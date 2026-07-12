"""
Revien Semantic Index — OPT-IN, LOCAL-FIRST hybrid vector retrieval.

Adds a vector-search layer over the existing graph retrieval so that a query
with NO keyword anchor still returns relevant results. Today, if the keyword/
entity extractor matches nothing, recall() returns empty — this closes that gap
by embedding the query and using the nearest stored nodes as ADDITIONAL anchors
for the same graph walk (union with keyword anchors), plus an optional semantic
score blend.

=========================== HARD CONTRACT ===========================
This layer is SPINE, not an extra: sqlite-vec + fastembed are CORE dependencies
(graph-only recall has no query-relevance signal — LoCoMo recall@10 0.05 vs
0.47 hybrid — so shipping without it ships degraded recall). The contract:

  * Imports stay GUARDED so a source install without the deps still runs:
    ``SemanticIndex.is_enabled`` is False, every method is an inert no-op, and
    graph retrieval / ingestion run unchanged — but the degrade is LOUD
    (stderr at engine construction + ``semantic_note`` on every recall
    response), never silent.

  * ``REVIEN_SEMANTIC`` gates activation. Default: enabled IFF sqlite-vec is
    importable. ``REVIEN_SEMANTIC=0`` force-disables. ``REVIEN_SEMANTIC=require``
    makes any missing dep or runtime failure a HARD ERROR instead of a degrade.

  * Embeddings stay in sync: the index registers a content listener on the
    store, so node label/content updates re-embed immediately and deletes drop
    the vector (no more stale-until-manual-reindex).

=========================== ARCHITECTURE ===========================
Storage  — sqlite-vec loadable extension (``sqlite_vec.load(conn)``) creates a
           ``vec0`` virtual table in the SAME SQLite db as the graph (no extra
           service). Node embeddings are stored keyed by node_id.

Embedding — pluggable ``EmbeddingProvider``:
              * FastEmbedProvider     — LOCAL default (BAAI/bge-small-en-v1.5,
                                        384-dim). No network on the default path.
              * OpenAIEmbeddingProvider — CLOUD, opt-in. Emits the SAME one-time
                                        disclosure style as leg 4's extractor
                                        ("sending text to <provider> ... leaves
                                        your machine").
            Provider selected via ``REVIEN_EMBEDDER`` (default "fastembed").
            Model overridable via ``REVIEN_EMBED_MODEL``.

Hybrid   — at recall: embed query -> vec0 top-K nearest node_ids -> union with
           keyword anchors. A semantic-similarity component (0..1) per node is
           also exposed so the engine can blend it into the score. When the
           layer is off, none of this runs.
"""

import os
import sqlite3
import struct
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


# ── Guarded heavy imports (the `semantic` extra) ──────────────────────
# sqlite-vec: vector virtual table. fastembed: local embeddings.
# If either is missing, the layer self-disables and recall/ingest are unchanged.
try:
    import sqlite_vec  # noqa: F401
    _SQLITE_VEC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    sqlite_vec = None
    _SQLITE_VEC_AVAILABLE = False

# fastembed availability is checked WITHOUT importing it: importing fastembed
# pulls in huggingface_hub machinery that can touch the network, and a hub
# endpoint that accepts connections but never answers turned that into a
# ~23-minute hang at IMPORT time (caught July 6 2026 — pytest collection and
# every fresh process stalled identically). The actual import happens lazily
# inside FastEmbedProvider._ensure_model, under offline-first control.
import importlib.util

_FASTEMBED_AVAILABLE = importlib.util.find_spec("fastembed") is not None

# The vector STORAGE requires sqlite-vec. A cloud embedder can supply vectors
# without fastembed, but with no local embedder and no sqlite-vec there is
# nothing to do. "Available" = storage backend present.
SEMANTIC_AVAILABLE = _SQLITE_VEC_AVAILABLE


# ── Config ────────────────────────────────────────────────────────────
DEFAULT_EMBEDDER = "fastembed"            # LOCAL, zero-network default
LOCAL_EMBEDDERS = ("fastembed",)
CLOUD_EMBEDDERS = ("openai",)
VALID_EMBEDDERS = LOCAL_EMBEDDERS + CLOUD_EMBEDDERS

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim
DEFAULT_LOCAL_DIM = 384
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"  # 1536-dim
DEFAULT_OPENAI_DIM = 1536
OPENAI_URL = "https://api.openai.com/v1/embeddings"
REQUEST_TIMEOUT = 30.0


def _semantic_enabled_by_env() -> bool:
    """REVIEN_SEMANTIC gate. Default: enabled iff the storage backend imports.

    Accepts 1/true/yes/on/require (force-on) and 0/false/no/off (force-off).
    Unset => default to availability. sqlite-vec is now a CORE dependency, so
    on a normal install this is on out of the box.
    """
    raw = os.environ.get("REVIEN_SEMANTIC")
    if raw is None:
        return SEMANTIC_AVAILABLE
    return raw.strip().lower() in ("1", "true", "yes", "on", "require", "required", "strict")


def _semantic_required() -> bool:
    """REVIEN_SEMANTIC=require: a missing/broken semantic layer is a hard error
    instead of a silent degrade to graph-only recall. For deployments where
    degraded recall quality is worse than a loud failure."""
    raw = os.environ.get("REVIEN_SEMANTIC", "")
    return raw.strip().lower() in ("require", "required", "strict")


# ── Cloud disclosure (mirrors leg-4 extractor_llm._disclose_cloud) ─────
_DISCLOSED_PROVIDERS: set = set()


def _disclose_cloud(provider: str) -> None:
    """One-time stderr warning when text leaves the machine for embeddings.

    Same style/voice as leg 4's extractor disclosure. Local embedders
    (fastembed) never call this.
    """
    if provider in _DISCLOSED_PROVIDERS:
        return
    _DISCLOSED_PROVIDERS.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending text to {provider} for embeddings "
        f"- this leaves your machine. Set REVIEN_EMBEDDER=fastembed (local) "
        f"to keep it on-device.\n"
    )
    sys.stderr.flush()


# ── Embedding provider abstraction ────────────────────────────────────
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract every embedder satisfies. Returns one float vector per text."""

    @property
    def dim(self) -> int: ...

    @property
    def is_cloud(self) -> bool: ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class FastEmbedProvider:
    """LOCAL embedder (default). BAAI/bge-small-en-v1.5, 384-dim, CPU-only.

    No network on the default path. Model loads lazily on first embed so that
    constructing the provider (and the whole engine) stays cheap and import-safe.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get(
            "REVIEN_EMBED_MODEL", DEFAULT_LOCAL_MODEL
        )
        self._model = None
        self._dim = DEFAULT_LOCAL_DIM  # bge-small is 384; refined after load

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError(
                "fastembed not installed (pip install revien[semantic])"
            )
        from fastembed import TextEmbedding
        # OFFLINE-FIRST via the per-call `local_files_only` PARAMETER — NOT the
        # HF_HUB_OFFLINE env var. huggingface_hub reads HF_HUB_OFFLINE into a
        # module constant at IMPORT time, so setting it before the first import
        # locks the whole process offline: the download fallback can never fire,
        # and on a COLD cache (every fresh `pip install`, every CI run) the model
        # can never be fetched — silently disabling the semantic spine for every
        # first-run user. `local_files_only` is honored per call, so the fallback
        # works: warm cache loads locally (zero network), cold cache downloads
        # once (the one legitimate, one-time fetch — see TELEMETRY.md).
        try:
            self._model = TextEmbedding(
                model_name=self.model_name, local_files_only=True
            )
        except Exception:
            self._model = TextEmbedding(model_name=self.model_name)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_cloud(self) -> bool:
        return False

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_model()
        vectors = [list(map(float, v)) for v in self._model.embed(list(texts))]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


class OpenAIEmbeddingProvider:
    """CLOUD embedder (opt-in). text-embedding-3-small, 1536-dim.

    Discloses ONCE before any text leaves the machine, in the same style as the
    leg-4 extractor. Uses stdlib urllib only — no new SDK dependency.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get(
            "REVIEN_EMBED_MODEL", DEFAULT_OPENAI_MODEL
        )
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self._dim = DEFAULT_OPENAI_DIM

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_cloud(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # Disclose BEFORE the network call (fires even if the request fails).
        _disclose_cloud("openai")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        import json
        import urllib.error
        import urllib.request

        payload = {"model": self.model_name, "input": list(texts)}
        req = urllib.request.Request(
            OPENAI_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover - network path
            body = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"openai HTTP {e.code}: {body}") from e

        vectors = [list(map(float, item["embedding"])) for item in data["data"]]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


def build_embedder(provider: Optional[str] = None) -> EmbeddingProvider:
    """Build the configured embedder. Default LOCAL fastembed.

    Selection precedence: explicit arg, else REVIEN_EMBEDDER, else "fastembed".
    Unknown values fall back to fastembed with a warning.
    """
    choice = (provider or os.environ.get("REVIEN_EMBEDDER", DEFAULT_EMBEDDER))
    choice = choice.lower().strip()

    if choice == "openai":
        return OpenAIEmbeddingProvider()
    if choice not in VALID_EMBEDDERS:
        sys.stderr.write(
            f"[revien] Unknown REVIEN_EMBEDDER={choice!r}; "
            f"valid: {', '.join(VALID_EMBEDDERS)}. Falling back to fastembed.\n"
        )
    return FastEmbedProvider()


def _serialize_f32(vector: Sequence[float]) -> bytes:
    """Pack a float vector to the little-endian float32 blob sqlite-vec wants."""
    return struct.pack(f"{len(vector)}f", *vector)


# ── The index ──────────────────────────────────────────────────────────
class SemanticIndex:
    """Optional hybrid vector layer over the graph.

    Self-disabling: when the `semantic` extra is absent OR REVIEN_SEMANTIC=0,
    ``is_enabled`` is False and every method no-ops, so recall()/ingest() behave
    exactly as before. The embedder is constructed lazily (first index/search)
    so an enabled-but-never-queried engine pays no model-load cost, and a
    misconfigured cloud key never breaks plain graph retrieval.
    """

    TABLE = "vec_nodes"
    # Deferred-embed queue (capture leg): plain SQLite table in the SAME db,
    # so queued work survives a daemon restart. Rows are node_ids only — label/
    # content are re-read from the store at drain time, so an edit made while
    # a node waits in the queue is embedded in its CURRENT form, never stale.
    PENDING_TABLE = "pending_embeds"
    # Per-search drain bound: keeps a pathological backlog from turning one
    # recall into a bulk reindex. Normal capture volume drains in one batch.
    PENDING_DRAIN_BATCH = 256

    def __init__(
        self,
        store,
        embedder: Optional[EmbeddingProvider] = None,
        enabled: Optional[bool] = None,
    ):
        self.store = store
        self._embedder = embedder
        self._embedder_built = embedder is not None
        self._table_ready = False
        self._dim: Optional[int] = None
        self._broken = False  # set if extension load fails at runtime
        self._broken_reason: Optional[str] = None
        self._pending_table_ready = False
        # How many deferred captures the MOST RECENT search() embedded —
        # surfaced on the recall response so a drain is visible, not silent.
        self._last_search_drained = 0

        # Resolve enablement: explicit arg wins, else env gate.
        if enabled is None:
            enabled = _semantic_enabled_by_env()
        self._enabled = bool(enabled) and SEMANTIC_AVAILABLE

        # REVIEN_SEMANTIC=require: refuse to construct a degraded engine.
        if _semantic_required() and not self._enabled:
            raise RuntimeError(
                "REVIEN_SEMANTIC=require but the semantic layer cannot start "
                f"(sqlite_vec importable: {_SQLITE_VEC_AVAILABLE}, "
                f"fastembed importable: {_FASTEMBED_AVAILABLE}). "
                "Install the missing dependency or unset REVIEN_SEMANTIC."
            )

        # Keep embeddings in sync with node edits/deletes. Without this, an
        # updated node kept its STALE vector until a manual reindex_all() —
        # vector search would keep matching the old content. Registration is
        # keyed, so the newest index over a store replaces the previous one
        # (they share the same vec table, so any live instance can serve).
        if self._enabled:
            self._register_store_listener()

    def _register_store_listener(self) -> None:
        """Wire this index to the store's content-change/delete hooks.
        Subclasses that force-enable after construction call this themselves."""
        if hasattr(self.store, "register_content_listener"):
            self.store.register_content_listener(
                "semantic_index",
                on_content_change=self._on_node_content_change,
                on_delete=self.remove_node,
            )

    # ── State ──────────────────────────────────────────────
    @property
    def is_enabled(self) -> bool:
        """True only when the extra is present, env allows it, and nothing
        has failed at runtime. The engine branches on this."""
        return self._enabled and not self._broken

    def status(self) -> Dict:
        return {
            "enabled": self.is_enabled,
            "extra_available": SEMANTIC_AVAILABLE,
            "sqlite_vec": _SQLITE_VEC_AVAILABLE,
            "fastembed": _FASTEMBED_AVAILABLE,
            "env_gate": _semantic_enabled_by_env(),
            "embedder": (
                "cloud:openai" if (self._embedder and self._embedder.is_cloud)
                else ("local:fastembed" if self._embedder_built else "unbuilt")
            ),
            "broken": self._broken,
            "broken_reason": self._broken_reason,
            "required": _semantic_required(),
            "dim": self._dim,
        }

    def inactive_reason(self) -> Optional[str]:
        """One-line human answer to 'why is semantic recall not running?'.
        None when the layer is active."""
        if self.is_enabled:
            return None
        if self._broken:
            return f"disabled after runtime error: {self._broken_reason}"
        if not _SQLITE_VEC_AVAILABLE:
            return "sqlite-vec not importable (core dependency missing?)"
        if not _semantic_enabled_by_env():
            return "force-disabled via REVIEN_SEMANTIC"
        return "disabled at construction (enabled=False)"

    # ── Lazy wiring ────────────────────────────────────────
    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = build_embedder()
            self._embedder_built = True
        return self._embedder

    def _load_extension(self, conn: sqlite3.Connection) -> None:
        """Load the sqlite-vec loadable extension onto the live connection."""
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def _ensure_table(self, dim: int) -> None:
        """Create the vec0 virtual table once, sized to the embedder dim."""
        if self._table_ready:
            return
        conn = self.store._get_conn()
        self._load_extension(conn)
        # Cosine distance: bge-small (and most sentence embedders) are trained
        # for cosine similarity. The default L2 metric compresses every pair
        # into a narrow band on these dense vectors, killing discrimination;
        # cosine separates the genuinely-relevant node from the rest.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.TABLE} "
            f"USING vec0(node_id TEXT PRIMARY KEY, "
            f"embedding float[{dim}] distance_metric=cosine)"
        )
        conn.commit()
        self._dim = dim
        self._table_ready = True

    def _safe_disable(self, exc: Exception) -> None:
        """A runtime failure must never break plain graph retrieval — UNLESS
        REVIEN_SEMANTIC=require, in which case a broken layer is a hard error
        (degraded recall is the failure mode that hides for weeks; require-mode
        deployments prefer the crash). Otherwise: mark broken, record WHY, and
        warn. The reason is surfaced through status() and every
        RetrievalResponse.semantic_note so the caller can see the degrade
        instead of silently getting graph-only results."""
        if _semantic_required():
            raise RuntimeError(
                f"semantic layer failed with REVIEN_SEMANTIC=require: {exc!r}"
            ) from exc
        self._broken = True
        self._broken_reason = repr(exc)
        sys.stderr.write(
            f"[revien.semantic] DISABLED after runtime error: {exc!r}. "
            f"Recall is now graph-only (keyword) retrieval - quality is "
            f"significantly degraded. Set REVIEN_SEMANTIC=require to make "
            f"this fatal instead.\n"
        )
        sys.stderr.flush()

    def _on_node_content_change(self, node_id: str, label: str, content: str) -> None:
        """Store listener: a node's label/content changed — refresh its vector."""
        self.index_node(node_id, label, content)

    def remove_node(self, node_id: str) -> None:
        """Store listener: node deleted — drop its vector so search can't
        return a ghost. Safe no-op when disabled or the table doesn't exist."""
        if not self.is_enabled or not self._table_ready:
            return
        try:
            conn = self.store._get_conn()
            conn.execute(f"DELETE FROM {self.TABLE} WHERE node_id = ?", (node_id,))
            conn.commit()
        except Exception as e:  # noqa: BLE001 - cleanup must not break deletes
            self._safe_disable(e)

    @staticmethod
    def _node_text(label: str, content: str) -> str:
        """Embedding text for a node: label carries the signal, content adds
        context. Mirrors what the keyword path searches over."""
        label = (label or "").strip()
        content = (content or "").strip()
        if content and content != label:
            return f"{label}. {content}"
        return label or content

    # ── Indexing ───────────────────────────────────────────
    def index_node(self, node_id: str, label: str, content: str) -> bool:
        """Embed and upsert a single node. No-op (returns False) when disabled.

        Failures self-disable the layer rather than propagating, so ingestion
        never crashes because of the optional semantic path.
        """
        if not self.is_enabled:
            return False
        try:
            text = self._node_text(label, content)
            if not text:
                return False
            embedder = self._get_embedder()
            vec = embedder.embed([text])[0]
            self._ensure_table(len(vec))
            conn = self.store._get_conn()
            # vec0 has no UPSERT; delete-then-insert keeps one row per node.
            conn.execute(f"DELETE FROM {self.TABLE} WHERE node_id = ?", (node_id,))
            conn.execute(
                f"INSERT INTO {self.TABLE}(node_id, embedding) VALUES (?, ?)",
                (node_id, _serialize_f32(vec)),
            )
            conn.commit()
            return True
        except Exception as e:  # noqa: BLE001 - optional layer must not break core
            self._safe_disable(e)
            return False

    def index_nodes(self, nodes: Sequence[Tuple[str, str, str]]) -> int:
        """Batch-embed (node_id, label, content) tuples. Returns count indexed."""
        if not self.is_enabled or not nodes:
            return 0
        try:
            texts = [self._node_text(lbl, ct) for (_id, lbl, ct) in nodes]
            keep = [(nid, t) for (nid, _l, _c), t in zip(nodes, texts) if t]
            if not keep:
                return 0
            embedder = self._get_embedder()
            vectors = embedder.embed([t for _nid, t in keep])
            if not vectors:
                return 0
            self._ensure_table(len(vectors[0]))
            conn = self.store._get_conn()
            for (nid, _t), vec in zip(keep, vectors):
                conn.execute(f"DELETE FROM {self.TABLE} WHERE node_id = ?", (nid,))
                conn.execute(
                    f"INSERT INTO {self.TABLE}(node_id, embedding) VALUES (?, ?)",
                    (nid, _serialize_f32(vec)),
                )
            conn.commit()
            return len(keep)
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return 0

    # ── Deferred embedding (capture leg: persist now, embed later) ────
    def _ensure_pending_table(self) -> None:
        """Plain table — no vec extension, no embedder. Creating it must work
        on a cold process where the model has never loaded: that is the point."""
        if self._pending_table_ready:
            return
        conn = self.store._get_conn()
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.PENDING_TABLE} "
            f"(node_id TEXT PRIMARY KEY, queued_at TEXT NOT NULL)"
        )
        conn.commit()
        self._pending_table_ready = True

    def defer_nodes(self, nodes: Sequence[Tuple[str, str, str]]) -> int:
        """Queue (node_id, label, content) tuples for later embedding instead
        of embedding now. The capture path uses this so an interactive ingest
        never blocks on a cold model load. Returns count queued.

        Only ids are stored; drain re-reads the node from the store, so the
        freshest label/content wins. No-op (0) when the layer is disabled —
        a disabled layer will never drain, and pretending to queue would
        promise semantic recall that can never arrive.
        """
        if not self.is_enabled or not nodes:
            return 0
        try:
            self._ensure_pending_table()
            conn = self.store._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            for node_id, _label, _content in nodes:
                conn.execute(
                    f"INSERT OR REPLACE INTO {self.PENDING_TABLE} "
                    f"(node_id, queued_at) VALUES (?, ?)",
                    (node_id, now),
                )
            conn.commit()
            return len(nodes)
        except Exception as e:  # noqa: BLE001 - queueing must not break ingest
            self._safe_disable(e)
            return 0

    def pending_count(self) -> int:
        """Deferred captures not yet embedded. 0 when disabled or none queued."""
        if not self.is_enabled:
            return 0
        try:
            conn = self.store._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (self.PENDING_TABLE,),
            ).fetchone()
            if not row or not row[0]:
                return 0
            self._pending_table_ready = True
            return int(
                conn.execute(f"SELECT COUNT(*) FROM {self.PENDING_TABLE}").fetchone()[0]
            )
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return 0

    def drain_pending(self, limit: Optional[int] = None) -> int:
        """Embed queued nodes and clear them from the queue. Returns count
        embedded. Rows whose node has since been deleted are dropped; rows are
        removed only after the embed batch succeeds, so a failure (which
        self-disables the layer) leaves the queue intact for a later process.
        """
        if not self.is_enabled:
            return 0
        if self.pending_count() == 0:
            return 0
        try:
            conn = self.store._get_conn()
            n = int(limit) if limit else self.PENDING_DRAIN_BATCH
            ids = [
                r[0]
                for r in conn.execute(
                    f"SELECT node_id FROM {self.PENDING_TABLE} "
                    f"ORDER BY queued_at LIMIT ?",
                    (n,),
                ).fetchall()
            ]
            if not ids:
                return 0
            nodes = self.store.get_nodes_bulk(ids)
            batch = [
                (nid, nodes[nid].label, nodes[nid].content)
                for nid in ids
                if nid in nodes
            ]
            embedded = self.index_nodes(batch) if batch else 0
            if self._broken:
                return embedded  # embed failed — keep the queue for later
            # Clear every fetched row: embedded ones, deleted-node ghosts, and
            # empty-text skips alike. Anything left behind would re-drain forever.
            conn.executemany(
                f"DELETE FROM {self.PENDING_TABLE} WHERE node_id = ?",
                [(nid,) for nid in ids],
            )
            conn.commit()
            return embedded
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return 0

    def pending_note(self) -> Optional[str]:
        """One-line recall-response note about deferred captures, or None when
        there is nothing to say (the common case — response shape unchanged)."""
        parts = []
        if self._last_search_drained:
            parts.append(
                f"embedded {self._last_search_drained} deferred capture(s) at search time"
            )
        remaining = self.pending_count()
        if remaining:
            parts.append(
                f"{remaining} capture(s) pending embedding (keyword-recallable only)"
            )
        return "; ".join(parts) if parts else None

    def reindex_all(self, batch_size: int = 256) -> Dict:
        """Backfill: embed every non-context node currently in the graph.

        Returns a summary dict. No-op summary when the layer is disabled.
        """
        if not self.is_enabled:
            return {"status": "disabled", "indexed": 0, **self.status()}
        try:
            from revien.graph.schema import NodeType

            all_nodes = self.store.list_nodes(limit=999999)
            batch: List[Tuple[str, str, str]] = []
            total = 0
            for node in all_nodes:
                # Index every node, including CONTEXT (verbatim turns) — the
                # coherent answer-bearing content for conversational memory.
                batch.append((node.node_id, node.label, node.content))
                if len(batch) >= batch_size:
                    total += self.index_nodes(batch)
                    batch = []
            if batch:
                total += self.index_nodes(batch)
            return {"status": "ok", "indexed": total, **self.status()}
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return {"status": "error", "indexed": 0, "error": repr(e)}

    # ── Search ─────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Embed the query and return [(node_id, similarity 0..1)] nearest first.

        Returns [] when disabled, when nothing is indexed, or on any runtime
        error (which self-disables the layer). Similarity = 1/(1+distance) from
        sqlite-vec's L2 distance, so higher = closer.
        """
        if not self.is_enabled or not query.strip():
            return []
        # Drain deferred captures FIRST so this very recall can see them —
        # "capture on the phone, ask at the desk" must not need two queries.
        # The model loads once either way (query embed needs it warm), and the
        # per-search batch bound keeps a backlog from stalling one recall.
        self._last_search_drained = self.drain_pending()
        if not self.is_enabled:  # drain failure self-disabled the layer
            return []
        try:
            embedder = self._get_embedder()
            qvec = embedder.embed([query])[0]
            # Nothing indexed yet -> table may not exist -> ensure it (sized to
            # the query dim) and return empty rather than erroring.
            self._ensure_table(len(qvec))
            conn = self.store._get_conn()
            rows = conn.execute(
                f"SELECT node_id, distance FROM {self.TABLE} "
                f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (_serialize_f32(qvec), int(top_k)),
            ).fetchall()
            return [(nid, 1.0 / (1.0 + float(dist))) for nid, dist in rows]
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return []

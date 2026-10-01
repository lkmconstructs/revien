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
    store, so node label/content updates are QUEUED for re-embed (drained by
    the next search / idle sweep — never inline under the store lock) and
    deletes drop the vector immediately (cheap SQL). No more
    stale-until-manual-reindex; freshness at query time is unchanged because
    search drains the queue first.

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
from contextlib import nullcontext
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


# ── Embed-context recipe (REVIEN_EMBED_CONTEXT) ───────────────────────
# What gets embedded for a CONTEXT (verbatim turn) node. "off" (default) is the
# node alone, exactly as before. "prev" prepends the preceding CONTEXT node of
# the same session_key so "I ran it last Saturday" carries its referent. Only
# the string handed to the embedder changes; stored node content never does.
EMBED_CONTEXT_OFF = "off"
EMBED_CONTEXT_PREV = "prev"
EMBED_CONTEXT_MAX_CHARS = 1200   # combined cap; the PREVIOUS turn is trimmed, never the current
_EMBED_CONTEXT_OFF_VALUES = ("", "0", "off", "false", "no", "none")


def embed_context_mode() -> str:
    """Current REVIEN_EMBED_CONTEXT setting: "off" (default) or "prev".
    Read per call so the env can change between operations. An unrecognised
    value is loud and treated as off."""
    raw = os.environ.get("REVIEN_EMBED_CONTEXT", "").strip().lower()
    if raw in _EMBED_CONTEXT_OFF_VALUES:
        return EMBED_CONTEXT_OFF
    if raw == EMBED_CONTEXT_PREV:
        return EMBED_CONTEXT_PREV
    sys.stderr.write(
        f"[revien.semantic] Unknown REVIEN_EMBED_CONTEXT={raw!r}; "
        f"valid: off, prev. Treating as off.\n"
    )
    return EMBED_CONTEXT_OFF


def _with_previous_turn(prev_content: str, content: str) -> str:
    """prev + newline + content, capped at EMBED_CONTEXT_MAX_CHARS. Over the
    cap, the previous turn loses characters from its LEFT (its tail survives);
    the current turn is never cut."""
    prev_content = (prev_content or "").strip()
    content = (content or "").strip()
    if not prev_content:
        return content
    room = EMBED_CONTEXT_MAX_CHARS - len(content) - 1
    if room <= 0:
        return content
    return f"{prev_content[-room:]}\n{content}"


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


class EmbedDimMismatch(RuntimeError):
    """The configured embedder's vector size differs from the stored vec
    table's. Inserting would corrupt or fail; `revien reindex` rebuilds."""


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
    # Recipe record: one key/value table in the same db. Holds the
    # embed-context mode the vectors in TABLE were built under.
    META_TABLE = "semantic_meta"

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
        # Open-time recipe/embedder warnings (shown by status() and the recall
        # response's semantic_note, not only stderr).
        self._open_warnings: List[str] = []
        self._rebuilding = False          # reindex_all may drop+recreate the vec table
        self._dim_mismatch = False        # broken BECAUSE of a dim change (reindex recovers)

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
        # vector search would keep matching the old content. Edits QUEUE a
        # re-embed (drained at next search — listeners run under the store
        # lock, so no model inference inline); deletes drop the vector
        # immediately. Registration is keyed, so the newest index over a
        # store replaces the previous one (they share the same vec table, so
        # any live instance can serve).
        if self._enabled:
            self._register_store_listener()
            self._warn_on_embed_context_mismatch()
            self._warn_on_embedder_mismatch()

    def _register_store_listener(self) -> None:
        """Wire this index to the store's content-change/delete hooks.
        Subclasses that force-enable after construction call this themselves."""
        if hasattr(self.store, "register_content_listener"):
            self.store.register_content_listener(
                "semantic_index",
                on_content_change=self._on_node_content_change,
                on_delete=self.remove_node,
                on_successor_change=self._on_successor_change,
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
            # Live provider state (None until built / when inactive). dim is
            # the provider's, refined after the first real embed.
            "embed_model": (getattr(self._embedder, "model_name", None)
                            if self.is_enabled else None),
            "embed_dim": (getattr(self._embedder, "dim", None)
                          if self.is_enabled and self._embedder is not None else None),
            # The CURRENT REVIEN_EMBED_CONTEXT recipe (what new vectors use).
            # The recipe the stored vectors were built under is in
            # semantic_meta; a difference is warned about at open.
            "embed_context": embed_context_mode() if self.is_enabled else None,
            # Open-time warnings (mixed recipe / embedder swapped since the
            # vectors were built). Empty list = none.
            "warnings": list(self._open_warnings),
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

    def warnings_note(self) -> Optional[str]:
        """Open-time warnings joined for a recall response; None when none."""
        return "; ".join(self._open_warnings) if self._open_warnings else None

    def _db(self):
        """The store's connection lock, held for each execute/commit block.

        The index rides the store's SHARED connection — an unlocked commit
        from the ingest thread would flush another thread's half-done store
        transaction (the audited-mutation guarantee). Embedding (model
        inference) stays OUTSIDE these blocks so the lock is never held for
        an embed. nullcontext keeps bare/mock stores working."""
        return getattr(self.store, "_lock", None) or nullcontext()

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

    def _stored_dim(self, conn) -> Optional[int]:
        """Dimension of the existing vec table (parsed from its DDL), or None
        when the table does not exist."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (self.TABLE,),
        ).fetchone()
        if row is None:
            return None
        import re
        m = re.search(r"float\[(\d+)\]", row[0] or "")
        return int(m.group(1)) if m else None

    def _ensure_table(self, dim: int) -> None:
        """Create the vec0 virtual table once, sized to the embedder dim.
        An existing table of a DIFFERENT dim raises EmbedDimMismatch (recall
        degrades loudly) unless a full reindex is rebuilding it."""
        if self._table_ready and self._dim is not None:
            return
        with self._db():
            conn = self.store._get_conn()
            self._load_extension(conn)
            stored = self._stored_dim(conn)
            if stored is not None and stored != dim:
                if not self._rebuilding:
                    raise EmbedDimMismatch(
                        f"vectors were built at dim {stored} but the configured "
                        f"embedder produces dim {dim}; run `revien reindex`")
                conn.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
                conn.commit()
                stored = None
            fresh = stored is None
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
            if fresh:
                self._record_embed_context(embed_context_mode())
                self._record_embedder(dim)
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
        if isinstance(exc, EmbedDimMismatch):
            self._dim_mismatch = True
            self._broken_reason = str(exc)
        sys.stderr.write(
            f"[revien.semantic] DISABLED after runtime error: {exc!r}. "
            f"Recall is now graph-only (keyword) retrieval - quality is "
            f"significantly degraded. Set REVIEN_SEMANTIC=require to make "
            f"this fatal instead.\n"
        )
        sys.stderr.flush()

    def _on_node_content_change(self, node_id: str, label: str, content: str) -> None:
        """Store listener: a node's label/content changed — QUEUE a re-embed.

        Queue, not embed: listeners fire while the store lock is held, and
        embedding means model inference (a cold fastembed load can hang for
        minutes) — inline it and every store consumer stalls behind one
        edit. The pending queue is a cheap INSERT; search() drains it before
        querying, so vector search sees the new content by the next recall
        (drain re-reads the node, so the freshest edit wins)."""
        self.defer_nodes([(node_id, label, content)])

    def _on_successor_change(self, node_id: str, label: str, content: str) -> None:
        """Store listener: the CONTEXT node AFTER an edited/deleted one. Under
        REVIEN_EMBED_CONTEXT=prev its vector embeds the predecessor's text, so
        it must be re-embedded or the old (possibly forgotten) text stays
        findable through it. No-op in other modes."""
        if embed_context_mode() == EMBED_CONTEXT_PREV:
            self.defer_nodes([(node_id, label, content)])

    def _ready_existing_table(self) -> bool:
        """Make an ALREADY-CREATED vec table usable on this connection.

        _table_ready is per-INSTANCE state: a fresh open of an existing db
        starts False and only _ensure_table (first index/search) flips it. A
        consumer that opens, deletes, and closes — every right-to-forget flow
        — never searches, so gating deletes on _table_ready alone leaked the
        deleted content's embedding forever (ghost vector). Probe
        sqlite_master for the real table instead; when present, load the
        extension (a vec0 DELETE needs it on the connection) and mark ready.
        Returns False when the table has never been created — nothing to
        delete from. _dim stays None on this path (unknown until an embed);
        _ensure_table still owns creation and sizing."""
        if self._table_ready:
            return True
        with self._db():
            conn = self.store._get_conn()
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (self.TABLE,),
            ).fetchone()
            if row is None:
                return False
            self._load_extension(conn)
        self._table_ready = True
        return True

    def remove_node(self, node_id: str) -> None:
        """Store listener: node deleted — drop its vector so search can't
        return a ghost. Safe no-op when disabled or the table was never
        created. Works on a fresh-opened index (see _ready_existing_table):
        right-to-forget must remove the embedding, not just the node row."""
        if not self.is_enabled:
            return
        try:
            if not self._ready_existing_table():
                return
            with self._db():
                conn = self.store._get_conn()
                conn.execute(f"DELETE FROM {self.TABLE} WHERE node_id = ?", (node_id,))
                conn.commit()
        except Exception as e:  # noqa: BLE001 - cleanup must not break deletes
            self._safe_disable(e)

    # ── Embed recipe record ────────────────────────────────
    def _recorded_embed_context(self) -> Optional[str]:
        """Mode the stored vectors were built under, or None when there are no
        vectors. A vec table with no recorded mode predates the knob: "off"."""
        with self._db():
            conn = self.store._get_conn()
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (self.TABLE,),
            ).fetchone() is None:
                return None
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (self.META_TABLE,),
            ).fetchone() is None:
                return EMBED_CONTEXT_OFF
            row = conn.execute(
                f"SELECT value FROM {self.META_TABLE} WHERE key = 'embed_context'"
            ).fetchone()
            return row[0] if row else EMBED_CONTEXT_OFF

    def _record_meta(self, **pairs) -> None:
        with self._db():
            conn = self.store._get_conn()
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.META_TABLE} "
                f"(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            for k, v in pairs.items():
                conn.execute(
                    f"INSERT OR REPLACE INTO {self.META_TABLE}(key, value) "
                    f"VALUES (?, ?)", (k, str(v)),
                )
            conn.commit()

    def _record_embed_context(self, mode: str) -> None:
        self._record_meta(embed_context=mode)

    def _record_embedder(self, dim: int) -> None:
        """Remember which model/dim the vectors in TABLE belong to."""
        model = getattr(self._embedder, "model_name", None)
        pairs = {"embed_dim": int(dim)}
        if model:
            pairs["embed_model"] = model
        self._record_meta(**pairs)

    def _recorded_meta(self, key: str) -> Optional[str]:
        with self._db():
            conn = self.store._get_conn()
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (self.META_TABLE,),
            ).fetchone() is None:
                return None
            row = conn.execute(
                f"SELECT value FROM {self.META_TABLE} WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def _warn_on_embed_context_mismatch(self) -> None:
        """One loud line at open when the stored vectors were built under a
        different REVIEN_EMBED_CONTEXT than the current one. Never rebuilds."""
        try:
            recorded = self._recorded_embed_context()
        except Exception:  # noqa: BLE001 - bare/mock stores, closed db
            return
        current = embed_context_mode()
        if recorded is not None and recorded != current:
            msg = (f"vectors were built under REVIEN_EMBED_CONTEXT={recorded} "
                   f"but the current setting is {current}. Mixed recipes "
                   f"degrade recall; run `revien reindex` to rebuild under "
                   f"the current setting.")
            self._open_warnings.append(msg)
            sys.stderr.write(f"[revien.semantic] WARNING: {msg}\n")
            sys.stderr.flush()

    def _warn_on_embedder_mismatch(self) -> None:
        """At open: the configured embedder model differs from the one the
        stored vectors were built with. Same-dim swaps keep working (warn
        only); a dim change is caught at the first embed and degrades recall
        to graph-only until `revien reindex`. Never loads a model."""
        try:
            recorded = self._recorded_meta("embed_model")
            if recorded is None:
                return
            emb = self._embedder if self._embedder is not None else build_embedder()
            current = getattr(emb, "model_name", None)
        except Exception:  # noqa: BLE001 - bare/mock stores, closed db
            return
        if current and current != recorded:
            msg = (f"vectors were built with {recorded}; the configured "
                   f"embedder is {current}. Run `revien reindex`.")
            self._open_warnings.append(msg)
            sys.stderr.write(f"[revien.semantic] WARNING: {msg}\n")
            sys.stderr.flush()

    def _embed_texts(self, items: Sequence[Tuple[str, str, str]]) -> List[str]:
        """The single place embed text is derived for stored nodes. Off: the
        node's own text (unchanged). prev: a CONTEXT node with a session_key
        is embedded as previous-turn + newline + this turn. Stored content is
        never touched. Claim nodes, session-less nodes and a session's first
        turn embed alone."""
        texts = [self._node_text(lbl, ct) for (_id, lbl, ct) in items]
        if embed_context_mode() != EMBED_CONTEXT_PREV or not items:
            return texts
        from revien.graph.schema import NodeType

        nodes = self.store.get_nodes_bulk([nid for (nid, _l, _c) in items])
        for i, (nid, _lbl, ct) in enumerate(items):
            node = nodes.get(nid)
            if (node is None or node.node_type != NodeType.CONTEXT
                    or not node.session_key or not (ct or "").strip()):
                continue
            prev = self.store.previous_context_in_session(node)
            if prev is not None:
                texts[i] = _with_previous_turn(prev.content, ct)
        return texts

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
            text = self._embed_texts([(node_id, label, content)])[0]
            if not text:
                return False
            embedder = self._get_embedder()
            vec = embedder.embed([text])[0]
            self._ensure_table(len(vec))
            with self._db():
                conn = self.store._get_conn()
                # Re-check the node still exists — embedding ran outside the
                # lock, and a delete in that window must not leave a ghost
                # vector search would keep returning.
                if conn.execute(
                    "SELECT 1 FROM nodes WHERE node_id = ?", (node_id,)
                ).fetchone() is None:
                    return False
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
            texts = self._embed_texts(nodes)
            keep = [(nid, t) for (nid, _l, _c), t in zip(nodes, texts) if t]
            if not keep:
                return 0
            embedder = self._get_embedder()
            vectors = embedder.embed([t for _nid, t in keep])
            if not vectors:
                return 0
            self._ensure_table(len(vectors[0]))
            indexed = 0
            with self._db():
                conn = self.store._get_conn()
                for (nid, _t), vec in zip(keep, vectors):
                    # Re-check the node still exists — embedding ran outside
                    # the lock, and a delete in that window (drain's TOCTOU)
                    # must not resurrect a ghost vector.
                    if conn.execute(
                        "SELECT 1 FROM nodes WHERE node_id = ?", (nid,)
                    ).fetchone() is None:
                        continue
                    conn.execute(f"DELETE FROM {self.TABLE} WHERE node_id = ?", (nid,))
                    conn.execute(
                        f"INSERT INTO {self.TABLE}(node_id, embedding) VALUES (?, ?)",
                        (nid, _serialize_f32(vec)),
                    )
                    indexed += 1
                conn.commit()
            return indexed
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return 0

    # ── Deferred embedding (capture leg: persist now, embed later) ────
    def _ensure_pending_table(self) -> None:
        """Plain table — no vec extension, no embedder. Creating it must work
        on a cold process where the model has never loaded: that is the point."""
        if self._pending_table_ready:
            return
        with self._db():
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
            with self._db():
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
            with self._db():
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
            n = int(limit) if limit else self.PENDING_DRAIN_BATCH
            with self._db():
                conn = self.store._get_conn()
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
            with self._db():
                conn = self.store._get_conn()
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
                f"{remaining} capture(s) pending embedding "
                f"(queued for the idle sweep / next recall)"
            )
        return "; ".join(parts) if parts else None

    def reindex_all(self, batch_size: int = 256) -> Dict:
        """Backfill: embed every node currently in the graph.

        The embed recipe (context mode, model, dim) is recorded ONLY when every
        batch succeeded; any failure returns status "partial" with the recipe
        untouched and a loud stderr line, so a half-built index can never
        claim to be built under the new recipe. A dim change (embedder swapped
        for one with a different vector size) is recovered here: the vec table
        is dropped and recreated under the new dim, then everything re-embeds.

        Returns a summary dict. No-op summary when the layer is disabled.
        """
        recovering = self._enabled and self._broken and self._dim_mismatch
        if not self.is_enabled and not recovering:
            return {"status": "disabled", "indexed": 0, **self.status()}
        if recovering:
            self._broken = False
            self._broken_reason = None
            self._dim_mismatch = False
            self._table_ready = False
            self._dim = None
        self._rebuilding = True
        try:
            all_nodes = self.store.list_nodes(limit=999999)
            batch: List[Tuple[str, str, str]] = []
            total = 0
            failed = False
            for node in all_nodes:
                # Index every node, including CONTEXT (verbatim turns) — the
                # coherent answer-bearing content for conversational memory.
                batch.append((node.node_id, node.label, node.content))
                if len(batch) >= batch_size:
                    total += self.index_nodes(batch)
                    batch = []
                    if not self.is_enabled:
                        failed = True
                        break
            if batch and not failed:
                total += self.index_nodes(batch)
            if not self.is_enabled:
                failed = True
            if failed:
                msg = (f"reindex FAILED part-way after {total} node(s): "
                       f"{self._broken_reason}. The embed recipe was NOT "
                       f"updated; vectors are a mix of old and new.")
                sys.stderr.write(f"[revien.semantic] ERROR: {msg}\n")
                sys.stderr.flush()
                return {"status": "partial", "indexed": total,
                        "error": self._broken_reason, **self.status()}
            if self._table_ready:
                self._record_embed_context(embed_context_mode())
                if self._dim:
                    self._record_embedder(self._dim)
            self._open_warnings = []
            return {"status": "ok", "indexed": total, **self.status()}
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return {"status": "error", "indexed": 0, "error": repr(e)}
        finally:
            self._rebuilding = False

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
            with self._db():
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

    def find_similar(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """Embed arbitrary text and return [(node_id, COSINE similarity)]
        nearest first. Unlike search() — whose 1/(1+distance) similarity is
        shaped for the recall score blend — this returns the raw cosine
        (1 - cosine_distance), so a caller's threshold reads exactly as the
        literature's does ("merge at >= 0.90"). Semantic dedup is the
        consumer: the candidate node is NOT yet stored, so every hit is an
        existing node. Drains the pending queue first for the same reason
        search() does — a paraphrase captured minutes ago must be findable —
        and callers on the defer-embed path skip this method entirely.
        Returns [] when disabled or on any runtime error (self-disabling)."""
        if not self.is_enabled or not text.strip():
            return []
        self.drain_pending()
        if not self.is_enabled:  # drain failure self-disabled the layer
            return []
        try:
            embedder = self._get_embedder()
            vec = embedder.embed([text])[0]
            self._ensure_table(len(vec))
            with self._db():
                conn = self.store._get_conn()
                rows = conn.execute(
                    f"SELECT node_id, distance FROM {self.TABLE} "
                    f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (_serialize_f32(vec), int(top_k)),
                ).fetchall()
            return [(nid, 1.0 - float(dist)) for nid, dist in rows]
        except Exception as e:  # noqa: BLE001
            self._safe_disable(e)
            return []

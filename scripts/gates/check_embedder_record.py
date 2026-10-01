"""G17: the semantic index records its embedder and recipe, refuses to mix,
and `revien reindex` recovers a dimension change.

CHECK: python scripts/gates/check_embedder_record.py
EXPECT: embedder record verification passed
"""
import contextlib
import hashlib
import io
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "1"
os.environ["REVIEN_RERANK"] = "0"
os.environ.pop("REVIEN_EMBED_CONTEXT", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.schema import Node, NodeType, SourceType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.retrieval.engine import RetrievalEngine  # noqa: E402
from revien.semantic.index import SEMANTIC_AVAILABLE, SemanticIndex  # noqa: E402

T0 = datetime(2023, 5, 7, tzinfo=timezone.utc)


def fail(msg):
    print(f"check_embedder_record: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


class BoW:
    """Stub embedder: deterministic bag-of-words, no model, no network."""
    is_cloud = False

    def __init__(self, dim=4, name="stub-a", fail_on_call=None):
        self.dim, self.model_name = dim, name
        self.seen, self.calls, self.fail_on_call = [], 0, fail_on_call

    def embed(self, texts):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("embedder died mid-reindex")
        out = []
        for t in texts:
            self.seen.append(t)
            v = [0.0] * self.dim
            for w in t.lower().replace(":", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def turn(store, content, seq, sess="c1:s1"):
    return store.add_node(Node(
        node_type=NodeType.CONTEXT, label=content[:200], content=content,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        created_at=T0 + timedelta(seconds=seq), last_accessed=T0,
        recorded_at=T0, session_key=sess))


def open_idx(path, **kw):
    store = GraphStore(db_path=path)
    return store, SemanticIndex(store, embedder=BoW(**kw), enabled=True)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def raw_meta(path):
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute("SELECT key, value FROM semantic_meta").fetchall())
    finally:
        conn.close()


def main_check(tmp):
    check(SEMANTIC_AVAILABLE, "sqlite-vec not importable; the semantic layer cannot be exercised")
    db = os.path.join(tmp, "a.db")

    # 1. Build at dim 4: the record is written.
    store, idx = open_idx(db, dim=4, name="stub-a")
    for i in range(3):
        turn(store, f"Alice: postgres billing note {i}", i)
    res, _ = quiet(idx.reindex_all)
    check(res["status"] == "ok" and res["indexed"] == 3, f"initial build failed: {res['status']}")
    store.close()
    meta = raw_meta(db)
    check(meta.get("embed_model") == "stub-a" and meta.get("embed_dim") == "4",
          f"semantic_meta lacks embedder record: {meta}")
    check(meta.get("embed_context") == "off", "semantic_meta lacks the context recipe")

    # 2. Positive control: same embedder reopens clean and recalls active.
    store, idx = open_idx(db, dim=4, name="stub-a")
    check(idx.status()["warnings"] == [], "same embedder reopened with a warning")
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    check(resp.semantic_active is True and not resp.semantic_note,
          "same embedder: recall not active/clean")
    store.close()

    # 3. Same-dim swap: warns in status(), keeps working.
    store, idx = open_idx(db, dim=4, name="stub-c")
    warns = idx.status()["warnings"]
    check(any("stub-a" in w and "stub-c" in w and "revien reindex" in w for w in warns),
          f"same-dim model swap did not warn: {warns}")
    store.close()

    # 4. Dim change: recall degrades to graph-only with a note naming reindex.
    store, idx = open_idx(db, dim=8, name="stub-b")
    warns = idx.status()["warnings"]
    check(any("stub-a" in w and "revien reindex" in w for w in warns),
          f"dim change did not warn at open: {warns}")
    resp, _ = quiet(lambda: RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3))
    check(resp.semantic_active is False, "dim change: recall still claims semantic_active")
    check("revien reindex" in (resp.semantic_note or ""), "dim change: note does not name reindex")
    # The stale index was not silently mixed: meta still describes the old build.
    check(raw_meta(db).get("embed_dim") == "4", "dim change rewrote the record without a reindex")

    # 5. Reindex recovers on the same degraded object; meta updated.
    res, _ = quiet(idx.reindex_all)
    check(res["status"] == "ok" and res["indexed"] == 3, f"reindex did not recover: {res['status']}")
    check(idx.is_enabled, "index still disabled after reindex")
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    check(resp.semantic_active is True and not resp.semantic_note,
          "after reindex: recall not active/clean")
    meta = raw_meta(db)
    check(meta.get("embed_model") == "stub-b" and meta.get("embed_dim") == "8",
          f"reindex did not update the record: {meta}")
    store.close()
    store, idx = open_idx(db, dim=8, name="stub-b")
    check(idx.status()["warnings"] == [], "fresh open after reindex still warns")
    store.close()

    # 6. Partial reindex: embedder dies on batch 2 -> partial, recipe unchanged.
    db2 = os.path.join(tmp, "b.db")
    store, idx = open_idx(db2, dim=4, name="stub-a")
    for i in range(5):
        turn(store, f"Alice: turn number {i} about postgres", i)
    check(quiet(idx.reindex_all)[0]["status"] == "ok", "partial fixture: initial build failed")
    store.close()
    before = raw_meta(db2)
    os.environ["REVIEN_EMBED_CONTEXT"] = "prev"
    try:
        store, idx = open_idx(db2, dim=4, name="stub-p", fail_on_call=2)
        res, err = quiet(idx.reindex_all, 2)
        check(res["status"] == "partial", f"dead embedder gave status {res['status']!r}, not partial")
        check("reindex FAILED" in err, "partial reindex was not loud on stderr")
        store.close()
    finally:
        os.environ.pop("REVIEN_EMBED_CONTEXT", None)
    after = raw_meta(db2)
    check(after == before, f"partial reindex changed the recorded recipe: {before} -> {after}")
    check(after.get("embed_context") == "off" and after.get("embed_model") == "stub-a",
          "partial reindex left the wrong recipe")

    # 7. REVIEN_EMBED_CONTEXT=prev: edit/delete of turn 2 re-embeds turn 3.
    os.environ["REVIEN_EMBED_CONTEXT"] = "prev"
    try:
        for mode in ("edit", "delete"):
            dbp = os.path.join(tmp, f"prev_{mode}.db")
            store, idx = open_idx(dbp, dim=4, name="stub-a")
            n1 = turn(store, "Alice: hello there", 1)
            n2 = turn(store, "Bob: my zebra passport number is 77341", 2)
            n3 = turn(store, "Alice: wow ok", 3)
            idx.index_nodes([(n.node_id, n.label, n.content) for n in (n1, n2, n3)])
            check("zebra passport number is 77341" in idx._embedder.seen[-1],
                  "prev mode: turn 3 was not embedded with turn 2")
            idx._embedder.seen.clear()
            if mode == "edit":
                store.update_node(n2.node_id, content="Bob: I like tea", label="Bob: I like tea")
                check(idx.pending_count() == 2, "prev edit: turn 2 and turn 3 not both re-queued")
                idx.drain_pending()
                check("Bob: I like tea\nAlice: wow ok" in idx._embedder.seen,
                      "prev edit: turn 3 not re-embedded with the edited turn 2")
            else:
                store.delete_node(n2.node_id)
                check(idx.pending_count() == 1, "prev delete: turn 3 not re-queued")
                idx.drain_pending()
                check(idx._embedder.seen == ["Alice: hello there\nAlice: wow ok"],
                      f"prev delete: turn 3 re-embedded wrongly: {idx._embedder.seen}")
            check(not any("zebra" in t for t in idx._embedder.seen),
                  f"prev {mode}: the old turn 2 text was re-embedded")
            store.close()
        # Control: off mode does not re-queue the successor on edit.
        os.environ.pop("REVIEN_EMBED_CONTEXT", None)
        dbo = os.path.join(tmp, "off.db")
        store, idx = open_idx(dbo, dim=4, name="stub-a")
        n1 = turn(store, "Alice: hello there", 1)
        n2 = turn(store, "Bob: first", 2)
        n3 = turn(store, "Alice: wow ok", 3)
        idx.index_nodes([(n.node_id, n.label, n.content) for n in (n1, n2, n3)])
        store.update_node(n2.node_id, content="Bob: second", label="Bob: second")
        check(idx.pending_count() == 1, "off mode re-queued the successor (control failed)")
        store.close()
    finally:
        os.environ.pop("REVIEN_EMBED_CONTEXT", None)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    try:
        main_check(tmp)
    except SystemExit:
        raise
    except Exception as exc:  # any crash is a failed gate
        fail(f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("embedder record verification passed")

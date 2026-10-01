"""Semantic index honesty about HOW its vectors were built. OFFLINE (stub
embedders; no model loads, no network).

S3  reindex_all records the recipe only when every batch succeeded.
S4  prev-mode: editing/deleting a turn re-embeds the NEXT turn (right to forget).
S5  the embedder (model, dim) is recorded in the store; a swap is loud in
    status() and semantic_note; a dim change degrades recall until
    `revien reindex`, which recovers by rebuilding the vec table.
"""

import io
import contextlib
import hashlib
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SEMANTIC_AVAILABLE, SemanticIndex

pytestmark = pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="sqlite-vec absent")

T0 = datetime(2023, 5, 7, tzinfo=timezone.utc)


class BoW:
    is_cloud = False

    def __init__(self, dim=16, name="stub-a", fail_on_call=None):
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


@pytest.fixture
def path():
    fd, p = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield p
    for _ in range(1):
        try:
            os.unlink(p)
        except OSError:  # pragma: no cover
            pass


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("REVIEN_EMBED_CONTEXT", raising=False)


def _turn(store, content, seq, sess="c1:s1"):
    return store.add_node(Node(
        node_type=NodeType.CONTEXT, label=content[:200], content=content,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        created_at=T0 + timedelta(seconds=seq), last_accessed=T0,
        recorded_at=T0, session_key=sess))


def _open(path, **kw):
    store = GraphStore(db_path=path)
    return store, SemanticIndex(store, embedder=BoW(**kw), enabled=True)


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ── S4 ────────────────────────────────────────────────────────────────

def test_prev_mode_edit_requeues_successor(path, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    store, idx = _open(path)
    n1 = _turn(store, "Alice: hello there", 1)
    n2 = _turn(store, "Bob: my zebra passport number is 77341", 2)
    n3 = _turn(store, "Alice: wow ok", 3)
    idx.index_nodes([(n.node_id, n.label, n.content) for n in (n1, n2, n3)])
    assert "zebra passport number is 77341" in idx._embedder.seen[-1]  # n3 carries n2
    idx._embedder.seen.clear()
    store.update_node(n2.node_id, content="Bob: I like tea", label="Bob: I like tea")
    assert idx.pending_count() == 2  # n2 itself AND its successor n3
    idx.drain_pending()
    assert "Bob: I like tea\nAlice: wow ok" in idx._embedder.seen
    assert not any("zebra" in t for t in idx._embedder.seen)
    store.close()


def test_prev_mode_delete_requeues_successor_so_forgotten_text_is_gone(path, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    store, idx = _open(path)
    n1 = _turn(store, "Alice: hello there", 1)
    n2 = _turn(store, "Bob: my zebra passport number is 77341", 2)
    n3 = _turn(store, "Alice: wow ok", 3)
    idx.index_nodes([(n.node_id, n.label, n.content) for n in (n1, n2, n3)])
    idx._embedder.seen.clear()
    store.delete_node(n2.node_id)
    assert idx.pending_count() == 1
    idx.drain_pending()
    assert idx._embedder.seen == ["Alice: hello there\nAlice: wow ok"]
    assert not any("zebra" in t for t in idx._embedder.seen)
    # and the deleted node is not findable by id
    assert n2.node_id not in [nid for nid, _ in idx.search("zebra passport 77341", top_k=5)]
    store.close()


def test_off_mode_does_not_requeue_successor(path):
    store, idx = _open(path)
    n1 = _turn(store, "Alice: hello there", 1)
    n2 = _turn(store, "Bob: first", 2)
    n3 = _turn(store, "Alice: wow ok", 3)
    idx.index_nodes([(n.node_id, n.label, n.content) for n in (n1, n2, n3)])
    store.update_node(n2.node_id, content="Bob: second", label="Bob: second")
    assert idx.pending_count() == 1  # only n2
    store.delete_node(n2.node_id)
    assert idx.pending_count() == 1
    store.close()


# ── S5 ────────────────────────────────────────────────────────────────

def _seeded(path, **kw):
    store, idx = _open(path, **kw)
    for i in range(3):
        _turn(store, f"Alice: postgres billing note {i}", i)
    assert idx.reindex_all()["status"] == "ok"
    store.close()


def test_embedder_recorded_in_store(path):
    _seeded(path, dim=16, name="stub-a")
    store = GraphStore(db_path=path)
    idx = SemanticIndex(store, embedder=BoW(), enabled=True)
    assert idx._recorded_meta("embed_model") == "stub-a"
    assert idx._recorded_meta("embed_dim") == "16"
    store.close()


def test_dim_change_degrades_loudly_then_reindex_recovers(path):
    _seeded(path, dim=16, name="stub-a")
    store, idx = _open(path, dim=32, name="stub-b")
    st = idx.status()
    assert any("stub-a" in w and "revien reindex" in w for w in st["warnings"])
    resp, err = _quiet(lambda: RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3))
    assert resp.semantic_active is False
    assert "revien reindex" in resp.semantic_note and "dim 16" in resp.semantic_note
    store.close()

    store, idx = _open(path, dim=32, name="stub-b")
    res, _ = _quiet(idx.reindex_all)
    assert res["status"] == "ok" and res["indexed"] == 3
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    assert resp.semantic_active is True and not resp.semantic_note
    assert idx._recorded_meta("embed_model") == "stub-b"
    assert idx._recorded_meta("embed_dim") == "32"
    store.close()
    # fresh open: clean
    store, idx = _open(path, dim=32, name="stub-b")
    assert idx.status()["warnings"] == []
    store.close()


def test_dim_change_recovers_even_when_the_session_already_degraded(path):
    """The SAME index object that tripped on the mismatch can run the reindex."""
    _seeded(path, dim=16, name="stub-a")
    store, idx = _open(path, dim=32, name="stub-b")
    _quiet(lambda: RetrievalEngine(store, semantic=idx).recall("postgres", top_n=3))
    assert not idx.is_enabled
    res, _ = _quiet(idx.reindex_all)
    assert res["status"] == "ok" and idx.is_enabled
    store.close()


def test_same_dim_swap_warns_keeps_working_and_reindex_clears(path):
    _seeded(path, dim=16, name="stub-a")
    store, idx = _open(path, dim=16, name="stub-c")
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    assert resp.semantic_active is True
    assert "stub-a" in (resp.semantic_note or "") and "revien reindex" in resp.semantic_note
    res, _ = _quiet(idx.reindex_all)
    assert res["status"] == "ok"
    store.close()
    store, idx = _open(path, dim=16, name="stub-c")
    assert idx.status()["warnings"] == []
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    assert not resp.semantic_note
    store.close()


def test_recipe_mismatch_surfaces_in_status_and_semantic_note(path, monkeypatch):
    _seeded(path, dim=16, name="stub-a")
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    store, idx = _open(path, dim=16, name="stub-a")
    assert any("REVIEN_EMBED_CONTEXT=off" in w for w in idx.status()["warnings"])
    resp = RetrievalEngine(store, semantic=idx).recall("postgres billing", top_n=3)
    assert resp.semantic_active is True
    assert "REVIEN_EMBED_CONTEXT=off" in resp.semantic_note
    store.close()

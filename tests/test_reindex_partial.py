"""S3: reindex_all records the embed recipe ONLY when every batch succeeded.
OFFLINE: stub embedder, no model loads, no network."""

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


# ── S3 ────────────────────────────────────────────────────────────────

def test_partial_reindex_never_records_the_new_recipe(path, monkeypatch):
    store, idx = _open(path)
    for i in range(5):
        _turn(store, f"Alice: turn number {i} about postgres", i)
    assert idx.reindex_all()["status"] == "ok"
    assert idx._recorded_embed_context() == "off"
    store.close()

    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    store, idx = _open(path, fail_on_call=2)  # batch 2 of 3 dies
    res, err = _quiet(idx.reindex_all, 2)
    assert res["status"] == "partial"
    assert res["indexed"] == 2
    assert "reindex FAILED" in err
    assert idx._recorded_embed_context() == "off", "recipe must stay the OLD one"
    store.close()



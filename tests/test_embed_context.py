"""
REVIEN_EMBED_CONTEXT: opt-in "prev" embeds a turn together with the turn before
it in the same session. OFFLINE -- the embedder is a stub that records the
strings it is asked to embed; no model loads.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.semantic import index as sem_index
from revien.semantic.index import (
    EMBED_CONTEXT_MAX_CHARS,
    SemanticIndex,
    SEMANTIC_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="sqlite-vec absent")

T0 = datetime(2023, 5, 7, tzinfo=timezone.utc)


class RecordingEmbedder:
    dim = 4
    is_cloud = False
    model_name = "stub"

    def __init__(self):
        self.seen = []

    def embed(self, texts):
        self.seen.extend(texts)
        return [[float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts]


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("REVIEN_EMBED_CONTEXT", raising=False)


def _turn(store, content, session="c1:s1", seq=0, recorded=T0, node_type=NodeType.CONTEXT):
    node = Node(
        node_type=node_type,
        label=content[:200],
        content=content,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        created_at=T0 + timedelta(seconds=seq),
        last_accessed=T0,
        recorded_at=recorded,
        session_key=session,
    )
    return store.add_node(node)


def _index_all(idx, store, nodes):
    idx.index_nodes([(n.node_id, n.label, n.content) for n in nodes])


def _mk(store):
    emb = RecordingEmbedder()
    return SemanticIndex(store, embedder=emb, enabled=True), emb


def test_off_strings_identical_to_today(store):
    idx, emb = _mk(store)
    nodes = [_turn(store, f"A: turn {i}", seq=i) for i in range(3)]
    _index_all(idx, store, nodes)
    assert emb.seen == [SemanticIndex._node_text(n.label, n.content) for n in nodes]
    for v in ("0", "off", ""):
        os.environ["REVIEN_EMBED_CONTEXT"] = v
        emb.seen.clear()
        _index_all(idx, store, nodes)
        assert emb.seen == [n.content for n in nodes]


def test_prev_joins_previous_turn(store, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    idx, emb = _mk(store)
    n1 = _turn(store, "Melanie: I signed up for a charity race.", seq=1)
    n2 = _turn(store, "Melanie: I ran it last Saturday.", seq=2)
    nosess = _turn(store, "Bob: no session here", session=None, seq=3)
    fact = _turn(store, "Melanie runs races", seq=4, node_type=NodeType.FACT)
    _index_all(idx, store, [n1, n2, nosess, fact])
    assert emb.seen == [
        n1.content,
        f"{n1.content}\n{n2.content}",
        nosess.content,
        fact.content,
    ]
    # stored content never changes
    assert store.get_node(n2.node_id).content == "Melanie: I ran it last Saturday."


def test_prev_single_index_node_and_drain_paths(store, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    idx, emb = _mk(store)
    n1 = _turn(store, "A: first", seq=1)
    n2 = _turn(store, "A: second", seq=2)
    idx.index_node(n2.node_id, n2.label, n2.content)
    assert emb.seen == ["A: first\nA: second"]
    # deferred queue drain goes through index_nodes
    emb.seen.clear()
    idx.defer_nodes([(n2.node_id, n2.label, n2.content)])
    assert idx.drain_pending() == 1
    assert emb.seen == ["A: first\nA: second"]


def test_listener_requeue_uses_prev(store, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    idx, emb = _mk(store)
    n1 = _turn(store, "A: first", seq=1)
    n2 = _turn(store, "A: second", seq=2)
    _index_all(idx, store, [n1, n2])
    emb.seen.clear()
    store.update_node(n2.node_id, content="A: second, edited")
    assert idx.pending_count() == 1
    idx.drain_pending()
    assert emb.seen == ["A: first\nA: second, edited"]


def test_long_previous_turn_trimmed_from_left_current_intact(store, monkeypatch):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    idx, emb = _mk(store)
    long_prev = "".join(chr(97 + (i % 26)) for i in range(3000)) + "TAILEND"
    n1 = _turn(store, long_prev, seq=1)
    cur = "B: " + "x" * 100
    n2 = _turn(store, cur, seq=2)
    idx.index_nodes([(n2.node_id, n2.label, n2.content)])
    text = emb.seen[0]
    assert len(text) == EMBED_CONTEXT_MAX_CHARS
    assert text.endswith("\n" + cur)
    assert "TAILEND" in text
    assert text.startswith(long_prev[-(EMBED_CONTEXT_MAX_CHARS - len(cur) - 1):][:20])

    # current turn alone longer than the cap: never cut, previous dropped
    emb.seen.clear()
    huge = "C: " + "y" * 2000
    n3 = _turn(store, huge, seq=3)
    idx.index_nodes([(n3.node_id, n3.label, n3.content)])
    assert emb.seen == [huge]


def test_previous_lookup_respects_session_and_order(store):
    a2 = _turn(store, "a2", session="c:s1", seq=20, recorded=T0 + timedelta(days=1))
    a1 = _turn(store, "a1", session="c:s1", seq=10, recorded=T0)  # older recorded_at, inserted later
    b1 = _turn(store, "b1", session="c:s2", seq=5, recorded=T0 + timedelta(days=5))
    assert store.previous_context_in_session(a1) is None
    assert store.previous_context_in_session(a2).node_id == a1.node_id
    assert store.previous_context_in_session(b1) is None
    # same recorded_at: created_at decides; equal both: insertion order
    c1 = _turn(store, "c1", session="c:s3", seq=1)
    c2 = _turn(store, "c2", session="c:s3", seq=2)
    c3 = _turn(store, "c3", session="c:s3", seq=2)
    assert store.previous_context_in_session(c2).node_id == c1.node_id
    assert store.previous_context_in_session(c3).node_id == c2.node_id
    # claim nodes and session-less nodes are never a "previous turn"
    f = _turn(store, "fact", session="c:s3", seq=3, node_type=NodeType.FACT)
    c4 = _turn(store, "c4", session="c:s3", seq=4)
    assert store.previous_context_in_session(c4).node_id == c3.node_id
    nosess = _turn(store, "x", session=None, seq=9)
    assert store.previous_context_in_session(nosess) is None


def test_reindex_rebuilds_under_current_mode(store, monkeypatch):
    idx, emb = _mk(store)
    n1 = _turn(store, "A: first", seq=1)
    n2 = _turn(store, "A: second", seq=2)
    _index_all(idx, store, [n1, n2])  # built under off
    assert idx._recorded_embed_context() == "off"

    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    emb.seen.clear()
    res = idx.reindex_all()
    assert res["status"] == "ok" and res["embed_context"] == "prev"
    assert "A: first\nA: second" in emb.seen
    assert idx._recorded_embed_context() == "prev"


def test_mode_mismatch_logs_warning_on_open(store, monkeypatch, capsys):
    idx, emb = _mk(store)
    _index_all(idx, store, [_turn(store, "A: first", seq=1)])
    capsys.readouterr()

    SemanticIndex(store, embedder=emb, enabled=True)  # same mode: silent
    assert "REVIEN_EMBED_CONTEXT" not in capsys.readouterr().err

    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    SemanticIndex(store, embedder=emb, enabled=True)
    err = capsys.readouterr().err
    assert err.count("built under REVIEN_EMBED_CONTEXT=off") == 1
    assert "revien reindex" in err


def test_no_vectors_no_warning(store, monkeypatch, capsys):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "prev")
    SemanticIndex(store, embedder=RecordingEmbedder(), enabled=True)
    assert "REVIEN_EMBED_CONTEXT" not in capsys.readouterr().err


def test_unknown_value_is_loud_and_off(monkeypatch, capsys):
    monkeypatch.setenv("REVIEN_EMBED_CONTEXT", "sideways")
    assert sem_index.embed_context_mode() == "off"
    assert "Unknown REVIEN_EMBED_CONTEXT" in capsys.readouterr().err


def test_legacy_db_without_session_index_opens(tmp_path):
    path = str(tmp_path / "legacy.db")
    s = GraphStore(db_path=path)
    s.close()
    conn = sqlite3.connect(path)
    conn.execute("DROP INDEX IF EXISTS idx_nodes_session")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    s = GraphStore(db_path=path)
    try:
        names = {r[0] for r in s._get_conn().execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_nodes_session" in names
    finally:
        s.close()

    # A db that predates the origin columns entirely (no session_key column):
    # the index must be created by the migration, after the ALTERs.
    path2 = str(tmp_path / "ancient.db")
    GraphStore(db_path=path2).close()
    conn = sqlite3.connect(path2)
    for idx in ("idx_nodes_session", "idx_nodes_origin_runtime",
                "idx_nodes_origin_source", "idx_nodes_project"):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    for col in ("origin_runtime", "origin_source", "project_key", "session_key"):
        conn.execute(f"ALTER TABLE nodes DROP COLUMN {col}")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    s = GraphStore(db_path=path2)
    try:
        cols = {r[1] for r in s._get_conn().execute("PRAGMA table_info(nodes)")}
        assert "session_key" in cols
        names = {r[0] for r in s._get_conn().execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_nodes_session" in names
    finally:
        s.close()

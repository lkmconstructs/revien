"""
Storage-core repair (R1): the invariants this leg exists to hold.

1. One reentrant lock over the shared connection — concurrent writers
   (Hermes worker thread, sync dream job, main-thread recalls) never race
   the same cursor. N threads hammering one GraphStore lose nothing.
2. Atomic audited mutations — a mutation and its audit row land in ONE
   commit. An audit failure rolls the mutation back; provenance is a
   required invariant, not best-effort bookkeeping.
3. Loud migrations — only the duplicate-column/already-exists idempotency
   race is swallowed; locked/disk/corruption errors raise at startup.
4. Transactional import with modes — refuse (default) / merge / replace,
   prevalidated first, one transaction, db untouched on any failure. The
   old shape committed the wipe FIRST, so a malformed edge mid-import left
   the original graph erased and half-replaced.

Temp-db pattern from test_capture.py (mkstemp + guarded unlink — Windows
WAL teardown races are known-environmental).
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.graph.schema import Edge, EdgeType, Graph, Node, NodeType
from revien.graph.store import (
    GraphStore, ImportRefusedError, ImportValidationError,
)

try:
    from test_capture import _QueueTestIndex
except ImportError:  # pragma: no cover - direct-run path
    from tests.test_capture import _QueueTestIndex


# ── Fixtures ──────────────────────────────────────────────

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


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fake_home(monkeypatch, tmp_dir):
    """Temp HOME so the CLI's config resolution never touches the real one."""
    home = tmp_dir / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    yield home


def _fact(label, content=None, node_id=None):
    kwargs = {"node_type": NodeType.FACT, "label": label,
              "content": content or f"Content of {label}."}
    if node_id:
        kwargs["node_id"] = node_id
    return Node(**kwargs)


def _graph(n_nodes=3, with_edge=True):
    nodes = [_fact(f"import {i}") for i in range(n_nodes)]
    edges = []
    if with_edge and n_nodes >= 2:
        edges.append(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=nodes[0].node_id,
            target_node_id=nodes[1].node_id,
        ))
    return Graph(nodes=nodes, edges=edges)


# ── 1. Concurrency: one lock, zero lost writes ────────────

class TestConcurrentAccess:
    N_THREADS = 3
    PER_THREAD = 200  # 600 total

    def test_concurrent_writes_lose_nothing(self, store):
        errors = []

        def writer(t):
            try:
                for i in range(self.PER_THREAD):
                    store.add_node(_fact(f"t{t}-n{i}"))
            except Exception as e:  # noqa: BLE001 - collected for the assert
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,))
                   for t in range(self.N_THREADS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], f"concurrent writes raised: {errors[:3]}"
        total = self.N_THREADS * self.PER_THREAD
        assert store.count_nodes() == total

        # Provenance invariant under concurrency: every row has its audit.
        conn = store._get_conn()
        audited = conn.execute(
            "SELECT COUNT(DISTINCT node_id) FROM audit_log WHERE op='create'"
        ).fetchone()[0]
        assert audited == total, "every write must carry its audit record"

    def test_readers_alongside_writers(self, store):
        errors = []
        stop = threading.Event()

        def writer(t):
            try:
                for i in range(100):
                    store.add_node(_fact(f"rw-t{t}-n{i}"))
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        per_reader_counts = [[], []]

        def reader(counts):
            try:
                while not stop.is_set():
                    counts.append(store.count_nodes())
                    nodes = store.list_nodes(limit=50)
                    if nodes:
                        # Consistent reads: every id list_nodes just returned
                        # must resolve via a bulk get — no torn rows, no
                        # phantom ids (writers only ever ADD in this test).
                        ids = [n.node_id for n in nodes]
                        got = store.get_nodes_bulk(ids)
                        assert set(got) == set(ids)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        writers = [threading.Thread(target=writer, args=(t,)) for t in range(3)]
        readers = [threading.Thread(target=reader, args=(c,))
                   for c in per_reader_counts]
        for th in readers + writers:
            th.start()
        for th in writers:
            th.join()
        stop.set()
        for th in readers:
            th.join()

        assert errors == [], f"concurrent read/write raised: {errors[:3]}"
        assert store.count_nodes() == 300
        # Counts each reader observed mid-flight never exceed the final total
        # and never go backwards (writers only add; every count it saw was a
        # committed state, never a torn one).
        for counts in per_reader_counts:
            assert all(c <= 300 for c in counts)
            assert counts == sorted(counts)


# ── 2. Atomic audited mutations ───────────────────────────

class TestAtomicAudit:
    def _boom(self, *a, **k):
        raise RuntimeError("audit boom")

    def test_add_node_rolls_back_on_audit_failure(self, store, monkeypatch):
        node = _fact("doomed")
        monkeypatch.setattr(store, "record_audit", self._boom)
        with pytest.raises(RuntimeError, match="audit boom"):
            store.add_node(node)
        monkeypatch.undo()
        assert store.get_node(node.node_id) is None, "mutation must roll back"
        # Store stays usable after the failure.
        kept = store.add_node(_fact("kept"))
        assert store.get_node(kept.node_id) is not None

    def test_update_node_rolls_back_on_audit_failure(self, store, monkeypatch):
        node = store.add_node(_fact("stable", content="original"))
        monkeypatch.setattr(store, "record_audit", self._boom)
        with pytest.raises(RuntimeError, match="audit boom"):
            store.update_node(node.node_id, content="mutated")
        monkeypatch.undo()
        assert store.get_node(node.node_id).content == "original"

    def test_set_node_validity_rolls_back_on_audit_failure(
        self, store, monkeypatch
    ):
        from datetime import datetime, timezone
        node = store.add_node(_fact("windowed"))
        monkeypatch.setattr(store, "record_audit", self._boom)
        with pytest.raises(RuntimeError, match="audit boom"):
            store.set_node_validity(
                node.node_id, valid_until=datetime.now(timezone.utc)
            )
        monkeypatch.undo()
        assert store.get_node(node.node_id).valid_until is None

    def test_mutation_and_audit_share_one_commit(self, store):
        # A crash between "mutation committed" and "audit committed" was the
        # defect. With one commit there is no such window: after add_node the
        # row AND its audit are both visible on a second connection.
        node = store.add_node(_fact("paired"))
        other = sqlite3.connect(store.db_path)
        try:
            n = other.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_id=?", (node.node_id,)
            ).fetchone()[0]
            a = other.execute(
                "SELECT COUNT(*) FROM audit_log WHERE node_id=? AND op='create'",
                (node.node_id,),
            ).fetchone()[0]
        finally:
            other.close()
        assert (n, a) == (1, 1)


# ── 3. Loud migrations ────────────────────────────────────

class _FailingConn:
    """Fake connection whose execute raises a chosen OperationalError."""

    def __init__(self, message):
        self._message = message

    def execute(self, *a, **k):
        raise sqlite3.OperationalError(self._message)

    def commit(self):
        pass


class TestLoudMigrations:
    def test_duplicate_column_still_swallowed(self, store, capsys):
        # The idempotency race stays a note, not a crash.
        store._migrate_add_validity_columns(
            _FailingConn("duplicate column name: valid_from")
        )
        assert "Migration note" in capsys.readouterr().out

    def test_real_errors_raise(self, store):
        # Locked db / disk / corruption must be loud at startup.
        for helper in (
            store._migrate_add_confidence_columns,
            store._migrate_add_modality_columns,
            store._migrate_add_temporal_columns,
            store._migrate_add_validity_columns,
        ):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                helper(_FailingConn("database is locked"))


# ── 4. Transactional import with modes ────────────────────

class TestImportModes:
    def test_empty_db_default_ok(self, store):
        g = _graph(3)
        result = store.import_graph(g)
        assert result["nodes_added"] == 3
        assert result["edges_added"] == 1
        assert store.count_nodes() == 3
        # Imported rows carry their audits (same transaction).
        assert [e["op"] for e in store.get_node_audit(g.nodes[0].node_id)] == ["create"]

    def test_refuse_leaves_db_byte_stable(self, store):
        store.add_node(_fact("resident"))
        # exported_at is stamped at export time; compare the stored content.
        before = store.export_graph().model_dump_json(exclude={"exported_at"})
        with pytest.raises(ImportRefusedError, match="not empty"):
            store.import_graph(_graph(2))
        after = store.export_graph().model_dump_json(exclude={"exported_at"})
        assert after == before

    def test_merge_skips_collisions_never_mutates(self, store):
        resident = store.add_node(_fact("resident", content="original content"))
        # Payload collides on the resident's id with DIFFERENT content.
        collider = _fact("resident", content="imposter content",
                         node_id=resident.node_id)
        newcomer = _fact("newcomer")
        g = Graph(nodes=[collider, newcomer], edges=[Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=newcomer.node_id,
            target_node_id=resident.node_id,  # edge onto existing memory: valid
        )])
        result = store.import_graph(g, mode="merge")
        assert result["nodes_added"] == 1
        assert result["skipped_nodes"] == 1
        assert result["edges_added"] == 1
        assert store.get_node(resident.node_id).content == "original content"
        assert store.get_node(newcomer.node_id) is not None

    def test_replace_swaps_graph(self, store):
        old = store.add_node(_fact("old resident"))
        g = _graph(2)
        result = store.import_graph(g, mode="replace")
        assert result["removed_nodes"] == 1
        assert result["nodes_added"] == 2
        assert store.get_node(old.node_id) is None
        assert store.count_nodes() == 2

    def test_replace_failure_rolls_back_to_original(self, store, monkeypatch):
        # Seed a real graph: nodes, an edge, and extra audit history — ALL of
        # it must survive a mid-import failure, not just the node rows.
        original = store.add_node(_fact("survivor", content="must survive"))
        second = store.add_node(_fact("survivor 2"))
        edge = store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=original.node_id,
            target_node_id=second.node_id,
        ))
        store.update_node(original.node_id, content="must survive")  # audit row
        conn = store._get_conn()
        audits_before = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

        def boom(e):
            raise RuntimeError("mid-import failure")
        monkeypatch.setattr(store, "add_edge", boom)

        with pytest.raises(RuntimeError, match="mid-import failure"):
            store.import_graph(_graph(3, with_edge=True), mode="replace")
        monkeypatch.undo()

        # ONE transaction: the wipe rolled back with the failed insert —
        # nodes AND edges AND audit rows all read back as before.
        assert store.count_nodes() == 2
        assert store.count_edges() == 1
        assert store.get_node(original.node_id).content == "must survive"
        assert store.get_edge(edge.edge_id) is not None
        audits_after = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        assert audits_after == audits_before, \
            "imported rows' audits must roll back with the import"

    def test_dangling_edge_fails_validation_db_untouched(self, store):
        keeper = store.add_node(_fact("keeper"))
        n = _fact("payload node")
        bad = Graph(nodes=[n], edges=[Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=n.node_id,
            target_node_id="no-such-node",
        )])
        with pytest.raises(ImportValidationError, match="unknown node IDs"):
            store.import_graph(bad, mode="replace")
        assert store.count_nodes() == 1
        assert store.get_node(keeper.node_id) is not None

    def test_duplicate_payload_ids_fail_validation(self, store):
        n = _fact("twin")
        dup = Graph(nodes=[n, _fact("twin two", node_id=n.node_id)], edges=[])
        with pytest.raises(ImportValidationError, match="duplicate node IDs"):
            store.import_graph(dup)
        assert store.count_nodes() == 0

    def test_bad_mode_rejected(self, store):
        with pytest.raises(ValueError, match="mode must be"):
            store.import_graph(_graph(1, with_edge=False), mode="overwrite")

    def test_clear_existing_compat_true_is_replace(self, store):
        store.add_node(_fact("old"))
        with pytest.warns(DeprecationWarning, match="clear_existing"):
            store.import_graph(_graph(2), clear_existing=True)
        assert store.count_nodes() == 2

    def test_clear_existing_compat_false_is_merge(self, store):
        resident = store.add_node(_fact("resident"))
        with pytest.warns(DeprecationWarning, match="clear_existing"):
            result = store.import_graph(_graph(2), clear_existing=False)
        assert result["nodes_added"] == 2
        assert store.get_node(resident.node_id) is not None
        assert store.count_nodes() == 3


# ── 5. Semantic index after import ────────────────────────

class TestImportSemantic:
    def _searchable_graph(self):
        # Labels drawn from _MockEmbedder's vocab so search has signal.
        a = _fact("dog", content="The dog park routine.")
        b = _fact("sunset", content="Sunset photo essay idea.")
        return Graph(nodes=[a, b], edges=[]), a, b

    def test_replace_drops_stale_vectors_and_queues_imports(self, store):
        sem = _QueueTestIndex(store)
        stale = store.add_node(_fact("bread", content="Bread starter notes."))
        sem.index_node(stale.node_id, stale.label, stale.content)
        assert stale.node_id in sem._vectors

        g, a, _b = self._searchable_graph()
        result = store.import_graph(g, mode="replace", semantic=sem)
        # Import does only cheap SQL under the lock: stale vector dropped
        # inline, imported nodes QUEUED — no embedding ran yet.
        assert result["semantic_rebuild"] == "queued (2 nodes pending embed)"
        assert stale.node_id not in sem._vectors
        assert sem.pending_count() == 2

        # search() drains the queue first, so the imported nodes are
        # searchable by the very next recall.
        hits = [nid for nid, _s in sem.search("dog")]
        assert a.node_id in hits
        assert set(sem._vectors) == {n.node_id for n in g.nodes}
        assert sem.pending_count() == 0

    def test_merge_queues_only_new_nodes(self, store):
        resident = store.add_node(_fact("revenue", content="Q3 revenue claim."))
        sem = _QueueTestIndex(store)
        g, a, b = self._searchable_graph()
        result = store.import_graph(g, mode="merge", semantic=sem)
        assert result["semantic_rebuild"] == "queued (2 nodes pending embed)"
        sem.drain_pending()
        assert set(sem._vectors) == {a.node_id, b.node_id}
        assert resident.node_id not in sem._vectors

    def test_no_index_reported_honestly(self, store):
        result = store.import_graph(_graph(1, with_edge=False))
        assert result["semantic_rebuild"] == "skipped (disabled)"

    def test_layer_breaking_mid_import_reported(self, store, monkeypatch):
        # If remove_node/defer_nodes self-disable the layer mid-operation,
        # the status must say FAILED — "queued (0 ...)" would read as a
        # clean no-op while recall silently degrades.
        sem = _QueueTestIndex(store)

        def broken_defer(nodes):
            sem._broken = True
            sem._broken_reason = "OperationalError('pending table corrupt')"
            return 0
        monkeypatch.setattr(sem, "defer_nodes", broken_defer)

        result = store.import_graph(_graph(2), semantic=sem)
        assert result["semantic_rebuild"].startswith(
            "failed (semantic layer self-disabled"
        )
        assert "pending table corrupt" in result["semantic_rebuild"]


# ── 6. REST surface ───────────────────────────────────────

class TestImportEndpoint:
    @pytest.fixture
    def client(self, tmp_dir):
        from revien.daemon.server import create_app
        app = create_app(db_path=str(tmp_dir / "api.db"))
        with TestClient(app) as c:
            c.app = app
            yield c

    def _seed(self, client, label="resident"):
        node = _fact(label)
        client.app.state.store.add_node(node)
        return node

    def _payload(self, n=2):
        return json.loads(_graph(n).model_dump_json())

    def test_default_refuses_nonempty_409(self, client):
        self._seed(client)
        resp = client.post("/v1/graph/import", json=self._payload())
        assert resp.status_code == 409
        assert client.app.state.store.count_nodes() == 1

    def test_merge_mode(self, client):
        resident = self._seed(client)
        resp = client.post("/v1/graph/import?mode=merge", json=self._payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["nodes_added"] == 2
        assert client.app.state.store.get_node(resident.node_id) is not None

    def test_replace_mode(self, client):
        resident = self._seed(client)
        resp = client.post("/v1/graph/import?mode=replace", json=self._payload())
        assert resp.status_code == 200
        assert resp.json()["removed_nodes"] == 1
        assert client.app.state.store.get_node(resident.node_id) is None

    def test_invalid_mode_400(self, client):
        resp = client.post("/v1/graph/import?mode=overwrite", json=self._payload())
        assert resp.status_code == 400

    def test_validation_failure_400(self, client):
        payload = self._payload(1)
        payload["edges"] = [{
            "edge_type": "related_to",
            "source_node_id": payload["nodes"][0]["node_id"],
            "target_node_id": "no-such-node",
        }]
        resp = client.post("/v1/graph/import", json=payload)
        assert resp.status_code == 400
        assert client.app.state.store.count_nodes() == 0


# ── 7. CLI surface ────────────────────────────────────────

class TestImportCli:
    def _run(self, *args):
        from revien.cli import main
        return CliRunner().invoke(main, list(args))

    def _write_graph(self, tmp_dir, n=3):
        g = _graph(n)
        path = tmp_dir / "graph.json"
        path.write_text(g.model_dump_json(indent=2), encoding="utf-8")
        return g, path

    def _occupied_db(self, tmp_dir, name="occupied.db"):
        db = tmp_dir / name
        s = GraphStore(db_path=str(db))
        resident = s.add_node(_fact("resident"))
        s.close()
        return db, resident

    def test_replace_flag(self, fake_home, tmp_dir):
        g, path = self._write_graph(tmp_dir)
        db, resident = self._occupied_db(tmp_dir)
        result = self._run("import", str(path), "--db", str(db), "--replace")
        assert result.exit_code == 0, result.output
        assert "Replaced existing graph" in result.output
        s = GraphStore(db_path=str(db))
        try:
            assert s.get_node(resident.node_id) is None
            assert s.count_nodes() == 3
        finally:
            s.close()

    def test_merge_and_replace_mutually_exclusive(self, fake_home, tmp_dir):
        _g, path = self._write_graph(tmp_dir)
        result = self._run("import", str(path), "--db",
                           str(tmp_dir / "x.db"), "--merge", "--replace")
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_merge_parity_with_store_mode(self, fake_home, tmp_dir):
        # CLI --merge now runs store mode="merge" — same skip/add semantics
        # the old duplicated loop pinned (see test_onramp for the full pins).
        g, path = self._write_graph(tmp_dir)
        db, resident = self._occupied_db(tmp_dir)
        result = self._run("import", str(path), "--db", str(db), "--merge")
        assert result.exit_code == 0, result.output
        assert "Merged 3 nodes, 1 edges" in result.output
        # Second merge: everything collides, nothing duplicated.
        result = self._run("import", str(path), "--db", str(db), "--merge")
        assert result.exit_code == 0, result.output
        assert "skipped 3 existing nodes" in result.output
        s = GraphStore(db_path=str(db))
        try:
            assert s.count_nodes() == 4
            assert s.get_node(resident.node_id) is not None
        finally:
            s.close()

    def test_refuse_message_unchanged(self, fake_home, tmp_dir):
        _g, path = self._write_graph(tmp_dir)
        db, _resident = self._occupied_db(tmp_dir)
        result = self._run("import", str(path), "--db", str(db))
        assert result.exit_code != 0
        assert "not empty" in result.output
        assert "--merge" in result.output
        assert "--replace" in result.output

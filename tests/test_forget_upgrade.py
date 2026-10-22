"""Invariant 7: databases made by earlier versions upgrade through migration
006 without error and without deleting on a guess (docs/FORGET_SPEC.md §3).

tests/fixtures/*.sql are real databases (see the README there): v0.3.0 (the
tag and commit 7757085), origin/main at 03217f4, origin/main at 03217f4 after
real use, origin/main at b905109 after real use (the first PR), and a
hand-built v0.2.x-shaped schema (the repository history starts at 0.3.0)."""
import importlib
import sqlite3
from pathlib import Path

import pytest

from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore

FIX = Path(__file__).parent / "fixtures"
DATABASES = {
    "v0.2.x-shape": "db_v02x_schema.sql",
    "v0.3.0-tag-7bf8738": "db_v030_tag_7bf8738.sql",
    "v0.3.0-7757085": "db_v030_7757085.sql",
    "main-03217f4": "db_main_03217f4.sql",
    "main-used-03217f4": "db_main_used_03217f4.sql",
    "main-b905109": "db_main_b905109.sql",
}


def made_by(name, tmp_path):
    path = tmp_path / f"{name}.db"
    conn = sqlite3.connect(path)
    conn.executescript((FIX / DATABASES[name]).read_text())
    conn.commit()
    conn.close()
    return str(path)


def snapshot(path):
    conn = sqlite3.connect(path)
    try:
        nodes = {r[0]: r[1:] for r in conn.execute(
            "SELECT node_id, node_type, label, content, source_id FROM nodes")}
        edges = {r[0]: r[1:] for r in conn.execute(
            "SELECT edge_id, edge_type, source_node_id, target_node_id FROM edges")}
        audit = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] \
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='audit_log'").fetchone() else 0
        return nodes, edges, audit
    finally:
        conn.close()


def migration():
    return importlib.import_module("revien.graph.migrations.006_forget_layer")


@pytest.mark.parametrize("name", sorted(DATABASES))
class TestUpgrade:
    def test_opens_and_keeps_every_row(self, name, tmp_path):
        path = made_by(name, tmp_path)
        before = snapshot(path)
        GraphStore(db_path=path).close()
        nodes, edges, audit = snapshot(path)
        assert nodes == before[0]            # same ids, same words: nothing deleted or rewritten
        assert edges == before[1]
        assert audit >= before[2]
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()
        assert {"node_provenance", "capture_units", "suppressions"} <= names

    def test_ownership_is_proven_only_where_it_can_be_and_never_guessed(self, name, tmp_path):
        path = made_by(name, tmp_path)
        GraphStore(db_path=path).close()
        s = GraphStore(db_path=path)
        try:
            for n in s.list_nodes(limit=10000):
                rows = s.get_provenance(n.node_id)
                assert rows, f"{n.node_id} has no ownership row at all"
                proven = [r for r in rows if not r["unit_id"].startswith("unproven:")]
                if proven:
                    # proof is an ingest key stamped on a context or skill node
                    assert n.node_type in (NodeType_CONTEXT, NodeType_SKILL)
                    assert (n.metadata or {}).get("ingest_key")
                    assert all(r["label_ok"] and r["wording_ok"] for r in proven)
                else:
                    assert all(not r["label_ok"] and not r["wording_ok"] for r in rows)
        finally:
            s.close()

    def test_a_forget_never_deletes_an_unproven_node_it_was_not_pointed_at(self, name, tmp_path):
        path = made_by(name, tmp_path)
        s = GraphStore(db_path=path)
        try:
            all_nodes = {n.node_id: n for n in s.list_nodes(limit=10000)}
            proven_ctx = [n for n in all_nodes.values()
                          if any(not r["unit_id"].startswith("unproven:")
                                 for r in s.get_provenance(n.node_id))]
            unproven = {nid for nid, n in all_nodes.items() if n not in proven_ctx}
            ops = GraphOperations(s)
            # a conversation's own record is refused: nothing is changed
            before = {n.node_id for n in s.list_nodes(limit=10000)}
            for ctx in proven_ctx:
                res = ops.forget_node(ctx.node_id)
                assert res["status"] == "refused" and res["reason"] == "whole_conversation"
            assert before == {n.node_id for n in s.list_nodes(limit=10000)}
            # forgetting one unproven node names it, and only it, for removal
            if unproven:
                target = sorted(unproven)[0]
                res = ops.forget_node(target)
                assert res["status"] in ("forgotten", "refused")
                after = {n.node_id for n in s.list_nodes(limit=10000)}
                assert (before - after) <= {target}
        finally:
            s.close()

    def test_forgetting_an_unproven_node_by_name_removes_only_it(self, name, tmp_path):
        path = made_by(name, tmp_path)
        s = GraphStore(db_path=path)
        try:
            unproven = [n for n in s.list_nodes(limit=10000)
                        if all(r["unit_id"].startswith("unproven:")
                               for r in s.get_provenance(n.node_id))]
            if not unproven:
                pytest.skip("no unproven node in this database")
            target = unproven[0]
            before = {n.node_id for n in s.list_nodes(limit=10000)}
            res = GraphOperations(s).forget_node(target.node_id)
            assert res["status"] in ("forgotten", "refused")
            after = {n.node_id for n in s.list_nodes(limit=10000)}
            if res["status"] == "forgotten":
                assert before - after == {target.node_id}
            else:
                assert before == after
        finally:
            s.close()

    def test_the_standalone_migration_is_idempotent(self, name, tmp_path):
        path = made_by(name, tmp_path)
        first = migration().migrate(path)
        assert {"node_provenance", "capture_units", "suppressions"} <= set(first["created"]) \
            or first["created"] == []        # the connect-time chain may have run first
        again = migration().migrate(path)
        assert again["created"] == [] and again["backfilled"] == 0
        conn = sqlite3.connect(path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
        finally:
            conn.close()

    def test_a_second_open_changes_nothing(self, name, tmp_path):
        path = made_by(name, tmp_path)
        GraphStore(db_path=path).close()

        def rows():
            c = sqlite3.connect(path)
            try:
                return (c.execute("SELECT * FROM node_provenance ORDER BY 1,2").fetchall(),
                        c.execute("SELECT * FROM capture_units ORDER BY 1").fetchall())
            finally:
                c.close()

        first = rows()
        GraphStore(db_path=path).close()
        assert rows() == first


def test_the_key_is_made_by_the_migration_and_lives_outside_the_database(tmp_path):
    from revien.forget import keys
    path = made_by("v0.3.0-tag-7bf8738", tmp_path)
    assert not keys.key_path().exists()
    migration().migrate(path)
    assert keys.key_path().exists() and len(keys.key_path().read_text().strip()) == 64
    assert keys.key_path().read_text().strip().encode() not in open(path, "rb").read()


from revien.graph.schema import NodeType  # noqa: E402

NodeType_CONTEXT = NodeType.CONTEXT
NodeType_SKILL = NodeType.SKILL

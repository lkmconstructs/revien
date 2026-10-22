"""Codex review of #6: contradicting identities, the identity a graph import
stores, Unicode case in the --everywhere prefilter, and restoring by unit id."""
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.forget import gate
from revien.graph.operations import GraphOperations
from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from tests import forget_seed as fs
from tests.test_forget_narrow import (LINES, NAME, contexts_holding, holders_of,
                                      nodes, sync, write_session)


def forget_name(db):
    s = GraphStore(db_path=db)
    ent = [n for n in s.list_nodes(limit=100)
           if n.node_type == NodeType.ENTITY and n.label == NAME][0]
    res = GraphOperations(s).forget_node(ent.node_id)
    s.close()
    assert res["status"] == "forgotten"
    return res


# ── 1. a different identity at the same path is a new origin ─────────────

class TestContradictingIdentitiesVetoAPathMatch:
    def test_a_claude_session_replaced_by_another_session_id_lands(self, tmp_path):
        root = tmp_path / "claude"
        db = str(tmp_path / "c.db")
        write_session(root, "proj-a", "s1", "uuid-OLD")
        sync(db, root)
        forget_name(db)
        assert not holders_of(db, NAME)
        # the same path, a different session inside it
        write_session(root, "proj-a", "s1", "uuid-NEW", LINES + [
            ("user", "A later remark about the parking permits.")])
        sync(db, root)
        assert contexts_holding(db, NAME), "another session at the old path is another origin"

    def test_the_same_session_id_in_a_rewritten_file_stays_out(self, tmp_path):
        root = tmp_path / "claude"
        db = str(tmp_path / "c2.db")
        write_session(root, "proj-a", "s1", "uuid-OLD")
        sync(db, root)
        forget_name(db)
        write_session(root, "proj-a", "s1", "uuid-OLD", LINES + [
            ("user", "A later remark about the parking permits.")])
        sync(db, root)
        assert not holders_of(db, NAME)

    def test_a_codex_rollout_replaced_by_another_sid_lands(self, tmp_path, monkeypatch):
        seeded = fs.seed("codex", tmp_path, monkeypatch)
        fs.forget_all(seeded.db, seeded.secret)
        lines = [json.dumps({"timestamp": "2026-07-01T11:00:00Z", "type": "session_meta",
                             "payload": {"id": "0197-codex-another", "cwd": "/work/proj-c"}})]
        for role, kind, text in (
                ("user", "input_text", "remember that the parking permit is blue."),
                ("assistant", "output_text",
                 f"Noted. We decided to use {seeded.secret} as the codename."),
                ("user", "input_text", "A later remark about the parking permits.")):
            lines.append(json.dumps({"timestamp": "2026-07-01T11:00:01Z", "type": "response_item",
                                     "payload": {"type": "message", "role": role,
                                                 "content": [{"type": kind, "text": text}]}}))
        seeded.source_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fs._age(seeded.source_file)
        seeded.recapture()
        blob = " ".join(n.content for n in nodes(seeded.db))
        assert seeded.secret in blob

    def test_the_same_codex_sid_in_a_renamed_rollout_stays_out(self, tmp_path, monkeypatch):
        seeded = fs.seed("codex", tmp_path, monkeypatch)
        fs.forget_all(seeded.db, seeded.secret)
        copy = seeded.source_file.parent / "rollout-renamed.jsonl"
        shutil.copy(seeded.source_file, copy)
        fs._age(copy)
        seeded.recapture()
        assert seeded.secret not in " ".join(n.content for n in nodes(seeded.db))

    def test_a_different_inode_at_the_same_path_is_vetoed_and_the_same_inode_binds(self, tmp_path):
        from revien.graph.units import identity_keys
        f = tmp_path / "a.md"
        f.write_text("x")
        keys = identity_keys(str(f))
        path_key = [k for k in keys if k.startswith("path:")][0]
        inode_key = [k for k in keys if k.startswith("inode:")]
        if not inode_key:
            pytest.skip("no inode on this platform")
        s = GraphStore(db_path=str(tmp_path / "i.db"))
        s.add_suppression("forget:i", [path_key, "inode:1:99999999"], [], ["a" * 64], [], "t")
        s.add_suppression("forget:j", [path_key, inode_key[0]], [], ["b" * 64], [], "t")
        long_fps = gate.known(s, gate.UnitRef(None, str(f)))[0]
        s.close()
        assert long_fps == {"b" * 64}


# ── 2. a graph import stores the canonical session identity ──────────────

def test_a_graph_import_of_a_claude_session_binds_a_forget_to_the_session_id(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    seeded = fs.seed("claude_code", tmp_path / "src", monkeypatch)
    s = GraphStore(db_path=seeded.db)
    graph = s.export_graph()
    s.close()
    db = str(tmp_path / "dst.db")
    d = GraphStore(db_path=db)
    d.import_graph(graph, mode="merge")
    target = [n for n in d.list_nodes(limit=100)
              if n.node_type == NodeType.DECISION and seeded.secret in n.content][0]
    res = GraphOperations(d).forget_node(target.node_id)
    d.close()
    assert [b["unit_id"] for b in res["bound_to"]] == ["session:claude-code:uuid-claude-1"], res["bound_to"]
    s = GraphStore(db_path=db)
    try:
        for n in s.list_nodes(limit=100):
            for r in s.get_provenance(n.node_id):
                assert "sess-1" not in r["unit_id"], r     # no filename-derived unit id
    finally:
        s.close()


# ── 3. Unicode case in the --everywhere prefilter ────────────────────────

class TestEverywhereFindsHoldersWhateverTheirCase:
    def _run(self, tmp_path, wording, holder_text):
        s = GraphStore(db_path=str(tmp_path / "u.db"))
        target = s.add_node(Node(node_type=NodeType.FACT, label="t", content=wording),
                            owner=("session:claude-code:A", 1, 1))
        holder = s.add_node(Node(node_type=NodeType.FACT, label="h", content=holder_text))
        res = GraphOperations(s).forget_node(target.node_id, everywhere=True)
        after = s.get_node(holder.node_id)
        s.close()
        return res, after

    def test_a_capital_accented_letter_matches_its_lowercase_form(self, tmp_path):
        res, after = self._run(tmp_path, "Élodie signed the lease.",
                               "ÉLODIE SIGNED THE LEASE. The key is under the mat.")
        assert after is not None and "ÉLODIE" not in after.content
        assert "The key is under the mat." in after.content
        assert after.node_id in {c["node_id"] for c in res["cut_in_place"]}

    def test_eszett_and_ss_are_not_the_same_wording(self, tmp_path):
        """Case-insensitive means simple case folding (the same rule the cut
        itself uses): 'Straße' and 'STRASSE' are different words."""
        res, after = self._run(tmp_path, "Our office is on Straße Nord.",
                               "Our office is on STRASSE NORD. Parking is free.")
        assert after.content.startswith("Our office is on STRASSE NORD")
        assert res["cut_in_place"] == []
        wanted = gate._split_windows(gate.window_fps_for(["Straße Nord"]))
        assert not gate.holds_wording("the STRASSE NORD office", wanted)
        assert gate.holds_wording("the STRAßE nord office", wanted)


# ── 4. restore by an exact unit id lifts every record bound to it ────────

def test_restore_by_a_unit_id_lifts_every_record_bound_to_it(tmp_path):
    db = str(tmp_path / "r.db")
    s = GraphStore(db_path=db)
    unit = "item:chatgpt:conversation:X"
    s.add_suppression("forget:one", [f"unit:{unit}"], [], ["a" * 64], [], "t")
    s.add_suppression("forget:two", [f"unit:{unit}"], [], ["b" * 64], [], "t")
    s.add_suppression("forget:other", ["unit:item:chatgpt:conversation:Y"], [], ["c" * 64], [], "t")
    s.close()
    out = CliRunner().invoke(main, ["restore", unit, "--db", db])
    assert out.exit_code == 0, out.output
    assert "Will lift 2 forget record(s)" in out.output
    s = GraphStore(db_path=db)
    try:
        assert [r["unit_id"] for r in s.list_suppressions("active")] == ["forget:other"]
    finally:
        s.close()


def test_a_bare_name_that_matches_several_records_is_still_ambiguous(tmp_path):
    from revien.forget.restore import plan_restore
    s = GraphStore(db_path=str(tmp_path / "r2.db"))
    s.add_suppression("forget:one", ["unit:file:/a/notes.md"], [], ["a" * 64], [], "t")
    s.add_suppression("forget:two", ["unit:file:/b/notes.md"], [], ["b" * 64], [], "t")
    plan = plan_restore(s, "notes.md")
    s.close()
    assert plan["status"] == "ambiguous" and len(plan["candidates"]) == 2

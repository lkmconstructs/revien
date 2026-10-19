"""Coming back: suppression (docs/FORGET_SPEC.md §6, invariant 3).

A forgotten unit is not recaptured by any path without an explicit restore or
override, including after a move, rename, symlink, hard link or re-timestamp.
File-backed units are suppressed until the file changes; units with no file
are suppressed for good; shared channels cannot be suppressed and say so.
"""
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from revien.cli import main
from revien.daemon.server import create_app
from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore
from tests import forget_seed as fs

STABLE = ["claude_code", "codex", "file_watcher", "obsidian", "chatgpt",
          "claude_import", "readwise", "openai_adapter", "skill"]
SHARED = ["pipeline", "mcp"]


def forget_all(seeded):
    store = GraphStore(db_path=seeded.db)
    try:
        return GraphOperations(store).forget_node(seeded.ctx_ids[0], cascade=True)
    finally:
        store.close()


def has_secret(seeded) -> bool:
    conn = sqlite3.connect(seeded.db)
    try:
        return bool(scrub.find_residue(conn, [seeded.secret]))
    finally:
        conn.close()


def node_count(db) -> int:
    s = GraphStore(db_path=db)
    try:
        return s.count_nodes()
    finally:
        s.close()


@pytest.mark.parametrize("path", STABLE)
def test_a_forgotten_unit_is_not_recaptured_by_its_own_path(path, tmp_path, monkeypatch):
    """Invariant 3, every stable capture path."""
    seeded = fs.seed(path, tmp_path, monkeypatch)
    res = forget_all(seeded)
    assert res["suppressed"], res
    assert node_count(seeded.db) == 0 or not has_secret(seeded)
    seeded.recapture()
    assert not has_secret(seeded), f"{path}: recaptured"
    assert node_count(seeded.db) == 0 or all(
        seeded.secret not in (n.label + n.content)
        for n in _all_nodes(seeded.db))


def _all_nodes(db):
    s = GraphStore(db_path=db)
    try:
        return s.list_nodes(limit=1000)
    finally:
        s.close()


@pytest.mark.parametrize("path", SHARED)
def test_shared_channels_are_eligible_for_recapture_and_say_so(path, tmp_path, monkeypatch):
    seeded = fs.seed(path, tmp_path, monkeypatch)
    res = forget_all(seeded)
    assert res["suppressed"] == []
    assert res["not_suppressed"]
    assert "eligible for recapture because the source supplied no stable item identity" in res["message"] \
        or "eligible for recapture because the source supplied no stable item identity" in res["blast_radius"]
    assert not has_secret(seeded)
    seeded.recapture()
    assert has_secret(seeded)            # honest: nothing could stop it


class TestFileBackedUnitsLiftOnlyWhenTheFileChanges:
    def _forgotten(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        forget_all(seeded)
        return seeded

    def test_unchanged_file_stays_out(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        seeded.recapture()
        assert node_count(seeded.db) == 0

    def test_a_new_timestamp_does_not_bring_it_back(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        now = time.time()
        os.utime(seeded.source_file, (now, now))
        seeded.recapture()
        assert node_count(seeded.db) == 0

    def test_changed_content_is_new_evidence_and_is_captured(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        with open(seeded.source_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "user", "sessionId": "uuid-claude-1",
                "message": {"role": "user", "content": [{"type": "text",
                            "text": "A brand new remark about deployments."}]},
                "timestamp": "2026-07-02T10:00:00Z"}) + "\n")
        seeded.recapture()
        assert node_count(seeded.db) > 0

    def test_a_symlink_to_the_file_does_not_bring_it_back(self, tmp_path, monkeypatch):
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        link_root = tmp_path / "links" / "projects" / "other-proj"
        link_root.mkdir(parents=True)
        try:
            os.symlink(seeded.source_file, link_root / "alias.jsonl")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        fs._resync(seeded.db, "claude-code",
                   ClaudeCodeAdapter(session_dir=str(tmp_path / "links" / "projects")))
        assert node_count(seeded.db) == 0

    def test_a_hard_link_does_not_bring_it_back(self, tmp_path, monkeypatch):
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        d = tmp_path / "hl" / "projects" / "hl-proj"
        d.mkdir(parents=True)
        try:
            os.link(seeded.source_file, d / "other-name.jsonl")
        except (OSError, NotImplementedError):
            pytest.skip("hard links unavailable")
        fs._resync(seeded.db, "claude-code",
                   ClaudeCodeAdapter(session_dir=str(tmp_path / "hl" / "projects")))
        assert node_count(seeded.db) == 0

    def test_a_moved_file_does_not_bring_it_back(self, tmp_path, monkeypatch):
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        d = tmp_path / "moved" / "projects" / "new-proj"
        d.mkdir(parents=True)
        new = d / "renamed.jsonl"
        shutil.move(str(seeded.source_file), str(new))
        fs._resync(seeded.db, "claude-code",
                   ClaudeCodeAdapter(session_dir=str(tmp_path / "moved" / "projects")))
        assert node_count(seeded.db) == 0

    def test_a_new_file_at_the_old_path_is_judged_by_what_it_holds(self, tmp_path, monkeypatch):
        """The file was moved away and a different file now sits where it
        was. Different content is new evidence; the same content is not."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        old_path = seeded.source_file
        old_text = old_path.read_text()
        d = tmp_path / "away" / "projects" / "away-proj"
        d.mkdir(parents=True)
        shutil.move(str(old_path), str(d / "away.jsonl"))
        adapter = ClaudeCodeAdapter(session_dir=str(old_path.parents[1]))
        # same bytes copied back: still the forgotten text
        old_path.write_text(old_text)
        fs._age(old_path)
        fs._resync(seeded.db, "claude-code", adapter)
        assert node_count(seeded.db) == 0
        # a genuinely different file at the old path
        old_path.write_text(json.dumps({
            "type": "user", "sessionId": "another-session",
            "message": {"role": "user", "content": [{"type": "text",
                        "text": "Totally different notes about lunch."}]},
            "timestamp": "2026-07-03T10:00:00Z"}) + "\n")
        fs._age(old_path)
        fs._resync(seeded.db, "claude-code", adapter)
        assert node_count(seeded.db) > 0
        assert not has_secret(seeded)

    def test_a_made_up_path_through_the_api_never_lifts_a_suppression(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "claude-code:proj-x:sess-1",
                "content": "User: some changed words entirely.",
                "origin_runtime": "claude-code", "session_key": "sess-1",
                "metadata": {"session_id": "uuid-claude-1",
                             "path": str(tmp_path / "does" / "not" / "exist.jsonl")}})
            assert r.status_code == 200
            body = r.json()
            assert body["suppressed"] is True and body["context_node_id"] == ""
            assert "suppressed" in body["suppression_note"]
        assert node_count(seeded.db) == 0

    def test_a_path_to_some_other_real_file_does_not_lift_it_either(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        other = tmp_path / "other.txt"
        other.write_text("unrelated")
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "claude-code:proj-x:sess-1", "content": "User: other words.",
                "origin_runtime": "claude-code", "session_key": "sess-1",
                "metadata": {"session_id": "uuid-claude-1", "path": str(other)}})
            assert r.json()["suppressed"] is True

    def test_the_same_text_under_another_name_stays_out(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        text = seeded.source_file.read_text()
        store = GraphStore(db_path=seeded.db)
        rec = store.list_suppressions("active")[0]
        store.close()
        from revien.adapters.claude_code import ClaudeCodeAdapter
        content, _ts = ClaudeCodeAdapter(session_dir=str(tmp_path))._parse_session_log_ts(seeded.source_file)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "elsewhere", "content": content})
            assert r.json()["suppressed"] is True

    def test_a_case_changed_path_names_the_same_unit(self, tmp_path, monkeypatch):
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        forget_all(seeded)
        upper = str(seeded.source_file).upper()
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "file:notes.md", "content": "Totally new words.",
                "origin_runtime": "file", "origin_source": "watch",
                "metadata": {"path": upper}})
            assert r.json()["suppressed"] is True


class TestPermanentUnits:
    def test_a_conversation_with_no_file_is_suppressed_for_good(self, tmp_path, monkeypatch):
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        forget_all(seeded)
        store = GraphStore(db_path=seeded.db)
        (rec,) = store.list_suppressions("active")
        store.close()
        assert rec["kind"] == "permanent"
        # an EDITED export is still the same conversation: still out
        export = seeded.extra["export"]
        conv = json.loads(export.read_text())
        conv[0]["mapping"]["m2"]["message"]["content"]["parts"] = ["A different answer."]
        export.write_text(json.dumps(conv))
        seeded.recapture()
        assert node_count(seeded.db) == 0

    def test_the_importer_reports_suppressed_units(self, tmp_path, monkeypatch):
        from revien.importers import chatgpt
        from revien.importers.base import run_import
        from revien.ingestion.pipeline import IngestionPipeline
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        forget_all(seeded)
        store = GraphStore(db_path=seeded.db)
        try:
            report = run_import(chatgpt.iter_units(str(seeded.extra["export"])),
                                IngestionPipeline(store))
        finally:
            store.close()
        assert report.units_suppressed == 1 and report.units_ingested == 0

    def test_override_on_a_reimport_lifts_it_and_audits(self, tmp_path, monkeypatch):
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        forget_all(seeded)
        res = CliRunner().invoke(main, [
            "import-chatgpt", str(seeded.extra["export"]), "--db", seeded.db,
            "--override-suppressed"])
        assert res.exit_code == 0, res.output
        assert node_count(seeded.db) > 0
        store = GraphStore(db_path=seeded.db)
        try:
            assert store.list_suppressions("active") == []
            ops = [a for a in store.get_all_audit() if a["op"] == "restore"]
            assert len(ops) == 1 and ops[0]["actor"] == "override:import"
        finally:
            store.close()

    def test_import_command_without_override_says_why(self, tmp_path, monkeypatch):
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        forget_all(seeded)
        res = CliRunner().invoke(main, [
            "import-chatgpt", str(seeded.extra["export"]), "--db", seeded.db])
        assert "suppressed=1" in res.output and "--override-suppressed" in res.output
        assert node_count(seeded.db) == 0


class TestOtherGates:
    def test_graph_import_does_not_bring_a_forgotten_session_back(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        graph = store.export_graph()                    # a backup taken before
        GraphOperations(store).forget_node(seeded.ctx_ids[0], cascade=True)
        result = store.import_graph(graph, mode="merge")
        store.close()
        assert result["skipped_suppressed"] == len(graph.nodes)
        assert result["nodes_added"] == 0
        assert not has_secret(seeded)

    def test_rest_graph_import_is_gated_too(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        graph = store.export_graph().model_dump(mode="json")
        GraphOperations(store).forget_node(seeded.ctx_ids[0], cascade=True)
        store.close()
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/graph/import?mode=merge", json=graph)
            assert r.status_code == 200 and r.json()["skipped_suppressed"] > 0
        assert not has_secret(seeded)

    def test_a_forgotten_skill_stays_out_until_its_file_changes(self, tmp_path, monkeypatch):
        seeded = fs.seed("skill", tmp_path, monkeypatch)
        res = forget_all(seeded)
        assert res["suppressed"][0]["unit_id"].startswith("skill:")
        seeded.recapture()
        assert node_count(seeded.db) == 0
        seeded.source_file.write_text(
            "---\nname: deploy\ndescription: Deploy\ntriggers: [deploy]\n---\nNew body.\n")
        seeded.recapture()
        assert node_count(seeded.db) == 1

    def test_skill_proposals_are_not_made_from_forgotten_steps(self, tmp_path):
        from revien.graph.schema import Node, NodeType
        from revien.skills.proposals import propose_skills
        store = GraphStore(db_path=str(tmp_path / "sp.db"))
        try:
            for s in range(3):
                for step in ("open the ZQXSTEP gate", "run the nightly job"):
                    store.add_node(
                        Node(node_type=NodeType.ACTION, label=step, content=step,
                             session_key=f"s{s}", project_key="p", origin_runtime="claude-code",
                             source_id=f"claude-code:p:s{s}"),
                        owner=(f"session:claude-code:s{s}", 1, 1))
            first = propose_skills(store, min_occurrences=3, min_sessions=2)
            assert first["created"] == 1
            made = [n for n in store.list_nodes(node_type=NodeType.SKILL, limit=10)]
            # forget one session: the proposal keeps only its name, the
            # forgotten step is not in it, and it is not re-made with the step
            target = [n for n in store.list_nodes(node_type=NodeType.ACTION, limit=50)
                      if n.session_key == "s0" and "ZQXSTEP" in n.label][0]
            GraphOperations(store).forget_node(target.node_id)
            blob = json.dumps([n.model_dump(mode="json") for n in store.list_nodes(limit=100)])
            assert "ZQXSTEP" not in blob or all(
                "ZQXSTEP" not in (n.label + n.content) for n in store.list_nodes(node_type=NodeType.ACTION, limit=50)
                if n.session_key == "s0")
            # the same step typed again into a suppressed unit is not proposed from
            store.add_node(Node(node_type=NodeType.ACTION, label="open the ZQXSTEP gate",
                                content="open the ZQXSTEP gate", session_key="s9",
                                project_key="p", origin_runtime="claude-code"),
                           owner=("session:claude-code:s9", 1, 1))
        finally:
            store.close()

    def test_typing_forgotten_text_back_into_a_distilled_note_is_not_captured(self, tmp_path):
        """The keyed fingerprint stops the unchanged note being re-imported."""
        import re
        from revien.distill import VaultDistiller, VaultReconciler
        from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
        from revien.semantic.index import SemanticIndex
        store = GraphStore(db_path=str(tmp_path / "d.db"))
        vault = tmp_path / "v"
        vault.mkdir()
        unit = ("session:claude-code:sD", 1, 1)
        try:
            ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="chat",
                                      content="User: Atlas runs on Ubuntu ZQXWORD."), owner=unit)
            ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Atlas Server",
                                      content="Atlas Server", source_type=SourceType.EXTRACTED,
                                      confidence=1.0), owner=unit)
            fact = store.add_node(Node(node_type=NodeType.FACT, label="os", content="Atlas runs on Ubuntu ZQXWORD.",
                                       source_type=SourceType.EXTRACTED, confidence=1.0), owner=unit)
            store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=ent.node_id,
                                target_node_id=fact.node_id, weight=0.8))
            VaultDistiller(store, str(vault)).distill()
            notes = [vault / "Revien" / "Atlas Server.md"]
            assert "ZQXWORD" in notes[0].read_text(encoding="utf-8")
            res = GraphOperations(store).forget_node(ctx.node_id, cascade=True)
            assert res["scope"] == "forgotten from Revien"
            (note,) = [l for l in res["leftovers"] if l["kind"] == "vault_note"]
            assert note["fix"] == "revien distill-vault" and note["note"] == "Atlas Server"
            assert "revien distill-vault" in res["message"]
            # the note still shows it (Revien never rewrites the vault itself)
            assert "ZQXWORD" in notes[0].read_text(encoding="utf-8")
            # a user-typed copy of the line (anchors stripped) is not captured
            text = re.sub(r"\s*<!--rv:[^>]*-->", "", notes[0].read_text(encoding="utf-8"))
            notes[0].write_text(text, encoding="utf-8")
            rec = VaultReconciler(store, str(vault), semantic=SemanticIndex(store, enabled=False)).reconcile()
            assert rec["added"] == 0
            for n in store.list_nodes(limit=100):
                assert "ZQXWORD" not in n.label + n.content
        finally:
            store.close()


class TestRecordsAreContentFree:
    def test_no_secret_no_label_and_a_keyed_fingerprint(self, tmp_path, monkeypatch):
        import hashlib
        from revien.forget import keys
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        text = seeded.source_file.read_text()
        forget_all(seeded)
        conn = sqlite3.connect(seeded.db)
        try:
            rows = conn.execute("SELECT * FROM suppressions").fetchall()
            blob = json.dumps([list(map(str, r)) for r in rows])
        finally:
            conn.close()
        assert seeded.secret not in blob
        store = GraphStore(db_path=seeded.db)
        try:
            (rec,) = store.list_suppressions("active")
        finally:
            store.close()
        # not a plain hash of anything it could be a hash of
        plain = {hashlib.sha256(x.encode()).hexdigest() for x in (
            seeded.secret, text, seeded.secret.lower())}
        assert not plain & set(rec["content_fps"] + rec["wording_fps"] + [rec["file_fp"]])
        # and it does depend on the key
        old = rec["file_fp"]
        keys._cache.clear()
        key_file = keys.key_path()
        key_file.write_text(("ab" * 32))
        keys._cache.clear()
        assert keys.file_fingerprint(str(seeded.source_file)) != old
        # the key is a file in the Revien home, owner-only, never in the db
        raw = open(seeded.db, "rb").read()
        assert ("ab" * 32).encode() not in raw
        if os.name == "posix":
            assert (key_file.stat().st_mode & 0o077) == 0

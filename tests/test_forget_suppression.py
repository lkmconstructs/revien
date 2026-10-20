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
    # nothing can suppress the SOURCE, but the forgotten sentences are on the
    # denylist: the same capture comes back without them
    assert "denylist" in res["not_suppressed"][0]
    seeded.recapture()
    texts = " ".join(n.content for n in _all_nodes(seeded.db))
    assert f"my safe code is {seeded.secret}" not in texts
    # a different statement about the same thing is not stopped
    s2 = GraphStore(db_path=seeded.db)
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
    out = IngestionPipeline(s2).ingest(IngestionInput(
        source_id="mcp", content=f"Please remember the vault {seeded.secret} too."))
    s2.close()
    assert out.suppressed is False and has_secret(seeded)


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
        # ...but the sentences that were forgotten stay out of it
        assert not has_secret(seeded)
        texts = " ".join(n.content for n in _all_nodes(seeded.db))
        assert "brand new remark" in texts

    def test_a_sentence_forgotten_from_a_growing_session_stays_out(self, tmp_path, monkeypatch):
        """A node-level forget: the session keeps growing, the file changes,
        and the new messages are captured - without the forgotten sentence
        and without touching the short sentences around it."""
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        target = [n for n in store.list_nodes(limit=50)
                  if n.node_type.value == "decision" and seeded.secret in n.content][0]
        GraphOperations(store).forget_node(target.node_id)
        store.close()
        with open(seeded.source_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "user", "sessionId": "uuid-claude-1",
                "message": {"role": "user", "content": [{"type": "text",
                            "text": "We also decided to use redis for caching."}]},
                "timestamp": "2026-07-02T10:00:00Z"}) + "\n")
        fs._age(seeded.source_file)
        seeded.recapture()
        texts = " ".join(n.content for n in _all_nodes(seeded.db))
        assert "redis for caching" in texts
        # the sentence that was forgotten is not back; a different sentence
        # of the same session that was not forgotten is
        assert f"decided to use {seeded.secret} as the codename" not in texts
        assert f"my safe code is {seeded.secret}" in texts

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

    def test_a_moved_file_that_then_changes_is_the_same_file_changed(self, tmp_path, monkeypatch):
        """The inode is how Revien knows a moved file is the file it forgot:
        changed, it is new evidence; without the inode it could not tell."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        d = tmp_path / "moved2" / "projects" / "new-proj"
        d.mkdir(parents=True)
        new = d / "renamed.jsonl"
        shutil.move(str(seeded.source_file), str(new))
        with open(new, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "user", "sessionId": "uuid-claude-1",
                "message": {"role": "user", "content": [{"type": "text",
                            "text": "Something new said after the move."}]},
                "timestamp": "2026-07-02T10:00:00Z"}) + "\n")
        fs._age(new)
        fs._resync(seeded.db, "claude-code",
                   ClaudeCodeAdapter(session_dir=str(tmp_path / "moved2" / "projects")))
        assert node_count(seeded.db) > 0

    def test_an_unchanged_file_stays_out_even_if_the_adapter_now_reads_it_differently(
            self, tmp_path, monkeypatch):
        """The file on disk decides, not what the parser makes of it today."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        adapter = ClaudeCodeAdapter(session_dir=str(seeded.extra["root"] / "projects"))
        real = adapter._parse_session_log_ts
        adapter._parse_session_log_ts = lambda p: (
            (lambda r: ((r[0] or "") + "\nAssistant: A line the parser invented.", r[1]))(real(p)))
        fs._resync(seeded.db, "claude-code", adapter)
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
            # recognised as a copy by content alone (it has no id and no
            # path): it is not refused whole, the sentences that held what was
            # forgotten are left out and the response says so
            body = r.json()
            assert body["suppressed"] is False
            assert body["left_out"]["matched_by"] == "content"
        assert not has_secret(seeded)

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


class TestAForgetThatLandsMidCapture:
    """The gate is asked again just before the write, under the lock a forget
    also takes, so a forget that lands while the extractor runs is not written
    over."""

    def _extractor_that_forgets(self, store, target_ref):
        from revien.ingestion.extractor import RuleBasedExtractor

        class Forgetting(RuleBasedExtractor):
            fired = False

            def extract(self, content, source_id=""):
                if not Forgetting.fired and target_ref["id"]:
                    Forgetting.fired = True
                    GraphOperations(store).forget_node(target_ref["id"], cascade=True)
                return super().extract(content, source_id)

        return Forgetting()

    def test_fresh_capture(self, tmp_path):
        from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
        f = tmp_path / "notes.md"
        f.write_text("We decided to use postgres for billing.")
        store = GraphStore(db_path=str(tmp_path / "m.db"))
        try:
            mk = lambda text: IngestionInput(
                source_id="file:notes.md", origin_runtime="file", origin_source="watch",
                content=text, metadata={"path": str(f)})
            first = IngestionPipeline(store).ingest(mk("We decided to use postgres for billing."))
            ref = {"id": first.context_node_id}
            f.write_text("We decided to use postgres for billing. And redis.")
            pipe2 = IngestionPipeline(store, extractor=self._extractor_that_forgets(store, ref))
            # the forget of the file's unit lands while the extractor runs
            out = pipe2.ingest(mk("We decided to use postgres for billing. And redis."))
            assert out.suppressed is True and out.context_node_id == ""
            assert store.count_nodes() == 0
        finally:
            store.close()

    def test_keyed_refresh(self, tmp_path):
        from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
        store = GraphStore(db_path=str(tmp_path / "m2.db"))
        try:
            key = "chatgpt:conversation:c9"
            mk = lambda text: IngestionInput(
                source_id=key, ingest_key=key, origin_runtime="chatgpt",
                session_key="c9", content=text)
            first = IngestionPipeline(store).ingest(mk("User: We decided to use postgres for billing."))
            ref = {"id": first.context_node_id}
            pipe2 = IngestionPipeline(store, extractor=self._extractor_that_forgets(store, ref))
            out = pipe2.ingest(mk("User: We decided to use postgres for billing. And redis."))
            assert out.suppressed is True
            assert store.count_nodes() == 0
        finally:
            store.close()


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
            assert len(ops) == 2 and {o["actor"] for o in ops} == {"override:import"}
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
        # the nodes of the forgotten unit, and the sentences that held what was
        # forgotten, do not come back (what remains of the context node is a
        # copy recognised by content only: the rest of it lands)
        assert result["skipped_suppressed"] >= len(graph.nodes) - 1
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
            def action(session, step):
                return store.add_node(
                    Node(node_type=NodeType.ACTION, label=step, content=step,
                         session_key=session, project_key="p", origin_runtime="claude-code",
                         source_id=f"claude-code:p:{session}"),
                    owner=(f"session:claude-code:{session}", 1, 1))

            # a unit that said the step is forgotten (node-level)
            ctx = store.add_node(Node(node_type=NodeType.CONTEXT, label="chat",
                                      content="User: first open the ZQXSTEP gate. then run the nightly job."),
                                 owner=("session:claude-code:gone", 1, 1))
            gone = action("gone", "open the ZQXSTEP gate")
            GraphOperations(store).forget_node(gone.node_id)
            # three other sessions use the same two steps in order
            for s in ("s1", "s2", "s3"):
                action(s, "open the ZQXSTEP gate")
                action(s, "run the nightly job")
            made = propose_skills(store, min_occurrences=3, min_sessions=2)
            assert made["detected"] >= 1
            assert made["created"] == 0 and made["suppressed"] >= 1
            assert store.list_nodes(node_type=NodeType.SKILL, limit=10) == []
        finally:
            store.close()

    def test_a_proposal_keeps_only_its_name_when_a_contributing_session_is_forgotten(self, tmp_path):
        from revien.graph.schema import Node, NodeType
        from revien.skills.proposals import propose_skills
        store = GraphStore(db_path=str(tmp_path / "sp2.db"))
        try:
            for s in ("s1", "s2", "s3"):
                store.add_node(
                    Node(node_type=NodeType.CONTEXT, label=f"chat {s}", content=f"User: chat {s}"),
                    owner=(f"session:claude-code:{s}", 1, 1))
                for step in ("open the gate", "run the nightly job"):
                    store.add_node(
                        Node(node_type=NodeType.ACTION, label=step, content=step,
                             session_key=s, project_key="p", origin_runtime="claude-code",
                             source_id=f"claude-code:p:{s}"),
                        owner=(f"session:claude-code:{s}", 1, 1))
            assert propose_skills(store, min_occurrences=3, min_sessions=2)["created"] == 1
            (prop,) = store.list_nodes(node_type=NodeType.SKILL, limit=10)
            assert {r["unit_id"] for r in store.get_provenance(prop.node_id)} == {
                "session:claude-code:s1", "session:claude-code:s2", "session:claude-code:s3"}
            s1 = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=10)
                  if n.label == "chat s1"][0]
            GraphOperations(store).forget_node(s1.node_id, cascade=True)
            after = store.get_node(prop.node_id)
            assert after is not None                      # two other sessions still vouch for the name
            assert after.metadata.get("steps") == [] and "cut_back_at" in after.metadata
            assert "open the gate" not in after.content
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
            # the note's subject and one claim are not part of the forgotten
            # unit, so the note stays reconcilable after the forget
            ent = store.add_node(Node(node_type=NodeType.ENTITY, label="Atlas Server",
                                      content="Atlas Server", source_type=SourceType.EXTRACTED,
                                      confidence=1.0))
            keep = store.add_node(Node(node_type=NodeType.FACT, label="ram", content="Has 64GB of RAM.",
                                       source_type=SourceType.EXTRACTED, confidence=1.0))
            fact = store.add_node(Node(node_type=NodeType.FACT, label="os",
                                       content="Atlas runs on Ubuntu ZQXWORD.",
                                       source_type=SourceType.EXTRACTED, confidence=1.0), owner=unit)
            for n in (keep, fact):
                store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=ent.node_id,
                                    target_node_id=n.node_id, weight=0.8))
            VaultDistiller(store, str(vault)).distill()
            note = vault / "Revien" / "Atlas Server.md"
            assert "ZQXWORD" in note.read_text(encoding="utf-8")
            res = GraphOperations(store).forget_node(ctx.node_id, cascade=True)
            assert res["scope"] == "forgotten from Revien"
            (leftover,) = [l for l in res["leftovers"] if l["kind"] == "vault_note"]
            assert leftover["fix"] == "revien distill-vault" and leftover["note"] == "Atlas Server"
            assert "revien distill-vault" in res["message"]
            # Revien never rewrites the vault itself: the note still shows it
            assert "ZQXWORD" in note.read_text(encoding="utf-8")
            # the user retypes the line without its anchor: not captured
            text = note.read_text(encoding="utf-8")
            text = re.sub(r"(- Atlas runs on Ubuntu ZQXWORD\.).*", r"\1", text)
            note.write_text(text, encoding="utf-8")
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
        file_bytes = seeded.source_file.read_bytes()
        # every text the record could be a plain hash of
        texts = set()
        for n in _all_nodes(seeded.db):
            texts.update((n.content, n.label, n.content.strip(), n.label.strip()))
            texts.update(part.strip() for part in n.content.split("\n"))
            texts.update(part.strip() for part in n.content.replace("\n", ". ").split(". "))
        texts.add(seeded.secret)
        plain = {hashlib.sha256(t.encode()).hexdigest() for t in texts if t}
        plain.add(hashlib.sha256(file_bytes).hexdigest())
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
        fps = set(rec["content_fps"] + rec["wording_fps"] + rec["sentence_fps"] + [rec["file_fp"]])
        assert len(fps) > 3 and all(len(f) == 64 for f in fps)
        assert not plain & fps, "a fingerprint is a plain hash of text that could be guessed"
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

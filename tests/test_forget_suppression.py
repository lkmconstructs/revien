"""Coming back: the denylist (docs/FORGET_SPEC.md §6, invariant 3).

A forget of one thing leaves a record of the sentences that held its wording,
bound to the identity of the source it was forgotten from. A capture that comes
from that source again (a re-sync, a grown session, a moved, renamed, linked or
re-timestamped file, the same conversation id) never brings those sentences
back, whatever else changed; the rest of the source is captured as usual. A
source with no identity of its own (a shared channel) cannot be bound, and the
response says so.
"""
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from revien.daemon.server import create_app
from revien.forget import scrub
from revien.graph.operations import GraphOperations
from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from tests import forget_seed as fs

STABLE = ["claude_code", "codex", "file_watcher", "obsidian", "chatgpt",
          "claude_import", "readwise", "openai_adapter"]
SHARED = ["pipeline", "mcp"]


def forget_all(seeded, **kw):
    res = fs.forget_all(seeded.db, seeded.secret, **kw)
    assert res["status"] == "forgotten", res
    return res


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


def _all_nodes(db):
    s = GraphStore(db_path=db)
    try:
        return s.list_nodes(limit=1000)
    finally:
        s.close()


def decision_of(seeded) -> str:
    """The wording of the decision the seed produced (what a user would
    forget)."""
    return [n.content for n in _all_nodes(seeded.db)
            if n.node_type == NodeType.DECISION and seeded.secret in n.content][0]


def texts(db) -> str:
    return " ".join(n.label + " " + n.content for n in _all_nodes(db))


@pytest.mark.parametrize("path", STABLE)
def test_a_forgotten_wording_is_not_recaptured_by_its_own_path(path, tmp_path, monkeypatch):
    """Invariant 3, every stable capture path."""
    seeded = fs.seed(path, tmp_path, monkeypatch)
    res = forget_all(seeded)
    assert res["bound_to"], res
    assert not has_secret(seeded)
    seeded.recapture()
    assert not has_secret(seeded), f"{path}: recaptured"
    assert seeded.secret not in texts(seeded.db)


@pytest.mark.parametrize("path", SHARED)
def test_shared_channels_cannot_be_bound_and_say_so(path, tmp_path, monkeypatch):
    seeded = fs.seed(path, tmp_path, monkeypatch)
    res = forget_all(seeded)
    assert res["bound_to"] == []
    assert res["not_bound"]
    assert "shared channel" in res["not_bound"][0]["note"]
    assert "shared channel" in res["message"] or "shared channel" in res["blast_radius"]
    assert not has_secret(seeded)
    # nothing identifies the source, so the same capture lands as written
    seeded.recapture()
    assert seeded.secret in texts(seeded.db)
    # and --everywhere is the way to keep it out of a shared channel
    forget_all(seeded, everywhere=True)
    seeded.recapture()
    assert seeded.secret not in texts(seeded.db)


class TestAnIdentityKeepsTheWordingOut:
    def _forgotten(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        forget_all(seeded)
        return seeded

    def test_an_unchanged_file_is_captured_without_the_wording(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        seeded.recapture()
        assert not has_secret(seeded)
        assert "parking permit" in texts(seeded.db)       # the rest of the source is captured

    def test_a_new_timestamp_does_not_bring_it_back(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        now = time.time()
        os.utime(seeded.source_file, (now, now))
        seeded.recapture()
        assert not has_secret(seeded)

    def test_changed_content_is_captured_and_the_wording_stays_out(self, tmp_path, monkeypatch):
        seeded = self._forgotten(tmp_path, monkeypatch)
        with open(seeded.source_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "user", "sessionId": "uuid-claude-1",
                "message": {"role": "user", "content": [{"type": "text",
                            "text": "A brand new remark about deployments."}]},
                "timestamp": "2026-07-02T10:00:00Z"}) + "\n")
        seeded.recapture()
        assert not has_secret(seeded)
        assert "brand new remark" in texts(seeded.db)

    def test_a_sentence_forgotten_from_a_growing_session_stays_out(self, tmp_path, monkeypatch):
        """The session keeps growing, the file changes, and the new messages
        are captured - without the forgotten sentence and without touching
        the sentences around it."""
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
        body = texts(seeded.db)
        assert "redis for caching" in body
        assert f"decided to use {seeded.secret} as the codename" not in body
        assert "parking permit is blue" in body

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
        assert not has_secret(seeded)

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
        assert not has_secret(seeded)

    def test_a_moved_file_does_not_bring_it_back(self, tmp_path, monkeypatch):
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        d = tmp_path / "moved" / "projects" / "new-proj"
        d.mkdir(parents=True)
        new = d / "renamed.jsonl"
        shutil.move(str(seeded.source_file), str(new))
        fs._resync(seeded.db, "claude-code",
                   ClaudeCodeAdapter(session_dir=str(tmp_path / "moved" / "projects")))
        assert not has_secret(seeded)

    def test_a_moved_file_that_then_changes_is_the_same_file_changed(self, tmp_path, monkeypatch):
        """The session id (and the inode) is how Revien knows a moved file is
        the source it forgot from: changed, its new lines are captured, and
        the wording is still kept out."""
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
        assert "Something new said after the move" in texts(seeded.db)
        assert not has_secret(seeded)

    def test_an_unchanged_file_stays_out_even_if_the_adapter_now_reads_it_differently(
            self, tmp_path, monkeypatch):
        """The identity decides, not what the parser makes of the text today."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        adapter = ClaudeCodeAdapter(session_dir=str(seeded.extra["root"] / "projects"))
        real = adapter._parse_session_log_ts
        adapter._parse_session_log_ts = lambda p: (
            (lambda r: ((r[0] or "") + "\nAssistant: A line the parser invented.", r[1]))(real(p)))
        fs._resync(seeded.db, "claude-code", adapter)
        assert not has_secret(seeded)

    def test_a_new_file_at_the_old_path_is_judged_by_its_identity(self, tmp_path, monkeypatch):
        """The file was moved away and a different file now sits where it
        was. A file carrying the same session id is the same source; a file
        with another session id is another source and lands as written."""
        from revien.adapters.claude_code import ClaudeCodeAdapter
        seeded = self._forgotten(tmp_path, monkeypatch)
        old_path = seeded.source_file
        old_text = old_path.read_text()
        d = tmp_path / "away" / "projects" / "away-proj"
        d.mkdir(parents=True)
        shutil.move(str(old_path), str(d / "away.jsonl"))
        adapter = ClaudeCodeAdapter(session_dir=str(old_path.parents[1]))
        # same bytes copied back: still the forgotten source
        old_path.write_text(old_text)
        fs._age(old_path)
        fs._resync(seeded.db, "claude-code", adapter)
        assert not has_secret(seeded)
        # a genuinely different file at the old path
        old_path.write_text(json.dumps({
            "type": "user", "sessionId": "another-session",
            "message": {"role": "user", "content": [{"type": "text",
                        "text": "Totally different notes about lunch."}]},
            "timestamp": "2026-07-03T10:00:00Z"}) + "\n")
        fs._age(old_path)
        fs._resync(seeded.db, "claude-code", adapter)
        assert "Totally different notes about lunch" in texts(seeded.db)
        assert not has_secret(seeded)

    def test_a_made_up_path_binds_nothing_but_the_session_id_still_does(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        sentence = f"Assistant: Noted. {decision_of(seeded)}"
        forget_all(seeded)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "claude-code:proj-x:sess-1",
                "content": f"User: some changed words entirely.\n{sentence}",
                "origin_runtime": "claude-code", "session_key": "sess-1",
                "metadata": {"session_id": "uuid-claude-1",
                             "path": str(tmp_path / "does" / "not" / "exist.jsonl")}})
            assert r.status_code == 200
            assert r.json()["suppressed"] is False
        assert "some changed words entirely" in texts(seeded.db)
        assert not has_secret(seeded)

    def test_a_path_to_some_other_real_file_binds_nothing(self, tmp_path, monkeypatch):
        """With no session id inside it, a capture is known by its file: a path
        to some other real file is that other file, not the forgotten one."""
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        wording = decision_of(seeded)
        forget_all(seeded)
        other = tmp_path / "other.txt"
        other.write_text("unrelated")
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "file:other.txt", "origin_runtime": "file",
                "origin_source": "watch",
                "content": wording, "metadata": {"path": str(other)}})
            assert r.json()["suppressed"] is False
        assert seeded.secret in texts(seeded.db)       # another file: lands as written

    def test_a_case_changed_path_names_the_same_unit(self, tmp_path, monkeypatch):
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        wording = decision_of(seeded)
        forget_all(seeded)
        upper = str(seeded.source_file).upper()
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/ingest", json={
                "source_id": "file:notes.md",
                "content": f"Totally new words. {wording}",
                "origin_runtime": "file", "origin_source": "watch",
                "metadata": {"path": upper}})
            assert r.json()["suppressed"] is False
        assert "Totally new words" in texts(seeded.db)
        assert not has_secret(seeded)


class TestAWatchedFileIsTheSameFileByItsInode:
    """A watched file has no id inside it: its identity is its real path and
    its inode, so a hard link or a moved file is still the file."""

    def test_a_hard_link_to_a_watched_file_is_the_same_file(self, tmp_path, monkeypatch):
        from revien.adapters.file_watcher import FileWatcherAdapter
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        forget_all(seeded)
        d2 = tmp_path / "linked"
        d2.mkdir()
        try:
            os.link(seeded.source_file, d2 / "other-name.md")
        except (OSError, NotImplementedError):
            pytest.skip("hard links unavailable")
        fs._resync(seeded.db, "files", FileWatcherAdapter(watch_dir=str(d2)))
        assert "parking permit" in texts(seeded.db)
        assert not has_secret(seeded)

    def test_a_moved_watched_file_is_the_same_file(self, tmp_path, monkeypatch):
        from revien.adapters.file_watcher import FileWatcherAdapter
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        forget_all(seeded)
        d2 = tmp_path / "moved"
        d2.mkdir()
        shutil.move(str(seeded.source_file), str(d2 / "renamed.md"))
        fs._resync(seeded.db, "files", FileWatcherAdapter(watch_dir=str(d2)))
        assert not has_secret(seeded)

    def test_a_copy_of_a_watched_file_is_another_file_and_lands(self, tmp_path, monkeypatch):
        from revien.adapters.file_watcher import FileWatcherAdapter
        seeded = fs.seed("file_watcher", tmp_path, monkeypatch)
        forget_all(seeded)
        d2 = tmp_path / "copied"
        d2.mkdir()
        shutil.copy(seeded.source_file, d2 / "copy.md")
        fs._age(d2 / "copy.md")
        fs._resync(seeded.db, "files", FileWatcherAdapter(watch_dir=str(d2)))
        assert has_secret(seeded)           # a different path and inode: not recognised


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
                    GraphOperations(store).forget_node(target_ref["id"])
                return super().extract(content, source_id)

        return Forgetting()

    def _decision(self, store, needle):
        return [n for n in store.list_nodes(node_type=NodeType.DECISION, limit=20)
                if needle in n.content][0]

    def test_fresh_capture(self, tmp_path):
        from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
        f = tmp_path / "notes.md"
        f.write_text("We decided to use postgres for billing.")
        store = GraphStore(db_path=str(tmp_path / "m.db"))
        try:
            mk = lambda text: IngestionInput(
                source_id="file:notes.md", origin_runtime="file", origin_source="watch",
                content=text, metadata={"path": str(f)})
            IngestionPipeline(store).ingest(mk("We decided to use postgres for billing."))
            ref = {"id": self._decision(store, "postgres for billing").node_id}
            f.write_text("We decided to use postgres for billing. And redis.")
            pipe2 = IngestionPipeline(store, extractor=self._extractor_that_forgets(store, ref))
            # the forget lands while the extractor runs
            out = pipe2.ingest(mk("We decided to use postgres for billing. And redis."))
            assert out.suppressed is False
            body = " ".join(n.label + " " + n.content for n in store.list_nodes(limit=100))
            assert "postgres for billing" not in body
            assert "redis" in body
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
            IngestionPipeline(store).ingest(mk("User: We decided to use postgres for billing."))
            ref = {"id": self._decision(store, "postgres for billing").node_id}
            pipe2 = IngestionPipeline(store, extractor=self._extractor_that_forgets(store, ref))
            out = pipe2.ingest(mk("User: We decided to use postgres for billing. And redis."))
            assert out.suppressed is False
            body = " ".join(n.label + " " + n.content for n in store.list_nodes(limit=100))
            assert "postgres for billing" not in body
            assert "redis" in body
        finally:
            store.close()


class TestAConversationThatIsEditedIsStillTheSameConversation:
    def test_an_edited_export_keeps_the_wording_out(self, tmp_path, monkeypatch):
        seeded = fs.seed("chatgpt", tmp_path, monkeypatch)
        wording = decision_of(seeded)
        forget_all(seeded)
        # an EDITED export is still the same conversation (same id): the
        # wording stays out of it, the new answer is captured
        export = seeded.extra["export"]
        conv = json.loads(export.read_text())
        conv[0]["mapping"]["m2"]["message"]["content"]["parts"] = [
            f"A different answer. {wording}"]
        export.write_text(json.dumps(conv))
        seeded.recapture()
        assert "A different answer" in texts(seeded.db)
        assert not has_secret(seeded)


class TestAnExactWordingCaptureIsRefusedOnlyAfterEverywhere:
    """forget --everywhere: a capture that is exactly the forgotten wording is
    refused whole, from any source, and a restore (or the import override)
    lifts that."""

    def _unit(self, content):
        from revien.importers.base import ImportUnit
        return ImportUnit(
            source_id="import:other:1", ingest_key="import:other:1", content=content,
            content_type="conversation", timestamp=None, origin_runtime="chatgpt",
            session_key="1")

    def _setup(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        target = [n for n in store.list_nodes(limit=50)
                  if n.node_type.value == "decision" and seeded.secret in n.content][0]
        wording = target.content
        GraphOperations(store).forget_node(target.node_id, everywhere=True)
        store.close()
        return seeded, wording

    def test_the_importer_reports_the_refusal(self, tmp_path, monkeypatch):
        from revien.importers.base import run_import
        from revien.ingestion.pipeline import IngestionPipeline
        seeded, wording = self._setup(tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        try:
            report = run_import([self._unit(wording)], IngestionPipeline(store))
        finally:
            store.close()
        assert report.units_suppressed == 1 and report.units_ingested == 0
        assert not has_secret(seeded)

    def test_override_on_a_reimport_lifts_it_and_audits(self, tmp_path, monkeypatch):
        from revien.importers.base import run_import
        from revien.ingestion.pipeline import IngestionPipeline
        seeded, wording = self._setup(tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        try:
            report = run_import([self._unit(wording)],
                                IngestionPipeline(store), override_suppression=True)
            assert report.units_suppressed == 0 and report.units_ingested == 1
            assert not [r for r in store.list_suppressions("active") if r["scope"] == "global"]
            ops = [a for a in store.get_all_audit() if a["op"] == "restore"]
            assert len(ops) == 1 and ops[0]["actor"] == "override:import"
        finally:
            store.close()


class TestOtherGates:
    def test_graph_import_does_not_bring_a_forgotten_sentence_back(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        graph = store.export_graph()                    # a backup taken before
        store.close()
        res = forget_all(seeded)
        store = GraphStore(db_path=seeded.db)
        result = store.import_graph(graph, mode="merge")
        store.close()
        # the nodes that held the sentence do not come back, and the sentence
        # is left out of the stored conversation that does
        assert result["skipped_suppressed"] >= res["count"]
        assert not has_secret(seeded)

    def test_rest_graph_import_is_gated_too(self, tmp_path, monkeypatch):
        seeded = fs.seed("claude_code", tmp_path, monkeypatch)
        store = GraphStore(db_path=seeded.db)
        graph = store.export_graph().model_dump(mode="json")
        store.close()
        forget_all(seeded)
        with TestClient(create_app(db_path=seeded.db)) as client:
            r = client.post("/v1/graph/import?mode=merge", json=graph)
            assert r.status_code == 200 and r.json()["skipped_suppressed"] > 0
        assert not has_secret(seeded)

    def _actions(self, store, sessions, steps):
        from revien.graph.schema import Node

        for s in sessions:
            for step in steps:
                store.add_node(
                    Node(node_type=NodeType.ACTION, label=step, content=step,
                         session_key=s, project_key="p", origin_runtime="claude-code",
                         source_id=f"claude-code:p:{s}"),
                    owner=(f"session:claude-code:{s}", 1, 1))

    def _gone(self, store):
        from revien.graph.schema import Node

        store.add_node(Node(node_type=NodeType.CONTEXT, label="chat",
                            content="User: first open the ZQXSTEP gate. then run the nightly job."),
                       owner=("session:claude-code:gone", 1, 1))
        return store.add_node(
            Node(node_type=NodeType.ACTION, label="open the ZQXSTEP gate",
                 content="open the ZQXSTEP gate", session_key="gone", project_key="p",
                 origin_runtime="claude-code", source_id="claude-code:p:gone"),
            owner=("session:claude-code:gone", 1, 1))

    def test_skill_proposals_are_not_made_from_wording_forgotten_everywhere(self, tmp_path):
        from revien.skills.proposals import propose_skills
        store = GraphStore(db_path=str(tmp_path / "sp.db"))
        try:
            gone = self._gone(store)
            GraphOperations(store).forget_node(gone.node_id, everywhere=True)
            # three other sessions use the same two steps in order, later
            self._actions(store, ("s1", "s2", "s3"),
                          ("open the ZQXSTEP gate", "run the nightly job"))
            made = propose_skills(store, min_occurrences=3, min_sessions=2)
            assert made["detected"] >= 1
            assert made["created"] == 0 and made["suppressed"] >= 1
            assert store.list_nodes(node_type=NodeType.SKILL, limit=10) == []
        finally:
            store.close()

    def test_a_plain_forget_does_not_stop_other_sessions_proposing_the_same_steps(self, tmp_path):
        from revien.skills.proposals import propose_skills
        store = GraphStore(db_path=str(tmp_path / "sp1.db"))
        try:
            gone = self._gone(store)
            GraphOperations(store).forget_node(gone.node_id)
            self._actions(store, ("s1", "s2", "s3"),
                          ("open the ZQXSTEP gate", "run the nightly job"))
            # those sessions hold their own copies: forget is per source
            assert propose_skills(store, min_occurrences=3, min_sessions=2)["created"] == 1
        finally:
            store.close()

    def test_a_proposal_keeps_only_its_name_when_a_contributing_step_is_forgotten(self, tmp_path):
        from revien.graph.schema import Node
        from revien.skills.proposals import propose_skills
        store = GraphStore(db_path=str(tmp_path / "sp2.db"))
        try:
            for s in ("s1", "s2", "s3"):
                store.add_node(
                    Node(node_type=NodeType.CONTEXT, label=f"chat {s}",
                         content="User: open the gate. run the nightly job."),
                    owner=(f"session:claude-code:{s}", 1, 1))
            self._actions(store, ("s1", "s2", "s3"), ("open the gate", "run the nightly job"))
            assert propose_skills(store, min_occurrences=3, min_sessions=2)["created"] == 1
            (prop,) = store.list_nodes(node_type=NodeType.SKILL, limit=10)
            assert {r["unit_id"] for r in store.get_provenance(prop.node_id)} == {
                "session:claude-code:s1", "session:claude-code:s2", "session:claude-code:s3"}
            step = [n for n in store.list_nodes(node_type=NodeType.ACTION, limit=20)
                    if n.session_key == "s1" and n.label == "open the gate"][0]
            GraphOperations(store).forget_node(step.node_id)
            after = store.get_node(prop.node_id)
            assert after is not None                      # two other sessions still vouch for the name
            assert after.metadata.get("steps") == [] and "cut_back_at" in after.metadata
            assert "open the gate" not in after.content
        finally:
            store.close()

    def test_typing_wording_forgotten_everywhere_back_into_a_distilled_note_is_not_captured(self, tmp_path):
        """A line retyped into a distilled note is a capture like any other:
        a forget --everywhere keeps its exact wording out."""
        import re
        from revien.distill import VaultDistiller, VaultReconciler
        from revien.graph.schema import Edge, EdgeType, Node, SourceType
        from revien.semantic.index import SemanticIndex
        store = GraphStore(db_path=str(tmp_path / "d.db"))
        vault = tmp_path / "v"
        vault.mkdir()
        unit = ("session:claude-code:sD", 1, 1)
        try:
            store.add_node(Node(node_type=NodeType.CONTEXT, label="chat",
                                content="User: Atlas runs on Ubuntu ZQXWORD."), owner=unit)
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
            res = GraphOperations(store).forget_node(fact.node_id, everywhere=True)
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
        # every text the record could be a plain hash of
        plain_texts = set()
        for n in _all_nodes(seeded.db):
            plain_texts.update((n.content, n.label, n.content.strip(), n.label.strip()))
            plain_texts.update(part.strip() for part in n.content.split("\n"))
            plain_texts.update(part.strip() for part in n.content.replace("\n", ". ").split(". "))
        plain_texts.add(seeded.secret)
        plain = {hashlib.sha256(t.encode()).hexdigest() for t in plain_texts if t}
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
        fps = set(rec["wording_fps"] + rec["sentence_fps"] + rec["short_fps"])
        assert len(fps) >= 2 and all(len(f) == 64 for f in fps)
        assert not plain & fps, "a fingerprint is a plain hash of text that could be guessed"
        # and it does depend on the key
        old = set(rec["wording_fps"])
        keys._cache.clear()
        key_file = keys.key_path()
        key_file.write_text(("ab" * 32))
        keys._cache.clear()
        assert keys.fingerprint(seeded.secret) not in old
        assert keys.fingerprints([seeded.secret]) != list(old)
        # the key is a file in the Revien home, owner-only, never in the db
        raw = open(seeded.db, "rb").read()
        assert ("ab" * 32).encode() not in raw
        if os.name == "posix":
            assert (key_file.stat().st_mode & 0o077) == 0

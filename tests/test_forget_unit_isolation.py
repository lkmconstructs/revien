"""A forget removes the forgotten unit and nothing else.

Covers the whole-session cascade's ownership rule (what belongs exclusively to
the forgotten unit), the colliding-identity cases (a basename or a vault path
shared by two different files), sub-agent units, the query text of recalls that
surfaced a forgotten node, the one-time scrub of a legacy training.db, and the
notices that state what was removed and what was kept.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.adapters.file_watcher import FileWatcherAdapter
from revien.daemon.scheduler import SyncScheduler
from revien.daemon.server import create_app
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

from tests.test_forget_followups import (
    UUID1, UUID2, _subagent_lines, bump, cc_session, run_async,
)

A = f"claude-code:proj-a:{UUID1}"
B = f"claude-code:proj-a:{UUID2}"


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.unlink(path + suffix)


@pytest.fixture
def store(db_path):
    s = GraphStore(db_path=db_path)
    yield s
    s.close()


def ingest(store, sid, text, key=True):
    return IngestionPipeline(store).ingest(IngestionInput(
        source_id=sid, ingest_key=sid if key else "", content=text,
        content_type="conversation"))


def raw(path):
    out = b""
    for suffix in ("", "-wal", "-journal"):
        f = Path(str(path) + suffix)
        if f.exists():
            out += f.read_bytes()
    return out


def ctx_of(store, sid):
    return [n for n in store.list_nodes(node_type=NodeType.CONTEXT,
                                        source_id=sid, limit=99)]


def linked(store, node_id):
    out = set()
    for e in store.get_edges_for_node(node_id):
        out.add(e.target_node_id if e.source_node_id == node_id
                else e.source_node_id)
    return out


def alive(store):
    return {n.node_id for n in store.list_nodes(limit=99999)}


# ---------------------------------------------------------------- ownership

class TestCascadeKeepsEveryOtherUnit:
    TXT_A = ("User: We decided to use Postgres for billing. Alice owns the "
             "payments service. We decided to rotate the vault key to "
             "Zanzibarquokka9 today. I prefer dark mode.\nAssistant: ok")
    TXT_B = ("User: We decided to use Postgres for billing. Alice owns the "
             "payments service. We chose Redis for caching. I prefer dark "
             "mode.\nAssistant: ok")

    def _two(self, store):
        ingest(store, A, self.TXT_A)
        ingest(store, B, self.TXT_B)
        return ctx_of(store, A)[0], ctx_of(store, B)[0]

    def test_session_cascade_does_not_delete_nodes_another_session_links(
            self, store, db_path):
        a, b = self._two(store)
        b_links = linked(store, b.node_id)
        a_links = linked(store, a.node_id)
        assert b_links & a_links, "control: the sessions share dedup'd nodes"
        ops = GraphOperations(store)
        prev = ops.forget_preview(a.node_id)
        out = ops.forget_node(a.node_id, cascade=True, reason="private")
        # the preview is exactly what the forget did
        assert set(prev["node_ids"]) == set(out["forgotten"])
        assert set(prev["kept_shared"]) == set(out["kept_shared"])
        # everything B links survives, and B's text is whole
        assert b_links <= alive(store)
        assert store.get_node(b.node_id).content == f"{self.TXT_B}" or \
            "Redis" in store.get_node(b.node_id).content
        # the forgotten session's own claim is gone everywhere
        assert b"Zanzibarquokka9" not in raw(db_path)
        # nothing still names the forgotten session
        assert not [n for n in store.list_nodes(limit=99999) if n.source_id == A]
        # what was kept is stated, with the node ids
        text = " ".join(out["suppression_notice"])
        assert "Kept:" in text and "Removed:" in text
        assert "Sibling facts already captured" not in text
        assert set(out["kept_shared"]) <= b_links
        assert all(v == [b.node_id] for v in out["kept_shared_linked_from"].values())

    def test_forgetting_the_last_owner_removes_a_shared_node(self, store, db_path):
        a, b = self._two(store)
        ops = GraphOperations(store)
        ops.forget_node(a.node_id, cascade=True, reason="private")
        assert b"Zanzibarquokka9" not in raw(db_path)
        assert b"Postgres" in raw(db_path), "control: B still holds the claim"
        out = ops.forget_node(b.node_id, cascade=True, reason="private")
        assert out["kept_shared"] == []
        left = [n for n in store.list_nodes(limit=99999)
                if not n.node_id.startswith(ops.TOMBSTONE_PREFIX)]
        assert left == []
        assert b"Postgres" not in raw(db_path)

    def test_a_node_merged_by_similarity_keeps_no_forgotten_wording(
            self, store, db_path):
        """Fuzzy dedup (edit distance 2) reuses the FIRST session's node for a
        different claim and keeps the first wording. Keeping that node would
        keep the forgotten session's text, so it goes and the response says so."""
        ingest(store, A, "User: We decided service3 uses queue Q3xray. ok")
        ingest(store, B, "User: We decided service4 uses queue Q4xray. ok")
        a, b = ctx_of(store, A)[0], ctx_of(store, B)[0]
        merged = [store.get_node(i) for i in linked(store, b.node_id)
                  if "Q3xray" in (store.get_node(i).content or "")]
        assert merged, "control: B links a node worded by A"
        out = GraphOperations(store).forget_node(
            a.node_id, cascade=True, reason="private")
        assert {m.node_id for m in merged} <= set(out["removed_unattested_shared"])
        assert b"Q3xray" not in raw(db_path)
        assert "Q4xray" in store.get_node(b.node_id).content
        assert "no other unit vouches for even the label" in \
            " ".join(out["suppression_notice"])

    def test_twelve_sessions_lose_nothing_that_is_not_accounted_for(self, tmp_path):
        shared = ["We decided to use Postgres for billing.",
                  "Alice owns the payments service.",
                  "I prefer pytest for tests.",
                  "We chose Redis for the session cache.",
                  "I prefer dark mode.", "We will deploy with Kubernetes.",
                  "Bob is the release manager.",
                  "We decided to keep migrations in alembic."]
        import random
        random.seed(7)
        base = str(tmp_path / "g.db")
        s = GraphStore(db_path=base)
        sess = []
        for i in range(12):
            sid = f"claude-code:proj:{i:08d}-1111-4111-8111-111111111111"
            picks = random.sample(shared, 3)
            ingest(s, sid, "User: " + " ".join(
                picks + [f"We decided service{i} uses queue Qzed{i}x.",
                         f"Owner{i} owns module{i}."]))
            sess.append(sid)
        s.close()
        total_links = lost_attested = 0
        for k, sid in enumerate(sess):
            db = str(tmp_path / f"c{k}.db")
            shutil.copy(base, db)
            st = GraphStore(db_path=db)
            ctxs = {x: ctx_of(st, x)[0].node_id for x in sess}
            others = {x: (c, linked(st, c)) for x, c in ctxs.items() if x != sid}
            out = GraphOperations(st).forget_node(
                ctxs[sid], cascade=True, reason="p")
            gone = set(out["forgotten"])
            unattested = set(out["removed_unattested_shared"])
            for x, (c, links) in others.items():
                assert c in alive(st), "another unit's context node was deleted"
                assert st.get_node(c).content
                total_links += len(links)
                lost_attested += len((links & gone) - unattested)
            st.close()
            assert f"service{k} uses".encode() not in raw(db)
        assert total_links > 1000
        assert lost_attested == 0


class TestCollidingIdentities:
    def _sync_files(self, store, root):
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("files", FileWatcherAdapter(
            watch_dir=str(root), recursive=True))
        return run_async(sched.sync_one("files"))

    def test_same_basename_in_two_folders_cascade_spares_the_other_file(
            self, store, db_path, tmp_path):
        w = tmp_path / "watch"
        (w / "projA").mkdir(parents=True)
        (w / "projB").mkdir(parents=True)
        (w / "projA" / "notes.md").write_text(
            "We decided to deploy projA on Fridays. The projA secret is Quorvex11.\n")
        (w / "projB" / "notes.md").write_text(
            "We decided projB uses Kafka for events. Bob owns projB billing.\n")
        self._sync_files(store, w)
        ctxs = ctx_of(store, "file:notes.md")
        assert len(ctxs) == 2, "control: two files share one source_id"
        a = [c for c in ctxs if "projA" in c.content][0]
        b = [c for c in ctxs if "projB" in c.content][0]
        b_links = linked(store, b.node_id)
        ops = GraphOperations(store)
        prev = ops.forget_preview(a.node_id)
        assert b.node_id not in prev["node_ids"]
        assert prev["other_contexts_same_source"] == [b.node_id]
        out = ops.forget_node(a.node_id, cascade=True, reason="private")
        assert b.node_id in alive(store) and b_links <= alive(store)
        assert "Kafka" in store.get_node(b.node_id).content
        assert b"Quorvex11" not in raw(db_path)
        assert b"Kafka" in raw(db_path)
        assert out["other_contexts_same_source"] == [b.node_id]
        # the tombstone holds this file's path next to the shared name, and
        # the response says the other file is still captured
        assert set(out["suppressed_identities"]) == {
            "file:notes.md", os.path.realpath(w / "projA" / "notes.md")}
        text = " ".join(out["suppression_notice"])
        assert b.node_id in text and "same source id" in text
        assert "still captured while this path exists" in text
        # disclosed consequence is real: only the forgotten file is skipped
        for f in (w / "projA" / "notes.md", w / "projB" / "notes.md"):
            bump(f, 5000)
        r = self._sync_files(store, w)
        assert r.get("items_suppressed", 0) == 1
        assert r["items_ingested"] == 1

    def test_vault_relative_path_collision_spares_the_other_note(
            self, store, db_path):
        v = "vault:daily/2026-07-01.md#standup"
        ingest(store, v, "We decided vault1 uses SQLite. Carol leads vault1. "
                         "The vault1 pin is Plinthor77.", key=False)
        ingest(store, v, "We decided vault2 uses Mongo. Dave leads vault2.",
               key=False)
        ctxs = ctx_of(store, v)
        assert len(ctxs) == 2
        one = [c for c in ctxs if "vault1" in c.content][0]
        two = [c for c in ctxs if "vault2" in c.content][0]
        out = GraphOperations(store).forget_node(
            one.node_id, cascade=True, reason="private")
        assert two.node_id in alive(store)
        assert "Mongo" in store.get_node(two.node_id).content
        assert b"Plinthor77" not in raw(db_path)
        assert out["other_contexts_same_source"] == [two.node_id]
        assert two.node_id not in out["forgotten"]

    def test_another_units_context_node_is_never_in_a_preview(self, store):
        a, b = A, B
        ingest(store, a, "User: We decided to use Postgres. ok")
        ingest(store, b, "User: We decided to use Postgres. ok")
        ca, cb = ctx_of(store, a)[0], ctx_of(store, b)[0]
        prev = GraphOperations(store).forget_preview(ca.node_id)
        assert cb.node_id not in prev["node_ids"]
        assert ca.node_id in prev["node_ids"]


class TestRestSurfaces:
    def test_preview_and_forget_report_kept_and_removed(self, db_path):
        s = GraphStore(db_path=db_path)
        ingest(s, A, TestCascadeKeepsEveryOtherUnit.TXT_A)
        ingest(s, B, TestCascadeKeepsEveryOtherUnit.TXT_B)
        a, b = ctx_of(s, A)[0], ctx_of(s, B)[0]
        s.close()
        with TestClient(create_app(db_path=db_path)) as c:
            p = c.get(f"/v1/nodes/{a.node_id}/forget/preview").json()
            assert p["kept_shared"] and b.node_id not in p["node_ids"]
            r = c.post(f"/v1/nodes/{a.node_id}/forget",
                       json={"cascade": True, "reason": "private"}).json()
            assert r["count"] == p["count"]
            assert r["kept_shared"] == p["kept_shared"]
            assert any(n.startswith("Kept:") for n in r["suppression_notice"])
            assert c.get(f"/v1/nodes/{b.node_id}").status_code == 200
        assert b"Zanzibarquokka9" not in raw(db_path)


# ----------------------------------------------------------------- notices

class TestNoticesAreAccurate:
    def test_shared_channel_cascade_says_the_facts_were_kept(self, store):
        ingest(store, "mcp", "We decided to use Zarnix for search.", key=False)
        c = [n for n in store.list_nodes(node_type=NodeType.CONTEXT,
                                         source_id="mcp", limit=9)][0]
        out = GraphOperations(store).forget_node(
            c.node_id, cascade=True, reason="private")
        assert out["extracted_facts_kept"] is True
        text = " ".join(out["suppression_notice"])
        assert "facts extracted from it were kept" in text
        assert any("Zarnix" in (n.content or "")
                   for n in store.list_nodes(limit=99))

    def test_extracted_node_of_a_channel_does_not_claim_cascade_forgets_a_session(
            self, store):
        ingest(store, "mcp", "We decided to use Zarnix for search.", key=False)
        fact = [n for n in store.list_nodes(limit=99)
                if n.node_type != NodeType.CONTEXT][0]
        out = GraphOperations(store).forget_node(fact.node_id, reason="private")
        text = " ".join(out["suppression_notice"])
        assert out["context_nodes_remaining"]
        assert "whole session" not in text
        assert "shared channel" in text and "never the facts extracted" in text

    def test_session_extracted_node_points_at_the_cascade(self, store):
        ingest(store, A, "User: We decided to use Zarnix for search. ok")
        fact = [n for n in store.list_nodes(limit=99)
                if n.node_type != NodeType.CONTEXT][0]
        out = GraphOperations(store).forget_node(fact.node_id, reason="private")
        text = " ".join(out["suppression_notice"])
        assert "forget the context node with cascade" in text
        assert "whole session" not in text

    def test_rest_delete_names_distilled_notes(self, db_path, tmp_path):
        from revien.distill import VaultDistiller
        s = GraphStore(db_path=db_path)
        ent = s.add_node(Node(node_type=NodeType.ENTITY, label="Acme",
                              content="Acme", source_id="s"))
        fact = s.add_node(Node(node_type=NodeType.FACT, label="k",
                               content="Acme stores Zanzibarquokka9 keys",
                               source_id="s"))
        s.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                        source_node_id=ent.node_id, target_node_id=fact.node_id))
        vault = tmp_path / "vault"
        vault.mkdir()
        VaultDistiller(s, str(vault)).distill()
        other = s.add_node(Node(node_type=NodeType.FACT, label="z",
                                content="never distilled", source_id="u"))
        s.close()
        with TestClient(create_app(db_path=db_path)) as c:
            r = c.delete(f"/v1/nodes/{fact.node_id}").json()
            assert r["distilled_notes_remaining"] == ["Acme"]
            text = " ".join(r["suppression_notice"])
            assert "distilled vault note" in text
            assert "next `revien distill-vault` removes it" in text
            r2 = c.delete(f"/v1/nodes/{other.node_id}").json()
            assert r2["distilled_notes_remaining"] == []
            assert "vault" not in " ".join(r2["suppression_notice"])


# --------------------------------------------------------------- sub-agents

class TestSubagentUnits:
    def _sync(self, store, root):
        sched = SyncScheduler(pipeline=IngestionPipeline(store))
        sched.register_adapter("cc", ClaudeCodeAdapter(session_dir=str(root)))
        return run_async(sched.sync_one("cc"))

    def _subagent(self, root, uuid, text, name="agent-a1b2c3d4", prefix=0):
        d = root / "projects" / "-home-me-acme" / uuid / "subagents"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.jsonl"
        body = _subagent_lines(uuid, text, False)
        if prefix:
            body = "\n".join(json.dumps({"type": "summary", "summary": f"s{i}"})
                             for i in range(prefix)) + "\n" + body
        p.write_text(body)
        return p

    def test_only_the_parent_is_forgotten_and_a_later_subagent_file_is_skipped(
            self, store, tmp_path):
        """Mutation C7 (drop the parent_session_id lookup in
        find_ingest_suppression) used to pass the suite: every earlier test
        also forgot the sub-agent's own unit. Here only the PARENT is
        forgotten and the sub-agent file, without the sidechain flag, appears
        afterwards."""
        root = tmp_path / "cc"
        main = cc_session(root, "-home-me-acme", UUID1,
                          "We rotate the Brimwort55 key.")
        bump(main, -1000)
        self._sync(store, root)
        ctx = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=9)][0]
        out = GraphOperations(store).forget_node(
            ctx.node_id, cascade=True, reason="private")
        assert out["subagent_contexts_forgotten"] == []
        assert not [n for n in store.list_nodes(limit=999)
                    if "Brimwort55" in (n.content or "")]
        # the sub-agent file is NEW: never ingested, never forgotten itself
        sub = self._subagent(root, UUID1, "Subtask: Brimwort55 rotation plan.")
        other = self._subagent(root, UUID2, "Subtask: write docs for Redis",
                               name="agent-ffff0000")
        r = self._sync(store, root)
        assert not [n for n in store.list_nodes(limit=999)
                    if "Brimwort55" in (n.content or "")], \
            "a later sub-agent file of a forgotten session was captured"
        assert r.get("items_suppressed", 0) >= 1
        assert any("Redis" in (n.content or "") for n in store.list_nodes(limit=999))

    def test_cascade_takes_the_sub_agent_units_already_ingested(
            self, store, db_path, tmp_path):
        root = tmp_path / "cc"
        main = cc_session(root, "-home-me-acme", UUID1, "We rotate the Brimwort55 key.")
        sub = self._subagent(root, UUID1, "Subtask: Brimwort55 rotation plan.")
        other = self._subagent(root, UUID2, "Subtask: write docs for Redis",
                               name="agent-ffff0000")
        for p in (main, sub, other):
            bump(p, -1000)
        self._sync(store, root)
        subs = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=99)
                if n.source_id.endswith("agent-a1b2c3d4")]
        assert subs and subs[0].metadata["parent_session_id"] == UUID1
        parent = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=99)
                  if n.source_id.endswith(UUID1)][0]
        redis_ctx = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=99)
                     if n.source_id.endswith("agent-ffff0000")][0]
        ops = GraphOperations(store)
        prev = ops.forget_preview(parent.node_id)
        assert subs[0].node_id in prev["node_ids"]
        assert prev["subagent_contexts"] == [subs[0].node_id]
        out = ops.forget_node(parent.node_id, cascade=True, reason="private")
        assert out["subagent_contexts_forgotten"] == [subs[0].node_id]
        assert b"Brimwort55" not in raw(db_path)
        assert redis_ctx.node_id in alive(store), \
            "an unrelated session's sub-agent unit was deleted"
        assert "sub-agent" in " ".join(out["suppression_notice"])

    def test_session_id_beyond_the_first_200_lines_is_still_read(
            self, store, tmp_path):
        root = tmp_path / "cc"
        main = cc_session(root, "-home-me-acme", UUID1, "We rotate the Brimwort55 key.")
        bump(main, -1000)
        self._sync(store, root)
        ctx = [n for n in store.list_nodes(node_type=NodeType.CONTEXT, limit=9)][0]
        GraphOperations(store).forget_node(ctx.node_id, cascade=True, reason="p")
        self._subagent(root, UUID1, "Subtask: Brimwort55 again.", prefix=450)
        items = run_async(ClaudeCodeAdapter(session_dir=str(root)).fetch_new_content(
            datetime(2020, 1, 1, tzinfo=timezone.utc)))
        sub = [i for i in items if i["source_id"].endswith("agent-a1b2c3d4")]
        assert sub and sub[0]["metadata"]["parent_session_id"] == UUID1
        r = self._sync(store, root)
        assert not [n for n in store.list_nodes(limit=999)
                    if "Brimwort55" in (n.content or "")]
        assert r.get("items_suppressed", 0) >= 1


# ------------------------------------------------------- recall query text

class TestUseJudgmentQueriesArePurged:
    QUERY = "is the staging vault code Kelvarynth used with billing"

    def test_every_judgment_of_the_recall_loses_the_query(self, store, db_path):
        n1 = store.add_node(Node(node_type=NodeType.FACT, label="vault",
                                 content="vault code Kelvarynth", source_id="s"))
        n2 = store.add_node(Node(node_type=NodeType.FACT, label="billing",
                                 content="Postgres for billing", source_id="s"))
        n3 = store.add_node(Node(node_type=NodeType.FACT, label="cache",
                                 content="Redis cache", source_id="s"))
        for nid, key in ((n1, "ep1"), (n2, "ep1"), (n3, "ep2")):
            store.record_use_judgment(
                episode_key=key, node_id=nid.node_id, used=True, reason="r",
                shared_terms=["x"], similarity=0.5,
                query=self.QUERY if key == "ep1" else "Redis question",
                mode="apply", applied=True)
        assert raw(db_path).count(self.QUERY.encode()) >= 2, "fixture"
        GraphOperations(store).forget_node(n1.node_id, reason="private")
        assert self.QUERY.encode() not in raw(db_path)
        rows = {r["node_id"]: r for r in store.list_use_judgments(limit=10)}
        assert n1.node_id not in rows
        assert rows[n2.node_id]["query"] == ""        # same recall: blanked
        assert rows[n2.node_id]["used"] is True        # evidence itself stays
        assert rows[n3.node_id]["query"] == "Redis question"  # other recall


# ----------------------------------------------------------- legacy training

class TestLegacyTrainingDbIsScrubbedOnce:
    def _legacy(self, path):
        c = sqlite3.connect(path)
        c.execute("PRAGMA secure_delete = OFF")
        c.execute("CREATE TABLE retrieval_signals (id INTEGER PRIMARY KEY "
                  "AUTOINCREMENT, query TEXT NOT NULL, node_id TEXT NOT NULL, "
                  "node_label TEXT, node_type TEXT, score REAL, "
                  "was_used INTEGER DEFAULT 0, timestamp TEXT NOT NULL)")
        for i in range(30):
            q = "what is Zephrolux991" if i == 5 else f"ordinary query {i}"
            c.execute("INSERT INTO retrieval_signals (query, node_id, node_label, "
                      "node_type, score, timestamp) VALUES (?,?,?,?,?,'t')",
                      (q, f"n{i}", f"label {i}", "fact", 0.5))
        c.commit()
        c.execute("DELETE FROM retrieval_signals WHERE id = 6")
        c.commit()
        free = c.execute("PRAGMA freelist_count").fetchone()[0]
        c.close()
        return free

    def test_residue_in_a_partly_used_page_is_removed_with_empty_freelist(
            self, tmp_path):
        from revien.neural.training import TrainingLoop
        tdb = tmp_path / "training.db"
        free = self._legacy(str(tdb))
        assert free == 0, "control: no page was freed, so the old check never ran"
        assert b"Zephrolux991" in tdb.read_bytes(), "control: residue exists"
        TrainingLoop(db_path=str(tdb), model_dir=str(tmp_path / "m"))
        assert b"Zephrolux991" not in tdb.read_bytes()
        c = sqlite3.connect(str(tdb))
        try:
            n = c.execute("SELECT count(*) FROM retrieval_signals").fetchone()[0]
            marker = c.execute("SELECT value FROM training_state WHERE key = "
                               "'secure_scrub_v1'").fetchone()
        finally:
            c.close()
        assert n == 29 and marker == ("1",)

    def test_the_rewrite_runs_once_per_file(self, tmp_path, monkeypatch):
        from revien.neural.training import TrainingLoop
        tdb = tmp_path / "training.db"
        TrainingLoop(db_path=str(tdb), model_dir=str(tmp_path / "m"))
        seen = []
        orig = TrainingLoop._connect

        def spy(self):
            conn = orig(self)
            real = conn.execute

            class W:
                def __getattr__(_s, k):
                    return getattr(conn, k)

                def execute(_s, sql, *a):
                    seen.append(sql)
                    return real(sql, *a)
            return W()
        monkeypatch.setattr(TrainingLoop, "_connect", spy)
        TrainingLoop(db_path=str(tdb), model_dir=str(tmp_path / "m"))
        assert "VACUUM" not in seen

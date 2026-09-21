"""
Skill proposals (thin WS3, leg D2).

Covers: repeated-ACTION-sequence detection (n-gram windows 2..4, session
grouping incl. the session_key=None date fallback, sub-pattern suppression
so one workflow yields one proposal not three), propose_skills' governance
contract (status=proposed/origin=engine only, never active; DERIVED_FROM
edges proposal->action; idempotent re-run; audit op skill_propose),
accept/decline (audit ops skill_accept/skill_decline; third decline
invalidates), the recall `skill_proposals` field (draft-only surfacing,
keyword match), and the daemon /v1/skills routes (GETs open on loopback,
POSTs gated by require_mutation_auth). Fictional names only (Mara/Theo/Sam,
Fernweh-Core).
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from revien.daemon import server as server_module
from revien.daemon.server import create_app
from revien.graph.schema import EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.skills.proposals import (
    accept_proposal,
    decline_proposal,
    detect_repeated_sequences,
    matching_proposals,
    normalize_label,
    propose_skills,
)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _action(store, label, project_key, session_key, recorded_at, content=None):
    return store.add_node(Node(
        node_type=NodeType.ACTION,
        label=label,
        content=content or label,
        project_key=project_key,
        session_key=session_key,
        recorded_at=recorded_at,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
    ))


def _seed_three_sessions(store, steps, project_key="fernweh-core"):
    """Same step sequence, once each in 3 distinct sessions. Returns the
    list of per-session lists of created ACTION nodes."""
    sessions = []
    for i, session_key in enumerate(("sess-a", "sess-b", "sess-c")):
        t0 = _BASE_TIME + timedelta(days=i)
        nodes = [
            _action(store, step, project_key, session_key, t0 + timedelta(minutes=m))
            for m, step in enumerate(steps)
        ]
        sessions.append(nodes)
    return sessions


# ── normalize_label ──────────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_label("Ping Theo, now!") == "ping theo now"

    def test_drops_leading_pronoun_contraction(self):
        assert normalize_label("I'll ping Theo") == "ping theo"

    def test_drops_leading_article(self):
        assert normalize_label("The migration script") == "migration script"

    def test_collapses_whitespace(self):
        assert normalize_label("  ping   theo  ") == "ping theo"

    def test_empty_and_none_safe(self):
        assert normalize_label("") == ""
        assert normalize_label(None) == ""


# ── detect_repeated_sequences ────────────────────────────────────────────

class TestDetectRepeatedSequences:
    def test_three_sessions_qualify(self, store):
        _seed_three_sessions(store, ["I'll ping Theo", "I'll run the migration"])

        patterns = detect_repeated_sequences(store)

        assert len(patterns) == 1
        p = patterns[0]
        assert p["occurrences"] == 3
        assert p["sessions"] == 3
        assert p["steps"] == ["ping theo", "run the migration"]
        assert p["project_key"] == "fernweh-core"
        assert len(p["node_ids"]) == 6  # 3 sessions x 2 steps, all distinct

    def test_single_session_never_qualifies(self, store):
        # Same 2-step pattern repeated 3x back-to-back, but ALL inside one
        # session — occurrences=3 would pass the count alone, but
        # min_sessions=2 must still refuse it.
        session_key = "sess-solo"
        steps = ["ping theo", "run migration"] * 3
        for m, step in enumerate(steps):
            _action(store, step, "fernweh-core", session_key, _BASE_TIME + timedelta(minutes=m))

        patterns = detect_repeated_sequences(store)
        assert patterns == []

    def test_session_key_none_falls_back_to_project_and_date(self, store):
        day = _BASE_TIME
        steps = ["I'll sync fernweh branches", "I'll notify Mara"]
        for session_idx in range(3):
            # Distinct calendar days => distinct fallback groups, session_key
            # left None throughout (the time-gap fallback path).
            t0 = day + timedelta(days=session_idx)
            for m, step in enumerate(steps):
                _action(store, step, "fernweh-core", None, t0 + timedelta(minutes=m))

        patterns = detect_repeated_sequences(store)
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 3
        assert patterns[0]["sessions"] == 3

    def test_longer_window_suppresses_sub_patterns(self, store):
        # A 4-step workflow repeated identically across 3 sessions would,
        # without suppression, ALSO register as two qualifying 3-grams and
        # three qualifying 2-grams of the very same workflow.
        steps = ["fetch data", "clean data", "train model", "evaluate results"]
        _seed_three_sessions(store, steps)

        patterns = detect_repeated_sequences(store)

        assert len(patterns) == 1, (
            "one workflow must yield one proposal, not three: "
            f"got window lengths {[len(p['steps']) for p in patterns]}"
        )
        assert patterns[0]["steps"] == steps
        assert len(patterns[0]["node_ids"]) == 12  # 3 sessions x 4 steps

    def test_invalidated_actions_excluded(self, store):
        from revien.graph.operations import GraphOperations

        sessions = _seed_three_sessions(store, ["ping theo", "run migration"])
        ops = GraphOperations(store)
        # Soft-invalidate one of the three sessions' occurrences.
        for node in sessions[0]:
            ops.invalidate_node(node.node_id, reason="test cleanup")

        patterns = detect_repeated_sequences(store)
        assert patterns == []  # down to 2 sessions, below the default min of 2...

    def test_below_threshold_not_proposed(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        patterns = detect_repeated_sequences(store, min_occurrences=4)
        assert patterns == []


# ── propose_skills governance ─────────────────────────────────────────────

class TestProposeSkillsGovernance:
    def test_creates_one_proposed_engine_skill(self, store):
        _seed_three_sessions(store, ["I'll ping Theo", "I'll run the migration"])

        summary = propose_skills(store)

        assert summary["detected"] == 1
        assert summary["created"] == 1
        proposals = store.list_nodes(node_type=NodeType.SKILL, limit=10)
        assert len(proposals) == 1
        node = proposals[0]
        assert node.metadata["status"] == "proposed"
        assert node.metadata["origin"] == "engine"
        assert "curated" not in node.metadata
        assert node.metadata["occurrences"] == 3
        assert node.metadata["sessions"] == 3
        assert node.metadata["declines"] == 0
        assert node.metadata["draft"] is False  # default rule-based extractor
        assert node.label.startswith("proposed: ")
        assert node.source_type.value == "inferred"

    def test_never_creates_active_status(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        propose_skills(store)
        proposals = store.list_nodes(node_type=NodeType.SKILL, limit=10)
        assert all(n.metadata.get("status") != "active" for n in proposals)

    def test_derived_from_edges_point_proposal_to_actions(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        summary = propose_skills(store)
        node = summary["proposals"][0]

        edges = [e for e in store.get_edges_for_node(node.node_id)
                 if e.edge_type == EdgeType.DERIVED_FROM]
        assert len(edges) == 6
        # The proposal (derived thing) is the SOURCE; each ACTION node (the
        # ancestor material) is the TARGET — confirmed against the actual
        # get_lineage/get_children_of convention in graph/operations.py,
        # which is the opposite of schema.py's literal docstring wording.
        assert all(e.source_node_id == node.node_id for e in edges)
        action_ids = {n.node_id for n in store.list_nodes(node_type=NodeType.ACTION, limit=50)}
        assert {e.target_node_id for e in edges} <= action_ids

    def test_records_skill_propose_audit(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        summary = propose_skills(store)
        node = summary["proposals"][0]

        history = store.get_node_audit(node.node_id)
        ops = [h["op"] for h in history]
        assert "skill_propose" in ops
        propose_entry = next(h for h in history if h["op"] == "skill_propose")
        assert propose_entry["before"] is None
        assert propose_entry["after"]["metadata"]["status"] == "proposed"

    def test_idempotent_rerun_no_duplicates(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        propose_skills(store)
        summary2 = propose_skills(store)

        assert summary2["created"] == 0
        assert summary2["updated"] == 1
        assert summary2["edges"] == 0  # all 6 already exist
        proposals = store.list_nodes(node_type=NodeType.SKILL, limit=10)
        assert len(proposals) == 1

    def test_rerun_with_new_session_updates_counts(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        propose_skills(store)

        # A 4th session of the SAME workflow.
        t0 = _BASE_TIME + timedelta(days=10)
        _action(store, "ping theo", "fernweh-core", "sess-d", t0)
        _action(store, "run migration", "fernweh-core", "sess-d", t0 + timedelta(minutes=1))

        summary2 = propose_skills(store)
        assert summary2["updated"] == 1
        assert summary2["edges"] == 2  # only the new session's 2 nodes are new

        proposals = store.list_nodes(node_type=NodeType.SKILL, limit=10)
        assert len(proposals) == 1
        assert proposals[0].metadata["occurrences"] == 4
        assert proposals[0].metadata["sessions"] == 4

    def test_rerun_after_accept_never_touches_label_or_content(self, store):
        """F2: once a proposal has been accepted (status != "proposed"), a
        re-run with more evidence must update ONLY the occurrences/sessions
        counters (and missing DERIVED_FROM edges) — label and content stay
        byte-identical, and status/origin/declines are untouched."""
        _seed_three_sessions(store, ["ping theo", "run migration"])
        summary = propose_skills(store)
        proposal = summary["proposals"][0]
        accepted = accept_proposal(store, proposal.node_id, actor="mara")
        assert accepted.metadata["status"] == "active"
        before_label, before_content = accepted.label, accepted.content

        # More evidence: a 4th, 5th, 6th session of the SAME pattern.
        for i, sess in enumerate(("sess-d", "sess-e", "sess-f")):
            t0 = _BASE_TIME + timedelta(days=10 + i)
            _action(store, "ping theo", "fernweh-core", sess, t0)
            _action(store, "run migration", "fernweh-core", sess, t0 + timedelta(minutes=1))

        summary2 = propose_skills(store)
        assert summary2["updated"] == 1

        after = store.get_node(proposal.node_id)
        assert after.label == before_label
        assert after.content == before_content
        assert after.metadata["status"] == "active"
        assert after.metadata["origin"] == "engine"
        assert after.metadata["declines"] == 0
        assert after.metadata["occurrences"] == 6
        assert after.metadata["sessions"] == 6

        history = store.get_node_audit(proposal.node_id)
        propose_entries = [h for h in history if h["op"] == "skill_propose"]
        assert len(propose_entries) == 2  # initial create + the frozen update
        last = propose_entries[-1]
        assert last["before"]["metadata"]["status"] == "active"
        assert last["after"]["metadata"]["status"] == "active"
        assert last["before"]["label"] == last["after"]["label"]
        assert last["before"]["content"] == last["after"]["content"]


# ── accept / decline ───────────────────────────────────────────────────────

class TestAcceptDecline:
    def _propose(self, store):
        _seed_three_sessions(store, ["ping theo", "run migration"])
        summary = propose_skills(store)
        return summary["proposals"][0]

    def test_accept_sets_active_origin_stays_engine(self, store):
        node = self._propose(store)
        updated = accept_proposal(store, node.node_id)

        assert updated.metadata["status"] == "active"
        assert updated.metadata["origin"] == "engine"

        history = store.get_node_audit(node.node_id)
        assert "skill_accept" in [h["op"] for h in history]
        accept_entry = next(h for h in history if h["op"] == "skill_accept")
        assert accept_entry["before"]["metadata"]["status"] == "proposed"
        assert accept_entry["after"]["metadata"]["status"] == "active"

    def test_decline_increments_and_audits(self, store):
        node = self._propose(store)
        updated = decline_proposal(store, node.node_id)

        assert updated.metadata["declines"] == 1
        assert updated.invalidated_at is None
        history = [h["op"] for h in store.get_node_audit(node.node_id)]
        assert history.count("skill_decline") == 1

    def test_third_decline_invalidates(self, store):
        node = self._propose(store)
        decline_proposal(store, node.node_id)
        decline_proposal(store, node.node_id)
        updated = decline_proposal(store, node.node_id)

        assert updated.metadata["declines"] == 3
        assert updated.invalidated_at is not None

        ops_log = [h["op"] for h in store.get_node_audit(node.node_id)]
        assert ops_log.count("skill_decline") == 3
        assert "invalidate" in ops_log

    def test_propose_rerun_does_not_resurrect_invalidated(self, store):
        node = self._propose(store)
        for _ in range(3):
            decline_proposal(store, node.node_id)

        propose_skills(store)  # same pattern still present in the graph
        refreshed = store.get_node(node.node_id)
        assert refreshed.invalidated_at is not None

    @pytest.mark.parametrize("fn", [accept_proposal, decline_proposal], ids=["accept", "decline"])
    @pytest.mark.parametrize(
        "state,match",
        [
            ("wrong_type", None),
            ("missing", None),
            ("already_active", "proposed"),
            ("invalidated", None),
        ],
    )
    def test_accept_decline_refusals(self, store, fn, state, match):
        """accept_proposal/decline_proposal both refuse (ValueError) against
        the same four non-live states: not a SKILL node, missing node,
        already active (message names the required "proposed" status), and
        3x-declined/invalidated."""
        if state == "wrong_type":
            fact = store.add_node(Node(
                node_type=NodeType.FACT, label="not a skill", content="x",
                source_type=SourceType.EXTRACTED, confidence=1.0,
            ))
            node_id = fact.node_id
        elif state == "missing":
            node_id = "does-not-exist"
        elif state == "already_active":
            node = self._propose(store)
            accept_proposal(store, node.node_id)
            node_id = node.node_id
        else:  # invalidated
            node = self._propose(store)
            for _ in range(3):
                decline_proposal(store, node.node_id)
            node_id = node.node_id

        with pytest.raises(ValueError, match=match):
            fn(store, node_id)


# ── recall: skill_proposals field ─────────────────────────────────────────

class TestRecallSkillProposals:
    def _draft_proposal(self, store, steps, label="proposed: ping theo -> sync fernweh"):
        return store.add_node(Node(
            node_type=NodeType.SKILL,
            label=label,
            content="## Steps\n\n1. ping theo\n2. sync fernweh",
            metadata={
                "origin": "engine", "status": "proposed", "pattern_hash": "abc123",
                "occurrences": 3, "sessions": 3, "declines": 0,
                "steps": steps, "draft": True,
            },
            source_type=SourceType.INFERRED, confidence=0.5,
            project_key="fernweh-core",
        ))

    def _skeleton_proposal(self, store, steps):
        return store.add_node(Node(
            node_type=NodeType.SKILL,
            label="proposed: ping theo -> sync fernweh",
            content="## Steps\n\n1. ping theo\n2. sync fernweh",
            metadata={
                "origin": "engine", "status": "proposed", "pattern_hash": "def456",
                "occurrences": 3, "sessions": 3, "declines": 0,
                "steps": steps, "draft": False,
            },
            source_type=SourceType.INFERRED, confidence=0.5,
            project_key="fernweh-core",
        ))

    def test_matching_proposals_requires_draft_true(self, store):
        self._draft_proposal(store, ["ping theo", "sync fernweh branches"])
        self._skeleton_proposal(store, ["ping theo", "sync fernweh branches"])

        results = matching_proposals(store, "sync the fernweh branches please")
        assert len(results) == 1
        assert results[0]["steps"] == "ping theo -> sync fernweh branches"

    def test_matching_proposals_no_keyword_overlap(self, store):
        self._draft_proposal(store, ["ping theo", "sync fernweh branches"])
        assert matching_proposals(store, "completely unrelated topic") == []

    def test_matching_proposals_uniform_row_keys(self, store):
        self._draft_proposal(store, ["ping theo", "sync fernweh branches"])
        results = matching_proposals(store, "fernweh branches")
        assert set(results[0].keys()) == {
            "node_id", "label", "occurrences", "sessions", "project_key", "steps",
        }

    def test_recall_response_always_carries_the_field(self, store):
        store.add_node(Node(
            node_type=NodeType.FACT, label="Fernweh branches use trunk-based dev",
            content="Fernweh-Core branches follow trunk-based development.",
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))
        self._draft_proposal(store, ["ping theo", "sync fernweh branches"])
        engine = RetrievalEngine(store)

        matched = engine.recall("sync fernweh branches")
        assert matched.skill_proposals != []

        unmatched = engine.recall("something else entirely")
        assert unmatched.skill_proposals == []


# ── daemon: /v1/skills ─────────────────────────────────────────────────────

@pytest.fixture
def daemon_client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    app = create_app(db_path=path)
    with TestClient(app) as c:
        yield c
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover - Windows WAL handle race
        pass


class TestSkillsDaemonRoutes:
    def test_get_skills_open_on_loopback(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        propose_skills(store)

        resp = daemon_client.get("/v1/skills")
        assert resp.status_code == 200
        rows = resp.json()
        assert any((r["metadata"] or {}).get("origin") == "engine" for r in rows)

    def test_get_skill_by_id(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        summary = propose_skills(store)
        node_id = summary["proposals"][0].node_id

        resp = daemon_client.get(f"/v1/skills/{node_id}")
        assert resp.status_code == 200
        assert resp.json()["node_id"] == node_id

    def test_get_skill_missing_404(self, daemon_client):
        resp = daemon_client.get("/v1/skills/does-not-exist")
        assert resp.status_code == 404

    def test_accept_open_on_loopback(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        resp = daemon_client.post(f"/v1/skills/{node_id}/accept")
        assert resp.status_code == 200
        assert resp.json()["metadata"]["status"] == "active"

    def test_decline_open_on_loopback(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        resp = daemon_client.post(f"/v1/skills/{node_id}/decline")
        assert resp.status_code == 200
        assert resp.json()["metadata"]["declines"] == 1

    def test_accept_remote_without_token_403(self, daemon_client, monkeypatch):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        # Force the remote-caller branch even though TestClient reports
        # "testclient" as the host (loopback-exempt by default) — same
        # technique as require_mutation_auth's own unit tests, applied at
        # the endpoint level to prove the route actually calls the gate.
        monkeypatch.setattr(server_module, "_LOOPBACK_HOSTS", set())

        resp = daemon_client.post(f"/v1/skills/{node_id}/accept")
        assert resp.status_code == 403

    def test_accept_remote_with_correct_token_200(self, daemon_client, monkeypatch):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        monkeypatch.setattr(server_module, "_LOOPBACK_HOSTS", set())
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "s3cret-pairing-token")

        resp = daemon_client.post(
            f"/v1/skills/{node_id}/accept",
            headers={"Authorization": "Bearer s3cret-pairing-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["metadata"]["status"] == "active"

    def test_decline_remote_without_token_403(self, daemon_client, monkeypatch):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        monkeypatch.setattr(server_module, "_LOOPBACK_HOSTS", set())
        resp = daemon_client.post(f"/v1/skills/{node_id}/decline")
        assert resp.status_code == 403

    def test_accept_already_active_is_409(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        first = daemon_client.post(f"/v1/skills/{node_id}/accept")
        assert first.status_code == 200
        second = daemon_client.post(f"/v1/skills/{node_id}/accept")
        assert second.status_code == 409

    def test_decline_already_active_is_409(self, daemon_client):
        store = daemon_client.app.state.store
        _seed_three_sessions(store, ["ping theo", "run migration"])
        node_id = propose_skills(store)["proposals"][0].node_id

        daemon_client.post(f"/v1/skills/{node_id}/accept")
        resp = daemon_client.post(f"/v1/skills/{node_id}/decline")
        assert resp.status_code == 409


# ── F6: ASCII-only labels/output (Windows console crash on U+2192) ────────


class TestAsciiArrowInsteadOfUnicode:
    def test_arrow_constant_is_ascii(self):
        from revien.skills import proposals as proposals_module
        assert proposals_module.ARROW == " -> "
        assert proposals_module.ARROW.isascii()

    def test_cli_skills_propose_output_is_pure_ascii(self, store):
        """Run the actual CLI command (CliRunner) and assert the full
        output stream encodes as pure ASCII — the concrete Windows console
        failure mode is a UnicodeEncodeError on cp1252/cp437 stdout."""
        from click.testing import CliRunner
        from revien.cli import main

        _seed_three_sessions(store, ["ping theo", "sync fernweh branches"])
        store.close()

        runner = CliRunner()
        result = runner.invoke(main, ["skills", "propose", "--db", store.db_path])
        assert result.exit_code == 0, result.output
        result.output.encode("ascii")  # must not raise
        assert "->" in result.output

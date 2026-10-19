"""
Usage signal (N1): recall episodes from real-shape Claude Code transcripts,
the evidence gathered for each recalled memory, and the observer's ledger,
shadow and apply behavior.

The judgment rule itself (revien/usage/judge.py) has its own contract in
tests/test_usage_judge.py; every test here injects a stand-in rule so the
plumbing is tested independently of the thresholds.
"""

import json
import os
import time

import pytest
from click.testing import CliRunner

from revien.adapters.claude_code import ClaudeCodeAdapter
from revien.cli import main
from revien.graph.schema import Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex
from revien.usage import (
    USAGE_CURSOR,
    UsageObserver,
    UseVerdict,
    distinctive_terms,
    extract_recall_episodes,
    gather_evidence,
    is_recall_tool,
    parse_recall_result,
    usage_mode,
)
from revien.usage.episodes import RecallEpisode, RecalledMemory

SOURCE_ID = "claude-code:billing:sess-1"


# ── Real-shape transcript builders ────────────────────────────────

def _prompt(text, **extra):
    return {"type": "user", "message": {"role": "user", "content": text},
            "timestamp": "2026-10-01T10:00:00Z", **extra}


def _assistant(blocks, stop_reason="tool_use", msg_id="m1", **extra):
    return {"type": "assistant",
            "message": {"id": msg_id, "role": "assistant", "content": blocks,
                        "stop_reason": stop_reason},
            **extra}


def _text(text):
    return {"type": "text", "text": text}


def _recall_call(tool_id, query, name="mcp__revien__revien_recall"):
    return {"type": "tool_use", "id": tool_id, "name": name,
            "input": {"query": query, "top_n": 5}}


def _recall_result(tool_id, memories, as_blocks=False, is_error=False):
    payload = json.dumps({
        "query": "q",
        "results": [
            {"node_id": nid, "node_type": "fact", "label": label,
             "content": content, "score": 0.9}
            for nid, label, content in memories
        ],
    })
    block = {"type": "tool_result", "tool_use_id": tool_id,
             "content": [_text(payload)] if as_blocks else payload}
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"role": "user", "content": [block]},
            "timestamp": "2026-10-01T10:00:05Z"}


def _lines(entries):
    return [json.dumps(e) for e in entries]


DB_MEMORY = (
    "n-db",
    "Chose PostgreSQL over SQLite for billing",
    "Billing API runs on PostgreSQL: concurrent writes from the worker pool "
    "ruled out SQLite.",
)
ANSWER = ("Billing is on PostgreSQL. The worker pool's concurrent writes are "
          "why SQLite was ruled out.")


def _session(answer=ANSWER, memory=DB_MEMORY, tool_id="toolu_1"):
    """Prompt -> recall -> result -> thinking + answer (end_turn)."""
    return [
        _prompt("Remind me which database the billing service uses?"),
        _assistant([_recall_call(tool_id, "billing service database")], msg_id="m1"),
        _recall_result(tool_id, [memory]),
        _assistant([{"type": "thinking", "thinking": "SECRET-THOUGHT",
                     "signature": "sig"}], stop_reason="end_turn", msg_id="m2"),
        _assistant([_text(answer)], stop_reason="end_turn", msg_id="m2"),
    ]


# ── Episodes ──────────────────────────────────────────────────────

class TestEpisodes:
    def test_closed_turn_yields_one_episode(self):
        episodes = extract_recall_episodes(_lines(_session()), SOURCE_ID)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.query == "billing service database"
        assert ep.prompt == "Remind me which database the billing service uses?"
        assert [m.node_id for m in ep.memories] == ["n-db"]
        assert ep.response == ANSWER
        assert ep.episode_key == f"{SOURCE_ID}#toolu_1"

    def test_thinking_is_never_part_of_the_answer(self):
        ep = extract_recall_episodes(_lines(_session()), SOURCE_ID)[0]
        assert "SECRET-THOUGHT" not in ep.response

    def test_turn_still_being_written_is_not_emitted(self):
        entries = _session()[:3] + [
            _assistant([_text("Let me check one more thing.")], stop_reason="tool_use",
                       msg_id="m2"),
        ]
        assert extract_recall_episodes(_lines(entries), SOURCE_ID) == []

    def test_new_prompt_closes_a_turn_without_stop_reason(self):
        entries = _session()[:3] + [
            _assistant([_text(ANSWER)], stop_reason=None, msg_id="m2"),
            _prompt("Thanks. Next question."),
        ]
        episodes = extract_recall_episodes(_lines(entries), SOURCE_ID)
        assert len(episodes) == 1 and episodes[0].response == ANSWER

    def test_end_turn_message_spanning_entries_is_collected_whole(self):
        entries = _session()[:3] + [
            _assistant([_text("First part.")], stop_reason="end_turn", msg_id="m2"),
            _assistant([_text("Second part.")], stop_reason="end_turn", msg_id="m2"),
        ]
        ep = extract_recall_episodes(_lines(entries), SOURCE_ID)[0]
        assert ep.response == "First part.\n\nSecond part."

    def test_answer_includes_text_between_tool_calls(self):
        entries = _session()[:3] + [
            _assistant([_text("Recall says PostgreSQL."),
                        {"type": "tool_use", "id": "toolu_9", "name": "Bash",
                         "input": {"command": "psql --version"}}], msg_id="m2"),
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_9",
                 "content": "psql 16.2"}]}},
            _assistant([_text("Confirmed: version 16.")], stop_reason="end_turn",
                       msg_id="m3"),
        ]
        ep = extract_recall_episodes(_lines(entries), SOURCE_ID)[0]
        assert ep.response == "Recall says PostgreSQL.\n\nConfirmed: version 16."
        assert "psql --version" not in ep.response  # tool input is not answer text

    def test_error_results_and_other_tools_are_ignored(self):
        errored = _session()
        errored[2] = _recall_result("toolu_1", [DB_MEMORY], is_error=True)
        assert extract_recall_episodes(_lines(errored), SOURCE_ID) == []

        other_tool = _session()
        other_tool[1] = _assistant([_recall_call("toolu_1", "q", name="Bash")])
        assert extract_recall_episodes(_lines(other_tool), SOURCE_ID) == []

    def test_meta_and_sidechain_entries_are_skipped(self):
        entries = _session()[:3] + [
            # Harness-injected: must not count as a human prompt closing the turn.
            _prompt("Another Claude session sent a message: ...", isMeta=True),
            # Subagent chatter: must not leak into the answer.
            _assistant([_text("SUBAGENT-TEXT")], stop_reason="end_turn",
                       msg_id="side", isSidechain=True),
        ] + _session()[3:]
        ep = extract_recall_episodes(_lines(entries), SOURCE_ID)[0]
        assert ep.response == ANSWER

    def test_two_recalls_in_one_turn_are_two_episodes(self):
        other = ("n-cache", "Cache is Redis", "Sessions are cached in Redis.")
        entries = [
            _prompt("What backs billing and sessions?"),
            _assistant([_recall_call("toolu_1", "billing database")], msg_id="m1"),
            _recall_result("toolu_1", [DB_MEMORY]),
            _assistant([_recall_call("toolu_2", "session cache")], msg_id="m2"),
            _recall_result("toolu_2", [other]),
            _assistant([_text("PostgreSQL for billing, Redis for sessions.")],
                       stop_reason="end_turn", msg_id="m3"),
        ]
        episodes = extract_recall_episodes(_lines(entries), SOURCE_ID)
        assert [e.tool_use_id for e in episodes] == ["toolu_1", "toolu_2"]

    def test_garbage_lines_do_not_break_parsing(self):
        lines = ["not json", "[1, 2]", ""] + _lines(_session())
        assert len(extract_recall_episodes(lines, SOURCE_ID)) == 1


class TestRecallResultParsing:
    def test_string_and_block_content(self):
        for as_blocks in (False, True):
            entry = _recall_result("t", [DB_MEMORY], as_blocks=as_blocks)
            content = entry["message"]["content"][0]["content"]
            assert [m.node_id for m in parse_recall_result(content)] == ["n-db"]

    def test_wrapped_and_unparseable_payloads(self):
        wrapped = json.dumps({"result": {"results": [{"node_id": "a", "label": "x"}]}})
        assert [m.node_id for m in parse_recall_result(wrapped)] == ["a"]
        assert parse_recall_result("Error executing tool revien_recall") == []
        assert parse_recall_result(json.dumps({"results": [{"label": "no id"}]})) == []

    def test_recall_tool_names(self):
        assert is_recall_tool("revien_recall")
        assert is_recall_tool("mcp__revien__revien_recall")
        assert is_recall_tool("mcp__memory__revien_recall")
        assert not is_recall_tool("mcp__revien__revien_store")
        assert not is_recall_tool("revien_recall_v2")


# ── Evidence ──────────────────────────────────────────────────────

class _VocabEmbedder:
    """Deterministic bag-of-vocabulary embedder (no model, no network)."""

    VOCAB = ["postgresql", "sqlite", "worker", "weather", "rain", "redis"]
    is_cloud = False
    dim = len(VOCAB)

    def embed(self, texts):
        return [[1.0 if w in (t or "").lower() else 0.0 for w in self.VOCAB]
                for t in texts]


def _episode(response, prompt="", query="", memories=(DB_MEMORY,)):
    return RecallEpisode(
        session_source_id=SOURCE_ID, tool_use_id="t", query=query, prompt=prompt,
        memories=[RecalledMemory(node_id=n, label=l, content=c) for n, l, c in memories],
        response=response,
    )


class TestEvidence:
    def test_distinctive_terms_drop_stopwords_and_short_words(self):
        terms = distinctive_terms("The API runs on port 8080 and it is fast")
        assert "8080" in terms  # digits always count
        assert "runs" in terms and "fast" in terms
        assert "the" not in terms and "api" not in terms and "is" not in terms

    def test_prompt_and_query_words_never_count(self):
        ep = _episode(ANSWER,
                      prompt="Which database does billing use?",
                      query="billing service database")
        (_, evidence), = gather_evidence(ep)
        assert "billing" not in evidence.shared_terms
        assert "postgresql" in evidence.shared_terms
        assert "concurrent" in evidence.shared_terms

    def test_answer_that_ignores_the_memory_shares_nothing(self):
        (_, evidence), = gather_evidence(_episode("It looks like rain tomorrow."))
        assert evidence.shared_terms == []
        assert evidence.memory_terms > 0

    def test_similarity_is_none_without_an_embedder(self):
        (_, evidence), = gather_evidence(_episode(ANSWER))
        assert evidence.similarity is None

    def test_similarity_is_the_best_paragraph_not_the_whole_answer(self):
        response = "The weather says rain.\n\nBilling runs on PostgreSQL, not SQLite."
        (_, evidence), = gather_evidence(_episode(response), _VocabEmbedder().embed)
        # Memory = {postgresql, sqlite, worker}; best paragraph = {postgresql,
        # sqlite}: cos = 2 / (sqrt(3) * sqrt(2)) ~ 0.816. Embedding the whole
        # answer at once would dilute it to 2 / (sqrt(3) * 2) ~ 0.577.
        assert evidence.similarity == pytest.approx(2 / (3 ** 0.5 * 2 ** 0.5))

    def test_a_failed_embed_leaves_similarity_none(self):
        (_, evidence), = gather_evidence(_episode(ANSWER), lambda texts: None)
        assert evidence.similarity is None


# ── Observer: ledger, shadow, apply ───────────────────────────────

def _used(evidence):
    return UseVerdict(used=True, reason="stand-in: always used")


def _unused(evidence):
    return UseVerdict(used=False, reason="stand-in: never used")


def _abstain(evidence):
    return UseVerdict(used=False, abstain=True, reason="stand-in: abstain")


@pytest.fixture
def store(tmp_path):
    s = GraphStore(db_path=str(tmp_path / "revien.db"))
    yield s
    s.close()


@pytest.fixture
def db_node(store):
    node = Node(node_id="n-db", node_type=NodeType.FACT, label=DB_MEMORY[1],
                content=DB_MEMORY[2], source_id="test",
                source_type=SourceType.EXTRACTED, confidence=1.0)
    store.add_node(node)
    return node


def _write_session(root, entries, project="billing", session="sess-1"):
    path = root / "projects" / project / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_lines(entries)) + "\n", encoding="utf-8")
    # A file modified in the same clock tick as the cursor stamp is re-read on
    # purpose; Windows' coarse clock makes that tick wide. Age the fixture so
    # cursor tests test the cursor (tests that need "newer" set it explicitly).
    old = time.time() - 60
    os.utime(path, (old, old))
    return path


@pytest.fixture
def adapter(tmp_path):
    _write_session(tmp_path, _session())
    return ClaudeCodeAdapter(session_dir=str(tmp_path / "projects"))


def _access_count(store, node_id="n-db"):
    return store.get_node(node_id).access_count


class TestObserver:
    def test_shadow_records_without_touching_the_graph(self, store, db_node, adapter):
        before = _access_count(store)
        report = UsageObserver(store, judge=_used, mode="shadow").run_claude_code(adapter)
        assert (report.files_scanned, report.episodes, report.judged) == (1, 1, 1)
        assert (report.used, report.applied) == (1, 0)
        assert _access_count(store) == before
        assert not any(a["op"] == "access" for a in store.get_node_audit("n-db"))
        (row,) = store.list_use_judgments()
        assert row["used"] and not row["applied"] and row["mode"] == "shadow"
        assert row["query"] == "billing service database"
        assert row["reason"] == "stand-in: always used"

    def test_apply_counts_a_use_exactly_once(self, store, db_node, adapter):
        engine = RetrievalEngine(store)
        before = _access_count(store)
        report = UsageObserver(store, engine=engine, judge=_used,
                               mode="apply").run_claude_code(adapter)
        assert report.applied == 1
        assert _access_count(store) == before + 1
        assert any(a["op"] == "access" for a in store.get_node_audit("n-db"))

        again = UsageObserver(store, engine=engine, judge=_used,
                              mode="apply").run_claude_code(adapter, full=True)
        assert (again.judged, again.already_judged, again.applied) == (0, 1, 0)
        assert _access_count(store) == before + 1

    def test_unused_is_recorded_and_never_applied(self, store, db_node, adapter):
        engine = RetrievalEngine(store)
        before = _access_count(store)
        report = UsageObserver(store, engine=engine, judge=_unused,
                               mode="apply").run_claude_code(adapter)
        assert (report.judged, report.unused, report.applied) == (1, 1, 0)
        assert _access_count(store) == before
        (row,) = store.list_use_judgments()
        assert not row["used"] and not row["applied"]

    def test_abstain_writes_nothing(self, store, db_node, adapter):
        report = UsageObserver(store, judge=_abstain).run_claude_code(adapter)
        assert (report.abstained, report.judged) == (1, 0)
        assert store.list_use_judgments() == []

    def test_missing_node_is_skipped(self, store, adapter):
        engine = RetrievalEngine(store)
        report = UsageObserver(store, engine=engine, judge=_used,
                               mode="apply").run_claude_code(adapter)
        assert (report.missing, report.judged, report.applied) == (1, 0, 0)
        assert store.list_use_judgments() == []

    def test_forgetting_a_memory_takes_its_judgments_with_it(
        self, store, db_node, adapter
    ):
        from revien.graph.operations import GraphOperations

        UsageObserver(store, judge=_used).run_claude_code(adapter)
        (row,) = store.list_use_judgments()
        assert row["shared_terms"]  # the memory's own words live in the ledger

        GraphOperations(store).forget_node("n-db", reason="test")
        assert store.list_use_judgments() == []

        # The transcript still holds the memory's text; a full rescan must not
        # write the forgotten words back.
        report = UsageObserver(store, judge=_used).run_claude_code(adapter, full=True)
        assert (report.missing, report.judged) == (1, 0)
        assert store.list_use_judgments() == []

    def test_grown_session_judges_only_the_new_episode(self, store, db_node, tmp_path):
        path = _write_session(tmp_path, _session())
        adapter = ClaudeCodeAdapter(session_dir=str(tmp_path / "projects"))
        UsageObserver(store, judge=_used).run_claude_code(adapter)

        _write_session(tmp_path, _session() + _session(tool_id="toolu_2"))
        future = time.time() + 60  # newer than the stored cursor
        os.utime(path, (future, future))
        report = UsageObserver(store, judge=_used).run_claude_code(adapter)
        assert (report.episodes, report.judged, report.already_judged) == (2, 1, 1)
        assert store.use_judgment_counts()["judged"] == 2

    def test_unchanged_sessions_are_not_reread(self, store, db_node, adapter):
        UsageObserver(store, judge=_used).run_claude_code(adapter)
        assert store.get_sync_cursor(USAGE_CURSOR) is not None
        report = UsageObserver(store, judge=_used).run_claude_code(adapter)
        assert report.files_scanned == 0

    def test_off_mode_does_nothing(self, store, db_node, adapter):
        report = UsageObserver(store, judge=_used, mode="off").run_claude_code(adapter)
        assert (report.files_scanned, report.judged) == (0, 0)
        assert store.get_sync_cursor(USAGE_CURSOR) is None

    def test_apply_without_an_engine_is_refused(self, store):
        with pytest.raises(ValueError, match="RetrievalEngine"):
            UsageObserver(store, mode="apply")

    def test_mode_resolution_falls_back_to_shadow(self, monkeypatch):
        assert usage_mode("APPLY") == "apply"
        assert usage_mode("bogus") == "shadow"
        monkeypatch.setenv("REVIEN_USAGE", "off")
        assert usage_mode() == "off"
        monkeypatch.setenv("REVIEN_USAGE", "aply")  # a typo never applies
        assert usage_mode() == "shadow"

    def test_semantic_layer_supplies_similarity(self, store, db_node, adapter):
        semantic = SemanticIndex(store, embedder=_VocabEmbedder(), enabled=True)
        report = UsageObserver(store, semantic=semantic,
                               judge=_used).run_claude_code(adapter)
        assert report.semantic_active
        (row,) = store.list_use_judgments()
        assert row["similarity"] is not None and row["similarity"] > 0.6

    def test_semantic_off_reports_inactive(self, store, db_node, adapter):
        report = UsageObserver(store, judge=_used).run_claude_code(adapter)
        assert not report.semantic_active
        (row,) = store.list_use_judgments()
        assert row["similarity"] is None


# ── CLI ───────────────────────────────────────────────────────────

class TestUsageCLI:
    def test_scan_then_list(self, tmp_path, store, db_node, monkeypatch):
        monkeypatch.setattr("revien.usage.observer.judge_use", _used)
        _write_session(tmp_path, _session())
        db_path = store.db_path
        store.close()
        runner = CliRunner()

        result = runner.invoke(main, ["usage", "scan", "--db", db_path,
                                      "--path", str(tmp_path / "projects")])
        assert result.exit_code == 0, result.output
        assert "1 new (1 used, 0 not used)" in result.output
        assert "shadow" in result.output

        result = runner.invoke(main, ["usage", "list", "--db", db_path])
        assert result.exit_code == 0, result.output
        assert "1 judgment(s): 1 used" in result.output
        assert DB_MEMORY[1][:40] in result.output

        result = runner.invoke(main, ["usage", "list", "--db", db_path,
                                      "--json-output"])
        payload = json.loads(result.output)
        assert payload["counts"]["used"] == 1
        assert payload["judgments"][0]["node_id"] == "n-db"

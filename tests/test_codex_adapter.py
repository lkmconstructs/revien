"""
Codex Adapter + connect installer tests.
Create mock Codex rollout JSONL logs (both verified line shapes).
Point the Codex adapter at them; verify parsing, since-filtering, health,
auto-detect via CODEX_HOME.
Then drive `revien connect codex` against a temp HOME/CODEX_HOME and verify
the config.json write and the config.toml append (idempotent, never creates).
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.adapters.codex import CodexAdapter, default_codex_home


# ── Mock Data ─────────────────────────────────────────────
# Line shapes verified against real Codex CLI rollout files (2026-05/07):
# every line is {"timestamp", "type", "payload"}; conversation lives in
# response_item lines with payload {"type": "message", "role", "content"}.

MOCK_CODEX_ROLLOUT = [
    {
        "timestamp": "2026-07-01T10:00:00.000Z",
        "type": "session_meta",
        "payload": {
            "id": "0197-test-session",
            "cwd": "C:\\Users\\dev\\my-codex-project",
            "cli_version": "0.99.0",
            "originator": "codex_cli_rs",
        },
    },
    {
        "timestamp": "2026-07-01T10:00:01.000Z",
        "type": "turn_context",
        "payload": {"cwd": "C:\\Users\\dev\\my-codex-project", "model": "gpt-5"},
    },
    {
        # Injected plumbing arrives as a user-role message — must be skipped.
        "timestamp": "2026-07-01T10:00:02.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text",
                         "text": "<environment_context>cwd: C:\\Users\\dev</environment_context>"}],
        },
    },
    {
        "timestamp": "2026-07-01T10:00:03.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text",
                         "text": "Switch the queue backend to Redis Streams, not RabbitMQ."}],
        },
    },
    {
        "timestamp": "2026-07-01T10:00:04.000Z",
        "type": "response_item",
        "payload": {"type": "reasoning", "summary": [], "content": None,
                    "encrypted_content": "opaque"},
    },
    {
        "timestamp": "2026-07-01T10:00:05.000Z",
        "type": "response_item",
        "payload": {"type": "function_call", "name": "shell",
                    "arguments": "{\"command\": [\"rm\", \"-rf\", \"queue\"]}",
                    "call_id": "call_1"},
    },
    {
        "timestamp": "2026-07-01T10:00:06.000Z",
        "type": "response_item",
        "payload": {"type": "function_call_output", "call_id": "call_1",
                    "output": "removed queue/"},
    },
    {
        "timestamp": "2026-07-01T10:00:07.000Z",
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "phase": "final",
            "content": [{"type": "output_text",
                         "text": "Done — Redis Streams is the queue backend now, consumer group per worker."}],
        },
    },
    {
        "timestamp": "2026-07-01T10:00:08.000Z",
        "type": "event_msg",
        "payload": {"type": "agent_message",
                    "message": "duplicate of the assistant response_item — skip"},
    },
]

# Older Codex versions wrote response items bare, no envelope.
MOCK_CODEX_ROLLOUT_BARE = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text",
                     "text": "Pin the tokenizer to version 0.15, the 0.16 release breaks offsets."}],
    },
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text",
                     "text": "Pinned tokenizer==0.15 in requirements."}],
    },
    {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": "system instructions — not conversation"}],
    },
]


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _write_rollout(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def mock_codex_home():
    """A fake CODEX_HOME with sessions in the real YYYY/MM/DD layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir) / "codex-home"
        _write_rollout(
            home / "sessions" / "2026" / "07" / "01" / "rollout-2026-07-01T10-00-00-abc123.jsonl",
            MOCK_CODEX_ROLLOUT,
        )
        _write_rollout(
            home / "sessions" / "2026" / "07" / "02" / "rollout-2026-07-02T09-00-00-def456.jsonl",
            MOCK_CODEX_ROLLOUT_BARE,
        )
        # A non-rollout JSONL in the tree must be ignored.
        _write_rollout(home / "sessions" / "index.jsonl", [{"type": "noise"}])
        yield home


@pytest.fixture
def epoch():
    return datetime(2020, 1, 1, tzinfo=timezone.utc)


# ── Codex Adapter Tests ───────────────────────────────────

class TestCodexAdapter:
    def test_reads_rollout_files(self, mock_codex_home, epoch):
        """Adapter should find both rollout files, and only rollout files."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        results = run_async(adapter.fetch_new_content(epoch))
        assert len(results) == 2, "Should parse exactly the two rollout files"

    def test_parses_envelope_shape(self, mock_codex_home, epoch):
        """response_item envelope: user/assistant text extracted, noise skipped."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        results = run_async(adapter.fetch_new_content(epoch))
        by_file = {r["metadata"]["session_file"]: r for r in results}
        content = by_file["rollout-2026-07-01T10-00-00-abc123.jsonl"]["content"]

        assert "User: Switch the queue backend to Redis Streams" in content
        assert "Assistant: Done — Redis Streams" in content
        # Noise classes, each verified absent:
        assert "environment_context" not in content, "Injected plumbing should be skipped"
        assert "rm" not in content, "function_call should be skipped"
        assert "removed queue/" not in content, "function_call_output should be skipped"
        assert "opaque" not in content, "reasoning should be skipped"
        assert "duplicate of the assistant" not in content, "event_msg should be skipped"

    def test_parses_bare_shape(self, mock_codex_home, epoch):
        """Bare {"type": "message"} lines (older Codex) parse too; developer role skipped."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        results = run_async(adapter.fetch_new_content(epoch))
        by_file = {r["metadata"]["session_file"]: r for r in results}
        content = by_file["rollout-2026-07-02T09-00-00-def456.jsonl"]["content"]

        assert "User: Pin the tokenizer" in content
        assert "Assistant: Pinned tokenizer==0.15" in content
        assert "system instructions" not in content, "developer role is not conversation"

    def test_project_name_from_session_meta(self, mock_codex_home, epoch):
        """Project comes from session_meta cwd; source_id is per-session
        (codex:<project>:<stem>), matching the claude_code adapter's
        granularity — sessions in one project must not share provenance."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        results = run_async(adapter.fetch_new_content(epoch))
        by_file = {r["metadata"]["session_file"]: r for r in results}

        with_meta = by_file["rollout-2026-07-01T10-00-00-abc123.jsonl"]
        assert with_meta["metadata"]["project"] == "my-codex-project"
        assert with_meta["source_id"] == (
            "codex:my-codex-project:rollout-2026-07-01T10-00-00-abc123"
        )

        # Bare file has no session_meta — project falls back to "unknown",
        # the stem still gives it a distinct per-session identity.
        bare = by_file["rollout-2026-07-02T09-00-00-def456.jsonl"]
        assert bare["source_id"] == (
            "codex:unknown:rollout-2026-07-02T09-00-00-def456"
        )

    def test_cwd_basename_is_cross_platform(self):
        """Regression (CI, 2026-07-13): session_meta cwd is stored in the
        session's NATIVE format, so a Windows cwd (C:\\...) must extract on a
        Linux host and vice-versa. Path(cwd).name is host-native and got the
        whole string on the wrong OS — split on both separators instead."""
        from revien.adapters.codex import _basename_cross_platform as b
        assert b("C:\\Users\\dev\\my-codex-project") == "my-codex-project"
        assert b("/home/user/my-project") == "my-project"
        assert b("C:\\proj\\") == "proj"      # trailing sep
        assert b("/srv/app/") == "app"
        assert b("D:/mixed/sep/proj") == "proj"  # forward slashes on Windows drive
        assert b("") == ""

    def test_produces_valid_ingestion_content(self, mock_codex_home, epoch):
        """Adapter output should have all required fields for ingestion."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        results = run_async(adapter.fetch_new_content(epoch))

        for result in results:
            assert "content" in result
            assert "content_type" in result
            assert "timestamp" in result
            assert "metadata" in result
            assert result["content_type"] == "conversation"
            assert result["metadata"]["adapter"] == "codex"

    def test_since_filter_works(self, mock_codex_home):
        """Content from before 'since' timestamp should be excluded."""
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        results = run_async(adapter.fetch_new_content(future))
        assert len(results) == 0, "Future since date should return no results"

    def test_health_check_valid_dir(self, mock_codex_home):
        adapter = CodexAdapter(session_dir=str(mock_codex_home / "sessions"))
        assert run_async(adapter.health_check()) is True

    def test_health_check_invalid_dir(self):
        adapter = CodexAdapter(session_dir="/nonexistent/path")
        assert run_async(adapter.health_check()) is False

    def test_auto_detect_honors_codex_home(self, mock_codex_home, epoch, monkeypatch):
        """With CODEX_HOME set, auto-detect must land on $CODEX_HOME/sessions."""
        monkeypatch.setenv("CODEX_HOME", str(mock_codex_home))
        assert default_codex_home() == mock_codex_home

        adapter = CodexAdapter()  # no session_dir — auto-detect
        assert adapter.session_dir == mock_codex_home / "sessions"
        results = run_async(adapter.fetch_new_content(epoch))
        assert len(results) == 2

    def test_auto_detect_missing_sessions(self, monkeypatch):
        """CODEX_HOME set but no sessions dir — adapter degrades, not crashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("CODEX_HOME", str(Path(tmpdir) / "empty-codex"))
            adapter = CodexAdapter()
            assert adapter.session_dir is None
            assert run_async(adapter.health_check()) is False
            epoch = datetime(2020, 1, 1, tzinfo=timezone.utc)
            assert run_async(adapter.fetch_new_content(epoch)) == []

    def test_garbage_lines_dont_crash(self, epoch):
        """Malformed JSON and non-dict lines are skipped, not fatal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = Path(tmpdir) / "sessions"
            rollout = sessions / "2026" / "07" / "03" / "rollout-garbage.jsonl"
            rollout.parent.mkdir(parents=True)
            rollout.write_text(
                "not json at all\n"
                "[1, 2, 3]\n"
                + json.dumps({"type": "response_item", "payload": "not-a-dict"}) + "\n"
                + json.dumps({
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": "Survived the noise."}]},
                }) + "\n",
                encoding="utf-8",
            )
            adapter = CodexAdapter(session_dir=str(sessions))
            results = run_async(adapter.fetch_new_content(epoch))
            assert len(results) == 1
            assert "Survived the noise." in results[0]["content"]


# ── CLI connect codex Tests ───────────────────────────────

@pytest.fixture
def cli_env(monkeypatch, mock_codex_home):
    """Temp HOME (for ~/.revien) + CODEX_HOME pointed at the mock. The MCP
    command resolver is pinned to the bare name so the config.toml assertions
    are deterministic and machine-independent (its real behavior — resolving an
    absolute exe path — is unit-tested separately in TestResolveRevienCommand)."""
    import revien.cli as _cli
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_home = Path(tmpdir) / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setenv("CODEX_HOME", str(mock_codex_home))
        monkeypatch.setattr(_cli, "_resolve_revien_command", lambda: "revien")
        yield fake_home


class TestConnectCodex:
    def _run(self, args=("connect", "codex")):
        from revien.cli import main
        runner = CliRunner()
        return runner.invoke(main, list(args))

    def test_writes_adapter_config(self, cli_env, mock_codex_home):
        result = self._run()
        assert result.exit_code == 0, result.output

        config = json.loads((cli_env / ".revien" / "config.json").read_text())
        assert config["adapters"]["codex"]["type"] == "codex"
        assert config["adapters"]["codex"]["session_dir"] == str(mock_codex_home / "sessions")

    def test_appends_mcp_block_once(self, cli_env, mock_codex_home):
        """config.toml exists: append [mcp_servers.revien] exactly once — run twice."""
        config_toml = mock_codex_home / "config.toml"
        config_toml.write_text('model = "gpt-5"\n', encoding="utf-8")

        first = self._run()
        assert first.exit_code == 0, first.output
        second = self._run()
        assert second.exit_code == 0, second.output

        text = config_toml.read_text(encoding="utf-8")
        assert text.count("[mcp_servers.revien]") == 1, "Append must be idempotent"
        assert text.startswith('model = "gpt-5"'), "Existing config must survive untouched"
        # TOML literal string (single quotes) — resolver pinned to bare name here.
        assert "command = 'revien'" in text
        assert 'args = ["mcp"]' in text
        assert "already present" in second.output

    def test_absent_toml_prints_instead_of_creating(self, cli_env, mock_codex_home):
        """No config.toml: never create it — print the block to paste."""
        config_toml = mock_codex_home / "config.toml"
        assert not config_toml.exists()

        result = self._run()
        assert result.exit_code == 0, result.output
        assert not config_toml.exists(), "connect must not create config.toml"
        assert "[mcp_servers.revien]" in result.output, "Block should be printed for pasting"

    def test_prints_agents_md_snippet(self, cli_env):
        result = self._run()
        assert result.exit_code == 0, result.output
        assert "AGENTS.md" in result.output
        assert "revien_recall" in result.output
        assert "revien_store" in result.output

    def test_explicit_path_wins(self, cli_env):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom = Path(tmpdir) / "elsewhere"
            custom.mkdir()
            result = self._run(("connect", "codex", "--path", str(custom)))
            assert result.exit_code == 0, result.output
            config = json.loads((cli_env / ".revien" / "config.json").read_text())
            assert config["adapters"]["codex"]["session_dir"] == str(custom)

    def test_commented_out_block_does_not_count_as_present(
        self, cli_env, mock_codex_home
    ):
        """A commented-out [mcp_servers.revien] must not block the append —
        presence means an ACTIVE table header, not substring containment."""
        config_toml = mock_codex_home / "config.toml"
        config_toml.write_text(
            'model = "gpt-5"\n# [mcp_servers.revien]\n# command = "old"\n',
            encoding="utf-8",
        )
        result = self._run()
        assert result.exit_code == 0, result.output
        text = config_toml.read_text(encoding="utf-8")
        active = [
            ln for ln in text.splitlines() if ln.strip() == "[mcp_servers.revien]"
        ]
        assert len(active) == 1, "Active block must be appended past the comment"

    def test_similarly_named_table_does_not_count_as_present(
        self, cli_env, mock_codex_home
    ):
        config_toml = mock_codex_home / "config.toml"
        config_toml.write_text(
            '[mcp_servers.revien-staging]\ncommand = "other"\n', encoding="utf-8"
        )
        result = self._run()
        assert result.exit_code == 0, result.output
        text = config_toml.read_text(encoding="utf-8")
        assert "[mcp_servers.revien]" in text
        assert "[mcp_servers.revien-staging]" in text, "Neighbor table untouched"

    def test_utf16_toml_left_untouched_not_crashed(self, cli_env, mock_codex_home):
        """PowerShell 5.1 default encoding: connect must neither crash nor
        append UTF-8 bytes into a UTF-16 file — print the block instead."""
        config_toml = mock_codex_home / "config.toml"
        config_toml.write_text('model = "gpt-5"\n', encoding="utf-16")
        before = config_toml.read_bytes()

        result = self._run()
        assert result.exit_code == 0, result.output
        assert config_toml.read_bytes() == before, "Non-UTF-8 file must be untouched"
        assert "[mcp_servers.revien]" in result.output, "Block printed for pasting"


class TestResolveRevienCommand:
    """The MCP command an installer writes for a client (Codex) to spawn. A
    user-site pip install often leaves revien.exe off PATH, so the resolver
    must find the absolute path rather than emit a bare name that won't spawn."""

    def test_prefers_which_when_on_path(self, monkeypatch):
        import revien.cli as cli
        monkeypatch.setattr(cli.shutil, "which", lambda n: r"C:\bin\revien.exe")
        assert cli._resolve_revien_command() == r"C:\bin\revien.exe"

    def test_finds_scripts_dir_when_not_on_path(self, monkeypatch):
        import revien.cli as cli
        monkeypatch.setattr(cli.shutil, "which", lambda n: None)
        name = "revien.exe" if os.name == "nt" else "revien"
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / name).write_text("", encoding="utf-8")
            monkeypatch.setattr(cli.sysconfig, "get_path", lambda p, s=None: tmpdir)
            assert cli._resolve_revien_command() == str(Path(tmpdir) / name)

    def test_falls_back_to_bare_name(self, monkeypatch):
        import revien.cli as cli
        monkeypatch.setattr(cli.shutil, "which", lambda n: None)
        monkeypatch.setattr(cli.sysconfig, "get_path", lambda p, s=None: "/nonexistent")
        assert cli._resolve_revien_command() == "revien"


# ── connect → start actually syncs (the registry) ──────────


class TestAdapterWiring:
    def test_build_adapter_from_config_known_types(self):
        from revien.adapters import (
            ClaudeCodeAdapter,
            CodexAdapter,
            FileWatcherAdapter,
            ObsidianVaultAdapter,
            build_adapter_from_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            assert isinstance(
                build_adapter_from_config(
                    {"type": "claude_code", "session_dir": tmpdir}
                ),
                ClaudeCodeAdapter,
            )
            assert isinstance(
                build_adapter_from_config({"type": "codex", "session_dir": tmpdir}),
                CodexAdapter,
            )
            assert isinstance(
                build_adapter_from_config(
                    {"type": "file_watcher", "watch_dir": tmpdir}
                ),
                FileWatcherAdapter,
            )
            assert isinstance(
                build_adapter_from_config({"type": "obsidian", "vault_dir": tmpdir}),
                ObsidianVaultAdapter,
            )

    def test_build_adapter_from_config_bad_entries(self):
        from revien.adapters import build_adapter_from_config

        assert build_adapter_from_config(None) is None
        assert build_adapter_from_config({}) is None
        assert build_adapter_from_config({"type": "nonsense"}) is None
        # Known type, missing required key — skipped, not raised.
        assert build_adapter_from_config({"type": "file_watcher"}) is None

    def test_sync_endpoint_runs_registered_adapters(self, mock_codex_home):
        """POST /v1/sync with a live scheduler ingests adapter content —
        the 'connect -> start -> sync' promise, end to end."""
        from fastapi.testclient import TestClient

        from revien.adapters import build_adapter_from_config
        from revien.daemon.scheduler import SyncScheduler
        from revien.daemon.server import create_app

        fd, db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            app = create_app(db_path=db)
            scheduler = SyncScheduler(pipeline=app.state.pipeline)
            adapter = build_adapter_from_config(
                {"type": "codex", "session_dir": str(mock_codex_home / "sessions")}
            )
            # Backdate last-sync so the mock sessions count as new.
            scheduler.register_adapter("codex", adapter)
            scheduler._last_sync["codex"] = datetime(2020, 1, 1, tzinfo=timezone.utc)
            app.state.scheduler = scheduler

            with TestClient(app) as client:
                before = client.get("/v1/health").json()["node_count"]
                resp = client.post("/v1/sync")
                assert resp.status_code == 200
                assert "item(s) ingested" in resp.json()["message"]
                after = client.get("/v1/health").json()["node_count"]
                assert after > before, "sync must ingest the adapter's content"
        finally:
            try:
                os.unlink(db)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

    def test_sync_endpoint_honest_without_adapters(self):
        from fastapi.testclient import TestClient

        from revien.daemon.server import create_app

        fd, db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            app = create_app(db_path=db)
            with TestClient(app) as client:
                resp = client.post("/v1/sync")
                assert resp.status_code == 200
                assert "No adapters registered" in resp.json()["message"]
        finally:
            try:
                os.unlink(db)
            except PermissionError:  # pragma: no cover - Windows WAL race
                pass

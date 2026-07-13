"""
LEG P4 — agent on-ramp batch tests.

1. CLAUDE.md snippet printed by `revien connect claude-code`
2. MCP client installers: connect cursor|windsurf|cline|continue|vscode
3. revien export <file> / revien import <file> round-trip
4. revien watch — interval snapshots, retention, gzip

Temp-HOME pattern from test_codex_adapter.py: Path.home monkeypatch +
tempfile (pytest tmp_path basetemp is broken on this box). Db files get the
mkdtemp + rmtree(ignore_errors=True) treatment — Windows WAL teardown races
are known-environmental.
"""

import gzip
import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.mcp_install import MCP_ENTRY, config_path_for


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def fake_home(monkeypatch):
    """Temp HOME; APPDATA redirected too so the VS Code / Cline paths
    resolve inside it."""
    tmpdir = tempfile.mkdtemp()
    try:
        home = Path(tmpdir) / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
        yield home
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _run(*args):
    from revien.cli import main
    return CliRunner().invoke(main, list(args))


def _make_db(db_path, n_nodes=3):
    """A small real graph: n fact nodes + one edge between the first two."""
    from revien.graph.schema import Edge, EdgeType, Node, NodeType
    from revien.graph.store import GraphStore

    store = GraphStore(db_path=str(db_path))
    nodes = []
    edge = None
    try:
        for i in range(n_nodes):
            node = Node(
                node_type=NodeType.FACT,
                label=f"fact {i}",
                content=f"Fact number {i} about the test graph.",
                source_id="test-onramp",
            )
            store.add_node(node)
            nodes.append(node)
        if n_nodes >= 2:
            edge = Edge(
                edge_type=EdgeType.RELATED_TO,
                source_node_id=nodes[0].node_id,
                target_node_id=nodes[1].node_id,
            )
            store.add_edge(edge)
    finally:
        store.close()
    return nodes, edge


# ── Item 1: CLAUDE.md snippet on connect claude-code ──────

class TestClaudeCodeSnippet:
    def test_snippet_printed_never_written(self, fake_home, tmp_dir):
        session_dir = tmp_dir / "claude-sessions"
        session_dir.mkdir()
        result = _run("connect", "claude-code", "--path", str(session_dir))
        assert result.exit_code == 0, result.output

        assert "CLAUDE.md" in result.output
        assert "revien_recall" in result.output, "active recall instruction"
        assert "revien_store" in result.output, "silent store instruction"
        assert "conversation start" in result.output
        assert "built-in memory" in result.output, "prefer-Revien instruction"
        # Printed, never written: no CLAUDE.md anywhere in the fake home.
        assert not list(fake_home.rglob("CLAUDE.md"))

    def test_snippet_under_ten_lines(self, fake_home, tmp_dir):
        session_dir = tmp_dir / "claude-sessions"
        session_dir.mkdir()
        result = _run("connect", "claude-code", "--path", str(session_dir))
        lines = result.output.splitlines()
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("## Memory (Revien)"))
        end = next(i for i, l in enumerate(lines)
                   if "built-in memory" in l)
        assert end - start + 1 < 10, "snippet block must stay under 10 lines"


# ── Item 2: MCP client installers ─────────────────────────

class TestCursorInstaller:
    """Cursor docs say 'Create ~/.cursor/mcp.json' — creation sanctioned."""

    def test_creates_when_missing(self, fake_home):
        result = _run("connect", "cursor")
        assert result.exit_code == 0, result.output
        cfg = fake_home / ".cursor" / "mcp.json"
        assert cfg.exists(), "Cursor's docs sanction creating the file"
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["mcpServers"]["revien"] == MCP_ENTRY
        assert "Created" in result.output

    def test_idempotent(self, fake_home):
        first = _run("connect", "cursor")
        assert first.exit_code == 0, first.output
        second = _run("connect", "cursor")
        assert second.exit_code == 0, second.output
        assert "already present" in second.output
        cfg = fake_home / ".cursor" / "mcp.json"
        # The server KEY appears once ('"revien":' — the bare string also
        # matches the "command": "revien" value, so count the key form).
        assert cfg.read_text(encoding="utf-8").count('"revien":') == 1

    def test_merges_preserving_existing_servers(self, fake_home):
        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({
            "mcpServers": {"github": {"command": "npx",
                                      "args": ["-y", "server-github"]}}
        }), encoding="utf-8")

        result = _run("connect", "cursor")
        assert result.exit_code == 0, result.output
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["mcpServers"]["github"]["command"] == "npx", \
            "existing servers must survive the merge"
        assert data["mcpServers"]["revien"] == MCP_ENTRY
        # Pretty-printed on write-back.
        assert '\n  "mcpServers"' in cfg.read_text(encoding="utf-8")

    def test_unparseable_json_refused_with_snippet(self, fake_home):
        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("{ this is not json", encoding="utf-8")
        before = cfg.read_bytes()

        result = _run("connect", "cursor")
        assert result.exit_code == 0, result.output
        assert cfg.read_bytes() == before, "unparseable file must be untouched"
        assert '"mcpServers"' in result.output, "paste-block printed"

    def test_utf16_refused_untouched(self, fake_home):
        """PowerShell 5.1 default: never rewrite a UTF-16 user file."""
        cfg = fake_home / ".cursor" / "mcp.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-16")
        before = cfg.read_bytes()

        result = _run("connect", "cursor")
        assert result.exit_code == 0, result.output
        assert cfg.read_bytes() == before
        assert '"mcpServers"' in result.output, "paste-block printed"


class TestWindsurfInstaller:
    """Windsurf docs cover EDITING mcp_config.json, not creating it."""

    def test_missing_not_created(self, fake_home):
        result = _run("connect", "windsurf")
        assert result.exit_code == 0, result.output
        cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
        assert not cfg.exists(), "must not create Windsurf's config"
        assert '"mcpServers"' in result.output, "paste-block printed"

    def test_merges_existing(self, fake_home):
        cfg = fake_home / ".codeium" / "windsurf" / "mcp_config.json"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

        result = _run("connect", "windsurf")
        assert result.exit_code == 0, result.output
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["mcpServers"]["revien"] == MCP_ENTRY


class TestClineInstaller:
    """cline_mcp_settings.json is created by the Cline extension."""

    def _cfg(self, fake_home):
        return (fake_home / "AppData" / "Roaming" / "Code" / "User"
                / "globalStorage" / "saoudrizwan.claude-dev" / "settings"
                / "cline_mcp_settings.json")

    def test_path_resolution(self, fake_home):
        assert config_path_for("cline") == self._cfg(fake_home)

    def test_missing_not_created(self, fake_home):
        result = _run("connect", "cline")
        assert result.exit_code == 0, result.output
        assert not self._cfg(fake_home).exists()
        assert "Cline" in result.output
        assert '"mcpServers"' in result.output, "paste-block printed"

    def test_merges_existing(self, fake_home):
        cfg = self._cfg(fake_home)
        cfg.parent.mkdir(parents=True)
        cfg.write_text(json.dumps(
            {"mcpServers": {"other": {"command": "x", "args": []}}}
        ), encoding="utf-8")

        result = _run("connect", "cline")
        assert result.exit_code == 0, result.output
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["mcpServers"]["revien"] == MCP_ENTRY
        assert "other" in data["mcpServers"], "existing servers preserved"


class TestVSCodeInstaller:
    """User-profile mcp.json; top-level key is 'servers', NOT 'mcpServers'."""

    def _user_dir(self, fake_home):
        return fake_home / "AppData" / "Roaming" / "Code" / "User"

    def test_no_user_dir_refused(self, fake_home):
        result = _run("connect", "vscode")
        assert result.exit_code == 0, result.output
        assert not (self._user_dir(fake_home) / "mcp.json").exists()
        assert not self._user_dir(fake_home).exists(), \
            "must not fabricate a VS Code install"
        assert '"servers"' in result.output, "paste-block printed"

    def test_creates_in_existing_user_dir(self, fake_home):
        self._user_dir(fake_home).mkdir(parents=True)
        result = _run("connect", "vscode")
        assert result.exit_code == 0, result.output
        cfg = self._user_dir(fake_home) / "mcp.json"
        assert cfg.exists()
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["servers"]["revien"] == MCP_ENTRY
        assert "mcpServers" not in data, "VS Code uses 'servers'"

    def test_merges_existing(self, fake_home):
        user_dir = self._user_dir(fake_home)
        user_dir.mkdir(parents=True)
        cfg = user_dir / "mcp.json"
        cfg.write_text(json.dumps(
            {"servers": {"github": {"type": "http", "url": "https://x"}}}
        ), encoding="utf-8")

        result = _run("connect", "vscode")
        assert result.exit_code == 0, result.output
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["servers"]["revien"] == MCP_ENTRY
        assert data["servers"]["github"]["url"] == "https://x"


class TestContinueInstaller:
    """~/.continue/config.yaml is auto-generated by Continue — never
    created here; append-only when no mcpServers block exists."""

    def _cfg(self, fake_home):
        return fake_home / ".continue" / "config.yaml"

    def test_missing_not_created(self, fake_home):
        result = _run("connect", "continue")
        assert result.exit_code == 0, result.output
        assert not self._cfg(fake_home).exists()
        assert "- name: revien" in result.output, "paste-block printed"

    def test_appends_block_when_no_mcp_servers(self, fake_home):
        cfg = self._cfg(fake_home)
        cfg.parent.mkdir(parents=True)
        original = "# my continue config\nmodels:\n  - name: gpt\n"
        cfg.write_text(original, encoding="utf-8")

        result = _run("connect", "continue")
        assert result.exit_code == 0, result.output
        text = cfg.read_text(encoding="utf-8")
        assert text.startswith(original), \
            "append-only: existing content (incl. comments) survives byte-identical"

        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(text)
        assert {"name": "revien", "command": "revien", "args": ["mcp"]} \
            in parsed["mcpServers"]

    def test_append_is_idempotent(self, fake_home):
        cfg = self._cfg(fake_home)
        cfg.parent.mkdir(parents=True)
        cfg.write_text("models: []\n", encoding="utf-8")

        first = _run("connect", "continue")
        assert first.exit_code == 0, first.output
        second = _run("connect", "continue")
        assert second.exit_code == 0, second.output
        assert "already present" in second.output
        assert cfg.read_text(encoding="utf-8").count("- name: revien") == 1

    def test_existing_mcp_servers_block_not_rewritten(self, fake_home):
        """A YAML rewrite would eat user comments — hand over the entry."""
        cfg = self._cfg(fake_home)
        cfg.parent.mkdir(parents=True)
        original = ("mcpServers:\n"
                    "  - name: other  # keep me\n"
                    "    command: foo\n")
        cfg.write_text(original, encoding="utf-8")

        result = _run("connect", "continue")
        assert result.exit_code == 0, result.output
        assert cfg.read_text(encoding="utf-8") == original, "file untouched"
        assert "- name: revien" in result.output, "entry printed for manual add"


# ── Item 3: export / import round-trip ────────────────────

class TestExportImport:
    def test_roundtrip_counts_and_sampled_node(self, fake_home, tmp_dir):
        from revien.graph.store import GraphStore

        db1 = tmp_dir / "source.db"
        nodes, edge = _make_db(db1, n_nodes=3)

        out = tmp_dir / "graph.json"
        result = _run("export", str(out), "--db", str(db1))
        assert result.exit_code == 0, result.output
        assert out.exists()

        # Exported payload is the /v1/graph schema: top-level nodes + edges.
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert set(payload.keys()) >= {"nodes", "edges"}
        assert len(payload["nodes"]) == 3
        assert len(payload["edges"]) == 1

        db2 = tmp_dir / "fresh.db"
        result = _run("import", str(out), "--db", str(db2))
        assert result.exit_code == 0, result.output

        store = GraphStore(db_path=str(db2))
        try:
            assert store.count_nodes() == 3
            assert store.count_edges() == 1
            sample = store.get_node(nodes[0].node_id)
            assert sample is not None
            assert sample.node_id == nodes[0].node_id
            assert sample.label == nodes[0].label
            assert sample.content == nodes[0].content
            assert sample.node_type == nodes[0].node_type
            assert sample.source_id == nodes[0].source_id
            assert store.get_edge(edge.edge_id) is not None
        finally:
            store.close()

    def test_import_refuses_nonempty_without_merge(self, fake_home, tmp_dir):
        from revien.graph.store import GraphStore

        db1 = tmp_dir / "source.db"
        _make_db(db1, n_nodes=2)
        out = tmp_dir / "graph.json"
        assert _run("export", str(out), "--db", str(db1)).exit_code == 0

        db2 = tmp_dir / "occupied.db"
        _make_db(db2, n_nodes=1)

        result = _run("import", str(out), "--db", str(db2))
        assert result.exit_code != 0, "non-empty target must refuse"
        assert "not empty" in result.output
        assert "--merge" in result.output

        store = GraphStore(db_path=str(db2))
        try:
            assert store.count_nodes() == 1, "refusal must not touch the db"
        finally:
            store.close()

    def test_merge_adds_and_skips_duplicates(self, fake_home, tmp_dir):
        from revien.graph.store import GraphStore

        db1 = tmp_dir / "source.db"
        _make_db(db1, n_nodes=3)
        out = tmp_dir / "graph.json"
        assert _run("export", str(out), "--db", str(db1)).exit_code == 0

        db2 = tmp_dir / "target.db"
        _make_db(db2, n_nodes=1)  # distinct uuids — nothing shared

        result = _run("import", str(out), "--db", str(db2), "--merge")
        assert result.exit_code == 0, result.output
        store = GraphStore(db_path=str(db2))
        try:
            assert store.count_nodes() == 4  # 1 existing + 3 imported
        finally:
            store.close()

        # Merge again: everything already there — skipped, not duplicated.
        result = _run("import", str(out), "--db", str(db2), "--merge")
        assert result.exit_code == 0, result.output
        assert "skipped 3 existing nodes" in result.output
        store = GraphStore(db_path=str(db2))
        try:
            assert store.count_nodes() == 4
            assert store.count_edges() == 1, "edge merged once, then skipped"
        finally:
            store.close()

    def test_import_rejects_garbage_file(self, fake_home, tmp_dir):
        bad = tmp_dir / "bad.json"
        bad.write_text("this is not a graph", encoding="utf-8")
        result = _run("import", str(bad), "--db", str(tmp_dir / "x.db"))
        assert result.exit_code != 0


# ── Item 4: revien watch ──────────────────────────────────

class TestWatch:
    def test_snapshot_created_and_loadable(self, tmp_dir):
        from revien.graph.store import GraphStore
        from revien.watch import snapshot_db, snapshot_dir_for

        db = tmp_dir / "mem.db"
        _make_db(db, n_nodes=3)

        snap = snapshot_db(db)
        assert snap.exists()
        assert snap.parent == snapshot_dir_for(db)
        assert snap.suffix == ".db"

        store = GraphStore(db_path=str(snap))
        try:
            assert store.count_nodes() == 3, "snapshot must be a loadable GraphStore"
            assert store.count_edges() == 1
        finally:
            store.close()

    def test_gzip_snapshot_loadable(self, tmp_dir):
        from revien.graph.store import GraphStore
        from revien.watch import snapshot_db

        db = tmp_dir / "mem.db"
        _make_db(db, n_nodes=2)

        snap = snapshot_db(db, use_gzip=True)
        assert snap.name.endswith(".db.gz")
        assert not Path(str(snap)[:-3]).exists(), "uncompressed copy removed"

        restored = tmp_dir / "restored.db"
        with gzip.open(snap, "rb") as f_in, open(restored, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        store = GraphStore(db_path=str(restored))
        try:
            assert store.count_nodes() == 2
        finally:
            store.close()

    def test_retention_prunes_oldest(self, tmp_dir):
        from revien.watch import prune_snapshots, snapshot_db, snapshot_dir_for

        db = tmp_dir / "mem.db"
        _make_db(db, n_nodes=1)

        taken = [snapshot_db(db) for _ in range(5)]
        snap_dir = snapshot_dir_for(db)
        removed = prune_snapshots(snap_dir, keep=2)
        assert removed == 3

        remaining = sorted(p.name for p in snap_dir.glob("*.db"))
        assert remaining == sorted(p.name for p in taken)[-2:], \
            "the two NEWEST snapshots survive"

    def test_prune_counts_gz_and_db_together(self, tmp_dir):
        from revien.watch import prune_snapshots, snapshot_db, snapshot_dir_for

        db = tmp_dir / "mem.db"
        _make_db(db, n_nodes=1)
        snapshot_db(db)
        snapshot_db(db, use_gzip=True)
        snapshot_db(db)
        removed = prune_snapshots(snapshot_dir_for(db), keep=2)
        assert removed == 1

    def test_cli_watch_snapshots_then_ctrl_c(self, fake_home, tmp_dir, monkeypatch):
        """One tick then Ctrl+C: sleep raises KeyboardInterrupt, the loop
        exits cleanly with one snapshot on disk."""
        from revien.watch import snapshot_dir_for

        db = tmp_dir / "mem.db"
        _make_db(db, n_nodes=1)

        def fake_sleep(_seconds):
            raise KeyboardInterrupt

        monkeypatch.setattr(time, "sleep", fake_sleep)
        result = _run("watch", "--db", str(db), "--interval", "1", "--keep", "3")
        assert result.exit_code == 0, result.output
        assert "snapshot" in result.output
        assert "Stopped." in result.output
        assert len(list(snapshot_dir_for(db).glob("*.db"))) == 1

    def test_cli_watch_missing_db_refuses(self, fake_home, tmp_dir):
        result = _run("watch", "--db", str(tmp_dir / "nope.db"))
        assert result.exit_code != 0


class TestContinueScanEdgeCases:
    """No-yaml fallback: the two states the adversarial review broke."""

    def test_flow_style_mcpservers_counts_as_key_exists(self):
        from revien.mcp_install import _continue_scan

        text = "mcpServers: [{name: other, command: foo}]\nmodels:\n  - name: gpt\n"
        key_exists, present = _continue_scan(text)
        assert key_exists, (
            "flow-style mcpServers must register — otherwise the append "
            "writes a duplicate top-level key that clobbers the user's "
            "servers on last-wins YAML parsing"
        )
        assert not present

    def test_revien_named_model_elsewhere_is_not_present(self):
        from revien.mcp_install import _continue_scan

        text = (
            "models:\n"
            "  - name: revien\n"
            "    provider: ollama\n"
            "mcpServers:\n"
            "  - name: other\n"
            "    command: foo\n"
        )
        key_exists, present = _continue_scan(text)
        assert key_exists
        assert not present, (
            "a model named revien outside the mcpServers block must not "
            "read as already-installed"
        )

    def test_revien_inside_block_is_present(self):
        from revien.mcp_install import _continue_scan

        text = "mcpServers:\n  - name: revien\n    command: revien\n"
        key_exists, present = _continue_scan(text)
        assert key_exists and present


class TestJsonMergeUnicode:
    def test_unicode_survives_merge_unescaped(self, tmp_dir):
        from revien.mcp_install import install_mcp_client

        path = tmp_dir / ".cursor" / "mcp.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"mcpServers": {"日本語サーバー": {"command": "café"}}},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outcome = install_mcp_client("cursor", override_path=path)
        assert outcome.status == "merged"
        text = path.read_text(encoding="utf-8")
        assert "日本語サーバー" in text, "unicode must not be escaped to \\uXXXX"
        assert "café" in text
        data = json.loads(text)
        assert "revien" in data["mcpServers"]
        assert "日本語サーバー" in data["mcpServers"]

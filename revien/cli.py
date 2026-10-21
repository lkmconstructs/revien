"""
Revien CLI — Command-line interface for the Revien memory engine.
Three lines to persistent memory:
    pip install revien
    revien connect claude-code
    revien start
"""

import json
import os
import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

try:
    import click
except ImportError:
    print("Click is required: pip install click")
    sys.exit(1)

# Single version source (R4): revien.__version__ feeds the CLI banner, the
# FastAPI app metadata, /v1/health, and the hermes plugin manifest alike.
from revien import __version__
from revien.dates import said_date


def _default_db_path() -> str:
    revien_dir = Path.home() / ".revien"
    revien_dir.mkdir(parents=True, exist_ok=True)
    return str(revien_dir / "revien.db")


def _resolve_revien_command() -> str:
    """Absolute path to the ``revien`` launcher when resolvable, else the bare
    name. An MCP client (Codex, etc.) spawns this command to start the stdio
    server; a user-site pip install often lands ``revien.exe`` in a Scripts dir
    that is NOT on PATH, so a bare ``revien`` fails to spawn. An absolute path
    is PATH-independent AND immune to a stale sibling ``revien`` package on the
    spawn cwd — the two failure modes a bare name hits on a dev box.
    """
    found = shutil.which("revien")
    if found:
        return found
    name = "revien.exe" if os.name == "nt" else "revien"
    seen = set()
    # This interpreter's script dirs — user scheme first (where --user installs
    # land), then the base scheme.
    schemes = []
    user_scheme = f"{os.name}_user"
    if user_scheme in sysconfig.get_scheme_names():
        schemes.append(user_scheme)
    schemes.append(sysconfig.get_default_scheme())
    for scheme in schemes:
        try:
            scripts = sysconfig.get_path("scripts", scheme)
        except (KeyError, ValueError):
            continue
        if not scripts or scripts in seen:
            continue
        seen.add(scripts)
        cand = Path(scripts) / name
        if cand.exists():
            return str(cand)
    return "revien"  # last resort: trust the user's PATH


def _get_config_path() -> Path:
    return Path.home() / ".revien" / "config.json"


def _load_config() -> dict:
    config_path = _get_config_path()
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {"adapters": {}, "db_path": _default_db_path()}


def _save_config(config: dict) -> None:
    config_path = _get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))


@click.group()
@click.version_option(version=__version__, prog_name="revien")
def main():
    """Revien — Memory that returns. Graph-based memory engine for AI systems."""
    pass


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=7437, help="Port to listen on")
@click.option("--db", default=None, help="Database path (default: ~/.revien/revien.db)")
@click.option("--sync-interval", default=6.0, help="Auto-sync interval in hours")
def start(host: str, port: int, db: Optional[str], sync_interval: float):
    """Start the Revien daemon."""
    from revien.daemon.daemon import RevienDaemon

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    click.echo(f"Starting Revien daemon on {host}:{port}")
    click.echo(f"Database: {db_path}")
    click.echo("Press Ctrl+C to stop.\n")

    daemon = RevienDaemon(
        host=host,
        port=port,
        db_path=db_path,
        sync_interval_hours=sync_interval,
        adapters=config.get("adapters", {}),
    )
    daemon.start()


@main.command()
@click.option("--db", default=None, help="Database path (default: ~/.revien/revien.db)")
def mcp(db: Optional[str]):
    """Run the Revien MCP server over stdio (for Claude Code, Codex, Cursor,
    and other MCP clients that spawn local servers).

    Own process, own connection — SQLite WAL lets it coexist with a running
    daemon on the same database. Requires the mcp extra:
    pip install revien[mcp]
    """
    from revien.mcp_server import MCP_AVAILABLE, build_mcp_server, close_mcp_server

    if not MCP_AVAILABLE:
        click.echo("The MCP SDK is not installed. Install with: pip install revien[mcp]")
        sys.exit(1)

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    # stdout is the protocol channel on stdio — anything human goes to stderr.
    click.echo(f"Revien MCP server (stdio) — database: {db_path}", err=True)
    server = build_mcp_server(db_path=db_path)
    try:
        server.run(transport="stdio")
    finally:
        # The own-stack GraphStore this process opened closes with it —
        # deterministically, not whenever GC gets around to the connection.
        close_mcp_server(server)


@main.command()
@click.argument("system")
@click.option("--path", default=None, help="Custom path for the adapter")
def connect(system: str, path: Optional[str]):
    """Connect an AI system to Revien. Supported: claude-code, codex,
    cursor, windsurf, cline, continue, vscode, hermes, file-watcher, api,
    obsidian."""
    config = _load_config()

    if system == "claude-code":
        from revien.adapters.claude_code import ClaudeCodeAdapter, CLAUDE_CODE_PATHS

        if path:
            session_dir = path
        else:
            # Auto-detect
            session_dir = None
            for p in CLAUDE_CODE_PATHS:
                if p.exists():
                    session_dir = str(p)
                    break

        if session_dir is None:
            click.echo("Could not auto-detect Claude Code session directory.")
            click.echo("Specify manually: revien connect claude-code --path /path/to/.claude")
            return

        config["adapters"]["claude-code"] = {
            "type": "claude_code",
            "session_dir": session_dir,
        }
        _save_config(config)
        click.echo(f"Connected Claude Code adapter.")
        click.echo(f"Session directory: {session_dir}")

        # CLAUDE.md snippet — printed, never written into a repo (mirror of
        # the codex branch's AGENTS.md snippet). Makes the agent an ACTIVE
        # memory user instead of a passive one.
        click.echo("Suggested CLAUDE.md snippet (add to your project's "
                   "CLAUDE.md or ~/.claude/CLAUDE.md):")
        click.echo(
            "\n## Memory (Revien)\n"
            "This machine runs Revien, a persistent memory engine.\n"
            "- At conversation start, call `revien_recall` with the task topic to load prior context.\n"
            "- When a decision is made or a fact is established, store it with `revien_store` — silently, no announcement.\n"
            "- Prefer Revien recall over built-in memory for cross-session context.\n"
        )
        click.echo("Run 'revien start' to begin syncing.")

    elif system == "codex":
        from revien.adapters.codex import default_codex_home

        codex_home = default_codex_home()

        if path:
            session_dir = path
        else:
            sessions = codex_home / "sessions"
            session_dir = str(sessions) if sessions.exists() else None

        if session_dir is None:
            click.echo("Could not auto-detect Codex session directory.")
            click.echo("Specify manually: revien connect codex --path /path/to/.codex/sessions")
            return

        config["adapters"]["codex"] = {
            "type": "codex",
            "session_dir": session_dir,
        }
        _save_config(config)
        click.echo("Connected Codex adapter.")
        click.echo(f"Session directory: {session_dir}")

        # MCP client config: append to ~/.codex/config.toml — never overwrite,
        # never create. If the file isn't there, hand over the block instead.
        # command is a TOML literal string (single quotes) so a resolved
        # absolute Windows path with backslashes needs no escaping.
        revien_cmd = _resolve_revien_command()
        mcp_block = (
            "\n[mcp_servers.revien]\n"
            f"command = '{revien_cmd}'\n"
            'args = ["mcp"]\n'
        )
        config_toml = codex_home / "config.toml"
        if config_toml.exists():
            # Foreign file: tolerate BOMs and PowerShell's UTF-16 default —
            # an unreadable file gets the paste-block, never a traceback.
            existing = None
            read_enc = None
            for enc in ("utf-8-sig", "utf-16"):
                try:
                    existing = config_toml.read_text(encoding=enc)
                    read_enc = enc
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            # Present = an ACTIVE table header line, not substring containment:
            # a commented-out block or [mcp_servers.revien-staging] must not
            # block the append.
            present = existing is not None and any(
                line.strip() == "[mcp_servers.revien]"
                for line in existing.splitlines()
            )
            if present:
                click.echo(f"MCP entry already present in {config_toml} — left untouched.")
            elif existing is None or read_enc != "utf-8-sig":
                # Unreadable OR non-UTF-8 (appending UTF-8 to a UTF-16 file
                # would corrupt it) — hand over the block, touch nothing.
                click.echo(
                    f"Not appending to {config_toml} "
                    f"({'unrecognized encoding' if existing is None else 'file is not UTF-8'}) "
                    f"— left untouched. Add this block yourself:"
                )
                click.echo(mcp_block)
            else:
                with open(config_toml, "a", encoding="utf-8") as f:
                    if existing and not existing.endswith("\n"):
                        f.write("\n")
                    f.write(mcp_block)
                click.echo(f"Appended to {config_toml}:")
                click.echo(mcp_block)
        else:
            click.echo(f"No config.toml found at {config_toml}.")
            click.echo("To give Codex live recall/store tools, add this to your Codex config.toml:")
            click.echo(mcp_block)

        # AGENTS.md snippet — printed, never written into a repo. That's the
        # user's call.
        click.echo("Suggested AGENTS.md snippet (add to your repo root or ~/.codex/AGENTS.md):")
        click.echo(
            "\n## Memory (Revien)\n"
            "This machine runs Revien, a persistent memory engine.\n"
            "- At session start, call `revien_recall` with the task topic to load prior context.\n"
            "- When a decision is made or a fact is established, store it with `revien_store` — silently, no announcement.\n"
            "- Prefer Revien recall over asking the user to repeat past decisions.\n"
        )
        click.echo("Run 'revien start' to begin syncing.")

    elif system == "hermes":
        # Hermes Agent (NousResearch) memory-provider install. Hermes discovers
        # provider plugins from ~/.hermes/plugins/memory/<name>/ — the VERIFIED
        # layout (developer-guide memory-provider-plugin.md, hermes-agent
        # v0.18.2). We drop that plugin dir here. (A pip entry-point discovery
        # path was considered but their docs name only filesystem discovery and
        # the loader source was unreachable to confirm one — so it is not
        # advertised.)
        #
        # Same discipline as the codex config.toml append: idempotent, and never
        # clobber a file we didn't write. Our generated files carry a marker; a
        # foreign __init__.py gets the paste-block, not an overwrite.
        marker = "# revien connect hermes"
        version = __version__
        init_py = (
            f"{marker} (generated) — Hermes memory-provider plugin for Revien.\n"
            "# Re-exports the provider + entry point from the installed revien package,\n"
            "# so the thin plugin dir tracks the library and needs no edits on upgrade.\n"
            "from revien.hermes_provider import RevienMemoryProvider, register\n\n"
            '__all__ = ["RevienMemoryProvider", "register"]\n'
        )
        plugin_yaml = (
            f"{marker} (generated)\n"
            "name: revien\n"
            f"version: {version}\n"
            "description: >-\n"
            "  Revien persistent memory graph as a Hermes memory provider —\n"
            "  local-first, single SQLite file, zero network egress.\n"
            "hooks:\n"
            "  - prefetch\n"
            "  - sync_turn\n"
            "  - on_session_end\n"
            "  - system_prompt_block\n"
        )
        readme_md = (
            "# Revien memory provider for Hermes Agent\n\n"
            "Backs Hermes' automatic external memory with an in-process Revien\n"
            "stack over `~/.revien/revien.db` (override with `REVIEN_DB`).\n\n"
            "Requires: `pip install revien[hermes]` and `hermes-agent`.\n"
            "Tested against hermes-agent v0.18.2 (2026.7.7.2).\n\n"
            "Activate:  `hermes memory setup`  then select `revien`.\n"
        )
        files = {
            "__init__.py": init_py,
            "plugin.yaml": plugin_yaml,
            "README.md": readme_md,
        }

        hermes_home = Path(path) if path else (Path.home() / ".hermes")
        plugin_dir = hermes_home / "plugins" / "memory" / "revien"

        existing_init = plugin_dir / "__init__.py"
        foreign = False
        if existing_init.exists():
            # Ours to refresh only if it carries our marker — else it's a user
            # file and we touch nothing.
            try:
                foreign = marker not in existing_init.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                foreign = True

        if foreign:
            click.echo(
                f"A plugin already exists at {plugin_dir} and was not written by "
                f"'revien connect hermes' — left untouched."
            )
            click.echo("To install manually, create these files yourself:")
            for fname, body in files.items():
                click.echo(f"\n----- {plugin_dir / fname} -----")
                click.echo(body)
        else:
            plugin_dir.mkdir(parents=True, exist_ok=True)
            for fname, body in files.items():
                (plugin_dir / fname).write_text(body, encoding="utf-8")
            click.echo(f"Installed Revien Hermes memory provider -> {plugin_dir}")
            click.echo("  wrote: __init__.py, plugin.yaml, README.md")

        # The plugin dir re-exports the provider, which needs the SDK-guarded
        # module importable. Point the user at activation + the dependency.
        click.echo(
            "\nThen activate in Hermes:  hermes memory setup  (select 'revien').\n"
            "Requires: pip install revien[hermes]  and  hermes-agent "
            "(tested: v0.18.2)."
        )

    elif system in ("cursor", "windsurf", "cline", "continue", "vscode"):
        # MCP client installers (LEG P4): write the Revien MCP server entry
        # into the tool's own config at its documented location. Same
        # discipline as the codex config.toml append above — merge, never
        # clobber; refuse with a paste-block instead of guessing.
        from revien.mcp_install import DISPLAY_NAMES, install_mcp_client

        out = install_mcp_client(
            system, override_path=Path(path) if path else None
        )
        name = DISPLAY_NAMES[system]
        if out.status == "created":
            click.echo(f"Created {out.path} with the Revien MCP entry.")
        elif out.status == "merged":
            click.echo(f"Added the Revien MCP entry to {out.path} "
                       f"(existing entries preserved).")
        elif out.status == "already":
            click.echo(f"Revien MCP entry already present in {out.path} "
                       f"— left untouched.")
        else:
            click.echo(f"Did not modify {out.path}: {out.detail}.")
            click.echo("Add this yourself:")
            click.echo("\n" + out.snippet)
        if out.status in ("created", "merged"):
            click.echo(f"Restart {name} to load the Revien MCP server "
                       f"(requires: pip install revien[mcp]).")

    elif system == "file-watcher":
        if not path:
            click.echo("File watcher requires a path: revien connect file-watcher --path /watch/dir")
            return
        config["adapters"]["file-watcher"] = {
            "type": "file_watcher",
            "watch_dir": path,
        }
        _save_config(config)
        click.echo(f"Connected file watcher adapter.")
        click.echo(f"Watch directory: {path}")

    elif system == "api":
        if not path:
            click.echo("API adapter requires a URL: revien connect api --path http://localhost:8080/conversations")
            return
        config["adapters"]["api"] = {
            "type": "generic_api",
            "url": path,
        }
        _save_config(config)
        click.echo(f"Connected generic API adapter.")
        click.echo(f"URL: {path}")

    elif system == "obsidian":
        if not path:
            click.echo("Obsidian requires a vault path: revien connect obsidian --path /path/to/vault")
            return
        vault = Path(path).expanduser()
        if not vault.is_dir():
            click.echo(f"Vault directory not found: {vault}")
            return
        config["adapters"]["obsidian"] = {
            "type": "obsidian",
            "vault_dir": str(vault),
        }
        _save_config(config)
        click.echo("Connected Obsidian vault adapter.")
        click.echo(f"Vault: {vault}")
        click.echo("Run 'revien sync-vault' to ingest the vault now.")

    else:
        click.echo(f"Unknown system: {system}")
        click.echo("Supported: claude-code, codex, cursor, windsurf, cline, "
                   "continue, vscode, hermes, file-watcher, api, obsidian")


@main.command()
@click.argument("query")
@click.option("--top", default=5, help="Number of results to return")
@click.option("--db", default=None, help="Database path")
@click.option("--as-of", "as_of", default=None,
              help="Bi-temporal query time (ISO-8601): what was true AT this "
                   "time — superseded facts whose validity window covers it "
                   "come back")
@click.option("--source", "source", multiple=True,
              help="Filter to one or more origin_runtime values "
                   "(claude-code/codex/hermes/ollama/openai/langchain/"
                   "obsidian/file/api/...). Repeatable: --source claude-code "
                   "--source codex. Omit for no filtering.")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--format", "output_format", type=click.Choice(["json", "toon"]),
              default="json", show_default=True,
              help="Wire format: 'toon' prints the full recall payload as "
                   "TOON (Token-Oriented Object Notation — same data, fewer "
                   "tokens for a consuming LLM); 'json' keeps the existing "
                   "behavior (human-readable, or JSON with --json-output)")
def recall(query: str, top: int, db: Optional[str], as_of: Optional[str],
           source: tuple, json_output: bool, output_format: str):
    """Query Revien memory from the command line."""
    from datetime import datetime
    from revien.graph.store import GraphStore
    from revien.retrieval.engine import RetrievalEngine

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    as_of_dt = None
    if as_of:
        try:
            as_of_dt = datetime.fromisoformat(as_of)
        except ValueError:
            click.echo(f"Invalid --as-of (ISO-8601 expected): {as_of}")
            return

    # Origin Layer (WS0 Leg B): multiple=True gives a tuple; empty tuple
    # means "no filter" (engine.recall(source=None) is the byte-identical
    # unfiltered path), one value stays a list of one — engine.recall
    # accepts either a bare string or a list, list is simplest here.
    source_filter = list(source) if source else None

    store = GraphStore(db_path=db_path)
    engine = RetrievalEngine(store)

    try:
        response = engine.recall(query, top_n=top, as_of=as_of_dt, source=source_filter)

        if output_format == "toon":
            if json_output:
                click.echo(
                    "note: --json-output is ignored with --format toon",
                    err=True,
                )
            # Same shape as POST /v1/recall — the payload a consuming LLM
            # ingests — serialized as TOON (LEG P2).
            from revien.toon import serialize_recall

            payload = {
                "query": response.query,
                "results": [
                    {
                        "node_id": r.node_id,
                        "node_type": r.node_type,
                        "label": r.label,
                        "content": r.content,
                        "score": r.score,
                        "score_breakdown": r.score_breakdown,
                        "path": r.path,
                        "origin_runtime": r.origin_runtime,
                        "origin_source": r.origin_source,
                        "project_key": r.project_key,
                        "recorded_at": r.recorded_at,
                        "recorded_at_source": r.recorded_at_source,
                    }
                    for r in response.results
                ],
                "nodes_examined": response.nodes_examined,
                "retrieval_time_ms": response.retrieval_time_ms,
                "semantic_active": response.semantic_active,
                "semantic_note": response.semantic_note,
                "skill_proposals": response.skill_proposals,
            }
            click.echo(serialize_recall(payload))
        elif json_output:
            output = {
                "query": response.query,
                "results": [
                    {
                        "node_id": r.node_id,
                        "node_type": r.node_type,
                        "label": r.label,
                        "content": r.content,
                        "score": r.score,
                        "score_breakdown": r.score_breakdown,
                        "origin_runtime": r.origin_runtime,
                        "origin_source": r.origin_source,
                        "project_key": r.project_key,
                        "recorded_at": r.recorded_at,
                        "recorded_at_source": r.recorded_at_source,
                    }
                    for r in response.results
                ],
                "nodes_examined": response.nodes_examined,
                "retrieval_time_ms": response.retrieval_time_ms,
                "skill_proposals": response.skill_proposals,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            if not response.results:
                click.echo("No memories found for that query.")
                return

            click.echo(f"\nQuery: {query}")
            click.echo(f"Found {len(response.results)} results "
                       f"({response.retrieval_time_ms:.1f}ms, "
                       f"{response.nodes_examined} nodes examined)\n")

            for i, r in enumerate(response.results, 1):
                runtime = r.origin_runtime or "unknown"
                said = said_date(r.recorded_at, r.recorded_at_source) or "-"
                click.echo(f"  [{i}] {r.label}")
                click.echo(f"      Type: {r.node_type} | Score: {r.score:.3f} "
                           f"| Runtime: {runtime} | Date: {said}")
                click.echo(f"      {r.content[:120]}{'...' if len(r.content) > 120 else ''}")
                click.echo()

            if response.skill_proposals:
                click.echo(f"{len(response.skill_proposals)} draft skill "
                           f"proposal(s) match this query:")
                for p in response.skill_proposals:
                    click.echo(f"  - {p['label']}  ({p['occurrences']}x across "
                               f"{p['sessions']} sessions)")
                click.echo()
    finally:
        store.close()


@main.command()
@click.option("--db", default=None, help="Database path")
@click.option("--all", "include_history", is_flag=True,
              help="Include tensions where one side was later invalidated")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def tensions(db: Optional[str], include_history: bool, json_output: bool):
    """List recognized coexisting tensions (B1) — pairs of claims that pull
    in opposite directions and BOTH remain true. 'What am I in tension with
    myself about?'"""
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        pairs = store.list_tension_pairs(live_only=not include_history)

        if json_output:
            click.echo(json.dumps({"count": len(pairs), "tensions": pairs}, indent=2))
            return

        if not pairs:
            click.echo("No tensions recorded. (Tension detection is opt-in: "
                       "set REVIEN_TENSION_BACKEND with REVIEN_CSL=1, or "
                       "resolve queue candidates as coexist.)")
            return

        click.echo(f"\n{len(pairs)} tension{'s' if len(pairs) != 1 else ''}:\n")
        for i, p in enumerate(pairs, 1):
            a, b = p["a"], p["b"]
            click.echo(f"  [{i}] {a['content'][:100]}")
            click.echo(f"   ⇄  {b['content'][:100]}")
            if p.get("source_context"):
                click.echo(f"      ({p['source_context']}, recognized {p['created_at'][:10]})")
            click.echo()
    finally:
        store.close()


@main.command()
@click.option("--db", default=None, help="Database path")
@click.option("--reindex", is_flag=True,
              help="Also backfill the semantic index (recovery, not routine)")
@click.option("--invalidate-orphans", "invalidate_orphans", is_flag=True,
              help="Soft-invalidate edgeless nodes (reversible; default is report-only)")
@click.option("--alias", "alias", is_flag=True,
              help="Also run evidence-backed ALIAS_OF inference (writes edges; opt-in)")
@click.option("--json-output", is_flag=True, help="Output the full report as JSON")
def dream(db: Optional[str], reindex: bool, invalidate_orphans: bool,
          alias: bool, json_output: bool):
    """Run the consolidation pass ("dream mode"): persist confidence decay,
    refresh communities, and report orphaned memories. Nothing is deleted;
    every change is audited and reported."""
    import json as _json
    from revien.consolidate import Consolidator
    from revien.graph.clustering import CommunityDetector
    from revien.graph.store import GraphStore
    from revien.semantic.index import SemanticIndex

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        clustering = CommunityDetector(db_path=db_path)
        semantic = SemanticIndex(store)
        report = Consolidator(
            store, semantic=semantic, clustering=clustering,
        ).run(reindex=reindex, invalidate_orphans=invalidate_orphans, alias=alias)

        if json_output:
            click.echo(_json.dumps(report.to_dict(), indent=2))
            return

        d = report.to_dict()
        click.echo(f"\nDream pass complete ({d['duration_ms']:.0f}ms, "
                   f"{d['nodes_examined']} nodes examined)\n")
        click.echo(f"  decay     : {d['decay']['nodes_decayed']} node(s) decayed")
        for item in d["decay"]["sample"][:5]:
            click.echo(f"              {item['label'][:50]}: "
                       f"{item['before']} -> {item['after']}")
        click.echo(f"  clusters  : {d['recluster']['communities']} communities"
                   if d["recluster"]["ran"] else "  clusters  : skipped")
        click.echo(f"  reindex   : {d['reindex']['result']}"
                   if d["reindex"]["ran"] else "  reindex   : skipped (use --reindex)")
        click.echo(f"  orphans   : {d['orphans']['found']} found, "
                   f"{d['orphans']['invalidated']} invalidated"
                   + ("" if invalidate_orphans else " (report-only; "
                      "--invalidate-orphans to act)"))
        for item in d["orphans"]["sample"][:5]:
            click.echo(f"              [{item['node_type']}] {item['label'][:50]}")
        if d["alias"]["ran"]:
            click.echo(f"  aliases   : {d['alias']['edges_created']} edge(s) created "
                       f"({d['alias']['candidates_considered']} candidate(s) considered)")
            for item in d["alias"]["sample"][:5]:
                click.echo(f"              {item['label_a'][:30]} <-> "
                           f"{item['label_b'][:30]} [{item['method']}]")
        else:
            click.echo("  aliases   : skipped (use --alias)"
                       + (f" — {d['alias']['note']}" if d["alias"]["note"] else ""))
        click.echo()
    finally:
        store.close()


@main.command()
@click.option("--db", default=None, help="Database path")
@click.option("--remove", "remove_edge_id", default=None,
              help="Soft-invalidate one alias edge by id (reversible, audited)")
@click.option("--all", "include_history", is_flag=True,
              help="Include reversed (removed) alias edges")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def aliases(db: Optional[str], remove_edge_id: Optional[str],
            include_history: bool, json_output: bool):
    """List evidence-backed ALIAS_OF edges, or reverse one with
    --remove <edge_id> (soft-invalidate; the edge is retained and auditable,
    never deleted). Aliases are created by 'revien dream --alias', by
    POST /v1/edges with edge_type=alias_of, or by the recall anchor
    expansion's own inference pass."""
    from revien.graph.operations import GraphOperations
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        if remove_edge_id:
            ops = GraphOperations(store)
            edge = ops.invalidate_edge(
                remove_edge_id, reason="cli_remove", construct_id="cli"
            )
            if edge is None:
                click.echo(f"No such edge: {remove_edge_id}")
                return
            click.echo(
                f"Removed alias edge {remove_edge_id} "
                f"(soft-invalidated — reversible, not deleted)."
            )
            return

        pairs = store.list_alias_pairs(live_only=not include_history)

        if json_output:
            click.echo(json.dumps({"count": len(pairs), "aliases": pairs}, indent=2))
            return

        if not pairs:
            click.echo("No alias edges recorded. (Run 'revien dream --alias', "
                       "or POST /v1/edges with edge_type=alias_of.)")
            return

        click.echo(f"\n{len(pairs)} alias pair{'s' if len(pairs) != 1 else ''}:\n")
        for i, p in enumerate(pairs, 1):
            a, b = p["a"], p["b"]
            click.echo(f"  [{i}] {a['label']} <-> {b['label']}"
                       + (f"  ({p['method']})" if p.get("method") else ""))
            if p.get("source_context"):
                click.echo(f"      {p['source_context']}")
            click.echo(f"      edge_id={p['edge_id']}")
            click.echo()
    finally:
        store.close()


@main.command()
@click.argument("content")
@click.option("--source", default="cli", help="Source identifier")
@click.option("--db", default=None, help="Database path")
def ingest(content: str, source: str, db: Optional[str]):
    """Manually ingest content into Revien."""
    from revien.graph.store import GraphStore
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    pipeline = IngestionPipeline(store)

    try:
        # Manual capture: when it was said is now, and that is what we stamp.
        result = pipeline.ingest(IngestionInput(
            source_id=source,
            content=content,
            timestamp=datetime.now(timezone.utc),
            timestamp_source="capture",
        ))
        if result.suppressed:
            click.echo("Not ingested: " + result.suppression_note)
            return
        click.echo(f"Ingested: {result.nodes_created} nodes, {result.edges_created} edges")
        click.echo(f"Graph total: {result.total_nodes_in_graph} nodes, "
                   f"{result.total_edges_in_graph} edges")
    finally:
        store.close()


@main.command(name="sync-vault")
@click.option("--vault", default=None, help="Vault path (defaults to the connected obsidian adapter)")
@click.option("--db", default=None, help="Database path")
@click.option("--folder", default="Revien", help="Distill folder to reconcile edits from (default: Revien)")
@click.option("--full", is_flag=True, help="Re-ingest every note, not just ones changed since last sync")
def sync_vault(vault: Optional[str], db: Optional[str], folder: str, full: bool):
    """One-shot Obsidian vault sync. First reconciles your edits to distilled
    notes back into the graph (correct / delete / add), then ingests new or
    changed vault notes as curated memory."""
    import asyncio
    from datetime import datetime, timezone

    from revien.adapters.obsidian import ObsidianVaultAdapter
    from revien.distill import VaultReconciler
    from revien.graph.store import GraphStore
    from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

    config = _load_config()
    vault_dir = vault or config.get("adapters", {}).get("obsidian", {}).get("vault_dir")
    if not vault_dir:
        click.echo("No vault configured. Run: revien connect obsidian --path /path/to/vault")
        return
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        # 1. Reconcile edits to distilled notes back into the graph (always —
        #    independent of whether there are new user notes to ingest).
        rec = VaultReconciler(store, vault_dir, folder=folder).reconcile()
        if rec["corrected"] or rec["added"] or rec["forgotten"]:
            click.echo(
                f"Reconciled edits: {rec['corrected']} corrected, "
                f"{rec['added']} added, {rec['forgotten']} forgotten "
                f"across {rec['notes_reconciled']} note(s)."
            )

        # 2. Ingest new / changed vault notes as curated memory.
        adapter = ObsidianVaultAdapter(vault_dir)
        last_sync_key = "obsidian_last_sync"
        since = datetime.fromtimestamp(0, tz=timezone.utc)
        if not full and config.get(last_sync_key):
            try:
                since = datetime.fromisoformat(config[last_sync_key])
            except ValueError:
                pass

        items = asyncio.run(adapter.fetch_new_content(since))
        pipeline = IngestionPipeline(store)
        nodes = edges = suppressed = 0
        for item in items:
            ts = None
            try:
                ts = datetime.fromisoformat(item["timestamp"])
            except (KeyError, ValueError, TypeError):
                pass
            out = pipeline.ingest(IngestionInput(
                source_id=item["source_id"],
                content=item["content"],
                content_type=item.get("content_type", "note"),
                timestamp=ts,
                # The adapter says whether the date is a frontmatter date
                # ("content") or just the file mtime; never assume.
                timestamp_source=item.get("timestamp_source") or "mtime",
                metadata=item.get("metadata", {}),
                links=item.get("links", []),
                curated=True,
            ))
            if out.suppressed:
                suppressed += 1
                continue
            nodes += out.nodes_created
            edges += out.edges_created
    finally:
        store.close()

    config[last_sync_key] = datetime.now(timezone.utc).isoformat()
    _save_config(config)
    if items:
        notes = len({i["metadata"]["note"] for i in items})
        click.echo(f"Vault sync: {notes} notes -> {len(items)} chunks, "
                   f"{nodes} nodes, {edges} edges (curated).")
        if suppressed:
            click.echo(f"{suppressed} chunk(s) were not ingested: a forget "
                       "suppressed them. 'revien restore' lifts that.")
    elif not (rec["corrected"] or rec["added"] or rec["forgotten"]):
        click.echo("Vault is up to date — nothing to reconcile or ingest.")


@main.command(name="reconcile-vault")
@click.option("--vault", default=None, help="Vault path (defaults to the connected obsidian adapter)")
@click.option("--db", default=None, help="Database path")
@click.option("--folder", default="Revien", help="Distill folder to reconcile from (default: Revien)")
def reconcile_vault(vault: Optional[str], db: Optional[str], folder: str):
    """Reconcile your edits to distilled notes back into the graph, without
    ingesting anything else. Correct a claim to supersede it, delete a line to
    forget it, add a line under a heading to teach it."""
    from revien.distill import VaultReconciler
    from revien.graph.store import GraphStore

    config = _load_config()
    vault_dir = vault or config.get("adapters", {}).get("obsidian", {}).get("vault_dir")
    if not vault_dir:
        click.echo("No vault configured. Run: revien connect obsidian --path /path/to/vault")
        return
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found.")
        return

    store = GraphStore(db_path=db_path)
    try:
        rec = VaultReconciler(store, vault_dir, folder=folder).reconcile()
    finally:
        store.close()
    click.echo(
        f"Reconciled: {rec['corrected']} corrected, {rec['added']} added, "
        f"{rec['forgotten']} forgotten across {rec['notes_reconciled']} note(s)."
    )


@main.command(name="distill-vault")
@click.option("--vault", default=None, help="Vault path (defaults to the connected obsidian adapter)")
@click.option("--db", default=None, help="Database path")
@click.option("--folder", default="Revien", help="Folder inside the vault to write into (default: Revien)")
@click.option("--min-claims", default=1, help="Minimum machine-side claims an entity needs to earn a note")
def distill_vault(vault: Optional[str], db: Optional[str], folder: str, min_claims: int):
    """Write Revien's memory INTO the vault as readable markdown — one note
    per entity, provenance on every claim, wikilinked into your graph.
    Read-only view: writes only inside the distill folder, only overwrites
    its own marked files, and distilled notes are never re-ingested."""
    from revien.distill import VaultDistiller
    from revien.graph.store import GraphStore

    config = _load_config()
    vault_dir = vault or config.get("adapters", {}).get("obsidian", {}).get("vault_dir")
    if not vault_dir:
        click.echo("No vault configured. Run: revien connect obsidian --path /path/to/vault")
        return
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' or 'revien ingest' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        summary = VaultDistiller(
            store, vault_dir, folder=folder, min_claims=min_claims
        ).distill()
    finally:
        store.close()

    if summary.get("status") != "ok":
        click.echo(f"Distill failed: {summary.get('error')}")
        return
    click.echo(f"Distilled {summary['notes']} entity notes -> {summary['out_dir']}")
    click.echo(f"  written: {summary['written']}  unchanged: {summary['unchanged']}  "
               f"pruned: {summary['pruned']}  vault-echo skipped: {summary['skipped_vault_echo']}")


@main.command()
@click.option("--db", default=None, help="Database path")
def reindex(db: Optional[str]):
    """Backfill semantic embeddings for existing nodes (opt-in semantic layer).

    Requires the `semantic` extra: pip install revien[semantic]. Without it (or
    with REVIEN_SEMANTIC=0) this reports the layer is disabled and does nothing.
    """
    from revien.graph.store import GraphStore
    from revien.semantic.index import SemanticIndex

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        semantic = SemanticIndex(store)
        if not semantic.is_enabled:
            st = semantic.status()
            click.echo("Semantic layer is disabled.")
            click.echo(f"  extra installed (sqlite-vec): {st['sqlite_vec']}")
            click.echo(f"  fastembed installed: {st['fastembed']}")
            click.echo(f"  env gate (REVIEN_SEMANTIC): {st['env_gate']}")
            click.echo("Install with: pip install revien[semantic]")
            return
        result = semantic.reindex_all()
        click.echo(f"Reindexed {result.get('indexed', 0)} nodes "
                   f"(status: {result.get('status')}).")
        if result.get("status") != "ok":
            click.echo(f"Reindex did NOT complete: {result.get('error')}. "
                       f"Run `revien reindex` again.", err=True)
            sys.exit(1)
    finally:
        store.close()


@main.command()
@click.option("--db", default=None, help="Database path")
def status(db: Optional[str]):
    """Show Revien status and graph statistics."""
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        node_count = store.count_nodes()
        edge_count = store.count_edges()
        click.echo(f"Revien Memory Engine v{__version__}")
        click.echo(f"Database: {db_path}")
        click.echo(f"Nodes: {node_count}")
        click.echo(f"Edges: {edge_count}")

        # Show adapter config
        adapters = config.get("adapters", {})
        if adapters:
            click.echo(f"Connected adapters: {', '.join(adapters.keys())}")
        else:
            click.echo("No adapters connected. Run 'revien connect <system>'")

        # Origin Layer (WS0 Leg B): per-runtime node counts. "unknown"
        # (NULL origin_runtime — pre-origin-layer or unrecognized source_id)
        # is a real bucket, not hidden — see
        # GraphStore.count_nodes_by_origin_runtime.
        by_runtime = store.count_nodes_by_origin_runtime()
        if by_runtime:
            click.echo("Nodes by runtime:")
            for runtime, count in by_runtime.items():
                click.echo(f"  {runtime}: {count}")

        by_source = store.count_nodes_by_recorded_at_source()
        if by_source:
            click.echo("Nodes by recorded_at_source:")
            for src, count in by_source.items():
                note = " (date not shown)" if src in ("mtime", "unlabeled", "undated") else ""
                click.echo(f"  {src}: {count}{note}")

        click.echo(f"Pairing token: {_pairing_token_status()}")
        from revien.semantic._ort import ort_telemetry_state
        click.echo(
            "ONNX Runtime telemetry: "
            + ("disabled" if ort_telemetry_state() else "not applicable")
        )
        store.expire_recall_receipts()
        flags = len(store.list_receipt_flags())
        if flags:
            click.echo(f"Usage flags: {flags} recall(s) expired unjudged "
                       "(revien usage flags)")
    finally:
        store.close()


def _pairing_token_status() -> str:
    """set (env) / set (file) / not set — mirrors pairing.configured_token's
    precedence without ever printing the token itself."""
    from revien import pairing

    if os.environ.get("REVIEN_CAPTURE_TOKEN", "").strip():
        return "set (env)"
    if pairing.load_token():
        return "set (file)"
    return "not set"


@main.command()
@click.option("--rotate", is_flag=True, help="Mint a new token, replacing any existing one.")
@click.option("--path", "show_path", is_flag=True, help="Print the token file path only — not the token.")
def token(rotate: bool, show_path: bool):
    """Print the pairing token for remote capture/mutation, minting one if absent.

    A remote /v1/ingest (or skill accept/decline) caller pairs by sending this
    token as `Authorization: Bearer <token>`. REVIEN_CAPTURE_TOKEN, if set,
    overrides whatever is minted here. The full token prints exactly once per
    invocation — nothing is echoed partially or masked.
    """
    from revien import pairing

    if show_path:
        click.echo(str(pairing.token_path()))
        return
    click.echo(pairing.mint_token(rotate=rotate))


@main.command()
@click.argument("file", required=False, type=click.Path())
@click.option("--output", "-o", default=None,
              help="Output file path (same as FILE; kept for compatibility)")
@click.option("--db", default=None, help="Database path")
def export(file: Optional[str], output: Optional[str], db: Optional[str]):
    """Export the full graph as portable JSON: revien export graph.json

    Same schema and code path as GET /v1/graph — the file `revien import`
    (and POST /v1/graph/import) accepts. No FILE prints to stdout.
    """
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        sys.exit(1)

    target = file or output
    store = GraphStore(db_path=db_path)
    try:
        graph = store.export_graph()
        json_str = graph.model_dump_json(indent=2)

        if target:
            Path(target).write_text(json_str, encoding="utf-8")
            # A file on disk is a copy a later forget cannot reach; remember
            # that one was written so the forget response can name it.
            _note_export(store)
            click.echo(f"Exported {len(graph.nodes)} nodes, "
                       f"{len(graph.edges)} edges to {target}")
        else:
            click.echo(json_str)
    finally:
        store.close()


def _note_export(store) -> None:
    """Record that an export was written to disk (count and time, no path)."""
    try:
        prev = json.loads(store.get_meta("exports_written") or "{}")
        store.set_meta("exports_written", json.dumps({
            "count": int(prev.get("count", 0)) + 1,
            "last": datetime.now(timezone.utc).isoformat()}))
    except Exception:  # noqa: BLE001 - a note must never fail an export
        pass


@main.command(name="import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database path")
@click.option("--merge", is_flag=True,
              help="Import into a non-empty database; nodes/edges whose IDs "
                   "already exist are skipped, never overwritten")
@click.option("--replace", "replace", is_flag=True,
              help="Replace the existing graph wholesale — delete + insert in "
                   "one transaction (a failure leaves the original intact)")
def import_(file: str, db: Optional[str], merge: bool, replace: bool):
    """Import a graph exported by `revien export`: revien import graph.json

    Same schema and code path as POST /v1/graph/import. Refuses a non-empty
    target database unless --merge or --replace — an import must never
    silently eat an existing graph.
    """
    from revien.graph.schema import Graph
    from revien.graph.store import (
        GraphStore, ImportRefusedError, ImportValidationError,
    )
    from revien.semantic.index import SemanticIndex

    if merge and replace:
        click.echo("--merge and --replace are mutually exclusive.")
        sys.exit(1)

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    try:
        data = json.loads(Path(file).read_text(encoding="utf-8-sig"))
        graph = Graph.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        click.echo(f"Not a JSON graph file: {e}")
        sys.exit(1)
    except Exception as e:  # pydantic ValidationError — schema mismatch
        click.echo(f"Invalid graph data: {e}")
        sys.exit(1)

    mode = "replace" if replace else ("merge" if merge else "refuse")

    store = GraphStore(db_path=db_path)
    try:
        existing_nodes = store.count_nodes()
        existing_edges = store.count_edges()
        semantic = SemanticIndex(store)

        try:
            result = store.import_graph(graph, mode=mode, semantic=semantic)
        except ImportRefusedError:
            click.echo(
                f"Target database is not empty ({existing_nodes} nodes, "
                f"{existing_edges} edges): {db_path}"
            )
            click.echo("Use --merge to import on top of it, --replace to "
                       "overwrite it, or --db to point at a fresh file.")
            sys.exit(1)
        except ImportValidationError as e:
            click.echo(f"Invalid graph data: {e}")
            sys.exit(1)

        if mode == "merge" and (existing_nodes or existing_edges):
            click.echo(f"Merged {result['nodes_added']} nodes, "
                       f"{result['edges_added']} edges "
                       f"(skipped {result['skipped_nodes']} existing nodes, "
                       f"{result['skipped_edges']} existing edges).")
        elif mode == "replace" and (existing_nodes or existing_edges):
            click.echo(f"Replaced existing graph ({result['removed_nodes']} nodes, "
                       f"{result['removed_edges']} edges removed).")
            click.echo(f"Imported {result['nodes_added']} nodes, "
                       f"{result['edges_added']} edges into {db_path}")
        else:
            click.echo(f"Imported {result['nodes_added']} nodes, "
                       f"{result['edges_added']} edges into {db_path}")

        if result["semantic_rebuild"] != "skipped (disabled)":
            click.echo(f"Semantic index: {result['semantic_rebuild']}")

        click.echo(f"Graph total: {store.count_nodes()} nodes, "
                   f"{store.count_edges()} edges")
    finally:
        store.close()


def _run_export_import(runtime: str, file: str, db: Optional[str],
                        dry_run: bool, limit: Optional[int],
                        override: bool = False) -> None:
    """Shared body for import-chatgpt / import-claude / import-readwise:
    resolve the store/pipeline the same way `ingest` does (cli.py's own
    ingest command, above), stream units from the right importer module,
    run them through run_import() under a progress bar, then print the
    same summary shape for all three so scripting against any of them
    looks identical."""
    from revien.graph.store import GraphStore
    from revien.importers import chatgpt, claude, readwise
    from revien.importers.base import run_import
    from revien.ingestion.pipeline import IngestionPipeline

    module = {"chatgpt": chatgpt, "claude": claude, "readwise": readwise}[runtime]

    try:
        units = list(module.iter_units(file))
    except (FileNotFoundError, ValueError) as e:
        click.echo(str(e))
        sys.exit(1)

    if limit is not None:
        units = units[:limit]

    if dry_run:
        # Never touch the store: no GraphStore/IngestionPipeline construction
        # at all, so a dry run can't accidentally create a fresh db file.
        report = run_import(units, pipeline=None, dry_run=True)
        click.echo("dry run: nothing written")
    else:
        config = _load_config()
        db_path = db or config.get("db_path", _default_db_path())
        store = GraphStore(db_path=db_path)
        pipeline = IngestionPipeline(store)
        try:
            with click.progressbar(
                length=len(units), label=f"Importing ({runtime})"
            ) as bar:
                report = run_import(
                    units, pipeline, dry_run=False,
                    progress=lambda _unit: bar.update(1),
                    override_suppression=override,
                )
        finally:
            store.close()

    click.echo(
        f"seen={report.units_seen} ingested={report.units_ingested} "
        f"unchanged={report.units_unchanged} denied={report.units_denied} "
        f"empty={report.units_skipped_empty} errors={len(report.errors)}"
        + (f" suppressed={report.units_suppressed}" if report.units_suppressed else "")
    )
    if report.units_suppressed:
        click.echo(f"{report.units_suppressed} unit(s) were not imported: a "
                   "forget suppressed them. Re-run with --override-suppressed "
                   "to import them anyway, or use 'revien restore'.")
    if not dry_run:
        click.echo(
            f"nodes_created={report.nodes_created} "
            f"edges_created={report.edges_created}"
        )
    for source_id, message in report.errors:
        click.echo(f"  error: {source_id}: {message}")

    # F11: only a hard failure is non-zero — errors happened AND nothing at
    # all was captured (ingested or confirmed-unchanged). A partial import
    # (some conversations ingested, one malformed one logged as an error) or
    # a clean idempotent re-run (all unchanged, 0 ingested, 0 errors) both
    # exit 0.
    if report.errors and (report.units_ingested + report.units_unchanged) == 0:
        sys.exit(1)


@main.command(name="import-chatgpt")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database path")
@click.option("--dry-run", is_flag=True,
              help="Parse and report counts; write nothing")
@click.option("--override-suppressed", is_flag=True,
              help="Import units a forget suppressed (lifts those "
                   "suppressions; audited)")
@click.option("--limit", default=None, type=int,
              help="Only process the first N conversations")
def import_chatgpt(file: str, db: Optional[str], dry_run: bool, limit: Optional[int],
                  override_suppressed: bool):
    """Import a ChatGPT export (conversations.json, or the .zip you
    downloaded straight from ChatGPT's Settings -> Data controls -> Export)
    through the ingestion pipeline: one unit per conversation, on the
    thread actually shown (edited-away branches are not ingested).
    Idempotent — re-running an unchanged export is a no-op."""
    _run_export_import("chatgpt", file, db, dry_run, limit, override_suppressed)


@main.command(name="import-claude")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database path")
@click.option("--dry-run", is_flag=True,
              help="Parse and report counts; write nothing")
@click.option("--override-suppressed", is_flag=True,
              help="Import units a forget suppressed (lifts those "
                   "suppressions; audited)")
@click.option("--limit", default=None, type=int,
              help="Only process the first N conversations")
def import_claude(file: str, db: Optional[str], dry_run: bool, limit: Optional[int],
                  override_suppressed: bool):
    """Import a Claude.ai export (conversations.json, or the .zip from
    Claude.ai's Settings -> Account -> Export data) through the ingestion
    pipeline: one unit per conversation. Idempotent — re-running an
    unchanged export is a no-op."""
    _run_export_import("claude", file, db, dry_run, limit, override_suppressed)


@main.command(name="import-readwise")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database path")
@click.option("--dry-run", is_flag=True,
              help="Parse and report counts; write nothing")
@click.option("--override-suppressed", is_flag=True,
              help="Import units a forget suppressed (lifts those "
                   "suppressions; audited)")
@click.option("--limit", default=None, type=int,
              help="Only process the first N highlights")
def import_readwise(file: str, db: Optional[str], dry_run: bool, limit: Optional[int],
                  override_suppressed: bool):
    """Import a Readwise highlights CSV export through the ingestion
    pipeline: one unit per highlight, linked to its book by an entity
    edge. Idempotent — re-running an unchanged export is a no-op."""
    _run_export_import("readwise", file, db, dry_run, limit, override_suppressed)


@main.command()
@click.option("--db", default=None, help="Database path")
@click.option("--interval", default=60.0, show_default=True,
              help="Minutes between snapshots")
@click.option("--keep", default=10, show_default=True,
              help="Snapshots retained; oldest pruned first")
@click.option("--gzip", "use_gzip", is_flag=True,
              help="Compress snapshots (.db.gz)")
def watch(db: Optional[str], interval: float, keep: int, use_gzip: bool):
    """Snapshot the database on an interval — the local backup loop.

    Copies the live db via SQLite's backup API (consistent even mid-write)
    to <db>.snapshots/<timestamp>.db, keeping the newest --keep. Runs in the
    foreground until Ctrl+C.
    """
    import time
    from datetime import datetime

    from revien.watch import prune_snapshots, snapshot_db, snapshot_dir_for

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        sys.exit(1)
    if keep < 1:
        click.echo("--keep must be at least 1.")
        sys.exit(1)
    if interval <= 0:
        click.echo("--interval must be positive.")
        sys.exit(1)

    snap_dir = snapshot_dir_for(db_path)
    click.echo(f"Watching {db_path}")
    click.echo(f"Snapshots: {snap_dir} — every {interval:g} min, "
               f"keeping {keep}{', gzipped' if use_gzip else ''}. "
               f"Ctrl+C to stop.")
    try:
        while True:
            snap = snapshot_db(db_path, use_gzip=use_gzip)
            pruned = prune_snapshots(snap_dir, keep=keep)
            stamp = datetime.now().strftime("%H:%M:%S")
            msg = f"[{stamp}] snapshot {snap.name}"
            if pruned:
                msg += f" (pruned {pruned} old)"
            click.echo(msg)
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


# ── skills (thin WS3, leg D1) ──────────────────────────────────────────────
# A group, not flat commands, so `propose`/`accept`/`decline` (leg D2, a
# parallel worktree) can land as siblings of `ingest`/`list`/`show` without
# touching anything above this line.

@main.group()
def skills():
    """Manage skills (SKILL.md-style procedures) as first-class memory."""
    pass


@skills.command(name="ingest")
@click.option("--path", "paths", multiple=True, type=click.Path(),
              help="Skill root to scan (repeatable). Default: ./.claude/skills "
                   "and ./.codex/skills.")
@click.option("--global", "include_global", is_flag=True,
              help="Also scan the global skill homes: ~/.claude/skills, "
                   "~/.codex/skills, ~/.hermes/skills.")
@click.option("--db", default=None, help="Database path")
def skills_ingest(paths: Tuple[str, ...], include_global: bool, db: Optional[str]):
    """Scan skill folders and ingest each SKILL.md as a SKILL node.

    Idempotent by path: re-running refreshes an unchanged skill's node in
    place (label/content/metadata) rather than duplicating it."""
    from revien.graph.store import GraphStore
    from revien.skills.ingest import ingest_roots

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        summary = ingest_roots(store, paths=list(paths) or None, include_global=include_global)
    finally:
        store.close()

    if summary["scanned"] == 0:
        click.echo("No SKILL.md files found under the scanned roots.")
        return
    click.echo(
        f"Scanned {summary['scanned']} skill(s): {summary['created']} created, "
        f"{summary['refreshed']} refreshed, {summary['edges']} entity/topic "
        f"edge(s) linked."
    )
    if summary.get("suppressed"):
        click.echo(f"{summary['suppressed']} skill(s) were not ingested: a "
                   "forget --everywhere banned their exact text. "
                   "'revien restore' lifts that.")


@skills.command(name="list")
@click.option("--project", default=None, help="Filter to one project_key")
@click.option("--status", default=None, help="Filter to one metadata status (e.g. active)")
@click.option("--format", "output_format", type=click.Choice(["json", "toon"]),
              default=None, help="Machine-readable output (default: human table)")
@click.option("--db", default=None, help="Database path")
def skills_list_cmd(project: Optional[str], status: Optional[str],
                     output_format: Optional[str], db: Optional[str]):
    """List ingested skills — name, origin, status, scope, version, project."""
    from revien.graph.store import GraphStore
    from revien.skills.ingest import list_skills

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        nodes = list_skills(store, project=project, status=status)
    finally:
        store.close()

    rows = [
        {
            "name": n.label,
            "origin": (n.metadata or {}).get("origin", ""),
            "status": (n.metadata or {}).get("status", ""),
            "scope": (n.metadata or {}).get("scope", ""),
            "version": (n.metadata or {}).get("version", ""),
            "project": n.project_key or "",
        }
        for n in nodes
    ]

    if output_format == "json":
        click.echo(json.dumps(rows, indent=2))
        return
    if output_format == "toon":
        from revien.toon import encode
        click.echo(encode({"skills": rows}))
        return

    if not rows:
        click.echo("No skills ingested. Run 'revien skills ingest' first.")
        return
    for r in rows:
        click.echo(
            f"  {r['name']}  [{r['origin']}/{r['status']}/{r['scope']}]"
            f"{' v' + r['version'] if r['version'] else ''}"
            f"{' (' + r['project'] + ')' if r['project'] else ''}"
        )


@skills.command(name="show")
@click.argument("name")
@click.option("--db", default=None, help="Database path")
def skills_show(name: str, db: Optional[str]):
    """Print one skill's body markdown (user-authored wins over engine-proposed)."""
    from revien.graph.store import GraphStore
    from revien.skills.ingest import show_skill

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        node = show_skill(store, name)
    finally:
        store.close()

    if node is None:
        click.echo(f"No skill named '{name}'.")
        sys.exit(1)
    click.echo(node.content)


@skills.command(name="propose")
@click.option("--min-occurrences", "min_occurrences", default=3, show_default=True,
              help="Minimum total occurrences of a step-sequence to propose it.")
@click.option("--min-sessions", "min_sessions", default=2, show_default=True,
              help="Minimum distinct sessions a step-sequence must appear in.")
@click.option("--db", default=None, help="Database path")
def skills_propose(min_occurrences: int, min_sessions: int, db: Optional[str]):
    """Detect repeated ACTION sequences and propose skills from them.

    Idempotent: re-running refreshes an unchanged pattern's occurrence/
    session counts in place rather than duplicating the proposal. Never
    creates an active skill — proposals land with status=proposed,
    origin=engine; accept them with 'revien skills accept <node_id>'."""
    from revien.graph.store import GraphStore
    from revien.skills.proposals import propose_skills

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        summary = propose_skills(
            store, min_occurrences=min_occurrences, min_sessions=min_sessions,
            progress=click.echo,
        )
    finally:
        store.close()

    if summary["detected"] == 0:
        click.echo("No repeated action sequences met the threshold.")
        return
    click.echo(
        f"Detected {summary['detected']} pattern(s): {summary['created']} "
        f"proposed, {summary['updated']} refreshed, {summary['edges']} "
        f"derived-from edge(s) linked."
    )
    for node in summary["proposals"]:
        click.echo(f"  {node.node_id}  {node.label}")


@skills.command(name="accept")
@click.argument("node_id")
@click.option("--db", default=None, help="Database path")
def skills_accept(node_id: str, db: Optional[str]):
    """Accept a proposed skill: status -> active. Origin stays engine —
    exactly one node_id per invocation, no bulk accept."""
    from revien.graph.store import GraphStore
    from revien.skills.proposals import accept_proposal

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        try:
            node = accept_proposal(store, node_id)
        except ValueError as e:
            click.echo(str(e))
            sys.exit(1)
    finally:
        store.close()
    click.echo(f"Accepted: {node.label} (status={node.metadata.get('status')})")


@skills.command(name="decline")
@click.argument("node_id")
@click.option("--db", default=None, help="Database path")
def skills_decline(node_id: str, db: Optional[str]):
    """Decline a proposed skill (declines += 1). The third decline
    soft-invalidates it. Exactly one node_id per invocation, no bulk
    decline."""
    from revien.graph.store import GraphStore
    from revien.skills.proposals import decline_proposal

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        try:
            node = decline_proposal(store, node_id)
        except ValueError as e:
            click.echo(str(e))
            sys.exit(1)
    finally:
        store.close()
    declines = (node.metadata or {}).get("declines", 0)
    if node.invalidated_at is not None:
        click.echo(f"Declined ({declines}x) and invalidated: {node.label}")
    else:
        click.echo(f"Declined ({declines}x): {node.label}")


@main.group()
def sync():
    """Control adapter sync positions."""
    pass


# Columns a Revien database must have in the tables it always creates. A
# table of the same name with other columns means a foreign database.
# The complete column sets of the oldest real Revien database (commit
# 7757085, 0.3.0) for nodes, edges and audit_log; later versions only add
# columns and tables. sync_cursors appeared later, so it is checked only when
# present. A file missing any of these is not a Revien database.
_REVIEN_TABLE_COLUMNS = {
    "nodes": {
        "node_id", "node_type", "label", "content", "source_id", "created_at",
        "last_accessed", "access_count", "metadata", "source_type",
        "confidence", "pinned", "confidence_set_at", "confidence_set_by",
        "source_context", "last_referenced", "invalidated_at",
        "source_modality", "answerable_by_text", "vision_processed",
        "recorded_at", "event_time_start", "event_time_end",
        "event_time_granularity", "event_time_confidence", "event_time_text",
        "valid_from", "valid_until",
    },
    "edges": {
        "edge_id", "edge_type", "source_node_id", "target_node_id", "weight",
        "created_at", "metadata", "confidence", "confidence_set_at",
        "confidence_set_by", "source_context",
    },
    "audit_log": {"id", "node_id", "op", "actor", "ts", "before_json",
                  "after_json"},
    "sync_cursors": {"adapter_name", "cursor", "updated_at"},
}
_REVIEN_REQUIRED_TABLES = {"nodes", "edges", "audit_log"}


def _readonly_uri(p: Path) -> str:
    """SQLite URI for a read that changes nothing on disk. A WAL-mode file
    opened plain read-only still CREATES -wal/-shm sidecars, so when no -wal
    file exists the file is opened immutable (no locks, no sidecars; with no
    -wal file the main file holds everything). When a -wal file exists (an
    open or crashed WAL database) it is opened read-only instead, because
    recent data lives in the WAL and immutable would not see it."""
    import urllib.parse
    resolved = p.resolve()
    quoted = urllib.parse.quote(str(resolved))
    wal = Path(str(resolved) + "-wal")
    if wal.exists():
        return "file:" + quoted + "?mode=ro"
    return "file:" + quoted + "?mode=ro&immutable=1"


def _sqlite_problem(db_path: str) -> Optional[str]:
    """One-line reason this path cannot be a Revien database, else None.
    Checked BEFORE opening a GraphStore, which would write a full schema into
    an empty or foreign file.

    A SQLite header alone is not enough (any app's database has one), so the
    file is opened READ-ONLY (see ``_readonly_uri``: it cannot be written and
    creates no journal, -wal or -shm file, even for a WAL-mode database) and
    its own catalog is read: it must hold the
    ``nodes``, ``edges`` and ``audit_log`` tables every Revien version has,
    each with every column the oldest real version has. ``sync_cursors``,
    added later, is checked only when present."""
    import sqlite3

    p = Path(db_path)
    if not p.exists():
        return "No Revien database found. Run 'revien start' first."
    if p.is_dir():
        return f"{db_path} is a directory, not a Revien database."
    try:
        with open(p, "rb") as f:
            head = f.read(16)
    except OSError as e:
        return f"Cannot read {db_path}: {e.strerror or e}"
    if not head:
        return f"{db_path} is empty, not a Revien database."
    if head != b"SQLite format 3\x00":
        return f"{db_path} is not a SQLite database."
    uri = _readonly_uri(p)
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            if not _REVIEN_REQUIRED_TABLES <= tables:
                return (f"{db_path} is a SQLite database but not a Revien "
                        "database (no nodes/edges/audit_log tables). "
                        "Not touching it.")
            for table, wanted in _REVIEN_TABLE_COLUMNS.items():
                if table not in tables:
                    continue
                have = {r[1] for r in conn.execute(
                    f"PRAGMA table_info({table})")}
                if not wanted <= have:
                    return (f"{db_path} has a '{table}' table that is not "
                            "Revien's, so it is not a Revien database. "
                            "Not touching it.")
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        return f"Cannot read {db_path}: {e}"
    except sqlite3.DatabaseError:
        return f"{db_path} is not a SQLite database."
    return None


@sync.command(name="reset")
@click.argument("adapter_name")
@click.option("--db", default=None, help="Database path")
@click.option("--orphaned", is_flag=True,
              help="Reset a saved sync position whose adapter is no longer in "
                   "your config (only works when a saved position exists)")
@click.option("--daemon-url", default="http://127.0.0.1:7437",
              help="Where to look for a running daemon (loopback addresses "
                   "only; default http://127.0.0.1:7437)")
@click.option("--yes", "yes", is_flag=True,
              help="Reset even though you have forgotten something: required "
                   "once this database has an explicit forget in it")
def sync_reset(adapter_name: str, db: Optional[str], orphaned: bool,
               daemon_url: str, yes: bool):
    """Clear a connected adapter's sync position so its next sync re-reads
    its full history (for example 'revien sync reset claude-code').

    Only adapters that tag every item with a stable key (claude-code, codex)
    can be reset: their sessions are refreshed in place, not duplicated, so
    the reset is safe to repeat. Other adapters would ingest everything again
    as new nodes, so they are refused. Nothing is deleted from the graph.
    A sync already running in a current daemon when you reset cannot undo
    the reset: its newer position is discarded. A daemon started before
    this protection existed cannot do that, so the reset is refused while
    one is running ("restart required"): it checks the daemon heartbeats in
    the database and asks the daemon at --daemon-url for /v1/health. No
    daemon answering means nothing to restart. The new position is used at
    the next sync (default every 6 hours).

    --orphaned resets a saved position whose adapter was removed from your
    config. Its type cannot be checked, so the reset is allowed only because
    a saved position exists.

    Once you have forgotten something, a reset needs --yes: the full re-read
    shows every source to Revien again. A source that was re-read is captured
    again minus the sentences that held what was forgotten from it (they stay
    on a keyed denylist, bound to that source's identity; with --everywhere,
    whatever the source)."""
    import sqlite3

    from revien.adapters import (
        _ADAPTER_CLASSES, adapter_type_rereads_idempotently)
    from revien.graph.store import GraphStore

    if adapter_name.startswith("usage:"):
        click.echo(f"'{adapter_name}' is the usage observer's position, not an "
                   "adapter. To rescan, run 'revien usage scan --full'.")
        raise SystemExit(1)

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    problem = _sqlite_problem(db_path)
    if problem:
        click.echo(problem)
        raise SystemExit(1)

    adapters = config.get("adapters") or {}
    connected = list(adapters.keys())
    entry = adapters.get(adapter_name)
    connected_txt = ", ".join(sorted(connected)) or "none"

    # Everything that can be decided from the config and a read-only look at
    # the database is decided BEFORE a GraphStore is opened: opening one
    # migrates (writes to) the database, and a mistyped name must not do that.
    if entry is None:
        if not orphaned:
            if _has_sync_row(db_path, adapter_name):
                click.echo(
                    f"'{adapter_name}' has a saved sync position but is not a "
                    "connected adapter in your config, so its type cannot be "
                    "checked and it is not reset. Connected adapters: "
                    f"{connected_txt}. Connect it again ('revien connect'), "
                    "use one of those names, or re-run with --orphaned to "
                    "reset the saved position anyway.")
            else:
                click.echo(f"No connected adapter named '{adapter_name}'. "
                           f"Connected adapters: {connected_txt}.")
            raise SystemExit(1)
        if not _has_sync_row(db_path, adapter_name):
            click.echo(f"No saved sync position for '{adapter_name}', so "
                       "--orphaned has nothing to reset. Connected adapters: "
                       f"{connected_txt}. Nothing was changed.")
            raise SystemExit(1)
    else:
        if orphaned:
            click.echo(f"'{adapter_name}' is a connected adapter, so "
                       "--orphaned does not apply. Run the command without it.")
            raise SystemExit(1)
        atype = entry.get("type", "") if isinstance(entry, dict) else None
        if not isinstance(atype, str) or atype not in _ADAPTER_CLASSES:
            click.echo(
                f"Cannot reset '{adapter_name}': its config entry has "
                + ("no usable 'type'" if not atype or not isinstance(atype, str)
                   else f"unknown type '{atype}'")
                + " (known types: " + ", ".join(sorted(_ADAPTER_CLASSES))
                + "). Fix the entry in the config file. Nothing was changed.")
            raise SystemExit(1)
        if not adapter_type_rereads_idempotently(atype):
            click.echo(
                f"Refusing to reset '{adapter_name}': its items carry no stable "
                "key, so a full re-read would add everything again as new "
                "nodes instead of refreshing them. Only claude-code and codex "
                "adapters can be reset.")
            raise SystemExit(1)

    # A reset re-reads every source. With an explicit forget in the database
    # that is not a routine step: say what it does and make the user mean it.
    forgets = _forget_count(db_path)
    if forgets and not yes:
        click.echo(
            f"This database has {forgets} forget record(s). A reset makes the "
            "next sync re-read every source in full. Each source is captured "
            "again minus the sentences that held what was forgotten from it "
            "(kept out by their keyed fingerprint, bound to that source's "
            "identity; from any source for a forget --everywhere). Re-run "
            "with --yes to reset anyway. "
            "Nothing was changed.")
        raise SystemExit(1)

    # A reset is only safe against a daemon that fences its sync positions.
    # Decided before anything is written (and before a GraphStore migrates).
    old = _old_daemon_problem(db_path, daemon_url)
    if old:
        click.echo(old)
        click.echo("Nothing was changed.")
        raise SystemExit(1)

    try:
        store = GraphStore(db_path=db_path)
    except (sqlite3.Error, OSError) as e:
        click.echo(f"Cannot open {db_path}: {e}")
        raise SystemExit(1)
    try:
        had_position = store.reset_sync_cursor(adapter_name)
    except sqlite3.Error as e:
        click.echo(f"Database error: {e}")
        raise SystemExit(1)
    finally:
        store.close()

    if had_position is None:
        click.echo(f"'{adapter_name}' has no saved sync position, so its next "
                   "sync already reads its full history. Nothing to reset.")
        return
    if not had_position:
        click.echo(f"'{adapter_name}' was already reset; its next sync reads "
                   "its full history. The reset was recorded again.")
    else:
        click.echo(f"Cleared the sync position for '{adapter_name}'.")
    if orphaned:
        click.echo("Its adapter type is unknown (it is not in your config). If "
                   "you connect it again as anything other than claude-code or "
                   "codex, its first sync adds everything as new nodes.")
    click.echo("It takes effect at the next sync (default every 6 hours), "
               "which re-reads the full history. Already captured sessions "
               "are refreshed in place, not duplicated. Nothing was deleted.")
    click.echo("A sync that is running right now in a current daemon cannot "
               "undo this reset: when it finishes, its position is "
               "discarded. If the daemon is not running, start it with "
               "'revien start'.")


def _old_daemon_problem(db_path: str, daemon_url: str,
                        need_features=None, what: str = "reset") -> Optional[str]:
    """Why an operation that depends on a daemon feature (sync-reset fencing,
    or the forget gate) must not run now, else None. Two independent sources:

    * the heartbeats current daemons write into THIS database: a live one that
      lacks the feature refuses;
    * ONE loopback health check (revien/loopback_probe.py, the single pinned
      module that may open a connection): an answer without a features list
      is a daemon from before fencing. Whether it serves this database cannot
      be told from its answer, so it refuses either way.

    No heartbeat and nothing listening: no daemon is running, nothing to
    restart. Something listening that does not answer as a current Revien
    daemon (a hang, an HTTP error, not JSON) refuses.

    ``what`` is "reset" (sync reset) or "forget"; it only changes the wording."""
    from revien.daemon.heartbeat import RESET_REQUIRES, live_heartbeats_readonly
    from revien.loopback_probe import probe_daemon

    need = set(RESET_REQUIRES if need_features is None else need_features)
    forgetting = what == "forget"
    again = "forget again" if forgetting else "run this command again"
    restart = ("Restart required: stop that daemon ('Ctrl+C' in its "
               "terminal), start it again with 'revien start', then "
               + again + ".")
    protects = ("capture a forgotten source again" if forgetting
                else "protect a reset from a sync that is running")
    for beat in live_heartbeats_readonly(_readonly_uri(Path(db_path))):
        if not need <= set(beat["features"]):
            if forgetting:
                return ("A daemon (pid " + str(beat["pid"]) + ") is serving "
                        "this database from before forget protection: it "
                        "would " + protects + ". " + restart)
            return ("A daemon (pid " + str(beat["pid"]) + ") is serving this "
                    "database but does not protect a reset from a sync that "
                    "is running. " + restart)
    health = probe_daemon(daemon_url)
    if health is None:
        return None
    if "unanswered" in health:
        # something accepted the connection and did not give a current Revien
        # health answer: it may be a daemon from before the protection, and
        # one that syncs afterwards writes the forgotten text back
        return (f"Something is listening at {daemon_url} but did not give a "
                f"current Revien health answer ({health['unanswered']}). It "
                "may be a Revien daemon that is busy, hung or from before "
                "this protection, so " + ("the forget" if forgetting else "this")
                + " is refused rather than guessed. " + restart)
    if "rejected" in health:
        if forgetting:
            return (f"--daemon-url {daemon_url!r} was not used "
                    f"({health['rejected']}). Only http://127.0.0.1:PORT (any "
                    "127.x.x.x), http://[::1]:PORT or http://localhost:PORT "
                    "can be checked, so the forget cannot tell whether a "
                    "daemon from before forget protection is running. Use a "
                    "loopback --daemon-url.")
        return (f"--daemon-url {daemon_url!r} was not used ({health['rejected']}). "
                "Only http://127.0.0.1:PORT (any 127.x.x.x), http://[::1]:PORT "
                "or http://localhost:PORT can be checked, so the reset cannot "
                "tell whether a daemon from before sync-reset protection is "
                "running. Use a loopback --daemon-url.")
    ver = health.get("version") or "unknown"
    if health["features"] is None:
        if forgetting:
            return (f"A Revien daemon answering at {daemon_url} (version "
                    f"{ver}) was started before forget protection existed. It "
                    "would capture the forgotten source again. The answer "
                    "does not say which database it serves, so this is "
                    "refused either way. " + restart)
        return (f"A Revien daemon answering at {daemon_url} (version {ver}) was "
                "started before sync-reset protection existed. A sync it is "
                "running would write its old position back and undo the "
                "reset, and it cannot read a reset position. The answer does "
                "not say which database it serves, so this is refused either "
                "way. " + restart)
    if not need <= set(health["features"]):
        if forgetting:
            return (f"The daemon answering at {daemon_url} (version {ver}) "
                    "does not report forget protection. " + restart)
        return (f"The daemon answering at {daemon_url} (version {ver}) does "
                "not report sync-reset protection. " + restart)
    return None


def _forget_count(db_path: str) -> int:
    """How many explicit forgets this database holds: suppression records
    plus forget audit entries. Read-only (see _readonly_uri): creates no
    files and does not migrate."""
    import sqlite3

    uri = _readonly_uri(Path(db_path))
    total = 0
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            for sql in ("SELECT COUNT(*) FROM suppressions",
                        "SELECT COUNT(*) FROM audit_log WHERE op = 'forget'"):
                try:
                    total += conn.execute(sql).fetchone()[0]
                except sqlite3.Error:
                    pass
        finally:
            conn.close()
    except sqlite3.Error:
        return 0
    return total


def _has_sync_row(db_path: str, name: str) -> bool:
    """True if sync_cursors holds a row for ``name``. Read-only (see
    _readonly_uri): creates no files."""
    import sqlite3

    uri = _readonly_uri(Path(db_path))
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=1.0)
        try:
            return conn.execute(
                "SELECT 1 FROM sync_cursors WHERE adapter_name = ?",
                (name,)).fetchone() is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


@main.command()
@click.argument("node_id", required=False)
@click.option("--wording", "wording", default=None, metavar="STRING",
              help="Forget a literal string (a key, a token, a password) "
                   "wherever it appears: every node, the audit trail, recall "
                   "records, training data and snapshots, and keep the same "
                   "characters out of every later capture. Use --wording - to "
                   "read the string from standard input, so it does not sit "
                   "in your shell history. The string is never stored, "
                   "logged or printed")
@click.option("--everywhere", is_flag=True,
              help="Also cut the exact forgotten wording (this node's own "
                   "wording) out of every other node that holds it, in place "
                   "(nothing is deleted), and keep it out of every source. "
                   "Without it, a forget is per source and the response names "
                   "the other sources that still hold the wording")
@click.option("--db", default=None, help="Database path")
@click.option("--yes", "yes", is_flag=True, help="Do not ask before forgetting")
@click.option("--preview", is_flag=True,
              help="Show what would be forgotten and change nothing")
@click.option("--daemon-url", default="http://127.0.0.1:7437",
              help="Where to look for a running daemon (loopback addresses "
                   "only; default http://127.0.0.1:7437)")
@click.option("--json-output", is_flag=True, help="Output the result as JSON")
def forget(node_id: Optional[str], wording: Optional[str], everywhere: bool,
           db: Optional[str], yes: bool, preview: bool, daemon_url: str,
           json_output: bool):
    """Forget one specific thing: a fact, an entity, a decision or a
    preference. Privacy, not housekeeping.

    Removes the node, cuts its wording out of the conversation it came from,
    scrubs every copy Revien keeps (audit history, recall records, vectors,
    training data, the database file and its write-ahead log), and keeps that
    wording out of later captures of the same source (named by its id, or its
    real path or inode). A copy of the source under a different identity is
    not recognised; --everywhere keeps the wording out of every source. The
    result says exactly what was removed, cut back and kept, what is left that
    Revien cannot reach (your own source file, exports, distilled notes), and
    which way back it does and does not close.

    A whole conversation is not one thing to forget: to remove it, run
    `revien delete NODE_ID`. To remove a secret that should never have been
    recorded, name its exact characters: `revien forget --wording -` (the
    string on standard input). For a leaked key, rotate it first and forget
    second: a forget cannot recall what already left this machine. A running
    daemon from before forget protection makes this refuse; restart it
    first."""
    import sqlite3

    from revien.graph.operations import GraphOperations
    from revien.graph.store import GraphStore

    if wording is not None:
        if node_id or everywhere:
            click.echo("--wording takes no NODE_ID and no --everywhere: it "
                       "always reaches every node and every later capture.")
            raise SystemExit(1)
        if wording == "-":
            wording = click.get_text_stream("stdin").read()
            if wording.endswith("\r\n"):
                wording = wording[:-2]
            elif wording.endswith("\n"):
                wording = wording[:-1]
    elif not node_id:
        click.echo("Give a NODE_ID, or --wording STRING.")
        raise SystemExit(1)

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    problem = _sqlite_problem(db_path)
    if problem:
        click.echo(problem)
        raise SystemExit(1)

    from revien.daemon.heartbeat import FORGET_REQUIRES

    old = _old_daemon_problem(db_path, daemon_url,
                              need_features=FORGET_REQUIRES, what="forget")
    if old:
        click.echo(old)
        click.echo("Nothing was changed.")
        raise SystemExit(1)

    try:
        store = GraphStore(db_path=db_path)
    except (sqlite3.Error, OSError) as e:
        click.echo(f"Cannot open {db_path}: {e}")
        raise SystemExit(1)
    if wording is not None:
        try:
            ops = GraphOperations(store)
            plan = ops.forget_wording(wording, actor="cli:forget", dry_run=True)
            if plan.get("status") == "refused":
                click.echo(plan["message"])
                raise SystemExit(1)
            if preview:
                if json_output:
                    click.echo(json.dumps(plan, indent=2))
                else:
                    click.echo(plan["blast_radius"])
                    for t, n in sorted(plan["would_cut_in"].items()):
                        click.echo(f"  {t}: {n} row(s)")
                return
            if not yes:
                click.echo(plan["blast_radius"])
                if not plan["would_cut_occurrences"]:
                    click.echo("The string is not in this database yet; it "
                               "would still be kept out of later captures.")
                if not click.confirm("Forget this string?"):
                    click.echo("Nothing was changed.")
                    raise SystemExit(1)
            result = ops.forget_wording(wording, actor="cli:forget")
        finally:
            store.close()
        if json_output:
            click.echo(json.dumps(result, indent=2))
            return
        click.echo(result["message"])
        if result["status"] in ("refused", "busy"):
            raise SystemExit(1)
        return

    try:
        ops = GraphOperations(store)
        plan = ops._forget_engine().forget(
            node_id, dry_run=True, everywhere=everywhere)
        status = plan.get("status")
        if status in ("not_found", "skipped"):
            click.echo(f"No node {node_id} to forget.")
            raise SystemExit(1)
        if status == "refused":
            click.echo(plan["message"])
            raise SystemExit(1)
        if preview:
            if json_output:
                click.echo(json.dumps(plan, indent=2))
            else:
                click.echo(plan["blast_radius"])
                click.echo(f"  would remove  : {len(plan['would_remove'])} node(s)")
                click.echo(f"  would cut back: {len(plan['would_cut_back'])} shared node(s)")
                click.echo(f"  would keep    : {len(plan['would_keep'])} (another unit vouches)")
                click.echo(f"  unproven      : {len(plan['unproven'])} kept and listed")
                for u in plan["not_bound"]:
                    click.echo(f"  note: {u['note']}")
                if plan.get("would_cut_in_place"):
                    click.echo(f"  would cut in place: {len(plan['would_cut_in_place'])} "
                               "node(s) of other sources")
                click.echo("  would stop coming back: "
                           + " | ".join(plan["reentry"]["stops"]))
                click.echo("  would not stop: "
                           + " | ".join(plan["reentry"]["does_not_stop"]))
            return
        if not yes:
            click.echo(plan["blast_radius"])
            click.echo(f"  {len(plan['would_remove'])} node(s) removed, "
                       f"{len(plan['would_cut_back'])} cut back, "
                       f"{len(plan['would_keep'])} kept.")
            if not click.confirm("Forget this?"):
                click.echo("Nothing was changed.")
                raise SystemExit(1)
        result = ops.forget_node(node_id, actor="cli:forget",
                                 everywhere=everywhere)
    finally:
        store.close()

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    if result["status"] in ("refused", "busy"):
        click.echo(result["message"])
        raise SystemExit(1)
    click.echo(result["message"])
    click.echo(f"  removed {result['removed_nodes']} node(s); the graph went "
               f"from {result['nodes_before']} to {result['nodes_after']} nodes "
               f"and {result['edges_before']} to {result['edges_after']} edges.")
    for u in result["not_bound"]:
        click.echo(f"  {u['note']}")


@main.command(name="delete")
@click.argument("node_id")
@click.option("--db", default=None, help="Database path")
@click.option("--yes", "yes", is_flag=True, help="Do not ask before deleting")
def delete_cmd(node_id: str, db: Optional[str], yes: bool):
    """Delete one node now (operational, not privacy).

    Removes the node and its edges and writes one content-free `delete` audit
    entry. It writes no forget record, so the content can be captured again if
    its source still holds it, and the facts taken from a conversation are
    separate nodes that stay. To keep a secret out for good, use
    `revien forget --wording`."""
    import sqlite3

    from revien.graph.operations import GraphOperations
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    problem = _sqlite_problem(db_path)
    if problem:
        click.echo(problem)
        raise SystemExit(1)
    try:
        store = GraphStore(db_path=db_path)
    except (sqlite3.Error, OSError) as e:
        click.echo(f"Cannot open {db_path}: {e}")
        raise SystemExit(1)
    try:
        node = store.get_node(node_id)
        if node is None:
            click.echo(f"No node {node_id} to delete.")
            raise SystemExit(1)
        if not yes and not click.confirm(
                f"Delete this {node.node_type.value} node ({node_id})?"):
            click.echo("Nothing was changed.")
            raise SystemExit(1)
        GraphOperations(store).delete_node(node_id, actor="cli:delete")
    finally:
        store.close()
    click.echo(f"Deleted {node_id}. It can be captured again if its source "
               "still holds it; no forget record was written.")


@main.command()
@click.argument("target", required=False)
@click.option("--db", default=None, help="Database path")
@click.option("--list", "list_only", is_flag=True,
              help="List the active forget records and change nothing")
@click.option("--yes", "yes", is_flag=True,
              help="For a pattern: lift what it lists without asking")
def restore(target: Optional[str], db: Optional[str], list_only: bool, yes: bool):
    """Let forgotten wording be captured again.

    'revien restore /path/to/file' lifts the forget records bound to that ONE
    file; 'revien restore forget:<id>' lifts one record by its id. A bare name
    never lifts several paths. To lift several, give an explicit pattern, for
    example: revien restore 'projA/**'. It lists everything it would lift and
    asks before lifting. Every restore is audited. It does not bring a removed
    node back."""
    import sqlite3

    from revien.forget.restore import plan_restore, restore as do_restore
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    problem = _sqlite_problem(db_path)
    if problem:
        click.echo(problem)
        raise SystemExit(1)
    try:
        store = GraphStore(db_path=db_path)
    except (sqlite3.Error, OSError) as e:
        click.echo(f"Cannot open {db_path}: {e}")
        raise SystemExit(1)
    try:
        if list_only or not target:
            from revien.forget.restore import liftable
            rows = liftable(store)
            if not rows:
                click.echo("No active forget records.")
                return
            click.echo(f"{len(rows)} active forget record(s):")
            for r in rows:
                if r.get("literal_fps"):
                    click.echo(
                        f"  #{r['id']} {r['unit_id']} (a secret string kept out "
                        f"of every source by forget --wording), since "
                        f"{r['created_at'][:19]}")
                    continue
                if r["scope"] == "global":
                    click.echo(
                        f"  #{r['id']} {r['unit_id']} (an exact wording kept "
                        f"out of every source by forget --everywhere), since "
                        f"{r['created_at'][:19]}")
                    continue
                where = ", ".join(sorted(
                    k[len("unit:"):] for k in r["path_keys"]
                    if k.startswith("unit:"))) or r["unit_id"]
                click.echo(
                    f"  #{r['id']} {r['unit_id']} ({len(r['sentence_fps'])} "
                    f"sentence(s) kept out of {where}), since "
                    f"{r['created_at'][:19]}")
            return
        plan = plan_restore(store, target)
        if plan["status"] in ("error", "none", "ambiguous"):
            click.echo(plan["message"])
            for c in plan.get("candidates", []):
                click.echo(f"  #{c['id']} {c['unit_id']} "
                           f"({', '.join(c['sources']) or 'every source'})")
            raise SystemExit(1)
        click.echo(f"Will lift {len(plan['lifts'])} forget record(s):")
        for item in plan["lifts"]:
            click.echo(f"  #{item['id']} {item['unit_id']} "
                       f"({', '.join(item['sources']) or 'every source'})")
        if plan["mode"] == "bulk" and not yes:
            if not click.confirm("Lift all of these?"):
                click.echo("Nothing was changed.")
                raise SystemExit(1)
        done = do_restore(store, target, actor="cli:restore", confirmed=True)
    finally:
        store.close()
    click.echo(f"Restored {len(done.get('lifted', []))} forget record(s). The "
               "wording can be captured again at the next sync.")


@main.group()
def usage():
    """Did recalled memory actually get used? Judges each revien_recall in
    your Claude Code sessions against the answer that followed it. Records
    only (shadow) unless REVIEN_USAGE=apply."""
    pass


@usage.command(name="scan")
@click.option("--db", default=None, help="Database path")
@click.option("--path", "session_dir", default=None, type=click.Path(),
              help="Claude Code projects directory (default: the connected "
                   "adapter's, else auto-detect)")
@click.option("--mode", type=click.Choice(["shadow", "apply"]), default=None,
              help="Override REVIEN_USAGE for this run")
@click.option("--full", is_flag=True,
              help="Rescan every session, not only those changed since the "
                   "last scan (safe: judged pairs are skipped)")
@click.option("--json-output", is_flag=True, help="Output the report as JSON")
def usage_scan(db: Optional[str], session_dir: Optional[str], mode: Optional[str],
               full: bool, json_output: bool):
    """Judge recall episodes in Claude Code sessions."""
    import json as _json
    from revien.adapters.claude_code import ClaudeCodeAdapter
    from revien.graph.store import GraphStore
    from revien.retrieval.engine import RetrievalEngine
    from revien.semantic.index import SemanticIndex
    from revien.usage import UsageObserver

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return
    if session_dir is None:
        connected = (config.get("adapters") or {}).get("claude-code") or {}
        session_dir = connected.get("session_dir")
    adapter = ClaudeCodeAdapter(session_dir=session_dir)

    store = GraphStore(db_path=db_path)
    try:
        semantic = SemanticIndex(store)
        engine = RetrievalEngine(store, semantic=semantic)
        report = UsageObserver(
            store, engine=engine, semantic=semantic, mode=mode,
        ).run_claude_code(adapter, full=full)
    finally:
        store.close()

    if json_output:
        click.echo(_json.dumps(report.model_dump(), indent=2))
        return
    click.echo(f"\nUsage scan ({report.mode}): {report.files_scanned} session "
               f"file(s), {report.episodes} recall episode(s)\n")
    click.echo(f"  judged    : {report.judged} new ({report.used} used, "
               f"{report.unused} not used)")
    click.echo(f"  abstained : {report.abstained}")
    click.echo(f"  skipped   : {report.already_judged} already judged, "
               f"{report.missing} no longer in the graph")
    if report.mode == "apply":
        click.echo(f"  applied   : {report.applied} use(s) counted")
    else:
        click.echo("  applied   : 0 (shadow records only; REVIEN_USAGE=apply "
                   "or --mode apply to count uses)")
    click.echo("  semantic  : " + ("active" if report.semantic_active else
               "OFF - no similarity signal, so the rule abstains"))
    click.echo()


@usage.command(name="list")
@click.option("--db", default=None, help="Database path")
@click.option("--limit", default=20, show_default=True, help="Rows to show")
@click.option("--filter", "which", type=click.Choice(["all", "used", "unused"]),
              default="all", show_default=True, help="Which judgments to show")
@click.option("--json-output", is_flag=True, help="Output rows as JSON")
def usage_list(db: Optional[str], limit: int, which: str, json_output: bool):
    """Show recent usage judgments with the evidence behind each."""
    import json as _json
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        used = {"all": None, "used": True, "unused": False}[which]
        rows = store.list_use_judgments(limit=limit, used=used)
        counts = store.use_judgment_counts()
        for row in rows:
            node = store.get_node(row["node_id"])
            row["label"] = node.label if node is not None else "(node no longer exists)"
    finally:
        store.close()

    if json_output:
        click.echo(_json.dumps({"counts": counts, "judgments": rows}, indent=2))
        return
    click.echo(f"\n{counts['judged']} judgment(s): {counts['used']} used, "
               f"{counts['unused']} not used, {counts['applied']} applied"
               + (f", {counts['apply_unknown']} apply unknown (in progress, "
                  "or interrupted: mark_used may have run)"
                  if counts["apply_unknown"] else "") + "\n")
    for row in rows:
        similarity = ("-" if row["similarity"] is None
                      else f"{row['similarity']:.2f}")
        verdict = "USED  " if row["used"] else "unused"
        mark = "  [apply unknown]" if row["apply_state"] == "unknown" else ""
        click.echo(f"  {verdict} sim={similarity} terms={len(row['shared_terms'])} "
                   f"[{row['mode']}] {row['label'][:60]}{mark}")
        if row["shared_terms"]:
            click.echo(f"         shared: {', '.join(row['shared_terms'][:8])}")
        click.echo(f"         query : {row['query'][:70]}")
        if row["reason"]:
            click.echo(f"         reason: {row['reason'][:70]}")
    click.echo()


@usage.command(name="flags")
@click.option("--db", default=None, help="Database path")
@click.option("--dismiss", "dismiss_id", default=None,
              help="Mark one flag (by recall_id) as seen. Kept, not deleted.")
@click.option("--all", "include_dismissed", is_flag=True,
              help="Also show flags already dismissed")
@click.option("--json-output", is_flag=True, help="Output flags as JSON")
def usage_flags(db: Optional[str], dismiss_id: Optional[str],
                include_dismissed: bool, json_output: bool):
    """Recalls whose receipt expired before their answer was judged."""
    import json as _json
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())
    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    try:
        store.expire_recall_receipts()
        if dismiss_id:
            if store.dismiss_receipt_flag(dismiss_id):
                click.echo(f"Dismissed {dismiss_id} (kept in the record).")
            else:
                click.echo(f"No open flag {dismiss_id}.")
                sys.exit(1)
            return
        flags = store.list_receipt_flags(include_dismissed)
        for flag in flags:
            flag["labels"] = []
            for node_id in flag["node_ids"]:
                node = store.get_node(node_id)
                flag["labels"].append(node.label if node is not None
                                      else "(node no longer exists)")
    finally:
        store.close()

    if json_output:
        click.echo(_json.dumps({"flags": flags}, indent=2))
        return
    if not flags:
        click.echo("No usage flags: every receipted recall was judged in time.")
        return
    click.echo(f"\n{len(flags)} recall(s) expired unjudged:\n")
    for flag in flags:
        status = " (dismissed)" if flag["status"] == "dismissed" else ""
        click.echo(f"  {flag['recall_id']}{status}  runtime={flag['runtime']}  "
                   f"expired {flag['expires_at'][:19]}")
        click.echo(f"         query : {flag['query'][:70]}")
        for label in flag["labels"][:8]:
            click.echo(f"         memory: {label[:66]}")
    click.echo()


if __name__ == "__main__":
    main()

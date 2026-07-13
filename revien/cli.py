"""
Revien CLI — Command-line interface for the Revien memory engine.
Three lines to persistent memory:
    pip install revien
    revien connect claude-code
    revien start
"""

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Click is required: pip install click")
    sys.exit(1)


def _default_db_path() -> str:
    revien_dir = Path.home() / ".revien"
    revien_dir.mkdir(parents=True, exist_ok=True)
    return str(revien_dir / "revien.db")


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
@click.version_option(version="0.3.0", prog_name="revien")
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
    from revien.mcp_server import MCP_AVAILABLE, build_mcp_server

    if not MCP_AVAILABLE:
        click.echo("The MCP SDK is not installed. Install with: pip install revien[mcp]")
        sys.exit(1)

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    # stdout is the protocol channel on stdio — anything human goes to stderr.
    click.echo(f"Revien MCP server (stdio) — database: {db_path}", err=True)
    server = build_mcp_server(db_path=db_path)
    server.run(transport="stdio")


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
        mcp_block = (
            "\n[mcp_servers.revien]\n"
            'command = "revien"\n'
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
        version = "0.3.0"
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
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--format", "output_format", type=click.Choice(["json", "toon"]),
              default="json", show_default=True,
              help="Wire format: 'toon' prints the full recall payload as "
                   "TOON (Token-Oriented Object Notation — same data, fewer "
                   "tokens for a consuming LLM); 'json' keeps the existing "
                   "behavior (human-readable, or JSON with --json-output)")
def recall(query: str, top: int, db: Optional[str], as_of: Optional[str],
           json_output: bool, output_format: str):
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

    store = GraphStore(db_path=db_path)
    engine = RetrievalEngine(store)

    try:
        response = engine.recall(query, top_n=top, as_of=as_of_dt)

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
                    }
                    for r in response.results
                ],
                "nodes_examined": response.nodes_examined,
                "retrieval_time_ms": response.retrieval_time_ms,
                "semantic_active": response.semantic_active,
                "semantic_note": response.semantic_note,
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
                    }
                    for r in response.results
                ],
                "nodes_examined": response.nodes_examined,
                "retrieval_time_ms": response.retrieval_time_ms,
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
                click.echo(f"  [{i}] {r.label}")
                click.echo(f"      Type: {r.node_type} | Score: {r.score:.3f}")
                click.echo(f"      {r.content[:120]}{'...' if len(r.content) > 120 else ''}")
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
                click.echo(f"      ({p['source_context']}, {p['created_at'][:10]})")
            click.echo()
    finally:
        store.close()


@main.command()
@click.option("--db", default=None, help="Database path")
@click.option("--reindex", is_flag=True,
              help="Also backfill the semantic index (recovery, not routine)")
@click.option("--invalidate-orphans", "invalidate_orphans", is_flag=True,
              help="Soft-invalidate edgeless nodes (reversible; default is report-only)")
@click.option("--json-output", is_flag=True, help="Output the full report as JSON")
def dream(db: Optional[str], reindex: bool, invalidate_orphans: bool,
          json_output: bool):
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
        ).run(reindex=reindex, invalidate_orphans=invalidate_orphans)

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
        result = pipeline.ingest(IngestionInput(
            source_id=source,
            content=content,
        ))
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
        nodes = edges = 0
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
                metadata=item.get("metadata", {}),
                links=item.get("links", []),
                curated=True,
            ))
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
        click.echo(f"Revien Memory Engine v0.1.0")
        click.echo(f"Database: {db_path}")
        click.echo(f"Nodes: {node_count}")
        click.echo(f"Edges: {edge_count}")

        # Show adapter config
        adapters = config.get("adapters", {})
        if adapters:
            click.echo(f"Connected adapters: {', '.join(adapters.keys())}")
        else:
            click.echo("No adapters connected. Run 'revien connect <system>'")
    finally:
        store.close()


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
            click.echo(f"Exported {len(graph.nodes)} nodes, "
                       f"{len(graph.edges)} edges to {target}")
        else:
            click.echo(json_str)
    finally:
        store.close()


@main.command(name="import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", default=None, help="Database path")
@click.option("--merge", is_flag=True,
              help="Import into a non-empty database; nodes/edges whose IDs "
                   "already exist are skipped, never overwritten")
def import_(file: str, db: Optional[str], merge: bool):
    """Import a graph exported by `revien export`: revien import graph.json

    Same schema and code path as POST /v1/graph/import. Refuses a non-empty
    target database unless --merge — an import must never silently eat an
    existing graph.
    """
    from revien.graph.schema import Graph
    from revien.graph.store import GraphStore

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

    store = GraphStore(db_path=db_path)
    try:
        existing_nodes = store.count_nodes()
        existing_edges = store.count_edges()

        if (existing_nodes or existing_edges) and not merge:
            click.echo(
                f"Target database is not empty ({existing_nodes} nodes, "
                f"{existing_edges} edges): {db_path}"
            )
            click.echo("Use --merge to import on top of it, or --db to "
                       "point at a fresh file.")
            sys.exit(1)

        if merge and (existing_nodes or existing_edges):
            added_n = skipped_n = added_e = skipped_e = 0
            for node in graph.nodes:
                if store.get_node(node.node_id) is not None:
                    skipped_n += 1
                else:
                    store.add_node(node)
                    added_n += 1
            for edge in graph.edges:
                if store.get_edge(edge.edge_id) is not None:
                    skipped_e += 1
                else:
                    store.add_edge(edge)
                    added_e += 1
            click.echo(f"Merged {added_n} nodes, {added_e} edges "
                       f"(skipped {skipped_n} existing nodes, "
                       f"{skipped_e} existing edges).")
        else:
            # Empty target: the same code path the daemon's import uses.
            store.import_graph(graph, clear_existing=False)
            click.echo(f"Imported {len(graph.nodes)} nodes, "
                       f"{len(graph.edges)} edges into {db_path}")

        click.echo(f"Graph total: {store.count_nodes()} nodes, "
                   f"{store.count_edges()} edges")
    finally:
        store.close()


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


if __name__ == "__main__":
    main()

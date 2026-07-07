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
@click.version_option(version="0.2.1", prog_name="revien")
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
    )
    daemon.start()


@main.command()
@click.argument("system")
@click.option("--path", default=None, help="Custom path for the adapter")
def connect(system: str, path: Optional[str]):
    """Connect an AI system to Revien. Supported: claude-code, file-watcher, api, obsidian."""
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
        click.echo("Run 'revien start' to begin syncing.")

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
        click.echo("Supported: claude-code, file-watcher, api, obsidian")


@main.command()
@click.argument("query")
@click.option("--top", default=5, help="Number of results to return")
@click.option("--db", default=None, help="Database path")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def recall(query: str, top: int, db: Optional[str], json_output: bool):
    """Query Revien memory from the command line."""
    from revien.graph.store import GraphStore
    from revien.retrieval.engine import RetrievalEngine

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    if not Path(db_path).exists():
        click.echo("No Revien database found. Run 'revien start' first.")
        return

    store = GraphStore(db_path=db_path)
    engine = RetrievalEngine(store)

    try:
        response = engine.recall(query, top_n=top)

        if json_output:
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
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--db", default=None, help="Database path")
def export(output: Optional[str], db: Optional[str]):
    """Export the full graph as JSON."""
    from revien.graph.store import GraphStore

    config = _load_config()
    db_path = db or config.get("db_path", _default_db_path())

    store = GraphStore(db_path=db_path)
    try:
        graph = store.export_graph()
        json_str = graph.model_dump_json(indent=2)

        if output:
            Path(output).write_text(json_str)
            click.echo(f"Graph exported to {output}")
        else:
            click.echo(json_str)
    finally:
        store.close()


if __name__ == "__main__":
    main()

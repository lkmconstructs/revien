"""G7: declared origin outside the fixed vocabulary is rejected at pipeline
and daemon; MCP store cannot claim the vault channel.

CHECK: python scripts/gates/check_origin_vocab.py
EXPECT: origin vocabulary verification passed
"""
import inspect
import os
import sys
import tempfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.store import GraphStore  # noqa: E402
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline  # noqa: E402


def fail(msg):
    print(f"check_origin_vocab: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path=db_path)
    pipeline = IngestionPipeline(store)

    # ── IngestionPipeline.ingest with a bogus origin_runtime -> ValueError ──
    raised = False
    try:
        pipeline.ingest(IngestionInput(
            source_id="claude-code:fernweh-core:sess-9",
            content="Theo said the deadline moved.",
            origin_runtime="TOTALLY-MADE-UP",
        ))
    except ValueError as e:
        raised = True
        if "TOTALLY-MADE-UP" not in str(e):
            fail(f"ValueError raised but does not name the offending value: {e}")
    if not raised:
        fail("pipeline.ingest() with an invalid origin_runtime did not raise ValueError")

    # ── daemon POST /v1/ingest with a bogus origin_runtime -> HTTP 400 ──
    from fastapi.testclient import TestClient
    from revien.daemon.server import create_app

    fd2, db_path2 = tempfile.mkstemp(suffix=".db")
    os.close(fd2)
    app = create_app(db_path=db_path2)
    client = TestClient(app)
    resp = client.post("/v1/ingest", json={
        "source_id": "custom-source",
        "content": "Sam decided to use Kafka for the event bus.",
        "origin_runtime": "TOTALLY-MADE-UP",
    })
    if resp.status_code != 400:
        fail(f"daemon /v1/ingest with an invalid origin_runtime returned "
             f"{resp.status_code}, expected 400")
    if "TOTALLY-MADE-UP" not in resp.json().get("detail", ""):
        fail(f"400 response detail does not name the offending value: {resp.json()}")

    # ── MCP revien_store tool has NO origin_source parameter ──
    from revien.mcp_server import _build_server

    src = inspect.getsource(_build_server)
    if "def revien_store(" not in src:
        fail("_build_server source does not define revien_store -- introspection target changed")
    tool_src = src.split("def revien_store(")[1].split(") -> Dict[str, Any]:")[0]
    if "origin_source" in tool_src:
        fail("revien_store's signature includes an 'origin_source' parameter "
             "-- callers must not be able to claim the vault channel")

    # ── a node stored via that MCP tool's call path has origin_source == "api" ──
    result = pipeline.ingest(IngestionInput(
        source_id="mcp",
        content="A durable fact from an LLM tool call about Fernweh-Core.",
        content_type="note",
        defer_embed=False,
        origin_runtime="obsidian",  # even claiming a vault-associated runtime
        origin_source="api",       # hard-set by the tool, never caller-controlled
        project_key=None,
        session_key=None,
    ))
    mcp_node = store.get_node(result.context_node_id)
    if mcp_node.origin_source != "api":
        fail(f"MCP-style store call produced origin_source={mcp_node.origin_source!r}, expected 'api'")
    if mcp_node.origin_source == "vault":
        fail("MCP-style store call was able to claim the vault channel")

    # ── declared origin gets origin_declared=True; derived origin does not ──
    declared = pipeline.ingest(IngestionInput(
        source_id="claude-code:fernweh-core:sess-declared",
        content="We decided to use PostgreSQL for the Fernweh-Core API.",
        origin_runtime="claude-code",
        origin_source="live",
    ))
    declared_nodes = store.list_nodes(source_id="claude-code:fernweh-core:sess-declared", limit=999)
    if not declared_nodes:
        fail("declared-origin ingest produced no nodes")
    for n in declared_nodes:
        if n.metadata.get("origin_declared") is not True:
            fail(
                f"node {n.node_id} from a DECLARED origin_runtime lacks "
                f"metadata.origin_declared=True (got {n.metadata.get('origin_declared')!r})"
            )

    derived = pipeline.ingest(IngestionInput(
        source_id="claude-code:fernweh-core:sess-derived",
        content="We decided to use SQLite for the cache layer.",
    ))
    derived_nodes = store.list_nodes(source_id="claude-code:fernweh-core:sess-derived", limit=999)
    if not derived_nodes:
        fail("derived-origin ingest produced no nodes")
    for n in derived_nodes:
        if "origin_declared" in (n.metadata or {}):
            fail(
                f"node {n.node_id} from a DERIVED origin_runtime (caller omitted it) "
                f"unexpectedly carries an origin_declared key: {n.metadata}"
            )

    store.close()
    try:
        os.unlink(db_path)
        os.unlink(db_path2)
    except PermissionError:
        pass

    print("origin vocabulary verification passed")


if __name__ == "__main__":
    main()

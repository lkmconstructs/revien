"""Every capture path is gated, and every writer records ownership
(docs/FORGET_SPEC.md §3, §6)."""
import ast
import inspect
from pathlib import Path

import pytest

from revien.forget import gate
from revien.ingestion.pipeline import IngestionPipeline
from tests import forget_seed as fs

REPO = Path(__file__).resolve().parents[1]

# add_node calls that deliberately carry no ownership row, and why.
UNOWNED_ADD_NODE = {
    "revien/forget/engine.py": "a content-free tombstone marker for lineage; it holds nothing to forget",
    "revien/graph/clustering.py": "networkx Graph.add_node, not a store write",
    "revien/skills/proposals.py": "ownership is recorded right after, in the same transaction, "
                                  "by _record_proposal_provenance",
}
# modules that write nodes with an owner and must also ask the gate first
OWNED_WRITERS_THAT_GATE_THEMSELVES = [
    "revien/adapters/openai_adapter.py", "revien/distill.py",
    "revien/skills/ingest.py", "revien/ingestion/pipeline.py",
]
# writers gated one level up: dedup is only ever called from the pipeline
GATED_BY_CALLER = {"revien/ingestion/dedup.py": "IngestionPipeline.ingest checks the gate before it extracts"}


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            yield node


def test_every_node_writer_records_who_wrote_it():
    unknown = []
    for path in sorted((REPO / "revien").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel in ("revien/graph/store.py",):
            continue
        for call in _calls(path):
            if call.func.attr == "add_node" and not any(k.arg == "owner" for k in call.keywords):
                if rel not in UNOWNED_ADD_NODE:
                    unknown.append(f"{rel}:{call.lineno}")
    assert not unknown, ("a writer creates a node without recording an owner "
                         f"(§3): {unknown}")


def test_every_owned_writer_asks_the_gate():
    for rel in OWNED_WRITERS_THAT_GATE_THEMSELVES:
        src = (REPO / rel).read_text(encoding="utf-8")
        assert ("forget_gate.check(" in src or "_gate.check(" in src), rel
    # the store's graph import asks it too
    assert "forget_gate.check" in inspect.getsource(
        __import__("revien.graph.store", fromlist=["GraphStore"]).GraphStore.import_graph)


def test_the_pipeline_asks_the_gate_before_it_extracts_anything():
    src = inspect.getsource(IngestionPipeline._ingest)
    assert src.index("_gate_check") < src.index("self.extractor.extract")
    assert src.index("_gate_check") < src.index("find_context_node_by_ingest_key")


@pytest.mark.parametrize("path", fs.ALL_PATHS)
def test_each_capture_path_reaches_the_gate(path, tmp_path, monkeypatch):
    """The inventory, run for real: seed through the path with the gate
    wrapped, and it must have been asked."""
    if path == "langchain":
        from revien.adapters.langchain_adapter import LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("langchain-core is not installed")
    calls = []
    real = gate.check

    def spy(store, unit, content):
        calls.append(unit.unit_id)
        return real(store, unit, content)

    monkeypatch.setattr(gate, "check", spy)
    fs.seed(path, tmp_path, monkeypatch)
    assert calls, f"{path}: the capture never asked the suppression gate"

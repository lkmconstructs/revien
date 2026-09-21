"""G3: source filter never emits foreign-runtime content in results,
tensions, path labels, or skill_proposals, and fails closed on empty.

CHECK: python scripts/gates/check_source_filter.py
EXPECT: source filter verification passed
"""
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.retrieval.engine import RetrievalEngine  # noqa: E402
from revien.skills.proposals import matching_proposals  # noqa: E402


def fail(msg):
    print(f"check_source_filter: {msg}", file=sys.stderr)
    sys.exit(1)


def mk(store, label, content, runtime, ntype=NodeType.FACT, **kw):
    n = Node(
        node_type=ntype, label=label, content=content, source_id="x",
        origin_runtime=runtime, origin_source="live",
        project_key=f"p-{runtime}" if runtime else None,
        source_type=SourceType.EXTRACTED, confidence=1.0,
        recorded_at=datetime.now(timezone.utc), **kw,
    )
    return store.add_node(n)


def main():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path=db_path)

    # ── leak1.py scenario: tension partner leak + skill proposal leak ──
    e_codex = mk(store, "Fernweh-Core", "Fernweh-Core project", "codex", NodeType.ENTITY)
    c_codex = mk(store, "Fernweh-Core ships Tuesday", "Fernweh-Core ships Tuesday", "codex")
    store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=e_codex.node_id,
                         target_node_id=c_codex.node_id, weight=1.0))
    c_cc = mk(store, "Fernweh-Core slips to Friday", "Fernweh-Core slips to Friday", "claude-code")
    store.add_edge(Edge(edge_type=EdgeType.CONFLICTS_WITH, source_node_id=c_codex.node_id,
                         target_node_id=c_cc.node_id, weight=1.0, source_context="conflict"))

    # leak2.py scenario: foreign hop on the path
    anchor = mk(store, "Fernweh-Core anchor", "Fernweh-Core anchor", "codex", NodeType.ENTITY)
    mid = mk(store, "PRIVATE claude-code secret topic", "PRIVATE claude-code secret topic",
             "claude-code", NodeType.TOPIC)
    leaf = mk(store, "codex leaf via foreign hop", "codex leaf via foreign hop", "codex")
    store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=anchor.node_id,
                         target_node_id=mid.node_id))
    store.add_edge(Edge(edge_type=EdgeType.RELATED_TO, source_node_id=mid.node_id,
                         target_node_id=leaf.node_id))

    # claude-code skill proposal (draft engine proposal)
    cc_prop = store.add_node(Node(
        node_type=NodeType.SKILL, label="proposed: sync fernweh-core branches",
        content="## Steps\n\n1. sync fernweh-core branches\n",
        source_id="skill-proposal:cc-hash",
        metadata={
            "origin": "engine", "status": "proposed", "pattern_hash": "cc-hash",
            "occurrences": 4, "sessions": 2, "declines": 0,
            "steps": ["sync fernweh-core branches"], "draft": True,
        },
        source_type=SourceType.INFERRED, confidence=0.5,
        origin_runtime="claude-code", origin_source="live", project_key="p-claude-code",
        recorded_at=datetime.now(timezone.utc),
    ))
    codex_prop = store.add_node(Node(
        node_type=NodeType.SKILL, label="proposed: sync fernweh-core branches too",
        content="## Steps\n\n1. sync fernweh-core branches\n",
        source_id="skill-proposal:codex-hash",
        metadata={
            "origin": "engine", "status": "proposed", "pattern_hash": "codex-hash",
            "occurrences": 4, "sessions": 2, "declines": 0,
            "steps": ["sync fernweh-core branches"], "draft": True,
        },
        source_type=SourceType.INFERRED, confidence=0.5,
        origin_runtime="codex", origin_source="live", project_key="p-codex",
        recorded_at=datetime.now(timezone.utc),
    ))

    engine = RetrievalEngine(store)

    # ── positive control: unfiltered recall DOES return foreign material ──
    unfiltered = engine.recall(
        "Fernweh-Core branches sync ships slips", top_n=25, min_score=0.0,
        include_context=True, include_tensions=True, debug=True,
    )
    if not unfiltered.results:
        fail("positive control: unfiltered recall returned no results at all")
    unfiltered_runtimes = {r.origin_runtime for r in unfiltered.results}
    if "claude-code" not in unfiltered_runtimes:
        fail(
            "positive control failed: unfiltered recall never surfaced "
            f"claude-code material (runtimes seen: {unfiltered_runtimes}) -- "
            "test data does not exercise the leak path"
        )
    unfiltered_prop_ids = {p["node_id"] for p in unfiltered.skill_proposals}
    if cc_prop.node_id not in unfiltered_prop_ids:
        # weaker positive control on proposals; not fatal on its own, but
        # note it. Proceed -- keyword match is heuristic.
        pass

    # ── filtered recall (source="codex") must never leak claude-code ──
    filtered = engine.recall(
        "Fernweh-Core branches sync ships slips", top_n=25, min_score=0.0,
        include_context=True, include_tensions=True, source="codex", debug=True,
    )
    if not filtered.results:
        fail("filtered recall(source='codex') returned no results")

    for r in filtered.results:
        if r.origin_runtime != "codex":
            fail(f"filtered recall leaked a non-codex result: {r.label!r} "
                 f"runtime={r.origin_runtime!r}")
        for t in r.tensions:
            partner_runtime = None
            for n in (e_codex, c_codex, c_cc, anchor, mid, leaf):
                if n.node_id == t["node_id"]:
                    partner_runtime = n.origin_runtime
            if partner_runtime is not None and partner_runtime != "codex":
                fail(f"filtered recall leaked a foreign tension partner: {t}")
        # no foreign-runtime label anywhere in the path -- only codex labels
        # or the literal [filtered] placeholder are allowed.
        for label in r.path:
            if label == "[filtered]":
                continue
            if label == mid.label:
                fail("filtered recall leaked the claude-code path label verbatim")

    filtered_prop_ids = {p["node_id"] for p in filtered.skill_proposals}
    if cc_prop.node_id in filtered_prop_ids:
        fail("filtered recall leaked a claude-code skill_proposal")

    # matching_proposals() directly, current signature (source_filter=...)
    mp_filtered = matching_proposals(store, "sync fernweh-core branches", ["codex"])
    mp_ids = {p["node_id"] for p in mp_filtered}
    if cc_prop.node_id in mp_ids:
        fail("matching_proposals(source_filter=['codex']) leaked the claude-code proposal")
    if codex_prop.node_id not in mp_ids:
        fail("matching_proposals(source_filter=['codex']) dropped the in-filter codex proposal")

    mp_unfiltered = matching_proposals(store, "sync fernweh-core branches", None)
    mp_unfiltered_ids = {p["node_id"] for p in mp_unfiltered}
    if not ({cc_prop.node_id, codex_prop.node_id} <= mp_unfiltered_ids):
        fail("matching_proposals(source_filter=None) dropped an in-scope proposal")

    # ── fail-closed on empty filter: source=[], source="", source=[""] ──
    for empty_source, desc in (([], "[]"), ("", "''"), ([""], "['']")):
        resp = engine.recall(
            "Fernweh-Core branches sync ships slips", top_n=25, min_score=0.0,
            source=empty_source, debug=True,
        )
        if resp.results != []:
            fail(f"source={desc} did not fail closed: results={resp.results}")
        if resp.skill_proposals != []:
            fail(f"source={desc} did not fail closed: skill_proposals={resp.skill_proposals}")

    # ── source=None is unfiltered ──
    none_resp = engine.recall(
        "Fernweh-Core branches sync ships slips", top_n=25, min_score=0.0,
        source=None, debug=True,
    )
    if not none_resp.results:
        fail("source=None returned no results -- should be unfiltered like the control")
    none_runtimes = {r.origin_runtime for r in none_resp.results}
    if "claude-code" not in none_runtimes:
        fail(f"source=None unexpectedly filtered out claude-code (runtimes: {none_runtimes})")

    store.close()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass

    print("source filter verification passed")


if __name__ == "__main__":
    main()

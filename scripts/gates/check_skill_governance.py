"""G5: skill proposals are engine-origin and proposed-only; single-id accept
keeps engine origin; accepted body is frozen; third decline invalidates;
every step audited.

CHECK: python scripts/gates/check_skill_governance.py
EXPECT: skill governance verification passed
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.schema import EdgeType, Node, NodeType, SourceType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.skills.proposals import accept_proposal, decline_proposal, propose_skills  # noqa: E402

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fail(msg):
    print(f"check_skill_governance: {msg}", file=sys.stderr)
    sys.exit(1)


def seed_actions(store, steps, sessions, project_key="fernweh-core", start_minute=0):
    n = start_minute
    node_ids = []
    for sess in sessions:
        for step in steps:
            node = store.add_node(Node(
                node_type=NodeType.ACTION, label=step, content=step,
                source_id=f"claude-code:{project_key}:{sess}", source_type=SourceType.EXTRACTED,
                confidence=1.0, origin_runtime="claude-code", origin_source="live",
                project_key=project_key, session_key=sess,
                recorded_at=BASE + timedelta(minutes=n),
            ))
            node_ids.append(node.node_id)
            n += 1
    return node_ids, n


def main():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path=db_path)

    # Seed ACTION nodes across 3 sessions: Mara/Theo/Sam doing tasks on
    # Fernweh-Core.
    steps = ["I'll ping Mara about the Fernweh-Core deploy",
             "I'll sync the Fernweh-Core branches"]
    action_ids, next_minute = seed_actions(store, steps, ("sess-mara", "sess-theo", "sess-sam"))

    # ── propose_skills: exactly one proposal, engine-origin, proposed ──
    summary = propose_skills(store)
    if summary["created"] != 1:
        fail(f"propose_skills created {summary['created']} proposals, expected 1")
    proposals = store.list_nodes(node_type=NodeType.SKILL, limit=10)
    if len(proposals) != 1:
        fail(f"expected exactly 1 SKILL node after propose_skills, got {len(proposals)}")
    proposal = proposals[0]
    if proposal.metadata.get("status") != "proposed":
        fail(f"proposal status == {proposal.metadata.get('status')!r}, expected 'proposed'")
    if proposal.metadata.get("origin") != "engine":
        fail(f"proposal origin == {proposal.metadata.get('origin')!r}, expected 'engine'")
    if "curated" in proposal.metadata:
        fail("engine proposal metadata carries a 'curated' key -- reserved for user skills")

    derived_edges = [
        e for e in store.get_edges_for_node(proposal.node_id)
        if e.edge_type == EdgeType.DERIVED_FROM and e.source_node_id == proposal.node_id
    ]
    if len(derived_edges) != len(action_ids):
        fail(
            f"DERIVED_FROM edge count == {len(derived_edges)}, "
            f"expected {len(action_ids)} (one per source ACTION node)"
        )

    # ── accept_proposal: active, origin stays engine ──
    accepted = accept_proposal(store, proposal.node_id, actor="mara")
    if accepted.metadata.get("status") != "active":
        fail(f"accepted status == {accepted.metadata.get('status')!r}, expected 'active'")
    if accepted.metadata.get("origin") != "engine":
        fail(f"accepted origin == {accepted.metadata.get('origin')!r}, expected 'engine'")

    before_label, before_content = accepted.label, accepted.content

    # ── re-run propose_skills with MORE evidence -> content frozen, ──
    # counters increased.
    more_ids, _ = seed_actions(store, steps, ("sess-4", "sess-5"), start_minute=next_minute)
    summary2 = propose_skills(store)
    if summary2["updated"] != 1:
        fail(f"re-propose after accept: updated == {summary2['updated']}, expected 1")

    after = store.get_node(proposal.node_id)
    if after.label != before_label:
        fail("accepted proposal's label changed on re-propose -- must be byte-frozen")
    if after.content != before_content:
        fail("accepted proposal's content changed on re-propose -- must be byte-frozen")
    if after.metadata.get("occurrences", 0) <= proposal.metadata.get("occurrences", 0):
        fail("re-propose with more evidence did not increase the occurrences counter")

    # ── second proposal, decline it 3 times ──
    steps2 = ["I'll run the Fernweh-Core benchmark", "I'll notify Sam of the results"]
    seed_actions(store, steps2, ("sess-a", "sess-b", "sess-c"), start_minute=next_minute + 100)
    summary3 = propose_skills(store)
    if summary3["created"] != 1:
        fail(f"second propose_skills created {summary3['created']} proposals, expected 1")
    second = [n for n in summary3["proposals"] if n.node_id != proposal.node_id]
    if not second:
        fail("could not find the second (new) proposal")
    second_proposal = second[0]

    last = None
    for i in range(3):
        last = decline_proposal(store, second_proposal.node_id, actor="mara")
    if last.metadata.get("declines") != 3:
        fail(f"after 3 declines, declines == {last.metadata.get('declines')}, expected 3")
    if last.invalidated_at is None:
        fail("after the 3rd decline, invalidated_at is still None")

    # accept() on an invalidated proposal must raise ValueError.
    raised = False
    try:
        accept_proposal(store, second_proposal.node_id)
    except ValueError:
        raised = True
    if not raised:
        fail("accept_proposal() on an invalidated proposal did not raise ValueError")

    # ── audit_log completeness ──
    history1 = store.get_node_audit(proposal.node_id)
    ops1 = [h["op"] for h in history1]
    if "skill_propose" not in ops1:
        fail("audit_log missing 'skill_propose' for the first proposal")
    if "skill_accept" not in ops1:
        fail("audit_log missing 'skill_accept' for the first proposal")

    history2 = store.get_node_audit(second_proposal.node_id)
    ops2 = [h["op"] for h in history2]
    if ops2.count("skill_decline") != 3:
        fail(f"audit_log has {ops2.count('skill_decline')} 'skill_decline' entries, expected 3")
    invalidate_entries = [h for h in history2 if h["op"] == "invalidate"]
    if not invalidate_entries:
        fail("audit_log missing an 'invalidate' entry for the 3x-declined proposal")
    inv = invalidate_entries[0]
    if inv.get("before") is None or inv.get("after") is None:
        fail(f"invalidate audit entry has null before/after: {inv}")

    # ── no bulk-accept escape hatch ──
    repo_root = Path(__file__).resolve().parents[2]
    cli_src = (repo_root / "revien" / "cli.py").read_text(encoding="utf-8")
    server_src = (repo_root / "revien" / "daemon" / "server.py").read_text(encoding="utf-8")
    for name, src in (("cli.py", cli_src), ("server.py", server_src)):
        idx = 0
        while True:
            idx = src.find("skills", idx)
            if idx == -1:
                break
            window = src[max(0, idx - 200):idx + 200]
            if "--all" in window and "skill" in window.lower():
                fail(f"{name}: found '--all' near 'skills' -- possible bulk-accept escape hatch")
            idx += 6

    import inspect
    from revien.skills import proposals as proposals_module
    for fn_name in ("accept_proposal", "decline_proposal"):
        fn = getattr(proposals_module, fn_name)
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            ann = str(param.annotation)
            if "List" in ann or "list" in ann.lower():
                fail(
                    f"{fn_name} has a list-typed parameter {pname!r} "
                    f"({ann}) -- possible bulk accept/decline escape hatch"
                )

    store.close()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass

    print("skill governance verification passed")


if __name__ == "__main__":
    main()

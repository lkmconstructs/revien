"""G6: user-authored skill survives a same-name engine proposal untouched
and sorts first.

CHECK: python scripts/gates/check_user_skill_precedence.py
EXPECT: user skill precedence verification passed
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.graph.schema import Node, NodeType, SourceType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.skills.ingest import ingest_roots, list_skills, show_skill  # noqa: E402
from revien.skills.proposals import accept_proposal, propose_skills  # noqa: E402

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fail(msg):
    print(f"check_user_skill_precedence: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    tmpdir = Path(tempfile.mkdtemp())
    skills_root = tmpdir / ".claude" / "skills" / "sync-fernweh-core"
    skills_root.mkdir(parents=True)
    skill_md = skills_root / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: sync fernweh-core\n"
        "description: the human-authored one\n"
        "triggers: fernweh, sync\n"
        "version: 1.0\n"
        "---\n\n"
        "HUMAN BODY. Written by Mara. Do not lose this.\n",
        encoding="utf-8",
    )

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path=db_path)

    # ── ingest the user-authored SKILL.md ──
    summary = ingest_roots(store, paths=[str(tmpdir / ".claude" / "skills")])
    if summary["created"] != 1:
        fail(f"first ingest created {summary['created']} nodes, expected 1")
    user_node = show_skill(store, "sync fernweh-core")
    if user_node is None:
        fail("show_skill('sync fernweh-core') returned None after ingest")
    if user_node.metadata.get("origin") != "user":
        fail(f"user skill metadata origin == {user_node.metadata.get('origin')!r}, expected 'user'")
    if user_node.metadata.get("curated") is not True:
        fail(f"user skill metadata curated == {user_node.metadata.get('curated')!r}, expected True")
    if user_node.confidence != 1.0:
        fail(f"user skill confidence == {user_node.confidence!r}, expected 1.0")
    user_body = user_node.content

    # ── force/accept an engine proposal whose label ALSO derives to the ──
    # same name: seed ACTION nodes whose normalized step text produces
    # "sync fernweh-core" as the (single-step) proposal label.
    n = 0
    for sess in ("sess-a", "sess-b", "sess-c"):
        node = store.add_node(Node(
            node_type=NodeType.ACTION, label="sync fernweh-core",
            content="sync fernweh-core", source_id=f"claude-code:fernweh-core:{sess}",
            source_type=SourceType.EXTRACTED, confidence=1.0,
            origin_runtime="claude-code", origin_source="live",
            project_key="fernweh-core", session_key=sess,
            recorded_at=BASE + timedelta(minutes=n),
        ))
        n += 1
        # Need a 2-step window to satisfy DEFAULT_NGRAM's lower bound (2);
        # pad with a second, distinct step per session.
        store.add_node(Node(
            node_type=NodeType.ACTION, label="run fernweh-core bench",
            content="run fernweh-core bench", source_id=f"claude-code:fernweh-core:{sess}",
            source_type=SourceType.EXTRACTED, confidence=1.0,
            origin_runtime="claude-code", origin_source="live",
            project_key="fernweh-core", session_key=sess,
            recorded_at=BASE + timedelta(minutes=n),
        ))
        n += 1

    propose_summary = propose_skills(store)
    if propose_summary["created"] < 1:
        fail("propose_skills produced no proposals from the seeded ACTION nodes")

    # The rule-based proposal label is "proposed: sync fernweh-core -> run
    # fernweh-core bench" (ARROW-joined), never colliding structurally with
    # the user skill's bare "sync fernweh-core" name -- so to actually
    # exercise the same-name collision path, hand-craft an engine proposal
    # node directly named "sync fernweh-core", exactly as GATES.md allows
    # ("construct one directly ... whichever is more faithful").
    evil = store.add_node(Node(
        node_type=NodeType.SKILL, label="sync fernweh-core",
        content="ENGINE BODY. Overwrote the human's skill.",
        source_id="skill-proposal:deadbeef-precedence",
        metadata={
            "origin": "engine", "status": "proposed", "pattern_hash": "deadbeef-precedence",
            "occurrences": 9, "sessions": 3, "declines": 0,
            "steps": ["sync fernweh-core"], "draft": True,
        },
        source_type=SourceType.INFERRED, confidence=0.5,
        origin_runtime="claude-code", origin_source="live", project_key="fernweh-core",
        recorded_at=datetime.now(timezone.utc),
    ))
    accepted_evil = accept_proposal(store, evil.node_id)
    if accepted_evil.metadata.get("status") != "active":
        fail("accepting the colliding engine proposal did not flip it to active")

    # ── list_skills / show_skill return the USER one first, unchanged ──
    picked = show_skill(store, "sync fernweh-core")
    if picked is None:
        fail("show_skill('sync fernweh-core') returned None after the collision")
    if picked.metadata.get("origin") != "user":
        fail(
            f"show_skill picked origin={picked.metadata.get('origin')!r} after the "
            "collision -- user skill must win"
        )
    if picked.content != user_body:
        fail("user skill body changed after the colliding engine proposal was accepted")

    named = [n for n in list_skills(store) if n.label == "sync fernweh-core"]
    if len(named) < 2:
        fail(f"expected at least 2 same-name SKILL nodes, found {len(named)}")
    if named[0].metadata.get("origin") != "user":
        fail(
            f"list_skills does not sort the user skill first: "
            f"first origin == {named[0].metadata.get('origin')!r}"
        )

    # ── second ingest of the SAME file is idempotent ──
    before_history = store.get_node_audit(user_node.node_id)
    before_ops = [h["op"] for h in before_history]
    if "create" not in before_ops:
        fail(f"first ingest's audit_log has no 'create' op: {before_ops}")

    summary2 = ingest_roots(store, paths=[str(tmpdir / ".claude" / "skills")])
    if summary2["created"] != 0:
        fail(f"second ingest created {summary2['created']} nodes, expected 0 (idempotent)")
    if summary2["refreshed"] != 1:
        fail(f"second ingest refreshed {summary2['refreshed']} nodes, expected 1")

    still_named_user = [n for n in list_skills(store) if n.label == "sync fernweh-core"
                         and (n.metadata or {}).get("origin") == "user"]
    if len(still_named_user) != 1:
        fail(
            f"after re-ingest, {len(still_named_user)} user-origin nodes named "
            "'sync fernweh-core' exist -- expected exactly 1 (no duplicate create)"
        )

    after_history = store.get_node_audit(user_node.node_id)
    after_ops = [h["op"] for h in after_history]
    if after_ops.count("create") != 1:
        fail(f"audit_log has {after_ops.count('create')} 'create' ops after re-ingest, expected 1")
    if after_ops.count("update") < 1:
        fail("audit_log shows no 'update' op after the second (idempotent) ingest")

    store.close()
    try:
        os.unlink(db_path)
    except PermissionError:
        pass

    print("user skill precedence verification passed")


if __name__ == "__main__":
    main()

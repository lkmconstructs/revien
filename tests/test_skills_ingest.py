"""
Skills ingest (thin WS3, leg D1).

Covers: the minimal frontmatter parser (inline/block triggers, folded
description, missing name fallback), idempotent ingest (re-run refreshes in
place rather than duplicating), scope/project/origin field placement,
best-effort RELATED_TO edges to matching entities, `skills show` returning
the body, and user-before-engine ordering. Fictional names only (Mara/Theo/
Sam, Fernweh-Core).
"""

import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from revien.cli import main
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.skills.frontmatter import parse_frontmatter
from revien.skills.ingest import (
    ingest_roots,
    ingest_skill_file,
    list_skills,
    show_skill,
    skill_ingest_key,
    sort_user_before_engine,
)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    os.unlink(path)


def _write_skill(dir_path: Path, name: str, frontmatter: str, body: str) -> Path:
    skill_dir = dir_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")
    return skill_md


FERNWEH_TRIGGER_SKILL = """\
name: fernweh-sync
description: >
  Sync Fernweh-Core state between Mara and Theo's branches before either one
  merges — catches drift early.
triggers: fernweh, sync branches, fernweh-core
version: "1.0"
"""

BLOCK_TRIGGER_SKILL = """\
name: sam-standup
description: Sam's daily standup checklist.
triggers:
  - standup
  - daily check-in
version: "0.2"
"""

NO_FRONTMATTER_SKILL = "Just a body, no frontmatter block at all.\n"

NAMELESS_SKILL = """\
description: Has no explicit name key.
"""


# ── frontmatter parser ──────────────────────────────────────────────────

class TestParseFrontmatter:
    def test_inline_triggers_and_folded_description(self):
        fm, body = parse_frontmatter(f"---\n{FERNWEH_TRIGGER_SKILL}---\nBody text.\n")
        assert fm["name"] == "fernweh-sync"
        assert "Fernweh-Core" in fm["description"]
        assert "catches drift early." in fm["description"]
        assert fm["triggers"] == ["fernweh", "sync branches", "fernweh-core"]
        assert fm["version"] == "1.0"
        assert body == "Body text.\n"

    def test_block_list_triggers(self):
        fm, body = parse_frontmatter(f"---\n{BLOCK_TRIGGER_SKILL}---\nChecklist body.\n")
        assert fm["name"] == "sam-standup"
        assert fm["triggers"] == ["standup", "daily check-in"]
        assert fm["version"] == "0.2"

    def test_no_frontmatter_block(self):
        fm, body = parse_frontmatter(NO_FRONTMATTER_SKILL)
        assert fm == {}
        assert body == NO_FRONTMATTER_SKILL

    def test_missing_name_key(self):
        fm, body = parse_frontmatter(f"---\n{NAMELESS_SKILL}---\nBody.\n")
        assert "name" not in fm
        assert fm["description"] == "Has no explicit name key."

    def test_unknown_keys_ignored(self):
        fm, body = parse_frontmatter(
            "---\nname: theo-notes\nauthor: Theo\ncolor: blue\n---\nBody.\n"
        )
        assert fm == {"name": "theo-notes"}


# ── ingest: node fields, scope/project/origin ──────────────────────────

class TestIngestFields:
    def test_project_scope_fields(self, store, tmp_path):
        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Steps here.")

        node, created, _edges = ingest_skill_file(
            store, skill_md, scope="project", project_key="mara-project", origin_runtime="claude-code"
        )

        assert created is True
        assert node.node_type == NodeType.SKILL
        assert node.label == "fernweh-sync"
        assert node.content == "Steps here."
        assert node.source_type == SourceType.EXTRACTED
        assert node.confidence == 1.0
        assert node.origin_runtime == "claude-code"
        assert node.origin_source == "vault"
        assert node.project_key == "mara-project"
        assert node.session_key is None
        assert node.metadata["description"].startswith("Sync Fernweh-Core")
        assert node.metadata["triggers"] == ["fernweh", "sync branches", "fernweh-core"]
        assert node.metadata["version"] == "1.0"
        assert node.metadata["origin"] == "user"
        assert node.metadata["status"] == "active"
        assert node.metadata["scope"] == "project"
        assert node.metadata["path"] == str(skill_md.resolve())

    def test_global_scope_has_no_project_key(self, store, tmp_path):
        skills_root = tmp_path / ".hermes" / "skills"
        skill_md = _write_skill(skills_root, "sam-standup", BLOCK_TRIGGER_SKILL, "Checklist.")

        node, created, _edges = ingest_skill_file(
            store, skill_md, scope="global", project_key=None, origin_runtime="hermes"
        )

        assert created is True
        assert node.metadata["scope"] == "global"
        assert node.project_key is None
        assert node.origin_runtime == "hermes"

    def test_unrecognized_root_has_no_origin_runtime(self, store, tmp_path):
        skills_root = tmp_path / "some-other-dir" / "skills"
        skill_md = _write_skill(skills_root, "theo-notes", "name: theo-notes\n", "Notes.")

        node, _created, _edges = ingest_skill_file(
            store, skill_md, scope="project", project_key="theo-project", origin_runtime=None
        )

        assert node.origin_runtime is None


# ── idempotency ──────────────────────────────────────────────────────────

class TestIdempotentIngest:
    def test_ingest_key_is_abs_path_prefixed(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("name: x\n", encoding="utf-8")
        assert skill_ingest_key(skill_md) == f"skill:{skill_md.resolve()}"

    def test_rerun_refreshes_not_duplicates(self, store, tmp_path):
        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Steps v1.")

        node1, created1, _ = ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")
        assert created1 is True

        # Edit the body and version, then re-ingest the same path.
        skill_md.write_text(
            "---\n" + FERNWEH_TRIGGER_SKILL.replace('"1.0"', '"1.1"') + "---\nSteps v2.\n",
            encoding="utf-8",
        )
        node2, created2, _ = ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")

        assert created2 is False
        assert node2.node_id == node1.node_id
        assert node2.content == "Steps v2."
        assert node2.metadata["version"] == "1.1"

        all_skills = store.list_nodes(node_type=NodeType.SKILL, limit=100)
        assert len(all_skills) == 1

    def test_rerun_records_update_audit(self, store, tmp_path):
        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Steps v1.")

        node1, _, _ = ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")
        node2, _, _ = ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")

        history = store.get_node_audit(node1.node_id)
        ops = [h["op"] for h in history]
        assert ops == ["create", "update"]


# ── entity/topic edges ────────────────────────────────────────────────────

class TestEntityEdges:
    def test_wikilink_and_trigger_match_existing_entities(self, store, tmp_path):
        fernweh_entity = store.add_node(Node(
            node_type=NodeType.ENTITY, label="Fernweh-Core", content="Fernweh-Core project",
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))
        standup_topic = store.add_node(Node(
            node_type=NodeType.TOPIC, label="standup", content="standup topic",
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))

        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(
            skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL,
            "See [[Fernweh-Core]] for context.",
        )

        node, _created, edges = ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")

        # wikilink match -> Fernweh-Core, trigger "standup" is NOT one of this
        # skill's triggers, so only the wikilink match is expected here.
        assert edges == 1
        skill_edges = store.get_edges_for_node(node.node_id)
        targets = {e.target_node_id for e in skill_edges if e.source_node_id == node.node_id}
        assert fernweh_entity.node_id in targets
        assert standup_topic.node_id not in targets

    def test_no_matching_entities_creates_no_edges(self, store, tmp_path):
        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(skills_root, "sam-standup", BLOCK_TRIGGER_SKILL, "No links here.")

        node, _created, edges = ingest_skill_file(store, skill_md, "project", "sam-project", "claude-code")

        assert edges == 0
        assert store.get_edges_for_node(node.node_id) == []


# ── list / show / ordering ────────────────────────────────────────────────

class TestListShowOrdering:
    def test_show_returns_body(self, store, tmp_path):
        skills_root = tmp_path / ".claude" / "skills"
        skill_md = _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "The body markdown.")
        ingest_skill_file(store, skill_md, "project", "mara-project", "claude-code")

        found = show_skill(store, "fernweh-sync")
        assert found is not None
        assert found.content == "The body markdown."

    def test_show_missing_skill_returns_none(self, store):
        assert show_skill(store, "does-not-exist") is None

    def test_user_before_engine_ordering(self, store):
        engine_node = store.add_node(Node(
            node_type=NodeType.SKILL, label="fernweh-sync", content="engine draft",
            metadata={"origin": "engine", "status": "proposed"},
            source_type=SourceType.INFERRED, confidence=0.5,
        ))
        user_node = store.add_node(Node(
            node_type=NodeType.SKILL, label="fernweh-sync", content="user version",
            metadata={"origin": "user", "status": "active"},
            source_type=SourceType.EXTRACTED, confidence=1.0,
        ))

        ordered = sort_user_before_engine([engine_node, user_node])
        assert ordered[0].node_id == user_node.node_id
        assert ordered[1].node_id == engine_node.node_id

        # show() picks the user-authored one even when the engine draft was
        # ingested first.
        found = show_skill(store, "fernweh-sync")
        assert found.node_id == user_node.node_id

    def test_list_filters_by_project_and_status(self, store, tmp_path):
        mara_root = tmp_path / "mara" / ".claude" / "skills"
        theo_root = tmp_path / "theo" / ".claude" / "skills"
        mara_md = _write_skill(mara_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Mara's copy.")
        theo_md = _write_skill(theo_root, "sam-standup", BLOCK_TRIGGER_SKILL, "Theo's copy.")

        ingest_skill_file(store, mara_md, "project", "mara-project", "claude-code")
        ingest_skill_file(store, theo_md, "project", "theo-project", "claude-code")

        mara_only = list_skills(store, project="mara-project")
        assert [n.label for n in mara_only] == ["fernweh-sync"]

        active_only = list_skills(store, status="active")
        assert {n.label for n in active_only} == {"fernweh-sync", "sam-standup"}

        none_match = list_skills(store, status="proposed")
        assert none_match == []


# ── ingest_roots: scanning + CLI ─────────────────────────────────────────

class TestIngestRoots:
    def test_ingest_roots_explicit_path_is_project_scoped(self, store, tmp_path, monkeypatch):
        skills_root = tmp_path / "skills"
        _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Body.")
        _write_skill(skills_root, "sam-standup", BLOCK_TRIGGER_SKILL, "Body 2.")

        cwd = tmp_path / "mara-project"
        cwd.mkdir()

        summary = ingest_roots(store, paths=[str(skills_root)], include_global=False, cwd=cwd)

        assert summary["scanned"] == 2
        assert summary["created"] == 2
        assert summary["refreshed"] == 0
        for node in summary["skills"]:
            assert node.project_key == "mara-project"
            assert node.metadata["scope"] == "project"

    def test_ingest_roots_missing_root_is_empty_not_error(self, store, tmp_path):
        summary = ingest_roots(store, paths=[str(tmp_path / "nope")], cwd=tmp_path)
        assert summary["scanned"] == 0
        assert summary["skills"] == []

    def test_ingest_roots_rerun_is_idempotent(self, store, tmp_path):
        skills_root = tmp_path / "skills"
        _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "Body.")

        first = ingest_roots(store, paths=[str(skills_root)], cwd=tmp_path)
        second = ingest_roots(store, paths=[str(skills_root)], cwd=tmp_path)

        assert first["created"] == 1
        assert second["created"] == 0
        assert second["refreshed"] == 1
        assert len(store.list_nodes(node_type=NodeType.SKILL, limit=100)) == 1


class TestSkillsCli:
    def test_cli_ingest_list_show(self, tmp_path, monkeypatch):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            project_dir = tmp_path / "mara-project"
            skills_root = project_dir / ".claude" / "skills"
            _write_skill(skills_root, "fernweh-sync", FERNWEH_TRIGGER_SKILL, "CLI body.")

            monkeypatch.chdir(project_dir.parent)
            project_dir.mkdir(exist_ok=True)
            monkeypatch.chdir(project_dir)

            runner = CliRunner()

            result = runner.invoke(main, ["skills", "ingest", "--db", db_path])
            assert result.exit_code == 0, result.output
            assert "1 created" in result.output

            result = runner.invoke(main, ["skills", "list", "--db", db_path])
            assert result.exit_code == 0, result.output
            assert "fernweh-sync" in result.output

            result = runner.invoke(main, ["skills", "list", "--db", db_path, "--format", "json"])
            assert result.exit_code == 0, result.output
            assert '"name": "fernweh-sync"' in result.output

            result = runner.invoke(main, ["skills", "show", "fernweh-sync", "--db", db_path])
            assert result.exit_code == 0, result.output
            assert "CLI body." in result.output

            result = runner.invoke(main, ["skills", "show", "nope-not-here", "--db", db_path])
            assert result.exit_code != 0
        finally:
            os.unlink(db_path)

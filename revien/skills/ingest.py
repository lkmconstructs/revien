"""
Revien Skills — ingest (thin WS3, leg D1).

Scans skill folders (each a directory containing a ``SKILL.md``) and turns
each one into a SKILL node: label = the skill's name, content = the body
verbatim (frontmatter stripped), metadata carries description/triggers/
version/origin/status/scope/path.

Idempotency is hand-rolled here rather than reused from the ingestion
pipeline's R3 refresh: the pipeline never accepts a "skill" content_type
without editing pipeline.py (out of scope for this leg — another agent owns
that file in parallel), so this module talks to GraphStore directly and
implements its own idempotency key, mirroring
GraphStore.find_context_node_by_ingest_key (store.py:1074) but scoped to
SKILL nodes: look up by ``ingest_key`` in metadata via
``list_nodes(node_type=SKILL)`` instead of a CONTEXT-only, store-owned
lookup. First-ingest records ``record_audit("create")`` (via
``store.add_node``); a re-ingest of an unchanged-path skill refreshes
label/content/metadata in place, which records ``record_audit("update")``
(via ``store.update_node``) — both audit calls are the store's own default
behavior, not something this module does by hand.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from revien.graph.schema import EdgeType, Edge, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.skills.frontmatter import parse_frontmatter

# Global skill roots and the runtime they belong to — the only three homes
# leg D1 knows about. Order matters only for de-duplication below.
_GLOBAL_ROOTS = (
    (Path.home() / ".claude" / "skills", "claude-code"),
    (Path.home() / ".codex" / "skills", "codex"),
    (Path.home() / ".hermes" / "skills", "hermes"),
)

# Project-scoped default roots, relative to the cwd.
_DEFAULT_PROJECT_ROOTS = (
    (Path(".claude") / "skills", "claude-code"),
    (Path(".codex") / "skills", "codex"),
)

# [[wikilink]] / [[wikilink|alias]] / [[wikilink#heading]] — same shape as
# adapters/obsidian.py's _WIKILINK_RE, redefined locally rather than
# importing a private name across modules.
_WIKILINK_RE = re.compile(r"\[\[([^\]\[#|]+)(?:#[^\]\[|]*)?(?:\|[^\]\[]*)?\]\]")


def _origin_runtime_for_root(root: Path) -> Optional[str]:
    """.claude/skills -> claude-code, .codex/skills -> codex,
    .hermes/skills -> hermes, anything else -> None (honest unknown)."""
    parent_name = root.parent.name.lower()
    return {
        ".claude": "claude-code",
        ".codex": "codex",
        ".hermes": "hermes",
    }.get(parent_name)


def resolve_roots(
    paths: Optional[List[str]] = None,
    include_global: bool = False,
    cwd: Optional[Path] = None,
) -> List[Tuple[Path, str, Optional[str]]]:
    """Turn CLI input into (root, scope, project_key) tuples.

    Explicit --path values are always project-scoped (project_key = cwd
    basename) — a caller pointing --path at a tmp_path fixture in tests, or
    at some other folder on disk, is naming a project's own skills folder,
    not a global one. --global additionally scans the three fixed global
    homes (~/.claude/skills, ~/.codex/skills, ~/.hermes/skills), on top of
    whatever --path gave (or the defaults, if --path was omitted).
    """
    cwd = cwd or Path.cwd()
    project_key = cwd.name
    roots: List[Tuple[Path, str, Optional[str]]] = []
    if paths:
        roots.extend((Path(p), "project", project_key) for p in paths)
    else:
        roots.extend(
            (cwd / rel, "project", project_key) for rel, _runtime in _DEFAULT_PROJECT_ROOTS
        )
    if include_global:
        roots.extend((root, "global", None) for root, _runtime in _GLOBAL_ROOTS)
    return roots


def discover_skill_files(root: Path) -> List[Path]:
    """Every SKILL.md under `root`, any depth (plugin-nested skills included).
    Missing root -> empty list, never an error."""
    if not root.exists():
        return []
    return sorted(root.rglob("SKILL.md"))


def skill_ingest_key(skill_md_path: Path) -> str:
    return f"skill:{skill_md_path.resolve()}"


def _find_skill_node_by_ingest_key(store: GraphStore, ingest_key: str) -> Optional[Node]:
    """SKILL-typed equivalent of GraphStore.find_context_node_by_ingest_key
    (store.py:1074), which is hard-coded to CONTEXT nodes. Paginates
    list_nodes(node_type=SKILL) rather than editing store.py to add a
    generic-node-type version."""
    offset = 0
    page = 500
    while True:
        batch = store.list_nodes(node_type=NodeType.SKILL, limit=page, offset=offset)
        if not batch:
            return None
        for node in batch:
            if (node.metadata or {}).get("ingest_key") == ingest_key:
                return node
        if len(batch) < page:
            return None
        offset += page


def _all_nodes(store: GraphStore, node_type: NodeType) -> List[Node]:
    out: List[Node] = []
    offset = 0
    page = 500
    while True:
        batch = store.list_nodes(node_type=node_type, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            return out
        offset += page


def build_skill_node(
    skill_md_path: Path,
    scope: str,
    project_key: Optional[str],
    origin_runtime: Optional[str],
    status: str = "active",
) -> Node:
    """Parse one SKILL.md and build the (not-yet-persisted) Node for it."""
    text = skill_md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    name = fm.get("name") or skill_md_path.parent.name
    metadata = {
        "description": fm.get("description", ""),
        "triggers": fm.get("triggers", []),
        "version": fm.get("version", ""),
        "origin": "user",
        # Kept so any future path that consults `curated` treats user
        # skills as ground truth. NOT currently load-bearing: SKILL nodes
        # never enter the ClaimGovernor's supersession candidates
        # (supersession_ingest.py's _existing_claims is CONTEXT-only), so
        # today no gate reads this flag before touching a skill. User
        # skills are safe today because SKILL nodes bypass the pipeline
        # entirely and dedup is same-type only.
        "curated": True,
        "status": status,
        "scope": scope,
        "path": str(skill_md_path.resolve()),
        "ingest_key": skill_ingest_key(skill_md_path),
    }
    return Node(
        node_type=NodeType.SKILL,
        label=name[:200],
        content=body.strip(),
        source_id=f"skill:{skill_md_path.resolve()}",
        metadata=metadata,
        source_type=SourceType.EXTRACTED,
        confidence=1.0,
        origin_runtime=origin_runtime,
        origin_source="vault",
        project_key=project_key,
        session_key=None,
        recorded_at=datetime.now(timezone.utc),
    )


def _link_matching_entities(store: GraphStore, skill_node: Node) -> int:
    """Best-effort SKILL -> ENTITY/TOPIC RELATED_TO edges for [[wikilinks]]
    in the body and exact (case-insensitive) trigger-word matches against
    existing entity/topic labels. Never creates a new ENTITY/TOPIC — a miss
    is just a miss."""
    wikilink_targets = {m.group(1).strip().lower() for m in _WIKILINK_RE.finditer(skill_node.content)}
    triggers = {t.strip().lower() for t in (skill_node.metadata or {}).get("triggers", []) if t.strip()}
    wanted = wikilink_targets | triggers
    if not wanted:
        return 0

    candidates = _all_nodes(store, NodeType.ENTITY) + _all_nodes(store, NodeType.TOPIC)
    by_label = {}
    for node in candidates:
        by_label.setdefault(node.label.strip().lower(), node)

    created = 0
    for target in wanted:
        match = by_label.get(target)
        if match is None or match.node_id == skill_node.node_id:
            continue
        store.add_edge(Edge(
            edge_type=EdgeType.RELATED_TO,
            source_node_id=skill_node.node_id,
            target_node_id=match.node_id,
            weight=0.8,
        ))
        created += 1
    return created


def ingest_skill_file(store: GraphStore, skill_md_path: Path, scope: str, project_key: Optional[str], origin_runtime: Optional[str]) -> Tuple[Node, bool, int]:
    """Ingest one SKILL.md. Returns (node, created, edges_created). Idempotent
    by ingest_key: an unchanged path refreshes label/content/metadata on the
    existing node in place rather than duplicating it."""
    fresh = build_skill_node(skill_md_path, scope, project_key, origin_runtime)
    ikey = fresh.metadata["ingest_key"]
    existing = _find_skill_node_by_ingest_key(store, ikey)

    if existing is None:
        node = store.add_node(fresh)
        created = True
    else:
        node = store.update_node(
            existing.node_id,
            label=fresh.label,
            content=fresh.content,
            metadata=fresh.metadata,
        )
        created = False

    edges = _link_matching_entities(store, node)
    return node, created, edges


def ingest_roots(
    store: GraphStore,
    paths: Optional[List[str]] = None,
    include_global: bool = False,
    cwd: Optional[Path] = None,
) -> Dict:
    """Scan every resolved root and ingest each SKILL.md found. Returns a
    summary dict: {"scanned", "created", "refreshed", "edges", "skills": [...]}"""
    roots = resolve_roots(paths=paths, include_global=include_global, cwd=cwd)
    summary = {"scanned": 0, "created": 0, "refreshed": 0, "edges": 0, "skills": []}
    for root, scope, project_key in roots:
        origin_runtime = _origin_runtime_for_root(root)
        for skill_md in discover_skill_files(root):
            node, created, edges = ingest_skill_file(store, skill_md, scope, project_key, origin_runtime)
            summary["scanned"] += 1
            summary["edges"] += edges
            if created:
                summary["created"] += 1
            else:
                summary["refreshed"] += 1
            summary["skills"].append(node)
    return summary


def sort_user_before_engine(nodes: List[Node]) -> List[Node]:
    """Same-name skills sort origin=user before origin=engine, everywhere
    skills are listed or recalled. Stable: ties keep their relative order
    (newest-first, when the caller already sorted that way)."""
    def key(node: Node) -> int:
        return 0 if (node.metadata or {}).get("origin") == "user" else 1
    return sorted(nodes, key=key)


def list_skills(
    store: GraphStore,
    project: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Node]:
    """All SKILL nodes, optionally filtered by project_key / metadata
    status, user-before-engine ordered."""
    nodes = _all_nodes(store, NodeType.SKILL)
    if project is not None:
        nodes = [n for n in nodes if n.project_key == project]
    if status is not None:
        nodes = [n for n in nodes if (n.metadata or {}).get("status") == status]
    return sort_user_before_engine(nodes)


def skill_index_row(node: Node) -> str:
    """The one-line index-row text recall shows for a SKILL node's
    `content` — NEVER the full body (that's what `skills show`/`revien
    skills show` is for). D1 leftover, wired into
    revien/retrieval/engine.py's result-building loop.

    A human-authored (D1) skill carries description/triggers metadata:
    "<description> — triggers: a, b". An engine proposal (D2) has no
    description — it carries a `steps` list instead, so it gets the same
    shape with its steps standing in for triggers. A node with neither
    (shouldn't happen, but never crash recall over it) falls back to its
    label."""
    md = node.metadata or {}
    description = md.get("description")
    if description is not None:
        triggers = md.get("triggers") or []
        trig_str = ", ".join(triggers)
        return f"{description} — triggers: {trig_str}" if trig_str else description
    steps = md.get("steps") or []
    if steps:
        return f"proposed skill — steps: {', '.join(steps)}"
    return node.label


def show_skill(store: GraphStore, name: str) -> Optional[Node]:
    """The highest-precedence (user-before-engine) SKILL node matching
    `name` case-insensitively, or None."""
    nodes = [n for n in _all_nodes(store, NodeType.SKILL) if n.label.strip().lower() == name.strip().lower()]
    if not nodes:
        return None
    return sort_user_before_engine(nodes)[0]

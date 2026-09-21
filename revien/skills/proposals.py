"""
Revien Skills — proposals (thin WS3, leg D2).

Detects a repeated ACTION sequence (the same few "I'll X" / "next I'll Y"
commitments, in the same order, across several sessions) and turns it into
an engine-origin SKILL proposal: a draft procedure the human can accept,
decline, or ignore. Nothing here ever creates an ACTIVE skill — origin=engine
proposals start (and, on accept, STAY) origin=engine; only their status
moves proposed -> active. A user-authored skill (origin=user, D1) is never
touched by this module: same-name collisions are avoided structurally,
because every proposal's label carries a "proposed: " prefix no hand-written
skill would use.

DERIVED_FROM direction (confirmed against the actual convention this
codebase runs, not the schema.py docstring's wording — see the module-level
note in graph/operations.py:get_lineage / get_children_of): the DERIVED
node is the edge's SOURCE, the ancestor it was derived from is the edge's
TARGET. A SKILL proposal is derived FROM its source ACTION nodes, so:
    edge.source_node_id = proposal.node_id (the SKILL, the derived thing)
    edge.target_node_id = action_node.node_id (the ancestor material)
This is what GraphOperations.get_children_of / get_lineage already walk;
wiring it the other way would make the proposal invisible to both.

Governance, spelled out because it is the product, not an implementation
detail:
  - propose_skills() only ever writes status="proposed", origin="engine".
  - accept_proposal() flips status -> "active"; origin stays "engine" —
    acceptance is not a claim of authorship.
  - Every propose/accept/decline call writes a record_audit row (before/after
    snapshots) with op "skill_propose" / "skill_accept" / "skill_decline",
    IN ADDITION TO whatever generic "create"/"update" row store.add_node /
    store.update_node already writes — the semantic op name is what a
    governance audit greps for.
  - The third decline invalidates the proposal (GraphOperations.
    invalidate_node) rather than deleting it — soft, reversible, same as
    every other invalidation in this codebase.
  - metadata never carries "curated" on an engine-origin node — that flag is
    reserved for human-authored (D1) skills; it is how the CSL gate tells
    the two apart.
"""

import hashlib
import re
from collections import defaultdict, namedtuple
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.ingestion.extractor import RuleBasedExtractor

ARROW = " -> "  # ASCII on purpose — U+2192 crashes Windows console encodings
# (cp1252/cp437) wherever a label or step-join reaches stdout/CliRunner.
# The one place this joiner string is spelled out.

# Leading pronoun/article tokens stripped during normalization — "i'll ping
# Theo" and "ping Theo" are the same step. Matched against a token AFTER
# punctuation stripping, so "i'll" and "ill" (apostrophe already gone) both
# need an entry.
_LEADING_WORDS = {
    "i", "ill", "im", "id", "we", "well", "wed", "youll", "you", "lets",
    "let", "the", "a", "an", "he", "hell", "she", "shell", "they", "theyll",
}

DEFAULT_MIN_OCCURRENCES = 3
DEFAULT_MIN_SESSIONS = 2
DEFAULT_NGRAM = (2, 4)
DECLINE_INVALIDATE_THRESHOLD = 3


def normalize_label(text: str) -> str:
    """Lowercase, strip punctuation, drop a leading article/pronoun,
    collapse whitespace. Used both for pattern-hashing (two differently-
    punctuated renderings of the same step must hash identically) and for
    recall's query/step keyword match."""
    lowered = (text or "").strip().lower().replace("’", "'")
    tokens = re.findall(r"[a-z0-9']+", lowered)
    # Apostrophes stripped everywhere, not just leading/trailing, so a
    # contraction collapses to one comparable token: "i'll" -> "ill" (which
    # is exactly what _LEADING_WORDS lists), "don't" -> "dont".
    tokens = [t.replace("'", "") for t in tokens]
    tokens = [t for t in tokens if t]
    while tokens and tokens[0] in _LEADING_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


# G8(c): the lean row detect_repeated_sequences actually needs per ACTION
# node — node_id/label/project_key/session_key/recorded_at/created_at — NOT
# a full Node (content, metadata dict, confidence, etc.). Same field names
# as Node's own attributes on purpose, so _session_group_key and the window
# logic below work unmodified against either a Node or an _ActionRow.
_ActionRow = namedtuple(
    "_ActionRow",
    "node_id label project_key session_key recorded_at created_at origin_runtime",
)


def _session_group_key(node: Union[Node, "_ActionRow"]) -> Tuple[Optional[str], str]:
    """(project_key, session_key) when the node has one; otherwise the
    documented time-gap fallback — (project_key, recorded_at's date) —
    since nodes with session_key None still cluster by "the same rough
    sitting" rather than being thrown out of grouping entirely."""
    if node.session_key:
        return (node.project_key, f"session:{node.session_key}")
    when = node.recorded_at or node.created_at
    date_str = when.date().isoformat() if when else "unknown"
    return (node.project_key, f"date:{date_str}")


def _is_contiguous_subsequence(short: List[str], long: List[str]) -> bool:
    """True iff `short` appears as a contiguous run inside `long` (and is
    strictly shorter — an equal-length different sequence is never a
    sub-pattern of itself)."""
    n, m = len(short), len(long)
    if n >= m:
        return False
    for i in range(0, m - n + 1):
        if long[i:i + n] == short:
            return True
    return False


def _all_action_rows(store: GraphStore) -> List[_ActionRow]:
    """Live (non-invalidated) ACTION nodes as lean tuples (G8c) — a 16.7k-
    ACTION corpus building a Python list of 16.7k FULL Node objects (each
    carrying content, metadata dict, confidence, timestamps it never uses
    here) was measured as unnecessary retained memory relative to what
    detect_repeated_sequences actually reads: node_id, label, project_key,
    session_key, recorded_at, created_at. Each page is reduced to tuples
    and the Node objects dropped before the next page is fetched. Page
    size 1000 (was 500) halves the round trips at that scale."""
    out: List[_ActionRow] = []
    offset = 0
    page = 1000
    while True:
        batch = store.list_nodes(node_type=NodeType.ACTION, limit=page, offset=offset)
        out.extend(
            _ActionRow(n.node_id, n.label, n.project_key, n.session_key,
                       n.recorded_at, n.created_at, n.origin_runtime)
            for n in batch if n.invalidated_at is None
        )
        if len(batch) < page:
            return out
        offset += page


def detect_repeated_sequences(
    store: GraphStore,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    ngram: Tuple[int, int] = DEFAULT_NGRAM,
) -> List[Dict]:
    """Find repeated ACTION step-sequences that qualify as a proposal.

    Live (non-invalidated) ACTION nodes are grouped by (project_key,
    session_key) — or the (project_key, recorded_at date) fallback when
    session_key is None — ordered by recorded_at then created_at within
    each group. Sliding windows of size 2..4 (inclusive, longest first) are
    hashed by their NORMALIZED step text (sha256 of the steps joined by
    "\\n"); a window qualifies when it occurs >= min_occurrences times
    total AND across >= min_sessions distinct groups.

    Sub-pattern suppression: when a longer qualifying window's step
    sequence contains a shorter qualifying window as a contiguous run, the
    shorter one is dropped — one workflow yields one proposal, not three
    (a 4-step qualifying pattern would otherwise also register as two
    3-step and three 2-step qualifying patterns of the SAME underlying
    workflow).

    Returns one dict per surviving pattern:
        {pattern_hash, steps (normalized), display_steps (original label
         text, same order), occurrences, sessions, project_key, node_ids}
    """
    nodes = _all_action_rows(store)

    groups: Dict[Tuple, List[Node]] = defaultdict(list)
    for n in nodes:
        groups[_session_group_key(n)].append(n)
    for key in groups:
        groups[key].sort(key=lambda n: (n.recorded_at or n.created_at, n.created_at))

    lo, hi = ngram
    acc: Dict[str, Dict] = {}

    for group_key, group_nodes in groups.items():
        project_key = group_key[0]
        normalized = [normalize_label(n.label) for n in group_nodes]
        for size in range(hi, lo - 1, -1):
            if len(group_nodes) < size:
                continue
            for i in range(0, len(group_nodes) - size + 1):
                window_norm = tuple(normalized[i:i + size])
                if any(not step for step in window_norm):
                    continue  # a step that normalized to nothing isn't a step
                phash = hashlib.sha256(
                    "\n".join(window_norm).encode("utf-8")
                ).hexdigest()
                window_nodes = group_nodes[i:i + size]
                entry = acc.setdefault(phash, {
                    "pattern_hash": phash,
                    "steps": list(window_norm),
                    "display_steps": [wn.label for wn in window_nodes],
                    "project_key": project_key,
                    "occurrences": 0,
                    "session_keys": set(),
                    "node_ids": set(),
                    "origin_runtimes": set(),
                })
                entry["occurrences"] += 1
                entry["session_keys"].add(group_key)
                entry["node_ids"].update(wn.node_id for wn in window_nodes)
                entry["origin_runtimes"].update(wn.origin_runtime for wn in window_nodes)

    qualifying = [
        e for e in acc.values()
        if e["occurrences"] >= min_occurrences and len(e["session_keys"]) >= min_sessions
    ]
    # Longest window first, so shorter contained patterns can be suppressed
    # against an already-kept longer one.
    qualifying.sort(key=lambda e: (-len(e["steps"]), -e["occurrences"], e["pattern_hash"]))

    kept: List[Dict] = []
    for entry in qualifying:
        if any(_is_contiguous_subsequence(entry["steps"], k["steps"]) for k in kept):
            continue
        kept.append(entry)

    results = [
        {
            "pattern_hash": e["pattern_hash"],
            "steps": e["steps"],
            "display_steps": e["display_steps"],
            "occurrences": e["occurrences"],
            "sessions": len(e["session_keys"]),
            "project_key": e["project_key"],
            "node_ids": sorted(e["node_ids"]),
            # G3: the proposal built from this pattern is stamped with
            # this origin_runtime so a filtered recall's skill_proposals
            # can be gated the same way every other node is. Only
            # attributed when EVERY source ACTION node agrees — a pattern
            # whose occurrences span more than one runtime (or include an
            # unknown-provenance node) gets None, an honest "don't know",
            # never a guess.
            "origin_runtime": (
                next(iter(e["origin_runtimes"]))
                if len(e["origin_runtimes"]) == 1
                else None
            ),
        }
        for e in kept
    ]
    results.sort(key=lambda r: (-r["occurrences"], -len(r["steps"]), r["pattern_hash"]))
    return results


def _skeleton_body(display_steps: List[str], occurrences: int, sessions: int) -> str:
    lines = ["## Steps", ""]
    lines.extend(f"{i}. {step}" for i, step in enumerate(display_steps, start=1))
    lines.append("")
    lines.append(f"_Observed {occurrences} times across {sessions} sessions._")
    return "\n".join(lines)


def _label_for(display_steps: List[str]) -> str:
    return ("proposed: " + ARROW.join(display_steps))[:200]


def _draft_body(
    extractor,
    is_llm: bool,
    display_steps: List[str],
    occurrences: int,
    sessions: int,
) -> Tuple[str, bool]:
    """(content, draft). Rule-based path (the default): always the
    skeleton, draft=False — recall must NOT surface these. LLM path
    (REVIEN_EXTRACTOR != rule-based, or a caller-injected non-rule-based
    extractor): draft=True is reported truthfully regardless of whether the
    optional `draft_skill_body` hook actually produced prose — an LLM
    backend was consulted, that's the fact draft=True records. Any failure
    (missing hook, exception, bad return) falls back to the skeleton text
    so a flaky/offline backend can never break proposal generation."""
    skeleton = _skeleton_body(display_steps, occurrences, sessions)
    if not is_llm:
        return skeleton, False
    draft_fn = getattr(extractor, "draft_skill_body", None)
    if draft_fn is None:
        return skeleton, True
    try:
        drafted = draft_fn(steps=display_steps, occurrences=occurrences, sessions=sessions)
    except Exception:
        return skeleton, True
    return (drafted or skeleton), True


def _resolve_extractor(extractor):
    """Same selection the ingestion pipeline makes (pipeline.py: `extractor
    or build_extractor()`), so proposals honour REVIEN_EXTRACTOR exactly
    the way the rest of ingestion does. `is_llm` is derived from the
    resolved type, not a second env read, so an injected stub extractor
    (tests) is trusted over whatever the environment happens to say."""
    from revien.ingestion.extractor_llm import build_extractor

    resolved = extractor or build_extractor()
    return resolved, not isinstance(resolved, RuleBasedExtractor)


def _load_proposal_index(store: GraphStore) -> Dict[str, Node]:
    """{pattern_hash: engine-origin SKILL node} — ONE pass over every SKILL
    node (G8a). This replaces the old per-pattern _find_proposal_by_
    pattern_hash, which page-scanned ALL SKILL nodes for EVERY pattern —
    O(patterns * skills). propose_skills builds this once per call and
    does O(1) dict lookups instead: O(patterns + skills)."""
    index: Dict[str, Node] = {}
    offset = 0
    page = 1000
    while True:
        batch = store.list_nodes(node_type=NodeType.SKILL, limit=page, offset=offset)
        for node in batch:
            md = node.metadata or {}
            phash = md.get("pattern_hash")
            if md.get("origin") == "engine" and phash:
                index[phash] = node
        if len(batch) < page:
            return index
        offset += page


def _existing_derived_targets(store: GraphStore, skill_node_id: str) -> set:
    return {
        edge.target_node_id
        for edge in store.get_edges_for_node(skill_node_id)
        if edge.edge_type == EdgeType.DERIVED_FROM and edge.source_node_id == skill_node_id
    }


def propose_skills(
    store: GraphStore,
    extractor=None,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    min_sessions: int = DEFAULT_MIN_SESSIONS,
    progress=None,
) -> Dict:
    """Detect qualifying repeated ACTION sequences and create/refresh one
    engine-origin SKILL proposal each. Idempotent by pattern_hash: a
    re-run on an unchanged pattern updates occurrences/sessions/content in
    place and adds only the DERIVED_FROM edges that don't already exist —
    it never duplicates a proposal node or an edge. An already-accepted
    proposal (status active) is never demoted back to "proposed" by a
    re-run; its origin stays "engine" either way.

    G8: O(n), not O(patterns * skills) — ``proposal_index`` is built with
    ONE pass over every SKILL node (see _load_proposal_index) instead of
    the old per-pattern page-scan.

    ``progress`` (optional): a ``callable(str)`` invoked with two status
    lines — one right after pattern detection ("patterns found: N"), one
    at the end ("proposals written: N") — the hook `revien skills propose`
    uses to print a progress line without this module importing click.
    """
    resolved_extractor, is_llm = _resolve_extractor(extractor)
    patterns = detect_repeated_sequences(
        store, min_occurrences=min_occurrences, min_sessions=min_sessions
    )
    if progress is not None:
        progress(f"patterns found: {len(patterns)}")
    summary = {"detected": len(patterns), "created": 0, "updated": 0, "edges": 0, "proposals": []}
    proposal_index = _load_proposal_index(store)

    # G8: ONE commit for the whole run, not one per add_node/record_audit/
    # add_edge call. At the 997-pattern / 24.7k-edge synthetic this was
    # measured as the actual first-run bottleneck (150s) — SQLite commits
    # fsync, and store.add_edge/add_node/record_audit each call
    # self._commit() individually outside a transaction. store.transaction()
    # is reentrant (nests by depth, only the outermost commit is real — see
    # its docstring) so every write below still gets its normal audit-row
    # guarantees, just batched into one disk sync at the end. A failure
    # partway rolls back the WHOLE batch, same all-or-nothing shape
    # import_graph already uses for its bulk writes.
    with store.transaction():
        for pattern in patterns:
            body, draft = _draft_body(
                resolved_extractor, is_llm,
                pattern["display_steps"], pattern["occurrences"], pattern["sessions"],
            )
            label = _label_for(pattern["display_steps"])
            existing = proposal_index.get(pattern["pattern_hash"])

            metadata = {
                "origin": "engine",
                "status": "proposed",
                "pattern_hash": pattern["pattern_hash"],
                "occurrences": pattern["occurrences"],
                "sessions": pattern["sessions"],
                "declines": (existing.metadata or {}).get("declines", 0) if existing else 0,
                "steps": pattern["steps"],
                "draft": draft,
                # Deliberately absent: "curated" — that flag is reserved for
                # human-authored (D1) skills; an engine proposal never sets it.
            }
            if existing is None:
                node = Node(
                    node_type=NodeType.SKILL,
                    label=label,
                    content=body,
                    source_id=f"skill-proposal:{pattern['pattern_hash']}",
                    metadata=metadata,
                    source_type=SourceType.INFERRED,
                    confidence=0.5,
                    project_key=pattern["project_key"],
                    # G3: origin_runtime is what lets a source-filtered
                    # recall's skill_proposals be gated at all — None (mixed
                    # or unknown source ACTION nodes) means this proposal
                    # never matches an active filter, same rule as every
                    # other node in the origin layer.
                    origin_runtime=pattern.get("origin_runtime"),
                    session_key=None,
                    recorded_at=datetime.now(timezone.utc),
                )
                node = store.add_node(node)
                store.record_audit(
                    node.node_id, "skill_propose",
                    before=None, after=node.model_dump(mode="json"),
                )
                summary["created"] += 1
            elif existing.invalidated_at is not None:
                # 3x-declined and soft-invalidated — leave it alone. Re-running
                # propose must not resurrect a proposal the human already
                # rejected three times.
                summary["proposals"].append(existing)
                continue
            elif (existing.metadata or {}).get("status") != "proposed":
                # The node has moved on (accepted -> "active", or any other
                # status a future path might set) — a re-run must NEVER touch
                # label/content/origin/status again once a human has acted on
                # it. Only the observational counters move: occurrences,
                # sessions, and any DERIVED_FROM edges the detector found that
                # aren't already wired (below). Everything else about the
                # existing node — label, content, status, origin, declines,
                # steps, draft — is carried forward untouched.
                before_snapshot = existing.model_dump(mode="json")
                frozen_metadata = dict(existing.metadata or {})
                frozen_metadata["occurrences"] = pattern["occurrences"]
                frozen_metadata["sessions"] = pattern["sessions"]
                node = store.update_node(
                    existing.node_id,
                    metadata=frozen_metadata,
                    _audit_op=None,
                )
                store.record_audit(
                    node.node_id, "skill_propose",
                    before=before_snapshot, after=node.model_dump(mode="json"),
                )
                summary["updated"] += 1
            else:
                before_snapshot = existing.model_dump(mode="json")
                node = store.update_node(
                    existing.node_id,
                    label=label, content=body, metadata=metadata,
                    _audit_op=None,
                )
                store.record_audit(
                    node.node_id, "skill_propose",
                    before=before_snapshot, after=node.model_dump(mode="json"),
                )
                summary["updated"] += 1

            already = _existing_derived_targets(store, node.node_id)
            for action_node_id in pattern["node_ids"]:
                if action_node_id in already:
                    continue
                store.add_edge(Edge(
                    edge_type=EdgeType.DERIVED_FROM,
                    source_node_id=node.node_id,
                    target_node_id=action_node_id,
                    weight=0.8,
                ))
                summary["edges"] += 1

            summary["proposals"].append(node)

    if progress is not None:
        progress(
            f"proposals written: {summary['created']} created, "
            f"{summary['updated']} updated"
        )
    return summary


def accept_proposal(store: GraphStore, node_id: str, actor: str = "") -> Node:
    """status -> "active". origin stays "engine" — acceptance records that
    a human approved the WORKFLOW, not that they wrote it. Audit op
    "skill_accept" with before/after snapshots.

    Refuses (ValueError) when the node is not a SKILL, is invalidated, or
    its status is not "proposed" — accept is a one-way door from proposed
    to active; it is not a way to resurrect a declined/invalidated
    proposal or to re-confirm an already-active/ingested skill.
    """
    node = store.get_node(node_id)
    if node is None:
        raise ValueError(f"No such node: {node_id}")
    if node.node_type != NodeType.SKILL:
        raise ValueError(f"Node {node_id} is not a SKILL node")
    if node.invalidated_at is not None:
        raise ValueError(f"Node {node_id} is invalidated and cannot be accepted")
    if (node.metadata or {}).get("status") != "proposed":
        raise ValueError(
            f"Node {node_id} status is "
            f"{(node.metadata or {}).get('status')!r}, not 'proposed' -- cannot accept"
        )

    before = node.model_dump(mode="json")
    metadata = dict(node.metadata or {})
    metadata["status"] = "active"
    updated = store.update_node(node_id, metadata=metadata, _audit_op=None)
    store.record_audit(
        node_id, "skill_accept", actor=actor,
        before=before, after=updated.model_dump(mode="json"),
    )
    return updated


def decline_proposal(store: GraphStore, node_id: str, actor: str = "", reason: str = "") -> Node:
    """declines += 1. Audit op "skill_decline" with before/after snapshots.
    The THIRD decline soft-invalidates the proposal via
    GraphOperations.invalidate_node (its own "invalidate" audit row rides
    alongside this one) — content is retained, just excluded from default
    recall/listing, same as every other invalidation in this codebase.

    Refuses (ValueError) when the node is not a SKILL, is invalidated, or
    its status is not "proposed" — decline only makes sense against a live
    proposal awaiting a decision.
    """
    node = store.get_node(node_id)
    if node is None:
        raise ValueError(f"No such node: {node_id}")
    if node.node_type != NodeType.SKILL:
        raise ValueError(f"Node {node_id} is not a SKILL node")
    if node.invalidated_at is not None:
        raise ValueError(f"Node {node_id} is already invalidated")
    if (node.metadata or {}).get("status") != "proposed":
        raise ValueError(
            f"Node {node_id} status is "
            f"{(node.metadata or {}).get('status')!r}, not 'proposed' -- cannot decline"
        )

    before = node.model_dump(mode="json")
    metadata = dict(node.metadata or {})
    declines = int(metadata.get("declines", 0)) + 1
    metadata["declines"] = declines
    updated = store.update_node(node_id, metadata=metadata, _audit_op=None)
    store.record_audit(
        node_id, "skill_decline", actor=actor,
        before=before, after=updated.model_dump(mode="json"),
    )

    if declines >= DECLINE_INVALIDATE_THRESHOLD:
        ops = GraphOperations(store)
        invalidated = ops.invalidate_node(
            node_id, reason=reason or "declined 3x", construct_id=actor
        )
        if invalidated is not None:
            updated = invalidated

    return updated


def _query_keywords(query: str) -> set:
    return {w for w in normalize_label(query).split(" ") if len(w) >= 4}


def matching_proposals(
    store: GraphStore,
    query: str,
    source_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """Draft (draft=True) engine-origin proposals relevant to a recall
    query — the recall response's `skill_proposals` field.

    Surfaced only when ALL of: status == "proposed", metadata["draft"] is
    True (an LLM actually drafted readable prose — the rule-based
    skeleton is never surfaced through recall), and the node is not
    invalidated. Relevance is a shared-keyword match (normalized, length
    >= 4) between the query and the proposal's normalized step labels —
    same normalize_label() used for pattern detection, so "Sync fernweh
    branches" recall-matches a proposal whose step said "I'll sync the
    Fernweh branches".

    G3 (RULING): ``source_filter`` is the SAME normalized filter recall()
    applies to results/tensions/path — None means unfiltered (byte-
    identical to before); ANY other value (including an empty list, G4's
    fail-closed case) means only proposals whose origin_runtime is IN the
    filter are returned, exactly the same membership rule used
    everywhere else in the origin layer. A proposal's origin_runtime
    reflects which runtime's ACTION nodes it was derived from (set at
    propose_skills() ingest time via the pattern's project/nodes) — a
    proposal built out of claude-code ACTIONs must never surface through
    a recall filtered to source="codex".

    G9: prefiltered in SQL (store.list_draft_proposed_skills) instead of
    a page-scan of every SKILL node — with zero draft proposals in the
    graph this costs ~one indexed-free LIKE scan against an empty result,
    regardless of how many accepted/declined/non-draft SKILL nodes exist.

    Rows use a FIXED key set (node_id, label, occurrences, sessions,
    project_key, steps) so toon.py can carry them as a uniform tabular
    array; `steps` is the normalized step list already joined by " -> "
    (ARROW) into one string, matching the plan's TOON column contract.
    """
    keywords = _query_keywords(query)
    if not keywords:
        return []

    out: List[Dict] = []
    offset = 0
    page = 1000
    while True:
        batch = store.list_draft_proposed_skills(limit=page, offset=offset)
        if not batch:
            break
        for node in batch:
            if node.invalidated_at is not None:
                continue
            md = node.metadata or {}
            # The LIKE prefilter can only narrow candidates, never confirm
            # them (a substring hit isn't a parsed-JSON guarantee) — these
            # three checks are the exact same confirmation the old full
            # scan did, just against a far smaller candidate set.
            if md.get("origin") != "engine":
                continue
            if md.get("status") != "proposed":
                continue
            if not md.get("draft"):
                continue
            if (source_filter is not None
                    and node.origin_runtime not in source_filter):
                continue
            steps = md.get("steps") or []
            step_words = set()
            for step in steps:
                step_words.update(w for w in step.split(" ") if len(w) >= 4)
            if not (keywords & step_words):
                continue
            out.append({
                "node_id": node.node_id,
                "label": node.label,
                "occurrences": md.get("occurrences", 0),
                "sessions": md.get("sessions", 0),
                "project_key": node.project_key,
                "steps": ARROW.join(steps),
            })
        if len(batch) < page:
            break
        offset += page
    return out

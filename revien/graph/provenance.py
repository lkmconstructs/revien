"""Write-time provenance: which unit supplied which node, and with what wording.

A forget cascade has to know what a unit "owns". Inferring that after the fact
from edges, ``source_id`` and substring matches missed one writer's shape in
every round (OpenAI messages chain by FOLLOWED_BY, entities are stamped but not
linked, an LLM description is attested by its label only, a node re-attributed
to an arbitrary unit outlives every unit that said its wording). So every writer
records it when it writes, in ``node_provenance`` (content-free: ids and two
booleans, never text):

    node_id     the node
    unit_id     the node_id of the CONTEXT node of the unit that wrote or merged
                into it ("~external" is the sentinel for a writer that is not a
                unit: a manually added node a unit later merged into, or a node
                that outlived a deliberate non-cascade forget of its unit)
    label_ok    this unit said the node's label (normalized-label equal)
    wording_ok  this unit said the node's stored wording (label and content);
                generated content ("Entity: X", "Topic: X (mentioned 3 times)")
                carries no wording, so for it label_ok implies wording_ok

A cascade removes the unit's rows, deletes nodes left with no row, and for the
rest keeps only what a live unit attests: wording_ok keeps the node as is,
label_ok alone keeps it with its content reset to the label, neither deletes
it. Existing databases are backfilled once from edges, ``source_id`` and the
live context text (``infer_rows``); the same inference covers a node that a
writer that does not record (an older build) left without any row.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

from .normalize import normalize_label
from .schema import Node, NodeType

EXTERNAL = "~external"

SENTENCE_TYPES = (NodeType.DECISION, NodeType.ACTION, NodeType.FACT,
                  NodeType.PREFERENCE)


def squash(text: str) -> str:
    return " " + re.sub(r"[\W_]+", " ", (text or "").lower()).strip() + " "


def is_generated_content(node_type: NodeType, label: str, content: str) -> bool:
    """True when the content is text Revien generated from the label (the regex
    extractor's "Entity: X", "Technology/tool: X", "Topic: X (mentioned N
    times)", or the label itself), so the label is the only wording."""
    c = (content or "").strip()
    if not c:
        return True
    lab = re.escape((label or "").strip())
    if re.fullmatch(lab, c, re.IGNORECASE):
        return True
    if node_type == NodeType.ENTITY and re.fullmatch(
            rf"(Entity|Technology/tool): {lab}", c, re.IGNORECASE):
        return True
    if node_type == NodeType.TOPIC and re.fullmatch(
            rf"Topic: {lab}( \(mentioned \d+ times\))?", c, re.IGNORECASE):
        return True
    return False


def generic_content(node_type: NodeType, label: str) -> str:
    """What a kept node's content is reset to when only its label is attested."""
    if node_type == NodeType.ENTITY:
        return f"Entity: {label}"
    if node_type == NodeType.TOPIC:
        return f"Topic: {label}"
    return label


def flags_for_merge(candidate: Node, existing: Node) -> Tuple[bool, bool]:
    """(label_ok, wording_ok) for a unit whose extractor proposed ``candidate``
    and whose dedup reused ``existing`` instead."""
    label_ok = (normalize_label(candidate.label)
                == normalize_label(existing.label)
                and bool(normalize_label(candidate.label)))
    if not label_ok:
        return False, False
    if (existing.node_type not in SENTENCE_TYPES
            and is_generated_content(existing.node_type, existing.label,
                                     existing.content)
            and is_generated_content(candidate.node_type, candidate.label,
                                     candidate.content)):
        return True, True
    return True, squash(candidate.content) == squash(existing.content)


def flags_for_mention(node: Node) -> Tuple[bool, bool]:
    """A unit's text names ``node`` by its label (declared link, gazetteer
    mention). It said the label; it said the stored wording only when that
    wording is generated from the label."""
    return True, is_generated_content(node.node_type, node.label, node.content)


def flags_in_text(node: Node, text: str) -> Tuple[bool, bool]:
    """Provable attestation of ``node`` by a unit's verbatim text."""
    hay = squash(text)
    lab = squash(node.label)
    label_ok = bool(lab.strip()) and lab in hay
    if node.node_type in SENTENCE_TYPES:
        wording_ok = label_ok and squash(node.content) in hay
    else:
        wording_ok = label_ok and is_generated_content(
            node.node_type, node.label, node.content)
    return label_ok, wording_ok


def infer_rows(conn, unit_ids: Optional[Iterable[str]] = None,
               node_ids: Optional[Iterable[str]] = None) -> int:
    """Infer and insert rows for nodes that have none.

    Scope: every row-less node (backfill) or, with ``unit_ids``, the row-less
    nodes those units' context nodes link or whose source_id they carry; with
    ``node_ids``, just those. A node gets a row for each live context node it
    is edge-linked to and for each context node carrying its source_id. A node
    whose source_id is not a unit's (a manual add, a shared channel) also gets
    the external row, because something other than a unit wrote it. Returns the
    number of nodes that received rows. Idempotent: only row-less nodes."""
    from .suppression import is_unit_source_id

    ctx_rows = conn.execute(
        "SELECT node_id, source_id, content FROM nodes WHERE node_type = 'context' "
        "AND node_id NOT LIKE 'tombstone:%'").fetchall()
    ctx_text = {r[0]: r[2] or "" for r in ctx_rows}
    ctx_src: Dict[str, List[str]] = {}
    for cid, sid, _ in ctx_rows:
        if sid:
            ctx_src.setdefault(sid, []).append(cid)

    if node_ids is not None:
        scope = set(node_ids)
    elif unit_ids is not None:
        units = list(unit_ids)
        scope = set()
        for i in range(0, len(units), 300):
            chunk = units[i:i + 300]
            marks = ",".join("?" * len(chunk))
            for (a, b) in conn.execute(
                    "SELECT source_node_id, target_node_id FROM edges WHERE "
                    f"source_node_id IN ({marks}) OR target_node_id IN ({marks})",
                    chunk * 2).fetchall():
                scope.add(a)
                scope.add(b)
            for (nid,) in conn.execute(
                    "SELECT n.node_id FROM nodes n WHERE n.source_id IN ("
                    f"SELECT source_id FROM nodes WHERE node_id IN ({marks}))",
                    chunk).fetchall():
                scope.add(nid)
        scope -= set(units)
    else:
        scope = None

    has_rows = {r[0] for r in conn.execute(
        "SELECT DISTINCT node_id FROM node_provenance").fetchall()}
    q = ("SELECT node_id, node_type, label, content, source_id FROM nodes "
         "WHERE node_type != 'context' AND node_id NOT LIKE 'tombstone:%'")
    done = 0
    for nid, ntype, label, content, sid in conn.execute(q).fetchall():
        if nid in has_rows or (scope is not None and nid not in scope):
            continue
        node = Node(node_id=nid, node_type=NodeType(ntype), label=label,
                    content=content, source_id=sid or "")
        linked: List[str] = []
        for (a, b) in conn.execute(
                "SELECT source_node_id, target_node_id FROM edges "
                "WHERE source_node_id = ? OR target_node_id = ?",
                (nid, nid)).fetchall():
            other = b if a == nid else a
            if other in ctx_text and other not in linked:
                linked.append(other)
        same = [c for c in ctx_src.get(sid or "", []) if c not in linked]
        units_of = linked + same
        rows: List[Tuple[str, int, int]] = []
        for cid in units_of:
            # The creator (it carries the unit's source_id) is trusted for
            # what its own extraction stored when it is the only unit with
            # that source; otherwise only the text proves it.
            creator = bool(sid) and cid in ctx_src.get(sid, [])
            lo, wo = flags_in_text(node, ctx_text[cid])
            if creator and len(ctx_src.get(sid, [])) == 1:
                lo = wo = True
            rows.append((cid, int(lo), int(wo)))
        if not (sid and is_unit_source_id(sid) and ctx_src.get(sid)):
            # Not written by a unit that still exists: a manual add, a shared
            # channel, or the survivor of a unit that is gone.
            if not (sid and is_unit_source_id(sid)):
                rows.append((EXTERNAL, 1, 1))
        if not rows:
            continue
        conn.executemany(
            "INSERT OR IGNORE INTO node_provenance "
            "(node_id, unit_id, label_ok, wording_ok) VALUES (?, ?, ?, ?)",
            [(nid, u, lo, wo) for (u, lo, wo) in rows])
        done += 1
    return done

"""
Revien Alias Resolution — evidence-backed ALIAS_OF edges between surface
forms and synonymous concepts, so recall anchored to one name reaches the
others.

WHY: "Sam", "Sam R.", and "sam@..." land in the graph as three separate
ENTITY nodes (extraction has no way to know they're the same person), and a
recall anchor that exact-matches one of them never walks to the other two —
their facts are simply unreachable from that query. The same gap holds for
conceptual synonyms with zero shared vocabulary ("offline mode" / "Roadmap
2026"). This module never merges those nodes (Sovereignty contract: nothing
destroyed, nothing silent — see consolidate.py's header) — it draws an
explicit, evidenced, REVERSIBLE ALIAS_OF edge and lets recall (engine.py
``_find_anchors``) union the alias's neighborhood into the anchor set.

Candidate generation is BLOCKED, never all-pairs — comparing every ENTITY to
every other ENTITY is O(n^2) label-similarity work for a signal (shared
surface form or shared meaning) that only ever fires between a handful of
NEAR entities. Two blocking rules generate the candidate SET fed to scoring:
  (a) share >=1 normalized token (name_form's usual shape: "sam" / "sam r"),
  (b) mutual top-K nearest neighbors by label embedding (conceptual aliasing:
      no shared vocabulary, but the meaning is close). Finding each entity's
      top-K neighbors still costs one cosine per OTHER entity — that part is
      inherent to nearest-neighbor search without an ANN index — but the
      CANDIDATE PAIRS scored below are bounded to blocked pairs only, not
      the full n^2 combination space. REVIEN_ALIAS_MAX_ENTITIES is the hard
      backstop: past that live-entity count the whole pass is skipped with a
      report note rather than silently degrading into a slow sweep.

Evidence scoring is PRECISION-FIRST: a false alias silently widens recall
across two DIFFERENT people/concepts, which poisons results forever until
someone notices and reverses it — worse than a missed alias, which just
costs the gap this leg exists to close. So both routes require
corroboration, never label similarity alone:
  * name_form   — normalized token-subset/containment ("sam" is a subset of
                  "sam r"), or a high fuzzy ratio — PLUS >=2 distinct shared
                  CONTEXT/TOPIC neighbors (same bar as conceptual, below) OR a
                  label embedding similarity clearing REVIEN_ALIAS_SIM_NAME
                  (0.85 default).
                    EXCEPTION — the SUBSET shape, at ANY token count: when
                    one label's normalized token set is a STRICT subset of
                    the other's (subset of "sam r" is "sam", but just as
                    much "new york times" contains "new york", "ford
                    foundation" contains "ford", "amazon rainforest"
                    contains "amazon"), co-occurrence can
                    NEVER draw the edge, however much of it exists. A
                    qualified superset is USUALLY a DIFFERENT entity that
                    merely shares the shorter word ("New York Times" the
                    newspaper is not "New York" the city) — and co-
                    occurrence is USELESS as a tiebreaker here because a
                    qualified superset co-occurs with its shorter name
                    constantly BY CONSTRUCTION (every mention of "the
                    Times" sits near "New York"; every mention of a laptop
                    sits near its owner's name). Occasionally the subset
                    shape IS the same entity in fuller form ("Sam" / "Sam
                    R."), but token shape alone cannot tell that case from
                    "New York" / "New York Times" — only real embedding
                    similarity can, so this shape REQUIRES embedding_sim >=
                    REVIEN_ALIAS_SIM_NAME and skips (with a report note, not
                    silently) when no embedder is available to supply it.
  * conceptual  — no name overlap at all, so evidence must be doubled: label
                  embedding similarity >= REVIEN_ALIAS_SIM_CONCEPT (0.90) AND
                  co-occurrence with >= REVIEN_ALIAS_COOC_MIN (2) distinct
                  shared CONTEXT/TOPIC neighbors. Similarity alone is cheap
                  and wrong often enough (near-synonyms that are NOT the same
                  thing) that it never draws an edge unassisted.

All five thresholds are env-tunable (mirrors the tension-backend / fence
env-gate convention elsewhere in this codebase) so a deployment can loosen or
tighten them without a code change; the defaults above are the conservative,
shipped behavior.

Guards (never negotiable, checked for every candidate pair):
  * CONFLICTS_WITH / CONTRADICTS between the pair -> never alias (a
    recognized tension is the opposite claim of "these are the same thing").
  * different node_types -> never alias.
  * either node soft-invalidated -> skipped before candidate generation even
    sees it.
  * a live ALIAS_OF edge already connects the pair (either direction) ->
    idempotent no-op, no duplicate edge.

Degrade path: the conceptual route needs label embeddings. When the semantic
layer is absent/disabled, or the embedder raises, conceptual scoring is
skipped with a report note and name_form (with co-occurrence corroboration)
still runs — evidence-based aliasing never depends on the heavy optional
stack to do SOME of its job.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from revien.graph.normalize import normalize_label
from revien.graph.operations import GraphOperations
from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore

# Cap per-item detail in the report — enough to review, never a dump (mirrors
# consolidate.py's REPORT_ITEM_CAP).
ALIAS_SAMPLE_CAP = 50

DEFAULT_MAX_ENTITIES = 20000
DEFAULT_TOP_K = 5
DEFAULT_SIM_NAME = 0.85
DEFAULT_SIM_CONCEPT = 0.90
DEFAULT_COOC_MIN = 2

# name_form is a corrected surface-form match (real weight, near-EXTRACTED
# confidence); conceptual is a softer synonym inference (real but lower
# weight) — the same shape as other inferred-edge defaults in this codebase.
NAME_FORM_WEIGHT = 0.8
CONCEPTUAL_WEIGHT = 0.6


def _env_float(name: str, default: float) -> float:
    """Read a float env override; malformed values fall back to the default
    (mirrors retrieval/scorer.py's _env_float — a bad knob must never crash
    the pass)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class AliasPassResult:
    """What one alias-inference pass actually did — feeds ConsolidationReport
    the way consolidate.py's other passes do. Nothing silent: `note` carries
    a skip/degrade reason whenever the pass didn't run to full strength."""
    ran: bool = False
    entities_considered: int = 0
    candidates_considered: int = 0
    edges_created: int = 0
    sample: List[Dict[str, Any]] = field(default_factory=list)
    note: Optional[str] = None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _structural_neighbor_ids(store: GraphStore, node_id: str) -> set:
    """Every node_id directly connected to node_id by any edge, either
    direction. Used to compute co-occurrence (shared CONTEXT/TOPIC
    neighbors) — the corroboration signal both alias routes need."""
    neighbors = set()
    for edge in store.get_edges_for_node(node_id):
        other = (edge.target_node_id if edge.source_node_id == node_id
                 else edge.source_node_id)
        neighbors.add(other)
    return neighbors


def _shared_context_topic_count(store: GraphStore, id_a: str, id_b: str) -> int:
    """Count of DISTINCT CONTEXT/TOPIC nodes connected to BOTH entities —
    "they co-occurred somewhere," the co-occurrence evidence term."""
    shared_ids = (_structural_neighbor_ids(store, id_a)
                  & _structural_neighbor_ids(store, id_b))
    if not shared_ids:
        return 0
    nodes = store.get_nodes_bulk(shared_ids)
    return sum(
        1 for nid in shared_ids
        if nid in nodes and nodes[nid].node_type in (NodeType.CONTEXT, NodeType.TOPIC)
    )


def _has_conflict_edge(store: GraphStore, id_a: str, id_b: str) -> bool:
    """True if a and b are already joined by CONFLICTS_WITH or CONTRADICTS —
    a recognized tension is the opposite claim of 'these are the same
    thing,' so it hard-blocks aliasing regardless of how strong the name/
    embedding evidence looks."""
    conflict_types = (EdgeType.CONFLICTS_WITH, EdgeType.CONTRADICTS)
    for edge in store.get_edges_for_node(id_a):
        if edge.edge_type in conflict_types:
            other = (edge.target_node_id if edge.source_node_id == id_a
                     else edge.source_node_id)
            if other == id_b:
                return True
    return False


def live_alias_edge_exists(store: GraphStore, id_a: str, id_b: str) -> bool:
    """True if a LIVE ALIAS_OF edge already connects a and b, either
    direction — the idempotency check. A REVERSED (invalidated) alias does
    not count: it may be re-evidenced and redrawn without archaeology."""
    for edge in store.get_edges_for_node(id_a):
        if edge.edge_type is EdgeType.ALIAS_OF and edge.invalidated_at is None:
            other = (edge.target_node_id if edge.source_node_id == id_a
                     else edge.source_node_id)
            if other == id_b:
                return True
    return False


def _name_overlap(norm_a: str, norm_b: str) -> bool:
    """Token-subset or containment after normalization — 'sam' is a subset
    of 'sam r'."""
    tokens_a, tokens_b = set(norm_a.split()), set(norm_b.split())
    if not tokens_a or not tokens_b:
        return False
    if tokens_a <= tokens_b or tokens_b <= tokens_a:
        return True
    return norm_a in norm_b or norm_b in norm_a


def _is_subset_shape(norm_a: str, norm_b: str) -> bool:
    """True when one label's normalized token set is a STRICT subset of the
    other's, at ANY token count — 'sam' is a subset of 'sam r'; 'john' of
    'john laptop'; 'new york' of 'new york times'; 'ford' of 'ford
    foundation'; 'amazon' of 'amazon rainforest'.

    This shape is the false-alias marker: a qualified superset is USUALLY a
    DIFFERENT entity that merely shares the shorter name/word ('New York'
    the city is not 'New York Times' the newspaper; 'Ford' the person is
    not 'Ford Foundation' the organization; 'John' is not "John's laptop").
    It is occasionally the SAME entity in a fuller form ('Sam' / 'Sam R.'),
    but token shape alone cannot tell those two cases apart, and co-
    occurrence is USELESS here too — a qualified superset co-occurs with
    its shorter name constantly by construction (every mention of "the
    Times" is also near "New York"). Only real embedding similarity can
    tell 'Sam R.' (same person) from 'New York Times' (different thing) —
    see the module header and the scoring loop for the mandatory-embedding
    rule this shape gets.

    Equal token sets (a == b, e.g. re-cased duplicates already caught by
    normalize_label) are NOT a strict subset of each other and fall through
    to _name_overlap's non-subset containment check instead.
    """
    tokens_a, tokens_b = set(norm_a.split()), set(norm_b.split())
    if not tokens_a or not tokens_b:
        return False
    return tokens_a < tokens_b or tokens_b < tokens_a


def run_alias_pass(
    store: GraphStore,
    ops: Optional[GraphOperations] = None,
    semantic: Optional[object] = None,
    max_entities: Optional[int] = None,
    top_k: Optional[int] = None,
    sim_name: Optional[float] = None,
    sim_concept: Optional[float] = None,
    cooc_min: Optional[int] = None,
    actor: str = "alias_pass",
) -> AliasPassResult:
    """Run one evidence-backed alias-inference pass over live ENTITY nodes.

    ``semantic`` is an optional SemanticIndex (or anything exposing
    ``is_enabled`` + ``_get_embedder()``) — when absent or disabled, the
    conceptual route is skipped with a note and name_form (with
    co-occurrence corroboration) still runs unaffected. Explicit args
    override the env knobs (see module header); None means "read the env".
    """
    ops = ops or GraphOperations(store)
    max_entities = (max_entities if max_entities is not None
                    else _env_int("REVIEN_ALIAS_MAX_ENTITIES", DEFAULT_MAX_ENTITIES))
    top_k = top_k if top_k is not None else _env_int("REVIEN_ALIAS_TOP_K", DEFAULT_TOP_K)
    sim_name = (sim_name if sim_name is not None
                else _env_float("REVIEN_ALIAS_SIM_NAME", DEFAULT_SIM_NAME))
    sim_concept = (sim_concept if sim_concept is not None
                   else _env_float("REVIEN_ALIAS_SIM_CONCEPT", DEFAULT_SIM_CONCEPT))
    cooc_min = (cooc_min if cooc_min is not None
                else _env_int("REVIEN_ALIAS_COOC_MIN", DEFAULT_COOC_MIN))

    result = AliasPassResult(ran=True)

    # Live ENTITY nodes only — invalidated (soft-deleted) entities never
    # enter candidate generation at all.
    entities: List[Node] = [
        n for n in store.list_nodes(node_type=NodeType.ENTITY, limit=999_999)
        if n.invalidated_at is None
    ]
    result.entities_considered = len(entities)

    if len(entities) > max_entities:
        result.ran = False
        result.note = (
            f"skipped: {len(entities)} live entities exceeds "
            f"REVIEN_ALIAS_MAX_ENTITIES={max_entities}"
        )
        return result

    if len(entities) < 2:
        result.note = "fewer than 2 live entities — nothing to pair"
        return result

    norm_labels = [normalize_label(n.label) for n in entities]

    # ── Blocking (a): shared normalized token ──────────────────────────
    token_index: Dict[str, List[int]] = {}
    for idx, norm in enumerate(norm_labels):
        for tok in norm.split():
            token_index.setdefault(tok, []).append(idx)

    pairs: set = set()
    for idx_list in token_index.values():
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a, b = idx_list[i], idx_list[j]
                pairs.add((min(a, b), max(a, b)))

    # ── Blocking (b): mutual top-K label-embedding neighbors ───────────
    # `vectors` (one per entity, same order) stays in scope through scoring
    # below, so a name_form pair's "OR embedding_sim >= sim_name"
    # corroboration can compute its OWN pairwise cosine directly — it must
    # not depend on that pair ALSO having been each other's mutual top-K
    # neighbor, which is a stricter, unrelated condition.
    vectors: Optional[List[List[float]]] = None
    if semantic is not None and getattr(semantic, "is_enabled", False):
        try:
            embedder = semantic._get_embedder()
            vectors = embedder.embed([n.label for n in entities])
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the pass
            vectors = None
            result.note = (
                f"conceptual aliasing disabled this pass: embedder failed "
                f"({exc!r}); name_form (with co-occurrence) still ran"
            )
        if vectors:
            n = len(entities)
            per_entity_topk: List[List[int]] = []
            for i in range(n):
                sims = sorted(
                    ((j, _cosine(vectors[i], vectors[j])) for j in range(n) if j != i),
                    key=lambda kv: kv[1], reverse=True,
                )[:top_k]
                per_entity_topk.append([j for j, _ in sims])
            for i in range(n):
                for j in per_entity_topk[i]:
                    if i in per_entity_topk[j]:  # MUTUAL top-K only
                        pairs.add((min(i, j), max(i, j)))
    else:
        result.note = (
            "semantic layer unavailable/disabled — conceptual aliasing "
            "needs label embeddings, skipped this pass; name_form "
            "(with co-occurrence) still ran"
        )

    # ── Evidence scoring (precision-first) ─────────────────────────────
    subset_skipped_no_embedder = 0
    for (i, j) in sorted(pairs):
        result.candidates_considered += 1
        a, b = entities[i], entities[j]

        if a.node_type != b.node_type:
            continue
        if a.invalidated_at is not None or b.invalidated_at is not None:
            continue
        if _has_conflict_edge(store, a.node_id, b.node_id):
            continue
        if live_alias_edge_exists(store, a.node_id, b.node_id):
            continue  # idempotent: already aliased, no duplicate

        norm_a, norm_b = norm_labels[i], norm_labels[j]
        sim = _cosine(vectors[i], vectors[j]) if vectors else None
        shared = _shared_context_topic_count(store, a.node_id, b.node_id)

        method: Optional[str] = None
        evidence_bits: List[str] = []
        weight = NAME_FORM_WEIGHT

        if _name_overlap(norm_a, norm_b):
            # Subset shape (ANY token count — "sam"/"sam r", "new york"/
            # "new york times", "ford"/"ford foundation" — one label's
            # tokens a strict subset of the other's) is the
            # false-alias marker: a qualified superset USUALLY names a
            # DIFFERENT entity that merely shares the shorter word ('New
            # York Times' is not 'New York'), occasionally the SAME entity
            # in fuller form ('Sam R.' is 'Sam'). Token shape can't tell
            # those apart, and co-occurrence is USELESS here too — a
            # qualified superset co-occurs with its shorter name
            # constantly BY CONSTRUCTION (every mention of "the Times" sits
            # near "New York"), so however much co-occurrence exists, it
            # NEVER draws this shape. Only real embedding similarity can.
            if _is_subset_shape(norm_a, norm_b):
                if sim is None:
                    subset_skipped_no_embedder += 1
                    continue
                if sim < sim_name:
                    continue
            else:
                # Non-subset name_form overlap (containment without a clean
                # token subset): corroboration bar matches conceptual's
                # (>= cooc_min distinct shared neighbors) OR sim >= sim_name.
                corroborated = shared >= cooc_min or (sim is not None and sim >= sim_name)
                if not corroborated:
                    continue
            method = "name_form"
            ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
            evidence_bits.append(
                f"name_form: {a.label!r} / {b.label!r} (normalized token "
                f"overlap, fuzzy ratio {ratio:.2f})"
            )
            if shared >= 1:
                evidence_bits.append(f"{shared} shared context/topic neighbor(s)")
            if sim is not None:
                evidence_bits.append(f"label embedding sim={sim:.3f}")
            weight = NAME_FORM_WEIGHT
        else:
            if sim is None or sim < sim_concept or shared < cooc_min:
                continue
            method = "conceptual"
            evidence_bits.append(
                f"conceptual: label embedding sim={sim:.3f} "
                f"(>= {sim_concept}), {shared} shared context/topic "
                f"neighbor(s) (>= {cooc_min})"
            )
            weight = CONCEPTUAL_WEIGHT

        edge = Edge(
            edge_type=EdgeType.ALIAS_OF,
            source_node_id=a.node_id,
            target_node_id=b.node_id,
            weight=weight,
            confidence=sim if sim is not None else 0.75,
            confidence_set_by=actor,
            source_context="; ".join(evidence_bits),
            metadata={
                "embedding_sim": sim,
                "cooccurrence": shared,
                "method": method,
            },
        )
        store.add_edge_audited(edge, actor=actor)
        result.edges_created += 1
        if len(result.sample) < ALIAS_SAMPLE_CAP:
            result.sample.append({
                "label_a": a.label,
                "label_b": b.label,
                "method": method,
                "evidence": edge.source_context,
            })

    if subset_skipped_no_embedder:
        subset_note = (
            f"{subset_skipped_no_embedder} subset-shaped name_form pair(s) "
            f"(e.g. 'new york' / 'new york times') skipped: no embedder "
            f"available to corroborate — co-occurrence alone never draws "
            f"that shape"
        )
        result.note = f"{result.note}; {subset_note}" if result.note else subset_note

    return result

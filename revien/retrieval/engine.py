"""
Revien Retrieval Engine — Query processing, graph walking, scoring, and ranking.
The full retrieval pipeline: parse query → find anchors → walk graph → score → rank.

Scoring layers (in apply order inside recall):
  1. Three-factor base score (recency + frequency + proximity)  [base, stdlib]
  2. Neural adjustment (TF-IDF + LogisticRegression)            [opt-in extra]
  3. Community boost (same-community-as-anchor bonus)           [leg 2]
  4. Confidence multiplier (post-factor, with lazy decay)       [leg 1, base]

Final: final_score = (neural_adjusted_base + community_boost) * effective_confidence

The neural layer is OPT-IN (pip install revien[neural]). Its import is guarded:
when numpy/scikit-learn are absent, neural is silently disabled and every other
layer runs unchanged.
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# One-shot flag for the graph-only degrade warning (per process, not per engine
# — bench runs construct hundreds of engines and one warning is the message).
_SEMANTIC_OFF_WARNED = False

from revien.graph.schema import EdgeType, Node, NodeType, SourceType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from revien.graph.clustering import CommunityDetector
from revien.ingestion.extractor import RuleBasedExtractor
# Neural reranker is opt-in. NeuralScorer/TrainingLoop import cleanly without
# the `neural` extra installed (NeuralScorer self-disables when numpy/sklearn
# are missing; TrainingLoop is pure sqlite3 stdlib).
from revien.neural.scorer_model import NeuralScorer
from revien.neural.training import TrainingLoop
# Semantic/vector layer is opt-in (pip install revien[semantic]). SemanticIndex
# imports cleanly without the extra and self-disables (is_enabled False), so
# recall() runs the unchanged graph path when it is absent or REVIEN_SEMANTIC=0.
from revien.semantic.index import SemanticIndex
from revien.semantic.rerank import CrossEncoderReranker
from revien.skills.ingest import skill_index_row
from revien.skills.proposals import matching_proposals
from .bm25 import bm25_rank
from .scorer import ScoreBreakdown, ScoringConfig, ThreeFactorScorer, _env_float
from .walker import GraphWalker


def _alias_expansion_enabled_by_env() -> bool:
    """ALIAS_OF anchor-expansion gate (alias leg). DEFAULT ON — same
    on-by-default convention as REVIEN_FENCE/REVIEN_RERANK. REVIEN_ALIAS=0
    restores pre-alias anchor selection byte-identically. Read PER CALL
    (like pipeline.py's _fence_enabled_by_env), not cached at construction,
    so a live deployment (or a test) can flip it without rebuilding the
    engine."""
    return os.environ.get("REVIEN_ALIAS", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


def rrf_fuse(
    ranked_lists: List[List[str]],
    k: float = 60.0,
    top_n: Optional[int] = None,
) -> List[str]:
    """Reciprocal Rank Fusion (LEG P1): fuse ranked candidate lists into one.

        rrf_score(item) = Σ_lists 1 / (k + rank_in_list)     # rank is 1-based

    k=60 is the canonical default; smaller k sharpens the head (top ranks
    dominate), larger k flattens toward consensus. Items appearing in several
    lists sum their contributions — that consensus is the whole point. Ties
    break deterministically by item id so a fused anchor set is stable
    across runs.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ids = [item for item, _ in fused]
    return ids[:top_n] if top_n is not None else ids


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """ISO-8601 string for a node timestamp. The speaker's own UTC offset is
    PRESERVED (an aware non-UTC time stays in its offset), so the first ten
    characters are the speaker's calendar day - converting to UTC would turn
    a 9pm-EDT message into the next day. Naive is taken as UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass
class RetrievalResult:
    """A single node result with its score and path."""
    node_id: str
    node_type: str
    label: str
    content: str
    score: float
    score_breakdown: Dict[str, float]
    path: List[str]
    # B1 tension surfacing: LIVE claims this node holds a CONFLICTS_WITH edge
    # to — the other side of a recognized tension. Populated ONLY when
    # recall(include_tensions=True); empty (and cost-free) otherwise.
    tensions: List[Dict[str, str]] = field(default_factory=list)
    # Origin Layer (WS0 Leg B): the node's provenance, carried on EVERY
    # result so a caller (or TOON's uniform-key tabular reshape) never has
    # to special-case an origin-less row. None is a valid, honest value —
    # "unknown provenance" — not an omission; the key is always present.
    origin_runtime: Optional[str] = None
    origin_source: Optional[str] = None
    project_key: Optional[str] = None
    # When the content was SAID (node.recorded_at), ISO-8601 in the speaker's own offset, or None
    # when the node has none. Never falls back to created_at (ingest time):
    # a wrong date is worse than none. Key always present (None allowed) so
    # TOON's uniform-key tabular reshape stays uniform.
    recorded_at: Optional[str] = None
    # Where recorded_at came from: content | capture | import | mtime, or None
    # for rows ingested before the source was recorded. Renderers show the
    # date (and the "when it was said" note) only for content/capture/import.
    recorded_at_source: Optional[str] = None


@dataclass
class RetrievalResponse:
    """Complete response to a retrieval query."""
    query: str
    results: List[RetrievalResult]
    nodes_examined: int
    retrieval_time_ms: float
    neural_active: bool = False
    # Semantic-layer visibility: recall quality differs ~10x between the hybrid
    # and graph-only paths (LoCoMo recall@10 0.47 vs 0.05), so every response
    # says WHICH path served it. semantic_note is None when active, else a
    # one-line reason the layer is off — the caller must be able to see a
    # degrade, not infer it from mysteriously worse results.
    semantic_active: bool = False
    semantic_note: Optional[str] = None
    # Populated only when recall(debug=True): anchor sets, walked distances,
    # per-node final scores, and filter reasons. This is what per-query
    # retrieval failure analysis (extraction/seed/walk/ranking miss) reads.
    diagnostics: Optional[Dict[str, Any]] = None
    # Skills leg D2: draft (LLM-authored, not the rule-based skeleton)
    # engine-origin skill proposals whose steps share a keyword with this
    # query. ALWAYS present (possibly empty) so a caller never has to
    # special-case its absence — same convention as origin_runtime/
    # origin_source/project_key on RetrievalResult. See
    # revien/skills/proposals.py:matching_proposals for the match rule.
    skill_proposals: List[Dict[str, Any]] = field(default_factory=list)


class RetrievalEngine:
    """
    Full retrieval pipeline:
    1. Parse query to extract entities/topics
    2. Find anchor nodes in the graph
    2b. Community-first routing — boost nodes in query-relevant communities
    3. Walk the graph from anchors
    4. Score all reachable nodes (three-factor + neural + community + confidence)
    5. Rank and return top N
    6. Log retrieval for neural training (if neural extra installed)
    """

    # Community membership boost — nodes in the same community as anchors score higher
    COMMUNITY_BOOST = 0.15

    # Semantic-FIRST ranking — when the opt-in semantic layer is enabled, a node
    # the query embeds close to is ranked PRIMARILY by that similarity, and the
    # three-factor graph composite only REFINES it (weight GRAPH_REFINE). This is
    # what makes recall query-discriminative: a genuinely relevant node outranks
    # high-frequency/high-proximity hub nodes, instead of being given a weak
    # additive nudge that the hubs swamp (the v1 bug: same hubs returned for
    # every query). When the layer is disabled this path is inert and the graph
    # composite is used unchanged.
    GRAPH_REFINE = 0.25
    # How many nearest neighbours to pull as semantic anchors/candidates.
    SEMANTIC_TOP_K = 30
    # Similarity floor for treating a node as a semantic match. bge-small scores
    # genuinely-relevant turns in roughly the 0.3-0.6 band, so the old 0.55 floor
    # rejected most real matches and recall fell back to hub-walking. 0.30 admits
    # real matches while excluding pure noise; the candidates are already the
    # top-K nearest, so relative rank — not an absolute threshold — carries the
    # signal.
    #
    # SCALE HONESTY (retrieval-door leg): against search()'s 1/(1+distance)
    # similarity (min 1/3) a 0.30 floor could never bind — dead code from the
    # day it shipped; every sweep of it measured noise. The head-admission
    # check keeps that (inert) comparison for byte-compatibility; the
    # over-fetch TAIL gate applies this floor on the RAW COSINE scale
    # (cosine = 2 - 1/sim), where 0.30 actually gates what may union into
    # the candidate set. See the 2a block in recall().
    SEMANTIC_SIM_FLOOR = 0.30
    # RRF fusion constant (LEG P1, active only under REVIEN_HYBRID=rrf).
    # Canonical default 60; REVIEN_RRF_K overrides for the sweep.
    RRF_K = 60.0
    # Hard ceiling on recall(top_n=...). Requests above it are capped LOUDLY
    # (stderr) — the old silent min(top_n, 20) made rank-depth evaluations
    # measure the clamp instead of the ranking.
    TOP_N_MAX = 200
    # Vector-union over-fetch factor (retrieval-door leg, see the 2a block in
    # recall). The vector fetch pulls factor*top_k, keeps the original top-k
    # admission unchanged, and extends with non-CONTEXT hits until top_k
    # returnable nodes are in. 1.0 = exact pre-leg behavior (kill switch for
    # A/B). Override: REVIEN_VECTOR_OVERFETCH. Fetch geometry stays sweepable
    # pending the server-side k-sweep verdict.
    VECTOR_OVERFETCH = 4.0
    # prefer_types recall hint: final-score multiplier
    # for nodes whose type the CALLER asked to favor for THIS query ("what
    # does this reader avoid" -> prefer_types=["preference"]). A soft boost,
    # not a filter — other types still surface, just outranked when preferred
    # ones compete. Inert when recall() gets no prefer_types. 1.5 is a
    # starting point, not a benched value; REVIEN_PREFER_BOOST for the sweep.
    PREFER_BOOST = 1.5

    def __init__(
        self,
        store: GraphStore,
        scoring_config: Optional[ScoringConfig] = None,
        max_depth: int = 3,
        clustering: Optional[CommunityDetector] = None,
        model_dir: Optional[str] = None,
        training_db: Optional[str] = None,
        semantic: Optional[SemanticIndex] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.store = store
        self.ops = GraphOperations(store)
        self.extractor = RuleBasedExtractor()
        # No explicit config -> defaults + env overrides (ScoringConfig.from_env;
        # unset env == exact defaults). This is what lets the bench sweep ranking
        # knobs without code edits.
        self.scorer = ThreeFactorScorer(scoring_config or ScoringConfig.from_env())
        # Weighted walk (A1): edge strength = weight, optionally * edge
        # confidence. The strength only moves scores when the scorer's
        # edge_weight_blend knob (REVIEN_EDGE_WEIGHT_BLEND) is > 0.
        self.walker = GraphWalker(
            store,
            max_depth=max_depth,
            use_edge_confidence=os.environ.get(
                "REVIEN_EDGE_CONFIDENCE_IN_WALK", "0"
            ).strip().lower() in ("1", "true", "yes", "on"),
        )
        self.max_depth = max_depth
        self.clustering = clustering

        # Ranking knobs (see class constants for semantics). Env-overridable for
        # sweeps; the class constants remain the shipped defaults. The miss
        # taxonomy says 72% of semantic-path misses are `outranked` — these are
        # the levers that decide that ranking.
        self.semantic_top_k = int(_env_float("REVIEN_SEMANTIC_TOP_K", self.SEMANTIC_TOP_K))
        self.semantic_sim_floor = _env_float(
            "REVIEN_SEMANTIC_SIM_FLOOR", self.SEMANTIC_SIM_FLOOR
        )
        self.graph_refine = _env_float("REVIEN_GRAPH_REFINE", self.GRAPH_REFINE)
        self.community_boost = _env_float("REVIEN_COMMUNITY_BOOST", self.COMMUNITY_BOOST)
        self.prefer_boost = _env_float("REVIEN_PREFER_BOOST", self.PREFER_BOOST)
        # < 1.0 makes no sense (can't under-fetch the head); clamp to the
        # kill-switch value instead of crashing on a bad experiment knob.
        self.vector_overfetch = max(
            1.0, _env_float("REVIEN_VECTOR_OVERFETCH", self.VECTOR_OVERFETCH)
        )
        # LEG P1 — RRF hybrid fusion gate (EXPERIMENTAL, default off). When
        # REVIEN_HYBRID=rrf, anchor selection is replaced by Reciprocal Rank
        # Fusion of the keyword-ranked and semantic-ranked candidate lists;
        # everything downstream (walk, scoring, rerank) is untouched. Unset
        # or any other value: the shipped anchor path runs byte-identical.
        self.hybrid_mode = os.environ.get("REVIEN_HYBRID", "").strip().lower()
        # BM25 lexical lane (production-validated overlay, see revien/retrieval
        # /bm25.py header — recall@10 0.5814 -> 0.6395 measured under
        # REVIEN_HYBRID=rrf). Read once at init, same convention as
        # hybrid_mode: unset/any-other-value keeps the shipped substring
        # keyword lane byte-identical; REVIEN_LEXICAL=bm25 swaps the ranking
        # math both _lexical_candidates call sites (RRF fusion list AND the
        # keyword-fallback anchor path) resolve to.
        requested_lexical = os.environ.get("REVIEN_LEXICAL", "").strip().lower()
        self.lexical_mode = "bm25" if requested_lexical == "bm25" else "keyword"
        # WHY: the RRF fusion call site (below, in recall()) capped the
        # lexical candidate list at ``self.semantic_top_k`` — but the
        # production overlay this lane was validated against (bm25.py's
        # header, 0.5814 -> 0.6395) ran the lexical list UNCAPPED
        # (limit=None). That's a capped-vs-uncapped confound: any recall
        # delta measured between the keyword and BM25 lanes under RRF was
        # never isolating the ranking math alone, it was also comparing a
        # capped list against what the overlay actually ran. This knob lets
        # a sweep pick apart the two effects independently.
        # REVIEN_LEXICAL_LIMIT unset -> None -> the RRF call site uses
        # self.semantic_top_k exactly as before (byte-identical). Set to a
        # positive int -> that value overrides the cap. Set to "0" ->
        # uncapped (None is passed through to _lexical_candidates, which
        # already treats None as "no cap" on both lanes). Same
        # never-raise contract as REVIEN_RRF_K just above: malformed values
        # fall back to the default (None / semantic_top_k) silently — a bad
        # experiment knob must never crash recall.
        _raw_lexical_limit = os.environ.get("REVIEN_LEXICAL_LIMIT")
        self.lexical_limit_override: Optional[int] = None
        if _raw_lexical_limit is not None:
            try:
                _parsed_lexical_limit = int(_raw_lexical_limit.strip())
            except (ValueError, AttributeError):
                _parsed_lexical_limit = None
            if _parsed_lexical_limit is not None and _parsed_lexical_limit >= 0:
                # 0 means "uncapped" downstream (None); store the sentinel
                # value itself and resolve it at the RRF call site so the
                # "unset" (None-override) and "0" (uncapped) states stay
                # distinguishable here.
                self.lexical_limit_override = _parsed_lexical_limit
        # A bad experiment knob must never crash recall (scorer.py's own
        # convention, _env_float's docstring) — this runs unconditionally,
        # even when REVIEN_HYBRID isn't "rrf", so a malformed or zero
        # REVIEN_RRF_K silently falls back to the shipped default instead of
        # raising, same as every other env-float knob in this class.
        _requested_rrf_k = _env_float("REVIEN_RRF_K", self.RRF_K)
        self.rrf_k = (
            _requested_rrf_k
            if math.isfinite(_requested_rrf_k) and _requested_rrf_k > 0
            else self.RRF_K
        )
        # Frequency feedback-loop gate — DEFAULT OFF (sweep-shipped July 2026):
        # recall() touching its own results made access_count a popularity
        # prior contaminated by the engine's own behavior (being returned →
        # higher frequency → returned again), measured at -21% recall@10 vs
        # honest frequency at full scale. Only mark_used() — a caller
        # confirming the memory was actually useful — feeds access_count.
        # Set REVIEN_TOUCH_ON_RECALL=1 to restore the old behavior.
        self.touch_on_recall = os.environ.get(
            "REVIEN_TOUCH_ON_RECALL", "0"
        ).strip().lower() in ("1", "true", "yes", "on")

        # Neural components — opt-in by EXPLICIT REQUEST, not by dependency
        # presence. "Opt-in = the extra is installed" was a landmine: any env
        # that happened to have sklearn (most do) silently activated score
        # adjustment from a GLOBAL per-machine model (~/.revien/models) that
        # the training loop retrains on whatever traffic the box has seen —
        # cross-tenant reranking of per-user stores, measured at -20pts
        # generic recall on the per-user conversational bench (0.7 with the
        # ambient model vs 0.9 without; the model had been retrained by the
        # bench runs themselves). Opt in with REVIEN_NEURAL=1, or implicitly
        # by passing model_dir/training_db (explicit args ARE intent). Off:
        # scoring is pure pass-through and NOTHING trains or writes models.
        self.neural_enabled = (
            model_dir is not None
            or training_db is not None
            or os.environ.get("REVIEN_NEURAL", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self.neural_scorer = NeuralScorer(model_dir=model_dir)
        self.training_loop = TrainingLoop(db_path=training_db, model_dir=model_dir)

        # Semantic/vector layer (SPINE — core deps as of the promote-to-spine
        # change). Constructs cleanly even when the deps are absent (source
        # install): SemanticIndex.is_enabled stays False and all of its methods
        # no-op, so recall() degrades to the graph-only path — but LOUDLY:
        self.semantic = semantic if semantic is not None else SemanticIndex(store)
        self._warn_if_semantic_inactive()

        # Cross-encoder head reranker (A1). DEFAULT ON since July 11 2026
        # ("smarter by default"): int8 model, depth 20 — measured lever for
        # the `outranked` bucket: with semantic-as-spine the top of the
        # ranking is all distance-0 anchors, and only a model that reads
        # query+candidate together can reorder them. REVIEN_RERANK=0 opts
        # out (pre-rerank path, byte-identical); degrades loudly on runtime
        # failure.
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()

    def _warn_if_semantic_inactive(self) -> None:
        """One warning per process when recall will run graph-only. Graph-only
        recall has no query-relevance signal beyond keyword overlap (LoCoMo
        recall@10: 0.05 vs 0.47 hybrid) — an engine silently running in that
        mode is the bug this warning exists to catch."""
        global _SEMANTIC_OFF_WARNED
        if self.semantic.is_enabled or _SEMANTIC_OFF_WARNED:
            return
        _SEMANTIC_OFF_WARNED = True
        sys.stderr.write(
            f"[revien] recall is running GRAPH-ONLY (keyword) retrieval - "
            f"semantic layer inactive: {self.semantic.inactive_reason()}. "
            f"Recall quality is significantly degraded. Set "
            f"REVIEN_SEMANTIC=require to make this fatal.\n"
        )
        sys.stderr.flush()

    def recall(
        self,
        query: str,
        top_n: int = 5,
        min_score: float = 0.01,
        now: Optional[datetime] = None,
        include_invalidated: bool = False,
        include_context: bool = False,
        include_tensions: bool = False,
        as_of: Optional[datetime] = None,
        debug: bool = False,
        prefer_types: Optional[List[str]] = None,
        source: Optional[Union[str, List[str]]] = None,
    ) -> RetrievalResponse:
        """
        Query the memory graph and return ranked results.

        Args:
            query: Natural language query
            top_n: Maximum results to return (default 5). Capped at
                TOP_N_MAX (200) with a stderr warning — never silently.
            min_score: Minimum composite score threshold
            now: Current time for recency scoring (defaults to UTC now)
            include_invalidated: When False (default), soft-invalidated nodes
                (invalidated_at set) are excluded from results. Set True to
                surface them. Provenance is non-destructive — invalidated nodes
                are retained and recoverable, just hidden from default recall.
                When no node is invalidated this flag changes nothing, so recall
                is byte-identical to the pre-6a behavior.
            include_tensions: When True, each result carries the live other
                side of any CONFLICTS_WITH edge it holds (B1 tension
                surfacing) in ``result.tensions``. Default False: zero extra
                queries, response byte-identical.
            as_of: Bi-temporal query time (B2) — "what was true AT this
                time?". Nodes whose validity window excludes as_of are
                filtered; a SUPERSEDED (invalidated) node whose closed window
                COVERS as_of is deliberately included — recovering the old
                truth is the point ("where did she live in March?").
                Invalidated nodes with no window stay hidden (their validity
                is unknown, not historical). When `now` is not given, recency
                scores relative to as_of, so ranking is coherent with the
                queried moment. Default None: byte-identical behavior.
            debug: When True, the response carries a ``diagnostics`` dict
                (anchor sets by origin, walked node distances, per-node final
                scores including sub-threshold ones, and filter reasons) so a
                caller can classify WHY a given node was or wasn't returned.
                Default False: zero overhead, response unchanged.
            prefer_types: Per-query type hint — node types (e.g.
                ["preference"]) whose final scores are multiplied by the
                prefer-boost knob (PREFER_BOOST / REVIEN_PREFER_BOOST) so a
                caller can express intent WITHOUT hard filtering: other types
                still surface, preferred ones win contested rankings. Unknown
                type strings are ignored. Composes with (multiplies on top
                of) the config-level type_weights prior, which applies to
                every query regardless of this hint. Default None: no-op,
                response byte-identical.
            source: Origin Layer (WS0 Leg B) filter — a single origin_runtime
                string (e.g. "claude-code") or a list of them. Applied as a
                SQL prefilter at every candidate/anchor source (entity-label
                lookup, keyword/BM25 lexical search) AND as a final gate on
                every walked node, so a node reached only through graph-walk
                expansion from an in-filter anchor can never leak a
                different runtime into the results. A node with
                origin_runtime None (unknown provenance) never matches a
                source filter. Default None: no filtering, response
                byte-identical.

        Returns:
            RetrievalResponse with ranked nodes and timing data
        """
        start_time = time.perf_counter()
        # Honest bound, not a silent lie: the old min(top_n, 20) quietly
        # returned 20 to a caller asking for 30 — an evaluation probing rank
        # depth measured the clamp, not the ranking. 200 bounds pathological
        # asks; anything above it is logged so the caller can see the cap.
        if top_n > self.TOP_N_MAX:
            sys.stderr.write(
                f"[revien] recall top_n={top_n} capped to {self.TOP_N_MAX}\n"
            )
            top_n = self.TOP_N_MAX

        if now is None:
            # An as_of query ranks relative to the queried moment — recency
            # scored from wall-clock now would bury the era being asked about.
            now = as_of if as_of is not None else datetime.now(timezone.utc)

        # Normalized once; empty/None means the hint is inert.
        preferred_types = (
            {t.strip().lower() for t in prefer_types} if prefer_types else None
        )
        # Origin Layer (WS0 Leg B) source filter, normalized once (G4):
        # None = unfiltered; any non-None value (including empty) = a
        # filter is present, and empty matches nothing.
        source_filter: Optional[List[str]] = None
        if source is not None:
            _raw = [source] if isinstance(source, str) else list(source)
            source_filter = [s for s in _raw if s]
        if as_of is not None and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        entity_anchor_ids: List[str] = []
        keyword_anchor_ids: List[str] = []
        semantic_sims: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        if self.hybrid_mode == "rrf":
            # LEG P1 — RRF hybrid fusion (REVIEN_HYBRID=rrf, experimental).
            # Fuses keyword/BM25-ranked and semantic-ranked candidate lists;
            # keyword search runs ALWAYS (not as a fallback) as one ranked
            # list, the semantic top-K (same floor as the shipped path) is
            # the other, and the RRF-fused top-N seeds the anchor set. The
            # walk + scoring + rerank pipeline downstream is unchanged.
            #
            # ENTITY-ANCHOR UNION (P1 regression receipt): earlier the fused
            # list REPLACED entity anchors wholesale, which measured +65
            # disconnected results on the eval — entity anchors had stopped
            # seeding the walk at all, so any node reachable only through an
            # entity match (and, downstream, its alias-expanded neighbors —
            # _find_anchors already unions ALIAS_OF neighbors onto entity
            # anchors under REVIEN_ALIAS) became silently unreachable. Entity
            # anchors are found the same way the shipped path finds them
            # (with alias expansion intact) and PREPENDED onto the fused set
            # below — this correction is unconditional (no separate flag):
            # it fixes a known regression, not an experiment of its own.
            #
            # ORDER IS A REAL EFFECT, not incidental: entity anchors go at
            # the FRONT (``entity_anchor_ids + fused_ids``, de-duped keeping
            # first occurrence), not appended. This changes
            # ``diagnostics["anchors"]["all"]`` ordering versus a plain
            # RRF-only list, and — because the walker seeds every anchor at
            # distance 0 with its own path entry — can flip which anchor's
            # label appears first in a shared result's ``path`` when the
            # SAME node is reachable from both an entity anchor and a fused
            # one. Entity anchors lead because they are the higher-precision
            # signal (exact/fuzzy label match, optionally alias-expanded)
            # versus the fused list's rank-fusion heuristic — ties should
            # read as "the entity match owns this," not as whichever list
            # rrf_fuse happened to place first.
            entity_anchor_ids = self._find_anchors(query, source_filter)
            # REVIEN_LEXICAL_LIMIT resolution: unset (override is None) keeps
            # the shipped cap (semantic_top_k) byte-identical; 0 resolves to
            # None here (uncapped, both lanes treat None that way); a
            # positive int overrides the cap directly.
            if self.lexical_limit_override is None:
                _lexical_limit = self.semantic_top_k
            elif self.lexical_limit_override == 0:
                _lexical_limit = None
            else:
                _lexical_limit = self.lexical_limit_override
            keyword_anchor_ids, bm25_scores = self._lexical_candidates(
                query, limit=_lexical_limit, source_filter=source_filter,
            )
            semantic_ranked: List[str] = []
            if self.semantic.is_enabled:
                sem_hits = [
                    (node_id, sim)
                    for node_id, sim in self.semantic.search(
                        query, top_k=self.semantic_top_k
                    )
                    if sim >= self.semantic_sim_floor
                ]
                # Semantic index carries no origin fields of its own (WS0 Leg
                # B) — filter its hits against the store so a filtered
                # recall's semantic lane can't anchor on another runtime.
                allowed = self._origin_allowed_ids(
                    [nid for nid, _ in sem_hits], source_filter
                )
                for node_id, sim in sem_hits:
                    if node_id not in allowed:
                        continue
                    semantic_sims[node_id] = sim
                    semantic_ranked.append(node_id)
            fused_ids = rrf_fuse(
                [keyword_anchor_ids, semantic_ranked],
                k=self.rrf_k,
                top_n=self.semantic_top_k,
            )
            anchor_ids = list(dict.fromkeys(entity_anchor_ids + fused_ids))
        else:
            # 1. Parse query — extract entities and topics
            entity_anchor_ids = self._find_anchors(query, source_filter)
            anchor_ids = list(entity_anchor_ids)

            # 2. If no anchors found, try keyword search across all nodes
            # (or BM25-ranked candidates under REVIEN_LEXICAL=bm25).
            if not anchor_ids:
                keyword_anchor_ids, bm25_scores = self._lexical_candidates(
                    query, source_filter=source_filter,
                )
                anchor_ids = list(keyword_anchor_ids)

            # 2a. Hybrid semantic anchors (opt-in). When the semantic layer is
            # enabled, embed the query, pull the nearest stored nodes, and UNION
            # them into the anchor set. This is what lets a keyword-less query
            # (no entity/keyword overlap with any node) still find relevant nodes.
            # When the layer is disabled, semantic_sims is empty and the anchor set
            # is byte-for-byte what it was before this leg.
            #
            # RETRIEVAL DOOR (vector-union leg, mirrors server patch-H): the
            # plain top-k fetch burned most of its budget on CONTEXT nodes
            # that result assembly then discards (include_context=False, the
            # default) — so an edge-poor preference node OUTSIDE the vector
            # top-k was unreachable at any walk depth, while the specific
            # query (which embeds right next to it) found it fine. Fix:
            # over-fetch (REVIEN_VECTOR_OVERFETCH, 1.0 = old behavior
            # byte-identical), keep the original top-k admission UNCHANGED
            # (CONTEXT hits still seed walks to their extracted neighbors),
            # then extend with non-CONTEXT hits from the over-fetch tail
            # until top-k non-CONTEXT nodes are unioned in. Tail admissions
            # are gated by SEMANTIC_SIM_FLOOR on the RAW COSINE scale — the
            # floor's original intent; on the 1/(1+distance) scale it could
            # never bind (min 1/3 > 0.30, dead code since the leg shipped).
            # cosine = 2 - 1/sim inverts the search() similarity shape.
            if self.semantic.is_enabled:
                fetch_k = self.semantic_top_k
                overfetch = self.vector_overfetch if not include_context else 1.0
                if overfetch > 1.0:
                    fetch_k = int(self.semantic_top_k * overfetch)
                hits = self.semantic.search(query, top_k=fetch_k)
                head = hits[: self.semantic_top_k]
                tail = hits[self.semantic_top_k:]
                # Semantic index carries no origin fields (WS0 Leg B) — a
                # source filter needs the store's own view of these hits, so
                # fetch it whenever filtering OR the tail is non-empty (the
                # tail admission already needed node_type from this bulk
                # fetch pre-Leg-B).
                hit_nodes = (
                    self.store.get_nodes_bulk([nid for nid, _ in hits])
                    if (tail or source_filter) else {}
                )
                # Same origin-filter membership test the RRF lane uses
                # (_origin_allowed_ids) — reuses hit_nodes instead of a
                # second bulk fetch when a filter is active.
                allowed = self._origin_allowed_ids(
                    [nid for nid, _ in hits], source_filter, nodes=hit_nodes
                )
                for node_id, sim in head:
                    # Only nodes that clear the floor act as semantic anchors. This
                    # is what lets a keyword-less query reach a genuinely-close node
                    # without near-uniform mild similarity reshuffling keyword hits.
                    if sim < self.semantic_sim_floor:
                        continue
                    if source_filter is not None and node_id not in allowed:
                        continue
                    semantic_sims[node_id] = sim
                    if node_id not in anchor_ids:
                        anchor_ids.append(node_id)
                if tail:
                    non_ctx_admitted = sum(
                        1 for nid, _ in head
                        if nid in semantic_sims
                        and nid in hit_nodes
                        and hit_nodes[nid].node_type != NodeType.CONTEXT
                    )
                    for node_id, sim in tail:
                        if non_ctx_admitted >= self.semantic_top_k:
                            break
                        node = hit_nodes.get(node_id)
                        if node is None or node.node_type == NodeType.CONTEXT:
                            continue
                        if source_filter is not None and node_id not in allowed:
                            continue
                        # Binding gate: raw cosine, not the 1/(1+d) shape.
                        if (2.0 - 1.0 / sim) < self.semantic_sim_floor:
                            continue
                        semantic_sims[node_id] = sim
                        if node_id not in anchor_ids:
                            anchor_ids.append(node_id)
                        non_ctx_admitted += 1

        # 2b. Community-first routing — identify which communities are relevant
        relevant_communities: set = set()
        if anchor_ids and self.clustering and self.clustering.is_clustered:
            relevant_communities = set(
                self.clustering.get_communities_for_anchors(anchor_ids)
            )

        # 3. Walk graph from anchors — ONE traversal yields distances, paths,
        # and path strengths (this used to be two full BFS passes per recall).
        if anchor_ids:
            node_distances, node_paths, node_strengths = self.walker.walk_full(
                anchor_ids
            )
        else:
            node_distances = {}
            node_paths = {}
            node_strengths = {}

        # 4. Score all reachable nodes. One bulk fetch for the whole walked
        # frontier — a SELECT per node here was a measured latency driver.
        scored_results: List[RetrievalResult] = []
        nodes_examined = len(node_distances)
        nodes_map = self.store.get_nodes_bulk(node_distances.keys())
        # Debug bookkeeping (leg: per-query failure analysis). Cheap dicts,
        # built only when asked for.
        diag_scores: Dict[str, float] = {}
        diag_filtered: Dict[str, str] = {}

        for node_id, distance in node_distances.items():
            node = nodes_map.get(node_id)
            if node is None:
                if debug:
                    diag_filtered[node_id] = "missing"
                continue

            # CONTEXT nodes are the verbatim turns — answer-bearing content for
            # conversational memory. Surface them by default; callers wanting only
            # distilled extract nodes can pass include_context=False.
            if node.node_type == NodeType.CONTEXT and not include_context:
                if debug:
                    diag_filtered[node_id] = "context_excluded"
                continue

            # Origin Layer (WS0 Leg B): the FINAL gate, re-checked here on
            # every walked node regardless of how it entered node_distances
            # (entity/keyword/BM25/semantic anchor, alias expansion, OR
            # graph-walk expansion off any of those). This is what makes
            # "filtered recall never leaks other runtimes" true even though
            # walk expansion can reach a neighbor with different provenance
            # than the anchor that seeded it — the SQL prefilters above keep
            # candidate/anchor sources cheap, this is what keeps the
            # guarantee. A node with origin_runtime None never matches.
            if source_filter is not None and node.origin_runtime not in source_filter:
                if debug:
                    diag_filtered[node_id] = "origin_filtered"
                continue

            # Bi-temporal filter (B2): with as_of set, validity windows decide.
            # A closed-window SUPERSEDED node covering as_of comes BACK — that
            # recovered old truth is what an as-of query exists for.
            if as_of is not None:
                vf, vu = node.valid_from, node.valid_until
                if vf is not None and vf.tzinfo is None:
                    vf = vf.replace(tzinfo=timezone.utc)
                if vu is not None and vu.tzinfo is None:
                    vu = vu.replace(tzinfo=timezone.utc)
                if vf is not None and vf > as_of:
                    if debug:
                        diag_filtered[node_id] = "not_yet_valid"
                    continue
                if vu is not None and vu <= as_of:
                    if debug:
                        diag_filtered[node_id] = "no_longer_valid"
                    continue
                # Invalidated with NO closed window: validity unknown, not
                # historical — keep the provenance default unless overridden.
                if (node.invalidated_at is not None and vu is None
                        and not include_invalidated):
                    if debug:
                        diag_filtered[node_id] = "invalidated"
                    continue
            # Provenance (leg 6a): exclude soft-invalidated nodes by default.
            # No-op when nothing is invalidated, so recall stays byte-identical.
            elif node.invalidated_at is not None and not include_invalidated:
                if debug:
                    diag_filtered[node_id] = "invalidated"
                continue

            # Three-factor base score (recency + frequency + proximity).
            # Recency scores CONTENT time — when the memory was said
            # (recorded_at), falling back to when it entered the graph
            # (created_at). Scoring last_accessed here made "recency" mean
            # recently-touched: it correlated with retrieval popularity, was
            # constant in any evaluation querying at a historical `now`, and
            # buried old-but-true facts behind whatever was returned last.
            breakdown = self.scorer.score(
                timestamp=node.recorded_at or node.created_at,
                access_count=node.access_count,
                graph_distance=distance,
                now=now,
                path_strength=node_strengths.get(node_id, 1.0),
                node_type=node.node_type.value,
            )

            # Neural adjustment (opt-in, REVIEN_NEURAL / explicit model_dir).
            # Gated HERE, not just inside adjust_score: an ambient trained
            # model on the machine must not rerank without explicit opt-in.
            if self.neural_enabled:
                base_score = self.neural_scorer.adjust_score(
                    base_score=breakdown.composite,
                    node_label=node.label,
                    query=query,
                )
            else:
                base_score = breakdown.composite

            # Community boost — nodes in same community as anchors get a boost
            community_boost = 0.0
            if relevant_communities and self.clustering:
                node_community = self.clustering.get_community(node_id)
                if node_community is not None and node_community in relevant_communities:
                    community_boost = self.community_boost

            # Layer 1 (leg 1): confidence multiplier as a post-factor.
            effective_confidence = self._effective_confidence(node, now)

            # Semantic-FIRST ranking (opt-in). When the semantic layer surfaced
            # this node for the query, similarity is the PRIMARY term and the
            # graph composite only refines it (GRAPH_REFINE) — so query-relevant
            # nodes outrank frequency/proximity hubs. semantic_sims is empty when
            # the layer is disabled, so sim is None and the graph-only expression
            # below runs unchanged (byte-identical to the pre-semantic path).
            #
            # BM25 composes the SAME way (bm25_scores is empty unless
            # REVIEN_LEXICAL=bm25): it's another query-relevance signal, not a
            # separate term, so a node with both takes the MAX rather than
            # double-counting overlapping evidence — mirrors the production
            # overlay's contract (bm25.py header; the overlay blended it at
            # this exact seam).
            sim = semantic_sims.get(node_id)
            bm25_score = bm25_scores.get(node_id)
            if sim is not None or bm25_score is not None:
                query_relevance = max(
                    score for score in (sim, bm25_score) if score is not None
                )
                final_score = (
                    query_relevance + self.graph_refine * base_score + community_boost
                ) * effective_confidence
            else:
                final_score = (base_score + community_boost) * effective_confidence

            # Node-type prior: applied HERE, on the final score, so
            # the prior shapes ranking identically on the semantic-first and
            # graph-only paths — multiplying the graph composite instead would
            # scale it by graph_refine on the semantic path and dilute it
            # exactly where the type-blind misses were measured. Unmapped
            # types weigh 1.0, so an empty map is byte-identical.
            type_weight = self.scorer.config.type_weight(node.node_type.value)
            final_score *= type_weight

            # Per-query prefer_types hint: soft boost, composes with the
            # config prior above. None/empty = inert.
            prefer_boosted = (
                preferred_types is not None
                and node.node_type.value in preferred_types
            )
            if prefer_boosted:
                final_score *= self.prefer_boost

            if debug:
                diag_scores[node_id] = final_score

            if final_score < min_score:
                continue

            # Build path labels (path nodes are all within the walked set,
            # so the bulk map already has them).
            #
            # G2 (origin filter must cover path labels too): a hop's own
            # label is provenance-bearing content just like a result's
            # label/content — a query filtered to "codex" that walks
            # anchor(codex) -> mid(claude-code) -> leaf(codex) must not
            # spell "mid"'s claude-code label into the response merely
            # because the FINAL node on that path is in-filter. When a
            # source filter is active, any path node outside the filter
            # (None origin included, same rule as everywhere else) is
            # replaced by the literal "[filtered]" placeholder — never the
            # real label — and the list keeps its original length so hop
            # counts (len(path)) stay honest.
            path_ids = node_paths.get(node_id, [node_id])
            path_labels = []
            for pid in path_ids:
                pnode = nodes_map.get(pid)
                if pnode is None:
                    continue
                if (source_filter is not None
                        and pnode.origin_runtime not in source_filter):
                    path_labels.append("[filtered]")
                else:
                    path_labels.append(pnode.label)

            score_breakdown = {
                "recency": breakdown.recency,
                "frequency": breakdown.frequency,
                "proximity": breakdown.proximity,
                "base_composite": breakdown.composite,
                "community_boost": community_boost,
                "effective_confidence": effective_confidence,
                "neural_adjusted": (
                    self.neural_enabled and self.neural_scorer.is_neural
                ),
            }
            # Only surface the semantic component when the opt-in layer is
            # enabled, so the disabled-path breakdown is byte-for-byte unchanged.
            if self.semantic.is_enabled:
                score_breakdown["semantic_sim"] = sim if sim is not None else 0.0
            # Same rule for BM25: only surfaced when the lane is actually
            # selected, so the keyword-lane breakdown stays byte-identical.
            if self.lexical_mode == "bm25":
                score_breakdown["bm25_score"] = (
                    bm25_score if bm25_score is not None else 0.0
                )
            # Same rule for path strength: only in the breakdown when the
            # weighted-walk blend is actually shaping the score.
            if self.scorer.config.edge_weight_blend > 0.0:
                score_breakdown["path_strength"] = round(
                    node_strengths.get(node_id, 1.0), 4
                )
            # Same rule for the type prior and prefer hint: keys appear only
            # when the multiplier actually moved this node's score.
            if type_weight != 1.0:
                score_breakdown["type_weight"] = type_weight
            if prefer_boosted:
                score_breakdown["prefer_boost"] = self.prefer_boost

            scored_results.append(RetrievalResult(
                node_id=node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                # D1 leftover: a SKILL result's `content` is the index-row
                # text ("<description> — triggers: a, b"), never the full
                # body — `skills show`/`revien skills show` is what returns
                # the body. Every other node type is unchanged.
                content=(
                    skill_index_row(node)
                    if node.node_type == NodeType.SKILL
                    else node.content
                ),
                score=final_score,
                score_breakdown=score_breakdown,
                path=path_labels,
                # Origin Layer (WS0 Leg B): populated on EVERY result, None
                # allowed — the key is always present so TOON's uniform-key
                # tabular reshape never has to special-case an origin-less row.
                origin_runtime=node.origin_runtime,
                origin_source=node.origin_source,
                project_key=node.project_key,
                recorded_at=_iso_utc(node.recorded_at),
                recorded_at_source=(node.metadata or {}).get("recorded_at_source"),
            ))

        # 5. Rank by final score, then (opt-in) cross-encoder rerank of the
        # HEAD — before the top_n slice, so a gold node at base rank ~26 can
        # be pulled into a top-10 (the outranked bucket's median lives there).
        scored_results.sort(key=lambda r: r.score, reverse=True)
        if self.reranker.is_enabled:
            scored_results = self.reranker.rerank(query, scored_results)
        top_results = scored_results[:top_n]

        # 5b. B1 tension surfacing (opt-in): attach the LIVE other side of any
        # CONFLICTS_WITH edge a returned claim carries, so a caller sees "she
        # said this — AND holds this opposing pull" in one response. Flag-off
        # is a no-op: zero queries, response byte-identical.
        if include_tensions and top_results:
            self._attach_tensions(top_results, source_filter)

        # 6. Touch retrieved nodes (update access tracking). Env-gated
        # (REVIEN_TOUCH_ON_RECALL=0 disables) because this is the frequency
        # feedback loop: being RETURNED bumps access_count, which raises the
        # frequency score, which gets the node returned again — retrieval
        # popularity masquerading as relevance. With the gate off, only
        # mark_used() (a caller confirming the memory was actually useful)
        # feeds the frequency signal. Default ON (shipped behavior unchanged)
        # pending the sweep verdict.
        if self.touch_on_recall:
            for result in top_results:
                self.ops.touch_node(result.node_id)

        # 7. Log retrieval for neural training — ONLY under explicit neural
        # opt-in. Unconditional "signals accumulate for later" is how the
        # ambient global model got trained in the first place.
        if self.neural_enabled:
            self.training_loop.log_retrieval(
                query=query,
                results=[
                    {
                        "node_id": r.node_id,
                        "label": r.label,
                        "node_type": r.node_type,
                        "score": r.score,
                    }
                    for r in top_results
                ],
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        diagnostics: Optional[Dict[str, Any]] = None
        if debug:
            diagnostics = {
                "anchors": {
                    "entity": entity_anchor_ids,
                    "keyword": keyword_anchor_ids,
                    "semantic": list(semantic_sims.keys()),
                    "all": list(anchor_ids),
                },
                "lexical_mode": self.lexical_mode,
                "bm25_scores": dict(bm25_scores),
                "node_distances": dict(node_distances),
                "scores": diag_scores,
                "filtered": diag_filtered,
                "max_depth": self.max_depth,
            }

        return RetrievalResponse(
            query=query,
            results=top_results,
            nodes_examined=nodes_examined,
            retrieval_time_ms=round(elapsed_ms, 2),
            neural_active=self.neural_enabled and self.neural_scorer.is_neural,
            semantic_active=self.semantic.is_enabled,
            # Active layer: note deferred-capture state (drained N at search
            # time / M still pending) when there is any — None otherwise, so
            # the response shape is unchanged for the common case.
            semantic_note=(
                self.semantic.inactive_reason()
                or "; ".join(n for n in (
                    getattr(self.semantic, "warnings_note", lambda: None)(),
                    self.semantic.pending_note()) if n) or None
            ),
            diagnostics=diagnostics,
            # Skills leg D2: always computed, always present (possibly
            # empty). G9: cheap because matching_proposals prefilters in
            # SQL (metadata LIKE on status/draft) before any Python-side
            # scan — NOT because "SKILL-typed nodes only" is inherently a
            # small set (a graph can carry thousands of accepted/declined
            # SKILL nodes that were never draft proposals; the old comment
            # here was wrong about why this was cheap). G3: source_filter
            # is the SAME normalized filter recall() applies everywhere
            # else — a filtered recall's skill_proposals never surfaces a
            # proposal derived from a foreign runtime's ACTION nodes.
            skill_proposals=matching_proposals(self.store, query, source_filter),
        )

    def _attach_tensions(
        self,
        results: List[RetrievalResult],
        source_filter: Optional[List[str]] = None,
    ) -> None:
        """Populate each result's `tensions` with the live counterpart of any
        CONFLICTS_WITH edge it carries. One edges query + one bulk node fetch
        per result that HAS such edges — tension edges are rare, so the
        common case is the single get_edges_for_node lookup.

        G1 (origin filter must cover tensions too): a tension partner is
        foreign content riding in on a result that's IN the filter — the
        result itself passed the gate at step 4, but its CONFLICTS_WITH
        counterpart never did, because tensions are attached AFTER that
        gate. When a source filter is active, a counterpart whose
        origin_runtime is not IN the filter is dropped — same rule as
        everywhere else, including a counterpart with origin_runtime None
        (unknown provenance never matches an active filter)."""
        for result in results:
            counterpart_ids = []
            for edge in self.store.get_edges_for_node(result.node_id):
                if edge.edge_type is not EdgeType.CONFLICTS_WITH:
                    continue
                other = (edge.target_node_id
                         if edge.source_node_id == result.node_id
                         else edge.source_node_id)
                counterpart_ids.append((other, edge.source_context))
            if not counterpart_ids:
                continue
            nodes = self.store.get_nodes_bulk([nid for nid, _ in counterpart_ids])
            for nid, context in counterpart_ids:
                node = nodes.get(nid)
                if node is None or node.invalidated_at is not None:
                    continue  # a superseded claim's tension is history
                if (source_filter is not None
                        and node.origin_runtime not in source_filter):
                    continue
                result.tensions.append({
                    "node_id": node.node_id,
                    "label": node.label,
                    "content": node.content,
                    "source_context": context,
                })

    def _effective_confidence(self, node: Node, now: datetime) -> float:
        """
        Layer 1: effective confidence used as the recall post-factor — PURE.

        INFERRED nodes decay -0.01/week since last_referenced (floored at
        DECAY_FLOOR); EXTRACTED, DERIVED, CORRECTED and pinned nodes are immune.
        Delegates the decay MATH to GraphOperations._compute_decayed_confidence,
        which performs NO writes — so recall NEVER persists or audits. Decay is
        only persisted by the explicit maintenance pass (_apply_decay). This
        keeps reads pure: no write latency, no decay-spam in the audit log.

        The `now` arg is accepted for API parity but unused — decay anchors to
        wall-clock now inside the pure helper.
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node.confidence
        return self.ops._compute_decayed_confidence(node)

    def _origin_allowed_ids(
        self,
        node_ids: List[str],
        source_filter: Optional[List[str]],
        nodes: Optional[Dict[str, Node]] = None,
    ) -> set:
        """Which of ``node_ids`` pass the origin_runtime filter (WS0 Leg B).

        Used for candidate sources that don't carry origin data of their
        own (the semantic/vector index) — a membership check against
        ``source_filter``, over a bulk node fetch. Returns every id
        unfiltered (as a set) when no filter is active, so callers can
        unconditionally intersect against this without a None-check.

        ``nodes`` (optional): a caller's own already-fetched id->Node bulk
        lookup (e.g. one it needed anyway for a node_type check alongside
        the filter), reused here instead of a second store round-trip.
        Omitted, this fetches its own.

        G4: the check is ``source_filter is None``, never a truthy check —
        an EMPTY-but-present filter (``source_filter == []``) must fail
        closed (match nothing), and ``not []`` is True, so a truthy check
        here would silently unfilter it."""
        if source_filter is None:
            return set(node_ids)
        if nodes is None:
            nodes = self.store.get_nodes_bulk(node_ids)
        return {
            nid for nid in node_ids
            if nid in nodes and nodes[nid].origin_runtime in source_filter
        }

    def _find_anchors(
        self, query: str, source_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract entities/topics from query and find matching nodes in the graph.
        These become the starting points for graph traversal.

        source_filter (WS0 Leg B): threaded straight into the entity-label
        lookups as a SQL prefilter — a filtered recall's entity anchor
        search never even sees another runtime's nodes.
        """
        # Use the same extraction logic as ingestion
        extraction = self.extractor.extract(query, source_id="__query__")
        anchor_ids = []

        # Look for extracted entities/topics in the graph
        for candidate in extraction.nodes:
            if candidate.node_type == NodeType.CONTEXT:
                continue  # Skip the query's own context node

            # Try exact match first
            existing = self.ops.find_node_by_label(
                candidate.label, node_type=candidate.node_type,
                origin_runtime=source_filter,
            )
            if existing:
                anchor_ids.append(existing.node_id)
                continue

            # Try fuzzy match
            fuzzy = self.ops.find_nodes_by_label_fuzzy(
                candidate.label, max_distance=3, origin_runtime=source_filter,
            )
            for match in fuzzy:
                if match.node_id not in anchor_ids:
                    anchor_ids.append(match.node_id)

        # Alias expansion (alias leg): union in the LIVE ALIAS_OF neighbors of
        # every anchor found above — "Sam" anchors also seed "Sam R." and
        # "sam@..." when evidence-backed ALIAS_OF edges connect them. ONE hop
        # only, off the ORIGINAL anchors — deliberately not transitive
        # (expanding an alias's alias's alias compounds false-positive risk
        # multiplicatively, and v1 has no reason to take that risk for a gap
        # that one hop already closes). Distinguishability: the walker seeds
        # every anchor (original AND alias-expanded) at distance 0 with its
        # own path entry, so a result reached ONLY through the alias node
        # carries that node's label as the first hop in `path` — no new
        # response field needed, the existing path already tells the story.
        if _alias_expansion_enabled_by_env():
            for anchor_id in list(anchor_ids):
                for alias_id in self._live_alias_neighbors(anchor_id):
                    if alias_id not in anchor_ids:
                        anchor_ids.append(alias_id)

        return anchor_ids

    # Defensive bound, not a normal-path limit: the inference pass (alias.py)
    # is precision-first and blocked, so it can't realistically fan one node
    # out to dozens of live aliases. This exists for the OTHER creation path
    # — POST /v1/edges lets any caller draw ALIAS_OF edges directly — so a
    # spammed/misbehaving caller can't turn one anchor into an unbounded
    # anchor-set (and unbounded walk-seed) blowup.
    MAX_ALIAS_NEIGHBORS_PER_ANCHOR = 25

    def _live_alias_neighbors(self, node_id: str) -> List[str]:
        """LIVE ALIAS_OF neighbors of one node, both directions, one hop.
        Skips invalidated edges (a reversed alias) AND invalidated/missing
        neighbor nodes — a superseded alias must not silently keep expanding
        recall. One get_edges_for_node call per anchor; anchor sets are
        small (typically <10), so this stays cheap. Capped at
        MAX_ALIAS_NEIGHBORS_PER_ANCHOR (see its comment)."""
        neighbor_ids: List[str] = []
        for edge in self.store.get_edges_for_node(node_id):
            if edge.edge_type is not EdgeType.ALIAS_OF or edge.invalidated_at is not None:
                continue
            other_id = (edge.target_node_id if edge.source_node_id == node_id
                        else edge.source_node_id)
            neighbor_ids.append(other_id)
            if len(neighbor_ids) >= self.MAX_ALIAS_NEIGHBORS_PER_ANCHOR:
                break
        if not neighbor_ids:
            return []
        others = self.store.get_nodes_bulk(neighbor_ids)
        return [nid for nid in neighbor_ids
                if nid in others and others[nid].invalidated_at is None]

    def _keyword_search(
        self, query: str, limit: Optional[int] = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Fallback: search node labels and content for query keywords.
        Used when entity extraction finds no anchors — and, under
        REVIEN_HYBRID=rrf (LEG P1), ALWAYS, as the keyword-ranked fusion
        list (with limit widened to match the semantic list length, or
        uncapped under REVIEN_LEXICAL_LIMIT=0 — see engine.py's __init__
        comment on that knob).
        Default limit=10 keeps the shipped fallback path byte-identical.
        ``limit=None`` requests every matching node (SQLite's LIMIT -1).

        source_filter (WS0 Leg B): SQL prefilter passed straight through to
        store.search_nodes_keyword — a filtered recall's keyword-fallback
        anchor search never considers another runtime's rows.
        """
        words = set(query.lower().split())
        # Remove very common words
        stop = {"what", "did", "we", "about", "the", "is", "are", "was",
                "were", "how", "when", "where", "why", "who", "do", "does",
                "a", "an", "in", "on", "at", "to", "for", "of", "with",
                "that", "this", "it", "and", "or", "but", "not", "no",
                "have", "has", "had", "be", "been", "will", "would",
                "can", "could", "should", "may", "might", "my", "our",
                "i", "you", "he", "she", "they", "me", "us", "them",
                "last", "next", "any", "some"}
        keywords = words - stop

        if not keywords:
            return []

        # SQL-side substring search (same semantics as the old Python scan:
        # any-keyword hit on label+content, newest first, CONTEXT excluded,
        # capped at `limit` anchors). The old list_nodes(limit=999999) full
        # scan was the single biggest recall latency driver (OPEN 2).
        # limit=None -> SQLite LIMIT -1, its documented "no limit" value —
        # only reachable via REVIEN_LEXICAL_LIMIT=0, never the default path.
        sql_limit = -1 if limit is None else limit
        matches = self.store.search_nodes_keyword(
            keywords, limit=sql_limit, exclude_context=True,
            origin_runtime=source_filter,
        )
        return [n.node_id for n in matches]

    def _bm25_candidates(
        self, query: str, limit: Optional[int] = 10,
        source_filter: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """BM25-lane counterpart to ``_keyword_search`` (REVIEN_LEXICAL=bm25).

        Ranks the SAME corpus ``_keyword_search`` scans conceptually — every
        non-CONTEXT node's label+content, live OR soft-invalidated; NEITHER
        lane filters ``invalidated_at`` here, that's a downstream ``recall()``
        filter, not a property of this candidate read — but by Okapi BM25
        term rarity and saturating term frequency instead of substring
        presence, so a rare exact identifier or phrase outranks a document
        that merely repeats a common query word more often (see bm25.py's
        header for the production numbers this lane is validated against).

        COST (disclose, don't bury): this reintroduces the exact
        ``list_nodes(limit=999999)``-then-scan-in-Python shape that OPEN 2
        (see ``_keyword_search``'s comment, ``store.py``'s
        ``search_nodes_keyword``) moved OFF of and into SQL — because BM25's
        document-frequency/average-length stats need the WHOLE corpus's
        tokens, not a pre-filtered slice, or rarity is measured against the
        wrong population. Measured ~2.2x recall latency at 4k nodes vs the
        keyword lane's SQL-side scan. O(corpus) per call, not paid unless
        REVIEN_LEXICAL=bm25 is explicitly selected.

        Returns ``(ranked_ids, scores)`` where scores are bounded to
        ``score / (score + 1)`` — this keeps them a comparable 0..1-ish
        query-relevance signal alongside semantic cosine similarity, so
        ``recall()``'s max-of-available-signals blend (mirrors the overlay's
        contract) isn't dominated by BM25's unbounded raw magnitude.

        source_filter (WS0 Leg B): SQL prefilter on the list_nodes corpus
        fetch — a filtered recall's BM25 stats (document frequency, average
        length) are scored over the FILTERED corpus, not the whole graph,
        which is the correct population for "rarity" under a runtime
        filter, and — same as every other candidate source — a node outside
        the filter can never enter the ranked list at all.
        """
        documents = [
            (node.node_id, f"{node.label} {node.content}")
            for node in self.store.list_nodes(
                limit=999999, origin_runtime=source_filter,
            )
            if node.node_type != NodeType.CONTEXT
        ]
        ranked = bm25_rank(query, documents, top_n=limit)
        scores = {node_id: score / (score + 1.0) for node_id, score in ranked}
        return [node_id for node_id, _score in ranked], scores

    def _lexical_candidates(
        self, query: str, limit: Optional[int] = 10,
        source_filter: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Dispatch to the selected lexical lane (REVIEN_LEXICAL) without
        touching either call site's shipped behavior when unset: the
        keyword-fallback anchor path and the RRF fusion list both resolve
        through here, so flipping the env var moves both at once. Unset (or
        any value other than "bm25") returns exactly what ``_keyword_search``
        returned before this lane existed, with an empty scores dict — the
        keyword path is byte-identical. ``limit=None`` (REVIEN_LEXICAL_LIMIT
        =0, resolved by the RRF call site) means uncapped on either lane.

        source_filter (WS0 Leg B): threaded through to whichever lane runs."""
        if self.lexical_mode == "bm25":
            return self._bm25_candidates(query, limit=limit, source_filter=source_filter)
        return self._keyword_search(query, limit=limit, source_filter=source_filter), {}

    def mark_used(self, node_id: str, query: Optional[str] = None) -> None:
        """
        Mark a node as actually used after retrieval.
        Call this when the user references or acts on retrieved information.
        Provides positive training signal AND reinforces edge weights along the path.
        """
        # 1. Log training signal — only under explicit neural opt-in (same
        # gate as recall's log_retrieval; no silent signal accumulation).
        if self.neural_enabled:
            self.training_loop.mark_used(node_id, query)

        # 2. Touch the node (bump access count + last_accessed)
        self.ops.touch_node(node_id)

        # 2b. Provenance hook (leg 6a): record the access in the audit trail.
        # touch_node suppresses its own generic "update"; this is the single
        # "access" entry. Defensive — never breaks the underlying op.
        accessed = self.store.get_node(node_id)
        if accessed is not None:
            self.store._record_node_audit(
                node_id, "access", after_node=accessed,
            )

        # 3. Reinforce edges connected to this node (retrieval-driven learning)
        REINFORCEMENT_DELTA = 0.05  # Small boost per usage
        edges = self.store.get_edges_for_node(node_id)
        for edge in edges:
            new_weight = min(1.0, edge.weight + REINFORCEMENT_DELTA)
            if new_weight != edge.weight:
                self.store.update_edge_weight(edge.edge_id, new_weight)

    def get_training_stats(self) -> Dict:
        """Get neural training statistics."""
        return {
            "training": self.training_loop.get_stats(),
            "scorer": self.neural_scorer.get_stats(),
        }

    def force_train(self) -> bool:
        """Force training run regardless of threshold. For testing/manual triggers.

        Returns False when the neural extra is not installed (training no-ops).
        """
        return self.training_loop.train()

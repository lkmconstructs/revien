"""
Revien Consolidation — "dream mode" (B3.1): the explicit maintenance pass.

WHY: several maintenance surfaces were designed but never fire on their own
(verified dangling): confidence decay (_apply_decay) is never auto-called, so
INFERRED claims keep stale confidence forever; community clustering only
refreshes on ingest thresholds; the semantic index has a manual backfill
(reindex_all) and nothing calls it; and nothing ever looks for orphaned
nodes. Dream mode is the single, observable pass that runs them.

Sovereignty contract (the rails, non-negotiable):
  * NOTHING runs silently. Every pass returns a ConsolidationReport listing
    exactly what changed — counts AND items (capped) — and every mutation
    goes through the existing audited operations. A dream you can't inspect
    is a black box, which is the failure mode this product exists to reject.
  * NOTHING is destroyed. Decay demotes toward a floor (never deletes);
    orphans are REPORTED, not removed — cleanup is an explicit opt-in that
    SOFT-invalidates (reversible, audited), never hard-deletes. Consent Is
    Law applies to the machine's own housekeeping.
  * No pass in B3.1 merges nodes, so the false-merge audit surface is not
    fed by this module — any future consolidation pass that merges MUST
    feed it (standing rule).
  * Scheduling is OPT-IN (REVIEN_DREAM_INTERVAL_HOURS unset = never runs
    unattended). Manual-first: `revien dream` / POST /v1/consolidate.

Passes (each individually toggleable):
  decay      — persist confidence decay for INFERRED, non-pinned, live nodes
               (pinned/EXTRACTED/DERIVED/CORRECTED immune, per Leg 1).
  recluster  — re-run community detection on the current graph.
  reindex    — semantic re-embed backfill. DEFAULT OFF: embeddings stay in
               sync via the store content listener; this is for recovering
               from a corrupted/absent vector table, not routine upkeep.
  orphans    — detect nodes with no edges. Report always; soft-invalidate
               only with invalidate_orphans=True.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from revien.graph.operations import GraphOperations
from revien.graph.schema import NodeType, SourceType
from revien.graph.store import GraphStore

# Cap per-item detail in reports: enough to review, never a dump.
REPORT_ITEM_CAP = 50


@dataclass
class ConsolidationReport:
    """What a dream pass actually did. Serializable, never silent."""
    started_at: str = ""
    duration_ms: float = 0.0
    nodes_examined: int = 0
    # decay
    decay_ran: bool = False
    nodes_decayed: int = 0
    decayed_sample: List[Dict[str, Any]] = field(default_factory=list)
    # clustering
    recluster_ran: bool = False
    communities: int = 0
    # semantic reindex (backfill)
    reindex_ran: bool = False
    reindex_result: Optional[Dict[str, Any]] = None
    # orphans
    orphans_found: int = 0
    orphan_sample: List[Dict[str, str]] = field(default_factory=list)
    orphans_invalidated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "nodes_examined": self.nodes_examined,
            "decay": {
                "ran": self.decay_ran,
                "nodes_decayed": self.nodes_decayed,
                "sample": self.decayed_sample,
            },
            "recluster": {"ran": self.recluster_ran, "communities": self.communities},
            "reindex": {"ran": self.reindex_ran, "result": self.reindex_result},
            "orphans": {
                "found": self.orphans_found,
                "sample": self.orphan_sample,
                "invalidated": self.orphans_invalidated,
            },
        }


class Consolidator:
    """Runs the dream pass. Construct once per store; call run() per pass."""

    def __init__(
        self,
        store: GraphStore,
        ops: Optional[GraphOperations] = None,
        semantic=None,
        clustering=None,
    ):
        self.store = store
        self.ops = ops or GraphOperations(store)
        self.semantic = semantic
        self.clustering = clustering

    # ── passes ────────────────────────────────────────────────────────────

    def _decay_pass(self, report: ConsolidationReport) -> None:
        """Persist decay for every eligible node. Eligibility mirrors the
        recall-path rules exactly (INFERRED, un-pinned, live) so the dream
        pass writes what recall has been computing on the fly."""
        report.decay_ran = True
        nodes = self.store.list_nodes(limit=1_000_000)
        report.nodes_examined = len(nodes)
        for node in nodes:
            if node.pinned or node.source_type != SourceType.INFERRED:
                continue
            if node.invalidated_at is not None:
                continue
            before = node.confidence
            after_node = self.ops._apply_decay(node)
            if after_node.confidence < before:
                report.nodes_decayed += 1
                if len(report.decayed_sample) < REPORT_ITEM_CAP:
                    report.decayed_sample.append({
                        "node_id": node.node_id,
                        "label": node.label,
                        "before": round(before, 3),
                        "after": round(after_node.confidence, 3),
                    })

    def _recluster_pass(self, report: ConsolidationReport) -> None:
        if self.clustering is None:
            return
        report.recluster_ran = True
        communities = self.clustering.run()
        report.communities = len(communities)

    def _reindex_pass(self, report: ConsolidationReport) -> None:
        if self.semantic is None or not getattr(self.semantic, "is_enabled", False):
            return
        report.reindex_ran = True
        report.reindex_result = self.semantic.reindex_all()

    def _orphan_pass(self, report: ConsolidationReport, invalidate: bool) -> None:
        """Report nodes with no edges. CONTEXT nodes are exempt — a verbatim
        turn is the raw record and stands alone by design; extraction may
        simply have found nothing in it. Invalidation (opt-in) is SOFT and
        reversible — never a delete."""
        orphan_ids = self.store.list_orphan_node_ids()
        for node_id in orphan_ids:
            node = self.store.get_node(node_id)
            if node is None or node.invalidated_at is not None:
                continue
            if node.node_type == NodeType.CONTEXT:
                continue
            report.orphans_found += 1
            if len(report.orphan_sample) < REPORT_ITEM_CAP:
                report.orphan_sample.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "label": node.label,
                })
            if invalidate:
                self.ops.invalidate_node(
                    node.node_id,
                    reason="orphan: no edges at consolidation",
                    construct_id="dream_consolidation",
                )
                report.orphans_invalidated += 1

    # ── entry point ───────────────────────────────────────────────────────

    def run(
        self,
        decay: bool = True,
        recluster: bool = True,
        reindex: bool = False,
        invalidate_orphans: bool = False,
    ) -> ConsolidationReport:
        report = ConsolidationReport(
            started_at=datetime.now(timezone.utc).isoformat()
        )
        t0 = time.perf_counter()
        if decay:
            self._decay_pass(report)
        if recluster:
            self._recluster_pass(report)
        if reindex:
            self._reindex_pass(report)
        self._orphan_pass(report, invalidate=invalidate_orphans)
        report.duration_ms = (time.perf_counter() - t0) * 1000
        return report

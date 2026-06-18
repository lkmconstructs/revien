"""
revien_bench.sovereignty — The differentiator assertions (design §2.D, equal weight to F1).

All checks are deterministic and run on the default (graph_only, extractive) path
with ZERO network and $0 cost. Each returns a structured result with pass/fail and
the measured evidence, so the runner can fold them into results JSON.

Checks
------
1. provenance_completeness — % of answer-supporting nodes that carry a traceable
   source_id AND dia_id (lineage back to the exact LoCoMo turn). Target 100%.
2. audit_integrity — the append-only audit_log row count equals what we expect
   (>= one 'create' per node, plus the dia_id-tag 'update's), and is monotonic /
   never rewritten. Verified via store.get_all_audit() / get_node_audit().
3. network_egress_zero — assert the configured embedder/extractor are LOCAL
   (REVIEN_EMBEDDER local, REVIEN_EXTRACTOR not a cloud backend), and that the
   measured cloud-call counter is 0. The headline path makes no outbound calls.
4. consent — two sub-tests:
     a. REVIEN_INGEST_DENY -> denied source_id produces 0 nodes.
     b. soft-invalidate -> an invalidated node is EXCLUDED from default recall
        but RECOVERABLE (include_invalidated=True surfaces it; content retained).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from revien.graph.schema import NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import (
    CLOUD_EMBEDDERS,
    DEFAULT_EMBEDDER,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: Dict = field(default_factory=dict)


# ── 1. Provenance completeness ────────────────────────────────────────────────
def provenance_completeness(store: GraphStore, node_ids: Sequence[str]) -> Check:
    """% of the given answer-supporting nodes with traceable source_id + dia_id."""
    total = 0
    traceable = 0
    missing: List[str] = []
    for nid in node_ids:
        node = store.get_node(nid)
        if node is None:
            continue
        total += 1
        md = node.metadata or {}
        if node.source_id and md.get("dia_id"):
            traceable += 1
        else:
            missing.append(nid)
    pct = (traceable / total * 100.0) if total else 100.0
    return Check(
        name="provenance_completeness",
        passed=(total == 0 or pct >= 100.0),
        detail={
            "nodes_checked": total,
            "traceable": traceable,
            "pct": round(pct, 2),
            "missing_lineage": missing[:10],
        },
    )


# ── 2. Audit integrity ────────────────────────────────────────────────────────
def audit_integrity(store: GraphStore, expected_min_creates: int) -> Check:
    """Append-only audit trail: a 'create' row exists for >= every node, ids are
    strictly increasing (never rewritten), and the row count is >= expected."""
    rows = store.get_all_audit()
    creates = sum(1 for r in rows if r.get("op") == "create")
    ids = [r["id"] for r in rows]
    monotonic = all(b > a for a, b in zip(ids, ids[1:])) if len(ids) > 1 else True
    node_count = store.count_nodes()
    passed = (
        monotonic
        and creates >= node_count            # >=1 create per surviving node
        and len(rows) >= expected_min_creates
    )
    return Check(
        name="audit_integrity",
        passed=passed,
        detail={
            "audit_rows": len(rows),
            "create_rows": creates,
            "node_count": node_count,
            "ids_monotonic": monotonic,
            "expected_min_rows": expected_min_creates,
        },
    )


# ── 3. Network egress == 0 ────────────────────────────────────────────────────
def network_egress_zero(cloud_calls: int = 0) -> Check:
    """Assert the default path is local and made zero cloud calls.

    `cloud_calls` is the runner's measured counter (always 0 on the headline
    path — no cloud answerer, local/absent embedder). We also inspect the env
    so a misconfiguration (REVIEN_EMBEDDER=openai) is caught as a FAIL.
    """
    embedder = (os.environ.get("REVIEN_EMBEDDER", DEFAULT_EMBEDDER) or DEFAULT_EMBEDDER).lower().strip()
    extractor = (os.environ.get("REVIEN_EXTRACTOR", "rule") or "rule").lower().strip()
    embedder_local = embedder not in CLOUD_EMBEDDERS
    # Any extractor other than the local "rule" default is treated as potentially
    # cloud-bound for this assertion (llm backends can be cloud).
    extractor_local = extractor in ("rule", "", "local")
    passed = embedder_local and extractor_local and cloud_calls == 0
    return Check(
        name="network_egress_zero",
        passed=passed,
        detail={
            "embedder": embedder,
            "embedder_local": embedder_local,
            "extractor": extractor,
            "extractor_local": extractor_local,
            "cloud_calls": cloud_calls,
        },
    )


# ── 4. Consent sub-tests ──────────────────────────────────────────────────────
def consent_deny_subtest() -> Check:
    """REVIEN_INGEST_DENY: a denied source_id must capture 0 nodes.

    Runs in a throwaway in-memory store so it never touches benchmark state.
    """
    prev = os.environ.get("REVIEN_INGEST_DENY")
    store = GraphStore(db_path=":memory:")
    try:
        os.environ["REVIEN_INGEST_DENY"] = "denied:turn"
        pipe = IngestionPipeline(store)
        before = store.count_nodes()
        out = pipe.ingest(
            IngestionInput(
                source_id="denied:turn",
                content="Alice: the secret password is hunter2 and we use PostgreSQL.",
                content_type="conversation",
            )
        )
        after = store.count_nodes()
        passed = out.nodes_created == 0 and after == before
        detail = {"nodes_created": out.nodes_created, "before": before, "after": after}
    finally:
        if prev is None:
            os.environ.pop("REVIEN_INGEST_DENY", None)
        else:
            os.environ["REVIEN_INGEST_DENY"] = prev
        store.close()
    return Check(name="consent_ingest_deny", passed=passed, detail=detail)


def consent_soft_invalidate_subtest() -> Check:
    """Soft-invalidate: invalidated node is excluded from default recall but
    recoverable (content retained; include_invalidated=True surfaces it)."""
    store = GraphStore(db_path=":memory:")
    try:
        pipe = IngestionPipeline(store)
        pipe.ingest(
            IngestionInput(
                source_id="conv:D1:1",
                content="Alice: We deployed the service on PostgreSQL last Tuesday.",
                content_type="conversation",
            )
        )
        engine = RetrievalEngine(store)
        q = "What database did we deploy on?"

        before = engine.recall(q, top_n=10)
        hit_before = any("postgres" in r.content.lower() for r in before.results)

        # Soft-invalidate every non-context node (mark stale, non-destructive).
        target_ids = [
            n.node_id for n in store.list_nodes(limit=999999)
            if n.node_type != NodeType.CONTEXT
        ]
        now = datetime.now(timezone.utc)
        for nid in target_ids:
            store.update_node(nid, invalidated_at=now, _audit_op="invalidate")

        after = engine.recall(q, top_n=10)
        hit_after_default = any("postgres" in r.content.lower() for r in after.results)

        recovered = engine.recall(q, top_n=10, include_invalidated=True)
        hit_recovered = any("postgres" in r.content.lower() for r in recovered.results)

        # Content retained in store regardless of invalidation.
        content_retained = all(
            store.get_node(nid) is not None and store.get_node(nid).content
            for nid in target_ids
        )

        passed = (
            hit_before                 # was retrievable
            and not hit_after_default  # excluded from default recall once stale
            and hit_recovered          # recoverable via include_invalidated
            and content_retained       # never deleted
        )
        detail = {
            "hit_before": hit_before,
            "hit_after_default": hit_after_default,
            "hit_recovered": hit_recovered,
            "content_retained": content_retained,
            "invalidated_nodes": len(target_ids),
        }
    finally:
        store.close()
    return Check(name="consent_soft_invalidate", passed=passed, detail=detail)


def run_consent_subtests() -> List[Check]:
    return [consent_deny_subtest(), consent_soft_invalidate_subtest()]


# ── Aggregate ─────────────────────────────────────────────────────────────────
def all_checks_passed(checks: Sequence[Check]) -> bool:
    return all(c.passed for c in checks)


def checks_to_dict(checks: Sequence[Check]) -> Dict:
    return {
        "all_passed": all_checks_passed(checks),
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks
        ],
    }

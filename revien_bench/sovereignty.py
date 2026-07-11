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
3. network_egress_zero — assert EVERY active backend is local: the extractor
   (REVIEN_EXTRACTOR), embedder (REVIEN_EMBEDDER), answerer (--answerer reader),
   and judge. PASSES only if none is a cloud provider; FAILS and names the
   off-device backend(s) if any sends data to a cloud API. The decision is made
   from the run config (env + answerer spec), NOT by intercepting HTTP — a cloud
   run must never be able to claim zero egress. The measured cloud-call counter
   is folded in as a secondary signal. The headline path makes no outbound calls.
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
from revien.ingestion.extractor_llm import (
    CLOUD_BACKENDS as CLOUD_EXTRACTORS,
    DEFAULT_EXTRACTOR,
)
from revien.semantic.index import (
    CLOUD_EMBEDDERS,
    DEFAULT_EMBEDDER,
)

# Answerer (reader) cloud providers. Imported from the answerer layer so there
# is ONE source of truth for "which reader backends leave the machine".
from .answerers import CLOUD_PROVIDERS as CLOUD_ANSWERERS, parse_provider


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
# A backend is LOCAL unless its provider is one of these cloud sets. The decision
# is made from the run config — never from "did we happen to observe HTTP" — so a
# cloud run can never claim zero egress just because a counter wasn't incremented.
_LOCAL_EXTRACTORS = ("rule", "", "local", "ollama")
_LOCAL_EMBEDDERS = ("", "local")  # plus anything not in CLOUD_EMBEDDERS
_LOCAL_ANSWERERS = ("extractive", "", "local", "ollama")
_LOCAL_JUDGES = ("rule", "f1", "extractive", "", "local", "none", "ollama")


def network_egress_zero(
    cloud_calls: int = 0,
    answerer: Optional[str] = None,
    judge: Optional[str] = None,
) -> Check:
    """PASS only if EVERY active backend is local; FAIL naming any cloud backend.

    Active backends inspected from the run CONFIG (not from intercepted HTTP):
      * extractor  — REVIEN_EXTRACTOR (default 'rule'); cloud set CLOUD_EXTRACTORS
      * embedder   — REVIEN_EMBEDDER  (default 'fastembed'); cloud set CLOUD_EMBEDDERS
      * answerer   — the --answerer reader spec; cloud set CLOUD_ANSWERERS
      * judge      — the scorer; the headline build uses local F1 (no cloud judge)

    If ANY backend is a cloud provider (openai/openrouter/anthropic/together/
    claude), the check FAILS and `detail["cloud_backends"]` names which component
    sent data off-device — even if `cloud_calls` is 0. The measured `cloud_calls`
    counter is a secondary safety signal: a positive count alone also fails.
    """
    embedder = (os.environ.get("REVIEN_EMBEDDER", DEFAULT_EMBEDDER) or DEFAULT_EMBEDDER).lower().strip()
    extractor = (os.environ.get("REVIEN_EXTRACTOR", DEFAULT_EXTRACTOR) or DEFAULT_EXTRACTOR).lower().strip()
    # The answerer is passed in (it is a CLI arg, not an env var); fall back to
    # the local default reader.
    answerer_spec = (answerer or "extractive").strip()
    answerer_provider, _ = parse_provider(answerer_spec)
    # The judge is local F1 in this build (no cloud-judge wiring yet); accept an
    # explicit override so the assertion stays honest if one is ever added.
    judge_name = (judge or "f1").strip().lower()

    # Local determination, per backend, from config.
    extractor_local = (extractor in _LOCAL_EXTRACTORS) and (extractor not in CLOUD_EXTRACTORS)
    embedder_local = (embedder in _LOCAL_EMBEDDERS or embedder not in CLOUD_EMBEDDERS)
    answerer_local = (answerer_provider in _LOCAL_ANSWERERS) and (answerer_provider not in CLOUD_ANSWERERS)
    judge_local = (judge_name in _LOCAL_JUDGES) and (judge_name not in CLOUD_ANSWERERS)

    # Name every component that leaves the machine (config-derived).
    cloud_backends: List[str] = []
    if not extractor_local:
        cloud_backends.append(f"extractor={extractor}")
    if not embedder_local:
        cloud_backends.append(f"embedder={embedder}")
    if not answerer_local:
        cloud_backends.append(f"answerer={answerer_provider}")
    if not judge_local:
        cloud_backends.append(f"judge={judge_name}")

    all_local = not cloud_backends
    # PASS requires every backend local AND no observed cloud call. A cloud
    # backend in config fails even if cloud_calls==0; a positive counter fails
    # even if config somehow looked local.
    passed = all_local and cloud_calls == 0
    return Check(
        name="network_egress_zero",
        passed=passed,
        detail={
            "extractor": extractor,
            "extractor_local": extractor_local,
            "embedder": embedder,
            "embedder_local": embedder_local,
            "answerer": answerer_provider,
            "answerer_local": answerer_local,
            "judge": judge_name,
            "judge_local": judge_local,
            "all_backends_local": all_local,
            "cloud_backends": cloud_backends,
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

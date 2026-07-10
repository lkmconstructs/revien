"""Claim Sovereignty Layer — Leg B: the assembly point.

Components verified in isolation are CONNECTED here into one path that runs on
every ingested claim:

    new claim node (a verbatim CONTEXT turn)
      -> classify (claim_type + durability + confidence)              [Leg 2.5]
      -> for each comparable existing claim:
           gate.evaluate(existing, new)                               [Leg 3]
             floor -> scope -> contradiction -> recognizer -> tripwire -> decide
           -> AUTO_SUPERSEDE        : soft-invalidate the existing claim
                                      (retained, excluded from recall), link
                                      new -CORRECTS-> old. The ONLY mutating path.
           -> CANDIDATE / VERSION_LOCKED : park the pair in the candidate queue
                                      for human review. Mutates NOTHING.
           -> COEXIST (B1)          : genuine tension between two affirmative
                                      claims — BOTH stay live, a CONFLICTS_WITH
                                      edge makes the tension first-class.
                                      Opt-in via REVIEN_TENSION_BACKEND.
           -> NO_CONFLICT           : leave both standing.

Design choices:
  * A "claim" = the verbatim CONTEXT node (the turn). Single-claim turns map
    cleanly; a compound turn is flagged by the classifier and the gate routes it
    to candidate. (Splitting a turn into multiple claims is future work.)
  * The classifier is DETERMINISTIC, so existing claims are re-derived from their
    text on the fly — no classification is persisted for correctness. Cheap.
  * The recognizer (Trigger 2, qwen/cloud for us) is the ONLY network touch and
    fires ONLY when the gate reaches a would-be-auto on a scoped contradiction —
    bounded, not per-ingest. When no recognizer is wired the gate runs the
    sensitive floor + distrust tripwire (interim, degraded-safety).
  * Opt-in: the pipeline runs this only when a governor is wired (REVIEN_CSL=1 or
    an injected ClaimGovernor), so default ingest is byte-for-byte unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.ingestion.claim_classifier import ClaimClassifier
from revien.supersession import (
    Claim,
    SupersessionAction,
    SupersessionGate,
    SupersessionMetrics,
)


@dataclass
class GovernanceOutcome:
    """What the gate decided for one (existing, new) claim pair, and what was done."""
    action: str               # auto_supersede | candidate | version_locked
    existing_node_id: str
    new_node_id: str
    existing_text: str
    new_text: str
    reason: str
    data_effect: str          # what actually happened to the data
    trace: List[str] = field(default_factory=list)


class ClaimGovernor:
    """Runs the full Claim Sovereignty path on each ingested claim node."""

    def __init__(self, store, ops, *, classifier=None, gate=None, metrics=None,
                 max_compare: int = 5000):
        self.store = store
        self.ops = ops
        self.classifier = classifier or ClaimClassifier()
        # Default gate = sensitive floor + default-on distrust tripwire, no
        # recognizer (interim). Inject a gate with recognizer=... to wire Trigger 2.
        self.gate = gate if gate is not None else SupersessionGate()
        self.metrics = metrics if metrics is not None else SupersessionMetrics()
        self.max_compare = max_compare

    def _existing_claims(self, new_node: Node) -> List[Node]:
        """Live (non-invalidated) CONTEXT claims other than the one just ingested."""
        nodes = self.store.list_nodes(node_type=NodeType.CONTEXT, limit=self.max_compare)
        return [
            n for n in nodes
            if n.node_id != new_node.node_id
            and n.invalidated_at is None
            and (n.content or "").strip()
        ]

    def govern(self, new_node: Node) -> List[GovernanceOutcome]:
        """Govern one freshly-ingested claim against existing memory."""
        text = (new_node.content or "").strip()
        if not text:
            return []
        new_claim = Claim(text=text, result=self.classifier.classify(text))

        outcomes: List[GovernanceOutcome] = []
        for existing_node in self._existing_claims(new_node):
            ex_text = existing_node.content.strip()
            existing_claim = Claim(text=ex_text, result=self.classifier.classify(ex_text))
            decision = self.gate.evaluate(existing_claim, new_claim)
            self.metrics.record(decision)

            if decision.action is SupersessionAction.NO_CONFLICT:
                continue

            # Curated shield (Obsidian vault leg): a human-curated claim is
            # NEVER silently auto-superseded by a machine-side claim. The
            # contradiction is real and preserved — it goes to the candidate
            # queue for the human's hands, not the gate's. Consent Is Law,
            # mechanized. A curated NEW claim superseding a machine claim is
            # allowed through unchanged (the human's word outranks ours).
            action = decision.action
            reason = decision.reason
            if (
                action is SupersessionAction.AUTO_SUPERSEDE
                and (existing_node.metadata or {}).get("curated")
                and not (new_node.metadata or {}).get("curated")
            ):
                action = SupersessionAction.CANDIDATE
                reason = f"curated_shield: {reason}"

            if action is SupersessionAction.AUTO_SUPERSEDE:
                self._supersede(existing_node, new_node)
                effect = f"existing soft-invalidated (superseded_by {new_node.node_id})"
            elif action is SupersessionAction.COEXIST:
                # B1: genuine tension — BOTH claims stay live, nothing queued,
                # nothing invalidated. The tension becomes first-class as a
                # CONFLICTS_WITH edge (non-mutating by contract, schema.py) so
                # recall/lineage can surface it. Additive-only, so the curated
                # shield does not apply.
                self._draw_tension(existing_node, new_node, reason)
                effect = "tension drawn (conflicts_with edge); BOTH claims live"
            else:  # CANDIDATE or VERSION_LOCKED -> queue, mutate nothing
                cid = self.store.add_candidate(
                    existing_node.node_id, new_node.node_id,
                    action.value, reason, " | ".join(decision.trace),
                )
                effect = f"queued for review (candidate #{cid}); BOTH claims preserved"

            outcomes.append(GovernanceOutcome(
                action=action.value,
                existing_node_id=existing_node.node_id,
                new_node_id=new_node.node_id,
                existing_text=existing_node.content,
                new_text=new_node.content,
                reason=reason,
                data_effect=effect,
                trace=list(decision.trace),
            ))
        return outcomes

    def _supersede(self, existing_node: Node, new_node: Node) -> None:
        """Soft-invalidate the existing claim and record the supersession link."""
        self.ops.invalidate_node(
            existing_node.node_id,
            reason=f"superseded_by:{new_node.node_id}",
            construct_id="csl_supersession",
        )
        try:
            self.store.add_edge(Edge(
                edge_type=EdgeType.CORRECTS,
                source_node_id=new_node.node_id,
                target_node_id=existing_node.node_id,
                weight=1.0,
            ))
        except Exception:  # noqa: BLE001 - the link is best-effort; invalidation is the fact
            pass

    def _draw_tension(
        self, existing_node: Node, new_node: Node, source_context: str
    ) -> None:
        """Draw the non-mutating tension edge between two live claims (B1).

        Idempotent per pair: re-ingesting either side must not stack duplicate
        tension edges (the governor re-compares every ingest)."""
        for e in self.store.get_edges_for_node(existing_node.node_id):
            if e.edge_type is EdgeType.CONFLICTS_WITH and (
                {e.source_node_id, e.target_node_id}
                == {existing_node.node_id, new_node.node_id}
            ):
                return
        self.store.add_edge(Edge(
            edge_type=EdgeType.CONFLICTS_WITH,
            source_node_id=new_node.node_id,
            target_node_id=existing_node.node_id,
            weight=0.8,
            confidence_set_by="csl_tension_gate",
            source_context=source_context,
        ))

    def coexist_candidate(self, candidate_id: int) -> bool:
        """Resolve a queued candidate as a COEXISTING tension (human's hands).

        Draws the CONFLICTS_WITH edge between the pair and marks the candidate
        resolved as "coexist". Mutates neither claim — both stay live. Returns
        False when the candidate is unknown or already resolved.
        """
        row = next(
            (c for c in self.store.list_candidates(unresolved_only=True)
             if c["id"] == candidate_id),
            None,
        )
        if row is None:
            return False
        existing_node = self.store.get_node(row["existing_node_id"])
        new_node = self.store.get_node(row["new_node_id"])
        if existing_node is None or new_node is None:
            return False
        self._draw_tension(
            existing_node, new_node, f"coexist_resolution:candidate#{candidate_id}"
        )
        self.store.resolve_candidate(candidate_id, "coexist")
        return True


def _env_recognizer():
    """Build the Trigger-2 recognizer from env, or None (gate runs floor+tripwire).

    Wired when REVIEN_SENSITIVITY_BACKEND is set AND the backend is reachable
    (e.g. a cloud key is present). Never raises — an unavailable recognizer
    yields None and the gate degrades to floor + tripwire.
    """
    if not os.environ.get("REVIEN_SENSITIVITY_BACKEND"):
        return None
    try:
        from revien.sensitivity_llm import LLMSensitivityRecognizer
        rec = LLMSensitivityRecognizer()
        return rec if rec.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _env_tension_recognizer():
    """Build the B1 tension recognizer from env, or None (gate unchanged).

    Wired when REVIEN_TENSION_BACKEND is set AND the backend is reachable.
    Mirrors _env_recognizer: never raises; unavailable yields None.
    """
    if not os.environ.get("REVIEN_TENSION_BACKEND"):
        return None
    try:
        from revien.tension import LLMTensionRecognizer
        rec = LLMTensionRecognizer()
        return rec if rec.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def build_governor(store, ops, recognizer="env", tension_recognizer="env") -> ClaimGovernor:
    """Factory: a governor with the default-on tripwire and optional recognizers.

    recognizer/tension_recognizer="env" (default) resolve from env; pass an
    explicit recognizer (or None) to override either.
    """
    rec = _env_recognizer() if recognizer == "env" else recognizer
    trec = (_env_tension_recognizer() if tension_recognizer == "env"
            else tension_recognizer)
    return ClaimGovernor(
        store, ops,
        gate=SupersessionGate(recognizer=rec, tension_recognizer=trec),
    )

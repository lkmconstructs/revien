"""
Revien Graph Operations — Higher-level operations on the graph store.
Convenience layer over raw CRUD.
Confidence layer: confidence tagging, reinforcement, decay, propagation
(2-hop bounded).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union

from .schema import Edge, EdgeType, Graph, Node, NodeType, SourceType
from .store import GraphStore


# ── Governance Layer (leg 6b): retention config ──────────────────────────────
# "Choose your storage." Nothing auto-deletes by default. The mode is read once
# from the environment per sweep so a deployment can flip storage policy without
# code changes.
RETENTION_KEEP = "keep"        # no-op sweep — current behavior (decay demotes only)
RETENTION_ARCHIVE = "archive"  # sweep soft-invalidates stale nodes (recoverable)
RETENTION_EXPIRE = "expire"    # sweep HARD-deletes stale nodes (+tombstone) — opt-in
VALID_RETENTION_MODES = (RETENTION_KEEP, RETENTION_ARCHIVE, RETENTION_EXPIRE)
DEFAULT_RETENTION_MODE = RETENTION_KEEP
DEFAULT_RETENTION_DAYS = 90


def get_retention_mode() -> str:
    """Resolve the retention mode from ``REVIEN_RETENTION`` (default ``keep``).

    Unknown values fall back to the safe default (``keep`` — nothing removed).
    """
    raw = (os.environ.get("REVIEN_RETENTION") or DEFAULT_RETENTION_MODE).strip().lower()
    return raw if raw in VALID_RETENTION_MODES else DEFAULT_RETENTION_MODE


def get_retention_days() -> int:
    """Resolve the retention window in days from ``REVIEN_RETENTION_DAYS``.

    Non-numeric / non-positive values fall back to the default window.
    """
    raw = os.environ.get("REVIEN_RETENTION_DAYS")
    if raw is None:
        return DEFAULT_RETENTION_DAYS
    try:
        days = int(raw)
        return days if days > 0 else DEFAULT_RETENTION_DAYS
    except (TypeError, ValueError):
        return DEFAULT_RETENTION_DAYS


class GraphOperations:
    """High-level graph operations built on GraphStore."""

    # Decay demotes but never deletes: INFERRED nodes decay toward this floor,
    # never to zero. Only explicit correction (correct_node) sets confidence to
    # 0.0. Aged memories stay retrievable but rank low — "store everything,
    # compact nothing."
    DECAY_FLOOR = 0.15

    def __init__(self, store: GraphStore):
        self.store = store

    # ── Confidence Layer: Scoring, Decay, Reinforcement ───────

    # metadata key: UTC ISO timestamp up to which decay has ALREADY been
    # charged against node.confidence. Metadata, not a column, so there is no
    # migration; absent on nodes never decayed (their reference is then
    # last_referenced/created_at, as before).
    DECAYED_AT_KEY = "confidence_decayed_at"

    def _decay_reference(self, node: Node) -> datetime:
        """The instant decay is charged FROM: the later of the node's last
        reference and the last APPLIED decay. Without the second term every
        pass re-charged the whole age against an already-decayed confidence
        (compounding on each consolidator run, and again on the read path)."""
        reference_time = node.last_referenced or node.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        raw = (node.metadata or {}).get(self.DECAYED_AT_KEY)
        if raw:
            try:
                applied = datetime.fromisoformat(raw)
                if applied.tzinfo is None:
                    applied = applied.replace(tzinfo=timezone.utc)
                if applied > reference_time:
                    reference_time = applied
            except (TypeError, ValueError):
                pass  # unreadable marker: fall back to the node's own times
        return reference_time

    def _compute_decayed_confidence(self, node: Node) -> float:
        """PURE: effective (decayed) confidence for a node — NO writes, NO audit.

        Decay rate: -0.01/week since last_referenced (or created_at), floored at
        DECAY_FLOOR (demote, never delete). Pinned and non-INFERRED nodes are
        immune (return stored confidence unchanged). This is the single source
        of the decay MATH and is safe on the read/recall path — it never touches
        the store, so reads never trigger writes or audit entries.
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node.confidence
        now = datetime.now(timezone.utc)
        reference_time = self._decay_reference(node)
        weeks_since = (now - reference_time).days / 7.0
        decay_amount = weeks_since * 0.01
        # Demote, don't delete: floor at DECAY_FLOOR; min() guard never raises a
        # node already sitting below the floor.
        return min(node.confidence, max(self.DECAY_FLOOR, node.confidence - decay_amount))

    def _apply_decay(self, node: Node) -> Node:
        """PERSIST decay (write + 'decay' audit) for a node.

        Use ONLY on an explicit maintenance/retention pass — NOT on the
        read/recall path, which uses the pure _compute_decayed_confidence so
        reads never write. Persists only when the change is meaningful (> 0.005).
        Demotes, never deletes (floored at DECAY_FLOOR); only correct_node zeroes.
        """
        if node.pinned or node.source_type != SourceType.INFERRED:
            return node
        new_confidence = self._compute_decayed_confidence(node)
        if node.confidence - new_confidence > 0.005:
            # Advance the marker by the WHOLE days charged (the math counts
            # whole days), so the remainder is not lost and a repeat pass at
            # the same instant charges nothing.
            charged_to = self._decay_reference(node) + timedelta(
                days=(datetime.now(timezone.utc) - self._decay_reference(node)).days)
            return self.store.update_node(
                node.node_id,
                confidence=new_confidence,
                confidence_set_at=datetime.now(timezone.utc),
                metadata={**(node.metadata or {}),
                          self.DECAYED_AT_KEY: charged_to.isoformat()},
                source_context="lazy_decay",
                _audit_op="decay",
            ) or node
        return node

    def reinforce_node(self, node_id: str, construct_id: str = "") -> Optional[Node]:
        """Reinforce a node's confidence after successful use.

        Increment confidence by +0.05 (capped at 1.0). Update last_referenced
        timestamp and audit fields.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc)
        new_confidence = min(1.0, node.confidence + 0.05)
        return self.store.update_node(
            node_id,
            confidence=new_confidence,
            confidence_set_at=now,
            confidence_set_by=construct_id,
            source_context="reinforced_after_use",
            last_referenced=now,
            metadata={**node.metadata, "_reinforced_by": construct_id},
            _audit_op="reinforce",
            _audit_actor=construct_id,
        )

    def correct_node(
        self, node_id: str, correction_context: str = "", construct_id: str = ""
    ) -> Optional[Node]:
        """Mark a node as CORRECTED due to conflicting information.

        Set source_type to CORRECTED and confidence to 0.0.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc)
        return self.store.update_node(
            node_id,
            source_type=SourceType.CORRECTED,
            confidence=0.0,
            confidence_set_at=now,
            confidence_set_by=construct_id,
            source_context=correction_context,
            metadata={
                **node.metadata,
                "_corrected_by": construct_id,
                "_corrected_at": now.isoformat(),
            },
            _audit_op="correct",
            _audit_actor=construct_id,
        )

    def invalidate_node(
        self, node_id: str, reason: str = "", construct_id: str = ""
    ) -> Optional[Node]:
        """Soft-invalidate a node: mark it stale without deleting anything.

        Sets ``invalidated_at`` to now and writes an audit entry. This is for
        correction / archive / supersession — the node's content is RETAINED and
        the row is never removed. Invalidated nodes are excluded from default
        recall but remain fully inspectable and recoverable. This is NOT deletion
        and NOT right-to-forget (that is leg 6b). Idempotent: re-invalidating an
        already-invalid node leaves the original timestamp untouched.
        """
        node = self.store.get_node(node_id)
        if node is None:
            return None
        if node.invalidated_at is not None:
            # Already stale — don't overwrite the original timestamp.
            return node

        now = datetime.now(timezone.utc)
        return self.store.update_node(
            node_id,
            invalidated_at=now,
            metadata={
                **node.metadata,
                "_invalidated_by": construct_id,
                "_invalidated_reason": reason,
                "_invalidated_at": now.isoformat(),
            },
            _audit_op="invalidate",
            _audit_actor=construct_id,
        )

    def invalidate_edge(
        self, edge_id: str, reason: str = "", construct_id: str = ""
    ) -> Optional[Edge]:
        """Soft-invalidate an edge: mark it stale without deleting anything.

        Edge analogue of invalidate_node (alias leg): sets ``invalidated_at``
        to now and writes an audit entry via ``store.update_edge``. The row
        is RETAINED — a wrong ALIAS_OF (or any future provenance-bearing
        edge) is reversible, not erased, and recall expansion / the alias
        CLI listing both skip invalidated edges by default. Idempotent:
        re-invalidating an already-invalid edge leaves the original
        timestamp untouched.
        """
        edge = self.store.get_edge(edge_id)
        if edge is None:
            return None
        if edge.invalidated_at is not None:
            # Already reversed — don't overwrite the original timestamp.
            return edge

        now = datetime.now(timezone.utc)
        return self.store.update_edge(
            edge_id,
            invalidated_at=now,
            metadata={
                **edge.metadata,
                "_invalidated_by": construct_id,
                "_invalidated_reason": reason,
                "_invalidated_at": now.isoformat(),
            },
            _audit_op="invalidate",
            _audit_actor=construct_id,
        )

    def get_lineage(self, node_id: str, max_depth: int = 10) -> Dict:
        """Trace a node's derivation chain via DERIVED_FROM edges.

        Walks DERIVED_FROM edges in the source→ancestor direction (a node is
        ``DERIVED_FROM`` its source, so we follow edges where this node is the
        SOURCE node to reach its ancestors). Bounded depth and cycle-safe.

        Returns:
            {
              "node_id": <id>,
              "ancestors": [ {node summary, "depth": n}, ... ],  # nearest first
              "truncated": bool,  # True if max_depth stopped the walk
            }
        """
        root = self.store.get_node(node_id)
        if root is None:
            return {"node_id": node_id, "ancestors": [], "truncated": False}

        ancestors: List[Dict] = []
        visited = {node_id}
        truncated = False
        # BFS over DERIVED_FROM edges, current node as edge SOURCE -> ancestor.
        frontier = [(node_id, 0)]
        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                # Stop expanding; flag truncation only if more would follow.
                edges = self.store.get_edges_for_node(current_id)
                if any(
                    e.edge_type == EdgeType.DERIVED_FROM
                    and e.source_node_id == current_id
                    and e.target_node_id not in visited
                    for e in edges
                ):
                    truncated = True
                continue

            for edge in self.store.get_edges_for_node(current_id):
                if edge.edge_type != EdgeType.DERIVED_FROM:
                    continue
                # current is DERIVED_FROM its source -> ancestor is the target.
                if edge.source_node_id != current_id:
                    continue
                ancestor_id = edge.target_node_id
                if ancestor_id in visited:
                    continue
                visited.add(ancestor_id)
                ancestor = self.store.get_node(ancestor_id)
                if ancestor is None:
                    continue
                ancestors.append({
                    "node_id": ancestor.node_id,
                    "node_type": ancestor.node_type.value,
                    "label": ancestor.label,
                    "depth": depth + 1,
                    "invalidated_at": (
                        ancestor.invalidated_at.isoformat()
                        if ancestor.invalidated_at else None
                    ),
                })
                frontier.append((ancestor_id, depth + 1))

        return {"node_id": node_id, "ancestors": ancestors, "truncated": truncated}

    # ── Three verbs (docs/FORGET_SPEC.md §1) ─────────────────
    #
    #   forget   privacy: the node, its wording and every copy are scrubbed,
    #            a content-free suppression record keeps the source from
    #            bringing it back (revien/forget/engine.py)
    #   delete   operational: remove this node now, leave a content-free
    #            delete audit entry, no suppression record
    #   cleanup  retention and other unattended maintenance, its own audit
    #            entry, no suppression record
    #
    # The verb is recorded by the code path that ran, as the audit op and a
    # fixed actor. A caller-supplied reason or construct id is never stored:
    # it is free text, and free text is where a secret ends up.

    TOMBSTONE_PREFIX = "tombstone:"

    @classmethod
    def _tombstone_id(cls, original_node_id: str) -> str:
        return f"{cls.TOMBSTONE_PREFIX}{original_node_id}"

    #: Where training.db lives besides the engine's own; set by the daemon.
    training_dbs: tuple = ()
    #: The community detector to rebuild after a forget; set by the daemon.
    clustering = None
    #: The semantic index whose vectors a forget must drop; set by the daemon.
    semantic = None
    #: Retrains the ranking model after a forget; set by the daemon when
    #: neural ranking is on. None: nothing retrains it automatically.
    retrain = None
    #: Says why the last retrain produced no model (a callable, or None).
    retrain_reason = None

    def _forget_engine(self):
        from revien.forget.engine import ForgetEngine
        dbs = list(self.training_dbs)
        default = os.path.join(os.path.expanduser("~"), ".revien", "training.db")
        if default not in dbs:
            dbs.append(default)
        retrain, reason = self.retrain, self.retrain_reason
        if retrain is None and os.environ.get("REVIEN_NEURAL", "0").strip().lower() \
                in ("1", "true", "yes", "on"):
            # a forget run from the CLI with neural ranking switched on
            from revien.neural.training import TrainingLoop
            loop = TrainingLoop(db_path=default)
            retrain, reason = loop.train, (lambda: loop.last_failure)
        return ForgetEngine(self.store, semantic=self.semantic,
                            training_dbs=dbs, clustering=self.clustering,
                            retrain=retrain, retrain_reason=reason)

    def forget_preview(self, node_id: str) -> Dict:
        """What a forget WOULD do, so it is never blind. Writes nothing."""
        out = self._forget_engine().forget(node_id, dry_run=True)
        out["exists"] = out.get("status") not in ("not_found", "skipped")
        return out

    def forget_node(
        self, node_id: str, reason: str = "",
        construct_id: str = "", actor: str = "api:forget",
        everywhere: bool = False,
    ) -> Dict:
        """The forget verb. ``reason`` and ``construct_id`` are accepted for
        compatibility and deliberately not stored. See ForgetEngine."""
        return self._forget_engine().forget(
            node_id, actor=actor, everywhere=everywhere)

    def delete_node(self, node_id: str, actor: str = "api:delete",
                    op: str = "delete") -> bool:
        """The delete verb: remove the node and its edges now. Leaves one
        content-free audit entry (node id, actor, time) and NO suppression
        record: if its source still holds the content, it can be captured
        again."""
        if self.store.get_node(node_id) is None:
            return False
        now = datetime.now(timezone.utc)
        with self.store.transaction():
            self.store.record_audit(
                node_id, op, actor=actor, before=None,
                after={"original_node_id": node_id, "verb": op,
                       "deleted_at": now.isoformat()},
                ts=now)
            return self.store.delete_node(node_id)

    # ── Governance Layer (leg 6b): retention sweep ───────────

    def _is_stale(self, node: Node, cutoff: datetime) -> bool:
        """A node is stale when its freshness clock predates the cutoff.

        Freshness clock = last_referenced if set, else created_at. Access /
        reinforce update last_referenced, so using a node resets its clock and
        keeps it out of the sweep. Pinned nodes and tombstone markers are NEVER
        stale (handled by the caller's immunity guard).
        """
        clock = node.last_referenced or node.created_at
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        return clock < cutoff

    def apply_retention(
        self,
        mode: Optional[str] = None,
        days: Optional[int] = None,
        construct_id: str = "",
    ) -> Dict:
        """A retention sweep a person asked for (POST /v1/retention/sweep).
        That is the DELETE verb: ``expire`` deletes stale nodes, each with a
        delete audit entry, and leaves no suppression record."""
        return self._retention_sweep(mode, days, op="delete",
                                     actor="api:retention_sweep")

    def run_scheduled_cleanup(
        self, mode: Optional[str] = None, days: Optional[int] = None,
    ) -> Dict:
        """The same sweep run unattended. That is automatic CLEANUP: its own
        audit op, no suppression record. Nothing schedules it today; a
        scheduler that does must call this, not apply_retention."""
        return self._retention_sweep(mode, days, op="cleanup",
                                     actor="scheduler:retention")

    def _retention_sweep(self, mode, days, op: str, actor: str) -> Dict:
        """Run one retention sweep under the selected storage policy.

        "Choose your storage." Resolves mode/window from the environment unless
        overridden by args.

          keep    - no-op. Nothing is removed (decay still only demotes).
          archive - soft-invalidate stale, unpinned, live nodes (REUSES leg-6a
                    ``invalidate_node``). Recoverable.
          expire  - HARD-delete those same stale nodes with an audit entry
                    under ``op``. The ONLY auto-destructive mode; opt-in.

        Pinned nodes (and tombstone markers) are ALWAYS immune. Returns counts:
        {"mode", "window_days", "scanned", "archived", "expired"}.
        """
        mode = (mode or get_retention_mode()).strip().lower()
        if mode not in VALID_RETENTION_MODES:
            mode = DEFAULT_RETENTION_MODE
        window = days if (days is not None and days > 0) else get_retention_days()

        result = {"mode": mode, "window_days": window,
                  "scanned": 0, "archived": 0, "expired": 0}
        if mode == RETENTION_KEEP:
            return result  # no-op sweep - nothing removed

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window)

        candidates = self.store.list_nodes(limit=999999)
        result["scanned"] = len(candidates)

        stale_ids: List[str] = []
        for node in candidates:
            if node.pinned:
                continue  # pinned is always immune (covers tombstone markers)
            if node.node_id.startswith(self.TOMBSTONE_PREFIX):
                continue  # belt-and-suspenders: never sweep a gravestone
            if mode == RETENTION_ARCHIVE and node.invalidated_at is not None:
                continue  # already archived - don't re-touch
            if self._is_stale(node, cutoff):
                stale_ids.append(node.node_id)

        if mode == RETENTION_ARCHIVE:
            for nid in stale_ids:
                updated = self.invalidate_node(
                    nid, reason="retention_archive", construct_id=actor
                )
                if updated is not None:
                    result["archived"] += 1
        elif mode == RETENTION_EXPIRE:
            for nid in stale_ids:
                if self.delete_node(nid, actor=actor, op=op):
                    result["expired"] += 1

        return result

    def export_everything(self) -> Dict:
        """Full portable snapshot: graph (nodes + edges) + the audit log.

        "Your data, portable." JSON-safe dict suitable for a single response
        body or a file dump. Reuses the store's graph export and audit reads.
        """
        graph = self.store.export_graph()
        return {
            "graph": graph.model_dump(mode="json"),
            "audit": self.store.get_all_audit(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }

    def propagate_confidence(
        self, source_node_id: str, max_hops: int = 2
    ) -> Dict[str, float]:
        """Propagate confidence from a source node to connected nodes.

        Confidence decays per hop and is capped at 2 hops max (bounded
        propagation). Returns dict of {node_id: new_confidence} for the nodes
        whose confidence would be raised by propagation.
        """
        if max_hops > 2:
            max_hops = 2

        source = self.store.get_node(source_node_id)
        if source is None:
            return {}

        updates: Dict[str, float] = {}
        visited = {source_node_id}
        frontier = [(source_node_id, source.confidence, 0)]  # (node_id, confidence, depth)

        while frontier:
            current_id, current_conf, depth = frontier.pop(0)
            if depth >= max_hops:
                continue

            edges = self.store.get_edges_for_node(current_id)
            for edge in edges:
                neighbor_id = (
                    edge.target_node_id
                    if edge.source_node_id == current_id
                    else edge.source_node_id
                )

                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # Propagated confidence decays by 0.1 per hop.
                propagated = current_conf * (1.0 - (0.1 * (depth + 1)))
                propagated = max(0.0, min(1.0, propagated))

                neighbor = self.store.get_node(neighbor_id)
                if neighbor and propagated > neighbor.confidence:
                    updates[neighbor_id] = propagated
                    frontier.append((neighbor_id, propagated, depth + 1))

        # Apply updates (mark propagated; confidence raise tracked in updates dict).
        for node_id in updates:
            node = self.store.get_node(node_id)
            if node is not None:
                self.store.update_node(
                    node_id,
                    metadata={**node.metadata, "_propagated": True},
                )

        return updates

    def retrieve_with_confidence(
        self,
        node_type: Optional[NodeType] = None,
        source_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict], List[float]]:
        """Retrieve nodes with confidence scoring.

        Applies lazy decay, then returns (nodes_as_dicts, confidence_scores)
        sorted by confidence descending.
        """
        raw_nodes = self.store.list_nodes(
            node_type=node_type, source_id=source_id, limit=limit, offset=offset
        )

        nodes_with_scores = []
        for node in raw_nodes:
            # Pure read: compute effective confidence without persisting/auditing.
            score = self._compute_decayed_confidence(node)
            view = node.model_dump()
            view["confidence"] = score
            nodes_with_scores.append((view, score))

        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        nodes = [n for n, _ in nodes_with_scores]
        scores = [s for _, s in nodes_with_scores]
        return nodes, scores

    def find_node_by_label(
        self, label: str, node_type: Optional[NodeType] = None,
        origin_runtime: Optional[Union[str, List[str]]] = None,
        project_key: Optional[str] = None,
    ) -> Optional[Node]:
        """Find a node by NORMALIZED label match, optionally filtered by type.

        Normalized, not raw-lowercase: 'Fernweh-Core' and 'Fernweh Core'
        are the same entity in different surface forms, and this lookup is
        where dedup, anchor selection, and link resolution all meet. The
        entity-fragmentation this closes was the top cause of unattached
        cross-corpus claims (vault eval attachment track).

        origin_runtime/project_key (WS0 Leg B): the SAME SQL prefilter as
        store.list_nodes, so a filtered recall's entity-anchor lookup never
        considers another runtime's nodes at all."""
        from revien.graph.normalize import normalize_label
        target = normalize_label(label)
        nodes = self.store.list_nodes(
            node_type=node_type, limit=999999,
            origin_runtime=origin_runtime, project_key=project_key,
        )
        for node in nodes:
            if normalize_label(node.label) == target:
                return node
        return None

    def find_nodes_by_label_fuzzy(
        self, label: str, max_distance: int = 5, min_ratio: float = 0.75,
        origin_runtime: Optional[Union[str, List[str]]] = None,
        project_key: Optional[str] = None,
    ) -> List[Node]:
        """Find nodes with similar labels using both Levenshtein distance
        and ratio-based matching. Ratio-based matching handles length
        differences better (e.g., 'PostgreSQL' vs 'Postgres'). Labels are
        normalized first so separator/case noise doesn't eat the distance
        budget.

        origin_runtime/project_key (WS0 Leg B): SQL prefilter, see
        find_node_by_label above."""
        from difflib import SequenceMatcher
        from revien.graph.normalize import normalize_label
        nodes = self.store.list_nodes(
            limit=999999, origin_runtime=origin_runtime, project_key=project_key,
        )
        matches = []
        target = normalize_label(label)
        for node in nodes:
            node_norm = normalize_label(node.label)
            dist = _levenshtein(node_norm, target)
            ratio = SequenceMatcher(None, node_norm, target).ratio()
            if dist < max_distance or ratio >= min_ratio:
                matches.append(node)
        return matches

    def connect_nodes(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 0.5,
    ) -> Edge:
        """Create an edge between two existing nodes."""
        edge = Edge(
            edge_type=edge_type,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=weight,
        )
        return self.store.add_edge(edge)

    def touch_node(self, node_id: str) -> Optional[Node]:
        """Increment access_count and update last_accessed."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        # Bookkeeping bump (access_count / last_accessed). Audit of access is
        # owned by the engine's mark_used hook (op="access"); suppress the
        # generic "update" entry here to avoid flooding the trail on every recall.
        return self.store.update_node(
            node_id,
            access_count=node.access_count + 1,
            last_accessed=datetime.now(timezone.utc),
            _audit_op=None,
        )

    def get_node_with_edges(self, node_id: str) -> Optional[Dict]:
        """Get a node with all its edges and connected node summaries."""
        node = self.store.get_node(node_id)
        if node is None:
            return None
        edges = self.store.get_edges_for_node(node_id)
        connected = []
        for edge in edges:
            other_id = (
                edge.target_node_id
                if edge.source_node_id == node_id
                else edge.source_node_id
            )
            other_node = self.store.get_node(other_id)
            if other_node:
                connected.append(
                    {
                        "node_id": other_node.node_id,
                        "node_type": other_node.node_type.value,
                        "label": other_node.label,
                        "edge_type": edge.edge_type.value,
                        "edge_weight": edge.weight,
                    }
                )
        return {
            "node": node.model_dump(),
            "edges": [e.model_dump() for e in edges],
            "connected_nodes": connected,
        }

    def get_subgraph(self, node_id: str, max_depth: int = 2) -> Graph:
        """Extract a subgraph around a node up to max_depth hops."""
        visited_nodes = set()
        visited_edges = set()
        frontier = {node_id}

        for _ in range(max_depth + 1):
            next_frontier = set()
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                visited_nodes.add(nid)
                edges = self.store.get_edges_for_node(nid)
                for edge in edges:
                    visited_edges.add(edge.edge_id)
                    other_id = (
                        edge.target_node_id
                        if edge.source_node_id == nid
                        else edge.source_node_id
                    )
                    if other_id not in visited_nodes:
                        next_frontier.add(other_id)
            frontier = next_frontier

        nodes = []
        for nid in visited_nodes:
            node = self.store.get_node(nid)
            if node:
                nodes.append(node)

        edges = []
        for eid in visited_edges:
            edge = self.store.get_edge(eid)
            if edge:
                edges.append(edge)

        return Graph(nodes=nodes, edges=edges)


def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

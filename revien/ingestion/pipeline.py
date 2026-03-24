"""
Revien Ingestion Pipeline — Orchestrates extraction, deduplication, and storage.
The main entry point for feeding content into the graph.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.graph.operations import GraphOperations
from .extractor import ExtractionResult, RuleBasedExtractor
from .dedup import Deduplicator


@dataclass
class IngestionInput:
    """Standard envelope for content ingestion."""
    source_id: str
    content: str
    content_type: str = "conversation"  # conversation | document | note | code
    timestamp: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class IngestionOutput:
    """Result of an ingestion operation."""
    context_node_id: str
    nodes_created: int
    nodes_deduplicated: int
    edges_created: int
    total_nodes_in_graph: int
    total_edges_in_graph: int


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    1. Extract nodes and edges from raw content
    2. Deduplicate nodes against existing graph
    3. Store new nodes and edges
    4. Return summary of what happened
    """

    def __init__(self, store: GraphStore):
        self.store = store
        self.ops = GraphOperations(store)
        self.extractor = RuleBasedExtractor()
        self.dedup = Deduplicator(store, self.ops)

    def ingest(self, input_data: IngestionInput) -> IngestionOutput:
        """
        Ingest raw content into the graph.
        Returns a summary of what was created/deduplicated.
        """
        # 1. Extract nodes and edges
        extraction = self.extractor.extract(
            content=input_data.content,
            source_id=input_data.source_id,
        )

        # 2. Deduplicate and store nodes, building ID mapping
        id_map = {}  # old_id -> actual_id (may differ if deduplicated)
        nodes_created = 0
        nodes_deduped = 0

        for candidate_node in extraction.nodes:
            old_id = candidate_node.node_id
            actual_node, is_new = self.dedup.deduplicate_node(candidate_node)
            id_map[old_id] = actual_node.node_id
            if is_new:
                nodes_created += 1
            else:
                nodes_deduped += 1

        # 3. Store edges with remapped IDs
        edges_created = 0
        for edge in extraction.edges:
            remapped_source = id_map.get(edge.source_node_id, edge.source_node_id)
            remapped_target = id_map.get(edge.target_node_id, edge.target_node_id)

            # Skip self-edges
            if remapped_source == remapped_target:
                continue

            # Skip duplicate edges
            if self._edge_exists(remapped_source, remapped_target, edge.edge_type):
                continue

            new_edge = Edge(
                edge_type=edge.edge_type,
                source_node_id=remapped_source,
                target_node_id=remapped_target,
                weight=edge.weight,
            )
            self.store.add_edge(new_edge)
            edges_created += 1

        # 4. Get context node ID from mapping
        context_id = id_map.get(
            extraction.context_node.node_id,
            extraction.context_node.node_id,
        )

        return IngestionOutput(
            context_node_id=context_id,
            nodes_created=nodes_created,
            nodes_deduplicated=nodes_deduped,
            edges_created=edges_created,
            total_nodes_in_graph=self.store.count_nodes(),
            total_edges_in_graph=self.store.count_edges(),
        )

    def _edge_exists(
        self, source_id: str, target_id: str, edge_type: EdgeType
    ) -> bool:
        """Check if an edge with this source/target/type already exists."""
        edges = self.store.get_edges_for_node(source_id)
        for e in edges:
            if (
                e.source_node_id == source_id
                and e.target_node_id == target_id
                and e.edge_type == edge_type
            ):
                return True
        return False

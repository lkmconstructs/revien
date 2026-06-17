"""
Revien Graph Schema — Pydantic models for nodes, edges, and the graph.
Memory is a graph, not a vector store.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    ENTITY = "entity"
    TOPIC = "topic"
    DECISION = "decision"
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    CONTEXT = "context"


class EdgeType(str, Enum):
    RELATED_TO = "related_to"
    DECIDED_IN = "decided_in"
    MENTIONED_BY = "mentioned_by"
    DEPENDS_ON = "depends_on"
    FOLLOWED_BY = "followed_by"
    CONTRADICTS = "contradicts"
    # Confidence audit
    CORRECTS = "corrects"
    # Provenance / lineage (leg 6a): target is derived FROM source.
    DERIVED_FROM = "derived_from"


class SourceType(str, Enum):
    """Confidence source — what evidence backs the node/edge."""
    EXTRACTED = "extracted"  # Ground truth from user or other source, confidence 1.0
    INFERRED = "inferred"  # Pattern-derived, confidence 0.5-0.8
    DERIVED = "derived"  # Computed from graph, inherits parent confidence
    CORRECTED = "corrected"  # Contradicted by later information, confidence 0.0


class Node(BaseModel):
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    label: str = Field(max_length=200)
    content: str
    source_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Confidence Layer
    source_type: SourceType = SourceType.INFERRED  # How the node was created
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Confidence 0.0-1.0
    pinned: bool = False  # Immune to decay if True
    confidence_set_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_set_by: str = ""  # Which construct set this confidence
    source_context: str = ""  # Why this confidence value
    last_referenced: Optional[datetime] = None  # For freshness/decay checks

    # Provenance Layer (leg 6a): soft-invalidation. Non-destructive staleness
    # marker — content is RETAINED. NULL means live; a timestamp means the node
    # has been marked stale (superseded/archived/corrected) and is excluded from
    # default recall. This is NOT deletion and NOT forget.
    invalidated_at: Optional[datetime] = None


class Edge(BaseModel):
    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Confidence Layer
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Edge-level confidence
    confidence_set_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_set_by: str = ""  # Which construct validated this edge
    source_context: str = ""  # Why this edge confidence


class Graph(BaseModel):
    """Full graph export/import format."""
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"

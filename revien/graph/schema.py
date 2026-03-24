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


class Edge(BaseModel):
    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edge_type: EdgeType
    source_node_id: str
    target_node_id: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Graph(BaseModel):
    """Full graph export/import format."""
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"

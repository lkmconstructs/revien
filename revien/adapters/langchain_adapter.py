# Copyright 2025 LKM Constructs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Revien LangChain Memory Adapter — Drop-in replacement for BaseMemory.

Uses Revien's graph-based retrieval instead of compaction-based summarization.
Relevance over recency. Every conversation is preserved in the graph forever.

Usage:
    from revien.adapters.langchain_adapter import RevienMemory

    memory = RevienMemory(graph_path="./my_revien_graph")
    chain = ConversationChain(llm=llm, memory=memory)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_core.memory import BaseMemory
    from pydantic import Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object  # type: ignore
    Field = None  # type: ignore


from revien.graph.store import GraphStore
from revien.graph.schema import EdgeType, NodeType
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine


class _MissingLangChainStub:
    """Stub for when langchain-core is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "langchain-core is required to use RevienMemory. "
            "Install it with: pip install langchain-core"
        )


class RevienMemory(BaseMemory if LANGCHAIN_AVAILABLE else _MissingLangChainStub):
    """
    LangChain-compatible memory backend powered by Revien's graph-based retrieval.

    Unlike traditional memory backends that compress old messages or keep only recent ones,
    RevienMemory stores every piece of conversational context in a persistent graph
    and retrieves what's RELEVANT to the current query, not what's RECENT.

    Three-factor scoring combines:
    - Recency: How fresh is this information?
    - Frequency: How often has it been accessed?
    - Proximity: How close is it in the graph to the current query?

    Attributes:
        store: Underlying GraphStore instance.
        retrieval_engine: RetrievalEngine for querying the graph.
        ingestion_pipeline: IngestionPipeline for storing conversations.
        session_scope: Optional session ID to isolate memory within a session.
        top_n: Number of results to retrieve per query (default 5).
        min_score: Minimum relevance threshold (default 0.01).
    """

    # LangChain BaseMemory compatibility
    human_prefix: str = "Human"
    ai_prefix: str = "AI"

    if LANGCHAIN_AVAILABLE:
        graph_path: str = Field(default="revien.db")
        session_scope: Optional[str] = Field(default=None)
        top_n: int = Field(default=5, ge=1, le=20)
        min_score: float = Field(default=0.01, ge=0.0, le=1.0)
        _store: GraphStore = None  # type: ignore
        _retrieval_engine: RetrievalEngine = None  # type: ignore
        _ingestion_pipeline: IngestionPipeline = None  # type: ignore

        class Config:
            arbitrary_types_allowed = True

    def __init__(
        self,
        graph_path: str = "revien.db",
        session_scope: Optional[str] = None,
        top_n: int = 5,
        min_score: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """
        Initialize RevienMemory.

        Args:
            graph_path: Path to the Revien SQLite database.
            session_scope: Optional session identifier. If set, memory operations
                are scoped to this session (stored in node metadata).
            top_n: Maximum number of relevant context nodes to retrieve (1-20).
            min_score: Minimum composite relevance score threshold (0.0-1.0).
            **kwargs: Additional arguments passed to BaseMemory (if available).
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required to use RevienMemory. "
                "Install it with: pip install langchain-core"
            )

        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(graph_path, str):
            raise ValueError("graph_path must be a string")
        if not (1 <= top_n <= 20):
            raise ValueError("top_n must be between 1 and 20")
        if not (0.0 <= min_score <= 1.0):
            raise ValueError("min_score must be between 0.0 and 1.0")

        self.graph_path = str(graph_path)
        self.session_scope = session_scope
        self.top_n = top_n
        self.min_score = min_score

        # Initialize graph infrastructure
        self._store = GraphStore(self.graph_path)
        self._retrieval_engine = RetrievalEngine(self._store)
        self._ingestion_pipeline = IngestionPipeline(self._store)

    @property
    def memory_variables(self) -> List[str]:
        """
        Return the list of memory variable names.

        LangChain expects this to be a list of strings that correspond
        to keys in the dictionary returned by load_memory_variables().

        Returns:
            ["history"] — a single variable containing the retrieved context.
        """
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        Load and return relevant context from the graph.

        This is the core advantage of graph-based memory: instead of returning
        the last N messages, we query the graph for what's RELEVANT to the
        current input.

        Args:
            inputs: Dictionary of input variables. The first value should be
                the user's current input/query.

        Returns:
            {"history": "<formatted context nodes>"}
        """
        # Extract the current user input from the inputs dict
        # Convention: first non-memory-variable input is the query
        current_input = self._extract_query_from_inputs(inputs)

        if not current_input:
            return {"history": ""}

        # Query the graph for relevant context
        try:
            response = self._retrieval_engine.recall(
                query=current_input,
                top_n=self.top_n,
                min_score=self.min_score,
            )
        except Exception as e:
            # Graceful degradation: if retrieval fails, return empty history
            return {"history": ""}

        # Format results as a readable context block
        history_text = self._format_retrieval_response(response)
        return {"history": history_text}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the input and output of an LLM call to the graph.

        Creates two nodes (input + output) connected by a FOLLOWED_BY edge,
        both tagged with the session scope if provided.

        Args:
            inputs: Input variables from the chain (contains user message).
            outputs: Output variables from the chain (contains AI response).
        """
        # Extract messages
        user_input = self._extract_query_from_inputs(inputs)
        ai_output = outputs.get("output", outputs.get("text", ""))

        if not user_input:
            return  # Nothing to save

        try:
            # Build metadata for session tracking
            metadata = {}
            if self.session_scope:
                metadata["session_id"] = self.session_scope

            # Create input node
            input_ingestion = IngestionInput(
                source_id=self.session_scope or "langchain",
                content=user_input,
                content_type="conversation",
                timestamp=datetime.now(timezone.utc),
                metadata={**metadata, "role": "user"},
            )

            input_result = self._ingestion_pipeline.ingest(input_ingestion)
            input_node_id = input_result.context_node_id

            # Create output node
            output_ingestion = IngestionInput(
                source_id=self.session_scope or "langchain",
                content=ai_output,
                content_type="conversation",
                timestamp=datetime.now(timezone.utc),
                metadata={**metadata, "role": "assistant"},
            )

            output_result = self._ingestion_pipeline.ingest(output_ingestion)
            output_node_id = output_result.context_node_id

            # Connect input → output with FOLLOWED_BY edge
            if input_node_id and output_node_id:
                self._ingestion_pipeline.store.add_edge(
                    self._create_edge_for_conversation(
                        input_node_id, output_node_id
                    )
                )
        except Exception:
            # Graceful degradation: if save fails, continue
            pass

    def clear(self) -> None:
        """
        Clear all memory from the graph.

        WARNING: This deletes all nodes and edges. Use with caution.
        Consider using session_scope to isolate conversations instead.
        """
        if not self._store:
            return

        try:
            # Get all nodes
            all_nodes = self._store.list_nodes(limit=999999)
            # Delete each node (edges cascade automatically)
            for node in all_nodes:
                self._store.delete_node(node.node_id)
        except Exception:
            pass

    def __del__(self) -> None:
        """Clean up database connection on deletion."""
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass

    # ── Private Helpers ────────────────────────────────────────

    def _extract_query_from_inputs(self, inputs: Dict[str, Any]) -> str:
        """
        Extract the user query from LangChain inputs dict.

        Convention: looks for 'input', 'question', 'query', or the first
        non-memory-variable string value.

        Args:
            inputs: Dictionary of input variables.

        Returns:
            The extracted query string, or empty string if none found.
        """
        # Common keys in LangChain chains
        for key in ("input", "question", "query", "text"):
            if key in inputs:
                val = inputs[key]
                if isinstance(val, str):
                    return val.strip()

        # Fallback: find the first string value
        for val in inputs.values():
            if isinstance(val, str):
                return val.strip()

        return ""

    def _format_retrieval_response(self, response: Any) -> str:
        """
        Format retrieval results into readable context for the LLM.

        Args:
            response: RetrievalResponse from the retrieval engine.

        Returns:
            Formatted string with retrieved context nodes.
        """
        if not response.results:
            return ""

        lines = [
            f"## Relevant Context (from {len(response.results)} nodes)\n"
        ]

        for i, result in enumerate(response.results, 1):
            lines.append(f"### Result {i}: {result.label}")
            lines.append(f"Type: {result.node_type} | Score: {result.score:.3f}")

            if result.score_breakdown:
                breakdown = result.score_breakdown
                lines.append(
                    f"Factors: "
                    f"recency={breakdown.get('recency', 0):.3f}, "
                    f"frequency={breakdown.get('frequency', 0):.3f}, "
                    f"proximity={breakdown.get('proximity', 0):.3f}"
                )

            if result.path:
                lines.append(f"Path: {' → '.join(result.path)}")

            lines.append(f"\n{result.content}\n")

        lines.append(f"[Retrieved in {response.retrieval_time_ms:.2f}ms]")
        return "\n".join(lines)

    def _create_edge_for_conversation(
        self, source_id: str, target_id: str
    ) -> Any:
        """
        Create an edge connecting two conversation nodes.

        Args:
            source_id: ID of the input node.
            target_id: ID of the output node.

        Returns:
            Edge object configured for conversational flow.
        """
        # Import here to avoid circular imports
        from revien.graph.schema import Edge

        return Edge(
            edge_type=EdgeType.FOLLOWED_BY,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=1.0,
        )

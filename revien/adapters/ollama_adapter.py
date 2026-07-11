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
Revien Ollama Adapter — Bridges Revien's graph memory to locally-running Ollama models.
Injects relevant graph context into Ollama prompts automatically.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

from revien.graph.schema import Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline, IngestionOutput
from revien.retrieval.engine import RetrievalEngine, RetrievalResponse


class OllamaAdapter:
    """
    Bridges Revien's graph memory to locally-running Ollama models.
    Injects relevant graph context into Ollama prompts automatically.

    The adapter maintains a persistent graph of conversations and decisions,
    retrieving and injecting relevant context before each API call.
    """

    def __init__(
        self,
        graph_path: str,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the Ollama adapter with graph memory.

        Args:
            graph_path: Path to the SQLite graph database.
            ollama_host: Ollama server URL (default: http://localhost:11434).
            model: Model name to use for inference (default: llama3).
            timeout: HTTP request timeout in seconds (default: 30.0).

        Raises:
            FileNotFoundError: If the graph path parent directory doesn't exist.
        """
        self.graph_path = graph_path
        self.ollama_host = ollama_host
        self.model = model
        self.timeout = timeout

        # Initialize Revien components
        self.store = GraphStore(db_path=graph_path)
        self.pipeline = IngestionPipeline(self.store)
        self.retrieval = RetrievalEngine(self.store)

        # HTTP client for Ollama communication (lazy init for environments
        # where proxy settings may interfere with client creation)
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            try:
                self._client = httpx.Client(timeout=self.timeout)
            except (ImportError, Exception):
                # Fallback: create client without proxy support
                self._client = httpx.Client(
                    timeout=self.timeout,
                    proxy=None,
                )
        return self._client

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        max_context_nodes: int = 5,
    ) -> str:
        """
        Send a message to Ollama with automatic Revien context injection.

        Retrieves relevant context from the memory graph, injects it into
        the system prompt, sends to Ollama, and stores both message and
        response in the graph for future retrieval.

        Args:
            message: The user message to send.
            system_prompt: Base system prompt. If None, a minimal prompt is used.
            max_context_nodes: Maximum nodes to retrieve as context (default: 5).

        Returns:
            The model's response as a string.

        Raises:
            RuntimeError: If Ollama is unreachable.
            httpx.HTTPError: For other network errors.
        """
        # Retrieve context from graph
        context_str = self.get_context_for_prompt(
            message, max_nodes=max_context_nodes
        )

        # Build system prompt with injected context
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        if context_str:
            full_system_prompt = f"{system_prompt}\n\n{context_str}"
        else:
            full_system_prompt = system_prompt

        # Call Ollama
        response_text = self._call_ollama(
            message=message,
            system_prompt=full_system_prompt,
        )

        # Ingest message and response into graph
        self._ingest_exchange(message, response_text)

        return response_text

    def ingest_ollama_history(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Ingest a chat history into the memory graph.

        Takes a list of message dicts with 'role' (user/assistant) and
        'content' fields and ingests them as a conversation into the graph.

        Args:
            history: List of dicts with 'role' and 'content' keys.
                     Example: [
                         {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi there!"}
                     ]

        Returns:
            Dictionary with ingestion statistics:
                - total_messages: int
                - nodes_created: int
                - nodes_deduplicated: int
                - edges_created: int
                - total_nodes_in_graph: int
                - total_edges_in_graph: int
        """
        if not history:
            return {
                "total_messages": 0,
                "nodes_created": 0,
                "nodes_deduplicated": 0,
                "edges_created": 0,
                "total_nodes_in_graph": self.store.count_nodes(),
                "total_edges_in_graph": self.store.count_edges(),
            }

        # Combine all messages into a single conversation string
        conversation_lines = []
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            conversation_lines.append(f"{role}: {content}")

        full_conversation = "\n".join(conversation_lines)

        # Ingest as a single conversation
        input_data = IngestionInput(
            source_id="ollama_history",
            content=full_conversation,
            content_type="conversation",
            timestamp=datetime.now(timezone.utc),
            metadata={"message_count": len(history)},
        )

        output = self.pipeline.ingest(input_data)

        return {
            "total_messages": len(history),
            "nodes_created": output.nodes_created,
            "nodes_deduplicated": output.nodes_deduplicated,
            "edges_created": output.edges_created,
            "total_nodes_in_graph": output.total_nodes_in_graph,
            "total_edges_in_graph": output.total_edges_in_graph,
        }

    def get_context_for_prompt(
        self, query: str, max_nodes: int = 5
    ) -> str:
        """
        Query the Revien graph and format results as a context block.

        Retrieves relevant nodes based on the query and formats them
        as a readable context block suitable for injection into a prompt.

        Args:
            query: The query string (typically a user message).
            max_nodes: Maximum nodes to return (default: 5).

        Returns:
            Formatted context string, or empty string if no results found.
        """
        response: RetrievalResponse = self.retrieval.recall(
            query, top_n=max_nodes
        )

        if not response.results:
            return ""

        lines = [
            "[Revien Memory Context]",
            "The following context is retrieved from persistent memory based on relevance to the current query:\n",
        ]

        for result in response.results:
            # Format score as percentage
            score_pct = int(result.score * 100)

            # Format time delta
            now = datetime.now(timezone.utc)
            node = self.store.get_node(result.node_id)
            if node:
                time_delta = self._format_time_delta(node.created_at, now)
            else:
                time_delta = "unknown time"

            # Build context line
            label = result.label or "Untitled"
            content_preview = result.content[:100]
            if len(result.content) > 100:
                content_preview += "..."

            line = f"- [Score: {score_pct}%] ({time_delta}) {label}: {content_preview}"
            lines.append(line)

        lines.append("\n[End Memory Context]")
        return "\n".join(lines)

    def health_check(self) -> bool:
        """
        Check if Ollama is reachable at the configured host.

        Returns:
            True if Ollama is responding, False otherwise.
        """
        try:
            response = self._get_client().get(
                f"{self.ollama_host}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return False

    # ── Private Methods ───────────────────────────────────────

    def _call_ollama(
        self,
        message: str,
        system_prompt: str,
    ) -> str:
        """
        Send a request to Ollama /api/chat endpoint.

        Args:
            message: The user message.
            system_prompt: The system prompt with injected context.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If Ollama is unreachable or returns an error.
            httpx.HTTPError: For network errors.
        """
        url = f"{self.ollama_host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "stream": False,
        }

        try:
            response = self._get_client().post(url, json=payload)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_host}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama request timed out ({self.timeout}s). "
                f"Server may be slow or unresponsive. Error: {e}"
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse Ollama response: {e}"
            ) from e

    def _ingest_exchange(self, message: str, response: str) -> None:
        """
        Ingest a user message and model response into the graph.

        Creates a conversation entry in the graph for later retrieval.

        Args:
            message: The user message.
            response: The model response.
        """
        now = datetime.now(timezone.utc)
        exchange = f"User: {message}\nAssistant: {response}"

        input_data = IngestionInput(
            source_id="ollama_chat",
            content=exchange,
            content_type="conversation",
            timestamp=now,
            metadata={
                "role": "exchange",
                "message_length": len(message),
                "response_length": len(response),
            },
        )

        try:
            self.pipeline.ingest(input_data)
        except Exception as e:
            # Log but don't fail — ingestion errors shouldn't break the chat
            logger.warning(f"Failed to ingest exchange into graph: {e}")

    def _format_time_delta(self, created_at: datetime, now: datetime) -> str:
        """
        Format a time delta as human-readable relative time.

        Args:
            created_at: The creation timestamp.
            now: The reference timestamp (usually now).

        Returns:
            Human-readable time delta (e.g., "2 days ago", "1 hour ago").
        """
        delta = now - created_at
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return "just now"
        if total_seconds < 3600:
            minutes = total_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        if total_seconds < 86400:
            hours = total_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if total_seconds < 604800:
            days = total_seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"

        weeks = total_seconds // 604800
        if weeks < 4:
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"

        months = total_seconds // 2592000
        if months < 12:
            return f"{months} month{'s' if months != 1 else ''} ago"

        years = total_seconds // 31536000
        return f"{years} year{'s' if years != 1 else ''} ago"

    def close(self) -> None:
        """
        Clean up resources (close HTTP client and graph store).

        Should be called when done with the adapter to ensure
        database connections are properly closed.
        """
        if self._client:
            self._client.close()
        if self.store:
            self.store.close()

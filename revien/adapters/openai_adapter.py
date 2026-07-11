# Licensed under the Apache License, Version 2.0
# See LICENSE file in the project root for full license text.

"""
Revien OpenAI Adapter — Ingests ChatGPT conversation exports into Revien's graph.
Supports both single conversation JSON and bulk export format (conversations.json).

The adapter parses OpenAI's standard conversation export format, creates nodes for each
message, preserves parent→child threading relationships, and generates summary edges
for cross-conversation analysis.

Thread safety: Uses watchdog for directory monitoring; each adapter instance should
be used from a single thread or externally synchronized.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from difflib import SequenceMatcher

from revien.graph.schema import Edge, EdgeType, Node, NodeType
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    """
    Ingests OpenAI ChatGPT conversation exports into Revien's graph.
    Supports both single conversation JSON and bulk export format.

    A conversation is parsed into a tree of message nodes, preserving the full
    parent→child structure. Messages from different conversations can be linked
    via content similarity edges.

    Example usage:
        adapter = OpenAIAdapter(graph_path="revien.db")
        stats = adapter.ingest_conversation("path/to/conversation.json")
        print(f"Ingested {stats['node_count']} nodes, {stats['edge_count']} edges")
    """

    # Minimum similarity threshold for cross-conversation edges (0.0 to 1.0)
    _SIMILARITY_THRESHOLD = 0.6

    def __init__(self, graph_path: str = "revien.db") -> None:
        """
        Initialize the OpenAI adapter with a graph store.

        Args:
            graph_path: Path to the SQLite database. Created if it doesn't exist.
        """
        self.store = GraphStore(graph_path)
        self.pipeline = IngestionPipeline(self.store)
        self._processed_conversations: Dict[str, bool] = {}

    def ingest_conversation(self, filepath: str) -> Dict[str, int]:
        """
        Parse a single OpenAI conversation JSON and ingest it into the graph.

        The conversation is parsed into a tree of message nodes. Each message becomes
        a node with metadata about its author role and timestamps. Parent→child
        relationships are preserved as FOLLOWED_BY edges.

        Args:
            filepath: Path to a single conversation JSON file (OpenAI export format).

        Returns:
            Dictionary with keys:
                - node_count: Number of nodes created/deduplicated
                - edge_count: Number of edges created
                - conversation_id: OpenAI conversation ID (for dedup tracking)
                - message_count: Total messages processed
                - system_messages: Count of system messages
                - skipped_messages: Count of messages with non-text content

        Raises:
            FileNotFoundError: If filepath doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
            ValueError: If JSON structure doesn't match OpenAI format.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Conversation file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._ingest_conversation_data(data, filepath.stem)

    def ingest_bulk_export(self, filepath: str) -> Dict[str, int]:
        """
        Parse a bulk export file (conversations.json) and ingest all conversations.

        The bulk export is an array of conversation objects. Each conversation is
        ingested separately. Cross-conversation edges are created where message
        content similarity exceeds the threshold.

        Deduplication: Conversations are tracked by their OpenAI conversation_id
        (stored in node metadata). Already-ingested conversations are skipped
        to avoid duplicate nodes.

        Args:
            filepath: Path to conversations.json (array of conversations).

        Returns:
            Dictionary with aggregate statistics:
                - total_conversations: Number of conversations in the file
                - conversations_ingested: Number newly ingested
                - conversations_skipped: Number already in graph (deduped)
                - total_nodes: Aggregate nodes created across all
                - total_edges: Aggregate edges created
                - total_messages: Total messages across all conversations
                - cross_conversation_edges: Edges linking different conversations

        Raises:
            FileNotFoundError: If filepath doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Bulk export file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Bulk export must be a JSON array of conversations")

        total_nodes = 0
        total_edges = 0
        conversations_ingested = 0
        conversations_skipped = 0
        total_messages = 0
        cross_conv_edges = 0

        conversation_nodes: List[Node] = []

        for i, conv_data in enumerate(data):
            conv_id = self._extract_conversation_id(conv_data)

            # Deduplication: skip if already ingested
            if self._is_conversation_ingested(conv_id):
                conversations_skipped += 1
                logger.debug(f"Skipping already-ingested conversation: {conv_id}")
                continue

            stats = self._ingest_conversation_data(conv_data, f"bulk_conv_{i}")
            total_nodes += stats.get("node_count", 0)
            total_edges += stats.get("edge_count", 0)
            total_messages += stats.get("message_count", 0)
            conversations_ingested += 1

            # Track conversation nodes for cross-conversation linking
            self._processed_conversations[conv_id] = True
            source_id = f"openai:conversation:{conv_id}"
            conversation_nodes.extend(
                self.store.list_nodes(source_id=source_id, limit=9999)
            )

        # Create cross-conversation edges where content similarity is high
        cross_conv_edges = self._link_similar_conversations(conversation_nodes)
        total_edges += cross_conv_edges

        return {
            "total_conversations": len(data),
            "conversations_ingested": conversations_ingested,
            "conversations_skipped": conversations_skipped,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_messages": total_messages,
            "cross_conversation_edges": cross_conv_edges,
        }

    def watch_export_directory(
        self, dirpath: str, callback: Optional[Callable] = None
    ) -> None:
        """
        Watch a directory for new OpenAI export files and auto-ingest them.

        Monitors dirpath for new .json files matching OpenAI's naming conventions
        (either single conversation files or conversations.json). When new files
        are detected, they are ingested automatically.

        Deduplication by conversation ID prevents re-ingesting the same
        conversation if files are re-dropped or copied multiple times.

        This is a blocking call. In a production system, it should be run in
        a background thread or async context.

        Args:
            dirpath: Directory to watch for new exports.
            callback: Optional callable(filepath, stats) called after each ingest.
                      Useful for logging or triggering downstream actions.

        Raises:
            ImportError: If watchdog is not installed.
            FileNotFoundError: If dirpath doesn't exist.
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            raise ImportError(
                "watchdog is required for directory monitoring. "
                "Install with: pip install watchdog"
            )

        dirpath = Path(dirpath)
        if not dirpath.exists() or not dirpath.is_dir():
            raise FileNotFoundError(f"Watch directory not found: {dirpath}")

        class OpenAIExportHandler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                filepath = Path(event.src_path)
                if filepath.suffix != ".json":
                    return

                logger.info(f"Detected new OpenAI export: {filepath}")
                try:
                    # Determine if it's a bulk or single export
                    if filepath.name == "conversations.json":
                        stats = self.adapter.ingest_bulk_export(str(filepath))
                    else:
                        stats = self.adapter.ingest_conversation(str(filepath))

                    logger.info(f"Ingested {filepath}: {stats}")
                    if callback:
                        callback(str(filepath), stats)
                except Exception as e:
                    logger.error(f"Failed to ingest {filepath}: {e}", exc_info=True)

        handler = OpenAIExportHandler()
        handler.adapter = self
        observer = Observer()
        observer.schedule(handler, dirpath, recursive=False)
        observer.start()

        try:
            observer.join()
        except KeyboardInterrupt:
            observer.stop()
            observer.join()

    # ──────────────────────────────────────────────────────────────────────
    # Private Implementation
    # ──────────────────────────────────────────────────────────────────────

    def _ingest_conversation_data(
        self, conv_data: Dict, source_prefix: str
    ) -> Dict[str, int]:
        """
        Internal: Parse a conversation object and ingest all its messages.

        Args:
            conv_data: Parsed conversation object from OpenAI export.
            source_prefix: Prefix for source_id (used for tracking).

        Returns:
            Dictionary with node_count, edge_count, etc.
        """
        conv_id = self._extract_conversation_id(conv_data)
        source_id = f"openai:conversation:{conv_id}"

        # Check for dedup before processing
        if self._is_conversation_ingested(conv_id):
            return {
                "node_count": 0,
                "edge_count": 0,
                "conversation_id": conv_id,
                "message_count": 0,
                "system_messages": 0,
                "skipped_messages": 0,
            }

        # Extract title and timestamps
        title = conv_data.get("title", "Untitled")
        create_time = self._unix_to_datetime(conv_data.get("create_time"))
        update_time = self._unix_to_datetime(conv_data.get("update_time"))

        # Create a root context node for the conversation
        context_node = Node(
            node_type=NodeType.CONTEXT,
            label=title[:200],  # Enforce max_length on label
            content=f"OpenAI conversation: {title}",
            source_id=source_id,
            created_at=create_time,
            last_accessed=update_time,
            metadata={
                "conversation_id": conv_id,
                "title": title,
                "adapter": "openai",
            },
        )
        self.store.add_node(context_node)

        # Build message tree
        mapping = conv_data.get("mapping", {})
        if not mapping:
            return {
                "node_count": 1,
                "edge_count": 0,
                "conversation_id": conv_id,
                "message_count": 0,
                "system_messages": 0,
                "skipped_messages": 0,
            }

        node_map: Dict[str, Node] = {}
        edges_to_create: List[Edge] = []
        message_count = 0
        system_messages = 0
        skipped_messages = 0

        # First pass: create all message nodes
        for node_id, node_data in mapping.items():
            if not self._is_valid_message_node(node_data):
                skipped_messages += 1
                continue

            msg = node_data.get("message", {})
            author_role = msg.get("author", {}).get("role", "unknown")
            content_parts = msg.get("content", {}).get("parts", [])
            msg_timestamp = self._unix_to_datetime(msg.get("create_time"))

            # Handle empty or null content
            content = "".join(str(p) for p in content_parts if p) or ""
            if not content.strip():
                content = f"[Empty message from {author_role}]"

            # Flag system messages
            is_system = author_role == "system"
            if is_system:
                system_messages += 1

            # Create node
            msg_node = Node(
                node_type=self._classify_message(author_role, content),
                label=self._truncate_label(content, 200),
                content=content,
                source_id=source_id,
                created_at=msg_timestamp,
                last_accessed=msg_timestamp,
                metadata={
                    "conversation_id": conv_id,
                    "author_role": author_role,
                    "content_type": msg.get("content", {}).get("content_type", "text"),
                    "openai_message_id": msg.get("id", ""),
                    "is_system": is_system,
                },
            )
            self.store.add_node(msg_node)
            node_map[node_id] = msg_node
            message_count += 1

        # Second pass: create parent→child edges (FOLLOWED_BY)
        for node_id, node_data in mapping.items():
            if node_id not in node_map:
                continue

            parent_id = node_data.get("parent")
            if parent_id and parent_id in node_map:
                edge = Edge(
                    edge_type=EdgeType.FOLLOWED_BY,
                    source_node_id=node_map[parent_id].node_id,
                    target_node_id=node_map[node_id].node_id,
                    weight=0.9,
                    metadata={
                        "conversation_id": conv_id,
                        "relation_type": "message_thread",
                    },
                )
                self.store.add_edge(edge)
                edges_to_create.append(edge)
            else:
                # Root message: link to context node
                edge = Edge(
                    edge_type=EdgeType.MENTIONED_BY,
                    source_node_id=context_node.node_id,
                    target_node_id=node_map[node_id].node_id,
                    weight=0.8,
                    metadata={
                        "conversation_id": conv_id,
                        "relation_type": "root_message",
                    },
                )
                self.store.add_edge(edge)
                edges_to_create.append(edge)

        # Mark conversation as processed
        self._processed_conversations[conv_id] = True

        return {
            "node_count": len(node_map) + 1,  # +1 for context node
            "edge_count": len(edges_to_create),
            "conversation_id": conv_id,
            "message_count": message_count,
            "system_messages": system_messages,
            "skipped_messages": skipped_messages,
        }

    def _link_similar_conversations(self, nodes: List[Node]) -> int:
        """
        Create edges between nodes from different conversations based on content similarity.

        Only creates RELATED_TO edges where similarity exceeds the threshold.

        Args:
            nodes: List of nodes from all conversations being ingested.

        Returns:
            Number of cross-conversation edges created.
        """
        if len(nodes) < 2:
            return 0

        edges_created = 0
        checked = set()

        for i, node1 in enumerate(nodes):
            if not node1.content or len(node1.content) < 10:
                continue

            conv1 = node1.metadata.get("conversation_id", "")

            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                if not node2.content or len(node2.content) < 10:
                    continue

                conv2 = node2.metadata.get("conversation_id", "")

                # Only link nodes from different conversations
                if conv1 == conv2 or conv1 == "" or conv2 == "":
                    continue

                pair = (node1.node_id, node2.node_id)
                if pair in checked:
                    continue
                checked.add(pair)

                similarity = self._compute_similarity(node1.content, node2.content)
                if similarity >= self._SIMILARITY_THRESHOLD:
                    edge = Edge(
                        edge_type=EdgeType.RELATED_TO,
                        source_node_id=node1.node_id,
                        target_node_id=node2.node_id,
                        weight=similarity,
                        metadata={
                            "cross_conversation": True,
                            "similarity_score": similarity,
                        },
                    )
                    self.store.add_edge(edge)
                    edges_created += 1

        return edges_created

    @staticmethod
    def _extract_conversation_id(conv_data: Dict) -> str:
        """
        Extract the OpenAI conversation ID.

        Tries multiple keys where the ID might be stored.
        Falls back to generating a hash if not found.

        Args:
            conv_data: Conversation object.

        Returns:
            Conversation ID string.
        """
        if "id" in conv_data:
            return str(conv_data["id"])
        if "conversation_id" in conv_data:
            return str(conv_data["conversation_id"])

        # Fallback: generate from title + create_time
        title = conv_data.get("title", "unknown")
        create_time = conv_data.get("create_time", 0)
        fallback_id = f"{title}_{int(create_time)}"
        return fallback_id

    @staticmethod
    def _is_valid_message_node(node_data: Dict) -> bool:
        """
        Validate that a node in the OpenAI mapping is a valid message.

        Args:
            node_data: Node object from OpenAI mapping.

        Returns:
            True if it contains a message with author info.
        """
        msg = node_data.get("message")
        if not msg:
            return False
        author = msg.get("author")
        if not author or "role" not in author:
            return False
        return True

    @staticmethod
    def _unix_to_datetime(timestamp: Optional[float]) -> datetime:
        """
        Convert Unix timestamp (float seconds) to datetime in UTC.

        Args:
            timestamp: Unix timestamp (seconds since epoch).

        Returns:
            datetime object in UTC timezone.
        """
        if not timestamp:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            return datetime.now(timezone.utc)

    @staticmethod
    def _classify_message(author_role: str, content: str) -> NodeType:
        """
        Classify a message into a NodeType based on author role and content.

        Args:
            author_role: "user", "assistant", "system", "tool", etc.
            content: Message content string.

        Returns:
            Appropriate NodeType.
        """
        if author_role == "system":
            return NodeType.CONTEXT
        elif author_role == "user":
            return NodeType.FACT
        elif author_role == "assistant":
            return NodeType.TOPIC
        elif author_role == "tool":
            return NodeType.CONTEXT
        else:
            return NodeType.FACT

    @staticmethod
    def _truncate_label(text: str, max_length: int) -> str:
        """
        Truncate text to max_length characters, preserving words.

        Args:
            text: Input text.
            max_length: Maximum length.

        Returns:
            Truncated text.
        """
        if len(text) <= max_length:
            return text
        truncated = text[: max_length - 3]
        # Try to break at word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        return truncated + "..."

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        """
        Compute similarity between two text strings using SequenceMatcher.

        Uses Python's built-in difflib for simple, dependency-free similarity.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0
        # Normalize: lowercase, take first 500 chars to avoid long texts
        t1 = text1.lower()[:500]
        t2 = text2.lower()[:500]
        matcher = SequenceMatcher(None, t1, t2)
        return matcher.ratio()

    def _is_conversation_ingested(self, conversation_id: str) -> bool:
        """
        Check if a conversation has already been ingested (by ID).

        Checks both the in-session cache and the database.

        Args:
            conversation_id: OpenAI conversation ID.

        Returns:
            True if already ingested.
        """
        # Check in-session cache first
        if conversation_id in self._processed_conversations:
            return True

        # Check database: look for context nodes with this conversation_id
        source_id = f"openai:conversation:{conversation_id}"
        nodes = self.store.list_nodes(source_id=source_id, limit=1)
        if nodes:
            self._processed_conversations[conversation_id] = True
            return True

        return False

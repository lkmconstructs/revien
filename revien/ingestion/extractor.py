"""
Revien Ingestion Extractor — Rule-based NLP extraction of nodes and edges.
Extracts entities, topics, decisions, facts, preferences, and events from raw text.
No GPU. No external models. Pure pattern matching for MVP.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

from revien.graph.schema import Edge, EdgeType, Node, NodeType


@dataclass
class ExtractionResult:
    """Container for all nodes and edges extracted from a single content block."""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    context_node: Node = None


# ── Pattern Definitions ───────────────────────────────────

# Decision markers — phrases that signal a choice was made
DECISION_PATTERNS = [
    r"(?:we |I |let'?s )?(?:decided|decide) (?:to |on |that )?(.{10,200}?)(?:\.|$)",
    r"(?:let'?s |we'?ll )?go(?:ing)? with (.{10,200}?)(?:\.|$)",
    r"(?:the |our )?(?:plan|approach|decision|choice) is (.{10,200}?)(?:\.|$)",
    r"(?:chosen|selected|picked) (?:approach|option|strategy):? (.{10,200}?)(?:\.|$)",
    r"confirmed[.:]? (.{10,200}?)(?:\.|$)",
]

# Preference markers — phrases that signal user preferences
PREFERENCE_PATTERNS = [
    r"(?:I |we )?prefer(?:s)? (.{5,200}?)(?:\.|$)",
    r"always use (.{5,200}?)(?:\.|$)",
    r"(?:I |we )?don'?t (?:like|want|use) (.{5,200}?)(?:\.|$)",
    r"(?:I |we )?(?:like|love|enjoy) (?:using |to use )?(.{5,200}?)(?:\.|$)",
    r"(?:never|avoid) (?:use |using )?(.{5,200}?)(?:\.|$)",
]

# Fact patterns — specific data points, config values, requirements
FACT_PATTERNS = [
    r"(?:uses?|using|runs? on|deployed (?:on|to)|hosted (?:on|at)) (.{5,200}?)(?:\.|$)",
    r"(?:the |our )?(?:server|database|backend|frontend|api|service|port) (?:is |runs |uses )?(.{5,200}?)(?:\.|$)",
    r"(?:not |instead of |rather than )(\w+),?\s*(?:use |using |we use )(.{5,200}?)(?:\.|$)",
    r"(\$[\d,]+(?:/\w+)?(?:\s+with\s+.{5,100}?)?)(?:\.|,|$)",
    r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?)",  # IP addresses
    r"(?:port|PORT)\s*(?:=|:|\s)\s*(\d{2,5})",
]

# Event patterns — things that happened at a specific time
EVENT_PATTERNS = [
    r"(?:deployed|launched|released|shipped|published|merged|pushed) (.{5,200}?)(?:\.|$)",
    r"(?:on |last |this )?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|yesterday),?\s*(?:we )?(.*?)(?:\.|$)",
]

# Entity extraction — capitalized multi-word phrases, tech terms, quoted terms
ENTITY_PATTERNS = [
    r'"([A-Z][^"]{1,80})"',  # Quoted capitalized terms
    r"'([A-Z][^']{1,80})'",  # Single-quoted capitalized terms
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",  # Multi-word proper nouns
]

# Known tech terms that should be extracted as entities
TECH_ENTITIES = {
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
    "python", "javascript", "typescript", "rust", "go", "java",
    "fastapi", "flask", "django", "express", "react", "vue", "angular",
    "docker", "kubernetes", "helm", "aws", "gcp", "azure", "tailscale",
    "claude", "claude code", "chatgpt", "openai", "anthropic",
    "langchain", "langgraph", "ollama", "lm studio",
    "github", "gitlab", "vercel", "netlify",
    "linux", "ubuntu", "wsl", "wsl2",
    "onnx", "spacy", "pytorch", "tensorflow",
    "jwt", "oauth", "graphql", "rest api",
    "pytest", "vitest", "jest", "mocha",
    "vite", "webpack", "tailwind css", "styled-components",
    "pyjwt", "asyncpg", "psycopg2",
    "ci/cd", "helm charts",
}

# Stop words for topic extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "that", "this", "it", "i",
    "we", "you", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "what", "which", "who",
    "whom", "these", "those", "am", "about", "up", "also", "make", "sure",
    "let", "s", "re", "ve", "ll", "t", "d", "m",
    "user", "assistant", "yes", "no", "okay", "ok", "sure", "thanks",
    "thank", "please", "well", "going", "get", "got", "thing", "things",
}


class RuleBasedExtractor:
    """
    Rule-based NLP extractor for MVP.
    Extracts nodes and edges from raw text using regex patterns and heuristics.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self._decision_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in DECISION_PATTERNS]
        self._preference_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PREFERENCE_PATTERNS]
        self._fact_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in FACT_PATTERNS]
        self._event_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in EVENT_PATTERNS]
        self._entity_re = [re.compile(p, re.MULTILINE) for p in ENTITY_PATTERNS]

    def extract(self, content: str, source_id: str = "") -> ExtractionResult:
        """
        Extract all nodes and edges from raw content.
        Returns an ExtractionResult with context node, extracted nodes, and edges.
        """
        result = ExtractionResult()
        now = datetime.now(timezone.utc)

        # 1. Create context node for the full interaction
        context_node = Node(
            node_type=NodeType.CONTEXT,
            label=self._make_context_label(content),
            content=content,
            source_id=source_id,
            created_at=now,
            last_accessed=now,
        )
        result.context_node = context_node
        result.nodes.append(context_node)

        # 2. Extract entities
        entities = self._extract_entities(content, source_id, now)
        for node in entities:
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, EdgeType.MENTIONED_BY
            ))

        # 3. Extract decisions
        decisions = self._extract_decisions(content, source_id, now)
        for node in decisions:
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, EdgeType.DECIDED_IN
            ))

        # 4. Extract facts
        facts = self._extract_facts(content, source_id, now)
        for node in facts:
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, EdgeType.RELATED_TO
            ))

        # 5. Extract preferences
        preferences = self._extract_preferences(content, source_id, now)
        for node in preferences:
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, EdgeType.RELATED_TO
            ))

        # 6. Extract topics
        topics = self._extract_topics(content, source_id, now)
        for node in topics:
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, EdgeType.RELATED_TO
            ))

        # 7. Infer entity-entity edges (co-occurrence)
        entity_nodes = [n for n in result.nodes if n.node_type == NodeType.ENTITY]
        for i in range(len(entity_nodes)):
            for j in range(i + 1, len(entity_nodes)):
                result.edges.append(self._make_edge(
                    entity_nodes[i].node_id,
                    entity_nodes[j].node_id,
                    EdgeType.RELATED_TO,
                    weight=0.3,
                ))

        # 8. Connect decisions to mentioned entities
        decision_nodes = [n for n in result.nodes if n.node_type == NodeType.DECISION]
        for dec in decision_nodes:
            for ent in entity_nodes:
                if ent.label.lower() in dec.content.lower():
                    result.edges.append(self._make_edge(
                        dec.node_id, ent.node_id, EdgeType.RELATED_TO, weight=0.7
                    ))

        return result

    # ── Extraction Methods ────────────────────────────────

    def _extract_entities(self, content: str, source_id: str, now: datetime) -> List[Node]:
        entities = {}  # label_lower -> Node

        # Tech entities by keyword match
        content_lower = content.lower()
        for tech in TECH_ENTITIES:
            if tech in content_lower:
                # Find the actual casing in text
                idx = content_lower.index(tech)
                actual = content[idx:idx + len(tech)]
                label = actual.strip()
                if label.lower() not in entities:
                    entities[label.lower()] = Node(
                        node_type=NodeType.ENTITY,
                        label=label,
                        content=f"Technology/tool: {label}",
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                    )

        # Regex-based entity extraction
        for pattern in self._entity_re:
            for match in pattern.finditer(content):
                label = match.group(1).strip()
                if len(label) < 2 or label.lower() in STOP_WORDS:
                    continue
                if label.lower() not in entities:
                    entities[label.lower()] = Node(
                        node_type=NodeType.ENTITY,
                        label=label,
                        content=f"Entity: {label}",
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                    )

        # Extract noun-phrase-like entities: "the enterprise tier", "the database layer"
        tier_pattern = re.compile(
            r"(?:the |our |an? )?(\w+(?:\s+\w+)?\s+(?:tier|layer|module|engine|system|service|platform|framework|tool))",
            re.IGNORECASE,
        )
        for match in tier_pattern.finditer(content):
            label = match.group(1).strip()
            if len(label) > 3 and label.lower() not in entities and label.lower() not in STOP_WORDS:
                entities[label.lower()] = Node(
                    node_type=NodeType.ENTITY,
                    label=label,
                    content=f"Entity: {label}",
                    source_id=source_id,
                    created_at=now,
                    last_accessed=now,
                )

        return list(entities.values())

    def _extract_decisions(self, content: str, source_id: str, now: datetime) -> List[Node]:
        decisions = []
        seen = set()
        for pattern in self._decision_re:
            for match in pattern.finditer(content):
                text = match.group(0).strip()
                captured = match.group(1).strip() if match.lastindex else text
                label = captured[:200]
                if label.lower() not in seen and len(label) > 10:
                    seen.add(label.lower())
                    decisions.append(Node(
                        node_type=NodeType.DECISION,
                        label=label,
                        content=text,
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                    ))
        return decisions

    def _extract_facts(self, content: str, source_id: str, now: datetime) -> List[Node]:
        facts = []
        seen = set()
        for pattern in self._fact_re:
            for match in pattern.finditer(content):
                text = match.group(0).strip()
                # Use largest captured group
                groups = [g for g in match.groups() if g]
                captured = max(groups, key=len) if groups else text
                label = captured.strip()[:200]
                if label.lower() not in seen and len(label) > 5:
                    seen.add(label.lower())
                    facts.append(Node(
                        node_type=NodeType.FACT,
                        label=label,
                        content=text,
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                    ))
        return facts

    def _extract_preferences(self, content: str, source_id: str, now: datetime) -> List[Node]:
        preferences = []
        seen = set()
        for pattern in self._preference_re:
            for match in pattern.finditer(content):
                text = match.group(0).strip()
                captured = match.group(1).strip() if match.lastindex else text
                label = captured[:200]
                if label.lower() not in seen and len(label) > 5:
                    seen.add(label.lower())
                    preferences.append(Node(
                        node_type=NodeType.PREFERENCE,
                        label=label,
                        content=text,
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                    ))
        return preferences

    def _extract_topics(self, content: str, source_id: str, now: datetime) -> List[Node]:
        """Extract topics via keyword frequency and co-occurrence."""
        # Tokenize and count
        words = re.findall(r"\b[a-z][a-z'-]+\b", content.lower())
        word_freq = {}
        for w in words:
            if w not in STOP_WORDS and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Bigram topics
        bigram_freq = {}
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in STOP_WORDS and w2 not in STOP_WORDS and len(w1) > 2 and len(w2) > 2:
                bigram = f"{w1} {w2}"
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1

        # Score: frequency > 1 or bigram frequency > 1
        topics = []
        seen = set()

        # Bigram topics first (more specific)
        for bigram, freq in sorted(bigram_freq.items(), key=lambda x: -x[1]):
            if freq >= 2 and bigram not in seen:
                seen.add(bigram)
                topics.append(Node(
                    node_type=NodeType.TOPIC,
                    label=bigram,
                    content=f"Topic: {bigram} (mentioned {freq} times)",
                    source_id=source_id,
                    created_at=now,
                    last_accessed=now,
                    metadata={"frequency": freq},
                ))

        # Single-word topics (high frequency only)
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
            if freq >= 3 and word not in seen:
                # Skip if it's part of a bigram topic
                if not any(word in t for t in seen):
                    seen.add(word)
                    topics.append(Node(
                        node_type=NodeType.TOPIC,
                        label=word,
                        content=f"Topic: {word} (mentioned {freq} times)",
                        source_id=source_id,
                        created_at=now,
                        last_accessed=now,
                        metadata={"frequency": freq},
                    ))

        return topics[:10]  # Cap at 10 topics per ingestion

    # ── Helpers ───────────────────────────────────────────

    def _make_context_label(self, content: str) -> str:
        """Generate a human-readable label for the context node."""
        # Use first meaningful line, trimmed
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines:
            first = lines[0]
            # Strip speaker prefixes
            first = re.sub(r"^(User|Assistant|Human|AI):\s*", "", first)
            return first[:200]
        return "Unnamed context"

    def _make_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 0.5,
    ) -> Edge:
        return Edge(
            edge_type=edge_type,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=weight,
        )

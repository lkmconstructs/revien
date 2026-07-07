"""
Revien LLM Extractor — pluggable, LOCAL-FIRST text extraction.

This is the opt-in upgrade path over RuleBasedExtractor. It is OFF by default:
`pip install revien` + ingest runs the regex extractor with zero config, zero
network, and no API key. The LLM path is enabled only when the operator sets
REVIEN_EXTRACTOR to a non-"rule" value, and any CLOUD use is disclosed loudly.

Contract: every extractor honours the same signature/return shape that the
existing RuleBasedExtractor already produces —
    extract(content: str, source_id: str = "") -> ExtractionResult
so ingestion/pipeline.py stays agnostic to which backend is wired.

Design:
- TextExtractor   — the interface (Protocol). RuleBasedExtractor satisfies it
                    as-is; no edits to extractor.py were needed.
- LLMExtractor    — model-primary, regex-MANDATORY-fallback. On ANY failure
                    (no key, timeout, bad JSON, connection refused, HTTP error)
                    it falls through to RuleBasedExtractor so ingestion never
                    crashes and never goes dark.
- Backends        — ollama (LOCAL, recommended), openrouter/openai/anthropic
                    (CLOUD, opt-in). All use stdlib urllib only — no SDKs, no
                    new pip dependencies.
- build_extractor — config-driven selection via REVIEN_EXTRACTOR, with the
                    rule fallback always attached.

The extraction prompt and JSON parsing are ported from the production server's
model_extractor.py (OpenRouter/Qwen) and generalised across backends.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from revien.graph.schema import Edge, EdgeType, Node, NodeType, SourceType
from .extractor import ExtractionResult, RuleBasedExtractor


# ── Config ────────────────────────────────────────────────
# Selection knob. Default "rule" = offline, zero-config, zero-network.
DEFAULT_EXTRACTOR = "rule"
LOCAL_BACKENDS = ("rule", "ollama")
CLOUD_BACKENDS = ("openrouter", "openai", "anthropic")
VALID_BACKENDS = ("rule",) + ("ollama",) + CLOUD_BACKENDS

REQUEST_TIMEOUT = 30.0

# Per-backend defaults. Model overridable via REVIEN_EXTRACTION_MODEL.
_BACKEND_DEFAULTS = {
    "ollama": {
        "url": os.environ.get("REVIEN_OLLAMA_URL", "http://localhost:11434/api/chat"),
        "model": "llama3.1",
        "key_env": None,  # local; no key
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3.6-plus",
        "key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-latest",
        "key_env": "ANTHROPIC_API_KEY",
    },
}


# ── Extraction prompt (ported from server model_extractor.py) ─────────
SYSTEM_PROMPT = """You extract structured memory from conversation/document text for a long-term memory graph. Return ONLY a JSON object, no prose, no markdown fences.

Extract these node types. Apply a HIGH bar — only items that are genuinely worth remembering, not every noun or phrase. An item earns a node only if a person reading the memory later would find it meaningful.

Schema:
{
  "entities": [{"label": "<=200 chars", "content": "one sentence on what it is"}],
  "facts": [{"label": "<=200 chars", "content": "the fact in context"}],
  "decisions": [{"label": "<=200 chars", "content": "what was decided and why"}],
  "preferences": [{"label": "<=200 chars", "content": "the preference"}],
  "topics": [{"label": "<=200 chars (a real recurring theme, NOT a single stray word)"}]
}

Rules:
- entities: people, systems, products, tools that MATTER to the discussion. Not every capitalized word. Not generic tech terms unless they're a real subject of discussion.
- facts: concrete, specific, durable data points (configs, requirements, states). Not conversational filler.
- decisions: actual choices made, not topics discussed.
- preferences: stable likes/dislikes/conventions, not one-off reactions.
- topics: substantive recurring threads. NEVER single isolated words. If it isn't a real theme, omit it.
- If a category has nothing worth keeping, return an empty list for it.
- Prefer FEWER, HIGHER-QUALITY nodes. Empty lists are correct when there's nothing meaningful.
"""


# ── Interface ─────────────────────────────────────────────
@runtime_checkable
class TextExtractor(Protocol):
    """
    The extraction contract the ingestion pipeline depends on.

    Any extractor (rule-based, LLM-backed, future) must accept raw content plus
    a source id and return an ExtractionResult holding a context node, typed
    nodes, and edges. RuleBasedExtractor already satisfies this Protocol.
    """

    def extract(self, content: str, source_id: str = "") -> ExtractionResult:
        ...


# ── Disclosure ────────────────────────────────────────────
# Fires ONCE per process, only for CLOUD backends. Local backends stay silent.
_DISCLOSED_PROVIDERS: set = set()


def _disclose_cloud(provider: str) -> None:
    """
    Emit a one-time stderr warning when captured text leaves the machine.
    Local backends (rule, ollama) never call this.
    """
    if provider in _DISCLOSED_PROVIDERS:
        return
    _DISCLOSED_PROVIDERS.add(provider)
    sys.stderr.write(
        f"WARNING: Revien is sending captured text to {provider} for extraction "
        f"- this leaves your machine. Set REVIEN_EXTRACTOR=rule (offline) or "
        f"=ollama (local) to keep it on-device.\n"
    )
    sys.stderr.flush()


# ── LLM Extractor ─────────────────────────────────────────
class LLMExtractor:
    """
    LLM-backed extractor. Same interface as RuleBasedExtractor.

    Model-primary, regex-MANDATORY-fallback: on any failure (no key, timeout,
    bad JSON, connection refused, HTTP error) it returns the RuleBasedExtractor
    result so ingestion never crashes. CLOUD backends disclose once to stderr.
    """

    def __init__(
        self,
        backend: str = "ollama",
        fallback: Optional[TextExtractor] = None,
    ):
        backend = (backend or "ollama").lower().strip()
        if backend not in _BACKEND_DEFAULTS:
            raise ValueError(
                f"Unknown LLM backend {backend!r}. "
                f"Expected one of: {', '.join(_BACKEND_DEFAULTS)}"
            )
        self.backend = backend
        self.fallback: TextExtractor = fallback or RuleBasedExtractor()

        # Degrade visibility (OPEN 3): per-call fallback lines scroll away in
        # a busy ingest; what masked the quota-429 leak was the ABSENCE of an
        # aggregate signal. Track state, escalate ONCE per process when the
        # LLM path has effectively stopped working.
        self.total_calls = 0
        self.total_fallbacks = 0
        self.consecutive_fallbacks = 0
        self._escalated = False

        cfg = _BACKEND_DEFAULTS[backend]
        self.url = cfg["url"]
        self.model = os.environ.get("REVIEN_EXTRACTION_MODEL", cfg["model"])
        self.key_env = cfg["key_env"]
        self.api_key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else ""
        self.is_cloud = backend in CLOUD_BACKENDS

    # Consecutive failures before the one-time "you are running on regex"
    # escalation. 3 = fast enough to catch a dead key on the first sync,
    # tolerant of a single transient timeout.
    ESCALATE_AFTER = 3

    def status(self) -> dict:
        """Aggregate extractor health for callers/daemons to surface."""
        return {
            "backend": self.backend,
            "total_calls": self.total_calls,
            "total_fallbacks": self.total_fallbacks,
            "consecutive_fallbacks": self.consecutive_fallbacks,
            "degraded": self.consecutive_fallbacks >= self.ESCALATE_AFTER,
        }

    def _note_fallback(self, reason: str) -> None:
        self.total_fallbacks += 1
        self.consecutive_fallbacks += 1
        sys.stderr.write(
            f"[LLMExtractor:{self.backend}] {reason}; "
            f"falling back to rule-based extraction\n"
        )
        if self.consecutive_fallbacks >= self.ESCALATE_AFTER and not self._escalated:
            self._escalated = True
            sys.stderr.write(
                f"[revien] WARNING: LLM extraction ({self.backend}) has failed "
                f"{self.consecutive_fallbacks} consecutive times - ingestion is "
                f"EFFECTIVELY RULE-BASED (degraded extraction quality) until the "
                f"backend recovers. Check the {self.key_env or 'backend'} key, "
                f"quota, and connectivity.\n"
            )
        sys.stderr.flush()

    # ── Public contract ───────────────────────────────────
    def extract(self, content: str, source_id: str = "") -> ExtractionResult:
        result = ExtractionResult()
        now = datetime.now(timezone.utc)

        # 1. Context node for the full interaction (always created).
        context_node = Node(
            node_type=NodeType.CONTEXT,
            label=self._make_context_label(content),
            content=content,
            source_id=source_id,
            created_at=now,
            last_accessed=now,
            source_type=SourceType.EXTRACTED,
            confidence=1.0,
            source_context="EXTRACTED",
        )
        result.context_node = context_node
        result.nodes.append(context_node)

        # 2. Disclose BEFORE any cloud network call (so the warning fires even
        #    if the request then fails). Local backends stay silent.
        if self.is_cloud:
            _disclose_cloud(self.backend)

        # 3. Try model extraction; fall back to regex on ANY failure — loudly,
        # with escalation once the failures are consecutive (OPEN 3).
        self.total_calls += 1
        try:
            parsed = self._call_model(content)
        except Exception as e:  # noqa: BLE001 — fallback must catch everything
            self._note_fallback(f"model call failed ({e!r})")
            return self.fallback.extract(content, source_id)

        if parsed is None:
            self._note_fallback("unparseable model output")
            return self.fallback.extract(content, source_id)

        # A success resets the consecutive counter (and re-arms escalation, so
        # a later sustained outage warns again).
        self.consecutive_fallbacks = 0
        self._escalated = False

        # 4. Build typed nodes from parsed JSON.
        self._build_typed_nodes(parsed, "entities", NodeType.ENTITY,
                                 EdgeType.MENTIONED_BY, result, context_node, source_id, now)
        self._build_typed_nodes(parsed, "facts", NodeType.FACT,
                                 EdgeType.RELATED_TO, result, context_node, source_id, now)
        self._build_typed_nodes(parsed, "decisions", NodeType.DECISION,
                                 EdgeType.DECIDED_IN, result, context_node, source_id, now)
        self._build_typed_nodes(parsed, "preferences", NodeType.PREFERENCE,
                                 EdgeType.RELATED_TO, result, context_node, source_id, now)
        self._build_typed_nodes(parsed, "topics", NodeType.TOPIC,
                                 EdgeType.RELATED_TO, result, context_node, source_id, now)

        # 5. Co-occurrence edges between entities (mirrors regex behaviour).
        entity_nodes = [n for n in result.nodes if n.node_type == NodeType.ENTITY]
        for i in range(len(entity_nodes)):
            for j in range(i + 1, len(entity_nodes)):
                result.edges.append(self._make_edge(
                    entity_nodes[i].node_id, entity_nodes[j].node_id,
                    EdgeType.RELATED_TO, weight=0.3,
                ))

        return result

    # ── Model call (dispatches per backend) ───────────────
    def _call_model(self, content: str) -> Optional[dict]:
        if self.key_env and not self.api_key:
            raise RuntimeError(f"{self.key_env} not set")

        snippet = content[:12000]  # bound cost/latency

        if self.backend == "anthropic":
            return self._call_anthropic(snippet)
        if self.backend == "ollama":
            return self._call_ollama(snippet)
        # openrouter + openai share the OpenAI chat-completions shape.
        return self._call_openai_compatible(snippet)

    def _call_openai_compatible(self, snippet: str) -> Optional[dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": snippet},
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = self._post_json(self.url, payload, headers)
        raw = data["choices"][0]["message"]["content"]
        return self._safe_parse(raw)

    def _call_anthropic(self, snippet: str) -> Optional[dict]:
        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0.1,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": snippet}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = self._post_json(self.url, payload, headers)
        # Anthropic returns content as a list of blocks.
        blocks = data.get("content") or []
        raw = "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict)
        )
        return self._safe_parse(raw)

    def _call_ollama(self, snippet: str) -> Optional[dict]:
        # LOCAL. Ollama's /api/chat with stream disabled returns one JSON object.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": snippet},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }
        headers = {"Content-Type": "application/json"}
        data = self._post_json(self.url, payload, headers)
        # /api/chat -> {"message": {"content": ...}}; /api/generate -> {"response": ...}
        if "message" in data and isinstance(data["message"], dict):
            raw = data["message"].get("content", "")
        else:
            raw = data.get("response", "")
        return self._safe_parse(raw)

    def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{self.backend} HTTP {e.code}: {body}") from e

    @staticmethod
    def _safe_parse(raw: str) -> Optional[dict]:
        """Parse JSON, tolerating accidental markdown fences."""
        if not raw:
            return None
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    # ── Node building ─────────────────────────────────────
    def _build_typed_nodes(self, parsed, key, node_type, edge_type,
                           result, context_node, source_id, now):
        items = parsed.get(key) or []
        if not isinstance(items, list):
            return
        seen = set()
        for item in items:
            if isinstance(item, str):
                label, body = item, ""
            elif isinstance(item, dict):
                label = (item.get("label") or "").strip()
                body = (item.get("content") or "").strip()
            else:
                continue
            if not label or len(label) < 2:
                continue
            key_l = label.lower()
            if key_l in seen:
                continue
            seen.add(key_l)

            node = Node(
                node_type=node_type,
                label=label[:200],
                content=body or label,
                source_id=source_id,
                created_at=now,
                last_accessed=now,
                source_type=SourceType.INFERRED,
                confidence=0.7,
                source_context="INFERRED",
                metadata={"extractor": f"llm:{self.backend}"},
            )
            result.nodes.append(node)
            result.edges.append(self._make_edge(
                node.node_id, context_node.node_id, edge_type
            ))

    # ── Helpers (mirror RuleBasedExtractor) ───────────────
    def _make_context_label(self, content: str) -> str:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        if lines:
            first = re.sub(r"^(User|Assistant|Human|AI):\s*", "", lines[0])
            return first[:200]
        return "Unnamed context"

    def _make_edge(self, source_id, target_id, edge_type, weight=0.5) -> Edge:
        return Edge(
            edge_type=edge_type,
            source_node_id=source_id,
            target_node_id=target_id,
            weight=weight,
        )


# ── Config-driven selection ───────────────────────────────
def build_extractor(backend: Optional[str] = None) -> TextExtractor:
    """
    Build the configured extractor with the MANDATORY rule fallback attached.

    Selection precedence: explicit `backend` arg, else env REVIEN_EXTRACTOR,
    else DEFAULT_EXTRACTOR ("rule"). Unknown values fall back to rule with a
    warning rather than crashing — ingestion must always work.

    - "rule"                       -> RuleBasedExtractor (offline, default)
    - "ollama"                     -> LLMExtractor (LOCAL, recommended upgrade)
    - "openrouter"/"openai"/"anthropic" -> LLMExtractor (CLOUD, discloses once)
    """
    choice = (backend or os.environ.get("REVIEN_EXTRACTOR", DEFAULT_EXTRACTOR))
    choice = choice.lower().strip()

    if choice == "rule":
        return RuleBasedExtractor()

    if choice not in VALID_BACKENDS:
        sys.stderr.write(
            f"[revien] Unknown REVIEN_EXTRACTOR={choice!r}; "
            f"valid: {', '.join(VALID_BACKENDS)}. Falling back to rule-based.\n"
        )
        return RuleBasedExtractor()

    # LLM backend selected. Construct it with the rule fallback baked in.
    try:
        return LLMExtractor(backend=choice, fallback=RuleBasedExtractor())
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[revien] Could not build LLM extractor ({e!r}); "
            f"falling back to rule-based.\n"
        )
        return RuleBasedExtractor()

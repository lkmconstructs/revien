"""
Shared request validation for Revien's protocol surfaces (R4).

One rule set, three faces: the REST daemon (``/v1/ingest`` / ``/v1/recall``),
the MCP tools (``revien_store`` / ``revien_recall``), and the Hermes provider's
``handle_tool_call``. Each surface translates :class:`ValidationError` into its
own protocol error (HTTP 400 / MCP ToolError / Hermes ``{"error": ...}``
payload) but the RULES live here so the surfaces cannot drift.

Message style follows the pre-existing ``as_of``/``format`` checks in
``revien/daemon/server.py``: ``Invalid <field> (<expectation>): <value>``.

Deliberately NOT applied inside ``IngestionPipeline.ingest`` or
``RetrievalEngine.recall``: existing green tests pin the permissive engine
behavior (``pipeline.ingest(content="")`` creates a bare context node in
tests/test_ingestion.py; ``engine.recall(top_n=100)`` clamps to 20 in
tests/test_retrieval.py), and the bench (revien_bench) drives the engine
directly. The engine's internal ``top_n`` clamp stays as the belt; this module
is the suspenders at the protocol edge.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

# The content types the ingestion pipeline documents
# (revien/ingestion/pipeline.py: "conversation | document | note | code").
VALID_CONTENT_TYPES = frozenset({"conversation", "document", "note", "code"})

# Recall result-window bounds. The engine clamps internally at 20; the API
# edge REJECTS out-of-range values instead of silently reshaping the request.
TOP_N_MIN = 1
TOP_N_MAX = 20


class ValidationError(ValueError):
    """A request failed protocol-surface validation.

    Subclasses ValueError so FastMCP's tool wrapper surfaces it as a ToolError
    exactly like the pre-existing bad-``as_of`` path (pinned by
    tests/test_mcp.py::test_bad_as_of_errors_cleanly).
    """


def validate_ingest(
    content: str,
    source_id: str,
    content_type: str,
    timestamp: Optional[str] = None,
) -> Optional[datetime]:
    """Validate an ingest/store request. Returns the parsed timestamp.

    Rules:
      - ``content`` non-blank after strip (a blank ingest creates an empty,
        unrecallable memory — refuse it at the edge).
      - ``source_id`` non-blank after strip (provenance must name a source).
      - ``content_type`` one of :data:`VALID_CONTENT_TYPES`.
      - ``timestamp`` None or ISO-8601 parseable — unparseable is an ERROR,
        not a silent drop (the old server.py behavior swallowed it).

    Returns the parsed ``datetime`` (or None when no timestamp was given) so
    callers never parse twice.

    Known, accepted gap: the non-blank checks use ``str.strip()``, which does
    not catch content made solely of zero-width or exotic unicode spaces.
    """
    if not (content or "").strip():
        raise ValidationError("Invalid content (non-blank text expected)")
    if not (source_id or "").strip():
        raise ValidationError("Invalid source_id (non-blank string expected)")
    if content_type not in VALID_CONTENT_TYPES:
        valid = "|".join(sorted(VALID_CONTENT_TYPES))
        raise ValidationError(
            f"Invalid content_type ({valid} expected): {content_type}"
        )
    if timestamp is None:
        return None
    try:
        # JS clients (Date.toISOString()) send a trailing "Z";
        # datetime.fromisoformat only accepts it from Python 3.11 — normalize
        # so a 3.10 daemon and a 3.12 daemon accept the same wire format.
        if isinstance(timestamp, str) and timestamp[-1:] in ("Z", "z"):
            timestamp_to_parse = timestamp[:-1] + "+00:00"
        else:
            timestamp_to_parse = timestamp
        return datetime.fromisoformat(timestamp_to_parse)
    except (ValueError, TypeError):
        raise ValidationError(
            f"Invalid timestamp (ISO-8601 expected): {timestamp}"
        )


def validate_recall(
    query: str,
    top_n: int,
    min_score: Optional[float] = None,
) -> None:
    """Validate a recall request.

    Rules:
      - ``query`` non-blank after strip.
      - ``TOP_N_MIN <= top_n <= TOP_N_MAX`` — out-of-range is rejected at the
        edge (no clamping, no negative-slice surprises); the engine's internal
        clamp remains as a second belt for direct callers.
      - ``min_score >= 0`` when provided (surfaces without the knob pass None).
    """
    if not (query or "").strip():
        raise ValidationError("Invalid query (non-blank text expected)")
    # Type-strictness note: the isinstance/bool checks below are belt for
    # DIRECT Python callers. The protocol faces coerce upstream before this
    # runs — pydantic (REST) and FastMCP (MCP) turn "5"->5 and reject
    # non-integers at the schema layer; the Hermes face applies the matching
    # coercion rules itself in handle_tool_call — so these branches are
    # normally unreachable from the wire.
    # bool is an int subclass; reject it explicitly rather than treating
    # True as top_n=1.
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise ValidationError(
            f"Invalid top_n (integer {TOP_N_MIN}..{TOP_N_MAX} expected): {top_n}"
        )
    if not (TOP_N_MIN <= top_n <= TOP_N_MAX):
        raise ValidationError(
            f"Invalid top_n ({TOP_N_MIN}..{TOP_N_MAX} expected): {top_n}"
        )
    if min_score is not None:
        # NaN fails every comparison (NaN < 0 is False), so a plain `< 0`
        # check would let NaN through and silently disable the score filter;
        # inf/-inf are equally meaningless as thresholds. Reject non-finite
        # outright.
        if not math.isfinite(min_score) or min_score < 0:
            raise ValidationError(
                f"Invalid min_score (finite number >= 0 expected): {min_score}"
            )

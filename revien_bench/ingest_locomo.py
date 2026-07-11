"""
revien_bench.ingest_locomo — Ingest ONE LoCoMo conversation into a GraphStore.

Uses the REAL Revien ingestion API (no shortcuts):

    pipe = IngestionPipeline(store, semantic=...)
    pipe.ingest(IngestionInput(
        source_id=f"{conv}:{dia_id}",
        content=f"{speaker}: {text}",
        content_type="conversation",
        timestamp=<session_date>,
        metadata={"dia_id": ..., "session": ...},
    ))

dia_id tagging
--------------
The rule-based extractor sets `source_id` on every node it creates but does NOT
copy IngestionInput.metadata onto extracted nodes. So to map retrieval hits back
to gold evidence dia_ids, we (a) make `source_id` carry the dia_id as
`"{conv}:{dia_id}"`, and (b) immediately after each turn's ingest, look up every
node with that source_id and stamp `dia_id` / `session` / `conv` into the node's
`metadata` via the store's real update path. This is the load-bearing link that
lets recall@k / MRR / nDCG be scored against LoCoMo `evidence`.

timestamp
---------
LoCoMo session dates are human strings ("7 May 2023", "1:56 pm on 8 May, 2023").
We parse them best-effort into a UTC datetime so recency/temporal scoring has a
real signal; if parsing fails we pass None (current behavior — created_at=now)
but ALWAYS keep the raw date string in metadata for traceability.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from revien.graph.schema import Modality
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline

from .loader import Conversation, Turn


# A small, dependency-free date parser for the LoCoMo session-date strings.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
_DATE_RE_ALT = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")


def parse_session_date(raw: str) -> Optional[datetime]:
    """Best-effort parse of a LoCoMo session date string -> aware UTC datetime."""
    if not raw:
        return None
    low = raw.lower()
    m = _DATE_RE.search(low)
    if m:
        day, mon_name, year = m.group(1), m.group(2), m.group(3)
        mon = _MONTHS.get(mon_name)
        if mon:
            try:
                return datetime(int(year), mon, int(day), tzinfo=timezone.utc)
            except ValueError:
                return None
    m = _DATE_RE_ALT.search(low)
    if m:
        mon_name, day, year = m.group(1), m.group(2), m.group(3)
        mon = _MONTHS.get(mon_name)
        if mon:
            try:
                return datetime(int(year), mon, int(day), tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _tag_nodes_with_dia_id(
    store: GraphStore, source_id: str, dia_id: str, session: int, conv_id: str
) -> int:
    """Stamp dia_id/session/conv into the metadata of every node for this turn.

    Returns the number of nodes tagged. Uses list_nodes(source_id=...) (an
    indexed query) + the real update_node path, so the tag is persisted and the
    audit trail records it.
    """
    nodes = store.list_nodes(source_id=source_id, limit=999999)
    tagged = 0
    for node in nodes:
        md = dict(node.metadata or {})
        md["dia_id"] = dia_id
        md["session"] = session
        md["conv"] = conv_id
        # Suppress the generic per-field "update" audit op flood: we DO want one
        # provenance row per tag so lineage is traceable, so keep the default op.
        store.update_node(node.node_id, metadata=md)
        tagged += 1
    return tagged


def ingest_conversation(
    conv: Conversation,
    store: GraphStore,
    semantic=None,
    use_blip_caption: bool = False,
) -> Dict:
    """Ingest every turn of ONE conversation into `store` via the real pipeline.

    Args:
        conv: parsed Conversation.
        store: a FRESH, isolated GraphStore (no cross-conversation leakage).
        semantic: optional SemanticIndex (passed straight to IngestionPipeline).
                  None => pipeline builds its own (self-disables without the extra).
        use_blip_caption: append BLIP caption text to image turns (default off).

    Returns a summary dict (turns ingested, nodes created, nodes tagged).
    """
    pipe = IngestionPipeline(store, semantic=semantic)

    turns_ingested = 0
    nodes_created = 0
    nodes_tagged = 0

    for turn in conv.turns:
        text = turn.text
        captioned = bool(use_blip_caption and turn.blip_caption)
        if captioned:
            text = f"{text} [image: {turn.blip_caption}]"
        content = f"{turn.speaker}: {text}"
        source_id = f"{conv.conv_id}:{turn.dia_id}"
        ts = parse_session_date(turn.session_date)

        # Leg 1 modality. An image turn is MIXED (speaker text + a photo). When we
        # appended the BLIP caption the image content is now IN the text — treat
        # that as a cheap vision pass (answerable_by_text + vision_processed).
        # Otherwise the image is dropped and its content is unavailable to text.
        if turn.has_image:
            modality = Modality.MIXED
            answerable = captioned
            vision_done = captioned
        else:
            modality = Modality.TEXT
            answerable = True
            vision_done = False

        out = pipe.ingest(
            IngestionInput(
                source_id=source_id,
                content=content,
                content_type="conversation",
                timestamp=ts,
                metadata={
                    "dia_id": turn.dia_id,
                    "session": turn.session,
                    "conv": conv.conv_id,
                    "session_date": turn.session_date,
                },
                source_modality=modality,
                answerable_by_text=answerable,
                vision_processed=vision_done,
            )
        )
        turns_ingested += 1
        nodes_created += out.nodes_created
        nodes_tagged += _tag_nodes_with_dia_id(
            store, source_id, turn.dia_id, turn.session, conv.conv_id
        )

    return {
        "conv_id": conv.conv_id,
        "turns_ingested": turns_ingested,
        "nodes_created": nodes_created,
        "nodes_tagged": nodes_tagged,
        "total_nodes": store.count_nodes(),
        "total_edges": store.count_edges(),
    }


def dia_ids_for_node(store: GraphStore, node_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (dia_id, conv) for a node from its metadata, else (None, None)."""
    node = store.get_node(node_id)
    if node is None:
        return (None, None)
    md = node.metadata or {}
    return (md.get("dia_id"), md.get("conv"))

"""
revien_bench.loader — Parse locomo10.json into Conversation / QA dataclasses.

REAL SCHEMA (verified against snap-research/locomo data/locomo10.json, 2026-06-18):

  Top level: JSON array of sample objects, each:
    {
      "qa": [ {question, answer|adversarial_answer, evidence:[dia_id], category}, ... ],
      "conversation": {
        "speaker_a": "<name>",
        "speaker_b": "<name>",
        "session_1_date_time": "<human date string>",
        "session_1": [ {speaker, dia_id, text, img_url?, blip_caption?}, ... ],
        "session_2_date_time": ...,
        "session_2": [ ... ],
        ... (variable number of sessions)
        ("session_summary"/"observation"/"event_summary" keys may exist; IGNORED)
      }
    }

  Turn:  {"speaker": str, "dia_id": "D<session>:<turn>", "text": str,
          "img_url"?: [str], "blip_caption"?: str}
  QA:    {"question": str, "answer": str|number, "evidence": ["D1:3", ...],
          "category": int}
         Category 5 (adversarial) stores the gold in "adversarial_answer"
         INSTEAD of "answer", and typically carries empty/absent evidence.

  Category map: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.

Defensiveness: this loader does NOT assume a stable top-level id key. It derives
a conversation id from any of sample_id/conv_id/id if present, else the array
index ("conv_0", "conv_1", ...). It tolerates missing date_times, missing
evidence, int/float answers, and the adversarial_answer field. Images are
dropped (text-only system); blip_caption text is OPTIONALLY appended (off by
default — kept faithful to design §1 "keep BLIP caption text only" as an opt-in).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CATEGORY_NAMES: Dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_SESSION_RE = re.compile(r"^session_(\d+)$")


@dataclass
class Turn:
    """One dialogue turn within a session."""
    dia_id: str
    speaker: str
    text: str
    session: int
    session_date: str = ""
    blip_caption: str = ""


@dataclass
class QA:
    """One LoCoMo QA item."""
    question: str
    answer: str            # gold (from `answer` or, for cat 5, `adversarial_answer`)
    category: int
    evidence: List[str] = field(default_factory=list)   # gold dia_ids
    is_adversarial: bool = False

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, f"category_{self.category}")


@dataclass
class Conversation:
    """One LoCoMo sample: a multi-session conversation + its QA set."""
    conv_id: str
    speaker_a: str
    speaker_b: str
    turns: List[Turn] = field(default_factory=list)
    qa: List[QA] = field(default_factory=list)
    # session index -> date string
    session_dates: Dict[int, str] = field(default_factory=dict)

    @property
    def last_session_date(self) -> str:
        if not self.session_dates:
            return ""
        return self.session_dates[max(self.session_dates)]


def _coerce_evidence(raw) -> List[str]:
    """Normalize the evidence field to a list of dia_id strings.

    LoCoMo evidence is usually a list of dia_ids, but be defensive about
    nested lists / single string / ints.
    """
    if raw is None:
        return []
    out: List[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for it in items:
        if isinstance(it, list):
            out.extend(str(x) for x in it)
        else:
            out.append(str(it))
    return out


def _parse_qa(raw_qa: dict) -> Optional[QA]:
    """Parse one QA dict, tolerating adversarial-answer and int/float gold."""
    if not isinstance(raw_qa, dict):
        return None
    question = str(raw_qa.get("question", "")).strip()
    if not question:
        return None
    category = raw_qa.get("category")
    try:
        category = int(category)
    except (TypeError, ValueError):
        category = 0
    is_adv = category == 5
    # Cat-5 gold lives in adversarial_answer; others in answer. Fall back across
    # both so a slightly different shape never silently drops the gold.
    gold = raw_qa.get("adversarial_answer")
    if gold is None:
        gold = raw_qa.get("answer")
    if gold is None and is_adv:
        gold = "not mentioned"  # canonical refusal target
    answer = "" if gold is None else str(gold)
    evidence = _coerce_evidence(raw_qa.get("evidence"))
    return QA(
        question=question,
        answer=answer,
        category=category,
        evidence=evidence,
        is_adversarial=is_adv,
    )


def _parse_conversation(raw_conv: dict, conv_id: str) -> Conversation:
    speaker_a = str(raw_conv.get("speaker_a", "Speaker A"))
    speaker_b = str(raw_conv.get("speaker_b", "Speaker B"))
    conv = Conversation(conv_id=conv_id, speaker_a=speaker_a, speaker_b=speaker_b)

    # Collect session_N arrays and session_N_date_time strings.
    for key, val in raw_conv.items():
        m = _SESSION_RE.match(key)
        if m and isinstance(val, list):
            sess_n = int(m.group(1))
            date_key = f"session_{sess_n}_date_time"
            sess_date = str(raw_conv.get(date_key, "") or "")
            conv.session_dates[sess_n] = sess_date
            for turn in val:
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text", "") or "").strip()
                dia_id = str(turn.get("dia_id", "") or "")
                speaker = str(turn.get("speaker", "") or "")
                if not text or not dia_id:
                    continue
                conv.turns.append(
                    Turn(
                        dia_id=dia_id,
                        speaker=speaker,
                        text=text,
                        session=sess_n,
                        session_date=sess_date,
                        blip_caption=str(turn.get("blip_caption", "") or ""),
                    )
                )

    # Order turns by (session, turn-index-from-dia_id) for stable ingestion.
    def _turn_key(t: Turn):
        # dia_id like "D1:3" -> (1, 3); fall back to (session, 0).
        mm = re.match(r"D(\d+):(\d+)", t.dia_id)
        if mm:
            return (int(mm.group(1)), int(mm.group(2)))
        return (t.session, 0)

    conv.turns.sort(key=_turn_key)
    return conv


def load_locomo(path: str | Path) -> List[Conversation]:
    """Load and parse the full LoCoMo-10 dataset into Conversation objects."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a top-level JSON array of samples, got {type(data).__name__}"
        )

    conversations: List[Conversation] = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        # Derive a stable conversation id, tolerating various id key names.
        conv_id = str(
            sample.get("sample_id")
            or sample.get("conv_id")
            or sample.get("id")
            or f"conv_{idx}"
        )
        raw_conv = sample.get("conversation", {})
        conv = _parse_conversation(raw_conv if isinstance(raw_conv, dict) else {}, conv_id)
        for raw_qa in sample.get("qa", []) or []:
            qa = _parse_qa(raw_qa)
            if qa is not None:
                conv.qa.append(qa)
        conversations.append(conv)
    return conversations


def schema_report(path: str | Path) -> dict:
    """Return a small structural report of the fetched file (for verification)."""
    convs = load_locomo(path)
    total_turns = sum(len(c.turns) for c in convs)
    total_qa = sum(len(c.qa) for c in convs)
    cat_counts: Dict[int, int] = {}
    for c in convs:
        for qa in c.qa:
            cat_counts[qa.category] = cat_counts.get(qa.category, 0) + 1
    return {
        "conversations": len(convs),
        "total_turns": total_turns,
        "total_qa": total_qa,
        "qa_by_category": {
            CATEGORY_NAMES.get(k, str(k)): v for k, v in sorted(cat_counts.items())
        },
        "sessions_per_conv": [len(c.session_dates) for c in convs],
    }


if __name__ == "__main__":
    import sys
    from pprint import pprint

    p = sys.argv[1] if len(sys.argv) > 1 else "data/locomo10.json"
    pprint(schema_report(p))

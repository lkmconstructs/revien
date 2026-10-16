"""
Usage signal (N1, metacognition tier 1): did an answer actually draw on the
memories recall handed it? See observer.py for modes and judge.py for the
rule.
"""

from .episodes import (
    RecallEpisode,
    RecalledMemory,
    extract_recall_episodes,
    is_recall_tool,
    parse_recall_result,
)
from .evidence import UseEvidence, distinctive_terms, gather_evidence
from .judge import UseVerdict, judge_use
from .observer import (
    RECEIPT_TTL,
    USAGE_CURSOR,
    JudgmentDetail,
    ObserveResult,
    Receipt,
    ReceiptError,
    UsageObserver,
    UsageReport,
    live_observation_enabled,
    usage_mode,
)

__all__ = [
    "JudgmentDetail",
    "ObserveResult",
    "RECEIPT_TTL",
    "Receipt",
    "ReceiptError",
    "live_observation_enabled",
    "RecallEpisode",
    "RecalledMemory",
    "USAGE_CURSOR",
    "UsageObserver",
    "UsageReport",
    "UseEvidence",
    "UseVerdict",
    "distinctive_terms",
    "extract_recall_episodes",
    "gather_evidence",
    "is_recall_tool",
    "judge_use",
    "parse_recall_result",
    "usage_mode",
]

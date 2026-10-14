"""
The usage observer (N1): recall episodes in, judgments out — and, in apply
mode, use.

Modes (``REVIEN_USAGE``):
  shadow (default) — record judgments only. Nothing in the graph changes, so
                     the rule can be calibrated on real sessions first.
  apply            — also call ``RetrievalEngine.mark_used()`` for each memory
                     judged used: access count +1, an "access" audit row, and
                     +0.05 on its edges.
  off              — the pass does nothing.
Anything unrecognized falls back to shadow: a typo must never start writing
to the graph.

Idempotent by construction: each (episode, memory) pair is claimed once in
the ``use_judgments`` ledger, so rescanning a grown session (or the same file
from two processes) never counts a use twice.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from revien.daemon.scheduler import EPOCH

from .episodes import RecallEpisode, RecalledMemory, extract_recall_episodes
from .evidence import EmbedFn, UseEvidence, gather_evidence
from .judge import UseVerdict, judge_use

MODES = ("off", "shadow", "apply")
# Persisted scan position for Claude Code sessions, in the sync_cursors table
# beside the adapters' own (independent of the ingest cursor).
USAGE_CURSOR = "usage:claude-code"
# How long a client has to hand an answer back. Past this an unjudged receipt
# becomes a flag (never silently dropped).
RECEIPT_TTL = timedelta(hours=24)


def usage_mode(value: Optional[str] = None) -> str:
    """Resolve the mode: an explicit value wins, else ``REVIEN_USAGE``, else
    shadow. Unrecognized values resolve to shadow."""
    raw = value if value is not None else os.environ.get("REVIEN_USAGE", "")
    raw = raw.strip().lower()
    return raw if raw in MODES else "shadow"


def live_observation_enabled() -> bool:
    """In-process observers (Hermes, Ollama) judge every turn, so they run
    only when ``REVIEN_USAGE`` is explicitly shadow or apply. Unset means no
    receipts and no judgments from those paths."""
    return os.environ.get("REVIEN_USAGE", "").strip().lower() in ("shadow", "apply")


class ReceiptError(ValueError):
    """A receipt that cannot be redeemed. ``code``: unknown, expired,
    already_redeemed, empty_answer."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Receipt(BaseModel):
    recall_id: str
    expires_at: datetime


class JudgmentDetail(BaseModel):
    """How one memory was handled."""

    node_id: str
    label: str = ""
    verdict: str  # used | unused | abstain
    shared_terms: List[str] = Field(default_factory=list)
    similarity: Optional[float] = None
    reason: str = ""
    applied: bool = False


class ObserveResult(BaseModel):
    """What POST /v1/usage/observe returns."""

    recall_id: str
    runtime: str
    mode: str
    semantic_active: bool
    judgments: List[JudgmentDetail] = Field(default_factory=list)
    skipped: Dict[str, int] = Field(default_factory=dict)


class UsageReport(BaseModel):
    """What one pass did. Counts are per (episode, memory) pair unless named
    otherwise."""

    mode: str
    semantic_active: bool = False
    files_scanned: int = 0
    episodes: int = 0  # closed recall episodes found
    judged: int = 0  # new judgments written to the ledger
    used: int = 0
    unused: int = 0
    abstained: int = 0  # the rule declined to judge; nothing written
    already_judged: int = 0  # skipped: judged on an earlier pass
    missing: int = 0  # skipped: the memory's node no longer exists
    applied: int = 0  # mark_used calls made (apply mode only)


class UsageObserver:
    """Runs the usage pass over Claude Code sessions."""

    def __init__(
        self,
        store: Any,
        engine: Any = None,
        semantic: Any = None,
        judge: Optional[Callable[[UseEvidence], UseVerdict]] = None,
        mode: Optional[str] = None,
    ):
        self.store = store
        self.engine = engine
        self.semantic = semantic
        # Resolved at construction, not import, so a caller (or a test) can
        # swap the module-level rule.
        self.judge = judge or judge_use
        self.mode = usage_mode(mode)
        if self.mode == "apply" and engine is None:
            raise ValueError(
                "REVIEN_USAGE=apply needs a RetrievalEngine: mark_used lives there"
            )

    def _embedder(self) -> Optional[EmbedFn]:
        if self.semantic is None or not self.semantic.is_enabled:
            return None
        return self.semantic.embed_texts

    def observe_episode(
        self, episode: RecallEpisode, report: UsageReport
    ) -> List[JudgmentDetail]:
        """Judge every not-yet-judged memory of one episode. Returns the
        verdict for each memory it judged or abstained on."""
        details: List[JudgmentDetail] = []
        fresh = []
        for memory in episode.memories:
            if self.store.has_use_judgment(episode.episode_key, memory.node_id):
                report.already_judged += 1
            elif self.store.get_node(memory.node_id) is None:
                # Forgotten or deleted since the recall. The transcript still
                # holds its text, so judging it would write the forgotten
                # words back into the ledger: skip, record nothing.
                report.missing += 1
            else:
                fresh.append(memory)
        if not fresh:
            return details  # nothing new: skip the embedding cost entirely
        scoped = episode.model_copy(update={"memories": fresh})
        for memory, evidence in gather_evidence(scoped, self._embedder()):
            verdict = self.judge(evidence)
            detail = JudgmentDetail(
                node_id=memory.node_id,
                label=memory.label,
                verdict="abstain" if verdict.abstain else (
                    "used" if verdict.used else "unused"),
                shared_terms=evidence.shared_terms,
                similarity=evidence.similarity,
                reason=verdict.reason,
            )
            if verdict.abstain:
                report.abstained += 1
                details.append(detail)
                continue
            apply = self.mode == "apply" and verdict.used
            claimed = self.store.record_use_judgment(
                episode_key=episode.episode_key,
                node_id=memory.node_id,
                used=verdict.used,
                reason=verdict.reason,
                shared_terms=evidence.shared_terms,
                similarity=evidence.similarity,
                query=episode.query,
                mode=self.mode,
                applied=apply,
            )
            if not claimed:  # another pass got there first
                report.already_judged += 1
                continue
            report.judged += 1
            if verdict.used:
                report.used += 1
            else:
                report.unused += 1
            if apply:
                self.engine.mark_used(memory.node_id, episode.query)
                report.applied += 1
            detail.applied = apply
            details.append(detail)
        return details

    # ── Receipts: any runtime ─────────────────────────────

    def issue_receipt(self, query: str, node_ids: List[str], runtime: str) -> Receipt:
        """Record a recall's receipt and return it. Callers hand the
        recall_id back with the answer via observe_answer()."""
        self.store.expire_recall_receipts()
        issued = datetime.now(timezone.utc)
        receipt = Receipt(
            recall_id="r_" + secrets.token_urlsafe(16),
            expires_at=issued + RECEIPT_TTL,
        )
        self.store.add_recall_receipt(
            recall_id=receipt.recall_id, runtime=runtime, query=query,
            node_ids=node_ids, issued_at=issued, expires_at=receipt.expires_at,
        )
        return receipt

    def observe_answer(
        self, recall_id: str, answer: str, prompt: str = ""
    ) -> ObserveResult:
        """Judge an answer against the memories its receipt says it was
        handed. Raises ReceiptError (unknown, expired, already redeemed,
        empty answer); nothing is written in those cases.

        Memories are compared as they are now, not as recalled — the
        receipt holds ids only. The receipt is closed only after judging,
        so a failure leaves it open to retry (or to expire into a flag)."""
        if not answer or not answer.strip():
            raise ReceiptError("empty_answer", "answer is empty")
        self.store.expire_recall_receipts()
        receipt = self.store.get_recall_receipt(recall_id)
        if receipt is None:
            raise ReceiptError("unknown", f"no receipt {recall_id!r}")
        if receipt["status"] == "redeemed":
            raise ReceiptError("already_redeemed",
                               f"receipt {recall_id!r} was already redeemed")
        if receipt["status"] != "open":
            raise ReceiptError("expired", f"receipt {recall_id!r} expired "
                               f"unjudged at {receipt['expires_at']} (flagged)")

        memories = []
        for node_id in receipt["node_ids"]:
            node = self.store.get_node(node_id)
            memories.append(RecalledMemory(
                node_id=node_id,
                label=node.label if node is not None else "",
                content=node.content if node is not None else "",
            ))
        runtime = receipt["runtime"] or "api"
        episode = RecallEpisode(
            session_source_id=f"receipt:{runtime}", tool_use_id=recall_id,
            query=receipt["query"], prompt=prompt, memories=memories,
            response=answer,
        )
        report = UsageReport(mode=self.mode,
                             semantic_active=self._embedder() is not None)
        judgments = self.observe_episode(episode, report)
        self.store.close_recall_receipt(recall_id, "redeemed")
        return ObserveResult(
            recall_id=recall_id, runtime=runtime, mode=self.mode,
            semantic_active=report.semantic_active, judgments=judgments,
            skipped={"already_judged": report.already_judged,
                     "missing": report.missing},
        )

    def observe_file(
        self, path: Path, session_source_id: str, report: UsageReport
    ) -> None:
        """Judge every closed recall episode in one session file."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            episodes = extract_recall_episodes(f, session_source_id)
        report.files_scanned += 1
        report.episodes += len(episodes)
        for episode in episodes:
            self.observe_episode(episode, report)

    def run_claude_code(self, adapter: Any, full: bool = False) -> UsageReport:
        """One pass over the adapter's session files changed since the last
        pass (all of them with ``full``). Same cursor discipline as the sync
        scheduler: stamp before reading, persist only after success."""
        report = UsageReport(
            mode=self.mode,
            semantic_active=self._embedder() is not None,
        )
        if self.mode == "off":
            return report
        since = None if full else self.store.get_sync_cursor(USAGE_CURSOR)
        started = datetime.now(timezone.utc)
        for path in adapter.session_files(since or EPOCH):
            self.observe_file(path, adapter.source_id_for(path), report)
        self.store.set_sync_cursor(USAGE_CURSOR, started)
        return report

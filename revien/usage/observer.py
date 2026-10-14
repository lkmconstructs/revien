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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from revien.daemon.scheduler import EPOCH

from .episodes import RecallEpisode, extract_recall_episodes
from .evidence import EmbedFn, UseEvidence, gather_evidence
from .judge import UseVerdict, judge_use

MODES = ("off", "shadow", "apply")
# Persisted scan position for Claude Code sessions, in the sync_cursors table
# beside the adapters' own (independent of the ingest cursor).
USAGE_CURSOR = "usage:claude-code"


def usage_mode(value: Optional[str] = None) -> str:
    """Resolve the mode: an explicit value wins, else ``REVIEN_USAGE``, else
    shadow. Unrecognized values resolve to shadow."""
    raw = value if value is not None else os.environ.get("REVIEN_USAGE", "")
    raw = raw.strip().lower()
    return raw if raw in MODES else "shadow"


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

    def observe_episode(self, episode: RecallEpisode, report: UsageReport) -> None:
        """Judge every not-yet-judged memory of one episode."""
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
            return  # nothing new: skip the embedding cost entirely
        scoped = episode.model_copy(update={"memories": fresh})
        for memory, evidence in gather_evidence(scoped, self._embedder()):
            verdict = self.judge(evidence)
            if verdict.abstain:
                report.abstained += 1
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

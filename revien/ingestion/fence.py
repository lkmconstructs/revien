"""
Revien Context Fence — strips recall re-entry out of ingested content BEFORE
it becomes new memory.

WHY: Revien's own retrieved memory gets injected back into prompts — a
system-reminder block the Claude Code harness wraps around user turns, the
``[Revien Memory Context]`` fence ollama_adapter builds from recall results,
the ``## Relevant memory (Revien)`` header hermes_provider prepends, the
``## Relevant Context (from N nodes)`` block langchain_adapter formats — and
every one of those routes eventually calls ``pipeline.ingest()`` on the WHOLE
exchange, injected text included. Left alone, the graph re-learns what it
already told you: recalled content gets re-extracted, re-indexed, and fed
back into the next recall with rising confidence it never earned on its
own. This module is the single choke point that stops that loop.

Deliberately marker-based only, NOT JSON/schema sniffing: a heuristic that
tries to detect "this looks like injected context" from structure is a
false-positive machine (a user pasting a markdown list starting with
"## Relevant..." loses real content). Every marker below is a literal string
this codebase itself emits or a harness is known to emit — strip exactly
those, nothing inferred.

Case-sensitive, on purpose: the markers are program-emitted constants, not
prose a user might casually capitalize differently. Loosening the match
trades false negatives (rare — the emitters are stable) for false positives
(a user's own text happening to contain the same words in different case).
"""

import re
from dataclasses import dataclass, field
from typing import List, Pattern, Tuple


@dataclass
class FenceResult:
    """What fencing did to one piece of content. Never silent — every
    stripped span is counted and attributed to a marker family so the
    caller can log (or refuse to ingest) instead of quietly rewriting text."""
    content: str
    stripped_spans: int = 0
    stripped_chars: int = 0
    markers: List[str] = field(default_factory=list)


# ── Marker patterns ───────────────────────────────────────────────────────
# (a) Claude Code harness system-reminders. Well-formed pairs, multiline,
# non-greedy so two adjacent reminders don't merge into one giant span.
_SYSTEM_REMINDER_PAIR = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL
)
_SYSTEM_REMINDER_OPEN = re.compile(r"<system-reminder>")

# (b) ollama_adapter's recall fence (get_context_for_prompt, ~line 223/248).
_MEMORY_CONTEXT_PAIR = re.compile(
    r"\[Revien Memory Context\].*?\[End Memory Context\]", re.DOTALL
)
_MEMORY_CONTEXT_OPEN = re.compile(r"\[Revien Memory Context\]")

# (c) hermes_provider's prefetch header (_format_context, ~line 617): the
# header line plus its contiguous "- " list lines, stopping at the first
# line that is neither a list line nor blank. re.MULTILINE so ^ anchors each
# line, not just the string start. The blank-line alternative is anchored
# with a lookahead to the next "\n" (or end of string) — WITHOUT it,
# `[ \t]*` alone is satisfied by a zero-width match on ANY line (blank or
# not), so the block would silently swallow one stray newline past every
# real boundary line instead of stopping cleanly.
_HERMES_HEADER_BLOCK = re.compile(
    r"^## Relevant memory \(Revien\).*(?:\n(?:-[ ].*|[ \t]*(?=\n|$)))*",
    re.MULTILINE,
)

# (d) langchain_adapter's memory block (_format_retrieval_response, ~line
# 341): the "## Relevant Context (from N nodes)" header through every
# "### Result" entry, stopping at the next h2 (a line starting "## ") or EOF.
_LANGCHAIN_CONTEXT_BLOCK = re.compile(
    r"^## Relevant Context \(from .*(?:\n(?!## ).*)*",
    re.MULTILINE,
)

# Collapse blank-line runs the STRIPPING leaves behind. 3+ newlines -> 2
# (one blank line), same as a human paragraph break. Only ever applied when
# something was actually stripped (see fence_content) — marker-free content
# passes through byte-identical, never silently rewritten for a newline run
# it walked in with. Idempotent: a text with no run of 3+ is unchanged by a
# second pass.
_NEWLINE_RUNS = re.compile(r"\n{3,}")


def _strip_paired(
    text: str, pair_re: Pattern, open_re: Pattern
) -> Tuple[str, int, int, int, int]:
    """Strip every well-formed marker pair, then handle a truncated tail —
    but ONLY when it's actually a truncated tail and not a mid-line false
    positive. If an OPENING marker survives with no matching close, that's
    the harness's own emission getting cut off mid-injection — but the
    harness always emits reminders at the START of a line (start-of-text or
    right after a "\\n"). A bare marker string sitting mid-line (e.g. quoted
    inside a code sample: ``if '<system-reminder>' in text:``) is not that —
    it is left untouched rather than nuking everything after it to EOF.

    Returns (text, pair_spans, pair_chars, truncated_spans, truncated_chars)
    — truncated counts are reported separately so the caller can attribute
    them to a distinct marker family; a whole-tail strip is a different
    (and much larger-blast-radius) event than a tidy pair removal and must
    never be disguised as one in the log."""
    pair_spans = 0
    pair_chars = 0

    def _sub(m: "re.Match") -> str:
        nonlocal pair_spans, pair_chars
        pair_spans += 1
        pair_chars += len(m.group(0))
        return ""

    text = pair_re.sub(_sub, text)

    trunc_spans = 0
    trunc_chars = 0
    m = open_re.search(text)
    if m is not None and (m.start() == 0 or text[m.start() - 1] == "\n"):
        trunc_spans = 1
        trunc_chars = len(text) - m.start()
        text = text[: m.start()]

    return text, pair_spans, pair_chars, trunc_spans, trunc_chars


def _strip_blocks(text: str, block_re: Pattern) -> Tuple[str, int, int]:
    """Strip every match of a self-terminating block pattern (one that
    already defines its own end — a boundary line or EOF — so there is no
    separate unclosed-tail case). Returns (text, spans, chars)."""
    spans = 0
    chars = 0

    def _sub(m: "re.Match") -> str:
        nonlocal spans, chars
        spans += 1
        chars += len(m.group(0))
        return ""

    text = block_re.sub(_sub, text)
    return text, spans, chars


def fence_content(text: str) -> FenceResult:
    """Strip every known recall-re-entry marker span out of ``text``.

    Idempotent: fencing already-fenced output is a no-op (the markers are
    gone, so nothing matches — and with nothing stripped, the newline
    collapse never runs, so there is no second-pass rewrite to even
    consider).
    """
    if not text:
        return FenceResult(content=text or "")

    content = text
    total_spans = 0
    total_chars = 0
    markers: List[str] = []

    def _apply_paired(c: str, pair_re: Pattern, open_re: Pattern, name: str) -> str:
        nonlocal total_spans, total_chars
        c, pair_spans, pair_chars, trunc_spans, trunc_chars = _strip_paired(
            c, pair_re, open_re
        )
        if pair_spans:
            total_spans += pair_spans
            total_chars += pair_chars
            markers.append(name)
        if trunc_spans:
            # Reported under its OWN marker family — a whole-tail strip is a
            # much bigger claim than "found a matched pair" and the log must
            # never blur the two together.
            total_spans += trunc_spans
            total_chars += trunc_chars
            markers.append(f"{name}_truncated")
        return c

    def _apply_blocks(c: str, block_re: Pattern, name: str) -> str:
        nonlocal total_spans, total_chars
        c, spans, chars = _strip_blocks(c, block_re)
        if spans:
            total_spans += spans
            total_chars += chars
            markers.append(name)
        return c

    content = _apply_paired(
        content, _SYSTEM_REMINDER_PAIR, _SYSTEM_REMINDER_OPEN, "system_reminder"
    )
    content = _apply_paired(
        content, _MEMORY_CONTEXT_PAIR, _MEMORY_CONTEXT_OPEN, "memory_context"
    )
    content = _apply_blocks(content, _HERMES_HEADER_BLOCK, "hermes_header")
    content = _apply_blocks(content, _LANGCHAIN_CONTEXT_BLOCK, "langchain_context")

    # Collapse newline runs ONLY when something was actually stripped — this
    # is cleanup for the holes stripping leaves, not a blanket rewrite of
    # marker-free content that happened to arrive with a 3+ newline run of
    # its own (that content is none of this module's business).
    if total_spans:
        content = _NEWLINE_RUNS.sub("\n\n", content)

    return FenceResult(
        content=content,
        stripped_spans=total_spans,
        stripped_chars=total_chars,
        markers=markers,
    )

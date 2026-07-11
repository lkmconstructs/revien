"""Claim Sovereignty Layer — Leg 2: ingestion-time temporal resolution.

Resolve the temporal expression in a turn ("yesterday", "last year", "7 May
2023") into WHEN-IT-HAPPENED, anchored to recorded_at (when it was SAID). The
output is always a [start, end] RANGE plus a granularity label and a confidence —
never a bare instant. This is the load-bearing honesty rule:

    "last year" is stored as the whole previous calendar year ("sometime in
    2025"), NOT a guessed day. A memory system that collapses a fuzzy reference
    to a false-precise date will later assert that date as fact — the exact
    failure the Claim Sovereignty Layer exists to prevent. Honesty over precision.

Unboundable expressions ("recently", "a while ago", "these days", "soon") resolve
to None — we do NOT invent a range for them. Relative expressions need an anchor;
without recorded_at only absolute dates resolve. Pure stdlib, deterministic, no
network, no model.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from revien.graph.schema import TemporalGranularity

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_NUMWORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Confidence anchors — RELATIVE ordering is the point: unambiguous absolute dates
# rank highest; coarse/ambiguous references rank lower so downstream gating can
# refuse to act on weak temporal evidence.
_C_ABS_DAY = 0.95
_C_ABS_MONTH = 0.90
_C_ABS_YEAR = 0.85
_C_YESTERDAY = 0.95
_C_AGO_DAY = 0.90
_C_AGO_WEEK = 0.80
_C_AGO_MONTH = 0.75
_C_AGO_YEAR = 0.70
_C_LAST_WEEK = 0.70
_C_LAST_MONTH = 0.70
_C_LAST_YEAR = 0.65  # the canonical fuzzy-range case
_C_THIS = 0.70
_C_NEXT = 0.55
_C_WEEKDAY = 0.60


@dataclass
class TemporalResolution:
    """A resolved event time as a half-open-ish [start, end] inclusive range."""
    start: datetime
    end: datetime
    granularity: TemporalGranularity
    confidence: float
    text: str  # the raw matched expression, kept verbatim


# ── range helpers (UTC, inclusive end-of-period) ──────────────────────────────
def _utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def _day_range(dt: datetime) -> Tuple[datetime, datetime]:
    return (_utc(dt.year, dt.month, dt.day),
            _utc(dt.year, dt.month, dt.day, 23, 59, 59))


def _month_range(year: int, month: int) -> Tuple[datetime, datetime]:
    last = calendar.monthrange(year, month)[1]
    return _utc(year, month, 1), _utc(year, month, last, 23, 59, 59)


def _year_range(year: int) -> Tuple[datetime, datetime]:
    return _utc(year, 1, 1), _utc(year, 12, 31, 23, 59, 59)


def _week_range(monday: datetime) -> Tuple[datetime, datetime]:
    start = _utc(monday.year, monday.month, monday.day)
    end_dt = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end_dt


def _shift_months(year: int, month: int, delta: int) -> Tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def _R(res: Tuple[datetime, datetime], g: TemporalGranularity, c: float, text: str
       ) -> TemporalResolution:
    return TemporalResolution(start=res[0], end=res[1], granularity=g,
                              confidence=c, text=text)


# ── absolute patterns (no anchor needed) ──────────────────────────────────────
_RE_DMY = re.compile(r"\b(\d{1,2})\s+([a-z]+)\.?,?\s+(\d{4})\b")
_RE_MDY = re.compile(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_RE_MY = re.compile(r"\b([a-z]+)\.?,?\s+(\d{4})\b")
_RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# ── relative patterns (need recorded_at) ──────────────────────────────────────
_RE_AGO = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(day|week|month|year)s?\s+ago\b")
_RE_LAST_UNIT = re.compile(r"\blast\s+(week|month|year)\b")
_RE_THIS_UNIT = re.compile(r"\bthis\s+(week|month|year)\b")
_RE_NEXT_UNIT = re.compile(r"\bnext\s+(week|month|year)\b")
_RE_LAST_WD = re.compile(r"\blast\s+(" + "|".join(_WEEKDAYS) + r")\b")
_RE_WD = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b")


def _resolve_absolute(t: str) -> Optional[TemporalResolution]:
    m = _RE_DMY.search(t)
    if m and m.group(2) in _MONTHS:
        day, mon, year = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
        try:
            return _R(_day_range(_utc(year, mon, day)), TemporalGranularity.DAY,
                      _C_ABS_DAY, m.group(0))
        except ValueError:
            return None
    m = _RE_MDY.search(t)
    if m and m.group(1) in _MONTHS:
        mon, day, year = _MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        try:
            return _R(_day_range(_utc(year, mon, day)), TemporalGranularity.DAY,
                      _C_ABS_DAY, m.group(0))
        except ValueError:
            return None
    m = _RE_MY.search(t)
    if m and m.group(1) in _MONTHS:
        mon, year = _MONTHS[m.group(1)], int(m.group(2))
        return _R(_month_range(year, mon), TemporalGranularity.MONTH,
                  _C_ABS_MONTH, m.group(0))
    m = _RE_YEAR.search(t)
    if m:
        return _R(_year_range(int(m.group(1))), TemporalGranularity.YEAR,
                  _C_ABS_YEAR, m.group(0))
    return None


def _resolve_relative(t: str, R: datetime) -> Optional[TemporalResolution]:
    if re.search(r"\byesterday\b", t):
        return _R(_day_range(R - timedelta(days=1)), TemporalGranularity.DAY,
                  _C_YESTERDAY, "yesterday")
    if re.search(r"\btoday\b", t):
        return _R(_day_range(R), TemporalGranularity.DAY, _C_YESTERDAY, "today")

    m = _RE_AGO.search(t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMWORDS[m.group(1)]
        unit = m.group(2)
        if unit == "day":
            return _R(_day_range(R - timedelta(days=n)), TemporalGranularity.DAY,
                      _C_AGO_DAY, m.group(0))
        if unit == "week":
            target = R - timedelta(weeks=n)
            mon = target - timedelta(days=target.weekday())
            return _R(_week_range(mon), TemporalGranularity.WEEK, _C_AGO_WEEK, m.group(0))
        if unit == "month":
            y, mo = _shift_months(R.year, R.month, -n)
            return _R(_month_range(y, mo), TemporalGranularity.MONTH, _C_AGO_MONTH, m.group(0))
        if unit == "year":
            return _R(_year_range(R.year - n), TemporalGranularity.YEAR, _C_AGO_YEAR, m.group(0))

    m = _RE_LAST_UNIT.search(t)
    if m:
        unit = m.group(1)
        if unit == "week":
            mon = R - timedelta(days=R.weekday() + 7)
            return _R(_week_range(mon), TemporalGranularity.WEEK, _C_LAST_WEEK, "last week")
        if unit == "month":
            y, mo = _shift_months(R.year, R.month, -1)
            return _R(_month_range(y, mo), TemporalGranularity.MONTH, _C_LAST_MONTH, "last month")
        if unit == "year":
            return _R(_year_range(R.year - 1), TemporalGranularity.YEAR, _C_LAST_YEAR, "last year")

    m = _RE_THIS_UNIT.search(t)
    if m:
        unit = m.group(1)
        if unit == "week":
            mon = R - timedelta(days=R.weekday())
            return _R(_week_range(mon), TemporalGranularity.WEEK, _C_THIS, "this week")
        if unit == "month":
            return _R(_month_range(R.year, R.month), TemporalGranularity.MONTH, _C_THIS, "this month")
        if unit == "year":
            return _R(_year_range(R.year), TemporalGranularity.YEAR, _C_THIS, "this year")

    m = _RE_NEXT_UNIT.search(t)
    if m:
        unit = m.group(1)
        if unit == "week":
            mon = R + timedelta(days=7 - R.weekday())
            return _R(_week_range(mon), TemporalGranularity.WEEK, _C_NEXT, "next week")
        if unit == "month":
            y, mo = _shift_months(R.year, R.month, 1)
            return _R(_month_range(y, mo), TemporalGranularity.MONTH, _C_NEXT, "next month")
        if unit == "year":
            return _R(_year_range(R.year + 1), TemporalGranularity.YEAR, _C_NEXT, "next year")

    m = _RE_LAST_WD.search(t) or _RE_WD.search(t)
    if m:
        wd = _WEEKDAYS[m.group(1)]
        # Most recent past occurrence of that weekday before R (>=1 day back).
        back = (R.weekday() - wd) % 7
        back = back or 7
        return _R(_day_range(R - timedelta(days=back)), TemporalGranularity.DAY,
                  _C_WEEKDAY, m.group(0))

    return None


def resolve_event_time(
    text: str, recorded_at: Optional[datetime]
) -> Optional[TemporalResolution]:
    """Resolve the primary temporal expression in `text` against `recorded_at`.

    Absolute dates resolve with or without an anchor and take priority; relative
    expressions need `recorded_at`. Returns None when nothing BOUNDABLE is found —
    vague references ("recently", "a while ago") are deliberately left unresolved
    rather than guessed into a false-precise date.
    """
    if not text:
        return None
    t = text.lower()
    absolute = _resolve_absolute(t)
    if absolute is not None:
        return absolute
    if recorded_at is not None:
        return _resolve_relative(t, recorded_at)
    return None

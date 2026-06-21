"""Claim Sovereignty Layer — Leg 2 resolver tests (relative->absolute, fuzzy->range).

The load-bearing assertions: "last year" resolves to a whole-year RANGE at lower
confidence (never a guessed day), and vague references resolve to None (never
invented). That is the honesty-over-precision thesis, tested.
"""

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from revien.graph.schema import TemporalGranularity
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.ingestion.temporal import resolve_event_time

# Anchor: 8 May 2023 (a Monday) — "recorded_at" for relative resolution.
R = datetime(2023, 5, 8, tzinfo=timezone.utc)


@pytest.fixture
def store():
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    fd.close()
    s = GraphStore(db_path=fd.name)
    yield s
    s.close()
    Path(fd.name).unlink(missing_ok=True)


def test_last_year_is_a_range_not_a_date():
    """THE honesty rule: 'last year' is the whole previous year, not a guessed day."""
    res = resolve_event_time("I painted a sunrise last year", R)
    assert res is not None
    assert res.granularity is TemporalGranularity.YEAR
    assert res.start == datetime(2022, 1, 1, tzinfo=timezone.utc)
    assert (res.start, res.end.month, res.end.day) == (
        datetime(2022, 1, 1, tzinfo=timezone.utc), 12, 31)
    assert (res.end - res.start).days >= 364  # spans the year
    assert res.confidence < 0.8               # weaker than an exact date
    assert res.text == "last year"            # raw expression kept verbatim


def test_vague_references_resolve_to_nothing():
    """The other half of the rule: unboundable references are NOT invented."""
    for vague in ["I painted recently", "a while ago", "these days",
                  "someday", "in the past", "a long time back"]:
        assert resolve_event_time(vague, R) is None


def test_yesterday():
    res = resolve_event_time("I saw it yesterday", R)
    assert res.granularity is TemporalGranularity.DAY
    assert res.start.date() == date(2023, 5, 7)
    assert res.end.date() == date(2023, 5, 7)


def test_numeric_and_word_days_ago():
    assert resolve_event_time("that was 3 days ago", R).start.date() == date(2023, 5, 5)
    assert resolve_event_time("two days ago we met", R).start.date() == date(2023, 5, 6)


def test_absolute_day_month_year():
    res = resolve_event_time("on 7 May 2023 we launched", R)
    assert res.granularity is TemporalGranularity.DAY
    assert res.start.date() == date(2023, 5, 7)
    assert res.confidence >= 0.9


def test_absolute_month_year_is_month_range():
    res = resolve_event_time("back in May 2023", R)
    assert res.granularity is TemporalGranularity.MONTH
    assert res.start == datetime(2023, 5, 1, tzinfo=timezone.utc)
    assert res.end.day == 31


def test_bare_year_is_year_range():
    res = resolve_event_time("it happened in 2022", R)
    assert res.granularity is TemporalGranularity.YEAR
    assert res.start.year == 2022


def test_last_month_is_prior_calendar_month():
    res = resolve_event_time("last month I started pottery", R)
    assert res.granularity is TemporalGranularity.MONTH
    assert res.start == datetime(2023, 4, 1, tzinfo=timezone.utc)
    assert res.end.day == 30


def test_last_week_is_a_seven_day_past_range():
    res = resolve_event_time("last week we went hiking", R)
    assert res.granularity is TemporalGranularity.WEEK
    assert (res.end - res.start).days == 6          # a 7-day span
    assert res.end.date() < R.date()                # entirely in the past


def test_relative_needs_an_anchor_but_absolute_does_not():
    assert resolve_event_time("yesterday", None) is None       # no recorded_at
    res = resolve_event_time("7 May 2023", None)               # absolute is fine
    assert res is not None and res.start.date() == date(2023, 5, 7)


def test_confidence_orders_exact_above_fuzzy():
    exact = resolve_event_time("7 May 2023", R)
    fuzzy = resolve_event_time("last year", R)
    assert exact.confidence > fuzzy.confidence


# ── pipeline integration ──────────────────────────────────────────────────────

def test_pipeline_attaches_event_time_range_to_turn(store):
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(
        source_id="t1",
        content="Melanie: I painted a sunrise last year.",
        timestamp=R,
    ))
    ctx = store.get_node(out.context_node_id)
    assert ctx.recorded_at == R
    assert ctx.event_time_granularity is TemporalGranularity.YEAR
    assert ctx.event_time_start.year == 2022
    assert ctx.event_time_end.year == 2022
    assert ctx.event_time_text == "last year"


def test_pipeline_leaves_event_time_null_for_vague_turn(store):
    pipe = IngestionPipeline(store)
    out = pipe.ingest(IngestionInput(
        source_id="t2",
        content="Melanie: I painted a sunrise recently.",
        timestamp=R,
    ))
    ctx = store.get_node(out.context_node_id)
    assert ctx.event_time_start is None
    assert ctx.event_time_granularity is None
    assert ctx.event_time_text == ""

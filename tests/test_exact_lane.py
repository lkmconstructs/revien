"""Exact lane (REVIEN_EXACT): detection rules, date matching on the structured
event-time fields, label matching, the bounded boost, the generic-title guard,
and the proof that switch-off / no-signal recall is byte-identical.

Synthetic graphs only. Similarities are set by a fixture embedder so a test
controls exactly what "meaning" says and what the lane adds.
"""

import dataclasses
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from revien.graph.schema import (
    Edge, EdgeType, Node, NodeType, SourceType, TemporalGranularity,
)
from revien.graph.store import GraphStore
from revien.retrieval.engine import RetrievalEngine
from revien.retrieval.exact import (
    BOOST_MAX, ExactConfig, ExactLane, lift_after_rerank, parse_signals,
)
from revien.semantic.rerank import CrossEncoderReranker

from tests.test_semantic import _InMemoryVectorIndex

UTC = timezone.utc
G = TemporalGranularity


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = GraphStore(db_path=path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except PermissionError:  # pragma: no cover
        pass


def add(store, label, content="", node_type=NodeType.FACT, **kw):
    return store.add_node(Node(
        node_type=node_type, label=label, content=content or label,
        source_type=SourceType.EXTRACTED, confidence=1.0, **kw))


def day(y, m, d):
    return dict(
        event_time_start=datetime(y, m, d, tzinfo=UTC),
        event_time_end=datetime(y, m, d, 23, 59, 59, tzinfo=UTC),
        event_time_granularity=G.DAY, event_time_confidence=0.95)


def month(y, m):
    import calendar
    last = calendar.monthrange(y, m)[1]
    return dict(
        event_time_start=datetime(y, m, 1, tzinfo=UTC),
        event_time_end=datetime(y, m, last, 23, 59, 59, tzinfo=UTC),
        event_time_granularity=G.MONTH, event_time_confidence=0.9)


class _Embedder:
    """sims: {text key -> cosine against the query axis}; any text containing
    a key embeds at that cosine. Queries listed in ``queries`` sit on the
    axis. Everything else is orthogonal (cosine 0, never a candidate)."""

    def __init__(self, sims, queries=()):
        self.sims, self.queries = sims, set(queries)

    dim = 3
    is_cloud = False

    def embed(self, texts):
        out = []
        for t in texts:
            tl = (t or "").lower()
            vec = [0.0, 0.0, 1.0]
            if tl in self.queries:
                vec = [1.0, 0.0, 0.0]
            else:
                for key, cos in self.sims.items():
                    if key in tl:
                        vec = [cos, math.sqrt(max(0.0, 1 - cos * cos)), 0.0]
                        break
            out.append(vec)
        return out


def engine_for(store, *, exact, sims=None, queries=(), reranker=None):
    sem = None
    if sims is not None:
        sem = _InMemoryVectorIndex(store, _Embedder(sims, queries))
        sem.reindex_all()
    return RetrievalEngine(store, semantic=sem, exact=exact, reranker=reranker)


def link(store, a, b, edge_type=EdgeType.RELATED_TO):
    return store.add_edge(Edge(edge_type=edge_type, source_node_id=a.node_id,
                               target_node_id=b.node_id, weight=0.6))


def kill(store, table, key, value):
    """Soft-invalidate a node or an edge the way the store does (a timestamp)."""
    conn = store._get_conn()
    conn.execute(f"UPDATE {table} SET invalidated_at = ? WHERE {key} = ?",
                 ("2026-01-01T00:00:00+00:00", value))
    conn.commit()


def ids_of(resp):
    return [r.node_id for r in resp.results]


def strip(resp):
    """A response as plain data, minus wall-clock time."""
    d = dataclasses.asdict(resp)
    d.pop("retrieval_time_ms")
    return d


def boosts(resp):
    return {r.node_id: r.score_breakdown["exact_boost"]
            for r in resp.results if "exact_boost" in r.score_breakdown}


# ── detection rules ───────────────────────────────────────────────────────────

class TestParse:
    @pytest.mark.parametrize("q", [
        "What happened on September 21?", "what about Sept 21", "Sep 21st",
        "sep. 21", "21st of September", "21 September", "9/21",
    ])
    def test_yearless_formats_resolve_to_sept_21(self, q):
        d = parse_signals(q).dates
        assert [(x.month, x.day, x.year, x.granularity) for x in d] == [
            (9, 21, None, "day")]

    @pytest.mark.parametrize("q", [
        "09/21/2026", "9/21/26", "2026-09-21", "2026/09/21",
        "21 September 2026", "September 21, 2026", "Sep 21st 2026",
        "21/09/2026",
    ])
    def test_year_formats_resolve_to_one_day(self, q):
        d = parse_signals(q).dates
        assert [(x.month, x.day, x.year) for x in d] == [(9, 21, 2026)]

    def test_month_year_is_a_month(self):
        d = parse_signals("notes from September 2026").dates
        assert [(x.month, x.day, x.year, x.granularity) for x in d] == [
            (9, None, 2026, "month")]

    def test_fraction_and_invalid_dates_are_not_dates(self):
        assert parse_signals("add 1/2 cup and 3/4 tsp").dates == []
        assert parse_signals("13/45 and Feb 30 and 2026-13-01").dates == []

    def test_may_is_a_month_only_capitalized(self):
        assert parse_signals("it may 5 times").dates == []
        assert len(parse_signals("on May 5").dates) == 1

    def test_dates_do_not_leak_into_names_or_ids(self):
        s = parse_signals("What did I do on September 21 2026?")
        assert s.ids == [] and [x.text for x in s.spans] == []

    def test_quoted_phrases(self):
        s = parse_signals('Find "Project Atlas" and “weekly sync”, not it\'s')
        assert s.quoted == ["Project Atlas", "weekly sync"]

    def test_ids_and_codes(self):
        s = parse_signals("see PR-1234, v0.3.0, user_id_map and REV-FIX but "
                          "not well-known, 21st, 3pm, 10:30 or 2026")
        assert s.ids == ["PR-1234", "v0.3.0", "user_id_map", "REV-FIX"]

    def test_names_mid_sentence_and_question_words_dropped(self):
        s = parse_signals("Did Priya tell Dana Reyes about it?")
        texts = {x.text for x in s.spans}
        assert {"Priya", "Dana Reyes", "Dana", "Reyes"} <= texts
        assert "Did" not in texts          # single question word at sentence start
        start = {x.text for x in s.spans if x.sentence_start}
        assert "Did Priya" in start and "Priya" not in start

    def test_no_signal_for_plain_paraphrase(self):
        s = parse_signals("what do I usually do when I feel overwhelmed at work")
        assert not (s.quoted or s.dates or s.ids or s.spans)


# ── date matching on the structured fields ───────────────────────────────────

@pytest.fixture
def dated(store):
    n = {}
    n["day"] = add(store, "Dentist visit", "Went in for a cleaning.", **day(2026, 9, 21))
    n["other_year"] = add(store, "Old trip", "A trip.", **day(2025, 9, 21))
    n["next_day"] = add(store, "Hiking", "A hike.", **day(2026, 9, 22))
    n["month"] = add(store, "Quarter planning", "Planning.", **month(2026, 9))
    n["other_month"] = add(store, "October sync", "Sync.", **month(2026, 10))
    # No event date, but the TEXT says the date: a date must never match it.
    n["texty"] = add(store, "Text only", "We met on September 21 2026 (9/21).")
    return n


def exact_for(store, q, **env):
    lane = ExactLane(store, ExactConfig.from_env())
    return lane.build(q, [])


class TestDates:
    SAME_DAY = ["September 21, 2026", "09/21/2026", "2026-09-21",
                "21 September 2026", "Sep 21st 2026", "9/21/26"]
    ANY_YEAR = ["September 21", "Sept 21", "Sep 21st", "9/21", "21 September"]

    def test_every_format_lands_on_the_same_memory(self, store, dated):
        for q in self.SAME_DAY:
            r = exact_for(store, f"what happened on {q}")
            assert r is not None, q
            assert r.boost[dated["day"].node_id] == pytest.approx(BOOST_MAX), q
            assert dated["other_year"].node_id not in r.boost, q
            assert dated["next_day"].node_id not in r.boost, q
        for q in self.ANY_YEAR:
            r = exact_for(store, f"what happened on {q}")
            assert r is not None, q
            # same memory, in every year the graph has
            assert r.boost[dated["day"].node_id] == pytest.approx(BOOST_MAX), q
            assert r.boost[dated["other_year"].node_id] == pytest.approx(BOOST_MAX), q
            assert dated["next_day"].node_id not in r.boost, q

    def test_all_formats_agree_exactly(self, store, dated):
        got = [exact_for(store, f"on {q}").boost for q in self.SAME_DAY]
        assert all(g == got[0] for g in got)
        got = [exact_for(store, f"on {q}").boost for q in self.ANY_YEAR]
        assert all(g == got[0] for g in got)

    def test_granularity_month_matches_any_day_at_lower_weight(self, store, dated):
        r = exact_for(store, "what happened on September 21, 2026")
        b = r.boost
        assert b[dated["month"].node_id] == pytest.approx(BOOST_MAX * 0.4)
        assert b[dated["month"].node_id] < b[dated["day"].node_id]
        # any day inside the month, not just the 21st
        r2 = exact_for(store, "what happened on 2026-09-03")
        assert dated["month"].node_id in r2.boost
        assert dated["day"].node_id not in r2.boost
        # a different month is not matched
        assert dated["month"].node_id not in exact_for(
            store, "on October 5, 2026").boost
        assert dated["other_month"].node_id in exact_for(
            store, "on October 5, 2026").boost

    def test_month_query_matches_day_nodes_at_month_weight(self, store, dated):
        r = exact_for(store, "notes from September 2026")
        assert r.boost[dated["day"].node_id] == pytest.approx(BOOST_MAX * 0.4)
        assert r.boost[dated["month"].node_id] == pytest.approx(BOOST_MAX * 0.4)
        assert dated["other_year"].node_id not in r.boost

    def test_undated_nodes_are_never_matched_by_a_date(self, store, dated):
        for q in self.SAME_DAY + self.ANY_YEAR:
            r = exact_for(store, f"on {q}")
            assert dated["texty"].node_id not in r.boost
            assert dated["texty"].node_id not in r.matches

    def test_date_in_recall_lifts_and_admits_the_dated_node(self, store, dated):
        eng = engine_for(store, exact=True)
        resp = eng.recall("what happened on 9/21/2026", top_n=10, debug=True)
        ids = [r.node_id for r in resp.results]
        assert dated["day"].node_id in ids
        assert dated["texty"].node_id not in ids
        assert resp.diagnostics["exact"]["detected"]["dates"][0]["year"] == 2026
        assert dated["day"].node_id in resp.diagnostics["exact"]["injected"]
        assert boosts(resp)[dated["day"].node_id] == pytest.approx(BOOST_MAX)

    def test_switch_off_ignores_dates(self, store, dated):
        resp = engine_for(store, exact=False).recall(
            "what happened on 9/21/2026", top_n=10, debug=True)
        assert not boosts(resp)
        assert "exact" not in resp.diagnostics
        assert dated["day"].node_id not in [r.node_id for r in resp.results]

    def test_feb_29_year_less_only_in_leap_years(self, store):
        leap = add(store, "Leap day", "x", **day(2024, 2, 29))
        add(store, "Other", "y", **day(2025, 2, 28))
        r = exact_for(store, "on Feb 29")
        assert list(r.boost) == [leap.node_id]


# ── names, titles, quotes, ids ───────────────────────────────────────────────

def said(store, label, when, source="content", **kw):
    meta = {"recorded_at_source": source} if source else {}
    return add(store, label, f"{label} text", recorded_at=when, metadata=meta, **kw)


class TestRecordedAtDates:
    D = datetime(2026, 9, 21, 22, 30, tzinfo=timezone(timedelta(hours=-4)))

    def test_recorded_at_matches_at_half_weight_by_local_date(self, store):
        # 22:30 at -04:00 is 02:30 UTC on the 22nd: the speaker's day is the 21st
        n = said(store, "Said it", self.D)
        r = exact_for(store, "what did I say on September 21, 2026")
        assert r.boost[n.node_id] == pytest.approx(BOOST_MAX * 0.5)
        assert r.matches[n.node_id][0]["field"] == "recorded_at"
        assert n.node_id not in exact_for(store, "on 2026-09-22").boost

    @pytest.mark.parametrize("src", ["content", "capture", "import"])
    def test_trusted_sources_match(self, store, src):
        n = said(store, "Said it", self.D, source=src)
        assert n.node_id in exact_for(store, "on 9/21/2026").boost

    @pytest.mark.parametrize("src", ["mtime", None])
    def test_untrusted_or_unlabeled_source_never_matches(self, store, src):
        n = said(store, "Said it", self.D, source=src)
        assert n.node_id not in exact_for(store, "on 9/21/2026").boost

    def test_created_at_is_never_used(self, store):
        n = add(store, "Ingested", "x")      # created_at is now; no dates at all
        today = datetime.now(timezone.utc).date()
        r = exact_for(store, f"on {today.isoformat()}")
        assert r is None or n.node_id not in r.boost

    def test_event_time_wins_over_recorded_at(self, store):
        n = said(store, "Both", self.D, **day(2026, 9, 21))
        r = exact_for(store, "on 2026-09-21")
        assert r.boost[n.node_id] == pytest.approx(BOOST_MAX)
        assert [m["field"] for m in r.matches[n.node_id]] == ["event_time"]

    def test_yearless_uses_recorded_years_and_reports_field(self, store):
        n = said(store, "Said it", self.D)
        r = exact_for(store, "on Sept 21")
        assert n.node_id in r.boost
        assert r.detected["dates"][0]["matched_by_recorded_at"] == [n.node_id]
        assert r.detected["dates"][0]["matched_by_event_time"] == []


class TestLabels:
    def test_exact_label_case_and_whitespace_insensitive(self, store):
        n = add(store, "Dana Reyes", "A person.", node_type=NodeType.ENTITY)
        for q in ["tell me about Dana   Reyes", "Tell me about DANA REYES?",
                  'what is "DANA REYES"']:
            r = exact_for(store, q)
            assert r is not None and n.node_id in r.boost, q

    def test_longest_span_wins_and_tokens_not_reused(self, store):
        short = add(store, "Dana", "x", node_type=NodeType.ENTITY)
        long = add(store, "Dana Reyes", "x", node_type=NodeType.ENTITY)
        r = exact_for(store, "tell me about Dana Reyes")
        assert long.node_id in r.boost and short.node_id not in r.boost

    def test_sentence_start_name_needs_a_known_label(self, store):
        n = add(store, "Revien", "x", node_type=NodeType.ENTITY)
        assert n.node_id in exact_for(store, "Revien is what?").boost
        assert exact_for(store, "Unknownthing is what?") is None

    def test_single_lowercase_title_query(self, store):
        n = add(store, "harbor", "x", node_type=NodeType.TOPIC)
        assert n.node_id in exact_for(store, "harbor").boost
        assert exact_for(store, "a walk by the harbor today") is None

    def test_id_matches_label_or_whole_token_in_label_not_content(self, store):
        eq = add(store, "PR-1234", "the fix")
        inlabel = add(store, "Merged PR-1234 to main", "x")
        content_only = add(store, "Some change", "see PR-1234 for details")
        sub = add(store, "PR-12345", "x")
        r = exact_for(store, "what is PR-1234")
        assert r.boost[eq.node_id] == pytest.approx(BOOST_MAX)
        assert r.boost[inlabel.node_id] == pytest.approx(BOOST_MAX * 0.7)
        assert content_only.node_id not in r.boost
        assert sub.node_id not in r.boost

    def test_node_id_typed_into_query(self, store):
        n = add(store, "Anything", "x")
        r = exact_for(store, f"open {n.node_id}")
        assert r is not None and r.boost[n.node_id] == pytest.approx(BOOST_MAX)

    def test_origin_filter_scopes_the_lookup(self, store):
        a = add(store, "Dana Reyes", "x", origin_runtime="codex")
        b = add(store, "Dana Reyes", "x", origin_runtime="claude-code")
        lane = ExactLane(store, ExactConfig.from_env())
        r = lane.build("about Dana Reyes", [], origin_runtime=["codex"])
        assert set(r.boost) == {a.node_id}


# ── the guard ─────────────────────────────────────────────────────────────────

class TestGenericGuard:
    def test_common_word_title_gets_nothing(self, store):
        n = add(store, "Notes", "x")
        r = exact_for(store, "what is in my Notes")
        assert n.node_id not in r.boost
        assert r.matches[n.node_id][0]["guard"] == "common_words"

    def test_title_shared_by_many_nodes_gets_nothing(self, store):
        ids = [add(store, "Zephyr", f"v{i}").node_id for i in range(5)]
        r = exact_for(store, "about Zephyr please")
        assert not set(ids) & set(r.boost)

    def test_df_scales_boost_down(self, store):
        solo = add(store, "Orinoco", "x")
        pair = [add(store, "Kestrel", f"v{i}").node_id for i in range(2)]
        r = exact_for(store, "the Orinoco and the Kestrel")
        assert r.boost[solo.node_id] == pytest.approx(BOOST_MAX * 0.8)
        for p in pair:
            assert r.boost[p] == pytest.approx(BOOST_MAX * 0.8 * 0.5)

    def test_threshold_is_configurable(self, store, monkeypatch):
        ids = [add(store, "Zephyr", f"v{i}").node_id for i in range(5)]
        monkeypatch.setenv("REVIEN_EXACT_MAX_DF", "5")
        r = ExactLane(store, ExactConfig.from_env()).build("about Zephyr please", [])
        assert set(ids) <= set(r.boost)

    def test_common_words_extendable(self, store, monkeypatch):
        n = add(store, "Orinoco", "x")
        monkeypatch.setenv("REVIEN_EXACT_COMMON_WORDS", "orinoco")
        r = ExactLane(store, ExactConfig.from_env()).build("the Orinoco", [])
        assert n.node_id not in r.boost

    def test_generic_title_does_not_hijack_ranking(self, store):
        sims = {"overwhelm": 0.80, "notes about": 0.45}
        want = add(store, "Coping", "overwhelm strategies for hard weeks")
        generic = [add(store, "Notes", f"notes about thing {i}") for i in range(4)]
        q = "how do I handle my Notes on overwhelm"
        off = engine_for(store, exact=False, sims=sims, queries=[q.lower()])
        on = engine_for(store, exact=True, sims=sims, queries=[q.lower()])
        r_off, r_on = off.recall(q, top_n=10), on.recall(q, top_n=10)
        assert [r.node_id for r in r_off.results] == [r.node_id for r in r_on.results]
        assert r_on.results[0].node_id == want.node_id
        assert not boosts(r_on)


# ── bounded boost, name lifting, injection ───────────────────────────────────

class TestBoost:
    def _graph(self, store):
        # decoys outrank the named node on meaning alone
        d1 = add(store, "Garden plans", "what to plant in the garden")
        d2 = add(store, "Garden tools", "rakes and hoes for the garden")
        target = add(store, "Dana Reyes", "Dana keeps bees.", node_type=NodeType.ENTITY)
        far = add(store, "Theo Brandt", "Theo fixes bikes.", node_type=NodeType.ENTITY)
        sims = {"garden plans": 0.70, "garden tools": 0.66, "dana keeps": 0.62,
                "theo fixes": 0.34}
        return d1, d2, target, far, sims

    def test_exact_name_lifts_the_right_node(self, store):
        d1, d2, target, far, sims = self._graph(store)
        q = "who is Dana Reyes"          # about the entity: the entity is lifted
        off = engine_for(store, exact=False, sims=sims, queries=[q.lower()])
        on = engine_for(store, exact=True, sims=sims, queries=[q.lower()])
        r_off, r_on = off.recall(q, top_n=10), on.recall(q, top_n=10)
        rank = lambda r, n: [x.node_id for x in r.results].index(n.node_id)
        assert rank(r_off, target) > 0
        assert rank(r_on, target) == 0
        assert boosts(r_on) == {target.node_id: pytest.approx(BOOST_MAX * 1.0)}
        # the breakdown key appears only where it applies
        assert all("exact_boost" not in x.score_breakdown
                   for x in r_on.results if x.node_id != target.node_id)
        assert all("exact_boost" not in x.score_breakdown for x in r_off.results)

    def test_boost_is_capped_not_a_promotion(self, store):
        d1, d2, target, far, sims = self._graph(store)
        # name a node whose meaning is far behind the leader
        q = "who is Theo Brandt"
        on = engine_for(store, exact=True, sims=sims, queries=[q.lower()])
        resp = on.recall(q, top_n=10)
        ids = [r.node_id for r in resp.results]
        assert ids[0] == d1.node_id
        if far.node_id in ids:
            assert ids.index(far.node_id) > 0
        b = boosts(resp)
        assert b and all(v <= BOOST_MAX + 1e-9 for v in b.values())

    def test_boost_cap_holds_when_signals_stack(self, store):
        n = add(store, "Dana Reyes", "x", node_type=NodeType.ENTITY,
                **day(2026, 9, 21))
        r = exact_for(store, 'see "Dana Reyes" and Dana Reyes on 2026-09-21')
        assert len(r.matches[n.node_id]) >= 3
        assert r.boost[n.node_id] == pytest.approx(BOOST_MAX)

    def test_cap_is_configurable(self, store, monkeypatch):
        n = add(store, "Dana Reyes", "x", node_type=NodeType.ENTITY)
        monkeypatch.setenv("REVIEN_EXACT_BOOST_MAX", "0.03")
        r = ExactLane(store, ExactConfig.from_env()).build("Dana Reyes", [])
        assert r.boost[n.node_id] == pytest.approx(0.03)

    def test_injection_is_bounded(self, store, monkeypatch):
        for i in range(8):
            add(store, f"Event {i}", "x", **day(2026, 9, 21))
        monkeypatch.setenv("REVIEN_EXACT_MAX_INJECT", "2")
        r = ExactLane(store, ExactConfig.from_env()).build("on 2026-09-21", [])
        assert len(r.boost) == 8 and len(r.inject_ids) == 2
        r0 = ExactLane(store, ExactConfig.from_env()).build(
            "on 2026-09-21", list(r.boost))
        assert r0.inject_ids == []     # already anchors: nothing to inject

    def test_crowded_date_is_scaled_down(self, store, monkeypatch):
        for i in range(10):
            add(store, f"Import {i}", "x", **day(2026, 9, 21))
        monkeypatch.setenv("REVIEN_EXACT_DATE_MAX_MATCHES", "5")
        r = ExactLane(store, ExactConfig.from_env()).build("on 2026-09-21", [])
        assert all(v == pytest.approx(BOOST_MAX * 0.5) for v in r.boost.values())


# ── false dates, ordinary words, possessives (fixes for the first build) ─────

class TestFalseDates:
    REAL = [
        ("September 21", (9, 21, None)), ("Sept 21st", (9, 21, None)),
        ("9/21", (9, 21, None)), ("09/21/2026", (9, 21, 2026)),
        ("2026-09-21", (9, 21, 2026)), ("21 September 2026", (9, 21, 2026)),
    ]

    @pytest.mark.parametrize("q,want", REAL)
    def test_real_dates_still_parse(self, q, want):
        for text in (q, f"what happened on {q}?"):
            d = parse_signals(text).dates
            assert [(x.month, x.day, x.year) for x in d] == [want], text

    @pytest.mark.parametrize("q", ["we are open 24/7", "24/7"])
    def test_24_7_is_not_a_date(self, q):
        assert parse_signals(q).dates == []

    @pytest.mark.parametrize("q", [
        "it was 8/10", "I give it 9/10", "a solid 1/10", "she scored 10/20",
        "8/10", "9/10", "1/10", "10/20", "5/5 stars", "10/10 would again",
    ])
    def test_scores_are_not_dates(self, q):
        assert parse_signals(q).dates == []

    @pytest.mark.parametrize("q", ["a 16/9 screen", "16/9", "21/9 ultrawide"])
    def test_ratios_with_no_month_are_not_dates(self, q):
        assert parse_signals(q).dates == []

    def test_ratio_rule_leaves_other_slash_pairs_alone(self):
        # a > b (a score never exceeds its denominator), b not a denominator
        for q, want in [("11/10", (11, 10)), ("12/25", (12, 25)),
                        ("3/7", (3, 7)), ("1/15", (1, 15))]:
            d = parse_signals(q).dates
            assert [(x.month, x.day) for x in d] == [want], q

    @pytest.mark.parametrize("q", [
        "release 2024.1.5", "v2024.01.05 shipped", "upgrade to 2026.09.21",
        "build 2024.1.5.2",
    ])
    def test_dotted_versions_are_not_dates(self, q):
        assert parse_signals(q).dates == []

    @pytest.mark.parametrize("q", [
        "we march 3 miles", "they march 3 miles to town", "it may 5 times",
        "jan 5 said hello", "I mar 3 things", "the dogs may 9 times a day",
    ])
    def test_lowercase_ambiguous_month_words_are_words(self, q):
        assert parse_signals(q).dates == []

    @pytest.mark.parametrize("q,want", [
        ("March 3", (3, 3)), ("May 5", (5, 5)), ("Jan 5", (1, 5)),
        ("MAR 3", (3, 3)), ("on march 3", (3, 3)), ("by may 5", (5, 5)),
        ("since jan 5", (1, 5)), ("from mar 3", (3, 3)),
        ("on the may 5", (5, 5)), ("the 5th of may", (5, 5)),
        ("3rd march", (3, 3)),
    ])
    def test_ambiguous_months_with_a_signal_still_parse(self, q, want):
        d = parse_signals(q).dates
        assert [(x.month, x.day) for x in d] == [want]

    def test_ambiguous_month_and_year(self):
        assert parse_signals("in march 2026").dates[0].granularity == "month"
        assert parse_signals("we march 2026 times").dates == []

    @pytest.mark.parametrize("q", [
        "what happened september 21", "sep. 21", "oct 3", "21 sept 2026",
        "dec 24", "aug 9", "nov 11", "apr 1",
    ])
    def test_unambiguous_month_words_parse_in_any_case(self, q):
        assert len(parse_signals(q).dates) == 1

    @pytest.mark.parametrize("q,want", [
        ("on 3/10", (3, 10)), ("by 8/10", (8, 10)), ("since 5/5", (5, 5)),
        ("until 10/10", (10, 10)), ("till 4/20", (4, 20)),
        ("before 3/10", (3, 10)), ("after 2/2", (2, 2)),
        ("through 5/5", (5, 5)), ("on the 3/10", (3, 10)),
        ("by the 3/10", (3, 10)), ("what happened on 3/10?", (3, 10)),
    ])
    def test_a_slash_pair_after_a_date_preposition_is_a_date(self, q, want):
        d = parse_signals(q).dates
        assert (d[0].month, d[0].day, d[0].year) == (*want, None), q

    @pytest.mark.parametrize("q,want", [
        ("03/10", (3, 10)), ("06/20", (6, 20)), ("05/05", (5, 5)),
        ("4/05", (4, 5)), ("what is 08/10", (8, 10)),
    ])
    def test_a_zero_padded_slash_pair_is_a_date(self, q, want):
        d = parse_signals(q).dates
        assert (d[0].month, d[0].day, d[0].year) == (*want, None), q

    @pytest.mark.parametrize("q,want", [
        ("3/10/2026", (3, 10, 2026)), ("5/5/26", (5, 5, 2026)),
        ("rated it 10/10/2026", (10, 10, 2026)),
    ])
    def test_a_slash_pair_with_a_year_is_a_date(self, q, want):
        d = parse_signals(q).dates
        assert (d[0].month, d[0].day, d[0].year) == want, q

    @pytest.mark.parametrize("q", [
        "8/10", "rated 9/10", "a 9/10 rating", "scored 10/20", "in 8/10 cases",
        "rated 9/10 by critics", "it got 5/5", "1/10 would not", "a 3/4 cup",
        "the score was 4/20 today",
    ])
    def test_without_a_preposition_a_score_stays_a_score(self, q):
        assert parse_signals(q).dates == [], q

    def test_a_preposition_does_not_make_an_impossible_date(self):
        for q in ["on 24/7", "from 16/9", "by 13/45", "on 2/30"]:
            assert parse_signals(q).dates == [], q

    def test_366_bare_month_day_forms(self):
        import calendar
        forms = [(m, d) for m in range(1, 13)
                 for d in range(1, calendar.monthrange(2000, m)[1] + 1)]
        assert len(forms) == 366
        ok = lambda q: len(parse_signals(q).dates) == 1
        # 50 of them read as ratios or scores with nothing to mark them
        assert sum(ok(f"{m}/{d}") for m, d in forms) == 316
        # a date preposition rescues every one
        for prep in ["on", "by", "since", "until", "till", "before", "after",
                     "through"]:
            assert all(ok(f"{prep} {m}/{d}") for m, d in forms), prep
        # these words do not ("a score of 8/10", "between 1/2 and 3/4")
        for word in ["in", "of", "from", "between", "during", "and", "to",
                     "score of", "out of"]:
            assert sum(ok(f"{word} {m}/{d}") for m, d in forms) == 316, word
        # zero-padding rescues all but the four with nothing to pad
        assert sum(ok(f"{m:02d}/{d:02d}") for m, d in forms) == 362

    def test_a_false_date_does_not_hide_a_real_one(self):
        d = parse_signals("we march 3 miles and then Sept 21 at 24/7 for 8/10")
        assert [(x.month, x.day) for x in d.dates] == [(9, 21)]


# ── a slash pair is a date only when something marks it as one ────────────────

def _dates(q):
    """(month, day, year) per date; a range is ("range", start, end)."""
    out = []
    for d in parse_signals(q).dates:
        if d.is_range:
            end_year = None if d.year is None else d.year + d.end_year_offset
            out.append(("range", (d.month, d.day, d.year),
                        (d.end_month, d.end_day, end_year)))
        else:
            out.append((d.month, d.day, d.year))
    return out


class TestSlashPairsNeedAMarker:
    """A wrong date is worse than none: a ratio-shaped a/b is a date only
    after a date preposition (on by before after since until till through),
    zero-padded, or with a year. Nothing else marks one."""

    @pytest.mark.parametrize("q", [
        # "of", "out of", "score of", "ratio of"
        "a score of 8/10", "a score of 8/10 on the quiz", "the ratio of 3/4",
        "out of 3/4", "of 4/5", "of 10/20", "the week of 3/10",
        # "between", "from", "during", "and", "to"
        "between 1/2 and 3/4 cup", "from 3/4 to 3/8", "from 1/2 to 3/4",
        "from 6/20", "during 6/6", "3/4 and 1/2",
        # a fraction OF something, even after a date preposition
        "3/4 of the team", "through 3/4 of the book", "on 1/2 of the days",
        "after 3/4 of the votes were counted", "by 2/3 of the members",
        # "by" an amount
        "reduced by 1/2", "cut by 1/3", "the price increased by 1/4",
        "we grew by 2/3", "multiply 3 by 3/4", "divided by 2/3",
        # no marker at all
        "in 2/3 cases", "rated 9/10", "8/10", "1/2 cup flour", "3/4",
    ])
    def test_not_a_date(self, q):
        assert _dates(q) == [], q

    @pytest.mark.parametrize("q,want", [
        ("on 3/4", [(3, 4, None)]), ("by 3/4", [(3, 4, None)]),
        ("before 3/4", [(3, 4, None)]), ("after 3/4", [(3, 4, None)]),
        ("since 3/4", [(3, 4, None)]), ("until 3/4", [(3, 4, None)]),
        ("till 3/4", [(3, 4, None)]), ("through 3/4", [(3, 4, None)]),
        ("On 3/4 we met", [(3, 4, None)]), ("due by the 3/10", [(3, 10, None)]),
        ("finished by 3/4", [(3, 4, None)]),
        ("what happened on 3/4?", [(3, 4, None)]),
        # zero-padded or with a year: a date whatever comes before
        ("03/04", [(3, 4, None)]), ("by 03/04", [(3, 4, None)]),
        ("from 03/04 to 03/08", [("range", (3, 4, None), (3, 8, None))]),
        ("the week of 03/10", [(3, 10, None)]),
        ("3/4/2025", [(3, 4, 2025)]),
        ("from 3/4/2026 to 3/8/2026", [("range", (3, 4, 2026), (3, 8, 2026))]),
        ("between 1/2/2026 and 3/4/2026", [(1, 2, 2026), (3, 4, 2026)]),
        # not ratio-shaped: read with no marker, as before
        ("12/25", [(12, 25, None)]), ("between 3/10 and 3/12", [(3, 12, None)]),
        # a score next to a real date: only the date
        ("a score of 8/10 on 3/4", [(3, 4, None)]),
    ])
    def test_a_date(self, q, want):
        assert _dates(q) == want, q

    @pytest.mark.parametrize("q,want", [
        ("Q3", []), ("Q3 2026 plan", []), ("the 90s", []),
        ("music from the 90s", []), ("2026-09-21", [(9, 21, 2026)]),
        ("what happened on 2026-09-21?", [(9, 21, 2026)]),
        ("May 5", [(5, 5, None)]), ("may 5", []), ("it may 5 times", []),
        ("May", []), ("March is coming", []), ("Did March call?", []),
        ("on march 3", [(3, 3, None)]), ("the 5th of may", [(5, 5, None)]),
        ("in march 2026", [(3, None, 2026)]), ("24/7 support", []),
        ("16/9 screen", []), ("Dana's notes", []),
    ])
    def test_unchanged_cases(self, q, want):
        assert _dates(q) == want, q

    def test_march_is_still_a_name_and_possessives_still_drop(self):
        assert "March" in {s.text for s in parse_signals("Did March call?").spans}
        assert {s.text for s in parse_signals("Dana's notes").spans} == {"Dana"}
        assert {s.text for s in parse_signals("Dana’s notes").spans} == {"Dana"}


class TestDetectionIsLinear:
    """Each date pattern is read in one left-to-right pass (the old reader
    restarted from the start after every date: "on 3/4 " * 20000 took 48.8 s),
    and the lane reads at most SCAN_CHARS characters of a query."""

    HOSTILE = {
        "slash dates": "on 3/4 " * 30000,
        "ratios": "3/4 " * 50000,
        "iso dates": "2026-09-21 " * 20000,
        "month days": "September 21 " * 16000,
        "ambiguous months": "on May 5 may 6 " * 14000,
        "quotes": '"Project Atlas" ' * 13000,
        "unbalanced quotes": '"' * 200000,
        "single quotes": "'a' 'bb' " * 23000,
        "names": " ".join(["Alpha", "Beta", "Gamma", "Delta"] * 9000),
        "codes": " ".join("PR-%d" % i for i in range(25000)),
        "possessives": "Dana's " * 29000,
        "dots": "a" + "." * 200000 + "a",
        "slashes": "1/" * 100000,
    }

    @pytest.mark.parametrize("name", sorted(HOSTILE))
    def test_a_200kb_hostile_query_builds_well_under_a_second(self, store, dated, name):
        import time
        from revien.retrieval.exact import SCAN_CHARS
        q = self.HOSTILE[name]
        assert len(q) >= 140_000
        lane = ExactLane(store, ExactConfig())
        t0 = time.perf_counter()
        r = lane.build(q, [])
        assert time.perf_counter() - t0 < 1.0, name
        if r is not None:
            assert r.detected["scanned_chars"] <= SCAN_CHARS

    @pytest.mark.parametrize("name", sorted(HOSTILE))
    def test_detection_without_the_bound_is_linear(self, name):
        # The bound is not what keeps detection fast: read in full, 200 KB of
        # any of these takes well under a second or two (quadratic took 48.8 s
        # at 140 KB). Generous threshold for a slow machine.
        import time
        from revien.retrieval.exact import parse_dates
        q = self.HOSTILE[name]
        t0 = time.perf_counter()
        parse_dates(q)
        parse_signals(q, max_chars=10 ** 9)
        assert time.perf_counter() - t0 < 6.0, name

    def test_every_date_in_a_long_query_is_read_without_the_bound(self):
        from revien.retrieval.exact import parse_dates
        dates, _ = parse_dates("on 3/4 " * 20000)
        assert len(dates) == 20000

    def test_a_long_query_is_read_at_both_ends(self, store, monkeypatch):
        from revien.retrieval.exact import SCAN_CHARS
        k = add(store, "Project Kestrel", "k")
        d = add(store, "Dana Reyes", "d")
        filler = "x " * 3000                       # 6000 characters
        assert len(filler) > SCAN_CHARS
        # the question comes last: the tail is read
        r = ExactLane(store, ExactConfig()).build(filler + "Project Kestrel", [])
        assert k.node_id in r.matches
        assert r.detected["truncated"] is True
        assert r.detected["scanned_chars"] <= SCAN_CHARS
        assert r.detected["query_chars"] == len(filler) + len("Project Kestrel")
        # and the head is read too
        r = ExactLane(store, ExactConfig()).build("Who is Dana Reyes? " + filler, [])
        assert d.node_id in r.matches
        # the middle is not
        q = filler + "Project Kestrel " + filler
        assert parse_signals(q).spans == []
        r = ExactLane(store, ExactConfig()).build(q, [])
        assert r is not None and r.boost == {} and r.inject_ids == []
        assert r.detected["truncated"] is True
        monkeypatch.setenv("REVIEN_EXACT_SCAN_CHARS", "100000")
        cfg = ExactConfig.from_env()
        assert cfg.scan_chars == 100000
        r = ExactLane(store, cfg).build(q, [])
        assert k.node_id in r.matches and "truncated" not in r.detected
        # inside the bound: read as before, and no cut reported
        r = ExactLane(store, ExactConfig()).build("who is Project Kestrel", [])
        assert "truncated" not in r.detected and "scanned_chars" not in r.detected

    def test_the_verifiers_case_question_last(self, store, dated):
        # 4.6 KB of filler, then the question: it used to be dropped silently
        d = add(store, "Dana Reyes", "d")
        q = "lorem ipsum " * 384 + "Who is Dana Reyes? ... on 9/21"
        assert len(q) > 4600
        r = ExactLane(store, ExactConfig()).build(q, [])
        assert r.detected["truncated"] is True
        dana = [x for x in r.detected["names"] if x["text"] == "Dana Reyes"]
        assert dana and dana[0]["matched"] == [d.node_id]
        assert r.detected["dates"][0]["text"] == "9/21"
        assert dated["day"].node_id in r.boost

    def test_a_cut_is_reported_even_with_no_signal(self, store):
        q = "lorem ipsum " * 600
        r = ExactLane(store, ExactConfig()).build(q, [])
        assert r is not None and r.boost == {} and r.matches == {}
        assert r.detected["truncated"] is True
        assert r.detected["query_chars"] == len(q)
        assert ExactLane(store, ExactConfig()).build("lorem ipsum", []) is None

    def test_the_cut_never_splits_a_token(self):
        from revien.retrieval.exact import SCAN_CHARS
        half = SCAN_CHARS - SCAN_CHARS // 2
        # a head that is one split token is dropped; the tail is read whole
        q = "a" * (SCAN_CHARS - 6) + " PR-123456 tail"
        sig = parse_signals(q)
        assert sig.ids == ["PR-123456"] and sig.truncated
        # the tail would start inside "QQ-123456": it starts after it instead
        q = "y " * 2000 + "QQ-123456" + " w" * (half // 2 - 4)
        sig = parse_signals(q)
        assert sig.ids == [] and sig.truncated
        assert len(" w" * (half // 2 - 4)) < half < len(" w" * (half // 2 - 4)) + 9
        # a date in the middle is not read
        q2 = "y " * 1500 + "on 2026-09-21 " + "y " * 1500
        assert parse_signals(q2).dates == []

    def test_head_and_tail_never_join(self):
        from revien.retrieval.exact import SCAN_CHARS
        h = SCAN_CHARS // 2
        # "Sept" ends the head and "21" starts the tail: not Sept 21
        head = "z " * ((h - 5) // 2) + "Sept"
        q = head + " " + "m " * 3000 + "21 " + "w " * ((SCAN_CHARS - h - 3) // 2 - 1)
        sig = parse_signals(q)
        assert sig.truncated and sig.dates == []
        # a quote opened in the head and closed in the tail is not a quote
        q = "z " * ((h - 9) // 2) + '"Project' + " m" * 3000 + ' Atlas"' + " w" * 1000
        assert parse_signals(q).quoted == []

    def test_a_cut_query_has_no_whole_query_label(self, store):
        add(store, "harbor", "h")
        q = "harbor" + " " * 5000 + "!"
        assert parse_signals(q, max_chars=10 ** 9).whole_query == "harbor"
        assert parse_signals(q).whole_query is None

    def test_past_the_bound_the_lane_adds_only_its_report(self, store):
        # The bound is the lane's own: recall still reads the whole query
        # (keyword recall finds the node). A signal in the middle of a long
        # query is not read by the lane, so results and scores are the same
        # as with it off; only diagnostics["exact"] says where it cut.
        k = add(store, "Project Kestrel", "kestrel notes")
        q = "kestrel " * 500 + "Project Kestrel " + "kestrel " * 500
        off = strip(engine_for(store, exact=False).recall(q, top_n=5, debug=True))
        on = strip(engine_for(store, exact=True).recall(q, top_n=5, debug=True))
        exact = on["diagnostics"].pop("exact")
        assert on == off and "exact" not in off["diagnostics"]
        assert exact["detected"]["truncated"] is True and exact["boost"] == {}
        assert k.node_id in [r["node_id"] for r in off["results"]]
        # without debug there is no diagnostics at all: identical
        off = strip(engine_for(store, exact=False).recall(q, top_n=5))
        on = strip(engine_for(store, exact=True).recall(q, top_n=5))
        assert on == off

    def test_a_blanked_date_never_joins_its_neighbours(self):
        # "Sept" and "5" sit either side of an ISO date: not "Sept 5"
        assert _dates("Sept 2026-09-21 5") == [(9, 21, 2026)]
        assert _dates("on Jan 2026-01-02 3") == [(1, 2, 2026)]
        # within one pattern, a date read earlier still reads as blank
        # before the next one ("on May 5 may 6": "may 6" follows "on")
        assert _dates("on May 5 may 6") == [(5, 5, None), (5, 6, None)]


class TestCodeCap:
    """Codes are looked for inside labels for at most MAX_CONTAINED_CODES per
    query, the longest first; the rest are reported, never silently empty."""

    def test_the_cap_keeps_the_most_distinctive_codes(self, store):
        from revien.retrieval.exact import MAX_CONTAINED_CODES
        assert MAX_CONTAINED_CODES == 16
        hit = add(store, "Fix for ABC-777 regression", "x")
        short = " ".join("Z-%d" % i for i in range(10, 50))    # 4 characters
        for q in (f"ABC-777 {short}", f"{short} ABC-777"):
            r = ExactLane(store, ExactConfig()).build(q, [])
            assert hit.node_id in r.boost, q                    # 7 chars: kept
            cap = r.detected["ids_capped"]
            assert cap["capped"] is True and cap["searched"] == 16
            assert cap["skipped"] == 41 - 16 == len(cap["skipped_codes"])
            assert "ABC-777" not in cap["skipped_codes"]

    def test_a_skipped_code_is_reported_as_skipped(self, store):
        hit = add(store, "Fix for ABC-77 regression", "x")
        longer = " ".join("ZZZZ-%d" % i for i in range(1000, 1016))   # 16 longer
        r = ExactLane(store, ExactConfig()).build(f"ABC-77 {longer}", [])
        assert hit.node_id not in r.boost
        entry = [x for x in r.detected["ids"] if x["text"] == "ABC-77"][0]
        assert entry["matched"] == [] and entry["contained_search"] == "skipped_by_cap"
        cap = r.detected["ids_capped"]
        assert cap == {"capped": True, "max_contained_codes": 16, "searched": 16,
                       "skipped": 1, "skipped_codes": ["ABC-77"]}
        searched = [x for x in r.detected["ids"] if x["text"] != "ABC-77"]
        assert all("contained_search" not in x for x in searched)

    def test_ties_keep_query_order(self, store):
        codes = ["QA-%02d" % i for i in range(20)]               # all 5 chars
        r = ExactLane(store, ExactConfig()).build(" ".join(codes), [])
        assert r.detected["ids_capped"]["skipped_codes"] == codes[16:]

    def test_the_cap_is_configurable(self, store, monkeypatch):
        hit = add(store, "Fix for ABC-77 regression", "x")
        longer = " ".join("ZZZZ-%d" % i for i in range(1000, 1016))
        monkeypatch.setenv("REVIEN_EXACT_MAX_CONTAINED_CODES", "64")
        cfg = ExactConfig.from_env()
        assert cfg.max_contained_codes == 64
        r = ExactLane(store, cfg).build(f"ABC-77 {longer}", [])
        assert hit.node_id in r.boost and "ids_capped" not in r.detected
        monkeypatch.setenv("REVIEN_EXACT_MAX_CONTAINED_CODES", "0")
        r = ExactLane(store, ExactConfig.from_env()).build("ABC-77", [])
        assert hit.node_id not in r.boost
        assert r.detected["ids_capped"]["skipped_codes"] == ["ABC-77"]

    def test_a_short_query_reports_no_cap(self, store):
        add(store, "Fix for ABC-77 regression", "x")
        r = ExactLane(store, ExactConfig()).build("ABC-77 and PR-1234", [])
        assert "ids_capped" not in r.detected
        assert all("contained_search" not in x for x in r.detected["ids"])


class TestDateEdges:
    """No date pattern reads across a slash, a hyphen or a dot into another
    number, at either end of the match."""

    @pytest.mark.parametrize("q,want", [
        # the verifier's three
        ("21 September 3/4", [(9, 21, None)]),
        ("the 3/4 May meeting", []),
        ("on 9/21 Mar", [(9, 21, None)]),
        # month day / day month against a slash, hyphen or dot
        ("September 3/4", []), ("May 5/6", []), ("May 5.5", []),
        ("v1.5 May release", []), ("21 September/October", []),
        ("Apr/May 5", []), ("3-May 5", []),
        # month year and the full forms
        ("Apr/May 2026", []), ("3-May 2026", []), ("September 2026/27", []),
        ("September 2026-27", []), ("21 Sept 2026/27", [(9, 21, None)]),
        ("Sept 21, 2026-2027", [(9, 21, None)]), ("5 May-2026", []),
        # slash pairs and slash dates
        ("1-3/4 cups", []), ("on 1-3/4", []), ("on 3/4.5", []),
        ("on 3/4-inch pipe", []), ("1-3/4/2026", []), ("9/21-26", []),
        ("2026-09-21-3", []), ("notes-2026-09-21", []),
    ])
    def test_no_date_across_an_edge(self, q, want):
        assert _dates(q) == want, q

    @pytest.mark.parametrize("q,want", [
        ("mid-May 2026", [(5, None, 2026)]), ("on 9/21-ish", [(9, 21, None)]),
        ("May 5.", [(5, 5, None)]), ("on May 5, 2026.", [(5, 5, 2026)]),
        ("(May 5)", [(5, 5, None)]), ("[9/21]", [(9, 21, None)]),
        ("2026-09-21T10:00", [(9, 21, 2026)]), ("on 1/2/2026.", [(1, 2, 2026)]),
        ("21 May, 2026", [(5, 21, 2026)]), ("May, 2026", [(5, None, 2026)]),
        ("from 9/21 to 9/23", [("range", (9, 21, None), (9, 23, None))]),
        ("September 21, 2026", [(9, 21, 2026)]), ("21 September 2026", [(9, 21, 2026)]),
        ("Sept 21st", [(9, 21, None)]), ("21st of September", [(9, 21, None)]),
        ("Sep. 21st 2026", [(9, 21, 2026)]), ("9/21/26", [(9, 21, 2026)]),
        ("2026/09/21", [(9, 21, 2026)]), ("12/25", [(12, 25, None)]),
        ("September 2026", [(9, None, 2026)]),
    ])
    def test_dates_with_clean_edges_still_parse(self, q, want):
        assert _dates(q) == want, q


# ── ranges: one signal for the whole span, read whole or not at all ──────────

def R(start, end):
    return [("range", start, end)]


class TestRanges:
    @pytest.mark.parametrize("q,want", [
        # what 2c40df5 / 11db52d read in part, now read whole
        ("what happened May 5-10", R((5, 5, None), (5, 10, None))),
        ("the Sept 21-23 trip", R((9, 21, None), (9, 23, None))),
        ("the 9/21-9/23 trip", R((9, 21, None), (9, 23, None))),
        ("12/24-12/26", R((12, 24, None), (12, 26, None))),
        ("9/21/2026-9/23/2026", R((9, 21, 2026), (9, 23, 2026))),
        ("21-23 September 2026", R((9, 21, 2026), (9, 23, 2026))),
        # hyphen, en dash, em dash and "to", spaced or not, all the same
        ("May 5 - 10", R((5, 5, None), (5, 10, None))),
        ("May 5–10", R((5, 5, None), (5, 10, None))),
        ("May 5 — 10", R((5, 5, None), (5, 10, None))),
        ("May 5 to 10", R((5, 5, None), (5, 10, None))),
        ("May 5 to May 10", R((5, 5, None), (5, 10, None))),
        ("9/21–9/23", R((9, 21, None), (9, 23, None))),
        ("9/21 to 9/23", R((9, 21, None), (9, 23, None))),
        ("from 9/21 to 9/23", R((9, 21, None), (9, 23, None))),
        ("2026-09-21 to 2026-09-23", R((9, 21, 2026), (9, 23, 2026))),
        ("May 5th-6th", R((5, 5, None), (5, 6, None))),
        ("21st to 23rd September", R((9, 21, None), (9, 23, None))),
        ("the 3-5 May meeting", R((5, 3, None), (5, 5, None))),
        ("4-5 May", R((5, 4, None), (5, 5, None))),
        # across a month and across a year
        ("Sept 28 - Oct 3", R((9, 28, None), (10, 3, None))),
        ("28 Sept - 3 Oct 2026", R((9, 28, 2026), (10, 3, 2026))),
        ("Dec 28 - Jan 3", R((12, 28, None), (1, 3, None))),
        ("Dec 28 - Jan 3, 2027", R((12, 28, 2026), (1, 3, 2027))),
        ("Dec 28, 2026 - Jan 3, 2027", R((12, 28, 2026), (1, 3, 2027))),
        ("Sept 21-23, 2026", R((9, 21, 2026), (9, 23, 2026))),
        ("21/09/2026-23/09/2026", R((9, 21, 2026), (9, 23, 2026))),
        ("Jan 1-31", R((1, 1, None), (1, 31, None))),
        ("Feb 28 - Mar 1", R((2, 28, None), (3, 1, None))),
        # a marked or padded ratio-shaped start
        ("on 3/4-3/8", R((3, 4, None), (3, 8, None))),
        ("from 03/04 to 03/08", R((3, 4, None), (3, 8, None))),
    ])
    def test_a_range(self, q, want):
        assert _dates(q) == want, q

    @pytest.mark.parametrize("q", [
        # reversed, empty, too long, impossible
        "May 10-5", "May 5-5", "May 1 - June 15", "Jan 1 - Feb 1",
        "Sept 21-45", "Feb 30 - Mar 2", "9/23-9/21", "21/9-23/9",
        "9/21/2026-9/23/2027",
        # the ratio rule on the start
        "1/2-3/4 cup", "from 3/4 to 3/8", "3/4-3/8", "rated it 9/10 - 10/10",
        "1/2 to 3/4 cup", "3/4-inch",
        # no range shape: no partial date either
        "9/21-26", "1-3/4 cups",
    ])
    def test_no_date_not_a_partial_one(self, q):
        assert _dates(q) == [], q

    def test_a_failed_range_leaves_no_code_behind(self):
        for q in ["Sept 21-45", "May 10-5", "the 9/23-9/21 trip", "ages 5-10",
                  "rooms 21-23", "call 555-1234", "1-3/4 cups", "9/21-26"]:
            assert parse_signals(q).ids == [], q

    def test_ranges_do_not_disturb_single_dates(self):
        assert _dates("on 9/21 and 9/23") == [(9, 21, None), (9, 23, None)]
        assert _dates("between 3/10 and 3/12") == [(3, 12, None)]
        assert _dates("May 5, 10 people") == [(5, 5, None)]
        assert _dates("Sept 21, 2026-27") == [(9, 21, None)]

    def test_the_span_is_a_store_range(self):
        from revien.retrieval.exact import parse_dates
        d = parse_dates("Dec 28 - Jan 3")[0][0]
        assert d.ranges_for_year(2026) == ("2026-12-28", "2027-01-03")
        d = parse_dates("Feb 28 - Feb 29")[0][0]
        assert d.ranges_for_year(2024) == ("2024-02-28", "2024-02-29")
        assert d.ranges_for_year(2025) is None
        assert parse_dates("Sept 21-23")[0][0].describe()["range_end"] == {
            "year": None, "month": 9, "day": 23}

    def test_a_day_inside_the_span_matches_like_a_day(self, store):
        inside = add(store, "Trip day two", "x", **day(2026, 9, 22))
        start = add(store, "Trip day one", "x", **day(2026, 9, 21))
        outside = add(store, "Day after", "x", **day(2026, 9, 24))
        mon = add(store, "September plans", "x", **month(2026, 9))
        r = ExactLane(store, ExactConfig()).build("the Sept 21-23 trip", [])
        assert r.boost[inside.node_id] == pytest.approx(BOOST_MAX)
        assert r.boost[start.node_id] == pytest.approx(BOOST_MAX)
        assert outside.node_id not in r.boost
        assert r.boost[mon.node_id] == pytest.approx(BOOST_MAX * 0.4)
        # one signal: never more than one exact day gives, however wide
        wide = ExactLane(store, ExactConfig()).build("Sept 1-30", [])
        exact = ExactLane(store, ExactConfig()).build("on 2026-09-22", [])
        assert wide.boost[inside.node_id] == exact.boost[inside.node_id]
        assert len(wide.detected["dates"]) == 1
        assert wide.matches[inside.node_id] == [
            m for m in wide.matches[inside.node_id] if m["signal"] == "date"]
        assert len(wide.matches[inside.node_id]) == 1

    def test_a_cross_year_range_matches_both_years(self, store):
        a = add(store, "New year eve", "x", **day(2026, 12, 31))
        b = add(store, "New year day", "x", **day(2027, 1, 1))
        r = ExactLane(store, ExactConfig()).build("Dec 28 - Jan 3", [])
        assert a.node_id in r.boost and b.node_id in r.boost

    @pytest.mark.parametrize("q,label", [
        ("what happened May 5-10", "Ages 5-10 swim class"),
        ("the Sept 21-23 trip", "Rooms 21-23 booking"),
        ("the 9/21-9/23 trip", "Seats 9/21-9/23"),
    ])
    def test_a_range_never_lifts_a_label_that_contains_its_numbers(self, store, q, label):
        hijack = add(store, label, "unrelated")
        trip = add(store, "Trip", "x", **day(2026, 9, 22))
        r = ExactLane(store, ExactConfig()).build(q, [])
        assert hijack.node_id not in r.boost and hijack.node_id not in r.inject_ids
        assert r.detected["ids"] == []
        if "Sept" in q or "9/2" in q:
            assert trip.node_id in r.boost


class TestQuoteAwareCut:
    """A cut never lands inside a quoted phrase: the text between two quotes
    is never read as one."""

    TAIL = '"Dentist appointment" Ridge hike "Garden planting season"'

    def _cut_at(self, shift, tail=None):
        from revien.retrieval.exact import SCAN_CHARS
        tail = tail or self.TAIL
        half = SCAN_CHARS - SCAN_CHARS // 2
        pad = half - (len(tail) - shift)
        return "lorem ipsum " * 400 + tail + " x" * (pad // 2) + " " * (pad % 2)

    @pytest.mark.parametrize("shift", [1, 3, 8, 15, 20])
    def test_a_tail_starting_inside_a_quote(self, store, shift):
        ridge = add(store, "Ridge hike", "x")
        garden = add(store, "Garden planting season", "x")
        q = self._cut_at(shift)
        sig = parse_signals(q)
        assert sig.truncated and sig.quoted == ["Garden planting season"]
        r = ExactLane(store, ExactConfig()).build(q, [])
        assert ridge.node_id not in r.boost and garden.node_id in r.boost

    def test_a_tail_starting_at_the_quote_reads_both(self):
        assert parse_signals(self._cut_at(0)).quoted == [
            "Dentist appointment", "Garden planting season"]

    @pytest.mark.parametrize("o,c", [("“", "”"), ("'", "'")])
    def test_curly_and_single_quotes_too(self, o, c):
        tail = f"{o}Dentist appointment{c} Ridge hike {o}Garden planting season{c}"
        assert parse_signals(self._cut_at(4, tail)).quoted == ["Garden planting season"]

    def test_the_head_cut_mirror(self):
        from revien.retrieval.exact import SCAN_CHARS, _scan_ends
        h = SCAN_CHARS // 2
        q = "y " * ((h - 10) // 2) + self.TAIL + " " + "m " * 3000
        head, _tail = _scan_ends(q, SCAN_CHARS)
        assert '"' not in head and len(head) < h
        assert parse_signals(q).quoted == []

    def test_a_stray_inch_mark_does_not_flip_the_quotes(self):
        # counting quote marks would be thrown by the 27" here; the lane's
        # own quote rule is not
        q = 'a 27" screen ' + "lorem ipsum " * 400 + '"Garden planting season" end'
        assert parse_signals(q).quoted == ["Garden planting season"]


class TestOrdinaryWords:
    @pytest.mark.parametrize("q", [
        "March is coming, what should I plant?", "Monday is the day",
        "OK, what next", "AI is everywhere", "Sunday plans?",
        "Notes from yesterday", "May I ask something",
    ])
    def test_sentence_initial_ordinary_words_are_not_names(self, q):
        assert parse_signals(q).spans == []

    def test_a_person_labelled_march_is_not_matched_by_the_month(self, store):
        n = add(store, "March", "A person.", node_type=NodeType.ENTITY)
        for q in ["March is coming, what should I plant?", "March", "March?"]:
            r = exact_for(store, q)
            assert r is None or n.node_id not in r.boost, q

    @pytest.mark.parametrize("label,q", [
        ("OK", "Is it OK?"), ("OK", "OK"), ("AI", "I use AI daily"),
        ("Monday", "What happens on Monday?"), ("Monday", "Monday"),
        ("March", "What did we do in March?"), ("Sat", "Is Sat free?"),
    ])
    def test_common_capitalized_labels_get_no_boost(self, store, label, q):
        n = add(store, label, "x", node_type=NodeType.TOPIC)
        r = exact_for(store, q)
        assert r is not None and n.node_id not in r.boost
        assert r.matches[n.node_id][0]["guard"] == "common_words"

    def test_the_lists_cover_months_weekdays_and_short_acronyms(self):
        from revien.retrieval.exact import COMMON_WORDS
        for w in ["january", "sept", "march", "monday", "fri", "ok", "ai", "tv",
                  "faq", "asap"]:
            assert w in COMMON_WORDS, w

    def test_a_quoted_common_word_is_an_explicit_title(self, store):
        n = add(store, "March", "A person.", node_type=NodeType.ENTITY)
        for q in ['tell me about "March"', 'who is "March"']:
            r = exact_for(store, q)
            assert r.boost[n.node_id] == pytest.approx(BOOST_MAX), q

    def test_a_quoted_common_word_is_still_under_the_df_guard(self, store):
        for i in range(4):
            add(store, "March", f"v{i}")
        r = exact_for(store, 'tell me about "March"')
        assert r.boost == {}

    def test_common_words_are_configurable_for_sentence_initial_names(
            self, store, monkeypatch):
        add(store, "Orinoco", "A river.")
        assert exact_for(store, "Orinoco is long") is not None
        monkeypatch.setenv("REVIEN_EXACT_COMMON_WORDS", "orinoco")
        lane = ExactLane(store, ExactConfig.from_env())
        assert lane.build("Orinoco is long", []) is None


class TestPossessives:
    def test_possessive_names_parse_without_the_s(self):
        for q in ["What did Priya's mom say", "What did Priya’s mom say"]:
            assert "Priya" in {x.text for x in parse_signals(q).spans}, q
        assert parse_signals("Priya's").whole_query == "Priya"
        assert parse_signals("Priya’s").whole_query == "Priya"

    def test_possessive_ends_the_name(self):
        texts = {x.text for x in parse_signals(
            "see Dana's Project Kestrel budget").spans}
        assert {"Dana", "Project Kestrel"} <= texts
        assert not any("Dana Project" in t for t in texts)

    @pytest.mark.parametrize("q", [
        "Priya's", "Priya’s", "tell me about Priya's",
        "tell me about Priya’s", "Who is Priya’s?",
    ])
    def test_possessive_matches_the_label(self, store, q):
        n = add(store, "Priya", "A person.", node_type=NodeType.ENTITY)
        r = exact_for(store, q)
        assert r is not None and n.node_id in r.boost, q

    def test_possessive_matches_in_a_topic_question(self, store):
        priya = add(store, "Priya", "A person.", node_type=NodeType.ENTITY)
        fact = add(store, "Garden bed", "x")
        link(store, priya, fact)
        r = exact_for(store, "What is Priya's favorite plant?")
        assert fact.node_id in r.boost and priya.node_id not in r.boost


# ── a named entity: about it, or asking something of it ──────────────────────

def _entity_world(store):
    dana = add(store, "Dana", "Dana is a colleague.", node_type=NodeType.ENTITY)
    gold = add(store, "Irrigation schedule",
               "Irrigation schedule drafted by Dana for the beds.")
    link(store, dana, gold)
    return dana, gold


ENTITY_SIMS = {"irrigation schedule": 0.70, "irrigation costs": 0.72,
               "dana is": 0.66}


class TestEntityVersusTopic:
    def test_a_named_person_does_not_hijack_a_question_about_something_else(
            self, store):
        dana, gold = _entity_world(store)
        q = "What did Dana write about irrigation?"
        off = engine_for(store, exact=False, sims=ENTITY_SIMS, queries=[q.lower()])
        on = engine_for(store, exact=True, sims=ENTITY_SIMS, queries=[q.lower()])
        r_off, r_on = off.recall(q, top_n=10), on.recall(q, top_n=10)
        assert ids_of(r_off)[0] == gold.node_id           # meaning puts gold first
        assert ids_of(r_on)[0] == gold.node_id            # the lane keeps it there
        assert dana.node_id not in boosts(r_on)           # the entity is not lifted
        assert boosts(r_on)[gold.node_id] > 0             # its neighbour is

    def test_the_right_neighbour_is_lifted_over_an_unconnected_lookalike(
            self, store):
        dana, gold = _entity_world(store)
        add(store, "Irrigation costs", "Irrigation costs for the season.")
        q = "What did Dana write about irrigation?"
        off = engine_for(store, exact=False, sims=ENTITY_SIMS, queries=[q.lower()])
        on = engine_for(store, exact=True, sims=ENTITY_SIMS, queries=[q.lower()])
        r_off, r_on = off.recall(q, top_n=10), on.recall(q, top_n=10)
        assert ids_of(r_off)[0] != gold.node_id
        assert ids_of(r_on)[0] == gold.node_id
        assert set(boosts(r_on)) == {gold.node_id}
        assert boosts(r_on)[gold.node_id] == pytest.approx(
            BOOST_MAX * 0.8 * 0.5)

    @pytest.mark.parametrize("q", [
        "Dana", "Dana?", "who is Dana", "Who is Dana?", "tell me about Dana",
        "What do you know about Dana", "Dana's", "Who's Dana?",
        "remind me who Dana is", "Tell me everything about Dana",
    ])
    def test_a_question_about_the_entity_boosts_the_entity(self, store, q):
        dana, gold = _entity_world(store)
        add(store, "Irrigation costs", "Irrigation costs for the season.")
        off = engine_for(store, exact=False, sims=ENTITY_SIMS, queries=[q.lower()])
        on = engine_for(store, exact=True, sims=ENTITY_SIMS, queries=[q.lower()])
        assert ids_of(off.recall(q, top_n=10))[0] != dana.node_id
        r_on = on.recall(q, top_n=10)
        assert ids_of(r_on)[0] == dana.node_id, q
        assert boosts(r_on) == {dana.node_id: pytest.approx(BOOST_MAX * 0.8)}

    @pytest.mark.parametrize("q", [
        "What did Dana write about irrigation?", "Dana's irrigation plan",
        "What is Dana's favorite tool", "irrigation notes from Dana",
        "What did Dana do on 9/21", 'Dana and the "Irrigation costs" file',
        "What did Dana say in PR-1234",
    ])
    def test_a_question_asking_something_of_the_entity_does_not(self, store, q):
        dana, gold = _entity_world(store)
        r = exact_for(store, q)
        assert dana.node_id not in r.boost
        assert r.matches[dana.node_id][0]["mode"] == "topic"

    def test_mode_is_reported(self, store):
        _entity_world(store)
        about = exact_for(store, "who is Dana")
        topic = exact_for(store, "What did Dana write about irrigation?")
        assert about.detected["names"][0]["mode"] == "entity"
        assert topic.detected["names"][0]["mode"] == "topic"
        assert topic.detected["names"][0]["neighbors"] == 1

    def test_framing_words_are_configurable(self, store, monkeypatch):
        dana, _ = _entity_world(store)
        assert dana.node_id not in exact_for(store, "Dana please elaborate").boost
        monkeypatch.setenv("REVIEN_EXACT_FRAMING_WORDS", "elaborate")
        lane = ExactLane(store, ExactConfig.from_env())
        assert dana.node_id in lane.build("Dana please elaborate", []).boost

    def test_quoted_titles_and_codes_still_boost_their_own_node(self, store):
        doc = add(store, "Project Kestrel", "Notes.")
        pr = add(store, "PR-1234", "A change.")
        r = exact_for(store, 'what is the status of "Project Kestrel" in PR-1234')
        assert r.boost[doc.node_id] == pytest.approx(BOOST_MAX)
        assert r.boost[pr.node_id] == pytest.approx(BOOST_MAX)


class TestNeighbors:
    Q = "What did Dana write about irrigation?"

    def test_one_hop_either_direction_live_and_in_scope_only(self, store):
        dana = add(store, "Dana", "x", node_type=NodeType.ENTITY)
        out_ = add(store, "Out", "x")
        in_ = add(store, "In", "x")
        alias = add(store, "Dana R.", "x")
        dead_edge = add(store, "DeadEdge", "x")
        dead_node = add(store, "DeadNode", "x")
        ctx = add(store, "Ctx", "x", node_type=NodeType.CONTEXT)
        two_hops = add(store, "TwoHops", "x")
        link(store, dana, out_)
        link(store, in_, dana)
        link(store, dana, alias, EdgeType.ALIAS_OF)
        e = link(store, dana, dead_edge)
        kill(store, "edges", "edge_id", e.edge_id)
        link(store, dana, dead_node)
        kill(store, "nodes", "node_id", dead_node.node_id)
        link(store, dana, ctx)
        link(store, out_, two_hops)
        lane = ExactLane(store, ExactConfig.from_env())
        r = lane.build(self.Q, [])
        assert set(r.boost) == {out_.node_id, in_.node_id}
        for v in r.boost.values():
            assert v == pytest.approx(BOOST_MAX * 0.8 * 0.5)
        assert r.inject_ids == []                  # lifted, never admitted
        r2 = lane.build(self.Q, [], include_context=True)
        assert ctx.node_id in r2.boost

    def test_origin_filter_scopes_the_neighbours(self, store):
        dana = add(store, "Dana", "x", node_type=NodeType.ENTITY,
                   origin_runtime="codex")
        mine = add(store, "Mine", "x", origin_runtime="codex")
        theirs = add(store, "Theirs", "x", origin_runtime="claude-code")
        link(store, dana, mine)
        link(store, dana, theirs)
        r = ExactLane(store, ExactConfig.from_env()).build(
            self.Q, [], origin_runtime=["codex"])
        assert set(r.boost) == {mine.node_id}

    def test_a_hub_is_scaled_down_then_ignored(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_EXACT_NEIGHBOR_MAX", "2")
        dana = add(store, "Dana", "x", node_type=NodeType.ENTITY)
        ns = [add(store, f"N{i}", "x") for i in range(3)]
        for n in ns:
            link(store, dana, n)
        lane = ExactLane(store, ExactConfig.from_env())
        r = lane.build(self.Q, [])
        assert all(r.boost[n.node_id] == pytest.approx(BOOST_MAX * 0.8 * 0.5 * 2 / 3)
                   for n in ns)
        for i in range(3, 20):                      # 20 neighbours > 8 * 2
            link(store, dana, add(store, f"N{i}", "x"))
        r = lane.build(self.Q, [])
        assert r.boost == {}
        assert r.detected["names"][0]["neighbor_guard"] == "hub_20_neighbors"

    def test_the_df_guard_applies_to_the_entity_label(self, store):
        pair = [add(store, "Dana", f"v{i}", node_type=NodeType.ENTITY)
                for i in range(2)]
        ns = [add(store, f"N{i}", "x") for i in range(2)]
        for p, n in zip(pair, ns):
            link(store, p, n)
        r = exact_for(store, self.Q)
        assert all(r.boost[n.node_id] == pytest.approx(BOOST_MAX * 0.8 * 0.5 * 0.5)
                   for n in ns)
        for i in range(2, 4):                       # four share the label: over 3
            p = add(store, "Dana", f"v{i}", node_type=NodeType.ENTITY)
            link(store, p, add(store, f"M{i}", "x"))
        assert exact_for(store, self.Q).boost == {}

    def test_neighbours_stack_but_stay_under_the_cap(self, store):
        a = add(store, "Dana", "x", node_type=NodeType.ENTITY)
        b = add(store, "Theo", "x", node_type=NodeType.ENTITY)
        both = add(store, "Shared", "x")
        link(store, a, both)
        link(store, b, both)
        r = exact_for(store, "What did Dana and Theo write about irrigation?")
        assert r.boost[both.node_id] == pytest.approx(
            2 * BOOST_MAX * 0.8 * 0.5)
        assert r.boost[both.node_id] <= BOOST_MAX

    def test_no_node_content_is_searched(self, store):
        dana = add(store, "Dana", "x", node_type=NodeType.ENTITY)
        mention = add(store, "Mentions", "Dana wrote about irrigation here.")
        r = exact_for(store, self.Q)
        assert mention.node_id not in r.boost and dana.node_id not in r.boost


# ── with the reranker on ─────────────────────────────────────────────────────

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
         "oscar"]


def _rerank_world(store):
    """15 nodes whose meaning orders them alpha > bravo > ... by 0.012 a step;
    the node labelled ``target`` is one of them. The query names it."""
    nodes = {}
    sims = {}
    for i, w in enumerate(WORDS):
        label = w.title()
        nodes[w] = add(store, label, f"{w} entry")
        sims[f"{w} entry"] = 0.80 - 0.012 * i
    return nodes, sims


def _permuted(target, position):
    """The reranker's order: base order with ``target`` moved to ``position``."""
    rest = [w for w in WORDS if w != target]
    return rest[:position] + [target] + rest[position:]


def _reranker(order):
    def scorer(query, texts):
        return [float(-order.index(t.split()[0])) for t in texts]
    return CrossEncoderReranker(scorer=scorer, top_k=20)


class TestWithTheRerankerOn:
    Q = "tell me about Hotel"          # about the entity labelled Hotel

    def _both(self, store, position):
        nodes, sims = _rerank_world(store)
        rr = _reranker(_permuted("hotel", position))

        def run(exact):
            eng = engine_for(store, exact=exact, sims=sims,
                             queries=[self.Q.lower()], reranker=rr)
            resp = eng.recall(self.Q, top_n=15, debug=True)
            assert all("rerank_score" in r.score_breakdown for r in resp.results)
            return resp
        return nodes, run(False), run(True)

    def test_the_lane_changes_the_order_inside_the_reranked_head(self, store):
        nodes, off, on = self._both(store, 3)
        hotel = nodes["hotel"].node_id
        assert ids_of(off).index(hotel) == 3            # the reranker's order
        assert ids_of(on).index(hotel) == 0             # 3 slots (0.036) < the boost
        assert boosts(on) == {hotel: pytest.approx(BOOST_MAX * 0.8)}
        assert on.diagnostics["exact"]["after_rerank"] == {hotel: [3, 0]}

    def test_the_cap_holds_after_reranking(self, store):
        nodes, off, on = self._both(store, 12)
        hotel = nodes["hotel"].node_id
        assert ids_of(off).index(hotel) == 12
        base = sorted((r.score for r in off.results), reverse=True)
        assert base[0] - base[12] > BOOST_MAX * 0.8     # the premise: out of reach
        order = ids_of(on)
        assert 0 < order.index(hotel) < 12              # it moved up, bounded
        # the reranker's own #1 is still #1: a node more than the cap behind
        # the leader cannot reach the top however exactly it matches
        assert order[0] == nodes[_permuted("hotel", 12)[0]].node_id
        # nobody else changed places relative to each other
        want = [nodes[w].node_id for w in _permuted("hotel", 12) if w != "hotel"]
        assert [i for i in order if i != hotel] == want
        # and it rose exactly as far as the boost allows: below every slot
        # more than the boost above its own
        above = sum(1 for s in base if s > base[12] + BOOST_MAX * 0.8)
        assert order.index(hotel) == above

    def test_a_query_with_no_match_leaves_the_reranked_order_alone(self, store):
        nodes, sims = _rerank_world(store)
        q = "tell me about Zorblax"
        rr = _reranker(_permuted("hotel", 5))
        a = engine_for(store, exact=False, sims=sims, queries=[q.lower()],
                       reranker=rr).recall(q, top_n=15, debug=True)
        b = engine_for(store, exact=True, sims=sims, queries=[q.lower()],
                       reranker=rr).recall(q, top_n=15, debug=True)
        b.diagnostics.pop("exact", None)
        assert strip(a) == strip(b)


class _R:
    """The three fields of a result lift_after_rerank reads."""

    def __init__(self, node_id, score, reranked=True):
        self.node_id, self.score = node_id, score
        self.score_breakdown = {"rerank_score": 1.0} if reranked else {}


class TestLiftAfterRerank:
    def test_unboosted_head_is_returned_as_is(self):
        rs = [_R("a", 0.5), _R("b", 0.4)]
        out, moves = lift_after_rerank(rs, 2, {})
        assert out is rs and moves == {}

    def test_a_head_the_reranker_did_not_touch_is_returned_as_is(self):
        rs = [_R("a", 0.5, False), _R("b", 0.45, False)]
        out, moves = lift_after_rerank(rs, 2, {"b": 0.05})
        assert out is rs and moves == {}

    def test_a_boost_smaller_than_the_gap_does_not_move_a_node(self):
        # reranked order a, b, c, d; d's score includes its boost
        rs = [_R("a", 0.50), _R("b", 0.46), _R("c", 0.43), _R("d", 0.30 + 0.10)]
        out, moves = lift_after_rerank(rs, 4, {"d": 0.10})
        assert [r.node_id for r in out] == ["a", "b", "c", "d"] and moves == {}

    def test_a_boost_larger_than_the_gap_passes_exactly_that_node(self):
        rs = [_R("a", 0.50), _R("b", 0.46), _R("c", 0.40), _R("d", 0.38 + 0.05)]
        out, moves = lift_after_rerank(rs, 4, {"d": 0.05})
        assert [r.node_id for r in out] == ["a", "b", "d", "c"]
        assert moves == {"d": (3, 2)}

    def test_the_cap_is_the_largest_lift(self):
        slots = [0.60, 0.58, 0.56, 0.54, 0.52, 0.50]
        rs = [_R(f"n{i}", s) for i, s in enumerate(slots)]
        rs[5] = _R("n5", slots[5] + 0.05)
        out, moves = lift_after_rerank(rs, 6, {"n5": 0.05})
        # n5 is dealt 0.50, +0.05 = 0.55 : above 0.54 and 0.52, below 0.56
        assert [r.node_id for r in out] == ["n0", "n1", "n2", "n5", "n3", "n4"]
        assert moves == {"n5": (5, 3)}

    def test_unboosted_nodes_keep_the_reranker_order_among_themselves(self):
        # a reranker order that disagrees with the scores everywhere
        scores = [0.30, 0.50, 0.40, 0.60]
        rs = [_R(f"n{i}", s) for i, s in enumerate(scores)]
        out, _ = lift_after_rerank(rs, 4, {"n3": 0.0001})
        assert [r.node_id for r in out if r.node_id != "n3"] == ["n0", "n1", "n2"]

    def test_score_fields_and_tail_are_untouched(self):
        rs = [_R("a", 0.5), _R("b", 0.45), _R("t", 0.1)]
        out, _ = lift_after_rerank(rs, 2, {"b": 0.1})
        assert out[-1] is rs[2]
        assert sorted((r.node_id, r.score) for r in out[:2]) == [("a", 0.5), ("b", 0.45)]


# ── with BM25 on ─────────────────────────────────────────────────────────────

class TestBm25LabelScale:
    def _engines(self, store, q, monkeypatch, bm25):
        if bm25:
            monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
            monkeypatch.setenv("REVIEN_HYBRID", "rrf")
        return engine_for(store, exact=True, sims=ENTITY_SIMS, queries=[q.lower()])

    def _world(self, store):
        dana, gold = _entity_world(store)
        add(store, "Irrigation costs", "Irrigation costs for the season.")
        dated = add(store, "Dentist", "x", **day(2026, 9, 21))
        return dana, gold, dated

    def test_label_boosts_are_halved_when_bm25_is_on(self, store, monkeypatch):
        dana, gold, dated = self._world(store)
        q = "who is Dana"
        plain = boosts(self._engines(store, q, monkeypatch, False).recall(q))
        assert plain == {dana.node_id: pytest.approx(BOOST_MAX * 0.8)}
        scaled = boosts(self._engines(store, q, monkeypatch, True).recall(q))
        assert scaled == {dana.node_id: pytest.approx(BOOST_MAX * 0.8 * 0.5)}

    def test_quotes_and_ids_are_scaled_too(self, store, monkeypatch):
        doc = add(store, "Project Kestrel", "Notes.")
        pr = add(store, "PR-1234", "A change.")
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = engine_for(store, exact=True)
        b = boosts(eng.recall('open "Project Kestrel" or PR-1234'))
        assert b[doc.node_id] == pytest.approx(BOOST_MAX * 0.5)
        assert b[pr.node_id] == pytest.approx(BOOST_MAX * 0.5)

    def test_dates_neighbours_and_typed_node_ids_are_not_scaled(
            self, store, monkeypatch):
        dana, gold, dated = self._world(store)
        monkeypatch.setenv("REVIEN_LEXICAL", "bm25")
        eng = engine_for(store, exact=True)
        assert boosts(eng.recall("what happened on 2026-09-21"))[
            dated.node_id] == pytest.approx(BOOST_MAX)
        topic = boosts(eng.recall("What did Dana write about irrigation?"))
        assert topic[gold.node_id] == pytest.approx(BOOST_MAX * 0.8 * 0.5)
        assert boosts(eng.recall(f"open {gold.node_id}"))[
            gold.node_id] == pytest.approx(BOOST_MAX)

    def test_scale_is_configurable(self, store, monkeypatch):
        dana, gold, dated = self._world(store)
        monkeypatch.setenv("REVIEN_EXACT_BM25_LABEL_SCALE", "0.25")
        q = "who is Dana"
        got = boosts(self._engines(store, q, monkeypatch, True).recall(q))
        assert got[dana.node_id] == pytest.approx(BOOST_MAX * 0.8 * 0.25)

    def test_the_lane_still_runs_with_bm25_and_rrf(self, store, monkeypatch):
        dana, gold, dated = self._world(store)
        q = "who is Dana"
        resp = self._engines(store, q, monkeypatch, True).recall(q)
        assert dana.node_id in boosts(resp)
        assert "bm25_score" in resp.results[0].score_breakdown


# ── guards and weights the first build did not pin down ──────────────────────

def span(y1, m1, d1, y2, m2, d2, gran):
    return dict(event_time_start=datetime(y1, m1, d1, tzinfo=UTC),
                event_time_end=datetime(y2, m2, d2, 23, 59, 59, tzinfo=UTC),
                event_time_granularity=gran, event_time_confidence=0.9)


class TestWeights:
    def test_granularity_weights_are_the_documented_ones(self, store):
        day_n = add(store, "Day", "x", **day(2026, 9, 21))
        week_n = add(store, "Week", "x", **span(2026, 9, 20, 2026, 9, 26, G.WEEK))
        month_n = add(store, "Month", "x", **month(2026, 9))
        year_n = add(store, "Year", "x", **span(2026, 1, 1, 2026, 12, 31, G.YEAR))
        fuzzy_n = add(store, "Fuzzy", "x", **span(2026, 9, 1, 2026, 10, 31, G.FUZZY))
        r = exact_for(store, "what happened on September 21, 2026")
        assert r.boost[day_n.node_id] == pytest.approx(BOOST_MAX * 1.0)
        assert r.boost[week_n.node_id] == pytest.approx(BOOST_MAX * 0.6)
        assert r.boost[month_n.node_id] == pytest.approx(BOOST_MAX * 0.4)
        assert r.boost[year_n.node_id] == pytest.approx(BOOST_MAX * 0.15)
        assert r.boost[fuzzy_n.node_id] == pytest.approx(BOOST_MAX * 0.1)
        b = r.boost
        assert (b[day_n.node_id] > b[week_n.node_id] > b[month_n.node_id]
                > b[year_n.node_id] > b[fuzzy_n.node_id])

    def test_label_share_limit_is_exactly_max_df(self, store):
        # default MAX_DF is 3: three nodes share the title and split the boost,
        # a fourth switches it off
        trio = [add(store, "Zephyr", f"v{i}").node_id for i in range(3)]
        r = exact_for(store, "about Zephyr please")
        for nid in trio:
            assert r.boost[nid] == pytest.approx(BOOST_MAX * 0.8 / 3)
        add(store, "Zephyr", "v3")
        assert exact_for(store, "about Zephyr please").boost == {}

    def test_label_share_limit_follows_the_setting(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_EXACT_MAX_DF", "1")
        a = add(store, "Zephyr", "v0")
        lane = ExactLane(store, ExactConfig.from_env())
        assert a.node_id in lane.build("about Zephyr please", []).boost
        add(store, "Zephyr", "v1")
        assert lane.build("about Zephyr please", []).boost == {}


# ── byte-identical when off / when there is no exact signal ──────────────────

NO_SIGNAL_QUERIES = [
    "how do I usually cope when everything feels like too much",
    "what did we decide about the plan for the new place",
    "When did I last talk about money worries",
    "What is the thing I avoid eating",
    "who do I trust with hard news",
    "remind me what makes me laugh",
    "anything about the garden and the bees",
    "tell me about my day",
    "and the plan",
    "",
]


def varied_graph(store):
    nodes = []
    ents = [("Dana Reyes", NodeType.ENTITY), ("Theo Brandt", NodeType.ENTITY),
            ("When", NodeType.TOPIC), ("What", NodeType.TOPIC),
            ("Tell", NodeType.TOPIC), ("notes", NodeType.TOPIC),
            ("plan", NodeType.TOPIC), ("money", NodeType.TOPIC),
            ("garden", NodeType.TOPIC)]
    for label, t in ents:
        nodes.append(add(store, label, f"{label} is a thing here.", node_type=t))
    nodes.append(add(store, "Coping", "overwhelm strategies for hard weeks",
                     node_type=NodeType.PREFERENCE))
    nodes.append(add(store, "Bees", "the bees live in the garden",
                     node_type=NodeType.FACT, **day(2026, 9, 21)))
    nodes.append(add(store, "Money worries", "worries about money and rent",
                     node_type=NodeType.FACT, **month(2026, 9)))
    nodes.append(add(store, "Old plan", "the plan for the new place",
                     node_type=NodeType.DECISION))
    nodes.append(add(store, "ctx", "verbatim turn about the garden and bees",
                     node_type=NodeType.CONTEXT))
    dead = add(store, "Retired idea", "plan for the new place",
               node_type=NodeType.FACT)
    store.invalidate_node(dead.node_id) if hasattr(store, "invalidate_node") else None
    for a, b in [(0, 1), (1, 9), (9, 10), (10, 11), (6, 12), (7, 13)]:
        store.add_edge(Edge(edge_type=EdgeType.RELATED_TO,
                            source_node_id=nodes[a].node_id,
                            target_node_id=nodes[b].node_id, weight=0.6))
    return nodes


SIMS = {"overwhelm": 0.72, "the bees": 0.55, "worries about money": 0.61,
        "the plan for the new": 0.5, "dana": 0.4, "theo": 0.38,
        "verbatim turn": 0.45, "garden": 0.42}


def _scramble(query, texts):
    """A deterministic stand-in for the cross-encoder that disagrees with the
    engine's own order almost everywhere."""
    return [float((sum(map(ord, t)) * 7919) % 1009) / 100.0 - i * 0.001
            for i, t in enumerate(texts)]


@pytest.fixture(params=["reranker_off", "reranker_on"])
def rr(request):
    if request.param == "reranker_off":
        return None
    return CrossEncoderReranker(scorer=_scramble, top_k=20)


class TestByteIdentical:
    def _both(self, store, q, rr, **kw):
        off = engine_for(store, exact=False, sims=SIMS, queries=NO_SIGNAL_QUERIES,
                         reranker=rr)
        on = engine_for(store, exact=True, sims=SIMS, queries=NO_SIGNAL_QUERIES,
                        reranker=rr)
        return off.recall(q, **kw), on.recall(q, **kw)

    def test_the_reranker_arm_really_reorders(self, store, rr):
        varied_graph(store)
        a, _ = self._both(store, "who do I trust with hard news", rr,
                          top_n=15, debug=True)
        has = any("rerank_score" in r.score_breakdown for r in a.results)
        assert has == (rr is not None)

    @pytest.mark.parametrize("q", NO_SIGNAL_QUERIES)
    def test_no_signal_queries_are_byte_identical(self, store, q, rr):
        varied_graph(store)
        for kw in ({}, {"top_n": 15, "debug": True},
                   {"top_n": 15, "debug": True, "include_context": True,
                    "include_invalidated": True}):
            a, b = self._both(store, q, rr, **kw)
            assert strip(a) == strip(b), (q, kw)
            assert repr(strip(a)) == repr(strip(b))
            if kw.get("debug"):
                assert "exact" not in b.diagnostics

    def test_no_signal_identical_without_semantic_layer(self, store, rr):
        varied_graph(store)
        for q in NO_SIGNAL_QUERIES:
            a = engine_for(store, exact=False, reranker=rr).recall(
                q, top_n=15, debug=True)
            b = engine_for(store, exact=True, reranker=rr).recall(
                q, top_n=15, debug=True)
            assert strip(a) == strip(b), q

    def test_generic_whole_query_title_is_reported_but_changes_no_result(
            self, store, rr):
        varied_graph(store)           # has a node labelled "notes"
        a, b = self._both(store, "notes", rr, top_n=15, debug=True)
        ex = b.diagnostics.pop("exact")
        assert ex["boost"] == {} and ex["injected"] == []
        assert ex["detected"]["whole_query"]["guard"] == "common_words"
        assert strip(a) == strip(b)

    def test_detected_but_unmatched_changes_no_result(self, store, rr):
        varied_graph(store)
        for q in ["what does Zorblax think", "did PR-9999 land",
                  "anything on 2031-01-05", 'about "No Such Title"',
                  "March is coming, what should I plant?",
                  "it was 8/10 and open 24/7", "release 2024.1.5"]:
            a, b = self._both(store, q, rr, top_n=15, debug=True)
            b.diagnostics.pop("exact", None)
            a.diagnostics.pop("exact", None)
            assert strip(a) == strip(b), q

    def test_a_named_entity_with_no_neighbours_changes_no_result(self, store, rr):
        varied_graph(store)
        # "Theo Brandt" has one neighbour; ask about someone with none
        add(store, "Quince", "x", node_type=NodeType.ENTITY)
        a, b = self._both(store, "What did Quince say about the garden?", rr,
                          top_n=15, debug=True)
        ex = b.diagnostics.pop("exact")
        assert ex["boost"] == {} and ex["injected"] == []
        a.diagnostics.pop("exact", None)
        assert strip(a) == strip(b)

    def test_switch_off_ignores_signals_entirely(self, store, rr):
        varied_graph(store)
        q = 'what does Dana Reyes say, "Bees", on 2026-09-21'
        plain = RetrievalEngine(store, semantic=None, exact=False, reranker=rr)
        resp = plain.recall(q, top_n=15, debug=True)
        assert "exact" not in resp.diagnostics
        assert not boosts(resp)

    def test_env_switch(self, store, monkeypatch):
        varied_graph(store)
        q = "what does Dana Reyes say"
        monkeypatch.delenv("REVIEN_EXACT", raising=False)
        assert RetrievalEngine(store).exact_enabled is False
        monkeypatch.setenv("REVIEN_EXACT", "on")
        eng = RetrievalEngine(store)
        assert eng.exact_enabled is True
        assert boosts(eng.recall(q, top_n=15))
        monkeypatch.setenv("REVIEN_EXACT", "off")
        assert RetrievalEngine(store).exact_enabled is False
        monkeypatch.setenv("REVIEN_EXACT", "0")
        assert RetrievalEngine(store).exact_enabled is False

    def test_a_failing_lane_degrades_loudly_to_switch_off(self, store, capsys, rr):
        varied_graph(store)
        q = "what does Dana Reyes say"
        eng = engine_for(store, exact=True, sims=SIMS, queries=[q.lower()],
                         reranker=rr)

        def boom(*a, **k):
            raise RuntimeError("injected")
        eng.exact_lane.build = boom
        got = eng.recall(q, top_n=15, debug=True)
        want = engine_for(store, exact=False, sims=SIMS, queries=[q.lower()],
                          reranker=rr).recall(q, top_n=15, debug=True)
        assert strip(got) == strip(want)
        assert "[revien.exact] lane failed" in capsys.readouterr().err

    def test_signal_with_switch_on_differs(self, store, rr):
        varied_graph(store)
        a, b = self._both(store, "what does Dana Reyes say", rr, top_n=15)
        assert boosts(b) and not boosts(a)


# ── migration 007 ─────────────────────────────────────────────────────────────

FIX = Path(__file__).parent / "fixtures"
FIXTURES = ["db_main_03217f4.sql", "db_main_b905109.sql", "db_v030_7757085.sql",
            "db_v02x_schema.sql"]
IDX = {"idx_nodes_label_lower", "idx_nodes_event_time", "idx_nodes_recorded_at"}


def _indexes(path):
    c = sqlite3.connect(path)
    try:
        return {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}, \
            c.execute("PRAGMA user_version").fetchone()[0]
    finally:
        c.close()


def _rows(path):
    c = sqlite3.connect(path)
    try:
        return c.execute(
            "SELECT node_id, label, content FROM nodes "
            "ORDER BY node_id").fetchall()
    finally:
        c.close()


class TestMigration007:
    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_upgrades_older_databases_and_is_idempotent(self, fixture, tmp_path):
        path = str(tmp_path / "old.db")
        c = sqlite3.connect(path)
        c.executescript((FIX / fixture).read_text())
        c.commit()
        c.close()
        before = _rows(path) if "nodes" in _tables(path) else None
        assert not (IDX & _indexes(path)[0])
        GraphStore(db_path=path).close()
        idx, uv = _indexes(path)
        assert IDX <= idx and uv >= 7
        if before is not None:
            assert _rows(path) == before
        snap = _catalogue(path)
        GraphStore(db_path=path).close()
        assert _catalogue(path) == snap

    def test_v6_database_and_dropped_index_are_repaired(self, store):
        path = store.db_path
        store.close()
        c = sqlite3.connect(path)
        for name in IDX:
            c.execute(f"DROP INDEX {name}")
        c.execute("PRAGMA user_version = 6")
        c.commit()
        c.close()
        GraphStore(db_path=path).close()
        idx, uv = _indexes(path)
        assert IDX <= idx and uv == 7
        # a marker never overrides the schema: drop one index at user_version 7
        c = sqlite3.connect(path)
        c.execute("DROP INDEX idx_nodes_label_lower")
        c.commit()
        c.close()
        GraphStore(db_path=path).close()
        assert IDX <= _indexes(path)[0]

    def test_standalone_script(self, tmp_path):
        import importlib
        mod = importlib.import_module("revien.graph.migrations.007_exact_lane_indexes")
        path = str(tmp_path / "x.db")
        GraphStore(db_path=path).close()
        c = sqlite3.connect(path)
        c.execute("DROP INDEX idx_nodes_event_time")
        c.commit()
        c.close()
        assert mod.migrate(path)["created"] == ["idx_nodes_event_time"]
        assert mod.migrate(path)["created"] == []

    def test_queries_use_the_indexes(self, store):
        add(store, "Dana", "x", **day(2026, 9, 21))
        conn = store._get_conn()
        plan = " ".join(str(r) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT node_id FROM nodes "
            "WHERE lower(label) IN ('dana')").fetchall())
        assert "idx_nodes_label_lower" in plan
        plan = " ".join(str(r) for r in conn.execute(
            "EXPLAIN QUERY PLAN SELECT node_id FROM nodes WHERE "
            "event_time_start IS NOT NULL AND (event_time_start < '2026-09-22' "
            "AND COALESCE(event_time_end, event_time_start) >= '2026-09-21')"
        ).fetchall())
        assert "idx_nodes_event_time" in plan


def _tables(path):
    c = sqlite3.connect(path)
    try:
        return {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        c.close()


def _catalogue(path):
    c = sqlite3.connect(path)
    try:
        return (sorted(c.execute(
            "SELECT type, name FROM sqlite_master ORDER BY 1, 2").fetchall()),
            c.execute("PRAGMA user_version").fetchone()[0])
    finally:
        c.close()

"""revien_bench.ab_exact: copies the database, runs off/on under one set of
settings, scores per category, and reports the gate."""
import hashlib
import json
import os

import pytest

from revien.graph.schema import Node, NodeType, SourceType, TemporalGranularity
from revien.graph.store import GraphStore
from revien_bench import ab_exact
from tests.test_exact_lane import _Embedder, day
from tests.test_semantic import _InMemoryVectorIndex

SIMS = {"garden plans": 0.70, "garden tools": 0.66, "dana keeps": 0.62,
        "a calm evening": 0.60, "dentist": 0.64}
Q_NAME = "who is Dana Reyes"
Q_PARA = "how do I like to spend a quiet night"
Q_DATE = "what happened on 9/21/2026"


def _add(store, label, content, **kw):
    return store.add_node(Node(node_type=NodeType.FACT, label=label,
                               content=content, source_type=SourceType.EXTRACTED,
                               confidence=1.0, **kw))


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    # tests/conftest.py does not cover revien_bench/tests: keep the cross
    # encoder out (offline, deterministic) and the home throwaway.
    monkeypatch.setenv("REVIEN_RERANK", "0")
    monkeypatch.setenv("REVIEN_HOME", str(tmp_path / "home"))


@pytest.fixture
def world(tmp_path):
    db = str(tmp_path / "src.db")
    s = GraphStore(db_path=db)
    ids = {
        "plans": _add(s, "Garden plans", "garden plans for spring").node_id,
        "tools": _add(s, "Garden tools", "garden tools list").node_id,
        "dana": _add(s, "Dana Reyes", "dana keeps bees").node_id,
        "calm": _add(s, "Evenings", "a calm evening with tea").node_id,
        "dentist": _add(s, "Dentist visit", "dentist cleaning", **day(2026, 9, 21)).node_id,
    }
    s.close()
    # Make the source a v6 database: opening it with GraphStore WOULD change
    # its bytes (migration 007 adds indexes), so only a copy can stay identical.
    import sqlite3
    c = sqlite3.connect(db)
    c.execute("DROP INDEX idx_nodes_label_lower")
    c.execute("DROP INDEX idx_nodes_event_time")
    c.execute("PRAGMA user_version = 6")
    c.commit()
    c.close()
    queries = [
        {"text": Q_NAME, "gold_ids": [ids["dana"]], "category": "exact name"},
        {"text": Q_PARA, "gold_ids": [ids["calm"]], "category": "paraphrase"},
        {"text": Q_DATE, "gold_ids": [ids["dentist"]], "category": "date mixed"},
    ]
    qpath = str(tmp_path / "q.json")
    json.dump(queries, open(qpath, "w"))
    return db, qpath, ids, tmp_path


def factory(store):
    emb = _Embedder(SIMS, queries=[Q_NAME.lower(), Q_PARA.lower(), Q_DATE.lower()])
    idx = _InMemoryVectorIndex(store, emb)
    idx.reindex_all()
    return idx


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_group_of():
    assert ab_exact.group_of("paraphrase") == "paraphrase"
    assert ab_exact.group_of("Date (mixed)") == "date"
    assert ab_exact.group_of("exact") == "exact"
    assert ab_exact.group_of("baseline") == "exact"
    assert ab_exact.group_of("misc") == "other"


def test_runs_on_a_copy_and_scores_each_group(world, monkeypatch):
    db, qpath, ids, _ = world
    monkeypatch.delenv("REVIEN_EXACT", raising=False)
    before = sha(db)
    rep = ab_exact.run_ab(db, ab_exact.load_queries(qpath),
                          semantic_factory=factory)
    assert sha(db) == before and rep["settings"]["source_untouched"]
    q = {r["text"]: r for r in rep["queries"]}
    # the name query is lifted by the lane; the paraphrase is identical
    assert q[Q_NAME]["off"]["rank"] > 1 and q[Q_NAME]["on"]["rank"] == 1
    # MRR is 1/rank of the first gold, averaged: pin the arithmetic end to end
    assert rep["by_group"]["exact"]["on"]["mrr"] == 1.0
    assert rep["by_group"]["exact"]["off"]["mrr"] == round(
        1.0 / q[Q_NAME]["off"]["rank"], 4)
    assert q[Q_PARA]["off"] == q[Q_PARA]["on"]
    assert (q[Q_DATE]["off"]["rank"] or 99) > 1 and q[Q_DATE]["on"]["rank"] == 1
    assert q[Q_PARA]["boosts_on"] == {} and q[Q_NAME]["boosts_on"]
    assert set(rep["by_group"]) == {"exact", "paraphrase", "date"}
    para = rep["by_group"]["paraphrase"]
    assert para["on"] == para["off"] and para["on"]["n"] == 1
    assert rep["by_group"]["exact"]["on"]["hit_at_1"] == 1
    assert rep["by_group"]["exact"]["off"]["hit_at_1"] == 0
    assert rep["settings"]["semantic_active"] is True
    assert rep["gate"]["paraphrase_on_ties_or_beats_off"] is True
    assert rep["gate"]["verdict"].startswith("NOT EVALUATED")
    assert q[Q_DATE]["lane_can_see_gold_date"] is True
    assert q[Q_DATE]["date_visibility"][0]["event_time_day_level"] is True
    assert "date_visibility" not in q[Q_NAME]
    assert "exact_detected" in q[Q_NAME] and "exact_detected" not in q[Q_PARA]
    assert not any(r.get("unstable_off") or r.get("unstable_on") for r in rep["queries"])


def test_gate_uses_the_fork_numbers(world, monkeypatch):
    db, qpath, ids, _ = world
    monkeypatch.delenv("REVIEN_EXACT", raising=False)
    qs = ab_exact.load_queries(qpath)
    ok = {Q_NAME: [ids["dana"]], Q_DATE: [ids["dentist"]]}
    rep = ab_exact.run_ab(db, qs, semantic_factory=factory, fork=ok)
    assert rep["gate"]["exact_and_date_on_ties_or_beats_fork"] is True
    assert rep["gate"]["verdict"] == "ELIGIBLE"
    # a fork that is better on one question than the lane is: stays off
    rep2 = ab_exact.run_ab(db, qs, semantic_factory=factory, fork={
        Q_NAME: [ids["dana"]], Q_DATE: [ids["dentist"]]}, top_n=10)
    assert rep2["gate"]["verdict"] == "ELIGIBLE"
    qs_hard = [dict(q) for q in qs]
    qs_hard[0]["gold"] = [ids["plans"]]       # lane (rank 2) vs fork rank 1
    rep3 = ab_exact.run_ab(db, qs_hard, semantic_factory=factory,
                           fork={Q_NAME: [ids["plans"]], Q_DATE: [ids["dentist"]]})
    assert rep3["gate"]["exact_and_date_on_ties_or_beats_fork"] is False
    assert rep3["gate"]["verdict"] == "STAYS OFF"


def test_date_visibility_distinguishes_cannot_see_from_missed(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from revien_bench.tests.test_ab_exact import _add as add_node
    db = str(tmp_path / "v.db")
    s = GraphStore(db_path=db)
    month_only = add_node(
        s, "Trip", "x", event_time_start=datetime(2026, 9, 1, tzinfo=timezone.utc),
        event_time_end=datetime(2026, 9, 30, tzinfo=timezone.utc),
        event_time_granularity=TemporalGranularity.MONTH)
    mtime = add_node(s, "File", "y", recorded_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
                     metadata={"recorded_at_source": "mtime"})
    said = add_node(s, "Said", "z", recorded_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
                    metadata={"recorded_at_source": "content"})
    s.close()
    qs = [{"text": "on 9/21/2026", "gold": [n.node_id], "category": "date"}
          for n in (month_only, mtime, said)]
    qs[1]["text"], qs[2]["text"] = "on Sept 21", "what on 2026-09-21"
    rep = ab_exact.run_ab(db, qs, allow_graph_only=True)
    see = [r["lane_can_see_gold_date"] for r in rep["queries"]]
    assert see == [False, False, True]
    assert rep["queries"][0]["date_visibility"][0]["coarse_event_time_only"]
    assert "CANNOT SEE" in ab_exact.render(rep)


def test_refuses_without_semantic_layer(world, monkeypatch):
    db, qpath, *_ = world

    class Off:
        is_enabled = False

        def inactive_reason(self):
            return "test: off"

    with pytest.raises(SystemExit) as e:
        ab_exact.run_ab(db, ab_exact.load_queries(qpath),
                        semantic_factory=lambda s: Off())
    assert "NOT active" in str(e.value)


def test_cli_end_to_end_graph_only(world, monkeypatch, capsys):
    db, qpath, ids, tmp = world
    for k in ("REVIEN_SEMANTIC", "REVIEN_SEMANTIC_SIM_FLOOR",
              "REVIEN_TOUCH_ON_RECALL", "REVIEN_EXACT", "REVIEN_RERANK"):
        monkeypatch.setenv(k, "0")
    monkeypatch.delenv("REVIEN_EXACT")
    out = str(tmp / "rep.json")
    before = sha(db)
    rc = ab_exact.main(["--db", db, "--queries", qpath, "--floor", "0.458",
                        "--allow-graph-only", "--out", out])
    assert rc == 0 and sha(db) == before
    assert os.environ["REVIEN_SEMANTIC_SIM_FLOOR"] == "0.458"
    rep = json.load(open(out))
    assert rep["settings"]["env"]["REVIEN_SEMANTIC_SIM_FLOOR"] == "0.458"
    text = capsys.readouterr().out
    assert "per-query ranks" in text and "verdict" in text


def test_query_file_formats(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"queries": [
        {"question": "q1", "gold_node_ids": "n1", "cat": "paraphrase"},
        {"q": "q2", "expected": [{"node_id": "n2"}], "type": "exact"}]}))
    qs = ab_exact.load_queries(str(p))
    assert qs == [{"text": "q1", "gold": ["n1"], "category": "paraphrase"},
                  {"text": "q2", "gold": ["n2"], "category": "exact"}]
    p.write_text(json.dumps([{"text": "x"}]))
    with pytest.raises(ValueError):
        ab_exact.load_queries(str(p))


# ── the arithmetic ────────────────────────────────────────────────────────────
def test_rank_and_reciprocal_rank_are_one_based_and_exact():
    ranked = ["a", "b", "c", "d", "e"]
    assert ab_exact.rank_of(ranked, ["a"], 10) == 1
    assert ab_exact.rank_of(ranked, ["c"], 10) == 3
    one = ab_exact.score_one(ranked, ["a"], 10)
    assert (one["rank"], one["rr"], one["hit1"], one["any"]) == (1, 1.0, 1, 1)
    third = ab_exact.score_one(ranked, ["c"], 10)
    assert third["rank"] == 3 and third["rr"] == pytest.approx(1 / 3)
    assert third["hit1"] == 0 and third["any"] == 1 and third["recall"] == 1.0
    # the first gold counts, whatever order gold is listed in
    assert ab_exact.score_one(ranked, ["e", "b"], 10)["rr"] == 0.5


def test_top_n_boundary_and_a_miss():
    ranked = ["a", "b", "c", "d", "e"]
    at_edge = ab_exact.score_one(ranked, ["c"], 3)
    assert at_edge["rank"] == 3 and at_edge["rr"] == pytest.approx(1 / 3)
    past = ab_exact.score_one(ranked, ["d"], 3)
    assert past["rank"] is None and past["rr"] == 0.0 and past["any"] == 0
    assert past["recall"] == 0.0
    assert ab_exact.score_one([], ["a"], 3)["rr"] == 0.0


def test_recall_is_a_fraction_of_gold_and_mrr_is_a_mean():
    ranked = ["a", "b", "c"]
    half = ab_exact.score_one(ranked, ["b", "zz"], 10)
    assert half["recall"] == 0.5 and half["rr"] == 0.5
    rows = [ab_exact.score_one(ranked, ["a"], 10),
            ab_exact.score_one(ranked, ["b"], 10),
            ab_exact.score_one(ranked, ["zz"], 10)]
    agg = ab_exact.aggregate(rows)
    assert agg["n"] == 3 and agg["hit_at_1"] == 1 and agg["any_at_n"] == 2
    assert agg["mrr"] == round((1 + 0.5 + 0) / 3, 4)
    assert agg["recall_at_n"] == round(2 / 3, 4)


# ── the gate ──────────────────────────────────────────────────────────────────
def _row(group, on_rank, fork_rank="skip", off_rank=1, category=None):
    def sc(r):
        return ab_exact.score_one(["x"] * (r - 1) + ["g"] if r else ["x"], ["g"], 10)
    row = {"group": group, "category": category or group, "on": sc(on_rank),
           "off": sc(off_rank), "text": f"{group} q"}
    if fork_rank != "skip":
        row["fork"] = sc(fork_rank)
    return row


def _gate(rows, fork=True):
    groups = sorted({r["group"] for r in rows})
    by_group = {g: {m: ab_exact.aggregate([r[m] for r in rows
                                           if r["group"] == g and m in r])
                    for m in ("off", "on", "fork")} for g in groups}
    return ab_exact.compute_gate(rows, by_group, fork)


def test_gate_compares_on_and_fork_over_the_same_questions():
    # Paired questions: the fork is better (rank 1 and 2 against 1 and 3).
    # The unpaired one is a rank-1 for "on": counted on one side only it
    # lifted every metric (Hit@1 2 vs 1, MRR 0.78 vs 0.75) and hid the loss.
    rows = [_row("paraphrase", 1), _row("exact", 1, 2), _row("exact", 3, 1),
            _row("exact", 1)]
    g = _gate(rows)
    assert g["exact_and_date_on_ties_or_beats_fork"] is False
    assert g["verdict"] == "STAYS OFF"
    assert g["exact_date_questions"] == 3
    assert g["exact_date_questions_with_fork_ranks"] == 2


def test_gate_with_unranked_questions_is_not_eligible():
    para = _row("paraphrase", 1)
    rows = [para, _row("exact", 1, 1), _row("exact", 1, 1), _row("exact", 1)]
    g = _gate(rows)
    assert g["exact_and_date_on_ties_or_beats_fork"] is True
    assert g["verdict"].startswith("NOT EVALUATED")
    assert "1 of 3" in g["verdict"]
    assert _gate([para, _row("exact", 1, 1), _row("exact", 1, 1)])["verdict"] == "ELIGIBLE"


def test_date_only_counts_toward_the_verdict():
    # names win big, dates lose by one place: together they tie or beat the
    # fork, the dates alone do not
    rows = [_row("paraphrase", 1), _row("exact", 1, 2), _row("exact", 1, 2),
            _row("date", 2, 1)]
    g = _gate(rows)
    assert g["exact_and_date_on_ties_or_beats_fork"] is True
    assert g["exact_only_on_ties_or_beats_fork"] is True
    assert g["date_only_on_ties_or_beats_fork"] is False
    assert g["verdict"] == "STAYS OFF"


def test_exact_only_counts_toward_the_verdict_too():
    rows = [_row("paraphrase", 1), _row("exact", 2, 1), _row("date", 1, 2),
            _row("date", 1, 2)]
    g = _gate(rows)
    assert g["exact_and_date_on_ties_or_beats_fork"] is True
    assert g["exact_only_on_ties_or_beats_fork"] is False
    assert g["verdict"] == "STAYS OFF"


def test_gate_needs_paraphrase_data_and_a_fork():
    rows = [_row("exact", 1, 1), _row("paraphrase", 1)]
    assert _gate(rows)["verdict"] == "ELIGIBLE"
    assert _gate([_row("exact", 1, 1)])["verdict"].startswith("NOT EVALUATED")
    assert _gate([_row("paraphrase", 1)], fork=False)["verdict"].startswith(
        "NOT EVALUATED")
    worse = [_row("paraphrase", 2, off_rank=1), _row("exact", 1, 1)]
    assert _gate(worse)["verdict"] == "STAYS OFF"


# ── every category against switch-off, each exact/date category against the fork ──
def test_name_plus_topic_is_grouped_as_exact():
    assert ab_exact.group_of("name_plus_topic") == "exact"


def test_a_name_plus_topic_regression_against_switch_off_stays_off():
    # The category is in the exact group, so it used to be compared with the
    # fork only. On ties the fork there (2 against 2) but is worse than off (1).
    def rows(npt_on):
        return [_row("paraphrase", 1),
                _row("exact", 1, 2, off_rank=2, category="name_title"),
                _row("exact", npt_on, 2, off_rank=1, category="name_plus_topic")]
    g = _gate(rows(2))
    assert g["exact_and_date_on_ties_or_beats_fork"] is True
    assert g["categories_on_ties_or_beats_off"]["name_plus_topic"] is False
    assert g["categories_on_ties_or_beats_off"]["name_title"] is True
    assert g["verdict"] == "STAYS OFF"
    assert _gate(rows(1))["verdict"] == "ELIGIBLE"        # no regression: fine


def test_every_category_must_tie_or_beat_switch_off_even_an_ungrouped_one():
    assert ab_exact.group_of("misc") == "other"
    rows = [_row("paraphrase", 1), _row("exact", 1, 1),
            _row("other", 2, off_rank=1, category="misc")]
    g = _gate(rows)
    assert g["categories_on_ties_or_beats_off"]["misc"] is False
    assert g["verdict"] == "STAYS OFF"
    rows[2] = _row("other", 1, off_rank=1, category="misc")
    assert _gate(rows)["verdict"] == "ELIGIBLE"


def test_each_exact_or_date_category_must_tie_or_beat_the_fork():
    # together the exact group beats the fork; the second category alone loses
    rows = [_row("paraphrase", 1),
            _row("exact", 1, 2, off_rank=2, category="name_title"),
            _row("exact", 1, 2, off_rank=2, category="name_title"),
            _row("exact", 2, 1, off_rank=2, category="name_plus_topic")]
    g = _gate(rows)
    assert g["exact_and_date_on_ties_or_beats_fork"] is True
    assert g["categories_on_ties_or_beats_fork"] == {
        "name_title": True, "name_plus_topic": False}
    assert g["verdict"] == "STAYS OFF"


def test_a_paraphrase_category_is_not_compared_with_the_fork():
    g = _gate([_row("paraphrase", 1), _row("exact", 1, 1)])
    assert "paraphrase" not in g["categories_on_ties_or_beats_fork"]
    assert g["categories_on_ties_or_beats_off"]["paraphrase"] is True


# ── date questions whose gold the lane cannot see ─────────────────────────────
def _unseen(off_rank, on_rank, see=False):
    def sc(r):
        return ab_exact.score_one(["x"] * (r - 1) + ["g"] if r else ["x"], ["g"], 10)
    return {"text": f"q {off_rank}->{on_rank}", "on": sc(on_rank), "off": sc(off_rank),
            "lane_can_see_gold_date": see}


def test_unseen_gold_summary_counts_the_cost():
    rows = [_unseen(5, 9), _unseen(4, 0), _unseen(3, 2), _unseen(1, 1),
            _unseen(0, 0), _unseen(0, 5), _unseen(2, 6, see=True),
            {"text": "no date", "on": {}, "off": {}}]
    u = ab_exact.unseen_gold_summary(rows)
    assert u["questions"] == 6                        # the seen and the dateless are not
    assert u["displaced"] == 2 and u["dropped_out_of_top_n"] == 1
    assert u["improved"] == 2                         # 3->2, and a miss found at 5
    assert u["unchanged"] == 2
    assert (u["hit_at_1_off"], u["hit_at_1_on"]) == (1, 1)
    assert [(d["rank_off"], d["rank_on"]) for d in u["displaced_questions"]] == [
        (5, 9), (4, None)]


def test_unseen_gold_summary_is_empty_without_unseen_rows():
    u = ab_exact.unseen_gold_summary([_unseen(2, 6, see=True)])
    assert u["questions"] == 0 and u["displaced"] == 0
    assert u["displaced_questions"] == []


def test_the_lane_pushing_down_an_unseen_gold_is_reported(tmp_path):
    db = str(tmp_path / "u.db")
    s = GraphStore(db_path=db)
    from datetime import datetime, timezone
    gold = _add(s, "Trip notes", "trip notes from the lake",
                recorded_at=datetime(2026, 9, 21, tzinfo=timezone.utc),
                metadata={"recorded_at_source": "mtime"})       # the lane cannot see it
    _add(s, "Dentist visit", "dentist cleaning", **day(2026, 9, 21))
    s.close()
    q = "what happened on 2026-09-21"
    qs = [{"text": q, "gold": [gold.node_id], "category": "date mixed"}]

    def fac(store):
        emb = _Embedder({"trip notes": 0.70, "dentist cleaning": 0.68},
                        queries=[q.lower()])
        idx = _InMemoryVectorIndex(store, emb)
        idx.reindex_all()
        return idx
    rep = ab_exact.run_ab(db, qs, semantic_factory=fac)
    row = rep["queries"][0]
    assert row["lane_can_see_gold_date"] is False
    assert (row["off"]["rank"], row["on"]["rank"]) == (1, 2)
    u = rep["date_unseen_gold"]
    assert (u["questions"], u["displaced"]) == (1, 1)
    assert u["displaced_questions"][0]["rank_off"] == 1
    assert u["displaced_questions"][0]["rank_on"] == 2
    out = ab_exact.render(rep)
    assert "pushed the gold DOWN on 1" in out and "displaced: 1 -> 2" in out

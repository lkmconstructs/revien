"""Word bonus (arm 6): REVIEN_HYBRID=bonus.

Anchors are chosen as with the switch unset. Every scored node gains
cap * (share of the query's distinct terms its label and content hold), on
top of the finished score and again after the reranker. Off unless asked.
"""

import pytest

from revien.retrieval.engine import RetrievalEngine
from tests.test_exact_lane import (  # noqa: F401  (store is a fixture)
    WORDS, _permuted, _rerank_world, _reranker, add, engine_for, ids_of,
    store, strip,
)

Q = "lunch at harbor street"            # terms: lunch, harbor, street


def _world(store):
    full = add(store, "Harbor lunch", "lunch on harbor street with the team")
    part = add(store, "Harbor meeting", "a meeting near the harbor")
    none = add(store, "Quarterly plan", "budget numbers for the quarter")
    sims = {"lunch on harbor": 0.70, "meeting near": 0.72, "budget numbers": 0.74}
    return (full, part, none), sims


def _recall(store, sims, monkeypatch, hybrid=None, cap=None, **kw):
    monkeypatch.setenv("REVIEN_RERANK", "0")
    monkeypatch.delenv("REVIEN_HYBRID", raising=False)
    monkeypatch.delenv("REVIEN_WORD_BONUS", raising=False)
    if hybrid is not None:
        monkeypatch.setenv("REVIEN_HYBRID", hybrid)
    if cap is not None:
        monkeypatch.setenv("REVIEN_WORD_BONUS", cap)
    eng = engine_for(store, exact=False, sims=sims, queries=[Q], **kw)
    return eng.recall(Q, top_n=10, debug=True)


def _scores(resp):
    return {r.node_id: r.score for r in resp.results}


def _bonuses(resp):
    return {r.node_id: r.score_breakdown["word_bonus"]
            for r in resp.results if "word_bonus" in r.score_breakdown}


class TestOffUnlessAsked:
    def test_unset_gives_no_bonus_and_no_diagnostics(self, store, monkeypatch):
        _, sims = _world(store)
        resp = _recall(store, sims, monkeypatch)
        assert _bonuses(resp) == {}
        assert "word_bonus" not in resp.diagnostics

    def test_rrf_gives_no_bonus(self, store, monkeypatch):
        _, sims = _world(store)
        assert _bonuses(_recall(store, sims, monkeypatch, hybrid="rrf")) == {}

    def test_bonus_mode_chooses_the_same_anchors_as_unset(self, store, monkeypatch):
        # With a zero cap nothing is added, so any difference would come from
        # the anchor path: there is none.
        _, sims = _world(store)
        off = _recall(store, sims, monkeypatch)
        zero = _recall(store, sims, monkeypatch, hybrid="bonus", cap="0")
        assert strip(zero) == strip(off)


    def test_bonus_mode_never_fuses_ranks(self, store, monkeypatch):
        import revien.retrieval.engine as engine_module

        def boom(*a, **k):
            raise AssertionError("rank fusion ran under REVIEN_HYBRID=bonus")
        monkeypatch.setattr(engine_module, "rrf_fuse", boom)
        _, sims = _world(store)
        _recall(store, sims, monkeypatch, hybrid="bonus")
        with pytest.raises(AssertionError, match="rank fusion ran"):
            _recall(store, sims, monkeypatch, hybrid="rrf")


class TestTheBonus:
    def test_it_is_the_share_of_query_terms_times_the_cap(self, store, monkeypatch):
        (full, part, none), sims = _world(store)
        on = _recall(store, sims, monkeypatch, hybrid="bonus")
        assert _bonuses(on) == {
            full.node_id: pytest.approx(0.10, abs=1e-6),
            part.node_id: pytest.approx(0.10 / 3, abs=1e-6),
        }
        assert on.diagnostics["word_bonus"]["query_terms"] == [
            "harbor", "lunch", "street"]

    def test_it_is_added_to_the_finished_score_and_lowers_nothing(
            self, store, monkeypatch):
        (full, part, none), sims = _world(store)
        off = _scores(_recall(store, sims, monkeypatch))
        on = _scores(_recall(store, sims, monkeypatch, hybrid="bonus"))
        assert on[full.node_id] - off[full.node_id] == pytest.approx(0.10)
        assert on[part.node_id] - off[part.node_id] == pytest.approx(0.10 / 3)
        assert on[none.node_id] == pytest.approx(off[none.node_id])
        assert all(on[nid] >= off[nid] for nid in off)

    def test_it_reorders_close_meanings_only(self, store, monkeypatch):
        (full, part, none), sims = _world(store)
        off = _recall(store, sims, monkeypatch)
        on = _recall(store, sims, monkeypatch, hybrid="bonus")
        assert ids_of(off)[:3] == [none.node_id, part.node_id, full.node_id]
        assert ids_of(on)[:3] == [full.node_id, part.node_id, none.node_id]

    def test_a_stopword_only_query_gives_no_bonus(self, store, monkeypatch):
        _, sims = _world(store)
        monkeypatch.setenv("REVIEN_RERANK", "0")
        monkeypatch.setenv("REVIEN_HYBRID", "bonus")
        eng = engine_for(store, exact=False, sims=sims, queries=["what is it"])
        resp = eng.recall("what is it", top_n=10, debug=True)
        assert _bonuses(resp) == {}
        assert resp.diagnostics["word_bonus"]["query_terms"] == []

    @pytest.mark.parametrize("raw, cap", [
        ("0.04", 0.04), ("abc", 0.10), ("-1", 0.10), ("nan", 0.10), ("inf", 0.10),
    ])
    def test_the_cap(self, store, monkeypatch, raw, cap):
        (full, _, _), sims = _world(store)
        on = _recall(store, sims, monkeypatch, hybrid="bonus", cap=raw)
        assert on.diagnostics["word_bonus"]["cap"] == cap
        assert _bonuses(on)[full.node_id] == pytest.approx(cap, abs=1e-6)

    def test_the_engine_reads_the_switch_once(self, store, monkeypatch):
        monkeypatch.setenv("REVIEN_HYBRID", "bonus")
        eng = RetrievalEngine(store)
        monkeypatch.delenv("REVIEN_HYBRID")
        assert eng.word_bonus == pytest.approx(0.10)
        assert RetrievalEngine(store).word_bonus == 0.0


class TestWithTheRerankerOn:
    QR = "hotel entry"      # every node holds "entry"; only Hotel holds both

    @pytest.fixture
    def world(self, store):
        return _rerank_world(store)

    def _run(self, store, world, monkeypatch, hybrid, exact=False, query=None):
        query = query or self.QR
        nodes, sims = world
        rr = _reranker(_permuted("hotel", 3))
        monkeypatch.delenv("REVIEN_HYBRID", raising=False)
        if hybrid:
            monkeypatch.setenv("REVIEN_HYBRID", hybrid)
        eng = engine_for(store, exact=exact, sims=sims,
                         queries=[query.lower()], reranker=rr)
        resp = eng.recall(query, top_n=15, debug=True)
        assert all("rerank_score" in r.score_breakdown for r in resp.results)
        return nodes, resp

    def test_the_bonus_acts_inside_the_reranked_head(self, store, world, monkeypatch):
        nodes, off = self._run(store, world, monkeypatch, None)
        hotel = nodes["hotel"].node_id
        assert ids_of(off).index(hotel) == 3            # the reranker's order
        _, on = self._run(store, world, monkeypatch, "bonus")
        # Hotel holds both terms (+0.10), the rest one (+0.05): 0.05 more than
        # its neighbours, which beats the 0.036 the three slots above it hold.
        assert ids_of(on).index(hotel) == 0
        assert on.diagnostics["word_bonus"]["after_rerank"][hotel] == [3, 0]

    def test_equal_bonuses_keep_the_reranker_order(self, store, world, monkeypatch):
        nodes, off = self._run(store, world, monkeypatch, None)
        _, on = self._run(store, world, monkeypatch, "bonus")
        hotel = nodes["hotel"].node_id
        assert ([n for n in ids_of(on) if n != hotel]
                == [n for n in ids_of(off) if n != hotel])

    def test_with_the_exact_lane_its_moves_report_only_its_own_nodes(
            self, store, world, monkeypatch):
        q = "tell me about Hotel"
        nodes, on = self._run(store, world, monkeypatch, "bonus", exact=True, query=q)
        hotel = nodes["hotel"].node_id
        assert ids_of(on)[0] == hotel
        exact_moves = on.diagnostics["exact"]["after_rerank"]
        assert set(exact_moves) <= {hotel}

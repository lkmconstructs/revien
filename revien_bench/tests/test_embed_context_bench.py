"""Bench side of REVIEN_EMBED_CONTEXT: cache key, layer status, session_key. OFFLINE."""

import json

from revien_bench import runner as R
from revien_bench.tests.test_checkpoint_resume import _FAKE_SHA
from revien_bench.tests.test_db_cache_layer_status import ACTIVE, _write_meta


def test_snapshot_miss_on_embed_context_change(tmp_path, capsys):
    m = tmp_path / "m.db.meta.json"
    _write_meta(m, layer_status={**ACTIVE, "embed_context": "off"})
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a",
                              embed_context="prev") is None
    out = capsys.readouterr().out
    assert "REVIEN_EMBED_CONTEXT=off" in out and "prev" in out
    assert m.exists()
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a",
                              embed_context="off") is not None


def test_legacy_snapshot_counts_as_off(tmp_path):
    m = tmp_path / "l.db.meta.json"
    _write_meta(m, layer_status=ACTIVE)  # no embed_context key: pre-knob
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a",
                              embed_context="off") is not None
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a",
                              embed_context="prev") is None


def test_layer_status_and_aggregate_carry_embed_context():
    class Sem:
        is_enabled = True

        def status(self):
            return {"embedder": "local:fastembed", "embed_model": "m",
                    "embed_dim": 4, "embed_context": "prev"}

    live = R._layer_status(Sem())
    assert live["embed_context"] == "prev"
    agg = R._aggregate_layer_status([live], True)
    assert agg["embed_context"] == "prev"


def test_ingest_sets_session_key(tmp_path):
    from revien.graph.store import GraphStore
    from revien.semantic.index import SemanticIndex
    from revien_bench.ingest_locomo import ingest_conversation
    from revien_bench.loader import Conversation, Turn
    turn = Turn(dia_id="D1:1", speaker="Ann", text="hello there", session=1,
                session_date="7 May 2023")
    conv = Conversation(conv_id="conv-1", speaker_a="Ann", speaker_b="Bo", turns=[turn])
    store = GraphStore(db_path=str(tmp_path / "b.db"))
    try:
        ingest_conversation(conv, store, semantic=SemanticIndex(store, enabled=False))
        keys = {n.session_key for n in store.list_nodes(limit=100)}
        assert keys == {"conv-1:s1"}
    finally:
        store.close()

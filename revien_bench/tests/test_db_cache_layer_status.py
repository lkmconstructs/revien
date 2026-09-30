"""
--db-cache layer-status guard. OFFLINE; never loads a real model.

A snapshot ingested while the semantic layer was silently DISABLED must never be
written, never be reused as a cache hit, and (unless --allow-degraded) must fail
the run with exit 3.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from revien_bench import runner as R
from revien_bench.tests.test_checkpoint_resume import _SpyAnswerer, _two_convs, _FAKE_SHA

ACTIVE = {"semantic_active": True, "rerank_active": False,
          "embedder": "local:fastembed", "embed_model": "model-a",
          "embed_dim": 384, "semantic_inactive_reason": None}
INACTIVE = {"semantic_active": False, "rerank_active": False,
            "embedder": "unbuilt",
            "semantic_inactive_reason": "disabled after runtime error: NoSuchFile"}


@pytest.fixture
def work_dir():
    d = Path(tempfile.mkdtemp(suffix="_dbcache_test"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def env(monkeypatch, work_dir):
    monkeypatch.setattr(R, "load_locomo", lambda _p: _two_convs())
    monkeypatch.setattr(R, "read_locked_hash", lambda: _FAKE_SHA)
    monkeypatch.setattr(R.A, "build_answerer", lambda _n: _SpyAnswerer())
    # Config requests semantic, but the index itself stays off (no model load).
    real = R.SemanticIndex
    monkeypatch.setattr(R, "SemanticIndex", lambda store: real(store, enabled=False))
    return work_dir


def _run(work_dir, config="semantic", **kw):
    return R.run_benchmark(
        config_name=config, answerer_name="extractive",
        dataset_path=work_dir / "ds.json", out_dir=work_dir / "results",
        fresh=True, db_cache=work_dir / "cache", **kw)


def _write_meta(path, **extra):
    meta = {"dataset_sha": _FAKE_SHA, "ingest_fp": R._ingest_fingerprint(),
            "conv_id": "D1", "turns_ingested": 2, "nodes_created": 3,
            "ingest_rate": 1.0}
    meta.update(extra)
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_meta_without_layer_status_is_miss(work_dir, capsys):
    m = work_dir / "x.db.meta.json"
    _write_meta(m)
    assert R._cache_load_meta(m, _FAKE_SHA, semantic_requested=False) is None
    assert "x.db.meta.json" in capsys.readouterr().out
    assert m.exists()  # never deleted


def test_meta_semantic_inactive_under_semantic_config_is_miss(work_dir, capsys):
    m = work_dir / "y.db.meta.json"
    _write_meta(m, layer_status=INACTIVE)
    assert R._cache_load_meta(m, _FAKE_SHA, semantic_requested=True) is None
    assert "y.db.meta.json" in capsys.readouterr().out
    assert m.exists()


def test_meta_with_active_layer_status_is_hit(work_dir):
    m = work_dir / "z.db.meta.json"
    _write_meta(m, layer_status=ACTIVE)
    assert R._cache_load_meta(m, _FAKE_SHA, semantic_requested=True) is not None
    # inactive layer is fine when semantic was not requested (graph_only)
    _write_meta(m, layer_status=INACTIVE)
    assert R._cache_load_meta(m, _FAKE_SHA, semantic_requested=False) is not None


def test_degraded_ingest_not_cached_with_allow_degraded(env, monkeypatch, capsys):
    monkeypatch.setattr(R, "_layer_status", lambda *a, **k: dict(INACTIVE))
    report = _run(env, allow_degraded=True)
    out = capsys.readouterr().out
    assert "[bench] db-cache: NOT caching D1" in out
    assert "semantic layer inactive during ingest" in out
    assert not list((env / "cache").glob("*")) if (env / "cache").exists() else True
    assert report["layer_status"]["semantic_active"] is False
    assert report["layer_status"]["allow_degraded"] is True


def test_degraded_run_exits_3_without_allow_degraded(env, monkeypatch, capsys):
    monkeypatch.setattr(R, "_layer_status", lambda *a, **k: dict(INACTIVE))
    with pytest.raises(SystemExit) as ei:
        _run(env)
    assert ei.value.code == 3
    assert "--allow-degraded" in capsys.readouterr().out


def test_active_layer_writes_meta_with_layer_status_and_results_carry_it(env, monkeypatch):
    monkeypatch.setattr(R, "_layer_status", lambda *a, **k: dict(ACTIVE))
    report = _run(env)
    metas = sorted((env / "cache").glob("*.meta.json"))
    assert len(metas) == 2
    for m in metas:
        assert json.loads(m.read_text(encoding="utf-8"))["layer_status"]["semantic_active"] is True
    ls = report["layer_status"]
    assert ls["semantic_requested"] is True and ls["semantic_active"] is True
    # second run hits the cache (meta has layer_status) and still reports it
    report2 = R.run_benchmark(
        config_name="semantic", answerer_name="extractive",
        dataset_path=env / "ds.json", out_dir=env / "results2",
        fresh=True, db_cache=env / "cache")
    assert report2["layer_status"]["semantic_active"] is True


def test_graph_only_records_layer_status_and_never_fails(env):
    report = _run(env, config="graph_only")
    assert report["layer_status"]["semantic_requested"] is False
    for m in (env / "cache").glob("*.meta.json"):
        assert "layer_status" in json.loads(m.read_text(encoding="utf-8"))


def test_layer_status_reads_live_object():
    class Fake:
        is_enabled = False
        def inactive_reason(self): return "disabled after runtime error: boom"
        def status(self): return {"embedder": "local:fastembed"}
    st = R._layer_status(Fake())
    assert st["semantic_active"] is False
    assert "boom" in st["semantic_inactive_reason"]
    assert st["embedder"] == "local:fastembed"


class _StubSem:
    is_enabled = True
    def status(self): return {"embedder": "local:fastembed"}


def test_layer_status_reports_rerank_depth_and_model_from_reranker(monkeypatch):
    monkeypatch.setenv("REVIEN_RERANK", "0")  # opt-out => disabled reranker
    from revien.semantic.rerank import CrossEncoderReranker
    rr = CrossEncoderReranker(model_name="stub/model", top_k=77,
                              scorer=lambda q, t: [0.0] * len(t))
    status = R._layer_status(_StubSem(), rr)
    assert status["rerank_active"] is True
    assert status["rerank_top_k"] == 77 and status["rerank_model"] == "stub/model"
    for none in (None, CrossEncoderReranker(top_k=5)):  # absent / disabled
        status = R._layer_status(_StubSem(), none)
        assert status["rerank_top_k"] is None and status["rerank_model"] is None


def test_snapshot_written_at_one_depth_is_hit_at_another(work_dir, monkeypatch):
    m = work_dir / "d.db.meta.json"
    _write_meta(m, layer_status={**ACTIVE, "rerank_active": True,
                                 "rerank_top_k": 20, "rerank_model": "a"})
    monkeypatch.setenv("REVIEN_RERANK_TOP_K", "100")
    assert R._cache_load_meta(m, _FAKE_SHA, semantic_requested=True) is not None


def test_env_overrides_capture_out_of_config_only(env, monkeypatch):
    monkeypatch.setenv("REVIEN_RERANK_TOP_K", "100")
    monkeypatch.setenv("REVIEN_BENCH_ALIAS", "0")  # process-only knob
    monkeypatch.setenv("NOT_REVIEN_X", "1")
    cfg_key = next(iter(R._load_config("graph_only").get("env") or {}), None)
    if cfg_key:  # config-set var: must be omitted even if also in process env
        monkeypatch.setenv(cfg_key, "zz")
    report = _run(env, config="graph_only")
    eo = report["env_overrides"]
    assert eo.get("REVIEN_RERANK_TOP_K") == "100"
    assert "NOT_REVIEN_X" not in eo and cfg_key not in eo
    assert eo.get("REVIEN_BENCH_ALIAS") == "0"
    assert "rerank_top_k" in report["layer_status"]


def test_report_renders_depth_and_env_overrides():
    from revien_bench import report as RP
    from revien_bench.tests.test_report import _base_report
    md = RP.render(_base_report(
        layer_status={"semantic_active": True, "rerank_active": True,
                      "rerank_top_k": 100, "embedder": "e"},
        env_overrides={"REVIEN_RERANK_TOP_K": "100"}))
    assert "rerank_top_k=100" in md
    assert "Env overrides" in md and "REVIEN_RERANK_TOP_K=100" in md
    assert "Env overrides" not in RP.render(_base_report())


# ── embedding-model key ─────────────────────────────────────────────


def test_snapshot_miss_under_other_model_hit_under_same(work_dir, capsys):
    m = work_dir / "m.db.meta.json"
    _write_meta(m, layer_status=ACTIVE)  # recorded with model-a
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-b") is None
    out = capsys.readouterr().out
    assert "m.db.meta.json" in out and "model-a" in out and "model-b" in out
    assert m.exists()  # never deleted
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a") is not None


def test_dim_compared_only_when_both_known(work_dir):
    m = work_dir / "d.db.meta.json"
    _write_meta(m, layer_status=ACTIVE)
    assert R._cache_load_meta(m, _FAKE_SHA, True, "model-a", 768) is None
    assert R._cache_load_meta(m, _FAKE_SHA, True, "model-a", 384) is not None
    assert R._cache_load_meta(m, _FAKE_SHA, True, "model-a", None) is not None


def test_legacy_meta_without_embed_model_is_miss_under_semantic(work_dir, capsys):
    m = work_dir / "l.db.meta.json"
    legacy = {k: v for k, v in ACTIVE.items() if not k.startswith("embed_")}
    _write_meta(m, layer_status=legacy)
    assert R._cache_load_meta(m, _FAKE_SHA, True, embed_model="model-a") is None
    assert "no embed_model" in capsys.readouterr().out


def test_graph_only_ignores_embed_model(work_dir):
    m = work_dir / "g.db.meta.json"
    _write_meta(m, layer_status=INACTIVE)
    assert R._cache_load_meta(m, _FAKE_SHA, False, embed_model="model-b") is not None


def test_layer_status_reads_live_index_state():
    class _Idx:
        is_enabled = True

        def status(self):
            return {"embedder": "local:fastembed", "embed_model": "model-x",
                    "embed_dim": 512}

    live = R._layer_status(_Idx())
    assert (live["embed_model"], live["embed_dim"]) == ("model-x", 512)

    class _Off(_Idx):
        is_enabled = False

        def inactive_reason(self):
            return "off"

    off = R._layer_status(_Off())
    assert off["embed_model"] is None and off["embed_dim"] is None


def test_run_end_to_end_model_switch_reingests(env, monkeypatch, capsys):
    class _Prov:
        def __init__(self, name):
            self.model_name = name

    monkeypatch.setattr(R, "_layer_status", lambda *a, **k: dict(ACTIVE))
    monkeypatch.setattr(R, "build_embedder", lambda: _Prov("model-a"))
    _run(env)
    capsys.readouterr()
    monkeypatch.setattr(R, "build_embedder", lambda: _Prov("model-b"))
    R.run_benchmark(
        config_name="semantic", answerer_name="extractive",
        dataset_path=env / "ds.json", out_dir=env / "results2",
        fresh=True, db_cache=env / "cache")
    assert "model-a, this run uses model-b" in capsys.readouterr().out


def test_report_renders_embed_model():
    from revien_bench import report as RP
    rep = {"layer_status": {"semantic_active": True, "rerank_active": False,
                            "embedder": "local:fastembed", "embed_model": "model-a"}}
    assert "embedder=local:fastembed:model-a" in RP.render(rep)
    assert R._embedder_label(rep["layer_status"]) == "local:fastembed:model-a"

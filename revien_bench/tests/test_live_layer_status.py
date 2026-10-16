"""B1: layer status is read LIVE after recall on a db-cache HIT as well as a
miss. A snapshot only proves what ingest saw; if the embedder dies before
recall, the run is degraded and must exit 3 (or be labelled with
--allow-degraded) -- never report semantic_active from the snapshot.
N10: the checkpoint fingerprint covers the bench's own source.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from revien_bench import runner as R
from revien_bench.tests.test_checkpoint_resume import _SpyAnswerer, _two_convs, _FAKE_SHA


class Stub:
    is_cloud = False
    dim = 4

    def __init__(self, name, fail=False):
        self.model_name = name
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("fastembed snapshot dir vanished")
        return [[float(len(t) % 7) + 0.1, 1.0, 0.0, 0.5] for t in texts]


@pytest.fixture
def wd(monkeypatch):
    d = Path(tempfile.mkdtemp(suffix="_live"))
    monkeypatch.setattr(R, "load_locomo", lambda _p: _two_convs())
    monkeypatch.setattr(R, "read_locked_hash", lambda: _FAKE_SHA)
    monkeypatch.setattr(R.A, "build_answerer", lambda _n: _SpyAnswerer())
    # The stub's dim (4) is not the default model's; dim is out of scope here.
    monkeypatch.setattr(R, "_resolve_embed_dim", lambda: None)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _use(monkeypatch, fail=False, enabled=True):
    from revien.semantic.index import SemanticIndex as real
    name = R._resolve_embed_model()
    monkeypatch.setattr(
        R, "SemanticIndex", lambda store: real(store, embedder=Stub(name, fail), enabled=enabled))


def _run(wd, out="results", **kw):
    return R.run_benchmark(
        config_name="semantic", answerer_name="extractive", dataset_path=wd / "ds.json",
        out_dir=wd / out, fresh=True, db_cache=wd / "cache", **kw)


def _prime_cache(wd, monkeypatch):
    _use(monkeypatch)
    r1 = _run(wd)
    assert r1["layer_status"]["semantic_active"] is True
    assert len(list((wd / "cache").glob("*.meta.json"))) == 2


def test_cache_hit_then_embedder_dies_at_recall_exits_3(wd, monkeypatch, capsys):
    _prime_cache(wd, monkeypatch)
    _use(monkeypatch, fail=True)  # cache hit skips ingest; recall embeds the query
    with pytest.raises(SystemExit) as ei:
        _run(wd, out="results2")
    assert ei.value.code == 3
    out = capsys.readouterr().out
    assert "during recall" in out  # proves it was the HIT path, not an ingest miss


def test_cache_hit_degraded_with_allow_degraded_reports_inactive(wd, monkeypatch):
    _prime_cache(wd, monkeypatch)
    _use(monkeypatch, fail=True)
    r2 = _run(wd, out="results2", allow_degraded=True)
    assert r2["layer_status"]["semantic_active"] is False
    assert r2["layer_status"]["allow_degraded"] is True


def test_cache_hit_with_disabled_index_is_not_reported_active(wd, monkeypatch):
    _prime_cache(wd, monkeypatch)
    _use(monkeypatch, enabled=False)
    with pytest.raises(SystemExit) as ei:
        _run(wd, out="results2")
    assert ei.value.code == 3


def test_cache_hit_healthy_still_reports_active(wd, monkeypatch):
    _prime_cache(wd, monkeypatch)
    _use(monkeypatch)
    r2 = _run(wd, out="results2")
    assert r2["layer_status"]["semantic_active"] is True


def test_run_fingerprint_changes_with_bench_source(monkeypatch):
    a = R._run_fingerprint()
    monkeypatch.setattr(R, "_bench_code_fingerprint", lambda: "deadbeef")
    assert R._run_fingerprint() != a


def test_bench_code_fingerprint_is_content_hash(tmp_path, monkeypatch):
    assert R._bench_code_fingerprint() == R._bench_code_fingerprint()
    assert len(R._bench_code_fingerprint()) == 8

"""G18: a benchmark run never produces a number from a degraded semantic
layer, on cache miss or cache hit.

CHECK: python scripts/gates/check_bench_degraded.py
EXPECT: bench degraded-run verification passed
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"
os.environ.pop("REVIEN_EMBED_CONTEXT", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.semantic.index import SEMANTIC_AVAILABLE, SemanticIndex  # noqa: E402
from revien_bench import answerers as A  # noqa: E402
from revien_bench import runner as R  # noqa: E402
from revien_bench.loader import QA, Conversation, Turn  # noqa: E402

FAKE_SHA = "deadbeef" * 8


def fail(msg):
    print(f"check_bench_degraded: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


class Stub:
    """Stub embedder; fail=True models a model dir that vanished."""
    is_cloud = False
    dim = 4

    def __init__(self, name, fail=False):
        self.model_name = name
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("fastembed snapshot dir vanished")
        return [[float(len(t) % 7) + 0.1, 1.0, 0.0, 0.5] for t in texts]


class Reader:
    name = "extractive"
    network_calls = 0
    cost_usd_estimate = 0.0

    def __init__(self):
        self._inner = A.ExtractiveAnswerer()

    def answer(self, ctx):
        return self._inner.answer(ctx)


def one_conv():
    c = Conversation(conv_id="D1", speaker_a="Alice", speaker_b="Bob")
    c.session_dates = {1: "7 May 2023"}
    c.turns = [
        Turn(dia_id="D1:1", speaker="Alice", session=1, session_date="7 May 2023",
             text="We deployed the backend on PostgreSQL."),
        Turn(dia_id="D1:2", speaker="Bob", session=1, session_date="7 May 2023",
             text="The JWT tokens live in Redis."),
    ]
    c.qa = [QA(question="What database for alpha?", answer="PostgreSQL", category=4,
               evidence=["D1:1"])]
    return c


def use(model="stub-model-a", fail=False):
    real = SemanticIndex
    R.SemanticIndex = lambda store: real(store, embedder=Stub(model, fail), enabled=True)
    R.build_embedder = lambda: Stub(model)


def run(wd, out, cache, **kw):
    buf = io.StringIO()
    code = None
    report = None
    with contextlib.redirect_stdout(buf):
        try:
            report = R.run_benchmark(
                config_name="semantic", answerer_name="extractive",
                dataset_path=wd / "ds.json", out_dir=wd / out, fresh=True,
                db_cache=wd / cache, **kw)
        except SystemExit as e:
            code = e.code
    return report, code, buf.getvalue()


def files(d):
    return sorted(p.name for p in d.glob("*")) if d.exists() else []


def result_jsons(wd, out):
    # a results report, not the resume checkpoint
    return [n for n in files(wd / out) if n.endswith(".json")]


def snapshots(wd, cache):
    return [n for n in files(wd / cache) if n.endswith(".meta.json") or n.endswith(".db")]


def main_check(wd):
    check(SEMANTIC_AVAILABLE, "sqlite-vec not importable; the semantic layer cannot be exercised")
    R.load_locomo = lambda _p: [one_conv()]
    R.read_locked_hash = lambda: FAKE_SHA
    R.A.build_answerer = lambda _n: Reader()
    R._resolve_embed_dim = lambda: None  # stub dim is not the default model's

    # 1. Healthy run: writes a snapshot recording the embedder and recipe.
    use()
    rep, code, _ = run(wd, "r1", "c1")
    check(code is None and rep is not None, f"healthy run failed (exit {code})")
    ls = rep["layer_status"]
    check(ls["semantic_active"] is True, "healthy run: semantic_active is not True")
    check(result_jsons(wd, "r1"), "healthy run wrote no results JSON")
    metas = sorted((wd / "c1").glob("*.meta.json"))
    check(len(metas) == 1, f"healthy run wrote {len(metas)} snapshot metas, expected 1")
    snap = json.loads(metas[0].read_text(encoding="utf-8"))["layer_status"]
    check(snap.get("semantic_active") is True and snap.get("embed_model") == "stub-model-a"
          and "embed_context" in snap and snap["embed_context"] == "off",
          f"snapshot layer_status lacks embed_model/embed_context: {snap}")
    check(ls.get("embed_model") == "stub-model-a" and ls.get("embed_context") == "off",
          "report layer_status lacks embed_model/embed_context")

    # 2. Dead embedder on a cache MISS: exit 3, no results JSON, no snapshot.
    use(fail=True)
    rep, code, out = run(wd, "r2", "c2")
    check(code == 3, f"dead embedder on MISS: exit {code!r}, expected 3")
    check(rep is None and not result_jsons(wd, "r2"), "dead embedder on MISS wrote a results JSON")
    check(not snapshots(wd, "c2"), f"dead embedder on MISS wrote a snapshot: {snapshots(wd, 'c2')}")
    check("--allow-degraded" in out, "MISS exit did not name --allow-degraded")

    # 3. Healthy snapshot (c1), then dead embedder on a cache HIT: exit 3, and
    #    no report may claim semantic_active True.
    use(fail=True)
    rep, code, out = run(wd, "r3", "c1")
    check(code == 3, f"dead embedder on HIT: exit {code!r}, expected 3")
    check("during recall" in out, "HIT exit was not the recall-time guard (cache not hit?)")
    for name in result_jsons(wd, "r3"):
        body = json.loads((wd / "r3" / name).read_text(encoding="utf-8"))
        check((body.get("layer_status") or {}).get("semantic_active") is not True,
              "a report from a dead embedder on HIT says semantic_active True")
    check(rep is None, "dead embedder on HIT returned a report")

    # 4. --allow-degraded completes but labels the run inactive; nothing cached.
    use(fail=True)
    rep, code, _ = run(wd, "r4", "c4", allow_degraded=True)
    check(code is None and rep is not None, f"--allow-degraded did not complete (exit {code})")
    check(rep["layer_status"]["semantic_active"] is False, "--allow-degraded: semantic_active not False")
    check(rep["layer_status"].get("allow_degraded") is True, "--allow-degraded not recorded")
    check(not snapshots(wd, "c4"), "--allow-degraded cached a degraded ingest")
    use(fail=True)
    rep, code, _ = run(wd, "r5", "c1", allow_degraded=True)  # HIT path
    check(code is None and rep["layer_status"]["semantic_active"] is False,
          "--allow-degraded on HIT did not report semantic_active False")

    # 5. Snapshot keying. Positive control: exact match is a HIT.
    use()
    with contextlib.redirect_stdout(io.StringIO()):
        rep, code, out = run(wd, "r6", "c1")
    check(code is None and "ignoring stale snapshot" not in out,
          "positive control: an exact-match snapshot was not a HIT")
    meta = metas[0]
    # the ingest fingerprint is env-sensitive: judge under the run's own config env
    prev_env = R._apply_env(R._load_config("semantic").get("env", {}))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hit = R._cache_load_meta(meta, FAKE_SHA, True, "stub-model-a", None, "off")
            other_model = R._cache_load_meta(meta, FAKE_SHA, True, "stub-model-b", None, "off")
            other_ctx = R._cache_load_meta(meta, FAKE_SHA, True, "stub-model-a", None, "prev")
    finally:
        R._restore_env(prev_env)
    check(hit is not None, "positive control: exact model+context match is not a HIT")
    check(other_model is None, "a snapshot from a different embed_model was reused")
    check(other_ctx is None, "a snapshot from a different embed_context was reused")
    # end to end: another model, and another context, each re-ingest.
    use("stub-model-b")
    rep, code, out = run(wd, "r7", "c1")
    check(code is None and "stub-model-a, this run uses stub-model-b" in out,
          "model switch did not re-ingest")
    use("stub-model-b")  # r7 re-snapshotted c1 under model-b / off
    os.environ["REVIEN_EMBED_CONTEXT"] = "prev"
    try:
        rep, code, out = run(wd, "r8", "c1")
    finally:
        os.environ.pop("REVIEN_EMBED_CONTEXT", None)
    check(code is None and "REVIEN_EMBED_CONTEXT=off, this run uses prev" in out,
          "context switch did not re-ingest")


if __name__ == "__main__":
    wd = Path(tempfile.mkdtemp(suffix="_benchgate"))
    try:
        main_check(wd)
    except SystemExit:
        raise
    except Exception as exc:  # any crash is a failed gate
        fail(f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    print("bench degraded-run verification passed")

"""ONNX Runtime telemetry is switched off before any model/session exists,
and reported as off only when it is.

Stub onnxruntime + stub fastembed; no real model loads, no network. The
subprocess tests run `python -c` with a throwaway HOME and every proxy pointed
at a closed loopback port, and exit at once (importing onnxruntime 1.29+
without the switch uploads about 9 s later; none of them lives that long)."""

import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import revien
from revien.semantic import _ort
from revien.semantic import index as idx
from revien.semantic import rerank as rr


def _fresh(monkeypatch, preimported=False, preset=None, env="1"):
    """A process state as `import revien` left it: onnxruntime imported
    before revien or not, what the variable held before revien touched it,
    and what it holds now (None = unset)."""
    monkeypatch.setattr(_ort, "_DONE", False)
    monkeypatch.setattr(_ort, "_SEEN_ENV", getattr(_ort, "_UNSEEN", None),
                        raising=False)
    monkeypatch.setattr(_ort, "_WARNED", False, raising=False)
    monkeypatch.setattr(revien, "_ORT_PREIMPORTED", preimported, raising=False)
    monkeypatch.setattr(revien, "_ORT_TELEMETRY_PRESET", preset, raising=False)
    if env is None:
        monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    else:
        monkeypatch.setenv("ORT_DISABLE_TELEMETRY", env)


@pytest.fixture
def seq(monkeypatch):
    calls = []
    ort = types.ModuleType("onnxruntime")
    ort.disable_telemetry_events = lambda: calls.append("disable")

    class TE:
        def __init__(self, *a, **k):
            calls.append("TextEmbedding")

    class TCE:
        def __init__(self, *a, **k):
            calls.append("TextCrossEncoder")

        @classmethod
        def add_custom_model(cls, *a, **k):
            pass

    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = TE
    rer = types.ModuleType("fastembed.rerank")
    ce = types.ModuleType("fastembed.rerank.cross_encoder")
    ce.TextCrossEncoder = TCE
    for name, mod in [("onnxruntime", ort), ("fastembed", fe),
                      ("fastembed.rerank", rer),
                      ("fastembed.rerank.cross_encoder", ce)]:
        monkeypatch.setitem(sys.modules, name, mod)
    _fresh(monkeypatch)
    monkeypatch.setattr(idx, "_FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(rr, "_FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(rr, "_register_builtin_variants", lambda: None)
    return calls


def test_disabled_once_and_before_models(seq):
    idx.FastEmbedProvider()._ensure_model()
    idx.FastEmbedProvider()._ensure_model()
    rr.CrossEncoderReranker()._ensure_model()
    rr.CrossEncoderReranker()._ensure_model()
    assert seq.count("disable") == 1
    assert seq[0] == "disable"
    assert seq == ["disable", "TextEmbedding", "TextEmbedding",
                   "TextCrossEncoder", "TextCrossEncoder"]


def test_reranker_alone_disables_first(seq):
    rr.CrossEncoderReranker()._ensure_model()
    assert seq == ["disable", "TextCrossEncoder"]


def test_missing_onnxruntime_false_no_raise(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises
    assert _ort.disable_ort_telemetry() is False
    assert _ort.ort_telemetry_state() is None


def test_raising_switch_never_raises(monkeypatch):
    ort = types.ModuleType("onnxruntime")

    def boom():
        raise RuntimeError("x")

    ort.disable_telemetry_events = boom
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    _fresh(monkeypatch)
    assert _ort.disable_ort_telemetry() is False


def test_sovereignty_detail_carries_field(seq):
    from revien_bench import sovereignty as S
    assert S.network_egress_zero().detail["ort_telemetry_disabled"] is True


def test_sovereignty_detail_none_when_absent(monkeypatch):
    from revien_bench import sovereignty as S
    _fresh(monkeypatch)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert S.network_egress_zero().detail["ort_telemetry_disabled"] is None


def test_semantic_status_field(seq):
    from revien.graph.store import GraphStore
    s = idx.SemanticIndex(GraphStore(db_path=":memory:"))
    assert s.status()["ort_telemetry_disabled"] is True


# ── the switch is the environment variable, set by `import revien` ─────────


def _child(code, env_value=None):
    """Run `code` in a fresh interpreter with this checkout first on the path,
    a throwaway HOME, and every proxy pointed at a closed loopback port."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.pop("ORT_DISABLE_TELEMETRY", None)
    if env_value is not None:
        env["ORT_DISABLE_TELEMETRY"] = env_value
    home = os.environ["HOME"]  # conftest gives every test a throwaway home
    env.update(HOME=home, USERPROFILE=home, NO_PROXY="", no_proxy="")
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        env[k] = "http://127.0.0.1:9"
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=root,
                          capture_output=True, text=True, timeout=120)


_SHOW = ("import revien, os; "
         "print(repr(os.environ.get('ORT_DISABLE_TELEMETRY')))")


def test_import_revien_sets_the_switch():
    out = _child(_SHOW)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "'1'"


@pytest.mark.parametrize("value", ["0", "false", "true", "1", "off"])
def test_explicit_value_is_kept(value):
    out = _child(_SHOW, env_value=value)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == repr(value)


def test_empty_value_counts_as_unset():
    out = _child(_SHOW, env_value="")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "'1'"


def test_child_process_inherits_the_switch():
    code = ("import revien, subprocess, sys; subprocess.run([sys.executable, "
            "'-c', 'import os; print(repr(os.environ.get(\"ORT_DISABLE_TELEMETRY\")))'])")
    out = _child(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "'1'"


def test_revien_package_import_pulls_in_no_onnxruntime():
    # the switch only works if nothing imports onnxruntime before it is set
    out = _child("import revien, sys; print('onnxruntime' in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# measured against onnxruntime 1.31.0 on Linux (see OFF_VALUES, ORT_WHITESPACE)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "Yes",
                                   "y", "on", "ON", " 1", "1 ", " true"])
def test_off_values(value):
    assert _ort.env_value_is_off(value) is True


@pytest.mark.parametrize("value", [None, "", "  ", "0", "false", "no", "off",
                                   "t", "2", "01", "1 1"])
def test_not_off_values(value):
    assert _ort.env_value_is_off(value) is False


# ORT trims exactly ASCII space, \t, \n, \r, \v and \f from either end
# (measured against onnxruntime 1.31.0: each of these gave 0 connections)
@pytest.mark.parametrize("value", ["\t1", "\n1", "\r1", "\v1", "\f1", "1\t",
                                   "1\n", "1\r", "1\v", "1\f", " \t1\t ",
                                   "\v\ftrue\r\n"])
def test_ascii_whitespace_is_trimmed_like_ort(value):
    assert _ort.env_value_is_off(value) is True


# ...and nothing else: Python's str.strip() would remove the first four, ORT
# does not (measured: a no-break space or \x1c before "1" gave 1 connection
# and the ~/.cache/Microsoft folder)
@pytest.mark.parametrize("value", ["\u00a01", "1\u00a0", "\x1c1", "\x1f1",
                                   "\u20031", "\x011", "\u2028true"])
def test_other_whitespace_is_not_trimmed(value):
    assert _ort.env_value_is_off(value) is False


@pytest.mark.parametrize("value", ["\u00a01", "\x1c1"])
def test_unicode_padded_one_is_reported_on(monkeypatch, caplog, value):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=False, preset=value, env=value)
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is False
    assert "does not treat as off" in _ort.ort_telemetry_problem()
    assert len(_warnings(caplog)) == 1


def test_ascii_whitespace_only_counts_as_unset():
    out = _child(_SHOW, env_value=" \t")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "'1'"


def test_no_break_space_is_a_value_not_empty():
    # ORT does not trim it, so it is the user's (not-off) value: kept, and
    # reported as on rather than silently replaced or called off
    out = _child(_SHOW, env_value="\u00a0")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == repr("\u00a0")


# ── honest reporting ────────────────────────────────────────────────────────


def _stub_ort(monkeypatch):
    ort = types.ModuleType("onnxruntime")
    ort.disable_telemetry_events = lambda: None
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)


def _warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "ONNX Runtime telemetry" in r.getMessage()]


def test_preimported_without_switch_is_false_and_warns_once(monkeypatch, caplog):
    _stub_ort(monkeypatch)
    # revien set the variable, but onnxruntime had already read the environment
    _fresh(monkeypatch, preimported=True, preset=None, env="1")
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is False
    assert _ort.disable_ort_telemetry() is False
    assert _ort.ort_telemetry_state() is False
    found = _warnings(caplog)
    assert len(found) == 1
    msg = found[0].getMessage()
    assert "imported before revien" in msg
    assert "set ORT_DISABLE_TELEMETRY=1 before importing onnxruntime" in msg
    assert "imported before revien" in _ort.ort_telemetry_problem()


def test_preimported_with_users_off_value_is_true(monkeypatch, caplog):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=True, preset="true", env="true")
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is True
    assert _ort.ort_telemetry_problem() is None
    assert _warnings(caplog) == []


def test_preimported_with_users_other_value_is_false(monkeypatch, caplog):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=True, preset="0", env="0")
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is False
    assert "ORT_DISABLE_TELEMETRY='0'" in _ort.ort_telemetry_problem()
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_users_explicit_not_off_value_is_reported_truthfully(monkeypatch, caplog, value):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=False, preset=value, env=value)
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is False
    problem = _ort.ort_telemetry_problem()
    assert f"ORT_DISABLE_TELEMETRY={value!r}" in problem
    assert "does not treat as off" in problem
    assert len(_warnings(caplog)) == 1


def test_value_is_read_when_onnxruntime_is_first_imported(monkeypatch):
    """Revien's own first import of onnxruntime snapshots the variable: a
    change after that does not change what ORT already read."""
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, env="1")
    assert _ort.ort_telemetry_state() is True
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    assert _ort.ort_telemetry_state() is True
    _fresh(monkeypatch, env=None)  # removed before onnxruntime was imported
    assert _ort.ort_telemetry_state() is False
    assert "unset when onnxruntime was imported" in _ort.ort_telemetry_problem()


def test_absent_onnxruntime_reports_none_and_no_warning(monkeypatch, caplog):
    _fresh(monkeypatch, preimported=True, preset=None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    caplog.set_level(logging.WARNING)
    assert _ort.ort_telemetry_state() is None
    assert _ort.ort_telemetry_problem() is None
    assert _warnings(caplog) == []


def test_real_preimport_reports_false():
    """A host that imports onnxruntime before revien: the real modules. The
    child exits at once and its proxies point at a closed port, so nothing is
    sent."""
    pytest.importorskip("onnxruntime")
    code = ("import onnxruntime, revien; "
            "from revien.semantic._ort import ort_telemetry_state as s; "
            "print(s(), s())")
    out = _child(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"
    assert out.stderr.count("ONNX Runtime telemetry is NOT off") == 1
    assert "ORT_DISABLE_TELEMETRY=1 before importing onnxruntime" in out.stderr


def test_real_default_path_reports_true():
    pytest.importorskip("onnxruntime")
    code = ("import revien, onnxruntime, os; "
            "from revien.semantic._ort import ort_telemetry_state as s; "
            "print(s(), os.environ.get('ORT_DISABLE_TELEMETRY'))")
    out = _child(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "True 1"
    assert "NOT off" not in out.stderr


# ── every surface reports the truth ─────────────────────────────────────────


def test_sovereignty_fails_when_telemetry_is_on(monkeypatch):
    from revien_bench import sovereignty as S
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=True, preset=None)
    chk = S.network_egress_zero()
    assert chk.passed is False
    assert chk.detail["ort_telemetry_disabled"] is False
    assert "imported before revien" in chk.detail["ort_telemetry_problem"]


def test_sovereignty_passes_when_telemetry_is_off(monkeypatch):
    from revien_bench import sovereignty as S
    _stub_ort(monkeypatch)
    _fresh(monkeypatch)
    chk = S.network_egress_zero()
    assert chk.passed is True
    assert chk.detail["ort_telemetry_disabled"] is True
    assert chk.detail["ort_telemetry_problem"] is None


def test_semantic_status_reports_false(monkeypatch):
    from revien.graph.store import GraphStore
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=False, preset="0", env="0")
    st = idx.SemanticIndex(GraphStore(db_path=":memory:")).status()
    assert st["ort_telemetry_disabled"] is False
    assert "ORT_DISABLE_TELEMETRY='0'" in st["ort_telemetry_problem"]


def _status_line(tmp_path):
    from click.testing import CliRunner
    from revien.cli import main
    from revien.graph.store import GraphStore
    db = str(tmp_path / "s.db")
    GraphStore(db_path=db).close()
    res = CliRunner().invoke(main, ["status", "--db", db])
    assert res.exit_code == 0, res.output
    return [l for l in res.output.splitlines()
            if l.startswith("ONNX Runtime telemetry:")]


def test_cli_status_says_not_disabled(monkeypatch, tmp_path):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch, preimported=True, preset=None)
    (line,) = _status_line(tmp_path)
    assert line.startswith("ONNX Runtime telemetry: NOT disabled: ")
    assert "ORT_DISABLE_TELEMETRY=1 before importing onnxruntime" in line


def test_cli_status_says_disabled(monkeypatch, tmp_path):
    _stub_ort(monkeypatch)
    _fresh(monkeypatch)
    assert _status_line(tmp_path) == ["ONNX Runtime telemetry: disabled"]


def test_cli_status_not_installed(monkeypatch, tmp_path):
    _fresh(monkeypatch)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert _status_line(tmp_path) == [
        "ONNX Runtime telemetry: not applicable (onnxruntime not installed)"]

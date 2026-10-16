"""scripts/gates/check_egress.py must catch the ways to reach the pinned
loopback probe: re-exporting its names and monkeypatching it. These tests call
the gate's functions (never its main(), which runs pytest and would recurse).
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "check_egress_gate", REPO / "scripts" / "gates" / "check_egress.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VERIFIER_REEXPORT = (
    "from revien.loopback_probe import http as _h\n"
    "def f():\n"
    "    return _h.client.HTTPSConnection('evil.example')\n"
)


class TestReExport:
    def test_the_verifier_shape_is_reported(self, gate):
        """The exact shape that passed the earlier gate."""
        assert gate.probe_reaches(VERIFIER_REEXPORT)

    @pytest.mark.parametrize("src", [
        "from revien.loopback_probe import socket as s\n",
        "from revien.loopback_probe import *\n",
        "from revien.loopback_probe import probe_daemon, http\n",
        "from revien.loopback_probe import probe_daemon as p\n",
        "import revien.loopback_probe\n",
        "import revien.loopback_probe as lp\nlp.http\n",
        "from revien import loopback_probe\n",
        "from . import loopback_probe\n",
        "from .loopback_probe import probe_daemon\n",
        "import importlib\nimportlib.import_module('revien.loopback_probe')\n",
        "import revien\nrevien.loopback_probe.http\n",
    ])
    def test_every_other_way_of_importing_it_is_reported(self, gate, src):
        assert gate.probe_reaches(src), src

    def test_the_old_network_scan_alone_does_not_see_the_verifier_shape(
            self, gate):
        """Why the new scan exists: nothing names a network module here."""
        assert gate.network_uses(VERIFIER_REEXPORT) == []

    def test_the_one_allowed_use_is_not_reported(self, gate):
        ok = ("def probe(url):\n"
              "    from revien.loopback_probe import probe_daemon\n"
              "    return probe_daemon(url)\n")
        assert gate.probe_reaches(ok) == []
        assert gate.probe_reaches(
            '"""Talks to revien/loopback_probe.py."""\nx = 1\n') == []


class TestMonkeypatch:
    @pytest.mark.parametrize("src", [
        "from revien.loopback_probe import probe_daemon\n"
        "probe_daemon.__globals__['socket'] = object()\n",
        "from revien.loopback_probe import probe_daemon\n"
        "probe_daemon.__code__ = (lambda u: None).__code__\n",
        "from revien.loopback_probe import probe_daemon\n"
        "setattr(probe_daemon, '__code__', None)\n",
        "from revien.loopback_probe import probe_daemon\n"
        "patch(probe_daemon)\n",
        "from revien.loopback_probe import probe_daemon\n"
        "p = probe_daemon\n",
        "import revien.loopback_probe as m\nm.probe_daemon = lambda u: None\n",
        "monkeypatch.setattr('revien.loopback_probe.probe_daemon', f)\n",
        "mock.patch('revien.loopback_probe.socket')\n",
        "import sys\nsys.modules['revien.loopback_probe'].x = 1\n",
    ])
    def test_patching_shapes_are_reported(self, gate, src):
        assert gate.probe_reaches(src), src


class TestTreeScan:
    def test_the_shipped_tree_is_clean(self, gate):
        assert gate.closure_violations(REPO) == []

    def test_a_re_export_added_to_cli_is_found_in_the_tree(
            self, gate, tmp_path):
        root = tmp_path / "repo"
        (root / "revien").mkdir(parents=True)
        shutil.copy(REPO / "revien" / "loopback_probe.py",
                    root / "revien" / "loopback_probe.py")
        (root / "revien" / "cli.py").write_text(VERIFIER_REEXPORT)
        hits = gate.closure_violations(root)
        assert hits and hits[0][0] == "revien/cli.py"

    def test_a_patch_added_to_a_script_is_found(self, gate, tmp_path):
        root = tmp_path / "repo"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "x.py").write_text(
            "from revien.loopback_probe import probe_daemon\n"
            "probe_daemon.__globals__['http'] = None\n")
        assert gate.closure_violations(root)


class TestPin:
    def test_the_pinned_file_matches(self, gate):
        real = (REPO / "revien" / "loopback_probe.py").read_bytes()
        assert gate.pin_problem("revien/loopback_probe.py", real) is None

    @pytest.mark.parametrize("edit", [
        lambda b: b.replace(b"import json", b"import json, os"),
        lambda b: b + b"\ndef probe_daemon(url):\n    return None\n",
        lambda b: b.replace(b"def probe_daemon(", b"@(lambda f: f)\ndef probe_daemon("),
        lambda b: b + b"\nsocket = 1\n",
    ])
    def test_any_edit_fails_the_pin(self, gate, edit):
        real = (REPO / "revien" / "loopback_probe.py").read_bytes()
        assert gate.pin_problem("revien/loopback_probe.py", edit(real))

    def test_other_files_are_not_pinned(self, gate):
        assert gate.pin_problem("revien/cli.py", b"x")


class TestSelfTests:
    def test_every_probe_control_comes_out_as_stated(self, gate):
        for name, src, want in gate.PROBE_CONTROLS:
            assert bool(gate.probe_reaches(src)) == want, name

    def test_listed_evasions_really_evade(self, gate):
        for name, src in gate.PROBE_EVASIONS:
            assert gate.probe_reaches(src) == [], name

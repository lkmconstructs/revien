"""G8: sovereignty and egress tests pass, and no changed source file imports
a network client on the default path.

CHECK: python scripts/gates/check_egress.py
EXPECT: egress verification passed
"""
import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def fail(msg):
    print(f"check_egress: {msg}", file=sys.stderr)
    sys.exit(1)


NETWORK_MODULES = (
    "httpx", "requests", "urllib.request", "urllib3", "socket", "aiohttp",
    "http.client", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc.client", "websockets", "websocket",
)
# A parent package whose network submodule is reached as an attribute:
# `import http` then `http.client.HTTPConnection(...)`.
_PARENTS = {m.split(".")[0] for m in NETWORK_MODULES if "." in m}
DYNAMIC = "<dynamic-import>"


def _matches(name: str):
    """The NETWORK_MODULES entry that `name` is or lies under, else None."""
    for m in NETWORK_MODULES:
        if name == m or name.startswith(m + "."):
            return m
    return None


class _Scan(ast.NodeVisitor):
    """Every network-client use in one module's source, as
    (function qualname or '<module>', module, statement text).

    Covers: `import a, b` (every name), `import http.client`, `from x import
    y` (including `from http import client`, `from urllib import request`,
    `import *`), a parent package used through an attribute (`import http` then
    `http.client...`), and the dynamic-import facility (`__import__`,
    `importlib.import_module`, `from importlib import import_module`, aliases
    of it): a literal argument names the module, anything else is
    `<dynamic-import>`. The position is the ENCLOSING FUNCTION, so an
    allowlist entry pins the one function that may do it."""

    def __init__(self, src: str):
        self.src = src
        self.scope = ["<module>"]
        self.alias = [{}]          # local name -> dotted module
        self.found = []

    def _line(self, node) -> str:
        seg = ast.get_source_segment(self.src, node) or ""
        return " ".join(seg.split())[:120]

    def _where(self) -> str:
        return ".".join(x for x in self.scope[1:]) or "<module>"

    def _add(self, module, node, stmt=None):
        self.found.append((self._where(), module, stmt or self._line(node)))

    def _func(self, node):
        self.scope.append(node.name)
        self.alias.append({})
        self.generic_visit(node)
        self.alias.pop()
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _func

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node):
        stmt = self._line(node)
        for a in node.names:
            if a.asname:
                self.alias[-1][a.asname] = a.name
            else:
                self.alias[-1][a.name.split(".")[0]] = a.name.split(".")[0]
            m = _matches(a.name)
            if m:
                self._add(m, node, stmt)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        stmt = self._line(node)
        base = node.module or ""
        for a in node.names:
            full = f"{base}.{a.name}" if a.name != "*" else base
            self.alias[-1][a.asname or a.name] = full
            m = _matches(full) or _matches(base)
            if m:
                self._add(m, node, stmt)
            if full in ("importlib.import_module", "importlib.__import__"):
                self._add(DYNAMIC, node, stmt)
        self.generic_visit(node)

    def _resolve(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        root = node.id
        for scope in reversed(self.alias):
            if root in scope:
                root = scope[root]
                break
        return ".".join([root] + list(reversed(parts)))

    def visit_Attribute(self, node):
        dotted = self._resolve(node)
        if dotted:
            m = _matches(dotted)
            # only the parent-package route: a bare `socket.socket` is already
            # reported by its import statement
            if m and "." in m and dotted.split(".")[0] in _PARENTS:
                self._add(m, node)
                return
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else "")
        resolved = self._resolve(f) if isinstance(f, ast.Attribute) else (
            next((sc[f.id] for sc in reversed(self.alias) if f.id in sc), f.id)
            if isinstance(f, ast.Name) else "")
        if (name in ("import_module", "__import__")
                or resolved in ("importlib.import_module", "__import__")):
            arg = node.args[0] if node.args else None
            lit = (arg.value if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str) else None)
            if lit is None:
                self._add(DYNAMIC, node)
            elif _matches(lit):
                self._add(_matches(lit), node)
        self.generic_visit(node)


def network_uses(src: str):
    """[(function, module, statement)] for every network-client use in `src`."""
    scan = _Scan(src)
    scan.visit(ast.parse(src))
    return scan.found


def new_network_uses(head_src: str, base_src: str):
    """Uses present in `head_src` and not already (same function, module and
    statement) in `base_src`."""
    base = list(network_uses(base_src)) if base_src else []
    out = []
    for use in network_uses(head_src):
        if use in base:
            base.remove(use)
        else:
            out.append(use)
    return out


# Network-client uses that are known and reviewed. Each entry pins the FILE,
# the one FUNCTION, the module and the exact statement; anything else, in the
# same file or elsewhere, fails the gate. The reason is the review: a new
# entry needs a test that pins the behaviour it justifies.
ALLOWED = {
    ("revien/cli.py", "_probe_daemon", "http.client", "import http.client"):
        "`revien sync reset` asks a LOOPBACK daemon for /v1/health; "
        "_probe_daemon refuses any non-loopback host before connecting "
        "(pinned by tests/test_forget_followups.py::"
        "TestResetSaysOnlyWhatItCanProve::test_probe_egress_is_loopback_only)",
}
PROBE_TEST = ("tests/test_forget_followups.py::TestResetSaysOnlyWhatItCanProve"
              "::test_probe_egress_is_loopback_only")


def violations_in(rel_path: str, head_src: str, base_src: str):
    """New network uses in this file that no ALLOWED entry pins."""
    hits = []
    for func, module, stmt in new_network_uses(head_src, base_src):
        # `http.client.HTTPConnection(...)` inside the pinned function is the
        # use of the pinned import, not a second import
        if (rel_path, func, module, stmt) in ALLOWED:
            continue
        if any(k[0] == rel_path and k[1] == func and k[2] == module
               and stmt.startswith(module) for k in ALLOWED):
            continue
        hits.append((rel_path, func, module, stmt))
    return hits


# Gate self-tests: each snippet must (not) be reported. E1..E5 are the
# shapes a one-line allowlist keyed by (file, module) used to miss.
CONTROLS = (
    ("E1 import http.client", "import http.client\n", {"http.client"}),
    ("E2 multi-name import",
     "import os, socket\n", {"socket"}),
    ("E3 parent package then attribute",
     "import http\ndef f():\n    return http.client.HTTPConnection('x')\n",
     {"http.client"}),
    ("E3b aliased parent package",
     "import http as h\ndef f():\n    return h.client.HTTPConnection('x')\n",
     {"http.client"}),
    ("E4 http.client in another function",
     "def probe():\n    import http.client\n\ndef other():\n"
     "    import http.client\n", {"http.client"}),
    ("E5 importlib literal",
     "import importlib\nm = importlib.import_module('socket')\n", {"socket"}),
    ("E5b __import__ literal", "m = __import__('socket')\n", {"socket"}),
    ("E5c importlib non-literal",
     "import importlib\ndef f(n):\n    return importlib.import_module(n)\n",
     {DYNAMIC}),
    ("E5d from importlib import import_module",
     "from importlib import import_module as im\nim('x' + 'y')\n", {DYNAMIC}),
    ("from x import y", "from http import client\n", {"http.client"}),
    ("from urllib import request", "from urllib import request\n",
     {"urllib.request"}),
    ("plain httpx", "import httpx\n", {"httpx"}),
    ("negative: os/json", "import os, json\nfrom pathlib import Path\n", set()),
    ("negative: urllib.parse", "import urllib.parse\n", set()),
    ("negative: importlib.util", "import importlib.util\n", set()),
    ("negative: literal safe dynamic import",
     "import importlib\nimportlib.import_module('json')\n", set()),
)


def _controls():
    for name, src, want in CONTROLS:
        got = {m for _f, m, _s in network_uses(src)}
        if got != want:
            fail(f"positive control {name!r}: got {sorted(got)}, want {sorted(want)}")
    # the allowlist pins the FUNCTION: the same import in another function of
    # the same file, or at module level, is a violation
    probe = ("revien/cli.py", "_probe_daemon", "http.client", "import http.client")
    ok = "def _probe_daemon():\n    import http.client\n    http.client.HTTPConnection('h')\n"
    if violations_in("revien/cli.py", ok, ""):
        fail("control: the pinned probe import was flagged")
    for name, src in (
            ("E4 other function", ok + "def other():\n    import http.client\n"),
            ("module level", "import http.client\n" + ok),
            ("E3 attribute elsewhere",
             ok + "import http\ndef other():\n    return http.client.HTTPConnection('x')\n"),
            ("socket inside the probe",
             "def _probe_daemon():\n    import http.client\n    import os, socket\n"),
            ("another file", ok)):
        path = "revien/other.py" if name == "another file" else "revien/cli.py"
        if not violations_in(path, src, ""):
            fail(f"control {name!r}: not reported as a violation (probe key {probe})")


def main():
    # ── self-tests FIRST, always run regardless of whether there is a diff
    # from main -- they prove the detector itself is not a no-op and that
    # the allowlist pins a function, not a file.
    _controls()

    # ── the allowlisted probe is covered by a test that must pass: it fails
    # if the loopback-only guard is removed from _probe_daemon ──
    probe = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--basetemp={tempfile.mkdtemp(prefix='revien-gate8p-')}", PROBE_TEST],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    if probe.returncode != 0 or "1 passed" not in probe.stdout:
        fail(f"loopback probe test did not pass (exit {probe.returncode}):\n"
             f"{probe.stdout[-2000:]}")

    # ── pytest -k "egress or sovereign" ──
    basetemp = tempfile.mkdtemp(prefix="revien-gate8-")
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            f"--basetemp={basetemp}", "-k", "egress or sovereign",
        ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        fail(
            "pytest -k 'egress or sovereign' failed "
            f"(exit {proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}"
        )
    summary_line = None
    for line in proc.stdout.splitlines():
        if re.search(r"\d+ passed", line):
            summary_line = line
    if summary_line is None:
        fail(f"pytest output has no '<N> passed' summary line:\n{proc.stdout[-2000:]}")
    m = re.search(r"(\d+) passed", summary_line)
    if not m or int(m.group(1)) <= 0:
        fail(f"pytest summary shows 0 (or unparseable) passed: {summary_line!r}")

    # ── scan every changed revien/**/*.py vs main, by AST ──
    diff_names = subprocess.run(
        ["git", "diff", "--name-only", "main", "--", "revien/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if diff_names.returncode != 0:
        fail(f"git diff --name-only main failed: {diff_names.stderr}")
    changed_files = [
        f for f in diff_names.stdout.splitlines() if f.strip().endswith(".py")
    ]

    violations = []
    for rel_path in changed_files:
        head_path = REPO_ROOT / rel_path
        if not head_path.exists():
            continue  # deleted in this branch
        head_src = head_path.read_text(encoding="utf-8")
        base = subprocess.run(
            ["git", "show", f"main:{rel_path}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        base_src = base.stdout if base.returncode == 0 else ""  # new file
        try:
            violations.extend(violations_in(rel_path, head_src, base_src))
        except SyntaxError as e:
            fail(f"{rel_path} does not parse: {e}")

    # An allowlist entry must still match the source it pins (a stale entry
    # would silently stop covering anything).
    for (rel_path, func, module, stmt), _why in ALLOWED.items():
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if (func, module, stmt) not in network_uses(src):
            fail(f"allowlist entry {rel_path}:{func}:{stmt} no longer matches the source")

    if violations:
        fail("changed files add a network-client use on the default path "
             f"that no allowlist entry pins (file, function, module, statement): {violations}")

    print("egress verification passed")


if __name__ == "__main__":
    main()

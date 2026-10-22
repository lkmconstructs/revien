"""G8: sovereignty and egress tests pass, and no changed source file imports
a network client on the default path.

How it holds:
* The one place Revien opens a connection by itself is revien/loopback_probe.py
  (an operation that depends on sync-reset fencing asks a daemon on this
  machine for /v1/health). The gate pins the SHA-256 of that WHOLE FILE. Any
  edit to it, a decorator, a default argument, a nested function, a second
  definition, a lambda, fails the gate until the hash in PINNED is changed on
  purpose, in a reviewed commit that also keeps tests/test_loopback_probe.py
  passing.
* Nothing else may reach that module. Every revien/, revien_bench/ and
  scripts/ file (not only changed ones) is parsed; the one allowed use is
  `from revien.loopback_probe import probe_daemon` with the name used only as
  a call. Any other import form, any attribute of the module, any alias,
  argument or attribute of probe_daemon, and any string naming the module or
  the function is a violation. That closes the re-export route
  (`from revien.loopback_probe import http as _h; _h.client.HTTPSConnection`)
  and the monkeypatch routes (`probe_daemon.__globals__[...] = ...`,
  `setattr(probe_daemon, ...)`, `monkeypatch.setattr("revien.loopback_probe...")`).
* Every other changed revien/**/*.py is scanned by AST against the base
  (origin/main when it exists, else main); any network-client use that is new
  there fails.

The scanners are heuristics and cannot see everything. EVASIONS and
PROBE_EVASIONS list the shapes shown to evade them (each has a snippet the
self-test runs and requires to stay unreported, so the lists cannot go stale)
and OUT_OF_SCOPE lists what they do not look at. All are printed on success.

CHECK: python scripts/gates/check_egress.py
EXPECT: egress verification passed
"""
import ast
import hashlib
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
    "httpx", "requests", "urllib.request", "urllib3", "socket", "_socket",
    "ssl", "socketserver", "aiohttp", "http.client", "ftplib", "smtplib",
    "poplib", "imaplib", "telnetlib", "xmlrpc.client", "websockets",
    "websocket", "pycurl", "grpc", "paramiko", "boto3", "botocore",
    "anthropic", "openai", "subprocess", "ctypes",
)
# Functions of an otherwise harmless module that open a connection.
NETWORK_CALLS = (
    "asyncio.open_connection", "asyncio.open_unix_connection",
    "asyncio.start_server", "asyncio.start_unix_server",
    "os.system", "os.popen", "os.execv", "os.execvp", "os.execl",
    "os.spawnv", "os.spawnl", "os.posix_spawn",
)
# Names that run or fetch code from a string or an attribute name.
DYNAMIC_NAMES = ("__import__", "exec", "eval", "__builtins__")
# Modules whose getattr()/vars() reach the dynamic-import facility.
DYNAMIC_ROOTS = ("importlib", "builtins", "sys")
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
        self._call_funcs = set()   # Attribute nodes that are being called

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
            if full in NETWORK_CALLS:
                self._add(full.split(".")[0], node, stmt)
            if full in ("importlib.import_module", "importlib.__import__",
                        "pkgutil.resolve_name"):
                self._add(DYNAMIC, node, stmt)
        self.generic_visit(node)

    def visit_Assign(self, node):
        """`h = http`, `h = http.client`, `f = importlib.import_module`: the
        new name resolves to what it was assigned (simple names only)."""
        self.generic_visit(node)
        dotted = self._resolve(node.value)
        if dotted:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.alias[-1][t.id] = dotted

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in DYNAMIC_NAMES \
                and not any(node.id in sc for sc in self.alias):
            # `__import__('x')` is judged by its argument in visit_Call;
            # exec/eval/__builtins__ and a bare reference never are
            if node.id == "__import__" and id(node) in self._call_funcs:
                return
            self._add(DYNAMIC, node)

    def visit_Subscript(self, node):
        """`sys.modules['http.client']` and `sys.modules[name]`."""
        if self._resolve(node.value) == "sys.modules":
            key = node.slice
            lit = key.value if (isinstance(key, ast.Constant)
                                and isinstance(key.value, str)) else None
            if lit is None:
                self._add(DYNAMIC, node)
            elif _matches(lit):
                self._add(_matches(lit), node)
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
            if dotted in NETWORK_CALLS:
                self._add(dotted.split(".")[0], node)
                return
            if dotted in ("builtins.__dict__", "builtins.__import__") \
                    and id(node) not in self._call_funcs:
                # the import machinery by attribute: builtins.__dict__[...]
                self._add(DYNAMIC, node)
                return
            if (dotted in ("importlib.import_module", "importlib.__import__",
                           "pkgutil.resolve_name")
                    and id(node) not in self._call_funcs):
                # a bare reference (`f = importlib.import_module`): the call
                # that follows is invisible; a direct call is judged in
                # visit_Call by its argument
                self._add(DYNAMIC, node)
                return
            m = _matches(dotted)
            # only the parent-package route: a bare `socket.socket` is already
            # reported by its import statement
            if m and "." in m and dotted.split(".")[0] in _PARENTS:
                self._add(m, node)
                return
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        self._call_funcs.add(id(f))
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else "")
        if name in ("getattr", "vars", "setattr", "delattr") and node.args:
            root = self._resolve(node.args[0])
            if root and root.split(".")[0] in DYNAMIC_ROOTS:
                self._add(DYNAMIC, node)
        if name in ("get", "setdefault", "pop") and isinstance(f, ast.Attribute) \
                and self._resolve(f.value) == "sys.modules":
            arg = node.args[0] if node.args else None
            lit = (arg.value if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str) else None)
            if lit is None:
                self._add(DYNAMIC, node)
            elif _matches(lit):
                self._add(_matches(lit), node)
        resolved = self._resolve(f) if isinstance(f, ast.Attribute) else (
            next((sc[f.id] for sc in reversed(self.alias) if f.id in sc), f.id)
            if isinstance(f, ast.Name) else "")
        if (name in ("import_module", "__import__", "resolve_name")
                or resolved in ("importlib.import_module", "__import__",
                                "pkgutil.resolve_name")):
            arg = node.args[0] if node.args else None
            lit = (arg.value if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str) else None)
            if lit is not None:
                lit = lit.replace(":", ".")  # pkgutil "pkg.mod:attr"
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


# The reviewed place that may open a connection, pinned by the SHA-256 of the
# WHOLE FILE (CRLF normalized). A change to anything in it, however small or
# wherever in the file, fails the gate until this hash is changed on purpose.
PINNED = {
    "revien/loopback_probe.py": "7bf209326ed6b4ec03e76ac457f8afcc860ce27c415c446e7a371c70f2088d56",
}
PROBE_TEST = "tests/test_loopback_probe.py"
PROBE_MODULE = "revien.loopback_probe"
PROBE_ENTRY = "probe_daemon"
# Where the closure scan looks (the whole tree, not only changed files).
CLOSURE_ROOTS = ("revien", "revien_bench", "scripts")


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def pin_problem(rel_path: str, data: bytes):
    """None when `rel_path` is the pinned file with the pinned content, else
    why not."""
    want = PINNED.get(rel_path)
    if want is None:
        return f"{rel_path} is not a pinned file"
    got = file_digest(data)
    if got != want:
        return (f"{rel_path} changed (sha256 {got}, pinned {want}); review the "
                "whole file, keep tests/test_loopback_probe.py passing, then "
                "update PINNED in scripts/gates/check_egress.py")
    return None


def violations_in(rel_path: str, head_src: str, base_src: str):
    """New network uses in this file. A pinned file is checked by its hash
    (`pin_problem`), not by the scanner: its whole content is the review."""
    if rel_path in PINNED:
        return []
    return [(rel_path, func, module, stmt)
            for func, module, stmt in new_network_uses(head_src, base_src)]


# Attribute names that reach into a function's or module's internals.
_ESCAPES = ("__globals__", "__code__", "__closure__", "__wrapped__",
            "__dict__", "__defaults__", "__kwdefaults__", "__builtins__",
            "__module__", "__self__", "__func__")


def _docstring_ids(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


# The one file outside the probe that may NAME its path. scripts/
# forget_mutation_check.py edits the text of a temp copy of
# revien/loopback_probe.py to prove its tests catch the change; it never
# imports or runs the probe. Only a string constant exactly equal to the path
# is allowed; every other reach (import, attribute, a longer string) still fails.
PATH_NAMERS = {"scripts/forget_mutation_check.py": "revien/loopback_probe.py"}

# In that file the path is a file-text target, nothing more, so the ways to
# LOAD a file as code are closed there: these imports, attributes and names
# are reported (a loader given the path string would run the probe).
LOADER_MODULES = ("runpy", "importlib", "imp", "pkgutil", "zipimport")
LOADER_ATTRS = ("load_module", "exec_module", "run_path", "run_module",
                "spec_from_file_location", "SourceFileLoader",
                "SourcelessFileLoader", "ExtensionFileLoader", "import_module",
                "load_source", "load_compiled")
LOADER_NAMES = ("exec", "eval", "compile", "__import__", "execfile")


def _loader_reaches(src: str):
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in LOADER_MODULES:
                    out.append((node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in LOADER_MODULES:
                out.append((node.lineno, f"from {node.module} import ..."))
            for a in node.names:
                if a.name in LOADER_ATTRS:
                    out.append((node.lineno, f"imports {a.name}"))
        elif isinstance(node, ast.Attribute) and node.attr in LOADER_ATTRS:
            out.append((node.lineno, f"attribute {node.attr}"))
        elif isinstance(node, ast.Name) and node.id in LOADER_NAMES:
            out.append((node.lineno, f"name {node.id}"))
    return out


def probe_reaches(src: str, allow_path: str = ""):
    """[(line, why)] for every way `src` touches the pinned probe module other
    than `from revien.loopback_probe import probe_daemon` used as a call."""
    tree = ast.parse(src)
    docs = _docstring_ids(tree)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    out = []
    entry_names = set()

    def flag(node, why):
        out.append((getattr(node, "lineno", 0), why))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == PROBE_MODULE or a.name.startswith(PROBE_MODULE + "."):
                    flag(node, f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            last = PROBE_MODULE.split(".")[-1]
            if node.level == 0 and (mod == PROBE_MODULE
                                    or mod.startswith(PROBE_MODULE + ".")):
                ok = (mod == PROBE_MODULE and len(node.names) == 1
                      and node.names[0].name == PROBE_ENTRY
                      and node.names[0].asname is None)
                if ok:
                    entry_names.add(PROBE_ENTRY)
                else:
                    flag(node, f"from {mod} import {', '.join(names)}")
            elif node.level == 0 and mod == "revien" and last in names:
                flag(node, f"from revien import {last}")
            elif node.level > 0 and (mod == last or mod.endswith("." + last)
                                     or (last in names)):
                flag(node, f"relative import of {last}")
        elif isinstance(node, ast.Attribute) and node.attr == PROBE_MODULE.split(".")[-1]:
            flag(node, "attribute access to the module")
        elif isinstance(node, ast.Name) and node.id == PROBE_MODULE.split(".")[-1]:
            flag(node, "name of the module")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docs
              and not (allow_path and node.value == allow_path)
              and (PROBE_MODULE.split(".")[-1] in node.value
                   or node.value == PROBE_ENTRY)):
            flag(node, f"string naming the probe: {node.value[:40]!r}")
    # The entry point may only be CALLED. Passed, stored, aliased or given an
    # attribute (probe_daemon.__globals__) it is a way into the module.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in entry_names:
            par = parents.get(id(node))
            if isinstance(par, ast.ImportFrom):
                continue
            if isinstance(par, ast.Call) and par.func is node:
                continue
            flag(node, f"{node.id} used other than as a plain call")
    return sorted(set(out))


def closure_violations(root: Path):
    """Every [(file, line, why)] under CLOSURE_ROOTS that reaches the probe
    other than by its one allowed use. The pinned file itself and this gate
    (which names the module in order to check for it) are skipped."""
    skip = {root / rel for rel in PINNED} | {Path(__file__).resolve()}
    found = []
    for top in CLOSURE_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.resolve() in skip or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                src = path.read_text(encoding="utf-8")
                hits = probe_reaches(src, PATH_NAMERS.get(rel, ""))
                if rel in PATH_NAMERS:
                    hits = sorted(set(hits) | {
                        (ln, f"{why} (a file that names the probe path may not load code)")
                        for ln, why in _loader_reaches(src)})
            except (SyntaxError, UnicodeDecodeError) as e:
                fail(f"{rel} could not be parsed for the probe scan: {e}")
            found.extend((rel, line, why) for line, why in hits)
    return found


# Gate self-tests: each snippet must (not) be reported.
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
    ("E3c alias made by assignment",
     "import http\nh = http\ndef f():\n    return h.client.HTTPConnection('x')\n",
     {"http.client"}),
    ("E3d alias chain by assignment",
     "import http\na = http\nb = a\nb.client.HTTPConnection('x')\n",
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
    ("E5e getattr on importlib",
     "import importlib\ndef f():\n    return getattr(importlib, 'import_module')('socket')\n",
     {DYNAMIC}),
    ("E5f getattr on builtins",
     "import builtins\ndef f():\n    return getattr(builtins, '__imp' + 'ort__')('socket')\n",
     {DYNAMIC}),
    ("E5g exec", "def f():\n    exec('import socket')\n", {DYNAMIC}),
    ("E5h eval", "def f():\n    return eval('__import__(\"socket\")')\n",
     {DYNAMIC}),
    ("E5i reference to __import__ without a call",
     "imp = __import__\nimp('socket')\n", {DYNAMIC, "socket"}),
    ("E5j attribute reference to import_module",
     "import importlib\nf = importlib.import_module\nf('socket')\n",
     {DYNAMIC, "socket"}),
    ("E5k pkgutil.resolve_name literal",
     "import pkgutil\npkgutil.resolve_name('http.client:HTTPSConnection')\n",
     {"http.client"}),
    ("E5l pkgutil.resolve_name non-literal",
     "import pkgutil\ndef f(n):\n    return pkgutil.resolve_name(n)\n", {DYNAMIC}),
    ("E5m from pkgutil import resolve_name",
     "from pkgutil import resolve_name\nresolve_name('socket')\n",
     {DYNAMIC, "socket"}),
    ("E5n builtins.__dict__ subscript",
     "import builtins\nbuiltins.__dict__['__imp' + 'ort__']('socket')\n",
     {DYNAMIC}),
    ("E5o builtins.__import__ attribute",
     "import builtins\nbuiltins.__import__('socket')\n", {"socket"}),
    ("E6 sys.modules literal",
     "import sys\ndef f():\n    return sys.modules['http.client'].HTTPConnection('x')\n",
     {"http.client"}),
    ("E6b sys.modules non-literal",
     "import sys\ndef f(n):\n    return sys.modules[n]\n", {DYNAMIC}),
    ("E6c sys.modules.get", "import sys\nsys.modules.get('socket')\n",
     {"socket"}),
    ("E7 _socket", "import _socket\n", {"_socket"}),
    ("E7b ssl", "import ssl\ndef f():\n    return ssl.create_connection(('x', 443))\n",
     {"ssl"}),
    ("E7c asyncio.open_connection",
     "import asyncio\nasync def f():\n    return await asyncio.open_connection('x', 80)\n",
     {"asyncio"}),
    ("E7d from asyncio import open_connection",
     "from asyncio import open_connection\n", {"asyncio"}),
    ("E7e lambda at module level",
     "import http\nf = lambda: http.client.HTTPSConnection('x')\n",
     {"http.client"}),
    ("E7f subprocess", "import subprocess\nsubprocess.run(['curl', 'x'])\n",
     {"subprocess"}),
    ("E7g os.system", "import os\nos.system('curl x')\n", {"os"}),
    ("from x import y", "from http import client\n", {"http.client"}),
    ("from urllib import request", "from urllib import request\n",
     {"urllib.request"}),
    ("plain httpx", "import httpx\n", {"httpx"}),
    ("negative: os/json", "import os, json\nfrom pathlib import Path\n", set()),
    ("negative: urllib.parse", "import urllib.parse\n", set()),
    ("negative: importlib.util", "import importlib.util\n", set()),
    ("negative: literal safe dynamic import",
     "import importlib\nimportlib.import_module('json')\n", set()),
    ("negative: sys.modules of a safe module",
     "import sys\nsys.modules['json']\n", set()),
    ("negative: asyncio.sleep",
     "import asyncio\nasync def f():\n    await asyncio.sleep(1)\n", set()),
)

# What the scanner provably does NOT report. Each snippet is run by the
# self-test and must stay unreported: if the scanner ever learns one, the
# self-test fails until the entry is removed, so this list cannot go stale.
EVASIONS = (
    ("alias through a container, a call or a parameter",
     "import http\nm = [http][0]\ndef f():\n    return m.client.HTTPConnection('x')\n"),
    ("a method of an object whose type the scanner cannot see",
     "async def f(loop):\n    return await loop.create_connection(None, 'x', 80)\n"),
    ("importing by file location, not by name (importlib.util)",
     "import importlib.util\nspec = importlib.util.find_spec('http.client')\n"
     "mod = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(mod)\n"),
    ("an importlib.machinery loader (SourceFileLoader(...).load_module())",
     "import importlib.machinery\nm = importlib.machinery.SourceFileLoader("
     "'c', '/usr/lib/python3/http/client.py').load_module()\n"),
    ("a module's __loader__ used to load another",
     "import json\nm = json.__loader__.load_module('http.client')\n"),
    ("operator.attrgetter to reach import_module",
     "import importlib, operator\n"
     "operator.attrgetter('import_module')(importlib)('socket')\n"),
    ("running a module through runpy",
     "import runpy\nrunpy.run_module('http.client')\n"),
    ("a network client the NETWORK_MODULES list does not name",
     "import pymysql\n"),
    ("a module that is not imported where it is used, reached through a "
     "variable set elsewhere",
     "def f(client):\n    return client.get('https://x.example')\n"),
)
# Self-tests for the probe-reach scan: (name, source, must it be reported).
PROBE_CONTROLS = (
    ("allowed: the one import used as a call",
     "def f(u):\n    from revien.loopback_probe import probe_daemon\n"
     "    return probe_daemon(u)\n", False),
    ("allowed: module-level import, call",
     "from revien.loopback_probe import probe_daemon\nx = probe_daemon('u')\n",
     False),
    ("allowed: docstring that names the module",
     '"""Uses revien/loopback_probe.py via probe_daemon."""\nx = 1\n', False),
    ("re-export: the shape a verifier used to pass the old gate",
     "from revien.loopback_probe import http as _h\n"
     "_h.client.HTTPSConnection('x')\n", True),
    ("re-export: socket", "from revien.loopback_probe import socket\n", True),
    ("re-export: star", "from revien.loopback_probe import *\n", True),
    ("re-export: two names",
     "from revien.loopback_probe import probe_daemon, json\n", True),
    ("re-export: aliased entry",
     "from revien.loopback_probe import probe_daemon as p\n", True),
    ("module import", "import revien.loopback_probe\n", True),
    ("module import, aliased",
     "import revien.loopback_probe as lp\nlp.http\n", True),
    ("from-package import", "from revien import loopback_probe\n", True),
    ("relative import of the module", "from . import loopback_probe\n", True),
    ("relative import from the module",
     "from .loopback_probe import probe_daemon\n", True),
    ("importlib by literal",
     "import importlib\nimportlib.import_module('revien.loopback_probe')\n", True),
    ("sys.modules by literal",
     "import sys\nsys.modules['revien.loopback_probe']\n", True),
    ("attribute chain from the package",
     "import revien\nrevien.loopback_probe.http\n", True),
    ("monkeypatch: __globals__ of the entry",
     "from revien.loopback_probe import probe_daemon\n"
     "probe_daemon.__globals__['socket'] = object()\n", True),
    ("monkeypatch: __code__ of the entry",
     "from revien.loopback_probe import probe_daemon\n"
     "probe_daemon.__code__ = (lambda u: None).__code__\n", True),
    ("monkeypatch: setattr on the entry",
     "from revien.loopback_probe import probe_daemon\n"
     "setattr(probe_daemon, '__code__', None)\n", True),
    ("monkeypatch: entry passed on",
     "from revien.loopback_probe import probe_daemon\n"
     "patch(probe_daemon)\n", True),
    ("monkeypatch: alias of the entry",
     "from revien.loopback_probe import probe_daemon\n"
     "p = probe_daemon\np('u')\n", True),
    ("monkeypatch: replace the module attribute",
     "import revien.loopback_probe as m\nm.probe_daemon = lambda u: None\n", True),
    ("monkeypatch: dotted path in a string",
     "monkeypatch.setattr('revien.loopback_probe.probe_daemon', f)\n", True),
    ("monkeypatch: patch a module global by string",
     "mock.patch('revien.loopback_probe.socket')\n", True),
    ("monkeypatch: the function name as a string",
     "getattr(mod, 'probe_daemon')\n", True),
)
# Shapes the probe-reach scan is shown NOT to catch. Run by the self-test and
# required to stay unreported, so this cannot go stale.
PROBE_EVASIONS = (
    ("a module name assembled at run time",
     "import revien\ngetattr(revien, 'loopback_' + 'probe')\n"),
    ("the module reached through a name the scanner does not follow",
     "import sys\nm = sys.modules[__name__.split('.')[0] + '.loop' + 'back_probe']\n"),
    ("a path assembled at run time (the mutation script once did this)",
     "p = 'revien/loopback' + '_probe.py'\n"),
    ("the probe file run as a script by a subprocess (not a loader)",
     "import subprocess\nsubprocess.run(['python', 'revien/loopback' + '_probe.py'])\n"),
)
# What the gate does not look at at all.
OUT_OF_SCOPE = (
    "non-Python files (a .pth file, a shell script, a build hook) and compiled "
    "extension modules",
    "code inside third-party packages the listed clients pull in",
    "a network client used in a file unchanged against the base ref (that scan "
    "covers the diff, not the tree), and a checkout with neither origin/main "
    "nor main to diff against (the probe-reach scan still covers the whole tree)",
    "tests/: they may import or patch the probe; the shipped package may not",
    "a runtime effect of data (a URL a config file supplies to an already "
    "allowed client): the gate looks for clients, not for destinations",
)


def _base_ref() -> str:
    """The ref the scan diffs against: REVIEN_GATE_BASE, else origin/main when
    it exists (a local `main` can lag it), else main."""
    for ref in (os.environ.get("REVIEN_GATE_BASE", ""), "origin/main", "main"):
        if ref and subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", ref],
                cwd=str(REPO_ROOT), capture_output=True).returncode == 0:
            return ref
    fail("no base ref to diff against (tried REVIEN_GATE_BASE, origin/main, main)")


def _controls():
    for name, src, want in CONTROLS:
        got = {m for _f, m, _s in network_uses(src)}
        if got != want:
            fail(f"positive control {name!r}: got {sorted(got)}, want {sorted(want)}")
    for name, src in EVASIONS:
        if network_uses(src):
            fail(f"EVASIONS entry {name!r} is now reported: remove it from "
                 "EVASIONS (and from the documented limits)")
    # a pinned file is checked by hash: any byte of difference is reported
    real = (REPO_ROOT / "revien" / "loopback_probe.py").read_bytes()
    if pin_problem("revien/loopback_probe.py", real):
        fail(pin_problem("revien/loopback_probe.py", real))
    for name, data in (
            ("added decorator", real.replace(b"def probe_daemon(", b"@(lambda f: f)\ndef probe_daemon(")),
            ("default argument", real.replace(b"probe_daemon(url)", b"probe_daemon(url, c=http.client.HTTPSConnection('x'))")),
            ("second def", real + b"\ndef probe_daemon(url):\n    return None\n"),
            ("lambda", real + b"\nleak = lambda: http.client.HTTPSConnection('x')\n"),
            ("extra import", real.replace(b"import json", b"import json, socket"))):
        if data == real:
            fail(f"pin control {name!r} did not change the file")
        if not pin_problem("revien/loopback_probe.py", data):
            fail(f"pin control {name!r}: a modified probe was accepted")
    if pin_problem("revien/cli.py", b"x") is None:
        fail("an unpinned file was treated as pinned")
    # the probe-reach scan: every control must come out as stated
    for name, src, want in PROBE_CONTROLS:
        got = bool(probe_reaches(src))
        if got != want:
            fail(f"probe control {name!r}: reported={got}, want {want}")
    for name, src in PROBE_EVASIONS:
        if probe_reaches(src):
            fail(f"PROBE_EVASIONS entry {name!r} is now reported: remove it "
                 "from PROBE_EVASIONS")
    # the pinned module offers nothing to re-export: its only public name is
    # the entry point, and it holds no network module at module level
    import importlib
    mod = importlib.import_module(PROBE_MODULE)
    public = sorted(n for n in vars(mod) if not n.startswith("__"))
    if public != [PROBE_ENTRY]:
        fail(f"{PROBE_MODULE} exposes {public}, want only [{PROBE_ENTRY!r}]")
    # a network use anywhere else is a violation, wherever it is
    for name, src in (("cli.py", "import http.client\n"),
                      ("other function", "def _probe_daemon():\n    import http.client\n")):
        if not violations_in("revien/cli.py", src, ""):
            fail(f"control {name!r}: not reported as a violation")


def main():
    # ── self-tests FIRST, always run regardless of whether there is a diff
    # from main -- they prove the detector itself is not a no-op and that
    # a change to the pinned probe file is reported.
    _controls()

    # ── the pinned probe is covered by a test that must pass: it fails
    # if the loopback-only guard is removed from it ──
    probe = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--basetemp={tempfile.mkdtemp(prefix='revien-gate8p-')}", PROBE_TEST],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    if probe.returncode != 0 or " passed" not in probe.stdout:
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

    # ── nothing but the one allowed call reaches the probe, anywhere ──
    reach = closure_violations(REPO_ROOT)
    if reach:
        fail("code reaches revien.loopback_probe other than by "
             "`from revien.loopback_probe import probe_daemon` used as a "
             f"call (file, line, how): {reach}")

    # ── scan every changed revien/**/*.py vs the base, by AST ──
    base_ref = _base_ref()
    diff_names = subprocess.run(
        ["git", "diff", "--name-only", base_ref, "--", "revien/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    if diff_names.returncode != 0:
        fail(f"git diff --name-only {base_ref} failed: {diff_names.stderr}")
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
            ["git", "show", f"{base_ref}:{rel_path}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
        )
        base_src = base.stdout if base.returncode == 0 else ""  # new file
        try:
            violations.extend(violations_in(rel_path, head_src, base_src))
        except SyntaxError as e:
            fail(f"{rel_path} does not parse: {e}")

    # Every pinned file is checked by hash on every run, diff or no diff.
    for rel_path in PINNED:
        path = REPO_ROOT / rel_path
        if not path.exists():
            fail(f"pinned file {rel_path} is missing")
        problem = pin_problem(rel_path, path.read_bytes())
        if problem:
            fail(problem)

    if violations:
        fail("changed files add a network-client use on the default path "
             f"(file, function, module, statement): {violations}")

    print("egress verification passed")
    print("limits, shapes the scanner does not report:")
    for name, _src in EVASIONS:
        print(f"  - {name}")
    for name, _src in PROBE_EVASIONS:
        print(f"  - (probe reach) {name}")
    print("limits, not examined by the gate:")
    for name in OUT_OF_SCOPE:
        print(f"  - {name}")


if __name__ == "__main__":
    main()

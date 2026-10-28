"""G21: the default path makes no request, native code included: ONNX
Runtime's telemetry is off before onnxruntime is first imported.

G8 (check_egress.py) hooks Python sockets and scans Python source, so a
compiled library that opens its own connection slips past it. ONNX Runtime
1.29.0 and later (first released 2026-08-17) on Linux and macOS is one: its
1DS SDK posts telemetry to Microsoft from C++ (libcurl), from the import alone
(the first upload about 9 s after `import onnxruntime`, no session needed),
unless ORT_DISABLE_TELEMETRY is set when onnxruntime is first imported. This
gate watches the process from outside instead: each scenario
runs in a fresh interpreter with a throwaway HOME and with HTTPS_PROXY,
HTTP_PROXY, https_proxy and http_proxy pointed at a logging listener on
127.0.0.1 (NO_PROXY empty), so a request libcurl makes arrives there as a
CONNECT line and goes no further. ORT_DISABLE_TELEMETRY is removed from every
scenario's environment, so only what the scenario itself does can set it.

(0) proxy self-check, independent of onnxruntime: a child makes a plain HTTPS
    request with Python's urllib, and with the curl command (libcurl) when one
    is installed, under the same proxy environment. Each must reach the
    listener, else the gate fails: a listener the proxy variables do not lead
    to would see nothing and pass everything.
(a) negative control: `import onnxruntime` and an InferenceSession on its
    bundled datasets/mul_1.onnx, no switch, alive 20 s. When the installed
    onnxruntime carries the 1DS code (its collector host, "OneCollector" or
    "ORT_DISABLE_TELEMETRY" is in a native library) this must show at least
    1 CONNECT, which proves the harness sees native egress.
    When it does not, the control is reported "not applicable for onnxruntime
    X" and skipped, not failed.
(b) `import revien` first, then the same session, alive 20 s: 0 requests, no
    $HOME/.cache/Microsoft, and revien reports the telemetry off.
(c) Revien's real default path: `revien ingest` and `revien recall` (the CLI's
    own main(), semantic layer and reranker at their defaults) in one process
    that stays alive 20 s after the recall: 0 requests and no
    $HOME/.cache/Microsoft. The
    scenario must show that an ONNX Runtime session really ran (the semantic
    layer is active and finds the ingested text), else it fails. Skipped,
    with the reason, only when the embedding model is not cached and cannot be
    fetched (an offline run with a cold cache); the models are checked, and
    fetched when missing, by a separate step with the switch set.
(d) honest report: onnxruntime imported BEFORE revien, no switch: revien
    must report the telemetry NOT off and warn once with the fix. The child
    exits at once, before the first upload is due, and runs under the same
    proxy, so anything ORT tries goes only to the listener; its requests are
    not judged.

Limits, printed on success: the listener sees only clients that honour the
proxy variables (libcurl, Python's urllib/requests/httpx do; a client that
ignores them, or that resolves DNS or opens raw sockets, is not seen), and
only during the time each scenario runs.

CHECK: python scripts/gates/check_native_egress.py
EXPECT: native egress verification passed
"""
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_VAR = "ORT_DISABLE_TELEMETRY"
# Strings the 1DS telemetry code compiles into the native library: the
# collector host, its endpoint name, and the variable that switches it off.
# Any one of them marks a build that carries it (1.29.0 and later; 1.22 to
# 1.28 carry none of them).
ONEDS_MARKERS = (b"mobile.events.data.microsoft.com", b"OneCollector",
                 b"ORT_DISABLE_TELEMETRY")
# The 1DS SDK uploads about 9 s after the import, then again ~5 s later.
SESSION_SECONDS = 20   # (a), (b): how long the process stays up after a session
ALIVE_SECONDS = 20     # (c): how long the process stays up after recall
SELFCHECK_HOST = "native-egress-selfcheck.invalid"


def fail(msg):
    print(f"check_native_egress: {msg}", file=sys.stderr)
    sys.exit(1)


class Listener:
    """A logging fake proxy on 127.0.0.1: records the first line of every
    request (`CONNECT host:443 HTTP/1.1` for HTTPS), answers nothing, closes."""

    def __init__(self):
        self.lines = []
        self._lock = threading.Lock()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(50)
        self.port = self._sock.getsockname()[1]
        self._stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True).start()

    def _one(self, conn):
        try:
            conn.settimeout(5)
            data = conn.recv(2048).split(b"\r\n")[0]
            with self._lock:
                self.lines.append(data.decode("ascii", "replace") or "<empty>")
        except OSError:
            with self._lock:
                self.lines.append("<connection, no request line>")
        finally:
            conn.close()

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def connects(self):
        return [line for line in self.lines if line.startswith("CONNECT ")]


def _child_env(home: Path, port=None, extra=None) -> dict:
    # Revien's defaults, whatever the caller's shell has set: no REVIEN_*
    # variable (semantic layer and reranker on, local embedder and extractor)
    env = {k: v for k, v in os.environ.items()
           if k != ENV_VAR and not k.startswith("REVIEN_")}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.update(HOME=str(home), USERPROFILE=str(home),
               XDG_CACHE_HOME=str(home / ".cache"),
               REVIEN_HOME=str(home / ".revien"), PYTHONIOENCODING="utf-8")
    if port is not None:
        proxy = f"http://127.0.0.1:{port}"
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env[k] = proxy
        env["NO_PROXY"] = env["no_proxy"] = ""
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


SESSION = (
    "import os, time\n"
    "import onnxruntime as o\n"
    "p = os.path.join(os.path.dirname(o.__file__), 'datasets', 'mul_1.onnx')\n"
    "s = o.InferenceSession(p, providers=['CPUExecutionProvider'])\n"
    "print('session ok', flush=True)\n"
    f"time.sleep({SESSION_SECONDS})\n"
)

SCENARIO_A = SESSION
SCENARIO_B = (
    "import revien\n"
    "from revien.semantic._ort import ort_telemetry_state\n"
    + SESSION +
    "print('STATE', ort_telemetry_state(), flush=True)\n"
)
SCENARIO_C = (
    "import json, sys, time\n"
    "db = sys.argv[1]\n"
    "from revien.cli import main\n"
    "texts = [\n"
    "    'The database migration finished on Tuesday after the index rebuild.',\n"
    "    'The migration plan moved the orders table to the new cluster.',\n"
    "    'Lunch on Friday is at the noodle place near the office.',\n"
    "]\n"
    "for args in ([['ingest', t, '--db', db] for t in texts]\n"
    "             + [['recall', 'when did the migration finish', '--db', db]]):\n"
    "    try:\n"
    "        main(args, standalone_mode=False)\n"
    "    except SystemExit as e:\n"
    "        if e.code not in (0, None):\n"
    "            raise\n"
    "from revien.graph.store import GraphStore\n"
    "from revien.semantic.index import SemanticIndex\n"
    "from revien.semantic._ort import ort_telemetry_state\n"
    "si = SemanticIndex(GraphStore(db_path=db))\n"
    "hits = len(si.search('migration finished', 3)) if si.is_enabled else 0\n"
    "print('RESULT ' + json.dumps({\n"
    "    'onnxruntime_loaded': 'onnxruntime' in sys.modules,\n"
    "    'semantic_enabled': si.is_enabled,\n"
    "    'semantic_hits': hits,\n"
    "    'ort_state': ort_telemetry_state(),\n"
    "}), flush=True)\n"
    f"time.sleep({ALIVE_SECONDS})\n"
)
SCENARIO_D = (
    "import onnxruntime\n"
    "import revien\n"
    "from revien.semantic._ort import ort_telemetry_state as s\n"
    "print('STATE', s(), s(), flush=True)\n"
)
SELFCHECK = (
    "import shutil, subprocess, urllib.request\n"
    "try:\n"
    f"    urllib.request.urlopen('https://{SELFCHECK_HOST}/', timeout=5)\n"
    "except Exception:\n"
    "    pass\n"
    "curl = shutil.which('curl')\n"
    "if curl:\n"
    f"    subprocess.run([curl, '-s', '-m', '5', 'https://curl.{SELFCHECK_HOST}/'],\n"
    "                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "print('CURL', bool(curl), flush=True)\n"
)
PREFETCH = (
    "import revien\n"
    "from revien.semantic.index import FastEmbedProvider\n"
    "from revien.semantic.rerank import CrossEncoderReranker\n"
    "FastEmbedProvider()._ensure_model()\n"
    "CrossEncoderReranker()._ensure_model()\n"
    "print('MODELS OK', flush=True)\n"
)


class Run:
    def __init__(self, name, code, args=(), proxied=True, extra=None):
        self.name = name
        self.home = Path(tempfile.mkdtemp(prefix=f"revien-gate21{name}-"))
        self.listener = Listener() if proxied else None
        env = _child_env(self.home, self.listener.port if proxied else None, extra)
        self.proc = subprocess.Popen(
            [sys.executable, "-c", code, *args], cwd=str(self.home), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace")
        self.out = self.err = ""

    def wait(self, timeout=600):
        try:
            self.out, self.err = self.proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.out, self.err = self.proc.communicate()
            fail(f"({self.name}) timed out after {timeout}s:\n{self.err[-2000:]}")
        time.sleep(0.5)  # let the listener log a request that raced the exit
        if self.listener:
            self.listener.close()
        return self

    @property
    def microsoft_dir(self) -> bool:
        return (self.home / ".cache" / "Microsoft").exists()

    def requests(self):
        return list(self.listener.lines) if self.listener else []

    def cleanup(self):
        shutil.rmtree(self.home, ignore_errors=True)


def _last_line(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1][:300] if lines else "nothing"


def _ort_version() -> str:
    try:
        from importlib import metadata
        for dist in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-silicon",
                     "onnxruntime-openvino", "onnxruntime-directml"):
            try:
                return metadata.version(dist)
            except metadata.PackageNotFoundError:
                continue
    except Exception:
        pass
    return "unknown"


def _oneds_marker(pkg_dir: Path):
    """The first ONEDS_MARKERS string found in a native library of the
    installed onnxruntime, else None (read from disk; onnxruntime is never
    imported here). Every shared library is read, and any other file over
    1 MB that is not source or data, so an odd library name is not missed."""
    native = (".so", ".pyd", ".dll", ".dylib")
    skip = (".py", ".pyi", ".pyc", ".onnx", ".json", ".txt", ".md")
    for path in sorted(pkg_dir.rglob("*")):
        try:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if not (suffix in native or ".so." in path.name
                    or (suffix not in skip and path.stat().st_size > (1 << 20))):
                continue
            with open(path, "rb") as f:
                prev = b""
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    window = prev[-64:] + chunk
                    for marker in ONEDS_MARKERS:
                        if marker in window:
                            return marker.decode()
                    prev = chunk
        except OSError:
            continue
    return None


def main():
    report = []
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        print("native egress verification passed")
        print("  onnxruntime is not installed: no native telemetry on the "
              "default path; (a) (b) (c) (d) not applicable")
        return
    pkg_dir = Path(list(spec.submodule_search_locations)[0])
    version = _ort_version()
    oneds = _oneds_marker(pkg_dir)
    if not (pkg_dir / "datasets" / "mul_1.onnx").exists():
        fail(f"onnxruntime {version} has no datasets/mul_1.onnx to build a session on")

    # (0) the proxy path itself, before anything is judged by its silence
    report.append(_proxy_selfcheck())

    # Models for (c): checked (and fetched when missing) with the switch set,
    # outside the watched scenarios. HF_HUB_OFFLINE first: a warm cache passes
    # without touching the network.
    pre = Run("p", PREFETCH, proxied=False,
              extra={ENV_VAR: "1", "HF_HUB_OFFLINE": "1"}).wait()
    models = "cached"
    if pre.proc.returncode != 0 or "MODELS OK" not in pre.out:
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
        why = _last_line(pre.err)
        pre.cleanup()
        pre = None if offline else Run("p", PREFETCH, proxied=False,
                                       extra={ENV_VAR: "1"}).wait()
        if pre is not None and pre.proc.returncode == 0 and "MODELS OK" in pre.out:
            models = "fetched"
        else:
            models = None
            if pre is not None:
                why = _last_line(pre.err)
    if pre is not None:
        pre.cleanup()

    runs = {
        "a": Run("a", SCENARIO_A),
        "b": Run("b", SCENARIO_B),
        "d": Run("d", SCENARIO_D),
    }
    if models:
        db = Path(tempfile.mkdtemp(prefix="revien-gate21db-")) / "c.db"
        # HF_HUB_OFFLINE is not the default: (c) runs without it, on the
        # warm cache the step above guaranteed
        runs["c"] = Run("c", SCENARIO_C, args=(str(db),),
                        extra={"HF_HUB_OFFLINE": None})
    for r in runs.values():
        r.wait()

    problems = []

    def check(name, fn):
        """Run one scenario's checks; every scenario is reported, pass or fail."""
        try:
            report.append(fn())
        except ScenarioFailed as e:
            report.append(f"{name} FAILED: {e}")
            problems.append(f"{name} {e}")

    try:
        check("(a)", lambda: _check_a(runs["a"], version, oneds))
        check("(b)", lambda: _check_b(runs["b"]))
        if "c" in runs:
            check("(c)", lambda: _check_c(runs["c"], models))
        else:
            report.append("(c) SKIPPED: the embedding model is not cached and "
                          "this run is offline (HF_HUB_OFFLINE set, or the "
                          "fetch failed); warm the model cache and rerun "
                          f"(model load said: {why})")
        check("(d)", lambda: _check_d(runs["d"]))
    finally:
        for r in runs.values():
            r.cleanup()
        if "c" in runs:
            shutil.rmtree(db.parent, ignore_errors=True)

    if problems:
        for line in report:
            print(f"  {line}", file=sys.stderr)
        fail("; ".join(problems))
    print("native egress verification passed")
    for line in report:
        print(f"  {line}")
    print("limits, not examined by the gate:")
    print("  - a client that ignores HTTPS_PROXY/HTTP_PROXY (raw sockets, DNS, "
          "a hard-coded proxy setting) is not seen by the listener")
    print("  - only what happens while each scenario runs (it stays up "
          f"{SESSION_SECONDS} s after the session, {ALIVE_SECONDS} s after the "
          "recall); a timer that fires later is not seen")
    print("  - events another onnxruntime process queued earlier in the real "
          "~/.cache/Microsoft/DeveloperTools/.onnxruntime/ (every scenario uses "
          "a throwaway HOME)")


class ScenarioFailed(Exception):
    pass


def _ran(r):
    if r.proc.returncode != 0:
        raise ScenarioFailed(f"exited {r.proc.returncode}: {r.err[-1500:]}")


def _proxy_selfcheck() -> str:
    """(0): a plain HTTPS request under the scenarios' proxy environment must
    reach the listener, by Python's urllib and, when installed, by curl."""
    r = Run("0", SELFCHECK).wait(timeout=120)
    if r.proc.returncode != 0:
        fail(f"(0) proxy self-check exited {r.proc.returncode}: {r.err[-1500:]}")
    want = [f"CONNECT {SELFCHECK_HOST}:443"]
    if "CURL True" in r.out:
        want.append(f"CONNECT curl.{SELFCHECK_HOST}:443")
    seen = r.requests()
    r.cleanup()
    missing = [w for w in want if not any(line.startswith(w) for line in seen)]
    if missing:
        fail(f"(0) proxy self-check: {missing} never reached the listener "
             f"(seen: {seen}), so a silent scenario would prove nothing")
    return ("(0) proxy self-check: a plain HTTPS request reached the listener "
            f"via {'urllib and curl' if len(want) == 2 else 'urllib (no curl installed)'}")


def _check_a(a, version, oneds):
    """Negative control: without the switch the harness must see the CONNECT."""
    _ran(a)
    if not oneds:
        return (f"(a) negative control: not applicable for onnxruntime {version} "
                "(none of the 1DS markers is in its native libraries); skipped")
    if not a.listener.connects:
        raise ScenarioFailed(
            f"onnxruntime {version} carries the 1DS telemetry code but a bare "
            f"session showed no CONNECT in {SESSION_SECONDS}s, so this harness "
            f"cannot see native egress here (requests seen: {a.requests()})")
    return (f"(a) negative control, onnxruntime {version} (carries "
            f"{oneds!r}) without the switch: {len(a.listener.connects)} CONNECT "
            f"({sorted(set(a.listener.connects))}), $HOME/.cache/Microsoft "
            f"created: {a.microsoft_dir}")


def _quiet(r, what):
    """0 requests through the proxy and no $HOME/.cache/Microsoft."""
    if r.requests():
        raise ScenarioFailed(f"{what}: {len(r.requests())} request(s) through "
                             f"the proxy: {r.requests()}")
    if r.microsoft_dir:
        raise ScenarioFailed(f"{what}: created $HOME/.cache/Microsoft")


def _check_b(b):
    _ran(b)
    _quiet(b, "`import revien` then a session")
    if "STATE True" not in b.out:
        raise ScenarioFailed(f"revien did not report the telemetry off: "
                             f"{b.out.strip()!r}")
    return ("(b) `import revien` then a session: 0 requests, no "
            "$HOME/.cache/Microsoft, reported off")


def _check_c(c, models):
    _ran(c)
    line = next((l for l in c.out.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise ScenarioFailed(f"printed no RESULT line: {c.out[-1500:]}")
    res = json.loads(line[len("RESULT "):])
    if not (res["onnxruntime_loaded"] and res["semantic_enabled"]
            and res["semantic_hits"] > 0):
        raise ScenarioFailed(
            f"no ONNX Runtime session ran on the default path, so 0 requests "
            f"would prove nothing: {res} {c.err[-1500:]}")
    _quiet(c, f"revien ingest + recall, alive {ALIVE_SECONDS}s")
    if res["ort_state"] is not True:
        raise ScenarioFailed(f"revien did not report the telemetry off: {res}")
    return (f"(c) revien ingest + recall (models {models}), alive "
            f"{ALIVE_SECONDS}s: 0 requests, no $HOME/.cache/Microsoft, "
            "semantic layer ran, reported off")


def _check_d(d):
    """onnxruntime imported before revien, no switch: reported NOT off."""
    _ran(d)
    if "STATE False False" not in d.out:
        raise ScenarioFailed(
            "onnxruntime imported before revien without the switch was not "
            f"reported as NOT off: {d.out.strip()!r}")
    if d.err.count("ONNX Runtime telemetry is NOT off") != 1 \
            or f"{ENV_VAR}=1 before importing onnxruntime" not in d.err:
        raise ScenarioFailed(f"expected exactly one warning naming the fix: "
                             f"{d.err[-1500:]!r}")
    return ("(d) onnxruntime imported before revien: reported NOT off, one "
            "warning naming the fix")


if __name__ == "__main__":
    main()

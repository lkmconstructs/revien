"""G8: sovereignty and egress tests pass, and no changed source file imports
a network client on the default path.

CHECK: python scripts/gates/check_egress.py
EXPECT: egress verification passed
"""
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
_ALT = "|".join(re.escape(m) for m in NETWORK_MODULES)
_IMPORT_RE = re.compile(
    r"^\+\s*(?:import\s+(" + _ALT + r")\b"
    r"|from\s+(" + _ALT + r")\b"
    # `from http import client` / `from urllib import request`
    r"|from\s+(http)\s+import\s+client\b"
    r"|from\s+(urllib)\s+import\s+request\b)"
)

# Network-client imports that are known and reviewed, keyed by file and module.
# Anything else added to revien/ fails the gate. The reason is the review:
# a new entry needs a test that pins the behaviour it justifies.
ALLOWED = {
    ("revien/cli.py", "http.client"):
        "`revien sync reset` asks a LOOPBACK daemon for /v1/health; "
        "_probe_daemon refuses any non-loopback host before connecting "
        "(pinned by tests/test_forget_followups.py::"
        "TestResetSaysOnlyWhatItCanProve::test_probe_egress_is_loopback_only)",
}
PROBE_TEST = ("tests/test_forget_followups.py::TestResetSaysOnlyWhatItCanProve"
              "::test_probe_egress_is_loopback_only")


def added_network_import(diff_line: str):
    """Return the module name if this unified-diff ADDED line ('+...',
    never a '+++' file header) introduces a new import of a network client
    module; else None."""
    if diff_line.startswith("+++"):
        return None
    if not diff_line.startswith("+"):
        return None
    m = _IMPORT_RE.match(diff_line)
    if not m:
        return None
    return m.group(1) or m.group(2) or m.group(3) or m.group(4)


def main():
    # ── positive control FIRST, always runs regardless of whether there's
    # a diff from main -- proves the detector itself isn't a no-op.
    control_hit = added_network_import("+import httpx")
    if control_hit != "httpx":
        fail(f"positive control failed: detector did not flag '+import httpx' (got {control_hit!r})")
    control_miss = added_network_import("+import os")
    if control_miss is not None:
        fail(f"positive control false-positive: '+import os' was flagged as {control_miss!r}")
    control_context_line = added_network_import("+++ b/revien/foo.py")
    if control_context_line is not None:
        fail("positive control false-positive: a '+++' file-header line was flagged")

    for line, want in (("+import http.client", "http.client"),
                       ("+    import http.client", "http.client"),
                       ("+from http import client", "http"),
                       ("+import socket", "socket")):
        got = added_network_import(line)
        if got != want:
            fail(f"positive control failed: {line!r} gave {got!r}, want {want!r}")

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

    # ── scan added lines of every changed revien/**/*.py vs main ──
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
    if not changed_files:
        # No diff from main -- handled gracefully; the positive control
        # above already proved the scan logic works.
        pass
    else:
        for rel_path in changed_files:
            diff_proc = subprocess.run(
                ["git", "diff", "main", "--", rel_path],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
            )
            if diff_proc.returncode != 0:
                fail(f"git diff main -- {rel_path} failed: {diff_proc.stderr}")
            for line in diff_proc.stdout.splitlines():
                hit = added_network_import(line)
                if hit and (rel_path, hit) not in ALLOWED:
                    violations.append((rel_path, hit, line))

    # Full-tree check as well: an allowlisted import must still exist where
    # the allowlist says (a stale entry would silently stop covering it).
    for (rel_path, module), _why in ALLOWED.items():
        src = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if not re.search(rf"^\s*import\s+{re.escape(module)}\b", src, re.M):
            fail(f"allowlist entry {rel_path}:{module} no longer matches the source")

    if violations:
        fail(f"changed files add a new network-client import on the default path: {violations}")

    print("egress verification passed")


if __name__ == "__main__":
    main()

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


NETWORK_MODULES = ("httpx", "requests", "urllib.request", "socket", "aiohttp")
_IMPORT_RE = re.compile(
    r"^\+\s*(?:import\s+(" + "|".join(re.escape(m) for m in NETWORK_MODULES) + r")\b"
    r"|from\s+(" + "|".join(re.escape(m) for m in NETWORK_MODULES) + r")\b)"
)


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
    return m.group(1) or m.group(2)


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
                if hit:
                    violations.append((rel_path, hit, line))

    if violations:
        fail(f"changed files add a new network-client import on the default path: {violations}")

    print("egress verification passed")


if __name__ == "__main__":
    main()

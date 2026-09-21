"""G9: CLI skills propose/list/accept and recall --source run under a
cp1252 console without a Unicode crash.

CHECK: python scripts/gates/check_cli_ascii.py
EXPECT: cli ascii verification passed
"""
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from revien.graph.schema import Node, NodeType, SourceType  # noqa: E402
from revien.graph.store import GraphStore  # noqa: E402
from revien.skills.proposals import propose_skills  # noqa: E402


def fail(msg):
    print(f"check_cli_ascii: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # ── verify `python -m revien.cli` actually works as an entry point ──
    cli_src = (REPO_ROOT / "revien" / "cli.py").read_text(encoding="utf-8")
    if '__name__ == "__main__"' not in cli_src:
        fail("revien/cli.py has no `if __name__ == '__main__':` guard -- "
             "`python -m revien.cli` would not run main()")

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphStore(db_path=db_path)

    # Seed a proposal so `skills list`/`skills propose`/`skills accept`
    # and `recall` all have something real to act on.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 0
    for sess in ("s1", "s2", "s3"):
        for step in ("I'll ping Mara about Fernweh-Core", "I'll sync Fernweh-Core branches"):
            store.add_node(Node(
                node_type=NodeType.ACTION, label=step, content=step,
                source_id=f"claude-code:fernweh-core:{sess}", source_type=SourceType.EXTRACTED,
                confidence=1.0, origin_runtime="claude-code", origin_source="live",
                project_key="fernweh-core", session_key=sess,
                recorded_at=base + timedelta(minutes=n),
            ))
            n += 1
    store.add_node(Node(
        node_type=NodeType.FACT, label="Fernweh-Core pricing",
        content="Fernweh-Core enterprise tier is $499/month.",
        source_id="x", source_type=SourceType.EXTRACTED, confidence=1.0,
        origin_runtime="claude-code", origin_source="live", project_key="fernweh-core",
    ))
    proposal_summary = propose_skills(store)
    if not proposal_summary["proposals"]:
        fail("seed propose_skills produced no proposals to exercise the CLI against")
    proposal_id = proposal_summary["proposals"][0].node_id
    store.close()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
    env["REVIEN_SEMANTIC"] = "0"
    env["REVIEN_RERANK"] = "0"

    commands = [
        [sys.executable, "-m", "revien.cli", "skills", "propose", "--db", db_path],
        [sys.executable, "-m", "revien.cli", "skills", "list", "--db", db_path],
        [sys.executable, "-m", "revien.cli", "skills", "accept", proposal_id, "--db", db_path],
        [sys.executable, "-m", "revien.cli", "recall", "Fernweh-Core",
         "--source", "claude-code", "--db", db_path],
    ]

    for cmd in commands:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
            encoding="cp1252", errors="replace", timeout=60,
        )
        if proc.returncode != 0:
            fail(f"{' '.join(cmd[2:])} exited {proc.returncode}\n"
                 f"stdout={proc.stdout}\nstderr={proc.stderr}")
        if "UnicodeEncodeError" in proc.stderr:
            fail(f"{' '.join(cmd[2:])} stderr contains UnicodeEncodeError: {proc.stderr}")
        if "UnicodeEncodeError" in proc.stdout:
            fail(f"{' '.join(cmd[2:])} stdout contains UnicodeEncodeError: {proc.stdout}")

    try:
        os.unlink(db_path)
    except PermissionError:
        pass

    print("cli ascii verification passed")


if __name__ == "__main__":
    main()

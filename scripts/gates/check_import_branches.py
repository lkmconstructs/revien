"""G14: ChatGPT import ingests only the displayed thread, never an
edited-away branch, and dry-run writes nothing.

CHECK: python scripts/gates/check_import_branches.py
EXPECT: import branch verification passed
"""
import json
import os
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from click.testing import CliRunner  # noqa: E402

from revien.cli import main as cli_main  # noqa: E402
from revien.importers import chatgpt as chatgpt_importer  # noqa: E402

B_TEXT = "Theo settles on branch B — the current, displayed reply."
D_PHRASE = "Sam's distinctive abandoned-branch phrase about Fernweh-Core."


def fail(msg):
    print(f"check_import_branches: {msg}", file=sys.stderr)
    sys.exit(1)


def zip_json(path: Path, data) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(data))


# root -> A(user) -> B(assistant, current) ; and root -> A -> C(user) -> D
# (assistant, longer abandoned branch, contains D_PHRASE).
def branching_mapping():
    return {
        "root": {"id": "root", "parent": None, "children": ["A"], "message": None},
        "A": {
            "id": "A", "parent": "root", "children": ["B", "C"],
            "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["Mara asks Theo and Sam."]},
                "create_time": 1712000000.0,
            },
        },
        "B": {
            "id": "B", "parent": "A", "children": [],
            "message": {
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [B_TEXT]},
                "create_time": 1712000010.0,
            },
        },
        "C": {
            "id": "C", "parent": "A", "children": ["D"],
            "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["A follow-up down the longer branch."]},
                "create_time": 1712000020.0,
            },
        },
        "D": {
            "id": "D", "parent": "C", "children": [],
            "message": {
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": [D_PHRASE]},
                "create_time": 1712000030.0,
            },
        },
    }


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="revien-gate14-"))

    # ── 1: current_node = B -> B's text present, D's phrase absent ─────────
    conv_current = {
        "id": "conv-current",
        "title": "Fernweh-Core branch pick",
        "create_time": 1712000000.0,
        "mapping": branching_mapping(),
        "current_node": "B",
    }
    zpath1 = tmpdir / "current.zip"
    zip_json(zpath1, [conv_current])

    units1 = list(chatgpt_importer.iter_units(str(zpath1)))
    if len(units1) != 1:
        fail(f"expected 1 unit, got {len(units1)}")
    content1 = units1[0].content
    if B_TEXT not in content1:
        fail("displayed thread (current_node=B) is missing B's text")
    if D_PHRASE in content1:
        fail("displayed thread (current_node=B) leaked the abandoned D branch")

    # ── 2: current_node absent -> longest path chosen (contains D's phrase) ──
    conv_no_current = {
        "id": "conv-no-current",
        "title": "Fernweh-Core branch pick (no current_node)",
        "create_time": 1712000000.0,
        "mapping": branching_mapping(),
        # current_node deliberately omitted.
    }
    zpath2 = tmpdir / "no_current.zip"
    zip_json(zpath2, [conv_no_current])

    units2 = list(chatgpt_importer.iter_units(str(zpath2)))
    if len(units2) != 1:
        fail(f"expected 1 unit, got {len(units2)}")
    content2 = units2[0].content
    if D_PHRASE not in content2:
        fail("missing current_node should fall back to the LONGEST path "
             "(root->A->C->D), but D's phrase is absent")

    # ── 3: parent cycle -> iter_units returns within 5s, never hangs ───────
    cyclic_mapping = {
        "root": {"id": "root", "parent": None, "children": ["X"], "message": None},
        "X": {
            "id": "X", "parent": "Y", "children": ["Y"],
            "message": {
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": ["X in a cycle."]},
                "create_time": 1712000000.0,
            },
        },
        "Y": {
            "id": "Y", "parent": "X", "children": ["X"],
            "message": {
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["Y in a cycle."]},
                "create_time": 1712000001.0,
            },
        },
    }
    conv_cycle = {
        "id": "conv-cycle",
        "title": "Fernweh-Core cyclic mapping",
        "create_time": 1712000000.0,
        "mapping": cyclic_mapping,
        "current_node": "X",  # X's parent is Y, Y's parent is X -> cycle
    }
    zpath3 = tmpdir / "cycle.zip"
    zip_json(zpath3, [conv_cycle])

    result = {}

    def run_cyclic():
        try:
            result["units"] = list(chatgpt_importer.iter_units(str(zpath3)))
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=run_cyclic, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    if thread.is_alive():
        fail("iter_units hung on a parent-cycle mapping (did not return within 5s)")
    if "error" in result:
        fail(f"iter_units raised on a parent-cycle mapping: {result['error']!r}")
    if "units" not in result or len(result["units"]) != 1:
        fail("cyclic-mapping conversation did not yield exactly 1 unit")

    # ── 4: --dry-run via CliRunner ──────────────────────────────────────────
    dry_conv_a = {
        "id": "conv-dry-a",
        "title": "Fernweh-Core dry run A",
        "create_time": 1712000000.0,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["u1"], "message": None},
            "u1": {
                "id": "u1", "parent": "root", "children": ["a1"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Mara's dry-run question A."]},
                    "create_time": 1712000000.0,
                },
            },
            "a1": {
                "id": "a1", "parent": "u1", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Theo's dry-run answer A."]},
                    "create_time": 1712000010.0,
                },
            },
        },
        "current_node": "a1",
    }
    dry_conv_b = {
        "id": "conv-dry-b",
        "title": "Fernweh-Core dry run B",
        "create_time": 1712000000.0,
        "mapping": {
            "root": {"id": "root", "parent": None, "children": ["u1"], "message": None},
            "u1": {
                "id": "u1", "parent": "root", "children": ["a1"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Sam's dry-run question B."]},
                    "create_time": 1712000000.0,
                },
            },
            "a1": {
                "id": "a1", "parent": "u1", "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Theo's dry-run answer B."]},
                    "create_time": 1712000010.0,
                },
            },
        },
        "current_node": "a1",
    }
    fixture_conversation_count = 2
    zpath4 = tmpdir / "dry_run.zip"
    zip_json(zpath4, [dry_conv_a, dry_conv_b])

    nonexistent_db = tmpdir / "does-not-exist" / "revien.db"
    if nonexistent_db.exists():
        fail("test setup error: nonexistent_db already exists")

    runner = CliRunner()
    invoke_result = runner.invoke(
        cli_main,
        ["import-chatgpt", str(zpath4), "--db", str(nonexistent_db), "--dry-run"],
    )
    if invoke_result.exit_code != 0:
        fail(f"--dry-run exit_code == {invoke_result.exit_code}, expected 0\n"
             f"output: {invoke_result.output}")
    if "nothing written" not in invoke_result.output:
        fail(f"--dry-run output missing 'nothing written': {invoke_result.output!r}")
    if nonexistent_db.exists():
        fail("--dry-run created the db path, which must never happen")
    if f"seen={fixture_conversation_count}" not in invoke_result.output:
        fail(
            f"--dry-run reported unit count did not match the fixture's "
            f"conversation count ({fixture_conversation_count}): "
            f"{invoke_result.output!r}"
        )

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("import branch verification passed")


if __name__ == "__main__":
    main()

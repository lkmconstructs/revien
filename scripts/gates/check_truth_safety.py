"""G19: the supersession gate never auto-fires unsafely, never breaches a
sensitive control, and never wrongfully erases.

CHECK: python scripts/gates/check_truth_safety.py
EXPECT: truth safety verification passed

Supersession numbers come from measure_supersession.measure() (structured).
measure_fact_update only has a printing main(), so its stdout is captured and
the labelled lines are parsed.
"""
import contextlib
import io
import os
import re
import sys
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def fail(msg):
    print(f"check_truth_safety: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


def grab(pattern, text, what):
    m = re.search(pattern, text)
    check(m is not None, f"could not parse '{what}' from fact-update output")
    return tuple(int(g) for g in m.groups())


def main_check():
    from revien_bench import measure_fact_update as FU  # noqa: E402
    from revien_bench import measure_supersession as SS  # noqa: E402

    r = SS.measure()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FU.main()  # exit code is the 90%-rate verdict, not a safety signal
    out = buf.getvalue()

    breaches, n_sens = grab(r"SENSITIVE CONTROLS[^\n]*?(\d+)/(\d+) breaches", out, "sensitive breaches")
    wrongful, n_restate = grab(r"(\d+)/(\d+) wrongful erases", out, "wrongful erases")
    n_unsafe = len(r["unsafe_autos"])
    gl_ok, gl_n = r["gate_logic_accuracy"]
    n_contra = r["n_contradiction"]

    # positive controls: an empty fixture set must not pass
    check(n_sens > 0, "no sensitive controls in the fact-update corpus")
    check(n_restate > 0, "no restatement/independent precision cases")
    check(n_contra > 0, "no real contradictions in the supersession fixtures")
    check(gl_n > 0, "gate engaged on zero fixtures")

    check(n_unsafe == 0, f"{n_unsafe} unsafe auto-fires")
    check(breaches == 0, f"{breaches}/{n_sens} sensitive controls breached")
    check(wrongful == 0, f"{wrongful}/{n_restate} wrongful erases")
    check(gl_ok == gl_n, f"gate-logic accuracy {gl_ok}/{gl_n}, expected 100%")

    print(f"unsafe auto-fires 0; sensitive breaches 0/{n_sens}; wrongful erases 0/{n_restate}; "
          f"gate-logic accuracy {gl_ok}/{gl_n}; real contradictions {n_contra}")


if __name__ == "__main__":
    try:
        main_check()
    except SystemExit:
        raise
    except Exception as exc:  # any crash is a failed gate
        fail(f"{type(exc).__name__}: {exc}")
    print("truth safety verification passed")

"""G20: truth-maintenance rates on the rule path do not regress below the
recorded floor.

CHECK: python scripts/gates/check_truth_rates.py
EXPECT: truth rates verification passed

Floors are today's (2026-10-02) measured values on the rule path. Raising them
is expected as recognizers improve; lowering them requires the owner's ruling.
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

FLOOR_UPDATED = 8      # of 68 ordinary fact-changes ending UPDATED (2026-10-02)
FLOOR_AUTO_FIRE = 6    # of 20 real contradictions auto-superseded (2026-10-02)


def fail(msg):
    print(f"check_truth_rates: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond, msg):
    if not cond:
        fail(msg)


def main_check():
    from revien_bench import measure_fact_update as FU  # noqa: E402
    from revien_bench import measure_supersession as SS  # noqa: E402

    r = SS.measure()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FU.main()  # no structured API; parse the labelled lines below
    out = buf.getvalue()

    m = re.search(r"UPDATED \(current truth held\): (\d+)/(\d+)", out)
    q = re.search(r"QUEUED  \([^)]*\):\s*(\d+)/(\d+)", out)
    s = re.search(r"STALE   \([^)]*\):\s*(\d+)/(\d+)", out)
    check(m and q and s, "could not parse UPDATED/QUEUED/STALE from fact-update output")
    upd, total = int(m.group(1)), int(m.group(2))
    queued, stale = int(q.group(1)), int(s.group(1))
    check(total > 0 and upd + queued + stale == total, "UPDATED/QUEUED/STALE do not sum to the corpus size")

    n_auto, n_contra = r["auto_fire_rate"]
    d = r["disposition"]
    check(n_contra > 0, "no real contradictions in the supersession fixtures")

    print(f"fact-update: UPDATED {upd}/{total}, QUEUED {queued}, STALE {stale} (floor UPDATED >= {FLOOR_UPDATED})")
    print(f"supersession: auto-fire {n_auto}/{n_contra} (floor >= {FLOOR_AUTO_FIRE}); "
          f"auto={d['auto_supersede']} candidate={d['candidate']} "
          f"version={d['version_locked']} missed={d['missed_no_conflict']}")

    check(upd >= FLOOR_UPDATED, f"UPDATED {upd}/{total} below floor {FLOOR_UPDATED}")
    check(n_auto >= FLOOR_AUTO_FIRE, f"auto-fire {n_auto}/{n_contra} below floor {FLOOR_AUTO_FIRE}")


if __name__ == "__main__":
    try:
        main_check()
    except SystemExit:
        raise
    except Exception as exc:  # any crash is a failed gate
        fail(f"{type(exc).__name__}: {exc}")
    print("truth rates verification passed")

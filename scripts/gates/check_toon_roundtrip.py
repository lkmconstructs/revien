"""G10: TOON recall round-trips origin fields, [filtered] path entries, and
skill_proposals losslessly.

CHECK: python scripts/gates/check_toon_roundtrip.py
EXPECT: toon roundtrip verification passed
"""
import os
import sys
from pathlib import Path

os.environ["REVIEN_SEMANTIC"] = "0"
os.environ["REVIEN_RERANK"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from revien.toon import parse_recall, serialize_recall  # noqa: E402


def fail(msg):
    print(f"check_toon_roundtrip: {msg}", file=sys.stderr)
    sys.exit(1)


def base_result(**overrides):
    r = {
        "node_id": "n-1",
        "node_type": "fact",
        "label": "Fernweh-Core pricing",
        "content": "Fernweh-Core enterprise tier is 499 dollars per month.",
        "score": 0.87,
        "score_breakdown": {"recency": 0.9, "confidence": 0.8},
        "path": ["n-1"],
        "origin_runtime": "claude-code",
        "origin_source": "live",
        "project_key": "Fernweh-Core",
        "recorded_at": "2023-05-07T00:00:00+00:00",
    }
    r.update(overrides)
    return r


def base_payload(results, skill_proposals):
    return {
        "query": "Fernweh-Core pricing",
        "results": results,
        "nodes_examined": len(results),
        "retrieval_time_ms": 1.23,
        "semantic_active": False,
        "semantic_note": None,
        "skill_proposals": skill_proposals,
    }


def main():
    variants = []

    # 1. origin fields as None
    variants.append(("origin fields None", base_payload(
        [base_result(origin_runtime=None, origin_source=None, project_key=None)], [],
    )))

    # 2. origin fields as ""
    variants.append(("origin fields empty string", base_payload(
        [base_result(origin_runtime="", origin_source="", project_key="")], [],
    )))

    # 3. origin fields as unicode strings
    variants.append(("origin fields unicode", base_payload(
        [base_result(
            label="Fernweh — cafe launch",
            origin_runtime="claude-code",
            project_key="Fernweh — cafe launch",
        )], [],
    )))

    # 4. path list containing the literal "[filtered]" string
    variants.append(("filtered path entry", base_payload(
        [base_result(path=["n-anchor", "[filtered]", "n-leaf"])], [],
    )))

    # 5. skill_proposals empty list
    variants.append(("empty skill_proposals", base_payload(
        [base_result()], [],
    )))

    # 6. skill_proposals populated, steps contain " -> " and commas
    variants.append(("populated skill_proposals", base_payload(
        [base_result()],
        [
            {
                "node_id": "skill-1",
                "label": "proposed: sync fernweh, ping mara -> notify sam",
                "occurrences": 4,
                "sessions": 2,
                "project_key": "Fernweh-Core",
                "steps": "sync fernweh, ping mara -> notify sam",
            },
            {
                "node_id": "skill-2",
                "label": "proposed: run bench -> report, done",
                "occurrences": 3,
                "sessions": 3,
                "project_key": None,
                "steps": "run bench -> report, done",
            },
        ],
    )))

    # 7. multiple results in one payload, mixed origin states + tensions key
    variants.append(("mixed multi-result with tensions", {
        "query": "Fernweh-Core status",
        "results": [
            base_result(node_id="n-a", origin_runtime="codex"),
            base_result(node_id="n-b", origin_runtime=None, origin_source=None, project_key=None,
                        path=["n-b", "[filtered]"]),
        ],
        "nodes_examined": 2,
        "retrieval_time_ms": 4.56,
        "semantic_active": True,
        "semantic_note": None,
        "skill_proposals": [],
    }))

    if len(variants) < 6:
        fail(f"only built {len(variants)} variants, need at least 6")

    for name, payload in variants:
        toon = serialize_recall(payload)
        back = parse_recall(toon)
        if back != payload:
            fail(f"round-trip mismatch for variant {name!r}:\n"
                 f"  original: {payload}\n"
                 f"  toon:     {toon}\n"
                 f"  parsed:   {back}")

    print("toon roundtrip verification passed")


if __name__ == "__main__":
    main()

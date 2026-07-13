"""
LEG P2 measurement: token counts, JSON vs TOON, on REAL recall payloads.

Measure, don't assert — this script produces the number the README line
publishes, whatever it turns out to be. It does NOT inherit the upstream
"~40%" claim.

Methodology:
  1. Seed a temp graph via the real daemon app (TestClient + /v1/ingest)
     with several multi-turn conversations (plain, delimiter-heavy,
     unicode).
  2. Run real POST /v1/recall queries; take each JSON payload exactly as a
     consuming LLM would receive it.
  3. Additionally, scan results/*.json for recall-SHAPED fixtures (a top-
     level "results" list of dicts with node_id) and include any found.
     (The eval-run outputs in results/ are metrics files, not recall
     payloads — expected to be skipped; the scan is here so future recall
     fixtures get picked up automatically.)
  4. Serialize each payload three ways — compact JSON
     (separators=(",", ":")), pretty JSON (indent=2, what the CLI prints),
     and TOON — and count tokens.
  5. Tokenizer: tiktoken cl100k_base when importable (exact). Fallback (NO
     new core dependency): bytes/4 heuristic (UTF-8 byte length / 4, the
     common rule-of-thumb) — clearly labeled APPROXIMATE in output.

Every TOON serialization is round-trip verified (parse == payload) before
it is counted, so the measured number is for a lossless encoding only.

Run:  python -m revien_bench.measure_toon
"""

import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revien.toon import parse_recall, serialize_recall  # noqa: E402

CONVERSATIONS = [
    (
        "bench-pricing",
        """User: We need to decide on the pricing for the enterprise tier.
Assistant: Based on our analysis, I recommend $499/month with a 20% annual discount.
User: That works. Let's go with that. Also, make sure the deployment uses PostgreSQL, not MySQL. We decided that last week.
Assistant: Confirmed. Enterprise tier at $499/month, 20% annual discount, PostgreSQL for the database layer. I'll update the architecture doc.""",
    ),
    (
        "bench-launch",
        """User: The launch checklist is: staging, canary, prod — in that order, no exceptions.
Assistant: Noted. Checklist order confirmed: staging, canary, prod.
User: Also the café rebrand ships with the 中文 locale and the "grand opening" banner 😀.
Assistant: Understood. Café rebrand includes the 中文 locale and the "grand opening" banner.
User: Deadline for the rebrand is March 15, and Sarah owns the rollout.
Assistant: Recorded: rebrand deadline March 15, Sarah owns the rollout.""",
    ),
    (
        "bench-infra",
        """User: Move the vector index to the new box, keep SQLite as the store of record.
Assistant: Understood — vector index migrates, SQLite remains the store of record.
User: Backups run nightly at 02:00 UTC via restic, retention 14 days.
Assistant: Noted: nightly restic backups at 02:00 UTC, 14-day retention.
User: And the API rate limit stays at 120 requests per minute per token.
Assistant: Confirmed: 120 requests/minute/token rate limit unchanged.""",
    ),
]

QUERIES = [
    "enterprise pricing decision",
    "which database was chosen",
    "launch checklist order",
    "café rebrand locale and banner",
    "who owns the rollout and when is the deadline",
    "backup schedule and retention",
    "API rate limit",
]


def _get_counter():
    """Returns (count_fn, label, exact)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s))), "tiktoken cl100k_base", True
    except Exception as exc:  # not installed, or encoding fetch failed
        print(f"[note] tiktoken unavailable ({exc!r}); "
              "falling back to APPROXIMATE bytes/4 heuristic")
        return (lambda s: max(1, len(s.encode("utf-8")) // 4)), \
            "APPROXIMATE bytes/4 heuristic", False


def _collect_api_payloads():
    from fastapi.testclient import TestClient

    from revien.daemon.server import create_app

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    payloads = []
    try:
        app = create_app(db_path=path)
        with TestClient(app) as client:
            for source_id, content in CONVERSATIONS:
                client.post("/v1/ingest", json={
                    "source_id": source_id, "content": content,
                })
            for q in QUERIES:
                payload = client.post(
                    "/v1/recall", json={"query": q, "top_n": 10}
                ).json()
                payloads.append((f"recall: {q!r}", payload))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass  # Windows: handle still held — temp file, OS cleans up
    return payloads


def _looks_like_recall_payload(obj):
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("results"), list)
        and all(isinstance(r, dict) and "node_id" in r for r in obj["results"])
    )


def _collect_fixture_payloads():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payloads, skipped = [], 0
    for fp in sorted(glob.glob(os.path.join(repo_root, "results", "*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            skipped += 1
            continue
        if _looks_like_recall_payload(obj):
            payloads.append((f"fixture: {os.path.basename(fp)}", obj))
        else:
            skipped += 1
    if skipped and not payloads:
        print(f"[note] results/*.json scanned: {skipped} file(s), none are "
              "recall-shaped fixtures (they are eval metrics) — skipped")
    return payloads


def main():
    count, tokenizer_label, exact = _get_counter()
    payloads = _collect_api_payloads() + _collect_fixture_payloads()

    if not payloads:
        print("No payloads collected — nothing to measure.")
        return 1

    header = f"{'payload':<48} {'json':>8} {'json-2sp':>9} {'toon':>8} {'vs json':>9} {'vs json-2sp':>12}"
    print(f"\nTokenizer: {tokenizer_label}"
          + ("" if exact else "  ** counts are APPROXIMATE **"))
    print(header)
    print("-" * len(header))

    tot_compact = tot_pretty = tot_toon = 0
    for name, payload in payloads:
        toon = serialize_recall(payload)
        assert parse_recall(toon) == payload, f"round-trip failed for {name}"
        compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        pretty = json.dumps(payload, ensure_ascii=False, indent=2)
        n_c, n_p, n_t = count(compact), count(pretty), count(toon)
        tot_compact += n_c
        tot_pretty += n_p
        tot_toon += n_t
        print(f"{name[:48]:<48} {n_c:>8} {n_p:>9} {n_t:>8} "
              f"{100.0 * (n_c - n_t) / n_c:>8.1f}% {100.0 * (n_p - n_t) / n_p:>11.1f}%")

    red_c = 100.0 * (tot_compact - tot_toon) / tot_compact
    red_p = 100.0 * (tot_pretty - tot_toon) / tot_pretty
    print("-" * len(header))
    print(f"{'TOTAL (' + str(len(payloads)) + ' payloads)':<48} "
          f"{tot_compact:>8} {tot_pretty:>9} {tot_toon:>8} {red_c:>8.1f}% {red_p:>11.1f}%")
    approx = "" if exact else " (approximate tokenizer)"

    def fmt(red):  # negative reduction = TOON came out LARGER; say so plainly
        return (f"-{red:.1f}%" if red >= 0
                else f"+{abs(red):.1f}% (TOON is LARGER)")

    print(f"\nMeasured token reduction, TOON vs compact JSON:  {fmt(red_c)}{approx}")
    print(f"Measured token reduction, TOON vs 2-space JSON:  {fmt(red_p)}{approx}")
    print("(json = compact separators, json-2sp = indent=2 as the CLI prints; "
          "TOON round-trip verified lossless on every payload before counting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

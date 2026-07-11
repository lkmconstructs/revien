"""
revien_bench.vault_eval — the vault's OWN eval. AND-not-OR, enforced here.

Measures recall over an Obsidian-vault corpus (plus a small conversation log
about vault entities) through the REAL ingest + recall paths. This produces
the VAULT number. It is a different corpus, a different gold space, and a
different results namespace (results/vault/) from the conversational LoCoMo
figures — the two are never averaged, compared side-by-side in one table, or
blended into a single headline. Publishing rule (maintainer's gate): no public
Obsidian claim without this eval's output attached.

Tracks:
  * retrieval: recall@k / MRR / nDCG@10 against gold ids
      - note:<relpath>   — any chunk of that note counts as a hit
      - conv:<turn_id>   — a specific conversation turn
  * miss taxonomy: same instrument as the conversational bench
    (never_extracted / no_anchors / walk_depth_miss / disconnected /
    filtered_out / outranked), gold space swapped in.
  * ATTACHMENT RATE: for each conversation turn with ground-truth intent,
    did the machine-side claim actually connect to the vault entity it is
    about? expect_attach=false rows are known-fragile label variants —
    reported separately, never dropped. This is the cross-corpus number the
    distill leg exposed ("Fernweh-Core" != "Fernweh Core").

Zero LLM, zero network on the default path. Extractive answering is out of
scope on purpose — retrieval is the claim being gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import revien
from revien.adapters.obsidian import ObsidianVaultAdapter
from revien.graph.schema import NodeType
from revien.graph.operations import GraphOperations
from revien.graph.store import GraphStore
from revien.ingestion.pipeline import IngestionInput, IngestionPipeline
from revien.retrieval.engine import RetrievalEngine
from revien.semantic.index import SemanticIndex

from . import failure_analysis as FA
from . import metrics as M

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
FIXTURE_VAULT = _PKG_DIR / "fixtures" / "vault"
FIXTURE_CONVS = _PKG_DIR / "fixtures" / "vault_conversations.json"
FIXTURE_QA = _PKG_DIR / "fixtures" / "vault_qa.json"
RECALL_KS = (1, 3, 5, 10)
RECALL_TOP_N = 10


def fixture_sha() -> str:
    """SHA-256 over the whole fixture corpus (vault + conversations + QA), so
    a result JSON is verifiably tied to the exact dataset that produced it."""
    h = hashlib.sha256()
    paths = sorted(FIXTURE_VAULT.rglob("*.md")) + [FIXTURE_CONVS, FIXTURE_QA]
    for p in paths:
        h.update(str(p.relative_to(_PKG_DIR)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


# ── Ingest ─────────────────────────────────────────────────────────────

def ingest_corpus(store: GraphStore, semantic: SemanticIndex) -> Dict:
    """Vault through the real adapter+pipeline (curated), then conversations
    through the same pipeline (machine-side). Returns ingest counts."""
    import asyncio

    pipeline = IngestionPipeline(store, semantic=semantic)

    adapter = ObsidianVaultAdapter(str(FIXTURE_VAULT))
    items = asyncio.run(
        adapter.fetch_new_content(datetime.fromtimestamp(0, tz=timezone.utc))
    )
    for item in items:
        ts = None
        try:
            ts = datetime.fromisoformat(item["timestamp"])
        except (KeyError, ValueError, TypeError):
            pass
        pipeline.ingest(IngestionInput(
            source_id=item["source_id"],
            content=item["content"],
            content_type="note",
            timestamp=ts,
            metadata=item.get("metadata", {}),
            links=item.get("links", []),
            curated=True,
        ))

    convs = json.loads(FIXTURE_CONVS.read_text(encoding="utf-8"))
    for turn in convs["turns"]:
        pipeline.ingest(IngestionInput(
            source_id=turn["id"],
            content=turn["text"],
            content_type="conversation",
            timestamp=datetime.fromisoformat(turn["timestamp"]),
            metadata={"conv": convs["conversation_id"]},
        ))

    return {"vault_chunks": len(items), "conversation_turns": len(convs["turns"])}


# ── Gold-space mapping ─────────────────────────────────────────────────

def _gold_id_for_node(node) -> Optional[str]:
    # The pipeline does NOT copy IngestionInput.metadata onto extracted nodes
    # (same trap ingest_locomo documents), so map through source_id — the
    # adapter encodes the note path as "vault:<relpath>#<heading-slug>".
    sid = node.source_id or ""
    if sid.startswith("vault:"):
        return f"note:{sid[len('vault:'):].split('#', 1)[0]}"
    if sid.startswith("vaultconv:"):
        return f"conv:{sid}"
    return None


def build_gold_map(store: GraphStore) -> Dict[str, List[str]]:
    """gold_id -> [node_ids], the vault eval's analogue of the dia_id map."""
    gold_map: Dict[str, List[str]] = {}
    for node in store.list_nodes(limit=999999):
        gid = _gold_id_for_node(node)
        if gid:
            gold_map.setdefault(gid, []).append(node.node_id)
    return gold_map


def _retrieved_gold_ids(store: GraphStore, results) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for r in results:
        node = store.get_node(r.node_id)
        gid = _gold_id_for_node(node) if node else None
        if gid and gid not in seen:
            seen.add(gid)
            ordered.append(gid)
    return ordered


# ── Attachment rate ────────────────────────────────────────────────────

def measure_attachment(store: GraphStore) -> Dict:
    """Did each conversation turn's memory connect to the vault entity it is
    about? A turn counts as attached when ANY node it produced shares an edge
    with the intended entity node (or dedup-merged into it)."""
    ops = GraphOperations(store)
    convs = json.loads(FIXTURE_CONVS.read_text(encoding="utf-8"))
    rows = []
    for turn in convs["turns"]:
        entity = ops.find_node_by_label(turn["intended_entity"], node_type=NodeType.ENTITY)
        attached = False
        if entity is not None:
            entity_neighbors = set(store.get_neighbors(entity.node_id))
            turn_nodes = store.list_nodes(source_id=turn["id"], limit=1000)
            attached = any(
                n.node_id in entity_neighbors or n.node_id == entity.node_id
                for n in turn_nodes
            )
        rows.append({
            "turn": turn["id"],
            "intended_entity": turn["intended_entity"],
            "expect_attach": turn["expect_attach"],
            "attached": attached,
        })

    expected = [r for r in rows if r["expect_attach"]]
    fragile = [r for r in rows if not r["expect_attach"]]
    return {
        "rows": rows,
        "attach_rate_expected": (
            round(sum(r["attached"] for r in expected) / len(expected), 4)
            if expected else None
        ),
        "attach_rate_fragile_variants": (
            round(sum(r["attached"] for r in fragile) / len(fragile), 4)
            if fragile else None
        ),
        "n_expected": len(expected),
        "n_fragile": len(fragile),
    }


# ── Eval ───────────────────────────────────────────────────────────────

def run_vault_eval(out_dir: Path, semantic_enabled: bool = True) -> Dict:
    fd, db_path = tempfile.mkstemp(suffix="_vault_eval.db")
    os.close(fd)
    store = GraphStore(db_path=db_path)
    try:
        # None -> default gate (spine on when available); False -> forced off.
        semantic = SemanticIndex(store, enabled=(None if semantic_enabled else False))
        ingest_counts = ingest_corpus(store, semantic)

        engine = RetrievalEngine(store, semantic=semantic)
        gold_map = build_gold_map(store)
        qa_items = json.loads(FIXTURE_QA.read_text(encoding="utf-8"))["qa"]

        per_q: List[Dict] = []
        latencies: List[float] = []
        for qa in qa_items:
            gold = set(qa["gold"])
            t0 = time.perf_counter()
            resp = engine.recall(
                qa["q"], top_n=RECALL_TOP_N, include_context=True, debug=True
            )
            ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(ms)

            retrieved = _retrieved_gold_ids(store, resp.results)
            row = {
                "question": qa["q"],
                "category": qa["category"],
                "gold_evidence": sorted(gold),
                "retrieved": retrieved,
                **{
                    f"recall@{k}": round(M.recall_at_k(retrieved, gold, k), 4)
                    for k in RECALL_KS
                },
                "mrr": round(M.mrr(retrieved, gold), 4),
                "ndcg@10": round(M.ndcg_at_k(retrieved, gold, 10), 4),
                "recall_latency_ms": round(ms, 3),
                "gold_miss_causes": FA.classify_misses(
                    store, resp.diagnostics, gold, retrieved, gold_map
                ),
                "category_name": qa["category"],  # aggregate compatibility
            }
            per_q.append(row)

        # Aggregate — vault namespace only, no shared keys with LoCoMo output.
        by_cat: Dict[str, List[Dict]] = {}
        for r in per_q:
            by_cat.setdefault(r["category"], []).append(r)
        retrieval = {
            f"recall@{k}": round(M.mean([r[f"recall@{k}"] for r in per_q]), 4)
            for k in RECALL_KS
        }
        retrieval["mrr"] = round(M.mean([r["mrr"] for r in per_q]), 4)
        retrieval["ndcg@10"] = round(M.mean([r["ndcg@10"] for r in per_q]), 4)
        per_category = {
            cat: {
                "n": len(rows),
                "recall@10": round(M.mean([r["recall@10"] for r in rows]), 4),
                "mrr": round(M.mean([r["mrr"] for r in rows]), 4),
            }
            for cat, rows in sorted(by_cat.items())
        }

        report = {
            "eval": "vault_fixture_v1",
            "corpus_note": (
                "VAULT corpus — separate number from conversational LoCoMo "
                "results; never blend or average the two."
            ),
            "semantic_enabled": semantic.is_enabled,
            "n_questions": len(per_q),
            "retrieval": retrieval,
            "per_category": per_category,
            "attachment": measure_attachment(store),
            "normalization_merges": FA.normalization_merge_report(store),
            "retrieval_failure_analysis": FA.aggregate_failures(per_q),
            "ingest": ingest_counts,
            "latency_ms": {"recall": M.latency_percentiles(latencies)},
            "dataset_sha256": fixture_sha(),
            "revien_version": revien.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "per_question": per_q,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"{ts}_vault_eval.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["_out_path"] = str(out_path)
        return report
    finally:
        store.close()
        try:
            os.unlink(db_path)
        except OSError:
            pass


def _print_summary(report: Dict) -> None:
    print("\n=== Revien VAULT eval (second corpus — its own number) ===")
    print(f"semantic      : {'on' if report['semantic_enabled'] else 'OFF (graph-only)'}")
    print(f"questions     : {report['n_questions']}   "
          f"(vault chunks {report['ingest']['vault_chunks']}, "
          f"conv turns {report['ingest']['conversation_turns']})")
    r = report["retrieval"]
    print(f"retrieval     : recall@1={r['recall@1']} @5={r['recall@5']} "
          f"@10={r['recall@10']} MRR={r['mrr']} nDCG@10={r['ndcg@10']}")
    for cat, v in report["per_category"].items():
        print(f"   {cat:12s} n={v['n']:<3d} recall@10={v['recall@10']} mrr={v['mrr']}")
    a = report["attachment"]
    print(f"attachment    : {a['attach_rate_expected']} on {a['n_expected']} clean-label "
          f"turns; {a['attach_rate_fragile_variants']} on {a['n_fragile']} fragile variants")
    nm = report.get("normalization_merges", {})
    print(f"norm merges   : {nm.get('count', 0)} normalization-only merges "
          f"(precision surface — review pairs in results JSON)")
    fa = report["retrieval_failure_analysis"]
    if fa.get("gold_items_missed"):
        causes = ", ".join(f"{c}={n}" for c, n in fa["by_cause"].items())
        print(f"miss taxonomy : {fa['gold_items_missed']} missed — {causes}")
    lat = report["latency_ms"]["recall"]
    print(f"recall latency: p50={lat['p50']}ms p90={lat['p90']}ms")
    print(f"dataset sha   : {report['dataset_sha256'][:16]}…")
    print(f"results JSON  : {report.get('_out_path')}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Revien vault eval (second corpus)")
    ap.add_argument("--out", default=str(_REPO_ROOT / "results" / "vault"))
    ap.add_argument("--no-semantic", action="store_true",
                    help="graph-only ablation run")
    args = ap.parse_args(argv)
    report = run_vault_eval(Path(args.out), semantic_enabled=not args.no_semantic)
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

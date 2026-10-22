"""Plant the secret into every surface a forget must scrub (docs/FORGET_SPEC.md
§4) and read each surface back afterwards."""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

from revien.graph.store import GraphStore
from revien.neural.training import TrainingLoop
from revien.semantic.index import SemanticIndex


class TinyEmbedder:
    dim = 4
    is_cloud = False

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def plant(db: str, secret: str, tmp: Path, node_ids: List[str]) -> Dict:
    """Write the secret to the audit snapshots (already there), the usage
    ledger, receipts, supersession candidates, the distill manifest, the
    vectors and training.db. Returns what was planted."""
    store = GraphStore(db_path=db)
    planted = {"training_db": str(tmp / "training.db")}
    try:
        now = datetime.now(timezone.utc)
        ids = [i for i in node_ids if store.get_node(i) is not None]
        target = ids[0]
        other = store.list_nodes(limit=50)
        others = [n.node_id for n in other if n.node_id not in ids]
        # usage ledger: one judgment of the target, one of another node in the
        # SAME recall; the query is the user's words, with the secret in them
        store.record_use_judgment(
            episode_key="sess#tool-1", node_id=target, used=True,
            reason="r", shared_terms=[secret.lower()], similarity=0.9,
            query=f"what is {secret}", mode="shadow", applied=False)
        if others:
            store.record_use_judgment(
                episode_key="sess#tool-1", node_id=others[0], used=False,
                reason="r", shared_terms=["other"], similarity=0.1,
                query=f"what is {secret}", mode="shadow", applied=False)
        store.add_recall_receipt(
            recall_id="rcpt-1", runtime="claude-code",
            query=f"tell me about {secret}", node_ids=[target],
            issued_at=now, expires_at=now + timedelta(hours=24))
        if others:
            store.add_candidate(target, others[0], "candidate",
                                reason=f"conflicts with {secret}",
                                trace=json.dumps({"claim": secret}))
        # two bystanders nobody forgets, with a candidate and a judgment of
        # their own that quote the secret (a recall where it was typed)
        from revien.graph.schema import Node, NodeType
        b1 = store.add_node(Node(node_type=NodeType.FACT, label="bystander one", content="plain words one"))
        b2 = store.add_node(Node(node_type=NodeType.FACT, label="bystander two", content="plain words two"))
        store.add_candidate(b1.node_id, b2.node_id, "candidate",
                            reason=f"asked about {secret}", trace=json.dumps({"q": secret}))
        store.record_use_judgment(
            episode_key="sess#tool-2", node_id=b1.node_id, used=False,
            reason="r", shared_terms=["plain"], similarity=0.2,
            query=f"remind me of {secret}", mode="shadow", applied=False)
        store.add_recall_receipt(
            recall_id="rcpt-2", runtime="claude-code",
            query=f"find {secret} please", node_ids=[b1.node_id],
            issued_at=now, expires_at=now + timedelta(hours=24))
        store.replace_distill_manifest("Some Note", [{
            "entity_id": target, "anchor_node_id": target,
            "current_node_id": target,
            "content_hash": "h", "section": "Facts"}])
        sem = SemanticIndex(store, embedder=TinyEmbedder(), enabled=True)
        for nid in ids:
            n = store.get_node(nid)
            sem.index_node(nid, n.label, n.content)
        loop = TrainingLoop(db_path=planted["training_db"],
                            model_dir=str(tmp / "models"))
        loop.log_retrieval(f"what is {secret}", [
            {"node_id": target, "label": f"code {secret}", "node_type": "fact",
             "score": 0.5}])
        planted["target"] = target
        planted["ids"] = ids
    finally:
        store.close()
    return planted


def table_text(db: str, table: str) -> str:
    conn = sqlite3.connect(db)
    try:
        return json.dumps([list(map(str, r)) for r in conn.execute(f'SELECT * FROM "{table}"')])
    finally:
        conn.close()


def raw_bytes(db: str) -> Dict[str, bytes]:
    out = {}
    for suffix in ("", "-wal", "-shm"):
        p = db + suffix
        out[suffix or "db"] = open(p, "rb").read() if os.path.exists(p) else b""
    return out

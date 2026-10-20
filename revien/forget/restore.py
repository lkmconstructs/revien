"""Restore: lift a forget record (docs/FORGET_SPEC.md §7).

A forget leaves a record of the sentences that held the forgotten wording,
bound to the source it was forgotten from (or, for ``forget --everywhere``, to
no source). ``revien restore <path>`` lifts the records bound to ONE file;
``revien restore forget:<id>`` lifts one record by its id; ``revien restore
<unit id>`` lifts the records bound to that unit. A bare name never fans out
across paths: if it names more than one record it is refused and the
candidates are listed. Lifting several at once needs an explicit pattern
(``revien restore 'projA/**'``), and what it would lift is listed first. Every
lift is audited. Only an explicit restore, or a re-import with the explicit
override flag, ever lifts a record. Lifting a record lets the wording be
captured again; it does not bring a removed node back.
"""

import fnmatch
import os
from typing import Dict, List

from revien.graph.units import identity_keys, path_unit_key

_WILD = set("*?[")


def is_pattern(arg: str) -> bool:
    return any(c in arg for c in _WILD)


def _audit_lift(store, record: Dict, actor: str, via: str) -> None:
    store.record_audit(
        f"suppression:{record['id']}", "restore", actor=actor,
        after={"unit_id": record["unit_id"], "scope": record["scope"],
               "via": via})


def lift_for_override(store, record: Dict) -> bool:
    """A re-import with the explicit override: lift and audit."""
    ok = store.lift_suppression(record["id"], "override:import")
    if ok:
        _audit_lift(store, record, "override:import", "override")
    return ok


def liftable(store) -> List[Dict]:
    """What a restore can lift: every active forget record."""
    return store.list_suppressions("active")


def _sources(rec: Dict) -> List[str]:
    return sorted(k[len("unit:"):] for k in rec["path_keys"]
                  if k.startswith("unit:"))


def _summarize(rec: Dict) -> Dict:
    return {"id": rec["id"], "unit_id": rec["unit_id"],
            "scope": rec["scope"], "sources": _sources(rec),
            "created_at": rec["created_at"],
            "sentences": len(rec.get("sentence_fps") or [])}


def _match_one(records: List[Dict], arg: str) -> List[Dict]:
    """Records bound to exactly this file or unit, or named by this id.
    Never a name search across paths."""
    found: Dict[int, Dict] = {}
    if os.path.exists(arg):
        keys = set(identity_keys(arg))
        unit_ids = {"file:" + path_unit_key(arg)}
        for rec in records:
            if keys & set(rec["path_keys"]) or any(
                    f"unit:{u}" in rec["path_keys"] for u in unit_ids):
                found[rec["id"]] = rec
        if found:
            return list(found.values())
    for rec in records:
        if rec["unit_id"] == arg or f"unit:{arg}" in rec["path_keys"]:
            found[rec["id"]] = rec
    return list(found.values())


def _match_pattern(records: List[Dict], pattern: str) -> List[Dict]:
    pat = pattern.replace("\\", "/").casefold()
    out = []
    for rec in records:
        candidates = [rec["unit_id"].casefold()] + [
            u.replace("\\", "/").casefold() for u in _sources(rec)]
        candidates += [k[len("path:"):].replace("\\", "/").casefold()
                       for k in rec["path_keys"] if k.startswith("path:")]
        for c in candidates:
            if fnmatch.fnmatchcase(c, pat) or fnmatch.fnmatchcase(c, "*/" + pat):
                out.append(rec)
                break
    return out


def plan_restore(store, arg: str) -> Dict:
    """What ``restore`` would lift for this argument. Lifts nothing."""
    records = liftable(store)
    if not arg or not arg.strip():
        return {"status": "error", "message": "Give a path to restore."}
    if is_pattern(arg):
        literal = arg.replace("*", "").replace("?", "").replace("[", "").replace("]", "")
        if not literal.strip("/\\. "):
            return {"status": "error", "mode": "bulk",
                    "message": "A pattern made only of wildcards would lift "
                               "everything. Give a narrower pattern, for "
                               "example 'projA/**'."}
        hits = _match_pattern(records, arg)
        return {"status": "ok", "mode": "bulk",
                "lifts": [_summarize(r) for r in hits]}
    hits = _match_one(records, arg)
    if hits:
        # an exact path, unit id or record id lifts every record bound to
        # it (the caller lists them first); only a bare name is ambiguous
        return {"status": "ok", "mode": "single",
                "lifts": [_summarize(r) for r in hits]}
    # A bare name: list candidates, never lift them.
    base = os.path.basename(arg).casefold()
    cands = [r for r in records if base and (
        r["unit_id"].casefold().endswith(base)
        or any(u.casefold().endswith(base) for u in _sources(r)))]
    if cands:
        return {"status": "ambiguous", "mode": "single",
                "candidates": [_summarize(r) for r in cands],
                "message": f"'{arg}' is a name, not a path. Restore by full "
                           "path so only that file is lifted."}
    return {"status": "none", "mode": "single", "lifts": [],
            "message": f"No active forget record matches '{arg}'."}


def restore(store, arg: str, actor: str = "cli:restore",
            confirmed: bool = True) -> Dict:
    """Lift what ``plan_restore`` lists. A bulk pattern lifts only when
    ``confirmed`` (the caller listed the plan first)."""
    plan = plan_restore(store, arg)
    if plan["status"] != "ok":
        return plan
    if plan["mode"] == "bulk" and not confirmed:
        plan["status"] = "needs_confirmation"
        return plan
    lifted = []
    for item in plan["lifts"]:
        rec = next(r for r in liftable(store) if r["id"] == item["id"])
        if store.lift_suppression(rec["id"], actor):
            _audit_lift(store, rec, actor, plan["mode"])
            lifted.append(item)
    plan["lifted"] = lifted
    plan["status"] = "restored" if lifted else "none"
    return plan

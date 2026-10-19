"""Restore: lift a suppression (docs/FORGET_SPEC.md §7).

``revien restore <path>`` lifts the suppression of ONE file. A bare name never
fans out across paths: if it names more than one suppression it is refused and
the candidates are listed. Lifting several at once needs an explicit pattern
(``revien restore 'projA/**'``), and what it would lift is listed first. Every
lift is audited. Only an explicit restore, or a re-import with the explicit
override flag, ever lifts a suppression.
"""

import fnmatch
import os
from typing import Dict, List, Optional

from revien.graph.units import identity_keys, path_unit_key, real_path

_WILD = set("*?[")


def is_pattern(arg: str) -> bool:
    return any(c in arg for c in _WILD)


def _audit_lift(store, record: Dict, actor: str, via: str) -> None:
    store.record_audit(
        f"suppression:{record['id']}", "restore", actor=actor,
        after={"unit_id": record["unit_id"], "kind": record["kind"],
               "via": via})


def _lift_linked_denylist(store, record: Dict, actor: str, via: str) -> None:
    """Lifting a source also lifts the forgotten-sentence denylist that was
    written with it: restoring a file means the user wants it back."""
    for w in store.list_suppressions("active", include_wording=True):
        if w["kind"] == "wording" and f"unit:{record['unit_id']}" in w["path_keys"]:
            if store.lift_suppression(w["id"], actor):
                _audit_lift(store, w, actor, via)


def lift_for_override(store, record: Dict) -> bool:
    """A re-import with the explicit override: lift and audit."""
    ok = store.lift_suppression(record["id"], "override:import")
    if ok:
        _audit_lift(store, record, "override:import", "override")
        _lift_linked_denylist(store, record, "override:import", "override")
    return ok


def liftable(store) -> List[Dict]:
    """What a restore can lift: every suppressed source, and the bans that a
    ``forget --everywhere`` put on an exact wording in every source (those are
    named ``forget:<id>``; the origin-bound denylist entries go with their
    source and are not listed on their own)."""
    return store.list_suppressions("active") + [
        w for w in store.list_suppressions("active", include_wording=True)
        if w["kind"] == "wording" and w.get("scope") == "global"]


def _summarize(rec: Dict) -> Dict:
    return {"id": rec["id"], "unit_id": rec["unit_id"], "kind": rec["kind"],
            "path": rec["path"], "created_at": rec["created_at"],
            "scope": rec.get("scope", "origin"),
            "sentences": len(rec.get("sentence_fps") or [])}


def _match_one(records: List[Dict], arg: str) -> List[Dict]:
    """Records that name exactly this file or unit. Never a name search
    across paths."""
    found: Dict[int, Dict] = {}
    exists = os.path.exists(arg)
    if exists:
        keys = set(identity_keys(arg))
        unit_ids = {"file:" + path_unit_key(arg), "skill:" + real_path(arg)}
        for rec in records:
            if rec["unit_id"] in unit_ids or keys & set(rec["path_keys"]):
                found[rec["id"]] = rec
        if found:
            return list(found.values())
    for rec in records:
        if rec["unit_id"] == arg:
            found[rec["id"]] = rec
        elif rec["path"] and os.path.normcase(rec["path"]).casefold() == \
                path_unit_key(arg):
            found[rec["id"]] = rec
    return list(found.values())


def _match_pattern(records: List[Dict], pattern: str) -> List[Dict]:
    pat = pattern.replace("\\", "/").casefold()
    out = []
    for rec in records:
        candidates = [rec["unit_id"].casefold()]
        if rec["path"]:
            candidates.append(rec["path"].replace("\\", "/").casefold())
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
        if len(hits) > 1 and not os.path.exists(arg):
            return {"status": "ambiguous", "mode": "single",
                    "candidates": [_summarize(r) for r in hits],
                    "message": f"'{arg}' names {len(hits)} suppressions. "
                               "Restore one by its full path, or lift several "
                               "with an explicit pattern."}
        return {"status": "ok", "mode": "single",
                "lifts": [_summarize(r) for r in hits]}
    # A bare name: list candidates, never lift them.
    base = os.path.basename(arg).casefold()
    cands = [r for r in records if base and (
        (r["path"] and os.path.basename(r["path"]).casefold() == base)
        or r["unit_id"].casefold().endswith(base))]
    if cands:
        return {"status": "ambiguous", "mode": "single",
                "candidates": [_summarize(r) for r in cands],
                "message": f"'{arg}' is a name, not a path. Restore by full "
                           "path so only that file is lifted."}
    return {"status": "none", "mode": "single", "lifts": [],
            "message": f"No active suppression matches '{arg}'."}


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
            _lift_linked_denylist(store, rec, actor, plan["mode"])
            lifted.append(item)
    plan["lifted"] = lifted
    plan["status"] = "restored" if lifted else "none"
    return plan

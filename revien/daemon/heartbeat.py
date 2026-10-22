"""What a running daemon says about itself, in the database it serves.

A daemon built with sync-reset fencing writes a heartbeat row into
``store_meta`` (key ``daemon_heartbeat:<pid>``): the features it has, its pid
and the time. It refreshes the row every ``HEARTBEAT_INTERVAL`` seconds and
removes it on a clean exit. A row older than ``STALE_AFTER`` seconds belongs
to a daemon that died without cleaning up and is ignored.

A daemon from before this change writes no heartbeat and its ``/v1/health``
carries no ``features`` list, which is how an operation that depends on
fencing (``revien sync reset``) recognises old code. The heartbeat answers a
different question from the health check: not "is something listening at
this address" but "which daemons are serving THIS database, and what can
they do". It holds no path, no content and no id derived from either.
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# What this build's daemon can do. "sync_reset_fence": its scheduler persists
# a sync position only if no reset happened meanwhile (sync_cursors.generation).
# "forget_suppression": every capture it makes goes through the suppression
# gate, so a forgotten unit does not come back through it.
DAEMON_FEATURES = ("sync_reset_fence", "usage_receipts", "forget_suppression",
                   "forget_literal")

# Features `revien sync reset` needs a daemon on the database to have.
RESET_REQUIRES = ("sync_reset_fence",)

# Features `revien forget` needs: a daemon without the gate would capture the
# forgotten source again, and one without the fence could undo a reset made
# because of it.
FORGET_REQUIRES = ("sync_reset_fence", "forget_suppression", "forget_literal")

HEARTBEAT_PREFIX = "daemon_heartbeat:"
HEARTBEAT_INTERVAL = 10.0   # seconds between refreshes
STALE_AFTER = 60.0          # seconds after which a row is ignored


def heartbeat_key(pid: Optional[int] = None) -> str:
    return f"{HEARTBEAT_PREFIX}{pid if pid is not None else os.getpid()}"


def write_heartbeat(store, features=DAEMON_FEATURES,
                    now: Optional[datetime] = None) -> None:
    """Record (or refresh) this process's heartbeat."""
    now = now or datetime.now(timezone.utc)
    store.set_meta(heartbeat_key(), json.dumps({
        "features": list(features),
        "pid": os.getpid(),
        "time": now.isoformat(),
    }))
    _drop_stale(store, now)


def clear_heartbeat(store) -> None:
    """Remove this process's heartbeat (clean shutdown). Never raises."""
    try:
        store.delete_meta(heartbeat_key())
    except Exception:  # noqa: BLE001 - shutting down; the row goes stale
        pass


def _parse(rows: Dict[str, str], now: datetime) -> List[dict]:
    live = []
    for _key, raw in rows.items():
        try:
            beat = json.loads(raw)
            when = datetime.fromisoformat(beat["time"])
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            feats = beat.get("features")
            pid = beat.get("pid")
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
        if not isinstance(feats, list):
            feats = []
        if now - when > timedelta(seconds=STALE_AFTER) or when - now > timedelta(
                seconds=STALE_AFTER):
            continue
        live.append({"pid": pid if isinstance(pid, int) else None,
                     "features": [f for f in feats if isinstance(f, str)],
                     "time": when.isoformat()})
    return live


def _drop_stale(store, now: datetime) -> None:
    for key, raw in store.list_meta(HEARTBEAT_PREFIX).items():
        if not _parse({key: raw}, now):
            store.delete_meta(key)


def live_heartbeats(store, now: Optional[datetime] = None) -> List[dict]:
    """Fresh heartbeats of daemons serving this database."""
    return _parse(store.list_meta(HEARTBEAT_PREFIX),
                  now or datetime.now(timezone.utc))


def live_heartbeats_readonly(readonly_uri: str,
                             now: Optional[datetime] = None) -> List[dict]:
    """Same, read through an already-built read-only SQLite URI, so a command
    can look before it opens (and so migrates) the database. A database with
    no ``store_meta`` table has no heartbeats."""
    try:
        conn = sqlite3.connect(readonly_uri, uri=True, timeout=1.0)
        try:
            rows = dict(conn.execute(
                "SELECT key, value FROM store_meta WHERE substr(key, 1, ?) = ?",
                (len(HEARTBEAT_PREFIX), HEARTBEAT_PREFIX)).fetchall())
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return _parse(rows, now or datetime.now(timezone.utc))


def daemons_lacking(store, need, exclude_pid: Optional[int] = None,
                    now: Optional[datetime] = None) -> List[dict]:
    """Live heartbeats (other than ``exclude_pid``) from daemons that do not
    report every feature in ``need``. A daemon refusing for its own sake
    passes its own pid so it does not refuse itself."""
    need = set(need)
    return [b for b in live_heartbeats(store, now)
            if b["pid"] != exclude_pid and not need <= set(b["features"])]

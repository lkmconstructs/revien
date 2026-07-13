"""
Interval snapshots of the Revien SQLite database — `revien watch`.

The OSS backup story: copy the live database to <db>.snapshots/<timestamp>.db
(or .db.gz) on a timer, retaining the newest N. Copies go through SQLite's
backup API (Connection.backup), which produces a consistent snapshot of a
live WAL database mid-write — a plain file copy of a hot db can capture a
torn state, so it is never used here.
"""

import gzip
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def snapshot_dir_for(db_path: PathLike) -> Path:
    """<db>.snapshots/ next to the database file."""
    p = Path(db_path)
    return p.with_name(p.name + ".snapshots")


def snapshot_db(
    db_path: PathLike,
    use_gzip: bool = False,
    out_dir: Optional[PathLike] = None,
) -> Path:
    """Take one consistent snapshot of the database. Returns the snapshot
    path (<timestamp>.db, or <timestamp>.db.gz with use_gzip)."""
    db = Path(db_path)
    out = Path(out_dir) if out_dir is not None else snapshot_dir_for(db)
    out.mkdir(parents=True, exist_ok=True)

    # Microseconds keep two snapshots in the same second distinct; UTC keeps
    # names monotonic across DST, so lexicographic order IS age order.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
    dest = out / f"{stamp}.db"

    # A failed or interrupted backup must not leave a torn snapshot behind —
    # it would count toward retention and read as a valid backup that isn't.
    try:
        src = sqlite3.connect(str(db))
        try:
            dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)  # safe on a live db; WAL contents included
            finally:
                dst.close()
        finally:
            src.close()

        if use_gzip:
            gz = Path(str(dest) + ".gz")
            try:
                with open(dest, "rb") as f_in, gzip.open(gz, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            except BaseException:
                gz.unlink(missing_ok=True)
                raise
            dest.unlink()
            return gz
        return dest
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def prune_snapshots(out_dir: PathLike, keep: int = 10) -> int:
    """Delete the oldest snapshots beyond `keep`. Returns how many were
    removed. Timestamped names sort lexicographically by age."""
    out = Path(out_dir)
    if keep < 1 or not out.is_dir():
        return 0
    snapshots = sorted(
        list(out.glob("*.db")) + list(out.glob("*.db.gz")),
        key=lambda p: p.name,
    )
    removed = 0
    for old in snapshots[:-keep] if len(snapshots) > keep else []:
        old.unlink()
        removed += 1
    return removed

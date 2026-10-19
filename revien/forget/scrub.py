"""Cutting forgotten wording out of stored text, and finding what is left.

``cut_text`` removes every occurrence of each forgotten string (case-insensitive,
Unicode NFC, longest string first). It cuts exactly the wording and nothing
else. ``find_residue`` is the check that follows a forget: it reads every text
column of every table (decoding the JSON ones, which escape non-ASCII and line
breaks, so a raw substring search would miss them) and reports where a
forgotten string still appears, instead of assuming it is gone.
"""

import json
import re
import sqlite3
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

MIN_CUT = 3  # shorter strings are never cut: they are not identifying wording


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_strings(strings: Iterable[str]) -> List[str]:
    """Distinct, stripped, at least MIN_CUT long, longest first."""
    seen, out = set(), []
    for s in strings:
        if not isinstance(s, str):
            continue
        s = _nfc(s).strip()
        if len(s) < MIN_CUT or s.casefold() in seen:
            continue
        seen.add(s.casefold())
        out.append(s)
    out.sort(key=len, reverse=True)
    return out


def _pattern(strings: List[str]):
    if not strings:
        return None
    return re.compile("|".join(re.escape(s) for s in strings), re.IGNORECASE)


def cut_text(text: str, strings: List[str]) -> Tuple[str, int]:
    """(text with every occurrence removed, number of occurrences)."""
    pat = _pattern(strings)
    if pat is None or not text:
        return text, 0
    normalized = _nfc(text)
    new, n = pat.subn("", normalized)
    if n == 0:
        return text, 0
    return new, n


def cut_json(value: Any, strings: List[str]) -> Tuple[Any, int]:
    """Cut inside every string of a decoded JSON value (keys left alone)."""
    pat = _pattern(strings)
    if pat is None:
        return value, 0
    total = 0

    def walk(v):
        nonlocal total
        if isinstance(v, str):
            new, n = pat.subn("", _nfc(v))
            if n:
                total += n
                return new
            return v
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk(value), total


def cut_json_text(raw: str, strings: List[str]) -> Tuple[str, int]:
    """Cut inside a JSON document held as text; non-JSON is cut as plain text."""
    if not raw:
        return raw, 0
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return cut_text(raw, strings)
    new, n = cut_json(value, strings)
    return (json.dumps(new) if n else raw), n


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
        try:
            inner = json.loads(value)
        except (TypeError, ValueError):
            inner = None
        if isinstance(inner, (dict, list)):
            yield from _iter_strings(inner)
    elif isinstance(value, list):
        for x in value:
            yield from _iter_strings(x)
    elif isinstance(value, dict):
        for x in value.values():
            yield from _iter_strings(x)


def contains(text: str, strings: List[str]) -> bool:
    pat = _pattern(strings)
    return bool(pat and text and pat.search(_nfc(text)))


# Tables whose rows are vector blobs or SQLite internals: not text stores.
_SKIP_PREFIXES = ("sqlite_", "vec_nodes_")


def find_residue(conn: sqlite3.Connection, strings: List[str]) -> Dict[str, int]:
    """{table: rows} where any forgotten string still appears in a text
    column. Reads the whole database; run once per forget."""
    pat = _pattern(strings)
    out: Dict[str, int] = {}
    if pat is None:
        return out
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND sql NOT LIKE "
        "'CREATE VIRTUAL%'")]
    for table in tables:
        if table.startswith(_SKIP_PREFIXES) or table == "vec_nodes":
            continue
        try:
            cur = conn.execute(f'SELECT * FROM "{table}"')
        except sqlite3.Error:
            continue
        hits = 0
        for row in cur:
            for v in row:
                if isinstance(v, str) and any(pat.search(_nfc(s)) for s in _iter_strings(v)):
                    hits += 1
                    break
                if isinstance(v, (bytes, bytearray)):
                    try:
                        if pat.search(_nfc(bytes(v).decode("utf-8", "ignore"))):
                            hits += 1
                            break
                    except Exception:  # noqa: BLE001
                        pass
        if hits:
            out[table] = hits
    return out

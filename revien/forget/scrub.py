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


def _bounded(s: str) -> str:
    """A regex for ``s`` as whole words: a word character at either end must
    not touch another word character, so 'API' never matches inside 'rapid'
    or 'Therapists'. Unicode-aware (\\w covers letters of every script)."""
    pre = r"(?<!\w)" if re.match(r"\w", s[:1]) else ""
    post = r"(?!\w)" if re.match(r"\w", s[-1:]) else ""
    return pre + re.escape(s) + post


def _pattern(strings: List[str]):
    if not strings:
        return None
    return re.compile("|".join(_bounded(s) for s in strings), re.IGNORECASE)


_TERMINATORS = ".!?"
_SPEAKER = re.compile(r"^(?:User|Assistant|Human|AI|System)\s*:\s*", re.IGNORECASE)


def _sentence_span(text: str, start: int, end: int):
    """The sentence around text[start:end]: from after the previous sentence
    end (or the start of the line) to the next sentence end (or the end of the
    line). A '.', '!' or '?' ends a sentence only when whitespace or the end of
    the text follows it, so 'blue.example.com' or '3.5' stay whole."""
    left = start
    while left > 0:
        ch = text[left - 1]
        if ch == "\n":
            break
        if ch.isspace() and left - 2 >= 0 and text[left - 2] in _TERMINATORS:
            break
        left -= 1
    right = end
    n = len(text)
    if end > start and text[end - 1] in _TERMINATORS and (end >= n or text[end].isspace()):
        # the wording already ends its sentence: do not run on into the next
        return left, right
    while right < n:
        ch = text[right]
        if ch == "\n":
            break
        if ch in _TERMINATORS and (right + 1 >= n or text[right + 1].isspace()):
            right += 1
            break
        right += 1
    return left, right


def _remove_spans(text: str, spans) -> str:
    spans.sort()
    merged = []
    for l, r in spans:
        if merged and l <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], r)
        else:
            merged.append([l, r])
    out, pos = [], 0
    for l, r in merged:
        line_start = text.rfind("\n", 0, l) + 1
        prefix = text[line_start:l]
        # a speaker label with nothing left to say goes with its sentence
        if prefix.strip() and _SPEAKER.fullmatch(prefix.strip() + " "):
            l = line_start
            prefix = ""
        line_end = text.find("\n", r)
        line_end = len(text) if line_end == -1 else line_end
        # the whole line was the sentence: take its line break too
        if not prefix.strip() and not text[r:line_end].strip():
            r = min(line_end + 1, len(text))
        elif not prefix.strip():
            # the sentence opened its line: the space before the next one goes
            while r < len(text) and text[r] in " \t":
                r += 1
        out.append(text[pos:l])
        pos = r
    out.append(text[pos:])
    return "".join(out)


MIN_SENTENCE = 20   # shorter sentences ("Noted.") are not on the global denylist
MIN_SHORT = 8       # ...they can be kept out of their own source, by identity


def iter_sentences(text: str):
    """(start, end, sentence) for each sentence of ``text``, in order. A
    sentence is a stretch of one line ending at '.', '!' or '?' followed by
    whitespace (or at the end of the line). A leading speaker label ("User:")
    is not part of the sentence text, so the same words fingerprint the same
    whether or not a transcript labels them."""
    pos = 0
    for line in text.split("\n"):
        line_start = pos
        pos += len(line) + 1
        i, n = 0, len(line)
        while i < n:
            j = i
            while j < n:
                if line[j] in _TERMINATORS and (j + 1 >= n or line[j + 1].isspace()):
                    j += 1
                    break
                j += 1
            raw = line[i:j]
            body = raw.strip()
            m = _SPEAKER.match(body)
            if m:
                body = body[m.end():]
            if body:
                yield line_start + i, line_start + j, body
            i = j
            while i < n and line[i].isspace():
                i += 1


def sentence_key(sentence: str) -> str:
    """What gets fingerprinted for a sentence: NFC, whitespace collapsed."""
    return " ".join(_nfc(sentence).split())


def strip_sentences(text: str, is_forgotten, min_len: int = MIN_SENTENCE) -> Tuple[str, int]:
    """Remove every sentence for which ``is_forgotten(sentence_key)`` is true
    (sentences shorter than ``min_len`` are never asked about). Returns
    (text without them, how many)."""
    spans = []
    for l, r, body in iter_sentences(text):
        key = sentence_key(body)
        if len(key) >= min_len and is_forgotten(key):
            spans.append([l, r])
    if not spans:
        return text, 0
    return _remove_spans(text, [tuple(x) for x in spans]), len(spans)


def cut_text(text: str, strings: List[str], sentences: bool = True) -> Tuple[str, int]:
    """(text with the forgotten wording cut out, number of occurrences).

    With ``sentences`` (the default) the whole sentence around each occurrence
    goes, because a forgotten sentence is what the user asked to lose
    (docs/FORGET_SPEC.md §2); a half-cut sentence would leave the rest of the
    secret behind. ``sentences=False`` cuts the matched characters only."""
    pat = _pattern(strings)
    if pat is None or not text:
        return text, 0
    normalized = _nfc(text)
    spans = []
    for m in pat.finditer(normalized):
        if sentences:
            spans.append(list(_sentence_span(normalized, m.start(), m.end())))
        else:
            spans.append([m.start(), m.end()])
    if not spans:
        return text, 0
    n = len(spans)
    return _remove_spans(normalized, [tuple(s) for s in spans]), n


def cut_json(value: Any, strings: List[str]) -> Tuple[Any, int]:
    """Cut inside every string of a decoded JSON value (keys left alone)."""
    total = 0

    def walk(v):
        nonlocal total
        if isinstance(v, str):
            new, n = cut_text(v, strings)
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


_WORD = re.compile(r"\w+", re.UNICODE)
MIN_TOKEN = 5


def tokens_of(text: str) -> set:
    return {w for w in _WORD.findall(_nfc(text or "").casefold()) if len(w) >= MIN_TOKEN}


def distinctive_tokens(conn: sqlite3.Connection, strings: List[str],
                       skip_side_fields=frozenset()) -> List[str]:
    """Words of the forgotten text that no surviving node contains.

    Recall echoes (a query typed before the forget, a candidate's reason) quote
    pieces of what was said, not whole sentences. A word that only the
    forgotten text had is what identifies such an echo; a word the rest of the
    graph still uses ("postgres") identifies nothing, so it is left alone. Call
    this AFTER the forgotten nodes are gone."""
    forgotten = set()
    for s in strings:
        forgotten |= tokens_of(s)
    if not forgotten:
        return []
    alive = set()
    for nid, label, content, meta in conn.execute(
            "SELECT node_id, label, content, metadata FROM nodes"):
        # The side fields of a node being cut are about to be cut with these
        # very words, so they cannot vouch for a word staying.
        side = "" if nid in skip_side_fields else (meta or "")
        alive |= tokens_of(f"{label}\n{content}\n{side}")
        if not (forgotten - alive):
            return []
    return sorted(forgotten - alive, key=len, reverse=True)


def compile_strings(strings: List[str]):
    """One compiled whole-word pattern for ``strings`` (None when empty)."""
    return _pattern(strings)


def nfc(text: str) -> str:
    return _nfc(text)


def contains(text: str, strings: List[str]) -> bool:
    pat = _pattern(strings)
    return bool(pat and text and pat.search(_nfc(text)))


# Tables whose rows are vector blobs or SQLite internals: not text stores.
_SKIP_PREFIXES = ("sqlite_", "vec_nodes_")


# Tables that hold identities, fingerprints and positions, never content.
NOT_CONTENT_TABLES = frozenset({
    "node_provenance", "capture_units", "suppressions", "store_meta",
    "sync_cursors", "distill_manifest"})


# Columns Revien writes itself, never the user's words: the audit op and actor
# ("forget", "api:forget", "cli:forget", a scheduler name). The one exception
# is the actor of a merge entry, which records the merged candidate's label.
_OWN_COLUMNS = {"audit_log": ("op", "actor")}
_MERGE_OPS = ("normalized_merge", "semantic_merge")


def _own_columns(table: str, cols: List[str]) -> set:
    return {cols.index(c) for c in _OWN_COLUMNS.get(table, ()) if c in cols}


def _is_content_cell(table: str, cols: List[str], row, i: int) -> bool:
    return (table == "audit_log" and cols[i] == "actor"
            and "op" in cols and row[cols.index("op")] in _MERGE_OPS)


def find_residue(conn: sqlite3.Connection, strings: List[str],
                 ignore=frozenset()) -> Dict[str, int]:
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
        if table.startswith(_SKIP_PREFIXES) or table == "vec_nodes" or table in ignore:
            continue
        try:
            cur = conn.execute(f'SELECT * FROM "{table}"')
        except sqlite3.Error:
            continue
        cols = [d[0] for d in cur.description]
        skip = _own_columns(table, cols)
        hits = 0
        for row in cur:
            for i, v in enumerate(row):
                if i in skip and not _is_content_cell(table, cols, row, i):
                    continue
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

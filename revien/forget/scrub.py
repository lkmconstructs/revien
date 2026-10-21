"""Cutting forgotten wording out of stored text, and finding what is left.

``cut_text`` removes every occurrence of each forgotten string (case-insensitive,
Unicode NFC, longest string first). It cuts exactly the wording and nothing
else. ``find_residue`` is the check that follows a forget: it reads every text
column of every table (decoding the JSON ones, which escape non-ASCII and line
breaks, so a raw substring search would miss them) and reports where a
forgotten string still appears, instead of assuming it is gone.
"""

import json
from typing import List, Optional
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


class Literal(str):
    """A string to cut as exact characters (``forget --wording``): case
    sensitive, no word boundaries (a secret sits inside ``?token=...&x`` and
    ``ghp_...``), and tolerant of a line break the text wrapped it at."""


_WRAP = re.compile(r"\r?\n[ \t]*")


def canonical(text: str) -> str:
    """The text with line wraps (a newline and the indent after it) removed:
    what a secret looks like however it was wrapped."""
    return _WRAP.sub("", text)


#: A line break (and the indent after it) is treated as a wrap of one long
#: line when the line before it has at least this many characters. After a
#: shorter line ("token" / "abc") the two are separate texts and are never
#: joined. The rule is a heuristic, stated exactly: a named string that
#: coincides with a join after a long line is cut across it.
MIN_WRAPPED_LINE = 20

#: A piece of a secret at least this long (and a quarter of it) left at the
#: edge of a field is reported, never cut: only the whole string is cut.
MIN_EDGE = 8


def unwrap(text: str) -> Tuple[str, Optional[List[int]]]:
    """(text with real wraps removed, original offset of every kept character).
    The offsets are None when nothing was removed. A wrap is a newline (and
    the indent after it) that ends a line of MIN_WRAPPED_LINE characters or
    more and is followed by more text on the next line."""
    breaks = []
    for m in _WRAP.finditer(text):
        start = text.rfind("\n", 0, m.start()) + 1
        if m.start() - start >= MIN_WRAPPED_LINE and m.end() < len(text) \
                and text[m.end()] not in "\r\n":
            breaks.append((m.start(), m.end()))
    if not breaks:
        return text, None
    kept, pos = [], 0
    for s, e in breaks:
        kept.extend(range(pos, s))
        pos = e
    kept.extend(range(pos, len(text)))
    return "".join(text[i] for i in kept), kept


class _Hit:
    def __init__(self, s, e):
        self._s, self._e = s, e

    def start(self):
        return self._s

    def end(self):
        return self._e


class _Pat:
    """A compiled pattern and how to normalise text before searching it."""

    def __init__(self, rx, literal: bool, parts=()):
        self.rx, self.literal, self.parts = rx, literal, list(parts)

    def norm(self, text: str) -> str:
        return text if self.literal else _nfc(text)

    def finditer(self, text):
        if not self.literal:
            return self.rx.finditer(text)
        canon, kept = unwrap(text)
        if kept is None:
            return self.rx.finditer(text)
        # offsets in the canonical text map back to the original: a match
        # that spans a wrap takes the wrap with it, and nothing else
        return [_Hit(kept[m.start()], kept[m.end() - 1] + 1)
                for m in self.rx.finditer(canon)]

    def search(self, text):
        for m in self.finditer(text):
            return m
        return None


def _pattern(strings: List[str]):
    if not strings:
        return None
    lits = [s for s in strings if isinstance(s, Literal)]
    if lits:
        if len(lits) != len(strings):
            raise ValueError("literal and word strings cannot be mixed")
        parts = sorted({canonical(s) for s in lits if canonical(s)}, key=len, reverse=True)
        if not parts:
            return None
        return _Pat(re.compile("|".join(re.escape(part) for part in parts)), True, parts)
    return _Pat(re.compile("|".join(_bounded(s) for s in strings), re.IGNORECASE), False)


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
    normalized = pat.norm(text)
    if pat.literal:
        # exactly the characters: nothing around them goes
        out, pos, n = [], 0, 0
        for m in pat.finditer(normalized):
            out.append(normalized[pos:m.start()])
            pos, n = m.end(), n + 1
        out.append(normalized[pos:])
        if not n:
            return text, 0
        return "".join(out), n
    spans = []
    for m in pat.finditer(normalized):
        if sentences and not pat.literal:
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


def edge_fragments(text: str, part: str, floor: int, report: bool = False):
    """[("end"|"start", length)] of the pieces of the string ``part`` that sit
    at the very edge of ``text``: the text ENDS with a proper prefix of it
    (where a label was cut off at the end of a line), or STARTS with a proper
    suffix of it. Only pieces of ``floor`` characters or more count. With
    ``report`` the other two edges (starts with a prefix, ends with a suffix)
    are looked at too: those are never cut, only reported."""
    found = []
    if not text or not part:
        return found
    tail, head = text.rstrip(), text.lstrip()
    for k in range(len(part) - 1, floor - 1, -1):
        if tail.endswith(part[:k]):
            found.append(("end", k))
            break
    for k in range(len(part) - 1, floor - 1, -1):
        if head.startswith(part[-k:]):
            found.append(("start", k))
            break
    if report:
        for k in range(len(part) - 1, floor - 1, -1):
            if head.startswith(part[:k]):
                found.append(("start-of-string", k))
                break
        for k in range(len(part) - 1, floor - 1, -1):
            if tail.endswith(part[-k:]):
                found.append(("end-of-string", k))
                break
    return found


def edge_floor(part: str) -> int:
    """The shortest piece that is cut: MIN_EDGE characters and a quarter of
    the string. Shorter pieces are only ever reported."""
    return max(MIN_EDGE, (len(part) + 3) // 4)


def aligned_edge_fragment(field: str, content: str, spans, part: str):
    """The piece of ``part`` that ``field`` (a label, a preview, a context
    line) kept of an occurrence cut from ``content``, or "".

    Exactness: the field must be a verbatim stretch of the content, and the
    piece must sit exactly where the field's text stops inside the occurrence
    (the field ends with the first k characters of an occurrence that starts
    right there), or begins exactly where the field's text starts inside one
    (it begins with the last k characters of an occurrence that ends right
    there). ``spans`` are the (start, end) of the occurrences in the original
    content. A piece that is merely equal to a prefix or suffix of the string
    somewhere in the field is not this, and is never cut."""
    t = (field or "").strip()
    if not t or not content or not spans:
        return ""
    floor = edge_floor(part)
    for kind, k in edge_fragments(t, part, floor):
        i = content.find(t)
        while i >= 0:
            if kind == "end":
                pos = i + len(t) - k
                if any(s0 == pos and e0 >= pos + k for s0, e0 in spans):
                    return part[:k]
            else:
                end = i + k
                if any(e0 == end and s0 <= i for s0, e0 in spans):
                    return part[-k:]
            i = content.find(t, i + 1)
    return ""


def cut_edge_fragment(text: str, part: str, fragment=None):
    """(text without the piece of ``part`` at its edge, the piece cut or "").
    The piece is a proper prefix of ``part`` at the end of the text, or a
    proper suffix at the start. With ``fragment`` only that exact piece is
    cut (an audit snapshot of the same field)."""
    if not isinstance(text, str) or not text:
        return text, ""
    if fragment is not None:
        t = text.rstrip()
        if t.endswith(fragment) and part.startswith(fragment):
            return t[:len(t) - len(fragment)], fragment
        h = text.lstrip()
        if h.startswith(fragment) and part.endswith(fragment):
            return h[len(fragment):], fragment
        return text, ""
    for kind, k in edge_fragments(text, part, edge_floor(part)):
        if kind == "end":
            t = text.rstrip()
            return t[:len(t) - k], part[:k]
        h = text.lstrip()
        return h[k:], part[-k:]
    return text, ""


def inside_longer_words(raw: str, strings: List[str], is_json: bool = False) -> int:
    """How many occurrences of a literal string sit INSIDE a longer word: a
    letter or digit right before it (when it starts with one) or right after
    it (when it ends with one). 'secret' in 'secretary' counts."""
    pat = _pattern(strings)
    if pat is None or not pat.literal or not raw:
        return 0
    lit = str(strings[0])
    first, last = lit[:1], lit[-1:]
    texts = [raw]
    if is_json:
        try:
            texts = list(_iter_strings(json.loads(raw)))
        except (TypeError, ValueError):
            texts = [raw]
    n = 0
    for t in texts:
        for m in pat.finditer(t):
            a, b = m.start(), m.end()
            before = t[a - 1] if a > 0 else ""
            after = t[b] if b < len(t) else ""
            if (first.isalnum() and before.isalnum()) or (last.isalnum() and after.isalnum()):
                n += 1
    return n


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
    return bool(pat and text and pat.search(pat.norm(text)))


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
                if isinstance(v, str) and any(pat.search(pat.norm(s)) for s in _iter_strings(v)):
                    hits += 1
                    break
                if isinstance(v, (bytes, bytearray)):
                    try:
                        if pat.search(pat.norm(bytes(v).decode("utf-8", "ignore"))):
                            hits += 1
                            break
                    except Exception:  # noqa: BLE001
                        pass
        if hits:
            out[table] = hits
    return out

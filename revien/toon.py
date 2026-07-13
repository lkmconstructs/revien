"""
TOON (Token-Oriented Object Notation) — optional wire format for recall
payloads (LEG P2). Independent Python port of the FORMAT from the TOON spec
(https://github.com/toon-format/spec, reference implementation
@toon-format/toon on npm, MIT). No JS dependency is taken; this module is
the implementation.

Implemented subset (everything the recall response schema needs, stated per
the leg's scope):

  - objects: ``key: value`` lines, nesting by 2-space indentation (spec §8)
  - inline arrays of primitives: ``key[N]: v1,v2`` (§9.2)
  - tabular arrays — uniform arrays of objects whose values are all
    primitive: ``key[N]{f1,f2}:`` header + one comma-delimited row per
    element at depth +1 (§9.3)
  - list arrays for non-uniform or nested element shapes: ``key[N]:`` with
    ``- `` items at depth +1 (§9.4). Recall ``results`` serialize this way
    because each result nests ``score_breakdown`` (object) and ``path``
    (array), which disqualifies the row from tabular form under §9.3's
    all-primitive rule.
  - empty arrays: ``key: []`` (legacy ``key[0]:`` accepted on parse, §9.1)
  - string quoting/escaping per §7.1–§7.2 (``\\`` ``\"`` ``\n`` ``\r``
    ``\t``, ``\\uXXXX`` for other control characters; strings quoted when
    empty, padded, keyword-like, numeric-like, hyphen-led, or containing
    the delimiter / colon / quote / backslash / brackets / braces /
    control chars)
  - key quoting per §7.3 (unquoted iff ``^[A-Za-z_][A-Za-z0-9_.]*$``)
  - scalars: numbers, ``true`` / ``false`` / ``null`` (§2)

Not implemented (not needed by the recall schema): alternative delimiters
(comma only — no tab/pipe headers), key folding, root-level arrays,
length-marker-free streaming forms.

Float precision — the documented choice: floats are emitted with Python's
``repr()``, the shortest representation that round-trips exactly through
``float()`` (IEEE-754 shortest-repr). One documented deviation from the
spec's canonical number form: integral floats keep their ``.0`` (``1.0``
stays ``"1.0"``, not ``"1"``) so a JSON float never comes back as an int —
the round-trip test asserts type-preserving equality. Decoders must accept
``1.0`` anyway, so output remains valid TOON. Non-finite floats (NaN/Inf)
raise, exactly as they would under strict JSON.

Recall flattening convention (documented, measured, lossless): TOON's token
win lives in the tabular form, and recall results are disqualified from it
by two nested fields — measured on real payloads, plain list-form TOON is
LARGER than compact JSON. So ``serialize_recall`` applies a recall-specific,
exactly-invertible reshape when (and only when) the payload matches the
recall schema precisely (top-level keys ``query, results, nodes_examined,
retrieval_time_ms, semantic_active, semantic_note`` in order; every result
``node_id, node_type, label, content, score, score_breakdown, path`` in
order; breakdown key order uniform across results; leaf values primitive):

  - each result's ``score_breakdown`` dict is flattened into dotted columns
    (``score_breakdown.recency`` … — dots are legal in unquoted keys, §7.3),
    making rows all-primitive and therefore tabular under §9.3;
  - each result's ``path`` array moves to a parallel top-level ``paths[N]``
    list array (one inline primitive array per result, same order).

``parse_recall`` detects the reshape by the reserved top-level ``paths``
key and inverts it exactly; payloads that do not match the schema (e.g.
``include_tensions=True``, empty results) are encoded in plain
spec-generic form, which is equally lossless. The output is valid TOON
either way — the reshape is a schema convention of THIS wire format, not a
spec extension.
"""

import re
from typing import Any, Dict, List, Tuple

DELIM = ","
INDENT = "  "

_UNQUOTED_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_NUMERIC_LIKE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+$")
# Characters that force quoting of an unquoted string value (§7.2).
_FORBIDDEN_IN_BARE = ':"\\[]{}' + DELIM


class ToonError(ValueError):
    """Raised on unencodable input or malformed TOON text."""


# ── Encoder ───────────────────────────────────────────────


def _is_primitive(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _escape(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _needs_quotes(s: str) -> bool:
    if s == "" or s != s.strip():
        return True
    if s in ("true", "false", "null"):
        return True
    if _NUMERIC_LIKE_RE.match(s):
        return True
    if s.startswith("-"):
        return True
    if any(ch in s for ch in _FORBIDDEN_IN_BARE):
        return True
    return any(ord(ch) < 0x20 for ch in s)


def _encode_string(s: str) -> str:
    if _needs_quotes(s):
        return '"' + _escape(s) + '"'
    return s


def _encode_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise ToonError("non-finite float is not representable: %r" % v)
        # repr() = shortest exact round-trip form; keeps '.0' on integral
        # floats (documented deviation from canonical form — type fidelity).
        return repr(v)
    if isinstance(v, str):
        return _encode_string(v)
    raise ToonError("not a TOON scalar: %r" % (type(v),))


def _encode_key(k: Any) -> str:
    if not isinstance(k, str):
        raise ToonError("TOON object keys must be strings, got %r" % (type(k),))
    if _UNQUOTED_KEY_RE.match(k):
        return k
    return '"' + _escape(k) + '"'


def _is_tabular(arr: List[Any]) -> bool:
    """§9.3: every element a non-empty object, identical key sequence,
    all values primitive."""
    if not arr:
        return False
    if not all(isinstance(v, dict) and v for v in arr):
        return False
    keys = list(arr[0].keys())
    for v in arr:
        if list(v.keys()) != keys:
            return False
        if not all(_is_primitive(x) for x in v.values()):
            return False
    return True


def _encode_field(key: str, value: Any, depth: int, lines: List[str]) -> None:
    pad = INDENT * depth
    ek = _encode_key(key)
    if _is_primitive(value):
        lines.append("%s%s: %s" % (pad, ek, _encode_scalar(value)))
    elif isinstance(value, dict):
        lines.append("%s%s:" % (pad, ek))
        _encode_object(value, depth + 1, lines)
    elif isinstance(value, (list, tuple)):
        _encode_array(ek, list(value), depth, lines)
    else:
        raise ToonError("unencodable value type: %r" % (type(value),))


def _encode_array(ek: str, arr: List[Any], depth: int, lines: List[str]) -> None:
    pad = INDENT * depth
    if not arr:
        lines.append("%s%s: []" % (pad, ek))
        return
    if all(_is_primitive(v) for v in arr):
        cells = DELIM.join(_encode_scalar(v) for v in arr)
        lines.append("%s%s[%d]: %s" % (pad, ek, len(arr), cells))
        return
    if _is_tabular(arr):
        fields = list(arr[0].keys())
        header = DELIM.join(_encode_key(f) for f in fields)
        lines.append("%s%s[%d]{%s}:" % (pad, ek, len(arr), header))
        rowpad = INDENT * (depth + 1)
        for row in arr:
            lines.append(rowpad + DELIM.join(_encode_scalar(row[f]) for f in fields))
        return
    # List form (§9.4) — non-uniform or nested elements.
    lines.append("%s%s[%d]:" % (pad, ek, len(arr)))
    for item in arr:
        _encode_list_item(item, depth + 1, lines)


def _encode_list_item(item: Any, depth: int, lines: List[str]) -> None:
    pad = INDENT * depth
    if _is_primitive(item):
        lines.append("%s- %s" % (pad, _encode_scalar(item)))
        return
    if isinstance(item, dict):
        if not item:
            lines.append("%s- {}" % pad)
            return
        # First field rides the hyphen line; the "- " prefix is exactly one
        # indent unit wide, so continuation fields at depth+1 align with it.
        sub: List[str] = []
        items_iter = iter(item.items())
        k0, v0 = next(items_iter)
        _encode_field(k0, v0, depth + 1, sub)
        lines.append("%s- %s" % (pad, sub[0][len(INDENT * (depth + 1)):]))
        lines.extend(sub[1:])
        for k, v in items_iter:
            _encode_field(k, v, depth + 1, lines)
        return
    if isinstance(item, (list, tuple)):
        arr = list(item)
        if not arr:
            lines.append("%s- []" % pad)
            return
        if all(_is_primitive(v) for v in arr):
            cells = DELIM.join(_encode_scalar(v) for v in arr)
            lines.append("%s- [%d]: %s" % (pad, len(arr), cells))
            return
        lines.append("%s- [%d]:" % (pad, len(arr)))
        for it in arr:
            _encode_list_item(it, depth + 1, lines)
        return
    raise ToonError("unencodable list item type: %r" % (type(item),))


def _encode_object(obj: Dict[str, Any], depth: int, lines: List[str]) -> None:
    for k, v in obj.items():
        _encode_field(k, v, depth, lines)


def encode(obj: Dict[str, Any]) -> str:
    """Encode a dict (root object document) to TOON text. No trailing
    newline (spec §12)."""
    if not isinstance(obj, dict):
        raise ToonError("TOON root must be an object (dict)")
    lines: List[str] = []
    _encode_object(obj, 0, lines)
    return "\n".join(lines)


# ── Decoder ───────────────────────────────────────────────


def _read_quoted(s: str, i: int) -> Tuple[str, int]:
    """Read a quoted string starting at s[i] == '\"'. Returns (value, index
    after closing quote)."""
    if s[i] != '"':
        raise ToonError("expected opening quote at %r" % s[i:])
    out = []
    i += 1
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == '"':
            return "".join(out), i + 1
        if ch == "\\":
            if i + 1 >= n:
                raise ToonError("dangling escape in %r" % s)
            e = s[i + 1]
            if e == "\\":
                out.append("\\")
            elif e == '"':
                out.append('"')
            elif e == "n":
                out.append("\n")
            elif e == "r":
                out.append("\r")
            elif e == "t":
                out.append("\t")
            elif e == "u":
                if i + 6 > n:
                    raise ToonError("truncated \\u escape in %r" % s)
                cp = int(s[i + 2:i + 6], 16)
                if 0xD800 <= cp <= 0xDFFF:
                    raise ToonError("lone surrogate in \\u escape")
                out.append(chr(cp))
                i += 6
                continue
            else:
                raise ToonError("invalid escape \\%s" % e)
            i += 2
            continue
        out.append(ch)
        i += 1
    raise ToonError("unterminated quoted string: %r" % s)


def _split_cells(s: str) -> List[str]:
    """Split a delimited cell list on the active (comma) delimiter,
    respecting quoted cells."""
    cells: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(s)
    in_q = False
    while i < n:
        ch = s[i]
        if in_q:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_q = False
            i += 1
            continue
        if ch == '"':
            in_q = True
            buf.append(ch)
            i += 1
            continue
        if ch == DELIM:
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if in_q:
        raise ToonError("unterminated quoted cell: %r" % s)
    cells.append("".join(buf).strip())
    return cells


def _parse_scalar(tok: str) -> Any:
    if tok.startswith('"'):
        val, idx = _read_quoted(tok, 0)
        if idx != len(tok):
            raise ToonError("trailing garbage after quoted string: %r" % tok)
        return val
    if tok == "null":
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    if _INT_RE.match(tok):
        return int(tok)
    if _NUMERIC_LIKE_RE.match(tok):
        return float(tok)
    if tok == "" or any(ch in tok for ch in _FORBIDDEN_IN_BARE):
        raise ToonError("malformed bare scalar: %r" % tok)
    return tok


def _parse_field_key(tok: str) -> str:
    """A tabular header field: quoted key or bare key."""
    if tok.startswith('"'):
        val, idx = _read_quoted(tok, 0)
        if idx != len(tok):
            raise ToonError("trailing garbage after quoted key: %r" % tok)
        return val
    if not _UNQUOTED_KEY_RE.match(tok):
        raise ToonError("invalid unquoted key: %r" % tok)
    return tok


def _parse_key(content: str) -> Tuple[str, str]:
    """Parse the leading key of a field line. Returns (key, rest) where
    rest begins with ':' or '['."""
    if content.startswith('"'):
        key, idx = _read_quoted(content, 0)
        return key, content[idx:]
    m = re.match(r"^[^:\[]+", content)
    if not m:
        raise ToonError("missing key in line: %r" % content)
    key = m.group(0)
    if not _UNQUOTED_KEY_RE.match(key):
        raise ToonError("invalid unquoted key: %r" % key)
    return key, content[len(key):]


def _read_braces(s: str) -> Tuple[str, str]:
    """s starts with '{'. Returns (inner, rest_after_closing_brace),
    respecting quoted field names."""
    if not s.startswith("{"):
        raise ToonError("expected '{' at %r" % s)
    i = 1
    n = len(s)
    in_q = False
    while i < n:
        ch = s[i]
        if in_q:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_q = False
            i += 1
            continue
        if ch == '"':
            in_q = True
            i += 1
            continue
        if ch == "}":
            return s[1:i], s[i + 1:]
        i += 1
    raise ToonError("unterminated field header: %r" % s)


class _Parser:
    def __init__(self, lines: List[Tuple[int, str]]):
        self.lines = lines
        self.i = 0

    def peek(self):
        return self.lines[self.i] if self.i < len(self.lines) else None

    def parse_object(self, depth: int) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        while True:
            nxt = self.peek()
            if nxt is None or nxt[0] < depth:
                break
            if nxt[0] > depth:
                raise ToonError("unexpected indent at: %r" % nxt[1])
            if nxt[1].startswith("- "):
                break  # list item belongs to an enclosing array
            self.i += 1
            key, value = self.parse_field(nxt[1], depth)
            obj[key] = value
        return obj

    def parse_field(self, content: str, depth: int) -> Tuple[str, Any]:
        key, rest = _parse_key(content)
        if rest.startswith("["):
            return key, self.parse_array_header(rest, depth)
        if rest == ":":
            nxt = self.peek()
            if nxt is not None and nxt[0] > depth and not nxt[1].startswith("- "):
                return key, self.parse_object(depth + 1)
            return key, {}  # empty object (§8)
        if rest.startswith(":"):
            valstr = rest[1:].strip()
            if valstr == "[]":
                return key, []
            return key, _parse_scalar(valstr)
        raise ToonError("malformed field line: %r" % content)

    def parse_array_header(self, rest: str, depth: int) -> List[Any]:
        m = re.match(r"^\[(\d+)\]", rest)
        if not m:
            raise ToonError("malformed array header: %r" % rest)
        n = int(m.group(1))
        rest2 = rest[m.end():]
        if rest2.startswith("{"):
            inner, rest3 = _read_braces(rest2)
            if rest3 != ":":
                raise ToonError("malformed tabular header: %r" % rest)
            fields = [_parse_field_key(c) for c in _split_cells(inner)]
            rows: List[Dict[str, Any]] = []
            for _ in range(n):
                nxt = self.peek()
                if nxt is None or nxt[0] != depth + 1:
                    raise ToonError("tabular row count mismatch (expected %d rows)" % n)
                self.i += 1
                cells = _split_cells(nxt[1])
                if len(cells) != len(fields):
                    raise ToonError("tabular row width mismatch: %r" % nxt[1])
                rows.append({f: _parse_scalar(c) for f, c in zip(fields, cells)})
            return rows
        if rest2 == ":":
            if n == 0:
                return []  # legacy empty form key[0]:
            return self.parse_list_items(n, depth + 1)
        if rest2.startswith(":"):
            inline = rest2[1:].strip()
            cells = _split_cells(inline)
            if len(cells) != n:
                raise ToonError(
                    "inline array length mismatch: declared %d, found %d" % (n, len(cells))
                )
            return [_parse_scalar(c) for c in cells]
        raise ToonError("malformed array header: %r" % rest)

    def parse_list_items(self, n: int, depth: int) -> List[Any]:
        items: List[Any] = []
        for _ in range(n):
            nxt = self.peek()
            if nxt is None or nxt[0] != depth or not nxt[1].startswith("- "):
                raise ToonError("list array item count mismatch (expected %d items)" % n)
            self.i += 1
            items.append(self.parse_list_item(nxt[1][2:], depth))
        return items

    def parse_list_item(self, rest: str, depth: int) -> Any:
        if rest == "{}":
            return {}
        if rest == "[]":
            return []
        if rest.startswith("["):
            return self.parse_array_header(rest, depth)
        # Distinguish object-first-field from bare primitive: an unquoted
        # primitive can never contain ':' or '[' (quoting rules force
        # quotes), and a quoted primitive consumes the whole token.
        if rest.startswith('"'):
            _, idx = _read_quoted(rest, 0)
            is_field = idx < len(rest) and rest[idx] in ":["
        else:
            is_field = (":" in rest) or ("[" in rest)
        if not is_field:
            return _parse_scalar(rest)
        # Object item: first field on the hyphen line (its own children sit
        # one level deeper than the hyphen), remaining fields at depth+1.
        key, value = self.parse_field(rest, depth + 1)
        obj: Dict[str, Any] = {key: value}
        obj.update(self.parse_object(depth + 1))
        return obj


def decode(text: str) -> Dict[str, Any]:
    """Parse TOON text (root object document, this module's subset) back to
    a dict."""
    lines: List[Tuple[int, str]] = []
    for raw in text.split("\n"):
        if raw.strip() == "":
            continue
        stripped = raw.lstrip(" ")
        indent = len(raw) - len(stripped)
        if indent % len(INDENT):
            raise ToonError("odd indentation (2-space units required): %r" % raw)
        if stripped[0] == "\t" or "\t" in raw[:indent]:
            raise ToonError("tabs are not valid indentation: %r" % raw)
        lines.append((indent // len(INDENT), stripped.rstrip(" ")))
    parser = _Parser(lines)
    obj = parser.parse_object(0)
    if parser.peek() is not None:
        raise ToonError("trailing content at: %r" % parser.peek()[1])
    return obj


# ── Recall wire-format face ───────────────────────────────

_RECALL_KEYS = ("query", "results", "nodes_examined", "retrieval_time_ms",
                "semantic_active", "semantic_note")
_RESULT_KEYS = ("node_id", "node_type", "label", "content", "score",
                "score_breakdown", "path")
_SB_PREFIX = "score_breakdown."
# Reserved by the flattening convention (see module docstring): its
# presence at top level is how parse_recall detects a reshaped document.
_PATHS_KEY = "paths"


def _recall_flatten_eligible(payload: Any) -> bool:
    """True iff payload matches the recall schema exactly, so the tabular
    reshape is exactly invertible. Anything else encodes spec-generic."""
    if not isinstance(payload, dict) or tuple(payload.keys()) != _RECALL_KEYS:
        return False
    results = payload["results"]
    if not isinstance(results, list) or not results:
        return False  # empty results: plain form is already minimal
    sb_keys = None
    for r in results:
        if not isinstance(r, dict) or tuple(r.keys()) != _RESULT_KEYS:
            return False
        sb, path = r["score_breakdown"], r["path"]
        if not isinstance(sb, dict) or not isinstance(path, list):
            return False
        if not all(isinstance(k, str) and k for k in sb):
            return False
        if not all(_is_primitive(v) for v in sb.values()):
            return False
        if not all(_is_primitive(p) for p in path):
            return False
        if not all(_is_primitive(r[k]) for k in
                   ("node_id", "node_type", "label", "content", "score")):
            return False
        keys = tuple(sb.keys())
        if sb_keys is None:
            sb_keys = keys
        elif keys != sb_keys:
            return False  # non-uniform breakdowns: rows would not be tabular
    return True


def _flatten_recall(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    paths: List[List[Any]] = []
    for r in payload["results"]:
        row = {k: r[k] for k in ("node_id", "node_type", "label", "content", "score")}
        for k, v in r["score_breakdown"].items():
            row[_SB_PREFIX + k] = v
        rows.append(row)
        paths.append(list(r["path"]))
    return {
        "query": payload["query"],
        "results": rows,
        _PATHS_KEY: paths,
        "nodes_examined": payload["nodes_examined"],
        "retrieval_time_ms": payload["retrieval_time_ms"],
        "semantic_active": payload["semantic_active"],
        "semantic_note": payload["semantic_note"],
    }


def _unflatten_recall(obj: Dict[str, Any]) -> Dict[str, Any]:
    paths = obj.pop(_PATHS_KEY)
    results = obj.get("results")
    if not isinstance(results, list) or not isinstance(paths, list) \
            or len(paths) != len(results):
        raise ToonError("malformed flattened recall document: "
                        "paths/results mismatch")
    rebuilt = []
    for row, path in zip(results, paths):
        if not isinstance(row, dict):
            raise ToonError("malformed flattened recall row: %r" % (row,))
        sb = {k[len(_SB_PREFIX):]: v for k, v in row.items()
              if k.startswith(_SB_PREFIX)}
        try:
            rebuilt.append({
                "node_id": row["node_id"],
                "node_type": row["node_type"],
                "label": row["label"],
                "content": row["content"],
                "score": row["score"],
                "score_breakdown": sb,
                "path": path,
            })
        except KeyError as exc:
            raise ToonError("flattened recall row missing column: %s" % exc)
    obj["results"] = rebuilt
    return obj


def serialize_recall(response_dict: Dict[str, Any]) -> str:
    """Serialize a POST /v1/recall response dict to TOON text. Applies the
    documented tabular reshape when the payload matches the recall schema
    (see module docstring); falls back to spec-generic encoding otherwise.
    Lossless either way.

    A top-level ``paths`` key is RESERVED by the reshape convention —
    parse_recall uses it to detect a flattened document, so a generic
    payload carrying one would silently round-trip to a DIFFERENT dict.
    Refused loudly instead. (Unreachable from /v1/recall and the CLI,
    which build the payload with a fixed key set.)"""
    if _recall_flatten_eligible(response_dict):
        return encode(_flatten_recall(response_dict))
    if isinstance(response_dict, dict) and _PATHS_KEY in response_dict:
        raise ToonError(
            f"top-level {_PATHS_KEY!r} is reserved by the recall reshape "
            f"convention; this payload cannot round-trip through "
            f"serialize_recall/parse_recall. Use encode()/decode() directly."
        )
    return encode(response_dict)


def parse_recall(toon_str: str) -> Dict[str, Any]:
    """Parse TOON text produced by serialize_recall back to the identical
    response dict (lossless round-trip — asserted by tests/test_toon.py,
    not inherited from the upstream README)."""
    obj = decode(toon_str)
    if _PATHS_KEY in obj:
        return _unflatten_recall(obj)
    return obj

"""The forget key and the keyed fingerprint.

A suppression record must let Revien recognise forgotten text without
keeping it. A plain hash of a short secret can be recovered by guessing, so
fingerprints are HMAC-SHA256 under a random 32-byte key. The key is a file in
the Revien home (``$REVIEN_HOME``, else ``~/.revien``), created owner-only,
and never written to the database: someone holding only ``revien.db`` cannot
check a guess against any fingerprint in it.
"""

import hashlib
import hmac
import os
import secrets
import threading
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

KEY_FILENAME = "forget.key"
_lock = threading.RLock()
_cache = {}


def revien_home() -> Path:
    override = os.environ.get("REVIEN_HOME", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".revien"


def key_path() -> Path:
    return revien_home() / KEY_FILENAME


def _write_key(path: Path, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key.hex().encode("ascii"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows ACLs do not honour POSIX bits; not fatal.


def load_key(create: bool = True) -> Optional[bytes]:
    """The key bytes. Created on first use when ``create`` is true, None when
    absent and ``create`` is false."""
    path = key_path()
    with _lock:
        cached = _cache.get(str(path))
        if cached is not None and path.exists():
            return cached
        if path.exists():
            raw = path.read_text().strip()
            try:
                key = bytes.fromhex(raw)
            except ValueError:
                key = b""
            if len(key) == 32:
                _cache[str(path)] = key
                return key
            if not create:
                return None
            # An unreadable key is replaced: fingerprints made under it can
            # no longer be matched, which fails open for old records, so say
            # so rather than silently minting.
            raise RuntimeError(
                f"{path} is not a valid Revien forget key (32 bytes, hex). "
                "Restore it from a backup; if it is lost, delete the file "
                "and every existing suppression fingerprint stops matching.")
        if not create:
            return None
        key = secrets.token_bytes(32)
        try:
            _write_key(path, key)
        except FileExistsError:
            return load_key(create)
        _cache[str(path)] = key
        return key


def normalize_text(text: str) -> bytes:
    """What gets fingerprinted: Unicode NFC, UTF-8. Whitespace and case are
    NOT folded: an identical fingerprint must mean identical text."""
    return unicodedata.normalize("NFC", text or "").encode("utf-8")


def fingerprint(text: str) -> str:
    key = load_key(create=True)
    return hmac.new(key, normalize_text(text), hashlib.sha256).hexdigest()


def fingerprints(texts: Iterable[str]) -> list:
    seen, out = set(), []
    for t in texts:
        if not t or not t.strip():
            continue
        fp = fingerprint(t)
        if fp not in seen:
            seen.add(fp)
            out.append(fp)
    return out


# ── a literal string, found inside any text without being stored ─────────
#
# A secret named with ``forget --wording`` must never land again in any new
# sentence. The record keeps no text: its length in characters, a keyed
# rolling hash (Rabin-Karp with a base derived from forget.key, so it can be
# slid along a capture in one pass) and a keyed HMAC of the string itself that
# confirms a candidate. The length is the one thing the record discloses.

_RK_MOD = (1 << 61) - 1


def _rk_base() -> int:
    key = load_key(create=True)
    raw = hmac.new(key, b"revien-literal-base", hashlib.sha256).digest()
    return int.from_bytes(raw[:8], "big") % (_RK_MOD - 3) + 2


def rolling_hash(text: str) -> int:
    base, h = _rk_base(), 0
    for ch in text:
        h = (h * base + ord(ch) + 1) % _RK_MOD
    return h


def literal_confirm(text: str) -> str:
    return fingerprint("literal:" + text)


def literal_entry(canonical_text: str) -> str:
    """``"<chars>:<rolling hash>:<hmac>"`` for a (canonical) secret."""
    return f"{len(canonical_text)}:{rolling_hash(canonical_text):x}:{literal_confirm(canonical_text)}"


def find_literals(text: str, entries) -> list:
    """[(start, end)] of every place in ``text`` where one of the entries'
    strings occurs (line wraps inside it ignored), as offsets into ``text``.
    Nothing but the entries is needed to find them."""
    wanted = {}
    for e in entries:
        n, roll, fp = e.split(":", 2)
        wanted.setdefault(int(n), {})[int(roll, 16)] = fp
    if not wanted or not text:
        return []
    from revien.forget.scrub import unwrap

    canon, kept = unwrap(text)
    if kept is None:
        kept = range(len(text))
    base = _rk_base()
    spans = []
    for n, rolls in wanted.items():
        if n > len(canon) or n <= 0:
            continue
        top = pow(base, n - 1, _RK_MOD)
        h = 0
        for ch in canon[:n]:
            h = (h * base + ord(ch) + 1) % _RK_MOD
        i = 0
        while True:
            fp = rolls.get(h)
            if fp is not None and literal_confirm(canon[i:i + n]) == fp:
                spans.append((kept[i], kept[i + n - 1] + 1))
            if i + n >= len(canon):
                break
            h = ((h - (ord(canon[i]) + 1) * top) * base + ord(canon[i + n]) + 1) % _RK_MOD
            i += 1
    return spans


def key_check() -> str:
    """A keyed fixed string: compare it with the one stored at the last forget
    to learn that forget.key was lost or replaced."""
    return fingerprint("revien-forget-key-check")

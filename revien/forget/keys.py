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

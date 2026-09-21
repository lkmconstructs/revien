"""Pairing token (Leg C): the credential that unlocks remote /v1/ingest and
remote mutation endpoints once REVIEN_CAPTURE_TOKEN isn't set explicitly.

REVIEN_CAPTURE_TOKEN remains the operator override (env wins, unconditionally
— CI/hosted deploys that set it never touch the filesystem). Absent that, the
token lives at ``$REVIEN_HOME/pairing.token`` (or ``~/.revien/pairing.token``)
so a host can mint one with `revien token` and hand it to a remote peer
without setting an env var. File permissions are tightened to owner-only
(0o600) where the OS honors that — Windows ACLs don't, so failures there are
swallowed rather than treated as fatal.
"""

import os
import secrets
from pathlib import Path
from typing import Optional


def _revien_home() -> Path:
    override = os.environ.get("REVIEN_HOME", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".revien"


def token_path() -> Path:
    """Where the pairing token file lives: $REVIEN_HOME/pairing.token, or
    ~/.revien/pairing.token when REVIEN_HOME is unset."""
    return _revien_home() / "pairing.token"


def load_token() -> Optional[str]:
    """The token on disk, or None if no file exists (or it's empty)."""
    path = token_path()
    if not path.exists():
        return None
    token = path.read_text().strip()
    return token or None


def _write_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # os.open with 0o600 so the file is owner-only from the moment it's
    # created — no window where it's world-readable before a later chmod.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Best-effort: Windows ACLs don't honor POSIX chmod bits. Not fatal.
        pass


def mint_token(rotate: bool = False) -> str:
    """Return the pairing token, minting one on first use.

    rotate=True always generates and persists a fresh token, replacing any
    existing one. rotate=False (default) returns the existing token if one
    is already on disk, minting only when absent.
    """
    path = token_path()
    if not rotate:
        existing = load_token()
        if existing:
            return existing
    minted = secrets.token_urlsafe(32)
    _write_token(path, minted)
    return minted


def configured_token() -> Optional[str]:
    """The active pairing token, or None if none is configured.

    Precedence: REVIEN_CAPTURE_TOKEN env (operator override, always wins when
    set) beats the pairing-token file. Neither present -> None, and callers
    treat that as "remote access disabled".
    """
    env_token = os.environ.get("REVIEN_CAPTURE_TOKEN", "").strip()
    if env_token:
        return env_token
    return load_token()

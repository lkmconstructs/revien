"""
revien_bench.fetch_locomo — Download LoCoMo-10, checksum, pin the SHA-256.

Fetches `data/locomo10.json` from the snap-research/locomo repo (raw GitHub),
computes the SHA-256, writes the file to `data/` (gitignored), and records the
hash in `revien_bench/DATASET.lock`. Network failure is handled cleanly: the
script reports the error and exits non-zero WITHOUT writing a partial file or a
bogus lock.

Run:  python -m revien_bench.fetch_locomo
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Raw GitHub URL for the canonical LoCoMo-10 QA dataset (Snap Research,
# Maharana et al., ACL 2024, arXiv:2402.17753). We fetch + checksum rather than
# vendor (licensing / Snap's research terms). Mirror candidates tried in order.
RAW_URLS = [
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "https://raw.githubusercontent.com/snap-research/locomo/master/data/locomo10.json",
]
REQUEST_TIMEOUT = 60.0

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
DATA_DIR = _REPO_ROOT / "data"
DATA_PATH = DATA_DIR / "locomo10.json"
LOCK_PATH = _PKG_DIR / "DATASET.lock"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "revien-bench/0.1"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def fetch(force: bool = False) -> str:
    """Fetch the dataset, write it, pin the hash. Returns the SHA-256 hex digest.

    Raises RuntimeError on network failure (no partial file written).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and not force:
        existing = DATA_PATH.read_bytes()
        digest = sha256_bytes(existing)
        print(f"[fetch_locomo] {DATA_PATH} already present ({len(existing)} bytes).")
        _write_lock(digest, len(existing), source="cache")
        print(f"[fetch_locomo] SHA-256: {digest}")
        return digest

    last_err: Exception | None = None
    for url in RAW_URLS:
        try:
            print(f"[fetch_locomo] downloading {url} ...")
            raw = _download(url)
            # Validate it parses as JSON before trusting it.
            json.loads(raw.decode("utf-8"))
            DATA_PATH.write_bytes(raw)
            digest = sha256_bytes(raw)
            _write_lock(digest, len(raw), source=url)
            print(f"[fetch_locomo] wrote {DATA_PATH} ({len(raw)} bytes)")
            print(f"[fetch_locomo] SHA-256: {digest}")
            return digest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            print(f"[fetch_locomo] failed via {url}: {e!r}", file=sys.stderr)
        except (ValueError, UnicodeDecodeError) as e:
            last_err = e
            print(f"[fetch_locomo] downloaded but not valid JSON via {url}: {e!r}",
                  file=sys.stderr)

    raise RuntimeError(
        f"Could not fetch LoCoMo dataset (no network / all mirrors failed). "
        f"Last error: {last_err!r}"
    )


def _write_lock(digest: str, size: int, source: str) -> None:
    lock = {
        "dataset": "locomo10.json",
        "sha256": digest,
        "bytes": size,
        "source": source,
        "note": "Snap Research LoCoMo-10 (arXiv:2402.17753). Fetched+checksummed, not vendored.",
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def read_locked_hash() -> str | None:
    """Return the pinned SHA-256 from DATASET.lock, or None if unset."""
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("sha256")
    except Exception:
        return None


def verify_local() -> bool:
    """True iff the on-disk dataset matches the pinned hash."""
    locked = read_locked_hash()
    if not locked or not DATA_PATH.exists():
        return False
    return sha256_bytes(DATA_PATH.read_bytes()) == locked


def main() -> int:
    force = "--force" in sys.argv
    try:
        fetch(force=force)
        return 0
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

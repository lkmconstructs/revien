"""The only code in Revien that opens a network connection by itself.

`revien sync reset` asks a daemon on THIS machine whether it is running the
current build. This module is the whole of that: it refuses any address that is
not loopback before it touches the network, and nothing else in the package may
import a network client (scripts/gates/check_egress.py fails on any new use
elsewhere). The gate pins the SHA-256 of this entire file, so any edit to it,
including a decorator, a default argument, a nested function or a second
definition, fails the gate until someone re-pins the hash on purpose.
"""
import json
import http.client
from urllib.parse import urlsplit

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def probe_daemon(url):
    """GET /v1/health from a loopback daemon. None when nothing answers, the
    address is not loopback, or the answer is not Revien's; else the decoded
    JSON object."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.scheme != "http" or host not in LOOPBACK_HOSTS:
            return None
        conn = http.client.HTTPConnection(host, parts.port or 80, timeout=1.5)
        try:
            conn.request("GET", (parts.path.rstrip("/") or "") + "/v1/health")
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8", "replace"))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - unreachable, refused, not JSON: all "none"
        return None
    if not isinstance(data, dict) or "node_count" not in data \
            or "uptime_seconds" not in data:
        return None
    return data

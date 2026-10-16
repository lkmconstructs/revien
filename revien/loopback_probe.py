"""The only code in Revien that opens a network connection by itself.

An operation that depends on sync-reset fencing (`revien sync reset`) asks
whether a daemon from before fencing is running on this machine. This module
is that whole question: one GET of /v1/health, on a loopback address.

Rules it keeps:
* The address is judged by what the host name RESOLVES to, not by its spelling:
  every resolved address must be loopback, and the socket is then opened to
  that resolved address (never to the name again), and its peer is checked
  once more after connecting.
* It never follows a redirect, never reads proxy settings (http.client has
  none) and sends nothing but the request line and Host header.
* It returns a three-key summary, never the response body.

It exposes exactly one name, `probe_daemon`; every import is inside it, so
there is nothing to re-export. scripts/gates/check_egress.py pins the SHA-256
of this whole file and fails if any other file imports this module in any form
but `from revien.loopback_probe import probe_daemon` used as a plain call, or
reaches into it (attribute, alias, string, monkeypatch).
"""


def probe_daemon(url):
    """GET /v1/health from a daemon on this machine.

    None: nothing answered, the address is not loopback, the answer was not
    HTTP 200, or it was not Revien's health JSON. Otherwise
    {"features": [str] or None, "version": str or None}; features is None
    when the daemon does not report any (code from before fencing)."""
    try:
        import ipaddress
        import json
        import socket
        from http.client import HTTPConnection
        from urllib.parse import urlsplit

        def loopback(addr):
            ip = ipaddress.ip_address(addr.split("%")[0])
            mapped = getattr(ip, "ipv4_mapped", None)
            return bool((mapped or ip).is_loopback)

        parts = urlsplit(url)
        host = parts.hostname
        if parts.scheme != "http" or not host or parts.username or parts.password:
            return None
        port = parts.port or 80
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if not found or not all(loopback(f[4][0]) for f in found):
            return None
        family, kind, proto, _name, sockaddr = found[0]
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(1.5)
            sock.connect(sockaddr)
            if not loopback(sock.getpeername()[0]):
                return None
            conn = HTTPConnection(sockaddr[0], port, timeout=1.5)
            conn.sock = sock
            sock = None
            try:
                conn.request("GET", "/v1/health", headers={"Connection": "close"})
                resp = conn.getresponse()
                body = resp.read(65537)
            finally:
                conn.close()
        finally:
            if sock is not None:
                sock.close()
        if resp.status != 200 or len(body) > 65536:
            return None
        data = json.loads(body.decode("utf-8", "replace"))
        if not isinstance(data, dict) or "node_count" not in data \
                or "uptime_seconds" not in data:
            return None
        feats = data.get("features")
        if not (isinstance(feats, list) and all(isinstance(f, str) for f in feats)):
            feats = None
        ver = data.get("version")
        return {"features": feats, "version": ver if isinstance(ver, str) else None}
    except Exception:  # noqa: BLE001 - unreachable, refused, not JSON: "none"
        return None

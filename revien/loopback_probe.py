"""The only code in Revien that opens a network connection by itself.

An operation that depends on sync-reset fencing (`revien sync reset`) asks
whether a daemon from before fencing is running on this machine. This module
is that whole question: one GET of /v1/health, on a loopback address.

Rules it keeps:
* Only an IP literal in 127.0.0.0/8 or ::1, or the literal name `localhost`
  (mapped to 127.0.0.1 here), is accepted. Every other host name is refused
  BEFORE any resolver call, so no name ever leaves the machine and no DNS
  answer can steer the connection. Any other address is refused too.
* A refused URL is reported as {"rejected": reason}, distinct from None
  (nothing answered), so the caller can refuse the operation instead of
  treating it as "no daemon".
* It never follows a redirect, never reads proxy settings (http.client has
  none), sends nothing but the request line and Host header, and re-checks
  the connected peer.
* On success it returns a two-key summary, never the response body.

It exposes exactly one name, `probe_daemon`; every import is inside it, so
there is nothing to re-export. scripts/gates/check_egress.py pins the SHA-256
of this whole file and fails if any other file imports this module in any form
but `from revien.loopback_probe import probe_daemon` used as a plain call.
"""


def probe_daemon(url):
    """GET /v1/health from a daemon on this machine.

    {"rejected": reason}: the URL is not an http URL on a loopback IP literal
    or `localhost`; nothing was resolved or connected.
    None: nothing answered, or the answer was not Revien's health JSON.
    Otherwise {"features": [str] or None, "version": str or None}; features is
    None when the daemon reports none (code from before fencing)."""
    try:
        import ipaddress
        import json
        import socket
        from http.client import HTTPConnection
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        host = parts.hostname
        if parts.scheme != "http" or not host or parts.username or parts.password:
            return {"rejected": "not a plain http URL with a host"}
        port = parts.port or 80
        if host.lower() == "localhost":
            ip = ipaddress.ip_address("127.0.0.1")
        else:
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                return {"rejected": "host is a name, not a loopback IP literal"}
        if ip.version == 4:
            ok = ip in ipaddress.ip_network("127.0.0.0/8")
        else:
            ok = ip == ipaddress.ip_address("::1")
        if not ok:
            return {"rejected": "address is not loopback"}
        family = socket.AF_INET if ip.version == 4 else socket.AF_INET6
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.settimeout(1.5)
            sock.connect((str(ip), port))
            peer = ipaddress.ip_address(sock.getpeername()[0].split("%")[0])
            if not peer.is_loopback:
                return None
            conn = HTTPConnection(str(ip), port, timeout=1.5)
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

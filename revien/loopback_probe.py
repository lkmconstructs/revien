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
  (nothing is listening), so the caller can refuse the operation instead of
  treating it as "no daemon". Something that accepted the connection but did
  not give a current health answer (it hung, answered 500, or answered
  something that is not Revien's health JSON) is {"unanswered": reason}: a
  daemon may be running, so the caller refuses too.
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
    None: nothing is listening there (the connection was refused).
    {"unanswered": reason}: something accepted the connection and did not
    give a Revien health answer in time (a hang, an HTTP error, not JSON).
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
        try:
            port = 80 if parts.port is None else parts.port
        except ValueError:
            return {"rejected": "port is not a number from 1 to 65535"}
        if not 1 <= port <= 65535:
            return {"rejected": "port is not a number from 1 to 65535"}
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
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            return None                # no such address family here
        try:
            sock.settimeout(1.5)
            try:
                sock.connect((str(ip), port))
            except OSError:
                return None            # nothing is listening
            peer = ipaddress.ip_address(sock.getpeername()[0].split("%")[0])
            if not peer.is_loopback:
                return {"unanswered": "the peer was not a loopback address"}
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
            return {"unanswered": f"HTTP {resp.status}"}
        data = json.loads(body.decode("utf-8", "replace"))
        if not isinstance(data, dict) or "node_count" not in data \
                or "uptime_seconds" not in data:
            return {"unanswered": "the answer was not Revien's health JSON"}
        feats = data.get("features")
        if not (isinstance(feats, list) and all(isinstance(f, str) for f in feats)):
            feats = None
        ver = data.get("version")
        return {"features": feats, "version": ver if isinstance(ver, str) else None}
    except Exception as exc:  # noqa: BLE001 - it accepted us and then hung or broke
        return {"unanswered": type(exc).__name__}

"""revien/loopback_probe.py: the one pinned module that opens a connection.

It must reach only loopback addresses (judged by what the host resolves to),
never follow a redirect, never use a proxy, and hand back a three-key summary.
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from revien.loopback_probe import probe_daemon

CURRENT = {"status": "healthy", "node_count": 3, "edge_count": 2,
           "uptime_seconds": 1.5, "version": "9.9.9",
           "features": ["sync_reset_fence"]}
OLD = {k: v for k, v in CURRENT.items() if k != "features"}


class _Server:
    def __init__(self, status=200, body=None, headers=None, raw=None):
        outer = self
        self.hits = []
        self.status, self.headers = status, headers or {}
        self.body = raw if raw is not None else json.dumps(
            body if body is not None else CURRENT).encode()

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits.append((self.path, dict(self.headers)))
                self.send_response(outer.status)
                for k, v in outer.headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, args=(0.02,),
                         daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(**kw):
        s = _Server(**kw)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _NoSockets:
    """Stand-in for socket.socket that proves no connection was attempted."""
    made = 0

    def __init__(self, *a, **k):
        type(self).made += 1
        raise AssertionError("a socket was created")


def _forbid_sockets(monkeypatch):
    _NoSockets.made = 0
    monkeypatch.setattr(socket, "socket", _NoSockets)
    monkeypatch.setattr(socket, "create_connection", _NoSockets)


def _resolve_to(monkeypatch, mapping):
    real = socket.getaddrinfo

    def fake(host, port, *a, **k):
        if host in mapping:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                    for ip in mapping[host]]
        return real(host, port, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake)


class TestAnswers:
    def test_current_daemon_reports_features_and_version(self, serve):
        s = serve()
        assert probe_daemon(s.url) == {
            "features": ["sync_reset_fence"], "version": "9.9.9"}
        assert s.hits[0][0] == "/v1/health"

    def test_old_daemon_has_no_features(self, serve):
        s = serve(body=OLD)
        assert probe_daemon(s.url) == {"features": None, "version": "9.9.9"}

    def test_malformed_features_count_as_none(self, serve):
        s = serve(body={**OLD, "features": "sync_reset_fence"})
        assert probe_daemon(s.url)["features"] is None
        s2 = serve(body={**OLD, "features": [1, 2]})
        assert probe_daemon(s2.url)["features"] is None

    def test_nothing_listening_is_none(self):
        assert probe_daemon(f"http://127.0.0.1:{_free_port()}") is None

    def test_the_body_is_never_returned(self, serve):
        s = serve(body={**CURRENT, "secret": "do-not-leak", "db_path": "/x"})
        assert set(probe_daemon(s.url)) == {"features", "version"}

    @pytest.mark.parametrize("status", [201, 204, 404, 500])
    def test_only_http_200_counts(self, serve, status):
        assert probe_daemon(serve(status=status).url) is None

    @pytest.mark.parametrize("raw", [
        b"not json", b"[]", b'{"hello": "world"}',
        json.dumps({"node_count": 1}).encode(),
    ])
    def test_not_revien_health_is_none(self, serve, raw):
        assert probe_daemon(serve(raw=raw).url) is None

    def test_oversized_answer_is_none(self, serve):
        big = json.dumps({**CURRENT, "pad": "x" * 70000}).encode()
        assert probe_daemon(serve(raw=big).url) is None

    def test_path_in_the_url_does_not_change_what_is_asked(self, serve):
        s = serve()
        assert probe_daemon(s.url + "/anything/else?x=1") is not None
        assert s.hits[0][0] == "/v1/health"

    @pytest.mark.parametrize("url", [
        "https://127.0.0.1:7437", "ftp://127.0.0.1", "127.0.0.1:7437", "",
        "http://", "http://user:pw@127.0.0.1:7437", "http://127.0.0.1:notaport",
    ])
    def test_bad_urls_are_rejected_or_none_without_a_connection(
            self, monkeypatch, url):
        _forbid_sockets(monkeypatch)
        _NoResolver.install(monkeypatch)
        assert probe_daemon(url) in (None, {"rejected": "not a plain http URL with a host"})
        assert _NoSockets.made == 0 and _NoResolver.calls == []


class _NoResolver:
    calls = []

    @classmethod
    def install(cls, monkeypatch):
        cls.calls = []

        def boom(*a, **k):
            cls.calls.append(a)
            raise AssertionError("the resolver was called")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        monkeypatch.setattr(socket, "gethostbyname", boom)
        monkeypatch.setattr(socket, "gethostbyname_ex", boom)


class TestLoopbackOnly:
    @pytest.mark.parametrize("url", [
        "http://192.0.2.1:7437", "http://10.0.0.5", "http://192.168.1.2:80",
        "http://203.0.113.9:80", "http://[2001:db8::1]:80", "http://0.0.0.0:7437",
        "http://169.254.169.254", "http://[::]:80", "http://[::ffff:10.0.0.5]:80",
        "http://[::ffff:203.0.113.9]:80", "http://[::ffff:127.0.0.1]:80",
        "http://128.0.0.1", "http://126.255.255.255",
    ])
    def test_non_loopback_literal_is_rejected_without_a_connection(
            self, monkeypatch, url):
        _forbid_sockets(monkeypatch)
        _NoResolver.install(monkeypatch)
        got = probe_daemon(url)
        assert set(got) == {"rejected"}
        assert _NoSockets.made == 0 and _NoResolver.calls == []

    @pytest.mark.parametrize("url", [
        "http://example.com", "http://example.com:7437", "http://looks-local.test",
        "http://localhost.example.com", "http://127.0.0.1.example.com",
        "http://my-box.test:7437", "http://LOCALHOST.evil.test", "http://local",
        "http://2130706433", "http://0x7f.1",
    ])
    def test_a_host_name_is_rejected_before_any_resolver_call(
            self, monkeypatch, url):
        _forbid_sockets(monkeypatch)
        _NoResolver.install(monkeypatch)
        got = probe_daemon(url)
        assert set(got) == {"rejected"}
        assert _NoResolver.calls == [] and _NoSockets.made == 0

    def test_localhost_is_mapped_to_127_0_0_1_without_a_resolver(
            self, monkeypatch, serve):
        s = serve()
        _NoResolver.install(monkeypatch)
        seen = []
        real = socket.socket

        class Spy(real):
            def connect(self, addr):
                seen.append(addr)
                return super().connect(addr)

        monkeypatch.setattr(socket, "socket", Spy)
        assert probe_daemon(f"http://localhost:{s.port}") is not None
        assert probe_daemon(f"http://LocalHost:{s.port}") is not None
        assert seen == [("127.0.0.1", s.port)] * 2
        assert _NoResolver.calls == []

    def test_any_address_in_127_8_works(self, serve):
        s = serve()
        assert probe_daemon(f"http://127.0.0.1:{s.port}") is not None
        assert "rejected" not in (probe_daemon(f"http://127.5.6.7:{s.port}") or {})

    def test_ipv6_loopback_literal_is_accepted_by_the_check(self, monkeypatch):
        _NoResolver.install(monkeypatch)
        got = probe_daemon(f"http://[::1]:{_free_port()}")
        assert got is None  # accepted, nothing listening (or no IPv6)

    def test_a_peer_that_is_not_loopback_is_dropped(self, monkeypatch, serve):
        s = serve()
        real = socket.socket

        class Lying(real):
            def getpeername(self):
                return ("203.0.113.9", s.port)

        monkeypatch.setattr(socket, "socket", Lying)
        assert probe_daemon(s.url) is None


class TestNoRedirectsNoProxies:
    @pytest.mark.parametrize("code", [301, 302, 307, 308])
    def test_redirects_are_not_followed(self, serve, code):
        target = serve()
        hop = serve(status=code, headers={
            "Location": f"http://127.0.0.1:{target.port}/v1/health"})
        assert probe_daemon(hop.url) is None
        assert hop.hits and target.hits == []

    def test_redirect_to_another_host_is_not_followed(self, serve):
        hop = serve(status=302, headers={"Location": "http://203.0.113.9/v1/health"})
        assert probe_daemon(hop.url) is None

    def test_proxy_settings_are_ignored(self, monkeypatch, serve):
        proxy = serve()
        target = serve()
        for var in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
                    "HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(var, proxy.url)
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        assert probe_daemon(target.url) is not None
        assert proxy.hits == []
        assert len(target.hits) == 1

    def test_sends_only_a_plain_get(self, serve):
        s = serve()
        probe_daemon(s.url)
        path, headers = s.hits[0]
        assert path == "/v1/health"
        assert not {k.lower() for k in headers} & {
            "authorization", "cookie", "proxy-authorization"}


class TestModuleSurface:
    def test_one_public_name_and_no_network_module_to_re_export(self):
        import revien.loopback_probe as mod
        assert sorted(n for n in vars(mod) if not n.startswith("__")) == [
            "probe_daemon"]
        for forbidden in ("socket", "http", "ssl", "json", "urlsplit",
                          "HTTPConnection", "ipaddress"):
            assert not hasattr(mod, forbidden)

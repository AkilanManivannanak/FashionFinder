import socket

import pytest

import url_fetch
from url_fetch import UnsafeURLError, fetch_image_bytes


def _resolve_to(ip):
    return lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"\x89PNG fake"):
        self.status_code, self.headers, self._body = status, headers or {"Content-Type": "image/png"}, body

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def close(self):
        pass


class FakeSession:
    def __init__(self, *responses):
        self.responses, self.urls = list(responses), []

    def get(self, url, **kw):
        assert kw["allow_redirects"] is False and kw["stream"] is True
        self.urls.append(url)
        return self.responses.pop(0)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a.jpg", "http:///nohost", "gopher://x"])
def test_rejects_non_http_urls(url):
    with pytest.raises(UnsafeURLError):
        fetch_image_bytes(url, session=FakeSession())


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.2", "169.254.169.254", "0.0.0.0", "::1", "fd00::1"])
def test_rejects_internal_addresses(monkeypatch, ip):
    monkeypatch.setattr(url_fetch.socket, "getaddrinfo", _resolve_to(ip))
    with pytest.raises(UnsafeURLError, match="private or internal"):
        fetch_image_bytes("http://looks-public.example/a.jpg", session=FakeSession(FakeResponse()))


def test_public_image_is_downloaded(monkeypatch):
    monkeypatch.setattr(url_fetch.socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    assert fetch_image_bytes("https://example.com/a.png", session=FakeSession(FakeResponse())) == b"\x89PNG fake"


def test_redirect_to_internal_address_is_refused(monkeypatch):
    ips = iter(["93.184.216.34", "127.0.0.1"])
    monkeypatch.setattr(url_fetch.socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(ips), port))])
    s = FakeSession(FakeResponse(302, {"Location": "http://internal.example/admin"}))
    with pytest.raises(UnsafeURLError, match="private or internal"):
        fetch_image_bytes("https://example.com/a.png", session=s)


def test_non_image_and_oversized_responses_are_refused(monkeypatch):
    monkeypatch.setattr(url_fetch.socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    with pytest.raises(UnsafeURLError, match="not an image"):
        fetch_image_bytes("https://example.com/a", session=FakeSession(FakeResponse(headers={"Content-Type": "text/html"})))
    monkeypatch.setattr(url_fetch, "MAX_BYTES", 10)
    with pytest.raises(UnsafeURLError, match="larger than"):
        fetch_image_bytes("https://example.com/a.png", session=FakeSession(FakeResponse(body=b"x" * 50)))

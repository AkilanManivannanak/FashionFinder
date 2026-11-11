"""
url_fetch.py
------------
Safe download of a user-supplied image URL for POST /search/url.

The endpoint fetches whatever URL the caller sends, so without checks it can be used to make
the server request internal addresses (SSRF), e.g. http://localhost:8001/... or the cloud
metadata service at 169.254.169.254, or to download huge files. This module:

  * allows only http/https URLs with a host name,
  * resolves the host and refuses private, loopback, link-local, multicast, reserved and
    unspecified addresses (checked again on every redirect, max 3 redirects),
  * streams the body and stops at MAX_BYTES,
  * requires an image/* content type when the server sends one.

Known limit: a DNS answer that changes between the check and the request (DNS rebinding)
is not prevented; pin the resolved IP in the request if that matters for a deployment.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

MAX_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT = (5, 10)  # connect, read seconds


class UnsafeURLError(ValueError):
    """The URL is not allowed or the download does not look like an acceptable image."""


def _check_host(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs are allowed.")
    if not parts.hostname:
        raise UnsafeURLError("URL has no host.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as e:
        raise UnsafeURLError(f"Cannot resolve host {parts.hostname!r}.") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            raise UnsafeURLError("URL points to a private or internal address.")


def fetch_image_bytes(url: str, session=None) -> bytes:
    """Download an image from a public http(s) URL, enforcing the limits above."""
    http = session or requests
    for _ in range(MAX_REDIRECTS + 1):
        _check_host(url)
        r = http.get(url, timeout=TIMEOUT, stream=True, allow_redirects=False,
                     headers={"User-Agent": "FashionFinder/1.0"})
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            url = urljoin(url, r.headers["Location"])
            r.close()
            continue
        if r.status_code != 200:
            raise UnsafeURLError(f"Image URL returned HTTP {r.status_code}.")
        ctype = r.headers.get("Content-Type", "")
        if ctype and not ctype.lower().startswith("image/"):
            raise UnsafeURLError(f"URL is not an image (Content-Type: {ctype}).")
        length = r.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > MAX_BYTES:
            raise UnsafeURLError("Image is larger than 10 MB.")
        chunks, total = [], 0
        for chunk in r.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                r.close()
                raise UnsafeURLError("Image is larger than 10 MB.")
            chunks.append(chunk)
        return b"".join(chunks)
    raise UnsafeURLError("Too many redirects.")

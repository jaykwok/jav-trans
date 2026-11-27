"""Localhost-only ASGI security middleware defending against DNS rebinding and CSRF."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# `localhost` and `test` are RFC 2606 reserved domains; `testserver` is
# Starlette/HTTPX's in-process ASGI default. None can be registered on public
# DNS, so allowing them keeps ASGI tests working while blocking real DNS
# rebinding and cross-site attacks.
ALLOWED_LOCAL_HOSTS: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "test", "testserver"}
)
_SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})
_ALLOWED_FETCH_SITES: frozenset[str] = frozenset({"same-origin", "none"})


def _valid_port(port_text: str) -> bool:
    if not port_text or not port_text.isascii() or not port_text.isdigit():
        return False
    port = int(port_text)
    return 1 <= port <= 65535


def extract_host_hostname(host_header: str) -> str | None:
    """Parse an HTTP ``Host`` header value and return its lowercased hostname."""
    raw = (host_header or "").strip()
    if not raw or any(ch.isspace() or ch in "/\\@\x00" for ch in raw):
        return None
    if raw.startswith("["):
        end = raw.find("]")
        if end <= 1:
            return None
        hostname = raw[1:end].lower()
        rest = raw[end + 1 :]
        if rest:
            if not rest.startswith(":") or not _valid_port(rest[1:]):
                return None
        return hostname
    colon_count = raw.count(":")
    if colon_count == 1:
        hostname, port_text = raw.split(":", 1)
        if not hostname or not _valid_port(port_text):
            return None
        return hostname.lower()
    if colon_count > 1:
        # Bare IPv6 address without port (e.g. ::1).
        return raw.lower()
    return raw.lower()


def extract_origin_hostname(origin_header: str) -> str | None:
    """Parse an HTTP ``Origin`` header and return its lowercased hostname."""
    raw = (origin_header or "").strip()
    if not raw or raw.lower() == "null" or any(ch.isspace() or ch == "\x00" for ch in raw):
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    if not parsed.hostname:
        return None
    try:
        _ = parsed.port
    except ValueError:
        return None
    return parsed.hostname.lower()


def _is_api_path(path: str) -> bool:
    return path == "/api" or path.startswith("/api/")


def validate_local_request(
    method: str,
    path: str,
    headers: Iterable[tuple[bytes, bytes]],
    *,
    allowed_hosts: frozenset[str] = ALLOWED_LOCAL_HOSTS,
) -> tuple[bool, str]:
    """Validate Host, Origin, and Sec-Fetch-Site headers for a local-only server."""
    host_values: list[str] = []
    origin_values: list[str] = []
    fetch_site_values: list[str] = []

    for raw_name, raw_value in headers:
        name = raw_name.decode("latin-1").lower()
        if name == "host":
            host_values.append(raw_value.decode("latin-1"))
        elif name == "origin":
            origin_values.append(raw_value.decode("latin-1"))
        elif name == "sec-fetch-site":
            fetch_site_values.append(raw_value.decode("latin-1"))

    if not host_values:
        return False, "Forbidden host"
    parsed_hosts: set[str] = set()
    for host_value in host_values:
        hostname = extract_host_hostname(host_value)
        if hostname is None or hostname not in allowed_hosts:
            return False, "Forbidden host"
        parsed_hosts.add(hostname)
    if len(parsed_hosts) != 1:
        return False, "Forbidden host"

    must_check_origin = method.upper() not in _SAFE_METHODS or _is_api_path(path)
    if must_check_origin:
        for origin_value in origin_values:
            origin_host = extract_origin_hostname(origin_value)
            if origin_host is None or origin_host not in allowed_hosts:
                return False, "Forbidden origin"
        for site_value in fetch_site_values:
            if site_value.strip().lower() not in _ALLOWED_FETCH_SITES:
                return False, "Forbidden cross-site request"

    return True, ""


class LocalSecurityMiddleware:
    """Pure ASGI middleware enforcing localhost Host / Origin / Sec-Fetch-Site.

    Implemented as raw ASGI (not BaseHTTPMiddleware) so SSE streaming responses
    like ``/api/events`` pass through unbuffered.
    """

    def __init__(
        self,
        app: Any,
        *,
        allowed_hosts: frozenset[str] = ALLOWED_LOCAL_HOSTS,
    ) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type == "http":
            method = str(scope.get("method") or "GET")
            path = str(scope.get("path") or "")
            headers = scope.get("headers") or ()
            ok, reason = validate_local_request(
                method,
                path,
                headers,
                allowed_hosts=self.allowed_hosts,
            )
            if not ok:
                log.warning(
                    "Blocked local security violation (%s) on %s %s",
                    reason,
                    method,
                    path,
                )
                response = JSONResponse(status_code=403, content={"detail": reason})
                await response(scope, receive, send)
                return
        elif scope_type == "websocket":
            path = str(scope.get("path") or "")
            headers = scope.get("headers") or ()
            ok, reason = validate_local_request(
                "POST",
                path,
                headers,
                allowed_hosts=self.allowed_hosts,
            )
            if not ok:
                log.warning(
                    "Blocked websocket security violation (%s) on %s",
                    reason,
                    path,
                )
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)

"""`web_fetch`: fetch HTTPS pages from allow-listed domains.

Ships disabled: with an empty allow-list every call is refused (the template's "stub").
When enabled it guards against SSRF: HTTPS only, allow-listed hosts only, every resolved
address must be public, redirects are not followed, size and time are capped.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Sequence
from html import unescape
from urllib.parse import urlsplit

import httpx

from harness.tools.registry import Permission, Tool, ToolError, make_tool

MAX_BYTES = 1_000_000
_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def host_allowed(host: str, allowed: Sequence[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in (a.lower() for a in allowed))


async def assert_public(host: str, port: int) -> None:
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"cannot resolve {host}") from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global:
            raise ToolError(f"{host} resolves to a non-public address")


def html_to_text(html: str) -> str:
    text = unescape(_TAGS.sub(" ", html))
    return re.sub(r"\s+", " ", text).strip()


def web_fetch_tool(
    allowed_domains: Sequence[str],
    *,
    timeout_s: float = 15.0,
    transport: httpx.AsyncBaseTransport | None = None,
    check_dns: bool = True,
) -> Tool:
    allowed = [d.strip().lower() for d in allowed_domains if d.strip()]

    async def web_fetch(url: str) -> str:
        """Fetch a web page over HTTPS and return its text content (HTML tags stripped).

        Only allow-listed domains are reachable; redirects are reported, not followed.
        """
        if not allowed:
            raise ToolError("web access is disabled (no allow-listed domains configured)")
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname:
            raise ToolError("only https:// URLs are allowed")
        if parts.username or parts.password:
            raise ToolError("URLs with credentials are not allowed")
        if not host_allowed(parts.hostname, allowed):
            raise ToolError(f"domain {parts.hostname} is not allow-listed")
        if check_dns:
            # ponytail: resolve-then-connect leaves a DNS-rebinding window; pin the
            # resolved IP in a custom transport if the allow-list includes untrusted DNS.
            await assert_public(parts.hostname, parts.port or 443)

        async with (
            httpx.AsyncClient(
                timeout=timeout_s, follow_redirects=False, transport=transport
            ) as client,
            client.stream("GET", url, headers={"User-Agent": "agent-harness/0.1"}) as resp,
        ):
            if resp.is_redirect:
                return f"HTTP {resp.status_code} redirect to {resp.headers.get('location')}"
            if resp.status_code >= 400:
                raise ToolError(f"HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "")
            if not ctype.startswith(("text/", "application/json", "application/xml")):
                raise ToolError(f"unsupported content type {ctype!r}")
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    break
        text = body[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace")
        return html_to_text(text) if "html" in ctype else text

    return make_tool(web_fetch, permission=Permission.READ, timeout_s=timeout_s + 5)

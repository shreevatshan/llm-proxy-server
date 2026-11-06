"""SSRF guard for upstream URLs a user (not the admin) supplied: BYOK instances.

A BYOK instance makes the proxy send requests -- carrying the owner's secrets --
to a URL the owner typed. Without a guard that is a server-side request forgery
primitive into the proxy's own network (the management API, cloud metadata at
169.254.169.254, other internal services).

Two layers:

* ``validate_user_upstream_url`` -- https only, and the host must resolve only
  to globally routable addresses. Run on create/update and again whenever the
  provider is built (startup, refresh), so a stored URL is re-checked.
* ``guarded_http_client`` -- an httpx client for BYOK providers that never
  follows redirects (a vetted public URL could 3xx to an internal one) and
  re-checks the destination of every request it sends, which also narrows DNS
  rebinding to the gap between that check and the connect.

Known limitation: the connect itself is not pinned to the checked address, so a
host with a very short TTL can still rebind between the check and the connect.
Admin-entered instances are not affected by any of this.
"""

import asyncio
import ipaddress
import socket
from typing import Iterable, Optional
from urllib.parse import urlparse

import httpx

# Credential-row fields that hold an upstream URL.
URL_FIELDS = ("endpoint", "base_url", "anthropic_base_url", "oauth_accounts_url")


class UnsafeUpstreamURL(ValueError):
    """The URL is not allowed as a BYOK upstream."""


def _is_public(ip: "ipaddress._BaseAddress") -> bool:
    # is_global already excludes private, loopback, link-local, CGNAT (100.64/10),
    # documentation and reserved ranges; the explicit checks are belt and braces.
    return (
        ip.is_global
        and not ip.is_loopback
        and not ip.is_private
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def _check_addresses(host: str, addresses: Iterable[str]) -> None:
    seen = False
    for address in addresses:
        seen = True
        try:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as e:
            raise UnsafeUpstreamURL(f"Host {host!r} resolved to an invalid address") from e
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not _is_public(ip):
            raise UnsafeUpstreamURL(
                f"Host {host!r} resolves to a non-public address; BYOK upstreams must be public"
            )
    if not seen:
        raise UnsafeUpstreamURL(f"Host {host!r} did not resolve")


def _parse(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https":
        raise UnsafeUpstreamURL("BYOK upstream URLs must use https")
    host = parsed.hostname
    if not host:
        raise UnsafeUpstreamURL("URL has no host")
    if parsed.username or parsed.password:
        raise UnsafeUpstreamURL("URL must not embed credentials")
    return host


def validate_user_upstream_url(url: Optional[str]) -> Optional[str]:
    """Raise UnsafeUpstreamURL unless ``url`` is an https URL on a public host.

    None/empty passes (the field is unset). Blocking: resolves the host.
    """
    if url in (None, ""):
        return url
    host = _parse(url)
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeUpstreamURL(f"Could not resolve host {host!r}") from e
    _check_addresses(host, (info[4][0] for info in infos))
    return url.strip()


def validate_credential_urls(fields: dict) -> None:
    """Validate every URL field present in a credentials payload or config."""
    for name in URL_FIELDS:
        value = fields.get(name)
        if value:
            try:
                validate_user_upstream_url(value)
            except UnsafeUpstreamURL as e:
                raise UnsafeUpstreamURL(f"{name}: {e}") from e


def validate_cred_row(cred) -> None:
    """Validate the URL columns of a provider_credentials row."""
    validate_credential_urls({name: getattr(cred, name, None) for name in URL_FIELDS})


async def _check_request(request: httpx.Request) -> None:
    host = request.url.host
    if request.url.scheme != "https":
        raise UnsafeUpstreamURL("BYOK upstream requests must use https")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeUpstreamURL(f"Could not resolve host {host!r}") from e
    _check_addresses(host, (info[4][0] for info in infos))


def guarded_http_client(timeout: Optional[float] = None) -> httpx.AsyncClient:
    """An httpx client for BYOK upstream calls: no redirects, every request re-checked."""
    kwargs = {
        "follow_redirects": False,
        "event_hooks": {"request": [_check_request]},
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return httpx.AsyncClient(**kwargs)


def http_client_for(config: Optional[dict], timeout: Optional[float] = None) -> Optional[httpx.AsyncClient]:
    """The guarded client for a BYOK provider config, else None (use the SDK default).

    Providers pass the result as ``http_client=`` to the openai/anthropic SDKs, which
    treat None as "build your own".
    """
    if config and config.get("byok"):
        return guarded_http_client(timeout=timeout)
    return None

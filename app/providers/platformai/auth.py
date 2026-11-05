"""Zoho OAuth access tokens for the PlatformAI gateway.

PlatformAI accepts only ``Authorization: Zoho-oauthtoken <access token>``. The
admin stores a long-lived refresh token; this module mints short-lived access
tokens from it (``POST {accounts}/oauth/v2/token`` with
``grant_type=refresh_token``), the way zllm's ``Cd`` does.

The cache is module-level, not per provider instance: the provider manager and
auto-sync rebuild provider instances on every admin change, and minting a new
token for each rebuild would trip Zoho's token rate limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx

from app.providers.base import ProviderHTTPError

logger = logging.getLogger(__name__)

# Re-use a token until this long before Zoho says it expires.
EXPIRY_SKEW_SECONDS = 60.0
DEFAULT_EXPIRES_IN = 3600.0

# A gateway 401 on a token younger than this is not an expiry problem (wrong
# portal, missing scope): re-minting would only burn Zoho's refresh quota.
MIN_TOKEN_AGE_FOR_REMINT_SECONDS = 60.0

# Back-off after a failed refresh: doubles per consecutive failure.
_BACKOFF_BASE = 30.0
_BACKOFF_MAX = 300.0
_RATE_LIMIT_BACKOFF_BASE = 120.0
_RATE_LIMIT_BACKOFF_MAX = 600.0

_RATE_LIMIT_RE = re.compile(r"too many|rate limit|rate-limit|try again after|temporarily bl|throttl", re.I)
_DEAD_GRANT_RE = re.compile(
    r"invalid[_ -]?grant|invalid[_ -]?token|invalid[_ -]?code|revoked|expired|malformed|invalid refresh",
    re.I,
)


@dataclass
class _TokenState:
    token: Optional[str] = None
    expires_at: float = 0.0
    minted_at: float = 0.0
    failures: int = 0
    blocked_until: float = 0.0
    last_error: Optional[ProviderHTTPError] = None
    lock: Optional[asyncio.Lock] = field(default=None, repr=False)


_STATES: Dict[Tuple[str, str], _TokenState] = {}


def _cache_key(provider_key: str, *credential: Optional[str]) -> Tuple[str, str]:
    """Key a token state on the provider and every credential field, so fixing
    the client ID, secret or accounts URL drops the old token and back-off."""
    material = "\0".join(part or "" for part in credential)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return provider_key, digest


def reset_token_cache() -> None:
    """Forget every cached token (tests)."""
    _STATES.clear()


def _auth_error(message: str) -> ProviderHTTPError:
    return ProviderHTTPError(
        502,
        message,
        body={"error": {"message": message, "type": "upstream_auth_error", "code": 502}},
    )


class PlatformAITokenManager:
    """Mints and caches access tokens for one PlatformAI provider credential."""

    def __init__(
        self,
        *,
        provider_key: str,
        accounts_url: str,
        client_id: str,
        refresh_token: str,
        client_secret: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        if not refresh_token:
            raise ValueError("PlatformAI refresh token is required")
        if not client_id:
            raise ValueError("PlatformAI OAuth client ID is required")
        self.accounts_url = accounts_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret or None
        self._refresh_token = refresh_token
        self._http = http_client
        self._key = _cache_key(provider_key, refresh_token, self.client_id, self.client_secret, self.accounts_url)

    @property
    def _state(self) -> _TokenState:
        state = _STATES.get(self._key)
        if state is None:
            state = _STATES[self._key] = _TokenState()
        return state

    def _lock(self) -> asyncio.Lock:
        state = self._state
        if state.lock is None:
            state.lock = asyncio.Lock()
        return state.lock

    def _fresh_token(self) -> Optional[str]:
        state = self._state
        if state.token and state.expires_at - EXPIRY_SKEW_SECONDS > time.time():
            return state.token
        return None

    async def get_token(self) -> str:
        token = self._fresh_token()
        if token:
            return token
        async with self._lock():
            # Another waiter may have refreshed while we queued.
            token = self._fresh_token()
            if token:
                return token
            return await self._refresh_locked()

    def invalidate(self, rejected_token: str) -> bool:
        """Handle a gateway 401 for ``rejected_token``; True when a retry may help.

        Only the token that was rejected is dropped, so a 401 on an old token
        does not throw away one another request has just minted. A token minted
        moments ago is kept: the 401 is not about expiry, and re-minting on
        every request would trip Zoho's refresh rate limit.
        """
        state = self._state
        if state.token != rejected_token:
            return state.token is not None or state.blocked_until <= time.time()
        if time.time() - state.minted_at < MIN_TOKEN_AGE_FOR_REMINT_SECONDS:
            return False
        state.token = None
        state.expires_at = 0.0
        return True

    async def _refresh_locked(self) -> str:
        state = self._state
        now = time.time()
        if state.blocked_until > now and state.last_error is not None:
            raise state.last_error
        try:
            token, expires_in = await self._request_token()
        except ProviderHTTPError as err:
            rate_limited = err.status_code == 429
            state.failures += 1
            base, cap = (
                (_RATE_LIMIT_BACKOFF_BASE, _RATE_LIMIT_BACKOFF_MAX)
                if rate_limited
                else (_BACKOFF_BASE, _BACKOFF_MAX)
            )
            delay = min(cap, base * (2 ** (state.failures - 1)))
            state.blocked_until = time.time() + delay
            state.last_error = err
            logger.warning(
                "PlatformAI token refresh failed (%s); pausing refresh for %.0fs",
                err.message,
                delay,
            )
            raise
        state.token = token
        state.minted_at = time.time()
        state.expires_at = state.minted_at + expires_in
        state.failures = 0
        state.blocked_until = 0.0
        state.last_error = None
        return token

    async def _request_token(self) -> Tuple[str, float]:
        data = {
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
            "grant_type": "refresh_token",
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret
        url = f"{self.accounts_url}/oauth/v2/token"
        try:
            if self._http is not None:
                response = await self._http.post(url, data=data, timeout=15.0)
            else:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.post(url, data=data)
        except httpx.HTTPError as exc:
            raise _auth_error(f"Could not reach Zoho accounts to refresh the PlatformAI token: {type(exc).__name__}")

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise _auth_error(f"Zoho token refresh returned a non-JSON response (HTTP {response.status_code})")

        error = payload.get("error")
        if error or response.status_code >= 400 or not payload.get("access_token"):
            # Zoho puts the reason in either field ("Access Denied" / "too many requests").
            detail = " — ".join(
                str(v) for v in (error, payload.get("error_description")) if v
            ) or f"HTTP {response.status_code}"
            if response.status_code == 429 or _RATE_LIMIT_RE.search(detail):
                raise ProviderHTTPError(
                    429,
                    "Zoho is rate-limiting PlatformAI token refreshes; try again shortly",
                    body={"error": {"message": "Zoho is rate-limiting PlatformAI token refreshes; try again shortly",
                                    "type": "rate_limit_error", "code": 429}},
                    headers={"retry-after": response.headers.get("retry-after", "120")},
                )
            if response.status_code == 401 or _DEAD_GRANT_RE.search(detail):
                raise _auth_error(
                    "PlatformAI refresh token rejected by Zoho accounts "
                    f"({detail}) — update the provider credentials"
                )
            raise _auth_error(f"Zoho token refresh failed: {detail}")

        try:
            expires_in = float(payload.get("expires_in") or DEFAULT_EXPIRES_IN)
        except (TypeError, ValueError):
            expires_in = DEFAULT_EXPIRES_IN
        return str(payload["access_token"]), expires_in

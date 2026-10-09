"""Shared httpx client and request handling for the web search backends.

One connection pool is reused by every backend; it is closed on shutdown from
``app.main.shared_shutdown``.
"""

import logging
import time
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

import httpx

from app.tracing import add_span_attributes, create_span
from app.websearch.results import SearchFailed, SearchOutcome

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(follow_redirects=True)
    return _client


async def close_client() -> None:
    global _client
    client, _client = _client, None
    if client is not None and not client.is_closed:
        await client.aclose()


async def search_json(
    query: str,
    url: str,
    params: dict,
    timeout_seconds: float,
    *,
    provider: str,
    label: str,
    parse: Callable[[dict], SearchOutcome],
    status_messages: Optional[Dict[int, str]] = None,
    not_json_message: Optional[str] = None,
) -> SearchOutcome:
    """GET a backend's JSON search endpoint and hand the payload to ``parse``.

    Transport errors, HTTP errors and non-JSON bodies are mapped to
    SearchFailed here; ``status_messages`` overrides the message for specific
    HTTP status codes. Never raises.
    """
    started = time.monotonic()
    with create_span(f"websearch.{provider}") as span:
        add_span_attributes(span, {
            "websearch.provider": provider,
            "websearch.host": urlparse(url).netloc,
            "websearch.query_length": len(query),
        })
        try:
            response = await get_client().get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException:
            return SearchFailed(query, "unavailable", f"{label} timed out after {timeout_seconds}s")
        except httpx.HTTPError as e:
            logger.warning("%s request failed: %s", label, e)
            return SearchFailed(query, "unavailable", f"{label} unreachable: {type(e).__name__}")

        add_span_attributes(span, {
            "websearch.status_code": response.status_code,
            "websearch.latency_ms": int((time.monotonic() - started) * 1000),
        })
        if response.status_code == 429:
            return SearchFailed(query, "too_many_requests", f"{label} rate limit reached (HTTP 429)")
        if status_messages and response.status_code in status_messages:
            return SearchFailed(query, "unavailable", status_messages[response.status_code])
        if response.status_code >= 400:
            return SearchFailed(query, "unavailable", f"{label} returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            return SearchFailed(query, "unavailable", not_json_message or f"{label} did not return JSON")
        if not isinstance(payload, dict):
            return SearchFailed(query, "unavailable", f"Unexpected {label} response shape")

        try:
            outcome = parse(payload)
        except Exception as e:
            logger.warning("%s response could not be parsed: %s", label, e, exc_info=True)
            return SearchFailed(query, "unavailable", f"Unexpected {label} response shape")
        if outcome.ok:
            add_span_attributes(span, {"websearch.result_count": len(outcome.results)})
        return outcome

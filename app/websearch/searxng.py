"""SearXNG search client used by web search interception.

Mirrors LiteLLM's SearXNG provider (llms/searxng/search/transformation.py),
but truncates results on our side: SearXNG itself ignores result counts.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union
from urllib.parse import urlparse

import httpx

from app.tracing import add_span_attributes, create_span
from app.websearch.settings import WebSearchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    date: Optional[str] = None


@dataclass(frozen=True)
class SearchSucceeded:
    query: str
    results: List[SearchResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class SearchFailed:
    query: str
    error_code: str  # too_many_requests | unavailable | invalid_tool_input | query_too_long
    message: str

    @property
    def ok(self) -> bool:
        return False


SearchOutcome = Union[SearchSucceeded, SearchFailed]

MAX_QUERY_CHARS = 500

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(follow_redirects=True)
    return _client


async def close_client() -> None:
    global _client
    client, _client = _client, None
    if client is not None and not client.is_closed:
        await client.aclose()


def search_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/search") else f"{base}/search"


class SearxngConfigError(Exception):
    """The instance's /config could not be fetched or parsed."""


def config_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/search"):
        base = base[: -len("/search")]
    return f"{base}/config"


async def fetch_engines(base_url: str, timeout_seconds: float) -> dict:
    """Engines and categories the instance has loaded, from its ``/config``.

    Returns ``{"engines": [{name, categories, enabled, shortcut}], "categories": [...]}``
    with engines sorted by name.
    """
    url = config_url(base_url)
    try:
        response = await _get_client().get(
            url, headers={"Accept": "application/json"}, timeout=timeout_seconds
        )
    except httpx.TimeoutException:
        raise SearxngConfigError(f"SearXNG timed out after {timeout_seconds}s")
    except httpx.HTTPError as e:
        raise SearxngConfigError(f"SearXNG unreachable: {type(e).__name__}")
    if response.status_code >= 400:
        raise SearxngConfigError(f"SearXNG returned HTTP {response.status_code} for /config")
    try:
        payload = response.json()
    except ValueError:
        raise SearxngConfigError("SearXNG /config did not return JSON")
    if not isinstance(payload, dict) or not isinstance(payload.get("engines"), list):
        raise SearxngConfigError("Unexpected SearXNG /config response shape")

    engines = []
    for item in payload["engines"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        categories = item.get("categories")
        engines.append({
            "name": item["name"],
            "categories": [c for c in categories if isinstance(c, str)] if isinstance(categories, list) else [],
            "enabled": bool(item.get("enabled")),
            "shortcut": item.get("shortcut") if isinstance(item.get("shortcut"), str) else None,
        })
    engines.sort(key=lambda e: e["name"])
    categories = payload.get("categories")
    if not isinstance(categories, list):
        categories = sorted({c for e in engines for c in e["categories"]})
    return {"engines": engines, "categories": [c for c in categories if isinstance(c, str)]}


def _build_params(query: str, cfg: WebSearchConfig) -> dict:
    params = {"q": query, "format": "json", "pageno": 1}
    # SearXNG unions engines with every engine in the sent categories, so an
    # explicit engine list must go alone.
    if cfg.engines:
        params["engines"] = cfg.engines
    elif cfg.categories:
        params["categories"] = cfg.categories
    if cfg.language:
        params["language"] = cfg.language
    if cfg.safesearch is not None:
        params["safesearch"] = cfg.safesearch
    if cfg.time_range:
        params["time_range"] = cfg.time_range
    return params


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def parse_results(payload: dict, cfg: WebSearchConfig) -> List[SearchResult]:
    """Map SearXNG ``results[]`` to SearchResult, deduped and truncated."""
    results: List[SearchResult] = []
    seen = set()
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                title=_truncate(item.get("title") or url, 300),
                url=url,
                snippet=_truncate(item.get("content") or "", cfg.max_snippet_chars),
                date=item.get("publishedDate") or item.get("pubdate") or None,
            )
        )
        if len(results) >= cfg.max_results:
            break
    return results


async def search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    """Run one SearXNG query. Never raises; failures come back as SearchFailed."""
    query = (query or "").strip()
    if not query:
        return SearchFailed(query, "invalid_tool_input", "No search query provided")
    if len(query) > MAX_QUERY_CHARS:
        return SearchFailed(query, "query_too_long", f"Query longer than {MAX_QUERY_CHARS} characters")
    if not cfg.searxng_base_url:
        return SearchFailed(query, "unavailable", "SearXNG base URL is not configured")

    headers = {"Accept": "application/json"}

    url = search_url(cfg.searxng_base_url)
    started = time.monotonic()
    with create_span("websearch.searxng") as span:
        add_span_attributes(span, {
            "websearch.host": urlparse(url).netloc,
            "websearch.query_length": len(query),
        })
        try:
            response = await _get_client().get(
                url,
                params=_build_params(query, cfg),
                headers=headers,
                timeout=cfg.timeout_seconds,
            )
        except httpx.TimeoutException:
            return SearchFailed(query, "unavailable", f"SearXNG timed out after {cfg.timeout_seconds}s")
        except httpx.HTTPError as e:
            logger.warning("SearXNG request failed: %s", e)
            return SearchFailed(query, "unavailable", f"SearXNG unreachable: {type(e).__name__}")

        add_span_attributes(span, {
            "websearch.status_code": response.status_code,
            "websearch.latency_ms": int((time.monotonic() - started) * 1000),
        })
        if response.status_code == 429:
            return SearchFailed(query, "too_many_requests", "SearXNG rate limit reached (HTTP 429)")
        if response.status_code == 403:
            # SearXNG answers 403 for format=json when "json" is not in
            # search.formats of its settings.yml.
            return SearchFailed(
                query,
                "unavailable",
                "SearXNG returned HTTP 403 (is 'json' enabled under search.formats in settings.yml?)",
            )
        if response.status_code >= 400:
            return SearchFailed(query, "unavailable", f"SearXNG returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            return SearchFailed(
                query,
                "unavailable",
                "SearXNG did not return JSON (is 'json' enabled under search.formats in settings.yml?)",
            )
        if not isinstance(payload, dict):
            return SearchFailed(query, "unavailable", "Unexpected SearXNG response shape")

        results = parse_results(payload, cfg)
        add_span_attributes(span, {"websearch.result_count": len(results)})
        return SearchSucceeded(query, results)


async def run_searches(queries: List[str], cfg: WebSearchConfig) -> List[SearchOutcome]:
    """Run queries concurrently, preserving order."""
    return list(await asyncio.gather(*(search(q, cfg) for q in queries)))


def format_outcome_text(outcome: SearchOutcome) -> str:
    """Text the model sees as the tool result (LiteLLM's Title/URL/Snippet blocks)."""
    if not outcome.ok:
        return f"Search failed: {outcome.message}"
    if not outcome.results:
        return f"No results found for: {outcome.query}"
    blocks = []
    for r in outcome.results:
        lines = [f"Title: {r.title}", f"URL: {r.url}"]
        if r.date:
            lines.append(f"Date: {r.date}")
        lines.append(f"Snippet: {r.snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

"""SearXNG search client used by web search interception.

Mirrors LiteLLM's SearXNG provider (llms/searxng/search/transformation.py),
but truncates results on our side: SearXNG itself ignores result counts.
"""

from typing import List

import httpx

from app.auth.models import WEBSEARCH_PROVIDER_LABELS, WEBSEARCH_PROVIDER_SEARXNG
from app.websearch.client import get_client, search_json
from app.websearch.results import (
    SearchFailed,
    SearchOutcome,
    SearchResult,
    SearchSucceeded,
    truncate,
)
from app.websearch.settings import WebSearchConfig

# SearXNG answers 403 (or an HTML page) for format=json when "json" is not in
# search.formats of its settings.yml.
_JSON_DISABLED_HINT = "is 'json' enabled under search.formats in settings.yml?"


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
        response = await get_client().get(
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
                title=truncate(item.get("title") or url, 300),
                url=url,
                snippet=truncate(item.get("content") or "", cfg.max_snippet_chars),
                date=item.get("publishedDate") or item.get("pubdate") or None,
            )
        )
        if len(results) >= cfg.max_results:
            break
    return results


async def search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    """Run one SearXNG query (already validated by ``backends.search``).

    Never raises; failures come back as SearchFailed.
    """
    if not cfg.searxng_base_url:
        return SearchFailed(query, "unavailable", "SearXNG base URL is not configured")

    return await search_json(
        query,
        search_url(cfg.searxng_base_url),
        _build_params(query, cfg),
        cfg.timeout_seconds,
        provider=WEBSEARCH_PROVIDER_SEARXNG,
        label=WEBSEARCH_PROVIDER_LABELS[WEBSEARCH_PROVIDER_SEARXNG],
        parse=lambda payload: SearchSucceeded(query, parse_results(payload, cfg)),
        status_messages={403: f"SearXNG returned HTTP 403 ({_JSON_DISABLED_HINT})"},
        not_json_message=f"SearXNG did not return JSON ({_JSON_DISABLED_HINT})",
    )

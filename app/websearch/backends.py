"""Dispatch a search to the configured backend.

The agentic loop calls these; it never learns which backend answered.
"""

import asyncio
from typing import Iterable, List
from urllib.parse import urlparse

from app.auth.models import WEBSEARCH_PROVIDER_FOURGET
from app.websearch import fourget, searxng
from app.websearch.results import MAX_QUERY_CHARS, SearchFailed, SearchOutcome, SearchSucceeded
from app.websearch.settings import WebSearchConfig


async def search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    """Validate the query, then run it on the selected backend. Never raises."""
    query = (query or "").strip()
    if not query:
        return SearchFailed(query, "invalid_tool_input", "No search query provided")
    if len(query) > MAX_QUERY_CHARS:
        return SearchFailed(query, "query_too_long", f"Query longer than {MAX_QUERY_CHARS} characters")
    if not (cfg.allowed_domains or cfg.blocked_domains):
        return await _backend_search(query, cfg)

    # Domain filters: steer the engine with site: operators (appended after the
    # length check, so long lists don't trip it), then filter the results by
    # host since engines honour the operators unevenly. Over-fetch to keep
    # max_results after filtering.
    backend_cfg = cfg.with_overrides(max_results=min(cfg.max_results * 3, 30))
    outcome = await _backend_search(_with_site_operators(query, cfg), backend_cfg)
    if not outcome.ok:
        return SearchFailed(query, outcome.error_code, outcome.message)
    kept = [
        r for r in outcome.results
        if (not cfg.allowed_domains or _matches_any(r.url, cfg.allowed_domains))
        and not _matches_any(r.url, cfg.blocked_domains)
    ]
    return SearchSucceeded(query, kept[: cfg.max_results])


async def _backend_search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    if cfg.provider == WEBSEARCH_PROVIDER_FOURGET:
        return await fourget.search(query, cfg)
    return await searxng.search(query, cfg)


def _with_site_operators(query: str, cfg: WebSearchConfig) -> str:
    if cfg.allowed_domains:
        sites = [f"site:{d}" for d in cfg.allowed_domains]
        return f"{query} {sites[0]}" if len(sites) == 1 else f"{query} ({' OR '.join(sites)})"
    return " ".join([query, *(f"-site:{d}" for d in cfg.blocked_domains)])


def _matches_any(url: str, domains: Iterable[str]) -> bool:
    """True when ``url`` falls under one of ``domains`` (``host[/path]``,
    subdomains included)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or "/"
    for entry in domains:
        entry = entry.strip().lower()
        if "://" in entry:
            entry = entry.split("://", 1)[1]
        domain, _, prefix = entry.partition("/")
        if domain.startswith("www."):
            domain = domain[4:]
        if not domain:
            continue
        if host != domain and not host.endswith("." + domain):
            continue
        if prefix and not path.lstrip("/").startswith(prefix.rstrip("/")):
            continue
        return True
    return False


async def run_searches(queries: List[str], cfg: WebSearchConfig) -> List[SearchOutcome]:
    """Run queries concurrently, preserving order."""
    return list(await asyncio.gather(*(search(q, cfg) for q in queries)))

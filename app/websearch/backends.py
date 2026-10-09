"""Dispatch a search to the configured backend.

The agentic loop calls these; it never learns which backend answered.
"""

import asyncio
from typing import List

from app.auth.models import WEBSEARCH_PROVIDER_FOURGET
from app.websearch import fourget, searxng
from app.websearch.results import MAX_QUERY_CHARS, SearchFailed, SearchOutcome
from app.websearch.settings import WebSearchConfig


async def search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    """Validate the query, then run it on the selected backend. Never raises."""
    query = (query or "").strip()
    if not query:
        return SearchFailed(query, "invalid_tool_input", "No search query provided")
    if len(query) > MAX_QUERY_CHARS:
        return SearchFailed(query, "query_too_long", f"Query longer than {MAX_QUERY_CHARS} characters")
    if cfg.provider == WEBSEARCH_PROVIDER_FOURGET:
        return await fourget.search(query, cfg)
    return await searxng.search(query, cfg)


async def run_searches(queries: List[str], cfg: WebSearchConfig) -> List[SearchOutcome]:
    """Run queries concurrently, preserving order."""
    return list(await asyncio.gather(*(search(q, cfg) for q in queries)))

"""4get search client used by web search interception.

4get is the alternative to SearXNG: it scrapes one chosen upstream engine per
request rather than aggregating. Its API always answers HTTP 200 -- failures
are reported in the ``status`` field -- so the status field, not the status
code, is what decides success here.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from app.auth.models import WEBSEARCH_PROVIDER_FOURGET, WEBSEARCH_PROVIDER_LABELS
from app.websearch.client import search_json
from app.websearch.results import (
    SearchFailed,
    SearchOutcome,
    SearchResult,
    SearchSucceeded,
    truncate,
)
from app.websearch.settings import WebSearchConfig

API_PATH = "/api/v1/web"

# 4get expresses "safe search" as a tri-state nsfw filter.
_NSFW_BY_SAFESEARCH = {0: "yes", 1: "maybe", 2: "no"}

# 4get's "newer" filter takes a date, not a range keyword, so the shared
# time_range setting is resolved against today.
_TIME_RANGE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

# A unix timestamp sent as a string, e.g. "1767225600" or "1767225600.0".
_NUMERIC = re.compile(r"-?\d+(\.\d+)?")


def api_url(base_url: str) -> str:
    """The search endpoint for an instance base URL.

    4get's Apache rewrites extensionless paths (``^([^\\.]+)$ -> $1.php``), so
    ``/api/v1/web`` is the correct path and ``.php`` must not be appended.
    """
    base = base_url.rstrip("/")
    return base if base.endswith(API_PATH) else f"{base}{API_PATH}"


def _newer_than(time_range: Optional[str]) -> Optional[str]:
    days = _TIME_RANGE_DAYS.get(time_range or "")
    if not days:
        return None
    return (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()


def _build_params(query: str, cfg: WebSearchConfig) -> dict:
    params = {"s": query}
    # Omitted -> the instance's own default scraper.
    if cfg.fourget_scraper:
        params["scraper"] = cfg.fourget_scraper
    if cfg.fourget_country:
        params["country"] = cfg.fourget_country
    if cfg.fourget_lang:
        params["lang"] = cfg.fourget_lang
    nsfw = _NSFW_BY_SAFESEARCH.get(cfg.safesearch)
    if nsfw:
        params["nsfw"] = nsfw
    newer = _newer_than(cfg.time_range)
    if newer:
        params["newer"] = newer
    return params


def _format_date(value) -> Optional[str]:
    """4get dates are usually unix timestamps, sometimes raw strings, often null."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if not _NUMERIC.fullmatch(value):
            return value  # an upstream string passed straight through
        value = float(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None
    return None


def _text(value) -> str:
    """A text field from a scraper result; scrapers occasionally emit non-strings."""
    return value if isinstance(value, str) else ""


def parse_results(payload: dict, cfg: WebSearchConfig) -> List[SearchResult]:
    """Map 4get ``web[]`` to SearchResult, deduped and truncated."""
    results: List[SearchResult] = []
    seen = set()
    for item in payload.get("web") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url or not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        results.append(
            SearchResult(
                title=truncate(_text(item.get("title")) or url, 300),
                url=url,
                snippet=truncate(_text(item.get("description")), cfg.max_snippet_chars),
                date=_format_date(item.get("date")),
            )
        )
        if len(results) >= cfg.max_results:
            break
    return results


async def search(query: str, cfg: WebSearchConfig) -> SearchOutcome:
    """Run one 4get query (already validated by ``backends.search``).

    Never raises; failures come back as SearchFailed.
    """
    if not cfg.fourget_base_url:
        return SearchFailed(query, "unavailable", "4get base URL is not configured")

    def parse(payload: dict) -> SearchOutcome:
        status = payload.get("status")
        if status != "ok":
            # The error text is the status field itself, e.g. "Invalid scraper"
            # or "The server administrator disabled the API!".
            message = status if isinstance(status, str) and status else "4get returned an unknown error"
            return SearchFailed(query, "unavailable", f"4get: {message}")
        return SearchSucceeded(query, parse_results(payload, cfg))

    # 4get's own API never sets a status code; an HTTP error comes from a
    # reverse proxy or the web server in front of it.
    return await search_json(
        query,
        api_url(cfg.fourget_base_url),
        _build_params(query, cfg),
        cfg.timeout_seconds,
        provider=WEBSEARCH_PROVIDER_FOURGET,
        label=WEBSEARCH_PROVIDER_LABELS[WEBSEARCH_PROVIDER_FOURGET],
        parse=parse,
        not_json_message="4get did not return JSON (is the API enabled?)",
    )

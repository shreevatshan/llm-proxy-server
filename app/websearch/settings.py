"""Request-safe snapshot of the admin-managed web search settings."""

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Optional

from app.auth.models import WEBSEARCH_API_SURFACES, WEBSEARCH_DEFAULTS

logger = logging.getLogger(__name__)

SURFACE_ANTHROPIC_MESSAGES = "anthropic_messages"
SURFACE_CHAT_COMPLETIONS = "chat_completions"
SURFACE_RESPONSES = "responses"

# A safety net in case the process ever runs several workers: a PUT reloads
# the snapshot of the worker that served it, the others catch up here.
_REFRESH_INTERVAL_SECONDS = 30


@dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool = False
    searxng_base_url: Optional[str] = None
    engines: Optional[str] = None
    categories: Optional[str] = WEBSEARCH_DEFAULTS["categories"]
    language: Optional[str] = WEBSEARCH_DEFAULTS["language"]
    safesearch: int = WEBSEARCH_DEFAULTS["safesearch"]
    time_range: Optional[str] = None
    max_results: int = WEBSEARCH_DEFAULTS["max_results"]
    max_snippet_chars: int = WEBSEARCH_DEFAULTS["max_snippet_chars"]
    timeout_seconds: int = WEBSEARCH_DEFAULTS["timeout_seconds"]
    max_agentic_loops: int = WEBSEARCH_DEFAULTS["max_agentic_loops"]
    max_queries_per_turn: int = WEBSEARCH_DEFAULTS["max_queries_per_turn"]
    apply_to: frozenset = field(default_factory=lambda: frozenset(WEBSEARCH_API_SURFACES))
    enabled_providers: frozenset = field(default_factory=frozenset)

    def with_overrides(self, **overrides) -> "WebSearchConfig":
        return replace(self, **overrides)


def _parse_json_list(raw, default) -> list:
    if raw is None:
        return list(default)
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return list(default)
    if not isinstance(decoded, list):
        return list(default)
    return [x for x in decoded if isinstance(x, str)]


def _pick(row, name):
    value = getattr(row, name, None)
    return WEBSEARCH_DEFAULTS[name] if value is None else value


def config_from_row(row) -> WebSearchConfig:
    """Build a config from a WebSearchSettings row (None -> defaults)."""
    if row is None:
        return WebSearchConfig()
    return WebSearchConfig(
        enabled=bool(row.enabled),
        searxng_base_url=row.searxng_base_url or None,
        engines=row.engines or None,
        categories=_pick(row, "categories"),
        language=_pick(row, "language"),
        safesearch=_pick(row, "safesearch"),
        time_range=row.time_range or None,
        max_results=_pick(row, "max_results"),
        max_snippet_chars=_pick(row, "max_snippet_chars"),
        timeout_seconds=_pick(row, "timeout_seconds"),
        max_agentic_loops=_pick(row, "max_agentic_loops"),
        max_queries_per_turn=_pick(row, "max_queries_per_turn"),
        apply_to=frozenset(_parse_json_list(row.apply_to, WEBSEARCH_DEFAULTS["apply_to"])),
        enabled_providers=frozenset(_parse_json_list(row.enabled_providers, WEBSEARCH_DEFAULTS["enabled_providers"])),
    )


class WebSearchSettingsCache:
    """Immutable-snapshot settings holder; reads never touch the database."""

    def __init__(self) -> None:
        self._config = WebSearchConfig()
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> WebSearchConfig:
        return self._config

    def set_config(self, config: WebSearchConfig) -> None:
        self._config = config

    async def load_from_database(self) -> None:
        from app.auth.database import AsyncSessionLocal, get_websearch_settings

        async with AsyncSessionLocal() as db:
            row = await get_websearch_settings(db)
        self._config = config_from_row(row)

    async def start(self) -> None:
        await self.load_from_database()
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        task, self._refresh_task = self._refresh_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(_REFRESH_INTERVAL_SECONDS)
            try:
                await self.load_from_database()
            except Exception:
                logger.warning("Web search settings refresh failed", exc_info=True)

    def is_active(self, surface: str, provider_key: Optional[str]) -> bool:
        """True when interception applies to this API surface and provider."""
        cfg = self._config
        if not cfg.enabled or not cfg.searxng_base_url:
            return False
        if surface not in cfg.apply_to:
            return False
        if "*" in cfg.enabled_providers:
            return True
        return bool(provider_key) and provider_key in cfg.enabled_providers


websearch_settings_cache = WebSearchSettingsCache()

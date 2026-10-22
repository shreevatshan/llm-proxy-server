"""Request-safe snapshot of the web search settings, backends and user choices."""

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from app.auth.models import (
    WEBSEARCH_API_SURFACES,
    WEBSEARCH_BACKEND_PARAMS,
    WEBSEARCH_DEFAULTS,
    WEBSEARCH_PARAM_FIELDS,
    WEBSEARCH_PARAM_GROUPS,
    WEBSEARCH_PROVIDER_FOURGET,
)

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
    provider: str = WEBSEARCH_DEFAULTS["provider"]
    # SearXNG backend
    searxng_base_url: Optional[str] = None
    engines: Optional[str] = None
    categories: Optional[str] = WEBSEARCH_DEFAULTS["categories"]
    language: Optional[str] = WEBSEARCH_DEFAULTS["language"]
    safesearch: int = WEBSEARCH_DEFAULTS["safesearch"]
    time_range: Optional[str] = None
    # 4get backend
    fourget_base_url: Optional[str] = None
    fourget_scraper: Optional[str] = WEBSEARCH_DEFAULTS["fourget_scraper"]
    fourget_lang: Optional[str] = None
    fourget_country: Optional[str] = None
    # Shared
    max_results: int = WEBSEARCH_DEFAULTS["max_results"]
    max_snippet_chars: int = WEBSEARCH_DEFAULTS["max_snippet_chars"]
    timeout_seconds: int = WEBSEARCH_DEFAULTS["timeout_seconds"]
    max_agentic_loops: int = WEBSEARCH_DEFAULTS["max_agentic_loops"]
    max_queries_per_turn: int = WEBSEARCH_DEFAULTS["max_queries_per_turn"]
    apply_to: frozenset = field(default_factory=lambda: frozenset(WEBSEARCH_API_SURFACES))
    enabled_providers: frozenset = field(default_factory=frozenset)
    # Per-request overrides taken from the client's native Anthropic
    # web_search tool (max_uses, allowed_domains, blocked_domains).
    max_uses: Optional[int] = None
    allowed_domains: Tuple[str, ...] = ()
    blocked_domains: Tuple[str, ...] = ()
    # Attach web_search_result_location citations to answer text that links a
    # search result, and answer Claude Code's standalone WebSearch request
    # with one search + one summarizing model call.
    emit_citations: bool = True
    claude_code_shortcircuit: bool = True
    # The backend the connection and parameters above came from.
    backend_id: Optional[int] = None
    backend_name: Optional[str] = None

    @property
    def base_url(self) -> Optional[str]:
        """Connection URL of the selected backend."""
        if self.provider == WEBSEARCH_PROVIDER_FOURGET:
            return self.fourget_base_url
        return self.searxng_base_url

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


def _set_params(row) -> Dict[str, Any]:
    """The parameter columns of a row that are set."""
    return {f: getattr(row, f) for f in WEBSEARCH_PARAM_FIELDS if getattr(row, f, None) is not None}


@dataclass(frozen=True)
class BackendSpec:
    """A websearch_backends row, detached from the session (id None: unsaved)."""
    id: Optional[int]
    name: str
    type: str
    base_url: str
    enabled: bool = True
    user_selectable: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)  # only the set ones

    @classmethod
    def from_row(cls, row) -> "BackendSpec":
        own = WEBSEARCH_BACKEND_PARAMS.get(row.type, ())
        return cls(
            id=row.id, name=row.name, type=row.type, base_url=row.base_url,
            enabled=bool(row.enabled), user_selectable=bool(row.user_selectable),
            params={f: v for f, v in _set_params(row).items() if f in own},
        )

    def effective_params(self) -> Dict[str, Any]:
        """Every parameter of the type, unset ones filled from WEBSEARCH_DEFAULTS."""
        return {f: self.params.get(f, WEBSEARCH_DEFAULTS[f]) for f in WEBSEARCH_BACKEND_PARAMS[self.type]}


@dataclass(frozen=True)
class UserPrefs:
    """A user's choices; None inherits the admin default."""
    backend_id: Optional[int] = None
    max_results: Optional[int] = None
    max_snippet_chars: Optional[int] = None
    params: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)  # backend id -> set params


def settings_config(row) -> WebSearchConfig:
    """The admin knobs and default limits of a WebSearchSettings row, with no
    backend applied (None -> defaults)."""
    if row is None:
        return WebSearchConfig()
    return WebSearchConfig(
        enabled=bool(row.enabled),
        max_results=_pick(row, "max_results"),
        max_snippet_chars=_pick(row, "max_snippet_chars"),
        timeout_seconds=_pick(row, "timeout_seconds"),
        max_agentic_loops=_pick(row, "max_agentic_loops"),
        max_queries_per_turn=_pick(row, "max_queries_per_turn"),
        apply_to=frozenset(_parse_json_list(row.apply_to, WEBSEARCH_DEFAULTS["apply_to"])),
        enabled_providers=frozenset(_parse_json_list(row.enabled_providers, WEBSEARCH_DEFAULTS["enabled_providers"])),
    )


def apply_backend(cfg: WebSearchConfig, backend: Optional[BackendSpec]) -> WebSearchConfig:
    """Point a config at a backend: its type, URL and parameters.

    None clears the connection, which leaves interception inactive.
    """
    if backend is None:
        return replace(cfg, searxng_base_url=None, fourget_base_url=None, backend_id=None, backend_name=None)
    values = {f: WEBSEARCH_DEFAULTS[f] for f in WEBSEARCH_PARAM_FIELDS}
    values.update(backend.effective_params())
    fourget = backend.type == WEBSEARCH_PROVIDER_FOURGET
    return replace(
        cfg,
        provider=backend.type,
        searxng_base_url=None if fourget else backend.base_url,
        fourget_base_url=backend.base_url if fourget else None,
        backend_id=backend.id,
        backend_name=backend.name,
        **values,
    )


def user_param_overrides(backend_type: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """The values a user's parameters replace on a backend of this type.

    Only the type's own parameters count. Setting any member of a
    WEBSEARCH_PARAM_GROUPS group takes the whole group from the user, the unset
    members as None: SearXNG ignores categories when engines is sent, so mixing an
    inherited engine list with the user's categories would drop the categories.
    """
    own = WEBSEARCH_BACKEND_PARAMS.get(backend_type, ())
    overrides = {f: v for f, v in params.items() if f in own and v is not None}
    for group in WEBSEARCH_PARAM_GROUPS:
        members = [f for f in group if f in own]
        if any(f in overrides for f in members):
            for f in members:
                overrides.setdefault(f, None)
    return overrides


@dataclass(frozen=True)
class WebSearchSnapshot:
    """Settings, backends and every user's choices at one moment."""
    default: WebSearchConfig = field(default_factory=WebSearchConfig)
    default_backend_id: Optional[int] = None
    backends: Mapping[int, BackendSpec] = field(default_factory=dict)
    users: Mapping[int, UserPrefs] = field(default_factory=dict)
    # Resolved configs by user id; valid for this snapshot's lifetime only.
    _resolved: Dict[int, WebSearchConfig] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_rows(cls, settings, backends, user_rows, param_rows) -> "WebSearchSnapshot":
        specs = {b.id: BackendSpec.from_row(b) for b in backends}
        default_id = getattr(settings, "default_backend_id", None)
        default_backend = specs.get(default_id)
        if default_backend is not None and not default_backend.enabled:
            default_backend = None
        params: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for p in param_rows:
            set_params = _set_params(p)
            if set_params:
                params.setdefault(p.user_id, {})[p.backend_id] = set_params
        users = {
            u.user_id: UserPrefs(
                backend_id=u.backend_id, max_results=u.max_results,
                max_snippet_chars=u.max_snippet_chars, params=params.pop(u.user_id, {}),
            )
            for u in user_rows
        }
        # Parameter rows whose user row was pruned still apply on the default backend.
        users.update({uid: UserPrefs(params=p) for uid, p in params.items()})
        return cls(
            default=apply_backend(settings_config(settings), default_backend),
            default_backend_id=default_id,
            backends=specs,
            users=users,
        )

    def is_selectable(self, backend_id: Optional[int]) -> bool:
        """Whether users may pick (and keep using) this backend."""
        b = self.backends.get(backend_id)
        return b is not None and b.enabled and (b.user_selectable or b.id == self.default_backend_id)

    def user_backend(self, user_id: Optional[int]) -> Tuple[Optional[BackendSpec], bool]:
        """(backend the user's searches go to, whether their own pick fell back)."""
        default = self.backends.get(self.default.backend_id)
        prefs = self.users.get(user_id) if user_id is not None else None
        if prefs is None or prefs.backend_id is None:
            return default, False
        if self.is_selectable(prefs.backend_id):
            return self.backends[prefs.backend_id], False
        return default, True

    def resolve(self, user_id: Optional[int]) -> WebSearchConfig:
        """The config a request by this user runs with (None -> the default)."""
        prefs = self.users.get(user_id) if user_id is not None else None
        if prefs is None:
            return self.default
        cached = self._resolved.get(user_id)
        if cached is not None:
            return cached
        cfg = self.default
        backend, _ = self.user_backend(user_id)
        if backend is not None:
            if backend.id != cfg.backend_id:
                cfg = apply_backend(cfg, backend)
            overrides = user_param_overrides(backend.type, prefs.params.get(backend.id, {}))
            if overrides:
                cfg = replace(cfg, **overrides)
        limits = {
            k: v for k, v in (("max_results", prefs.max_results), ("max_snippet_chars", prefs.max_snippet_chars))
            if v is not None
        }
        if limits:
            cfg = replace(cfg, **limits)
        self._resolved[user_id] = cfg
        return cfg


async def load_snapshot(db) -> WebSearchSnapshot:
    from app.auth.database import load_websearch_snapshot_rows

    return WebSearchSnapshot.from_rows(*await load_websearch_snapshot_rows(db))


class WebSearchSettingsCache:
    """Immutable-snapshot settings holder; reads never touch the database."""

    def __init__(self) -> None:
        self._snapshot = WebSearchSnapshot()
        self._refresh_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> WebSearchConfig:
        """The default config: what a user with no choices of their own gets."""
        return self._snapshot.default

    @property
    def snapshot(self) -> WebSearchSnapshot:
        return self._snapshot

    def set_config(self, config: WebSearchConfig) -> None:
        """Install a default config with no backends or user choices (tests)."""
        self._snapshot = WebSearchSnapshot(default=config, default_backend_id=config.backend_id)

    def resolve_config(self, user_id: Optional[int]) -> WebSearchConfig:
        return self._snapshot.resolve(user_id)

    async def load_from_database(self) -> None:
        from app.auth.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            self._snapshot = await load_snapshot(db)

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
        """True when interception applies to this API surface and provider.

        Global, not per user: users can change where searches go but not whether
        they run, and every user falls back to the default backend checked here.
        """
        cfg = self._snapshot.default
        if not cfg.enabled or not cfg.base_url:
            return False
        if surface not in cfg.apply_to:
            return False
        if "*" in cfg.enabled_providers:
            return True
        return bool(provider_key) and provider_key in cfg.enabled_providers


websearch_settings_cache = WebSearchSettingsCache()

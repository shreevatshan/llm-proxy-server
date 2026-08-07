"""Fast, request-safe resolution of admin-managed model aliases."""

import json
from contextvars import ContextVar
from typing import Optional

from app.auth.database import AsyncSessionLocal, get_all_model_aliases
from app.auth.models import MODEL_ALIAS_API_SURFACES


original_model_name: ContextVar[Optional[str]] = ContextVar(
    "original_model_name", default=None
)

# The API surface (openai / anthropic / azure_openai) the current request
# arrived on. Set once per request by the tracking middleware. None means
# "unscoped" — callers without request context apply every mapping (fail-open).
current_api_surface: ContextVar[Optional[str]] = ContextVar(
    "current_api_surface", default=None
)

_ALL_SURFACES = frozenset(MODEL_ALIAS_API_SURFACES)


def _parse_apis(raw) -> frozenset[str]:
    """Decode the stored JSON array; NULL/empty/malformed ⇒ every surface."""
    if not raw:
        return _ALL_SURFACES
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return _ALL_SURFACES
    if not isinstance(decoded, list):
        return _ALL_SURFACES
    surfaces = frozenset(decoded) & _ALL_SURFACES
    return surfaces or _ALL_SURFACES


class ModelAliasResolver:
    """Immutable-snapshot alias resolver; reads never touch the database."""

    def __init__(self) -> None:
        self._aliases: dict[str, tuple[str, frozenset[str]]] = {}

    async def load_from_database(self) -> None:
        async with AsyncSessionLocal() as db:
            aliases = await get_all_model_aliases(db)
        self._aliases = {
            row.alias: (row.target_model_id, _parse_apis(row.apis))
            for row in aliases
            if row.enabled
        }

    def resolve(self, name: Optional[str], api: Optional[str] = None) -> Optional[str]:
        if name is None:
            return None
        entry = self._aliases.get(name)
        if entry is None:
            return name
        target, apis = entry
        if api is not None and api not in apis:
            return name          # this surface is unaffected
        return target


model_alias_resolver = ModelAliasResolver()


def apply_alias(name: Optional[str], api: Optional[str] = None) -> Optional[str]:
    """Resolve *name* and retain its client-facing value when it changes."""
    if api is None:
        api = current_api_surface.get()
    mapped = model_alias_resolver.resolve(name, api)
    if name is not None and mapped != name:
        original_model_name.set(name)
    return mapped


def echo_model_name(request) -> Optional[str]:
    """Return the name supplied by the client, falling back to routed model."""
    return original_model_name.get() or request.model

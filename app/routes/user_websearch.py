"""A user's own web search choices: which backend their searches go to and its
search parameters. Anything left unset follows the admin defaults.

Whether web search runs at all, and its cost knobs (rounds, queries, timeout),
stay with the admin: this API cannot reach them.
"""

import logging
import time
from typing import Dict, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.admin import AdminUser
from app.auth.database import delete_user_websearch, get_db, get_user_websearch, put_user_websearch
from app.auth.middleware import get_current_active_user
from app.auth.models import (
    WEBSEARCH_BACKEND_PARAMS, WEBSEARCH_LIMITS, WEBSEARCH_PROVIDER_SEARXNG,
    User, UserWebSearchUpdate,
)
from app.websearch.settings import WebSearchSnapshot, load_snapshot, websearch_settings_cache

logger = logging.getLogger(__name__)

# Prefixed /auth/, so the CSRF middleware already covers it.
router = APIRouter(prefix="/auth/websearch", tags=["websearch"])

# Engine lists change rarely; caching them keeps users from hammering an instance.
_ENGINES_TTL_SECONDS = 600
_engines_cache: Dict[str, Tuple[float, dict]] = {}


def _require_user(current_user) -> User:
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The admin account sets web search defaults in the admin dashboard",
        )
    return current_user


def _params_of(row, backend_type: str) -> dict:
    return {f: getattr(row, f, None) for f in WEBSEARCH_BACKEND_PARAMS.get(backend_type, ())}


async def _payload(db: AsyncSession, user_id: int) -> dict:
    """Everything the preferences tab shows. Backend URLs are never included:
    they are often internal hostnames."""
    snapshot: WebSearchSnapshot = await load_snapshot(db)
    row, by_backend = await get_user_websearch(db, user_id)
    default = snapshot.default
    visible = sorted(
        (b for b in snapshot.backends.values() if snapshot.is_selectable(b.id)),
        key=lambda b: b.name.lower(),
    )
    backend, fell_back = snapshot.user_backend(user_id)
    chosen = snapshot.backends.get(row.backend_id) if row else None
    cfg = snapshot.resolve(user_id)
    params_by_backend = {
        str(bid): _params_of(p, snapshot.backends[bid].type)
        for bid, p in by_backend.items() if bid in snapshot.backends
    }
    return {
        "enabled": default.enabled and bool(default.base_url),
        "limits": {k: {"min": lo, "max": hi} for k, (lo, hi) in WEBSEARCH_LIMITS.items()},
        "defaults": {
            "backend_id": default.backend_id,
            "max_results": default.max_results,
            "max_snippet_chars": default.max_snippet_chars,
        },
        "backends": [
            {
                "id": b.id,
                "name": b.name,
                "type": b.type,
                "is_default": b.id == default.backend_id,
                "params": b.effective_params(),
            }
            for b in visible
        ],
        "selection": {
            "backend_id": row.backend_id if row else None,
            "max_results": row.max_results if row else None,
            "max_snippet_chars": row.max_snippet_chars if row else None,
            "params_by_backend": params_by_backend,
        },
        "effective": {
            "backend_id": cfg.backend_id,
            "backend_name": cfg.backend_name,
            "type": backend.type if backend else None,
            "fell_back": fell_back,
            "chosen_backend_id": row.backend_id if row else None,
            # The pick may not be in `backends` (unavailable), but its name is shown.
            "chosen_backend_name": chosen.name if chosen else None,
            "max_results": cfg.max_results,
            "max_snippet_chars": cfg.max_snippet_chars,
            "params": {f: getattr(cfg, f) for f in WEBSEARCH_BACKEND_PARAMS[backend.type]} if backend else {},
        },
    }


@router.get("")
async def get_my_websearch(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """The caller's web search choices, the backends they can pick, and what
    their searches currently use."""
    user = _require_user(current_user)
    return await _payload(db, user.id)


@router.put("")
async def put_my_websearch(
    body: UserWebSearchUpdate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Replace the caller's choices. The parameters are for the backend their
    searches go to: the chosen one, else the default; those for other backends
    are kept. A pick that became unavailable may be sent back unchanged, so
    saving other fields does not drop it (it applies again once restored)."""
    user = _require_user(current_user)
    snapshot = await load_snapshot(db)
    prefs = snapshot.users.get(user.id)
    kept = prefs is not None and body.backend_id == prefs.backend_id
    if body.backend_id is not None and not kept and not snapshot.is_selectable(body.backend_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="That backend is not available; pick another one or use the default",
        )
    target = body.backend_id if snapshot.is_selectable(body.backend_id) else snapshot.default.backend_id
    params = body.params_for(snapshot.backends[target].type) if target is not None else {}
    await put_user_websearch(
        db, user.id, body.backend_id,
        {"max_results": body.max_results, "max_snippet_chars": body.max_snippet_chars},
        target, params,
    )
    await websearch_settings_cache.load_from_database()
    return await _payload(db, user.id)


@router.delete("")
async def reset_my_websearch(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Drop all of the caller's choices, for every backend."""
    user = _require_user(current_user)
    await delete_user_websearch(db, user.id)
    await websearch_settings_cache.load_from_database()
    return await _payload(db, user.id)


@router.get("/engines")
async def get_backend_engines(
    backend_id: int = Query(..., description="A SearXNG backend the caller can pick"),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Engines and categories of a SearXNG backend, for the parameter pickers."""
    from app.websearch.searxng import SearxngConfigError, fetch_engines

    _require_user(current_user)
    snapshot = await load_snapshot(db)
    backend = snapshot.backends.get(backend_id)
    if backend is None or not snapshot.is_selectable(backend_id) or backend.type != WEBSEARCH_PROVIDER_SEARXNG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No SearXNG backend with that id")

    cached: Optional[Tuple[float, dict]] = _engines_cache.get(backend.base_url)
    if cached and time.monotonic() - cached[0] < _ENGINES_TTL_SECONDS:
        return cached[1]
    try:
        data = await fetch_engines(backend.base_url, snapshot.default.timeout_seconds)
    except SearxngConfigError as e:
        # The message names the instance URL; users get a generic one.
        logger.warning("Engine list for web search backend %s failed: %s", backend.name, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=f"{backend.name} did not return its engine list")
    _engines_cache[backend.base_url] = (time.monotonic(), data)
    return data

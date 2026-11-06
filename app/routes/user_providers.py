"""A user's own provider instances: BYOK (bring your own key).

A user adds an upstream account of their own (Azure, Google, Bedrock, PlatformAI
or any OpenAI/Anthropic-compatible endpoint) and reaches it through the proxy
with their normal proxy API key. Only they can use it; it is exempt from the
admin's quotas, and its traffic is still recorded as usage (see app/byok.py).

Every endpoint is scoped to the caller: a provider key the caller does not own
reads as not found (404), exactly like one that does not exist. Prefixed
/auth/, so the CSRF middleware already covers it; authentication is the session
(JWT) only, so a proxy API key cannot manage instances.
"""

import asyncio
import logging
import time
import urllib.parse
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import secret_box
from app.auth.admin import AdminUser
from app.auth.database import (
    count_user_providers, create_provider_credentials, delete_provider_credentials,
    get_db, get_model_configuration, get_models_by_provider, get_provider_credentials,
    get_user_provider_credential, get_user_provider_credentials, toggle_model_configuration,
    toggle_provider_credentials, update_provider_credentials,
)
from app.auth.middleware import get_current_active_user
from app.auth.models import (
    User, UserProviderCreate, UserProviderModelToggle, UserProviderResponse, UserProviderUpdate,
)
from app.byok import (
    BYOK_PROVIDER_TYPES, MAX_BYOK_INSTANCES_PER_USER, byok_instance_name, provider_key_of_model, slug_of,
)
from app.providers.auto_sync import sync_byok_provider
from app.providers.provider_manager import provider_manager
from app.providers.provider_service import (
    check_platformai_create, credentials_response_fields, infer_azure_supported_apis,
    platformai_update_deployments, provider_update_data,
)
from app.providers.url_guard import URL_FIELDS, UnsafeUpstreamURL, validate_credential_urls

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/providers", tags=["byok"])

# Seconds between a user's calls that reach an upstream (sync) or rebuild their
# instance (create/update/toggle/delete). In-memory, per process.
SYNC_COOLDOWN_SECONDS = 30.0
WRITE_COOLDOWN_SECONDS = 2.0

_last_action: Dict[Tuple[int, str], float] = {}
_user_locks: Dict[int, asyncio.Lock] = {}


def _require_user(current_user) -> User:
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The admin account manages providers in the admin dashboard",
        )
    return current_user


def _user_lock(user_id: int) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = _user_locks[user_id] = asyncio.Lock()
    return lock


def _cooldown(user_id: int, action: str, seconds: float) -> None:
    """429 when this user did ``action`` less than ``seconds`` ago; else record it."""
    now = time.monotonic()
    last = _last_action.get((user_id, action))
    if last is not None and now - last < seconds:
        wait = int(seconds - (now - last)) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Please wait {wait}s before trying again",
            headers={"Retry-After": str(wait)},
        )
    _last_action[(user_id, action)] = now


def _audit(user: User, action: str, cred, extra: str = "") -> None:
    """One log line per change: who, what, which upstream host. Never secrets."""
    host = None
    for field in URL_FIELDS:
        value = getattr(cred, field, None)
        if value:
            host = urllib.parse.urlparse(value).hostname
            break
    logger.info(
        "BYOK %s by user %s (id %s): %s type=%s host=%s%s",
        action, user.username, user.id, cred.provider_key, cred.provider_type, host,
        f" {extra}" if extra else "",
    )


async def _owned(db: AsyncSession, user_id: int, provider_key: str):
    cred = await get_user_provider_credential(db, user_id, urllib.parse.unquote(provider_key))
    if cred is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")
    return cred


async def _check_urls(fields: dict) -> None:
    """The SSRF guard (blocking DNS, so off the event loop). 422 on refusal."""
    try:
        await asyncio.to_thread(validate_credential_urls, fields)
    except UnsafeUpstreamURL as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


def _check_required_credentials(provider_type: str, data: dict) -> None:
    """A BYOK instance must carry its own credentials (no server fallbacks). 422."""
    def given(name):
        value = data.get(name)
        return isinstance(value, str) and bool(value.strip())

    missing = []
    if provider_type == "azure":
        missing = [n for n in ("endpoint", "api_key") if not given(n)]
    elif provider_type == "google":
        missing = [n for n in ("api_key",) if not given(n)]
    elif provider_type == "bedrock":
        missing = [n for n in ("access_key_id", "secret_access_key") if not given(n)]
    elif provider_type == "custom":
        if not (given("base_url") or given("anthropic_base_url")):
            missing = ["base_url"]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required field(s): {', '.join(missing)}",
        )


async def _response(db: AsyncSession, cred, user_id: int) -> UserProviderResponse:
    models = await get_models_by_provider(db, cred.provider_key)
    sync = provider_manager.get_sync_status(cred.provider_key) or {}
    load_error = provider_manager.byok_load_errors.get(cred.provider_key)
    if load_error is None and not secret_box.can_decrypt(cred):
        load_error = "Stored credentials cannot be decrypted; re-enter them"
    return UserProviderResponse(
        **credentials_response_fields(cred),
        slug=slug_of(cred.instance_name, user_id),
        admin_disabled=bool(cred.admin_disabled),
        model_count=len(models),
        enabled_model_count=sum(1 for m in models if m.is_enabled),
        load_error=load_error,
        sync_state=sync.get("state"),
        sync_error=sync.get("error"),
        last_sync_at=sync.get("last_sync_at"),
    )


@router.get("")
async def list_my_providers(
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """The caller's BYOK instances, plus what the add form needs."""
    user = _require_user(current_user)
    rows = await get_user_provider_credentials(db, user.id)
    return {
        "user_id": user.id,
        "instance_prefix": byok_instance_name(user.id, ""),
        "max_instances": MAX_BYOK_INSTANCES_PER_USER,
        "provider_types": list(BYOK_PROVIDER_TYPES),
        "providers": [await _response(db, cred, user.id) for cred in rows],
    }


@router.get("/detail", response_model=UserProviderResponse)
async def get_my_provider(
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    user = _require_user(current_user)
    return await _response(db, await _owned(db, user.id, provider_key), user.id)


@router.post("", response_model=UserProviderResponse)
async def create_my_provider(
    body: UserProviderCreate,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Add an instance: validate, store (secrets encrypted), then load and sync just it."""
    user = _require_user(current_user)
    _cooldown(user.id, "write", WRITE_COOLDOWN_SECONDS)

    infer_azure_supported_apis(body)
    try:
        check_platformai_create(body)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    if body.provider_type == "custom" and not body.supported_apis:
        body.supported_apis = ["openai"]

    raw = body.model_dump(exclude_unset=False)
    _check_required_credentials(body.provider_type, raw)
    await _check_urls(raw)

    instance_name = byok_instance_name(user.id, body.slug)
    provider_key = f"{body.provider_name}:{instance_name}"

    # Cap check and insert under one per-user lock: concurrent creates cannot overshoot.
    async with _user_lock(user.id):
        if await count_user_providers(db, user.id) >= MAX_BYOK_INSTANCES_PER_USER:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"You can add at most {MAX_BYOK_INSTANCES_PER_USER} providers",
            )
        if await get_provider_credentials(db, provider_key) is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="You already have a provider with that name")

        import json
        fields = secret_box.encrypt_fields({
            name: getattr(body, name)
            for name in (
                "endpoint", "api_key", "discovery_api_version", "azure_backend", "region",
                "access_key_id", "secret_access_key", "base_url", "deployments",
                "openai_deployments", "anthropic_deployments", "deployment_groups",
                "dynamic_discovery", "anthropic_base_url", "anthropic_auth_style",
                "oauth_client_id", "oauth_client_secret", "oauth_refresh_token",
                "oauth_accounts_url", "pai_token_id",
            )
        })
        if body.provider_type == "azure":
            fields["azure_backend"] = fields.get("azure_backend") or "openai"
        try:
            cred = await create_provider_credentials(
                db=db,
                provider_type=body.provider_type,
                instance_name=instance_name,
                enabled=body.enabled,
                provider_name=body.provider_name,
                supported_apis=json.dumps(body.supported_apis) if body.supported_apis else None,
                owner_user_id=user.id,
                admin_disabled=False,
                **fields,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    _audit(user, "create", cred)
    await sync_byok_provider(db, cred.provider_key)
    await db.refresh(cred)
    return await _response(db, cred, user.id)


@router.put("", response_model=UserProviderResponse)
async def update_my_provider(
    body: UserProviderUpdate,
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit an instance. Its identity (provider name, instance name) is fixed."""
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key)
    _cooldown(user.id, "write", WRITE_COOLDOWN_SECONDS)

    if body.instance_name is not None and body.instance_name != cred.instance_name:
        raise HTTPException(status_code=422, detail="A provider's name cannot be changed; add a new one instead")
    if body.provider_name is not None and body.provider_name != cred.provider_name:
        raise HTTPException(status_code=422, detail="A provider's name cannot be changed; add a new one instead")

    update_data = provider_update_data(body, exclude=("instance_name", "provider_name"))
    if "enabled" in update_data and cred.admin_disabled:
        update_data.pop("enabled")  # the admin's switch wins; toggle refuses it outright
    try:
        platformai_update_deployments(update_data, cred)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    merged = {name: getattr(cred, name, None) for name in URL_FIELDS}
    merged.update({k: v for k, v in update_data.items() if k in URL_FIELDS})
    await _check_urls(merged)

    update_data = secret_box.encrypt_fields(update_data)
    try:
        cred = await update_provider_credentials(db, cred.provider_key, **update_data)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    if cred is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Provider not found")

    _audit(user, "update", cred)
    await sync_byok_provider(db, cred.provider_key)
    await db.refresh(cred)
    return await _response(db, cred, user.id)


@router.put("/toggle", response_model=UserProviderResponse)
async def toggle_my_provider(
    enabled: bool = Query(...),
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Switch an instance on or off. Refused while the admin has it switched off."""
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key)
    if cred.admin_disabled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This provider was disabled by an administrator",
        )
    _cooldown(user.id, "write", WRITE_COOLDOWN_SECONDS)
    await toggle_provider_credentials(db, cred.provider_key, enabled)
    if enabled:
        await sync_byok_provider(db, cred.provider_key)
    else:
        await provider_manager.remove_byok_provider(cred.provider_key, forget_owner=False)
        provider_manager.model_cache.update_single_provider_config(cred.provider_key, False)
    await db.refresh(cred)
    _audit(user, "enable" if enabled else "disable", cred)
    return await _response(db, cred, user.id)


@router.post("/sync", response_model=UserProviderResponse)
async def sync_my_provider(
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-discover an instance's models (contacts the upstream; rate limited)."""
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key)
    _cooldown(user.id, f"sync:{cred.provider_key}", SYNC_COOLDOWN_SECONDS)
    await sync_byok_provider(db, cred.provider_key)
    await db.refresh(cred)
    return await _response(db, cred, user.id)


@router.get("/models")
async def list_my_provider_models(
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key)
    models = await get_models_by_provider(db, cred.provider_key)
    return {
        "provider_key": cred.provider_key,
        "models": [
            {"model_id": m.model_id, "model_name": m.model_name, "is_enabled": bool(m.is_enabled)}
            for m in sorted(models, key=lambda m: m.model_name or m.model_id)
        ],
    }


@router.put("/models/toggle")
async def toggle_my_provider_model(
    body: UserProviderModelToggle,
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Switch one model of the caller's own instance on or off."""
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key_of_model(body.model_id) or "")
    model = await get_model_configuration(db, body.model_id)
    if model is None or model.provider_key != cred.provider_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    await toggle_model_configuration(db, body.model_id, body.enabled)
    return {"model_id": body.model_id, "enabled": body.enabled}


@router.delete("")
async def delete_my_provider(
    provider_key: str = Query(...),
    current_user=Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an instance and its models. Usage history is kept."""
    user = _require_user(current_user)
    cred = await _owned(db, user.id, provider_key)
    _cooldown(user.id, "write", WRITE_COOLDOWN_SECONDS)
    key = cred.provider_key
    _audit(user, "delete", cred)
    await delete_provider_credentials(db, key)
    await provider_manager.remove_byok_provider(key, forget_owner=True)
    return {"message": "Provider deleted", "provider_key": key}

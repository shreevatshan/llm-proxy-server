"""Admin routes for user management and system administration."""

from fastapi import APIRouter, Request, Depends, HTTPException, status, Response, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from typing import Annotated, List, Optional, Literal
import os
import json
import urllib.parse
import logging
import io

from app.auth.database import (
    get_db, get_user_by_username, get_user_by_email, create_user, get_user_by_id, permanently_delete_user,
    get_all_provider_configurations, get_all_model_configurations, get_models_by_provider,
    create_or_update_provider_configuration, create_or_update_model_configuration, get_model_configuration,
    get_all_model_aliases, get_model_alias, get_model_alias_by_id, upsert_model_alias,
    delete_model_alias, delete_model_aliases, reorder_model_aliases,
    toggle_provider_configuration, toggle_model_configuration, bulk_toggle_all_models,
    search_models_and_providers, get_all_provider_credentials, get_provider_credentials,
    create_provider_credentials, update_provider_credentials, delete_provider_credentials,
    clear_all_model_configurations, admin_reset_user_password, update_user_profile, get_usage_aggregates, get_usage_years,
    delete_usage_records,
    count_usage_requests,
    usage_user_ids_for_pool,
    load_user_management_settings, upsert_user_management_settings,
    load_pool_settings, upsert_pool_settings,
    list_model_groups, create_model_group, update_model_group, delete_model_group,
    set_group_members, get_model_group_limits, model_group_overview,
    reload_reference_caches, clear_model_group_members, clear_outages,
    get_all_outages, get_outage, create_outages, update_outage_description, delete_outages,
    get_websearch_settings, upsert_websearch_settings,
    list_websearch_backends, get_websearch_backend, create_websearch_backend,
    update_websearch_backend, delete_websearch_backend, count_backend_users,
    normalize_new_username, get_profile, get_profile_by_name, list_profiles, create_profile,
    update_profile, delete_profile, assign_users_profile, list_profile_member_ids, list_profile_members,
    upsert_profile_limits, upsert_profile_group_limit, set_group_limit_for_all_profiles,
    get_profile_limits, set_profile_model_mode, list_profile_model_exceptions,
    upsert_profile_model_exception, delete_profile_model_exception, set_profile_model_exceptions_bulk,
)
from app.auth.webhook import send_signup_webhook
from app.auth.middleware import get_current_admin
from app.auth.models import (
    User, AdminUserCreate, UserResponse, DEFAULT_PROFILE_ID,
    ProfileCreate, ProfileUpdate, ProfileResponse, ProfileLimitsUpdate, ProfileOverallLimitsUpdate,
    ProfileLimitsResponse,
    ProfileMemberResponse, UserProfileAssign, ProviderConfigurationResponse, ModelConfigurationResponse,
    ModelManagementTree, ToggleRequest, BulkToggleRequest, BulkUserActionRequest, ModelSearchResponse,
    ProviderCredentialsCreate, ProviderCredentialsUpdate, ProviderCredentialsResponse,
    AdminPasswordReset, AdminUserModify, VALID_AZURE_BACKENDS,
    UserManagementSettingsResponse, UserManagementSettingsUpdate,
    ModelGroupCreate, ModelGroupUpdate, ModelGroupMembersUpdate, ModelGroupResponse,
    ModelGroupMemberResponse, ProfileGroupLimitResponse,
    ModelAliasUpsert, ModelAliasResponse, ModelAliasReorder, ModelAliasBulkDelete, ModelAliasTestMatch,
    MODEL_ALIAS_API_SURFACES, validate_alias_pattern,
    OutageCreate, OutageUpdate, OutageBulkResolve, OutageResponse,
    AdminPoolResponse, AdminPoolLimitsUpdate, PoolSettingsResponse, PoolSettingsUpdate,
    WebSearchSettingsResponse, WebSearchSettingsUpdate, WebSearchTestRequest, WebSearchEnginesRequest,
    WebSearchBackendCreate, WebSearchBackendUpdate, WebSearchBackendResponse,
    FOURGET_WEB_SCRAPERS, WEBSEARCH_PROVIDER_LABELS, WEBSEARCH_PROVIDER_SEARXNG,
)
from app.auth.admin import AdminUser, authenticate_admin, is_admin_enabled, get_admin_email
from app.auth.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.providers.provider_manager import provider_manager
from app.providers.auto_sync import auto_sync_on_provider_change, sync_provider_models
from app.providers.azure_deployments import merge_azure_deployments, normalize_azure_deployments, normalize_deployment_groups, validate_deployment_names
from datetime import timedelta, datetime
from pydantic import BaseModel
import asyncio

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/frontend/templates")

logger = logging.getLogger(__name__)


def _cookie_secure() -> bool:
    """Whether auth cookies should carry the Secure flag.

    Defaults to True (production-safe); set COOKIE_SECURE=false for local
    plaintext-HTTP development.
    """
    return os.getenv("COOKIE_SECURE", "true").lower() != "false"


# Placeholder returned instead of stored secrets (see _mask_secret).
_SECRET_MASK = "********"

# Secret fields that _mask_secret masks in read/detail responses. A client
# echoing the mask back on update means "leave unchanged", so these values
# must never be persisted (see _strip_masked_secrets).
_MASKED_SECRET_FIELDS = ("api_key", "secret_access_key")


def _mask_secret(value: Optional[str]) -> Optional[str]:
    """Mask a stored secret for read/detail responses (write-only semantics).

    Returns a fixed placeholder when a value is set (so the UI knows a secret
    exists) and None when it is absent. The real value is never returned.
    """
    return _SECRET_MASK if value else None


def _strip_masked_secrets(update_data: dict) -> dict:
    """Drop secret fields whose submitted value is the mask placeholder.

    Detail responses return secrets as ``_SECRET_MASK``; if a client round-trips
    that literal on an update, persisting it would corrupt the stored credential.
    Treat it as "unchanged" and strip it before writing.
    """
    for field in _MASKED_SECRET_FIELDS:
        if update_data.get(field) == _SECRET_MASK:
            update_data.pop(field)
    return update_data


def _provider_update_data(provider_data: BaseModel, *, exclude: tuple = ()) -> dict:
    """Column values an update/upsert request writes.

    Only fields the client set (or validators derived) that are not None;
    masked secrets mean "unchanged" and are dropped; ``supported_apis`` is
    stored as JSON text.
    """
    update_data = {
        field: value
        for field, value in provider_data.model_dump(exclude_unset=True).items()
        if value is not None and field not in exclude
    }
    _strip_masked_secrets(update_data)
    if isinstance(update_data.get('supported_apis'), list):
        update_data['supported_apis'] = json.dumps(update_data['supported_apis'])
    return update_data


def _parse_azure_deployment_fields(raw_deployments):
    groups = normalize_azure_deployments(raw_deployments)
    return groups, merge_azure_deployments(groups)


class AdminLogin(BaseModel):
    username: str
    password: str


@router.get("/", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page."""
    if not is_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin interface is disabled"
        )
    
    import time
    return templates.TemplateResponse(
        "auth/admin_login.html",
        {
            "request": request,
            "title": "Admin Login - LLM Proxy Server",
            "cache_version": str(int(time.time()))
        }
    )


@router.post("/login")
async def admin_login(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Admin login endpoint."""
    if not is_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin interface is disabled"
        )
    
    # Get form data from request
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required"
        )
    
    # Authenticate admin
    admin_user = authenticate_admin(username, password)
    if not admin_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": admin_user.username}, 
        expires_delta=access_token_expires,
        is_admin=True
    )
    
    # Redirect to admin dashboard
    from fastapi.responses import RedirectResponse
    redirect_response = RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)
    
    # Set HTTP-only cookie for web interface on the redirect response
    redirect_response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax"
    )
    
    return redirect_response


# Model Management API Endpoints

@router.get("/models", response_model=ModelManagementTree)
async def get_models_management(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get model management tree with providers and models."""
    try:
        # Get all providers from the unified ProviderCredentials system
        provider_credentials = await get_all_provider_credentials(db)
        models = await get_all_model_configurations(db)

        # Index models by provider_key once (O(M)) instead of re-scanning the
        # full model list per provider (O(P×M)).
        models_by_provider: dict = {}
        for m in models:
            models_by_provider.setdefault(m.provider_key, []).append(m)

        # Build provider response with model counts
        provider_responses = []
        for provider in provider_credentials:
            provider_models = models_by_provider.get(provider.provider_key, [])
            enabled_models = [m for m in provider_models if m.is_enabled]
            
            # Parse supported_apis for list view
            _supported_apis = None
            if provider.supported_apis:
                try:
                    _supported_apis = json.loads(provider.supported_apis)
                except (ValueError, TypeError):
                    _supported_apis = ["openai"]

            provider_response = ProviderConfigurationResponse(
                id=provider.id,
                provider_key=provider.provider_key,
                provider_type=provider.provider_type,
                instance_name=provider.instance_name,
                provider_name=provider.provider_name,
                enabled=provider.enabled,  # Using unified ProviderCredentials.enabled field
                model_count=len(provider_models),
                enabled_model_count=len(enabled_models),
                supported_apis=_supported_apis,
                created_at=provider.created_at,
                updated_at=provider.updated_at
            )
            provider_responses.append(provider_response)
        
        # Calculate totals
        total_models = len(models)
        enabled_models = len([m for m in models if m.is_enabled])
        
        return ModelManagementTree(
            providers=provider_responses,
            total_models=total_models,
            enabled_models=enabled_models
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model management data: {str(e)}"
        )


@router.get("/models/provider", response_model=List[ModelConfigurationResponse])
async def get_provider_models(
    provider_key: str = Query(..., description="Provider key to get models for"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get all models for a specific provider using query parameter."""
    try:
        # No need to URL decode since Query parameter handles this automatically
        models = await get_models_by_provider(db, provider_key)
        return [ModelConfigurationResponse.from_orm(model) for model in models]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider models: {str(e)}"
        )


@router.post("/models/sync")
async def sync_models_from_providers(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Sync models from providers to database."""
    try:
        # First, get all provider credentials from database to know which providers are valid
        valid_providers = await get_all_provider_credentials(db)
        valid_provider_keys = {p.provider_key for p in valid_providers}
        
        # Force a complete provider manager refresh to handle provider renames
        logger.info("Forcing complete provider manager refresh before sync...")
        await provider_manager.refresh_providers_from_database()
        
        # Get all models from provider manager (this will use the refreshed providers)
        all_models = await provider_manager._fetch_all_models()
        
        # Filter models to only include those from valid providers in database
        valid_models = []
        synced_model_ids = []
        for model in all_models:
            # Extract provider key from model
            if '/' in model.id:
                provider_key = model.id.split('/', 1)[0]
            elif hasattr(model, 'owned_by') and model.owned_by:
                provider_key = model.owned_by
            elif hasattr(model, 'provider') and model.provider:
                provider_key = model.provider
            else:
                logger.warning(f"Cannot determine provider for model {model.id}")
                continue

            # Only include models from providers that exist in database
            if provider_key in valid_provider_keys:
                valid_models.append(model)
                synced_model_ids.append(model.id)
            else:
                logger.debug(f"Skipping model {model.id} from provider {provider_key} (not in database)")
        
        # Track providers that actually had models synced
        synced_providers = set()
        synced_models = 0
        
        for model in valid_models:
            # Extract provider key and model name
            if '/' in model.id:
                provider_key = model.id.split('/', 1)[0]
                model_name = model.id.split('/', 1)[1]
            elif hasattr(model, 'owned_by') and model.owned_by:
                provider_key = model.owned_by
                model_name = model.id
            elif hasattr(model, 'provider') and model.provider:
                provider_key = model.provider
                model_name = model.id
            else:
                logger.warning(f"Cannot determine provider for model {model.id}, skipping")
                continue
            
            # Track which provider this model belongs to
            synced_providers.add(provider_key)
            
            # Check if model already exists to preserve its enabled state
            existing_model = await get_model_configuration(db, model.id)
            is_enabled = existing_model.is_enabled if existing_model else True
            
            await create_or_update_model_configuration(
                db, model.id, provider_key, model_name, is_enabled, commit=False
            )
            synced_models += 1

        # One commit for the whole sync rather than one write-lock acquisition per model.
        await db.commit()

        # Providers that were fetched but listed nothing (error, timeout or empty),
        # and providers that weren't fetched at all (disabled). Their saved models
        # aren't stale — the provider just wasn't reached — so keep them out of the
        # stale list, where "Remove" would drop their admin-set enabled states.
        fetched_provider_keys = set(provider_manager.providers.keys())
        failed_providers = []
        for key in sorted((valid_provider_keys & fetched_provider_keys) - synced_providers):
            sync_status = provider_manager.get_sync_status(key) or {}
            failed_providers.append({
                "provider_key": key,
                "error": sync_status.get("error") or "No models returned",
            })
        unfetched_provider_keys = valid_provider_keys - fetched_provider_keys

        # Identify stale models (models in DB but not synced)
        from app.auth.database import identify_stale_models
        stale_models = await identify_stale_models(
            db,
            synced_model_ids,
            exclude_provider_keys={p["provider_key"] for p in failed_providers} | unfetched_provider_keys,
        )

        # Determine the appropriate message
        if len(valid_provider_keys) == 0:
            message = "No providers found in database"
        elif len(synced_providers) == 0:
            message = f"No models found from {len(valid_provider_keys)} providers in database"
        else:
            message = f"Synced {synced_models} models from {len(synced_providers)} providers"
        if failed_providers:
            message += f"; {len(failed_providers)} provider(s) failed to sync"
        
        logger.info(f"Sync completed: {message}")
        logger.info(f"Total providers in DB: {len(valid_provider_keys)}")
        logger.info(f"Providers with models: {len(synced_providers)}")
        logger.info(f"Models synced: {synced_models}")
        logger.info(f"Stale models found: {len(stale_models)}")
        if failed_providers:
            logger.warning(f"Providers failed to sync: {failed_providers}")

        # Update the cache with all synced models
        provider_manager.model_cache.update_models(valid_models)
        logger.info(f"Cache updated with {len(valid_models)} models")
        
        return {
            "message": message,
            "providers_synced": len(synced_providers),
            "models_synced": synced_models,
            "total_providers_in_db": len(valid_provider_keys),
            "stale_models": stale_models,
            "stale_count": len(stale_models),
            "failed_providers": failed_providers,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync models: {str(e)}"
        )


class RemoveStaleModelsRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_ids: List[str]


@router.post("/models/remove-stale")
async def remove_stale_models(
    request: RemoveStaleModelsRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Remove stale models from the database."""
    try:
        from app.auth.database import delete_stale_models
        
        deleted_count = await delete_stale_models(db, request.model_ids)
        # Group members, access exceptions and aliases naming them were cleaned up too.
        await reload_reference_caches()

        return {
            "message": f"Successfully removed {deleted_count} stale model(s)",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove stale models: {str(e)}"
        )


@router.post("/models/refresh")
async def refresh_models_from_providers(
    current_admin: AdminUser = Depends(get_current_admin)
):
    """Clear all model configurations and refresh with current models from providers."""
    try:
        # Use the provider manager's refresh method
        result = await provider_manager.refresh_models_from_providers()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return {
            "message": f"Model refresh completed: cleared {result['cleared']} old models, created {result['created']} new models",
            "cleared": result["cleared"],
            "created": result["created"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh models: {str(e)}"
        )


@router.post("/system/reinit")
async def reinit_system(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reinitialize system by clearing ALL provider and model configurations from database."""
    try:
        # Clear all model configurations
        models_cleared = await clear_all_model_configurations(db)
        
        # Clear all provider credentials
        providers = await get_all_provider_credentials(db)
        providers_cleared = len(providers)
        
        for provider in providers:
            await db.delete(provider)

        # Nothing is left for a group to name; empty them rather than keep stale members.
        await clear_model_group_members(db)
        await clear_outages(db)
        await db.commit()

        # Refresh provider manager from database (should now be empty)
        await provider_manager.refresh_providers_from_database()
        await reload_reference_caches()
        
        return {
            "message": f"System reinitialized: cleared {providers_cleared} providers and {models_cleared} models",
            "providers_cleared": providers_cleared,
            "models_cleared": models_cleared
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reinitialize system: {str(e)}"
        )


@router.put("/models/provider/toggle")
async def toggle_provider(
    toggle_data: ToggleRequest,
    provider_key: str = Query(..., description="Provider key to toggle"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Turn a provider on or off. Its models keep their own flags: a model is served
    only while both it and its provider are on."""
    try:
        # Log for debugging
        logging.info(f"Admin provider toggle request - Provider Key: {provider_key}, Enabled: {toggle_data.enabled}")
        
        # Use toggle_provider_configuration which properly updates cache
        success = await toggle_provider_configuration(db, provider_key, toggle_data.enabled)
        if not success:
            logging.warning(f"Provider not found in database: {provider_key}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider not found: {provider_key}"
            )
        # The registry only loads enabled providers, and dispatch goes through it, so
        # the change takes effect now rather than at the next sync or provider edit.
        await provider_manager.refresh_providers_from_database()
        
        return {
            "message": f"Provider {provider_key} {'enabled' if toggle_data.enabled else 'disabled'}",
            "provider_key": provider_key,
            "enabled": toggle_data.enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to toggle provider {provider_key}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle provider: {str(e)}"
        )


@router.put("/models/toggle")
async def toggle_model(
    toggle_data: ToggleRequest,
    model_id: str = Query(..., description="Model ID to toggle"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Toggle individual model using query parameter."""
    try:
        # Log for debugging
        logging.info(f"Admin model toggle (query) request - Model ID: {model_id}, Enabled: {toggle_data.enabled}")
        
        # Check if model exists in database before trying to toggle
        from app.auth.database import get_model_configuration
        existing_model = await get_model_configuration(db, model_id)
        logging.info(f"Model lookup result for '{model_id}': {existing_model is not None}")
        
        if not existing_model:
            logging.warning(f"Model not found: {model_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}"
            )
        
        success = await toggle_model_configuration(db, model_id, toggle_data.enabled)
        if not success:
            logging.warning(f"Model toggle failed for: {model_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to toggle model: {model_id}"
            )
        
        return {
            "message": f"Model {model_id} {'enabled' if toggle_data.enabled else 'disabled'}",
            "model_id": model_id,
            "enabled": toggle_data.enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to toggle model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle model: {str(e)}"
        )


@router.put("/models/bulk-toggle")
async def bulk_toggle_models(
    bulk_data: BulkToggleRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Enable or disable all models."""
    try:
        if bulk_data.action not in ["enable_all", "disable_all"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Action must be 'enable_all' or 'disable_all'"
            )
        
        enabled = bulk_data.action == "enable_all"
        success = await bulk_toggle_all_models(db, enabled)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to bulk toggle models"
            )
        
        # Refresh cache
        await provider_manager.refresh_model_configurations()
        
        return {
            "message": f"All models {'enabled' if enabled else 'disabled'}",
            "action": bulk_data.action,
            "enabled": enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to bulk toggle models: {str(e)}"
        )


@router.get("/models/find", response_model=ModelSearchResponse)
async def search_models(
    q: str,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Search models and providers."""
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 2 characters long"
            )
        
        results = await search_models_and_providers(db, q.strip())
        
        model_responses = [ModelConfigurationResponse.from_orm(model) for model in results["models"]]
        provider_responses = []
        
        for provider in results["providers"]:
            provider_models = await get_models_by_provider(db, provider.provider_key)
            enabled_models = [m for m in provider_models if m.is_enabled]
            
            # Parse supported_apis for search results
            _search_apis = None
            if provider.supported_apis:
                try:
                    _search_apis = json.loads(provider.supported_apis)
                except (ValueError, TypeError):
                    _search_apis = ["openai"]

            provider_response = ProviderConfigurationResponse(
                id=provider.id,
                provider_key=provider.provider_key,
                provider_type=provider.provider_type,
                instance_name=provider.instance_name,
                provider_name=provider.provider_name,
                enabled=provider.enabled,  # Changed from is_enabled to enabled
                model_count=len(provider_models),
                enabled_model_count=len(enabled_models),
                supported_apis=_search_apis,
                created_at=provider.created_at,
                updated_at=provider.updated_at
            )
            provider_responses.append(provider_response)
        
        return ModelSearchResponse(
            models=model_responses,
            providers=provider_responses,
            total_results=len(model_responses) + len(provider_responses)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search models: {str(e)}"
        )


@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin dashboard page."""
    # Get all users from database
    result = await db.execute(select(User))
    users = result.scalars().all()
    user_settings = await load_user_management_settings(db)

    import time
    # Create response with security headers to prevent caching
    response = templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "admin": current_admin,
            "users": users,
            "user_settings": user_settings,
            "admin_email": get_admin_email(),
            "title": "Admin Dashboard - LLM Proxy Server",
            "cache_version": str(int(time.time())),
            # 4get has no discovery endpoint; this is the list the form validates against.
            "fourget_scrapers": FOURGET_WEB_SCRAPERS,
            "websearch_provider_labels": WEBSEARCH_PROVIDER_LABELS,
        }
    )
    
    # Add security headers to prevent caching of sensitive admin pages
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


@router.get("/models/search", response_class=HTMLResponse)
async def admin_search_page(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin)
):
    """Admin search page for providers and models."""
    import time
    # Create response with security headers to prevent caching
    response = templates.TemplateResponse(
        "admin/models_search.html",
        {
            "request": request,
            "admin": current_admin,
            "title": "Search - Admin Dashboard - LLM Proxy Server",
            "cache_version": str(int(time.time()))
        }
    )
    
    # Add security headers to prevent caching of sensitive admin pages
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


@router.get("/users/settings", response_model=UserManagementSettingsResponse)
async def get_user_management_settings_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get user account policies (e.g. whether users may delete their own account)."""
    return await load_user_management_settings(db)


@router.put("/users/settings", response_model=UserManagementSettingsResponse)
async def update_user_management_settings_endpoint(
    body: UserManagementSettingsUpdate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update user account policies."""
    row = await upsert_user_management_settings(db, body.allow_self_delete, current_admin.username)
    return UserManagementSettingsResponse.model_validate(row)


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all users (API endpoint)."""
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [UserResponse.from_orm(user) for user in users]


@router.post("/users", response_model=UserResponse)
async def create_user_admin(
    user_data: AdminUserCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new, already-active user in a profile (Default when none is given)."""
    from app.auth.database import is_reserved_username

    # Same normalisation as signup, so both paths store and compare the same string.
    try:
        user_data.username = normalize_new_username(user_data.username)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    user_data.email = user_data.email.strip()

    # Check if username already exists. The config-based admin is not in the users
    # table but owns its name for usage/quota purposes (is_reserved_username).
    if await get_user_by_username(db, user_data.username) or is_reserved_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    existing_email = await get_user_by_email(db, user_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate password length
    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )

    profile_id = user_data.profile_id or DEFAULT_PROFILE_ID
    if await get_profile(db, profile_id) is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile not found")

    try:
        user = await create_user(
            db, user_data.username, user_data.email, user_data.password, profile_id=profile_id,
        )
    except Exception:
        logger.exception("Admin user creation failed for %s", user_data.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    from app.profiles import profile_registry
    from app.rate_limit import rate_limit_tracker
    profile_registry.set_users_profile([user.id], profile_id)
    # A periodic reload whose SELECT predates the INSERT could install its user map
    # over the patch above; reloading under the tracker's lock settles it.
    await rate_limit_tracker.reload_config()

    # Send signup webhook notification
    await send_signup_webhook(
        username=user.username,
        email=user.email,
        signup_mode="admin_created",
        user_id=user.id,
        is_pending=False
    )
    return UserResponse.from_orm(user)


async def _move_users_to_profile(db: AsyncSession, user_ids: List[int], profile_id: int) -> List[int]:
    """Assign users to a profile, and make the move visible to enforcement at once.

    A pool's daily limit is the sum of its members' limits, so moving a member changes
    their pool's limit too: the pools are resettled, which also reloads the snapshot.
    """
    from app.profiles import profile_registry
    from app.routes.pools import resettle_pools_after_commit

    moved = await assign_users_profile(db, user_ids, profile_id)
    profile_registry.set_users_profile(moved, profile_id)
    if moved:
        await resettle_pools_after_commit(db, user_ids=moved)
    return moved


@router.put("/users/profile", response_model=UserResponse)
async def set_user_profile_endpoint(
    body: UserProfileAssign,
    user_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Move one user to another profile."""
    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if await get_profile(db, body.profile_id) is None:
        raise HTTPException(status_code=400, detail="Profile not found")
    await _move_users_to_profile(db, [user_id], body.profile_id)
    await db.refresh(user)
    return UserResponse.from_orm(user)


@router.delete("/users")
async def delete_user(
    user_id: int = Query(..., description="User ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete a user (admin only) using query parameter."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    try:
        # Soft delete by setting is_active to False
        user.is_active = False
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        logger.error("Failed to deactivate user %s: %s", user_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )

    # Invalidate cached auth entries so the deactivated user's API keys stop
    # working immediately (the cache would otherwise keep serving them).
    from app.auth.cache import auth_cache
    auth_cache.invalidate_user_api_keys(user.id)
    auth_cache.invalidate_user_by_id(user.id)

    # is_active changes the pool's limit: a deactivated member cannot send, so
    # their limit no longer counts toward the sum. Re-apportion whatever pool they
    # are in -- a no-op when they are in none.
    #
    # Outside the try, as in the bulk path: the change is already committed, so a
    # settlement failure is not "the change did not happen" and must not be
    # reported as one. The rollback in the handler above would be a no-op against
    # it anyway, leaving the caller told the operation failed when it did not.
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, user_ids=[user_id])

    return {"message": f"User {user.username} has been deactivated"}


@router.put("/users/activate")
async def activate_user(
    user_id: int = Query(..., description="User ID to activate"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Activate a user (admin only) using query parameter."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    try:
        user.is_active = True
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate user: {str(e)}"
        )

    # is_active changes the pool's limit: a deactivated member cannot send, so
    # their limit no longer counts toward the sum. Re-apportion whatever pool they
    # are in -- a no-op when they are in none.
    #
    # Outside the try, as in the bulk path: the change is already committed, so a
    # settlement failure is not "the change did not happen" and must not be
    # reported as one. The rollback in the handler above would be a no-op against
    # it anyway, leaving the caller told the operation failed when it did not.
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, user_ids=[user_id])

    return {"message": f"User {user.username} has been activated"}


@router.put("/users/approve")
async def approve_user(
    user_id: int = Query(..., description="User ID to approve"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Approve a pending user registration (admin only)."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if not getattr(user, 'is_pending_approval', False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not pending approval"
        )

    try:
        user.is_active = True
        user.is_pending_approval = False
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve user: {str(e)}"
        )

    # is_active changes the pool's limit: a deactivated member cannot send, so
    # their limit no longer counts toward the sum. Re-apportion whatever pool they
    # are in -- a no-op when they are in none.
    #
    # Outside the try, as in the bulk path: the change is already committed, so a
    # settlement failure is not "the change did not happen" and must not be
    # reported as one. The rollback in the handler above would be a no-op against
    # it anyway, leaving the caller told the operation failed when it did not.
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, user_ids=[user_id])

    return {"message": f"User {user.username} has been approved and activated"}


@router.put("/users/reset-password")
async def reset_user_password(
    password_data: AdminPasswordReset,
    user_id: int = Query(..., description="User ID to reset password for"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reset a user's password (admin only) using query parameter."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check if user is an OAuth user
    if user.oauth_provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot reset password for OAuth user (authenticated via {user.oauth_provider})"
        )

    # Validate password length
    if len(password_data.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )

    try:
        success = await admin_reset_user_password(db, user_id, password_data.new_password)
        if success:
            return {"message": f"Password reset successfully for user {user.username}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset password"
            )
    except Exception as e:
        logging.error(f"Error resetting password for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset password: {str(e)}"
        )


@router.put("/users/modify")
async def modify_user(
    modify_data: AdminUserModify,
    user_id: int = Query(..., description="User ID to modify"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Modify a user's username, email, and/or password (admin only).

    Username can be changed for any user. Email and password can only be
    changed for non-OAuth users.
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    is_oauth = bool(user.oauth_provider)

    # OAuth users may only have their username changed.
    if is_oauth and (modify_data.email is not None or modify_data.new_password is not None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot modify email or password for OAuth user (authenticated via {user.oauth_provider})"
        )

    # Validate new password length if provided.
    if modify_data.new_password is not None:
        if len(modify_data.new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )

    try:
        # Update username/email via the existing helper (handles uniqueness).
        if modify_data.username is not None or modify_data.email is not None:
            await update_user_profile(
                db, user_id,
                username=modify_data.username,
                email=modify_data.email
            )

        # Update password separately if provided.
        if modify_data.new_password is not None:
            success = await admin_reset_user_password(db, user_id, modify_data.new_password)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update password"
                )

        # Invalidate cached auth entries so renamed/re-emailed users aren't served stale.
        from app.auth.cache import auth_cache
        auth_cache.invalidate_user_api_keys(user.id)
        auth_cache.invalidate_user_by_id(user.id)

        await db.refresh(user)
        return {"message": f"User {user.username} updated successfully"}
    except ValueError as e:
        # Uniqueness conflicts from update_user_profile.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error modifying user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify user: {str(e)}"
        )


@router.delete("/users/permanent")
async def permanently_delete_user_endpoint(
    user_id: int = Query(..., description="User ID to permanently delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Permanently delete a user and all associated data (admin only) using query parameter."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    try:
        # Settle the pool, delete the account and its usage rows, drop buffered counts,
        # invalidate the tracker -- one implementation shared with the bulk and
        # self-service deletes. See delete_user_account.
        from app.routes.pools import delete_user_account
        success = await delete_user_account(db, user_id)
        if success:
            return {"message": f"User {user.username} has been permanently deleted"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to permanently delete user"
            )
    except Exception as e:
        logging.error(f"Error permanently deleting user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to permanently delete user: {str(e)}"
        )


# Bounds the per-user delete loop below, which cannot run as a single transaction.
MAX_BULK_USER_IDS = 200


@router.post("/users/bulk-action")
async def bulk_user_action(
    bulk_data: BulkUserActionRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Apply approve/deactivate/activate/delete/assign_profile to several users at once (admin only).

    Mirrors the single-user endpoints above so bulk and single behaviour never
    diverge. A well-formed request always returns 200: per-user problems are
    reported in ``failed`` rather than failing the whole batch.
    """
    action = bulk_data.action
    if action not in ("approve", "deactivate", "activate", "delete", "assign_profile"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be 'approve', 'deactivate', 'activate', 'delete' or 'assign_profile'"
        )
    if action == "assign_profile":
        if bulk_data.profile_id is None or await get_profile(db, bulk_data.profile_id) is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Profile not found")

    # Dedupe while preserving the order the admin selected them in.
    user_ids = list(dict.fromkeys(bulk_data.user_ids))
    if not user_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No users selected"
        )
    if len(user_ids) > MAX_BULK_USER_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process more than {MAX_BULK_USER_IDS} users at once"
        )

    from app.auth.cache import auth_cache

    succeeded: List[dict] = []
    failed: List[dict] = []

    if action == "assign_profile":
        found = set(await _move_users_to_profile(db, user_ids, bulk_data.profile_id))
        for user_id in user_ids:
            if user_id in found:
                succeeded.append({"id": user_id, "username": None})
            else:
                failed.append({"id": user_id, "username": None, "error": "User not found"})
    elif action == "delete":
        # permanently_delete_user commits internally, so this cannot be one
        # transaction; partial completion is possible and is reported per user.
        from app.routes.pools import delete_user_account

        for user_id in user_ids:
            user = await get_user_by_id(db, user_id)
            if not user:
                failed.append({"id": user_id, "username": None, "error": "User not found"})
                continue

            username = user.username
            try:
                # Same sequence as the single-user endpoint above; see delete_user_account.
                if await delete_user_account(db, user_id):
                    succeeded.append({"id": user_id, "username": username})
                else:
                    failed.append({"id": user_id, "username": username, "error": "User not found"})
            except Exception as e:
                logger.error("Failed to permanently delete user %s: %s", user_id, e, exc_info=True)
                failed.append({"id": user_id, "username": username, "error": "Failed to delete user"})
    else:
        # Resolve everything first, then mutate and commit once.
        targets: List[User] = []
        for user_id in user_ids:
            user = await get_user_by_id(db, user_id)
            if not user:
                failed.append({"id": user_id, "username": None, "error": "User not found"})
                continue

            awaiting_approval = getattr(user, "is_pending_approval", False)

            if action == "activate" and awaiting_approval:
                # is_pending_approval wins over is_active in the UI, so activating
                # here would report success while the row still reads "Pending".
                failed.append({
                    "id": user_id,
                    "username": user.username,
                    "error": "User is pending approval — use Approve"
                })
                continue

            if action == "approve" and not awaiting_approval:
                # Same rejection the single-user endpoint gives, so a mixed
                # selection reports exactly which users were not approvable.
                failed.append({
                    "id": user_id,
                    "username": user.username,
                    "error": "User is not pending approval"
                })
                continue

            targets.append(user)

        if targets:
            try:
                for user in targets:
                    user.is_active = action in ("activate", "approve")
                    if action == "approve":
                        user.is_pending_approval = False
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error("Failed to %s users in bulk: %s", action, e, exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to {action} users"
                )

            for user in targets:
                succeeded.append({"id": user.id, "username": user.username})
                if action == "deactivate":
                    # Drop cached auth so the user's API keys stop working now.
                    auth_cache.invalidate_user_api_keys(user.id)
                    auth_cache.invalidate_user_by_id(user.id)

            # One re-apportionment pass for the whole batch, not one per user: every
            # toggle here moved a member's contribution to their pool's limit.
            from app.routes.pools import resettle_pools_after_commit
            await resettle_pools_after_commit(db, user_ids=[u.id for u in targets])

    past_tense = {
        "approve": "approved",
        "deactivate": "deactivated",
        "activate": "activated",
        "delete": "deleted",
        "assign_profile": "moved",
    }[action]
    message = f"{len(succeeded)} user{'' if len(succeeded) == 1 else 's'} {past_tense}"
    if failed:
        message += f", {len(failed)} failed"

    return {
        "message": message,
        "action": action,
        "succeeded": succeeded,
        "failed": failed,
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
    }


# ==================== User Profile API Endpoints ====================
#
# A profile carries everything that used to be global or per-user: overall and group
# rate limits, model access, and its own model mappings. Every user is in exactly one.

async def _require_profile(db: AsyncSession, profile_id: int):
    profile = await get_profile(db, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return profile


async def _profile_response(db: AsyncSession, profile_id: int) -> ProfileResponse:
    rows = await list_profiles(db, profile_id=profile_id)
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return ProfileResponse(**rows[0])


async def _resettle_profile(db: AsyncSession, profile_id: int) -> None:
    """A profile's limits changed: re-apportion every pool one of its members is in.

    resettle_pools_after_commit refreshes the tracker (and so the profile snapshot)
    before settling, which is what makes the new limits apply on the next request.
    """
    from app.routes.pools import resettle_pools_after_commit
    member_ids = await list_profile_member_ids(db, [profile_id])
    if member_ids:
        await resettle_pools_after_commit(db, user_ids=member_ids)
    else:
        from app.rate_limit import rate_limit_tracker
        await rate_limit_tracker.refresh_now()


@router.get("/profiles", response_model=List[ProfileResponse])
async def list_profiles_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every profile with its member count, limits, access mode and mapping count."""
    return [ProfileResponse(**row) for row in await list_profiles(db)]


@router.post("/profiles", response_model=ProfileResponse)
async def create_profile_endpoint(
    body: ProfileCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a profile, optionally as a copy of another (limits, access, mappings)."""
    from sqlalchemy.exc import IntegrityError
    if await get_profile_by_name(db, body.name):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Profile '{body.name}' already exists")
    if body.clone_from_id is not None:
        await _require_profile(db, body.clone_from_id)
    try:
        row = await create_profile(db, body.name, body.description, current_admin.username,
                                   clone_from_id=body.clone_from_id)
    except IntegrityError:
        # Lost a race with a concurrent create of the same (case-folded) name.
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Profile '{body.name}' already exists")
    from app.rate_limit import rate_limit_tracker
    await rate_limit_tracker.reload_config()  # no members yet: no counts change
    if body.clone_from_id is not None:
        from app.model_alias import model_alias_resolver
        await model_alias_resolver.load_from_database()
    return await _profile_response(db, row.id)


@router.put("/profiles", response_model=ProfileResponse)
async def update_profile_endpoint(
    body: ProfileUpdate,
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Rename a profile or change its description."""
    from sqlalchemy.exc import IntegrityError
    await _require_profile(db, profile_id)
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields:
        clash = await get_profile_by_name(db, fields["name"])
        if clash is not None and clash.id != profile_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Profile '{fields['name']}' already exists")
    try:
        await update_profile(db, profile_id, fields, current_admin.username)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Profile name already exists")
    from app.rate_limit import rate_limit_tracker
    await rate_limit_tracker.reload_config()
    return await _profile_response(db, profile_id)


@router.delete("/profiles")
async def delete_profile_endpoint(
    profile_id: int = Query(...),
    reassign_to: int = Query(DEFAULT_PROFILE_ID, description="Profile that receives the members"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a profile; its members move to `reassign_to` (Default unless given)."""
    if profile_id == DEFAULT_PROFILE_ID:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The default profile cannot be deleted")
    if reassign_to == profile_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Members cannot be reassigned to the profile being deleted")
    profile = await _require_profile(db, profile_id)
    if await get_profile(db, reassign_to) is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reassignment profile not found")

    moved = await delete_profile(db, profile_id, reassign_to)

    from app.profiles import profile_registry
    from app.model_alias import model_alias_resolver
    from app.routes.pools import resettle_pools_after_commit
    profile_registry.set_users_profile(moved, reassign_to)
    if moved:
        await resettle_pools_after_commit(db, user_ids=moved)
    else:
        from app.rate_limit import rate_limit_tracker
        await rate_limit_tracker.refresh_now()
    await model_alias_resolver.load_from_database()
    return {"message": f"Profile '{profile.name}' deleted", "moved_count": len(moved)}


@router.get("/profiles/limits", response_model=ProfileLimitsResponse)
async def get_profile_limits_endpoint(
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """A profile's limit for every model group, the built-in Default group first."""
    await _require_profile(db, profile_id)
    return ProfileLimitsResponse(
        profile=await _profile_response(db, profile_id),
        groups=await get_profile_limits(db, profile_id),
    )


@router.put("/profiles/limits", response_model=ProfileResponse)
async def update_profile_limits_endpoint(
    body: ProfileOverallLimitsUpdate,
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a profile's Default-group RPM/RPD and its max concurrent requests
    (null = unlimited; omitted = unchanged).

    For RPM/RPD, the same as PUT /profiles/model-group-limits with group_id=0. The
    concurrency cap is profile-wide and has no per-group form.
    """
    await _require_profile(db, profile_id)
    fields_set = body.model_fields_set
    await upsert_profile_limits(db, profile_id, body.rpm_limit, body.rpd_limit,
                                current_admin.username, fields_set,
                                max_concurrent=body.max_concurrent_requests)
    if fields_set & {"rpm_limit", "rpd_limit"}:
        await _resettle_profile(db, profile_id)
    else:
        # Only the concurrency cap changed: no count depends on it, so there is
        # nothing to re-apportion -- the snapshot just has to pick it up.
        from app.rate_limit import rate_limit_tracker
        await rate_limit_tracker.reload_config()
    return await _profile_response(db, profile_id)


async def _require_group(db, group_id: int) -> None:
    """404 unless group_id is a model group or the built-in Default group."""
    from app.profiles import DEFAULT_GROUP_ID
    if group_id != DEFAULT_GROUP_ID and await get_model_group_limits(db, group_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")


@router.put("/profiles/model-group-limits", response_model=ProfileLimitsResponse)
async def update_profile_model_group_limit_endpoint(
    body: ProfileLimitsUpdate,
    profile_id: int = Query(...),
    group_id: int = Query(..., description="Model group id; 0 for the Default group"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set one profile's RPM/RPD for a group (null = unlimited)."""
    await _require_profile(db, profile_id)
    await _require_group(db, group_id)
    await upsert_profile_group_limit(db, profile_id, group_id, body.rpm_limit, body.rpd_limit,
                                     current_admin.username, body.model_fields_set)
    await _resettle_profile(db, profile_id)
    return ProfileLimitsResponse(
        profile=await _profile_response(db, profile_id),
        groups=await get_profile_limits(db, profile_id),
    )


@router.put("/profiles/model-group-limits/all")
async def update_model_group_limit_all_profiles_endpoint(
    body: ProfileLimitsUpdate,
    group_id: int = Query(..., description="Model group id; 0 for the Default group"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Give every profile the same limits for one group."""
    await _require_group(db, group_id)
    await set_group_limit_for_all_profiles(db, group_id, body.rpm_limit, body.rpd_limit,
                                           current_admin.username, fields_set=body.model_fields_set)
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)
    # Echo only what changed: an omitted field kept each profile's own value, and
    # reporting it as null would read as "set to unlimited".
    return {"group_id": group_id, **body.model_dump(include=body.model_fields_set)}


@router.get("/profiles/members", response_model=List[ProfileMemberResponse])
async def list_profile_members_endpoint(
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """The profile's members with their live per-minute and today's request counts
    in the Default group."""
    from app.rate_limit import rate_limit_tracker
    await _require_profile(db, profile_id)
    users = await list_profile_members(db, profile_id)
    out = []
    # Sequential: one AsyncSession must not be used concurrently.
    for u in users:
        st = await rate_limit_tracker.get_user_status(u.id, u.username)
        out.append(ProfileMemberResponse(
            user_id=u.id, username=u.username, email=u.email, is_active=bool(u.is_active),
            current_rpm_count=st.rpm_count, current_rpd_count=st.rpd_count,
        ))
    return out


# ==================== Web Search Interception (SearXNG / 4get) ====================

async def _reload_websearch() -> None:
    """Apply a change to requests now (other workers catch up on their refresh)."""
    from app.websearch.settings import websearch_settings_cache
    await websearch_settings_cache.load_from_database()


def _websearch_response(row, backends) -> WebSearchSettingsResponse:
    from app.websearch.settings import settings_config

    cfg = settings_config(row)
    default_id = getattr(row, "default_backend_id", None)
    default = next((b for b in backends if b.id == default_id), None)
    return WebSearchSettingsResponse(
        enabled=cfg.enabled,
        default_backend_id=default.id if default else None,
        default_backend={"id": default.id, "name": default.name, "type": default.type} if default else None,
        max_results=cfg.max_results,
        max_snippet_chars=cfg.max_snippet_chars,
        timeout_seconds=cfg.timeout_seconds,
        max_agentic_loops=cfg.max_agentic_loops,
        max_queries_per_turn=cfg.max_queries_per_turn,
        apply_to=sorted(cfg.apply_to),
        enabled_providers=sorted(cfg.enabled_providers),
        updated_at=getattr(row, "updated_at", None),
        updated_by=getattr(row, "updated_by", None),
    )


def _websearch_values(body: WebSearchSettingsUpdate) -> dict:
    return body.model_dump()


def _backend_response(row, default_id: Optional[int], user_counts: dict) -> WebSearchBackendResponse:
    return WebSearchBackendResponse(
        id=row.id, name=row.name, type=row.type, base_url=row.base_url,
        enabled=bool(row.enabled), user_selectable=bool(row.user_selectable),
        engines=row.engines, categories=row.categories, language=row.language,
        fourget_scraper=row.fourget_scraper, fourget_lang=row.fourget_lang,
        fourget_country=row.fourget_country, safesearch=row.safesearch, time_range=row.time_range,
        is_default=row.id == default_id,
        user_count=user_counts.get(row.id, 0),
        updated_at=row.updated_at, updated_by=row.updated_by,
    )


async def _backend_responses(db: AsyncSession) -> List[WebSearchBackendResponse]:
    settings = await get_websearch_settings(db)
    default_id = getattr(settings, "default_backend_id", None)
    counts = await count_backend_users(db)
    return [_backend_response(b, default_id, counts) for b in await list_websearch_backends(db)]


async def _one_backend_response(db: AsyncSession, backend_id: int) -> WebSearchBackendResponse:
    return next(b for b in await _backend_responses(db) if b.id == backend_id)


async def _check_backend_name(db: AsyncSession, name: str, backend_id: Optional[int] = None) -> None:
    """409 when another backend has the name, ignoring case."""
    from sqlalchemy import func
    from app.auth.models import WebSearchBackend
    # lower() equality, like the unique index: ilike would read "_" as a wildcard.
    q = select(WebSearchBackend.id).where(func.lower(WebSearchBackend.name) == name.lower())
    if backend_id is not None:
        q = q.where(WebSearchBackend.id != backend_id)
    if (await db.execute(q.limit(1))).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"A backend named '{name}' already exists")


async def _require_backend(db: AsyncSession, backend_id: int):
    row = await get_websearch_backend(db, backend_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Web search backend not found")
    return row


async def _is_default_backend(db: AsyncSession, backend_id: int) -> bool:
    settings = await get_websearch_settings(db)
    return getattr(settings, "default_backend_id", None) == backend_id


_DEFAULT_BACKEND_LOCKED = "Choose another default backend first"


@router.get("/websearch/settings", response_model=WebSearchSettingsResponse)
async def get_websearch_settings_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get web search interception settings and the defaults users inherit."""
    return _websearch_response(await get_websearch_settings(db), await list_websearch_backends(db))


@router.put("/websearch/settings", response_model=WebSearchSettingsResponse)
async def update_websearch_settings_endpoint(
    body: WebSearchSettingsUpdate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Replace web search interception settings and reload the request-time snapshot."""
    if body.default_backend_id is not None:
        backend = await get_websearch_backend(db, body.default_backend_id)
        if backend is None or not backend.enabled:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The default backend must be an existing, enabled backend",
            )
    row = await upsert_websearch_settings(db, _websearch_values(body), current_admin.username)
    await _reload_websearch()
    return _websearch_response(row, await list_websearch_backends(db))


@router.get("/websearch/backends", response_model=List[WebSearchBackendResponse])
async def list_websearch_backends_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every search backend, with whether it is the default and how many users picked it."""
    return await _backend_responses(db)


@router.post("/websearch/backends", response_model=WebSearchBackendResponse)
async def create_websearch_backend_endpoint(
    body: WebSearchBackendCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Add a backend. The first one becomes the default."""
    await _check_backend_name(db, body.name)
    try:
        row = await create_websearch_backend(db, body.column_values(), current_admin.username)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="The name was taken by a concurrent edit; reload and retry")
    await _reload_websearch()
    return await _one_backend_response(db, row.id)


@router.put("/websearch/backends", response_model=WebSearchBackendResponse)
async def update_websearch_backend_endpoint(
    body: WebSearchBackendUpdate,
    backend_id: int = Query(..., description="Backend ID to update"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Replace a backend's definition. Its type is fixed: users' parameters for it
    only make sense on that type."""
    row = await _require_backend(db, backend_id)
    if body.type != row.type:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="A backend's type cannot change; add a new backend instead")
    if not body.enabled and await _is_default_backend(db, backend_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_DEFAULT_BACKEND_LOCKED)
    await _check_backend_name(db, body.name, backend_id)
    try:
        await update_websearch_backend(db, backend_id, body.column_values(), current_admin.username)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="The name was taken by a concurrent edit; reload and retry")
    await _reload_websearch()
    return await _one_backend_response(db, backend_id)


@router.put("/websearch/backends/toggle", response_model=WebSearchBackendResponse)
async def toggle_websearch_backend_endpoint(
    toggle_data: ToggleRequest,
    backend_id: int = Query(..., description="Backend ID to enable or disable"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Switch a backend on or off. While off, users who picked it use the default."""
    await _require_backend(db, backend_id)
    if not toggle_data.enabled and await _is_default_backend(db, backend_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_DEFAULT_BACKEND_LOCKED)
    await update_websearch_backend(db, backend_id, {"enabled": toggle_data.enabled}, current_admin.username)
    await _reload_websearch()
    return await _one_backend_response(db, backend_id)


@router.delete("/websearch/backends")
async def delete_websearch_backend_endpoint(
    backend_id: int = Query(..., description="Backend ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a backend. Users who picked it go back to the default, and their
    parameters for it are removed."""
    if await _is_default_backend(db, backend_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_DEFAULT_BACKEND_LOCKED)
    if not await delete_websearch_backend(db, backend_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Web search backend not found")
    await _reload_websearch()
    return {"message": f"Web search backend {backend_id} deleted"}


@router.post("/websearch/test")
async def test_websearch_endpoint(
    body: WebSearchTestRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Run one live query against an unsaved backend, a stored one, or the default."""
    import time as _time
    from app.websearch.backends import search
    from app.websearch.settings import BackendSpec, apply_backend, load_snapshot

    snapshot = await load_snapshot(db)
    cfg = snapshot.default
    if body.settings is not None:
        values = body.settings.model_dump(exclude={"enabled", "default_backend_id"})
        values["apply_to"] = frozenset(values["apply_to"])
        values["enabled_providers"] = frozenset(values["enabled_providers"])
        cfg = cfg.with_overrides(**values)
    if body.backend is not None:
        params = body.backend.params_for(body.backend.type)
        cfg = apply_backend(cfg, BackendSpec(
            id=body.backend_id, name=body.backend.name, type=body.backend.type, base_url=body.backend.base_url,
            params={k: v for k, v in params.items() if v is not None},
        ))
    elif body.backend_id is not None:
        backend = snapshot.backends.get(body.backend_id)
        if backend is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Web search backend not found")
        cfg = apply_backend(cfg, backend)
    if not cfg.base_url:
        raise HTTPException(status_code=400, detail="No backend to test: add one, or set the default backend")

    started = _time.monotonic()
    outcome = await search(body.query, cfg)
    latency_ms = int((_time.monotonic() - started) * 1000)
    if not outcome.ok:
        return {"ok": False, "latency_ms": latency_ms, "error_code": outcome.error_code, "error": outcome.message}
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "result_count": len(outcome.results),
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "date": r.date}
            for r in outcome.results[:3]
        ],
    }


@router.post("/websearch/engines")
async def list_websearch_engines(
    body: WebSearchEnginesRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Engines and categories loaded on a SearXNG instance (from its /config)."""
    from app.websearch.searxng import SearxngConfigError, fetch_engines
    from app.websearch.settings import load_snapshot

    snapshot = await load_snapshot(db)
    base_url = body.base_url
    if base_url is None and body.backend_id is not None:
        backend = snapshot.backends.get(body.backend_id)
        if backend is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Web search backend not found")
        if backend.type != WEBSEARCH_PROVIDER_SEARXNG:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"{backend.name} is not a SearXNG backend")
        base_url = backend.base_url
    if base_url is None:
        # The default when it is SearXNG, else the first enabled SearXNG backend:
        # before multiple backends, the stored SearXNG URL answered here even while
        # 4get was the provider.
        base_url = snapshot.default.searxng_base_url or next(
            (b.base_url for b in sorted(snapshot.backends.values(), key=lambda b: b.id)
             if b.enabled and b.type == WEBSEARCH_PROVIDER_SEARXNG),
            None,
        )
    if not base_url:
        raise HTTPException(status_code=400, detail="SearXNG base URL is not configured")
    try:
        return await fetch_engines(base_url, snapshot.default.timeout_seconds)
    except SearxngConfigError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/websearch/providers")
async def list_websearch_providers(
    current_admin: AdminUser = Depends(get_current_admin),
):
    """Provider keys for the interception multi-select."""
    items = [
        {
            "provider_key": getattr(provider, "full_provider_name", key),
            "provider_type": getattr(provider, "provider_type", ""),
        }
        for key, provider in sorted(provider_manager.providers.items())
    ]
    return {"providers": items}


@router.get("/models/all", response_model=List[ModelConfigurationResponse])
async def get_all_models(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return all model configurations as a flat list."""
    models = await get_all_model_configurations(db)
    return [ModelConfigurationResponse.from_orm(m) for m in models]


# ==================== Outage Management API Endpoints ====================
#
# An outage takes a provider instance (every model on it, including ones synced
# later) or one model out of service for every caller; requests go to the fallback
# mappings, or fail with 503 quoting the description. Resolving deletes the row.

def _outage_responses(rows, provider_keys, model_ids) -> List[OutageResponse]:
    """Serialise outage rows with what each one takes out of service right now."""
    from datetime import timezone
    cached = [m.id for m in provider_manager.model_cache.get_models()]
    out = []
    for row in rows:
        if row.scope == "provider":
            prefix = f"{row.target}/"
            affected = sum(1 for model_id in cached if model_id.startswith(prefix))
            exists = row.target in provider_keys
        else:
            exists = row.target in model_ids or provider_manager.model_cache.has_model_id(row.target)
            affected = 1 if exists else 0
        out.append(OutageResponse(
            id=row.id, scope=row.scope, target=row.target, description=row.description,
            created_by=row.created_by,
            created_at=row.created_at.replace(tzinfo=timezone.utc).timestamp() if row.created_at else None,
            affected_models=affected, target_exists=exists,
        ))
    return out


async def _outage_targets(db: AsyncSession):
    """(provider keys, model ids) an outage may name."""
    from app.auth.models import ModelConfiguration
    provider_keys = {p.provider_key for p in await get_all_provider_credentials(db)}
    model_ids = set((await db.execute(select(ModelConfiguration.model_id))).scalars().all())
    return provider_keys, model_ids


async def _reload_outages() -> None:
    from app.outages import outage_registry
    await outage_registry.load_from_database()


@router.get("/outages", response_model=List[OutageResponse])
async def list_outages_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every active outage."""
    return _outage_responses(await get_all_outages(db), *await _outage_targets(db))


@router.post("/outages", response_model=List[OutageResponse])
async def create_outages_endpoint(
    body: OutageCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Put provider instances and/or models in outage. A target already in outage
    keeps its start time and takes the new description."""
    provider_keys, model_ids = await _outage_targets(db)
    unknown = [pk for pk in body.provider_keys if pk not in provider_keys]
    unknown += [
        m for m in body.model_ids
        if m not in model_ids and not provider_manager.model_cache.has_model_id(m)
    ]
    if unknown:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown provider instance(s) or model(s): {unknown}")
    rows = await create_outages(db, body.provider_keys, body.model_ids, body.description, current_admin.username)
    await _reload_outages()
    logger.info("Outage reported by %s for %s: %s", current_admin.username,
                [r.target for r in rows], body.description)
    return _outage_responses(rows, provider_keys, model_ids)


@router.put("/outages", response_model=OutageResponse)
async def update_outage_endpoint(
    body: OutageUpdate,
    outage_id: int = Query(..., description="Outage ID to update"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Change an outage's description."""
    row = await update_outage_description(db, outage_id, body.description)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Outage not found")
    await _reload_outages()
    return _outage_responses([row], *await _outage_targets(db))[0]


@router.delete("/outages")
async def resolve_outage_endpoint(
    outage_id: int = Query(..., description="Outage ID to resolve"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Resolve an outage: its models are served again."""
    row = await get_outage(db, outage_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Outage not found")
    target = row.target
    await delete_outages(db, [outage_id])
    await _reload_outages()
    logger.info("Outage of %s resolved by %s", target, current_admin.username)
    return {"message": f"Outage of {target} resolved"}


@router.post("/outages/bulk-resolve")
async def bulk_resolve_outages_endpoint(
    body: OutageBulkResolve,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Resolve several outages at once; ids already resolved are reported in ``missing``."""
    ids = list(dict.fromkeys(body.ids))
    deleted = await delete_outages(db, ids)
    if deleted:
        await _reload_outages()
    gone = set(deleted)
    return {
        "deleted": deleted,
        "missing": [outage_id for outage_id in ids if outage_id not in gone],
        "message": f"Resolved {len(deleted)} outage{'' if len(deleted) == 1 else 's'}",
    }


# ==================== Model Alias Management API Endpoints ====================
#
# Every mapping belongs to a profile, and a profile's members are mapped with its
# list alone. `profile_id` defaults to Default throughout.

#
# Each profile has two lists, selected by `kind`: the regular mappings ("alias"),
# applied when the requested model is available to the caller, and the fallback
# mappings ("fallback"), applied when it is not.

_KindQuery = Query("alias", pattern="^(alias|fallback)$", description="Which list: alias or fallback")


def _alias_responses(rows, profile_id: int) -> List[ModelAliasResponse]:
    """Serialise rows; exact regular rows also report whether their source is available.

    A regular mapping only applies to an available model, so one whose alias names
    no model the profile's members can use never fires (it belongs in the fallback
    list). Only exact rows name a single model; patterns are left unflagged.
    """
    from app.model_resolution import is_model_available, may_use_for_profile
    may_use = may_use_for_profile(profile_id)
    out = []
    for row in rows:
        resp = ModelAliasResponse.model_validate(row)
        if resp.kind == "alias" and resp.match_type == "exact":
            # Judged without outages: a mapping that applies once an outage is
            # resolved is not misplaced, and must not be flagged to be moved.
            resp.source_available = any(
                is_model_available(resp.alias, may_use, surface, ignore_outage=True) for surface in resp.apis
            )
        out.append(resp)
    return out


@router.get("/model-aliases", response_model=List[ModelAliasResponse])
async def list_model_aliases_endpoint(
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Profile whose mappings to list"),
    kind: str = _KindQuery,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List one list of a profile, including disabled aliases."""
    await _require_profile(db, profile_id)
    return _alias_responses(await get_all_model_aliases(db, profile_id, kind=kind), profile_id)


@router.post("/model-aliases", response_model=ModelAliasResponse)
async def upsert_model_alias_endpoint(
    body: ModelAliasUpsert,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create or update an alias after validating its pattern and canonical target.

    With `id`, that row is updated in place (and may be renamed, or moved to the
    other list by a different `kind`); otherwise the row is matched by its alias
    text within the profile's `kind` list.
    """
    from sqlalchemy.exc import IntegrityError
    await _require_profile(db, body.profile_id)
    if body.id is not None:
        existing = await get_model_alias_by_id(db, body.id)
        if existing is None or existing.profile_id != body.profile_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    else:
        existing = await get_model_alias(db, body.alias, body.profile_id, body.kind or "alias")
    # Like match_type, an omitted kind keeps the row where it is.
    kind = body.kind or (existing.kind if existing else None) or "alias"

    # The schema only validates the pattern when match_type is sent; re-check it
    # against the type the row will actually have.
    effective_type = body.match_type or (existing.match_type if existing else None) or "exact"
    try:
        validate_alias_pattern(body.alias, body.target_model_id, effective_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if await get_model_configuration(db, body.target_model_id) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown target model: {body.target_model_id}",
        )
    # Passing the loaded row's id lets upsert_model_alias reuse it from the
    # session. A rename onto another row's alias trips the unique constraint.
    try:
        row = await upsert_model_alias(
            db, body.alias, body.target_model_id, body.enabled, body.apis,
            match_type=body.match_type, alias_id=existing.id if existing else None,
            profile_id=body.profile_id, kind=kind,
        )
    except IntegrityError:
        await db.rollback()
        list_name = "fallback mapping" if kind == "fallback" else "mapping"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A {list_name} for '{body.alias}' already exists",
        )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return _alias_responses([row], body.profile_id)[0]


@router.put("/model-aliases/order", response_model=List[ModelAliasResponse])
async def reorder_model_aliases_endpoint(
    body: ModelAliasReorder,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set one list's evaluation order; `ids` must list every alias of it exactly once."""
    await _require_profile(db, body.profile_id)
    if not await reorder_model_aliases(db, body.ids, body.profile_id, body.kind):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mapping list changed — refresh and retry",
        )
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return _alias_responses(
        await get_all_model_aliases(db, body.profile_id, kind=body.kind), body.profile_id,
    )


@router.post("/model-aliases/bulk-delete")
async def bulk_delete_model_aliases_endpoint(
    body: ModelAliasBulkDelete,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete several aliases of one list at once and reload the resolver.

    Ids no longer in the list (already deleted, or another list's) are skipped and
    reported in ``missing`` rather than failing the batch.
    """
    await _require_profile(db, body.profile_id)
    ids = list(dict.fromkeys(body.ids))
    deleted = await delete_model_aliases(db, ids, body.profile_id, body.kind)
    if deleted:
        from app.model_alias import model_alias_resolver
        await model_alias_resolver.load_from_database()
    gone = set(deleted)
    return {
        "deleted": deleted,
        "missing": [alias_id for alias_id in ids if alias_id not in gone],
        "message": f"Deleted {len(deleted)} mapping{'' if len(deleted) == 1 else 's'}",
    }


@router.get("/model-aliases/test", response_model=List[ModelAliasTestMatch])
async def match_model_alias_endpoint(
    model: str = Query(..., min_length=1),
    api: Optional[str] = Query(None),
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Test as a member of this profile"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Report which enabled mapping *model* would hit, per API surface.

    Uses the live resolver and model availability, so the answer is what a member
    of `profile_id` gets: the regular list when the name is available to them, the
    fallback list when it is not. Without `api`, every surface is reported (rather
    than an unscoped match, which ignores API scoping and so never happens for a
    real request). Only `profile_id`'s own mappings are tried, as for its members.
    """
    if api is not None and api not in MODEL_ALIAS_API_SURFACES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown API surface: {api}")
    await _require_profile(db, profile_id)
    from app.model_alias import model_alias_resolver
    from app.model_resolution import is_model_available, may_use_for_profile
    may_use = may_use_for_profile(profile_id)
    results = []
    for surface in ([api] if api else MODEL_ALIAS_API_SURFACES):
        available = is_model_available(model, may_use, surface)
        kind = "alias" if available else "fallback"
        rule = model_alias_resolver.match(model, surface, profile_id, kind)
        if rule is None:
            results.append(ModelAliasTestMatch(
                api=surface, available=available, kind=kind, matched=False, resolved_model=model,
            ))
        else:
            results.append(ModelAliasTestMatch(
                api=surface, available=available, kind=kind, matched=True, resolved_model=rule.target,
                id=rule.id, alias=rule.pattern, match_type=rule.match_type, target_model_id=rule.target,
            ))
    return results


@router.delete("/model-aliases/{alias:path}")
async def delete_model_alias_endpoint(
    alias: str,
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Profile whose list holds the alias"),
    kind: str = _KindQuery,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete an alias and reload the in-memory resolver immediately."""
    if not await delete_model_alias(db, alias, profile_id, kind):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return {"message": f"Model alias '{alias}' deleted"}


# ==================== Model Group Management API Endpoints ====================
#
# One kind of group. Members are explicit models and/or whole provider instances;
# instance membership is live (models synced later join by themselves) and an explicit
# model beats an instance (rate_limit_tracker.group_for). Every model no group claims
# is in the built-in Default group, id 0, which has no row and cannot be edited.

def _group_responses(overview: dict) -> List[ModelGroupResponse]:
    """Default first, then every named group, with member status and live coverage."""
    from app.profiles import DEFAULT_GROUP_ID
    from app.rate_limit import provider_key_of

    models, providers = overview["models"], overview["providers"]
    by_instance = overview["by_instance"]
    names = {g.id: g.name for g in overview["groups"]}

    def model_status(model_id: str) -> str:
        if model_id not in models:
            return "missing"
        if providers.get(provider_key_of(model_id)) is False:
            return "provider_disabled"
        return "ok" if models[model_id] else "model_disabled"

    out = [ModelGroupResponse(
        id=DEFAULT_GROUP_ID, name="Default", is_default=True,
        description="Every model no other group claims",
        effective_models=overview["effective"][DEFAULT_GROUP_ID],
    )]
    for g in overview["groups"]:
        members = []
        for pk in sorted(m.provider_key for m in g.instance_members):
            members.append(ModelGroupMemberResponse(
                kind="instance", key=pk,
                status="missing" if pk not in providers else ("ok" if providers[pk] else "provider_disabled"),
                covers=[mid for mid in overview["effective"][g.id] if provider_key_of(mid) == pk],
            ))
        for mid in sorted(m.model_id for m in g.members):
            owner = by_instance.get(provider_key_of(mid))
            members.append(ModelGroupMemberResponse(
                kind="model", key=mid, status=model_status(mid),
                overrides_group=names.get(owner) if owner not in (None, g.id) else None,
            ))
        out.append(ModelGroupResponse(
            id=g.id, name=g.name, description=g.description, enabled=g.enabled,
            member_count=len(g.members) + len(g.instance_members),
            model_ids=sorted(m.model_id for m in g.members),
            provider_keys=sorted(m.provider_key for m in g.instance_members),
            members=members, effective_models=overview["effective"][g.id],
            updated_at=g.updated_at, updated_by=g.updated_by,
        ))
    return out


async def _check_group_name(db: AsyncSession, name: str, group_id: Optional[int] = None) -> None:
    """409 when the name is taken (ignoring case) or is the reserved "Default"."""
    from app.auth.models import ModelGroup as ModelGroupTable
    if name.strip().lower() == "default":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="'Default' is the built-in group's name")
    q = select(ModelGroupTable).where(ModelGroupTable.name.ilike(name))
    if group_id is not None:
        q = q.where(ModelGroupTable.id != group_id)
    if (await db.execute(q)).scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Model group '{name}' already exists")


async def _check_members(
    db: AsyncSession, body: ModelGroupMembersUpdate, group_id: Optional[int] = None,
) -> None:
    """Validate a member list before it replaces a group's (or a new group's) members.

    Only members being *added* must exist: one the group already has may have been
    deleted or renamed upstream since, and the admin must still be able to save the
    group -- with it kept, or unticked -- rather than be locked out of editing it.
    Each model and each instance may belong to one group only (409, naming the owner).
    """
    from app.auth.models import (
        ModelConfiguration, ModelGroup as ModelGroupTable, ModelGroupInstanceMember, ModelGroupMember,
    )
    current_models, current_instances = set(), set()
    if group_id is not None:
        current_models = set((await db.execute(
            select(ModelGroupMember.model_id).where(ModelGroupMember.group_id == group_id)
        )).scalars().all())
        current_instances = set((await db.execute(
            select(ModelGroupInstanceMember.provider_key).where(ModelGroupInstanceMember.group_id == group_id)
        )).scalars().all())

    new_models = [m for m in body.model_ids if m not in current_models]
    if new_models:
        found = set((await db.execute(
            select(ModelConfiguration.model_id).where(ModelConfiguration.model_id.in_(new_models))
        )).scalars().all())
        missing = [m for m in new_models if m not in found]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"model_id(s) not found: {missing}")
    new_instances = [pk for pk in body.provider_keys if pk not in current_instances]
    if new_instances:
        found = {p.provider_key for p in await get_all_provider_credentials(db)}
        missing = [pk for pk in new_instances if pk not in found]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"provider_key(s) not found: {missing}")

    conflicts = []
    for member, column, keys, kind in (
        (ModelGroupMember, ModelGroupMember.model_id, body.model_ids, "model"),
        (ModelGroupInstanceMember, ModelGroupInstanceMember.provider_key, body.provider_keys, "instance"),
    ):
        if not keys:
            continue
        q = (select(column, ModelGroupTable.id, ModelGroupTable.name)
             .join(ModelGroupTable, ModelGroupTable.id == member.group_id).where(column.in_(keys)))
        if group_id is not None:
            q = q.where(member.group_id != group_id)
        conflicts += [{"kind": kind, "key": key, "current_group_id": gid, "current_group": gname}
                      for key, gid, gname in (await db.execute(q)).all()]
    if conflicts:
        names = ", ".join(f"{c['key']} (in '{c['current_group']}')" for c in conflicts)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail={"message": f"Already in another group: {names}", "conflicts": conflicts})


def _reject_default(group_id: int) -> None:
    from app.profiles import DEFAULT_GROUP_ID
    if group_id == DEFAULT_GROUP_ID:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="The Default group is built in: it holds every model no other group claims")


@router.get("/model-groups")
async def get_model_groups(
    group_id: Optional[int] = Query(None, description="Omit to list all groups; provide to fetch one (0 = Default)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List every model group, Default first, with members and the models each governs."""
    groups = _group_responses(await model_group_overview(db))
    if group_id is None:
        return groups
    for g in groups:
        if g.id == group_id:
            return g
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")


@router.post("/model-groups")
async def create_model_group_endpoint(
    body: ModelGroupCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a model group with its members.

    Each profile's limits for it start as the strictest limits its members had until
    now (see database.create_model_group), so moving models in never loosens them.
    """
    await _check_group_name(db, body.name)
    await _check_members(db, body)
    try:
        group = await create_model_group(db, body.name, body.description, current_admin.username,
                                         body.model_ids, body.provider_keys)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="A member or the name was taken by a concurrent edit; reload and retry")

    # Members moved out of Default: the tracker must see them now, and pools settled
    # by the old split re-apportioned (resettle refreshes the snapshot first).
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)
    return next(g for g in _group_responses(await model_group_overview(db)) if g.id == group.id)


@router.put("/model-groups")
async def update_model_group_endpoint(
    group_id: int = Query(..., description="Group ID to update"),
    body: ModelGroupUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Rename or update the description of a model group, and optionally replace its
    members in the same commit (see ModelGroupUpdate)."""
    _reject_default(group_id)
    body = body or ModelGroupUpdate()
    fields = body.model_dump(exclude_unset=True, exclude={"model_ids", "provider_keys"})
    members = body.members()
    if members is not None and await list_model_groups(db, group_id=group_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")
    if "name" in fields:
        await _check_group_name(db, fields["name"], group_id)
    if members is not None:
        await _check_members(db, members, group_id)

    try:
        group = await update_model_group(
            db, group_id, fields, current_admin.username,
            members=None if members is None else (members.model_ids, members.provider_keys),
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="A member or the name was taken by a concurrent edit; reload and retry")
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    from app.rate_limit import rate_limit_tracker
    if members is None:
        # The name is in every 429 the group returns. No re-settle: neither field is
        # an apportionment weight.
        await rate_limit_tracker.reload_config()
    else:
        rate_limit_tracker.invalidate_group(group_id)
        # Refreshes the snapshot before settling, so the new membership applies at once.
        from app.routes.pools import resettle_pools_after_commit
        await resettle_pools_after_commit(db, all_pools=True)
    return next(g for g in _group_responses(await model_group_overview(db)) if g.id == group_id)


@router.put("/model-groups/toggle")
async def toggle_model_group_endpoint(
    toggle_data: ToggleRequest,
    group_id: int = Query(..., description="Group ID to enable or disable"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Switch a model group on or off. While off, its models are not served, like a
    disabled model; profiles that override the global config still get them."""
    _reject_default(group_id)
    group = await update_model_group(db, group_id, {"enabled": toggle_data.enabled}, current_admin.username)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    from app.rate_limit import rate_limit_tracker
    # Access only: membership and limits are unchanged, so no count or settlement moves.
    await rate_limit_tracker.reload_config()
    return next(g for g in _group_responses(await model_group_overview(db)) if g.id == group_id)


@router.delete("/model-groups")
async def delete_model_group_endpoint(
    group_id: int = Query(..., description="Group ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a model group; its models return to the Default group."""
    _reject_default(group_id)
    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_group(group_id)

    deleted = await delete_model_group(db, group_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)
    return {"message": f"Model group {group_id} deleted"}


@router.put("/model-groups/members")
async def set_model_group_members_endpoint(
    group_id: int = Query(..., description="Group ID to update members for"),
    body: ModelGroupMembersUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Replace a group's members: explicit models and whole provider instances."""
    _reject_default(group_id)
    if await list_model_groups(db, group_id=group_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")
    body = body or ModelGroupMembersUpdate()
    await _check_members(db, body, group_id)
    try:
        await set_group_members(db, group_id, body.model_ids, body.provider_keys)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="A member was claimed by a concurrent edit; reload and retry")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_group(group_id)
    # Refreshes the snapshot before settling, so the new membership applies at once.
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)

    return {"group_id": group_id, "model_ids": body.model_ids, "provider_keys": body.provider_keys}


# ==================== Profile Model Access API Endpoints ====================

class UserModelAccessItem(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    model_name: str
    provider_key: str
    globally_enabled: bool          # ModelConfiguration.is_enabled
    blocked_by_group: Optional[str] = None  # name of the disabled group that blocks it
    group_id: int = 0               # the model group governing it (0 = Default)
    is_exception: bool             # explicit profile override exists
    is_allowed: Optional[bool]      # exception value; None if no exception
    effective_allowed: bool         # final decision (global AND policy/exception)


class ProfileGroupAccessItem(BaseModel):
    group_id: int                   # 0 = the built-in Default group
    name: str
    is_default: bool = False
    enabled: bool = True            # the group's global switch (Model Groups tab)


class ProfileModelAccessResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    profile_id: int
    name: str
    member_count: int = 0
    mode: str                       # default|allow|deny|custom
    models: List[UserModelAccessItem]
    groups: List[ProfileGroupAccessItem] = []   # Default first; models carry their group_id


class UserModelPolicyUpdate(BaseModel):
    mode: Literal["default", "allow", "deny", "custom"]


class UserModelExceptionUpdate(BaseModel):
    is_allowed: bool


def _global_ok(model: ModelConfigurationResponse) -> bool:
    """Whether the model passes the global gate.

    Must mirror the cache's decision exactly: a model is globally available only when
    its own enabled flag AND its provider's enabled flag are both on. Checking just
    ``model.is_enabled`` here would disagree with the cache whenever a whole provider is
    globally disabled, causing every follow-global override to be mis-flagged as a
    profile exception (spurious pencils)."""
    return provider_manager.model_cache._passes_global_gate(model.model_id)


def _compute_access_item(
    model: ModelConfigurationResponse,
    mode: str,
    exception_map: dict,
) -> UserModelAccessItem:
    """Build one model's access state under a profile, mirroring the cache decision."""
    from app.rate_limit import rate_limit_tracker
    has_override = model.model_id in exception_map
    exc_val = exception_map.get(model.model_id)

    if mode == "allow":
        effective = True
        is_exception = False
    elif mode == "deny":
        effective = False
        is_exception = False
    elif mode == "custom":
        # Any stored override is a profile exception (pencil); otherwise follow global.
        if has_override:
            effective = bool(exc_val)
            is_exception = True
        else:
            effective = _global_ok(model)
            is_exception = False
    else:  # "default"
        effective = _global_ok(model)
        is_exception = False

    return UserModelAccessItem(
        model_id=model.model_id,
        model_name=model.model_name,
        provider_key=model.provider_key,
        globally_enabled=model.is_enabled,
        blocked_by_group=rate_limit_tracker.blocking_group(model.model_id),
        is_exception=is_exception,
        is_allowed=exc_val if is_exception else None,
        effective_allowed=effective,
    )


async def _build_profile_model_access(
    db: AsyncSession, profile_id: int, models=None, overview: Optional[dict] = None,
) -> ProfileModelAccessResponse:
    """Assemble the full model-access view of one profile.

    A caller that already loaded the model rows or the group overview passes them in;
    an access edit changes neither.
    """
    from app.profiles import DEFAULT_GROUP_ID
    profile = await _profile_response(db, profile_id)
    if models is None:
        models = await get_all_model_configurations(db)
    exception_map = await list_profile_model_exceptions(db, profile_id)
    if overview is None:
        overview = await model_group_overview(db)
    group_of = {mid: gid for gid, mids in overview["effective"].items() for mid in mids}
    items = []
    for m in models:
        item = _compute_access_item(ModelConfigurationResponse.from_orm(m), profile.model_access_mode, exception_map)
        item.group_id = group_of.get(m.model_id, DEFAULT_GROUP_ID)
        items.append(item)
    groups = [ProfileGroupAccessItem(group_id=DEFAULT_GROUP_ID, name="Default", is_default=True)]
    groups += [ProfileGroupAccessItem(group_id=g.id, name=g.name, enabled=g.enabled) for g in overview["groups"]]
    return ProfileModelAccessResponse(
        profile_id=profile.id, name=profile.name, member_count=profile.member_count,
        mode=profile.model_access_mode, models=items, groups=groups,
    )


async def _reload_access() -> None:
    """Access rules are read from the profile snapshot; reload it after an edit."""
    from app.rate_limit import rate_limit_tracker
    await rate_limit_tracker.reload_config()


@router.get("/profiles/model-access", response_model=ProfileModelAccessResponse)
async def get_profile_model_access(
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return a profile's access mode plus per-model effective access."""
    await _require_profile(db, profile_id)
    return await _build_profile_model_access(db, profile_id)


@router.put("/profiles/model-access/policy", response_model=ProfileModelAccessResponse)
async def update_profile_model_policy(
    body: UserModelPolicyUpdate,
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set the profile's model-access mode (default | allow | deny | custom).

    Stored per-model overrides are kept but only consulted in 'custom' mode; switching
    to allow/deny/default leaves them inert, and returning to custom reactivates them.
    """
    await _require_profile(db, profile_id)
    await set_profile_model_mode(db, profile_id, body.mode, current_admin.username)
    await _reload_access()
    return await _build_profile_model_access(db, profile_id)


@router.put("/profiles/model-access", response_model=ProfileModelAccessResponse)
async def upsert_profile_model_access_exception(
    body: UserModelExceptionUpdate,
    profile_id: int = Query(...),
    model_id: str = Query(..., description="Model ID, e.g. 'azure:primary/gpt-4'"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set an explicit per-model access override for a profile.

    Toggling an individual model always puts the profile in 'custom' mode. When it
    was previously in allow/deny, the current effective state of every model is first
    seeded as explicit overrides so the switch is loss-less (e.g. 'deny all' then
    enabling one model leaves only that model on). From 'default'/'custom' no seeding
    is needed — untoggled models keep following the global config.
    """
    profile = await _require_profile(db, profile_id)
    if not await get_model_configuration(db, model_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    prev_mode = profile.model_access_mode or "default"
    if prev_mode in ("allow", "deny"):
        all_models = await get_all_model_configurations(db)
        overrides = {m.model_id: prev_mode == "allow" for m in all_models}
        overrides[model_id] = body.is_allowed
        await set_profile_model_exceptions_bulk(db, profile_id, overrides, current_admin.username)
    else:
        await upsert_profile_model_exception(db, profile_id, model_id, body.is_allowed, current_admin.username)
    if prev_mode != "custom":
        await set_profile_model_mode(db, profile_id, "custom", current_admin.username)
    await _reload_access()
    return await _build_profile_model_access(db, profile_id)


@router.put("/profiles/model-access/group", response_model=ProfileModelAccessResponse)
async def set_profile_group_access(
    body: UserModelExceptionUpdate,
    profile_id: int = Query(...),
    group_id: int = Query(..., description="Model group ID (0 = Default)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Allow or deny every model a group governs right now, for one profile.

    The same per-model overrides as toggling each model by hand, written in one go, so
    the profile switches to 'custom' the same way (seeded loss-lessly from allow/deny).
    A model that joins the group later follows the profile's rules like any new model.
    """
    profile = await _require_profile(db, profile_id)
    overview = await model_group_overview(db)
    if group_id not in overview["effective"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")
    members = overview["effective"][group_id]
    if not members:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The group has no models")

    models = await get_all_model_configurations(db)
    prev_mode = profile.model_access_mode or "default"
    if prev_mode in ("allow", "deny"):
        overrides = {m.model_id: prev_mode == "allow" for m in models}
    else:
        overrides = await list_profile_model_exceptions(db, profile_id)
    overrides.update({mid: body.is_allowed for mid in members})
    await set_profile_model_exceptions_bulk(db, profile_id, overrides, current_admin.username)
    if prev_mode != "custom":
        await set_profile_model_mode(db, profile_id, "custom", current_admin.username)
    await _reload_access()
    return await _build_profile_model_access(db, profile_id, models=models, overview=overview)


@router.delete("/profiles/model-access")
async def delete_profile_model_access_exception(
    profile_id: int = Query(...),
    model_id: str = Query(..., description="Model ID to reset to the profile's default"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove an explicit per-model override; the model reverts to the profile's default."""
    profile = await _require_profile(db, profile_id)
    if not await delete_profile_model_exception(db, profile_id, model_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No override found for this profile and model")
    await _reload_access()
    return {"message": f"Override removed for profile {profile.name} on model {model_id}"}


@router.post("/logout")
async def admin_logout():
    """Admin logout endpoint."""
    from fastapi.responses import RedirectResponse
    redirect_response = RedirectResponse(url="/admin/", status_code=status.HTTP_302_FOUND)
    
    # Delete the cookie with the same parameters used when setting it
    redirect_response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax"
    )
    from app.auth.csrf import clear_csrf_cookie
    clear_csrf_cookie(redirect_response)
    
    return redirect_response


# Provider Credentials Management API Endpoints

@router.get("/providers", response_model=List[ProviderConfigurationResponse])
async def list_provider_credentials(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all provider credentials with model counts."""
    try:
        logging.debug("Fetching all provider credentials...")
        credentials = await get_all_provider_credentials(db)
        logging.debug(f"Found {len(credentials)} provider credentials")

        models = await get_all_model_configurations(db)
        logging.debug(f"Found {len(models)} model configurations")

        # Index models by provider_key once (O(M)) instead of re-scanning the
        # full model list per provider (O(P×M)).
        models_by_provider: dict = {}
        for m in models:
            models_by_provider.setdefault(m.provider_key, []).append(m)

        response_list = []

        for cred in credentials:
            # Calculate model counts for this provider
            provider_models = models_by_provider.get(cred.provider_key, [])
            enabled_models = [m for m in provider_models if m.is_enabled]

            logging.debug(f"Provider {cred.provider_key}: {len(provider_models)} models, {len(enabled_models)} enabled")
            
            # Parse deployments JSON if present
            deployments = None
            if cred.deployments_json:
                try:
                    deployments = json.loads(cred.deployments_json)
                except json.JSONDecodeError:
                    deployments = None
            
            sync_status = provider_manager.get_sync_status(cred.provider_key) or {}

            # Use ProviderConfigurationResponse which includes model_count fields
            response = ProviderConfigurationResponse(
                id=cred.id,
                provider_key=cred.provider_key,
                provider_type=cred.provider_type,
                instance_name=cred.instance_name,
                provider_name=cred.provider_name,  # Include provider_name for OpenAI-compatible providers
                enabled=cred.enabled,
                model_count=len(provider_models),
                enabled_model_count=len(enabled_models),
                sync_state=sync_status.get("state"),
                sync_error=sync_status.get("error"),
                last_sync_at=sync_status.get("last_sync_at"),
                created_at=cred.created_at,
                updated_at=cred.updated_at
            )
            response_list.append(response)
        
        logging.info(f"📤 Returning {len(response_list)} providers to frontend")
        return response_list
    except Exception as e:
        logging.error(f"❌ Failed to list provider credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list provider credentials"
        )


@router.get("/providers/detail", response_model=ProviderCredentialsResponse)
async def get_provider_credential(
    provider_key: str = Query(..., description="Provider key to retrieve"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get specific provider credentials using query parameter."""
    try:
        # URL decode the provider_key
        decoded_provider_key = urllib.parse.unquote(provider_key)
        
        credentials = await get_provider_credentials(db, decoded_provider_key)
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider credentials not found: {decoded_provider_key}"
            )
        
        # Parse deployments JSON if present
        deployment_groups, deployments = _parse_azure_deployment_fields(credentials.deployments_json)

        # Get dynamic_discovery from database
        # For backward compatibility: if not stored, infer from deployments (legacy behavior)
        dynamic_discovery = credentials.dynamic_discovery
        if dynamic_discovery is None and credentials.provider_type == 'azure':
            # Legacy: infer from deployments if not explicitly set
            dynamic_discovery = not bool(deployments)
        
        # Parse supported_apis JSON
        supported_apis = None
        if credentials.supported_apis:
            try:
                supported_apis = json.loads(credentials.supported_apis)
            except (json.JSONDecodeError, TypeError):
                supported_apis = ["openai"]

        return ProviderCredentialsResponse(
            id=credentials.id,
            provider_key=credentials.provider_key,
            provider_type=credentials.provider_type,
            instance_name=credentials.instance_name,
            enabled=credentials.enabled,
            endpoint=credentials.endpoint,
            api_key=_mask_secret(credentials.api_key),
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=_mask_secret(credentials.secret_access_key),
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            deployment_groups=deployment_groups.get("groups", []),
            provider_name=credentials.provider_name,
            dynamic_discovery=dynamic_discovery,
            supported_apis=supported_apis,
            anthropic_base_url=credentials.anthropic_base_url,
            anthropic_auth_style=credentials.anthropic_auth_style,
            created_at=credentials.created_at,
            updated_at=credentials.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get provider credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get provider credentials"
        )


@router.post("/providers", response_model=ProviderCredentialsResponse)
async def create_provider_credential(
    provider_data: ProviderCredentialsCreate,
    upsert: bool = Query(False, description="Update if provider exists instead of failing"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create new provider credentials."""
    try:
        logging.info(
            "Creating provider %s:%s (type=%s, upsert=%s)",
            provider_data.provider_name, provider_data.instance_name,
            provider_data.provider_type, upsert,
        )
        
        # Ensure provider_name is set - if not provided, default to provider_type for specialized providers
        if not provider_data.provider_name:
            # This should not happen now that provider_name is required, but keep for safety
            provider_data.provider_name = provider_data.provider_type
        
        # Generate provider key using provider_name:instance_name format
        provider_key = f"{provider_data.provider_name}:{provider_data.instance_name}"
        logging.info(f"🔍 Checking if provider exists: {provider_key}")
        
        existing = await get_provider_credentials(db, provider_key)
        if existing:
            if upsert:
                logging.info(f"🔄 Provider exists, updating due to upsert=True: {provider_key}")
                # Auto-infer supported_apis for Azure upsert
                if provider_data.provider_type == "azure" and not provider_data.supported_apis:
                    inferred_apis = ["openai"]
                    if (
                        provider_data.azure_backend == "foundry"
                        and provider_data.anthropic_deployments
                        and len(provider_data.anthropic_deployments) > 0
                    ):
                        inferred_apis.append("anthropic")
                    provider_data.supported_apis = inferred_apis

                # Convert create data to update data (provider_type is fixed)
                update_data = _provider_update_data(provider_data, exclude=("provider_type",))

                credentials = await update_provider_credentials(db, provider_key, **update_data)
                if not credentials:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Provider credentials not found: {provider_key}"
                    )
                
                # Automatically sync models for the updated provider
                logging.info(f"🔄 Starting auto-sync for updated provider: {credentials.provider_key}")
                sync_result = await auto_sync_on_provider_change(db, credentials.provider_key, action="update")
                logging.info(f"Auto-sync result for updated provider {credentials.provider_key}: {sync_result}")
                
            else:
                logging.warning(f"❌ Provider already exists: {provider_key}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Provider already exists: {provider_key}. Use upsert=true to update existing provider."
                )
        else:
            # Create provider credentials
            logging.info(f"💾 Creating new provider credentials in database...")

            # Auto-infer supported_apis for Azure providers
            if provider_data.provider_type == "azure" and not provider_data.supported_apis:
                inferred_apis = ["openai"]
                if (
                    provider_data.azure_backend == "foundry"
                    and provider_data.anthropic_deployments
                    and len(provider_data.anthropic_deployments) > 0
                ):
                    inferred_apis.append("anthropic")
                provider_data.supported_apis = inferred_apis

            # Serialize supported_apis for storage
            supported_apis_json = None
            if provider_data.supported_apis:
                supported_apis_json = json.dumps(provider_data.supported_apis)

            credentials = await create_provider_credentials(
                db=db,
                provider_type=provider_data.provider_type,
                instance_name=provider_data.instance_name,
                enabled=provider_data.enabled,
                endpoint=provider_data.endpoint,
                api_key=provider_data.api_key,
                discovery_api_version=provider_data.discovery_api_version,
                azure_backend=provider_data.azure_backend or ("openai" if provider_data.provider_type == "azure" else None),
                region=provider_data.region,
                access_key_id=provider_data.access_key_id,
                secret_access_key=provider_data.secret_access_key,
                base_url=provider_data.base_url,
                provider_name=provider_data.provider_name,
                deployments=provider_data.deployments,
                openai_deployments=provider_data.openai_deployments,
                anthropic_deployments=provider_data.anthropic_deployments,
                deployment_groups=provider_data.deployment_groups,
                dynamic_discovery=provider_data.dynamic_discovery,
                supported_apis=supported_apis_json,
                anthropic_base_url=provider_data.anthropic_base_url,
                anthropic_auth_style=provider_data.anthropic_auth_style,
            )
            
            logging.info(f"✅ Provider credentials created: {credentials.provider_key}")
            
            # Automatically sync models for the new provider
            logging.info(f"🔄 Starting auto-sync for new provider: {credentials.provider_key}")
            sync_result = await auto_sync_on_provider_change(db, credentials.provider_key, action="create")
            logging.info(f"Auto-sync result for new provider {credentials.provider_key}: {sync_result}")
        
        # Parse deployments JSON for response
        deployment_groups, deployments = _parse_azure_deployment_fields(credentials.deployments_json)
        
        # Parse supported_apis JSON for response
        supported_apis = None
        if credentials.supported_apis:
            try:
                supported_apis = json.loads(credentials.supported_apis)
            except (json.JSONDecodeError, TypeError):
                supported_apis = ["openai"]

        response = ProviderCredentialsResponse(
            id=credentials.id,
            provider_key=credentials.provider_key,
            provider_type=credentials.provider_type,
            instance_name=credentials.instance_name,
            provider_name=credentials.provider_name,
            enabled=credentials.enabled,
            endpoint=credentials.endpoint,
            api_key=_mask_secret(credentials.api_key),
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=_mask_secret(credentials.secret_access_key),
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            deployment_groups=deployment_groups.get("groups", []),
            dynamic_discovery=credentials.dynamic_discovery,
            supported_apis=supported_apis,
            anthropic_base_url=credentials.anthropic_base_url,
            anthropic_auth_style=credentials.anthropic_auth_style,
            created_at=credentials.created_at,
            updated_at=credentials.updated_at
        )
        
        # Add sync result to response for debugging
        if "error" in sync_result:
            logging.warning(f"Auto-sync failed for provider {credentials.provider_key}: {sync_result['error']}")
        else:
            logging.info(f"Auto-sync successful: {sync_result.get('message', 'Models synced')}")
        
        return response
    except HTTPException:
        raise
    except ValueError as e:
        # Handle provider already exists error
        if "Provider already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}"
            )
    except Exception as e:
        logging.error(f"Unexpected error creating provider credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create provider credentials"
        )


@router.put("/providers", response_model=ProviderCredentialsResponse)
async def update_provider_credential(
    provider_data: ProviderCredentialsUpdate,
    provider_key: str = Query(..., description="Provider key to update"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Update provider credentials using query parameter."""
    try:
        # URL decode the provider_key
        decoded_provider_key = urllib.parse.unquote(provider_key)
        
        update_data = _provider_update_data(provider_data)

        credentials = await update_provider_credentials(db, decoded_provider_key, **update_data)
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider credentials not found: {decoded_provider_key}"
            )
        
        # Automatically sync models for the updated provider
        sync_result = await auto_sync_on_provider_change(db, credentials.provider_key, action="update")
        logging.info(f"Auto-sync result for updated provider {credentials.provider_key}: {sync_result}")
        if credentials.provider_key != decoded_provider_key:
            # A rename rewrote group members, aliases and access exceptions too.
            await reload_reference_caches()
        
        # Parse deployments JSON for response
        deployment_groups, deployments = _parse_azure_deployment_fields(credentials.deployments_json)
        
        # Parse supported_apis JSON for update response
        supported_apis = None
        if credentials.supported_apis:
            try:
                supported_apis = json.loads(credentials.supported_apis)
            except (json.JSONDecodeError, TypeError):
                supported_apis = ["openai"]

        response = ProviderCredentialsResponse(
            id=credentials.id,
            provider_key=credentials.provider_key,
            provider_type=credentials.provider_type,
            instance_name=credentials.instance_name,
            provider_name=credentials.provider_name,
            enabled=credentials.enabled,
            endpoint=credentials.endpoint,
            api_key=_mask_secret(credentials.api_key),
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=_mask_secret(credentials.secret_access_key),
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            deployment_groups=deployment_groups.get("groups", []),
            dynamic_discovery=credentials.dynamic_discovery,
            supported_apis=supported_apis,
            anthropic_base_url=credentials.anthropic_base_url,
            anthropic_auth_style=credentials.anthropic_auth_style,
            created_at=credentials.created_at,
            updated_at=credentials.updated_at
        )
        
        # Add sync result to response for debugging
        if "error" in sync_result:
            logging.warning(f"Auto-sync failed for provider {credentials.provider_key}: {sync_result['error']}")
        else:
            logging.info(f"Auto-sync successful: {sync_result.get('message', 'Models synced')}")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update provider credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update provider credentials"
        )


@router.delete("/providers")
async def delete_provider_credential(
    provider_key: str = Query(..., description="Provider key to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Delete provider credentials using query parameter."""
    try:
        # URL decode the provider_key
        decoded_provider_key = urllib.parse.unquote(provider_key)
        
        success = await delete_provider_credentials(db, decoded_provider_key)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider credentials not found: {decoded_provider_key}"
            )
        
        # Remove the provider from the provider manager and refresh from database
        await provider_manager.remove_provider(decoded_provider_key)
        
        # Refresh provider manager to ensure it only has valid providers from database
        await provider_manager.refresh_providers_from_database()
        # Group members, aliases and access exceptions naming it were cleaned up too.
        await reload_reference_caches()

        return {"message": f"Provider credentials deleted: {decoded_provider_key}"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete provider credentials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete provider credentials"
        )


@router.get("/providers/export")
async def export_provider_config(
    include_secrets: bool = Query(
        False,
        description="Include plaintext api_key/secret_access_key in the export "
                    "(opt-in). Defaults to False so secrets are not written to disk.",
    ),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Export all provider credentials and model configurations as a JSON file."""
    try:
        credentials = await get_all_provider_credentials(db)
        model_configs = await get_all_model_configurations(db)

        providers_export = []
        for cred in credentials:
            deployment_groups, deployments = _parse_azure_deployment_fields(cred.deployments_json)

            supported_apis = None
            if cred.supported_apis:
                try:
                    supported_apis = json.loads(cred.supported_apis)
                except (json.JSONDecodeError, TypeError):
                    supported_apis = ["openai"]

            providers_export.append({
                "provider_key": cred.provider_key,
                "provider_type": cred.provider_type,
                "instance_name": cred.instance_name,
                "provider_name": cred.provider_name,
                "enabled": cred.enabled,
                "endpoint": cred.endpoint,
                "api_key": cred.api_key if include_secrets else None,
                "discovery_api_version": cred.discovery_api_version,
                "azure_backend": cred.azure_backend or ("openai" if cred.provider_type == "azure" else None),
                "region": cred.region,
                "access_key_id": cred.access_key_id,
                "secret_access_key": cred.secret_access_key if include_secrets else None,
                "base_url": cred.base_url,
                "deployments": deployments,
                "openai_deployments": deployment_groups.get("openai", []),
                "anthropic_deployments": deployment_groups.get("anthropic", []),
                "deployment_groups": deployment_groups.get("groups", []),
                "dynamic_discovery": cred.dynamic_discovery,
                "supported_apis": supported_apis,
                "anthropic_base_url": cred.anthropic_base_url,
                "anthropic_auth_style": cred.anthropic_auth_style,
            })

        models_export = [
            {
                "model_id": m.model_id,
                "provider_key": m.provider_key,
                "model_name": m.model_name,
                "is_enabled": m.is_enabled,
            }
            for m in model_configs
        ]

        from datetime import timezone
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "providers": providers_export,
            "model_configurations": models_export,
        }

        json_bytes = json.dumps(export_data, indent=2, default=str).encode("utf-8")
        return StreamingResponse(
            io.BytesIO(json_bytes),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=llm-proxy-config.json"},
        )
    except Exception as e:
        logging.error(f"Failed to export provider config: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export provider config"
        )


class ImportResult(BaseModel):
    imported: int
    skipped: int
    overwritten: int
    errors: List[str]


@router.post("/providers/import", response_model=ImportResult)
async def import_provider_config(
    file: UploadFile = File(...),
    overwrite: bool = Query(False, description="Overwrite existing providers instead of skipping"),
    sync_models: bool = Query(True, description="Sync models for each imported provider"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Import provider credentials and model configurations from a JSON file."""
    try:
        content = await file.read()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON file: {str(e)}"
            )

        if not isinstance(data, dict) or "providers" not in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid export format: missing 'providers' key"
            )

        imported = 0
        skipped = 0
        overwritten = 0
        errors = []

        for provider_data in data.get("providers", []):
            try:
                provider_key = provider_data.get("provider_key")
                if not provider_key:
                    errors.append("Skipped entry with missing provider_key")
                    continue

                # Validate Azure-specific fields
                if provider_data.get("provider_type") == "azure":
                    ab = provider_data.get("azure_backend")
                    if ab and ab not in VALID_AZURE_BACKENDS:
                        errors.append(f"Provider '{provider_key}': invalid azure_backend '{ab}', must be one of {VALID_AZURE_BACKENDS}")
                        continue

                    try:
                        if provider_data.get("deployment_groups") is not None:
                            # Groups (even []) are the only deployment source;
                            # the per-bucket lists alongside them are derived.
                            groups = normalize_deployment_groups(provider_data["deployment_groups"], strict=True)
                            derived = normalize_azure_deployments({"groups": groups})
                            provider_data["deployment_groups"] = groups
                            provider_data["openai_deployments"] = derived["openai"]
                            provider_data["anthropic_deployments"] = derived["anthropic"]
                            provider_data["deployments"] = merge_azure_deployments(derived)
                        else:
                            validate_deployment_names(provider_data.get("openai_deployments"))
                            validate_deployment_names(provider_data.get("anthropic_deployments"))
                            validate_deployment_names(provider_data.get("deployments"))
                    except ValueError as e:
                        errors.append(f"Provider '{provider_key}': {str(e)}")
                        continue

                    if provider_data.get("dynamic_discovery") is True:
                        has_deps = (
                            provider_data.get("deployments")
                            or provider_data.get("openai_deployments")
                            or provider_data.get("anthropic_deployments")
                            or provider_data.get("deployment_groups")
                        )
                        if has_deps:
                            errors.append(f"Provider '{provider_key}': cannot specify deployments with dynamic_discovery enabled")
                            continue

                existing = await get_provider_credentials(db, provider_key)

                if existing and not overwrite:
                    skipped += 1
                    continue

                supported_apis_val = provider_data.get("supported_apis")
                # Auto-infer supported_apis for Azure imports
                if provider_data.get("provider_type") == "azure" and not supported_apis_val:
                    supported_apis_val = ["openai"]
                    if (
                        provider_data.get("azure_backend") == "foundry"
                        and provider_data.get("anthropic_deployments")
                    ):
                        supported_apis_val.append("anthropic")
                supported_apis_json = json.dumps(supported_apis_val) if supported_apis_val else None

                # Shared field set for both create and update-in-place.
                provider_kwargs = dict(
                    provider_name=provider_data["provider_name"],
                    enabled=provider_data.get("enabled", True),
                    endpoint=provider_data.get("endpoint"),
                    api_key=provider_data.get("api_key"),
                    discovery_api_version=provider_data.get("discovery_api_version"),
                    azure_backend=provider_data.get("azure_backend") or (
                        "openai" if provider_data.get("provider_type") == "azure" else None
                    ),
                    region=provider_data.get("region"),
                    access_key_id=provider_data.get("access_key_id"),
                    secret_access_key=provider_data.get("secret_access_key"),
                    base_url=provider_data.get("base_url"),
                    deployments=provider_data.get("deployments"),
                    openai_deployments=provider_data.get("openai_deployments"),
                    anthropic_deployments=provider_data.get("anthropic_deployments"),
                    deployment_groups=provider_data.get("deployment_groups"),
                    dynamic_discovery=provider_data.get("dynamic_discovery"),
                    supported_apis=supported_apis_json,
                    anthropic_base_url=provider_data.get("anthropic_base_url"),
                    anthropic_auth_style=provider_data.get("anthropic_auth_style"),
                )

                if existing and overwrite:
                    # Update in place instead of delete+create. delete and create
                    # commit in separate transactions, so a failed create after a
                    # committed delete permanently loses the existing provider.
                    # update_provider_credentials rolls back on failure.
                    credentials = await update_provider_credentials(
                        db, provider_key,
                        provider_type=provider_data["provider_type"],
                        instance_name=provider_data["instance_name"],
                        **provider_kwargs,
                    )
                else:
                    credentials = await create_provider_credentials(
                        db=db,
                        provider_type=provider_data["provider_type"],
                        instance_name=provider_data["instance_name"],
                        **provider_kwargs,
                    )

                if sync_models:
                    try:
                        await auto_sync_on_provider_change(db, credentials.provider_key, action="create")
                    except Exception as sync_err:
                        logging.warning(f"Model sync failed for {provider_key}: {sync_err}")

                if existing:
                    overwritten += 1
                else:
                    imported += 1

            except Exception as e:
                errors.append(f"Failed to import provider '{provider_data.get('provider_key', '?')}': {str(e)}")

        # Restore model enabled/disabled states. Exports from before the provider and
        # model flags were decoupled carry every model of a disabled provider as off,
        # because the old provider toggle wrote through to them. When all of a disabled
        # provider's models are off that way, the flags say nothing about the models:
        # import them as on, so re-enabling the provider restores them (the same rule
        # as the startup migration, _normalise_coupled_model_flags).
        model_rows = data.get("model_configurations", [])
        disabled_providers = {
            p.get("provider_key") for p in data.get("providers", [])
            if isinstance(p, dict) and p.get("enabled") is False
        }
        coupled = {
            pk for pk in disabled_providers
            if (rows := [m for m in model_rows if m.get("provider_key") == pk])
            and not any(m.get("is_enabled", True) for m in rows)
        }
        for model_data in model_rows:
            try:
                await create_or_update_model_configuration(
                    db,
                    model_id=model_data["model_id"],
                    provider_key=model_data["provider_key"],
                    model_name=model_data["model_name"],
                    is_enabled=model_data.get("is_enabled", True) or model_data["provider_key"] in coupled,
                )
            except Exception as e:
                logging.warning(f"Failed to restore model config '{model_data.get('model_id')}': {e}")

        # Refresh provider manager
        await provider_manager.refresh_providers_from_database()

        return ImportResult(imported=imported, skipped=skipped, overwritten=overwritten, errors=errors)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to import provider config: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import provider config"
        )


# ==================== Active Requests ====================


@router.get("/requests/active")
async def get_active_requests(
    current_admin: AdminUser = Depends(get_current_admin),
):
    """Return current snapshot of all active requests."""
    from app.request_tracker import request_tracker
    return {
        "active_requests": request_tracker.get_active_requests(),
        "summary": request_tracker.get_summary(),
    }


@router.get("/requests/stream")
async def stream_active_requests(
    request: Request,
    current_admin: AdminUser = Depends(get_current_admin),
):
    """SSE stream of active request events."""
    from app.request_tracker import request_tracker

    async def event_generator():
        queue = await request_tracker.subscribe()
        try:
            snapshot = json.dumps({
                "event": "snapshot",
                "active_requests": request_tracker.get_active_requests(),
                "summary": request_tracker.get_summary(),
            })
            yield f"data: {snapshot}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if data is None:
                    break

                yield f"data: {data}\n\n"
        finally:
            await request_tracker.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ==================== Usage ====================


@router.get("/usage/years")
async def get_usage_years_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return list of years that have any usage data."""
    years = await get_usage_years(db)
    return {"years": years}


@router.get("/usage")
async def get_usage(
    view: Optional[str] = Query(None, description="'user', 'model' or 'pool' for drill-down"),
    id: Optional[str] = Query(None, description="Identity value to drill into"),
    # Annotated so a direct call (tests) gets a real None, not the Query marker.
    user_id: Annotated[Optional[int], Query(description="Exact user_id when view=user; disambiguates a shared label")] = None,
    window: str = Query("30d", pattern="^(24h|today|yesterday|7d|30d|month|all)$", description="Time window: 24h | today | yesterday | 7d | 30d | month | all"),
    year: Optional[int] = Query(None, description="Year (required when window=month)"),
    month: Optional[int] = Query(None, description="Month 1-12 (required when window=month)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated usage for the requested time window.

    - No view/id: top-level per_user + per_model + per_pool + totals.
    - ?view=user&id=alice: breakdown by model for alice.
    - ?view=model&id=gpt-4: breakdown by user for gpt-4.
    - ?view=pool&id=3: breakdown by member for pool 3.
    - window=24h|today|yesterday|7d|30d|month
    - window=month requires year and month params.
    """
    from app.request_tracker import request_tracker
    from app.auth.database import (
        get_usage_by_user_and_model, get_usage_earliest_date, get_usage_timeseries,
        get_usage_totals_by_pool,
    )
    from app.routes.pools import member_spans, pool_membership_map, window_bounds
    await request_tracker.flush_pending()

    # Pool drill-down: the members' own per-user split, scoped to one pool.
    #
    # Built from the same primitives build_pool_usage folds rather than by calling it,
    # because that function returns per_model / per_group / per_member_per_model too —
    # the full user x model cross product, all of which would be discarded here — and
    # it resolves the group tables to do it. Its drill-down path also 403s on
    # non-membership, which is not the right answer for an admin.
    #
    # Scoped by the rows' pool_id, like build_pool_usage: a member's rows count only
    # for the requests they sent while in the pool.
    if view == "pool":
        pool_id = _parse_pool_id(id)
        membership = await pool_membership_map(db)
        if pool_id not in membership:
            raise HTTPException(status_code=404, detail="Pool not found")
        pool = membership[pool_id]
        identities = pool["members"]

        cross = await get_usage_by_user_and_model(
            db, pool_id=pool_id, window=window, year=year, month=month,
        )
        folded: dict = {}
        for r in cross:
            name = r["user_identity"]
            prev = folded.get(name)
            folded[name] = (r["user_type"], (prev[1] if prev else 0) + r["request_count"])

        # Zero-filled from the current roster *plus* anyone whose stint overlaps the
        # window, so every member shows up even with no traffic (an absent row reads as
        # "not in the pool") and a leaver's contribution has a row to land in.
        lo, hi = window_bounds(window, year, month)
        spans = await member_spans(db, pool_id, lo=lo, hi=hi)
        listed = list(dict.fromkeys(identities + [s[0] for s in spans] + list(folded)))
        breakdown = [
            {
                "user_identity": name,
                "user_type": folded.get(name, (None, 0))[0],
                "request_count": folded.get(name, (None, 0))[1],
            }
            for name in listed
        ]
        breakdown.sort(key=lambda r: (-r["request_count"], r["user_identity"]))

        return {
            "window": window,
            "pool": {"id": pool_id, "name": pool["name"], "member_count": len(identities)},
            "breakdown": breakdown,
            "timeseries": await get_usage_timeseries(
                db, pool_ids=[pool_id], window=window, year=year, month=month,
            ),
            "earliest_date": await get_usage_earliest_date(db),
        }

    filter_user_id: Optional[int] = None
    filter_model: Optional[str] = None

    if view == "user" and id is not None:
        filter_user_id = await _resolve_usage_user_id(db, id, user_id)
    elif view == "model" and id is not None:
        filter_model = id

    result = await get_usage_aggregates(
        db,
        filter_user_id=filter_user_id,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["timeseries"] = await get_usage_timeseries(
        db,
        filter_user_id=filter_user_id,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["earliest_date"] = await get_usage_earliest_date(db)

    # Pools are a third axis over the same traffic, so per_pool rides along in the same
    # payload as per_user / per_model and the By Pool toggle costs no extra request.
    #
    # Not a fold of the per_user rows: those are each member's *whole* total for the
    # window, which would credit a pool with everything its members sent before they
    # joined. The rows' own pool_id is the honest attribution, and it is the same
    # number the pool's drill-down shows, so the two views cannot disagree. Only pools
    # that still exist are listed; rows of a deleted pool keep its id but stay in the
    # members' own totals rather than surfacing as a phantom pool.
    if view is None:
        membership = await pool_membership_map(db)
        pool_ids = list(membership)
        counts = await get_usage_totals_by_pool(
            db, pool_ids, window=window, year=year, month=month,
        )

        # The By Pool chart's own series. `timeseries` above covers *all* traffic, which
        # is the right series for By User and By Model -- every request has a user and a
        # model, so those two only re-partition the same bars. Pools don't partition it:
        # unpooled users exist, and a member's pre-join rows aren't the pool's. Reusing
        # the overall series there drew a chart that disagreed with the table under it.
        result["pool_timeseries"] = await get_usage_timeseries(
            db, pool_ids=pool_ids, window=window, year=year, month=month,
        )

        result["per_pool"] = sorted(
            (
                {
                    "pool_id": pool_id,
                    "name": pool["name"],
                    "member_count": len(pool["members"]),
                    # Carried so the delete confirm can name the members. Bounded by
                    # POOL_MEMBERS_HARD_CAP, so this is a handful of strings.
                    "members": pool["members"],
                    "request_count": counts.get(pool_id, 0),
                }
                for pool_id, pool in membership.items()
            ),
            key=lambda p: (-p["request_count"], p["name"]),
        )

    return result


def _parse_pool_id(raw: Optional[str]) -> int:
    """Pool ids arrive as strings on a query string shared with user/model identities."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="id must be a pool id when view=pool")


async def _resolve_usage_user_id(
    db: AsyncSession, name: str, user_id: Optional[int] = None,
) -> int:
    """Map a row from the usage UI to the user_id the usage tables are keyed by.

    The By User rows carry their user_id, and the UI sends it back; that is the only
    unambiguous key. A label alone can name two identities: is_reserved_username only
    rejects the admin's name while the admin is enabled, so a user who registered it
    during a disabled stint -- or under an earlier ADMIN_USERNAME -- shares the label
    with the admin's own rows.

    Without a user_id (a hand-built URL, an older client), the name is resolved with
    `users` first, matching the rekey migration's rule that a real account wins a
    collision. The config admin is not a users row; its traffic lives under
    ADMIN_USAGE_USER_ID, found by its current name or, since admin rows are never
    relabelled, by any label its history was written under. A name that matches none of
    these is a 404: a deleted user's rows went with the account.
    """
    from app.auth.database import is_reserved_username, usage_has_admin_label
    from app.auth.models import ADMIN_USAGE_USER_ID

    if user_id is not None:
        if user_id == ADMIN_USAGE_USER_ID or await get_user_by_id(db, user_id) is not None:
            return user_id
        raise HTTPException(status_code=404, detail="User not found")

    user = await get_user_by_username(db, name)
    if user is not None:
        return user.id
    if is_reserved_username(name) or await usage_has_admin_label(db, name):
        return ADMIN_USAGE_USER_ID
    raise HTTPException(status_code=404, detail="User not found")


@router.delete("/usage")
async def delete_usage(
    view: str = Query(..., description="'user', 'model' or 'pool'"),
    id: str = Query(..., description="Identity value, or pool id when view=pool"),
    # Annotated so a direct call (tests) gets a real None, not the Query marker.
    user_id: Annotated[Optional[int], Query(description="Exact user_id when view=user; disambiguates a shared label")] = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete all usage data, for all time, for one user, one model or one pool.

    Removes the rows from the hourly, daily and monthly tables together — leaving any
    of them behind would make the data reappear as soon as the time window changes.

    view=pool deletes the rows *attributed to the pool* (their pool_id), which is
    exactly the number the By Pool table shows. What its members sent outside the pool
    is theirs and stays.

    The identity is passed as a query parameter rather than a path segment because
    model ids contain slashes (e.g. 'openai/gpt-4o').
    """
    from contextlib import AsyncExitStack

    from app.request_tracker import request_tracker
    from app.rate_limit import rate_limit_tracker
    from app.routes.pools import (
        pool_guard, pool_ids_for_users, pool_membership_map, resettle_after_usage_purge,
    )

    if view not in ("user", "model", "pool"):
        raise HTTPException(status_code=400, detail="view must be 'user', 'model' or 'pool'")
    if not id or not id.strip():
        raise HTTPException(status_code=400, detail="id is required")

    # Resolve the axis value the tables are keyed by, and who is affected by it.
    members: List[str] = []
    resettle_user_ids: List[int] = []
    if view == "pool":
        pool_id = _parse_pool_id(id)
        membership = await pool_membership_map(db)
        if pool_id not in membership:
            raise HTTPException(status_code=404, detail="Pool not found")
        members = membership[pool_id]["members"]
        # Current members for the confirm text, but everyone with rows in the pool for
        # the resettle: a user who left earlier today still owns rows stamped with it,
        # and the purge takes those too. Leaving them out would strand their new pool's
        # ledger charging traffic that no longer exists for the rest of the day.
        resettle_user_ids = list(dict.fromkeys(
            membership[pool_id]["member_ids"] + await usage_user_ids_for_pool(db, pool_id)
        ))
        axis, value = "pool", pool_id
    elif view == "user":
        user_id = await _resolve_usage_user_id(db, id, user_id)
        resettle_user_ids = [user_id]
        axis, value = "user", user_id
    else:
        axis, value = "model", id

    # The purge spans both places usage lives — the DB rows and the counts still
    # buffered in the tracker — so the flush is held off across the whole sequence.
    # Otherwise one landing between the two writes its pre-purge snapshot straight
    # back into the tables that were just cleared.
    #
    # The pool locks are taken first, outside the pause, because resettle_after_usage_purge
    # below acquires them and settlement takes pause_flush internally: acquiring in the
    # other order here and in a concurrent settle is the AB-BA cycle pool_lock's docstring
    # warns about. Re-acquiring them inside is free — they are re-entrant per task.
    async with AsyncExitStack() as pool_stack:
        for pid in await pool_ids_for_users(db, resettle_user_ids):
            await pool_stack.enter_async_context(pool_guard(pid))

        async with request_tracker.pause_flush():
            # Drain the buffer first so counts recorded in the last cycle are deleted too.
            await request_tracker.flush_pending()

            try:
                # Counted before the delete: this is the request count the admin saw in
                # the table, not the number of rows, which is spread over three tables
                # holding the same traffic at different granularities.
                total = await count_usage_requests(db, axis, value)
                deleted = await delete_usage_records(db, axis, value)
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Failed to delete usage for {view} '{id}': {e}")
                raise HTTPException(status_code=500, detail="Failed to delete usage data")

            # Anything buffered between the flush and the commit would otherwise flush
            # back into the tables we just cleared.
            dropped = await request_tracker.drop_buffered_usage(axis, value)

            # Strictly after the drop above: settle_pool flushes the tracker itself, so
            # running it first would write those buffered counts back to the DB, where
            # drop_buffered_usage can no longer reach them. A pooled user's ledger charges
            # them for traffic that no longer exists until this runs.
            if resettle_user_ids:
                await resettle_after_usage_purge(db, resettle_user_ids)

    # RPD is a SUM over today's rows, and the counts are cached per identity.
    # resettle_after_usage_purge already invalidated and re-snapshotted the pool paths.
    if view == "model":
        # The same settlement staleness exists here in principle, but a model's traffic
        # spans an unbounded set of pools; dropping every RPD count is the accepted
        # answer, and the numbers converge on the next settle.
        rate_limit_tracker.invalidate_all_rpd()
    elif view == "user":
        rate_limit_tracker.invalidate_identity(id)

    logger.info(
        f"Admin '{current_admin.username}' deleted usage for {view} '{id}': "
        f"{deleted} (+{dropped} buffered requests dropped)"
    )
    return {"deleted": deleted, "total": total, "members": members}


# =========================================================================== #
# Request pools — read-only visibility plus two force actions
#
# Everything a pool member can see about their own pool, an admin can see about
# every pool. The payloads come from app/routes/pools.py so the admin's picture
# cannot drift from the members'; only the authorization step differs.
# =========================================================================== #


@router.get("/pools", response_model=List[AdminPoolResponse])
async def admin_get_pools(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Every pool with its owner, members, and per-tier limit / used / charged / carry."""
    from app.routes.pools import admin_list_pools
    try:
        return await admin_list_pools(db)
    except Exception as e:
        logger.error(f"Admin pool listing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load pools")


@router.get("/pools/settings", response_model=PoolSettingsResponse)
async def admin_get_pool_settings(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Global pool policy: max members per pool and the per-member share cap."""
    return await load_pool_settings(db)


@router.put("/pools/settings", response_model=PoolSettingsResponse)
async def admin_update_pool_settings(
    body: PoolSettingsUpdate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update the global pool policy every pool without its own override inherits."""
    from app.rate_limit import rate_limit_tracker
    row = await upsert_pool_settings(
        db, body.max_members, body.max_member_share_pct, current_admin.username,
    )
    # Only the caps moved, not any count, so the cached RPD counts stay valid.
    await rate_limit_tracker.reload_config()
    logger.info(
        f"Admin '{current_admin.username}' set pool settings: max_members={row.max_members}, "
        f"max_member_share_pct={row.max_member_share_pct}"
    )
    return PoolSettingsResponse.model_validate(row)


@router.put("/pools/limits")
async def admin_update_pool_limits(
    body: AdminPoolLimitsUpdate,
    pool_id: int = Query(..., description="Pool ID"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Override max members and/or the share cap for one pool. null inherits the global value."""
    from app.routes.pools import admin_set_pool_limits
    from app.rate_limit import rate_limit_tracker
    try:
        pool = await admin_set_pool_limits(
            db, pool_id, body.max_members, body.max_member_share_pct, body.model_fields_set,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin pool limits update failed for pool {pool_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update pool limits")
    await rate_limit_tracker.reload_config()
    logger.info(
        f"Admin '{current_admin.username}' set pool '{pool.name}' overrides: "
        f"max_members={pool.max_members}, max_member_share_pct={pool.max_member_share_pct}"
    )
    return {
        "message": f"Updated {pool.name}",
        "max_members": pool.max_members,
        "max_member_share_pct": pool.max_member_share_pct,
    }


@router.get("/pools/usage")
async def admin_get_pool_usage(
    pool_id: int = Query(..., description="Pool ID"),
    window: str = Query("30d", pattern="^(24h|today|yesterday|7d|30d|month|all)$"),
    view: Optional[str] = Query(None, pattern="^(user|model|group)$"),
    id: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """The same payload a member sees, for any pool. The admin skips the membership check."""
    from app.routes.pools import build_pool_usage
    try:
        return await build_pool_usage(
            db, pool_id, window=window, view=view, target=id, year=year, month=month,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin pool usage read failed for pool {pool_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load pool usage")


@router.delete("/pools")
async def admin_delete_pool_endpoint(
    pool_id: int = Query(..., description="Pool ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Settle every member, then force-delete the pool."""
    from app.routes.pools import admin_delete_pool
    try:
        name = await admin_delete_pool(db, pool_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin pool delete failed for pool {pool_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete pool")
    logger.info(f"Admin '{current_admin.username}' deleted pool '{name}'")
    return {"message": f"Deleted {name}"}


@router.delete("/pools/members")
async def admin_remove_pool_member(
    user_id: int = Query(..., description="User ID to remove from their pool"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Settle, then force-remove a user from whatever pool they are in."""
    from app.routes.pools import admin_remove_member
    try:
        username = await admin_remove_member(db, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin pool member removal failed for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to remove pool member")
    logger.info(f"Admin '{current_admin.username}' removed '{username}' from their pool")
    return {"message": f"Removed {username}"}

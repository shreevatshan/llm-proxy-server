"""Admin routes for user management and system administration."""

from fastapi import APIRouter, Request, Depends, HTTPException, status, Response, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
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
    delete_model_alias, reorder_model_aliases,
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
    set_group_members, get_model_group_limits,
    list_instance_groups, create_instance_group, update_instance_group, delete_instance_group,
    set_instance_group_members, get_instance_group_limits,
    get_websearch_settings, upsert_websearch_settings,
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
    ProfileCreate, ProfileUpdate, ProfileResponse, ProfileLimitsUpdate, ProfileLimitsResponse,
    ProfileMemberResponse, UserProfileAssign, ProviderConfigurationResponse, ModelConfigurationResponse,
    ModelManagementTree, ToggleRequest, BulkToggleRequest, BulkUserActionRequest, ModelSearchResponse,
    ProviderCredentialsCreate, ProviderCredentialsUpdate, ProviderCredentialsResponse,
    AdminPasswordReset, AdminUserModify, VALID_AZURE_BACKENDS,
    UserManagementSettingsResponse, UserManagementSettingsUpdate,
    ModelGroupCreate, ModelGroupUpdate, ModelGroupMembersUpdate, ModelGroupResponse,
    InstanceGroupCreate, InstanceGroupUpdate, InstanceGroupMembersUpdate, InstanceGroupResponse,
    ModelAliasUpsert, ModelAliasResponse, ModelAliasReorder, ModelAliasTestMatch,
    MODEL_ALIAS_API_SURFACES, validate_alias_pattern,
    AdminPoolResponse, AdminPoolLimitsUpdate, PoolSettingsResponse, PoolSettingsUpdate,
    WebSearchSettingsResponse, WebSearchSettingsUpdate, WebSearchTestRequest, WebSearchEnginesRequest,
    FOURGET_WEB_SCRAPERS, WEBSEARCH_PROVIDER_LABELS,
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
        
        await db.commit()
        
        # Refresh provider manager from database (should now be empty)
        await provider_manager.refresh_providers_from_database()
        
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
    """Toggle provider and all its models using unified ProviderCredentials system (query parameter version)."""
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
    """A profile's overall limits plus its limit for every model and instance group."""
    await _require_profile(db, profile_id)
    groups = await get_profile_limits(db, profile_id)
    return ProfileLimitsResponse(profile=await _profile_response(db, profile_id), **groups)


@router.put("/profiles/limits", response_model=ProfileResponse)
async def update_profile_limits_endpoint(
    body: ProfileLimitsUpdate,
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a profile's overall RPM/RPD (null = unlimited; omitted = unchanged)."""
    await _require_profile(db, profile_id)
    await upsert_profile_limits(db, profile_id, body.rpm_limit, body.rpd_limit,
                                current_admin.username, body.model_fields_set)
    await _resettle_profile(db, profile_id)
    return await _profile_response(db, profile_id)


async def _set_profile_group_limit(db, admin, profile_id, group_id, body, instance: bool):
    await _require_profile(db, profile_id)
    getter = get_instance_group_limits if instance else get_model_group_limits
    if await getter(db, group_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=("Instance" if instance else "Model") + " group not found")
    await upsert_profile_group_limit(db, profile_id, group_id, body.rpm_limit, body.rpd_limit,
                                     admin.username, body.model_fields_set, instance=instance)
    await _resettle_profile(db, profile_id)
    groups = await get_profile_limits(db, profile_id)
    return ProfileLimitsResponse(profile=await _profile_response(db, profile_id), **groups)


async def _set_group_limit_everywhere(db, admin, group_id, body, instance: bool):
    getter = get_instance_group_limits if instance else get_model_group_limits
    if await getter(db, group_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=("Instance" if instance else "Model") + " group not found")
    await set_group_limit_for_all_profiles(db, group_id, body.rpm_limit, body.rpd_limit,
                                           admin.username, instance=instance,
                                           fields_set=body.model_fields_set)
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)
    # Echo only what changed: an omitted field kept each profile's own value, and
    # reporting it as null would read as "set to unlimited".
    return {"group_id": group_id, **body.model_dump(include=body.model_fields_set)}


@router.put("/profiles/model-group-limits", response_model=ProfileLimitsResponse)
async def update_profile_model_group_limit_endpoint(
    body: ProfileLimitsUpdate,
    profile_id: int = Query(...),
    group_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set one profile's RPM/RPD for a model group (null = unlimited)."""
    return await _set_profile_group_limit(db, current_admin, profile_id, group_id, body, instance=False)


@router.put("/profiles/instance-group-limits", response_model=ProfileLimitsResponse)
async def update_profile_instance_group_limit_endpoint(
    body: ProfileLimitsUpdate,
    profile_id: int = Query(...),
    group_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set one profile's RPM/RPD for an instance group (null = unlimited)."""
    return await _set_profile_group_limit(db, current_admin, profile_id, group_id, body, instance=True)


@router.put("/profiles/model-group-limits/all")
async def update_model_group_limit_all_profiles_endpoint(
    body: ProfileLimitsUpdate,
    group_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Give every profile the same limits for one model group."""
    return await _set_group_limit_everywhere(db, current_admin, group_id, body, instance=False)


@router.put("/profiles/instance-group-limits/all")
async def update_instance_group_limit_all_profiles_endpoint(
    body: ProfileLimitsUpdate,
    group_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Give every profile the same limits for one instance group."""
    return await _set_group_limit_everywhere(db, current_admin, group_id, body, instance=True)


@router.get("/profiles/members", response_model=List[ProfileMemberResponse])
async def list_profile_members_endpoint(
    profile_id: int = Query(...),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """The profile's members with their live per-minute and today's request counts."""
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

def _websearch_response(row) -> WebSearchSettingsResponse:
    from app.websearch.settings import config_from_row

    cfg = config_from_row(row)
    return WebSearchSettingsResponse(
        enabled=cfg.enabled,
        provider=cfg.provider,
        searxng_base_url=cfg.searxng_base_url,
        engines=cfg.engines,
        categories=cfg.categories,
        language=cfg.language,
        safesearch=cfg.safesearch,
        time_range=cfg.time_range,
        fourget_base_url=cfg.fourget_base_url,
        fourget_scraper=cfg.fourget_scraper,
        fourget_lang=cfg.fourget_lang,
        fourget_country=cfg.fourget_country,
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


@router.get("/websearch/settings", response_model=WebSearchSettingsResponse)
async def get_websearch_settings_endpoint(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get web search interception settings (API key masked)."""
    return _websearch_response(await get_websearch_settings(db))


@router.put("/websearch/settings", response_model=WebSearchSettingsResponse)
async def update_websearch_settings_endpoint(
    body: WebSearchSettingsUpdate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Replace web search interception settings and reload the request-time snapshot."""
    row = await upsert_websearch_settings(db, _websearch_values(body), current_admin.username)
    from app.websearch.settings import config_from_row, websearch_settings_cache
    websearch_settings_cache.set_config(config_from_row(row))
    return _websearch_response(row)


@router.post("/websearch/test")
async def test_websearch_endpoint(
    body: WebSearchTestRequest,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Run one live query against the selected backend, with stored or unsaved settings."""
    import time as _time
    from app.websearch.backends import search
    from app.websearch.settings import config_from_row

    stored = config_from_row(await get_websearch_settings(db))
    cfg = stored
    if body.settings is not None:
        values = _websearch_values(body.settings)
        values["apply_to"] = frozenset(values["apply_to"])
        values["enabled_providers"] = frozenset(values["enabled_providers"])
        cfg = stored.with_overrides(**values)
    if not cfg.base_url:
        label = WEBSEARCH_PROVIDER_LABELS.get(cfg.provider, cfg.provider)
        raise HTTPException(status_code=400, detail=f"{label} base URL is not configured")

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
    """Engines and categories loaded on the SearXNG instance (from its /config)."""
    from app.websearch.searxng import SearxngConfigError, fetch_engines
    from app.websearch.settings import config_from_row

    stored = config_from_row(await get_websearch_settings(db))
    base_url = body.searxng_base_url or stored.searxng_base_url
    if not base_url:
        raise HTTPException(status_code=400, detail="SearXNG base URL is not configured")
    try:
        return await fetch_engines(base_url, stored.timeout_seconds)
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


# ==================== Model Alias Management API Endpoints ====================
#
# Every mapping belongs to a profile, and a profile's members are mapped with its
# list alone. `profile_id` defaults to Default throughout.

@router.get("/model-aliases", response_model=List[ModelAliasResponse])
async def list_model_aliases_endpoint(
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Profile whose mappings to list"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List one profile's aliases, including disabled aliases."""
    await _require_profile(db, profile_id)
    return [ModelAliasResponse.model_validate(row) for row in await get_all_model_aliases(db, profile_id)]


@router.post("/model-aliases", response_model=ModelAliasResponse)
async def upsert_model_alias_endpoint(
    body: ModelAliasUpsert,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create or update an alias after validating its pattern and canonical target.

    With `id`, that row is updated in place (and may be renamed); otherwise the
    row is matched by its alias text within the profile's list.
    """
    from sqlalchemy.exc import IntegrityError
    await _require_profile(db, body.profile_id)
    if body.id is not None:
        existing = await get_model_alias_by_id(db, body.id)
        if existing is None or existing.profile_id != body.profile_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    else:
        existing = await get_model_alias(db, body.alias, body.profile_id)

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
            profile_id=body.profile_id,
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A mapping for '{body.alias}' already exists",
        )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return ModelAliasResponse.model_validate(row)


@router.put("/model-aliases/order", response_model=List[ModelAliasResponse])
async def reorder_model_aliases_endpoint(
    body: ModelAliasReorder,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set one profile's evaluation order; `ids` must list every alias of it exactly once."""
    await _require_profile(db, body.profile_id)
    if not await reorder_model_aliases(db, body.ids, body.profile_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Mapping list changed — refresh and retry",
        )
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return [ModelAliasResponse.model_validate(row) for row in await get_all_model_aliases(db, body.profile_id)]


@router.get("/model-aliases/test", response_model=List[ModelAliasTestMatch])
async def match_model_alias_endpoint(
    model: str = Query(..., min_length=1),
    api: Optional[str] = Query(None),
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Test as a member of this profile"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Report which enabled mapping *model* would hit, per API surface.

    Uses the live resolver, so the answer is exactly what requests get. Without
    `api`, every surface is reported (rather than an unscoped match, which
    ignores API scoping and so never happens for a real request). Only
    `profile_id`'s own mappings are tried, as for its members.
    """
    if api is not None and api not in MODEL_ALIAS_API_SURFACES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown API surface: {api}")
    await _require_profile(db, profile_id)
    from app.model_alias import model_alias_resolver
    results = []
    for surface in ([api] if api else MODEL_ALIAS_API_SURFACES):
        rule = model_alias_resolver.match(model, surface, profile_id)
        if rule is None:
            results.append(ModelAliasTestMatch(api=surface, matched=False, resolved_model=model))
        else:
            results.append(ModelAliasTestMatch(
                api=surface, matched=True, resolved_model=rule.target, id=rule.id,
                alias=rule.pattern, match_type=rule.match_type, target_model_id=rule.target,
            ))
    return results


@router.delete("/model-aliases/{alias:path}")
async def delete_model_alias_endpoint(
    alias: str,
    profile_id: int = Query(DEFAULT_PROFILE_ID, description="Profile whose list holds the alias"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete an alias and reload the in-memory resolver immediately."""
    if not await delete_model_alias(db, alias, profile_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model alias not found")
    from app.model_alias import model_alias_resolver
    await model_alias_resolver.load_from_database()
    return {"message": f"Model alias '{alias}' deleted"}


# ==================== Model Group Management API Endpoints ====================

@router.get("/model-groups")
async def get_model_groups(
    group_id: Optional[int] = Query(None, description="Omit to list all groups; provide to fetch one"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all model groups, or fetch a single group's details including members."""
    def _to_response(g) -> ModelGroupResponse:
        return ModelGroupResponse(
            id=g.id,
            name=g.name,
            description=g.description,
            member_count=len(g.members),
            members=[m.model_id for m in g.members],
            updated_at=g.updated_at,
            updated_by=g.updated_by,
        )

    if group_id is not None:
        group = await list_model_groups(db, group_id=group_id)
        if group is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")
        return _to_response(group)

    groups = await list_model_groups(db)
    return [_to_response(g) for g in groups]


@router.post("/model-groups")
async def create_model_group_endpoint(
    body: ModelGroupCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new model group."""
    from sqlalchemy.future import select as sa_select
    from app.auth.models import ModelGroup as ModelGroupTable

    existing = await db.execute(
        sa_select(ModelGroupTable).where(ModelGroupTable.name.ilike(body.name))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Model group '{body.name}' already exists")

    group = await create_model_group(db, body.name, body.description, current_admin.username)
    # Limits live on profiles. The create form's values (if any) apply to every
    # profile, so a new group is never silently unlimited for everyone.
    if body.rpm_default is not None or body.rpd_default is not None:
        await set_group_limit_for_all_profiles(
            db, group.id, body.rpm_default, body.rpd_default, current_admin.username,
        )

    from app.rate_limit import rate_limit_tracker
    # No re-settle: a group created now has no members, so scope_for_model maps
    # nothing into it, its usage is 0 and settlement would find delta == 0.
    # Its limits start mattering when set_model_group_members_endpoint runs.

    return ModelGroupResponse(
        id=group.id, name=group.name, description=group.description,
        member_count=0, members=[], updated_at=group.updated_at, updated_by=group.updated_by,
    )


@router.put("/model-groups")
async def update_model_group_endpoint(
    group_id: int = Query(..., description="Group ID to update"),
    body: ModelGroupUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Rename or update description of a model group."""
    from sqlalchemy.future import select as sa_select
    from app.auth.models import ModelGroup as ModelGroupTable

    fields = {}
    if body:
        update_data = body.model_dump(exclude_unset=True)
        if "name" in update_data:
            existing = await db.execute(
                sa_select(ModelGroupTable).where(
                    ModelGroupTable.name.ilike(update_data["name"]),
                    ModelGroupTable.id != group_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Model group name '{update_data['name']}' already exists")
        fields = update_data

    group = await update_model_group(db, group_id, fields, current_admin.username)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_group(group_id)
    # No re-settle: ModelGroupUpdate carries name and description only, neither of
    # which is an apportionment weight.

    refreshed = await list_model_groups(db, group_id=group_id)
    return ModelGroupResponse(
        id=refreshed.id, name=refreshed.name, description=refreshed.description,
        member_count=len(refreshed.members), members=[m.model_id for m in refreshed.members],
        updated_at=refreshed.updated_at, updated_by=refreshed.updated_by,
    )


@router.delete("/model-groups")
async def delete_model_group_endpoint(
    group_id: int = Query(..., description="Group ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a model group (cascades members and per-user overrides)."""
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
    """Replace the member list for a model group. Validates each model_id exists and is not in another group."""
    from app.auth.models import ModelGroupMember as ModelGroupMemberTable, ModelConfiguration
    from sqlalchemy.future import select as sa_select

    # Confirm group exists
    group = await list_model_groups(db, group_id=group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    model_ids = body.model_ids if body else []

    # Validate all model_ids exist
    if model_ids:
        result = await db.execute(sa_select(ModelConfiguration).where(ModelConfiguration.model_id.in_(model_ids)))
        found = {r.model_id for r in result.scalars().all()}
        missing = [mid for mid in model_ids if mid not in found]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model_id(s) not found: {missing}",
            )

    # Check for conflicts: model_id already in another group
    if model_ids:
        result = await db.execute(
            sa_select(ModelGroupMemberTable).where(
                ModelGroupMemberTable.model_id.in_(model_ids),
                ModelGroupMemberTable.group_id != group_id,
            )
        )
        conflicts = [
            {"model_id": r.model_id, "current_group_id": r.group_id}
            for r in result.scalars().all()
        ]
        if conflicts:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"conflicts": conflicts})

    await set_group_members(db, group_id, model_ids)

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_group(group_id)
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)

    return {"group_id": group_id, "model_ids": model_ids}


# ==================== Profile Model Access API Endpoints ====================

class UserModelAccessItem(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    model_name: str
    provider_key: str
    globally_enabled: bool          # ModelConfiguration.is_enabled
    is_exception: bool              # explicit profile override exists
    is_allowed: Optional[bool]      # exception value; None if no exception
    effective_allowed: bool         # final decision (global AND policy/exception)


class ProfileModelAccessResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    profile_id: int
    name: str
    member_count: int = 0
    mode: str                       # default|allow|deny|custom
    models: List[UserModelAccessItem]


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
        is_exception=is_exception,
        is_allowed=exc_val if is_exception else None,
        effective_allowed=effective,
    )


async def _build_profile_model_access(db: AsyncSession, profile_id: int) -> ProfileModelAccessResponse:
    """Assemble the full model-access view of one profile."""
    profile = await _profile_response(db, profile_id)
    models = await get_all_model_configurations(db)
    exception_map = await list_profile_model_exceptions(db, profile_id)
    items = [
        _compute_access_item(ModelConfigurationResponse.from_orm(m), profile.model_access_mode, exception_map)
        for m in models
    ]
    return ProfileModelAccessResponse(
        profile_id=profile.id, name=profile.name, member_count=profile.member_count,
        mode=profile.model_access_mode, models=items,
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


# ==================== Instance Group Management API Endpoints ====================

@router.get("/instance-groups")
async def get_instance_groups(
    group_id: Optional[int] = Query(None, description="Omit to list all groups; provide to fetch one"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all instance groups, or fetch a single group's details including members."""
    def _to_response(g) -> InstanceGroupResponse:
        return InstanceGroupResponse(
            id=g.id,
            name=g.name,
            description=g.description,
            member_count=len(g.members),
            members=[m.provider_key for m in g.members],
            updated_at=g.updated_at,
            updated_by=g.updated_by,
        )

    if group_id is not None:
        group = await list_instance_groups(db, group_id=group_id)
        if group is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")
        return _to_response(group)

    groups = await list_instance_groups(db)
    return [_to_response(g) for g in groups]


@router.post("/instance-groups")
async def create_instance_group_endpoint(
    body: InstanceGroupCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new instance group."""
    from sqlalchemy.future import select as sa_select
    from app.auth.models import InstanceGroup as InstanceGroupTable

    existing = await db.execute(
        sa_select(InstanceGroupTable).where(InstanceGroupTable.name.ilike(body.name))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Instance group '{body.name}' already exists")

    group = await create_instance_group(db, body.name, body.description, current_admin.username)
    if body.rpm_default is not None or body.rpd_default is not None:
        await set_group_limit_for_all_profiles(
            db, group.id, body.rpm_default, body.rpd_default, current_admin.username, instance=True,
        )

    from app.rate_limit import rate_limit_tracker
    # No re-settle: an empty group folds no usage into its scope, so delta == 0.
    # See create_model_group_endpoint.

    return InstanceGroupResponse(
        id=group.id, name=group.name, description=group.description,
        member_count=0, members=[], updated_at=group.updated_at, updated_by=group.updated_by,
    )


@router.put("/instance-groups")
async def update_instance_group_endpoint(
    group_id: int = Query(..., description="Group ID to update"),
    body: InstanceGroupUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Rename or update description of an instance group."""
    from sqlalchemy.future import select as sa_select
    from app.auth.models import InstanceGroup as InstanceGroupTable

    fields = {}
    if body:
        update_data = body.model_dump(exclude_unset=True)
        if "name" in update_data:
            existing = await db.execute(
                sa_select(InstanceGroupTable).where(
                    InstanceGroupTable.name.ilike(update_data["name"]),
                    InstanceGroupTable.id != group_id,
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Instance group name '{update_data['name']}' already exists")
        fields = update_data

    group = await update_instance_group(db, group_id, fields, current_admin.username)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_instance_group(group_id)
    # No re-settle: InstanceGroupUpdate is name and description only.

    refreshed = await list_instance_groups(db, group_id=group_id)
    return InstanceGroupResponse(
        id=refreshed.id, name=refreshed.name, description=refreshed.description,
        member_count=len(refreshed.members), members=[m.provider_key for m in refreshed.members],
        updated_at=refreshed.updated_at, updated_by=refreshed.updated_by,
    )


@router.delete("/instance-groups")
async def delete_instance_group_endpoint(
    group_id: int = Query(..., description="Group ID to delete"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete an instance group (cascades members and per-user overrides)."""
    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_instance_group(group_id)

    deleted = await delete_instance_group(db, group_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)
    return {"message": f"Instance group {group_id} deleted"}


@router.put("/instance-groups/members")
async def set_instance_group_members_endpoint(
    group_id: int = Query(..., description="Group ID to update members for"),
    body: InstanceGroupMembersUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Replace the member list for an instance group. Validates each provider_key exists and is not in another group."""
    from app.auth.models import InstanceGroupMember as InstanceGroupMemberTable, ProviderCredentials
    from sqlalchemy.future import select as sa_select

    # Confirm group exists
    group = await list_instance_groups(db, group_id=group_id)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    provider_keys = body.provider_keys if body else []

    # Validate all provider_keys exist
    if provider_keys:
        result = await db.execute(sa_select(ProviderCredentials).where(ProviderCredentials.provider_key.in_(provider_keys)))
        found = {r.provider_key for r in result.scalars().all()}
        missing = [pk for pk in provider_keys if pk not in found]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"provider_key(s) not found: {missing}",
            )

    # Check for conflicts: provider_key already in another group
    if provider_keys:
        result = await db.execute(
            sa_select(InstanceGroupMemberTable).where(
                InstanceGroupMemberTable.provider_key.in_(provider_keys),
                InstanceGroupMemberTable.group_id != group_id,
            )
        )
        conflicts = [
            {"provider_key": r.provider_key, "current_group_id": r.group_id}
            for r in result.scalars().all()
        ]
        if conflicts:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"conflicts": conflicts})

    await set_instance_group_members(db, group_id, provider_keys)

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_instance_group(group_id)
    from app.routes.pools import resettle_pools_after_commit
    await resettle_pools_after_commit(db, all_pools=True)

    return {"group_id": group_id, "provider_keys": provider_keys}


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

        # Restore model enabled/disabled states
        for model_data in data.get("model_configurations", []):
            try:
                await create_or_update_model_configuration(
                    db,
                    model_id=model_data["model_id"],
                    provider_key=model_data["provider_key"],
                    model_name=model_data["model_name"],
                    is_enabled=model_data.get("is_enabled", True),
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

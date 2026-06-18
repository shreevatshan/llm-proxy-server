"""Admin routes for user management and system administration."""

from fastapi import APIRouter, Request, Depends, HTTPException, status, Response, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
import json
import urllib.parse
import logging
import io

from app.auth.database import (
    get_db, get_user_by_username, get_user_by_email, create_user, get_user_by_id, permanently_delete_user,
    get_all_provider_configurations, get_all_model_configurations, get_models_by_provider,
    create_or_update_provider_configuration, create_or_update_model_configuration, get_model_configuration,
    toggle_provider_configuration, toggle_model_configuration, bulk_toggle_all_models,
    search_models_and_providers, get_all_provider_credentials, get_provider_credentials,
    create_provider_credentials, update_provider_credentials, delete_provider_credentials,
    clear_all_model_configurations, admin_reset_user_password, get_usage_aggregates, get_usage_years,
    get_global_rate_limit, upsert_global_rate_limit,
    get_user_rate_limit, upsert_user_rate_limit, delete_user_rate_limit,
    list_model_groups, create_model_group, update_model_group, delete_model_group,
    set_group_members, get_model_group_limits, update_model_group_limits,
    list_user_group_rate_limits, upsert_user_group_rate_limit, delete_user_group_rate_limit,
    list_instance_groups, create_instance_group, update_instance_group, delete_instance_group,
    set_instance_group_members, get_instance_group_limits, update_instance_group_limits,
    list_user_instance_group_rate_limits, upsert_user_instance_group_rate_limit, delete_user_instance_group_rate_limit,
)
from app.auth.webhook import send_signup_webhook
from app.auth.middleware import get_current_admin
from app.auth.models import (
    User, UserCreate, UserResponse, ProviderConfigurationResponse, ModelConfigurationResponse,
    ModelManagementTree, ToggleRequest, BulkToggleRequest, ModelSearchResponse,
    ProviderCredentialsCreate, ProviderCredentialsUpdate, ProviderCredentialsResponse,
    AdminPasswordReset, VALID_AZURE_BACKENDS,
    GlobalRateLimitResponse, GlobalRateLimitUpdate,
    UserRateLimitResponse, UserRateLimitUpdate,
    ModelGroupCreate, ModelGroupUpdate, ModelGroupLimitsUpdate, ModelGroupMembersUpdate,
    ModelGroupResponse, UserModelGroupRateLimitResponse, UserModelGroupRateLimitUpdate,
    InstanceGroupCreate, InstanceGroupUpdate, InstanceGroupLimitsUpdate, InstanceGroupMembersUpdate,
    InstanceGroupResponse, UserInstanceGroupRateLimitResponse, UserInstanceGroupRateLimitUpdate,
)
from app.auth.admin import AdminUser, authenticate_admin, is_admin_enabled, get_admin_email
from app.auth.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.providers.provider_manager import provider_manager
from app.providers.auto_sync import auto_sync_on_provider_change, sync_provider_models
from app.providers.azure_deployments import merge_azure_deployments, normalize_azure_deployments, validate_deployment_names
from datetime import timedelta, datetime
from pydantic import BaseModel
import asyncio

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="app/frontend/templates")


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
        secure=False,  # Set to True in production with HTTPS
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
        
        # Build provider response with model counts
        provider_responses = []
        for provider in provider_credentials:
            provider_models = [m for m in models if m.provider_key == provider.provider_key]
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
        print("Forcing complete provider manager refresh before sync...")
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
                print(f"Warning: Cannot determine provider for model {model.id}")
                continue
            
            # Only include models from providers that exist in database
            if provider_key in valid_provider_keys:
                valid_models.append(model)
                synced_model_ids.append(model.id)
            else:
                print(f"Debug: Skipping model {model.id} from provider {provider_key} (not in database)")
        
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
                print(f"Warning: Cannot determine provider for model {model.id}, skipping")
                continue
            
            # Track which provider this model belongs to
            synced_providers.add(provider_key)
            
            # Check if model already exists to preserve its enabled state
            existing_model = await get_model_configuration(db, model.id)
            is_enabled = existing_model.is_enabled if existing_model else True
            
            await create_or_update_model_configuration(
                db, model.id, provider_key, model_name, is_enabled
            )
            synced_models += 1
        
        # Identify stale models (models in DB but not synced)
        from app.auth.database import identify_stale_models
        stale_models = await identify_stale_models(db, synced_model_ids)
        
        # Determine the appropriate message
        if len(valid_provider_keys) == 0:
            message = "No providers found in database"
        elif len(synced_providers) == 0:
            message = f"No models found from {len(valid_provider_keys)} providers in database"
        else:
            message = f"Synced {synced_models} models from {len(synced_providers)} providers"
        
        print(f"Sync completed: {message}")
        print(f"Total providers in DB: {len(valid_provider_keys)}")
        print(f"Providers with models: {len(synced_providers)}")
        print(f"Models synced: {synced_models}")
        print(f"Stale models found: {len(stale_models)}")
        
        # Update the cache with all synced models
        provider_manager.model_cache.update_models(valid_models)
        print(f"Cache updated with {len(valid_models)} models")
        
        return {
            "message": message,
            "providers_synced": len(synced_providers),
            "models_synced": synced_models,
            "total_providers_in_db": len(valid_provider_keys),
            "stale_models": stale_models,
            "stale_count": len(stale_models)
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
            # Let's also check what models are actually in the database with similar names
            all_models = await get_all_model_configurations(db)
            similar_models = [m.model_id for m in all_models if 'gemini' in m.model_id.lower()][:5]
            logging.warning(f"Model not found: {model_id}. Similar models in DB: {similar_models}")
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
    
    import time
    # Create response with security headers to prevent caching
    response = templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "admin": current_admin,
            "users": users,
            "admin_email": get_admin_email(),
            "title": "Admin Dashboard - LLM Proxy Server",
            "cache_version": str(int(time.time()))
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
    user_data: UserCreate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user (admin only)."""
    # Check if username already exists
    existing_user = await get_user_by_username(db, user_data.username)
    if existing_user:
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
    
    # Create user
    try:
        user = await create_user(db, user_data.username, user_data.email, user_data.password)

        # Send signup webhook notification
        await send_signup_webhook(
            username=user.username,
            email=user.email,
            signup_mode="admin_created",
            user_id=user.id,
            is_pending=False
        )

        return UserResponse.from_orm(user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


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
        return {"message": f"User {user.username} has been deactivated"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


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
        return {"message": f"User {user.username} has been activated"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate user: {str(e)}"
        )


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
        return {"message": f"User {user.username} has been approved and activated"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve user: {str(e)}"
        )


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
        success = await permanently_delete_user(db, user_id)
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


# ==================== Rate Limit Management API Endpoints ====================

@router.get("/rate-limits/defaults", response_model=GlobalRateLimitResponse)
async def get_rate_limit_defaults(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get global rate limit defaults."""
    row = await get_global_rate_limit(db)
    if row is None:
        return GlobalRateLimitResponse()
    return GlobalRateLimitResponse.model_validate(row)


@router.put("/rate-limits/defaults", response_model=GlobalRateLimitResponse)
async def update_rate_limit_defaults(
    body: GlobalRateLimitUpdate,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update global rate limit defaults."""
    row = await upsert_global_rate_limit(db, body.rpm_default, body.rpd_default, current_admin.username)
    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_defaults()
    await rate_limit_tracker.refresh_now()
    return GlobalRateLimitResponse.model_validate(row)


@router.get("/rate-limits/users")
async def get_rate_limit_users(
    user_id: Optional[int] = Query(None, description="User ID to get limits for (omit for all users)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get rate limit settings for one user (user_id provided) or all users."""
    from app.rate_limit import rate_limit_tracker
    from sqlalchemy.future import select as sa_select

    global_row = await get_global_rate_limit(db)
    rpm_default = global_row.rpm_default if global_row else None
    rpd_default = global_row.rpd_default if global_row else None

    async def _build_response(user: User) -> UserRateLimitResponse:
        override = await get_user_rate_limit(db, user.id)
        effective_rpm = (override.rpm_limit if override and override.rpm_limit is not None else rpm_default)
        effective_rpd = (override.rpd_limit if override and override.rpd_limit is not None else rpd_default)
        status_obj = await rate_limit_tracker.get_user_status(user.id, user.username)
        return UserRateLimitResponse(
            user_id=user.id,
            username=user.username,
            email=user.email,
            rpm_limit=override.rpm_limit if override else None,
            rpd_limit=override.rpd_limit if override else None,
            effective_rpm=effective_rpm,
            effective_rpd=effective_rpd,
            current_rpm_count=status_obj.rpm_count,
            current_rpd_count=status_obj.rpd_count,
        )

    if user_id is not None:
        user = await get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        return await _build_response(user)

    result = await db.execute(sa_select(User))
    users = result.scalars().all()
    return list(await asyncio.gather(*[_build_response(u) for u in users]))


@router.put("/rate-limits/users", response_model=UserRateLimitResponse)
async def update_user_rate_limit(
    body: UserRateLimitUpdate,
    user_id: int = Query(..., description="User ID to update rate limit for"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set or update per-user rate limit override."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    override = await upsert_user_rate_limit(
        db, user_id,
        rpm=body.rpm_limit,
        rpd=body.rpd_limit,
        admin_username=current_admin.username,
        fields_set=body.model_fields_set,
    )

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user(user_id)
    await rate_limit_tracker.refresh_now()

    global_row = await get_global_rate_limit(db)
    rpm_default = global_row.rpm_default if global_row else None
    rpd_default = global_row.rpd_default if global_row else None
    effective_rpm = override.rpm_limit if override.rpm_limit is not None else rpm_default
    effective_rpd = override.rpd_limit if override.rpd_limit is not None else rpd_default

    return UserRateLimitResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        rpm_limit=override.rpm_limit,
        rpd_limit=override.rpd_limit,
        effective_rpm=effective_rpm,
        effective_rpd=effective_rpd,
    )


@router.delete("/rate-limits/users")
async def delete_user_rate_limit_override(
    user_id: int = Query(..., description="User ID to remove rate limit override for"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove per-user rate limit override; user falls back to global defaults."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await delete_user_rate_limit(db, user_id)

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user(user_id)
    await rate_limit_tracker.refresh_now()

    return {"message": f"Rate limit override removed for user {user.username}"}


@router.get("/models/all", response_model=List[ModelConfigurationResponse])
async def get_all_models(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return all model configurations as a flat list."""
    models = await get_all_model_configurations(db)
    return [ModelConfigurationResponse.from_orm(m) for m in models]


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
            rpm_default=g.rpm_default,
            rpd_default=g.rpd_default,
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

    group = await create_model_group(db, body.name, body.description, body.rpm_default, body.rpd_default, current_admin.username)

    from app.rate_limit import rate_limit_tracker
    await rate_limit_tracker.refresh_now()

    return ModelGroupResponse(
        id=group.id, name=group.name, description=group.description,
        rpm_default=group.rpm_default, rpd_default=group.rpd_default,
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
    await rate_limit_tracker.refresh_now()

    refreshed = await list_model_groups(db, group_id=group_id)
    return ModelGroupResponse(
        id=refreshed.id, name=refreshed.name, description=refreshed.description,
        rpm_default=refreshed.rpm_default, rpd_default=refreshed.rpd_default,
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

    await rate_limit_tracker.refresh_now()
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
    await rate_limit_tracker.refresh_now()

    return {"group_id": group_id, "model_ids": model_ids}


@router.get("/model-groups/limits")
async def get_model_group_limits_endpoint(
    group_id: int = Query(..., description="Group ID"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get the RPM/RPD defaults for a model group."""
    row = await get_model_group_limits(db, group_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")
    return {"group_id": group_id, "rpm_default": row.rpm_default, "rpd_default": row.rpd_default}


@router.put("/model-groups/limits")
async def update_model_group_limits_endpoint(
    group_id: int = Query(..., description="Group ID"),
    body: ModelGroupLimitsUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update the RPM/RPD defaults for a model group."""
    rpm = body.rpm_default if body else None
    rpd = body.rpd_default if body else None

    group = await update_model_group_limits(db, group_id, rpm, rpd, current_admin.username)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_group(group_id)
    await rate_limit_tracker.refresh_now()

    return {"group_id": group_id, "rpm_default": group.rpm_default, "rpd_default": group.rpd_default}


@router.get("/model-groups/users")
async def get_model_group_user_overrides(
    group_id: int = Query(..., description="Group ID"),
    user_id: Optional[int] = Query(None, description="User ID (omit for all users in this group)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List per-user overrides for a model group."""
    group_row = await get_model_group_limits(db, group_id)
    if group_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    overrides = await list_user_group_rate_limits(db, group_id, user_id=user_id)


    async def _build(ov):
        user = await get_user_by_id(db, ov.user_id)
        if not user:
            return None
        effective_rpm = ov.rpm_limit if ov.rpm_limit is not None else group_row.rpm_default
        effective_rpd = ov.rpd_limit if ov.rpd_limit is not None else group_row.rpd_default
        return UserModelGroupRateLimitResponse(
            user_id=user.id, username=user.username, email=user.email,
            group_id=group_id,
            rpm_limit=ov.rpm_limit, rpd_limit=ov.rpd_limit,
            effective_rpm=effective_rpm, effective_rpd=effective_rpd,
        )

    import asyncio as _asyncio
    results = await _asyncio.gather(*[_build(ov) for ov in overrides])
    return [r for r in results if r is not None]


@router.put("/model-groups/users")
async def upsert_model_group_user_override(
    group_id: int = Query(..., description="Group ID"),
    user_id: int = Query(..., description="User ID"),
    body: UserModelGroupRateLimitUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set or update a per-user rate limit override for a model group."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    group_row = await get_model_group_limits(db, group_id)
    if group_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model group not found")

    rpm = body.rpm_limit if body else None
    rpd = body.rpd_limit if body else None
    fields_set = body.model_fields_set if body else set()

    override = await upsert_user_group_rate_limit(
        db, user_id, group_id, rpm, rpd, current_admin.username, fields_set
    )

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user_group(user_id, group_id)
    await rate_limit_tracker.refresh_now()

    effective_rpm = override.rpm_limit if override.rpm_limit is not None else group_row.rpm_default
    effective_rpd = override.rpd_limit if override.rpd_limit is not None else group_row.rpd_default

    return UserModelGroupRateLimitResponse(
        user_id=user.id, username=user.username, email=user.email,
        group_id=group_id,
        rpm_limit=override.rpm_limit, rpd_limit=override.rpd_limit,
        effective_rpm=effective_rpm, effective_rpd=effective_rpd,
    )


@router.delete("/model-groups/users")
async def delete_model_group_user_override(
    group_id: int = Query(..., description="Group ID"),
    user_id: int = Query(..., description="User ID"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove per-user override; user reverts to group default."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    deleted = await delete_user_group_rate_limit(db, user_id, group_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No override found for this user and group")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user_group(user_id, group_id)
    await rate_limit_tracker.refresh_now()

    return {"message": f"Override removed for user {user.username} on group {group_id}"}


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
            rpm_default=g.rpm_default,
            rpd_default=g.rpd_default,
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

    group = await create_instance_group(db, body.name, body.description, body.rpm_default, body.rpd_default, current_admin.username)

    from app.rate_limit import rate_limit_tracker
    await rate_limit_tracker.refresh_now()

    return InstanceGroupResponse(
        id=group.id, name=group.name, description=group.description,
        rpm_default=group.rpm_default, rpd_default=group.rpd_default,
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
    await rate_limit_tracker.refresh_now()

    refreshed = await list_instance_groups(db, group_id=group_id)
    return InstanceGroupResponse(
        id=refreshed.id, name=refreshed.name, description=refreshed.description,
        rpm_default=refreshed.rpm_default, rpd_default=refreshed.rpd_default,
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

    await rate_limit_tracker.refresh_now()
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
    await rate_limit_tracker.refresh_now()

    return {"group_id": group_id, "provider_keys": provider_keys}


@router.get("/instance-groups/limits")
async def get_instance_group_limits_endpoint(
    group_id: int = Query(..., description="Group ID"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get the RPM/RPD defaults for an instance group."""
    row = await get_instance_group_limits(db, group_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")
    return {"group_id": group_id, "rpm_default": row.rpm_default, "rpd_default": row.rpd_default}


@router.put("/instance-groups/limits")
async def update_instance_group_limits_endpoint(
    group_id: int = Query(..., description="Group ID"),
    body: InstanceGroupLimitsUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update the RPM/RPD defaults for an instance group."""
    rpm = body.rpm_default if body else None
    rpd = body.rpd_default if body else None

    group = await update_instance_group_limits(db, group_id, rpm, rpd, current_admin.username)
    if group is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_instance_group(group_id)
    await rate_limit_tracker.refresh_now()

    return {"group_id": group_id, "rpm_default": group.rpm_default, "rpd_default": group.rpd_default}


@router.get("/instance-groups/users")
async def get_instance_group_user_overrides(
    group_id: int = Query(..., description="Group ID"),
    user_id: Optional[int] = Query(None, description="User ID (omit for all users in this group)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """List per-user overrides for an instance group."""
    group_row = await get_instance_group_limits(db, group_id)
    if group_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    overrides = await list_user_instance_group_rate_limits(db, group_id, user_id=user_id)

    async def _build(ov):
        user = await get_user_by_id(db, ov.user_id)
        if not user:
            return None
        effective_rpm = ov.rpm_limit if ov.rpm_limit is not None else group_row.rpm_default
        effective_rpd = ov.rpd_limit if ov.rpd_limit is not None else group_row.rpd_default
        return UserInstanceGroupRateLimitResponse(
            user_id=user.id, username=user.username, email=user.email,
            group_id=group_id,
            rpm_limit=ov.rpm_limit, rpd_limit=ov.rpd_limit,
            effective_rpm=effective_rpm, effective_rpd=effective_rpd,
        )

    import asyncio as _asyncio
    results = await _asyncio.gather(*[_build(ov) for ov in overrides])
    return [r for r in results if r is not None]


@router.put("/instance-groups/users")
async def upsert_instance_group_user_override(
    group_id: int = Query(..., description="Group ID"),
    user_id: int = Query(..., description="User ID"),
    body: UserInstanceGroupRateLimitUpdate = None,
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set or update a per-user rate limit override for an instance group."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    group_row = await get_instance_group_limits(db, group_id)
    if group_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instance group not found")

    rpm = body.rpm_limit if body else None
    rpd = body.rpd_limit if body else None
    fields_set = body.model_fields_set if body else set()

    override = await upsert_user_instance_group_rate_limit(
        db, user_id, group_id, rpm, rpd, current_admin.username, fields_set
    )

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user_instance_group(user_id, group_id)
    await rate_limit_tracker.refresh_now()

    effective_rpm = override.rpm_limit if override.rpm_limit is not None else group_row.rpm_default
    effective_rpd = override.rpd_limit if override.rpd_limit is not None else group_row.rpd_default

    return UserInstanceGroupRateLimitResponse(
        user_id=user.id, username=user.username, email=user.email,
        group_id=group_id,
        rpm_limit=override.rpm_limit, rpd_limit=override.rpd_limit,
        effective_rpm=effective_rpm, effective_rpd=effective_rpd,
    )


@router.delete("/instance-groups/users")
async def delete_instance_group_user_override(
    group_id: int = Query(..., description="Group ID"),
    user_id: int = Query(..., description="User ID"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Remove per-user override; user reverts to instance-group default."""
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    deleted = await delete_user_instance_group_rate_limit(db, user_id, group_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No override found for this user and group")

    from app.rate_limit import rate_limit_tracker
    rate_limit_tracker.invalidate_user_instance_group(user_id, group_id)
    await rate_limit_tracker.refresh_now()

    return {"message": f"Override removed for user {user.username} on instance group {group_id}"}


@router.post("/logout")
async def admin_logout():
    """Admin logout endpoint."""
    from fastapi.responses import RedirectResponse
    redirect_response = RedirectResponse(url="/admin/", status_code=status.HTTP_302_FOUND)
    
    # Delete the cookie with the same parameters used when setting it
    redirect_response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax"
    )
    
    return redirect_response


# Provider Credentials Management API Endpoints

@router.get("/providers", response_model=List[ProviderConfigurationResponse])
async def list_provider_credentials(
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db)
):
    """List all provider credentials with model counts."""
    try:
        logging.info("🔍 Fetching all provider credentials...")
        credentials = await get_all_provider_credentials(db)
        logging.info(f"📋 Found {len(credentials)} provider credentials")
        
        models = await get_all_model_configurations(db)
        logging.info(f"📋 Found {len(models)} model configurations")
        
        response_list = []
        
        for cred in credentials:
            # Calculate model counts for this provider
            provider_models = [m for m in models if m.provider_key == cred.provider_key]
            enabled_models = [m for m in provider_models if m.is_enabled]
            
            logging.info(f"Provider {cred.provider_key}: {len(provider_models)} models, {len(enabled_models)} enabled")
            
            # Parse deployments JSON if present
            deployments = None
            if cred.deployments_json:
                try:
                    deployments = json.loads(cred.deployments_json)
                except json.JSONDecodeError:
                    deployments = None
            
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
                created_at=cred.created_at,
                updated_at=cred.updated_at
            )
            response_list.append(response)
        
        logging.info(f"📤 Returning {len(response_list)} providers to frontend")
        return response_list
    except Exception as e:
        logging.error(f"❌ Failed to list provider credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list provider credentials: {str(e)}"
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
            api_key=credentials.api_key,
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=credentials.secret_access_key,
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            provider_name=credentials.provider_name,
            dynamic_discovery=dynamic_discovery,
            supported_apis=supported_apis,
            created_at=credentials.created_at,
            updated_at=credentials.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider credentials: {str(e)}"
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
        logging.info(f"🚀 Creating provider with data: {provider_data}, upsert={upsert}")
        
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

                # Convert create data to update data
                update_data = {}
                for field, value in provider_data.dict(exclude_unset=True).items():
                    if value is not None and field != "provider_type":  # Don't update provider_type
                        update_data[field] = value

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
                dynamic_discovery=provider_data.dynamic_discovery,
                supported_apis=supported_apis_json
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
            api_key=credentials.api_key,
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=credentials.secret_access_key,
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            dynamic_discovery=credentials.dynamic_discovery,
            supported_apis=supported_apis,
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
        logging.error(f"Unexpected error creating provider credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create provider credentials: {str(e)}"
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
        
        # Prepare update data (only include non-None values)
        update_data = {}
        for field, value in provider_data.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        # Serialize supported_apis list to JSON string for storage
        if 'supported_apis' in update_data and isinstance(update_data['supported_apis'], list):
            update_data['supported_apis'] = json.dumps(update_data['supported_apis'])
        
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
            api_key=credentials.api_key,
            discovery_api_version=credentials.discovery_api_version,
            azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
            region=credentials.region,
            access_key_id=credentials.access_key_id,
            secret_access_key=credentials.secret_access_key,
            base_url=credentials.base_url,
            deployments=deployments,
            openai_deployments=deployment_groups.get("openai", []),
            anthropic_deployments=deployment_groups.get("anthropic", []),
            dynamic_discovery=credentials.dynamic_discovery,
            supported_apis=supported_apis,
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider credentials: {str(e)}"
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete provider credentials: {str(e)}"
        )


@router.get("/providers/export")
async def export_provider_config(
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
                "api_key": cred.api_key,
                "discovery_api_version": cred.discovery_api_version,
                "azure_backend": cred.azure_backend or ("openai" if cred.provider_type == "azure" else None),
                "region": cred.region,
                "access_key_id": cred.access_key_id,
                "secret_access_key": cred.secret_access_key,
                "base_url": cred.base_url,
                "deployments": deployments,
                "openai_deployments": deployment_groups.get("openai", []),
                "anthropic_deployments": deployment_groups.get("anthropic", []),
                "dynamic_discovery": cred.dynamic_discovery,
                "supported_apis": supported_apis,
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
        logging.error(f"Failed to export provider config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export provider config: {str(e)}"
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

                    if provider_data.get("dynamic_discovery") is True:
                        has_deps = (
                            provider_data.get("deployments")
                            or provider_data.get("openai_deployments")
                            or provider_data.get("anthropic_deployments")
                        )
                        if has_deps:
                            errors.append(f"Provider '{provider_key}': cannot specify deployments with dynamic_discovery enabled")
                            continue

                    try:
                        validate_deployment_names(provider_data.get("openai_deployments"))
                        validate_deployment_names(provider_data.get("anthropic_deployments"))
                        validate_deployment_names(provider_data.get("deployments"))
                    except ValueError as e:
                        errors.append(f"Provider '{provider_key}': {str(e)}")
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

                if existing and overwrite:
                    # Delete existing provider first so we can re-create cleanly
                    await delete_provider_credentials(db, provider_key)

                credentials = await create_provider_credentials(
                    db=db,
                    provider_type=provider_data["provider_type"],
                    instance_name=provider_data["instance_name"],
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
                    dynamic_discovery=provider_data.get("dynamic_discovery"),
                    supported_apis=supported_apis_json,
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
        logging.error(f"Failed to import provider config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import provider config: {str(e)}"
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
    view: Optional[str] = Query(None, description="'user' or 'model' for drill-down"),
    id: Optional[str] = Query(None, description="Identity value to drill into"),
    window: str = Query("30d", description="Time window: 24h | today | yesterday | 7d | 30d | month | all"),
    year: Optional[int] = Query(None, description="Year (required when window=month)"),
    month: Optional[int] = Query(None, description="Month 1-12 (required when window=month)"),
    current_admin: AdminUser = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated usage for the requested time window.

    - No view/id: top-level per_user + per_model + totals.
    - ?view=user&id=alice: breakdown by model for alice.
    - ?view=model&id=gpt-4: breakdown by user for gpt-4.
    - window=24h|today|yesterday|7d|30d|month
    - window=month requires year and month params.
    """
    from app.request_tracker import request_tracker
    from app.auth.database import get_usage_earliest_date
    await request_tracker.flush_pending()

    filter_user: Optional[str] = None
    filter_model: Optional[str] = None

    if view == "user" and id is not None:
        filter_user = id
    elif view == "model" and id is not None:
        filter_model = id

    result = await get_usage_aggregates(
        db,
        group_by="user",
        filter_user=filter_user,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["earliest_date"] = await get_usage_earliest_date(db)
    return result

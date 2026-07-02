"""Dashboard routes for web interface."""

import json
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict

from app.auth.database import get_db
from app.auth.middleware import get_current_active_user, get_current_user_or_admin, get_current_user_optional
from app.auth.models import User
from app.auth.admin import AdminUser
from typing import Union

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
templates = Jinja2Templates(directory="app/frontend/templates")


def _extract_endpoints_from_routers(routers, prefix: str = "", tag_suffix: str = "") -> List[Dict[str, str]]:
    """Extract endpoint metadata from FastAPI router objects.

    Introspects the actual registered routes so the dashboard endpoint
    listing stays in sync automatically when new routes are added.

    Args:
        routers: Iterable of APIRouter (or modules with a .router attribute).
        prefix: Path prefix to prepend (e.g. "/openai" for v1 Azure routes).
        tag_suffix: Suffix to append to the description (e.g. " (v1)").

    Returns:
        List of {method, path, desc} dicts sorted by path then method.
    """
    endpoints: List[Dict[str, str]] = []
    seen = set()  # deduplicate (method, path) pairs
    for router_or_mod in routers:
        router = getattr(router_or_mod, "router", router_or_mod)
        for route in router.routes:
            if not hasattr(route, "methods"):
                continue
            for method in sorted(route.methods):
                if method in ("HEAD", "OPTIONS"):
                    continue
                path = prefix + route.path
                key = (method, path)
                if key in seen:
                    continue
                seen.add(key)
                # Derive a human-readable description from route metadata
                summary = getattr(route, "summary", "") or ""
                name = getattr(route, "name", "") or ""
                desc = summary or name.replace("_", " ").title()
                if tag_suffix:
                    desc = f"{desc} {tag_suffix}".strip()
                endpoints.append({"method": method, "path": path, "desc": desc})
    endpoints.sort(key=lambda e: (e["path"], e["method"]))
    return endpoints


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    current_user_or_admin: Union[User, AdminUser] = Depends(get_current_user_or_admin)
):
    """Dashboard home page (requires authentication). Redirects admin users to admin dashboard."""
    # If user is admin, redirect to admin dashboard
    if isinstance(current_user_or_admin, AdminUser):
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)

    import time
    from app.config import config
    from app.providers.provider_manager import provider_manager

    # Extract endpoints and models for the user dashboard tabs
    from app.routes import (
        models, chat, completions, embeddings, images, audio, responses,
        anthropic_messages, anthropic_models,
        azure_openai,
    )
    openai_routers = [models, chat, completions, embeddings, images, audio, responses]
    anthropic_routers = [anthropic_messages, anthropic_models]
    azure_legacy_routers = [azure_openai]

    openai_endpoints = _extract_endpoints_from_routers(openai_routers)
    anthropic_endpoints = _extract_endpoints_from_routers(anthropic_routers)
    azure_endpoints = (
        _extract_endpoints_from_routers(azure_legacy_routers)
        + _extract_endpoints_from_routers(openai_routers, prefix="/openai")
    )

    # Strip /v1 prefix from OpenAI endpoint paths
    for ep in openai_endpoints:
        if ep["path"].startswith("/v1"):
            ep["path"] = ep["path"][3:]

    # Get enabled models with supported_apis for badge rendering
    enabled_models = provider_manager.model_cache.get_enabled_models()
    models_list = []
    for model in enabled_models:
        provider = provider_manager.providers.get(model.provider)
        supported_apis = provider.get_supported_apis_for_model(model.id) if provider else ["openai"]
        models_list.append({
            "id": model.id,
            "object": model.object,
            "created": model.created,
            "owned_by": model.owned_by,
            "supported_apis": supported_apis,
        })

    # Regular user - show normal dashboard
    return templates.TemplateResponse(
        "dashboard/authenticated.html",
        {
            "request": request,
            "user": current_user_or_admin,
            "title": "LLM Proxy Server Dashboard",
            "cache_version": str(int(time.time())),
            "domain": config.server.domain,
            "openai_port": config.server.openai_port,
            "anthropic_port": config.server.anthropic_port,
            "azure_openai_port": config.server.azure_openai_port,
            "management_port": config.server.management_port,
            "openai_endpoints_json": json.dumps(openai_endpoints),
            "anthropic_endpoints_json": json.dumps(anthropic_endpoints),
            "azure_endpoints_json": json.dumps(azure_endpoints),
            "models_json": json.dumps(models_list),
        }
    )


@router.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    current_user_or_admin: Union[User, AdminUser] = Depends(get_current_user_or_admin)
):
    """User profile page (requires authentication). Redirects admin users to admin dashboard."""
    # If user is admin, redirect to admin dashboard
    if isinstance(current_user_or_admin, AdminUser):
        return RedirectResponse(url="/admin/dashboard", status_code=status.HTTP_302_FOUND)
    
    import time
    # Regular user - show profile page
    return templates.TemplateResponse(
        "dashboard/profile.html",
        {
            "request": request,
            "user": current_user_or_admin,
            "title": "Profile - LLM Proxy Server",
            "cache_version": str(int(time.time()))
        }
    )


@router.get("/endpoints", response_class=HTMLResponse)
async def endpoints_page(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """User endpoints page - view available OpenAI and Anthropic API endpoints."""
    from app.config import config
    import time

    # Auto-generate endpoint lists by introspecting the actual routers.
    # This ensures the dashboard always reflects the real registered routes.
    from app.routes import (
        models, chat, completions, embeddings, images, audio, responses,
        anthropic_messages, anthropic_models,
        azure_openai,
    )
    openai_routers = [models, chat, completions, embeddings, images, audio, responses]
    anthropic_routers = [anthropic_messages, anthropic_models]
    azure_legacy_routers = [azure_openai]

    openai_endpoints = _extract_endpoints_from_routers(openai_routers)
    anthropic_endpoints = _extract_endpoints_from_routers(anthropic_routers)
    # Azure = legacy deployment-based routes + v1 (same OpenAI routers under /openai prefix)
    azure_endpoints = (
        _extract_endpoints_from_routers(azure_legacy_routers)
        + _extract_endpoints_from_routers(openai_routers, prefix="/openai")
    )

    # Strip /v1 prefix from OpenAI endpoint paths — the base URL already
    # includes /v1, so paths shown to the user should be relative (e.g.
    # "/chat/completions" not "/v1/chat/completions").
    for ep in openai_endpoints:
        if ep["path"].startswith("/v1"):
            ep["path"] = ep["path"][3:]  # remove leading /v1

    response = templates.TemplateResponse(
        "dashboard/endpoints.html",
        {
            "request": request,
            "user": current_user,
            "title": "Available Endpoints - LLM Proxy Server",
            "cache_version": str(int(time.time())),
            "domain": config.server.domain,
            "openai_port": config.server.openai_port,
            "anthropic_port": config.server.anthropic_port,
            "azure_openai_port": config.server.azure_openai_port,
            "management_port": config.server.management_port,
            "openai_endpoints_json": json.dumps(openai_endpoints),
            "anthropic_endpoints_json": json.dumps(anthropic_endpoints),
            "azure_openai_endpoints_json": json.dumps(azure_endpoints),
        }
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.get("/models", response_class=HTMLResponse)
async def models_search_page(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """User models search page - view and copy available model names."""
    import time
    # Create response with security headers to prevent caching
    response = templates.TemplateResponse(
        "dashboard/models_search.html",
        {
            "request": request,
            "user": current_user,
            "title": "Available Models - LLM Proxy Server",
            "cache_version": str(int(time.time()))
        }
    )
    
    # Add security headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response


@router.get("/api/models")
async def list_models_for_dashboard(
    current_user: Union[User, AdminUser] = Depends(get_current_user_or_admin)
):
    """List all enabled models with supported API info for the dashboard."""
    from app.providers.provider_manager import provider_manager

    models = provider_manager.model_cache.get_enabled_models()
    result = []
    for model in models:
        provider = provider_manager.providers.get(model.provider)
        supported_apis = provider.get_supported_apis_for_model(model.id) if provider else ["openai"]
        result.append({
            "id": model.id,
            "object": model.object,
            "created": model.created,
            "owned_by": model.owned_by,
            "supported_apis": supported_apis,
        })
    return {"data": result}

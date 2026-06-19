"""
LLM Proxy Server - Multi-Server Application Factory

Creates four FastAPI applications sharing the same provider_manager,
model_cache, database, and auth system:

1. OpenAI API Server (port 11440) - OpenAI-compatible API endpoints
2. Anthropic API Server (port 2027) - Anthropic Messages API endpoints
3. Azure OpenAI API Server (port 11439) - Azure OpenAI-compatible API endpoints
4. Management Server (port 8765) - Admin panel, user login, dashboard
"""

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from typing import Optional, Union
import asyncio
import json
import logging
import os
import time
import uuid

from app.config import config
from app.auth.database import init_database
from app.auth.middleware import get_current_user_optional
from app.auth.models import User
from app.auth.admin import AdminUser
from app.tracing import (
    init_tracing,
    instrument_database,
    create_span,
    add_span_attributes,
    set_span_error
)

logger = logging.getLogger(__name__)

# ==================== Shared State ====================

_startup_lock = asyncio.Lock()
_startup_complete = False


async def shared_startup():
    """Initialize shared resources used by all three servers.

    Protected by _startup_lock because asyncio.gather starts all three
    server lifespans concurrently — without the lock, multiple coroutines
    could pass the _startup_complete check before the first one finishes.
    """
    global _startup_complete
    async with _startup_lock:
        if _startup_complete:
            return

        with create_span("app.startup") as startup_span:
            from app.auth.database import engine, AsyncSessionLocal
            instrument_database(engine)

            with create_span("app.startup.init_database") as db_span:
                await init_database()
                add_span_attributes(db_span, {
                    "database.type": "sqlite",
                    "database.location": "data/llm_proxy.db"
                })

            with create_span("app.startup.init_auth_cache") as auth_cache_span:
                from app.auth.cache import auth_cache
                auth_cache.set_db_session_factory(AsyncSessionLocal)
                await auth_cache.start()
                add_span_attributes(auth_cache_span, {
                    "auth_cache.flush_interval": auth_cache.FLUSH_INTERVAL,
                    "auth_cache.api_key_ttl": auth_cache.API_KEY_TTL,
                    "auth_cache.user_ttl": auth_cache.USER_TTL
                })

            with create_span("app.startup.init_transformation_manager"):
                from app.transformation import initialize_transformation_manager
                initialize_transformation_manager(config.transformation)

            from app.providers.provider_manager import provider_manager

            with create_span("app.startup.init_provider_manager") as span:
                await provider_manager.initialize()
                add_span_attributes(span, {
                    "provider.count": len(provider_manager.providers)
                })

            with create_span("app.startup.init_model_cache_refresh") as mc_span:
                await provider_manager.model_cache.start()
                add_span_attributes(mc_span, {
                    "model_cache.refresh_interval": provider_manager.model_cache.REFRESH_INTERVAL
                })

            with create_span("app.startup.init_request_tracker"):
                from app.request_tracker import request_tracker
                await request_tracker.start()

            with create_span("app.startup.init_rate_limit_tracker"):
                from app.rate_limit import rate_limit_tracker
                rate_limit_tracker.set_db_session_factory(AsyncSessionLocal)
                await rate_limit_tracker.start()

        _startup_complete = True


async def shared_shutdown():
    """Cleanup shared resources."""
    print("Shutting down application...")

    SHUTDOWN_TIMEOUT_SECONDS = int(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", "10"))

    from app.providers.provider_manager import provider_manager
    from app.auth.cache import auth_cache
    from app.request_tracker import request_tracker
    from app.rate_limit import rate_limit_tracker

    # Phase 1: Stop background tasks and caches (may be using provider clients)
    phase1_tasks = [
        asyncio.create_task(provider_manager.model_cache.stop()),
        asyncio.create_task(auth_cache.stop()),
        asyncio.create_task(provider_manager.cleanup_background_tasks()),
        asyncio.create_task(request_tracker.stop()),
        asyncio.create_task(rate_limit_tracker.stop()),
    ]

    try:
        print(f"Waiting for {len(phase1_tasks)} shutdown tasks (timeout: {SHUTDOWN_TIMEOUT_SECONDS}s)...")
        await asyncio.wait_for(
            asyncio.gather(*phase1_tasks, return_exceptions=True),
            timeout=SHUTDOWN_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        print(f"Warning: Shutdown timeout ({SHUTDOWN_TIMEOUT_SECONDS}s) exceeded, forcing shutdown")
        for task in phase1_tasks:
            if not task.done():
                task.cancel()
    except Exception as e:
        print(f"Error during shutdown phase 1: {e}")

    # Phase 2: Close provider HTTP clients (safe now that nothing is using them)
    try:
        await provider_manager.close_provider_clients()
    except Exception as e:
        print(f"Error closing provider clients: {e}")

    # Phase 3: Dispose the SQLAlchemy engines. aiosqlite runs each connection on
    # a non-daemon worker thread that only terminates when the connection is
    # closed; the async engine's pool keeps it open, so without disposal that
    # thread blocks interpreter shutdown. Safe here because every DB user
    # (request_tracker, auth_cache, rate_limit_tracker, model_cache) was stopped
    # in Phase 1 and provider clients closed in Phase 2.
    try:
        from app.auth.database import engine, sync_engine
        await engine.dispose()
        sync_engine.dispose()
    except Exception as e:
        print(f"Error disposing DB engines: {e}")

    print("Application shutdown complete")


def _add_cors(app: FastAPI):
    """Add standard CORS middleware to an app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _instrument_app(app: FastAPI, name: str):
    """Instrument a FastAPI app with OpenTelemetry."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    try:
        FastAPIInstrumentor.instrument_app(app)
        print(f"✓ FastAPI instrumentation enabled for {name}")
    except Exception as e:
        print(f"⚠ FastAPI instrumentation failed for {name}: {e}")


_TRACKED_PREFIXES = ("/v1/", "/openai/deployments/")
_EXCLUDED_PATHS = ("/v1/messages/count_tokens",)


def _add_request_tracking(app: FastAPI, server_name: str):
    """Add request tracking middleware to an API server."""
    from app.request_tracker import request_tracker

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        path = request.url.path
        if not any(path.startswith(p) for p in _TRACKED_PREFIXES):
            return await call_next(request)
        if any(path == ep for ep in _EXCLUDED_PATHS):
            return await call_next(request)

        request_id = uuid.uuid4().hex

        model = None
        is_streaming = False
        if request.method == "POST":
            try:
                body_bytes = await request.body()
                body_json = json.loads(body_bytes)
                model = body_json.get("model")
                is_streaming = bool(body_json.get("stream", False))
            except Exception:
                body_bytes = b""

            # For Azure-style deployment paths the deployment name lives in the URL,
            # not the body (the Azure SDK omits the "model" field).  Fall back to
            # reconstructing it from the path so usage isn't recorded as "unknown".
            # Path shape: /openai/deployments/{provider}/{deployment}/...
            if not model and path.startswith("/openai/deployments/"):
                parts = path.split("/")
                if len(parts) >= 5 and parts[3] and parts[4]:
                    model = f"{parts[3]}/{parts[4]}"

            # No need to replace request._receive here.
            # Starlette's BaseHTTPMiddleware wraps the request in a
            # _CachedRequest whose wrapped_receive() automatically
            # re-serves self._body (set by our request.body() call above)
            # to downstream handlers.  Replacing _receive breaks its
            # disconnect-detection protocol and causes:
            #   RuntimeError: Unexpected message received: http.request

        request.state.tracking_request_id = request_id
        request.state.model = model

        await request_tracker.start_request(
            request_id=request_id,
            server=server_name,
            endpoint=path,
            method=request.method,
            model=model,
            user_identity="unknown",
            user_type="unknown",
            is_streaming=is_streaming,
        )

        try:
            response = await call_next(request)

            if hasattr(response, "body_iterator"):
                original_iterator = response.body_iterator

                async def tracking_iterator():
                    try:
                        async for chunk in original_iterator:
                            yield chunk
                    finally:
                        tracking_final = getattr(request.state, "tracking_final", None)
                        if isinstance(tracking_final, dict):
                            await request_tracker.end_request(
                                request_id,
                                status=tracking_final.get("status", "completed"),
                                termination_reason=tracking_final.get("termination_reason"),
                                error=tracking_final.get("error"),
                            )
                        else:
                            await request_tracker.end_request(request_id, status="completed")

                response.body_iterator = tracking_iterator()
            else:
                await request_tracker.end_request(request_id, status="completed")

            return response
        except Exception as exc:
            await request_tracker.end_request(request_id, status="errored", error=str(exc))
            raise


# ==================== OpenAI API Server ====================

def create_openai_app() -> FastAPI:
    """Create the OpenAI-compatible API server (port 11440)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await shared_startup()
        yield

    openai_app = FastAPI(
        title="LLM Proxy Server - OpenAI API",
        description="OpenAI-compatible API for multiple LLM providers",
        version="1.0.0",
        lifespan=lifespan
    )

    _add_cors(openai_app)
    _add_request_tracking(openai_app, "openai")
    _instrument_app(openai_app, "OpenAI API")

    from app.rate_limit import RateLimitExceeded
    from fastapi.responses import JSONResponse as _JSONResponse

    @openai_app.exception_handler(RateLimitExceeded)
    async def _openai_rl_handler(req, exc: RateLimitExceeded):
        return _JSONResponse(status_code=429, content=exc.body, headers=exc.headers)

    from app.routes import models, chat, completions, embeddings, images, audio, responses
    openai_app.include_router(models.router)
    openai_app.include_router(chat.router)
    openai_app.include_router(completions.router)
    openai_app.include_router(embeddings.router)
    openai_app.include_router(images.router)
    openai_app.include_router(audio.router)
    openai_app.include_router(responses.router)

    @openai_app.get("/health")
    async def health_check():
        from app.providers.provider_manager import provider_manager
        return {
            "status": "healthy",
            "server": "openai",
            "port": config.server.openai_port,
            "enabled_providers": provider_manager.get_enabled_providers(),
        }

    @openai_app.get("/api")
    async def api_info():
        return {
            "message": "LLM Proxy Server - OpenAI API",
            "version": "1.0.0",
            "endpoints": [
                "/v1/models", "/v1/chat/completions", "/v1/completions",
                "/v1/embeddings", "/v1/images/generations",
                "/v1/audio/speech", "/v1/responses",
            ]
        }

    @openai_app.get("/")
    async def root_redirect():
        """
        Redirect to management server's root page.
        This endpoint is added to provide an option to redirect
        to the management server when OpenAI root page is visited.
        """
        return RedirectResponse(
            url=f"http://{config.server.domain}:{config.server.management_port}/",
            status_code=302
        )

    return openai_app


# ==================== Anthropic API Server ====================

def create_anthropic_app() -> FastAPI:
    """Create the Anthropic Messages API server (port 2027)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await shared_startup()
        yield

    anthropic_app = FastAPI(
        title="LLM Proxy Server - Anthropic API",
        description="Anthropic Messages API for multiple LLM providers",
        version="1.0.0",
        lifespan=lifespan
    )

    _add_cors(anthropic_app)
    _add_request_tracking(anthropic_app, "anthropic")
    _instrument_app(anthropic_app, "Anthropic API")

    from app.rate_limit import RateLimitExceeded
    from fastapi.responses import JSONResponse as _JSONResponse

    @anthropic_app.exception_handler(RateLimitExceeded)
    async def _anthropic_rl_handler(req, exc: RateLimitExceeded):
        return _JSONResponse(status_code=429, content=exc.body, headers=exc.headers)

    from app.routes import anthropic_messages, anthropic_models
    anthropic_app.include_router(anthropic_messages.router)
    anthropic_app.include_router(anthropic_models.router)

    @anthropic_app.get("/health")
    async def health_check():
        from app.providers.provider_manager import provider_manager
        return {
            "status": "healthy",
            "server": "anthropic",
            "port": config.server.anthropic_port,
            "enabled_providers": provider_manager.get_enabled_providers(),
        }

    @anthropic_app.get("/api")
    async def api_info():
        return {
            "message": "LLM Proxy Server - Anthropic API",
            "version": "1.0.0",
            "endpoints": ["/v1/messages", "/v1/messages/count_tokens", "/v1/models"]
        }

    @anthropic_app.get("/")
    async def root_redirect():
        """
        Redirect to management server's root page.
        This endpoint is added to provide an option to redirect
        to the management server when Anthropic root page is visited.
        """
        return RedirectResponse(
            url=f"http://{config.server.domain}:{config.server.management_port}/",
            status_code=302
        )

    return anthropic_app


# ==================== Azure OpenAI API Server ====================

def create_azure_openai_app() -> FastAPI:
    """Create the Azure OpenAI-compatible API server (port 11439)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await shared_startup()
        yield

    azure_openai_app = FastAPI(
        title="LLM Proxy Server - Azure OpenAI API",
        description="Azure OpenAI-compatible API with deployment-based URLs",
        version="1.0.0",
        lifespan=lifespan
    )

    _add_cors(azure_openai_app)
    _add_request_tracking(azure_openai_app, "azure_openai")
    _instrument_app(azure_openai_app, "Azure OpenAI API")

    from app.rate_limit import RateLimitExceeded
    from fastapi.responses import JSONResponse as _JSONResponse

    @azure_openai_app.exception_handler(RateLimitExceeded)
    async def _azure_rl_handler(req, exc: RateLimitExceeded):
        return _JSONResponse(status_code=429, content=exc.body, headers=exc.headers)

    from app.routes import azure_openai
    azure_openai_app.include_router(azure_openai.router)

    # Mount standard OpenAI routers under /openai prefix → /openai/v1/*
    # Azure's v1 API (GA Aug 2025) uses OpenAI-compatible paths under /openai/v1/.
    # By reusing the existing route modules with a prefix, we get all v1 endpoints
    # (chat/completions, models, embeddings, responses, etc.) with zero code duplication.
    # See: https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle
    from app.routes import models, chat, completions, embeddings, images, audio, responses
    azure_openai_app.include_router(models.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(chat.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(completions.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(embeddings.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(images.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(audio.router, prefix="/openai", tags=["azure_openai_v1"])
    azure_openai_app.include_router(responses.router, prefix="/openai", tags=["azure_openai_v1"])

    @azure_openai_app.middleware("http")
    async def v1_api_middleware(request: Request, call_next):
        """Middleware for /openai/v1/ requests on the Azure server.

        Handles two v1-specific behaviors:
        1. Sets preserve_upstream_model so model names from Azure pass through
           untouched (e.g. "gpt-4.1-2025-04-14").
        2. Extracts aoai-* preview feature headers and stashes them in a
           ContextVar for the provider layer to forward to Azure.
        """
        if request.url.path.startswith("/openai/v1/"):
            from app.providers.openai_compatible import (
                preserve_upstream_model,
                extra_request_headers,
            )
            preserve_upstream_model.set(True)

            # Extract Azure-specific preview headers (e.g. "aoai-evals: preview")
            preview_headers = {
                k: v for k, v in request.headers.items()
                if k.lower().startswith("aoai-")
            }
            if preview_headers:
                extra_request_headers.set(preview_headers)

        response = await call_next(request)
        return response

    @azure_openai_app.get("/health")
    async def health_check():
        from app.providers.provider_manager import provider_manager
        return {
            "status": "healthy",
            "server": "azure_openai",
            "port": config.server.azure_openai_port,
            "enabled_providers": provider_manager.get_enabled_providers(),
        }

    @azure_openai_app.get("/api")
    async def api_info():
        return {
            "message": "LLM Proxy Server - Azure OpenAI API",
            "version": "1.0.0",
            "endpoints": [
                # Legacy deployment-based endpoints
                "/openai/models",
                "/openai/deployments/{provider_name}",
                "/openai/deployments/{provider_name}/{deployment}/chat/completions",
                "/openai/deployments/{provider_name}/{deployment}/completions",
                "/openai/deployments/{provider_name}/{deployment}/embeddings",
                "/openai/deployments/{provider_name}/{deployment}/images/generations",
                "/openai/deployments/{provider_name}/{deployment}/audio/speech",
                "/openai/deployments/{provider_name}/{deployment}/audio/transcriptions",
                "/openai/deployments/{provider_name}/{deployment}/audio/translations",
                "/openai/deployments/{provider_name}/responses",
                # v1 API (OpenAI-compatible paths — no api-version required)
                "/openai/v1/models",
                "/openai/v1/chat/completions",
                "/openai/v1/completions",
                "/openai/v1/embeddings",
                "/openai/v1/images/generations",
                "/openai/v1/audio/speech",
                "/openai/v1/audio/transcriptions",
                "/openai/v1/responses",
            ]
        }

    @azure_openai_app.get("/")
    async def root_redirect():
        """Redirect to management server's root page."""
        return RedirectResponse(
            url=f"http://{config.server.domain}:{config.server.management_port}/",
            status_code=302
        )

    return azure_openai_app


# ==================== Management Server ======================================

def create_management_app() -> FastAPI:
    """Create the management server (port 8765)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await shared_startup()
        yield
        await shared_shutdown()

    mgmt_app = FastAPI(
        title="LLM Proxy Server - Management",
        description="Admin panel and user management for LLM Proxy Server",
        version="1.0.0",
        lifespan=lifespan
    )

    _add_cors(mgmt_app)
    _instrument_app(mgmt_app, "Management")

    templates = Jinja2Templates(directory="app/frontend/templates")
    mgmt_app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

    from app.routes import auth, dashboard, admin
    mgmt_app.include_router(auth.router)
    mgmt_app.include_router(dashboard.router)
    mgmt_app.include_router(admin.router)

    @mgmt_app.get("/")
    async def root(
        request: Request,
        current_user: Optional[Union[User, AdminUser]] = Depends(get_current_user_optional),
    ):
        if current_user:
            if isinstance(current_user, AdminUser):
                return RedirectResponse(url="/admin/dashboard", status_code=302)
            else:
                return RedirectResponse(url="/dashboard/", status_code=302)
        else:
            import time as _time
            return templates.TemplateResponse(
                "dashboard/home.html",
                {"request": request, "user": None, "title": "LLM Proxy Server", "cache_version": str(int(_time.time()))}
            )

    @mgmt_app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        import time as _time
        return templates.TemplateResponse(
            "auth/login.html", {"request": request, "title": "Login - LLM Proxy Server", "cache_version": str(int(_time.time()))}
        )

    @mgmt_app.get("/signup", response_class=HTMLResponse)
    async def signup_page(request: Request):
        import time as _time
        return templates.TemplateResponse(
            "auth/signup.html", {"request": request, "title": "Sign Up - LLM Proxy Server", "cache_version": str(int(_time.time()))}
        )

    @mgmt_app.get("/health")
    async def health_check():
        from app.providers.provider_manager import provider_manager
        return {
            "status": "healthy",
            "server": "management",
            "port": config.server.management_port,
            "enabled_providers": provider_manager.get_enabled_providers(),
            "server_config": {
                "openai_port": config.server.openai_port,
                "anthropic_port": config.server.anthropic_port,
                "azure_openai_port": config.server.azure_openai_port,
                "management_port": config.server.management_port,
            }
        }

    return mgmt_app

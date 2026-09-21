"""Authentication middleware for API key validation."""

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from typing import Optional, Union
from .database import get_api_key, get_user_by_username, AsyncSessionLocal
from .models import User, APIKey
from .auth import verify_token
from .admin import AdminUser, authenticate_admin, get_admin_username
from .cache import auth_cache, CachedAPIKey, CachedUser
from app.api_envelope import envelope_for
from app.rate_limit import RateLimitExceeded
from app.tracing import (
    create_span, add_span_attributes, set_span_error,
    create_http_attributes, create_auth_attributes,
    AuthAttributes, set_trace_user
)
from opentelemetry import trace

security = HTTPBearer(auto_error=False)

# Endpoints that don't invoke the model — skip rate limiting for these
RATE_LIMIT_SKIP_PATHS = frozenset({
    "/v1/messages/count_tokens",
    "/v1/models",
    "/openai/models",
    "/openai/v1/models",
})


def get_owner_user_id(auth) -> Optional[int]:
    """Resolve the owning user id from an authenticated principal.

    Returns None for admins (they bypass per-user ownership checks). For an
    API key the owner is ``user_id`` (not the key id); for a user it is ``id``.
    Mirrors the id resolution used elsewhere (see the rate-limit path).
    """
    if isinstance(auth, AdminUser):
        return None
    uid = getattr(auth, "user_id", None)
    if uid is None:
        uid = getattr(auth, "id", None)
    return int(uid) if uid is not None else None


async def verify_response_ownership(response_id: str, auth) -> None:
    """Enforce Responses API ownership (IDOR guard).

    Admins bypass. A mapping with no recorded owner (legacy/pre-migration rows,
    or admin-created responses) is treated as unowned and allowed. Otherwise the
    caller must match the recorded ``user_id`` or the request is rejected with a
    404 (not 403, to avoid confirming the response exists).
    """
    if isinstance(auth, AdminUser):
        return
    from .database import AsyncSessionLocal as _SessionLocal, get_response_provider_mapping
    async with _SessionLocal() as db:
        mapping = await get_response_provider_mapping(db, response_id)
    if mapping is not None and mapping.user_id is not None:
        if mapping.user_id != get_owner_user_id(auth):
            raise HTTPException(status_code=404, detail="Response not found")


async def _enforce_rate_limit(
    request: Request, auth_result, envelope_override: Optional[str] = None
) -> None:
    """Check rate limits for the authenticated user. Raises RateLimitExceeded on deny."""
    from .admin import AdminUser as _AdminUser
    if isinstance(auth_result, _AdminUser):
        return
    # Normalize a trailing slash so e.g. "/v1/models/" matches "/v1/models".
    normalized_path = request.url.path.rstrip("/") or "/"
    if normalized_path in RATE_LIMIT_SKIP_PATHS:
        return

    user_id = getattr(auth_result, "user_id", None) or getattr(auth_result, "id", None)
    username = (
        getattr(auth_result, "username", None)
        or (f"key:{auth_result.id}" if hasattr(auth_result, "id") else None)
    )
    if user_id is None or username is None:
        return

    from app.rate_limit import rate_limit_tracker, RateLimitExceeded
    # Precedence (most-specific wins): instance group > model group > overall.
    # If the request's instance belongs to an instance group, or the model belongs
    # to a model group, that group's limit governs and is enforced at the route
    # level (enforce_group_rate_limit). Skip the overall gate here so an unlimited
    # group means truly unlimited. If the model is unknown, fall through to the
    # overall limit (the safe, stricter default).
    model = getattr(request.state, "model", None)
    if model:
        provider_key = model.split("/", 1)[0] if "/" in model else None
        if (provider_key and rate_limit_tracker.instance_belongs_to_group(provider_key)) \
                or rate_limit_tracker.model_belongs_to_group(model):
            return
        # Prefix-less name: the canonical id is not chosen until the route
        # resolves it (app/model_resolution.py), which runs after this. Skip the
        # overall gate only when EVERY candidate it could resolve to is grouped.
        # Deciding from a single provisional pick would be unsound: if the pick
        # were grouped but the route landed on an ungrouped candidate,
        # check_group_limit returns None and the request would be governed by no
        # limit at all. A mixed pool falls through here and is group-checked
        # again at the route -- double-governed, never ungoverned.
        from app.model_resolution import candidate_ids  # local: keeps app.auth leaf-ward
        candidates = candidate_ids(model)
        if candidates and all(
            rate_limit_tracker.instance_belongs_to_group(c.split("/", 1)[0])
            or rate_limit_tracker.model_belongs_to_group(c)
            for c in candidates
        ):
            return
    decision = await rate_limit_tracker.check_and_increment(user_id, username)
    if not decision.allowed:
        envelope = envelope_for(request.url.path, envelope_override)
        if envelope == "anthropic":
            raise RateLimitExceeded.anthropic(decision)
        if envelope == "azure":
            raise RateLimitExceeded.azure(decision)
        raise RateLimitExceeded.openai(decision)


def _resolve_identity(auth_result) -> tuple[Optional[str], Optional[str]]:
    """Normalise an auth result to the ``(identity, kind)`` pair we report it under.

    The isinstance order is load-bearing: AdminUser is checked first because its
    ``id`` is None, so the APIKey branch would render it as "key:None". An
    unrecognised auth object yields ``(None, None)`` and is reported nowhere.
    """
    if isinstance(auth_result, AdminUser):
        return auth_result.username, "admin"
    if isinstance(auth_result, (APIKey, CachedAPIKey)):
        return getattr(auth_result, 'username', None) or f"key:{auth_result.id}", "api_key"
    if isinstance(auth_result, (User, CachedUser)):
        return auth_result.username, "user"
    return None, None


async def _update_tracking_identity(request: Request, auth_result) -> None:
    """Update the request tracker with the authenticated user's identity.

    Doubles as the single choke point that publishes the username to the tracing
    layer: every successful auth path in this module, plus _authenticate_azure,
    calls this, so the OTel spans and the usage rows can never disagree.
    """
    try:
        identity, kind = _resolve_identity(auth_result)
    except Exception:
        # Reading .username/.id can raise on a detached ORM object
        # (DetachedInstanceError / MissingGreenlet) -- the JWT cache-miss path
        # hands us a User whose session is already closed. Identity reporting is
        # best-effort and must never turn a successful auth into a 500.
        identity, kind = None, None

    # Deliberately above the tracking_request_id guard: requests outside
    # _TRACKED_PREFIXES -- notably the OpenAI routers re-mounted under /openai on
    # the Azure port -- never get a tracking id, but they do reach providers and
    # still need user.id on their spans. set_trace_user() swallows its own errors,
    # so this cannot fail the request.
    if identity is not None:
        set_trace_user(identity)

    if not hasattr(request, "state") or not hasattr(request.state, "tracking_request_id"):
        return
    try:
        from app.request_tracker import request_tracker

        if identity is not None:
            await request_tracker.update_identity(
                request.state.tracking_request_id, identity, kind
            )
    except Exception:
        pass


async def get_api_key_from_request(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    
    return None


async def get_current_user_from_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Union[User, AdminUser, CachedUser]:
    """Get current user from JWT token (for web interface)."""
    token = None
    
    # First try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    else:
        # If no Authorization header, try to get token from cookie
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = verify_token(token)
    
    # Check if this is an admin token
    if token_data.is_admin and token_data.username == get_admin_username():
        # Return admin user (not from database)
        from .admin import get_admin_config
        admin_config = get_admin_config()
        return AdminUser(admin_config.username, admin_config.email)
    
    # Try cache first for regular user
    cached_user = auth_cache.get_cached_user(token_data.username)
    if cached_user and cached_user.is_active:
        # Return cached user directly - no DB query needed
        return cached_user
    
    # Cache miss - fetch from database
    async with AsyncSessionLocal() as db:
        user = await get_user_by_username(db, username=token_data.username)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Cache the user for future requests
    auth_cache.cache_user(user)

    return user


async def get_current_active_user(current_user: Union[User, AdminUser, CachedUser] = Depends(get_current_user_from_token)) -> Union[User, AdminUser, CachedUser]:
    """Get current active user."""
    # Admin users are always active
    if isinstance(current_user, AdminUser):
        return current_user
    
    # Check if regular user is active
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_user_or_admin(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Union[User, AdminUser, CachedUser]:
    """Get current user (regular or admin) from JWT token."""
    token = None
    
    # First try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    else:
        # If no Authorization header, try to get token from cookie
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token_data = verify_token(token)
        
        # Check if this is an admin token
        if token_data.is_admin and token_data.username == get_admin_username():
            # Return admin user (not from database)
            from .admin import get_admin_config
            admin_config = get_admin_config()
            return AdminUser(admin_config.username, admin_config.email)
        
        # Try cache first for regular user
        cached_user = auth_cache.get_cached_user(token_data.username)
        if cached_user and cached_user.is_active:
            # Return cached user directly - no DB query needed
            return cached_user
        
        # Cache miss - fetch from database
        async with AsyncSessionLocal() as db:
            user = await get_user_by_username(db, username=token_data.username)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Cache the user
        auth_cache.cache_user(user)

        return user
    except HTTPException:
        raise
    except Exception as e:
        # If token verification fails, raise authentication error
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[Union[User, AdminUser, CachedUser]]:
    """Get current user (regular or admin) from JWT token, return None if not authenticated."""
    token = None
    
    # First try to get token from Authorization header
    if credentials:
        token = credentials.credentials
    else:
        # If no Authorization header, try to get token from cookie
        token = request.cookies.get("access_token")
    
    if not token:
        return None
    
    try:
        token_data = verify_token(token)
        
        # Check if this is an admin token
        if token_data.is_admin and token_data.username == get_admin_username():
            # Return admin user (not from database)
            from .admin import get_admin_config
            admin_config = get_admin_config()
            return AdminUser(admin_config.username, admin_config.email)
        
        # Try cache first for regular user
        cached_user = auth_cache.get_cached_user(token_data.username)
        if cached_user and cached_user.is_active:
            # Return cached user directly - no DB query needed
            return cached_user
        
        # Cache miss - fetch from database
        async with AsyncSessionLocal() as db:
            user = await get_user_by_username(db, username=token_data.username)
        if user is None or not user.is_active:
            return None

        # Cache the user
        auth_cache.cache_user(user)

        return user
    except Exception:
        # If token verification fails, return None instead of raising an error
        return None


async def authenticate_jwt_or_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Union[User, AdminUser, APIKey, CachedAPIKey, CachedUser]:
    """Authenticate request using either JWT token or API key."""
    # Create initial attributes using semantic conventions
    initial_attributes = create_http_attributes(request.method, str(request.url))
    initial_attributes.update(create_auth_attributes("dual_auth", "pending"))
    
    with create_span(
        "auth.authenticate_jwt_or_api_key",
        kind=trace.SpanKind.INTERNAL,
        attributes=initial_attributes
    ) as span:
        try:
            token = None
            
            # First try to get token from Authorization header
            if credentials:
                token = credentials.credentials
            else:
                # If no Authorization header, try to get token from cookie
                token = request.cookies.get("access_token")
            
            # If we have a token, try JWT authentication first
            if token:
                try:
                    # Try to verify as JWT token
                    token_data = verify_token(token)
                    
                    # Check if this is an admin token
                    if token_data.is_admin and token_data.username == get_admin_username():
                        # Return admin user (not from database)
                        from .admin import get_admin_config
                        admin_config = get_admin_config()
                        add_span_attributes(span, create_auth_attributes(
                            method="jwt_admin",
                            result="success",
                            user_id=admin_config.username
                        ))
                        admin_user = AdminUser(admin_config.username, admin_config.email)
                        await _update_tracking_identity(request, admin_user)
                        await _enforce_rate_limit(request, admin_user)
                        return admin_user
                    
                    # Try cache first for regular user
                    cached_user = auth_cache.get_cached_user(token_data.username)
                    if cached_user and cached_user.is_active:
                        # Cache hit - return cached user directly
                        add_span_attributes(span, create_auth_attributes(
                            method="jwt_user",
                            result="success",
                            user_id=str(cached_user.id)
                        ))
                        add_span_attributes(span, {"auth.cache_hit": True})
                        await _update_tracking_identity(request, cached_user)
                        await _enforce_rate_limit(request, cached_user)
                        return cached_user

                    # Cache miss - fetch from database
                    async with AsyncSessionLocal() as db:
                        user = await get_user_by_username(db, username=token_data.username)
                    if user is not None and user.is_active:
                        auth_cache.cache_user(user)
                        add_span_attributes(span, create_auth_attributes(
                            method="jwt_user",
                            result="success",
                            user_id=str(user.id)
                        ))
                        add_span_attributes(span, {"auth.cache_hit": False})
                        await _update_tracking_identity(request, user)
                        await _enforce_rate_limit(request, user)
                        return user

                except RateLimitExceeded:
                    raise
                except (JWTError, HTTPException):
                    # JWT verification failed / not a valid JWT: fall through to API key.
                    # DB/infra errors are NOT swallowed here (propagate to the 500 handler).
                    pass
            
            # If JWT authentication failed or no token from cookies, try API key authentication
            if credentials and token:
                try:
                    # Try cache first for API key
                    cached = auth_cache.get_cached_api_key(token)
                    if cached and cached.is_active and cached.user_is_active:
                        # Cache hit - mark as used (batched update) and return cached object
                        auth_cache.mark_api_key_used(token)
                        
                        masked_key = token[:8] + "..." if len(token) > 8 else "***"
                        add_span_attributes(span, create_auth_attributes(
                            method="api_key",
                            result="success",
                            api_key_prefix=masked_key,
                            api_key_id=str(cached.id),
                            user_id=str(cached.user_id),
                            api_key_name=cached.name
                        ))
                        add_span_attributes(span, {"auth.cache_hit": True})

                        await _update_tracking_identity(request, cached)
                        await _enforce_rate_limit(request, cached)
                        return cached

                    # Cache miss - try to verify as API key from DB
                    async with AsyncSessionLocal() as db:
                        db_api_key = await get_api_key(db, token)
                        if db_api_key:
                            from .database import get_user_by_id
                            owner = await get_user_by_id(db, db_api_key.user_id)
                            owner_username = owner.username if owner else None
                    if db_api_key:
                        # Cache the API key with the owner's username
                        cached_key = auth_cache.cache_api_key(token, db_api_key, username=owner_username)
                        auth_cache.mark_api_key_used(token)

                        # Add masked API key to span (first 8 chars + "...")
                        masked_key = token[:8] + "..." if len(token) > 8 else "***"
                        add_span_attributes(span, create_auth_attributes(
                            method="api_key",
                            result="success",
                            api_key_prefix=masked_key,
                            api_key_id=str(db_api_key.id),
                            user_id=str(db_api_key.user_id),
                            api_key_name=db_api_key.name
                        ))
                        add_span_attributes(span, {"auth.cache_hit": False})

                        await _update_tracking_identity(request, cached_key)
                        await _enforce_rate_limit(request, cached_key)
                        return db_api_key
                except RateLimitExceeded:
                    raise
                except HTTPException:
                    # Invalid API key: fall through to the 401 below.
                    # DB/infra errors are NOT swallowed here (propagate to the 500 handler).
                    pass
            
            # If no authentication method worked, raise 401
            add_span_attributes(span, create_auth_attributes("dual_auth", "failed"))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Provide either a valid JWT token (in Authorization header or cookie) or API key (in Authorization header as 'Bearer <key>').",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        except HTTPException as e:
            set_span_error(span, e)
            raise
        except Exception as e:
            from app.rate_limit import RateLimitExceeded as _RLE
            if isinstance(e, _RLE):
                raise
            set_span_error(span, e)
            add_span_attributes(span, create_auth_attributes("dual_auth", "error"))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error"
            )


async def authenticate_anthropic_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Union[User, AdminUser, APIKey, CachedAPIKey, CachedUser]:
    """Authenticate Anthropic-compatible API requests.

    Accepts credentials from either the standard Anthropic `x-api-key` header
    or the existing `Authorization: Bearer` header. When both are present,
    `Authorization` takes precedence to preserve backward compatibility.
    """
    x_api_key = request.headers.get("x-api-key")
    if x_api_key and not credentials:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=x_api_key)
    return await authenticate_jwt_or_api_key(request, credentials)


async def get_current_admin(
    request: Request
) -> AdminUser:
    """Get current admin user from JWT token (admin-only routes)."""
    token = None
    
    # First try to get token from Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
    else:
        # If no Authorization header, try to get token from cookie
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token_data = verify_token(token)
        
        # Check if this is an admin token
        admin_username = get_admin_username()
        
        if not token_data.is_admin or token_data.username != admin_username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Return admin user
        from .admin import get_admin_config
        admin_config = get_admin_config()
        return AdminUser(admin_config.username, admin_config.email)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )

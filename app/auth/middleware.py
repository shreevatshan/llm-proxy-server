"""Authentication middleware for API key validation."""

from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Union
from .database import get_api_key, get_user_by_username, AsyncSessionLocal
from .models import User, APIKey
from .auth import verify_token
from .admin import AdminUser, authenticate_admin, get_admin_username
from .cache import auth_cache, CachedAPIKey, CachedUser
from app.rate_limit import RateLimitExceeded
from app.tracing import (
    create_span, add_span_attributes, set_span_error,
    create_http_attributes, create_auth_attributes,
    AuthAttributes
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


def _envelope_for(path: str, override: Optional[str]) -> str:
    if override:
        return override
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/openai/"):
        return "azure"
    return "openai"


async def _enforce_rate_limit(
    request: Request, auth_result, envelope_override: Optional[str] = None
) -> None:
    """Check rate limits for the authenticated user. Raises RateLimitExceeded on deny."""
    from .admin import AdminUser as _AdminUser
    if isinstance(auth_result, _AdminUser):
        return
    if request.url.path in RATE_LIMIT_SKIP_PATHS:
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
    decision = await rate_limit_tracker.check_and_increment(user_id, username)
    if not decision.allowed:
        envelope = _envelope_for(request.url.path, envelope_override)
        if envelope == "anthropic":
            raise RateLimitExceeded.anthropic(decision)
        if envelope == "azure":
            raise RateLimitExceeded.azure(decision)
        raise RateLimitExceeded.openai(decision)


async def _update_tracking_identity(request: Request, auth_result) -> None:
    """Update the request tracker with the authenticated user's identity."""
    if not hasattr(request, "state") or not hasattr(request.state, "tracking_request_id"):
        return
    try:
        from app.request_tracker import request_tracker
        from .admin import AdminUser

        request_id = request.state.tracking_request_id
        if isinstance(auth_result, AdminUser):
            await request_tracker.update_identity(request_id, auth_result.username, "admin")
        elif isinstance(auth_result, (APIKey, CachedAPIKey)):
            identity = getattr(auth_result, 'username', None) or f"key:{auth_result.id}"
            await request_tracker.update_identity(request_id, identity, "api_key")
        elif isinstance(auth_result, (User, CachedUser)):
            await request_tracker.update_identity(request_id, auth_result.username, "user")
    except Exception:
        pass


async def get_api_key_from_request(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    
    return None


async def authenticate_api_key(
    request: Request,
) -> Union[APIKey, CachedAPIKey]:
    """Authenticate request using API key."""
    # Create initial attributes using semantic conventions
    initial_attributes = create_http_attributes(request.method, str(request.url))
    initial_attributes.update(create_auth_attributes("api_key", "pending"))
    
    with create_span(
        "auth.authenticate_api_key",
        kind=trace.SpanKind.INTERNAL,
        attributes=initial_attributes
    ) as span:
        try:
            api_key = await get_api_key_from_request(request)
            
            if not api_key:
                add_span_attributes(span, create_auth_attributes("api_key", "missing_key"))
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Provide it in Authorization header as 'Bearer <key>'.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Add masked API key to span (first 8 chars + "...")
            masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
            
            # Try to get from cache first
            cached = auth_cache.get_cached_api_key(api_key)
            if cached and cached.is_active:
                # Cache hit - mark as used (batched update) and return cached object
                # No DB query needed - the cached object has all required attributes
                auth_cache.mark_api_key_used(api_key)
                
                add_span_attributes(span, create_auth_attributes(
                    method="api_key",
                    result="success",
                    api_key_prefix=masked_key,
                    api_key_id=str(cached.id),
                    user_id=str(cached.user_id),
                    api_key_name=cached.name
                ))
                add_span_attributes(span, {"auth.cache_hit": True})
                
                # Return cached object directly - it's compatible with APIKey interface
                return cached
            
            # Cache miss - fetch from database
            async with AsyncSessionLocal() as db:
                db_api_key = await get_api_key(db, api_key)
            if not db_api_key:
                add_span_attributes(span, create_auth_attributes(
                    "api_key", "invalid_key", api_key_prefix=masked_key
                ))
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Cache the API key for future requests
            auth_cache.cache_api_key(api_key, db_api_key)
            auth_cache.mark_api_key_used(api_key)
            
            # Add successful authentication details using semantic conventions
            add_span_attributes(span, create_auth_attributes(
                method="api_key",
                result="success",
                api_key_prefix=masked_key,
                api_key_id=str(db_api_key.id),
                user_id=str(db_api_key.user_id),
                api_key_name=db_api_key.name
            ))
            add_span_attributes(span, {"auth.cache_hit": False})
            
            return db_api_key
            
        except HTTPException as e:
            set_span_error(span, e)
            raise
        except Exception as e:
            set_span_error(span, e)
            add_span_attributes(span, create_auth_attributes("api_key", "error"))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error"
            )


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
    if user is None:
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
        if user is None:
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
        if user is None:
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
                    if user:
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
                except Exception:
                    # JWT verification failed, try API key authentication
                    pass
            
            # If JWT authentication failed or no token from cookies, try API key authentication
            if credentials and token:
                try:
                    # Try cache first for API key
                    cached = auth_cache.get_cached_api_key(token)
                    if cached and cached.is_active:
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
                except Exception:
                    # API key verification also failed
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

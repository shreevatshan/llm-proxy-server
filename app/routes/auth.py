"""Authentication routes for user management."""

from fastapi import APIRouter, HTTPException, status, Depends, Response, Request, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
from typing import List, Optional
import json
import secrets

from app.auth.database import (
    get_db, create_user, authenticate_user, get_user_by_username,
    get_user_by_email, is_reserved_username,
    create_api_key, get_user_api_keys, delete_api_key,
    update_user_profile, update_user_password, permanently_delete_user, verify_password,
    get_oauth_user_by_provider_id, create_oauth_user, update_oauth_user,
    is_self_delete_allowed, normalize_new_username,
    get_user_by_id, link_oauth_identity, user_has_provider_identity,
    OAuthEmailConflict, OAuthAlreadyLinked,
)
from app.auth.webhook import send_signup_webhook
from app.auth.models import (
    UserCreate, UserLogin, UserResponse, Token, APIKeyCreate,
    APIKeyResponse, APIKeyListResponse, UserUpdate, PasswordUpdate, AccountDelete,
    ZohoOAuthCallback, PendingWithdraw, OAuthLinkConfirm,
    MyQuotasResponse, QuotaSectionResponse, PoolSummary,
)
from app.auth.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.auth.middleware import get_current_active_user, get_current_user_or_admin
from app.auth.cookies import (
    cookie_secure, set_session_cookie, set_temp_cookie, clear_temp_cookie,
    OAUTH_LINK_COOKIE, PENDING_WITHDRAW_COOKIE,
)
from app.auth.flash import set_flash
from app.auth.purpose_tokens import (
    PURPOSE_OAUTH_LINK, PURPOSE_PENDING_WITHDRAW,
    issue_purpose_token, read_purpose_token, user_binding_claims, claims_match_user,
)
from app.auth.admin import authenticate_admin, is_admin_enabled, AdminUser, get_admin_email
from app.auth.models import User
from app.auth.zoho_oauth import zoho_oauth
from fastapi.responses import HTMLResponse
from typing import Union
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Name of the short-lived cookie holding the OAuth `state` value for CSRF
# protection during the ZOHO OAuth redirect round-trip.
_OAUTH_STATE_COOKIE = "zoho_oauth_state"


@router.post("/signup")
async def signup(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Normalize before anything reads the name: the form trims client-side, so
        # validating, looking up and storing the stripped value is what keeps the
        # length bound, the uniqueness checks and the stored name all talking about
        # the same string. Validated before any lookup so a bad name fails fast
        # rather than at INSERT time on a length-enforcing DB backend.
        try:
            user_data.username = normalize_new_username(user_data.username)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(e),
                headers={"X-Error-Field": "username"},
            )

        # Check if username already exists. The config-based admin is not in the
        # users table but owns its name for usage/quota purposes (is_reserved_username).
        if await get_user_by_username(db, user_data.username) or is_reserved_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
                headers={"X-Error-Field": "username"},
            )

        # Check if email already exists
        existing_email = await get_user_by_email(db, user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
                headers={"X-Error-Field": "email"},
            )
        
        # Validate password length
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long",
                headers={"X-Error-Field": "password"},
            )
        
        # Test password hashing before creating user
        logger.info(f"Testing password hashing for user: {user_data.username}")
        from app.auth.database import get_password_hash
        try:
            test_hash = get_password_hash(user_data.password)
            logger.info(f"Password hashing successful, hash length: {len(test_hash)}")
        except Exception as hash_error:
            logger.error(f"Password hashing failed: {str(hash_error)}")
            logger.error(f"Hash error traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Create user with pending approval (manual signups require admin approval)
        logger.info(f"Creating user: {user_data.username}")
        user = await create_user(db, user_data.username, user_data.email, user_data.password, is_pending=True)
        logger.info(f"User created successfully: {user.id}")

        # Send signup webhook notification
        await send_signup_webhook(
            username=user.username,
            email=user.email,
            signup_mode="manual",
            user_id=user.id,
            is_pending=True
        )

        response_data = UserResponse.from_orm(user)
        # Include admin email so the frontend can display it in the pending message
        return {**response_data.model_dump(), "admin_email": get_admin_email()}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Signup failed for user {user_data.username}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin, response: Response, db: AsyncSession = Depends(get_db)):
    """Authenticate regular user and return access token."""
    # Only authenticate regular users, not admin users
    user = await authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        admin_email = get_admin_email()
        if getattr(user, 'is_pending_approval', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your account is pending admin approval. Please contact {admin_email} for access.",
                headers={"X-Account-Status": "pending"},
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your account has been deactivated. Please contact {admin_email} for assistance.",
            headers={"X-Account-Status": "deactivated"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Set HTTP-only cookie for web interface
    set_session_cookie(response, access_token)

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/form", response_model=Token)
async def login_form(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """Authenticate regular user using form data (OAuth2 compatible)."""
    # Only authenticate regular users, not admin users
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        admin_email = get_admin_email()
        if getattr(user, 'is_pending_approval', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Your account is pending admin approval. Please contact {admin_email} for access.",
                headers={"X-Account-Status": "pending"},
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your account has been deactivated. Please contact {admin_email} for assistance.",
            headers={"X-Account-Status": "deactivated"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(response: Response):
    """Logout user by clearing the authentication cookie."""
    response.delete_cookie(key="access_token")
    from app.auth.csrf import clear_csrf_cookie
    clear_csrf_cookie(response)
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user(current_user: Union[User, AdminUser] = Depends(get_current_active_user)):
    """Get current user information."""
    if isinstance(current_user, AdminUser):
        # Return admin user info in a compatible format
        return {
            "id": None,  # Admin users don't have database IDs
            "username": current_user.username,
            "email": current_user.email,
            "created_at": current_user.created_at,
            "is_active": current_user.is_active,
            "is_admin": True
        }
    else:
        # Regular user
        return UserResponse.from_orm(current_user)


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_user_api_key(
    api_key_data: APIKeyCreate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key for the current user."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot create API keys"
        )
    
    try:
        api_key = await create_api_key(db, current_user.id, api_key_data.name)
        return APIKeyResponse.from_orm(api_key)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys", response_model=List[APIKeyListResponse])
async def list_user_api_keys(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    response: Response = None
):
    """List all API keys for the current user."""
    if isinstance(current_user, AdminUser):
        # Admin users don't have API keys
        return []
    
    # Add cache control headers to prevent caching
    if response:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    try:
        api_keys = await get_user_api_keys(db, current_user.id)
        return [
            APIKeyListResponse(
                id=key.id,
                name=key.name,
                api_key_preview=(key.key_prefix + "...") if key.key_prefix else "***",
                created_at=key.created_at,
                last_used=key.last_used,
                is_active=key.is_active
            )
            for key in api_keys
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys"
        )


@router.get("/api-keys/detail", response_model=APIKeyResponse)
async def get_user_api_key(
    api_key_id: int = Query(..., description="API key ID to retrieve"),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific API key by ID (masked) using query parameter.

    The full key is only shown once, at creation time. Read/detail responses
    mask the stored value so it cannot be re-retrieved later.
    """
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot access API keys"
        )
    
    try:
        # Get the API key and verify it belongs to the current user
        from sqlalchemy.future import select
        from app.auth.models import APIKey
        
        result = await db.execute(
            select(APIKey).where(
                APIKey.id == api_key_id,
                APIKey.user_id == current_user.id,
                APIKey.is_active == True
            )
        )
        api_key = result.scalar_one_or_none()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )

        # Mask the stored key — never re-expose the full value after creation.
        # The stored value is a SHA-256 hash; show the saved plaintext prefix.
        masked = (api_key.key_prefix + "...") if api_key.key_prefix else "***"
        return APIKeyResponse(
            id=api_key.id,
            name=api_key.name,
            api_key=masked,
            created_at=api_key.created_at,
            last_used=api_key.last_used,
            is_active=api_key.is_active,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key"
        )


@router.delete("/api-keys")
async def delete_user_api_key(
    api_key_id: int = Query(..., description="API key ID to delete"),
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an API key using query parameter."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot delete API keys"
        )
    
    success = await delete_api_key(db, api_key_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    return {"message": "API key deleted successfully"}


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    profile_data: UserUpdate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user profile information."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot update profile through this endpoint"
        )
    
    # Validate that at least one field is provided
    if not profile_data.username and not profile_data.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field (username or email) must be provided"
        )
    
    try:
        updated_user = await update_user_profile(
            db, 
            current_user.id, 
            username=profile_data.username,
            email=profile_data.email
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.from_orm(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.put("/password")
async def update_password(
    password_data: PasswordUpdate,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user password."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot update password through this endpoint"
        )
    
    # Validate new password length
    if len(password_data.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters long"
        )
    
    try:
        success = await update_user_password(
            db,
            current_user.id,
            password_data.current_password,
            password_data.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )


@router.delete("/account")
async def delete_account(
    delete_data: AccountDelete,
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    response: Response = None,
    db: AsyncSession = Depends(get_db)
):
    """Permanently delete user account and all associated data."""
    if isinstance(current_user, AdminUser):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin users cannot delete account through this endpoint"
        )

    # Checked before the confirmation text (and outside the try below, whose
    # broad except would turn this 403 into a 500).
    if not await is_self_delete_allowed(db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Account deletion has been disabled by the administrator. "
                f"Please contact {get_admin_email()} to delete your account."
            ),
            # Lets the profile page tell this apart from other 403s (e.g. CSRF).
            headers={"X-Account-Deletion-Disabled": "true"},
        )

    # Verify confirmation text before deletion
    if delete_data.confirmation != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation text is incorrect. Please type 'DELETE' exactly."
        )

    try:
        # Same sequence as the admin delete endpoints -- settle the pool, delete the
        # account and its usage, drop buffered counts, then invalidate -- kept in one
        # place so the three paths cannot drift. See delete_user_account.
        from app.routes.pools import delete_user_account
        success = await delete_user_account(db, current_user.id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Clear authentication cookie
        if response:
            response.delete_cookie(key="access_token")

        return {"message": "Account deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )


@router.get("/usage")
async def get_user_usage(
    view: Optional[str] = Query(None, description="'model' for drill-down by model"),
    id: Optional[str] = Query(None, description="Model name to drill into"),
    window: str = Query("30d", pattern="^(24h|today|yesterday|7d|30d|month|all)$", description="Time window: 24h | today | yesterday | 7d | 30d | month | all"),
    year: Optional[int] = Query(None, description="Year (required when window=month)"),
    month: Optional[int] = Query(None, description="Month 1-12 (required when window=month)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Return usage data for the authenticated user.

    - No view/id: top-level per_model + totals.
    - ?view=model&id=gpt-4: breakdown by model context (optional MVP+).
    - window=24h|today|yesterday|7d|30d|month|all
    - window=month requires year and month params.
    """
    from app.auth.database import get_usage_aggregates, get_usage_earliest_date, get_usage_timeseries
    from app.request_tracker import request_tracker

    await request_tracker.flush_pending()

    filter_model: Optional[str] = None
    if view == "model" and id is not None:
        filter_model = id

    result = await get_usage_aggregates(
        db,
        filter_user_id=current_user.id,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["timeseries"] = await get_usage_timeseries(
        db,
        filter_user_id=current_user.id,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["earliest_date"] = await get_usage_earliest_date(db, filter_user_id=current_user.id)
    return result


@router.get("/quotas", response_model=MyQuotasResponse)
async def get_my_quotas(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's read-only quota information as a unified list of
    sections, one quota bucket each.

    Every model the user can access is placed under exactly one section, resolved
    the way enforcement resolves it (rate_limit_tracker.group_for): the named model
    group that claims it, or else the built-in Default group, governed by the
    profile's own limits. Admins are exempt and receive is_admin=True, sections=[].
    """
    from app.auth.database import list_model_groups
    from app.profiles import DEFAULT_GROUP_ID, profile_registry
    from app.rate_limit import rate_limit_tracker
    from app.providers.provider_manager import provider_manager

    # Admins bypass all rate limits; AdminUser has no integer DB id
    if isinstance(current_user, AdminUser):
        return MyQuotasResponse(is_admin=True, sections=[])

    # Accessible models for this user (same filter as GET /v1/models and /auth/api/models).
    accessible = await provider_manager.get_all_models(user_id=current_user.id)
    models_by_group: dict[int, list[str]] = {}
    for m in accessible:
        models_by_group.setdefault(rate_limit_tracker.group_for(m.id), []).append(m.id)

    # Pool membership, resolved once. When pooled, the daily limit of every section is
    # the sum of the members' limits for that same section and the count covers them
    # all; the per-minute limit stays the caller's own throughout.
    pool_id, member_count, pooled_default_rpd = rate_limit_tracker.pooled_rpd_limits(
        current_user.id, current_user.username
    )
    is_pooled = pool_id is not None

    async def _share(scope_kind, scope_id, pool_rpd, rpd_remaining):
        """(share_cap, share_used, rpd_remaining) once the pool's per-member cap applies.

        A member can be stopped by their share before the pool runs dry, so "left today"
        is whichever of the two runs out first -- what the limiter will actually allow.
        """
        if not is_pooled:
            return None, None, rpd_remaining
        cap, used = await rate_limit_tracker.member_share_status(
            current_user.id, current_user.username, scope_kind, scope_id, pool_rpd,
        )
        if cap is None:
            return None, None, rpd_remaining
        own_left = max(0, cap - used)
        if rpd_remaining is not None:
            own_left = min(own_left, rpd_remaining)
        return cap, used, own_left

    async def _section(group_id, name, description, scope_kind) -> QuotaSectionResponse:
        # From the same profile snapshot enforcement reads, so the two never disagree.
        eff_rpm, eff_rpd = rate_limit_tracker.group_limits_for(current_user.id, group_id)
        cnt = await rate_limit_tracker.get_group_rpd_count(
            current_user.id, current_user.username, group_id,
        )
        if is_pooled:
            # An unlimited member makes the pool unlimited on that tier, which the
            # tracker reports as None -- the same value an unlimited section uses.
            eff_rpd = (pooled_default_rpd if group_id == DEFAULT_GROUP_ID
                       else rate_limit_tracker.pooled_group_rpd_limit(current_user.id, group_id))
        rpd_remaining = max(0, eff_rpd - cnt) if eff_rpd is not None else None
        scope_id = 0 if group_id == DEFAULT_GROUP_ID else group_id
        share_cap, share_used, rpd_remaining = await _share(scope_kind, scope_id, eff_rpd, rpd_remaining)
        return QuotaSectionResponse(
            name=name, description=description,
            rpm_limit=eff_rpm, rpd_limit=eff_rpd,
            rpd_count=cnt, rpd_remaining=rpd_remaining,
            models=models_by_group.get(group_id, []),
            is_default=group_id == DEFAULT_GROUP_ID,
            is_pooled=is_pooled,
            pooled_from_members=member_count if is_pooled else 0,
            carry_adjustment=rate_limit_tracker.carry_for(current_user.id, scope_kind, scope_id),
            share_cap=share_cap, share_used=share_used,
        )

    # Named groups, alphabetically (list_model_groups orders by name), then Default.
    sections = [
        await _section(group.id, group.name, group.description, "model_group")
        for group in await list_model_groups(db)
        if models_by_group.get(group.id)
    ]
    if models_by_group.get(DEFAULT_GROUP_ID):
        sections.append(await _section(
            DEFAULT_GROUP_ID, "Default", "Models not assigned to any other group", "overall",
        ))

    # The pool summary is a header chip, so only the name is needed beyond what the
    # tracker snapshot already carries.
    pool_summary = None
    if is_pooled:
        from app.auth.models import RequestPool
        from sqlalchemy.future import select
        row = await db.execute(select(RequestPool).where(RequestPool.id == pool_id))
        pool_obj = row.scalar_one_or_none()
        if pool_obj is not None:
            pool_summary = PoolSummary(
                id=pool_obj.id, name=pool_obj.name, member_count=member_count
            )

    return MyQuotasResponse(
        is_admin=False, sections=sections, pool=pool_summary,
        profile_name=profile_registry.for_user(current_user.id).name,
        max_concurrent_requests=rate_limit_tracker.concurrency_limit_for(current_user.id),
        in_flight=rate_limit_tracker.inflight_for(current_user.id),
    )


# ZOHO OAuth Routes

# Lifetimes of the short-lived auth-flow cookies (both scoped to /auth/).
_OAUTH_LINK_TTL_SECONDS = 600
_PENDING_WITHDRAW_TTL_SECONDS = 900

_ZOHO = "zoho"


def _oauth_redirect(url: str, message: Optional[str] = None, level: str = "danger",
                    actions=(), source: Optional[str] = None) -> RedirectResponse:
    """The one exit of the ZOHO OAuth flow: a 303 that always clears the one-time
    state cookie and, when given, carries *message* as a flash (never in the URL)."""
    resp = RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(key=_OAUTH_STATE_COOKIE)
    if message:
        set_flash(resp, message, level=level, actions=actions, source=source)
    return resp


def _issue_session(response: Response, user: User) -> None:
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    set_session_cookie(response, access_token)


def _set_withdraw_cookie(response: Response, user: User) -> None:
    """Let a pending user withdraw their signup without a password (OAuth users have none)."""
    token = issue_purpose_token(
        PURPOSE_PENDING_WITHDRAW,
        user_binding_claims(user),
        timedelta(seconds=_PENDING_WITHDRAW_TTL_SECONDS),
    )
    set_temp_cookie(response, PENDING_WITHDRAW_COOKIE, token, max_age=_PENDING_WITHDRAW_TTL_SECONDS)


def _pending_message(prefix: str = "") -> str:
    return (
        f"{prefix}Your account is pending admin approval. Please contact {get_admin_email()} "
        "for access. If you no longer want this account, you can withdraw your signup."
    )


def _deactivated_message() -> str:
    return f"Your account has been deactivated. Please contact {get_admin_email()} for assistance."


def _is_pending(user: User) -> bool:
    return bool(getattr(user, "is_pending_approval", False)) and not user.is_active


def _finish_oauth_login(user: User) -> RedirectResponse:
    """Sign an OAuth-authenticated user in, or explain why they cannot be."""
    if user.is_active:
        resp = _oauth_redirect("/dashboard/")
        _issue_session(resp, user)
        return resp
    if _is_pending(user):
        resp = _oauth_redirect(
            "/login", _pending_message(), level="warning",
            actions=["withdraw_pending"], source="oauth",
        )
        _set_withdraw_cookie(resp, user)
        return resp
    return _oauth_redirect("/login", _deactivated_message(), level="warning")


def _not_you_hint() -> str:
    return f" If you didn't create this account, contact {get_admin_email()}."


async def _oauth_email_match(db: AsyncSession, user: User, user_info) -> RedirectResponse:
    """A ZOHO identity whose email belongs to an existing account: offer an explicit link."""
    admin_email = get_admin_email()
    if user.oauth_provider == _ZOHO and user.oauth_sub == user_info.sub:
        # The account already records this exact identity (its OAuthUser row is
        # missing): restore the row rather than asking again. A matching provider
        # sub is stronger evidence than the email, so no email verification is needed.
        try:
            await link_oauth_identity(
                db, user, _ZOHO, user_info.sub, email=user_info.email, name=user_info.name,
                first_name=user_info.first_name, last_name=user_info.last_name,
                picture=user_info.picture, raw_data=json.dumps(user_info.model_dump()),
            )
        except OAuthAlreadyLinked:
            # Raced with a concurrent callback; a rollback may have expired `user`.
            oauth_user = await get_oauth_user_by_provider_id(db, _ZOHO, user_info.sub)
            if oauth_user is None:
                raise
            user = oauth_user.user
        return _finish_oauth_login(user)
    if not user_info.email_verified:
        return _oauth_redirect(
            "/login",
            "An account with this email already exists, but ZOHO has not verified this email "
            "address, so it cannot be linked. Please sign in with your password instead."
            + _not_you_hint(),
            level="warning",
        )
    if await user_has_provider_identity(db, user, _ZOHO):
        return _oauth_redirect(
            "/login",
            "An account with this email is already linked to a different ZOHO account. "
            f"Please contact {admin_email} for help, including if you didn't create this account.",
            level="warning",
        )
    if not user.hashed_password:
        return _oauth_redirect(
            "/login",
            "An account with this email already exists but cannot be linked here. "
            f"Please contact {admin_email} for help, including if you didn't create this account.",
            level="warning",
        )
    claims = user_binding_claims(user)
    claims.update({
        "provider": _ZOHO,
        "psub": user_info.sub,
        "pemail": user_info.email,
        "pname": (user_info.name or "")[:100],
        "pfirst": (user_info.first_name or None) and user_info.first_name[:50],
        "plast": (user_info.last_name or None) and user_info.last_name[:50],
        "ppic": (user_info.picture or None) and user_info.picture[:500],
    })
    token = issue_purpose_token(PURPOSE_OAUTH_LINK, claims, timedelta(seconds=_OAUTH_LINK_TTL_SECONDS))
    resp = _oauth_redirect("/auth/zoho/link")
    set_temp_cookie(resp, OAUTH_LINK_COOKIE, token, max_age=_OAUTH_LINK_TTL_SECONDS)
    return resp


@router.get("/zoho/login")
async def zoho_login(request: Request):
    """Initiate ZOHO OAuth login."""
    # Re-check if ZOHO OAuth is configured at runtime
    from app.auth.zoho_oauth import get_zoho_oauth
    current_zoho_oauth = get_zoho_oauth()

    if not current_zoho_oauth:
        return _oauth_redirect(
            "/login", "ZOHO sign-in is not available. Please sign in with your username and password.",
            level="warning",
        )

    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(32)

    # Get authorization URL
    try:
        auth_url = await current_zoho_oauth.get_authorization_url(state=state)
    except Exception:
        logger.exception("Failed to build the ZOHO authorization URL")
        return _oauth_redirect("/login", "ZOHO sign-in is temporarily unavailable. Please try again.")

    # Store the state in a short-lived HttpOnly cookie so the callback can
    # verify it matches, preventing login CSRF / session fixation.
    redirect_response = RedirectResponse(url=auth_url)
    redirect_response.set_cookie(
        key=_OAUTH_STATE_COOKIE,
        value=state,
        max_age=600,  # 10 minutes — long enough for the OAuth round-trip
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
    )
    return redirect_response


@router.get("/zoho/callback")
async def zoho_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None,
                        location: Optional[str] = None, error: Optional[str] = None,
                        error_description: Optional[str] = None,
                        db: AsyncSession = Depends(get_db)):
    """Handle ZOHO OAuth callback.

    Every exit is a 303 (see _oauth_redirect) with a fixed flash message; provider
    and exception text is logged, never shown.
    """
    # Re-check if ZOHO OAuth is configured at runtime
    from app.auth.zoho_oauth import get_zoho_oauth
    current_zoho_oauth = get_zoho_oauth()

    if not current_zoho_oauth:
        return _oauth_redirect(
            "/login", "ZOHO sign-in is not available. Please sign in with your username and password.",
            level="warning",
        )

    if error:
        logger.info(
            "ZOHO OAuth returned error=%r description=%r",
            error[:100], (error_description or "")[:200],
        )
        if error == "access_denied":
            return _oauth_redirect("/login", "ZOHO sign-in was cancelled.", level="info")
        return _oauth_redirect("/login", "ZOHO sign-in failed. Please try again.")

    # Validate the OAuth `state` against the value set at /zoho/login (CSRF
    # protection). Reject if missing or mismatched; the cookie is cleared either way.
    expected_state = request.cookies.get(_OAUTH_STATE_COOKIE)
    # Compare as bytes: compare_digest raises TypeError on non-ASCII str, and
    # `state` is attacker-controlled (would turn hostile input into a 500).
    if not expected_state or not state or not secrets.compare_digest(
        state.encode("utf-8"), expected_state.encode("utf-8")
    ):
        return _oauth_redirect(
            "/login", "Your ZOHO sign-in request expired or was invalid. Please try signing in again.",
            level="warning",
        )

    if not code:
        return _oauth_redirect("/login", "ZOHO sign-in did not complete. Please try again.", level="warning")

    try:
        user_info = await current_zoho_oauth.get_user_info(code, location)
    except Exception:
        logger.exception("ZOHO OAuth: fetching user info failed")
        return _oauth_redirect("/login", "We couldn't verify your ZOHO sign-in. Please try again.")

    try:
        raw_data = json.dumps(user_info.model_dump())

        # A. This ZOHO identity is already linked: refresh its details and sign in.
        oauth_user = await get_oauth_user_by_provider_id(db, _ZOHO, user_info.sub)
        if oauth_user:
            await update_oauth_user(
                db,
                oauth_user.id,
                email=user_info.email,
                name=user_info.name,
                first_name=user_info.first_name,
                last_name=user_info.last_name,
                picture=user_info.picture,
                raw_data=raw_data
            )
            user = oauth_user.user
            # Ensure oauth_provider is set on the user (for users migrated or linked later)
            if not user.oauth_provider:
                user.oauth_provider = _ZOHO
                user.oauth_sub = user_info.sub
                await db.commit()
                await db.refresh(user)
            return _finish_oauth_login(user)

        # B. The email belongs to an existing account: linking must be explicit.
        existing = await get_user_by_email(db, user_info.email)
        if existing:
            return await _oauth_email_match(db, existing, user_info)

        # C. A brand-new user.
        email_domain = user_info.email.split('@')[-1].lower() if '@' in user_info.email else ''
        is_pending = email_domain != 'zohocorp.com'
        try:
            user, oauth_user = await create_oauth_user(
                db,
                provider=_ZOHO,
                provider_user_id=user_info.sub,
                email=user_info.email,
                name=user_info.name,
                first_name=user_info.first_name,
                last_name=user_info.last_name,
                picture=user_info.picture,
                raw_data=raw_data,
                is_pending=is_pending,
                email_verified=bool(user_info.email_verified),
            )
        except OAuthEmailConflict:
            # Raced with another signup on the same email: offer the link instead.
            existing = await get_user_by_email(db, user_info.email)
            if existing is None:
                raise
            return await _oauth_email_match(db, existing, user_info)
        except OAuthAlreadyLinked:
            # Raced with a concurrent callback for the same identity.
            oauth_user = await get_oauth_user_by_provider_id(db, _ZOHO, user_info.sub)
            if oauth_user is None:
                raise
            return _finish_oauth_login(oauth_user.user)

        # Send signup webhook notification for new OAuth users
        await send_signup_webhook(
            username=user.username,
            email=user.email,
            signup_mode="oauth",
            user_id=user.id,
            is_pending=is_pending,
            oauth_provider=_ZOHO
        )
        return _finish_oauth_login(user)

    except Exception:
        logger.exception("ZOHO OAuth callback failed")
        try:
            await db.rollback()
        except Exception:
            logger.exception("Rollback after ZOHO OAuth failure failed")
        return _oauth_redirect("/login", "ZOHO sign-in failed due to a server error. Please try again.")


def _mask_username(username: str) -> str:
    username = username or ""
    keep = 1 if len(username) <= 3 else 2
    return username[:keep] + "***"


def _mask_email(email: str) -> str:
    email = email or ""
    local, sep, domain = email.partition("@")
    if not sep:
        return _mask_username(email)
    return (local[:1] or "") + "***@" + domain


async def _load_link_target(request: Request, db: AsyncSession):
    """Resolve the pending OAuth link request from its cookie.

    Returns ``(claims, user, problem)``; *problem* is None when the link may go
    ahead, ``"invalid"`` when the request expired / was tampered with / no longer
    matches the account, or ``"linked"`` when the account already has a ZOHO identity.
    """
    claims = read_purpose_token(request.cookies.get(OAUTH_LINK_COOKIE), PURPOSE_OAUTH_LINK)
    if not claims or claims.get("provider") != _ZOHO or not claims.get("psub") or not claims.get("pemail"):
        return None, None, "invalid"
    uid = claims.get("uid")
    if not isinstance(uid, int):
        return claims, None, "invalid"
    user = await get_user_by_id(db, uid)
    # id + username + created_at: SQLite can hand a deleted user's id to a new account.
    if user is None or not claims_match_user(claims, user):
        return claims, None, "invalid"
    if (user.email or "").lower() != str(claims.get("pemail")).lower():
        return claims, user, "invalid"
    if not user.hashed_password:
        return claims, user, "invalid"
    if await user_has_provider_identity(db, user, _ZOHO):
        return claims, user, "linked"
    return claims, user, None


_LINK_EXPIRED_MESSAGE = "This account-link request has expired. Please continue with ZOHO again."
_LINK_ALREADY_MESSAGE = "This account is already linked to a ZOHO account."


@router.get("/zoho/link", response_class=HTMLResponse)
async def zoho_link_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Ask the user whether to link their ZOHO identity to the existing account."""
    from app.routes.auth_pages import render_auth_page

    claims, user, problem = await _load_link_target(request, db)
    if problem:
        resp = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        clear_temp_cookie(resp, OAUTH_LINK_COOKIE)
        if problem == "linked":
            set_flash(resp, _LINK_ALREADY_MESSAGE + " Please continue with ZOHO to sign in.", level="info")
        else:
            set_flash(resp, _LINK_EXPIRED_MESSAGE, level="warning")
        return resp
    return render_auth_page(request, "auth/link_account.html", {
        "title": "Link ZOHO Account - LLM Proxy Server",
        "masked_username": _mask_username(user.username),
        "masked_email": _mask_email(user.email),
        "zoho_name": claims.get("pname") or "",
        "account_pending": _is_pending(user),
        "admin_email": get_admin_email(),
    })


@router.post("/zoho/link")
async def zoho_link_confirm(body: OAuthLinkConfirm, request: Request, response: Response,
                            db: AsyncSession = Depends(get_db)):
    """Link the ZOHO identity from the link cookie after verifying the account password."""
    claims, user, problem = await _load_link_target(request, db)
    if problem == "linked":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=_LINK_ALREADY_MESSAGE)
    if problem:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_LINK_EXPIRED_MESSAGE)

    if not verify_password(body.password, user.hashed_password):
        # The link cookie is kept so the user can retry.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect password")

    try:
        await link_oauth_identity(
            db, user, _ZOHO, claims["psub"],
            email=claims["pemail"],
            name=claims.get("pname") or claims["pemail"],
            first_name=claims.get("pfirst"),
            last_name=claims.get("plast"),
            picture=claims.get("ppic"),
        )
    except OAuthAlreadyLinked:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This ZOHO account is already linked to an account.",
        )
    except Exception:
        logger.exception("Linking ZOHO identity to user %s failed", user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to link accounts. Please try again.",
        )

    clear_temp_cookie(response, OAUTH_LINK_COOKIE)
    logger.info("Linked ZOHO identity to user %s", user.id)
    if user.is_active:
        _issue_session(response, user)
        return {"message": "Accounts linked.", "redirect": "/dashboard/"}
    if _is_pending(user):
        _set_withdraw_cookie(response, user)
        set_flash(
            response, _pending_message("Your ZOHO account is now linked. "),
            level="warning", actions=["withdraw_pending"], source="oauth",
        )
    else:
        set_flash(response, "Your ZOHO account is now linked. " + _deactivated_message(), level="warning")
    return {"message": "Accounts linked.", "redirect": "/login"}


@router.post("/zoho/link/decline")
async def zoho_link_decline(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Decline linking: no account is created; explain how to proceed instead."""
    _claims, user, _problem = await _load_link_target(request, db)
    clear_temp_cookie(response, OAUTH_LINK_COOKIE)
    admin_email = get_admin_email()
    message = "No account was linked or created."
    if user is not None:
        if _is_pending(user):
            message += (
                " To use ZOHO sign-in instead, sign in with your password, choose "
                "\"Withdraw signup\", then continue with ZOHO again."
            )
        elif user.is_active and await is_self_delete_allowed(db):
            message += (
                " To use ZOHO sign-in with a new account, sign in with your password, delete the "
                "existing account from your Profile page, then continue with ZOHO again."
            )
        else:
            message += (
                " To use ZOHO sign-in with a new account, contact "
                f"{admin_email} to remove the existing account."
            )
    message += f" If you didn't create that account, contact {admin_email}."
    set_flash(response, message, level="info")
    return {"message": "No account was linked.", "redirect": "/login"}


_WITHDRAW_EXPIRED_MESSAGE = "This withdraw option has expired. Please sign in again to withdraw."


@router.post("/pending/withdraw")
async def withdraw_pending_signup(body: PendingWithdraw, request: Request, response: Response,
                                  db: AsyncSession = Depends(get_db)):
    """Withdraw (permanently delete) a signup that is still pending approval.

    Identified by username + password, or (no credentials given) by the short-lived
    pending_withdraw cookie set after a ZOHO sign-in. Deliberately not gated by the
    admin's self-delete setting: a pending account was never granted access.
    """
    if body.username or body.password:
        if not body.username or not body.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password are required.",
            )
        user = await authenticate_user(db, body.username, body.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
    else:
        claims = read_purpose_token(
            request.cookies.get(PENDING_WITHDRAW_COOKIE), PURPOSE_PENDING_WITHDRAW
        )
        uid = claims.get("uid") if claims else None
        user = await get_user_by_id(db, uid) if isinstance(uid, int) else None
        if user is None or not claims_match_user(claims, user):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_WITHDRAW_EXPIRED_MESSAGE)

    if not _is_pending(user):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a signup that is still pending approval can be withdrawn.",
        )

    user_id = user.id
    try:
        from app.routes.pools import delete_user_account
        success = await delete_user_account(db, user_id)
    except Exception:
        logger.exception("Withdrawing pending signup for user %s failed", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to withdraw your signup. Please try again.",
        )
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_WITHDRAW_EXPIRED_MESSAGE)

    logger.info("Pending signup withdrawn for user %s", user_id)
    clear_temp_cookie(response, PENDING_WITHDRAW_COOKIE)
    message = "Your signup has been withdrawn and your data deleted."
    set_flash(response, message, level="success")
    return {"message": message, "redirect": "/login"}


@router.get("/zoho/status")
async def zoho_oauth_status():
    """Check if ZOHO OAuth is configured and available."""
    # Re-check if ZOHO OAuth is configured at runtime
    from app.auth.zoho_oauth import get_zoho_oauth
    current_zoho_oauth = get_zoho_oauth()
    
    if not current_zoho_oauth:
        return {
            "available": False,
            "message": "ZOHO OAuth is not configured. Please set ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET environment variables."
        }
    
    return {
        "available": True,
        "message": "ZOHO OAuth is available",
        "redirect_uri": current_zoho_oauth.config.redirect_uri,
        "scopes": current_zoho_oauth.config.scope
    }


@router.get("/debug/auth-status")
async def debug_auth_status(request: Request):
    """Debug endpoint to check authentication status and cookies.

    Disabled by default: it reflects token internals (a token/account oracle),
    so it is gated behind AUTH_DEBUG_ENDPOINT_ENABLED=true and returns 404
    otherwise. The token preview is never returned.
    """
    if os.getenv("AUTH_DEBUG_ENDPOINT_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    cookies = dict(request.cookies)
    access_token = cookies.get("access_token")

    result = {
        "cookies": list(cookies.keys()),
        "has_access_token": bool(access_token),
    }

    if access_token:
        try:
            from app.auth.auth import verify_token
            token_data = verify_token(access_token)
            result["token_valid"] = True
            result["username"] = token_data.username
            result["is_admin"] = token_data.is_admin
        except Exception:
            result["token_valid"] = False

    return result

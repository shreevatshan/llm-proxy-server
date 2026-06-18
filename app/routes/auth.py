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
    get_user_by_email, create_api_key, get_user_api_keys, delete_api_key,
    update_user_profile, update_user_password, permanently_delete_user, verify_password,
    get_oauth_user_by_provider_id, create_oauth_user, update_oauth_user
)
from app.auth.webhook import send_signup_webhook
from app.auth.models import (
    UserCreate, UserLogin, UserResponse, Token, APIKeyCreate,
    APIKeyResponse, APIKeyListResponse, UserUpdate, PasswordUpdate, AccountDelete,
    ZohoOAuthCallback,
    MyQuotasResponse, QuotaOverallResponse, QuotaGroupResponse, QuotaInstanceGroupResponse,
)
from app.auth.auth import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.auth.middleware import get_current_active_user, get_current_user_or_admin
from app.auth.admin import authenticate_admin, is_admin_enabled, AdminUser, get_admin_email
from app.auth.models import User
from app.auth.zoho_oauth import zoho_oauth
from typing import Union

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup")
async def signup(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    import traceback
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
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
                detail=f"Password hashing failed: {str(hash_error)}"
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
            detail=f"Failed to create user: {str(e)}"
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
                detail=f"Your account is pending admin approval. Please contact {admin_email} for access."
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your account has been deactivated. Please contact {admin_email} for assistance."
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Set HTTP-only cookie for web interface
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax"
    )
    
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
                detail=f"Your account is pending admin approval. Please contact {admin_email} for access."
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Your account has been deactivated. Please contact {admin_email} for assistance."
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
                api_key_preview=key.api_key[:8] + "..." if len(key.api_key) > 8 else key.api_key,
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
    """Get a specific API key by ID (returns full API key) using query parameter."""
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
        
        return APIKeyResponse.from_orm(api_key)
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

    # Verify confirmation text before deletion
    if delete_data.confirmation != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation text is incorrect. Please type 'DELETE' exactly."
        )

    try:
        success = await permanently_delete_user(db, current_user.id)

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
    window: str = Query("30d", description="Time window: 24h | today | yesterday | 7d | 30d | month | all"),
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
    from app.auth.database import get_usage_aggregates, get_usage_earliest_date
    from app.request_tracker import request_tracker

    await request_tracker.flush_pending()

    filter_model: Optional[str] = None
    if view == "model" and id is not None:
        filter_model = id

    result = await get_usage_aggregates(
        db,
        group_by="model",
        filter_user=current_user.username,
        filter_model=filter_model,
        window=window,
        year=year,
        month=month,
    )
    result["earliest_date"] = await get_usage_earliest_date(db, filter_user=current_user.username)
    return result


@router.get("/quotas", response_model=MyQuotasResponse)
async def get_my_quotas(
    current_user: Union[User, AdminUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's read-only quota information.

    - overall: effective RPM/RPD limits + current usage counts
    - groups:  each model group's effective RPM/RPD limits (override or group default) + member models
    - Admins are exempt from rate limits and receive is_admin=True with no quota data.
    """
    from app.auth.models import ModelGroup
    from app.auth.database import (
        list_model_groups, get_user_group_rate_limit,
        list_instance_groups, get_user_instance_group_rate_limit,
    )
    from app.rate_limit import rate_limit_tracker

    # Admins bypass all rate limits; AdminUser has no integer DB id
    if isinstance(current_user, AdminUser):
        return MyQuotasResponse(is_admin=True, overall=None, groups=[], instance_groups=[])

    # --- Overall request-level quota (read-only, no increment) ---
    status_obj = await rate_limit_tracker.get_user_status(current_user.id, current_user.username)
    overall = QuotaOverallResponse(
        rpm_limit=status_obj.rpm_limit,
        rpm_count=status_obj.rpm_count,
        rpm_remaining=status_obj.rpm_remaining,
        rpd_limit=status_obj.rpd_limit,
        rpd_count=status_obj.rpd_count,
        rpd_remaining=status_obj.rpd_remaining,
    )

    # --- Per-model-group quotas ---
    group_rows = await list_model_groups(db)  # eager-loads .members
    quota_groups: list[QuotaGroupResponse] = []
    for group in group_rows:
        # Resolve effective limit: per-user override if set, else group default
        ov = await get_user_group_rate_limit(db, current_user.id, group.id)
        eff_rpm = ov.rpm_limit if (ov and ov.rpm_limit is not None) else group.rpm_default
        eff_rpd = ov.rpd_limit if (ov and ov.rpd_limit is not None) else group.rpd_default
        models = [m.model_id for m in group.members]
        cnt = await rate_limit_tracker.get_group_rpd_count(current_user.username, models, group.id)
        rpd_remaining = max(0, eff_rpd - cnt) if eff_rpd is not None else None
        quota_groups.append(QuotaGroupResponse(
            name=group.name,
            description=group.description,
            rpm_limit=eff_rpm,
            rpd_limit=eff_rpd,
            rpd_count=cnt,
            rpd_remaining=rpd_remaining,
            models=models,
        ))

    # --- Per-instance-group quotas ---
    instance_group_rows = await list_instance_groups(db)  # eager-loads .members
    quota_instance_groups: list[QuotaInstanceGroupResponse] = []
    for group in instance_group_rows:
        ov = await get_user_instance_group_rate_limit(db, current_user.id, group.id)
        eff_rpm = ov.rpm_limit if (ov and ov.rpm_limit is not None) else group.rpm_default
        eff_rpd = ov.rpd_limit if (ov and ov.rpd_limit is not None) else group.rpd_default
        instances = [m.provider_key for m in group.members]
        cnt = await rate_limit_tracker.get_instance_group_rpd_count(current_user.username, instances, group.id)
        rpd_remaining = max(0, eff_rpd - cnt) if eff_rpd is not None else None
        quota_instance_groups.append(QuotaInstanceGroupResponse(
            name=group.name,
            description=group.description,
            rpm_limit=eff_rpm,
            rpd_limit=eff_rpd,
            rpd_count=cnt,
            rpd_remaining=rpd_remaining,
            instances=instances,
        ))

    return MyQuotasResponse(
        is_admin=False, overall=overall,
        groups=quota_groups, instance_groups=quota_instance_groups,
    )


# ZOHO OAuth Routes
@router.get("/zoho/login")
async def zoho_login(request: Request):
    """Initiate ZOHO OAuth login."""
    # Re-check if ZOHO OAuth is configured at runtime
    from app.auth.zoho_oauth import get_zoho_oauth
    current_zoho_oauth = get_zoho_oauth()
    
    if not current_zoho_oauth:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="ZOHO OAuth is not configured. Please set ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET environment variables."
        )
    
    # Generate state parameter for security
    state = secrets.token_urlsafe(32)
    
    # Store state in session (you might want to use a more secure session storage)
    # For now, we'll pass it through the OAuth flow
    
    # Get authorization URL
    auth_url = await current_zoho_oauth.get_authorization_url(state=state)
    
    return RedirectResponse(url=auth_url)


@router.get("/zoho/callback")
async def zoho_callback(code: str, state: str = None, location: str = None, 
                       db: AsyncSession = Depends(get_db)):
    """Handle ZOHO OAuth callback."""
    # Re-check if ZOHO OAuth is configured at runtime
    from app.auth.zoho_oauth import get_zoho_oauth
    current_zoho_oauth = get_zoho_oauth()
    
    if not current_zoho_oauth:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="ZOHO OAuth is not configured"
        )
    
    try:
        # Get user info from ZOHO
        user_info = await current_zoho_oauth.get_user_info(code, location)
        
        # Check if OAuth user already exists
        oauth_user = await get_oauth_user_by_provider_id(db, "zoho", user_info.sub)

        is_new_user = False
        if oauth_user:
            # Existing OAuth user - update their information
            await update_oauth_user(
                db,
                oauth_user.id,
                email=user_info.email,
                name=user_info.name,
                first_name=user_info.first_name,
                last_name=user_info.last_name,
                picture=user_info.picture,
                raw_data=json.dumps(user_info.model_dump())
            )
            user = oauth_user.user
            # Ensure oauth_provider is set on the user (for users migrated or linked later)
            if not user.oauth_provider:
                user.oauth_provider = "zoho"
                user.oauth_sub = user_info.sub
                await db.commit()
                await db.refresh(user)
        else:
            # New OAuth user - create user and OAuth record
            is_new_user = True
            email_domain = user_info.email.split('@')[-1].lower() if '@' in user_info.email else ''
            is_pending = email_domain != 'zohocorp.com'
            user, oauth_user = await create_oauth_user(
                db,
                provider="zoho",
                provider_user_id=user_info.sub,
                email=user_info.email,
                name=user_info.name,
                first_name=user_info.first_name,
                last_name=user_info.last_name,
                picture=user_info.picture,
                raw_data=json.dumps(user_info.model_dump()),
                is_pending=is_pending
            )

            # Send signup webhook notification for new OAuth users
            await send_signup_webhook(
                username=user.username,
                email=user.email,
                signup_mode="oauth",
                user_id=user.id,
                is_pending=is_pending,
                oauth_provider="zoho"
            )

        # Block pending users before issuing a token
        if not user.is_active:
            admin_email = get_admin_email()
            from urllib.parse import quote
            msg = f"Your account is pending admin approval. Please contact {admin_email} for access."
            return RedirectResponse(url=f"/login?error={quote(msg)}", status_code=302)

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )

        # Create redirect response
        redirect_response = RedirectResponse(url="/dashboard/", status_code=302)

        # Set HTTP-only cookie for web interface
        redirect_response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )

        return redirect_response
        
    except Exception as e:
        # Redirect to login page with error
        error_msg = f"OAuth authentication failed: {str(e)}" if str(e) else "OAuth authentication failed: Unknown error"
        from urllib.parse import quote
        return RedirectResponse(url=f"/login?error={quote(error_msg)}", status_code=302)


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
    """Debug endpoint to check authentication status and cookies."""
    cookies = dict(request.cookies)
    access_token = cookies.get("access_token")
    
    result = {
        "cookies": list(cookies.keys()),
        "has_access_token": bool(access_token),
        "access_token_preview": access_token[:20] + "..." if access_token else None
    }
    
    if access_token:
        try:
            from app.auth.auth import verify_token
            token_data = verify_token(access_token)
            result["token_valid"] = True
            result["username"] = token_data.username
            result["is_admin"] = token_data.is_admin
        except Exception as e:
            result["token_valid"] = False
            result["token_error"] = str(e)
    
    return result

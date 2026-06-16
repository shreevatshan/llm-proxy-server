"""Database connection and operations for authentication."""

import os
import secrets
import logging
from pathlib import Path
from sqlalchemy import create_engine, event, String
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, selectinload
from sqlalchemy.future import select
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from .models import Base, User, APIKey, ModelConfiguration, ProviderCredentials, OAuthUser, ResponseProviderMapping, RequestUsage, RequestUsageHourly, RequestUsageMonthly, UserRateLimit, GlobalRateLimit, ModelGroup, ModelGroupMember, UserModelGroupRateLimit
from app.providers.azure_deployments import serialize_azure_deployments

# Initialize logger
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/llm_proxy.db")
SYNC_DATABASE_URL = DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")

# Database timeout and retry configuration
DATABASE_BUSY_TIMEOUT = int(os.getenv("DATABASE_BUSY_TIMEOUT", "5"))  # seconds
DB_RETRY_MAX_ATTEMPTS = int(os.getenv("DB_RETRY_MAX_ATTEMPTS", "3"))
DB_RETRY_BACKOFF_MS = int(os.getenv("DB_RETRY_BACKOFF_MS", "100"))  # milliseconds

# Ensure database directory exists before creating engines
def ensure_db_directory():
    """Ensure the database directory exists and has proper permissions.
    Works for both Docker and native environments."""
    db_url = DATABASE_URL
    if db_url.startswith("sqlite"):
        # Extract the file path from the URL (remove sqlite+aiosqlite:///)
        db_path = db_url.split("///")[-1]
        db_dir = os.path.dirname(db_path) or "."
        
        # Resolve to absolute path
        db_dir = os.path.abspath(db_dir)
        db_path = os.path.abspath(db_path)
        
        # Create directory with open permissions
        old_umask = os.umask(0)
        try:
            os.makedirs(db_dir, mode=0o777, exist_ok=True)
        finally:
            os.umask(old_umask)
        
        # Ensure directory permissions (handles existing directories)
        try:
            os.chmod(db_dir, 0o777)
        except Exception:
            pass
        
        # Set permissive umask before SQLAlchemy creates the database file
        os.umask(0o000)


# Ensure directory exists before creating engines
ensure_db_directory()

# Create engines with proper configuration for SQLite async
# - pool_pre_ping: Verify connections before use
# - pool_recycle: Recycle connections to prevent stale ones  
# - connect_args: Enable check_same_thread=False for SQLite + async
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,  # Recycle connections after 5 minutes
    connect_args={"check_same_thread": False, "timeout": DATABASE_BUSY_TIMEOUT}
)
sync_engine = create_engine(
    SYNC_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": DATABASE_BUSY_TIMEOUT}
)


# Enable WAL mode for concurrent reads + single writer (eliminates most lock contention)
def _set_sqlite_pragmas(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


if DATABASE_URL.startswith("sqlite"):
    event.listen(engine.sync_engine, "connect", _set_sqlite_pragmas)
    event.listen(sync_engine, "connect", _set_sqlite_pragmas)


# Session makers
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# Password hashing with Docker-compatible configuration
# Use pbkdf2_sha256 which is built into Python and works reliably in Docker
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"], 
    deprecated="auto",
    pbkdf2_sha256__rounds=100000
)


# Database retry decorator for handling lock contention
def with_db_retry(max_attempts: int = DB_RETRY_MAX_ATTEMPTS, backoff_ms: int = DB_RETRY_BACKOFF_MS):
    """Decorator to retry database operations with exponential backoff on lock errors."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            from sqlalchemy.exc import OperationalError
            import asyncio
            
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    if "database is locked" in str(e).lower():
                        last_error = e
                        if attempt < max_attempts - 1:
                            # Exponential backoff: 100ms, 200ms, 400ms, etc.
                            wait_time = (backoff_ms / 1000) * (2 ** attempt)
                            logger.warning(f"Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                            await asyncio.sleep(wait_time)
                            continue
                    raise  # Re-raise if not a lock error or max attempts reached
            
            # Max attempts reached
            raise last_error
        return wrapper
    return decorator


def create_tables():
    """Create database tables synchronously."""
    Base.metadata.create_all(bind=sync_engine)


async def create_tables_async():
    """Create database tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using pbkdf2_sha256."""
    return pwd_context.hash(password)


def generate_api_key() -> str:
    """Generate a secure API key without prefix."""
    return secrets.token_urlsafe(32)


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get user by username with OAuth accounts eagerly loaded."""
    result = await db.execute(
        select(User)
        .where(User.username == username)
        .options(selectinload(User.oauth_accounts))
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


@with_db_retry()
async def create_user(db: AsyncSession, username: str, email: str, password: str, is_pending: bool = False) -> User:
    """Create a new user. If is_pending=True, the account requires admin approval before it becomes active."""
    hashed_password = get_password_hash(password)
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_active=not is_pending,
        is_pending_approval=is_pending
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    user = await get_user_by_username(db, username)
    if not user:
        return None
    if not user.hashed_password:  # OAuth user without password
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# OAuth user functions
async def get_oauth_user_by_provider_id(db: AsyncSession, provider: str, provider_user_id: str) -> Optional[OAuthUser]:
    """Get OAuth user by provider and provider user ID."""
    result = await db.execute(
        select(OAuthUser).options(selectinload(OAuthUser.user)).where(
            OAuthUser.provider == provider,
            OAuthUser.provider_user_id == provider_user_id
        )
    )
    return result.scalar_one_or_none()


async def create_oauth_user(db: AsyncSession, provider: str, provider_user_id: str, email: str, name: str,
                           first_name: Optional[str] = None, last_name: Optional[str] = None,
                           picture: Optional[str] = None, raw_data: Optional[str] = None,
                           is_pending: bool = False) -> tuple[User, OAuthUser]:
    """Create a new user with OAuth authentication."""
    # Generate a unique username from email or name
    username = email.split('@')[0] if '@' in email else name.lower().replace(' ', '_')

    # Check if username exists and make it unique if needed
    base_username = username
    counter = 1
    while await get_user_by_username(db, username):
        username = f"{base_username}_{counter}"
        counter += 1

    # Check if email exists
    existing_user_by_email = await get_user_by_email(db, email)
    if existing_user_by_email:
        # Email exists but might be for OAuth linking
        user = existing_user_by_email
        # Update oauth_provider if not already set (linking existing account to OAuth)
        if not user.oauth_provider:
            user.oauth_provider = provider
            user.oauth_sub = provider_user_id
    else:
        # Create new user
        user = User(
            username=username,
            email=email,
            hashed_password=None,  # No password for OAuth users
            oauth_provider=provider,
            oauth_sub=provider_user_id,
            is_active=not is_pending,
            is_pending_approval=is_pending
        )
        db.add(user)
        await db.flush()  # Flush to get the user ID
    
    # Create OAuth user record
    oauth_user = OAuthUser(
        user_id=user.id,
        provider=provider,
        provider_user_id=provider_user_id,
        email=email,
        name=name,
        first_name=first_name,
        last_name=last_name,
        picture=picture,
        raw_data=raw_data
    )
    db.add(oauth_user)
    await db.commit()
    await db.refresh(user)
    await db.refresh(oauth_user)
    
    return user, oauth_user


async def update_oauth_user(db: AsyncSession, oauth_user_id: int, email: Optional[str] = None, 
                           name: Optional[str] = None, first_name: Optional[str] = None, 
                           last_name: Optional[str] = None, picture: Optional[str] = None, 
                           raw_data: Optional[str] = None) -> Optional[OAuthUser]:
    """Update OAuth user information."""
    result = await db.execute(select(OAuthUser).where(OAuthUser.id == oauth_user_id))
    oauth_user = result.scalar_one_or_none()
    if not oauth_user:
        return None
    
    if email is not None:
        oauth_user.email = email
    if name is not None:
        oauth_user.name = name
    if first_name is not None:
        oauth_user.first_name = first_name
    if last_name is not None:
        oauth_user.last_name = last_name
    if picture is not None:
        oauth_user.picture = picture
    if raw_data is not None:
        oauth_user.raw_data = raw_data
    
    oauth_user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(oauth_user)
    
    return oauth_user


async def get_api_key(db: AsyncSession, api_key: str) -> Optional[APIKey]:
    """Get API key from database."""
    result = await db.execute(
        select(APIKey).where(APIKey.api_key == api_key, APIKey.is_active == True)
    )
    return result.scalar_one_or_none()


async def update_api_key_last_used(db: AsyncSession, api_key: str):
    """Update the last used timestamp for an API key."""
    result = await db.execute(select(APIKey).where(APIKey.api_key == api_key))
    db_api_key = result.scalar_one_or_none()
    if db_api_key:
        db_api_key.last_used = datetime.utcnow()
        await db.commit()


@with_db_retry()
async def create_api_key(db: AsyncSession, user_id: int, name: str) -> APIKey:
    """Create a new API key for a user."""
    api_key = generate_api_key()
    db_api_key = APIKey(
        user_id=user_id,
        api_key=api_key,
        name=name
    )
    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)
    return db_api_key


async def get_user_api_keys(db: AsyncSession, user_id: int) -> list[APIKey]:
    """Get all API keys for a user."""
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == user_id, APIKey.is_active == True)
    )
    return result.scalars().all()


async def delete_api_key(db: AsyncSession, api_key_id: int, user_id: int) -> bool:
    """Delete an API key (soft delete by setting is_active to False)."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == api_key_id, APIKey.user_id == user_id)
    )
    db_api_key = result.scalar_one_or_none()
    if db_api_key:
        # Invalidate the cache entry before soft-deleting
        from .cache import auth_cache
        auth_cache.invalidate_api_key(db_api_key.api_key)
        
        db_api_key.is_active = False
        await db.commit()
        return True
    return False


async def update_user_profile(db: AsyncSession, user_id: int, username: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
    """Update user profile information."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return None
    
    try:
        if username is not None:
            # Check if username is already taken by another user
            existing_user = await get_user_by_username(db, username)
            if existing_user and existing_user.id != user_id:
                raise ValueError("Username already taken")
            user.username = username
        
        if email is not None:
            # Check if email is already taken by another user
            existing_user = await get_user_by_email(db, email)
            if existing_user and existing_user.id != user_id:
                raise ValueError("Email already taken")
            user.email = email
        
        await db.commit()
        await db.refresh(user)
        return user
    except Exception as e:
        await db.rollback()
        raise e


async def update_user_password(db: AsyncSession, user_id: int, current_password: str, new_password: str) -> bool:
    """Update user password after verifying current password."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False

    # Verify current password
    if not verify_password(current_password, user.hashed_password):
        return False

    try:
        # Update password
        user.hashed_password = get_password_hash(new_password)
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def admin_reset_user_password(db: AsyncSession, user_id: int, new_password: str) -> bool:
    """Reset user password by admin without requiring current password."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False

    try:
        # Update password
        user.hashed_password = get_password_hash(new_password)
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def get_global_rate_limit(db: AsyncSession) -> Optional[GlobalRateLimit]:
    """Return the singleton global rate limit row (id=1), or None if not yet seeded."""
    result = await db.execute(select(GlobalRateLimit).where(GlobalRateLimit.id == 1))
    return result.scalar_one_or_none()


async def upsert_global_rate_limit(
    db: AsyncSession, rpm: Optional[int], rpd: Optional[int], admin_username: str
) -> GlobalRateLimit:
    """Create or update the global rate limit singleton."""
    row = await get_global_rate_limit(db)
    if row is None:
        row = GlobalRateLimit(id=1)
        db.add(row)
    row.rpm_default = rpm
    row.rpd_default = rpd
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def get_user_rate_limit(db: AsyncSession, user_id: int) -> Optional[UserRateLimit]:
    """Return the per-user rate limit override for the given user, or None."""
    result = await db.execute(select(UserRateLimit).where(UserRateLimit.user_id == user_id))
    return result.scalar_one_or_none()


async def upsert_user_rate_limit(
    db: AsyncSession,
    user_id: int,
    rpm: Optional[int],
    rpd: Optional[int],
    admin_username: str,
    fields_set: set,
) -> UserRateLimit:
    """Create or update a per-user rate limit override.

    Only updates fields present in fields_set so callers can distinguish
    "set to null (clear)" from "field not provided (no change)".
    """
    row = await get_user_rate_limit(db, user_id)
    if row is None:
        row = UserRateLimit(user_id=user_id)
        db.add(row)
    if "rpm_limit" in fields_set:
        row.rpm_limit = rpm
    if "rpd_limit" in fields_set:
        row.rpd_limit = rpd
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_user_rate_limit(db: AsyncSession, user_id: int) -> bool:
    """Remove the per-user rate limit override; user falls back to global defaults."""
    row = await get_user_rate_limit(db, user_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def permanently_delete_user(db: AsyncSession, user_id: int) -> bool:
    """Permanently delete a user and all associated data from the database."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False
    
    try:
        # Invalidate user and their API keys from cache
        from .cache import auth_cache
        auth_cache.invalidate_user(user.username)
        auth_cache.invalidate_user_api_keys(user_id)
        
        # First delete all associated API keys
        result = await db.execute(select(APIKey).where(APIKey.user_id == user_id))
        api_keys = result.scalars().all()
        for api_key in api_keys:
            await db.delete(api_key)
        
        # Delete all associated OAuth user records
        from app.auth.models import OAuthUser
        result = await db.execute(select(OAuthUser).where(OAuthUser.user_id == user_id))
        oauth_users = result.scalars().all()
        for oauth_user in oauth_users:
            await db.delete(oauth_user)
        
        # Finally delete the user
        await db.delete(user)
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e


# Model Management Database Operations (Updated to use unified ProviderCredentials system)

async def get_all_provider_configurations(db: AsyncSession) -> List[ProviderCredentials]:
    """Get all provider credentials (for backward compatibility with admin interface)."""
    result = await db.execute(select(ProviderCredentials))
    return result.scalars().all()


async def create_or_update_provider_configuration(
    db: AsyncSession, 
    provider_key: str, 
    provider_type: str, 
    provider_name: str,
    is_enabled: bool = True
) -> ProviderCredentials:
    """Create or update a provider configuration (uses ProviderCredentials now)."""
    existing = await get_provider_credentials(db, provider_key)
    
    if existing:
        existing.provider_type = provider_type
        existing.provider_name = provider_name
        existing.enabled = is_enabled  # Changed from is_enabled to enabled
        existing.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing)
        return existing
    else:
        provider_config = ProviderCredentials(
            provider_key=provider_key,
            provider_type=provider_type,
            provider_name=provider_name,
            enabled=is_enabled  # Changed from is_enabled to enabled
        )
        db.add(provider_config)
        await db.commit()
        await db.refresh(provider_config)
        return provider_config


async def get_model_configuration(db: AsyncSession, model_id: str) -> Optional[ModelConfiguration]:
    """Get model configuration by model ID."""
    result = await db.execute(
        select(ModelConfiguration).where(ModelConfiguration.model_id == model_id)
    )
    return result.scalar_one_or_none()


async def get_all_model_configurations(db: AsyncSession) -> List[ModelConfiguration]:
    """Get all model configurations."""
    result = await db.execute(select(ModelConfiguration))
    return result.scalars().all()


async def get_models_by_provider(db: AsyncSession, provider_key: str) -> List[ModelConfiguration]:
    """Get all models for a specific provider."""
    result = await db.execute(
        select(ModelConfiguration).where(ModelConfiguration.provider_key == provider_key)
    )
    return result.scalars().all()


async def create_or_update_model_configuration(
    db: AsyncSession,
    model_id: str,
    provider_key: str,
    model_name: str,
    is_enabled: bool = True
) -> ModelConfiguration:
    """Create or update a model configuration."""
    existing = await get_model_configuration(db, model_id)
    
    if existing:
        existing.provider_key = provider_key
        existing.model_name = model_name
        existing.is_enabled = is_enabled
        existing.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing)
        return existing
    else:
        model_config = ModelConfiguration(
            model_id=model_id,
            provider_key=provider_key,
            model_name=model_name,
            is_enabled=is_enabled
        )
        db.add(model_config)
        await db.commit()
        await db.refresh(model_config)
        return model_config


async def toggle_provider_configuration(db: AsyncSession, provider_key: str, enabled: bool) -> bool:
    """Toggle provider configuration and all its models (uses ProviderCredentials now)."""
    try:
        # Update provider
        provider = await get_provider_credentials(db, provider_key)
        if not provider:
            return False
        
        provider.enabled = enabled  # Changed from is_enabled to enabled
        provider.updated_at = datetime.utcnow()
        
        # Update all models under this provider
        models = await get_models_by_provider(db, provider_key)
        for model in models:
            model.is_enabled = enabled
            model.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("provider_toggle", provider_key=provider_key, enabled=enabled)
        
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def toggle_model_configuration(db: AsyncSession, model_id: str, enabled: bool) -> bool:
    """Toggle model configuration and auto-enable provider if needed (uses ProviderCredentials now)."""
    try:
        model = await get_model_configuration(db, model_id)
        if not model:
            return False
        
        model.is_enabled = enabled
        model.updated_at = datetime.utcnow()
        
        # If enabling a model, auto-enable its provider
        if enabled:
            provider = await get_provider_credentials(db, model.provider_key)
            if provider and not provider.enabled:  # Changed from is_enabled to enabled
                provider.enabled = True  # Changed from is_enabled to enabled
                provider.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("model_toggle", model_id=model_id, enabled=enabled, provider_key=model.provider_key)
        
        return True
    except Exception as e:
        await db.rollback()
        raise e


@with_db_retry()
async def bulk_toggle_all_models(db: AsyncSession, enabled: bool) -> bool:
    """Enable or disable all models and providers (uses ProviderCredentials now)."""
    try:
        # Update all providers
        providers = await get_all_provider_configurations(db)
        for provider in providers:
            provider.enabled = enabled  # Changed from is_enabled to enabled
            provider.updated_at = datetime.utcnow()
        
        # Update all models
        models = await get_all_model_configurations(db)
        for model in models:
            model.is_enabled = enabled
            model.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("bulk_toggle", enabled=enabled)
        
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def search_models_and_providers(db: AsyncSession, query: str) -> Dict[str, List]:
    """Search models and providers by query string (uses ProviderCredentials now)."""
    query_lower = query.lower()
    
    # Search models
    models_result = await db.execute(
        select(ModelConfiguration).where(
            ModelConfiguration.model_name.ilike(f"%{query}%")
        )
    )
    models = models_result.scalars().all()
    
    # Search providers
    providers_result = await db.execute(
        select(ProviderCredentials).where(
            (ProviderCredentials.instance_name.ilike(f"%{query}%")) |
            (ProviderCredentials.provider_name.ilike(f"%{query}%")) |
            (ProviderCredentials.provider_type.ilike(f"%{query}%")) |
            (ProviderCredentials.provider_key.ilike(f"%{query}%"))
        )
    )
    providers = providers_result.scalars().all()
    
    return {
        "models": models,
        "providers": providers
    }


async def get_model_configurations_dict(db: AsyncSession) -> Dict[str, bool]:
    """Get model configurations as a dictionary for caching."""
    models = await get_all_model_configurations(db)
    return {model.model_id: model.is_enabled for model in models}


async def get_provider_configurations_dict(db: AsyncSession) -> Dict[str, bool]:
    """Get provider configurations as a dictionary for caching (uses ProviderCredentials now)."""
    providers = await get_all_provider_configurations(db)
    return {provider.provider_key: provider.enabled for provider in providers}  # Changed from is_enabled to enabled


# Provider Credentials Database Operations

async def get_provider_credentials(db: AsyncSession, provider_key: str) -> Optional[ProviderCredentials]:
    """Get provider credentials by provider key."""
    result = await db.execute(
        select(ProviderCredentials).where(ProviderCredentials.provider_key == provider_key)
    )
    return result.scalar_one_or_none()


async def get_all_provider_credentials(db: AsyncSession) -> List[ProviderCredentials]:
    """Get all provider credentials."""
    result = await db.execute(select(ProviderCredentials))
    return result.scalars().all()


async def create_provider_credentials(
    db: AsyncSession,
    provider_type: str,
    instance_name: str,
    enabled: bool = True,
    **kwargs
) -> ProviderCredentials:
    """Create new provider credentials."""
    import json
    from sqlalchemy.exc import IntegrityError
    
    # Generate provider key using provider_name:instance_name format
    # provider_name is now required and should be set to provider_type for specialized providers
    provider_name = kwargs.get('provider_name')
    if not provider_name:
        raise ValueError("provider_name is required")

    if provider_type == "azure" and not kwargs.get("azure_backend"):
        kwargs["azure_backend"] = "openai"
    
    provider_key = f"{provider_name}:{instance_name}"
    
    # Double-check if provider already exists (for race condition safety)
    existing = await get_provider_credentials(db, provider_key)
    if existing:
        raise ValueError(f"Provider already exists: {provider_key}")
    
    # Handle deployments JSON conversion
    deployments = kwargs.pop('deployments', None)
    openai_deployments = kwargs.pop('openai_deployments', None)
    anthropic_deployments = kwargs.pop('anthropic_deployments', None)
    deployments_json = None
    if provider_type == "azure":
        deployments_json = serialize_azure_deployments(
            deployments=deployments,
            openai_deployments=openai_deployments,
            anthropic_deployments=anthropic_deployments,
        )
    elif deployments:
        deployments_json = json.dumps(deployments)
    
    credentials = ProviderCredentials(
        provider_key=provider_key,
        provider_type=provider_type,
        instance_name=instance_name,
        enabled=enabled,
        deployments_json=deployments_json,
        **kwargs
    )
    
    try:
        db.add(credentials)
        await db.commit()
        await db.refresh(credentials)
        return credentials
    except IntegrityError as e:
        await db.rollback()
        if "UNIQUE constraint failed: provider_credentials.provider_key" in str(e):
            raise ValueError(f"Provider already exists: {provider_key}")
        else:
            raise e


async def update_provider_credentials(
    db: AsyncSession,
    provider_key: str,
    **kwargs
) -> Optional[ProviderCredentials]:
    """Update provider credentials with proper handling of provider key changes."""
    import json
    
    credentials = await get_provider_credentials(db, provider_key)
    if not credentials:
        return None
    
    try:
        if credentials.provider_type == "azure" and "azure_backend" in kwargs and kwargs["azure_backend"] is None:
            kwargs["azure_backend"] = credentials.azure_backend or "openai"

        # Check if instance_name or provider_name is being changed (which affects provider_key)
        new_instance_name = kwargs.get('instance_name')
        new_provider_name = kwargs.get('provider_name')
        
        key_will_change = False
        new_provider_key = None
        
        if new_instance_name and new_instance_name != credentials.instance_name:
            key_will_change = True
        if new_provider_name and new_provider_name != credentials.provider_name:
            key_will_change = True
        
        if key_will_change:
            # Generate new provider key using provider_name:instance_name format
            final_instance_name = new_instance_name or credentials.instance_name
            final_provider_name = new_provider_name or credentials.provider_name
            new_provider_key = f"{final_provider_name}:{final_instance_name}"
            
            # Check if new provider key already exists
            existing_new = await get_provider_credentials(db, new_provider_key)
            if existing_new:
                raise ValueError(f"Provider with key '{new_provider_key}' already exists")
            
            # Perform provider rename with model migration
            return await _rename_provider_with_models(db, provider_key, new_provider_key, **kwargs)
        
        # Normal update (no provider key change)
        # Handle deployments JSON conversion
        deployments = kwargs.pop('deployments', None)
        openai_deployments = kwargs.pop('openai_deployments', None)
        anthropic_deployments = kwargs.pop('anthropic_deployments', None)
        if credentials.provider_type == "azure":
            if (
                deployments is not None
                or openai_deployments is not None
                or anthropic_deployments is not None
            ):
                kwargs['deployments_json'] = serialize_azure_deployments(
                    deployments=deployments,
                    openai_deployments=openai_deployments,
                    anthropic_deployments=anthropic_deployments,
                )
        elif deployments is not None:
            kwargs['deployments_json'] = json.dumps(deployments)
        
        # Update fields
        for field, value in kwargs.items():
            if value is not None and hasattr(credentials, field):
                setattr(credentials, field, value)
        
        credentials.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(credentials)
        return credentials
    except Exception as e:
        await db.rollback()
        raise e


async def _rename_provider_with_models(
    db: AsyncSession,
    old_provider_key: str,
    new_provider_key: str,
    **kwargs
) -> ProviderCredentials:
    """Rename a provider and migrate all associated models atomically."""
    import json
    
    # Get the existing provider
    old_credentials = await get_provider_credentials(db, old_provider_key)
    if not old_credentials:
        raise ValueError(f"Provider not found: {old_provider_key}")
    
    try:
        # Get all models associated with the old provider
        models = await get_models_by_provider(db, old_provider_key)
        
        # Handle deployments JSON conversion
        deployments = kwargs.pop('deployments', None)
        openai_deployments = kwargs.pop('openai_deployments', None)
        anthropic_deployments = kwargs.pop('anthropic_deployments', None)
        if old_credentials.provider_type == "azure":
            if (
                deployments is not None
                or openai_deployments is not None
                or anthropic_deployments is not None
            ):
                kwargs['deployments_json'] = serialize_azure_deployments(
                    deployments=deployments,
                    openai_deployments=openai_deployments,
                    anthropic_deployments=anthropic_deployments,
                )
        elif deployments is not None:
            kwargs['deployments_json'] = json.dumps(deployments)
        
        # Create new provider credentials with updated key
        new_credentials = ProviderCredentials(
            provider_key=new_provider_key,
            provider_type=old_credentials.provider_type,
            instance_name=kwargs.get('instance_name', old_credentials.instance_name),
            enabled=kwargs.get('enabled', old_credentials.enabled),
            endpoint=kwargs.get('endpoint', old_credentials.endpoint),
            api_key=kwargs.get('api_key', old_credentials.api_key),
            api_version=kwargs.get('api_version', old_credentials.api_version),
            azure_backend=kwargs.get('azure_backend', old_credentials.azure_backend),
            region=kwargs.get('region', old_credentials.region),
            access_key_id=kwargs.get('access_key_id', old_credentials.access_key_id),
            secret_access_key=kwargs.get('secret_access_key', old_credentials.secret_access_key),
            base_url=kwargs.get('base_url', old_credentials.base_url),
            deployments_json=kwargs.get('deployments_json', old_credentials.deployments_json),
            subscription_id=kwargs.get('subscription_id', old_credentials.subscription_id),
            resource_group=kwargs.get('resource_group', old_credentials.resource_group),
            account_name=kwargs.get('account_name', old_credentials.account_name),
            client_id=kwargs.get('client_id', old_credentials.client_id),
            client_secret=kwargs.get('client_secret', old_credentials.client_secret),
            tenant_id=kwargs.get('tenant_id', old_credentials.tenant_id),
            provider_name=kwargs.get('provider_name', old_credentials.provider_name),
            dynamic_discovery=kwargs.get('dynamic_discovery', old_credentials.dynamic_discovery),
            supported_apis=kwargs.get('supported_apis', old_credentials.supported_apis),
        )
        
        # Add new provider to session
        db.add(new_credentials)
        await db.flush()  # Flush to get the new provider in the session
        
        # Update all model records to use the new provider key
        for model in models:
            # Update model_id to use new provider key
            old_model_id = model.model_id
            if '/' in old_model_id:
                model_name_part = old_model_id.split('/', 1)[1]
                new_model_id = f"{new_provider_key}/{model_name_part}"
            else:
                new_model_id = f"{new_provider_key}/{model.model_name}"
            
            model.model_id = new_model_id
            model.provider_key = new_provider_key
            model.updated_at = datetime.utcnow()
        
        # Delete the old provider
        await db.delete(old_credentials)
        
        # Commit all changes atomically
        await db.commit()
        await db.refresh(new_credentials)
        
        print(f"Provider renamed: {old_provider_key} -> {new_provider_key}")
        print(f"Updated {len(models)} model records")
        
        return new_credentials
        
    except Exception as e:
        await db.rollback()
        print(f"Error renaming provider {old_provider_key} to {new_provider_key}: {e}")
        raise e


@with_db_retry()
async def delete_provider_credentials(db: AsyncSession, provider_key: str) -> bool:
    """Delete provider credentials and all associated models."""
    credentials = await get_provider_credentials(db, provider_key)
    if not credentials:
        return False
    
    try:
        # First delete all models associated with this provider
        models = await get_models_by_provider(db, provider_key)
        for model in models:
            await db.delete(model)
        
        # Then delete the provider credentials
        await db.delete(credentials)
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def toggle_provider_credentials(db: AsyncSession, provider_key: str, enabled: bool) -> bool:
    """Toggle provider credentials enabled state."""
    credentials = await get_provider_credentials(db, provider_key)
    if not credentials:
        return False
    
    try:
        credentials.enabled = enabled
        credentials.updated_at = datetime.utcnow()
        await db.commit()
        return True
    except Exception as e:
        await db.rollback()
        raise e


@with_db_retry()
async def clear_all_model_configurations(db: AsyncSession) -> int:
    """Clear all model configurations from the database."""
    try:
        models = await get_all_model_configurations(db)
        count = len(models)
        
        for model in models:
            await db.delete(model)
        
        await db.commit()
        return count
    except Exception as e:
        await db.rollback()
        raise e


@with_db_retry()
async def bulk_create_model_configurations(db: AsyncSession, models_data: List[Dict]) -> int:
    """Bulk create model configurations from a list of model data."""
    try:
        created_count = 0
        
        for model_data in models_data:
            model_config = ModelConfiguration(
                model_id=model_data['model_id'],
                provider_key=model_data['provider_key'],
                model_name=model_data['model_name'],
                is_enabled=model_data.get('is_enabled', True)
            )
            db.add(model_config)
            created_count += 1
        
        await db.commit()
        return created_count
    except Exception as e:
        await db.rollback()
        raise e


async def refresh_models_from_providers(db: AsyncSession, fresh_models: List[Dict]) -> Dict[str, int]:
    """Clear all existing models and replace with fresh models from providers."""
    try:
        # Clear existing models
        cleared_count = await clear_all_model_configurations(db)
        
        # Create new models
        created_count = await bulk_create_model_configurations(db, fresh_models)
        
        # Trigger cache update
        await _update_cache_after_database_change("model_sync")
        
        return {
            "cleared": cleared_count,
            "created": created_count
        }
    except Exception as e:
        await db.rollback()
        raise e


async def identify_stale_models(db: AsyncSession, current_model_ids: List[str]) -> List[Dict]:
    """Identify models in database that are not in the current list of model IDs."""
    try:
        all_models = await get_all_model_configurations(db)
        current_ids_set = set(current_model_ids)
        
        stale_models = []
        for model in all_models:
            if model.model_id not in current_ids_set:
                stale_models.append({
                    "model_id": model.model_id,
                    "provider_key": model.provider_key,
                    "model_name": model.model_name,
                    "is_enabled": model.is_enabled,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                })
        
        return stale_models
    except Exception as e:
        raise e


async def delete_stale_models(db: AsyncSession, stale_model_ids: List[str]) -> int:
    """Delete specific models from the database by their model IDs."""
    try:
        deleted_count = 0
        
        for model_id in stale_model_ids:
            model = await get_model_configuration(db, model_id)
            if model:
                await db.delete(model)
                deleted_count += 1
        
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("model_sync")
        
        return deleted_count
    except Exception as e:
        await db.rollback()
        raise e


async def _update_cache_after_database_change(operation: str, **kwargs) -> None:
    """Update cache after database changes to maintain real-time consistency."""
    try:
        # Import here to avoid circular imports
        from app.providers.provider_manager import provider_manager
        
        if operation == "model_toggle":
            # Update single model in cache
            model_id = kwargs.get("model_id")
            enabled = kwargs.get("enabled")
            provider_key = kwargs.get("provider_key")
            
            if model_id and enabled is not None:
                provider_manager.model_cache.update_single_model_config(model_id, enabled)
                print(f"Cache updated: model {model_id} {'enabled' if enabled else 'disabled'}")
                
                # If enabling model, also ensure provider is enabled in cache
                if enabled and provider_key:
                    provider_manager.model_cache.update_single_provider_config(provider_key, True)
        
        elif operation == "provider_toggle":
            # Update provider and all its models in cache
            provider_key = kwargs.get("provider_key")
            enabled = kwargs.get("enabled")
            
            if provider_key and enabled is not None:
                provider_manager.model_cache.update_provider_and_models_config(provider_key, enabled)
                print(f"Cache updated: provider {provider_key} and its models {'enabled' if enabled else 'disabled'}")
        
        elif operation == "bulk_toggle":
            # Refresh entire cache from database
            await provider_manager.refresh_model_configurations()
            print("Cache updated: bulk toggle operation")
        
        elif operation == "model_sync":
            # Refresh entire cache from database
            await provider_manager.refresh_model_configurations()
            print("Cache updated: model sync operation")
        
        elif operation == "provider_create":
            # Invalidate provider cache to trigger reload
            provider_key = kwargs.get("provider_key")
            if provider_key:
                provider_manager.model_cache.invalidate_provider(provider_key)
                print(f"Cache invalidated for new provider: {provider_key}")
        
        elif operation == "provider_delete":
            # Remove provider and its models from cache
            provider_key = kwargs.get("provider_key")
            if provider_key:
                provider_manager.model_cache.invalidate_provider(provider_key)
                provider_manager.model_cache.update_single_provider_config(provider_key, False)
                print(f"Cache updated: provider {provider_key} deleted")
        
        elif operation == "full_refresh":
            # Complete cache refresh
            await provider_manager.model_cache.refresh_cache_from_database()
            print("Cache updated: full refresh")
            
    except Exception as e:
        print(f"Error updating cache after {operation}: {e}")
        # Don't raise the exception to avoid breaking database operations


# ==================== RESPONSES API PROVIDER MAPPING ====================

async def store_response_provider_mapping(
    db: AsyncSession, response_id: str, provider_key: str, model_name: str = None
) -> ResponseProviderMapping:
    """Store a response_id -> provider mapping for Responses API routing."""
    try:
        mapping = ResponseProviderMapping(
            response_id=response_id,
            provider_key=provider_key,
            model_name=model_name
        )
        db.add(mapping)
        await db.commit()
        await db.refresh(mapping)
        return mapping
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to store response provider mapping: {e}")
        raise


async def get_response_provider_mapping(
    db: AsyncSession, response_id: str
) -> Optional[ResponseProviderMapping]:
    """Look up which provider created a given response_id."""
    result = await db.execute(
        select(ResponseProviderMapping).where(ResponseProviderMapping.response_id == response_id)
    )
    return result.scalar_one_or_none()


async def delete_response_provider_mapping(
    db: AsyncSession, response_id: str
) -> bool:
    """Delete a response_id -> provider mapping (e.g., when response is deleted)."""
    try:
        result = await db.execute(
            select(ResponseProviderMapping).where(ResponseProviderMapping.response_id == response_id)
        )
        mapping = result.scalar_one_or_none()
        if mapping:
            await db.delete(mapping)
            await db.commit()
            return True
        return False
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete response provider mapping: {e}")
        raise


async def flush_request_usage(rows: list[dict]) -> None:
    """Bulk upsert daily usage rows, incrementing request_count on conflict."""
    if not rows:
        return
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    async with AsyncSessionLocal() as db:
        try:
            stmt = sqlite_insert(RequestUsage).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "user_identity", "model", "server"],
                set_={"request_count": RequestUsage.request_count + stmt.excluded.request_count},
            )
            await db.execute(stmt)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to flush request usage: {e}")
            raise


async def flush_request_usage_hourly(rows: list[dict]) -> None:
    """Bulk upsert hourly usage rows, incrementing request_count on conflict."""
    if not rows:
        return
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    async with AsyncSessionLocal() as db:
        try:
            stmt = sqlite_insert(RequestUsageHourly).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["date", "hour", "user_identity", "model", "server"],
                set_={"request_count": RequestUsageHourly.request_count + stmt.excluded.request_count},
            )
            await db.execute(stmt)
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to flush hourly request usage: {e}")
            raise


async def prune_hourly_usage() -> None:
    """Delete hourly rows older than yesterday (~48h retention)."""
    from sqlalchemy import delete
    from datetime import timedelta
    from app import time_utils
    cutoff = time_utils.local_today() - timedelta(days=1)
    async with AsyncSessionLocal() as db:
        try:
            await db.execute(delete(RequestUsageHourly).where(RequestUsageHourly.date < cutoff))
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to prune hourly usage: {e}")
            raise


async def rollup_to_monthly() -> None:
    """Roll up fully-aged months from request_usage into request_usage_monthly.

    A month is eligible when its last calendar day is at least 30 days before today,
    ensuring we never roll up a partially-complete month.
    """
    from sqlalchemy import func, delete, text
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from datetime import date, timedelta
    from app import time_utils
    import calendar

    today = time_utils.local_today()

    async with AsyncSessionLocal() as db:
        try:
            # Find distinct (year, month) pairs present in the daily table
            q = select(
                func.strftime('%Y', RequestUsage.date).label('yr'),
                func.strftime('%m', RequestUsage.date).label('mo'),
            ).group_by(
                func.strftime('%Y', RequestUsage.date),
                func.strftime('%m', RequestUsage.date),
            )
            ym_rows = (await db.execute(q)).all()

            for ym in ym_rows:
                year, month = int(ym.yr), int(ym.mo)
                last_day_of_month = date(year, month, calendar.monthrange(year, month)[1])
                if last_day_of_month >= today - timedelta(days=30):
                    continue  # month not fully aged yet

                # Aggregate all daily rows for this month
                agg_q = select(
                    RequestUsage.user_identity,
                    RequestUsage.user_type,
                    RequestUsage.model,
                    RequestUsage.server,
                    func.sum(RequestUsage.request_count).label("request_count"),
                ).where(
                    func.strftime('%Y', RequestUsage.date) == str(year),
                    func.strftime('%m', RequestUsage.date) == f"{month:02d}",
                ).group_by(
                    RequestUsage.user_identity,
                    RequestUsage.user_type,
                    RequestUsage.model,
                    RequestUsage.server,
                )
                agg_rows = (await db.execute(agg_q)).all()

                if not agg_rows:
                    continue

                monthly_rows = [
                    {
                        "year": year,
                        "month": month,
                        "user_identity": r.user_identity,
                        "user_type": r.user_type,
                        "model": r.model,
                        "server": r.server,
                        "request_count": r.request_count,
                    }
                    for r in agg_rows
                ]

                # Upsert into monthly table
                stmt = sqlite_insert(RequestUsageMonthly).values(monthly_rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["year", "month", "user_identity", "model", "server"],
                    set_={"request_count": RequestUsageMonthly.request_count + stmt.excluded.request_count},
                )
                await db.execute(stmt)

                # Delete the source daily rows for this month
                await db.execute(
                    delete(RequestUsage).where(
                        func.strftime('%Y', RequestUsage.date) == str(year),
                        func.strftime('%m', RequestUsage.date) == f"{month:02d}",
                    )
                )
                await db.commit()
                logger.info(f"Rolled up {len(monthly_rows)} groups for {year}-{month:02d} into monthly table")

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to roll up monthly usage: {e}")
            raise


def _build_usage_where(window: str, year: Optional[int], month: Optional[int]):
    """Return (table, where_clauses) for the given window over the daily table.

    Returns None for table when the window should use the monthly table directly.
    For 'month' window with year/month, returns clauses for both daily and monthly tables
    so the caller can UNION them.
    """
    from datetime import timedelta
    from sqlalchemy import func
    from app import time_utils
    today = time_utils.local_today()

    if window == "all":
        return None, []  # handled inline in get_usage_aggregates (UNION daily + monthly, no date filter)
    if window == "24h":
        # Handled separately via RequestUsageHourly
        return None, []
    if window == "today":
        return RequestUsage, [RequestUsage.date == today]
    if window == "yesterday":
        return RequestUsage, [RequestUsage.date == today - timedelta(days=1)]
    if window == "7d":
        return RequestUsage, [RequestUsage.date >= today - timedelta(days=6)]
    if window == "30d":
        return RequestUsage, [RequestUsage.date >= today - timedelta(days=29)]
    if window == "month" and year and month:
        return None, []  # handled inline in get_usage_aggregates
    return RequestUsage, [RequestUsage.date >= today - timedelta(days=29)]


async def get_usage_earliest_date(db: AsyncSession, filter_user: Optional[str] = None) -> Optional[str]:
    """Return the earliest date for which any usage data exists, as an ISO string (YYYY-MM-DD).

    Checks the daily table first (exact dates), then falls back to the monthly rollup
    (returns the first day of the earliest year/month found there).
    When filter_user is given, scopes to that user only.
    """
    from sqlalchemy import func
    from datetime import date

    daily_q = select(func.min(RequestUsage.date))
    monthly_q = select(func.min(RequestUsageMonthly.year), func.min(RequestUsageMonthly.month))

    if filter_user:
        daily_q = daily_q.where(RequestUsage.user_identity == filter_user)
        monthly_q = select(
            func.min(RequestUsageMonthly.year),
            func.min(RequestUsageMonthly.month),
        ).where(RequestUsageMonthly.user_identity == filter_user)

    daily_min = (await db.execute(daily_q)).scalar()
    if daily_min:
        return daily_min.isoformat() if hasattr(daily_min, 'isoformat') else str(daily_min)

    monthly_row = (await db.execute(monthly_q)).first()
    if monthly_row and monthly_row[0] is not None:
        return date(int(monthly_row[0]), int(monthly_row[1]), 1).isoformat()

    return None


async def get_usage_years(db: AsyncSession) -> list[int]:
    """Return sorted list of years with any usage data (daily or monthly tables)."""
    from sqlalchemy import func, union_all, literal_column
    from datetime import date

    q_daily = select(
        func.strftime('%Y', RequestUsage.date).label('yr')
    ).group_by(func.strftime('%Y', RequestUsage.date))

    q_monthly = select(
        RequestUsageMonthly.year.cast(String).label('yr')
    ).group_by(RequestUsageMonthly.year)

    # Execute both and merge in Python (SQLite async union is awkward with labels)
    daily_years = {int(r.yr) for r in (await db.execute(q_daily)).all()}
    monthly_years = {int(r.yr) for r in (await db.execute(q_monthly)).all()}
    return sorted(daily_years | monthly_years)


async def get_usage_aggregates(
    db: AsyncSession,
    group_by: str = "user",
    filter_user: Optional[str] = None,
    filter_model: Optional[str] = None,
    window: str = "30d",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """Return aggregated usage data for the requested time window.

    window: '24h' | 'today' | 'yesterday' | '7d' | '30d' | 'month'
    year, month: required when window='month'
    group_by: 'user' | 'model' (top-level only)
    filter_user / filter_model: drill-down
    """
    from sqlalchemy import func, union_all
    from datetime import date, timedelta
    from app import time_utils

    today = time_utils.local_today()

    # ------------------------------------------------------------------ #
    # Build the base query (or queries for month UNION) depending on window
    # ------------------------------------------------------------------ #

    def _daily_where():
        if window == "today":
            return [RequestUsage.date == today]
        if window == "yesterday":
            return [RequestUsage.date == today - timedelta(days=1)]
        if window == "7d":
            return [RequestUsage.date >= today - timedelta(days=6)]
        return [RequestUsage.date >= today - timedelta(days=29)]  # default 30d

    async def _query_24h(extra_where):
        now_utc = time_utils.local_now()
        cutoff_dt = now_utc - timedelta(hours=24)
        cutoff_date = cutoff_dt.date()
        cutoff_hour = cutoff_dt.hour

        where = [
            (RequestUsageHourly.date > cutoff_date) |
            ((RequestUsageHourly.date == cutoff_date) & (RequestUsageHourly.hour >= cutoff_hour))
        ] + extra_where
        return where

    async def _exec_top_level_query(where_clauses, use_hourly=False, use_monthly=False):
        """Run user + model top-level queries and return (per_user, per_model)."""
        tbl_u = RequestUsageHourly if use_hourly else (RequestUsageMonthly if use_monthly else RequestUsage)
        tbl_m = tbl_u

        def _user_q(tbl, where):
            return (
                select(
                    tbl.user_identity,
                    tbl.user_type,
                    func.sum(tbl.request_count).label("request_count"),
                )
                .where(*where)
                .group_by(tbl.user_identity, tbl.user_type)
                .order_by(func.sum(tbl.request_count).desc(), tbl.user_identity)
            )

        def _model_q(tbl, where):
            return (
                select(
                    tbl.model,
                    func.sum(tbl.request_count).label("request_count"),
                )
                .where(*where)
                .group_by(tbl.model)
                .order_by(func.sum(tbl.request_count).desc(), tbl.model)
            )

        user_rows = (await db.execute(_user_q(tbl_u, where_clauses))).all()
        model_rows = (await db.execute(_model_q(tbl_m, where_clauses))).all()
        return (
            [{"user_identity": r.user_identity, "user_type": r.user_type, "request_count": r.request_count} for r in user_rows],
            [{"model": r.model, "request_count": r.request_count} for r in model_rows],
        )

    async def _exec_drilldown_query(where_clauses, use_hourly=False, use_monthly=False):
        tbl = RequestUsageHourly if use_hourly else (RequestUsageMonthly if use_monthly else RequestUsage)
        if filter_user is not None:
            q = (
                select(tbl.model, func.sum(tbl.request_count).label("request_count"))
                .where(*where_clauses, tbl.user_identity == filter_user)
                .group_by(tbl.model)
                .order_by(func.sum(tbl.request_count).desc(), tbl.model)
            )
            rows = (await db.execute(q)).all()
            return [{"model": r.model, "request_count": r.request_count} for r in rows]
        else:
            q = (
                select(tbl.user_identity, tbl.user_type, func.sum(tbl.request_count).label("request_count"))
                .where(*where_clauses, tbl.model == filter_model)
                .group_by(tbl.user_identity, tbl.user_type)
                .order_by(func.sum(tbl.request_count).desc(), tbl.user_identity)
            )
            rows = (await db.execute(q)).all()
            return [{"user_identity": r.user_identity, "user_type": r.user_type, "request_count": r.request_count} for r in rows]

    # ------------------------------------------------------------------ #
    # 24h window — strict rolling 24h using the hourly table only.
    # Requests made before request_usage_hourly was introduced will not
    # appear; this resolves naturally within 24h of first deploy.
    # ------------------------------------------------------------------ #
    if window == "24h":
        now_utc = time_utils.local_now()
        cutoff_dt = now_utc - timedelta(hours=24)
        cutoff_date = cutoff_dt.date()
        cutoff_hour = cutoff_dt.hour
        base_where = [
            (RequestUsageHourly.date > cutoff_date) |
            ((RequestUsageHourly.date == cutoff_date) & (RequestUsageHourly.hour >= cutoff_hour))
        ]

        if filter_user is not None or filter_model is not None:
            breakdown = await _exec_drilldown_query(base_where, use_hourly=True)
            return {"window": window, "breakdown": breakdown}

        per_user, per_model = await _exec_top_level_query(base_where, use_hourly=True)
        total_requests = sum(r["request_count"] for r in per_user)
        return {
            "window": window,
            "per_user": per_user,
            "per_model": per_model,
            "totals": {"requests": total_requests, "unique_users": len(per_user), "unique_models": len(per_model)},
        }

    # ------------------------------------------------------------------ #
    # Month window — UNION daily rows (not yet rolled up) + monthly rows
    # ------------------------------------------------------------------ #
    if window == "month" and year and month:
        yr_str = str(year)
        mo_str = f"{month:02d}"

        daily_where = [
            func.strftime('%Y', RequestUsage.date) == yr_str,
            func.strftime('%m', RequestUsage.date) == mo_str,
        ]
        monthly_where = [
            RequestUsageMonthly.year == year,
            RequestUsageMonthly.month == month,
        ]

        async def _month_query_user():
            d_rows = (await db.execute(
                select(RequestUsage.user_identity, RequestUsage.user_type, func.sum(RequestUsage.request_count).label("rc"))
                .where(*daily_where).group_by(RequestUsage.user_identity, RequestUsage.user_type)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(*monthly_where).group_by(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                k = (r.user_identity, r.user_type)
                combined[k] = combined.get(k, 0) + r.rc
            return sorted(
                [{"user_identity": k[0], "user_type": k[1], "request_count": v} for k, v in combined.items()],
                key=lambda x: -x["request_count"]
            )

        async def _month_query_model():
            d_rows = (await db.execute(
                select(RequestUsage.model, func.sum(RequestUsage.request_count).label("rc"))
                .where(*daily_where).group_by(RequestUsage.model)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.model, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(*monthly_where).group_by(RequestUsageMonthly.model)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                combined[r.model] = combined.get(r.model, 0) + r.rc
            return sorted(
                [{"model": m, "request_count": c} for m, c in combined.items()],
                key=lambda x: -x["request_count"]
            )

        if filter_user is not None:
            d_rows = (await db.execute(
                select(RequestUsage.model, func.sum(RequestUsage.request_count).label("rc"))
                .where(*daily_where, RequestUsage.user_identity == filter_user)
                .group_by(RequestUsage.model)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.model, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(*monthly_where, RequestUsageMonthly.user_identity == filter_user)
                .group_by(RequestUsageMonthly.model)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                combined[r.model] = combined.get(r.model, 0) + r.rc
            breakdown = sorted(
                [{"model": m, "request_count": c} for m, c in combined.items()],
                key=lambda x: -x["request_count"]
            )
            return {"window": window, "year": year, "month": month, "breakdown": breakdown}

        if filter_model is not None:
            d_rows = (await db.execute(
                select(RequestUsage.user_identity, RequestUsage.user_type, func.sum(RequestUsage.request_count).label("rc"))
                .where(*daily_where, RequestUsage.model == filter_model)
                .group_by(RequestUsage.user_identity, RequestUsage.user_type)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(*monthly_where, RequestUsageMonthly.model == filter_model)
                .group_by(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                k = (r.user_identity, r.user_type)
                combined[k] = combined.get(k, 0) + r.rc
            breakdown = sorted(
                [{"user_identity": k[0], "user_type": k[1], "request_count": v} for k, v in combined.items()],
                key=lambda x: -x["request_count"]
            )
            return {"window": window, "year": year, "month": month, "breakdown": breakdown}

        per_user = await _month_query_user()
        per_model = await _month_query_model()
        total_requests = sum(r["request_count"] for r in per_user)
        return {
            "window": window,
            "year": year,
            "month": month,
            "per_user": per_user,
            "per_model": per_model,
            "totals": {"requests": total_requests, "unique_users": len(per_user), "unique_models": len(per_model)},
        }

    # ------------------------------------------------------------------ #
    # All-time window — UNION daily rows + monthly rollups, no date filter
    # ------------------------------------------------------------------ #
    if window == "all":
        async def _all_query_user():
            d_rows = (await db.execute(
                select(RequestUsage.user_identity, RequestUsage.user_type, func.sum(RequestUsage.request_count).label("rc"))
                .group_by(RequestUsage.user_identity, RequestUsage.user_type)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .group_by(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                k = (r.user_identity, r.user_type)
                combined[k] = combined.get(k, 0) + r.rc
            return sorted(
                [{"user_identity": k[0], "user_type": k[1], "request_count": v} for k, v in combined.items()],
                key=lambda x: -x["request_count"]
            )

        async def _all_query_model():
            d_rows = (await db.execute(
                select(RequestUsage.model, func.sum(RequestUsage.request_count).label("rc"))
                .group_by(RequestUsage.model)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.model, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .group_by(RequestUsageMonthly.model)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                combined[r.model] = combined.get(r.model, 0) + r.rc
            return sorted(
                [{"model": m, "request_count": c} for m, c in combined.items()],
                key=lambda x: -x["request_count"]
            )

        if filter_user is not None:
            d_rows = (await db.execute(
                select(RequestUsage.model, func.sum(RequestUsage.request_count).label("rc"))
                .where(RequestUsage.user_identity == filter_user)
                .group_by(RequestUsage.model)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.model, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(RequestUsageMonthly.user_identity == filter_user)
                .group_by(RequestUsageMonthly.model)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                combined[r.model] = combined.get(r.model, 0) + r.rc
            breakdown = sorted(
                [{"model": m, "request_count": c} for m, c in combined.items()],
                key=lambda x: -x["request_count"]
            )
            return {"window": window, "breakdown": breakdown}

        if filter_model is not None:
            d_rows = (await db.execute(
                select(RequestUsage.user_identity, RequestUsage.user_type, func.sum(RequestUsage.request_count).label("rc"))
                .where(RequestUsage.model == filter_model)
                .group_by(RequestUsage.user_identity, RequestUsage.user_type)
            )).all()
            m_rows = (await db.execute(
                select(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type, func.sum(RequestUsageMonthly.request_count).label("rc"))
                .where(RequestUsageMonthly.model == filter_model)
                .group_by(RequestUsageMonthly.user_identity, RequestUsageMonthly.user_type)
            )).all()
            combined: dict = {}
            for r in list(d_rows) + list(m_rows):
                k = (r.user_identity, r.user_type)
                combined[k] = combined.get(k, 0) + r.rc
            breakdown = sorted(
                [{"user_identity": k[0], "user_type": k[1], "request_count": v} for k, v in combined.items()],
                key=lambda x: -x["request_count"]
            )
            return {"window": window, "breakdown": breakdown}

        per_user = await _all_query_user()
        per_model = await _all_query_model()
        total_requests = sum(r["request_count"] for r in per_user)
        return {
            "window": window,
            "per_user": per_user,
            "per_model": per_model,
            "totals": {"requests": total_requests, "unique_users": len(per_user), "unique_models": len(per_model)},
        }

    # ------------------------------------------------------------------ #
    # Daily-table windows: today, yesterday, 7d, 30d
    # ------------------------------------------------------------------ #
    base_where = _daily_where()

    if filter_user is not None or filter_model is not None:
        breakdown = await _exec_drilldown_query(base_where)
        return {"window": window, "breakdown": breakdown}

    per_user, per_model = await _exec_top_level_query(base_where)
    total_requests = sum(r["request_count"] for r in per_user)
    return {
        "window": window,
        "per_user": per_user,
        "per_model": per_model,
        "totals": {"requests": total_requests, "unique_users": len(per_user), "unique_models": len(per_model)},
    }


async def init_database():
    """Initialize the database and create tables asynchronously."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create tables asynchronously
    await create_tables_async()
    
    # Run auto-migrations for schema updates
    await _run_auto_migrations()
    
    print("Database initialized successfully!")


async def _run_auto_migrations():
    """Auto-migrate database schema for new columns and renamed provider types.
    
    This handles:
    1. Adding provider_credentials columns if missing
    2. Renaming provider_type 'openai_compatible' to 'custom' with default supported_apis
    """
    from sqlalchemy import text
    
    async with engine.begin() as conn:
        # Check if is_pending_approval column exists on users table
        try:
            result = await conn.execute(text("PRAGMA table_info(users)"))
            columns = [row[1] for row in result.fetchall()]
            if 'is_pending_approval' not in columns:
                logger.info("Auto-migration: Adding 'is_pending_approval' column to users")
                await conn.execute(text(
                    "ALTER TABLE users ADD COLUMN is_pending_approval BOOLEAN DEFAULT 0"
                ))
                logger.info("Auto-migration: 'is_pending_approval' column added successfully")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add is_pending_approval column: {e}")

        # Check if supported_apis column exists
        try:
            result = await conn.execute(text("PRAGMA table_info(provider_credentials)"))
            columns = [row[1] for row in result.fetchall()]

            if 'supported_apis' not in columns:
                logger.info("Auto-migration: Adding 'supported_apis' column to provider_credentials")
                await conn.execute(text(
                    "ALTER TABLE provider_credentials ADD COLUMN supported_apis TEXT DEFAULT '[\"openai\"]'"
                ))
                logger.info("Auto-migration: 'supported_apis' column added successfully")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add supported_apis column: {e}")

        try:
            result = await conn.execute(text("PRAGMA table_info(provider_credentials)"))
            columns = [row[1] for row in result.fetchall()]

            if 'azure_backend' not in columns:
                logger.info("Auto-migration: Adding 'azure_backend' column to provider_credentials")
                await conn.execute(text(
                    "ALTER TABLE provider_credentials ADD COLUMN azure_backend TEXT DEFAULT 'openai'"
                ))
                logger.info("Auto-migration: 'azure_backend' column added successfully")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add azure_backend column: {e}")
        
        # Rename provider_type 'openai_compatible' to 'custom' and set default supported_apis
        try:
            result = await conn.execute(text(
                "SELECT COUNT(*) FROM provider_credentials WHERE provider_type = 'openai_compatible'"
            ))
            count = result.scalar()
            if count and count > 0:
                logger.info(f"Auto-migration: Renaming {count} 'openai_compatible' providers to 'custom'")
                await conn.execute(text(
                    "UPDATE provider_credentials SET provider_type = 'custom', "
                    "supported_apis = '[\"openai\"]' "
                    "WHERE provider_type = 'openai_compatible'"
                ))
                logger.info("Auto-migration: Provider type rename completed")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not rename provider types: {e}")
        
        # Set default supported_apis for providers that have NULL
        try:
            await conn.execute(text(
                "UPDATE provider_credentials SET azure_backend = 'openai' "
                "WHERE provider_type = 'azure' AND (azure_backend IS NULL OR azure_backend = '')"
            ))
            await conn.execute(text(
                "UPDATE provider_credentials SET supported_apis = '[\"openai\"]' "
                "WHERE supported_apis IS NULL AND provider_type IN ('azure', 'google')"
            ))
            await conn.execute(text(
                "UPDATE provider_credentials SET supported_apis = '[\"openai\", \"anthropic\"]' "
                "WHERE supported_apis IS NULL AND provider_type = 'bedrock'"
            ))
            await conn.execute(text(
                "UPDATE provider_credentials SET supported_apis = '[\"openai\"]' "
                "WHERE supported_apis IS NULL AND provider_type = 'custom'"
            ))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not set default supported_apis: {e}")

        # Create user_rate_limits table if missing
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_rate_limits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                    rpm_limit INTEGER,
                    rpd_limit INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_by VARCHAR(50)
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_user_rate_limits_user_id ON user_rate_limits (user_id)"
            ))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not create user_rate_limits table: {e}")

        # Create global_rate_limits table and seed the singleton row
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS global_rate_limits (
                    id INTEGER PRIMARY KEY,
                    rpm_default INTEGER,
                    rpd_default INTEGER,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_by VARCHAR(50)
                )
            """))
            await conn.execute(text(
                "INSERT OR IGNORE INTO global_rate_limits (id, rpm_default, rpd_default) VALUES (1, NULL, NULL)"
            ))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not create global_rate_limits table: {e}")


def init_database_sync():
    """Initialize the database and create tables synchronously (for backward compatibility)."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Create tables
    create_tables()

    print("Database initialized successfully!")


# ---------------------------------------------------------------------------
# Model-group helpers
# ---------------------------------------------------------------------------

async def list_model_groups(db: AsyncSession, group_id: Optional[int] = None):
    """Return all model groups (with members loaded), or a single group if group_id given."""
    from sqlalchemy.orm import selectinload as _sil
    q = select(ModelGroup).options(_sil(ModelGroup.members))
    if group_id is not None:
        q = q.where(ModelGroup.id == group_id)
    result = await db.execute(q)
    rows = result.scalars().all()
    return rows[0] if group_id is not None and rows else (None if group_id is not None else rows)


async def create_model_group(
    db: AsyncSession, name: str, description: Optional[str],
    rpm_default: Optional[int], rpd_default: Optional[int], admin_username: str,
) -> ModelGroup:
    row = ModelGroup(
        name=name, description=description,
        rpm_default=rpm_default, rpd_default=rpd_default,
        updated_by=admin_username,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def update_model_group(
    db: AsyncSession, group_id: int, fields: dict, admin_username: str,
) -> Optional[ModelGroup]:
    result = await db.execute(select(ModelGroup).where(ModelGroup.id == group_id))
    row = result.scalar_one_or_none()
    if row is None:
        return None
    for k, v in fields.items():
        setattr(row, k, v)
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_model_group(db: AsyncSession, group_id: int) -> bool:
    result = await db.execute(select(ModelGroup).where(ModelGroup.id == group_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def set_group_members(db: AsyncSession, group_id: int, model_ids: list) -> list:
    """Replace the member list for a group. Returns list of ModelGroupMember rows."""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(ModelGroupMember).where(ModelGroupMember.group_id == group_id))
    new_members = [ModelGroupMember(group_id=group_id, model_id=mid) for mid in model_ids]
    db.add_all(new_members)
    await db.commit()
    return new_members


async def get_model_group_limits(db: AsyncSession, group_id: int) -> Optional[ModelGroup]:
    result = await db.execute(select(ModelGroup).where(ModelGroup.id == group_id))
    return result.scalar_one_or_none()


async def update_model_group_limits(
    db: AsyncSession, group_id: int, rpm_default: Optional[int], rpd_default: Optional[int],
    admin_username: str,
) -> Optional[ModelGroup]:
    return await update_model_group(
        db, group_id,
        {"rpm_default": rpm_default, "rpd_default": rpd_default},
        admin_username,
    )


async def get_user_group_rate_limit(
    db: AsyncSession, user_id: int, group_id: int,
) -> Optional[UserModelGroupRateLimit]:
    result = await db.execute(
        select(UserModelGroupRateLimit).where(
            UserModelGroupRateLimit.user_id == user_id,
            UserModelGroupRateLimit.group_id == group_id,
        )
    )
    return result.scalar_one_or_none()


async def list_user_group_rate_limits(
    db: AsyncSession, group_id: int, user_id: Optional[int] = None,
):
    q = select(UserModelGroupRateLimit).where(UserModelGroupRateLimit.group_id == group_id)
    if user_id is not None:
        q = q.where(UserModelGroupRateLimit.user_id == user_id)
    result = await db.execute(q)
    return result.scalars().all()


async def upsert_user_group_rate_limit(
    db: AsyncSession, user_id: int, group_id: int,
    rpm: Optional[int], rpd: Optional[int],
    admin_username: str, fields_set: set,
) -> UserModelGroupRateLimit:
    row = await get_user_group_rate_limit(db, user_id, group_id)
    if row is None:
        row = UserModelGroupRateLimit(user_id=user_id, group_id=group_id)
        db.add(row)
    if "rpm_limit" in fields_set:
        row.rpm_limit = rpm
    if "rpd_limit" in fields_set:
        row.rpd_limit = rpd
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_user_group_rate_limit(db: AsyncSession, user_id: int, group_id: int) -> bool:
    row = await get_user_group_rate_limit(db, user_id, group_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True

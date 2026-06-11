"""Database connection and operations for authentication."""

import os
import secrets
import logging
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, selectinload
from sqlalchemy.future import select
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from .models import Base, User, APIKey, ModelConfiguration, ProviderCredentials, OAuthUser, ResponseProviderMapping, RequestUsage, UserRateLimit, GlobalRateLimit
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
                           picture: Optional[str] = None, raw_data: Optional[str] = None) -> tuple[User, OAuthUser]:
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
            oauth_sub=provider_user_id
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
    """Bulk upsert usage rows, incrementing request_count on conflict."""
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


async def prune_request_usage(days: int = 30) -> None:
    """Delete usage rows older than `days` days."""
    from sqlalchemy import delete
    from datetime import date, timedelta
    cutoff = date.today() - timedelta(days=days)
    async with AsyncSessionLocal() as db:
        try:
            await db.execute(delete(RequestUsage).where(RequestUsage.date < cutoff))
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to prune request usage: {e}")
            raise


async def get_usage_aggregates(
    db: AsyncSession,
    group_by: str = "user",
    filter_user: Optional[str] = None,
    filter_model: Optional[str] = None,
) -> dict:
    """Return aggregated usage data for the last 30 days.

    group_by: 'user' or 'model'
    filter_user: when set, group by model for this user (drill-down)
    filter_model: when set, group by user for this model (drill-down)
    """
    from sqlalchemy import func, text
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=29)

    base_q = select(RequestUsage).where(RequestUsage.date >= cutoff)

    if filter_user is not None:
        # Drill-down: split by model for a specific user
        q = (
            select(
                RequestUsage.model,
                func.sum(RequestUsage.request_count).label("request_count"),
            )
            .where(RequestUsage.date >= cutoff)
            .where(RequestUsage.user_identity == filter_user)
            .group_by(RequestUsage.model)
            .order_by(func.sum(RequestUsage.request_count).desc(), RequestUsage.model)
        )
        rows = (await db.execute(q)).all()
        result = [{"model": r.model, "request_count": r.request_count} for r in rows]
        shown_since = await _usage_shown_since(db, cutoff)
        return {"shown_since": shown_since, "breakdown": result}

    if filter_model is not None:
        # Drill-down: split by user for a specific model
        q = (
            select(
                RequestUsage.user_identity,
                RequestUsage.user_type,
                func.sum(RequestUsage.request_count).label("request_count"),
            )
            .where(RequestUsage.date >= cutoff)
            .where(RequestUsage.model == filter_model)
            .group_by(RequestUsage.user_identity, RequestUsage.user_type)
            .order_by(func.sum(RequestUsage.request_count).desc(), RequestUsage.user_identity)
        )
        rows = (await db.execute(q)).all()
        result = [
            {"user_identity": r.user_identity, "user_type": r.user_type, "request_count": r.request_count}
            for r in rows
        ]
        shown_since = await _usage_shown_since(db, cutoff)
        return {"shown_since": shown_since, "breakdown": result}

    # Top-level: group by user or model
    if group_by == "model":
        q = (
            select(
                RequestUsage.model,
                func.sum(RequestUsage.request_count).label("request_count"),
            )
            .where(RequestUsage.date >= cutoff)
            .group_by(RequestUsage.model)
            .order_by(func.sum(RequestUsage.request_count).desc(), RequestUsage.model)
        )
        rows = (await db.execute(q)).all()
        per_model = [{"model": r.model, "request_count": r.request_count} for r in rows]
    else:
        per_model = None

    q_user = (
        select(
            RequestUsage.user_identity,
            RequestUsage.user_type,
            func.sum(RequestUsage.request_count).label("request_count"),
        )
        .where(RequestUsage.date >= cutoff)
        .group_by(RequestUsage.user_identity, RequestUsage.user_type)
        .order_by(func.sum(RequestUsage.request_count).desc(), RequestUsage.user_identity)
    )
    user_rows = (await db.execute(q_user)).all()
    per_user = [
        {"user_identity": r.user_identity, "user_type": r.user_type, "request_count": r.request_count}
        for r in user_rows
    ]

    if per_model is None:
        q_model = (
            select(
                RequestUsage.model,
                func.sum(RequestUsage.request_count).label("request_count"),
            )
            .where(RequestUsage.date >= cutoff)
            .group_by(RequestUsage.model)
            .order_by(func.sum(RequestUsage.request_count).desc(), RequestUsage.model)
        )
        model_rows = (await db.execute(q_model)).all()
        per_model = [{"model": r.model, "request_count": r.request_count} for r in model_rows]

    total_requests = sum(r["request_count"] for r in per_user)
    shown_since = await _usage_shown_since(db, cutoff)

    return {
        "shown_since": shown_since,
        "per_user": per_user,
        "per_model": per_model,
        "totals": {
            "requests": total_requests,
            "unique_users": len(per_user),
            "unique_models": len(per_model),
        },
    }


async def _usage_shown_since(db: AsyncSession, cutoff) -> Optional[str]:
    from sqlalchemy import func
    q = select(func.min(RequestUsage.date)).where(RequestUsage.date >= cutoff)
    result = await db.execute(q)
    min_date = result.scalar_one_or_none()
    return str(min_date) if min_date else str(cutoff)


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

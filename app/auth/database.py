"""Database connection and operations for authentication."""

import os
import json
import secrets
import hashlib
import functools
import logging
from pathlib import Path
from sqlalchemy import create_engine, event, false, String
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, selectinload
from sqlalchemy.future import select
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from .models import Base, User, APIKey, ModelConfiguration, ModelAlias, ProviderCredentials, OAuthUser, ResponseProviderMapping, RequestUsage, RequestUsageHourly, RequestUsageMonthly, UserRateLimit, GlobalRateLimit, ModelGroup, ModelGroupMember, UserModelGroupRateLimit, InstanceGroup, InstanceGroupMember, UserInstanceGroupRateLimit, UserModelAccessPolicy, UserModelAccessException, WebSearchSettings
from app.providers.azure_deployments import serialize_azure_deployments

# Initialize logger
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/llm_proxy.db")
SYNC_DATABASE_URL = DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")

# Database timeout and retry configuration
DATABASE_BUSY_TIMEOUT = int(os.getenv("DATABASE_BUSY_TIMEOUT", "5"))  # seconds
# Async connection pool sizing. SQLite has a single writer, so raising these only
# lets more sessions queue on the write lock; tune with care.
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))  # seconds
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
        
        # Create the directory owner-only (0o700). The SQLite file holds password
        # hashes, API-key hashes and provider secrets, so it must not be world-readable.
        old_umask = os.umask(0o077)
        try:
            os.makedirs(db_dir, mode=0o700, exist_ok=True)
        finally:
            # Restore the process-wide umask so we don't leak permissive defaults
            # to every file the process later creates.
            os.umask(old_umask)


# Ensure directory exists before creating engines
ensure_db_directory()

# Create engines with proper configuration for SQLite async
# - pool_pre_ping: Verify connections before use
# - pool_recycle: Recycle connections to prevent stale ones  
# - connect_args: Enable check_same_thread=False for SQLite + async
engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_timeout=DATABASE_POOL_TIMEOUT,
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
    # Enforce foreign keys so ON DELETE CASCADE actually fires (off by default in SQLite).
    cursor.execute("PRAGMA foreign_keys=ON")
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
    pbkdf2_sha256__rounds=600000  # OWASP guidance for PBKDF2-HMAC-SHA256; older hashes re-hash on next login
)


# Database retry decorator for handling lock contention
def with_db_retry(max_attempts: int = DB_RETRY_MAX_ATTEMPTS, backoff_ms: int = DB_RETRY_BACKOFF_MS):
    """Decorator to retry database operations with exponential backoff on lock errors."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from sqlalchemy.exc import OperationalError
            from sqlalchemy.ext.asyncio import AsyncSession
            import asyncio

            # Locate the session (if any) so we can roll it back between attempts;
            # a failed flush/commit leaves it in a poisoned state that would raise
            # PendingRollbackError on the next call otherwise.
            session = next(
                (a for a in args if isinstance(a, AsyncSession)),
                kwargs.get("db"),
            )

            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    if "database is locked" in str(e).lower():
                        last_error = e
                        if attempt < max_attempts - 1:
                            if isinstance(session, AsyncSession):
                                try:
                                    await session.rollback()
                                except Exception:
                                    pass
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


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest used to store and look up API keys.

    Deterministic so the hash stays indexable; the plaintext key is never
    persisted (shown once at creation time only).
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


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


def is_reserved_username(username: str) -> bool:
    """True if the name belongs to the config-based admin account.

    The admin is not a row in `users`, so uniqueness checks against that table miss
    it entirely — but admin traffic *is* recorded under this name (see
    _update_tracking_identity), and both usage and RPD are keyed by the username
    string. A second account holding the name would share the admin's daily quota,
    and a later rename would carry the admin's whole usage history away with it
    (see rename_usage_identity). Every path that assigns a username must reject it.
    """
    # Imported lazily: app.auth.admin imports from this module.
    from app.auth.admin import get_admin_username, is_admin_enabled
    return is_admin_enabled() and username == get_admin_username()


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
    if not user or not user.hashed_password:
        # Run a dummy verification so a missing/passwordless account takes the same
        # time as a real one, closing the username-enumeration timing side channel.
        pwd_context.dummy_verify()
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
                           is_pending: bool = False, email_verified: bool = False) -> tuple[User, OAuthUser]:
    """Create a new user with OAuth authentication.

    An existing local account is auto-linked by email ONLY when the provider
    asserts the email is verified (email_verified=True). Otherwise linking is
    refused to prevent account takeover via an unverified attacker-controlled email.
    """
    # Generate a unique username from email or name
    username = email.split('@')[0] if '@' in email else name.lower().replace(' ', '_')

    # Check if username exists and make it unique if needed. The admin name is not
    # in `users` but is just as taken (see is_reserved_username), so suffix past it too.
    base_username = username
    counter = 1
    while is_reserved_username(username) or await get_user_by_username(db, username):
        username = f"{base_username}_{counter}"
        counter += 1

    # Check if email exists
    existing_user_by_email = await get_user_by_email(db, email)
    if existing_user_by_email:
        # Only auto-link to an existing local account when the provider has
        # verified ownership of the email; otherwise refuse (email is unique, so
        # a separate account with the same email cannot be created either).
        if not email_verified:
            raise ValueError(
                "An account with this email already exists and the provider has not "
                "verified the email address, so it cannot be linked automatically."
            )
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
    """Get API key from database. Keys are stored as SHA-256 hashes.

    Only resolves when both the key and its owning user are active, so
    deactivating a user immediately disables all of their API keys.
    """
    key_hash = hash_api_key(api_key)
    result = await db.execute(
        select(APIKey)
        .join(User, APIKey.user_id == User.id)
        .where(
            APIKey.api_key == key_hash,
            APIKey.is_active == True,
            User.is_active == True,
        )
    )
    return result.scalar_one_or_none()


async def update_api_key_last_used(db: AsyncSession, api_key: str):
    """Update the last used timestamp for an API key (accepts the plaintext key)."""
    key_hash = hash_api_key(api_key)
    result = await db.execute(select(APIKey).where(APIKey.api_key == key_hash))
    db_api_key = result.scalar_one_or_none()
    if db_api_key:
        db_api_key.last_used = datetime.utcnow()
        await db.commit()


@with_db_retry()
async def create_api_key(db: AsyncSession, user_id: int, name: str) -> APIKey:
    """Create a new API key for a user.

    Only the SHA-256 hash and a short display prefix are persisted. The returned
    object carries the plaintext key in-memory so the caller can show it once;
    it is never stored and cannot be recovered afterwards.
    """
    plaintext = generate_api_key()
    db_api_key = APIKey(
        user_id=user_id,
        api_key=hash_api_key(plaintext),
        key_prefix=plaintext[:8],
        name=name
    )
    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)
    # Surface the plaintext once for the create response (in-memory only, not persisted).
    # Detach first so this override can never be flushed back over the stored hash.
    db.expunge(db_api_key)
    db_api_key.api_key = plaintext
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
        # Invalidate the cache entry before soft-deleting. The DB (and thus this
        # row) holds the SHA-256 hash while the cache is keyed by plaintext, so
        # evict via the hash-aware helper.
        from .cache import auth_cache
        auth_cache.invalidate_api_key_by_hash(db_api_key.api_key)
        
        db_api_key.is_active = False
        await db.commit()
        return True
    return False


async def update_user_profile(db: AsyncSession, user_id: int, username: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
    """Update user profile information.

    A username change also relabels that user's request usage rows (see
    rename_usage_identity). Usage is keyed by user_id, so nothing moves and the daily
    quota is untouched; the relabel only keeps the tables reading the current name.
    This is the single chokepoint for renames — both the admin path and the
    self-service profile path go through here.
    """
    # Imported lazily: request_tracker and rate_limit import from this module.
    from contextlib import AsyncExitStack
    from app.request_tracker import request_tracker
    from app.rate_limit import rate_limit_tracker
    from app.auth.cache import auth_cache

    user = await get_user_by_id(db, user_id)
    if not user:
        return None

    old_username = user.username
    renaming = username is not None and username != old_username

    async with AsyncExitStack() as stack:
        if renaming:
            # A rename rewrites usage in both places it lives — the DB rows and the
            # counts still buffered in the tracker. Hold off the flush across the
            # whole sequence: one landing between the two halves writes its
            # pre-rename snapshot back under the old name, re-creating the orphan
            # rows the SQL just cleaned up and double-billing those requests.
            await stack.enter_async_context(request_tracker.pause_flush())

        try:
            if username is not None:
                # Check if username is already taken by another user
                existing_user = await get_user_by_username(db, username)
                if existing_user and existing_user.id != user_id:
                    raise ValueError("Username already taken")
                # The check above cannot see the config-based admin; taking that
                # name would merge the admin's usage history into this user's rows.
                if is_reserved_username(username):
                    raise ValueError("Username already taken")
                if renaming:
                    # Drain buffered counts into the DB first so the backfill below
                    # covers them. Anything buffered after this point is relocated
                    # in-memory once the transaction commits. Best-effort: on failure
                    # the counts simply stay buffered and are moved by that later step.
                    try:
                        await request_tracker.flush_pending()
                    except Exception as e:
                        logger.warning(f"Usage flush before rename of '{old_username}' failed: {e}")
                user.username = username

            if email is not None:
                # Check if email is already taken by another user
                existing_user = await get_user_by_email(db, email)
                if existing_user and existing_user.id != user_id:
                    raise ValueError("Email already in use")
                user.email = email

            if renaming:
                # Same transaction as the username change: both land or neither does.
                # Usage is keyed by user_id, so this only rewrites the display label.
                await rename_usage_identity(db, user_id, username)

            await db.commit()
            await db.refresh(user)
        except Exception as e:
            await db.rollback()
            raise e

        if renaming:
            # Post-commit, still inside the flush pause: relocate counts that never
            # reached the DB, and drop cache entries keyed by either name.
            try:
                await request_tracker.rename_identity(user_id, username)
            except Exception as e:
                logger.warning(f"In-memory usage relabel of '{old_username}' failed: {e}")
            rate_limit_tracker.invalidate_identity(old_username)
            rate_limit_tracker.invalidate_identity(username)
            # The snapshot's identity->pool index is keyed by username, so until it
            # is rebuilt a lookup by the new name finds no pool -- and the next
            # invalidate_identity() for it would leave the pool's RPD caches standing.
            try:
                await rate_limit_tracker.refresh_now()
            except Exception as e:
                logger.warning(f"Rate-limit snapshot refresh after rename failed: {e}")
            auth_cache.invalidate_user(old_username)
            auth_cache.invalidate_user(username)
            # Cached API keys carry the owner's username, and that is the identity
            # recorded for API-key traffic (see _update_tracking_identity). The 30s
            # validity refresh only re-checks is_active, so without this the old name
            # keeps being written for the length of the key TTL — recreating the rows
            # the rename just moved, and billing RPD to a name with no rows left.
            auth_cache.invalidate_user_api_keys(user_id)

    return user


async def update_user_password(db: AsyncSession, user_id: int, current_password: str, new_password: str) -> bool:
    """Update user password after verifying current password."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False

    # OAuth-only users have no password to verify/change.
    if not user.hashed_password:
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


async def get_websearch_settings(db: AsyncSession) -> Optional[WebSearchSettings]:
    """Return the singleton web search settings row (id=1), or None if never saved."""
    result = await db.execute(select(WebSearchSettings).where(WebSearchSettings.id == 1))
    return result.scalar_one_or_none()


async def upsert_websearch_settings(
    db: AsyncSession, values: dict, admin_username: str
) -> WebSearchSettings:
    """Create or update the web search settings singleton.

    ``values`` maps column names to already-validated values; list values are
    stored as JSON.
    """
    row = await get_websearch_settings(db)
    if row is None:
        row = WebSearchSettings(id=1)
        db.add(row)
    for key, value in values.items():
        if isinstance(value, list):
            value = json.dumps(value)
        setattr(row, key, value)
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


# ==================== Per-User Model Access ====================

async def get_user_model_policy(db: AsyncSession, user_id: int) -> Optional[UserModelAccessPolicy]:
    """Return the per-user model-access default policy, or None (⇒ allow-all)."""
    result = await db.execute(
        select(UserModelAccessPolicy).where(UserModelAccessPolicy.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def upsert_user_model_policy(
    db: AsyncSession,
    user_id: int,
    mode: str,
    admin_username: str,
) -> UserModelAccessPolicy:
    """Create or update a user's model-access policy mode (default|allow|deny|custom)."""
    row = await get_user_model_policy(db, user_id)
    if row is None:
        row = UserModelAccessPolicy(user_id=user_id)
        db.add(row)
    row.mode = mode
    # Keep the legacy boolean roughly in sync for any old readers.
    row.default_allow = mode in ("allow", "default")
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_user_model_policy(db: AsyncSession, user_id: int) -> bool:
    """Remove a user's policy row; user reverts to the allow-all default."""
    row = await get_user_model_policy(db, user_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def list_user_model_exceptions(db: AsyncSession, user_id: int) -> List[UserModelAccessException]:
    """Return all explicit per-model access exceptions for a user."""
    result = await db.execute(
        select(UserModelAccessException).where(UserModelAccessException.user_id == user_id)
    )
    return list(result.scalars().all())


async def get_user_model_exception(
    db: AsyncSession, user_id: int, model_id: str
) -> Optional[UserModelAccessException]:
    """Return the explicit exception for (user, model), or None."""
    result = await db.execute(
        select(UserModelAccessException).where(
            UserModelAccessException.user_id == user_id,
            UserModelAccessException.model_id == model_id,
        )
    )
    return result.scalar_one_or_none()


async def upsert_user_model_exception(
    db: AsyncSession,
    user_id: int,
    model_id: str,
    is_allowed: bool,
    admin_username: str,
) -> UserModelAccessException:
    """Create or update an explicit per-model access exception for a user."""
    row = await get_user_model_exception(db, user_id, model_id)
    if row is None:
        row = UserModelAccessException(user_id=user_id, model_id=model_id)
        db.add(row)
    row.is_allowed = is_allowed
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_user_model_exception(db: AsyncSession, user_id: int, model_id: str) -> bool:
    """Remove an explicit exception; the (user, model) reverts to policy default."""
    row = await get_user_model_exception(db, user_id, model_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def set_user_model_exceptions_bulk(
    db: AsyncSession,
    user_id: int,
    overrides: Dict[str, bool],
    admin_username: str,
    replace: bool = False,
) -> None:
    """Upsert many per-model overrides in one transaction.

    When replace=True, clears all existing exceptions for the user first (used when
    seeding a fresh 'custom' baseline from allow/deny). Otherwise merges in place.
    """
    if replace:
        existing = await list_user_model_exceptions(db, user_id)
        for row in existing:
            await db.delete(row)
        await db.flush()

    now = datetime.utcnow()
    if replace:
        # Fresh slate: insert directly, no per-row lookup needed.
        for model_id, is_allowed in overrides.items():
            db.add(UserModelAccessException(
                user_id=user_id,
                model_id=model_id,
                is_allowed=is_allowed,
                updated_by=admin_username,
                updated_at=now,
            ))
    else:
        for model_id, is_allowed in overrides.items():
            row = await get_user_model_exception(db, user_id, model_id)
            if row is None:
                row = UserModelAccessException(user_id=user_id, model_id=model_id)
                db.add(row)
            row.is_allowed = is_allowed
            row.updated_by = admin_username
            row.updated_at = now

    await db.commit()


async def get_all_user_model_policies(db: AsyncSession) -> List[UserModelAccessPolicy]:
    """Return every user's model-access policy row (for cache warm-up)."""
    result = await db.execute(select(UserModelAccessPolicy))
    return list(result.scalars().all())


async def get_all_user_model_exceptions(db: AsyncSession) -> List[UserModelAccessException]:
    """Return every per-model access exception across all users (for cache warm-up)."""
    result = await db.execute(select(UserModelAccessException))
    return list(result.scalars().all())


async def permanently_delete_user(db: AsyncSession, user_id: int) -> bool:
    """Permanently delete a user and all associated data from the database.

    NOTE: users.id is a plain INTEGER PRIMARY KEY (no AUTOINCREMENT), so SQLite may
    reuse a deleted user's rowid for a future user. Adding AUTOINCREMENT is a schema
    migration deliberately left out of scope here; deleting dependent rows explicitly
    below prevents a reused id from inheriting stale overrides/policies.
    """
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

        # Explicitly delete all other dependent rows. SQLite does not reliably fire
        # ON DELETE CASCADE (and rows are keyed by user_id), so remove them by hand
        # to avoid orphaned rate-limit / model-access records.
        from sqlalchemy import delete as _sql_delete
        for _model in (
            UserRateLimit,
            UserModelGroupRateLimit,
            UserInstanceGroupRateLimit,
            UserModelAccessPolicy,
            UserModelAccessException,
        ):
            await db.execute(_sql_delete(_model).where(_model.user_id == user_id))

        # Usage rows are keyed by user_id and go with the account, so a later user
        # who takes this username inherits neither the history nor today's quota.
        # Callers hold request_tracker.pause_flush() around this and drop the
        # buffered counts afterwards; see permanently_delete_user_endpoint.
        await purge_user_usage(db, user_id)

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


async def get_all_model_aliases(db: AsyncSession) -> List[ModelAlias]:
    """Return all model aliases in stable display order."""
    result = await db.execute(select(ModelAlias).order_by(ModelAlias.alias))
    return list(result.scalars().all())


async def get_model_alias(db: AsyncSession, alias: str) -> Optional[ModelAlias]:
    """Return an alias by its exact client-facing name."""
    result = await db.execute(select(ModelAlias).where(ModelAlias.alias == alias))
    return result.scalar_one_or_none()


async def upsert_model_alias(
    db: AsyncSession, alias: str, target_model_id: str, enabled: bool, apis
) -> ModelAlias:
    """Create or update a model alias."""
    row = await get_model_alias(db, alias)
    if row is None:
        row = ModelAlias(alias=alias)
        db.add(row)
    row.target_model_id = target_model_id
    row.enabled = enabled
    row.apis = json.dumps(list(apis))
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_model_alias(db: AsyncSession, alias: str) -> bool:
    """Delete a model alias by name."""
    row = await get_model_alias(db, alias)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


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
    is_enabled: bool = True,
    commit: bool = True,
) -> ModelConfiguration:
    """Create or update a model configuration.

    With ``commit=False`` the change is only staged on the session, so bulk
    callers can write many rows in one transaction (one SQLite write-lock
    acquisition) and commit once themselves.
    """
    existing = await get_model_configuration(db, model_id)
    
    if existing:
        existing.provider_key = provider_key
        existing.model_name = model_name
        existing.is_enabled = is_enabled
        existing.updated_at = datetime.utcnow()
        if commit:
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
        if commit:
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
            discovery_api_version=kwargs.get('discovery_api_version', old_credentials.discovery_api_version),
            azure_backend=kwargs.get('azure_backend', old_credentials.azure_backend),
            region=kwargs.get('region', old_credentials.region),
            access_key_id=kwargs.get('access_key_id', old_credentials.access_key_id),
            secret_access_key=kwargs.get('secret_access_key', old_credentials.secret_access_key),
            base_url=kwargs.get('base_url', old_credentials.base_url),
            deployments_json=kwargs.get('deployments_json', old_credentials.deployments_json),
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
    db: AsyncSession, response_id: str, provider_key: str, model_name: str = None,
    user_id: int = None
) -> ResponseProviderMapping:
    """Store a response_id -> provider mapping for Responses API routing.

    ``user_id`` records the creating user so retrieve/delete/cancel/input_items
    can enforce ownership; it is None for admin-created responses.
    """
    try:
        mapping = ResponseProviderMapping(
            response_id=response_id,
            provider_key=provider_key,
            model_name=model_name,
            user_id=user_id
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


def _usage_upsert(table, rows: list[dict], index_elements: list[str]):
    """Build one increment-on-conflict bulk upsert against a usage table.

    The conflict target must name the table's unique constraint columns exactly, in
    order; SQLite rejects any other target. user_identity and user_type are labels
    outside the key, so the latest flush wins them: after a rename the very next flush
    relabels the rows it touches, and the relabel in rename_usage_identity covers the
    rest.
    """
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    stmt = sqlite_insert(table).values(rows)
    return stmt.on_conflict_do_update(
        index_elements=index_elements,
        set_={
            "request_count": table.request_count + stmt.excluded.request_count,
            "user_identity": stmt.excluded.user_identity,
            "user_type": stmt.excluded.user_type,
        },
    )


USAGE_DAILY_KEY = ["date", "user_id", "model", "server", "pool_id"]
USAGE_HOURLY_KEY = ["date", "hour", "user_id", "model", "server", "pool_id"]
USAGE_MONTHLY_KEY = ["year", "month", "user_id", "model", "server", "pool_id"]


async def flush_usage_rows(hourly_rows: list[dict], daily_rows: list[dict]) -> None:
    """Write both usage tables in ONE transaction, incrementing on conflict.

    The two writes have to be atomic. request_tracker subtracts its buffer only after
    this returns, so a failure that committed one table and raised on the other would
    leave the buffer unsubtracted, and the next cycle would re-add the half that did
    land -- inflating it permanently and silently splitting the 'today' chart (which
    reads hourly) from the 'today' table (which reads daily).
    """
    if not hourly_rows and not daily_rows:
        return
    async with AsyncSessionLocal() as db:
        try:
            if hourly_rows:
                await db.execute(_usage_upsert(RequestUsageHourly, hourly_rows, USAGE_HOURLY_KEY))
            if daily_rows:
                await db.execute(_usage_upsert(RequestUsage, daily_rows, USAGE_DAILY_KEY))
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to flush request usage: {e}")
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


async def purge_stale_pool_rows() -> None:
    """Delete request-pool ledger and carry rows from days before today.

    Both tables are day-scoped and every read filters on usage_date == local_today(),
    so stale rows are already inert; this only keeps them from growing without bound.
    Runs on the same schedule as rollup_to_monthly().
    """
    from sqlalchemy import delete
    from app import time_utils
    from app.auth.models import RequestPoolLedger, UserRpdCarry

    today = time_utils.local_today()
    async with AsyncSessionLocal() as db:
        try:
            for model in (RequestPoolLedger, UserRpdCarry):
                await db.execute(delete(model).where(model.usage_date < today))
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Pool ledger/carry purge failed: {e}")


def _month_bounds(year: int, month: int):
    """(first_day, last_day) of a calendar month, for index-friendly date range filters."""
    from datetime import date, timedelta
    first = date(year, month, 1)
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return first, nxt - timedelta(days=1)


async def rollup_to_monthly() -> None:
    """Roll up fully-aged months from request_usage into request_usage_monthly.

    A month is eligible when its last calendar day is at least 30 days before today,
    ensuring we never roll up a partially-complete month. The rollup groups on the
    monthly key -- (user_id, model, server, pool_id) -- so pool attribution survives
    it exactly; user_identity / user_type ride along as labels.
    """
    from sqlalchemy import func, delete
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

                first_day, last_day = _month_bounds(year, month)
                in_month = [RequestUsage.date >= first_day, RequestUsage.date <= last_day]

                agg_q = select(
                    RequestUsage.user_id,
                    RequestUsage.pool_id,
                    func.max(RequestUsage.user_identity).label("user_identity"),
                    func.max(RequestUsage.user_type).label("user_type"),
                    RequestUsage.model,
                    RequestUsage.server,
                    func.sum(RequestUsage.request_count).label("request_count"),
                ).where(*in_month).group_by(
                    RequestUsage.user_id,
                    RequestUsage.pool_id,
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
                        "user_id": r.user_id,
                        "pool_id": r.pool_id,
                        "user_identity": r.user_identity,
                        "user_type": r.user_type,
                        "model": r.model,
                        "server": r.server,
                        "request_count": r.request_count,
                    }
                    for r in agg_rows
                ]

                await db.execute(_usage_upsert(RequestUsageMonthly, monthly_rows, USAGE_MONTHLY_KEY))

                # Delete the source daily rows for this month
                await db.execute(delete(RequestUsage).where(*in_month))
                await db.commit()
                logger.info(f"Rolled up {len(monthly_rows)} groups for {year}-{month:02d} into monthly table")

        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to roll up monthly usage: {e}")
            raise


# The three usage tables, each keyed by user_id (+ its own time columns). Every
# maintenance operation that touches "all of a user's usage" has to hit all three.
_USAGE_TABLES = (RequestUsage, RequestUsageHourly, RequestUsageMonthly)


async def rename_usage_identity(db: AsyncSession, user_id: int, new: str) -> dict[str, int]:
    """Relabel every usage row of ``user_id`` with the new username.

    Usage is keyed by user_id, so a rename moves nothing and can never collide: this
    only rewrites the user_identity display label so the tables read the current name.
    Runs in the caller's transaction and never commits, so the username change and the
    relabel succeed or fail together.

    Returns a {table: rows_relabelled} map for logging.
    """
    from sqlalchemy import update

    relabelled: dict[str, int] = {}
    for table in _USAGE_TABLES:
        result = await db.execute(
            update(table)
            .where(table.user_id == user_id, table.user_identity != new)
            .values(user_identity=new)
        )
        if result.rowcount:
            relabelled[table.__tablename__] = result.rowcount
    if relabelled:
        logger.info(f"Relabelled usage rows of user {user_id} to '{new}': {relabelled}")
    return relabelled


# Which column each purge/count axis binds against. The axis picks one of three column
# literals; the value itself is always a bound parameter.
_USAGE_AXIS_COLUMNS = {"user": "user_id", "model": "model", "pool": "pool_id"}


async def delete_usage_records(db: AsyncSession, axis: str, value) -> dict[str, int]:
    """Delete every usage row for one user (axis='user', value=user_id), one model
    (axis='model', value=model id) or one pool (axis='pool', value=pool_id).

    Usage is spread across three tables with different retention -- hourly (~48h),
    daily, and the monthly rollup that is kept forever -- so a purge has to hit all
    three or the data reappears the moment the admin widens the time window.

    Runs in the caller's transaction and never commits, letting the tests drive it
    against any database.

    Side effect worth knowing: RPD is a SUM over today's request_usage rows, so deleting
    a user's or a pool's rows resets the corresponding daily quota.

    Returns a {table: rows_deleted} map for logging.
    """
    from sqlalchemy import delete

    column = _USAGE_AXIS_COLUMNS[axis]
    deleted: dict[str, int] = {}

    for table in _USAGE_TABLES:
        result = await db.execute(
            delete(table).where(getattr(table, column) == value)
        )
        deleted[table.__tablename__] = result.rowcount or 0

    if any(deleted.values()):
        logger.info(f"Deleted usage rows for {axis} '{value}': {deleted}")
    return deleted


async def purge_user_usage(db: AsyncSession, user_id: int) -> dict[str, int]:
    """Delete all usage of a user being permanently deleted. Same transaction as the caller.

    Rows are keyed by user_id, so this is what stops a later account that takes the
    same username from inheriting the history -- and, if it happens the same day, the
    consumed daily quota.
    """
    return await delete_usage_records(db, "user", user_id)


async def count_usage_requests(db: AsyncSession, axis: str, value) -> int:
    """Return how many requests are recorded for one user, model or pool.

    Sums request_count over the daily and monthly tables only. rollup_to_monthly
    deletes the daily rows it aggregates, so those two are disjoint and together
    cover all of time; the hourly table is a second copy of the last ~48h of daily
    and would double-count. Summing rowcounts across all three instead (as a delete
    result does) counts the same traffic up to three times.
    """
    from sqlalchemy import func

    column = _USAGE_AXIS_COLUMNS[axis]
    total = 0
    for table in (RequestUsage, RequestUsageMonthly):
        result = await db.execute(
            select(func.coalesce(func.sum(table.request_count), 0))
            .where(getattr(table, column) == value)
        )
        total += result.scalar_one() or 0
    return int(total)


async def usage_user_ids_for_pool(db: AsyncSession, pool_id: int) -> List[int]:
    """Every user_id that has rows stamped with this pool, current member or not.

    Membership is the wrong source for "who is affected if this pool's usage is
    purged": rows carry the pool the sender was in when the request completed, so a
    user who has since left still owns rows here. Settling only today's members would
    leave that user's ledger charging traffic the purge just deleted.
    """
    ids: set = set()
    for table in _USAGE_TABLES:
        result = await db.execute(
            select(table.user_id).where(table.pool_id == pool_id).distinct()
        )
        ids.update(int(uid) for uid in result.scalars().all() if uid is not None)
    return sorted(ids)


async def usage_has_admin_label(db: AsyncSession, label: str) -> bool:
    """True if any usage row is the config admin's under this display name.

    The admin has no `users` row, so its rows keep whatever ADMIN_USERNAME was in force
    when they were written and nothing relabels them afterwards. Without this, renaming
    or disabling the admin turns its history into a row the UI can neither open nor
    delete: the label no longer resolves through is_reserved_username, and there is no
    account to fall back to.
    """
    from app.auth.models import ADMIN_USAGE_USER_ID

    for table in _USAGE_TABLES:
        result = await db.execute(
            select(table.user_id)
            .where(table.user_id == ADMIN_USAGE_USER_ID, table.user_identity == label)
            .limit(1)
        )
        if result.first() is not None:
            return True
    return False


async def get_usage_earliest_date(db: AsyncSession, filter_user_id: Optional[int] = None) -> Optional[str]:
    """Return the earliest date for which any usage data exists, as an ISO string (YYYY-MM-DD).

    Checks the daily table first (exact dates), then falls back to the monthly rollup
    (returns the first day of the earliest year/month found there).
    When filter_user_id is given, scopes to that user only.
    """
    from sqlalchemy import func
    from datetime import date

    daily_q = select(func.min(RequestUsage.date))
    # Order by (year, month) rather than min(year)/min(month) independently —
    # independent mins pick the wrong date across year boundaries.
    monthly_q = select(RequestUsageMonthly.year, RequestUsageMonthly.month)

    if filter_user_id is not None:
        daily_q = daily_q.where(RequestUsage.user_id == filter_user_id)
        monthly_q = monthly_q.where(RequestUsageMonthly.user_id == filter_user_id)

    monthly_q = monthly_q.order_by(
        RequestUsageMonthly.year.asc(), RequestUsageMonthly.month.asc()
    ).limit(1)

    # Compute earliest from both sources — the daily table may only hold recent
    # rows while older data lives solely in the monthly rollup, so we must take
    # the minimum across both rather than short-circuiting on the daily table.
    candidates = []

    daily_min = (await db.execute(daily_q)).scalar()
    if daily_min:
        candidates.append(
            daily_min if hasattr(daily_min, 'isoformat')
            else date.fromisoformat(str(daily_min))
        )

    monthly_row = (await db.execute(monthly_q)).first()
    if monthly_row and monthly_row[0] is not None:
        candidates.append(date(int(monthly_row[0]), int(monthly_row[1]), 1))

    if not candidates:
        return None

    return min(candidates).isoformat()


async def get_usage_years(db: AsyncSession) -> list[int]:
    """Return sorted list of years with any usage data (daily or monthly tables)."""
    from sqlalchemy import func

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


def _fold_identities(rows: list, *, by_model: bool = False) -> list:
    """Collapse (user_id, user_identity, user_type) rows into one row per person.

    The key is user_id; user_identity is the label the rows carry (the current
    username, or the admin name for ADMIN_USAGE_USER_ID) and user_type is a delivery
    detail. Neither is part of any usage table's unique key, so one person's rows can
    still differ in user_type: alice through the web UI in the morning and through an
    API key in the afternoon. The badge reads "mixed" when more than one type was seen.
    """
    folded: dict = {}
    for r in rows:
        key = (r["user_id"], r["model"]) if by_model else (r["user_id"],)
        entry = folded.get(key)
        if entry is None:
            entry = folded[key] = {"types": set(), "count": 0, "label": None}
        if r.get("user_type"):
            entry["types"].add(r["user_type"])
        if r.get("user_identity"):
            entry["label"] = r["user_identity"]
        entry["count"] += r["request_count"]

    out = []
    for key, entry in folded.items():
        types = entry["types"]
        row = {
            "user_id": key[0],
            "user_identity": entry["label"],
            "user_type": next(iter(types)) if len(types) == 1 else ("mixed" if types else None),
        }
        if by_model:
            row["model"] = key[1]
        row["request_count"] = int(entry["count"])
        out.append(row)

    if by_model:
        return sorted(out, key=lambda x: (-x["request_count"], x["user_identity"] or "", x["model"]))
    return sorted(out, key=lambda x: (-x["request_count"], x["user_identity"] or ""))


def _user_rows_q(tbl, where):
    """Per-person aggregate on one usage table: (user_id, user_identity, user_type, sum)."""
    from sqlalchemy import func
    return (
        select(
            tbl.user_id,
            tbl.user_identity,
            tbl.user_type,
            func.sum(tbl.request_count).label("request_count"),
        )
        .where(*where)
        .group_by(tbl.user_id, tbl.user_identity, tbl.user_type)
    )


def _model_rows_q(tbl, where):
    """Per-model aggregate on one usage table: (model, sum)."""
    from sqlalchemy import func
    return (
        select(tbl.model, func.sum(tbl.request_count).label("request_count"))
        .where(*where)
        .group_by(tbl.model)
    )


def _user_row_dicts(rows) -> list:
    return [
        {"user_id": r.user_id, "user_identity": r.user_identity,
         "user_type": r.user_type, "request_count": r.request_count}
        for r in rows
    ]


def _merge_model_rows(*row_sets) -> list:
    combined: dict = {}
    for rows in row_sets:
        for r in rows:
            combined[r.model] = combined.get(r.model, 0) + int(r.request_count)
    return sorted(
        [{"model": m, "request_count": c} for m, c in combined.items()],
        key=lambda x: (-x["request_count"], x["model"]),
    )


def _window_tables(window: str, year: Optional[int], month: Optional[int]):
    """Which usage tables a window reads, each with its own WHERE clauses.

    Returns [(table, [clauses])]. Mirrors the retention design: hourly for the rolling
    24h, daily for today/yesterday/7d/30d, and daily UNION monthly for the month and
    all-time windows (a month sits in exactly one of the two, so the union never
    double-counts). Every usage reader goes through this so the table, the chart and
    the pool views can never disagree about what a window contains.
    """
    from datetime import timedelta
    from app import time_utils

    today = time_utils.local_today()

    if window == "24h":
        cutoff_dt = time_utils.local_now() - timedelta(hours=24)
        cutoff_date, cutoff_hour = cutoff_dt.date(), cutoff_dt.hour
        return [(RequestUsageHourly, [
            (RequestUsageHourly.date > cutoff_date)
            | ((RequestUsageHourly.date == cutoff_date) & (RequestUsageHourly.hour >= cutoff_hour))
        ])]
    if window == "month" and year and month:
        first_day, last_day = _month_bounds(year, month)
        return [
            (RequestUsage, [RequestUsage.date >= first_day, RequestUsage.date <= last_day]),
            (RequestUsageMonthly, [RequestUsageMonthly.year == year, RequestUsageMonthly.month == month]),
        ]
    if window == "all":
        return [(RequestUsage, []), (RequestUsageMonthly, [])]
    if window == "today":
        return [(RequestUsage, [RequestUsage.date == today])]
    if window == "yesterday":
        return [(RequestUsage, [RequestUsage.date == today - timedelta(days=1)])]
    if window == "7d":
        return [(RequestUsage, [RequestUsage.date >= today - timedelta(days=6)])]
    return [(RequestUsage, [RequestUsage.date >= today - timedelta(days=29)])]  # default 30d


async def get_usage_aggregates(
    db: AsyncSession,
    filter_user_id: Optional[int] = None,
    filter_model: Optional[str] = None,
    window: str = "30d",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """Return aggregated usage data for the requested time window.

    window: '24h' | 'today' | 'yesterday' | '7d' | '30d' | 'month' | 'all'
    year, month: required when window='month'

    With neither filter set the result is the top level: per_user, per_model and
    totals. Either filter set makes it a drill-down returning only 'breakdown' --
    both filters restrict the rows, and filter_model decides the axis (users who
    used that model; otherwise the models that user used).

    Both filters restrict; the drill target only picks the output axis. /auth/usage
    always pins filter_user_id to the caller, so treating it as the target would make
    ?view=model&id=X return every model the caller used instead of their usage of X --
    and the chart, which honours filter_model, would disagree with the table beside it.
    """
    def _filters(tbl):
        where = []
        if filter_user_id is not None:
            where.append(tbl.user_id == filter_user_id)
        if filter_model is not None:
            where.append(tbl.model == filter_model)
        return where

    tables = _window_tables(window, year, month)
    result: dict = {"window": window}
    if window == "month" and year and month:
        result.update({"year": year, "month": month})

    if filter_user_id is not None or filter_model is not None:
        if filter_model is not None:
            # Users who used this model (optionally: one user's usage of it).
            rows: list = []
            for tbl, where in tables:
                rows.extend((await db.execute(_user_rows_q(tbl, where + _filters(tbl)))).all())
            result["breakdown"] = _fold_identities(_user_row_dicts(rows))
        else:
            row_sets = [
                (await db.execute(_model_rows_q(tbl, where + _filters(tbl)))).all()
                for tbl, where in tables
            ]
            result["breakdown"] = _merge_model_rows(*row_sets)
        return result

    user_rows: list = []
    model_row_sets: list = []
    for tbl, where in tables:
        user_rows.extend((await db.execute(_user_rows_q(tbl, where))).all())
        model_row_sets.append((await db.execute(_model_rows_q(tbl, where))).all())

    per_user = _fold_identities(_user_row_dicts(user_rows))
    per_model = _merge_model_rows(*model_row_sets)
    result.update({
        "per_user": per_user,
        "per_model": per_model,
        "totals": {
            "requests": sum(r["request_count"] for r in per_user),
            "unique_users": len(per_user),
            "unique_models": len(per_model),
        },
    })
    return result


async def get_usage_by_user_and_model(
    db: AsyncSession,
    user_ids: Optional[list] = None,
    *,
    pool_id: Optional[int] = None,
    window: str = "30d",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> list[dict]:
    """Return the user x model cross-product of request counts.

    [{user_id, user_identity, user_type, model, request_count}], restricted to
    `user_ids` (every row those users sent, in any pool or none) and/or to `pool_id`
    (every row stamped with that pool, whoever sent it). Both given: that pool's rows
    from those users. Neither given: nothing -- an unscoped cross-product is never
    what a caller wants.

    Uses the same window/table selection as get_usage_aggregates. Settlement depends
    on window="today" reading the daily table, so the two stay in lockstep.

    One query per table serves every pool view: per-member totals, pool-wide
    per-model, per-member per-model and per-group are all folds of this one result
    set in Python. Settlement (app/auth/pools.py) uses window="today" with user_ids
    and NO pool filter -- a member's charge is what they sent today wherever they were.
    """
    from sqlalchemy import func

    if user_ids is None and pool_id is None:
        return []
    if user_ids is not None:
        user_ids = list(dict.fromkeys(user_ids))
        if not user_ids:
            return []

    def _rows_q(tbl, where):
        scope = []
        if user_ids is not None:
            scope.append(tbl.user_id.in_(user_ids))
        if pool_id is not None:
            scope.append(tbl.pool_id == pool_id)
        return (
            select(
                tbl.user_id,
                tbl.user_identity,
                tbl.user_type,
                tbl.model,
                func.sum(tbl.request_count).label("rc"),
            )
            .where(*scope, *where)
            .group_by(tbl.user_id, tbl.user_identity, tbl.user_type, tbl.model)
        )

    collected: list = []
    for tbl, where in _window_tables(window, year, month):
        collected.extend(
            {"user_id": r.user_id, "user_identity": r.user_identity,
             "user_type": r.user_type, "model": r.model, "request_count": r.rc}
            for r in (await db.execute(_rows_q(tbl, where))).all()
        )
    return _fold_identities(collected, by_model=True)


async def get_usage_totals_by_pool(
    db: AsyncSession,
    pool_ids: list,
    window: str = "30d",
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> dict:
    """{pool_id: request_count} over the window, for the given pools, in one query per table.

    Feeds the admin By Pool table. Pools absent from the result had no traffic; pools
    not in `pool_ids` (deleted ones whose rows still carry their id) are left out so
    they never surface as phantom rows.
    """
    from sqlalchemy import func

    pool_ids = list(dict.fromkeys(pool_ids))
    if not pool_ids:
        return {}
    totals: dict = {}
    for tbl, where in _window_tables(window, year, month):
        rows = (await db.execute(
            select(tbl.pool_id, func.sum(tbl.request_count).label("rc"))
            .where(tbl.pool_id.in_(pool_ids), *where)
            .group_by(tbl.pool_id)
        )).all()
        for r in rows:
            totals[r.pool_id] = totals.get(r.pool_id, 0) + int(r.rc or 0)
    return totals


async def get_usage_timeseries(
    db: AsyncSession,
    filter_user_id: Optional[int] = None,
    filter_model: Optional[str] = None,
    window: str = "30d",
    year: Optional[int] = None,
    month: Optional[int] = None,
    user_ids: Optional[list] = None,
    pool_ids: Optional[list] = None,
) -> list[dict]:
    """Return ordered, zero-filled time buckets of request counts for the window.

    Returns a list of {"label": str, "count": int} ordered chronologically. Mirrors
    the window/table selection of get_usage_aggregates so the chart stays in lockstep
    with the table:
      - 24h          -> one bucket per hour (rolling 24h) from the hourly table
      - today/yest.  -> 24 hourly buckets for that day (falls back to one daily bucket
                        if no hourly rows exist, e.g. a freshly-deployed instance)
      - 7d/30d       -> one bucket per day, zero-filled across the range
      - month        -> one bucket per day of the month (or a single monthly bucket
                        once the month has been rolled up and daily rows deleted)
      - all          -> one bucket per month (YYYY-MM), union of daily + monthly tables

    filter_user_id / filter_model scope the series to a single user or model
    (drill-down). user_ids limits it to a set of people without singling one out;
    pool_ids limits it to rows stamped with any of those pools, which is how the By
    Pool chart shows exactly what the pools consumed. All of them compose.
    """
    from sqlalchemy import func
    from datetime import date, timedelta
    from app import time_utils

    today = time_utils.local_today()

    def _apply_filters(q, tbl):
        if filter_user_id is not None:
            q = q.where(tbl.user_id == filter_user_id)
        if filter_model is not None:
            q = q.where(tbl.model == filter_model)
        if user_ids is not None:
            q = q.where(tbl.user_id.in_(list(dict.fromkeys(user_ids))) if user_ids else false())
        if pool_ids is not None:
            q = q.where(tbl.pool_id.in_(list(dict.fromkeys(pool_ids))) if pool_ids else false())
        return q

    # ------------------------------------------------------------------ #
    # 24h — rolling window, hourly buckets
    # ------------------------------------------------------------------ #
    if window == "24h":
        now = time_utils.local_now()
        cutoff_dt = now - timedelta(hours=24)
        cutoff_date = cutoff_dt.date()
        cutoff_hour = cutoff_dt.hour

        q = (
            select(
                RequestUsageHourly.date,
                RequestUsageHourly.hour,
                func.sum(RequestUsageHourly.request_count).label("rc"),
            )
            .where(
                (RequestUsageHourly.date > cutoff_date)
                | ((RequestUsageHourly.date == cutoff_date) & (RequestUsageHourly.hour >= cutoff_hour))
            )
            .group_by(RequestUsageHourly.date, RequestUsageHourly.hour)
        )
        q = _apply_filters(q, RequestUsageHourly)
        rows = (await db.execute(q)).all()
        counts: dict = {(r.date, r.hour): r.rc for r in rows}

        # Build ordered hourly slots from the cutoff hour through the current hour.
        # That spans 25 slots: the rolling 24h boundary lands mid-hour, so both the
        # partial cutoff hour and the partial current hour are included.
        start = cutoff_dt.replace(minute=0, second=0, microsecond=0)
        buckets = []
        for i in range(25):
            slot = start + timedelta(hours=i)
            key = (slot.date(), slot.hour)
            buckets.append({"label": slot.strftime("%Y-%m-%d %H:00"), "count": int(counts.get(key, 0))})
        return buckets

    # ------------------------------------------------------------------ #
    # today / yesterday — hourly buckets for a single day, daily fallback
    # ------------------------------------------------------------------ #
    if window in ("today", "yesterday"):
        target = today if window == "today" else today - timedelta(days=1)

        q = (
            select(RequestUsageHourly.hour, func.sum(RequestUsageHourly.request_count).label("rc"))
            .where(RequestUsageHourly.date == target)
            .group_by(RequestUsageHourly.hour)
        )
        q = _apply_filters(q, RequestUsageHourly)
        rows = (await db.execute(q)).all()

        if rows:
            counts = {int(r.hour): int(r.rc) for r in rows}
            return [
                {"label": f"{target.isoformat()} {h:02d}:00", "count": counts.get(h, 0)}
                for h in range(24)
            ]

        # Fallback: single daily bucket from the daily table.
        dq = (
            select(func.sum(RequestUsage.request_count).label("rc"))
            .where(RequestUsage.date == target)
        )
        dq = _apply_filters(dq, RequestUsage)
        total = (await db.execute(dq)).scalar() or 0
        return [{"label": target.isoformat(), "count": int(total)}]

    # ------------------------------------------------------------------ #
    # 7d / 30d — daily buckets, zero-filled across the range
    # ------------------------------------------------------------------ #
    if window in ("7d", "30d"):
        span = 7 if window == "7d" else 30
        start = today - timedelta(days=span - 1)

        q = (
            select(RequestUsage.date, func.sum(RequestUsage.request_count).label("rc"))
            .where(RequestUsage.date >= start)
            .group_by(RequestUsage.date)
        )
        q = _apply_filters(q, RequestUsage)
        rows = (await db.execute(q)).all()
        counts = {r.date.isoformat(): int(r.rc) for r in rows}

        return [
            {"label": (start + timedelta(days=i)).isoformat(),
             "count": counts.get((start + timedelta(days=i)).isoformat(), 0)}
            for i in range(span)
        ]

    # ------------------------------------------------------------------ #
    # month — daily buckets for the month, or a single monthly bucket
    # ------------------------------------------------------------------ #
    if window == "month" and year and month:
        first_day, last_day = _month_bounds(year, month)

        q = (
            select(RequestUsage.date, func.sum(RequestUsage.request_count).label("rc"))
            .where(RequestUsage.date >= first_day, RequestUsage.date <= last_day)
            .group_by(RequestUsage.date)
        )
        q = _apply_filters(q, RequestUsage)
        rows = (await db.execute(q)).all()

        if rows:
            counts = {r.date.isoformat(): int(r.rc) for r in rows}
            days_in_month = (last_day - first_day).days + 1
            return [
                {"label": date(year, month, d).isoformat(),
                 "count": counts.get(date(year, month, d).isoformat(), 0)}
                for d in range(1, days_in_month + 1)
            ]

        # Rolled-up month: single bucket from the monthly table.
        mq = (
            select(func.sum(RequestUsageMonthly.request_count).label("rc"))
            .where(RequestUsageMonthly.year == year, RequestUsageMonthly.month == month)
        )
        mq = _apply_filters(mq, RequestUsageMonthly)
        total = (await db.execute(mq)).scalar() or 0
        return [{"label": f"{year}-{month:02d}", "count": int(total)}]

    # ------------------------------------------------------------------ #
    # all — monthly buckets, union of daily + monthly tables
    # ------------------------------------------------------------------ #
    if window == "all":
        dq = (
            select(
                func.strftime('%Y-%m', RequestUsage.date).label("ym"),
                func.sum(RequestUsage.request_count).label("rc"),
            )
            .group_by(func.strftime('%Y-%m', RequestUsage.date))
        )
        dq = _apply_filters(dq, RequestUsage)
        d_rows = (await db.execute(dq)).all()

        mq = (
            select(
                RequestUsageMonthly.year,
                RequestUsageMonthly.month,
                func.sum(RequestUsageMonthly.request_count).label("rc"),
            )
            .group_by(RequestUsageMonthly.year, RequestUsageMonthly.month)
        )
        mq = _apply_filters(mq, RequestUsageMonthly)
        m_rows = (await db.execute(mq)).all()

        combined: dict = {}
        for r in d_rows:
            combined[r.ym] = combined.get(r.ym, 0) + int(r.rc)
        for r in m_rows:
            ym = f"{int(r.year)}-{int(r.month):02d}"
            combined[ym] = combined.get(ym, 0) + int(r.rc)

        return [{"label": ym, "count": combined[ym]} for ym in sorted(combined)]

    # ------------------------------------------------------------------ #
    # Fallback (unknown window) — behave like 30d
    # ------------------------------------------------------------------ #
    start = today - timedelta(days=29)
    q = (
        select(RequestUsage.date, func.sum(RequestUsage.request_count).label("rc"))
        .where(RequestUsage.date >= start)
        .group_by(RequestUsage.date)
    )
    q = _apply_filters(q, RequestUsage)
    rows = (await db.execute(q)).all()
    counts = {r.date.isoformat(): int(r.rc) for r in rows}
    return [
        {"label": (start + timedelta(days=i)).isoformat(),
         "count": counts.get((start + timedelta(days=i)).isoformat(), 0)}
        for i in range(30)
    ]


# --------------------------------------------------------------------------- #
# Usage-table rekey migration: username-string keys -> (user_id, pool_id) keys
#
# Pre-rekey usage rows are keyed by the username string and carry no pool. The
# rebuild below maps every identity to a user id (the config admin to
# ADMIN_USAGE_USER_ID, legacy "key:<id>" identities to the key's owner), merges rows
# that collapse onto one key, and back-fills pool_id from the membership intervals.
# Identities that resolve to nothing -- deleted users, deleted keys -- are dropped,
# matching the purge-on-delete policy; the file backup taken first is the safety net.
# --------------------------------------------------------------------------- #

_USAGE_TABLE_NAMES = ("request_usage", "request_usage_hourly", "request_usage_monthly")


def _sqlite_db_path() -> Optional[str]:
    """Filesystem path of the SQLite database, or None for other engines / in-memory."""
    if not DATABASE_URL.startswith("sqlite"):
        return None
    path = DATABASE_URL.split("///")[-1]
    if not path or path.startswith(":memory:"):
        return None
    return os.path.abspath(path)


def _backup_sqlite_file(tag: str) -> Optional[str]:
    """Copy the live database with VACUUM INTO (safe under WAL) and return the path.

    Runs on the stdlib driver outside any SQLAlchemy transaction: VACUUM cannot run
    inside one. Blocking; callers hand it to a thread.
    """
    import sqlite3
    import time as _time

    path = _sqlite_db_path()
    if path is None or not os.path.exists(path):
        return None
    dest = f"{path}.{tag}-{_time.strftime('%Y%m%d-%H%M%S')}.bak"
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"VACUUM INTO '{dest.replace(chr(39), chr(39) * 2)}'")
    finally:
        conn.close()
    return dest


async def _usage_tables_needing_rekey(conn) -> list:
    """Names of usage tables that exist but still lack the user_id column."""
    from sqlalchemy import text

    pending = []
    for table in _USAGE_TABLE_NAMES:
        rows = (await conn.execute(text(f"PRAGMA table_info({table})"))).fetchall()
        columns = [r[1] for r in rows]
        if columns and "user_id" not in columns:
            pending.append(table)
    return pending


async def _build_usage_identity_map(conn) -> None:
    """Create and fill the temp table mapping legacy identity strings to (user_id, label)."""
    from sqlalchemy import text
    from app.auth.models import ADMIN_USAGE_USER_ID

    await conn.execute(text("DROP TABLE IF EXISTS _usage_identity_map"))
    await conn.execute(text(
        "CREATE TEMP TABLE _usage_identity_map ("
        " user_identity TEXT PRIMARY KEY, user_id INTEGER NOT NULL, label TEXT NOT NULL)"
    ))

    entries: dict = {}
    users = (await conn.execute(text("SELECT id, username FROM users"))).fetchall()
    for uid, username in users:
        entries[username] = (int(uid), username)
    keys = (await conn.execute(text(
        "SELECT k.id, k.user_id, u.username FROM api_keys k JOIN users u ON u.id = k.user_id"
    ))).fetchall()
    for key_id, uid, username in keys:
        entries.setdefault(f"key:{key_id}", (int(uid), username))

    from app.auth.admin import get_admin_username, is_admin_enabled
    if is_admin_enabled():
        admin_name = get_admin_username()
        if admin_name:
            # setdefault, not assignment: is_reserved_username only rejects this name
            # while the admin is enabled, so a user who registered it during a disabled
            # stint -- or under an earlier ADMIN_USERNAME -- holds a real `users` row
            # under it. Overwriting would rekey that account's whole history to
            # ADMIN_USAGE_USER_ID, irreversibly. The map is keyed by identity alone and
            # cannot split a genuine collision, so the real account wins and the admin's
            # own rows fold into it -- misattributed, but nothing is lost.
            if admin_name in entries:
                logger.warning(
                    "Usage rekey: '%s' is both the admin username and a real account; "
                    "admin usage recorded under that name will be attributed to the account",
                    admin_name,
                )
            entries.setdefault(admin_name, (ADMIN_USAGE_USER_ID, admin_name))

    # Only the config admin ever writes user_type 'admin', so rows labelled that way
    # belong to it whatever the admin account is called -- or whether it is enabled --
    # at the moment this migration runs. Without this, migrating with the admin
    # disabled would silently drop its whole history as unattributable.
    for table in _USAGE_TABLE_NAMES:
        rows = await conn.execute(text(
            f"SELECT DISTINCT user_identity FROM {table} WHERE user_type = 'admin'"
        ))
        for (identity,) in rows.fetchall():
            if identity:
                entries.setdefault(identity, (ADMIN_USAGE_USER_ID, identity))

    for identity, (uid, label) in entries.items():
        await conn.execute(
            text("INSERT INTO _usage_identity_map (user_identity, user_id, label) "
                 "VALUES (:i, :u, :l)"),
            {"i": identity, "u": uid, "l": label},
        )


async def _rekey_usage_table(conn, table: str) -> dict:
    """Rebuild one usage table onto the (user_id, pool_id) key. Returns a stats dict.

    Copy-out rather than RENAME: a renamed table keeps its indexes under their old
    names, which then collide with the CREATE INDEX statements of the new table.
    Nothing references the usage tables by foreign key, so DROP is safe.
    """
    from sqlalchemy import text

    time_cols = {
        "request_usage": ("date",),
        "request_usage_hourly": ("date", "hour"),
        "request_usage_monthly": ("year", "month"),
    }[table]
    tcols = ", ".join(f"m.{c}" for c in time_cols)
    ins_tcols = ", ".join(time_cols)

    before = (await conn.execute(text(
        f"SELECT COUNT(*), COALESCE(SUM(request_count), 0) FROM {table}"
    ))).one()
    orphans = (await conn.execute(text(
        f"SELECT COUNT(*), COALESCE(SUM(request_count), 0) FROM {table} "
        f"WHERE user_identity NOT IN (SELECT user_identity FROM _usage_identity_map)"
    ))).one()

    await conn.execute(text(f"DROP TABLE IF EXISTS {table}_mig"))
    await conn.execute(text(f"CREATE TABLE {table}_mig AS SELECT * FROM {table}"))
    await conn.execute(text(f"DROP TABLE {table}"))
    await conn.run_sync(lambda sync_conn: Base.metadata.tables[table].create(sync_conn))

    await conn.execute(text(
        f"INSERT INTO {table} ({ins_tcols}, user_id, pool_id, user_identity, user_type, "
        f"model, server, request_count) "
        f"SELECT {tcols}, im.user_id, 0, im.label, MAX(m.user_type), m.model, m.server, "
        f"SUM(m.request_count) "
        f"FROM {table}_mig m JOIN _usage_identity_map im ON im.user_identity = m.user_identity "
        f"GROUP BY {tcols}, im.user_id, m.model, m.server"
    ))
    await conn.execute(text(f"DROP TABLE {table}_mig"))

    after = (await conn.execute(text(
        f"SELECT COUNT(*), COALESCE(SUM(request_count), 0) FROM {table}"
    ))).one()
    return {
        "rows_before": before[0], "requests_before": before[1],
        "rows_after": after[0], "requests_after": after[1],
        "orphan_rows_dropped": orphans[0], "orphan_requests_dropped": orphans[1],
    }


async def _backfill_usage_pool_ids(conn, tables: list) -> int:
    """Stamp pool_id onto pre-rekey rows from the membership intervals. Returns rows updated.

    Day-granular for daily/hourly and month-granular for the rollup -- the precision the
    interval table has. Stints are applied oldest first with a `pool_id = 0` guard, so
    on a day a user left one pool for another the earlier stint keeps the day.

    That guard decides the whole day, not just the part before the move: a usage row is
    one row per day, so traffic sent after joining the later pool is stamped with the
    earlier one too. There is no finer split available -- the rows being backfilled
    predate pool_id entirely -- and picking the earlier stint at least keeps the day
    attributed to the pool that held the member for the start of it.
    """
    from sqlalchemy import text
    from app import time_utils

    today = time_utils.local_today()
    intervals = (await conn.execute(text(
        "SELECT pool_id, user_id, joined_on, left_on FROM pool_membership_intervals "
        "ORDER BY joined_on, id"
    ))).fetchall()
    updated = 0
    for pool_id, user_id, joined_on, left_on in intervals:
        lo = joined_on if isinstance(joined_on, str) else joined_on.isoformat()
        hi_date = left_on or today
        hi = hi_date if isinstance(hi_date, str) else hi_date.isoformat()
        params = {"p": pool_id, "u": user_id, "lo": lo, "hi": hi}
        for table in tables:
            if table == "request_usage_monthly":
                lo_ord = int(lo[:4]) * 12 + int(lo[5:7])
                hi_ord = int(hi[:4]) * 12 + int(hi[5:7])
                result = await conn.execute(text(
                    f"UPDATE {table} SET pool_id = :p WHERE user_id = :u AND pool_id = 0 "
                    f"AND (year * 12 + month) BETWEEN :lo_ord AND :hi_ord"
                ), {"p": pool_id, "u": user_id, "lo_ord": lo_ord, "hi_ord": hi_ord})
            else:
                result = await conn.execute(text(
                    f"UPDATE {table} SET pool_id = :p WHERE user_id = :u AND pool_id = 0 "
                    f"AND date >= :lo AND date <= :hi"
                ), params)
            updated += result.rowcount or 0
    return updated


async def _check_usage_timezone(conn) -> None:
    """Record the zone the usage buckets are computed in; shout if it has changed."""
    from sqlalchemy import text
    from app.config import config

    configured = config.server.timezone
    row = (await conn.execute(text(
        "SELECT value FROM usage_meta WHERE key = 'timezone'"
    ))).first()
    if row is None:
        await conn.execute(
            text("INSERT INTO usage_meta (key, value, updated_at) "
                 "VALUES ('timezone', :tz, CURRENT_TIMESTAMP)"),
            {"tz": configured},
        )
        return
    if row[0] != configured:
        logger.error(
            "TIMEZONE is '%s' but the usage tables were bucketed in '%s'. Day and hour "
            "boundaries of existing rows no longer line up with new ones. Restore the "
            "old value, or accept the discontinuity by updating usage_meta.timezone.",
            configured, row[0],
        )


_POOL_ID_HIGH_WATER_KEY = "pool_id_high_water"


async def _seed_pool_id_high_water(conn) -> None:
    """Record the highest pool id ever seen, if no high-water mark exists yet.

    Usage rows keep a dissolved pool's id, so the seed covers them as well as live
    pools: an id that only survives in usage is still spent.
    """
    from sqlalchemy import text

    await conn.execute(text(
        "INSERT OR IGNORE INTO usage_meta (key, value, updated_at) "
        "SELECT :k, CAST(MAX("
        "  (SELECT COALESCE(MAX(id), 0) FROM request_pools),"
        "  (SELECT COALESCE(MAX(pool_id), 0) FROM request_usage),"
        "  (SELECT COALESCE(MAX(pool_id), 0) FROM request_usage_hourly),"
        "  (SELECT COALESCE(MAX(pool_id), 0) FROM request_usage_monthly)"
        ") AS TEXT), CURRENT_TIMESTAMP"
    ), {"k": _POOL_ID_HIGH_WATER_KEY})


async def allocate_pool_id(db: AsyncSession) -> int:
    """Return a pool id that no pool, live or dissolved, has ever had.

    request_pools.id is a plain INTEGER PRIMARY KEY, so SQLite would hand a dissolved
    pool's id to the next pool created -- and that pool would inherit every usage row
    still stamped with it, in its usage views and in an admin's per-pool purge.
    AUTOINCREMENT would fix that, but only by rebuilding a table that four others
    cascade from. A high-water mark in usage_meta gives the same guarantee.

    The UPDATE comes before the read so this transaction holds SQLite's write lock
    by the time it reads: two concurrent creates cannot be handed the same id.
    """
    from sqlalchemy import text

    await _seed_pool_id_high_water(db)
    await db.execute(text(
        "UPDATE usage_meta SET value = CAST(MAX("
        "  CAST(value AS INTEGER), (SELECT COALESCE(MAX(id), 0) FROM request_pools)"
        ") + 1 AS TEXT), updated_at = CURRENT_TIMESTAMP WHERE key = :k"
    ), {"k": _POOL_ID_HIGH_WATER_KEY})
    value = (await db.execute(
        text("SELECT value FROM usage_meta WHERE key = :k"), {"k": _POOL_ID_HIGH_WATER_KEY}
    )).scalar_one()
    return int(value)


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
    3. Rebuilding the usage tables onto (user_id, pool_id) keys, with a file backup first
    """
    import asyncio
    from sqlalchemy import text

    # The usage rekey rewrites three tables and drops unresolvable rows, so the file is
    # backed up before anything else runs. VACUUM INTO cannot run inside a transaction,
    # hence outside the engine.begin() block below.
    async with engine.connect() as conn:
        usage_rekey_pending = await _usage_tables_needing_rekey(conn)
    if usage_rekey_pending:
        try:
            backup = await asyncio.to_thread(_backup_sqlite_file, "pre-usage-rekey")
        except Exception as e:
            # Without a backup the rekey's orphan drop is unrecoverable, and without
            # the rekey every usage write fails. Neither is acceptable: stop here.
            raise RuntimeError(
                f"usage tables {usage_rekey_pending} need rekeying but the pre-migration "
                f"backup failed: {e}"
            ) from e
        if not backup:
            # A None return is not a success: the live file could not be resolved (a
            # relocated or non-file-backed DATABASE_URL), so there is nothing to roll
            # back to. Same bargain as the raise above -- refuse rather than drop rows
            # with no recovery path.
            raise RuntimeError(
                f"usage tables {usage_rekey_pending} need rekeying but the database file "
                f"could not be located to back up first"
            )
        logger.info("Auto-migration: database backed up to %s", backup)

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

        # Add 'user_id' column to response_provider_mappings for the Responses
        # API ownership (IDOR) check. Nullable: pre-migration rows keep NULL and
        # are treated as unowned by the enforcement code.
        try:
            result = await conn.execute(text("PRAGMA table_info(response_provider_mappings)"))
            columns = [row[1] for row in result.fetchall()]
            if columns and 'user_id' not in columns:
                logger.info("Auto-migration: Adding 'user_id' column to response_provider_mappings")
                await conn.execute(text(
                    "ALTER TABLE response_provider_mappings ADD COLUMN user_id INTEGER"
                ))
                logger.info("Auto-migration: 'user_id' column added to response_provider_mappings")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add user_id column to response_provider_mappings: {e}")

        # Add 'mode' column to user_model_access_policies and backfill from the
        # legacy default_allow boolean (True -> allow, False -> deny).
        try:
            result = await conn.execute(text("PRAGMA table_info(user_model_access_policies)"))
            columns = [row[1] for row in result.fetchall()]
            if columns and 'mode' not in columns:
                logger.info("Auto-migration: Adding 'mode' column to user_model_access_policies")
                await conn.execute(text(
                    "ALTER TABLE user_model_access_policies ADD COLUMN mode TEXT DEFAULT 'default'"
                ))
                if 'default_allow' in columns:
                    await conn.execute(text(
                        "UPDATE user_model_access_policies SET mode = 'allow' WHERE default_allow = 1"
                    ))
                    await conn.execute(text(
                        "UPDATE user_model_access_policies SET mode = 'deny' WHERE default_allow = 0"
                    ))
                logger.info("Auto-migration: 'mode' column added and backfilled")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add mode column: {e}")

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

        # Add 'apis' column to model_aliases so mappings can be scoped to
        # specific API surfaces. NULL (legacy rows) means "all surfaces".
        try:
            result = await conn.execute(text("PRAGMA table_info(model_aliases)"))
            columns = [row[1] for row in result.fetchall()]

            if columns and 'apis' not in columns:
                logger.info("Auto-migration: Adding 'apis' column to model_aliases")
                await conn.execute(text(
                    "ALTER TABLE model_aliases ADD COLUMN apis TEXT DEFAULT '[\"openai\", \"anthropic\", \"azure_openai\"]'"
                ))
                logger.info("Auto-migration: 'apis' column added successfully")
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add apis column to model_aliases: {e}")

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

        # Add the backend-selection columns to websearch_settings. NULL means
        # "use the default" (provider -> searxng), so existing rows keep working.
        # On a fresh database create_tables_async() has already made these and
        # this is a no-op.
        try:
            result = await conn.execute(text("PRAGMA table_info(websearch_settings)"))
            columns = [row[1] for row in result.fetchall()]
            if columns:
                for name, ddl in (
                    ('provider', "ALTER TABLE websearch_settings ADD COLUMN provider TEXT"),
                    ('fourget_base_url', "ALTER TABLE websearch_settings ADD COLUMN fourget_base_url TEXT"),
                    ('fourget_scraper', "ALTER TABLE websearch_settings ADD COLUMN fourget_scraper TEXT"),
                    ('fourget_lang', "ALTER TABLE websearch_settings ADD COLUMN fourget_lang TEXT"),
                    ('fourget_country', "ALTER TABLE websearch_settings ADD COLUMN fourget_country TEXT"),
                ):
                    if name not in columns:
                        logger.info(f"Auto-migration: Adding '{name}' column to websearch_settings")
                        await conn.execute(text(ddl))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add websearch backend columns: {e}")

        # Migrate api_version → discovery_api_version and drop legacy Azure AD columns.
        try:
            result = await conn.execute(text("PRAGMA table_info(provider_credentials)"))
            columns = [row[1] for row in result.fetchall()]

            if 'discovery_api_version' not in columns:
                logger.info("Auto-migration: Adding 'discovery_api_version' column to provider_credentials")
                await conn.execute(text(
                    "ALTER TABLE provider_credentials ADD COLUMN discovery_api_version TEXT"
                ))
                if 'api_version' in columns:
                    # Preserve any previously configured api_version as the discovery version.
                    await conn.execute(text(
                        "UPDATE provider_credentials SET discovery_api_version = api_version "
                        "WHERE provider_type = 'azure' AND api_version IS NOT NULL"
                    ))
                logger.info("Auto-migration: 'discovery_api_version' column added")

            # Drop the old api_version and Azure AD service-principal columns when present.
            for old_col in ('api_version', 'subscription_id', 'resource_group',
                            'account_name', 'client_id', 'client_secret', 'tenant_id'):
                if old_col in columns:
                    try:
                        await conn.execute(text(
                            f"ALTER TABLE provider_credentials DROP COLUMN {old_col}"
                        ))
                        logger.info(f"Auto-migration: Dropped obsolete column '{old_col}'")
                    except Exception as drop_err:
                        logger.warning(f"Auto-migration: Could not drop column '{old_col}': {drop_err}")
        except Exception as e:
            logger.warning(f"Auto-migration: discovery_api_version migration failed: {e}")

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

        # Create pool_membership_intervals and backfill one open stint per member.
        #
        # Pool usage is reconstructed from these spans, so without a backfill every
        # existing pool would report zero history the moment this ships. Departures
        # that happened *before* this migration are unrecoverable -- nothing recorded
        # them -- so pool usage history effectively begins here: a member who left
        # last week simply has no interval, and their traffic is excluded.
        try:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pool_membership_intervals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pool_id INTEGER NOT NULL REFERENCES request_pools(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    joined_on DATE NOT NULL,
                    left_on DATE
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_pmi_pool_user "
                "ON pool_membership_intervals (pool_id, user_id)"
            ))

            # joined_at is a naive UTC datetime; usage dates are local. Convert per
            # row rather than in SQL so the TIMEZONE setting is honoured.
            from datetime import timezone as _timezone
            from app import time_utils as _tu

            rows = (await conn.execute(text(
                "SELECT m.pool_id, m.user_id, m.joined_at FROM request_pool_members m "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM pool_membership_intervals i"
                "  WHERE i.pool_id = m.pool_id AND i.user_id = m.user_id"
                ")"
            ))).fetchall()
            for pool_id, user_id, joined_at in rows:
                if isinstance(joined_at, str):
                    joined_at = datetime.fromisoformat(joined_at)
                if joined_at is None:
                    joined_on = _tu.local_today()
                else:
                    if joined_at.tzinfo is None:
                        joined_at = joined_at.replace(tzinfo=_timezone.utc)
                    joined_on = joined_at.astimezone(_tu.get_tz()).date()
                await conn.execute(
                    text("INSERT INTO pool_membership_intervals "
                         "(pool_id, user_id, joined_on, left_on) "
                         "VALUES (:p, :u, :j, NULL)"),
                    {"p": pool_id, "u": user_id, "j": joined_on},
                )
            if rows:
                logger.info(
                    "Auto-migration: Opened %d pool membership interval(s)", len(rows)
                )
        except Exception as e:
            logger.warning(
                f"Auto-migration: Could not create pool_membership_intervals: {e}"
            )

        # Hash any plaintext API keys in place (SHA-256) and backfill key_prefix.
        # Existing plaintext keys keep working: the lookup path hashes the incoming
        # key, and here we replace each stored plaintext value with its hash.
        # Detection: a already-migrated value is a 64-char lowercase hex digest.
        try:
            result = await conn.execute(text("PRAGMA table_info(api_keys)"))
            columns = [row[1] for row in result.fetchall()]
            if columns:
                if 'key_prefix' not in columns:
                    logger.info("Auto-migration: Adding 'key_prefix' column to api_keys")
                    await conn.execute(text(
                        "ALTER TABLE api_keys ADD COLUMN key_prefix VARCHAR(16)"
                    ))

                import re as _re
                _hex64 = _re.compile(r'^[0-9a-f]{64}$')
                rows = (await conn.execute(
                    text("SELECT id, api_key, key_prefix FROM api_keys")
                )).fetchall()
                migrated = 0
                for row in rows:
                    key_id, key_val, prefix = row[0], row[1], row[2]
                    if key_val and not _hex64.match(key_val):
                        await conn.execute(
                            text("UPDATE api_keys SET api_key = :h, key_prefix = :p WHERE id = :i"),
                            {"h": hash_api_key(key_val), "p": prefix or key_val[:8], "i": key_id},
                        )
                        migrated += 1
                if migrated:
                    logger.info("Auto-migration: Hashed %d plaintext API key(s)", migrated)
        except Exception as e:
            logger.warning(f"Auto-migration: Could not hash existing API keys: {e}")

        # Enforce uniqueness of (provider, provider_user_id) on oauth_users.
        # The model declares a UniqueConstraint, but that only applies to freshly
        # created tables — pre-existing deployments need this index. If existing
        # duplicate rows make creation fail, warn and continue (do not crash).
        try:
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_oauth_provider_user "
                "ON oauth_users (provider, provider_user_id)"
            ))
        except Exception as e:
            logger.warning(
                "Auto-migration: Could not create unique index on "
                "oauth_users(provider, provider_user_id) — likely duplicate rows "
                "exist; deduplicate them manually to enforce uniqueness: %s", e
            )

        # Enforce case-insensitive uniqueness of request_pools.name. The column's
        # own UNIQUE is case-sensitive, so "Alpha" and "alpha" both fit in it even
        # though app/routes/pools.py treats them as the same name. Pre-existing
        # deployments need this index created explicitly; fresh ones get it from the
        # model. If existing rows already collide, warn and continue rather than
        # crash startup -- the route-level ilike checks still block new collisions.
        try:
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_request_pools_name_lower "
                "ON request_pools (lower(name))"
            ))
        except Exception as e:
            logger.warning(
                "Auto-migration: Could not create case-insensitive unique index on "
                "request_pools(name) — likely pool names differing only by case "
                "exist; rename one of them to enforce uniqueness: %s", e
            )

        # Rebuild the usage tables onto (user_id, pool_id) keys. Runs last: it needs
        # the users, api_keys and pool_membership_intervals tables in their final
        # shape, and the backup was taken before this block opened. Not wrapped in a
        # swallow-all try: a half-migrated usage schema would break every write, so a
        # failure here must abort startup and leave the transaction rolled back.
        if usage_rekey_pending:
            await _build_usage_identity_map(conn)
            for table in usage_rekey_pending:
                stats = await _rekey_usage_table(conn, table)
                logger.info("Auto-migration: rekeyed %s: %s", table, stats)
            stamped = await _backfill_usage_pool_ids(conn, usage_rekey_pending)
            logger.info("Auto-migration: back-filled pool_id on %d usage row(s)", stamped)
            await conn.execute(text("DROP TABLE IF EXISTS _usage_identity_map"))

        try:
            await _check_usage_timezone(conn)
        except Exception as e:
            logger.warning(f"Auto-migration: usage timezone check failed: {e}")

        # Seed at startup, while the usage buffer is still empty, so the mark also
        # covers ids of pools dissolved before it existed. allocate_pool_id seeds too.
        try:
            await _seed_pool_id_high_water(conn)
        except Exception as e:
            logger.warning(f"Auto-migration: could not seed the pool id high-water mark: {e}")


def init_database_sync():
    """Initialize the database and create tables synchronously (for backward compatibility).

    NOTE: Unlike the async init_database(), this only runs create_all and does NOT
    run _run_auto_migrations(), so on a pre-existing DB it leaves the schema
    unmigrated. init_database() (async) is the authoritative initializer; prefer it.
    """
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


# ---------------------------------------------------------------------------
# Instance-group helpers (mirror the model-group helpers, keyed on provider_key)
# ---------------------------------------------------------------------------

async def list_instance_groups(db: AsyncSession, group_id: Optional[int] = None):
    """Return all instance groups (with members loaded), or a single group if group_id given."""
    from sqlalchemy.orm import selectinload as _sil
    q = select(InstanceGroup).options(_sil(InstanceGroup.members))
    if group_id is not None:
        q = q.where(InstanceGroup.id == group_id)
    result = await db.execute(q)
    rows = result.scalars().all()
    return rows[0] if group_id is not None and rows else (None if group_id is not None else rows)


async def create_instance_group(
    db: AsyncSession, name: str, description: Optional[str],
    rpm_default: Optional[int], rpd_default: Optional[int], admin_username: str,
) -> InstanceGroup:
    row = InstanceGroup(
        name=name, description=description,
        rpm_default=rpm_default, rpd_default=rpd_default,
        updated_by=admin_username,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def update_instance_group(
    db: AsyncSession, group_id: int, fields: dict, admin_username: str,
) -> Optional[InstanceGroup]:
    result = await db.execute(select(InstanceGroup).where(InstanceGroup.id == group_id))
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


async def delete_instance_group(db: AsyncSession, group_id: int) -> bool:
    result = await db.execute(select(InstanceGroup).where(InstanceGroup.id == group_id))
    row = result.scalar_one_or_none()
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def set_instance_group_members(db: AsyncSession, group_id: int, provider_keys: list) -> list:
    """Replace the member list for an instance group. Returns list of InstanceGroupMember rows."""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(InstanceGroupMember).where(InstanceGroupMember.group_id == group_id))
    new_members = [InstanceGroupMember(group_id=group_id, provider_key=pk) for pk in provider_keys]
    db.add_all(new_members)
    await db.commit()
    return new_members


async def get_instance_group_limits(db: AsyncSession, group_id: int) -> Optional[InstanceGroup]:
    result = await db.execute(select(InstanceGroup).where(InstanceGroup.id == group_id))
    return result.scalar_one_or_none()


async def update_instance_group_limits(
    db: AsyncSession, group_id: int, rpm_default: Optional[int], rpd_default: Optional[int],
    admin_username: str,
) -> Optional[InstanceGroup]:
    return await update_instance_group(
        db, group_id,
        {"rpm_default": rpm_default, "rpd_default": rpd_default},
        admin_username,
    )


async def get_user_instance_group_rate_limit(
    db: AsyncSession, user_id: int, group_id: int,
) -> Optional[UserInstanceGroupRateLimit]:
    result = await db.execute(
        select(UserInstanceGroupRateLimit).where(
            UserInstanceGroupRateLimit.user_id == user_id,
            UserInstanceGroupRateLimit.group_id == group_id,
        )
    )
    return result.scalar_one_or_none()


async def list_user_instance_group_rate_limits(
    db: AsyncSession, group_id: int, user_id: Optional[int] = None,
):
    q = select(UserInstanceGroupRateLimit).where(UserInstanceGroupRateLimit.group_id == group_id)
    if user_id is not None:
        q = q.where(UserInstanceGroupRateLimit.user_id == user_id)
    result = await db.execute(q)
    return result.scalars().all()


async def upsert_user_instance_group_rate_limit(
    db: AsyncSession, user_id: int, group_id: int,
    rpm: Optional[int], rpd: Optional[int],
    admin_username: str, fields_set: set,
) -> UserInstanceGroupRateLimit:
    row = await get_user_instance_group_rate_limit(db, user_id, group_id)
    if row is None:
        row = UserInstanceGroupRateLimit(user_id=user_id, group_id=group_id)
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


async def delete_user_instance_group_rate_limit(db: AsyncSession, user_id: int, group_id: int) -> bool:
    row = await get_user_instance_group_rate_limit(db, user_id, group_id)
    if row is None:
        return False
    await db.delete(row)
    await db.commit()
    return True

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
from typing import Optional, List, Dict, Sequence, Tuple
from .models import Base, User, APIKey, ModelConfiguration, ModelAlias, ProviderCredentials, OAuthUser, ResponseProviderMapping, RequestUsage, RequestUsageHourly, RequestUsageMonthly, UserRateLimit, GlobalRateLimit, ModelGroup, ModelGroupMember, ModelGroupInstanceMember, UserModelGroupRateLimit, UserModelAccessPolicy, UserModelAccessException, WebSearchSettings, WebSearchBackend, UserWebSearchSettings, UserWebSearchBackendParams, WEBSEARCH_BACKEND_PARAMS, WEBSEARCH_PARAM_FIELDS, UserManagementSettings, UserManagementSettingsResponse, PoolSettings, PoolSettingsResponse, UserProfile, ProfileModelGroupLimit, ProfileModelAccessException, DEFAULT_PROFILE_ID, USERNAME_MIN_LENGTH, USERNAME_MAX_LENGTH
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


def normalize_new_username(username: str) -> str:
    """Strip a username for a new account and check its length; raises ValueError.

    Normalised before anything reads the name: forms trim client-side, so validating,
    looking up and storing the stripped value keeps the length bound, the uniqueness
    checks and the stored name all talking about the same string.
    """
    username = username.strip()
    if len(username) < USERNAME_MIN_LENGTH:
        raise ValueError(f"Username must be at least {USERNAME_MIN_LENGTH} characters long")
    if len(username) > USERNAME_MAX_LENGTH:
        raise ValueError(f"Username must be at most {USERNAME_MAX_LENGTH} characters long")
    return username


@with_db_retry()
async def create_user(
    db: AsyncSession, username: str, email: str, password: str, is_pending: bool = False,
    profile_id: Optional[int] = None,
) -> User:
    """Create a new user. If is_pending=True, the account requires admin approval before it becomes active."""
    hashed_password = get_password_hash(password)
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_active=not is_pending,
        is_pending_approval=is_pending,
        profile_id=profile_id or DEFAULT_PROFILE_ID,
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


async def get_user_management_settings(db: AsyncSession) -> Optional[UserManagementSettings]:
    """Return the singleton user management settings row (id=1), or None if never saved."""
    result = await db.execute(select(UserManagementSettings).where(UserManagementSettings.id == 1))
    return result.scalar_one_or_none()


async def upsert_user_management_settings(
    db: AsyncSession, allow_self_delete: bool, admin_username: str
) -> UserManagementSettings:
    """Create or update the user management settings singleton."""
    row = await get_user_management_settings(db)
    if row is None:
        row = UserManagementSettings(id=1)
        db.add(row)
    row.allow_self_delete = allow_self_delete
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def load_user_management_settings(db: AsyncSession) -> UserManagementSettingsResponse:
    """Effective user management settings: the saved row, or the model defaults if never saved."""
    row = await get_user_management_settings(db)
    if row is None:
        return UserManagementSettingsResponse()
    return UserManagementSettingsResponse.model_validate(row)


async def is_self_delete_allowed(db: AsyncSession) -> bool:
    """Whether regular users may delete their own account (allowed until an admin says otherwise)."""
    return (await load_user_management_settings(db)).allow_self_delete


async def get_pool_settings(db: AsyncSession) -> Optional[PoolSettings]:
    """Return the singleton pool settings row (id=1), or None if never saved."""
    result = await db.execute(select(PoolSettings).where(PoolSettings.id == 1))
    return result.scalar_one_or_none()


async def upsert_pool_settings(
    db: AsyncSession, max_members: int, max_member_share_pct: Optional[int], admin_username: str
) -> PoolSettings:
    """Create or update the pool settings singleton."""
    row = await get_pool_settings(db)
    if row is None:
        row = PoolSettings(id=1)
        db.add(row)
    row.max_members = max_members
    row.max_member_share_pct = max_member_share_pct
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def load_pool_settings(db: AsyncSession) -> PoolSettingsResponse:
    """Effective pool settings: the saved row, or the model defaults if never saved."""
    row = await get_pool_settings(db)
    if row is None:
        return PoolSettingsResponse()
    return PoolSettingsResponse.model_validate(row)


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


async def list_websearch_backends(db: AsyncSession) -> List[WebSearchBackend]:
    """Every web search backend, by name."""
    from sqlalchemy import func
    result = await db.execute(select(WebSearchBackend).order_by(func.lower(WebSearchBackend.name)))
    return list(result.scalars().all())


async def get_websearch_backend(db: AsyncSession, backend_id: int) -> Optional[WebSearchBackend]:
    result = await db.execute(select(WebSearchBackend).where(WebSearchBackend.id == backend_id))
    return result.scalar_one_or_none()


async def create_websearch_backend(db: AsyncSession, values: dict, admin_username: str) -> WebSearchBackend:
    """Create a backend. With no default backend yet, it becomes the default."""
    row = WebSearchBackend(**values, updated_by=admin_username)
    db.add(row)
    await db.flush()
    settings = await get_websearch_settings(db)
    if settings is None:
        settings = WebSearchSettings(id=1, updated_by=admin_username)
        db.add(settings)
    if settings.default_backend_id is None:
        settings.default_backend_id = row.id
    await db.commit()
    await db.refresh(row)
    return row


async def update_websearch_backend(
    db: AsyncSession, backend_id: int, values: dict, admin_username: str,
) -> Optional[WebSearchBackend]:
    row = await get_websearch_backend(db, backend_id)
    if row is None:
        return None
    for key, value in values.items():
        setattr(row, key, value)
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_websearch_backend(db: AsyncSession, backend_id: int) -> bool:
    """Delete a backend with its users' parameters for it; users who picked it
    go back to the default. SQLite does not enforce the foreign keys, so the
    dependent rows are handled here, in the same commit."""
    from sqlalchemy import delete as _sql_delete, update as _sql_update
    row = await get_websearch_backend(db, backend_id)
    if row is None:
        return False
    await db.execute(_sql_delete(UserWebSearchBackendParams).where(
        UserWebSearchBackendParams.backend_id == backend_id
    ))
    await db.execute(_sql_update(UserWebSearchSettings).where(
        UserWebSearchSettings.backend_id == backend_id
    ).values(backend_id=None))
    await db.delete(row)
    await db.commit()
    return True


async def count_backend_users(db: AsyncSession) -> Dict[int, int]:
    """{backend_id: number of users who picked it}."""
    from sqlalchemy import func
    result = await db.execute(
        select(UserWebSearchSettings.backend_id, func.count())
        .where(UserWebSearchSettings.backend_id.is_not(None))
        .group_by(UserWebSearchSettings.backend_id)
    )
    return dict(result.all())


async def get_user_websearch(
    db: AsyncSession, user_id: int,
) -> Tuple[Optional[UserWebSearchSettings], Dict[int, UserWebSearchBackendParams]]:
    """A user's web search row (or None) and their parameter rows by backend id."""
    row = (await db.execute(
        select(UserWebSearchSettings).where(UserWebSearchSettings.user_id == user_id)
    )).scalar_one_or_none()
    params = (await db.execute(
        select(UserWebSearchBackendParams).where(UserWebSearchBackendParams.user_id == user_id)
    )).scalars().all()
    return row, {p.backend_id: p for p in params}


async def put_user_websearch(
    db: AsyncSession, user_id: int, backend_id: Optional[int], limits: dict,
    params_backend_id: Optional[int], params: dict,
) -> None:
    """Replace a user's backend choice and limits, and their parameters for one
    backend (the others are kept). Rows left with nothing set are removed."""
    from sqlalchemy.exc import IntegrityError
    try:
        await _put_user_websearch(db, user_id, backend_id, limits, params_backend_id, params)
    except IntegrityError:
        # A concurrent save (another tab) inserted the same row first. Once more
        # finds it and updates it: the last save wins, as for any PUT.
        await db.rollback()
        await _put_user_websearch(db, user_id, backend_id, limits, params_backend_id, params)


async def _put_user_websearch(
    db: AsyncSession, user_id: int, backend_id: Optional[int], limits: dict,
    params_backend_id: Optional[int], params: dict,
) -> None:
    row, by_backend = await get_user_websearch(db, user_id)
    if params_backend_id is not None:
        params_row = by_backend.get(params_backend_id)
        if any(v is not None for v in params.values()):
            if params_row is None:
                params_row = UserWebSearchBackendParams(user_id=user_id, backend_id=params_backend_id)
                db.add(params_row)
                by_backend[params_backend_id] = params_row
            for field in WEBSEARCH_PARAM_FIELDS:
                setattr(params_row, field, params.get(field))
            params_row.updated_at = datetime.utcnow()
        elif params_row is not None:
            await db.delete(params_row)
            del by_backend[params_backend_id]

    if backend_id is None and all(v is None for v in limits.values()) and not by_backend:
        if row is not None:
            await db.delete(row)
    else:
        if row is None:
            row = UserWebSearchSettings(user_id=user_id)
            db.add(row)
        row.backend_id = backend_id
        for key, value in limits.items():
            setattr(row, key, value)
        row.updated_at = datetime.utcnow()
    await db.commit()


async def delete_user_websearch(db: AsyncSession, user_id: int) -> None:
    """Drop all of a user's web search choices (back to the admin defaults)."""
    from sqlalchemy import delete as _sql_delete
    for model in (UserWebSearchBackendParams, UserWebSearchSettings):
        await db.execute(_sql_delete(model).where(model.user_id == user_id))
    await db.commit()


async def load_websearch_snapshot_rows(db: AsyncSession):
    """(settings row or None, backends, user rows, user parameter rows), for the
    request-time snapshot in app.websearch.settings."""
    settings = await get_websearch_settings(db)
    backends = await list_websearch_backends(db)
    users = (await db.execute(select(UserWebSearchSettings))).scalars().all()
    params = (await db.execute(select(UserWebSearchBackendParams))).scalars().all()
    return settings, backends, list(users), list(params)


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
            UserModelAccessPolicy,
            UserModelAccessException,
            UserWebSearchSettings,
            UserWebSearchBackendParams,
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


async def get_all_model_aliases(
    db: AsyncSession, profile_id: int = DEFAULT_PROFILE_ID, all_profiles: bool = False,
) -> List[ModelAlias]:
    """Return one profile's model aliases (Default's unless given) in evaluation order.

    all_profiles=True returns every profile's rows, for the resolver.
    """
    q = select(ModelAlias)
    if not all_profiles:
        q = q.where(ModelAlias.profile_id == profile_id)
    result = await db.execute(q.order_by(ModelAlias.priority, ModelAlias.id))
    return list(result.scalars().all())


async def get_model_alias(db: AsyncSession, alias: str, profile_id: int = DEFAULT_PROFILE_ID) -> Optional[ModelAlias]:
    """Return an alias by its exact client-facing name within one profile's list."""
    result = await db.execute(
        select(ModelAlias).where(ModelAlias.alias == alias, ModelAlias.profile_id == profile_id)
    )
    return result.scalar_one_or_none()


async def get_model_alias_by_id(db: AsyncSession, alias_id: int) -> Optional[ModelAlias]:
    """Return an alias by primary key (from the session's identity map if already loaded)."""
    return await db.get(ModelAlias, alias_id)


async def upsert_model_alias(
    db: AsyncSession, alias: str, target_model_id: str, enabled: bool, apis,
    match_type: Optional[str] = None, alias_id: Optional[int] = None,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> Optional[ModelAlias]:
    """Create or update a model alias in *profile_id*'s list.

    With *alias_id*, that row is updated (and may be renamed); returns None if it
    does not exist in this profile. Otherwise the row is looked up by *alias*. New rows
    go to the bottom of their profile's priority order. A *match_type* of None keeps an
    existing row's type. Two concurrent creates may share a priority; ties break by id.
    """
    from sqlalchemy import func
    if alias_id is not None:
        row = await get_model_alias_by_id(db, alias_id)
        if row is None or row.profile_id != profile_id:
            return None
        row.alias = alias
    else:
        row = await get_model_alias(db, alias, profile_id)
    if row is None:
        max_priority = (await db.execute(
            select(func.max(ModelAlias.priority)).where(ModelAlias.profile_id == profile_id)
        )).scalar()
        row = ModelAlias(
            alias=alias,
            profile_id=profile_id,
            match_type=match_type or "exact",
            priority=(max_priority if max_priority is not None else -1) + 1,
        )
        db.add(row)
    elif match_type is not None:
        row.match_type = match_type
    row.target_model_id = target_model_id
    row.enabled = enabled
    row.apis = json.dumps(list(apis))
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def reorder_model_aliases(db: AsyncSession, ids: List[int], profile_id: int = DEFAULT_PROFILE_ID) -> bool:
    """Set priorities to the order of *ids*, which must name every alias of the profile once.

    Returns False (writing nothing) when *ids* does not match the stored rows, so a
    client working from a stale list cannot silently drop rows from the order.
    """
    rows = list((await db.execute(
        select(ModelAlias).where(ModelAlias.profile_id == profile_id)
    )).scalars().all())
    by_id = {row.id: row for row in rows}
    if len(ids) != len(rows) or set(ids) != set(by_id):
        return False
    for position, alias_id in enumerate(ids):
        by_id[alias_id].priority = position
    await db.commit()
    return True


async def delete_model_alias(db: AsyncSession, alias: str, profile_id: int = DEFAULT_PROFILE_ID) -> bool:
    """Delete a model alias by name from one profile's list."""
    row = await get_model_alias(db, alias, profile_id)
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
    """Turn a provider on or off.

    The provider flag is a master switch over its models' own flags, never a bulk
    write to them: a model is served only when both are on (cache._passes_global_gate),
    so switching a provider back on restores each model's individual setting.
    """
    try:
        provider = await get_provider_credentials(db, provider_key)
        if not provider:
            return False

        provider.enabled = enabled
        provider.updated_at = datetime.utcnow()
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("provider_toggle", provider_key=provider_key, enabled=enabled)
        
        return True
    except Exception as e:
        await db.rollback()
        raise e


async def toggle_model_configuration(db: AsyncSession, model_id: str, enabled: bool) -> bool:
    """Turn one model on or off. Never touches its provider's flag; see
    toggle_provider_configuration."""
    try:
        model = await get_model_configuration(db, model_id)
        if not model:
            return False

        model.is_enabled = enabled
        model.updated_at = datetime.utcnow()
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("model_toggle", model_id=model_id, enabled=enabled, provider_key=model.provider_key)
        
        return True
    except Exception as e:
        await db.rollback()
        raise e


@with_db_retry()
async def bulk_toggle_all_models(db: AsyncSession, enabled: bool) -> bool:
    """Enable or disable every model. Provider flags are left alone; see
    toggle_provider_configuration."""
    try:
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
    deployment_groups = kwargs.pop('deployment_groups', None)
    deployments_json = None
    if provider_type == "azure":
        deployments_json = serialize_azure_deployments(
            deployments=deployments,
            openai_deployments=openai_deployments,
            anthropic_deployments=anthropic_deployments,
            deployment_groups=deployment_groups,
        )
    elif deployments:
        deployments_json = json.dumps(deployments)
    
    for field in _CLEARABLE_PROVIDER_FIELDS:
        if kwargs.get(field) == "":
            kwargs[field] = None

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
        for field, value in _provider_update_fields(credentials.provider_type, kwargs).items():
            if hasattr(credentials, field):
                setattr(credentials, field, value)

        credentials.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(credentials)
        return credentials
    except Exception as e:
        await db.rollback()
        raise e


# Optional provider fields where an empty string from the admin form means
# "clear this override" rather than "leave unchanged".
_CLEARABLE_PROVIDER_FIELDS = frozenset({"anthropic_base_url", "anthropic_auth_style"})


def _provider_update_fields(provider_type: str, kwargs: dict) -> dict:
    """The column values an update writes, shared by update and rename.

    ``None`` means "leave unchanged" and is dropped (so an import that omits
    secrets keeps the stored ones); an empty string clears the optional
    overrides in _CLEARABLE_PROVIDER_FIELDS. The deployment inputs are folded
    into ``deployments_json``: once any of them is supplied it is always
    written, so an empty configuration (e.g. ``deployment_groups=[]`` when
    switching to discovery) stores NULL instead of keeping the old rows.
    """
    import json

    kwargs = dict(kwargs)
    deployments = kwargs.pop('deployments', None)
    openai_deployments = kwargs.pop('openai_deployments', None)
    anthropic_deployments = kwargs.pop('anthropic_deployments', None)
    deployment_groups = kwargs.pop('deployment_groups', None)

    fields = {}
    for field, value in kwargs.items():
        if field in _CLEARABLE_PROVIDER_FIELDS and value == "":
            fields[field] = None
        elif value is not None:
            fields[field] = value

    if provider_type == "azure":
        if (
            deployments is not None
            or openai_deployments is not None
            or anthropic_deployments is not None
            or deployment_groups is not None
        ):
            fields['deployments_json'] = serialize_azure_deployments(
                deployments=deployments,
                openai_deployments=openai_deployments,
                anthropic_deployments=anthropic_deployments,
                deployment_groups=deployment_groups,
            )
    elif deployments is not None:
        fields['deployments_json'] = json.dumps(deployments)
    return fields


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

        # Same field semantics as a normal update: None keeps the old value.
        fields = _provider_update_fields(old_credentials.provider_type, kwargs)

        # Create new provider credentials with updated key
        new_credentials = ProviderCredentials(
            provider_key=new_provider_key,
            provider_type=old_credentials.provider_type,
            instance_name=fields.get('instance_name', old_credentials.instance_name),
            enabled=fields.get('enabled', old_credentials.enabled),
            endpoint=fields.get('endpoint', old_credentials.endpoint),
            api_key=fields.get('api_key', old_credentials.api_key),
            discovery_api_version=fields.get('discovery_api_version', old_credentials.discovery_api_version),
            azure_backend=fields.get('azure_backend', old_credentials.azure_backend),
            region=fields.get('region', old_credentials.region),
            access_key_id=fields.get('access_key_id', old_credentials.access_key_id),
            secret_access_key=fields.get('secret_access_key', old_credentials.secret_access_key),
            base_url=fields.get('base_url', old_credentials.base_url),
            deployments_json=fields.get('deployments_json', old_credentials.deployments_json),
            provider_name=fields.get('provider_name', old_credentials.provider_name),
            dynamic_discovery=fields.get('dynamic_discovery', old_credentials.dynamic_discovery),
            supported_apis=fields.get('supported_apis', old_credentials.supported_apis),
            anthropic_base_url=fields.get('anthropic_base_url', old_credentials.anthropic_base_url),
            anthropic_auth_style=fields.get('anthropic_auth_style', old_credentials.anthropic_auth_style),
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

        await _rename_provider_references(db, old_provider_key, new_provider_key)

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

        await _remove_model_references(db, provider_keys=[provider_key])

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


async def identify_stale_models(
    db: AsyncSession,
    current_model_ids: List[str],
    exclude_provider_keys: Optional[set] = None,
) -> List[Dict]:
    """Identify models in database that are not in the current list of model IDs.

    Models of providers in ``exclude_provider_keys`` (e.g. ones whose sync failed
    or that are disabled) are never reported: they weren't listed because the
    provider wasn't reached, not because they were removed upstream.
    """
    try:
        all_models = await get_all_model_configurations(db)
        current_ids_set = set(current_model_ids)
        excluded = set(exclude_provider_keys or ())

        stale_models = []
        for model in all_models:
            if model.provider_key in excluded:
                continue
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

        await _remove_model_references(db, model_ids=stale_model_ids)
        await db.commit()
        
        # Trigger cache update
        await _update_cache_after_database_change("model_sync")
        
        return deleted_count
    except Exception as e:
        await db.rollback()
        raise e


# ---------------------------------------------------------------------------
# Keeping references to models and providers in step with them. Group members,
# access exceptions, aliases and web-search settings all name models and
# instances by string, with no foreign key, so a rename or a deliberate removal
# has to carry them along by hand. Only those two paths call these: a model sync
# deletes and re-creates the same ids, and must leave references alone.
# ---------------------------------------------------------------------------

def _on_instance(column, provider_key: str):
    """SQL predicate: `column` is a model id on this instance. substr, not LIKE, so
    '_' in a provider_key is not a wildcard."""
    from sqlalchemy import func
    prefix = f"{provider_key}/"
    return func.substr(column, 1, len(prefix)) == prefix


async def _rename_provider_references(db: AsyncSession, old_key: str, new_key: str) -> None:
    """Point every reference at old_key's models and instance at new_key. No commit."""
    import json
    from sqlalchemy import func, update as sa_update

    def moved(column):
        return new_key + "/" + func.substr(column, len(old_key) + 2)

    for model, column in (
        (ModelGroupMember, ModelGroupMember.model_id),
        (ProfileModelAccessException, ProfileModelAccessException.model_id),
        (ModelAlias, ModelAlias.target_model_id),
    ):
        await db.execute(
            sa_update(model).where(_on_instance(column, old_key))
            .values({column.key: moved(column)}).execution_options(synchronize_session=False)
        )
    await db.execute(
        sa_update(ModelGroupInstanceMember).where(ModelGroupInstanceMember.provider_key == old_key)
        .values(provider_key=new_key).execution_options(synchronize_session=False)
    )
    settings = (await db.execute(select(WebSearchSettings))).scalars().all()
    for row in settings:
        keys = json.loads(row.enabled_providers) if row.enabled_providers else None
        if isinstance(keys, list) and old_key in keys:
            row.enabled_providers = json.dumps([new_key if k == old_key else k for k in keys])


async def _remove_model_references(
    db: AsyncSession, model_ids: Sequence[str] = (), provider_keys: Sequence[str] = (),
) -> None:
    """Drop references to removed models (or every model of removed instances). No commit.

    Group members and access exceptions are deleted. Aliases are switched off rather
    than deleted, so the admin still sees what used to be mapped and can repoint it.
    """
    import json
    from sqlalchemy import delete as sa_delete, or_, update as sa_update

    def matching(column):
        conds = [column.in_(list(model_ids))] if model_ids else []
        conds += [_on_instance(column, pk) for pk in provider_keys]
        return or_(*conds)

    if not model_ids and not provider_keys:
        return
    await db.execute(sa_delete(ModelGroupMember).where(matching(ModelGroupMember.model_id)))
    await db.execute(sa_delete(ProfileModelAccessException).where(
        matching(ProfileModelAccessException.model_id)))
    await db.execute(
        sa_update(ModelAlias).where(matching(ModelAlias.target_model_id))
        .values(enabled=False).execution_options(synchronize_session=False)
    )
    if provider_keys:
        await db.execute(sa_delete(ModelGroupInstanceMember).where(
            ModelGroupInstanceMember.provider_key.in_(list(provider_keys))))
        for row in (await db.execute(select(WebSearchSettings))).scalars().all():
            keys = json.loads(row.enabled_providers) if row.enabled_providers else None
            if isinstance(keys, list) and any(k in provider_keys for k in keys):
                row.enabled_providers = json.dumps([k for k in keys if k not in provider_keys])


async def clear_model_group_members(db: AsyncSession) -> None:
    """Empty every group's member lists (system reinit, which drops every model). No commit."""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(ModelGroupMember))
    await db.execute(sa_delete(ModelGroupInstanceMember))


async def reload_reference_caches() -> None:
    """Reload what the reference cleanups above can change: groups and access
    exceptions (the rate-limit snapshot), aliases, and web-search settings."""
    from app.rate_limit import rate_limit_tracker
    from app.model_alias import model_alias_resolver
    from app.websearch.settings import websearch_settings_cache
    for reload in (rate_limit_tracker.refresh_now, model_alias_resolver.load_from_database,
                   websearch_settings_cache.load_from_database):
        try:
            await reload()
        except Exception as e:
            logger.warning("Reference cache reload (%s) failed: %s", getattr(reload, "__qualname__", reload), e)


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

        elif operation == "provider_toggle":
            # The provider flag only; its models keep their own flags.
            provider_key = kwargs.get("provider_key")
            enabled = kwargs.get("enabled")

            if provider_key and enabled is not None:
                provider_manager.model_cache.update_single_provider_config(provider_key, enabled)
                print(f"Cache updated: provider {provider_key} {'enabled' if enabled else 'disabled'}")
        
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


async def get_usage_report_aggregates(db: AsyncSession, start, end) -> dict:
    """Per-user and per-model totals over the inclusive local-date range [start, end].

    Reads request_usage alone: rollup_to_monthly only moves a month whose last day is
    more than 30 days old, so any range with start >= today - 30 is still entirely in
    the daily table. Callers must keep to that bound. Unlike the dashboard's '30d'
    window this takes explicit dates, so a report can end at yesterday and stay final.
    """
    where = [RequestUsage.date >= start, RequestUsage.date <= end]
    user_rows = (await db.execute(_user_rows_q(RequestUsage, where))).all()
    model_rows = (await db.execute(_model_rows_q(RequestUsage, where))).all()

    per_user = _fold_identities(_user_row_dicts(user_rows))
    return {
        "per_user": per_user,
        "per_model": _merge_model_rows(model_rows),
        "total": sum(r["request_count"] for r in per_user),
    }


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

    # Moving everyone onto profiles retires the per-user overrides and global
    # defaults. Nothing is deleted, so a failed backup only warns.
    async with engine.connect() as conn:
        profiles_pending = await _users_need_profile_column(conn)
    if profiles_pending:
        try:
            backup = await asyncio.to_thread(_backup_sqlite_file, "pre-profiles")
            if backup:
                logger.info("Auto-migration: database backed up to %s", backup)
        except Exception as e:
            logger.warning("Auto-migration: pre-profiles backup failed: %s", e)

    # Folding instance groups into model groups drops the old tables, so it gets a
    # backup of its own and, like the usage rekey, refuses to run without one.
    async with engine.connect() as conn:
        groups_unify_pending = await _instance_groups_pending(conn)
    if groups_unify_pending:
        try:
            backup = await asyncio.to_thread(_backup_sqlite_file, "pre-group-unify")
        except Exception as e:
            raise RuntimeError(
                f"instance groups need folding into model groups but the pre-migration "
                f"backup failed: {e}"
            ) from e
        if not backup:
            raise RuntimeError(
                "instance groups need folding into model groups but the database file "
                "could not be located to back up first"
            )
        logger.info("Auto-migration: database backed up to %s", backup)

    # Moving web search onto backend rows deletes nothing (the old columns stay),
    # so a failed backup only warns.
    async with engine.connect() as conn:
        websearch_backends_pending = await _websearch_backends_pending(conn)
    if websearch_backends_pending:
        try:
            backup = await asyncio.to_thread(_backup_sqlite_file, "pre-websearch-backends")
            if backup:
                logger.info("Auto-migration: database backed up to %s", backup)
        except Exception as e:
            logger.warning("Auto-migration: pre-websearch-backends backup failed: %s", e)

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

        # Native-Anthropic overrides for custom providers.
        for column_name, column_type in (("anthropic_base_url", "VARCHAR(500)"), ("anthropic_auth_style", "VARCHAR(20)")):
            try:
                result = await conn.execute(text("PRAGMA table_info(provider_credentials)"))
                columns = [row[1] for row in result.fetchall()]
                if column_name not in columns:
                    logger.info("Auto-migration: Adding '%s' column to provider_credentials", column_name)
                    await conn.execute(text(
                        f"ALTER TABLE provider_credentials ADD COLUMN {column_name} {column_type}"
                    ))
            except Exception as e:
                logger.warning("Auto-migration: Could not add %s column: %s", column_name, e)

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

        # Add 'match_type' and 'priority' to model_aliases for pattern mappings
        # evaluated in admin-defined order. Existing rows become exact matches and
        # keep their previous (alphabetical) display order as their priority.
        try:
            result = await conn.execute(text("PRAGMA table_info(model_aliases)"))
            columns = [row[1] for row in result.fetchall()]

            if columns and 'match_type' not in columns:
                logger.info("Auto-migration: Adding 'match_type' column to model_aliases")
                await conn.execute(text(
                    "ALTER TABLE model_aliases ADD COLUMN match_type VARCHAR(16) NOT NULL DEFAULT 'exact'"
                ))
            if columns and 'priority' not in columns:
                logger.info("Auto-migration: Adding 'priority' column to model_aliases")
                await conn.execute(text(
                    "ALTER TABLE model_aliases ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
                ))
                ids = [row[0] for row in (await conn.execute(
                    text("SELECT id FROM model_aliases ORDER BY alias")
                )).fetchall()]
                for position, alias_id in enumerate(ids):
                    await conn.execute(
                        text("UPDATE model_aliases SET priority = :p WHERE id = :id"),
                        {"p": position, "id": alias_id},
                    )
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add match_type/priority columns to model_aliases: {e}")

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

        # Move the single SearXNG / 4get configuration onto websearch_backends rows.
        # Not wrapped in a swallow-all try: a half-done move would leave web search
        # pointing at nothing, so a failure aborts startup with the transaction
        # rolled back.
        await _migrate_websearch_backends(conn, websearch_backends_pending)

        # Per-pool admin overrides of PoolSettings. NULL inherits the global value, so
        # existing pools keep today's behaviour.
        try:
            result = await conn.execute(text("PRAGMA table_info(request_pools)"))
            columns = [row[1] for row in result.fetchall()]
            if columns:
                for name, ddl in (
                    ('max_members', "ALTER TABLE request_pools ADD COLUMN max_members INTEGER"),
                    ('max_member_share_pct', "ALTER TABLE request_pools ADD COLUMN max_member_share_pct INTEGER"),
                ):
                    if name not in columns:
                        logger.info(f"Auto-migration: Adding '{name}' column to request_pools")
                        await conn.execute(text(ddl))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add request_pools override columns: {e}")

        # Model groups can be switched off; existing groups stay on.
        try:
            result = await conn.execute(text("PRAGMA table_info(model_groups)"))
            columns = [row[1] for row in result.fetchall()]
            if columns and 'enabled' not in columns:
                logger.info("Auto-migration: Adding 'enabled' column to model_groups")
                await conn.execute(text("ALTER TABLE model_groups ADD COLUMN enabled BOOLEAN NOT NULL DEFAULT 1"))
        except Exception as e:
            logger.warning(f"Auto-migration: Could not add model_groups.enabled column: {e}")

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

        await _migrate_profiles(conn, profiles_pending)

        # Fold instance groups into model groups (as instance members). Runs after
        # _migrate_profiles, which reads the old instance tables on a pre-profiles
        # database. Not wrapped in a swallow-all try: it drops tables, so a failure
        # must abort startup with the transaction rolled back.
        await _unify_model_groups(conn, groups_unify_pending)

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


async def _websearch_backends_pending(conn) -> bool:
    """True when websearch_settings predates multiple backends.

    A fresh database gets default_backend_id from create_all, so only a table
    created before it lacks the column.
    """
    from sqlalchemy import text

    columns = [r[1] for r in (await conn.execute(text("PRAGMA table_info(websearch_settings)"))).fetchall()]
    return bool(columns) and "default_backend_id" not in columns


# The single-backend columns of websearch_settings, read once by the migration.
# Older databases may lack the 4get ones (added in a later release).
_LEGACY_WEBSEARCH_COLUMNS = (
    "provider", "searxng_base_url", "engines", "categories", "language", "safesearch",
    "time_range", "fourget_base_url", "fourget_scraper", "fourget_lang", "fourget_country",
)


async def _migrate_websearch_backends(conn, pending: bool) -> None:
    """Turn the stored SearXNG and 4get configurations into backend rows.

    Each configured URL becomes a backend with its parameters, and the backend of
    the old `provider` becomes the default. Adding default_backend_id happens in
    the same transaction, which is what makes this run once. The old columns are
    left in place, so the previous release still finds its settings.
    """
    from sqlalchemy import text

    if not pending:
        return
    await conn.execute(text("ALTER TABLE websearch_settings ADD COLUMN default_backend_id INTEGER"))
    columns = {r[1] for r in (await conn.execute(text("PRAGMA table_info(websearch_settings)"))).fetchall()}
    legacy = [c for c in _LEGACY_WEBSEARCH_COLUMNS if c in columns]
    row = (await conn.execute(text(
        f"SELECT {', '.join(['enabled'] + legacy)} FROM websearch_settings WHERE id = 1"
    ))).mappings().first()
    if row is None:
        logger.info("Auto-migration: web search was never configured; no backends to create")
        return
    row = dict(row)

    created = {}
    if not (await conn.execute(text("SELECT 1 FROM websearch_backends LIMIT 1"))).first():
        for backend_type, name, url_column in (
            ("searxng", "SearXNG", "searxng_base_url"),
            ("fourget", "4get", "fourget_base_url"),
        ):
            url = (row.get(url_column) or "").strip().rstrip("/")
            if not url:
                continue
            params = {p: row.get(p) for p in WEBSEARCH_BACKEND_PARAMS[backend_type]}
            names = ", ".join(params)
            await conn.execute(text(
                "INSERT INTO websearch_backends (name, type, base_url, enabled, user_selectable, "
                f"{names}, created_at, updated_at, updated_by) VALUES (:name, :type, :url, 1, 1, "
                f"{', '.join(':' + p for p in params)}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'migration')"
            ), {"name": name, "type": backend_type, "url": url, **params})
            created[backend_type] = (await conn.execute(
                text("SELECT id FROM websearch_backends WHERE name = :name"), {"name": name}
            )).scalar_one()

    # Backends already there (an old settings table restored next to them) are
    # not duplicated, but one of them is still the default rather than none.
    by_type = dict(created)
    for backend_type, backend_id in (await conn.execute(
        text("SELECT type, id FROM websearch_backends WHERE enabled = 1 ORDER BY id")
    )).fetchall():
        by_type.setdefault(backend_type, backend_id)
    default_id = by_type.get(row.get("provider") or "searxng") or next(iter(by_type.values()), None)
    if default_id is not None:
        await conn.execute(text("UPDATE websearch_settings SET default_backend_id = :d WHERE id = 1"), {"d": default_id})
    elif row.get("enabled"):
        # Unreachable with the old form's validation (enabled needed a URL), but an
        # enabled feature with nowhere to send searches must not stay on.
        await conn.execute(text("UPDATE websearch_settings SET enabled = 0 WHERE id = 1"))
        logger.warning("Auto-migration: web search was enabled without a backend URL; turned it off")
    logger.info(
        "Auto-migration: web search backends created: %s; default backend id %s",
        ", ".join(sorted(created)) or "none", default_id,
    )


async def _users_need_profile_column(conn) -> bool:
    """True when the users table exists but predates user profiles."""
    from sqlalchemy import text

    columns = [r[1] for r in (await conn.execute(text("PRAGMA table_info(users)"))).fetchall()]
    return bool(columns) and "profile_id" not in columns


_MODEL_ALIAS_COLUMNS = (
    "id, alias, target_model_id, enabled, apis, match_type, priority, created_at, updated_at"
)


async def _make_aliases_profile_scoped(conn) -> None:
    """Add model_aliases.profile_id and move alias uniqueness to (profile, alias).

    Safe to run on every start. The mappings that predate profiles were the one list
    every user had, so they become Default's. Deployments carry the old uniqueness as a
    separate UNIQUE index, which is simply dropped. A column-level UNIQUE (very old DDL)
    cannot be dropped in SQLite, so that table is rebuilt from the current model.
    """
    from sqlalchemy import text

    columns = [r[1] for r in (await conn.execute(text("PRAGMA table_info(model_aliases)"))).fetchall()]
    if not columns:
        return
    if "profile_id" not in columns:
        logger.info("Auto-migration: Adding 'profile_id' column to model_aliases (all mappings -> Default)")
        # SQLite cannot ADD a NOT NULL REFERENCES column; the UPDATE below fills it.
        await conn.execute(text(
            "ALTER TABLE model_aliases ADD COLUMN profile_id INTEGER "
            "REFERENCES user_profiles(id) ON DELETE CASCADE"
        ))
    # Before any rebuild: the rebuilt table's profile_id is NOT NULL.
    await conn.execute(text(
        "UPDATE model_aliases SET profile_id = :id WHERE profile_id IS NULL"
    ), {"id": DEFAULT_PROFILE_ID})

    rebuild = False
    for _seq, name, unique, origin, *_ in (await conn.execute(text("PRAGMA index_list(model_aliases)"))).fetchall():
        if not unique:
            continue
        index_cols = [r[2] for r in (await conn.execute(text(f"PRAGMA index_info('{name}')"))).fetchall()]
        if index_cols != ["alias"]:
            continue
        if origin == "c":
            logger.info("Auto-migration: Making %s non-unique (aliases are unique per profile)", name)
            await conn.execute(text(f'DROP INDEX "{name}"'))
        else:
            rebuild = True

    if rebuild:
        logger.info("Auto-migration: Rebuilding model_aliases to drop column-level UNIQUE(alias)")
        await conn.execute(text("DROP TABLE IF EXISTS model_aliases_mig"))
        await conn.execute(text(
            f"CREATE TABLE model_aliases_mig AS SELECT {_MODEL_ALIAS_COLUMNS}, profile_id FROM model_aliases"
        ))
        await conn.execute(text("DROP TABLE model_aliases"))
        await conn.run_sync(lambda sc: Base.metadata.tables["model_aliases"].create(sc))
        await conn.execute(text(
            f"INSERT INTO model_aliases ({_MODEL_ALIAS_COLUMNS}, profile_id) "
            f"SELECT {_MODEL_ALIAS_COLUMNS}, profile_id FROM model_aliases_mig"
        ))
        await conn.execute(text("DROP TABLE model_aliases_mig"))

    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_model_aliases_alias ON model_aliases (alias)"))
    await conn.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_aliases_scope_alias "
        "ON model_aliases (profile_id, alias)"
    ))


# Set when this start folded instance groups into model groups. The precedence flip
# (an explicit model now beats an instance) moves some of today's usage between
# scopes, so startup re-settles every pool once; see app.main.
GROUPS_UNIFIED_THIS_START = False


async def _instance_groups_pending(conn) -> bool:
    from sqlalchemy import text
    return bool((await conn.execute(text(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'instance_groups'"
    ))).first())


def _unique_group_name(base: str, taken: set) -> str:
    """`base`, or `base (2)`, `base (3)`... -- unique case-insensitively, within 64 chars."""
    name, n = base[:64], 1
    while name.lower() in taken or name.lower() == "default":
        n += 1
        suffix = f" ({n})"
        name = base[:64 - len(suffix)] + suffix
    taken.add(name.lower())
    return name


async def _normalise_coupled_model_flags(conn) -> int:
    """Undo the old provider toggle's write-through to model flags.

    The old toggle forced every model of a provider off with it, so a disabled
    provider whose models are *all* off says nothing about the models; switch them
    back on so re-enabling the provider restores them. A provider with mixed flags was
    edited model by model, so it is left alone. Returns the models switched on.
    """
    from sqlalchemy import text
    rows = (await conn.execute(text(
        "SELECT p.provider_key, "
        "  SUM(CASE WHEN m.is_enabled THEN 1 ELSE 0 END), COUNT(m.id) "
        "FROM provider_credentials p JOIN model_configurations m ON m.provider_key = p.provider_key "
        "WHERE NOT p.enabled GROUP BY p.provider_key"
    ))).fetchall()
    switched, mixed = 0, []
    for provider_key, on, total in rows:
        if on == 0:
            await conn.execute(text(
                "UPDATE model_configurations SET is_enabled = 1 WHERE provider_key = :pk"
            ), {"pk": provider_key})
            switched += total
        elif on < total:
            mixed.append(provider_key)
    if mixed:
        logger.warning(
            "Auto-migration: disabled providers with individually disabled models were left "
            "as they are (their model flags now apply when the provider is re-enabled): %s", mixed,
        )
    return switched


async def _unify_model_groups(conn, instance_groups_pending: bool) -> None:
    """Bring model groups onto the single-concept schema.

    Every start: names must be unique case-insensitively and "Default" is reserved for
    the built-in group, so clashing rows are renamed before the lower(name) index goes on.

    Once, while the instance_groups table still exists: each instance group becomes a
    model group whose members are instances, carrying its per-profile limits and its
    pool ledger and carry rows across; then the old tables are dropped.
    """
    from sqlalchemy import text
    global GROUPS_UNIFIED_THIS_START

    taken: set = set()
    for gid, name in (await conn.execute(text("SELECT id, name FROM model_groups ORDER BY id"))).fetchall():
        new = _unique_group_name(name, taken)
        if new != name:
            logger.warning("Auto-migration: renaming model group %r to %r (names are unique "
                           "ignoring case, and 'Default' is reserved)", name, new)
            await conn.execute(text("UPDATE model_groups SET name = :n WHERE id = :id"), {"n": new, "id": gid})

    if instance_groups_pending:
        tables = {r[0] for r in (await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table'")
        )).fetchall()}
        explicit = dict((await conn.execute(text(
            "SELECT m.model_id, g.name FROM model_group_members m JOIN model_groups g ON g.id = m.group_id"
        ))).fetchall())
        groups = (await conn.execute(text(
            "SELECT id, name, description, created_at, updated_at, updated_by FROM instance_groups ORDER BY id"
        ))).fetchall()
        for old_id, name, description, created_at, updated_at, updated_by in groups:
            new_name = name if name.lower() not in taken and name.lower() != "default" \
                else _unique_group_name(f"{name} (instance)", taken)
            taken.add(new_name.lower())
            await conn.execute(text(
                "INSERT INTO model_groups (name, description, created_at, updated_at, updated_by) "
                "VALUES (:n, :d, :c, :u, :b)"
            ), {"n": new_name, "d": description, "c": created_at, "u": updated_at, "b": updated_by})
            new_id = (await conn.execute(text("SELECT last_insert_rowid()"))).scalar_one()
            members = [r[0] for r in (await conn.execute(text(
                "SELECT provider_key FROM instance_group_members WHERE group_id = :g"
            ), {"g": old_id})).fetchall()]
            for pk in members:
                await conn.execute(text(
                    "INSERT INTO model_group_instance_members (group_id, provider_key, created_at) "
                    "VALUES (:g, :pk, CURRENT_TIMESTAMP)"
                ), {"g": new_id, "pk": pk})
                for model_id, owner in explicit.items():
                    if model_id.startswith(pk + "/"):
                        logger.warning(
                            "Auto-migration: %s is in model group %r and its instance was in instance "
                            "group %r; the model group's limits now govern it (explicit models win)",
                            model_id, owner, new_name,
                        )
            if "profile_instance_group_limits" in tables:
                await conn.execute(text(
                    "INSERT INTO profile_model_group_limits "
                    "(profile_id, group_id, rpm_limit, rpd_limit, updated_at, updated_by) "
                    "SELECT profile_id, :new, rpm_limit, rpd_limit, updated_at, updated_by "
                    "FROM profile_instance_group_limits WHERE group_id = :old"
                ), {"new": new_id, "old": old_id})
            else:
                # A database from before profiles: the group's own defaults applied to
                # everyone, and _migrate_profiles had no limits table to seed them into.
                await conn.execute(text(
                    "INSERT INTO profile_model_group_limits "
                    "(profile_id, group_id, rpm_limit, rpd_limit, updated_at, updated_by) "
                    "SELECT :pid, :new, rpm_default, rpd_default, CURRENT_TIMESTAMP, 'migration' "
                    "FROM instance_groups WHERE id = :old "
                    "AND (rpm_default IS NOT NULL OR rpd_default IS NOT NULL)"
                ), {"pid": DEFAULT_PROFILE_ID, "new": new_id, "old": old_id})
            for table in ("request_pool_ledger", "user_rpd_carries"):
                if table in tables:
                    # model_groups ids are reused once the highest is deleted, and these
                    # rows have no foreign key: any already under new_id belong to a
                    # deleted group. Drop them, or they collide with (or pass for) the
                    # rows moved across.
                    await conn.execute(text(
                        f"DELETE FROM {table} WHERE scope_kind = 'model_group' AND scope_id = :new"
                    ), {"new": new_id})
                    await conn.execute(text(
                        f"UPDATE {table} SET scope_kind = 'model_group', scope_id = :new "
                        f"WHERE scope_kind = 'instance_group' AND scope_id = :old"
                    ), {"new": new_id, "old": old_id})
            logger.info("Auto-migration: instance group %r -> model group %r (id %s) with %d instance(s)",
                        name, new_name, new_id, len(members))

        switched = await _normalise_coupled_model_flags(conn)
        if switched:
            logger.info("Auto-migration: re-enabled %d model(s) the old provider toggle had switched "
                        "off; a disabled provider still keeps them out of service", switched)

        # Children first: foreign keys are enforced.
        for table in ("instance_group_members", "profile_instance_group_limits",
                      "user_instance_group_rate_limits", "instance_groups"):
            await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        GROUPS_UNIFIED_THIS_START = True

    try:
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_groups_name_lower ON model_groups (lower(name))"
        ))
    except Exception as e:
        logger.warning("Auto-migration: could not create case-insensitive unique index on "
                       "model_groups(name): %s", e)


async def _migrate_profiles(conn, profiles_pending: bool) -> None:
    """Bring a database onto user profiles.

    The one-time part runs only when users.profile_id is missing, and is deliberately
    not wrapped in a swallow-all try: until that column exists every User query fails,
    so a failure must abort startup with the transaction rolled back. The ALTER comes
    last, so a failure anywhere earlier leaves the trigger in place for the next start.

    Seeding Default from the old global and group defaults keeps every existing user's
    limits as they were. Per-user overrides are not carried over -- profiles replace
    them -- and the rows are left in place, unread, with a count logged.
    """
    from sqlalchemy import text

    await conn.execute(text(
        "INSERT OR IGNORE INTO user_profiles (id, name, description, model_access_mode, created_at, updated_at) "
        "VALUES (:id, 'Default', 'Built-in profile for new users', 'default', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    ), {"id": DEFAULT_PROFILE_ID})
    await conn.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_profiles_name_lower ON user_profiles (lower(name))"
    ))

    await _make_aliases_profile_scoped(conn)

    if profiles_pending:
        tables = {r[0] for r in (await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table'")
        )).fetchall()}
        if "global_rate_limits" in tables:
            await conn.execute(text(
                "UPDATE user_profiles SET "
                "rpm_limit = (SELECT rpm_default FROM global_rate_limits WHERE id = 1), "
                "rpd_limit = (SELECT rpd_default FROM global_rate_limits WHERE id = 1) "
                "WHERE id = :id"
            ), {"id": DEFAULT_PROFILE_ID})
        for groups, limits in (
            ("model_groups", "profile_model_group_limits"),
            ("instance_groups", "profile_instance_group_limits"),
        ):
            # profile_instance_group_limits is no longer created by create_all; it
            # exists only on a database that had instance groups with profiles.
            if groups in tables and limits in tables:
                await conn.execute(text(
                    f"INSERT OR IGNORE INTO {limits} "
                    f"(profile_id, group_id, rpm_limit, rpd_limit, updated_at, updated_by) "
                    f"SELECT :id, id, rpm_default, rpd_default, CURRENT_TIMESTAMP, 'migration' "
                    f"FROM {groups} WHERE rpm_default IS NOT NULL OR rpd_default IS NOT NULL"
                ), {"id": DEFAULT_PROFILE_ID})

        dropped = {}
        for table in ("user_rate_limits", "user_model_group_rate_limits",
                      "user_instance_group_rate_limits", "user_model_access_policies",
                      "user_model_access_exceptions"):
            if table in tables:
                count = (await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar()
                if count:
                    dropped[table] = count
        if dropped:
            logger.warning(
                "Auto-migration: user profiles replace per-user overrides; these rows are "
                "kept but no longer applied: %s", dropped,
            )

        logger.info("Auto-migration: Adding 'profile_id' column to users (all users -> Default)")
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN profile_id INTEGER REFERENCES user_profiles(id)"
        ))

    await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_profile_id ON users (profile_id)"))
    # Cheap repair on every start: nothing may point at a profile that is not there.
    await conn.execute(text(
        "UPDATE users SET profile_id = :id "
        "WHERE profile_id IS NULL OR profile_id NOT IN (SELECT id FROM user_profiles)"
    ), {"id": DEFAULT_PROFILE_ID})


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
    """Return all model groups (with both member kinds loaded), or one if group_id given."""
    from sqlalchemy.orm import selectinload as _sil
    q = select(ModelGroup).options(_sil(ModelGroup.members), _sil(ModelGroup.instance_members)).order_by(ModelGroup.name)
    if group_id is not None:
        q = q.where(ModelGroup.id == group_id)
    result = await db.execute(q)
    rows = result.scalars().all()
    return rows[0] if group_id is not None and rows else (None if group_id is not None else rows)


async def create_model_group(
    db: AsyncSession, name: str, description: Optional[str], admin_username: str,
    model_ids: Sequence[str] = (), provider_keys: Sequence[str] = (),
) -> ModelGroup:
    """Create a group with its members, in one transaction.

    Each profile's limits for it start as the strictest of the limits that governed
    the members until now, so they never get looser or become unlimited for everyone.
    That is the profile's Default limits for members no group claimed, and the owning
    group's limits for a model carved out of another group's instance.
    """
    from app.profiles import DEFAULT_GROUP_ID
    from app.rate_limit import provider_key_of

    # Instances belong to one group at most, so a new instance member was in Default;
    # an explicit model was in whichever group holds its instance, if any.
    instance_owner = dict((await db.execute(
        select(ModelGroupInstanceMember.provider_key, ModelGroupInstanceMember.group_id)
    )).all())
    sources = {instance_owner.get(provider_key_of(mid), DEFAULT_GROUP_ID) for mid in model_ids}
    if provider_keys or not sources:
        sources.add(DEFAULT_GROUP_ID)
    group_limits = {
        (r.profile_id, r.group_id): (r.rpm_limit, r.rpd_limit)
        for r in (await db.execute(select(ProfileModelGroupLimit).where(
            ProfileModelGroupLimit.group_id.in_(sources - {DEFAULT_GROUP_ID})
        ))).scalars()
    }

    def strictest(values):
        values = [v for v in values if v is not None]
        return min(values) if values else None

    row = ModelGroup(name=name, description=description, updated_by=admin_username)
    db.add(row)
    await db.flush()
    db.add_all([ModelGroupMember(group_id=row.id, model_id=mid) for mid in model_ids])
    db.add_all([ModelGroupInstanceMember(group_id=row.id, provider_key=pk) for pk in provider_keys])
    for profile in (await db.execute(select(UserProfile))).scalars().all():
        limits = [
            (profile.rpm_limit, profile.rpd_limit) if gid == DEFAULT_GROUP_ID
            else group_limits.get((profile.id, gid), (None, None))
            for gid in sources
        ]
        rpm, rpd = strictest(l[0] for l in limits), strictest(l[1] for l in limits)
        if rpm is not None or rpd is not None:
            db.add(ProfileModelGroupLimit(
                profile_id=profile.id, group_id=row.id, rpm_limit=rpm,
                rpd_limit=rpd, updated_by=admin_username,
            ))
    await db.commit()
    await db.refresh(row)
    return row


async def update_model_group(
    db: AsyncSession, group_id: int, fields: dict, admin_username: str,
    members: Optional[Tuple[Sequence[str], Sequence[str]]] = None,
) -> Optional[ModelGroup]:
    """Update a group's fields and, when `members` is (model_ids, provider_keys), replace
    its members too -- in one commit, so a rejected member list leaves the name alone."""
    result = await db.execute(select(ModelGroup).where(ModelGroup.id == group_id))
    row = result.scalar_one_or_none()
    if row is None:
        return None
    for k, v in fields.items():
        setattr(row, k, v)
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    if members is not None:
        await _replace_group_members(db, group_id, *members)
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


async def set_group_members(
    db: AsyncSession, group_id: int, model_ids: Sequence[str], provider_keys: Sequence[str] = (),
) -> None:
    """Replace both member lists for a group."""
    await _replace_group_members(db, group_id, model_ids, provider_keys)
    await db.commit()


async def _replace_group_members(
    db: AsyncSession, group_id: int, model_ids: Sequence[str], provider_keys: Sequence[str],
) -> None:
    """Replace both member lists for a group. No commit."""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(ModelGroupMember).where(ModelGroupMember.group_id == group_id))
    await db.execute(sa_delete(ModelGroupInstanceMember).where(ModelGroupInstanceMember.group_id == group_id))
    # Flush the deletes first: a key moving between this group's own rows must not
    # meet itself in the unique index.
    await db.flush()
    db.add_all([ModelGroupMember(group_id=group_id, model_id=mid) for mid in model_ids])
    db.add_all([ModelGroupInstanceMember(group_id=group_id, provider_key=pk) for pk in provider_keys])


async def model_group_overview(db: AsyncSession) -> dict:
    """Resolve every model to its group from the database, the way
    rate_limit_tracker.group_for does at request time.

    Returns {"groups": [ModelGroup rows], "models": {model_id: is_enabled},
    "providers": {provider_key: enabled}, "effective": {group_id: [model_id]}}, with
    the Default group under DEFAULT_GROUP_ID. Used by the admin views, which must show
    stale members and live instance coverage that the tracker snapshot does not keep.
    """
    from app.profiles import DEFAULT_GROUP_ID
    from app.rate_limit import provider_key_of

    groups = await list_model_groups(db)
    models = {m.model_id: bool(m.is_enabled) for m in (await db.execute(select(ModelConfiguration))).scalars()}
    providers = {
        p.provider_key: bool(p.enabled)
        for p in (await db.execute(select(ProviderCredentials))).scalars()
    }
    by_model = {m.model_id: g.id for g in groups for m in g.members}
    by_instance = {m.provider_key: g.id for g in groups for m in g.instance_members}
    effective: Dict[int, List[str]] = {g.id: [] for g in groups}
    effective[DEFAULT_GROUP_ID] = []
    for model_id in sorted(models):
        gid = by_model.get(model_id)
        if gid is None:
            gid = by_instance.get(provider_key_of(model_id), DEFAULT_GROUP_ID)
        effective[gid].append(model_id)
    return {"groups": groups, "models": models, "providers": providers, "effective": effective,
            "by_model": by_model, "by_instance": by_instance}


async def get_model_group_limits(db: AsyncSession, group_id: int) -> Optional[ModelGroup]:
    result = await db.execute(select(ModelGroup).where(ModelGroup.id == group_id))
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# User-profile helpers. (Not "user_profile": update_user_profile above is the
# username/email edit and predates these.)
# ---------------------------------------------------------------------------

def _profile_id_expr():
    """users.profile_id with NULL read as the default profile."""
    from sqlalchemy import func
    return func.coalesce(User.profile_id, DEFAULT_PROFILE_ID)


async def get_profile(db: AsyncSession, profile_id: int) -> Optional[UserProfile]:
    result = await db.execute(select(UserProfile).where(UserProfile.id == profile_id))
    return result.scalar_one_or_none()


async def get_profile_by_name(db: AsyncSession, name: str) -> Optional[UserProfile]:
    """Case-insensitive lookup, matching the uniqueness index."""
    from sqlalchemy import func
    result = await db.execute(select(UserProfile).where(func.lower(UserProfile.name) == name.lower()))
    return result.scalar_one_or_none()


async def list_profiles(db: AsyncSession, profile_id: Optional[int] = None) -> List[dict]:
    """Every profile (or just `profile_id`) with its member, exception and mapping
    counts, Default first."""
    from sqlalchemy import func

    def only(query, column):
        return query if profile_id is None else query.where(column == profile_id)

    profiles = (await db.execute(
        only(select(UserProfile), UserProfile.id).order_by(UserProfile.id)
    )).scalars().all()
    members = dict((await db.execute(
        only(select(_profile_id_expr(), func.count()).select_from(User), _profile_id_expr())
        .group_by(_profile_id_expr())
    )).all())
    exceptions = dict((await db.execute(
        only(select(ProfileModelAccessException.profile_id, func.count()),
             ProfileModelAccessException.profile_id)
        .group_by(ProfileModelAccessException.profile_id)
    )).all())
    aliases = dict((await db.execute(
        only(select(ModelAlias.profile_id, func.count()), ModelAlias.profile_id)
        .group_by(ModelAlias.profile_id)
    )).all())
    return [
        {
            "id": p.id, "name": p.name, "description": p.description,
            "is_default": p.id == DEFAULT_PROFILE_ID,
            "member_count": members.get(p.id, 0),
            "rpm_limit": p.rpm_limit, "rpd_limit": p.rpd_limit,
            "model_access_mode": p.model_access_mode or "default",
            "exception_count": exceptions.get(p.id, 0),
            "alias_count": aliases.get(p.id, 0),
            "updated_at": p.updated_at, "updated_by": p.updated_by,
        }
        for p in profiles
    ]


async def create_profile(
    db: AsyncSession, name: str, description: Optional[str], admin_username: str,
    clone_from_id: Optional[int] = None,
) -> UserProfile:
    """Create a profile, optionally copying everything another profile configures.

    A clone copies the overall limits, the access mode and exceptions, every group
    limit and the profile's own model mappings, in one transaction.
    """
    row = UserProfile(name=name, description=description, updated_by=admin_username)
    db.add(row)
    await db.flush()
    if clone_from_id is not None:
        src = await get_profile(db, clone_from_id)
        if src is not None:
            row.rpm_limit, row.rpd_limit = src.rpm_limit, src.rpd_limit
            row.model_access_mode = src.model_access_mode
            for lim in (await db.execute(select(ProfileModelGroupLimit)
                                         .where(ProfileModelGroupLimit.profile_id == src.id))).scalars():
                db.add(ProfileModelGroupLimit(profile_id=row.id, group_id=lim.group_id, rpm_limit=lim.rpm_limit,
                                              rpd_limit=lim.rpd_limit, updated_by=admin_username))
            for ex in (await db.execute(select(ProfileModelAccessException)
                                        .where(ProfileModelAccessException.profile_id == src.id))).scalars():
                db.add(ProfileModelAccessException(profile_id=row.id, model_id=ex.model_id,
                                                   is_allowed=ex.is_allowed, updated_by=admin_username))
            for al in await get_all_model_aliases(db, profile_id=src.id):
                db.add(ModelAlias(profile_id=row.id, alias=al.alias, target_model_id=al.target_model_id,
                                  enabled=al.enabled, apis=al.apis, match_type=al.match_type,
                                  priority=al.priority))
    await db.commit()
    await db.refresh(row)
    return row


async def update_profile(
    db: AsyncSession, profile_id: int, fields: dict, admin_username: str,
) -> Optional[UserProfile]:
    row = await get_profile(db, profile_id)
    if row is None:
        return None
    for k, v in fields.items():
        setattr(row, k, v)
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return row


async def delete_profile(db: AsyncSession, profile_id: int, reassign_to: int) -> List[int]:
    """Move a profile's members to *reassign_to*, then delete it with everything it owns.

    Returns the ids of the users moved. Callers validate both ids first; the default
    profile is refused here too, as the last line of defence. Child rows are deleted
    explicitly rather than trusted to ON DELETE CASCADE (see permanently_delete_user).
    """
    from sqlalchemy import delete as sa_delete, update as sa_update
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("the default profile cannot be deleted")
    if reassign_to == profile_id:
        raise ValueError("members cannot be reassigned to the profile being deleted")
    moved = list((await db.execute(
        select(User.id).where(_profile_id_expr() == profile_id)
    )).scalars().all())
    if moved:
        await db.execute(sa_update(User).where(_profile_id_expr() == profile_id).values(profile_id=reassign_to))
    for model in (ProfileModelGroupLimit, ProfileModelAccessException, ModelAlias):
        await db.execute(sa_delete(model).where(model.profile_id == profile_id))
    await db.execute(sa_delete(UserProfile).where(UserProfile.id == profile_id))
    await db.commit()
    return moved


async def assign_users_profile(db: AsyncSession, user_ids: List[int], profile_id: int) -> List[int]:
    """Put these users in *profile_id*; returns the ids that exist (and were moved)."""
    from sqlalchemy import update as sa_update
    found = list((await db.execute(select(User.id).where(User.id.in_(user_ids)))).scalars().all())
    if found:
        await db.execute(sa_update(User).where(User.id.in_(found)).values(profile_id=profile_id))
    await db.commit()
    return found


async def list_profile_member_ids(db: AsyncSession, profile_ids: List[int]) -> List[int]:
    result = await db.execute(select(User.id).where(_profile_id_expr().in_(profile_ids)))
    return list(result.scalars().all())


async def list_profile_members(db: AsyncSession, profile_id: int) -> List[User]:
    result = await db.execute(
        select(User).where(_profile_id_expr() == profile_id).order_by(User.username)
    )
    return list(result.scalars().all())


async def upsert_profile_limits(
    db: AsyncSession, profile_id: int, rpm: Optional[int], rpd: Optional[int],
    admin_username: str, fields_set: set,
) -> Optional[UserProfile]:
    """Set a profile's Default-group limits; only fields in fields_set change."""
    fields = {}
    if "rpm_limit" in fields_set:
        fields["rpm_limit"] = rpm
    if "rpd_limit" in fields_set:
        fields["rpd_limit"] = rpd
    return await update_profile(db, profile_id, fields, admin_username)


async def _upsert_profile_group_row(
    db: AsyncSession, model, profile_id: int, group_id: int,
    rpm: Optional[int], rpd: Optional[int], admin_username: str, fields_set: set,
) -> None:
    row = (await db.execute(select(model).where(
        model.profile_id == profile_id, model.group_id == group_id,
    ))).scalar_one_or_none()
    if row is None:
        row = model(profile_id=profile_id, group_id=group_id)
        db.add(row)
    if "rpm_limit" in fields_set:
        row.rpm_limit = rpm
    if "rpd_limit" in fields_set:
        row.rpd_limit = rpd
    if row.rpm_limit is None and row.rpd_limit is None:
        # Both unlimited is what a missing row already means.
        if row in db.new:
            db.expunge(row)
        else:
            await db.delete(row)
        return
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()


async def upsert_profile_group_limit(
    db: AsyncSession, profile_id: int, group_id: int, rpm: Optional[int], rpd: Optional[int],
    admin_username: str, fields_set: set,
) -> None:
    """Set one profile's limits for a group. DEFAULT_GROUP_ID writes the profile's own columns."""
    from app.profiles import DEFAULT_GROUP_ID
    if group_id == DEFAULT_GROUP_ID:
        await upsert_profile_limits(db, profile_id, rpm, rpd, admin_username, fields_set)
        return
    await _upsert_profile_group_row(
        db, ProfileModelGroupLimit, profile_id, group_id, rpm, rpd, admin_username, fields_set,
    )
    await db.commit()


async def set_group_limit_for_all_profiles(
    db: AsyncSession, group_id: int, rpm: Optional[int], rpd: Optional[int],
    admin_username: str, fields_set: set = frozenset({"rpm_limit", "rpd_limit"}),
) -> None:
    """Give every profile the same limits for one group (DEFAULT_GROUP_ID included).

    Only fields in fields_set change, so a limit the admin left blank keeps each
    profile's own value.
    """
    from app.profiles import DEFAULT_GROUP_ID
    for pid in (await db.execute(select(UserProfile.id))).scalars().all():
        if group_id == DEFAULT_GROUP_ID:
            row = await get_profile(db, pid)
            if "rpm_limit" in fields_set:
                row.rpm_limit = rpm
            if "rpd_limit" in fields_set:
                row.rpd_limit = rpd
            row.updated_by = admin_username
            row.updated_at = datetime.utcnow()
        else:
            await _upsert_profile_group_row(
                db, ProfileModelGroupLimit, pid, group_id, rpm, rpd, admin_username, fields_set,
            )
    await db.commit()


async def get_profile_limits(db: AsyncSession, profile_id: int) -> List[dict]:
    """The profile's limits for every group, Default first, set or not.

    member_count is the number of models each group governs right now.
    """
    from app.profiles import DEFAULT_GROUP_ID
    overview = await model_group_overview(db)
    profile = await get_profile(db, profile_id)
    limits = {
        row.group_id: row for row in (await db.execute(
            select(ProfileModelGroupLimit).where(ProfileModelGroupLimit.profile_id == profile_id)
        )).scalars()
    }
    out = [{
        "group_id": DEFAULT_GROUP_ID, "name": "Default", "is_default": True,
        "member_count": len(overview["effective"][DEFAULT_GROUP_ID]),
        "rpm_limit": profile.rpm_limit if profile else None,
        "rpd_limit": profile.rpd_limit if profile else None,
    }]
    out += [
        {
            "group_id": g.id, "name": g.name, "is_default": False, "enabled": g.enabled,
            "member_count": len(overview["effective"].get(g.id, [])),
            "rpm_limit": limits[g.id].rpm_limit if g.id in limits else None,
            "rpd_limit": limits[g.id].rpd_limit if g.id in limits else None,
        }
        for g in overview["groups"]
    ]
    return out


async def set_profile_model_mode(
    db: AsyncSession, profile_id: int, mode: str, admin_username: str,
) -> Optional[UserProfile]:
    return await update_profile(db, profile_id, {"model_access_mode": mode}, admin_username)


async def list_profile_model_exceptions(db: AsyncSession, profile_id: int) -> Dict[str, bool]:
    result = await db.execute(
        select(ProfileModelAccessException).where(ProfileModelAccessException.profile_id == profile_id)
    )
    return {row.model_id: row.is_allowed for row in result.scalars()}


async def upsert_profile_model_exception(
    db: AsyncSession, profile_id: int, model_id: str, is_allowed: bool, admin_username: str,
) -> None:
    row = (await db.execute(select(ProfileModelAccessException).where(
        ProfileModelAccessException.profile_id == profile_id,
        ProfileModelAccessException.model_id == model_id,
    ))).scalar_one_or_none()
    if row is None:
        row = ProfileModelAccessException(profile_id=profile_id, model_id=model_id)
        db.add(row)
    row.is_allowed = is_allowed
    row.updated_by = admin_username
    row.updated_at = datetime.utcnow()
    await db.commit()


async def delete_profile_model_exception(db: AsyncSession, profile_id: int, model_id: str) -> bool:
    from sqlalchemy import delete as sa_delete
    result = await db.execute(sa_delete(ProfileModelAccessException).where(
        ProfileModelAccessException.profile_id == profile_id,
        ProfileModelAccessException.model_id == model_id,
    ))
    await db.commit()
    return bool(result.rowcount)


async def set_profile_model_exceptions_bulk(
    db: AsyncSession, profile_id: int, overrides: Dict[str, bool], admin_username: str,
) -> None:
    """Replace every exception of a profile in one transaction (seeding a custom baseline)."""
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(ProfileModelAccessException).where(
        ProfileModelAccessException.profile_id == profile_id
    ))
    now = datetime.utcnow()
    for model_id, is_allowed in overrides.items():
        db.add(ProfileModelAccessException(
            profile_id=profile_id, model_id=model_id, is_allowed=is_allowed,
            updated_by=admin_username, updated_at=now,
        ))
    await db.commit()

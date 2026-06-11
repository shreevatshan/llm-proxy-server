"""Authentication cache for API keys and JWT tokens with batched database updates.

Uses asyncio.Lock for async-safe locking instead of threading.RLock to prevent
potential deadlocks when mixing async and sync code in FastAPI's async context.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Optional, Set, Any
from datetime import datetime
from dataclasses import dataclass, field

from app.tracing import create_span, add_span_attributes, get_tracer, safe_detach
from opentelemetry import trace
from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)


@dataclass
class CachedAPIKey:
    """
    Cached API key data that can be used in place of the SQLAlchemy APIKey model.

    This class provides the same attributes as the APIKey model, allowing it to be
    used interchangeably in routes that accept Union[User, AdminUser, APIKey].
    """
    id: int
    user_id: int
    api_key: str
    name: str
    is_active: bool
    cached_at: float = field(default_factory=time.time)
    last_used_updated: bool = False  # Track if last_used needs DB update
    username: Optional[str] = None   # Owner's username (for dashboard display)

    # These properties make CachedAPIKey compatible with SQLAlchemy APIKey
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


@dataclass
class CachedUser:
    """
    Cached user data from JWT token validation.
    
    This class provides the same attributes as the User model, allowing it to be
    used interchangeably in routes that accept Union[User, AdminUser, APIKey].
    """
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool = False
    cached_at: float = field(default_factory=time.time)
    
    # These properties make CachedUser compatible with SQLAlchemy User
    hashed_password: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # OAuth fields
    oauth_provider: Optional[str] = None
    oauth_accounts: list = field(default_factory=list)  # List of OAuth accounts


class AuthCache:
    """
    Thread-safe cache for API keys and JWT user lookups.
    
    Features:
    - Caches validated API keys to avoid DB lookups on every request
    - Caches user lookups for JWT token validation
    - Batches last_used timestamp updates and flushes every 30 seconds
    - Refreshes validity status (is_active) every 30 seconds
    - Flushes pending updates on shutdown
    
    TTL vs Refresh:
    - TTL (10-15 min): Handles memory cleanup and removes stale entries
    - Refresh (30 sec): Handles security - detects deactivated keys/users quickly
    """
    
    # Cache TTL in seconds - longer now since validity refresh handles security
    # TTL is mainly for memory management and stale entry cleanup
    API_KEY_TTL = 900   # 15 minutes
    USER_TTL = 600      # 10 minutes
    
    # Flush/refresh interval (30 seconds)
    # - Flushes last_used timestamps to DB
    # - Refreshes is_active status from DB to detect deactivated keys/users
    FLUSH_INTERVAL = 30
    
    def __init__(self):
        self._api_key_cache: Dict[str, CachedAPIKey] = {}
        self._user_cache: Dict[str, CachedUser] = {}  # keyed by username
        self._api_keys_to_update: Set[str] = set()  # API keys with pending last_used updates
        # Use asyncio.Lock for async-safe locking (initialized lazily in async context)
        self._lock: Optional[asyncio.Lock] = None
        # Threading lock for safe initialization of asyncio lock
        self._init_lock = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._db_session_factory = None
    
    def _ensure_lock(self):
        """Ensure asyncio lock is initialized (must be called from async context).
        
        Uses double-checked locking pattern with threading lock for thread-safety.
        """
        if self._lock is None:
            with self._init_lock:
                if self._lock is None:  # Double-check after acquiring lock
                    self._lock = asyncio.Lock()
    
    def set_db_session_factory(self, session_factory):
        """Set the database session factory for flushing updates."""
        self._db_session_factory = session_factory
    
    async def start(self):
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("AuthCache started with %d second flush interval", self.FLUSH_INTERVAL)
    
    async def stop(self):
        """Stop the background flush task and flush pending updates."""
        with create_span(
            "auth_cache.shutdown",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "auth_cache.pending_updates": len(self._api_keys_to_update),
                "auth_cache.cached_api_keys": len(self._api_key_cache),
                "auth_cache.cached_users": len(self._user_cache)
            }
        ) as span:
            self._running = False
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # Final flush on shutdown
            await self._flush_pending_updates()
            
            add_span_attributes(span, {"auth_cache.shutdown_complete": True})
            logger.info("AuthCache stopped, pending updates flushed")
    
    async def _flush_loop(self):
        """Background loop to flush pending updates and refresh validity every FLUSH_INTERVAL seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.FLUSH_INTERVAL)
                # Detach from parent trace context so each flush cycle gets its own trace
                token = otel_context.attach(otel_context.Context())
                try:
                    await self._flush_pending_updates()
                    await self._refresh_validity_status()
                finally:
                    safe_detach(token)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in auth cache flush loop: %s", e)
    
    async def _flush_pending_updates(self):
        """Flush pending last_used updates to the database."""
        if not self._db_session_factory:
            logger.warning("No DB session factory set, skipping flush")
            return
        
        # Get API keys to update atomically (use snapshot and clear)
        self._ensure_lock()
        async with self._lock:
            api_keys_to_update = list(self._api_keys_to_update)
            self._api_keys_to_update.clear()
        
        if not api_keys_to_update:
            return
        
        with create_span(
            "auth_cache.flush_last_used",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "auth_cache.keys_to_update": len(api_keys_to_update),
                "auth_cache.operation": "batch_update_last_used"
            }
        ) as span:
            logger.info("Flushing last_used updates for %d API keys", len(api_keys_to_update))
            
            try:
                async with self._db_session_factory() as db:
                    from sqlalchemy import update
                    from .models import APIKey
                    
                    # Batch update all API keys at once
                    now = datetime.utcnow()
                    await db.execute(
                        update(APIKey)
                        .where(APIKey.api_key.in_(api_keys_to_update))
                        .values(last_used=now)
                    )
                    await db.commit()
                    
                    add_span_attributes(span, {
                        "auth_cache.flush_success": True,
                        "auth_cache.keys_updated": len(api_keys_to_update)
                    })
                    logger.debug("Successfully flushed last_used for %d API keys", len(api_keys_to_update))
            except Exception as e:
                add_span_attributes(span, {
                    "auth_cache.flush_success": False,
                    "auth_cache.error": str(e)
                })
                logger.error("Error flushing API key last_used updates: %s", e)
                # Re-add failed keys for next flush attempt (async)
                async with self._lock:
                    self._api_keys_to_update.update(api_keys_to_update)
    
    async def _refresh_validity_status(self):
        """Refresh validity status of cached API keys and users from the database."""
        if not self._db_session_factory:
            return
        
        # Get list of cached API keys and users to check (snapshot reads - atomic in Python)
        cached_api_keys = list(self._api_key_cache.keys())
        cached_usernames = list(self._user_cache.keys())
        
        if not cached_api_keys and not cached_usernames:
            return
        
        with create_span(
            "auth_cache.refresh_validity",
            kind=trace.SpanKind.INTERNAL,
            attributes={
                "auth_cache.api_keys_to_check": len(cached_api_keys),
                "auth_cache.users_to_check": len(cached_usernames),
                "auth_cache.operation": "refresh_validity"
            }
        ) as span:
            invalidated_api_keys = 0
            invalidated_users = 0
            
            try:
                async with self._db_session_factory() as db:
                    from sqlalchemy import select
                    from .models import APIKey, User
                    
                    # Check API keys validity
                    if cached_api_keys:
                        result = await db.execute(
                            select(APIKey.api_key, APIKey.is_active)
                            .where(APIKey.api_key.in_(cached_api_keys))
                        )
                        valid_keys = {row.api_key: row.is_active for row in result.fetchall()}
                        
                        # Invalidate keys that are no longer active or don't exist
                        self._ensure_lock()
                        async with self._lock:
                            for api_key in cached_api_keys:
                                if api_key not in valid_keys or not valid_keys[api_key]:
                                    if api_key in self._api_key_cache:
                                        del self._api_key_cache[api_key]
                                        self._api_keys_to_update.discard(api_key)
                                        invalidated_api_keys += 1
                                        logger.debug("Invalidated API key from cache: %s...", api_key[:8])
                    
                    # Check users validity
                    if cached_usernames:
                        result = await db.execute(
                            select(User.username, User.is_active)
                            .where(User.username.in_(cached_usernames))
                        )
                        valid_users = {row.username: row.is_active for row in result.fetchall()}
                        
                        # Invalidate users that are no longer active or don't exist
                        async with self._lock:
                            for username in cached_usernames:
                                if username not in valid_users or not valid_users[username]:
                                    if username in self._user_cache:
                                        del self._user_cache[username]
                                        invalidated_users += 1
                                        logger.debug("Invalidated user from cache: %s", username)
                    
                    add_span_attributes(span, {
                        "auth_cache.refresh_success": True,
                        "auth_cache.invalidated_api_keys": invalidated_api_keys,
                        "auth_cache.invalidated_users": invalidated_users
                    })
                    
                    if invalidated_api_keys > 0 or invalidated_users > 0:
                        logger.info("Validity refresh: invalidated %d API keys, %d users", 
                                   invalidated_api_keys, invalidated_users)
                    
            except Exception as e:
                add_span_attributes(span, {
                    "auth_cache.refresh_success": False,
                    "auth_cache.error": str(e)
                })
                logger.error("Error refreshing validity status: %s", e)
    
    # API Key Cache Methods
    # Note: These sync methods use atomic dict operations which are thread-safe in Python
    # For async contexts, use the async versions or rely on atomic reference swaps
    
    def get_cached_api_key(self, api_key: str) -> Optional[CachedAPIKey]:
        """Get cached API key if valid and not expired (sync, uses atomic dict lookup)."""
        cached = self._api_key_cache.get(api_key)
        if cached:
            # Check TTL
            if time.time() - cached.cached_at < self.API_KEY_TTL:
                return cached
            else:
                # Expired, remove from cache (atomic pop)
                self._api_key_cache.pop(api_key, None)
        return None
    
    async def get_cached_api_key_async(self, api_key: str) -> Optional[CachedAPIKey]:
        """Get cached API key if valid and not expired (async version with lock)."""
        self._ensure_lock()
        async with self._lock:
            cached = self._api_key_cache.get(api_key)
            if cached:
                if time.time() - cached.cached_at < self.API_KEY_TTL:
                    return cached
                else:
                    del self._api_key_cache[api_key]
            return None
    
    def cache_api_key(self, api_key: str, db_api_key: Any, username: Optional[str] = None) -> CachedAPIKey:
        """Cache an API key after successful DB lookup (sync, uses atomic dict assignment)."""
        cached = CachedAPIKey(
            id=db_api_key.id,
            user_id=db_api_key.user_id,
            api_key=api_key,
            name=db_api_key.name,
            is_active=db_api_key.is_active,
            cached_at=time.time(),
            username=username,
            created_at=db_api_key.created_at,
            last_used=db_api_key.last_used
        )
        # Atomic dict assignment is thread-safe in Python
        self._api_key_cache[api_key] = cached
        return cached
    
    def mark_api_key_used(self, api_key: str):
        """Mark an API key as used (will be flushed to DB later)."""
        # Set add is atomic in Python
        self._api_keys_to_update.add(api_key)
    
    def invalidate_api_key(self, api_key: str):
        """Remove an API key from cache (e.g., when revoked)."""
        # Atomic operations
        self._api_key_cache.pop(api_key, None)
        self._api_keys_to_update.discard(api_key)
    
    def invalidate_user_api_keys(self, user_id: int):
        """Invalidate all cached API keys for a user."""
        # Take snapshot and iterate
        keys_to_remove = [
            key for key, cached in list(self._api_key_cache.items())
            if cached.user_id == user_id
        ]
        for key in keys_to_remove:
            self._api_key_cache.pop(key, None)
    
    # User Cache Methods (for JWT validation)
    
    def get_cached_user(self, username: str) -> Optional[CachedUser]:
        """Get cached user if valid and not expired (sync, uses atomic dict lookup)."""
        cached = self._user_cache.get(username)
        if cached:
            # Check TTL
            if time.time() - cached.cached_at < self.USER_TTL:
                return cached
            else:
                # Expired, remove from cache (atomic pop)
                self._user_cache.pop(username, None)
        return None
    
    async def get_cached_user_async(self, username: str) -> Optional[CachedUser]:
        """Get cached user if valid and not expired (async version with lock)."""
        self._ensure_lock()
        async with self._lock:
            cached = self._user_cache.get(username)
            if cached:
                if time.time() - cached.cached_at < self.USER_TTL:
                    return cached
                else:
                    del self._user_cache[username]
            return None
    
    def cache_user(self, db_user: Any, is_admin: bool = False) -> CachedUser:
        """Cache a user after successful DB lookup (sync, uses atomic dict assignment)."""
        # Check for oauth_accounts relationship
        oauth_accounts = []
        if hasattr(db_user, 'oauth_accounts'):
            try:
                oauth_accounts = list(db_user.oauth_accounts) if db_user.oauth_accounts else []
            except Exception:
                oauth_accounts = []
        
        cached = CachedUser(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            is_active=db_user.is_active,
            is_admin=is_admin,
            cached_at=time.time(),
            hashed_password=db_user.hashed_password if hasattr(db_user, 'hashed_password') else None,
            created_at=db_user.created_at if hasattr(db_user, 'created_at') else None,
            oauth_provider=db_user.oauth_provider if hasattr(db_user, 'oauth_provider') else None,
            oauth_accounts=oauth_accounts
        )
        # Atomic dict assignment is thread-safe in Python
        self._user_cache[db_user.username] = cached
        return cached
    
    def invalidate_user(self, username: str):
        """Remove a user from cache (atomic pop)."""
        self._user_cache.pop(username, None)
    
    def invalidate_user_by_id(self, user_id: int):
        """Remove a user from cache by ID (uses snapshot iteration)."""
        username_to_remove = None
        # Take snapshot for iteration
        for username, cached in list(self._user_cache.items()):
            if cached.id == user_id:
                username_to_remove = username
                break
        if username_to_remove:
            self._user_cache.pop(username_to_remove, None)
    
    # Cache Statistics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (uses snapshot reads - atomic in Python)."""
        return {
            "api_key_cache_size": len(self._api_key_cache),
            "user_cache_size": len(self._user_cache),
            "pending_last_used_updates": len(self._api_keys_to_update),
            "api_key_ttl": self.API_KEY_TTL,
            "user_ttl": self.USER_TTL,
            "flush_interval": self.FLUSH_INTERVAL
        }
    
    def clear(self):
        """Clear all caches (for testing or emergency)."""
        # Clear operations are atomic in Python
        self._api_key_cache.clear()
        self._user_cache.clear()
        self._api_keys_to_update.clear()
    
    async def clear_async(self):
        """Clear all caches with async lock."""
        self._ensure_lock()
        async with self._lock:
            self._api_key_cache.clear()
            self._user_cache.clear()
            self._api_keys_to_update.clear()


# Global singleton instance
auth_cache = AuthCache()

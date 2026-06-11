"""Per-user request rate limiting (RPM and RPD).

Single-process only: counters live in-memory. RPM buckets reset on restart
(60s window — acceptable). RPD survives via the RequestUsage table.

If the app is ever scaled to multiple workers, swap _minute_buckets for
a shared Redis counter.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 30  # seconds — how often to reload DB config
_RPD_TTL = 5           # seconds — local cache TTL for today-count DB reads


@dataclass
class RateLimitDecision:
    allowed: bool
    rpm_limit: Optional[int]
    rpm_remaining: Optional[int]
    rpd_limit: Optional[int]
    rpd_remaining: Optional[int]
    retry_after_seconds: int
    limited_by: Optional[str]  # "rpm" | "rpd" | None


@dataclass
class _MinuteBucket:
    window: int  # int(time.time() // 60)
    count: int


@dataclass
class _UserOverride:
    user_id: int
    rpm_limit: Optional[int]
    rpd_limit: Optional[int]


@dataclass
class _GlobalDefaults:
    rpm_default: Optional[int]
    rpd_default: Optional[int]


@dataclass
class UserStatus:
    """Read-only snapshot of a user's current rate limit state."""
    rpm_limit: Optional[int]
    rpm_count: int
    rpm_remaining: Optional[int]
    rpd_limit: Optional[int]
    rpd_count: int
    rpd_remaining: Optional[int]


@dataclass
class _RpdCacheEntry:
    count: int
    expires_at: float


@dataclass
class _UserLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimitExceeded(Exception):
    """Raised by _enforce_rate_limit; caught by the per-app exception handler."""

    def __init__(self, body: dict, headers: dict):
        self.body = body
        self.headers = headers
        super().__init__("Rate limit exceeded")

    @classmethod
    def openai(cls, decision: RateLimitDecision) -> "RateLimitExceeded":
        msg = _limit_message(decision)
        return cls(
            body={"error": {
                "message": msg,
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "param": None,
            }},
            headers=_rl_headers(decision),
        )

    @classmethod
    def anthropic(cls, decision: RateLimitDecision) -> "RateLimitExceeded":
        msg = _limit_message(decision)
        return cls(
            body={"type": "error", "error": {"type": "rate_limit_error", "message": msg}},
            headers=_rl_headers(decision),
        )

    @classmethod
    def azure(cls, decision: RateLimitDecision) -> "RateLimitExceeded":
        # Azure uses the OpenAI error envelope with an Azure-style code
        msg = _limit_message(decision)
        return cls(
            body={"error": {
                "message": msg,
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "param": None,
            }},
            headers=_rl_headers(decision),
        )


def _limit_message(d: RateLimitDecision) -> str:
    if d.limited_by == "rpm":
        limit = d.rpm_limit
        return (
            f"Rate limit exceeded: {limit} requests per minute. "
            f"Retry after {d.retry_after_seconds} seconds."
        )
    limit = d.rpd_limit
    return (
        f"Rate limit exceeded: {limit} requests per day. "
        f"Retry after {d.retry_after_seconds} seconds."
    )


def _rl_headers(d: RateLimitDecision) -> dict:
    limit = d.rpm_limit if d.limited_by == "rpm" else d.rpd_limit
    return {
        "Retry-After": str(d.retry_after_seconds),
        "X-RateLimit-Limit-Requests": str(limit) if limit is not None else "unlimited",
        "X-RateLimit-Remaining-Requests": "0",
        "X-RateLimit-Reset-Requests": str(d.retry_after_seconds),
    }


def _seconds_until_utc_midnight() -> int:
    now = datetime.now(tz=timezone.utc)
    from datetime import timedelta
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(1, int((next_midnight - now).total_seconds()))


class RateLimitTracker:
    def __init__(self) -> None:
        self._lock: Optional[asyncio.Lock] = None
        self._init_lock = asyncio.Lock.__new__(asyncio.Lock)  # placeholder; see _get_lock
        self._minute_buckets: Dict[int, _MinuteBucket] = {}
        self._overrides: Dict[int, _UserOverride] = {}
        self._defaults = _GlobalDefaults(rpm_default=None, rpd_default=None)
        self._rpd_cache: Dict[str, _RpdCacheEntry] = {}
        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._db_session_factory: Optional[Callable] = None
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None

    def set_db_session_factory(self, factory: Callable) -> None:
        self._db_session_factory = factory

    async def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_user_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    async def start(self) -> None:
        self._lock = asyncio.Lock()
        self._running = True
        await self._load_config()
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("RateLimitTracker started")

    async def stop(self) -> None:
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("RateLimitTracker stopped")

    async def _refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_REFRESH_INTERVAL)
            await self._load_config()
            self._evict_stale_buckets()

    async def _load_config(self) -> None:
        if self._db_session_factory is None:
            return
        try:
            from app.auth.database import get_global_rate_limit, AsyncSessionLocal
            from app.auth.models import UserRateLimit
            from sqlalchemy.future import select

            async with self._db_session_factory() as db:
                global_row = await get_global_rate_limit(db)
                result = await db.execute(select(UserRateLimit))
                override_rows = result.scalars().all()

            lock = await self._get_lock()
            async with lock:
                if global_row:
                    self._defaults = _GlobalDefaults(
                        rpm_default=global_row.rpm_default,
                        rpd_default=global_row.rpd_default,
                    )
                self._overrides = {
                    row.user_id: _UserOverride(
                        user_id=row.user_id,
                        rpm_limit=row.rpm_limit,
                        rpd_limit=row.rpd_limit,
                    )
                    for row in override_rows
                }
        except Exception as e:
            logger.warning(f"RateLimitTracker: config reload failed: {e}")

    def _evict_stale_buckets(self) -> None:
        current_window = int(time.time() // 60)
        stale = [uid for uid, b in self._minute_buckets.items() if b.window < current_window - 1]
        for uid in stale:
            del self._minute_buckets[uid]

    def invalidate_user(self, user_id: int) -> None:
        self._overrides.pop(user_id, None)
        self._minute_buckets.pop(user_id, None)

    def invalidate_defaults(self) -> None:
        self._defaults = _GlobalDefaults(rpm_default=None, rpd_default=None)

    async def refresh_now(self) -> None:
        """Force an immediate config reload from DB (called after admin edits)."""
        await self._load_config()

    async def get_user_status(self, user_id: int, username: str) -> "UserStatus":
        """Read-only — returns current usage without incrementing."""
        lock = await self._get_lock()
        async with lock:
            override = self._overrides.get(user_id)
            rpm = override.rpm_limit if (override and override.rpm_limit is not None) else self._defaults.rpm_default
            rpd = override.rpd_limit if (override and override.rpd_limit is not None) else self._defaults.rpd_default

            now = time.time()
            current_window = int(now // 60)
            bucket = self._minute_buckets.get(user_id)
            rpm_count = (bucket.count if bucket and bucket.window == current_window else 0)

        rpd_count = await self._get_today_count(username)

        rpm_remaining = max(0, rpm - rpm_count) if rpm is not None else None
        rpd_remaining = max(0, rpd - rpd_count) if rpd is not None else None
        return UserStatus(
            rpm_limit=rpm,
            rpm_count=rpm_count,
            rpm_remaining=rpm_remaining,
            rpd_limit=rpd,
            rpd_count=rpd_count,
            rpd_remaining=rpd_remaining,
        )

    async def check_and_increment(self, user_id: int, username: str) -> RateLimitDecision:
        global_lock = await self._get_lock()
        async with global_lock:
            override = self._overrides.get(user_id)
            rpm = override.rpm_limit if (override and override.rpm_limit is not None) else self._defaults.rpm_default
            rpd = override.rpd_limit if (override and override.rpd_limit is not None) else self._defaults.rpd_default

        # Per-user lock serializes the RPM check + RPD check + increment for this user,
        # eliminating the TOCTOU race where two concurrent requests both pass RPD.
        user_lock = self._get_user_lock(user_id)
        async with user_lock:
            now = time.time()
            current_window = int(now // 60)

            # Always maintain the RPM bucket so get_user_status can report live counts
            # even when no RPM limit is configured.
            bucket = self._minute_buckets.get(user_id)
            if bucket is None or bucket.window != current_window:
                bucket = _MinuteBucket(window=current_window, count=0)
                self._minute_buckets[user_id] = bucket

            # RPM check
            if rpm is not None and bucket.count >= rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=rpm, rpm_remaining=0,
                    rpd_limit=rpd, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="rpm",
                )

            # RPD check (DB call while holding per-user lock — prevents concurrent over-admission)
            if rpd is not None:
                today_count = await self._get_today_count(username)
                if today_count >= rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=rpm, rpm_remaining=None,
                        rpd_limit=rpd, rpd_remaining=0,
                        retry_after_seconds=_seconds_until_utc_midnight(),
                        limited_by="rpd",
                    )

            # All checks passed — increment
            bucket.count += 1
            rpm_remaining = (rpm - bucket.count) if rpm is not None else None

        rpd_count = await self._get_today_count(username)
        rpd_remaining = max(0, rpd - rpd_count) if rpd is not None else None

        return RateLimitDecision(
            allowed=True,
            rpm_limit=rpm, rpm_remaining=rpm_remaining,
            rpd_limit=rpd, rpd_remaining=rpd_remaining,
            retry_after_seconds=0,
            limited_by=None,
        )

    async def _get_today_count(self, user_identity: str) -> int:
        """Return today's total request count with a 5-second TTL cache."""
        now = time.time()
        entry = self._rpd_cache.get(user_identity)
        if entry and entry.expires_at > now:
            return entry.count

        try:
            from app.request_tracker import request_tracker
            count = await request_tracker.get_today_count(user_identity)
        except Exception:
            count = 0

        self._rpd_cache[user_identity] = _RpdCacheEntry(
            count=count, expires_at=now + _RPD_TTL
        )
        return count


rate_limit_tracker = RateLimitTracker()

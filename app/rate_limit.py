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
from typing import Callable, Dict, List, Optional, Tuple
from app import time_utils

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
    limited_by: Optional[str]  # "rpm" | "rpd" | "group_rpm" | "group_rpd" | None
    # Group context (set when limited_by is group_rpm or group_rpd, or for headers)
    group_id: Optional[int] = None
    group_name: Optional[str] = None
    group_rpm_limit: Optional[int] = None
    group_rpm_remaining: Optional[int] = None
    group_rpd_limit: Optional[int] = None
    group_rpd_remaining: Optional[int] = None


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
class _ModelGroupSnapshot:
    group_id: int
    name: str
    rpm_default: Optional[int]
    rpd_default: Optional[int]
    model_ids: List[str]


@dataclass
class _UserGroupOverride:
    user_id: int
    group_id: int
    rpm_limit: Optional[int]
    rpd_limit: Optional[int]


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


def _human_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        parts = [f"{minutes} minute{'s' if minutes != 1 else ''}"]
        if secs:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")
        return " ".join(parts)
    hours, mins = divmod(minutes, 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
    if mins:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    return " ".join(parts)


def _limit_message(d: RateLimitDecision) -> str:
    if d.limited_by == "group_rpm":
        return (
            f"Rate limit exceeded for model group '{d.group_name}': {d.group_rpm_limit} requests per minute. "
            f"Retry after {_human_duration(d.retry_after_seconds)}."
        )
    if d.limited_by == "group_rpd":
        return (
            f"Rate limit exceeded for model group '{d.group_name}': {d.group_rpd_limit} requests per day. "
            f"Retry after {_human_duration(d.retry_after_seconds)}."
        )
    if d.limited_by == "rpm":
        limit = d.rpm_limit
        return (
            f"Rate limit exceeded: {limit} requests per minute. "
            f"Retry after {_human_duration(d.retry_after_seconds)}."
        )
    limit = d.rpd_limit
    return (
        f"Rate limit exceeded: {limit} requests per day. "
        f"Retry after {_human_duration(d.retry_after_seconds)}."
    )


def _rl_headers(d: RateLimitDecision) -> dict:
    if d.limited_by in ("group_rpm", "group_rpd"):
        limit = d.group_rpm_limit if d.limited_by == "group_rpm" else d.group_rpd_limit
    else:
        limit = d.rpm_limit if d.limited_by == "rpm" else d.rpd_limit
    return {
        "Retry-After": str(d.retry_after_seconds),
        "X-RateLimit-Limit-Requests": str(limit) if limit is not None else "unlimited",
        "X-RateLimit-Remaining-Requests": "0",
        "X-RateLimit-Reset-Requests": str(d.retry_after_seconds),
    }


def _seconds_until_utc_midnight() -> int:
    return max(1, int(time_utils.seconds_until_local_midnight()))


def _effective_limit(user_val: Optional[int], group_val: Optional[int]) -> Optional[int]:
    """Return the stricter (minimum) of two nullable limits. None means unlimited."""
    if user_val is None:
        return group_val
    if group_val is None:
        return user_val
    return min(user_val, group_val)


class RateLimitTracker:
    def __init__(self) -> None:
        self._lock: Optional[asyncio.Lock] = None
        self._init_lock = asyncio.Lock.__new__(asyncio.Lock)  # placeholder; see _get_lock
        self._minute_buckets: Dict[int, _MinuteBucket] = {}
        # Group RPM buckets keyed by (user_id, group_id)
        self._group_minute_buckets: Dict[Tuple[int, int], _MinuteBucket] = {}
        self._overrides: Dict[int, _UserOverride] = {}
        self._defaults = _GlobalDefaults(rpm_default=None, rpd_default=None)
        self._rpd_cache: Dict[str, _RpdCacheEntry] = {}
        # Group RPD cache keyed by (user_identity, group_id)
        self._group_rpd_cache: Dict[Tuple[str, int], _RpdCacheEntry] = {}
        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._db_session_factory: Optional[Callable] = None
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        # Model-group state
        self._groups: Dict[int, _ModelGroupSnapshot] = {}        # group_id → snapshot
        self._model_to_group: Dict[str, int] = {}               # model_id → group_id
        self._user_group_overrides: Dict[Tuple[int, int], _UserGroupOverride] = {}  # (user_id, group_id)

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
            from app.auth.models import UserRateLimit, ModelGroup, ModelGroupMember, UserModelGroupRateLimit
            from sqlalchemy.future import select
            from sqlalchemy.orm import selectinload

            async with self._db_session_factory() as db:
                global_row = await get_global_rate_limit(db)
                result = await db.execute(select(UserRateLimit))
                override_rows = result.scalars().all()

                # Load model groups with members
                result = await db.execute(
                    select(ModelGroup).options(selectinload(ModelGroup.members))
                )
                group_rows = result.scalars().all()

                # Load per-user group overrides
                result = await db.execute(select(UserModelGroupRateLimit))
                user_group_rows = result.scalars().all()

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
                self._groups = {
                    g.id: _ModelGroupSnapshot(
                        group_id=g.id,
                        name=g.name,
                        rpm_default=g.rpm_default,
                        rpd_default=g.rpd_default,
                        model_ids=[m.model_id for m in g.members],
                    )
                    for g in group_rows
                }
                self._model_to_group = {
                    m.model_id: g.id
                    for g in group_rows
                    for m in g.members
                }
                self._user_group_overrides = {
                    (row.user_id, row.group_id): _UserGroupOverride(
                        user_id=row.user_id,
                        group_id=row.group_id,
                        rpm_limit=row.rpm_limit,
                        rpd_limit=row.rpd_limit,
                    )
                    for row in user_group_rows
                }
        except Exception as e:
            logger.warning(f"RateLimitTracker: config reload failed: {e}")

    def _evict_stale_buckets(self) -> None:
        current_window = int(time.time() // 60)
        stale = [uid for uid, b in self._minute_buckets.items() if b.window < current_window - 1]
        for uid in stale:
            del self._minute_buckets[uid]
        stale_g = [k for k, b in self._group_minute_buckets.items() if b.window < current_window - 1]
        for k in stale_g:
            del self._group_minute_buckets[k]

    def invalidate_user(self, user_id: int) -> None:
        self._overrides.pop(user_id, None)
        self._minute_buckets.pop(user_id, None)

    def invalidate_defaults(self) -> None:
        self._defaults = _GlobalDefaults(rpm_default=None, rpd_default=None)

    def invalidate_group(self, group_id: int) -> None:
        """Remove cached group snapshot and all related RPM buckets."""
        snap = self._groups.pop(group_id, None)
        if snap:
            for mid in snap.model_ids:
                self._model_to_group.pop(mid, None)
        # Remove group RPM buckets for this group
        stale = [k for k in self._group_minute_buckets if k[1] == group_id]
        for k in stale:
            del self._group_minute_buckets[k]
        # Remove group RPD cache entries for this group
        stale_rpd = [k for k in self._group_rpd_cache if k[1] == group_id]
        for k in stale_rpd:
            del self._group_rpd_cache[k]

    def invalidate_user_group(self, user_id: int, group_id: int) -> None:
        self._user_group_overrides.pop((user_id, group_id), None)
        self._group_minute_buckets.pop((user_id, group_id), None)
        self._group_rpd_cache.pop((str(user_id), group_id), None)

    async def refresh_now(self) -> None:
        """Force an immediate config reload from DB (called after admin edits)."""
        await self._load_config()

    def _resolve_group_limits(
        self, user_id: int, group: _ModelGroupSnapshot
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (effective_group_rpm, effective_group_rpd) for user_id in group."""
        override = self._user_group_overrides.get((user_id, group.group_id))
        g_rpm = override.rpm_limit if (override and override.rpm_limit is not None) else group.rpm_default
        g_rpd = override.rpd_limit if (override and override.rpd_limit is not None) else group.rpd_default
        return g_rpm, g_rpd

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

    async def check_group_limit(
        self, user_id: int, username: str, model_id: str
    ) -> Optional[RateLimitDecision]:
        """Check only model-group limits. Returns a denied decision or None if allowed/no group.

        Called from route handlers after model resolution, where the request-level
        limits have already been enforced by auth middleware.
        """
        global_lock = await self._get_lock()
        async with global_lock:
            group_id = self._model_to_group.get(model_id)
            if group_id is None:
                return None
            group = self._groups.get(group_id)
            if group is None:
                return None
            g_rpm, g_rpd = self._resolve_group_limits(user_id, group)

        if g_rpm is None and g_rpd is None:
            return None

        user_lock = self._get_user_lock(user_id)
        async with user_lock:
            now = time.time()
            current_window = int(now // 60)

            gk = (user_id, group.group_id)
            group_bucket = self._group_minute_buckets.get(gk)
            if group_bucket is None or group_bucket.window != current_window:
                group_bucket = _MinuteBucket(window=current_window, count=0)
                self._group_minute_buckets[gk] = group_bucket

            # Group RPM check
            if g_rpm is not None and group_bucket.count >= g_rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=None, rpm_remaining=None,
                    rpd_limit=None, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="group_rpm",
                    group_id=group.group_id,
                    group_name=group.name,
                    group_rpm_limit=g_rpm,
                    group_rpm_remaining=0,
                    group_rpd_limit=g_rpd,
                )

            # Group RPD check
            if g_rpd is not None:
                group_today_count = await self._get_today_group_count(username, group.model_ids, group.group_id)
                if group_today_count >= g_rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=None, rpm_remaining=None,
                        rpd_limit=None, rpd_remaining=None,
                        retry_after_seconds=_seconds_until_utc_midnight(),
                        limited_by="group_rpd",
                        group_id=group.group_id,
                        group_name=group.name,
                        group_rpm_limit=g_rpm,
                        group_rpd_limit=g_rpd,
                        group_rpd_remaining=0,
                    )

            # Passed — increment group RPM bucket
            group_bucket.count += 1

        return None  # group checks passed

    async def check_and_increment(
        self, user_id: int, username: str, model_id: Optional[str] = None
    ) -> RateLimitDecision:
        global_lock = await self._get_lock()
        async with global_lock:
            override = self._overrides.get(user_id)
            rpm = override.rpm_limit if (override and override.rpm_limit is not None) else self._defaults.rpm_default
            rpd = override.rpd_limit if (override and override.rpd_limit is not None) else self._defaults.rpd_default

            # Resolve group limits (if model belongs to a group)
            group: Optional[_ModelGroupSnapshot] = None
            g_rpm: Optional[int] = None
            g_rpd: Optional[int] = None
            if model_id:
                group_id = self._model_to_group.get(model_id)
                if group_id is not None:
                    group = self._groups.get(group_id)
                    if group:
                        g_rpm, g_rpd = self._resolve_group_limits(user_id, group)

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

            # Maintain group RPM bucket
            group_bucket: Optional[_MinuteBucket] = None
            if group:
                gk = (user_id, group.group_id)
                group_bucket = self._group_minute_buckets.get(gk)
                if group_bucket is None or group_bucket.window != current_window:
                    group_bucket = _MinuteBucket(window=current_window, count=0)
                    self._group_minute_buckets[gk] = group_bucket

            # 1) Request RPM check
            if rpm is not None and bucket.count >= rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=rpm, rpm_remaining=0,
                    rpd_limit=rpd, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="rpm",
                    group_id=group.group_id if group else None,
                    group_name=group.name if group else None,
                    group_rpm_limit=g_rpm,
                    group_rpd_limit=g_rpd,
                )

            # 2) Group RPM check
            if group and group_bucket is not None and g_rpm is not None and group_bucket.count >= g_rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=rpm, rpm_remaining=None,
                    rpd_limit=rpd, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="group_rpm",
                    group_id=group.group_id,
                    group_name=group.name,
                    group_rpm_limit=g_rpm,
                    group_rpm_remaining=0,
                    group_rpd_limit=g_rpd,
                )

            # 3) Request RPD check
            if rpd is not None:
                today_count = await self._get_today_count(username)
                if today_count >= rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=rpm, rpm_remaining=None,
                        rpd_limit=rpd, rpd_remaining=0,
                        retry_after_seconds=_seconds_until_utc_midnight(),
                        limited_by="rpd",
                        group_id=group.group_id if group else None,
                        group_name=group.name if group else None,
                        group_rpm_limit=g_rpm,
                        group_rpd_limit=g_rpd,
                    )

            # 4) Group RPD check
            if group and g_rpd is not None:
                group_today_count = await self._get_today_group_count(username, group.model_ids, group.group_id)
                if group_today_count >= g_rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=rpm, rpm_remaining=None,
                        rpd_limit=rpd, rpd_remaining=None,
                        retry_after_seconds=_seconds_until_utc_midnight(),
                        limited_by="group_rpd",
                        group_id=group.group_id,
                        group_name=group.name,
                        group_rpm_limit=g_rpm,
                        group_rpd_limit=g_rpd,
                        group_rpd_remaining=0,
                    )

            # All checks passed — increment both RPM buckets
            bucket.count += 1
            if group_bucket is not None:
                group_bucket.count += 1
            rpm_remaining = (rpm - bucket.count) if rpm is not None else None
            g_rpm_remaining = (g_rpm - group_bucket.count) if (g_rpm is not None and group_bucket is not None) else None

        rpd_count = await self._get_today_count(username)
        rpd_remaining = max(0, rpd - rpd_count) if rpd is not None else None

        g_rpd_remaining: Optional[int] = None
        if group and g_rpd is not None:
            group_today_count = await self._get_today_group_count(username, group.model_ids, group.group_id)
            g_rpd_remaining = max(0, g_rpd - group_today_count)

        return RateLimitDecision(
            allowed=True,
            rpm_limit=rpm, rpm_remaining=rpm_remaining,
            rpd_limit=rpd, rpd_remaining=rpd_remaining,
            retry_after_seconds=0,
            limited_by=None,
            group_id=group.group_id if group else None,
            group_name=group.name if group else None,
            group_rpm_limit=g_rpm,
            group_rpm_remaining=g_rpm_remaining,
            group_rpd_limit=g_rpd,
            group_rpd_remaining=g_rpd_remaining,
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

    async def _get_today_group_count(
        self, user_identity: str, model_ids: List[str], group_id: int
    ) -> int:
        """Return today's total request count for all models in the group, with TTL cache."""
        now = time.time()
        cache_key = (user_identity, group_id)
        entry = self._group_rpd_cache.get(cache_key)
        if entry and entry.expires_at > now:
            return entry.count

        try:
            from app.request_tracker import request_tracker
            count = await request_tracker.get_today_group_count(user_identity, model_ids)
        except Exception:
            count = 0

        self._group_rpd_cache[cache_key] = _RpdCacheEntry(
            count=count, expires_at=now + _RPD_TTL
        )
        return count


rate_limit_tracker = RateLimitTracker()

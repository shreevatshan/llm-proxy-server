"""Per-user request rate limiting (RPM and RPD).

Single-process only: counters live in-memory. RPM buckets reset on restart
(60s window — acceptable). RPD survives via the RequestUsage table.

If the app is ever scaled to multiple workers, swap _minute_buckets for
a shared Redis counter.

Request pools share one daily quota across several users: a pooled user's RPD limit is
the sum of the members' limits and their count is the sum of the members' consumption.
Only RPD is pooled — every RPM structure here stays strictly per-user.

Pooled RPD is approximate in the same way per-user RPD already is, only wider.
check_and_increment holds a *per-user* lock across [RPM check → RPD check → RPM
increment], so two members of a pool hold different locks and can both pass the RPD
check concurrently. That lock never made RPD exact anyway: the count comes from a 5s TTL
cache over a buffer flushed every 60s, and a request is only recorded at end_request,
after the upstream call returns. The real overshoot window is already "everything in
flight, plus the TTL" for a single user; pooling multiplies that existing bound by pool
size rather than introducing a new class of error. A per-pool lock is deliberately not
added: held inside the per-user lock it would serialise the whole pool's traffic through
one mutex and add a lock-ordering hazard, to buy a guarantee the surrounding design does
not offer. Exact pooled RPD needs a durable shared counter — the same change as going
multi-worker, noted above.
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
# How long a stale (expired) RPD cache entry is retained after expiry so it can
# be served as a last-known value on a transient DB failure, before eviction
# reclaims it. Bounds per-identity memory growth in long-lived processes.
_RPD_CACHE_RETENTION = 300  # seconds


def _user_scope_key(username: str) -> str:
    """RPD cache key for an unpooled user.

    Both kinds of scope key share one keyspace, and a username is an arbitrary
    string (validated for length only), so an unprefixed username could be spelled
    exactly like a pool key — a user registered as "pool:3" would otherwise read and
    write pool 3's cached day count. Prefixing both sides makes that unrepresentable.
    """
    return f"user:{username}"


def _pool_scope_key(pool_id: int) -> str:
    """RPD cache key shared by every member of a pool. See _user_scope_key."""
    return f"pool:{pool_id}"


@dataclass
class RateLimitDecision:
    allowed: bool
    rpm_limit: Optional[int]
    rpm_remaining: Optional[int]
    rpd_limit: Optional[int]
    rpd_remaining: Optional[int]
    retry_after_seconds: int
    limited_by: Optional[str]  # "rpm"|"rpd"|"group_rpm"|"group_rpd"|"instance_group_rpm"|"instance_group_rpd"|None
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
class _InstanceGroupSnapshot:
    group_id: int
    name: str
    rpm_default: Optional[int]
    rpd_default: Optional[int]
    provider_keys: List[str]


@dataclass
class _UserInstanceGroupOverride:
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
    if d.limited_by == "instance_group_rpm":
        return (
            f"Rate limit exceeded for instance group '{d.group_name}': {d.group_rpm_limit} requests per minute. "
            f"Retry after {_human_duration(d.retry_after_seconds)}, or use an instance outside this group. "
            f"View your allowed quota in the console."
        )
    if d.limited_by == "instance_group_rpd":
        return (
            f"Rate limit exceeded for instance group '{d.group_name}': {d.group_rpd_limit} requests per day. "
            f"Retry after {_human_duration(d.retry_after_seconds)}, or use an instance outside this group. "
            f"View your allowed quota in the console."
        )
    if d.limited_by == "group_rpm":
        return (
            f"Rate limit exceeded for model group '{d.group_name}': {d.group_rpm_limit} requests per minute. "
            f"Retry after {_human_duration(d.retry_after_seconds)}, or use a model outside this group. "
            f"View your allowed quota in the console."
        )
    if d.limited_by == "group_rpd":
        return (
            f"Rate limit exceeded for model group '{d.group_name}': {d.group_rpd_limit} requests per day. "
            f"Retry after {_human_duration(d.retry_after_seconds)}, or use a model outside this group. "
            f"View your allowed quota in the console."
        )
    if d.limited_by == "rpm":
        limit = d.rpm_limit
        return (
            f"Rate limit exceeded: {limit} requests per minute. "
            f"Retry after {_human_duration(d.retry_after_seconds)}. "
            f"View your allowed quota in the console."
        )
    limit = d.rpd_limit
    return (
        f"Rate limit exceeded: {limit} requests per day. "
        f"Retry after {_human_duration(d.retry_after_seconds)}. "
        f"View your allowed quota in the console."
    )


def _rl_headers(d: RateLimitDecision) -> dict:
    if d.limited_by in ("group_rpm", "group_rpd", "instance_group_rpm", "instance_group_rpd"):
        limit = d.group_rpm_limit if d.limited_by in ("group_rpm", "instance_group_rpm") else d.group_rpd_limit
    else:
        limit = d.rpm_limit if d.limited_by == "rpm" else d.rpd_limit
    return {
        "Retry-After": str(d.retry_after_seconds),
        "X-RateLimit-Limit-Requests": str(limit) if limit is not None else "unlimited",
        "X-RateLimit-Remaining-Requests": "0",
        "X-RateLimit-Reset-Requests": str(d.retry_after_seconds),
    }


def _seconds_until_local_midnight() -> int:
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
        self._minute_buckets: Dict[int, _MinuteBucket] = {}
        # Group RPM buckets keyed by (user_id, group_id)
        self._group_minute_buckets: Dict[Tuple[int, int], _MinuteBucket] = {}
        self._overrides: Dict[int, _UserOverride] = {}
        self._defaults = _GlobalDefaults(rpm_default=None, rpd_default=None)
        self._rpd_cache: Dict[str, _RpdCacheEntry] = {}            # scope_key → entry
        # Group RPD cache keyed by (scope_key, group_id)
        self._group_rpd_cache: Dict[Tuple[str, int], _RpdCacheEntry] = {}
        self._user_locks: Dict[int, asyncio.Lock] = {}
        self._db_session_factory: Optional[Callable] = None
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        # Model-group state
        self._groups: Dict[int, _ModelGroupSnapshot] = {}        # group_id → snapshot
        self._model_to_group: Dict[str, int] = {}               # model_id → group_id
        self._user_group_overrides: Dict[Tuple[int, int], _UserGroupOverride] = {}  # (user_id, group_id)
        # Instance-group state (keyed on provider_key)
        self._instance_groups: Dict[int, _InstanceGroupSnapshot] = {}    # group_id → snapshot
        self._provider_to_group: Dict[str, int] = {}                     # provider_key → group_id
        self._instance_group_minute_buckets: Dict[Tuple[int, int], _MinuteBucket] = {}  # (user_id, group_id)
        self._instance_group_rpd_cache: Dict[Tuple[str, int], _RpdCacheEntry] = {}      # (scope_key, group_id)
        self._user_instance_group_overrides: Dict[Tuple[int, int], _UserInstanceGroupOverride] = {}  # (user_id, group_id)
        # Request-pool state. Only RPD is pooled; every RPM structure above stays
        # strictly per-user. Usernames are cached alongside ids because RPD counting is
        # keyed by the username string (RequestUsage.user_identity), not by user_id.
        self._user_to_pool: Dict[int, int] = {}                    # user_id → pool_id
        self._pool_members: Dict[int, List[Tuple[int, str]]] = {}  # pool_id → [(user_id, username)]
        self._identity_to_pool: Dict[str, int] = {}                # username → pool_id
        self._carries: Dict[Tuple[int, str, int], int] = {}        # (user_id, scope_kind, scope_id) → carry
        self._carries_date = None                                  # local day the carries were loaded for

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
            from app.auth.models import (
                UserRateLimit, ModelGroup, ModelGroupMember, UserModelGroupRateLimit,
                InstanceGroup, InstanceGroupMember, UserInstanceGroupRateLimit,
                RequestPool, RequestPoolMember, User, UserRpdCarry,
            )
            from app import time_utils
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

                # Load instance groups with members
                result = await db.execute(
                    select(InstanceGroup).options(selectinload(InstanceGroup.members))
                )
                instance_group_rows = result.scalars().all()

                # Load per-user instance-group overrides
                result = await db.execute(select(UserInstanceGroupRateLimit))
                user_instance_group_rows = result.scalars().all()

                # Load pool membership joined to User for the usernames the RPD
                # counters are keyed by.
                result = await db.execute(
                    select(RequestPoolMember.pool_id, RequestPoolMember.user_id, User.username)
                    .join(User, User.id == RequestPoolMember.user_id)
                )
                pool_member_rows = result.all()

                # Carries are day-scoped; only today's can ever apply.
                carry_day = time_utils.local_today()
                result = await db.execute(
                    select(UserRpdCarry).where(UserRpdCarry.usage_date == carry_day)
                )
                carry_rows = result.scalars().all()

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
                self._instance_groups = {
                    g.id: _InstanceGroupSnapshot(
                        group_id=g.id,
                        name=g.name,
                        rpm_default=g.rpm_default,
                        rpd_default=g.rpd_default,
                        provider_keys=[m.provider_key for m in g.members],
                    )
                    for g in instance_group_rows
                }
                self._provider_to_group = {
                    m.provider_key: g.id
                    for g in instance_group_rows
                    for m in g.members
                }
                self._user_instance_group_overrides = {
                    (row.user_id, row.group_id): _UserInstanceGroupOverride(
                        user_id=row.user_id,
                        group_id=row.group_id,
                        rpm_limit=row.rpm_limit,
                        rpd_limit=row.rpd_limit,
                    )
                    for row in user_instance_group_rows
                }
                pool_members: Dict[int, List[Tuple[int, str]]] = {}
                user_to_pool: Dict[int, int] = {}
                identity_to_pool: Dict[str, int] = {}
                for pool_id, uid, uname in pool_member_rows:
                    pool_members.setdefault(pool_id, []).append((uid, uname))
                    user_to_pool[uid] = pool_id
                    identity_to_pool[uname] = pool_id
                self._pool_members = pool_members
                self._user_to_pool = user_to_pool
                self._identity_to_pool = identity_to_pool
                self._carries = {
                    (row.user_id, row.scope_kind, row.scope_id): row.carry
                    for row in carry_rows
                }
                self._carries_date = carry_day
        except Exception as e:
            logger.warning(f"RateLimitTracker: config reload failed: {e}")

    def _evict_stale_buckets(self) -> None:
        now = time.time()
        current_window = int(now // 60)
        stale = [uid for uid, b in self._minute_buckets.items() if b.window < current_window - 1]
        for uid in stale:
            del self._minute_buckets[uid]
        stale_g = [k for k, b in self._group_minute_buckets.items() if b.window < current_window - 1]
        for k in stale_g:
            del self._group_minute_buckets[k]
        stale_ig = [k for k, b in self._instance_group_minute_buckets.items() if b.window < current_window - 1]
        for k in stale_ig:
            del self._instance_group_minute_buckets[k]

        # Sweep RPD cache entries whose last-known value has aged out past the
        # retention window (these otherwise grow one entry per identity/group).
        cutoff = now - _RPD_CACHE_RETENTION
        for cache in (self._rpd_cache, self._group_rpd_cache, self._instance_group_rpd_cache):
            expired = [k for k, e in cache.items() if e.expires_at < cutoff]
            for k in expired:
                del cache[k]

        # Sweep idle per-user locks: no active RPM bucket and not currently held.
        # Skipping locked() locks avoids evicting a lock a request is holding
        # while suspended at an await inside `async with user_lock`.
        idle_locks = [
            uid for uid, lk in self._user_locks.items()
            if uid not in self._minute_buckets and not lk.locked()
        ]
        for uid in idle_locks:
            del self._user_locks[uid]

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
        # The RPD cache is keyed by (scope_key, group_id), where scope_key is "user:{name}"
        # or "pool:{id}" — never str(user_id), so a targeted pop cannot match. Drop every
        # entry for the group instead: over-broad, but correct, and it repopulates on the
        # next request. Without this, lowering a user's group limit stayed invisible for
        # the length of the TTL.
        for key in [k for k in self._group_rpd_cache if k[1] == group_id]:
            self._group_rpd_cache.pop(key, None)

    def invalidate_instance_group(self, group_id: int) -> None:
        """Remove cached instance-group snapshot and all related RPM buckets."""
        snap = self._instance_groups.pop(group_id, None)
        if snap:
            for pk in snap.provider_keys:
                self._provider_to_group.pop(pk, None)
        stale = [k for k in self._instance_group_minute_buckets if k[1] == group_id]
        for k in stale:
            del self._instance_group_minute_buckets[k]
        stale_rpd = [k for k in self._instance_group_rpd_cache if k[1] == group_id]
        for k in stale_rpd:
            del self._instance_group_rpd_cache[k]

    def invalidate_user_instance_group(self, user_id: int, group_id: int) -> None:
        self._user_instance_group_overrides.pop((user_id, group_id), None)
        self._instance_group_minute_buckets.pop((user_id, group_id), None)
        # Same keying mismatch as invalidate_user_group — see the note there.
        for key in [k for k in self._instance_group_rpd_cache if k[1] == group_id]:
            self._instance_group_rpd_cache.pop(key, None)

    async def refresh_now(self) -> None:
        """Force an immediate config reload from DB (called after admin edits)."""
        await self._load_config()

    def model_belongs_to_group(self, model_id: str) -> bool:
        """True if the model is mapped to a configured model group."""
        return bool(model_id) and model_id in self._model_to_group

    def instance_belongs_to_group(self, provider_key: str) -> bool:
        """True if the provider instance is mapped to a configured instance group."""
        return bool(provider_key) and provider_key in self._provider_to_group

    def grouped_keys(self) -> Tuple[set, set]:
        """Return (grouped_model_ids, grouped_provider_keys) currently configured.

        A request is excluded from the overall (ungrouped) quota when its model is
        in a model group OR its instance is in an instance group — the same predicate
        the auth middleware uses to skip the overall gate.
        """
        return set(self._model_to_group.keys()), set(self._provider_to_group.keys())

    def _resolve_group_limits(
        self, user_id: int, group: _ModelGroupSnapshot
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (effective_group_rpm, effective_group_rpd) for user_id in group."""
        override = self._user_group_overrides.get((user_id, group.group_id))
        g_rpm = override.rpm_limit if (override and override.rpm_limit is not None) else group.rpm_default
        g_rpd = override.rpd_limit if (override and override.rpd_limit is not None) else group.rpd_default
        return g_rpm, g_rpd

    def _resolve_instance_group_limits(
        self, user_id: int, group: _InstanceGroupSnapshot
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (effective_group_rpm, effective_group_rpd) for user_id in an instance group."""
        override = self._user_instance_group_overrides.get((user_id, group.group_id))
        g_rpm = override.rpm_limit if (override and override.rpm_limit is not None) else group.rpm_default
        g_rpd = override.rpd_limit if (override and override.rpd_limit is not None) else group.rpd_default
        return g_rpm, g_rpd

    # -- request pooling ---------------------------------------------------
    #
    # A pool shares one daily quota: its limit is the sum of its members' limits and
    # its count is the sum of their consumption. Only RPD is pooled — every RPM path
    # above keeps reading the caller's own user_id.

    def _rpd_scope(self, user_id: int, username: str) -> Tuple[str, List[str], List[int], Optional[int]]:
        """Return (cache_scope_key, identities_to_count, member_user_ids, pool_id).

        Unpooled: ("user:alice", ["alice"], [7], None)
        Pooled:   ("pool:3", ["alice","bob","carol"], [7,8,9], 3)

        The scope key is what the RPD caches are keyed by, so all members of a pool
        share one cache entry — one DB read per pool per TTL, not one per member.
        Both forms are prefixed so the two keyspaces cannot collide; see
        _user_scope_key.
        """
        pool_id = self._user_to_pool.get(user_id)
        members = self._pool_members.get(pool_id) if pool_id is not None else None
        if not members:
            return _user_scope_key(username), [username], [user_id], None
        return (_pool_scope_key(pool_id), [u for _, u in members], [i for i, _ in members], pool_id)

    def _own_rpd_limit(self, user_id: int) -> Optional[int]:
        """One user's own effective overall RPD — their override, else the global default."""
        override = self._overrides.get(user_id)
        if override and override.rpd_limit is not None:
            return override.rpd_limit
        return self._defaults.rpd_default

    def _pooled_rpd_limit(self, member_ids: List[int], per_member) -> Optional[int]:
        """Sum the per-member effective RPD. None from any member ⇒ unlimited pool.

        `per_member` is a callable resolving one member's effective limit for the tier
        being checked; it is evaluated under the global lock, where the snapshot dicts
        this reads are already held.
        """
        total = 0
        for uid in member_ids:
            value = per_member(uid)
            if value is None:
                return None
            total += value
        return total

    def _carry_sum(self, member_ids: List[int], scope_kind: str, scope_id: int) -> int:
        """Total day-scoped RPD adjustment for these members on one scope.

        Carries are written by pool settlement so a member is charged only for what the
        pool spent while they were in it. A carry from a previous local day must never
        apply, so a stale snapshot reads as zero rather than yesterday's numbers.
        """
        if not self._carries:
            return 0
        from app import time_utils
        if self._carries_date != time_utils.local_today():
            return 0
        return sum(self._carries.get((uid, scope_kind, scope_id), 0) for uid in member_ids)

    def invalidate_pool(self, pool_id: int) -> None:
        """Drop every cached RPD count for a pool, across all three tiers.

        Called on every composition change: the membership that produced the cached
        count no longer exists, so serving it for the rest of the TTL would enforce
        against a pool that is already gone.
        """
        scope_key = _pool_scope_key(pool_id)
        self._rpd_cache.pop(scope_key, None)
        for cache in (self._group_rpd_cache, self._instance_group_rpd_cache):
            for key in [k for k in cache if k[0] == scope_key]:
                cache.pop(key, None)

    async def get_user_status(self, user_id: int, username: str) -> "UserStatus":
        """Read-only — returns current usage without incrementing."""
        lock = await self._get_lock()
        async with lock:
            override = self._overrides.get(user_id)
            rpm = override.rpm_limit if (override and override.rpm_limit is not None) else self._defaults.rpm_default

            scope_key, identities, member_ids, pool_id = self._rpd_scope(user_id, username)
            if pool_id is None:
                rpd = override.rpd_limit if (override and override.rpd_limit is not None) else self._defaults.rpd_default
            else:
                rpd = self._pooled_rpd_limit(member_ids, self._own_rpd_limit)

            now = time.time()
            current_window = int(now // 60)
            bucket = self._minute_buckets.get(user_id)
            rpm_count = (bucket.count if bucket and bucket.window == current_window else 0)

        rpd_count = await self._get_today_count(scope_key, identities, member_ids)

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
            g_rpm, _own_rpd = self._resolve_group_limits(user_id, group)

            # RPD may be pooled across the members' group limits; RPM stays per-user.
            scope_key, identities, member_ids, pool_id = self._rpd_scope(user_id, username)
            if pool_id is None:
                g_rpd = _own_rpd
            else:
                g_rpd = self._pooled_rpd_limit(
                    member_ids, lambda uid: self._resolve_group_limits(uid, group)[1]
                )

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
                group_today_count = await self._get_today_group_count(
                    scope_key, identities, member_ids, group.model_ids, group.group_id
                )
                if group_today_count >= g_rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=None, rpm_remaining=None,
                        rpd_limit=None, rpd_remaining=None,
                        retry_after_seconds=_seconds_until_local_midnight(),
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

    async def check_instance_group_limit(
        self, user_id: int, username: str, provider_key: str
    ) -> Optional[RateLimitDecision]:
        """Check only instance-group limits. Returns a denied decision or None if allowed/no group.

        Called from route handlers after the instance (provider_key) is resolved.
        Takes precedence over model-group limits.
        """
        global_lock = await self._get_lock()
        async with global_lock:
            group_id = self._provider_to_group.get(provider_key)
            if group_id is None:
                return None
            group = self._instance_groups.get(group_id)
            if group is None:
                return None
            g_rpm, _own_rpd = self._resolve_instance_group_limits(user_id, group)

            # RPD may be pooled across the members' group limits; RPM stays per-user.
            scope_key, identities, member_ids, pool_id = self._rpd_scope(user_id, username)
            if pool_id is None:
                g_rpd = _own_rpd
            else:
                g_rpd = self._pooled_rpd_limit(
                    member_ids, lambda uid: self._resolve_instance_group_limits(uid, group)[1]
                )

        if g_rpm is None and g_rpd is None:
            return None

        user_lock = self._get_user_lock(user_id)
        async with user_lock:
            now = time.time()
            current_window = int(now // 60)

            gk = (user_id, group.group_id)
            group_bucket = self._instance_group_minute_buckets.get(gk)
            if group_bucket is None or group_bucket.window != current_window:
                group_bucket = _MinuteBucket(window=current_window, count=0)
                self._instance_group_minute_buckets[gk] = group_bucket

            # Instance-group RPM check
            if g_rpm is not None and group_bucket.count >= g_rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=None, rpm_remaining=None,
                    rpd_limit=None, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="instance_group_rpm",
                    group_id=group.group_id,
                    group_name=group.name,
                    group_rpm_limit=g_rpm,
                    group_rpm_remaining=0,
                    group_rpd_limit=g_rpd,
                )

            # Instance-group RPD check
            if g_rpd is not None:
                group_today_count = await self._get_today_instance_group_count(
                    scope_key, identities, member_ids, group.provider_keys, group.group_id
                )
                if group_today_count >= g_rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=None, rpm_remaining=None,
                        rpd_limit=None, rpd_remaining=None,
                        retry_after_seconds=_seconds_until_local_midnight(),
                        limited_by="instance_group_rpd",
                        group_id=group.group_id,
                        group_name=group.name,
                        group_rpm_limit=g_rpm,
                        group_rpd_limit=g_rpd,
                        group_rpd_remaining=0,
                    )

            # Passed — increment instance-group RPM bucket
            group_bucket.count += 1

        return None  # instance-group checks passed

    def invalidate_identity(self, identity: str) -> None:
        """Drop every cached RPD count for a user_identity.

        Called on both sides of a username change. The RPD caches are keyed by the
        username string, so without this the new name could serve a stale count for
        the length of the TTL right after a rename.

        A pooled user's counts live under the pool's scope key rather than their own, so
        the pool's entries go too — otherwise a rename leaves the pool serving a count
        computed from the old username for the length of the TTL.
        """
        scope_key = _user_scope_key(identity)
        self._rpd_cache.pop(scope_key, None)
        for cache in (self._group_rpd_cache, self._instance_group_rpd_cache):
            for key in [k for k in cache if k[0] == scope_key]:
                cache.pop(key, None)
        pool_id = self._identity_to_pool.get(identity)
        if pool_id is not None:
            self.invalidate_pool(pool_id)

    def invalidate_all_rpd(self) -> None:
        """Drop every cached RPD count, for all identities.

        Used when usage rows are deleted for a model: that touches the day counts of
        every user who called it, and the caches are keyed by identity rather than by
        model, so there is no narrower invalidation available.
        """
        self._rpd_cache.clear()
        self._group_rpd_cache.clear()
        self._instance_group_rpd_cache.clear()

    async def check_and_increment(
        self, user_id: int, username: str
    ) -> RateLimitDecision:
        """Enforce the per-user overall (ungrouped) RPM/RPD limits and increment
        the RPM bucket.

        Model-group and instance-group limits are enforced separately by
        check_group_limit / check_instance_group_limit from the route handlers.
        The sole caller (auth middleware) passes no model, so this method
        deliberately handles only the overall quota — the previous per-model
        group branches here were unreachable dead code and have been removed.
        """
        global_lock = await self._get_lock()
        async with global_lock:
            override = self._overrides.get(user_id)
            rpm = override.rpm_limit if (override and override.rpm_limit is not None) else self._defaults.rpm_default

            # RPD may be pooled; RPM above is always this user's own.
            scope_key, identities, member_ids, pool_id = self._rpd_scope(user_id, username)
            if pool_id is None:
                rpd = override.rpd_limit if (override and override.rpd_limit is not None) else self._defaults.rpd_default
            else:
                rpd = self._pooled_rpd_limit(member_ids, self._own_rpd_limit)

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

            # 1) Request RPM check
            if rpm is not None and bucket.count >= rpm:
                retry_after = max(1, 60 - int(now - current_window * 60))
                return RateLimitDecision(
                    allowed=False,
                    rpm_limit=rpm, rpm_remaining=0,
                    rpd_limit=rpd, rpd_remaining=None,
                    retry_after_seconds=retry_after,
                    limited_by="rpm",
                )

            # 2) Request RPD check
            if rpd is not None:
                today_count = await self._get_today_count(scope_key, identities, member_ids)
                if today_count >= rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=rpm, rpm_remaining=None,
                        rpd_limit=rpd, rpd_remaining=0,
                        retry_after_seconds=_seconds_until_local_midnight(),
                        limited_by="rpd",
                    )

            # All checks passed — increment the RPM bucket
            bucket.count += 1
            rpm_remaining = (rpm - bucket.count) if rpm is not None else None

        rpd_count = await self._get_today_count(scope_key, identities, member_ids)
        rpd_remaining = max(0, rpd - rpd_count) if rpd is not None else None

        return RateLimitDecision(
            allowed=True,
            rpm_limit=rpm, rpm_remaining=rpm_remaining,
            rpd_limit=rpd, rpd_remaining=rpd_remaining,
            retry_after_seconds=0,
            limited_by=None,
        )

    async def get_group_rpd_count(
        self, user_id: int, user_identity: str, model_ids: List[str], group_id: int
    ) -> int:
        """Read-only — today's request count for a model group (TTL-cached, no increment).

        Pooled callers get the whole pool's count, matching what enforcement sees.
        """
        scope_key, identities, member_ids, _ = self._rpd_scope(user_id, user_identity)
        return await self._get_today_group_count(
            scope_key, identities, member_ids, model_ids, group_id
        )

    async def get_instance_group_rpd_count(
        self, user_id: int, user_identity: str, provider_keys: List[str], group_id: int
    ) -> int:
        """Read-only — today's request count for an instance group (TTL-cached, no increment).

        Pooled callers get the whole pool's count, matching what enforcement sees.
        """
        scope_key, identities, member_ids, _ = self._rpd_scope(user_id, user_identity)
        return await self._get_today_instance_group_count(
            scope_key, identities, member_ids, provider_keys, group_id
        )

    def pooled_rpd_limits(self, user_id: int, username: str) -> Tuple[Optional[int], int, Optional[int]]:
        """Return (pool_id, member_count, pooled_overall_rpd) for the quotas endpoint.

        pool_id is None when the user is not pooled, in which case the caller should
        keep using their own limit.
        """
        pool_id = self._user_to_pool.get(user_id)
        members = self._pool_members.get(pool_id) if pool_id is not None else None
        if not members:
            return None, 1, None
        member_ids = [uid for uid, _ in members]
        return pool_id, len(members), self._pooled_rpd_limit(member_ids, self._own_rpd_limit)

    def pooled_group_rpd_limit(self, user_id: int, group_id: int, instance: bool = False) -> Optional[int]:
        """Pooled effective RPD for one group tier, or None when unlimited/unpooled.

        Returns None both for "no pool" and for "unlimited pool"; callers that need to
        tell them apart check pooled_rpd_limits()[0] first.
        """
        pool_id = self._user_to_pool.get(user_id)
        members = self._pool_members.get(pool_id) if pool_id is not None else None
        if not members:
            return None
        member_ids = [uid for uid, _ in members]
        if instance:
            group = self._instance_groups.get(group_id)
            if group is None:
                return None
            return self._pooled_rpd_limit(
                member_ids, lambda uid: self._resolve_instance_group_limits(uid, group)[1]
            )
        group = self._groups.get(group_id)
        if group is None:
            return None
        return self._pooled_rpd_limit(
            member_ids, lambda uid: self._resolve_group_limits(uid, group)[1]
        )

    # -- accessors for settlement (app/auth/pools.py) -----------------------
    #
    # Settlement must agree with enforcement about every member's limit and about which
    # scope a model's rows fall under, so it reads both from this same snapshot rather
    # than re-deriving them from the DB.

    def settlement_scopes(self) -> List[Tuple[str, int]]:
        """Every scope a pool can be settled on: ('overall', 0) plus each group."""
        scopes: List[Tuple[str, int]] = [("overall", 0)]
        scopes.extend(("model_group", gid) for gid in self._groups)
        scopes.extend(("instance_group", gid) for gid in self._instance_groups)
        return scopes

    def scope_for_model(self, model_id: str) -> Optional[Tuple[str, int]]:
        """Which group scope a model's usage rows count against, or None if ungrouped.

        Instance groups take precedence over model groups, matching the enforcement
        order in check_instance_group_limit / check_group_limit. None means the model is
        ungrouped and its rows count against the overall quota -- grouped rows do not,
        because get_today_count excludes them from the overall count.
        """
        if not model_id:
            return None
        provider_key = model_id.split('/', 1)[0] if '/' in model_id else model_id
        gid = self._provider_to_group.get(provider_key)
        if gid is not None:
            return ("instance_group", gid)
        gid = self._model_to_group.get(model_id)
        if gid is not None:
            return ("model_group", gid)
        return None

    def member_limit_for_scope(
        self, user_id: int, scope_kind: str, scope_id: int
    ) -> Optional[int]:
        """One member's own effective RPD on a scope. None means unlimited.

        A group the user is not a member of still resolves through the group's default,
        exactly as enforcement does -- the group's membership is over models/instances,
        not users.
        """
        if scope_kind == "overall":
            return self._own_rpd_limit(user_id)
        if scope_kind == "model_group":
            group = self._groups.get(scope_id)
            return None if group is None else self._resolve_group_limits(user_id, group)[1]
        if scope_kind == "instance_group":
            group = self._instance_groups.get(scope_id)
            return None if group is None else self._resolve_instance_group_limits(user_id, group)[1]
        return None

    def carry_for(self, user_id: int, scope_kind: str, scope_id: int = 0) -> int:
        """One user's own day-scoped carry on a scope — for explaining a count in the UI."""
        return self._carry_sum([user_id], scope_kind, scope_id)

    async def _get_today_count(
        self, scope_key: str, identities: List[str], member_ids: List[int]
    ) -> int:
        """Return today's effective total request count with a 5-second TTL cache.

            effective = max(0, rows for identities + carries for members)

        For an unpooled user the scope key is "user:{their name}" and `identities` is
        just them, so this is the pre-pool behaviour exactly. For a pooled user the key
        is "pool:{id}" and every member's rows are counted in one query.

        The carry sum is folded in before caching, so the cache stores the effective
        value; carries only change on settlement, which invalidates the entry. The
        max(0, …) matters because deleting usage rows can leave a negative carry behind.

        On a DB/read failure we log and serve the last-known cached count (even
        if expired) instead of caching a fabricated 0 — caching 0 would disable
        the daily limit for the whole TTL on every transient error (fail-open).
        """
        now = time.time()
        entry = self._rpd_cache.get(scope_key)
        if entry and entry.expires_at > now:
            return entry.count

        try:
            from app.request_tracker import request_tracker
            rows = await request_tracker.get_today_count(identities)
        except Exception:
            logger.error(
                "RPD count read failed for %s; serving last-known count",
                scope_key, exc_info=True,
            )
            # Serve the stale cached value if present; otherwise 0 but do NOT
            # cache it so the next request retries the DB immediately.
            return entry.count if entry else 0

        count = max(0, rows + self._carry_sum(member_ids, "overall", 0))
        self._rpd_cache[scope_key] = _RpdCacheEntry(
            count=count, expires_at=now + _RPD_TTL
        )
        return count

    async def _get_today_group_count(
        self, scope_key: str, identities: List[str], member_ids: List[int],
        model_ids: List[str], group_id: int,
    ) -> int:
        """Return today's effective request count for all models in the group, TTL-cached."""
        now = time.time()
        cache_key = (scope_key, group_id)
        entry = self._group_rpd_cache.get(cache_key)
        if entry and entry.expires_at > now:
            return entry.count

        try:
            from app.request_tracker import request_tracker
            rows = await request_tracker.get_today_group_count(identities, model_ids)
        except Exception:
            logger.error(
                "Group RPD count read failed for %s group %s; serving last-known count",
                scope_key, group_id, exc_info=True,
            )
            return entry.count if entry else 0

        count = max(0, rows + self._carry_sum(member_ids, "model_group", group_id))
        self._group_rpd_cache[cache_key] = _RpdCacheEntry(
            count=count, expires_at=now + _RPD_TTL
        )
        return count

    async def _get_today_instance_group_count(
        self, scope_key: str, identities: List[str], member_ids: List[int],
        provider_keys: List[str], group_id: int,
    ) -> int:
        """Return today's effective request count across all instances in the group, TTL-cached."""
        now = time.time()
        cache_key = (scope_key, group_id)
        entry = self._instance_group_rpd_cache.get(cache_key)
        if entry and entry.expires_at > now:
            return entry.count

        try:
            from app.request_tracker import request_tracker
            rows = await request_tracker.get_today_instance_group_count(identities, provider_keys)
        except Exception:
            logger.error(
                "Instance-group RPD count read failed for %s group %s; serving last-known count",
                scope_key, group_id, exc_info=True,
            )
            return entry.count if entry else 0

        count = max(0, rows + self._carry_sum(member_ids, "instance_group", group_id))
        self._instance_group_rpd_cache[cache_key] = _RpdCacheEntry(
            count=count, expires_at=now + _RPD_TTL
        )
        return count


rate_limit_tracker = RateLimitTracker()

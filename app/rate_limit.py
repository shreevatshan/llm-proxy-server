"""Per-user request rate limiting (RPM and RPD) and concurrency (requests in flight).

Single-process only: counters live in-memory. RPM buckets reset on restart
(60s window — acceptable). RPD survives via the RequestUsage table. In-flight counts
start at zero, which is exact: a restart ends every request.

If the app is ever scaled to multiple workers, swap _minute_buckets and _inflight for
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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple
from app import time_utils
from app.concurrency import TaskReentrantLock, lock_in_use
from app.profiles import DEFAULT_GROUP_ID, profile_registry

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 30  # seconds — how often to reload DB config
_RPD_TTL = 5           # seconds — local cache TTL for today-count DB reads
# How long a stale (expired) RPD cache entry is retained after expiry so it can
# be served as a last-known value on a transient DB failure, before eviction
# reclaims it. Bounds per-identity memory growth in long-lived processes.
_RPD_CACHE_RETENTION = 300  # seconds


def provider_key_of(model_id: Optional[str]) -> Optional[str]:
    """The instance (provider_key) a model id belongs to.

    Model ids are '{provider_key}/{model_name}' (e.g. 'azure:primary/gpt-4'); an id
    with no '/' is treated as its own provider_key. The one parser every group lookup,
    usage count and settlement shares, so they cannot disagree about an instance.
    """
    if not model_id:
        return None
    return model_id.split("/", 1)[0]


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
    # "rpm"|"rpd" (Default group)|"group_rpm"|"group_rpd"|"concurrency"|None
    limited_by: Optional[str]
    # Group context (set when limited_by is group_rpm or group_rpd, or for headers)
    group_id: Optional[int] = None
    group_name: Optional[str] = None
    group_rpm_limit: Optional[int] = None
    group_rpm_remaining: Optional[int] = None
    group_rpd_limit: Optional[int] = None
    group_rpd_remaining: Optional[int] = None
    # True when the daily quota that denied the request is a shared pool quota.
    # Only set on RPD denials, where it suppresses the "join a pool" hint that
    # would otherwise be advice the user has already taken.
    pooled: bool = False
    # Set when the denial is this member's share cap inside a pool rather than the
    # pool's own total: the cap that was hit and the percent it came from.
    pool_share_cap: Optional[int] = None
    pool_share_pct: Optional[int] = None
    # Set when limited_by is "concurrency": the profile's cap on requests in flight.
    concurrency_limit: Optional[int] = None


@dataclass
class _MinuteBucket:
    window: int  # int(time.time() // 60)
    count: int


@dataclass
class _ModelGroupSnapshot:
    group_id: int
    name: str
    model_ids: List[str]                                     # explicit model members
    provider_keys: List[str] = field(default_factory=list)  # instance members (live)
    enabled: bool = True                                     # off = its models are not served


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


def _pooling_hint(d: RateLimitDecision) -> str:
    """Suffix suggesting request pooling, for RPD denials only.

    A daily quota is the one limit pooling can raise: members share the sum of
    their RPD limits. Omitted when the quota that denied the request is already
    a pool's, so a pooled user is never told to do what they have done.
    """
    if d.pooled:
        return ""
    return (
        " If you need a larger daily quota, try request pooling — join or create a pool "
        "from the Request Pool tab in the console to share a combined daily quota with "
        "other users."
    )


def _share_message(d: RateLimitDecision, scope: str, pool_limit: Optional[int]) -> str:
    """Denial text for a member who used their share of the pool rather than the pool."""
    return (
        f"Rate limit exceeded: you have used your share of your pool's daily quota{scope} — "
        f"{d.pool_share_cap} of the pool's {pool_limit} requests per day "
        f"(members may use up to {d.pool_share_pct}% each, and never less than their own limit). "
        f"Retry after {_human_duration(d.retry_after_seconds)}. "
        f"View your allowed quota in the console."
    )


def _limit_message(d: RateLimitDecision) -> str:
    if d.limited_by == "concurrency":
        n = d.concurrency_limit
        return (
            f"Concurrency limit exceeded: at most {n} request{'s' if n != 1 else ''} "
            f"may be in progress at once. Retry when an earlier request finishes. "
            f"View your allowed quota in the console."
        )
    if d.pool_share_cap is not None:
        if d.limited_by == "group_rpd":
            return _share_message(d, f" for model group '{d.group_name}'", d.group_rpd_limit)
        return _share_message(d, "", d.rpd_limit)
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
            f"View your allowed quota in the console." + _pooling_hint(d)
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
        f"View your allowed quota in the console." + _pooling_hint(d)
    )


def _rl_headers(d: RateLimitDecision) -> dict:
    if d.limited_by == "concurrency":
        # Not the -Requests headers: those describe the per-minute window, and this
        # denial says nothing about it.
        return {
            "Retry-After": str(d.retry_after_seconds),
            "X-RateLimit-Limit-Concurrent-Requests": str(d.concurrency_limit),
            "X-RateLimit-Remaining-Concurrent-Requests": "0",
        }
    if d.pool_share_cap is not None:
        # The member's own ceiling is the limit they actually ran into.
        limit = d.pool_share_cap
    elif d.limited_by in ("group_rpm", "group_rpd"):
        limit = d.group_rpm_limit if d.limited_by == "group_rpm" else d.group_rpd_limit
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
        self._rpd_cache: Dict[str, _RpdCacheEntry] = {}            # scope_key → entry
        # Group RPD cache keyed by (scope_key, group_id)
        self._group_rpd_cache: Dict[Tuple[str, int], _RpdCacheEntry] = {}
        # Single-flight for RPD cache refills: (cache kind, cache key) → [lock, holders].
        # Concurrent misses on one key share a single DB read instead of each taking a
        # pooled connection. Entries are dropped once nobody holds or awaits them.
        self._rpd_inflight: Dict[Tuple[str, Any], list] = {}
        self._user_locks: Dict[int, asyncio.Lock] = {}
        # One lock per pool, serialising composition changes and settlement. See
        # pool_lock() for the ordering rule that makes it safe.
        self._pool_locks: Dict[int, TaskReentrantLock] = {}
        self._db_session_factory: Optional[Callable] = None
        # Serialises _load_config: a periodic reload whose SELECTs predate an admin's
        # commit must not install its stale snapshot after that admin's refresh_now.
        self._reload_lock: Optional[asyncio.Lock] = None
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        # Model-group state. A group claims models explicitly and/or whole instances;
        # see group_for() for how a model id resolves.
        self._groups: Dict[int, _ModelGroupSnapshot] = {}        # group_id → snapshot
        self._model_to_group: Dict[str, int] = {}               # explicit model_id → group_id
        self._provider_to_group: Dict[str, int] = {}            # instance provider_key → group_id
        # Request-pool state. Only RPD is pooled; every RPM structure above stays
        # strictly per-user. RPD counting itself goes through the member user ids
        # (RequestUsage.user_id); usernames are cached alongside them so
        # invalidate_identity() can find the pool a name belongs to.
        self._user_to_pool: Dict[int, int] = {}                    # user_id → pool_id
        self._pool_members: Dict[int, List[Tuple[int, str]]] = {}  # pool_id → [(user_id, username)]
        self._identity_to_pool: Dict[str, int] = {}                # username → pool_id
        # Pooled members whose account is deactivated. Kept beside _pool_members rather
        # than folded into it: they are still members for counting purposes (the rows
        # they sent before being disabled are real) but contribute no limit, and every
        # existing (uid, uname) unpack site stays untouched.
        self._inactive_members: Set[int] = set()
        self._carries: Dict[Tuple[int, str, int], int] = {}        # (user_id, scope_kind, scope_id) → carry
        self._carries_date = None                                  # local day the carries were loaded for
        # Per-member share cap (percent of the pool's limit on any scope). The global
        # default and the per-pool overrides are kept apart so a pool missing from the
        # snapshot still falls back to the default. None = uncapped.
        self._pool_share_default: Optional[int] = None
        self._pool_share_override: Dict[int, int] = {}             # pool_id → pct
        # Requests each user has in progress (user_id → count), whether or not their
        # profile caps it. Taken and released through app.concurrency_slots, which ties
        # each slot to one ASGI call. Keys are dropped at zero, so this never grows
        # past the users with something in flight.
        self._inflight: Dict[int, int] = {}

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

    def pool_lock(self, pool_id: int) -> TaskReentrantLock:
        """Serialise composition changes and settlement for one pool.

        settle_pool reads the pool's usage, apportions it and writes a ledger row per
        member. Two of those interleaving -- a join racing a leave, or two accepts --
        apportion against a membership that is already stale by the time they write.
        The conflict-safe upserts keep the writes from colliding; this keeps them from
        describing different pools.

        ORDERING: acquire this **before** request_tracker.pause_flush(), never after.
        settle_pool takes pause_flush internally, so a caller holding the flush mutex
        while waiting here would deadlock against a settle holding this and waiting for
        the flush. Every call site below follows that order.

        Single-process, like the rest of this module -- see the note at the top of the
        file. Across workers the database constraints remain the real guarantee.
        """
        if pool_id not in self._pool_locks:
            self._pool_locks[pool_id] = TaskReentrantLock()
        return self._pool_locks[pool_id]

    async def start(self) -> None:
        self._lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
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
        if self._reload_lock is None:
            self._reload_lock = asyncio.Lock()
        async with self._reload_lock:
            await self._load_config_locked()

    async def _load_config_locked(self) -> None:
        try:
            from app.auth.models import (
                ModelGroup, RequestPoolMember, User, UserRpdCarry,
            )
            from app.profiles import profile_registry
            from app import time_utils
            from sqlalchemy.future import select
            from sqlalchemy.orm import selectinload

            async with self._db_session_factory() as db:
                # Every limit value comes from the user's profile; the registry is
                # shared with model access, so reloading it here keeps both in step.
                await profile_registry.load(db)

                # Load model groups with both kinds of member
                result = await db.execute(
                    select(ModelGroup).options(
                        selectinload(ModelGroup.members),
                        selectinload(ModelGroup.instance_members),
                    )
                )
                group_rows = result.scalars().all()

                # Load pool membership joined to User for the usernames the RPD
                # counters are keyed by.
                result = await db.execute(
                    select(
                        RequestPoolMember.pool_id,
                        RequestPoolMember.user_id,
                        User.username,
                        User.is_active,
                    )
                    .join(User, User.id == RequestPoolMember.user_id)
                )
                pool_member_rows = result.all()

                # Carries are day-scoped; only today's can ever apply.
                carry_day = time_utils.local_today()
                result = await db.execute(
                    select(UserRpdCarry).where(UserRpdCarry.usage_date == carry_day)
                )
                carry_rows = result.scalars().all()

                share_default, share_override = await self._load_pool_share(db)

            lock = await self._get_lock()
            async with lock:
                self._groups = {
                    g.id: _ModelGroupSnapshot(
                        group_id=g.id,
                        name=g.name,
                        model_ids=[m.model_id for m in g.members],
                        provider_keys=[m.provider_key for m in g.instance_members],
                        enabled=bool(g.enabled),
                    )
                    for g in group_rows
                }
                self._model_to_group = {
                    m.model_id: g.id
                    for g in group_rows
                    for m in g.members
                }
                self._provider_to_group = {
                    m.provider_key: g.id
                    for g in group_rows
                    for m in g.instance_members
                }
                pool_members: Dict[int, List[Tuple[int, str]]] = {}
                user_to_pool: Dict[int, int] = {}
                identity_to_pool: Dict[str, int] = {}
                inactive_members: Set[int] = set()
                for pool_id, uid, uname, is_active in pool_member_rows:
                    pool_members.setdefault(pool_id, []).append((uid, uname))
                    user_to_pool[uid] = pool_id
                    identity_to_pool[uname] = pool_id
                    if not is_active:
                        inactive_members.add(uid)
                self._pool_members = pool_members
                self._user_to_pool = user_to_pool
                self._identity_to_pool = identity_to_pool
                self._inactive_members = inactive_members
                self._carries = {
                    (row.user_id, row.scope_kind, row.scope_id): row.carry
                    for row in carry_rows
                }
                self._carries_date = carry_day
                self._pool_share_default = share_default
                self._pool_share_override = share_override
        except Exception as e:
            logger.warning(f"RateLimitTracker: config reload failed: {e}")

    async def _load_pool_share(self, db) -> Tuple[Optional[int], Dict[int, int]]:
        """Read the share-cap policy: (global pct, {pool_id: override pct}).

        Isolated from the rest of the reload: if these reads fail (say the override
        columns were never migrated) the cap falls back to "none" instead of taking
        groups, membership and carries down with it.
        """
        try:
            from app.auth.models import PoolSettings, RequestPool
            from sqlalchemy.future import select

            result = await db.execute(
                select(PoolSettings.max_member_share_pct).where(PoolSettings.id == 1)
            )
            default = result.scalar_one_or_none()
            result = await db.execute(
                select(RequestPool.id, RequestPool.max_member_share_pct)
                .where(RequestPool.max_member_share_pct.is_not(None))
            )
            return default, {pid: pct for pid, pct in result.all()}
        except Exception as e:
            logger.warning(f"RateLimitTracker: pool share settings read failed: {e}")
            return None, {}

    def _evict_stale_buckets(self) -> None:
        now = time.time()
        current_window = int(now // 60)
        stale = [uid for uid, b in self._minute_buckets.items() if b.window < current_window - 1]
        for uid in stale:
            del self._minute_buckets[uid]
        stale_g = [k for k, b in self._group_minute_buckets.items() if b.window < current_window - 1]
        for k in stale_g:
            del self._group_minute_buckets[k]

        # Sweep RPD cache entries whose last-known value has aged out past the
        # retention window (these otherwise grow one entry per identity/group).
        cutoff = now - _RPD_CACHE_RETENTION
        for cache in (self._rpd_cache, self._group_rpd_cache):
            expired = [k for k, e in cache.items() if e.expires_at < cutoff]
            for k in expired:
                del cache[k]

        # Sweep idle per-user locks: no active RPM bucket, and nobody holding or waiting
        # for the lock. Evicting one in use would let the next caller build a new lock and
        # enter the critical section alongside whoever holds or is about to take the old one.
        idle_locks = [
            uid for uid, lk in self._user_locks.items()
            if uid not in self._minute_buckets and not lock_in_use(lk)
        ]
        for uid in idle_locks:
            del self._user_locks[uid]

        # Pool locks have no per-request bucket to key idleness off, so "not in use" is
        # the whole test. It must count waiters, not just the holder: a released lock
        # reads unlocked before its next waiter has taken it (see lock_in_use).
        for pid in [p for p, lk in self._pool_locks.items() if not lk.in_use()]:
            del self._pool_locks[pid]

    def invalidate_group(self, group_id: int) -> None:
        """Remove cached group snapshot and all related RPM buckets."""
        snap = self._groups.pop(group_id, None)
        if snap:
            for mid in snap.model_ids:
                self._model_to_group.pop(mid, None)
            for pk in snap.provider_keys:
                self._provider_to_group.pop(pk, None)
        # Remove group RPM buckets for this group
        stale = [k for k in self._group_minute_buckets if k[1] == group_id]
        for k in stale:
            del self._group_minute_buckets[k]
        # Remove group RPD cache entries for this group
        stale_rpd = [k for k in self._group_rpd_cache if k[1] == group_id]
        for k in stale_rpd:
            del self._group_rpd_cache[k]

    async def refresh_now(self) -> None:
        """Force an immediate config reload from DB (called after admin edits).

        Also drops the cached RPD counts. Every caller is on a path where something the
        counts depend on just changed -- a limit edit, a composition change, a purge --
        so serving a count cached before that change for the rest of _RPD_TTL would
        enforce against numbers that are already gone. In particular _load_config is what
        re-reads UserRpdCarry into the snapshot, and a carry rewritten by settlement is
        invisible until the cached count expires.

        The periodic reload in _refresh_loop calls _load_config directly and is
        deliberately left alone: it is not reacting to a change, so it keeps its cache.
        """
        await self._load_config()
        self.invalidate_all_rpd()

    async def reload_config(self) -> None:
        """Reload config from DB but keep the cached RPD counts.

        For admin edits no count depends on -- access rules, profile names, a new
        user or an empty new profile. Limit, composition and usage changes must use
        refresh_now instead.
        """
        await self._load_config()

    def group_for(self, model_id: Optional[str]) -> int:
        """The model group governing a model id; DEFAULT_GROUP_ID when none claims it.

        An explicit model member wins over an instance member, so one model can be
        carved out of an instance another group claims. Instance membership is
        resolved here on every call rather than expanded when the group is saved,
        so a model synced from that instance later joins the group by itself.
        """
        if not model_id:
            return DEFAULT_GROUP_ID
        gid = self._model_to_group.get(model_id)
        if gid is not None:
            return gid
        gid = self._provider_to_group.get(provider_key_of(model_id))
        return DEFAULT_GROUP_ID if gid is None else gid

    def blocking_group(self, model_id: Optional[str]) -> Optional[str]:
        """Name of the switched-off group governing a model, or None when it is served.

        Follows group_for, so a model carved out of a disabled instance's group into an
        enabled one is still served. Default cannot be switched off.
        """
        snap = self._groups.get(self.group_for(model_id))
        return snap.name if snap is not None and not snap.enabled else None

    def group_name(self, group_id: int) -> Optional[str]:
        if group_id == DEFAULT_GROUP_ID:
            return "Default"
        snap = self._groups.get(group_id)
        return snap.name if snap else None

    def grouped_keys(self) -> Tuple[set, set]:
        """Return (explicit_model_ids, instance_provider_keys) claimed by named groups.

        A row belongs to the Default group exactly when neither set claims it; see
        request_tracker.get_today_count.
        """
        return set(self._model_to_group.keys()), set(self._provider_to_group.keys())

    # -- limit values -------------------------------------------------------
    #
    # Every limit is the user's profile's (app.profiles); None means unlimited. These
    # four are the only readers, so enforcement, pooling and settlement can never
    # disagree about a user's limit.

    def _own_rpm_limit(self, user_id: int) -> Optional[int]:
        """One user's own overall RPM — their profile's."""
        return profile_registry.for_user(user_id).rpm

    def _resolve_group_limits(
        self, user_id: int, group: _ModelGroupSnapshot
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (effective_group_rpm, effective_group_rpd) for user_id in group."""
        return self.group_limits_for(user_id, group.group_id)

    def group_limits_for(
        self, user_id: int, group_id: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """(rpm, rpd) this user's profile sets for a group; the one lookup enforcement and display share.

        DEFAULT_GROUP_ID resolves to the profile's own rpm/rpd.
        """
        return profile_registry.for_user(user_id).model_group_limit(group_id)

    # -- concurrency --------------------------------------------------------
    #
    # Synchronous on purpose: with no await between the check and the increment, the
    # pair is atomic on the one event loop every port shares.

    def concurrency_limit_for(self, user_id: int) -> Optional[int]:
        """Requests this user's profile lets them have in progress at once; None = unlimited."""
        return profile_registry.for_user(user_id).max_concurrent

    def inflight_for(self, user_id: int) -> int:
        return self._inflight.get(user_id, 0)

    def try_acquire_slot(self, user_id: int) -> Optional[RateLimitDecision]:
        """Take an in-flight slot, or return the denial if the user is at their cap.

        Unlimited users are counted too, so a cap set while they have requests open
        applies to them at once rather than only after those finish.
        """
        limit = self.concurrency_limit_for(user_id)
        current = self._inflight.get(user_id, 0)
        if limit is not None and current >= limit:
            return RateLimitDecision(
                allowed=False,
                rpm_limit=None, rpm_remaining=None, rpd_limit=None, rpd_remaining=None,
                retry_after_seconds=1,
                limited_by="concurrency",
                concurrency_limit=limit,
            )
        self._inflight[user_id] = current + 1
        return None

    def release_slot(self, user_id: int) -> None:
        current = self._inflight.get(user_id, 0)
        if current <= 1:
            self._inflight.pop(user_id, None)
        else:
            self._inflight[user_id] = current - 1

    # -- request pooling ---------------------------------------------------
    #
    # A pool shares one daily quota: its limit is the sum of its members' limits and
    # its count is the sum of their consumption. Only RPD is pooled — every RPM path
    # above keeps reading the caller's own user_id.

    def _rpd_scope(self, user_id: int, username: str) -> Tuple[str, List[int], Optional[int]]:
        """Return (cache_scope_key, member_user_ids, pool_id).

        Unpooled: ("user:alice", [7], None)
        Pooled:   ("pool:3", [7,8,9], 3)

        The member ids are what the usage tables are keyed by, so they are what the
        today-counts are read for. The scope key is what the RPD caches are keyed by,
        so all members of a pool share one cache entry — one DB read per pool per TTL,
        not one per member. Both forms are prefixed so the two keyspaces cannot
        collide; see _user_scope_key.
        """
        pool_id = self._user_to_pool.get(user_id)
        members = self._pool_members.get(pool_id) if pool_id is not None else None
        if not members:
            return _user_scope_key(username), [user_id], None
        return _pool_scope_key(pool_id), [i for i, _ in members], pool_id

    def pool_id_for_user(self, user_id: int) -> int:
        """The pool this user is in per the current snapshot, or 0 when unpooled.

        request_tracker stamps this onto every usage row as it is recorded, which is
        what makes pool usage a plain filter. The snapshot is re-read after every
        composition change (_invalidate -> refresh_now), so the stamp tracks joins and
        leaves as soon as they commit.
        """
        return self._user_to_pool.get(user_id) or 0

    def _own_rpd_limit(self, user_id: int) -> Optional[int]:
        """One user's own effective overall RPD — their profile's."""
        return profile_registry.for_user(user_id).rpd

    def _pooled_rpd_limit(self, member_ids: List[int], per_member) -> Optional[int]:
        """Sum the per-member effective RPD. None from any member ⇒ unlimited pool.

        `per_member` is a callable resolving one member's effective limit for the tier
        being checked; it is evaluated under the global lock, where the snapshot dicts
        this reads are already held.

        Deactivated members are skipped, limit and unlimited alike. A disabled account
        cannot send a request -- get_api_key joins on User.is_active and every
        middleware path gates on it -- so counting its limit would hand a quota nobody
        can spend to whoever it was pooled with, and one disabled unlimited account
        would make the pool unlimited forever. Their rows still count: _rpd_scope keeps
        returning every member id, because traffic they sent while active was real.
        """
        total = 0
        for uid in member_ids:
            if uid in self._inactive_members:
                continue
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

    def share_pct_for(self, pool_id: Optional[int]) -> Optional[int]:
        """The per-member share percent in force for a pool; None when uncapped."""
        if pool_id is None:
            return None
        return self._pool_share_override.get(pool_id, self._pool_share_default)

    def member_share_cap(
        self, pool_id: Optional[int], own_limit: Optional[int], pool_limit: Optional[int]
    ) -> Optional[int]:
        """The most of a pool's daily limit one member may consume on one scope.

            cap = max(own_limit, floor(pool_limit * pct / 100))

        The floor at own_limit means pooling never leaves a member worse off than going
        solo; the pool-total check still runs first, so the pool itself is never
        exceeded. None (uncapped) when no percent applies, at 100%, or when either limit
        is unlimited -- a percentage of "unlimited" bounds nothing, and an unlimited
        member's own floor is unlimited too.
        """
        pct = self.share_pct_for(pool_id)
        if pct is None or pct >= 100 or own_limit is None or pool_limit is None:
            return None
        return max(own_limit, pool_limit * pct // 100)

    def invalidate_pool(self, pool_id: int) -> None:
        """Drop every cached RPD count for a pool, across all three tiers.

        Called on every composition change: the membership that produced the cached
        count no longer exists, so serving it for the rest of the TTL would enforce
        against a pool that is already gone.
        """
        scope_key = _pool_scope_key(pool_id)
        self._rpd_cache.pop(scope_key, None)
        for key in [k for k in self._group_rpd_cache if k[0] == scope_key]:
            self._group_rpd_cache.pop(key, None)

    async def get_user_status(self, user_id: int, username: str) -> "UserStatus":
        """Read-only — returns current usage without incrementing."""
        lock = await self._get_lock()
        async with lock:
            rpm = self._own_rpm_limit(user_id)

            scope_key, member_ids, pool_id = self._rpd_scope(user_id, username)
            if pool_id is None:
                rpd = self._own_rpd_limit(user_id)
            else:
                rpd = self._pooled_rpd_limit(member_ids, self._own_rpd_limit)

            now = time.time()
            current_window = int(now // 60)
            bucket = self._minute_buckets.get(user_id)
            rpm_count = (bucket.count if bucket and bucket.window == current_window else 0)

        rpd_count = await self._get_today_count(scope_key, member_ids)

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
        """Check a named model group's limits. Returns a denied decision, or None if
        allowed or the model is in the Default group.

        Called from route handlers after model resolution and the access check (see
        app.rate_limit_dep.enforce_access_and_limits). The Default group is checked
        by check_and_increment instead.
        """
        global_lock = await self._get_lock()
        async with global_lock:
            group_id = self.group_for(model_id)
            if group_id == DEFAULT_GROUP_ID:
                return None
            group = self._groups.get(group_id)
            if group is None:
                return None
            g_rpm, _own_rpd = self._resolve_group_limits(user_id, group)

            # RPD may be pooled across the members' group limits; RPM stays per-user.
            scope_key, member_ids, pool_id = self._rpd_scope(user_id, username)
            share_cap = None
            if pool_id is None:
                g_rpd = _own_rpd
            else:
                g_rpd = self._pooled_rpd_limit(
                    member_ids, lambda uid: self._resolve_group_limits(uid, group)[1]
                )
                share_cap = self.member_share_cap(pool_id, _own_rpd, g_rpd)
            share_pct = self.share_pct_for(pool_id)

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
                    scope_key, member_ids, group
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
                        pooled=pool_id is not None,
                    )
                # The pool has room, but this member may have used their share of it.
                if share_cap is not None:
                    own_count = await self._get_today_group_count(
                        _user_scope_key(username), [user_id], group
                    )
                    if own_count >= share_cap:
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
                            pooled=True,
                            pool_share_cap=share_cap,
                            pool_share_pct=share_pct,
                        )

            # Passed — increment group RPM bucket
            group_bucket.count += 1

        return None  # group checks passed

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
        for key in [k for k in self._group_rpd_cache if k[0] == scope_key]:
            self._group_rpd_cache.pop(key, None)
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

    async def check_and_increment(
        self, user_id: int, username: str
    ) -> RateLimitDecision:
        """Enforce the Default group's RPM/RPD (the profile's own limits) and
        increment the RPM bucket.

        Named groups are enforced by check_group_limit. Callers are the routes, for a
        model in the Default group (app.rate_limit_dep.enforce_access_and_limits), and
        the auth middleware, for requests that name no model at all.
        """
        global_lock = await self._get_lock()
        async with global_lock:
            rpm = self._own_rpm_limit(user_id)

            # RPD may be pooled; RPM above is always this user's own.
            scope_key, member_ids, pool_id = self._rpd_scope(user_id, username)
            share_cap = None
            if pool_id is None:
                rpd = self._own_rpd_limit(user_id)
            else:
                rpd = self._pooled_rpd_limit(member_ids, self._own_rpd_limit)
                share_cap = self.member_share_cap(pool_id, self._own_rpd_limit(user_id), rpd)
            share_pct = self.share_pct_for(pool_id)

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
                today_count = await self._get_today_count(scope_key, member_ids)
                if today_count >= rpd:
                    return RateLimitDecision(
                        allowed=False,
                        rpm_limit=rpm, rpm_remaining=None,
                        rpd_limit=rpd, rpd_remaining=0,
                        retry_after_seconds=_seconds_until_local_midnight(),
                        limited_by="rpd",
                        pooled=pool_id is not None,
                    )
                # The pool has room, but this member may have used their share of it.
                if share_cap is not None:
                    own_count = await self._get_today_count(_user_scope_key(username), [user_id])
                    if own_count >= share_cap:
                        return RateLimitDecision(
                            allowed=False,
                            rpm_limit=rpm, rpm_remaining=None,
                            rpd_limit=rpd, rpd_remaining=0,
                            retry_after_seconds=_seconds_until_local_midnight(),
                            limited_by="rpd",
                            pooled=True,
                            pool_share_cap=share_cap,
                            pool_share_pct=share_pct,
                        )

            # All checks passed — increment the RPM bucket
            bucket.count += 1
            rpm_remaining = (rpm - bucket.count) if rpm is not None else None

        rpd_count = await self._get_today_count(scope_key, member_ids)
        rpd_remaining = max(0, rpd - rpd_count) if rpd is not None else None

        return RateLimitDecision(
            allowed=True,
            rpm_limit=rpm, rpm_remaining=rpm_remaining,
            rpd_limit=rpd, rpd_remaining=rpd_remaining,
            retry_after_seconds=0,
            limited_by=None,
        )

    async def get_group_rpd_count(
        self, user_id: int, user_identity: str, group_id: int
    ) -> int:
        """Read-only — today's request count for a group (TTL-cached, no increment).

        DEFAULT_GROUP_ID reads the Default count. Pooled callers get the whole pool's
        count, matching what enforcement sees.
        """
        scope_key, member_ids, _ = self._rpd_scope(user_id, user_identity)
        if group_id == DEFAULT_GROUP_ID:
            return await self._get_today_count(scope_key, member_ids)
        group = self._groups.get(group_id)
        if group is None:
            return 0
        return await self._get_today_group_count(scope_key, member_ids, group)

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

    def pooled_group_rpd_limit(self, user_id: int, group_id: int) -> Optional[int]:
        """Pooled effective RPD for one group tier, or None when unlimited/unpooled.

        Returns None both for "no pool" and for "unlimited pool"; callers that need to
        tell them apart check pooled_rpd_limits()[0] first.
        """
        pool_id = self._user_to_pool.get(user_id)
        members = self._pool_members.get(pool_id) if pool_id is not None else None
        if not members:
            return None
        member_ids = [uid for uid, _ in members]
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
        """Every scope a pool can be settled on: ('overall', 0) -- the Default group --
        plus each named group."""
        scopes: List[Tuple[str, int]] = [("overall", 0)]
        scopes.extend(("model_group", gid) for gid in self._groups)
        return scopes

    def scope_for_model(self, model_id: str) -> Optional[Tuple[str, int]]:
        """Which named group scope a model's usage rows count against, or None for the
        Default group (scope ('overall', 0)).

        Resolves through group_for, the same lookup enforcement uses.
        """
        gid = self.group_for(model_id)
        if gid == DEFAULT_GROUP_ID:
            return None
        return ("model_group", gid)

    def member_limit_for_scope(
        self, user_id: int, scope_kind: str, scope_id: int
    ) -> Optional[int]:
        """One member's own effective RPD on a scope. None means unlimited.

        A group the user is not a member of still resolves through their profile's limit
        for it, exactly as enforcement does -- the group's membership is over
        models/instances, not users.
        """
        if scope_kind == "overall":
            return self._own_rpd_limit(user_id)
        if scope_kind == "model_group":
            group = self._groups.get(scope_id)
            return None if group is None else self._resolve_group_limits(user_id, group)[1]
        return None

    async def member_share_status(
        self, user_id: int, username: str, scope_kind: str, scope_id: int,
        pool_limit: Optional[int],
    ) -> Tuple[Optional[int], Optional[int]]:
        """(share_cap, share_used) for a pooled user on one scope, for display.

        Both None when unpooled or uncapped. share_used is the member's own effective
        count -- rows plus their carry -- read through the same cached reader the
        enforcement path uses, so the quota page and the 429 agree.
        """
        pool_id = self._user_to_pool.get(user_id)
        if pool_id is None:
            return None, None
        cap = self.member_share_cap(
            pool_id, self.member_limit_for_scope(user_id, scope_kind, scope_id), pool_limit,
        )
        if cap is None:
            return None, None
        own_key = _user_scope_key(username)
        if scope_kind == "model_group":
            group = self._groups.get(scope_id)
            if group is None:
                return None, None
            used = await self._get_today_group_count(own_key, [user_id], group)
        else:
            used = await self._get_today_count(own_key, [user_id])
        return cap, used

    def carry_for(self, user_id: int, scope_kind: str, scope_id: int = 0) -> int:
        """One user's own day-scoped carry on a scope — for explaining a count in the UI."""
        return self._carry_sum([user_id], scope_kind, scope_id)

    async def _get_today_count(
        self, scope_key: str, member_ids: List[int]
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
        async def load() -> int:
            from app.request_tracker import request_tracker
            rows = await request_tracker.get_today_count(member_ids)
            return max(0, rows + self._carry_sum(member_ids, "overall", 0))

        return await self._read_rpd_cached(
            "overall", self._rpd_cache, scope_key, load,
            "RPD count read failed for %s; serving last-known count", (scope_key,),
        )

    async def _read_rpd_cached(
        self, kind: str, cache: Dict[Any, _RpdCacheEntry], cache_key: Any,
        load: Callable[[], Awaitable[int]], error_msg: str, error_args: tuple,
    ) -> int:
        """Serve an RPD count from `cache`, refilling it with one `load()` per key.

        Requests that miss together wait on the first one's read and then take the
        value it cached, so a burst for one user or pool opens one DB session, not one
        per request.

        On a load failure the last-known entry is served even if expired; with none,
        0 is returned but NOT cached, so the next request retries immediately.
        Requests already waiting when a load fails share that failure rather than
        each retrying in turn, which would queue them behind one DB timeout apiece.
        """
        entry = cache.get(cache_key)
        if entry and entry.expires_at > time.time():
            return entry.count

        flight_key = (kind, cache_key)
        flight = self._rpd_inflight.get(flight_key)
        if flight is None:
            # [lock, holders, monotonic time of the last failed load]
            flight = self._rpd_inflight[flight_key] = [asyncio.Lock(), 0, None]
        flight[1] += 1
        arrived = time.monotonic()
        try:
            async with flight[0]:
                # Another request may have refilled the entry while we waited.
                entry = cache.get(cache_key)
                now = time.time()
                if entry and entry.expires_at > now:
                    return entry.count
                if flight[2] is not None and flight[2] >= arrived:
                    return entry.count if entry else 0
                try:
                    count = await load()
                except Exception:
                    flight[2] = time.monotonic()
                    logger.error(error_msg, *error_args, exc_info=True)
                    return entry.count if entry else 0
                cache[cache_key] = _RpdCacheEntry(count=count, expires_at=now + _RPD_TTL)
                return count
        finally:
            flight[1] -= 1
            if flight[1] == 0 and self._rpd_inflight.get(flight_key) is flight:
                del self._rpd_inflight[flight_key]

    async def _get_today_group_count(
        self, scope_key: str, member_ids: List[int], group: _ModelGroupSnapshot,
    ) -> int:
        """Return today's effective request count for every model the group governs, TTL-cached."""
        group_id = group.group_id

        async def load() -> int:
            from app.request_tracker import request_tracker
            rows = await request_tracker.get_today_group_count(
                member_ids, group.model_ids, group.provider_keys
            )
            return max(0, rows + self._carry_sum(member_ids, "model_group", group_id))

        return await self._read_rpd_cached(
            "model_group", self._group_rpd_cache, (scope_key, group_id), load,
            "Group RPD count read failed for %s group %s; serving last-known count",
            (scope_key, group_id),
        )


rate_limit_tracker = RateLimitTracker()

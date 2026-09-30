"""
Real-time active request tracking for the admin dashboard.

Provides an in-memory store of all in-flight LLM API requests and
broadcasts events to SSE subscribers when requests start or complete.
"""

import asyncio
import json
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Optional
from app import time_utils
from app.concurrency import TaskReentrantLock

logger = logging.getLogger(__name__)


# Promoted to app.concurrency so pool settlement can use the same primitive; the
# private name stays for this module's own call sites.
_TaskReentrantLock = TaskReentrantLock


@dataclass
class ActiveRequest:
    request_id: str
    server: str              # "openai" | "anthropic" | "azure_openai"
    endpoint: str            # e.g. "/v1/chat/completions"
    method: str              # "POST", "GET", etc.
    model: Optional[str]     # extracted from request body
    user_identity: str       # username, API key name, or "unknown"
    user_type: str           # "user" | "api_key" | "admin" | "unknown"
    is_streaming: bool
    start_time: float        # time.time()
    status: str = "in_progress"
    # Attribution key for usage rows: users.id, ADMIN_USAGE_USER_ID for the config
    # admin, None until authentication has run (such a request is never counted).
    user_id: Optional[int] = None


# Slots of the usage buffer key. Everything that reads or rewrites the buffer goes
# through these names rather than bare indexes.
_K_DATE, _K_HOUR, _K_USER_ID, _K_IDENTITY, _K_TYPE, _K_MODEL, _K_SERVER, _K_POOL = range(8)


class RequestTracker:
    # Seconds between DB flushes. Bounds how much usage a hard kill can lose; the
    # flush is one upsert per table, so it is cheap to run often.
    FLUSH_INTERVAL = 10

    def __init__(self):
        self._active: dict[str, ActiveRequest] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._usage_buffer: dict[tuple, int] = defaultdict(int)
        self._usage_lock = asyncio.Lock()
        # Serialises whole flushes. _usage_lock only guards the buffer itself and is
        # released across the DB write; this one spans snapshot → write → subtract.
        self._flush_mutex = _TaskReentrantLock()
        self._flush_task: Optional[asyncio.Task] = None
        self._last_rollup_at: float = 0.0

    async def start(self):
        self._running = True
        self._flush_task = asyncio.create_task(self._usage_flush_loop())
        logger.info("RequestTracker started")

    async def stop(self):
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush on shutdown
        await self._do_flush()
        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            self._subscribers.clear()
            self._active.clear()
        logger.info("RequestTracker stopped")

    async def flush_pending(self):
        """Flush buffered usage counts to the DB immediately."""
        await self._do_flush()

    @asynccontextmanager
    async def pause_flush(self):
        """Hold off the usage flush for the duration of the block.

        Operations that rewrite usage in both places it lives — the DB rows and the
        buffered counts that have not reached them yet — must run under this. A
        flush is not atomic: it snapshots the buffer, writes it, then subtracts. One
        landing between the two halves writes its pre-change snapshot under the old
        identity, re-creating exactly the rows the caller just moved or purged.

        Re-entrant, so flush_pending()/rename_identity()/drop_buffered_usage() can
        still be called from inside the block.
        """
        async with self._flush_mutex:
            yield

    async def _usage_flush_loop(self):
        while self._running:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            await self._do_flush()

    async def _do_flush(self):
        # Serialised against other flushes: the DB write happens outside
        # _usage_lock, so two overlapping flushes would take the same snapshot and
        # apply the increment-on-conflict upserts twice, inflating usage and the
        # RPD counts derived from it.
        async with self._flush_mutex:
            await self._flush_locked()

    async def _flush_locked(self):
        """Flush body. Caller holds _flush_mutex."""
        async with self._usage_lock:
            if not self._usage_buffer:
                return
            snapshot = dict(self._usage_buffer)

        # Buffer key: (date, hour, user_id, user_identity, user_type, model, server, pool_id)
        hourly_rows = [
            {
                "date": key[_K_DATE],
                "hour": key[_K_HOUR],
                "user_id": key[_K_USER_ID],
                "user_identity": key[_K_IDENTITY],
                "user_type": key[_K_TYPE],
                "model": key[_K_MODEL],
                "server": key[_K_SERVER],
                "pool_id": key[_K_POOL],
                "request_count": count,
            }
            for key, count in snapshot.items()
        ]

        # Collapse hourly rows into daily rows: drop the hour, sum the rest. Two hourly
        # keys differing only in user_type land on the same daily upsert key -- the
        # label of whichever comes last wins, which is what the upsert does anyway.
        daily_map: dict[tuple, int] = defaultdict(int)
        daily_labels: dict[tuple, tuple] = {}
        for key, count in snapshot.items():
            daily_key = (key[_K_DATE], key[_K_USER_ID], key[_K_MODEL], key[_K_SERVER], key[_K_POOL])
            daily_map[daily_key] += count
            daily_labels[daily_key] = (key[_K_IDENTITY], key[_K_TYPE])
        daily_rows = [
            {
                "date": k[0],
                "user_id": k[1],
                "user_identity": daily_labels[k][0],
                "user_type": daily_labels[k][1],
                "model": k[2],
                "server": k[3],
                "pool_id": k[4],
                "request_count": count,
            }
            for k, count in daily_map.items()
        ]

        try:
            from app.auth.database import (
                flush_usage_rows, prune_hourly_usage, rollup_to_monthly,
                purge_stale_pool_rows,
            )
            # The increment-on-conflict upserts are the only ops that consume the
            # buffered counts. Run just those in the guarded block so a later failure
            # in prune/rollup cannot leave the buffer uncleared and cause the next
            # cycle to re-add the same counts (double-counting inflates usage and RPD
            # decisions). Both tables are written in one transaction, so this either
            # consumes the snapshot entirely or leaves it untouched for the retry --
            # a half-applied flush would re-add the committed half forever.
            await flush_usage_rows(hourly_rows, daily_rows)
        except Exception as e:
            logger.error(f"Usage flush failed, will retry next cycle: {e}")
            return

        # Both upserts committed — subtract exactly the flushed snapshot under the
        # lock (increments that arrived during the write are preserved).
        async with self._usage_lock:
            for key, count in snapshot.items():
                self._usage_buffer[key] -= count
                if self._usage_buffer[key] <= 0:
                    del self._usage_buffer[key]

        # Maintenance (prune old hourly rows, throttled monthly rollup) is
        # best-effort and independent of the buffer; its failure must not trigger
        # a re-flush of already-committed usage.
        try:
            await prune_hourly_usage()
            if time.time() - self._last_rollup_at >= 3600:
                await rollup_to_monthly()
                await purge_stale_pool_rows()
                self._last_rollup_at = time.time()
        except Exception as e:
            logger.error(f"Usage maintenance (prune/rollup) failed: {e}")

    async def start_request(
        self,
        request_id: str,
        server: str,
        endpoint: str,
        method: str,
        model: Optional[str],
        user_identity: str,
        user_type: str,
        is_streaming: bool,
    ) -> None:
        entry = ActiveRequest(
            request_id=request_id,
            server=server,
            endpoint=endpoint,
            method=method,
            model=model,
            user_identity=user_identity,
            user_type=user_type,
            is_streaming=is_streaming,
            start_time=time.time(),
        )
        async with self._lock:
            self._active[request_id] = entry
        await self._broadcast("request_started", entry)

    async def end_request(
        self,
        request_id: str,
        status: str = "completed",
        termination_reason: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        # Accumulate usage (skip unauthenticated requests).
        # Only successful requests count toward usage / rate limits: a 2xx
        # response ("completed") or a request the client cancelled mid-flight
        # ("cancelled", work was already done). Errored requests (4xx/5xx,
        # stream/timeout failures) must NOT consume quota.
        # Requests with no model (metadata/listing endpoints like GET
        # /v1/models, /v1/responses/{id}, etc.) are not real model usage and
        # are skipped so they don't surface as an "unknown" model row.
        #
        # The pop and the buffer increment happen together under _lock. Splitting them
        # leaves a window where the request is in neither _active nor the buffer, and
        # those two are exactly what rename_identity sweeps: a rename landing in the gap
        # relabels nothing, and the old label this then buffers rides the next flush's
        # on-conflict update straight over the row rename_usage_identity just fixed.
        # _usage_lock is never held across a DB write, so nesting it here is cheap.
        _COUNTED_STATUSES = ("completed", "cancelled")
        async with self._lock:
            entry = self._active.pop(request_id, None)
            if entry is not None and (
                entry.model and entry.user_type != "unknown"
                and entry.user_id is not None and status in _COUNTED_STATUSES
            ):
                now_local = time_utils.local_now()
                key = (
                    now_local.date(),
                    now_local.hour,
                    entry.user_id,
                    entry.user_identity,
                    entry.user_type,
                    entry.model or "unknown",
                    entry.server,
                    # The pool the sender is in *now*: pool usage is a filter on this
                    # column, so a request completed after leaving is not the pool's.
                    self._pool_id_for(entry.user_id),
                )
                async with self._usage_lock:
                    self._usage_buffer[key] += 1
        if entry is None:
            return

        entry.status = status
        data = self._serialize(entry)
        if termination_reason:
            data["termination_reason"] = termination_reason
        if error:
            data["error"] = error
        if status == "completed":
            event_type = "request_completed"
        elif status == "cancelled":
            event_type = "request_cancelled"
        else:
            event_type = "request_errored"
        await self._broadcast_raw(event_type, data)

    @staticmethod
    def _pool_id_for(user_id: int) -> int:
        """The pool a user is in right now, per the rate limiter's snapshot (0 if none)."""
        try:
            from app.rate_limit import rate_limit_tracker
            return rate_limit_tracker.pool_id_for_user(user_id)
        except Exception:
            return 0

    async def update_identity(
        self,
        request_id: str,
        user_identity: str,
        user_type: str,
        user_id: Optional[int] = None,
    ) -> None:
        async with self._lock:
            entry = self._active.get(request_id)
            if entry is None:
                return
            entry.user_identity = user_identity
            entry.user_type = user_type
            entry.user_id = user_id
            data = self._serialize(entry)
        await self._broadcast_raw("request_updated", data)

    async def rename_identity(self, user_id: int, new: str) -> None:
        """Relabel buffered and in-flight usage of one user with their new username.

        Called after a username change has been committed. The DB rows are relabelled
        by rename_usage_identity; this covers the counts that have not reached the DB
        yet — anything buffered since the pre-rename flush, plus requests that were
        already in flight when the rename happened. Attribution is by user_id so
        nothing is at stake for quotas; without this, though, the next flush would
        write new (date, hour) rows under the old label.
        """
        # Excludes a concurrent flush: one caught mid-write would finish writing its
        # pre-rename snapshot under the old label and then subtract that snapshot from
        # buffer entries this method has already moved.
        async with self._flush_mutex:
            # Both sweeps under _lock, nesting _usage_lock inside it exactly as
            # end_request does. Held together they cover every counted request: one
            # still in flight is relabelled in _active, one already completed is
            # relabelled in the buffer, and end_request cannot be between the two.
            async with self._lock:
                for entry in self._active.values():
                    if entry.user_id == user_id:
                        entry.user_identity = new

                async with self._usage_lock:
                    stale = [
                        key for key in self._usage_buffer
                        if key[_K_USER_ID] == user_id and key[_K_IDENTITY] != new
                    ]
                    for key in stale:
                        relabelled = key[:_K_IDENTITY] + (new,) + key[_K_IDENTITY + 1:]
                        # The new label may already have a buffered count for this key.
                        self._usage_buffer[relabelled] += self._usage_buffer.pop(key)

    async def drop_buffered_usage(self, axis: str, value) -> int:
        """Discard buffered counts for one user (axis='user', value=user_id), one model
        (axis='model') or one pool (axis='pool', value=pool_id).

        Called after an admin purges usage rows, or a user is deleted. Counts buffered
        since the last flush have not reached the DB yet, so without this they land
        on the next cycle and recreate the rows that were just deleted.

        Returns the number of requests dropped, for logging.
        """
        slot = {"user": _K_USER_ID, "model": _K_MODEL, "pool": _K_POOL}[axis]

        # Excludes a concurrent flush, which would otherwise write its pre-purge
        # snapshot into the tables the caller just cleared.
        async with self._flush_mutex:
            async with self._usage_lock:
                stale = [key for key in self._usage_buffer if key[slot] == value]
                return sum(self._usage_buffer.pop(key) for key in stale)

    async def forget_user(self, user_id: int) -> int:
        """Stop attributing any usage to a user whose account was just deleted.

        drop_buffered_usage alone only clears what has already completed. A request
        still in flight -- typically a long stream -- ends later and buffers a fresh
        count under the deleted id, and the next flush writes it back. users.id is not
        AUTOINCREMENT, so SQLite hands that id to the next account registered, which
        would inherit the rows and have today's portion charged against its RPD.

        So, like rename_identity, both places are swept under _lock and _usage_lock
        together: buffered counts are dropped, and in-flight entries are detached
        (user_id=None is never counted, see end_request). end_request cannot land
        between the two sweeps. The entries stay in _active so the live view still
        shows them finishing.

        Returns the number of buffered requests dropped, for logging.
        """
        async with self._flush_mutex:
            async with self._lock:
                for entry in self._active.values():
                    if entry.user_id == user_id:
                        entry.user_id = None
                async with self._usage_lock:
                    stale = [key for key in self._usage_buffer if key[_K_USER_ID] == user_id]
                    return sum(self._usage_buffer.pop(key) for key in stale)

    async def update_streaming(
        self,
        request_id: str,
        is_streaming: bool,
    ) -> None:
        """Update the active request streaming mode and notify subscribers."""
        async with self._lock:
            entry = self._active.get(request_id)
            if entry is None or entry.is_streaming == is_streaming:
                return
            entry.is_streaming = is_streaming
            data = self._serialize(entry)
        await self._broadcast_raw("request_updated", data)

    async def update_model(self, request_id: str, model: str) -> None:
        """Correct the model on an in-flight request and notify subscribers.

        The tracking middleware records the name the client sent; the route
        resolves it to a canonical '{provider_key}/{name}' id after
        authentication (app/model_resolution.py). entry.model is a component of
        the usage key built in end_request, so without this the same logical
        model would split across a bare row and a prefixed row.

        Only _lock is needed: the usage key is built strictly later, and nothing
        is buffered at this point.
        """
        if not model:
            return
        async with self._lock:
            entry = self._active.get(request_id)
            if entry is None or entry.model == model:
                return
            entry.model = model
            data = self._serialize(entry)
        await self._broadcast_raw("request_updated", data)

    def get_active_requests(self) -> list[dict]:
        snapshot = dict(self._active)
        return [self._serialize(r) for r in snapshot.values()]

    def get_summary(self) -> dict:
        snapshot = list(self._active.values())
        by_server: dict[str, int] = {}
        by_model: dict[str, int] = {}
        for r in snapshot:
            by_server[r.server] = by_server.get(r.server, 0) + 1
            if r.model:
                by_model[r.model] = by_model.get(r.model, 0) + 1
        return {
            "total": len(snapshot),
            "by_server": by_server,
            "by_model": by_model,
        }

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    @staticmethod
    def _id_set(user_ids) -> set:
        """Normalise a single user id or a sequence of them into a set.

        Request pools count several users against one shared daily quota, so the
        today-count readers take either form.
        """
        if isinstance(user_ids, int):
            return {user_ids}
        return set(user_ids)

    async def get_today_count(self, user_ids) -> int:
        """Return total ungrouped requests today for these users (buffer + DB).

        `user_ids` is one user id or, for a pooled user, the ids of everyone sharing
        the pool's quota — counted in one query rather than one per member. Rows are
        counted wherever they were sent (any pool_id): a member's consumption is theirs
        whichever pool they were in at the time.

        Requests whose model belongs to a model group, or whose instance (provider_key
        prefix) belongs to an instance group, are excluded — those are governed by the
        group's own limit and never counted against the overall quota, matching the
        auth middleware's overall-gate skip.
        """
        ids = self._id_set(user_ids)
        if not ids:
            return 0
        today = time_utils.local_today()

        # Grouped model_ids and provider_keys to exclude from the overall count.
        try:
            from app.rate_limit import rate_limit_tracker
            grouped_models, grouped_providers = rate_limit_tracker.grouped_keys()
        except Exception:
            grouped_models, grouped_providers = set(), set()

        def _is_grouped(model) -> bool:
            if not model:
                return False
            if model in grouped_models:
                return True
            prefix = model.split('/', 1)[0] if '/' in model else model
            return prefix in grouped_providers

        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                if (key[_K_DATE] == today and key[_K_USER_ID] in ids
                        and not _is_grouped(key[_K_MODEL])):
                    buffered += count

        try:
            from sqlalchemy.future import select
            from sqlalchemy import func, or_, and_, not_
            from app.auth.database import AsyncSessionLocal
            from app.auth.models import RequestUsage
            async with AsyncSessionLocal() as db:
                conditions = [
                    RequestUsage.date == today,
                    RequestUsage.user_id.in_(list(ids)),
                ]
                # Exclude grouped models (exact match) and grouped instances (prefix match).
                exclude = []
                if grouped_models:
                    exclude.append(RequestUsage.model.in_(list(grouped_models)))
                for pk in grouped_providers:
                    exclude.append(RequestUsage.model.like(f"{pk}/%"))
                    exclude.append(RequestUsage.model == pk)
                if exclude:
                    conditions.append(not_(or_(*exclude)))
                result = await db.execute(
                    select(func.sum(RequestUsage.request_count)).where(and_(*conditions))
                )
                db_count = result.scalar() or 0
        except Exception:
            logger.error(
                "get_today_count DB read failed for %s; returning buffered-only count",
                user_ids, exc_info=True,
            )
            db_count = 0

        return buffered + db_count

    async def get_today_group_count(self, user_ids, model_ids: list) -> int:
        """Return total requests today across all model_ids in a group (buffer + DB).

        Models whose provider is in an instance group are excluded, mirroring the
        exclusion get_today_count applies. Instance groups take precedence everywhere
        else -- scope_for_model resolves such a model to the instance group, the auth
        middleware only ever runs the instance-group gate for it, and settlement folds
        it one way -- so counting it here as well would inflate a model-group number
        that its own gate will never see, and split the quota display from enforcement.
        get_today_instance_group_count needs no matching exclusion: nothing outranks an
        instance group.

        `user_ids` is one user id or, for a pooled user, the ids of everyone sharing
        the pool's quota.
        """
        if not model_ids:
            return 0
        ids = self._id_set(user_ids)
        if not ids:
            return 0
        today = time_utils.local_today()
        model_set = set(model_ids)

        try:
            from app.rate_limit import rate_limit_tracker
            _grouped_models, grouped_providers = rate_limit_tracker.grouped_keys()
        except Exception:
            grouped_providers = set()

        def _instance_grouped(model) -> bool:
            if not model:
                return False
            prefix = model.split('/', 1)[0] if '/' in model else model
            return prefix in grouped_providers

        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                if (key[_K_DATE] == today and key[_K_USER_ID] in ids
                        and key[_K_MODEL] in model_set and not _instance_grouped(key[_K_MODEL])):
                    buffered += count

        try:
            from sqlalchemy.future import select
            from sqlalchemy import func, or_, not_
            from app.auth.database import AsyncSessionLocal
            from app.auth.models import RequestUsage
            async with AsyncSessionLocal() as db:
                conditions = [
                    RequestUsage.date == today,
                    RequestUsage.user_id.in_(list(ids)),
                    RequestUsage.model.in_(model_ids),
                ]
                exclude = []
                for pk in grouped_providers:
                    exclude.append(RequestUsage.model.like(f"{pk}/%"))
                    exclude.append(RequestUsage.model == pk)
                if exclude:
                    conditions.append(not_(or_(*exclude)))
                result = await db.execute(
                    select(func.sum(RequestUsage.request_count)).where(*conditions)
                )
                db_count = result.scalar() or 0
        except Exception:
            logger.error(
                "get_today_group_count DB read failed for %s; returning buffered-only count",
                user_ids, exc_info=True,
            )
            db_count = 0

        return buffered + db_count

    async def get_today_instance_group_count(self, user_ids, provider_keys: list) -> int:
        """Return total requests today across all instances (provider_keys) in a group (buffer + DB).

        Instance membership matches the stored full model id by prefix: a model id is
        '{provider_key}/{model_name}', so membership is tested against the part before
        the first '/'.

        `user_ids` is one user id or, for a pooled user, the ids of everyone sharing
        the pool's quota.
        """
        if not provider_keys:
            return 0
        ids = self._id_set(user_ids)
        if not ids:
            return 0
        today = time_utils.local_today()
        pk_set = set(provider_keys)
        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                model = key[_K_MODEL]
                if key[_K_DATE] == today and key[_K_USER_ID] in ids and model:
                    prefix = model.split('/', 1)[0] if '/' in model else model
                    if prefix in pk_set:
                        buffered += count

        try:
            from sqlalchemy.future import select
            from sqlalchemy import func, or_
            from app.auth.database import AsyncSessionLocal
            from app.auth.models import RequestUsage
            async with AsyncSessionLocal() as db:
                conditions = [RequestUsage.model.like(f"{pk}/%") for pk in provider_keys]
                # Also match a bare provider_key with no model suffix, just in case.
                conditions += [RequestUsage.model == pk for pk in provider_keys]
                result = await db.execute(
                    select(func.sum(RequestUsage.request_count)).where(
                        RequestUsage.date == today,
                        RequestUsage.user_id.in_(list(ids)),
                        or_(*conditions),
                    )
                )
                db_count = result.scalar() or 0
        except Exception:
            logger.error(
                "get_today_instance_group_count DB read failed for %s; returning buffered-only count",
                user_ids, exc_info=True,
            )
            db_count = 0

        return buffered + db_count

    async def _broadcast(self, event_type: str, entry: ActiveRequest) -> None:
        await self._broadcast_raw(event_type, self._serialize(entry))

    async def _broadcast_raw(self, event_type: str, request_data: dict) -> None:
        if not self._subscribers:
            return
        payload = json.dumps({"event": event_type, "request": request_data})
        dead: list[asyncio.Queue] = []
        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(queue)
            for q in dead:
                self._subscribers.discard(q)
                # Signal end-of-stream so the consumer's queue.get() unblocks and
                # the SSE handler closes, instead of hanging on keepalives forever
                # (mirrors stop()). Best-effort: make room if the queue is full.
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(None)
                    except Exception:
                        pass

    @staticmethod
    def _serialize(entry: ActiveRequest) -> dict:
        return asdict(entry)


request_tracker = RequestTracker()

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
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from typing import Optional
from app import time_utils

logger = logging.getLogger(__name__)


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


class RequestTracker:
    FLUSH_INTERVAL = 60  # seconds between DB flushes

    def __init__(self):
        self._active: dict[str, ActiveRequest] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._usage_buffer: dict[tuple, int] = defaultdict(int)
        self._usage_lock = asyncio.Lock()
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

    async def _usage_flush_loop(self):
        while self._running:
            await asyncio.sleep(self.FLUSH_INTERVAL)
            await self._do_flush()

    async def _do_flush(self):
        async with self._usage_lock:
            if not self._usage_buffer:
                return
            snapshot = dict(self._usage_buffer)

        # Buffer key: (date, hour, user_identity, user_type, model, server)
        hourly_rows = [
            {
                "date": key[0],
                "hour": key[1],
                "user_identity": key[2],
                "user_type": key[3],
                "model": key[4],
                "server": key[5],
                "request_count": count,
            }
            for key, count in snapshot.items()
        ]

        # Collapse hourly rows into daily rows (sum counts for same date/user/model/server)
        daily_map: dict[tuple, int] = defaultdict(int)
        for key, count in snapshot.items():
            daily_key = (key[0], key[2], key[3], key[4], key[5])  # drop hour
            daily_map[daily_key] += count
        daily_rows = [
            {
                "date": k[0],
                "user_identity": k[1],
                "user_type": k[2],
                "model": k[3],
                "server": k[4],
                "request_count": count,
            }
            for k, count in daily_map.items()
        ]

        try:
            from app.auth.database import flush_request_usage, flush_request_usage_hourly, prune_hourly_usage, rollup_to_monthly
            await flush_request_usage_hourly(hourly_rows)
            await flush_request_usage(daily_rows)
            await prune_hourly_usage()

            # Throttle rollup to once per hour
            if time.time() - self._last_rollup_at >= 3600:
                await rollup_to_monthly()
                self._last_rollup_at = time.time()

            # Only clear after successful commit so get_today_count never sees a gap
            async with self._usage_lock:
                for key, count in snapshot.items():
                    self._usage_buffer[key] -= count
                    if self._usage_buffer[key] <= 0:
                        del self._usage_buffer[key]
        except Exception as e:
            logger.error(f"Usage flush failed, will retry next cycle: {e}")

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
        async with self._lock:
            entry = self._active.pop(request_id, None)
        if entry is None:
            return

        # Accumulate usage (skip unauthenticated requests)
        if entry.user_type != "unknown":
            now_local = time_utils.local_now()
            key = (
                now_local.date(),
                now_local.hour,
                entry.user_identity,
                entry.user_type,
                entry.model or "unknown",
                entry.server,
            )
            async with self._usage_lock:
                self._usage_buffer[key] += 1

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

    async def update_identity(
        self,
        request_id: str,
        user_identity: str,
        user_type: str,
    ) -> None:
        async with self._lock:
            entry = self._active.get(request_id)
            if entry is None:
                return
            entry.user_identity = user_identity
            entry.user_type = user_type
            data = self._serialize(entry)
        await self._broadcast_raw("request_updated", data)

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

    async def get_today_count(self, user_identity: str) -> int:
        """Return total ungrouped requests today for user_identity (buffer + DB).

        Requests whose model belongs to a model group, or whose instance (provider_key
        prefix) belongs to an instance group, are excluded — those are governed by the
        group's own limit and never counted against the overall quota, matching the
        auth middleware's overall-gate skip.
        """
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
                # key = (date, hour, user_identity, user_type, model, server)
                if key[0] == today and key[2] == user_identity and not _is_grouped(key[4]):
                    buffered += count

        try:
            from sqlalchemy.future import select
            from sqlalchemy import func, or_, and_, not_
            from app.auth.database import AsyncSessionLocal
            from app.auth.models import RequestUsage
            async with AsyncSessionLocal() as db:
                conditions = [
                    RequestUsage.date == today,
                    RequestUsage.user_identity == user_identity,
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
            db_count = 0

        return buffered + db_count

    async def get_today_group_count(self, user_identity: str, model_ids: list) -> int:
        """Return total requests today across all model_ids in a group (buffer + DB)."""
        if not model_ids:
            return 0
        today = time_utils.local_today()
        model_set = set(model_ids)
        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                # key = (date, hour, user_identity, user_type, model, server)
                if key[0] == today and key[2] == user_identity and key[4] in model_set:
                    buffered += count

        try:
            from sqlalchemy.future import select
            from sqlalchemy import func
            from app.auth.database import AsyncSessionLocal
            from app.auth.models import RequestUsage
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(func.sum(RequestUsage.request_count)).where(
                        RequestUsage.date == today,
                        RequestUsage.user_identity == user_identity,
                        RequestUsage.model.in_(model_ids),
                    )
                )
                db_count = result.scalar() or 0
        except Exception:
            db_count = 0

        return buffered + db_count

    async def get_today_instance_group_count(self, user_identity: str, provider_keys: list) -> int:
        """Return total requests today across all instances (provider_keys) in a group (buffer + DB).

        Instance membership matches the stored full model id by prefix: a model id is
        '{provider_key}/{model_name}', so membership is tested against the part before
        the first '/'.
        """
        if not provider_keys:
            return 0
        today = time_utils.local_today()
        pk_set = set(provider_keys)
        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                # key = (date, hour, user_identity, user_type, model, server)
                model = key[4]
                if key[0] == today and key[2] == user_identity and model:
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
                        RequestUsage.user_identity == user_identity,
                        or_(*conditions),
                    )
                )
                db_count = result.scalar() or 0
        except Exception:
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

    @staticmethod
    def _serialize(entry: ActiveRequest) -> dict:
        return asdict(entry)


request_tracker = RequestTracker()

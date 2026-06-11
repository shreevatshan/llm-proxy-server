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
from datetime import date
from typing import Optional

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

        rows = [
            {
                "date": key[0],
                "user_identity": key[1],
                "user_type": key[2],
                "model": key[3],
                "server": key[4],
                "request_count": count,
            }
            for key, count in snapshot.items()
        ]
        try:
            from app.auth.database import flush_request_usage, prune_request_usage
            await flush_request_usage(rows)
            await prune_request_usage(days=30)
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
            key = (
                date.today(),
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
        """Return total requests today for user_identity (buffer + DB)."""
        today = date.today()
        buffered = 0
        async with self._usage_lock:
            for key, count in self._usage_buffer.items():
                if key[0] == today and key[1] == user_identity:
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

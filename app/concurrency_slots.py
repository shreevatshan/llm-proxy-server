"""Per-user concurrency slots, each held for exactly the lifetime of one ASGI call.

A route takes a slot when it charges the request (charge_with_slot, which wraps the
RPM/RPD check), and ConcurrencySlotMiddleware gives it back when the call returns -- however it
ends: completed, raised, cancelled, or a send() failing mid-stream. Releasing from the
response body iterator instead would leave the last of those to garbage collection
(see the tracking middleware in app.main).

The middleware is pure ASGI, not BaseHTTPMiddleware, so it spawns no task and leaves
the contextvar arrangement the tracking middleware depends on alone. It must sit inside
the tracking middleware; _add_request_tracking registers it so.

A leaf module: rate_limit_tracker is imported lazily, as in app.auth.middleware.
"""

import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

SLOT_SCOPE_KEY = "llmproxy.concurrency_slot"


class _SlotHolder:
    """The slot one ASGI call holds, if any. release() is idempotent."""

    __slots__ = ("user_id",)

    def __init__(self) -> None:
        self.user_id: Optional[int] = None

    def release(self) -> None:
        if self.user_id is None:
            return
        from app.rate_limit import rate_limit_tracker
        user_id, self.user_id = self.user_id, None
        rate_limit_tracker.release_slot(user_id)


class ConcurrencySlotMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        holder = _SlotHolder()
        scope[SLOT_SCOPE_KEY] = holder
        try:
            await self.app(scope, receive, send)
        finally:
            holder.release()


_warned_unwired = False


def _holder(request) -> Optional[_SlotHolder]:
    holder = (getattr(request, "scope", None) or {}).get(SLOT_SCOPE_KEY)
    return holder if isinstance(holder, _SlotHolder) else None


async def charge_with_slot(request, user_id: int, charge: Callable[[], Awaitable]):
    """Take a slot for this request, then run *charge* (the RPM/RPD check, returning a
    RateLimitDecision or None). Returns the denial, else None.

    Concurrency goes first, so a request refused for it spends no RPM/RPD. A slot this
    call took is given back if *charge* denies. Taking a slot twice in one request is a
    no-op, so a route that enforces more than once is still counted once -- and a later
    check's denial leaves the slot an earlier one took alone.
    """
    global _warned_unwired
    holder = _holder(request)
    took = False
    if holder is None:
        # Every metered route is on an app _add_request_tracking wired up, so this is
        # a miswired app (or a test's bare request): the request is not counted.
        if not _warned_unwired:
            _warned_unwired = True
            logger.warning("concurrency_slots: request has no slot holder; concurrency "
                           "is not limited on this app")
    elif holder.user_id is None:
        from app.rate_limit import rate_limit_tracker
        denied = rate_limit_tracker.try_acquire_slot(user_id)
        if denied is not None:
            return denied
        holder.user_id = user_id
        took = True

    decision = await charge()
    if decision is not None and not decision.allowed:
        if took:
            holder.release()
        return decision
    return None

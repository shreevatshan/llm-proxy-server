"""Shared async primitives.

Small enough to be an import-time leaf: nothing here imports application modules, so
any layer can use it without a cycle.
"""

import asyncio


def lock_in_use(lock: asyncio.Lock) -> bool:
    """True while a task holds `lock` or is queued for it. The test for eviction.

    locked() alone is not enough: release() clears it before the next waiter has run to
    take the lock. Evicting in that gap lets the next caller build a fresh lock and walk
    in while the woken waiter walks in through the old one. asyncio keeps no public view
    of its queue, so this reads _waiters (a deque, or None before anyone has waited).
    """
    return lock.locked() or bool(getattr(lock, "_waiters", None))


class TaskReentrantLock:
    """An asyncio.Lock that the task already holding it may re-acquire.

    Used where a public coroutine takes the lock and callers also hold it across a
    wider critical section of their own -- the usage flush and RequestTracker's
    pause_flush(), and pool settlement, which loops over pools that each lock
    themselves. A plain Lock would self-deadlock on that nesting.

    Re-entrancy is per task, which is the whole point: it removes self-deadlock and
    deliberately does nothing for cross-task cycles, which remain the caller's problem
    to solve by acquiring in a consistent order.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._depth = 0

    def locked(self) -> bool:
        """True while any task holds this."""
        return self._lock.locked()

    def in_use(self) -> bool:
        """True while any task holds this or is waiting for it. The test for eviction."""
        return lock_in_use(self._lock)

    async def __aenter__(self):
        task = asyncio.current_task()
        if self._owner is not None and self._owner is task:
            self._depth += 1
            return self
        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, *exc_info):
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()
        return False

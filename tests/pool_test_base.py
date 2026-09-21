"""Shared fixture for the request-pool tests: a throwaway DB wired to a live tracker.

Settlement reads every member's limit from the rate-limit snapshot rather than from the
DB, so a pool test that stubbed the tracker would be testing arithmetic the real system
never performs. These tests instead point the real tracker at the same temporary
database and refresh it, which is also what exercises the snapshot-loading code.
"""

import os
import tempfile
import unittest
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.auth.models import (
    Base, RequestPool, RequestPoolMember, RequestUsage, User,
)

DAY = date(2026, 4, 7)


class PoolTestCase(unittest.IsolatedAsyncioTestCase):
    """A temp SQLite database, a real RateLimitTracker bound to it, and pool helpers."""

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self.db = self._session_factory()

        # Pin "today" so seeded usage and day-scoped ledger rows agree without
        # depending on when the suite runs.
        self._real_local_today = time_utils.local_today
        time_utils.local_today = lambda: DAY

        # request_tracker's today-count readers and the usage flush open their own
        # sessions from the module-level factory, so the test database has to be
        # installed there too or they would read the real one.
        from app.auth import database as auth_database
        self._real_session_local = auth_database.AsyncSessionLocal
        auth_database.AsyncSessionLocal = self._session_factory

        from app.rate_limit import rate_limit_tracker
        from app.request_tracker import request_tracker

        self.tracker = rate_limit_tracker
        self._saved_factory = self.tracker._db_session_factory
        self.tracker.set_db_session_factory(self._session_factory)

        # The usage buffer is process-global; settlement flushes it, so a test that
        # left rows in it would leak counts into the next one.
        self.request_tracker = request_tracker
        self._saved_buffer = dict(request_tracker._usage_buffer)
        request_tracker._usage_buffer.clear()

    async def asyncTearDown(self):
        from app.auth import database as auth_database
        auth_database.AsyncSessionLocal = self._real_session_local
        time_utils.local_today = self._real_local_today
        self.tracker.set_db_session_factory(self._saved_factory)
        self.tracker._user_to_pool = {}
        self.tracker._pool_members = {}
        self.tracker._identity_to_pool = {}
        self.tracker._carries = {}
        self.tracker._carries_date = None
        self.tracker._overrides = {}
        self.tracker._rpd_cache = {}
        self.tracker._group_rpd_cache = {}
        self.tracker._instance_group_rpd_cache = {}
        self.request_tracker._usage_buffer.clear()
        self.request_tracker._usage_buffer.update(self._saved_buffer)
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- seeding ---------------------------------------------------------

    async def make_user(self, username, rpd_limit=None):
        """Create a user, optionally with a personal overall RPD override."""
        from app.auth.models import UserRateLimit

        user = User(
            username=username, email=f"{username}@example.test",
            hashed_password="x", is_active=True,
        )
        self.db.add(user)
        await self.db.flush()
        if rpd_limit is not None:
            self.db.add(UserRateLimit(user_id=user.id, rpm_limit=None, rpd_limit=rpd_limit))
        await self.db.commit()
        return user

    async def seed_usage(self, username, count, *, model="p/m", server="openai", day=DAY):
        self.db.add(RequestUsage(
            date=day, user_identity=username, user_type="user",
            model=model, server=server, request_count=count,
        ))
        await self.db.commit()

    async def make_pool(self, name, owner, members=()):
        """Create a pool with an owner and, optionally, extra members already in it.

        Bypasses the join endpoint deliberately: most settlement tests want a given
        starting composition, not a history of joins.
        """
        pool = RequestPool(name=name, owner_user_id=owner.id)
        self.db.add(pool)
        await self.db.flush()
        for user in (owner,) + tuple(members):
            self.db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
        await self.db.commit()
        await self.refresh()
        return pool

    async def refresh(self):
        """Re-read the tracker snapshot from the test database."""
        await self.tracker.refresh_now()

    # -- reading ---------------------------------------------------------

    async def charged(self, pool_id, user_id, scope=("overall", 0)):
        from app.auth import pools as pool_settlement
        ledger = await pool_settlement._load_ledger(self.db, pool_id, DAY)
        return ledger.get((user_id, scope[0], scope[1]), 0)

    async def carry(self, user_id, scope=("overall", 0)):
        from app.auth import pools as pool_settlement
        carries = await pool_settlement._load_carries(self.db, [user_id], DAY)
        return carries.get((user_id, scope[0], scope[1]), 0)

    async def effective_used(self, user):
        """What the rate limiter reads as this user's daily consumption."""
        await self.refresh()
        status = await self.tracker.get_user_status(user.id, user.username)
        return status.rpd_count

    async def remaining(self, user):
        await self.refresh()
        status = await self.tracker.get_user_status(user.id, user.username)
        return status.rpd_remaining

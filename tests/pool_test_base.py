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
    Base, PoolMembershipInterval, RequestPool, RequestPoolMember, RequestUsage, User,
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
        # The alias resolver bound its own reference at import; profile endpoints
        # reload it, and must read this database rather than the real one.
        import app.model_alias as model_alias_module
        self._real_alias_session = model_alias_module.AsyncSessionLocal
        model_alias_module.AsyncSessionLocal = self._session_factory
        self._saved_aliases = (
            model_alias_module.model_alias_resolver._snapshot,
            model_alias_module.model_alias_resolver._profile_snapshots,
            model_alias_module.model_alias_resolver._fallback_snapshot,
            model_alias_module.model_alias_resolver._fallback_profile_snapshots,
        )

        # Provider renames and removals reload the outage registry from this database.
        from app.outages import outage_registry
        self._saved_outages = outage_registry._state

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

        # Limits live on profiles. A test asking for a per-user RPD gets a profile per
        # distinct value; group limits are written into every profile, so they behave
        # like the group-wide defaults these tests were written against.
        self._limit_profiles = {}   # rpd_limit -> profile id
        self._group_limits = []     # (group_id, rpm, rpd)

    async def asyncTearDown(self):
        from app.auth import database as auth_database
        auth_database.AsyncSessionLocal = self._real_session_local
        import app.model_alias as model_alias_module
        model_alias_module.AsyncSessionLocal = self._real_alias_session
        (model_alias_module.model_alias_resolver._snapshot,
         model_alias_module.model_alias_resolver._profile_snapshots,
         model_alias_module.model_alias_resolver._fallback_snapshot,
         model_alias_module.model_alias_resolver._fallback_profile_snapshots) = self._saved_aliases
        from app.outages import outage_registry
        outage_registry._state = self._saved_outages
        time_utils.local_today = self._real_local_today
        self.tracker.set_db_session_factory(self._saved_factory)
        self.tracker._user_to_pool = {}
        self.tracker._pool_members = {}
        self.tracker._identity_to_pool = {}
        self.tracker._inactive_members = set()
        self.tracker._pool_locks = {}
        self.tracker._carries = {}
        self.tracker._carries_date = None
        self.tracker._pool_share_default = None
        self.tracker._pool_share_override = {}
        # A switched-off group blocks its models in the global gate; never let one
        # outlive its test.
        self.tracker._groups = {}
        self.tracker._model_to_group = {}
        self.tracker._provider_to_group = {}
        from app.profiles import profile_registry
        profile_registry.reset()
        # User ids repeat across tests, so per-minute counts must not carry over.
        self.tracker._minute_buckets = {}
        self.tracker._group_minute_buckets = {}
        self.tracker._inflight = {}
        self.tracker._rpd_cache = {}
        self.tracker._group_rpd_cache = {}
        self.request_tracker._usage_buffer.clear()
        self.request_tracker._usage_buffer.update(self._saved_buffer)
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- seeding ---------------------------------------------------------

    async def make_user(self, username, rpd_limit=None):
        """Create a user, optionally in a profile whose overall RPD is `rpd_limit`."""
        from app.auth.models import DEFAULT_PROFILE_ID

        profile_id = DEFAULT_PROFILE_ID
        if rpd_limit is not None:
            profile_id = await self.profile_with_rpd(rpd_limit)
        user = User(
            username=username, email=f"{username}@example.test",
            hashed_password="x", is_active=True, profile_id=profile_id,
        )
        self.db.add(user)
        await self.db.commit()
        return user

    async def profile_with_rpd(self, rpd_limit):
        """A profile (shared per distinct value) whose overall RPD is `rpd_limit`."""
        from app.auth.models import UserProfile

        if rpd_limit not in self._limit_profiles:
            profile = UserProfile(name=f"rpd-{rpd_limit}", rpd_limit=rpd_limit)
            self.db.add(profile)
            await self.db.flush()
            for gid, rpm, rpd in self._group_limits:
                self._add_group_limit(profile.id, gid, rpm, rpd)
            await self.db.commit()
            self._limit_profiles[rpd_limit] = profile.id
        return self._limit_profiles[rpd_limit]

    async def make_profile(self, name, **limits):
        """A profile with explicit limits, carrying every group limit set so far."""
        from app.auth.models import UserProfile

        profile = UserProfile(name=name, **limits)
        self.db.add(profile)
        await self.db.flush()
        for gid, rpm, rpd in self._group_limits:
            self._add_group_limit(profile.id, gid, rpm, rpd)
        await self.db.commit()
        await self.refresh()
        return profile

    async def set_profile(self, user, profile_id):
        user.profile_id = profile_id
        await self.db.commit()
        await self.refresh()

    async def set_group_limit(self, profile_id, group, *, rpm=None, rpd=None):
        """Give one profile its own limit on a group, replacing the shared one."""
        from sqlalchemy import delete as sa_delete
        from app.auth.models import ProfileModelGroupLimit

        model = ProfileModelGroupLimit
        await self.db.execute(sa_delete(model).where(
            model.profile_id == profile_id, model.group_id == group.id,
        ))
        self._add_group_limit(profile_id, group.id, rpm, rpd)
        await self.db.commit()
        await self.refresh()

    def _add_group_limit(self, profile_id, group_id, rpm, rpd):
        from app.auth.models import ProfileModelGroupLimit

        if rpm is None and rpd is None:
            return
        self.db.add(ProfileModelGroupLimit(
            profile_id=profile_id, group_id=group_id, rpm_limit=rpm, rpd_limit=rpd,
        ))

    async def _limit_group_in_every_profile(self, group_id, rpm, rpd):
        from sqlalchemy import select
        from app.auth.models import UserProfile

        self._group_limits.append((group_id, rpm, rpd))
        for pid in (await self.db.execute(select(UserProfile.id))).scalars().all():
            self._add_group_limit(pid, group_id, rpm, rpd)

    async def user_id_of(self, username):
        from sqlalchemy import select
        return (await self.db.execute(
            select(User.id).where(User.username == username)
        )).scalar_one()

    async def seed_usage(self, username, count, *, model="p/m", server="openai", day=DAY,
                         pool_id=None):
        """Record `count` requests for a user, the way the tracker would have.

        Rows are keyed by user_id and stamped with the pool the sender is in *now*
        (0 when unpooled), exactly as end_request stamps a completed request. Pass
        `pool_id` explicitly to seed a row from before a join (0) or from a stint in a
        pool the user has since left.
        """
        from sqlalchemy import select

        uid = await self.user_id_of(username)
        if pool_id is None:
            pool_id = (await self.db.execute(
                select(RequestPoolMember.pool_id).where(RequestPoolMember.user_id == uid)
            )).scalar_one_or_none() or 0
        self.db.add(RequestUsage(
            date=day, user_id=uid, pool_id=pool_id, user_identity=username,
            user_type="user", model=model, server=server, request_count=count,
        ))
        await self.db.commit()

    async def make_pool(self, name, owner, members=(), *, joined_on=DAY):
        """Create a pool with an owner and, optionally, extra members already in it.

        Bypasses the join endpoint deliberately: most settlement tests want a given
        starting composition, not a history of joins. The membership intervals pool
        *usage* reads are opened here too, because a member with no interval has no
        usage as far as those views are concerned — `joined_on` moves that start date
        back for tests that care about pre-join traffic.
        """
        pool = RequestPool(name=name, owner_user_id=owner.id)
        self.db.add(pool)
        await self.db.flush()
        for user in (owner,) + tuple(members):
            self.db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
            self.db.add(PoolMembershipInterval(
                pool_id=pool.id, user_id=user.id, joined_on=joined_on,
            ))
        await self.db.commit()
        await self.refresh()
        return pool

    async def join_pool(self, pool, user, *, joined_on=DAY):
        """Add one member to an existing pool, with their own stint start."""
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
        self.db.add(PoolMembershipInterval(
            pool_id=pool.id, user_id=user.id, joined_on=joined_on,
        ))
        await self.db.commit()
        await self.refresh()

    async def leave_pool_on(self, pool, user, left_on=DAY):
        """Close a member's stint and drop their membership row, as a departure does."""
        from sqlalchemy import delete as sa_delete, update as sa_update

        await self.db.execute(
            sa_update(PoolMembershipInterval)
            .where(
                PoolMembershipInterval.pool_id == pool.id,
                PoolMembershipInterval.user_id == user.id,
                PoolMembershipInterval.left_on.is_(None),
            )
            .values(left_on=left_on)
        )
        await self.db.execute(
            sa_delete(RequestPoolMember).where(
                RequestPoolMember.pool_id == pool.id,
                RequestPoolMember.user_id == user.id,
            )
        )
        await self.db.commit()
        await self.refresh()

    async def make_model_group(self, name, model_id, rpd_default):
        """A model group containing one model, with the tracker refreshed onto it."""
        from app.auth.models import ModelGroup, ModelGroupMember

        group = ModelGroup(name=name)
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupMember(group_id=group.id, model_id=model_id))
        await self._limit_group_in_every_profile(group.id, None, rpd_default)
        await self.db.commit()
        await self.refresh()
        return group

    async def make_instance_group(self, name, provider_key, rpd_default):
        """A model group whose one member is a whole provider instance, with the
        tracker refreshed."""
        from app.auth.models import ModelGroup, ModelGroupInstanceMember

        group = ModelGroup(name=name)
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupInstanceMember(group_id=group.id, provider_key=provider_key))
        await self._limit_group_in_every_profile(group.id, None, rpd_default)
        await self.db.commit()
        await self.refresh()
        return group

    async def deactivate(self, user):
        """Flip a user's is_active off and let the tracker see it."""
        user.is_active = False
        await self.db.commit()
        await self.refresh()

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

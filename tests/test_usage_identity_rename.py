"""Tests for what a username change does to request usage.

Usage rows are keyed by user_id, so a rename moves nothing and can never collide or
reset a quota. What it must do is relabel the rows -- and the counts still buffered
in the tracker -- so every usage view reads the current name.
"""

import asyncio
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.auth import database
from app.auth.database import rename_usage_identity, update_user_profile
from app.auth.models import (
    ADMIN_USAGE_USER_ID, Base, RequestUsage, RequestUsageHourly, RequestUsageMonthly, User,
)
from app.request_tracker import _K_IDENTITY, ActiveRequest, RequestTracker

DAY = date(2026, 3, 4)


class UsageDBTestCase(unittest.IsolatedAsyncioTestCase):
    """Base: a throwaway SQLite database plus helpers for seeding usage rows.

    Seeding is by identity label for readability; the user_id the rows are keyed by is
    the matching users row when one exists, else a synthetic id handed out in first-seen
    order starting at 1 (so a test with no users rows gets alice == 1).
    """

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
        self._synthetic_ids: dict = {}

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    async def uid(self, identity) -> int:
        """The user_id usage rows for this identity are keyed by."""
        real = (await self.db.execute(
            select(User.id).where(User.username == identity)
        )).scalar_one_or_none()
        if real is not None:
            return real
        if identity not in self._synthetic_ids:
            self._synthetic_ids[identity] = len(self._synthetic_ids) + 1
        return self._synthetic_ids[identity]

    async def _seed(self, identity, count, *, model="p/m", server="openai",
                    day=DAY, hour=9, user_type="user", user_id=None, pool_id=0):
        """Add one row per usage table for this identity."""
        uid = user_id if user_id is not None else await self.uid(identity)
        common = dict(
            user_id=uid, pool_id=pool_id, user_identity=identity, user_type=user_type,
            model=model, server=server, request_count=count,
        )
        self.db.add_all([
            RequestUsage(date=day, **common),
            RequestUsageHourly(date=day, hour=hour, **common),
            RequestUsageMonthly(year=day.year, month=day.month, **common),
        ])
        await self.db.commit()
        return uid

    async def _rows(self, table, identity):
        """Return (row_count, summed request_count) for an identity label."""
        result = (await self.db.execute(
            select(func.count(table.id), func.sum(table.request_count))
            .where(table.user_identity == identity)
        )).one()
        return result[0], result[1] or 0

    async def _rows_by_id(self, table, user_id):
        result = (await self.db.execute(
            select(func.count(table.id), func.sum(table.request_count))
            .where(table.user_id == user_id)
        )).one()
        return result[0], result[1] or 0

    async def _assert_moved(self, old, new, expected_total):
        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, old), (0, 0))
                self.assertEqual(await self._rows(table, new), (1, expected_total))


class UsageRenameDBTests(UsageDBTestCase):
    """Exercises rename_usage_identity against a real (temporary) database."""

    async def test_rename_relabels_every_table(self):
        uid = await self._seed("old.name", 7)

        relabelled = await rename_usage_identity(self.db, uid, "new.name")
        await self.db.commit()

        await self._assert_moved("old.name", "new.name", 7)
        self.assertEqual(
            relabelled,
            {"request_usage": 1, "request_usage_hourly": 1, "request_usage_monthly": 1},
        )

    async def test_rows_are_relabelled_not_moved(self):
        """Keyed by user_id: the same rows, same count, new label -- nothing merges."""
        uid = await self._seed("old.name", 7, model="p/shared")
        await self._seed("old.name", 3, model="p/only-old", user_id=uid)

        await rename_usage_identity(self.db, uid, "new.name")
        await self.db.commit()

        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "old.name"), (0, 0))
                self.assertEqual(await self._rows(table, "new.name"), (2, 10))
                self.assertEqual(await self._rows_by_id(table, uid), (2, 10))

    async def test_other_users_are_untouched(self):
        uid = await self._seed("old.name", 7)
        await self._seed("bystander", 4)

        await rename_usage_identity(self.db, uid, "new.name")
        await self.db.commit()

        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "bystander"), (1, 4))

    async def test_rename_to_the_current_label_is_a_noop(self):
        uid = await self._seed("same.name", 7)

        relabelled = await rename_usage_identity(self.db, uid, "same.name")
        await self.db.commit()

        self.assertEqual(relabelled, {})
        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "same.name"), (1, 7))

    async def test_rename_with_no_usage_rows(self):
        relabelled = await rename_usage_identity(self.db, 4242, "new.name")
        self.assertEqual(relabelled, {})

    async def test_daily_quota_is_unaffected_by_a_rename(self):
        """RPD is a SUM over today's rows by user_id; the label is irrelevant to it."""
        today = time_utils.local_today()
        uid = await self._seed("old.name", 40, day=today)

        tracker = RequestTracker()
        # get_today_count resolves the session factory at call time.
        with patch("app.auth.database.AsyncSessionLocal", self._session_factory):
            self.assertEqual(await tracker.get_today_count(uid), 40)

            await rename_usage_identity(self.db, uid, "new.name")
            await self.db.commit()

            self.assertEqual(await tracker.get_today_count(uid), 40)
        self.assertEqual(await self._rows(RequestUsage, "new.name"), (1, 40))


class UpdateUserProfileTests(UsageDBTestCase):
    """The rename chokepoint both the admin and self-service paths go through."""

    async def _add_user(self, username, email="u@example.com"):
        user = User(username=username, email=email, hashed_password="x")
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def test_renaming_a_user_relabels_their_usage(self):
        user = await self._add_user("old.name")
        await self._seed("old.name", 12)

        updated = await update_user_profile(self.db, user.id, username="new.name")

        self.assertEqual(updated.username, "new.name")
        await self._assert_moved("old.name", "new.name", 12)
        self.assertEqual(await self._rows_by_id(RequestUsage, user.id), (1, 12))

    async def test_rename_drops_the_stale_rpd_cache_entry(self):
        from app.rate_limit import rate_limit_tracker, _RpdCacheEntry, _user_scope_key

        user = await self._add_user("old.name")
        rate_limit_tracker._rpd_cache[_user_scope_key("old.name")] = _RpdCacheEntry(count=9, expires_at=1e12)
        try:
            await update_user_profile(self.db, user.id, username="new.name")
            self.assertNotIn(_user_scope_key("old.name"), rate_limit_tracker._rpd_cache)
            self.assertNotIn(_user_scope_key("new.name"), rate_limit_tracker._rpd_cache)
        finally:
            rate_limit_tracker._rpd_cache.pop(_user_scope_key("old.name"), None)

    async def test_rename_drops_the_owners_cached_api_keys(self):
        """API-key traffic is labelled with the username cached on the key.

        The 30s validity refresh only re-checks is_active, so a stale entry would
        keep writing the old label for the length of the key TTL.
        """
        from app.auth.cache import auth_cache, CachedAPIKey

        user = await self._add_user("old.name")
        other = await self._add_user("bystander", email="b@example.com")
        auth_cache._api_key_cache["sk-mine"] = CachedAPIKey(
            id=1, user_id=user.id, api_key="sk-mine", name="mine",
            is_active=True, username="old.name",
        )
        auth_cache._api_key_cache["sk-theirs"] = CachedAPIKey(
            id=2, user_id=other.id, api_key="sk-theirs", name="theirs",
            is_active=True, username="bystander",
        )
        try:
            await update_user_profile(self.db, user.id, username="new.name")

            self.assertNotIn("sk-mine", auth_cache._api_key_cache)
            # Only the renamed user's keys are dropped.
            self.assertIn("sk-theirs", auth_cache._api_key_cache)
        finally:
            auth_cache._api_key_cache.pop("sk-mine", None)
            auth_cache._api_key_cache.pop("sk-theirs", None)

    async def test_updating_only_the_email_leaves_usage_alone(self):
        user = await self._add_user("old.name")
        await self._seed("old.name", 12)

        await update_user_profile(self.db, user.id, email="new@example.com")

        self.assertEqual(await self._rows(RequestUsage, "old.name"), (1, 12))

    async def test_taking_another_users_name_is_rejected(self):
        await self._add_user("taken", email="taken@example.com")
        user = await self._add_user("old.name")
        await self._seed("old.name", 12)

        with self.assertRaises(ValueError):
            await update_user_profile(self.db, user.id, username="taken")

        # Neither the name nor the label changed.
        await self.db.refresh(user)
        self.assertEqual(user.username, "old.name")
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (1, 12))

    async def test_taking_the_admin_username_is_rejected(self):
        """The admin is config-only, so the users-table check cannot see it.

        Admin traffic is recorded under ADMIN_USAGE_USER_ID with the admin name as its
        label. Two accounts sharing the name would make every by-name lookup (the
        admin usage drill-down, the delete-by-user route) ambiguous.
        """
        user = await self._add_user("old.name")
        await self._seed("root", 500, user_id=ADMIN_USAGE_USER_ID, user_type="admin")
        await self._seed("old.name", 12)

        with patch("app.auth.admin.is_admin_enabled", return_value=True), \
             patch("app.auth.admin.get_admin_username", return_value="root"):
            with self.assertRaises(ValueError):
                await update_user_profile(self.db, user.id, username="root")

        await self.db.refresh(user)
        self.assertEqual(user.username, "old.name")
        self.assertEqual(await self._rows(RequestUsage, "root"), (1, 500))
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (1, 12))

    async def test_a_flush_mid_rename_cannot_leave_the_old_label_behind(self):
        """Renaming has to exclude the flush, which is not atomic.

        A flush snapshots the buffer, writes it, then subtracts. One caught in that
        window writes its snapshot under the old label after the SQL relabel ran, and
        the in-memory relabel then misses the counts it already moved.
        """
        from app.request_tracker import request_tracker

        user = await self._add_user("old.name")
        key = (DAY, 9, user.id, "old.name", "user", "p/m", "openai", 0)
        request_tracker._usage_buffer[key] = 4

        real_flush = database.flush_usage_rows
        writes = 0

        async def slow_write(hourly_rows, daily_rows):
            # Stall only the first write, leaving it outstanding across the rename.
            nonlocal writes
            writes += 1
            if writes == 1:
                await asyncio.sleep(0.3)
            await real_flush(hourly_rows, daily_rows)

        async def noop(*args, **kwargs):
            return None

        try:
            with patch("app.auth.database.AsyncSessionLocal", self._session_factory), \
                 patch("app.auth.database.flush_usage_rows", slow_write), \
                 patch("app.auth.database.prune_hourly_usage", noop), \
                 patch("app.auth.database.rollup_to_monthly", noop):
                racing = asyncio.create_task(request_tracker.flush_pending())
                await asyncio.sleep(0.01)  # let it reach the DB write
                await update_user_profile(self.db, user.id, username="new.name")
                await racing
        finally:
            request_tracker._usage_buffer.clear()

        # Counted exactly once, and only under the new label.
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (0, 0))
        self.assertEqual(await self._rows(RequestUsage, "new.name"), (1, 4))
        self.assertEqual(await self._rows_by_id(RequestUsage, user.id), (1, 4))


class TrackerRenameIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Counts that have not reached the database yet must be relabelled too."""

    def _key(self, identity, *, user_id=1, model="p/m", hour=9, pool_id=0):
        # (date, hour, user_id, user_identity, user_type, model, server, pool_id)
        return (DAY, hour, user_id, identity, "user", model, "openai", pool_id)

    async def test_buffered_counts_follow_the_rename(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("old.name")] = 3
        tracker._usage_buffer[self._key("old.name", model="p/other")] = 2

        await tracker.rename_identity(1, "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("new.name")], 3)
        self.assertEqual(tracker._usage_buffer[self._key("new.name", model="p/other")], 2)
        self.assertNotIn(self._key("old.name"), tracker._usage_buffer)

    async def test_buffered_counts_sum_into_an_existing_key(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("old.name")] = 3
        tracker._usage_buffer[self._key("new.name")] = 5

        await tracker.rename_identity(1, "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("new.name")], 8)
        self.assertNotIn(self._key("old.name"), tracker._usage_buffer)

    async def test_other_users_are_left_alone(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("bystander", user_id=2)] = 4

        await tracker.rename_identity(1, "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("bystander", user_id=2)], 4)

    async def test_in_flight_requests_are_relabelled(self):
        tracker = RequestTracker()
        tracker._active["req-1"] = ActiveRequest(
            request_id="req-1", server="openai", endpoint="/v1/chat/completions",
            method="POST", model="p/m", user_identity="old.name", user_type="user",
            is_streaming=True, start_time=0.0, user_id=1,
        )
        tracker._active["req-2"] = ActiveRequest(
            request_id="req-2", server="openai", endpoint="/v1/chat/completions",
            method="POST", model="p/m", user_identity="bystander", user_type="user",
            is_streaming=False, start_time=0.0, user_id=2,
        )

        await tracker.rename_identity(1, "new.name")

        self.assertEqual(tracker._active["req-1"].user_identity, "new.name")
        self.assertEqual(tracker._active["req-2"].user_identity, "bystander")

    async def test_rename_to_same_name_is_a_noop(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("same.name")] = 3

        await tracker.rename_identity(1, "same.name")

        self.assertEqual(tracker._usage_buffer[self._key("same.name")], 3)

    async def test_a_completion_racing_the_rename_cannot_buffer_the_old_label(self):
        """end_request pops from _active and then buffers the count. If those are two
        steps, the request is momentarily in neither structure -- and those two are
        exactly what this method sweeps, so a rename landing in the gap relabels
        nothing. The old label would then ride the next flush's on-conflict update
        straight over the row rename_usage_identity had just relabelled.

        Both are pinned behind _lock to make the interleaving deterministic: whichever
        order they run in, nothing may be left buffered under the old name.
        """
        tracker = RequestTracker()
        tracker._broadcast_raw = AsyncMock()
        tracker._active["req-1"] = ActiveRequest(
            request_id="req-1", server="openai", endpoint="/v1/chat/completions",
            method="POST", model="p/m", user_identity="old.name", user_type="user",
            is_streaming=False, start_time=0.0, user_id=1,
        )

        await tracker._lock.acquire()
        completion = asyncio.create_task(tracker.end_request("req-1", status="completed"))
        rename = asyncio.create_task(tracker.rename_identity(1, "new.name"))
        # Let both reach the point where they need _lock before handing it over.
        for _ in range(10):
            await asyncio.sleep(0)
        tracker._lock.release()
        await asyncio.gather(completion, rename)

        self.assertEqual(
            [key[_K_IDENTITY] for key in tracker._usage_buffer],
            ["new.name"],
        )


class RateLimitInvalidateIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_rpd_caches_are_dropped_for_the_identity(self):
        from app.rate_limit import RateLimitTracker, _RpdCacheEntry, _user_scope_key

        tracker = RateLimitTracker()
        entry = _RpdCacheEntry(count=5, expires_at=1e12)
        old, bystander = _user_scope_key("old.name"), _user_scope_key("bystander")
        tracker._rpd_cache[old] = entry
        tracker._rpd_cache[bystander] = entry
        tracker._group_rpd_cache[(old, 1)] = entry
        tracker._group_rpd_cache[(bystander, 1)] = entry
        tracker._instance_group_rpd_cache[(old, 2)] = entry

        tracker.invalidate_identity("old.name")

        self.assertNotIn(old, tracker._rpd_cache)
        self.assertNotIn((old, 1), tracker._group_rpd_cache)
        self.assertNotIn((old, 2), tracker._instance_group_rpd_cache)
        self.assertIn(bystander, tracker._rpd_cache)
        self.assertIn((bystander, 1), tracker._group_rpd_cache)


if __name__ == "__main__":
    unittest.main()

"""Tests for moving request usage when a username changes.

Usage rows are keyed by the username string rather than user_id, so a rename has
to carry them along — otherwise the history is orphaned and the user's daily
quota silently resets.
"""

import asyncio
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.auth import database
from app.auth.database import rename_usage_identity, update_user_profile
from app.auth.models import (
    Base, RequestUsage, RequestUsageHourly, RequestUsageMonthly, User,
)
from app.request_tracker import ActiveRequest, RequestTracker

DAY = date(2026, 3, 4)


class UsageDBTestCase(unittest.IsolatedAsyncioTestCase):
    """Base: a throwaway SQLite database plus helpers for seeding usage rows."""

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

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    async def _seed(self, identity, count, *, model="p/m", server="openai",
                    day=DAY, hour=9, user_type="user"):
        """Add one row per usage table for this identity."""
        self.db.add_all([
            RequestUsage(
                date=day, user_identity=identity, user_type=user_type,
                model=model, server=server, request_count=count,
            ),
            RequestUsageHourly(
                date=day, hour=hour, user_identity=identity, user_type=user_type,
                model=model, server=server, request_count=count,
            ),
            RequestUsageMonthly(
                year=day.year, month=day.month, user_identity=identity,
                user_type=user_type, model=model, server=server, request_count=count,
            ),
        ])
        await self.db.commit()

    async def _rows(self, table, identity):
        """Return (row_count, summed request_count) for an identity."""
        result = (await self.db.execute(
            select(func.count(table.id), func.sum(table.request_count))
            .where(table.user_identity == identity)
        )).one()
        return result[0], result[1] or 0

    async def _assert_moved(self, old, new, expected_total):
        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, old), (0, 0))
                self.assertEqual(await self._rows(table, new), (1, expected_total))


class UsageRenameDBTests(UsageDBTestCase):
    """Exercises rename_usage_identity against a real (temporary) database."""

    async def test_clean_rename_moves_every_table(self):
        await self._seed("old.name", 7)

        moved = await rename_usage_identity(self.db, "old.name", "new.name")
        await self.db.commit()

        await self._assert_moved("old.name", "new.name", 7)
        self.assertEqual(
            moved,
            {"request_usage": 1, "request_usage_hourly": 1, "request_usage_monthly": 1},
        )

    async def test_conflicting_rows_are_merged_not_duplicated(self):
        # Same day/hour/month, model and server — every uniqueness key collides.
        await self._seed("old.name", 7)
        await self._seed("new.name", 5)

        await rename_usage_identity(self.db, "old.name", "new.name")
        await self.db.commit()

        # One surviving row per table holding the summed count.
        await self._assert_moved("old.name", "new.name", 12)

    async def test_rows_differing_only_by_time_key_do_not_merge(self):
        # Hourly rows are unique per (date, hour, ...): a different hour must stay
        # a separate row rather than being folded in.
        await self._seed("old.name", 7, hour=9)
        await self._seed("new.name", 5, hour=10, day=DAY, model="p/m")

        await rename_usage_identity(self.db, "old.name", "new.name")
        await self.db.commit()

        # Daily and monthly keys still collide, so those merge to one row.
        self.assertEqual(await self._rows(RequestUsage, "new.name"), (1, 12))
        self.assertEqual(await self._rows(RequestUsageMonthly, "new.name"), (1, 12))
        # Hourly keeps both hours.
        self.assertEqual(await self._rows(RequestUsageHourly, "new.name"), (2, 12))
        self.assertEqual(await self._rows(RequestUsageHourly, "old.name"), (0, 0))

    async def test_mixed_conflicting_and_clean_rows(self):
        await self._seed("old.name", 7, model="p/shared")
        await self._seed("old.name", 3, model="p/only-old")
        await self._seed("new.name", 5, model="p/shared")

        await rename_usage_identity(self.db, "old.name", "new.name")
        await self.db.commit()

        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "old.name"), (0, 0))
                # p/shared merged to 12, p/only-old carried over at 3.
                self.assertEqual(await self._rows(table, "new.name"), (2, 15))

    async def test_other_identities_are_untouched(self):
        await self._seed("old.name", 7)
        await self._seed("bystander", 4)

        await rename_usage_identity(self.db, "old.name", "new.name")
        await self.db.commit()

        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "bystander"), (1, 4))

    async def test_rename_to_same_name_is_a_noop(self):
        await self._seed("same.name", 7)

        moved = await rename_usage_identity(self.db, "same.name", "same.name")
        await self.db.commit()

        self.assertEqual(moved, {})
        for table in (RequestUsage, RequestUsageHourly, RequestUsageMonthly):
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "same.name"), (1, 7))

    async def test_rename_with_no_usage_rows(self):
        moved = await rename_usage_identity(self.db, "ghost", "new.name")
        self.assertEqual(moved, {})

    async def test_daily_quota_survives_the_rename(self):
        """The bug this all exists for: RPD is a COUNT over the current username."""
        today = time_utils.local_today()
        await self._seed("old.name", 40, day=today)

        tracker = RequestTracker()
        # get_today_count resolves the session factory at call time.
        with patch("app.auth.database.AsyncSessionLocal", self._session_factory):
            self.assertEqual(await tracker.get_today_count("old.name"), 40)
            self.assertEqual(await tracker.get_today_count("new.name"), 0)

            await rename_usage_identity(self.db, "old.name", "new.name")
            await self.db.commit()

            # The quota follows the user instead of resetting.
            self.assertEqual(await tracker.get_today_count("new.name"), 40)
            self.assertEqual(await tracker.get_today_count("old.name"), 0)


class UpdateUserProfileTests(UsageDBTestCase):
    """The rename chokepoint both the admin and self-service paths go through."""

    async def _add_user(self, username, email="u@example.com"):
        user = User(username=username, email=email, hashed_password="x")
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def test_renaming_a_user_moves_their_usage(self):
        user = await self._add_user("old.name")
        await self._seed("old.name", 12)

        updated = await update_user_profile(self.db, user.id, username="new.name")

        self.assertEqual(updated.username, "new.name")
        await self._assert_moved("old.name", "new.name", 12)

    async def test_rename_drops_the_stale_rpd_cache_entry(self):
        from app.rate_limit import rate_limit_tracker, _RpdCacheEntry

        user = await self._add_user("old.name")
        rate_limit_tracker._rpd_cache["old.name"] = _RpdCacheEntry(count=9, expires_at=1e12)
        try:
            await update_user_profile(self.db, user.id, username="new.name")
            self.assertNotIn("old.name", rate_limit_tracker._rpd_cache)
            self.assertNotIn("new.name", rate_limit_tracker._rpd_cache)
        finally:
            rate_limit_tracker._rpd_cache.pop("old.name", None)

    async def test_rename_drops_the_owners_cached_api_keys(self):
        """API-key traffic is recorded under the username cached on the key.

        The 30s validity refresh only re-checks is_active, so a stale entry would
        keep writing the old name for the length of the key TTL.
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

        # Neither the name nor the usage moved.
        await self.db.refresh(user)
        self.assertEqual(user.username, "old.name")
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (1, 12))

    async def test_taking_the_admin_username_is_rejected(self):
        """The admin is config-only, so the users-table check cannot see it.

        Admin traffic is still recorded under that name, and rename_usage_identity
        merges the source rows into the target and deletes them — so without this
        guard a self-service rename absorbs the admin's whole history and quota.
        """
        user = await self._add_user("old.name")
        await self._seed("root", 500)      # the admin's accumulated usage
        await self._seed("old.name", 12)

        with patch("app.auth.admin.is_admin_enabled", return_value=True), \
             patch("app.auth.admin.get_admin_username", return_value="root"):
            with self.assertRaises(ValueError):
                await update_user_profile(self.db, user.id, username="root")

        await self.db.refresh(user)
        self.assertEqual(user.username, "old.name")
        self.assertEqual(await self._rows(RequestUsage, "root"), (1, 500))
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (1, 12))

    async def test_a_flush_mid_rename_cannot_resurrect_the_old_identity(self):
        """Renaming has to exclude the flush, which is not atomic.

        A flush snapshots the buffer, writes it, then subtracts. One caught in that
        window re-writes its snapshot under the old name after the SQL has moved
        the rows away, and the pre-rename drain re-reads the same un-subtracted
        counts — billing those requests to both names.
        """
        from app.request_tracker import request_tracker

        user = await self._add_user("old.name")
        key = (DAY, 9, "old.name", "user", "p/m", "openai")
        request_tracker._usage_buffer[key] = 4

        real_daily = database.flush_request_usage
        writes = 0

        async def slow_daily(rows):
            # Stall only the first write, leaving it outstanding across the rename.
            nonlocal writes
            writes += 1
            if writes == 1:
                await asyncio.sleep(0.3)
            await real_daily(rows)

        async def noop(*args, **kwargs):
            return None

        try:
            with patch("app.auth.database.AsyncSessionLocal", self._session_factory), \
                 patch("app.auth.database.flush_request_usage", slow_daily), \
                 patch("app.auth.database.prune_hourly_usage", noop), \
                 patch("app.auth.database.rollup_to_monthly", noop):
                racing = asyncio.create_task(request_tracker.flush_pending())
                await asyncio.sleep(0.01)  # let it reach the DB write
                await update_user_profile(self.db, user.id, username="new.name")
                await racing
        finally:
            request_tracker._usage_buffer.clear()

        # Counted exactly once, and only under the new name.
        self.assertEqual(await self._rows(RequestUsage, "old.name"), (0, 0))
        self.assertEqual(await self._rows(RequestUsage, "new.name"), (1, 4))


class TrackerRenameIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Counts that have not reached the database yet must move too."""

    def _key(self, identity, *, model="p/m", hour=9):
        # (date, hour, user_identity, user_type, model, server)
        return (DAY, hour, identity, "user", model, "openai")

    async def test_buffered_counts_follow_the_rename(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("old.name")] = 3
        tracker._usage_buffer[self._key("old.name", model="p/other")] = 2

        await tracker.rename_identity("old.name", "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("new.name")], 3)
        self.assertEqual(tracker._usage_buffer[self._key("new.name", model="p/other")], 2)
        self.assertNotIn(self._key("old.name"), tracker._usage_buffer)

    async def test_buffered_counts_sum_into_an_existing_key(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("old.name")] = 3
        tracker._usage_buffer[self._key("new.name")] = 5

        await tracker.rename_identity("old.name", "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("new.name")], 8)
        self.assertNotIn(self._key("old.name"), tracker._usage_buffer)

    async def test_other_identities_are_left_alone(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("bystander")] = 4

        await tracker.rename_identity("old.name", "new.name")

        self.assertEqual(tracker._usage_buffer[self._key("bystander")], 4)

    async def test_in_flight_requests_are_reassigned(self):
        tracker = RequestTracker()
        tracker._active["req-1"] = ActiveRequest(
            request_id="req-1", server="openai", endpoint="/v1/chat/completions",
            method="POST", model="p/m", user_identity="old.name", user_type="user",
            is_streaming=True, start_time=0.0,
        )
        tracker._active["req-2"] = ActiveRequest(
            request_id="req-2", server="openai", endpoint="/v1/chat/completions",
            method="POST", model="p/m", user_identity="bystander", user_type="user",
            is_streaming=False, start_time=0.0,
        )

        await tracker.rename_identity("old.name", "new.name")

        self.assertEqual(tracker._active["req-1"].user_identity, "new.name")
        self.assertEqual(tracker._active["req-2"].user_identity, "bystander")

    async def test_rename_to_same_name_is_a_noop(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("same.name")] = 3

        await tracker.rename_identity("same.name", "same.name")

        self.assertEqual(tracker._usage_buffer[self._key("same.name")], 3)


class RateLimitInvalidateIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_rpd_caches_are_dropped_for_the_identity(self):
        from app.rate_limit import RateLimitTracker, _RpdCacheEntry

        tracker = RateLimitTracker()
        entry = _RpdCacheEntry(count=5, expires_at=1e12)
        tracker._rpd_cache["old.name"] = entry
        tracker._rpd_cache["bystander"] = entry
        tracker._group_rpd_cache[("old.name", 1)] = entry
        tracker._group_rpd_cache[("bystander", 1)] = entry
        tracker._instance_group_rpd_cache[("old.name", 2)] = entry

        tracker.invalidate_identity("old.name")

        self.assertNotIn("old.name", tracker._rpd_cache)
        self.assertNotIn(("old.name", 1), tracker._group_rpd_cache)
        self.assertNotIn(("old.name", 2), tracker._instance_group_rpd_cache)
        self.assertIn("bystander", tracker._rpd_cache)
        self.assertIn(("bystander", 1), tracker._group_rpd_cache)


if __name__ == "__main__":
    unittest.main()

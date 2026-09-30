"""Tests for purging a user's, model's or pool's usage data from the admin usage tab.

Usage lives in three tables with different retention (hourly, daily, monthly
rollup), so a purge that misses one makes the data reappear as soon as the admin
widens the time window. These tests pin that down, plus the in-memory buffer that
would otherwise flush the deleted counts straight back in.
"""

import unittest
from datetime import date
from unittest.mock import patch

from sqlalchemy import func, select

from app import time_utils
from app.auth.database import count_usage_requests, delete_usage_records
from app.auth.models import RequestUsage, RequestUsageHourly, RequestUsageMonthly
from app.request_tracker import RequestTracker

from tests.test_usage_identity_rename import DAY, UsageDBTestCase

USAGE_TABLES = (RequestUsage, RequestUsageHourly, RequestUsageMonthly)


class DeleteUsageRecordsTests(UsageDBTestCase):
    """Exercises delete_usage_records against a real (temporary) database."""

    async def _model_rows(self, table, model):
        """Return (row_count, summed request_count) for a model."""
        result = (await self.db.execute(
            select(func.count(table.id), func.sum(table.request_count))
            .where(table.model == model)
        )).one()
        return result[0], result[1] or 0

    async def _pool_rows(self, table, pool_id):
        result = (await self.db.execute(
            select(func.count(table.id), func.sum(table.request_count))
            .where(table.pool_id == pool_id)
        )).one()
        return result[0], result[1] or 0

    async def _total_rows(self, table):
        return (await self.db.execute(select(func.count(table.id)))).scalar_one()

    async def test_delete_by_user_clears_every_table(self):
        uid = await self._seed("alice", 7)

        deleted = await delete_usage_records(self.db, "user", uid)
        await self.db.commit()

        self.assertEqual(
            deleted,
            {"request_usage": 1, "request_usage_hourly": 1, "request_usage_monthly": 1},
        )
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "alice"), (0, 0))

    async def test_delete_by_user_spans_days_models_and_pools(self):
        uid = await self._seed("alice", 7, model="p/one")
        await self._seed("alice", 3, model="p/two")
        await self._seed("alice", 5, day=date(2025, 11, 2), hour=4)
        await self._seed("alice", 2, model="p/one", pool_id=9)

        deleted = await delete_usage_records(self.db, "user", uid)
        await self.db.commit()

        # Four seeds, four rows per table -- the monthly rows differ by month or pool.
        self.assertEqual(deleted["request_usage"], 4)
        self.assertEqual(deleted["request_usage_hourly"], 4)
        self.assertEqual(deleted["request_usage_monthly"], 4)
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "alice"), (0, 0))

    async def test_delete_by_user_leaves_other_users_alone(self):
        uid = await self._seed("alice", 7)
        await self._seed("bystander", 4)

        await delete_usage_records(self.db, "user", uid)
        await self.db.commit()

        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "bystander"), (1, 4))

    async def test_delete_by_model_clears_every_table_across_users(self):
        await self._seed("alice", 7, model="p/gone")
        await self._seed("bob", 4, model="p/gone")
        await self._seed("alice", 2, model="p/kept")

        deleted = await delete_usage_records(self.db, "model", "p/gone")
        await self.db.commit()

        self.assertEqual(
            deleted,
            {"request_usage": 2, "request_usage_hourly": 2, "request_usage_monthly": 2},
        )
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._model_rows(table, "p/gone"), (0, 0))
                self.assertEqual(await self._model_rows(table, "p/kept"), (1, 2))

    async def test_delete_by_pool_removes_only_rows_attributed_to_it(self):
        """What its members sent outside the pool -- before joining, in another pool,
        or after leaving -- is theirs and stays."""
        await self._seed("alice", 7, pool_id=3)
        await self._seed("bob", 4, pool_id=3)
        await self._seed("alice", 2, pool_id=0, model="p/before-joining")
        await self._seed("bob", 1, pool_id=8, model="p/other-pool")

        deleted = await delete_usage_records(self.db, "pool", 3)
        await self.db.commit()

        self.assertEqual(
            deleted,
            {"request_usage": 2, "request_usage_hourly": 2, "request_usage_monthly": 2},
        )
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._pool_rows(table, 3), (0, 0))
                self.assertEqual(await self._rows(table, "alice"), (1, 2))
                self.assertEqual(await self._rows(table, "bob"), (1, 1))

    async def test_deleting_an_unknown_value_removes_nothing(self):
        await self._seed("alice", 7)

        deleted = await delete_usage_records(self.db, "user", 4242)
        await self.db.commit()

        self.assertEqual(
            deleted,
            {"request_usage": 0, "request_usage_hourly": 0, "request_usage_monthly": 0},
        )
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._rows(table, "alice"), (1, 7))

    async def test_value_is_bound_not_interpolated(self):
        """A model id is free-form text; it must never reach the SQL as syntax."""
        await self._seed("alice", 7)

        deleted = await delete_usage_records(
            self.db, "model", "p/m'; DROP TABLE request_usage; --"
        )
        await self.db.commit()

        self.assertEqual(set(deleted.values()), {0})
        for table in USAGE_TABLES:
            with self.subTest(table=table.__tablename__):
                self.assertEqual(await self._total_rows(table), 1)

    async def test_an_unknown_axis_is_rejected(self):
        with self.assertRaises(KeyError):
            await delete_usage_records(self.db, "user_identity", "alice")

    async def test_deleting_todays_rows_resets_the_daily_quota(self):
        """RPD is a SUM over today's request_usage rows — the documented side effect."""
        today = time_utils.local_today()
        uid = await self._seed("alice", 40, day=today)

        tracker = RequestTracker()
        # get_today_count resolves the session factory at call time.
        with patch("app.auth.database.AsyncSessionLocal", self._session_factory):
            self.assertEqual(await tracker.get_today_count(uid), 40)

            await delete_usage_records(self.db, "user", uid)
            await self.db.commit()

            self.assertEqual(await tracker.get_today_count(uid), 0)


class CountUsageRequestsTests(UsageDBTestCase):
    """What the delete endpoint reports back, so the toast means something.

    Summing the per-table rowcounts a delete returns counts the same traffic up to
    three times, because hourly/daily/monthly are three representations of it.

    Seeds tables individually rather than via _seed: the real layout is daily +
    its hourly copy of the last ~48h, with monthly holding only aged traffic that
    rollup_to_monthly has already deleted from daily.
    """

    async def _seed_recent(self, identity, count, *, model="p/m", day=DAY, hour=9, pool_id=0):
        """Traffic still in the daily table, mirrored in hourly."""
        uid = await self.uid(identity)
        common = dict(
            user_id=uid, pool_id=pool_id, user_identity=identity, user_type="user",
            model=model, server="openai", request_count=count,
        )
        self.db.add_all([
            RequestUsage(date=day, **common),
            RequestUsageHourly(date=day, hour=hour, **common),
        ])
        await self.db.commit()
        return uid

    async def _seed_aged(self, identity, count, *, model="p/m", year=2024, month=1, pool_id=0):
        """Traffic that has been rolled up: monthly only."""
        uid = await self.uid(identity)
        self.db.add(RequestUsageMonthly(
            year=year, month=month, user_id=uid, pool_id=pool_id,
            user_identity=identity, user_type="user",
            model=model, server="openai", request_count=count,
        ))
        await self.db.commit()
        return uid

    async def test_counts_requests_not_rows(self):
        uid = await self._seed_recent("alice", 7)

        self.assertEqual(await count_usage_requests(self.db, "user", uid), 7)

    async def test_hourly_rows_are_not_double_counted(self):
        # One day, two hours: hourly holds two rows, daily one.
        uid = await self._seed_recent("alice", 10)
        self.db.add(RequestUsageHourly(
            date=DAY, hour=10, user_id=uid, pool_id=0, user_identity="alice",
            user_type="user", model="p/m", server="openai", request_count=4,
        ))
        await self.db.commit()

        self.assertEqual(await count_usage_requests(self.db, "user", uid), 10)

    async def test_daily_and_monthly_are_summed_together(self):
        """The two are disjoint, so aged traffic must still be counted."""
        uid = await self._seed_aged("alice", 90)
        await self._seed_recent("alice", 7)

        self.assertEqual(await count_usage_requests(self.db, "user", uid), 97)

    async def test_counts_by_model_across_users(self):
        await self._seed_recent("alice", 7, model="p/gone")
        await self._seed_recent("bob", 3, model="p/gone")
        await self._seed_recent("alice", 5, model="p/kept")

        self.assertEqual(await count_usage_requests(self.db, "model", "p/gone"), 10)

    async def test_counts_by_pool_across_members_and_months(self):
        await self._seed_recent("alice", 7, pool_id=3)
        await self._seed_aged("bob", 90, pool_id=3)
        await self._seed_recent("alice", 5, pool_id=0, model="p/unpooled")

        self.assertEqual(await count_usage_requests(self.db, "pool", 3), 97)

    async def test_unknown_value_counts_zero(self):
        await self._seed_recent("alice", 7)

        self.assertEqual(await count_usage_requests(self.db, "user", 4242), 0)

    async def test_value_is_bound_not_interpolated(self):
        await self._seed_recent("alice", 7)

        total = await count_usage_requests(
            self.db, "model", "p/m'; DROP TABLE request_usage; --"
        )

        self.assertEqual(total, 0)
        self.assertEqual(await self._rows(RequestUsage, "alice"), (1, 7))


class TrackerDropBufferedUsageTests(unittest.IsolatedAsyncioTestCase):
    """Counts that have not reached the database yet must be dropped too."""

    def _key(self, identity, *, user_id=1, model="p/m", hour=9, pool_id=0):
        # (date, hour, user_id, user_identity, user_type, model, server, pool_id)
        return (DAY, hour, user_id, identity, "user", model, "openai", pool_id)

    async def test_drops_only_the_purged_user(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("alice")] = 3
        tracker._usage_buffer[self._key("alice", model="p/other")] = 2
        tracker._usage_buffer[self._key("bystander", user_id=2)] = 5

        dropped = await tracker.drop_buffered_usage("user", 1)

        self.assertEqual(dropped, 5)
        self.assertEqual(list(tracker._usage_buffer), [self._key("bystander", user_id=2)])

    async def test_drops_only_the_purged_model(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("alice", model="p/gone")] = 3
        tracker._usage_buffer[self._key("bob", user_id=2, model="p/gone")] = 4
        tracker._usage_buffer[self._key("alice", model="p/kept")] = 6

        dropped = await tracker.drop_buffered_usage("model", "p/gone")

        self.assertEqual(dropped, 7)
        self.assertEqual(list(tracker._usage_buffer), [self._key("alice", model="p/kept")])

    async def test_drops_only_the_purged_pool(self):
        tracker = RequestTracker()
        tracker._usage_buffer[self._key("alice", pool_id=3)] = 3
        tracker._usage_buffer[self._key("bob", user_id=2, pool_id=3)] = 4
        tracker._usage_buffer[self._key("alice", pool_id=0)] = 6

        dropped = await tracker.drop_buffered_usage("pool", 3)

        self.assertEqual(dropped, 7)
        self.assertEqual(list(tracker._usage_buffer), [self._key("alice", pool_id=0)])

    async def test_nothing_buffered_is_a_noop(self):
        tracker = RequestTracker()

        self.assertEqual(await tracker.drop_buffered_usage("user", 1), 0)
        self.assertEqual(tracker._usage_buffer, {})


class TrackerForgetUserTests(unittest.IsolatedAsyncioTestCase):
    """A deleted user's in-flight requests must not re-add counts when they end.

    users.id is not AUTOINCREMENT, so a count buffered under the deleted id after the
    purge would be flushed and then inherited by the next account given that id.
    """

    async def _start(self, tracker, request_id, identity, user_id):
        await tracker.start_request(
            request_id, "openai", "/v1/chat/completions", "POST",
            "p/m", identity, "user", True,
        )
        await tracker.update_identity(request_id, identity, "user", user_id=user_id)

    async def test_a_stream_ending_after_the_delete_is_not_counted(self):
        tracker = RequestTracker()
        await self._start(tracker, "r1", "alice", 1)
        await self._start(tracker, "r2", "bob", 2)
        with patch.object(RequestTracker, "_pool_id_for", staticmethod(lambda _uid: 0)):
            await tracker.forget_user(1)
            await tracker.end_request("r1")
            await tracker.end_request("r2")

        self.assertEqual(
            [key[2] for key in tracker._usage_buffer], [2],
            "only the surviving user's request is buffered",
        )
        self.assertEqual(tracker._active, {})

    async def test_already_buffered_counts_are_dropped(self):
        tracker = RequestTracker()
        tracker._usage_buffer[(DAY, 9, 1, "alice", "user", "p/m", "openai", 0)] = 3
        tracker._usage_buffer[(DAY, 9, 2, "bob", "user", "p/m", "openai", 0)] = 4

        self.assertEqual(await tracker.forget_user(1), 3)
        self.assertEqual(
            list(tracker._usage_buffer), [(DAY, 9, 2, "bob", "user", "p/m", "openai", 0)]
        )


if __name__ == "__main__":
    unittest.main()

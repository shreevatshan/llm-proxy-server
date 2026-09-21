"""Tests for the /admin/usage and /auth/usage aggregate reader.

Three things the reader used to get wrong:

  * one person split into several rows, because the reads grouped by user_type as
    well as user_identity and user_type is not part of any usage table's key;
  * a drill-down that ignored filter_model whenever filter_user was set -- which
    /auth/usage always sets -- so ?view=model&id=X returned every model the caller
    had ever used, while the chart beside it honoured the filter;
  * the chart (hourly) and the table (daily) silently disagreeing about "today".
"""

import os
import tempfile
import unittest
from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.auth.database import get_usage_aggregates, get_usage_timeseries
from app.auth.models import Base, RequestUsage, RequestUsageHourly


class AggregateTestCase(unittest.IsolatedAsyncioTestCase):
    """A throwaway DB and a pinned 'today', so the windows land where seeded."""

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.db = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )()

        self._real_today = time_utils.local_today
        self.today = self._real_today()   # a real date: 'today' windows key off it

    async def asyncTearDown(self):
        time_utils.local_today = self._real_today
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def seed(self, identity, count, *, user_type="user", model="p/m",
                   server="openai", day=None):
        self.db.add(RequestUsage(
            date=day or self.today, user_identity=identity, user_type=user_type,
            model=model, server=server, request_count=count,
        ))
        await self.db.commit()

    async def seed_hourly(self, identity, count, *, hour=9, user_type="user",
                          model="p/m", server="openai", day=None):
        self.db.add(RequestUsageHourly(
            date=day or self.today, hour=hour, user_identity=identity,
            user_type=user_type, model=model, server=server, request_count=count,
        ))
        await self.db.commit()


class OnePersonOneRowTests(AggregateTestCase):
    """user_type is a display hint, not an attribution key."""

    async def test_one_identity_across_delivery_paths_is_one_row(self):
        # The same person, the web UI one day and an API key the next.
        await self.seed("alice", 3, user_type="user", day=self.today - timedelta(days=1))
        await self.seed("alice", 5, user_type="api_key", day=self.today)

        result = await get_usage_aggregates(self.db, window="7d")

        self.assertEqual(len(result["per_user"]), 1)
        self.assertEqual(result["per_user"][0]["user_identity"], "alice")
        self.assertEqual(result["per_user"][0]["request_count"], 8)
        self.assertEqual(result["totals"]["unique_users"], 1)
        self.assertEqual(result["totals"]["requests"], 8)

    async def test_the_badge_reads_mixed_when_paths_differ(self):
        await self.seed("alice", 3, user_type="user", day=self.today - timedelta(days=1))
        await self.seed("alice", 5, user_type="api_key", day=self.today)

        result = await get_usage_aggregates(self.db, window="7d")

        self.assertEqual(result["per_user"][0]["user_type"], "mixed")

    async def test_a_single_path_keeps_its_own_label(self):
        await self.seed("alice", 3, user_type="api_key")

        result = await get_usage_aggregates(self.db, window="today")

        self.assertEqual(result["per_user"][0]["user_type"], "api_key")

    async def test_distinct_identities_stay_distinct(self):
        # user_identity is what separates the admin from everyone else.
        await self.seed("root", 4, user_type="admin")
        await self.seed("alice", 3, user_type="user")

        result = await get_usage_aggregates(self.db, window="today")

        self.assertEqual(
            {r["user_identity"]: r["request_count"] for r in result["per_user"]},
            {"root": 4, "alice": 3},
        )
        self.assertEqual(result["totals"]["unique_users"], 2)

    async def test_the_model_drilldown_folds_identities_too(self):
        await self.seed("alice", 3, user_type="user", model="p/m",
                        day=self.today - timedelta(days=1))
        await self.seed("alice", 5, user_type="api_key", model="p/m")

        result = await get_usage_aggregates(self.db, window="7d", filter_model="p/m")

        self.assertEqual(result["breakdown"],
                         [{"user_identity": "alice", "user_type": "mixed",
                           "request_count": 8}])


class DrilldownFilterCompositionTests(AggregateTestCase):
    """/auth/usage pins filter_user to the caller, so both filters must apply."""

    async def _seed_two_models(self):
        await self.seed("alice", 7, model="p/wanted")
        await self.seed("alice", 4, model="p/other")
        await self.seed("bob", 9, model="p/wanted")

    async def test_both_filters_restrict_and_filter_model_picks_the_axis(self):
        await self._seed_two_models()

        result = await get_usage_aggregates(
            self.db, window="today", filter_user="alice", filter_model="p/wanted",
        )

        # Alice's usage of p/wanted alone -- not every model she used, and not
        # bob's traffic on the same model.
        self.assertEqual(result["breakdown"],
                         [{"user_identity": "alice", "user_type": "user",
                           "request_count": 7}])

    async def test_filter_user_alone_still_breaks_down_by_model(self):
        await self._seed_two_models()

        result = await get_usage_aggregates(self.db, window="today", filter_user="alice")

        self.assertEqual(result["breakdown"],
                         [{"model": "p/wanted", "request_count": 7},
                          {"model": "p/other", "request_count": 4}])

    async def test_filter_model_alone_lists_everyone_on_that_model(self):
        await self._seed_two_models()

        result = await get_usage_aggregates(self.db, window="today", filter_model="p/wanted")

        self.assertEqual([r["user_identity"] for r in result["breakdown"]],
                         ["bob", "alice"])

    async def test_the_chart_and_the_table_agree_on_the_drilldown(self):
        """The bug was visible as a chart that disagreed with the table beside it."""
        await self._seed_two_models()
        await self.seed_hourly("alice", 7, model="p/wanted")
        await self.seed_hourly("alice", 4, model="p/other")
        await self.seed_hourly("bob", 9, model="p/wanted")

        result = await get_usage_aggregates(
            self.db, window="today", filter_user="alice", filter_model="p/wanted",
        )
        series = await get_usage_timeseries(
            self.db, window="today", filter_user="alice", filter_model="p/wanted",
        )

        self.assertEqual(sum(b["request_count"] for b in result["breakdown"]),
                         sum(p["count"] for p in series))

    async def test_composition_holds_for_the_month_window(self):
        await self._seed_two_models()

        result = await get_usage_aggregates(
            self.db, window="month", year=self.today.year, month=self.today.month,
            filter_user="alice", filter_model="p/wanted",
        )

        self.assertEqual(result["breakdown"],
                         [{"user_identity": "alice", "user_type": "user",
                           "request_count": 7}])

    async def test_composition_holds_for_the_all_window(self):
        await self._seed_two_models()

        result = await get_usage_aggregates(
            self.db, window="all", filter_user="alice", filter_model="p/wanted",
        )

        self.assertEqual(result["breakdown"],
                         [{"user_identity": "alice", "user_type": "user",
                           "request_count": 7}])


class ChartTableParityTests(AggregateTestCase):
    """The 'today' table reads daily rows and the chart reads hourly ones.

    They only agree while both tables are written in the same transaction -- which
    is what the single-transaction flush guarantees. A partial flush shows up here
    as a chart that has drifted away from the number above it.
    """

    async def test_today_totals_match_the_hourly_series(self):
        for hour, count in ((0, 2), (9, 5), (23, 4)):
            await self.seed_hourly("alice", count, hour=hour)
        await self.seed("alice", 11)

        result = await get_usage_aggregates(self.db, window="today")
        series = await get_usage_timeseries(self.db, window="today")

        self.assertEqual(sum(p["count"] for p in series), result["totals"]["requests"])

    async def test_parity_holds_across_several_users_and_models(self):
        rows = [("alice", "p/a", 3), ("alice", "p/b", 4), ("bob", "p/a", 5)]
        for identity, model, count in rows:
            await self.seed(identity, count, model=model)
            await self.seed_hourly(identity, count, model=model, hour=9)

        result = await get_usage_aggregates(self.db, window="today")
        series = await get_usage_timeseries(self.db, window="today")

        self.assertEqual(result["totals"]["requests"], 12)
        self.assertEqual(sum(p["count"] for p in series), 12)


if __name__ == "__main__":
    unittest.main()

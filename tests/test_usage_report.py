"""Tests for the daily usage report sent to the notification webhook.

The report's windows both end at yesterday, unlike the dashboard's '30d' which
includes today, so these pin the boundaries as well as the top-N cut, the
formatting and the 08:00 scheduling across DST.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import time_utils
from app.auth import webhook
from app.auth.models import Base, RequestUsage
from app.config import config
from app.request_tracker import request_tracker
from app.usage_report import (
    UsageReportScheduler,
    build_usage_report,
    format_usage_report,
    next_run_at,
    seconds_until,
)


class ReportTestCase(unittest.IsolatedAsyncioTestCase):
    """A throwaway DB and a fixed 'today', so the windows land where seeded."""

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self.db = self.session_factory()
        self.today = time_utils.local_today()

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def seed(self, user_id, identity, count, *, days_ago=1, model="p/m",
                   user_type="user"):
        self.db.add(RequestUsage(
            date=self.today - timedelta(days=days_ago), user_id=user_id, pool_id=0,
            user_identity=identity, user_type=user_type,
            model=model, server="openai", request_count=count,
        ))
        await self.db.commit()


class WindowTests(ReportTestCase):
    async def test_windows_end_at_yesterday_and_span_thirty_days(self):
        await self.seed(1, "alice", 1, days_ago=0)     # today: in neither
        await self.seed(1, "alice", 10, days_ago=1)    # yesterday: in both
        await self.seed(1, "alice", 100, days_ago=30)  # first day of the 30
        await self.seed(1, "alice", 1000, days_ago=31) # one day too old

        report = await build_usage_report(self.db, self.today)

        self.assertEqual(report["yesterday"]["total"], 10)
        self.assertEqual(report["last_30_days"]["total"], 110)
        self.assertEqual(report["last_30_days"]["start"], self.today - timedelta(days=30))
        self.assertEqual(report["last_30_days"]["end"], self.today - timedelta(days=1))

    async def test_top_five_sorted_by_count_then_name(self):
        for i, count in enumerate([5, 70, 30, 30, 1, 50, 2], start=1):
            await self.seed(i, f"user{i}", count, model=f"p/model{i}")

        report = await build_usage_report(self.db, self.today)
        users = report["yesterday"]["top_users"]
        models = report["yesterday"]["top_models"]

        self.assertEqual(
            [u["user_identity"] for u in users],
            ["user2", "user6", "user3", "user4", "user1"],
        )
        self.assertEqual(
            [m["model"] for m in models],
            ["p/model2", "p/model6", "p/model3", "p/model4", "p/model1"],
        )
        # The cut only trims the lists; the totals still count everyone.
        self.assertEqual(report["yesterday"]["total"], 188)

    async def test_one_person_across_delivery_paths_is_one_entry(self):
        await self.seed(1, "alice", 3, user_type="user")
        await self.seed(1, "alice", 4, user_type="api_key", model="p/other")

        report = await build_usage_report(self.db, self.today)

        self.assertEqual(len(report["yesterday"]["top_users"]), 1)
        self.assertEqual(report["yesterday"]["top_users"][0]["request_count"], 7)


class FormatTests(ReportTestCase):
    async def test_text_carries_totals_ranges_and_rankings(self):
        await self.seed(1, "alice", 1200, model="p/big")
        await self.seed(2, "bob", 34, model="p/small", days_ago=5)

        text = format_usage_report(await build_usage_report(self.db, self.today))
        yesterday = (self.today - timedelta(days=1)).isoformat()
        start = (self.today - timedelta(days=30)).isoformat()

        self.assertIn(f"Yesterday ({yesterday}): 1,200", text)
        self.assertIn(f"Last 30 days ({start} → {yesterday}): 1,234", text)
        self.assertIn("1. alice — 1,200", text)
        self.assertIn("2. bob — 34", text)
        self.assertIn("1. p/big — 1,200", text)

    async def test_empty_database_reports_no_requests(self):
        text = format_usage_report(await build_usage_report(self.db, self.today))

        self.assertIn("Yesterday", text)
        self.assertEqual(text.count("_No requests_"), 4)


class ScheduleTests(unittest.TestCase):
    TZ = ZoneInfo("America/New_York")

    def test_before_the_hour_runs_today(self):
        now = datetime(2026, 6, 10, 7, 59, tzinfo=self.TZ)
        self.assertEqual(next_run_at(now, 8), datetime(2026, 6, 10, 8, 0, tzinfo=self.TZ))

    def test_at_or_after_the_hour_runs_tomorrow(self):
        for now in (datetime(2026, 6, 10, 8, 0, tzinfo=self.TZ),
                    datetime(2026, 6, 10, 23, 0, tzinfo=self.TZ)):
            self.assertEqual(next_run_at(now, 8), datetime(2026, 6, 11, 8, 0, tzinfo=self.TZ))

    def test_spring_forward_day_is_23_hours(self):
        now = datetime(2026, 3, 7, 8, 0, tzinfo=self.TZ)   # DST starts 2026-03-08
        target = next_run_at(now, 8)
        self.assertEqual((target.month, target.day, target.hour), (3, 8, 8))
        self.assertEqual(seconds_until(target, now), 23 * 3600)

    def test_fall_back_day_is_25_hours(self):
        now = datetime(2026, 10, 31, 8, 0, tzinfo=self.TZ)  # DST ends 2026-11-01
        target = next_run_at(now, 8)
        self.assertEqual((target.month, target.day, target.hour), (11, 1, 8))
        self.assertEqual(seconds_until(target, now), 25 * 3600)


class SchedulerTests(ReportTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._real_url = config.webhook.notification_webhook_url
        self._real_flush = request_tracker.flush_pending
        self._real_send = webhook.send_usage_report_webhook

    async def asyncTearDown(self):
        config.webhook.notification_webhook_url = self._real_url
        request_tracker.flush_pending = self._real_flush
        webhook.send_usage_report_webhook = self._real_send
        await super().asyncTearDown()

    async def test_no_webhook_starts_no_task(self):
        config.webhook.notification_webhook_url = None
        scheduler = UsageReportScheduler()

        await scheduler.start()

        self.assertIsNone(scheduler._task)

    async def test_webhook_starts_task_and_stop_cancels_it(self):
        config.webhook.notification_webhook_url = "https://example.invalid/hook"
        scheduler = UsageReportScheduler()

        await scheduler.start()
        self.assertIsNotNone(scheduler._task)
        await scheduler.stop()

        self.assertIsNone(scheduler._task)

    async def test_run_once_flushes_then_sends_the_report(self):
        config.webhook.notification_webhook_url = "https://example.invalid/hook"
        request_tracker.flush_pending = AsyncMock()
        webhook.send_usage_report_webhook = AsyncMock(return_value=True)
        await self.seed(1, "alice", 9)

        sent = await UsageReportScheduler()._run_once(session_factory=self.session_factory)

        self.assertTrue(sent)
        request_tracker.flush_pending.assert_awaited_once()
        webhook.send_usage_report_webhook.assert_awaited_once()
        text = webhook.send_usage_report_webhook.await_args.args[0]
        self.assertIn("1. alice — 9", text)


if __name__ == "__main__":
    unittest.main()

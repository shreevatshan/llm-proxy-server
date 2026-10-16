"""Daily usage report, posted to the notification webhook at REPORT_HOUR local time.

Runs only when NOTIFICATION_WEBHOOK_URL is set. Both windows end at yesterday, so
every number in a report is final no matter when during the day it is read. A run
missed while the process was down is skipped, not caught up on the next start.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)

REPORT_HOUR = 8
TOP_N = 5
REPORT_DAYS = 30
# Re-read the wall clock at least this often while waiting, so a suspended host or
# a clock step cannot push the run hours past its slot.
_MAX_SLEEP_SECONDS = 900


async def build_usage_report(db, today: date) -> dict:
    """Totals and top users/models for yesterday and for the REPORT_DAYS ending yesterday."""
    from app.auth.database import get_usage_report_aggregates

    yesterday = today - timedelta(days=1)
    month_start = today - timedelta(days=REPORT_DAYS)

    windows = {}
    for key, start in (("yesterday", yesterday), ("last_30_days", month_start)):
        agg = await get_usage_report_aggregates(db, start, yesterday)
        windows[key] = {
            "start": start,
            "end": yesterday,
            "total": agg["total"],
            "top_users": agg["per_user"][:TOP_N],
            "top_models": agg["per_model"][:TOP_N],
        }
    return windows


def _ranked(rows: list, label_key: str) -> list:
    if not rows:
        return ["_No requests_"]
    return [f"{i}. {r[label_key]} — {r['request_count']:,}" for i, r in enumerate(rows, 1)]


def format_usage_report(report: dict) -> str:
    """Render a build_usage_report() result as chat-friendly plain text."""
    day, month = report["yesterday"], report["last_30_days"]
    lines = [
        f"📊 Daily Usage Report — {day['end'].isoformat()}",
        f"🌐 Domain: {config.server.domain}",
        "",
        "📈 Total Requests",
        f"• Yesterday ({day['end'].isoformat()}): {day['total']:,}",
        f"• Last {REPORT_DAYS} days ({month['start'].isoformat()} → {month['end'].isoformat()}): "
        f"{month['total']:,}",
    ]
    sections = (
        ("👤", "Users", "top_users", "user_identity"),
        ("🤖", "Models", "top_models", "model"),
    )
    for icon, noun, rows_key, label_key in sections:
        for title, window in (("Yesterday", day), (f"Last {REPORT_DAYS} Days", month)):
            lines += ["", f"{icon} Top {TOP_N} {noun} — {title}"]
            lines += _ranked(window[rows_key], label_key)
    return "\n".join(lines)


def next_run_at(now: datetime, hour: int) -> datetime:
    """The next local `hour`:00 strictly after `now` (an aware datetime)."""
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        # Aware-datetime arithmetic steps the wall clock, not elapsed time, so this
        # lands on tomorrow's `hour`:00 even when a DST change makes it 23h or 25h away.
        target += timedelta(days=1)
    return target


def seconds_until(target: datetime, now: datetime) -> float:
    """Real seconds from `now` to `target`.

    Subtracting two datetimes that share a tzinfo ignores any offset change between
    them, so compare in UTC to get elapsed time across a DST transition.
    """
    return (target.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()


class UsageReportScheduler:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if not config.webhook.notification_webhook_url:
            logger.info("Daily usage report disabled: no notification webhook configured")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        from app import time_utils

        # The first slot is the next one after start-up: a restart past REPORT_HOUR
        # waits for tomorrow rather than sending today's report late.
        target = next_run_at(time_utils.local_now(), REPORT_HOUR)
        logger.info(f"Daily usage report scheduled for {target.isoformat()}")
        while True:
            remaining = seconds_until(target, time_utils.local_now())
            if remaining > 0:
                await asyncio.sleep(min(remaining, _MAX_SLEEP_SECONDS))
                continue
            try:
                await self._run_once()
            except Exception:
                logger.exception("Daily usage report failed")
            target = next_run_at(time_utils.local_now(), REPORT_HOUR)
            logger.info(f"Next daily usage report scheduled for {target.isoformat()}")

    async def _run_once(self, session_factory=None) -> bool:
        from app import time_utils
        from app.auth.webhook import send_usage_report_webhook
        from app.request_tracker import request_tracker

        if session_factory is None:
            from app.auth.database import AsyncSessionLocal
            session_factory = AsyncSessionLocal

        # Yesterday's last few seconds may still be sitting in the tracker's buffer.
        await request_tracker.flush_pending()
        async with session_factory() as db:
            report = await build_usage_report(db, time_utils.local_today())
        return await send_usage_report_webhook(format_usage_report(report))


usage_report_scheduler = UsageReportScheduler()

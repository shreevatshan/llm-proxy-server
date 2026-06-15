"""Timezone-aware time helpers.

All day-boundary decisions (usage bucketing, rate-limit resets, dashboard
windows) must go through these helpers so that the TIMEZONE env var controls
what "today" means everywhere consistently.
"""

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo


def get_tz() -> ZoneInfo:
    from app.config import config
    return ZoneInfo(config.server.timezone)


def local_now() -> datetime:
    return datetime.now(tz=get_tz())


def local_today() -> date:
    return local_now().date()


def local_hour() -> int:
    return local_now().hour


def seconds_until_local_midnight() -> float:
    now = local_now()
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (next_midnight - now).total_seconds()

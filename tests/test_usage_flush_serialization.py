"""Tests that the usage flush cannot overlap itself or a rename/purge.

A flush is three steps — snapshot the buffer, write it to the DB, subtract the
snapshot — and the buffer lock is released across the write. Anything that runs
in that window sees counts that are recorded in neither place yet:

  * a second flush takes the same snapshot and applies the increment-on-conflict
    upserts twice, inflating usage and the RPD limits derived from it;
  * a rename or purge finishes, and then the in-flight write lands, re-creating
    exactly the rows it just moved or deleted.

RequestTracker.pause_flush() and the flush mutex close that window.
"""

import asyncio
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.auth.models import RequestUsage, RequestUsageHourly
from app.request_tracker import RequestTracker

from tests.test_usage_identity_rename import DAY, UsageDBTestCase

# Buffer key: (date, hour, user_identity, user_type, model, server)
KEY = (DAY, 9, "alice", "user", "p/m", "openai")


async def _noop(*args, **kwargs):
    return None


class SlowFlushTestCase(UsageDBTestCase):
    """Base: points the flush at the temp DB and stretches its write out."""

    @contextmanager
    def slow_flush(self, delay=0.3):
        """Stall the *first* flush's write so another task can race into it.

        Only the first is delayed: a racer that also stalled would finish in the
        same order anyway, and the point is to have one write still outstanding
        while later work commits. _do_flush imports these lazily from
        app.auth.database, so patching the module attributes is what takes effect.
        Maintenance (prune/rollup) runs after every flush and is irrelevant here.
        """
        import app.auth.database as database

        real_flush = database.flush_usage_rows
        writes = 0

        async def slow_write(hourly_rows, daily_rows):
            nonlocal writes
            writes += 1
            if writes == 1:
                await asyncio.sleep(delay)
            await real_flush(hourly_rows, daily_rows)

        with patch("app.auth.database.AsyncSessionLocal", self._session_factory), \
             patch("app.auth.database.flush_usage_rows", slow_write), \
             patch("app.auth.database.prune_hourly_usage", _noop), \
             patch("app.auth.database.rollup_to_monthly", _noop):
            yield


class OverlappingFlushTests(SlowFlushTestCase):
    async def test_two_concurrent_flushes_write_the_counts_once(self):
        tracker = RequestTracker()
        tracker._usage_buffer[KEY] = 5

        with self.slow_flush():
            await asyncio.gather(tracker.flush_pending(), tracker.flush_pending())

        # Serialised, the second flush finds an empty buffer and writes nothing.
        self.assertEqual(await self._rows(RequestUsage, "alice"), (1, 5))
        self.assertEqual(await self._rows(RequestUsageHourly, "alice"), (1, 5))
        self.assertEqual(tracker._usage_buffer, {})

    async def test_counts_arriving_during_a_flush_are_kept(self):
        """The subtract must remove the snapshot, not the whole buffer."""
        tracker = RequestTracker()
        tracker._usage_buffer[KEY] = 5

        with self.slow_flush():
            flush = asyncio.create_task(tracker.flush_pending())
            await asyncio.sleep(0.01)  # let it reach the DB write
            tracker._usage_buffer[KEY] += 2
            await flush

        self.assertEqual(await self._rows(RequestUsage, "alice"), (1, 5))
        self.assertEqual(tracker._usage_buffer[KEY], 2)


class FlushVersusPurgeTests(SlowFlushTestCase):
    async def test_a_purge_under_pause_flush_is_not_undone(self):
        """What DELETE /admin/usage does: flush, delete, drop the buffer.

        Unguarded, the purge's own flush_pending() re-snapshots counts the racing
        flush has not subtracted yet, and that racing write then lands after the
        DELETE and puts the rows straight back.
        """
        from app.auth.database import delete_usage_records

        await self._seed("alice", 11)
        tracker = RequestTracker()
        tracker._usage_buffer[KEY] = 5

        with self.slow_flush():
            # A flush already in flight, stalled mid-write, when the purge starts.
            racing = asyncio.create_task(tracker.flush_pending())
            await asyncio.sleep(0.01)  # let it reach the DB write

            async with tracker.pause_flush():
                await tracker.flush_pending()
                await delete_usage_records(self.db, "user", "alice")
                await self.db.commit()
                dropped = await tracker.drop_buffered_usage("user", "alice")

            await racing

        # The racing flush drained before the purge began, so its counts were
        # deleted along with the rest instead of landing afterwards.
        self.assertEqual(await self._rows(RequestUsage, "alice"), (0, 0))
        self.assertEqual(await self._rows(RequestUsageHourly, "alice"), (0, 0))
        self.assertEqual(dropped, 0)
        self.assertEqual(tracker._usage_buffer, {})


class FlushInsideResettleTests(SlowFlushTestCase):
    """The pool purge adds a third flush inside pause_flush(), and its position matters.

    DELETE /admin/usage?view=pool ends by re-settling every affected pool, because the
    ledger still charges members for the rows just deleted. settle_pool calls
    flush_pending() itself, so that re-settle is a flush running after the purge has
    already committed — and drop_buffered_usage only clears the in-memory buffer. Run it
    before the drop and anything buffered since the purge's own flush is written to the
    DB, where the drop can no longer reach it.
    """

    async def _purge(self, tracker, *, resettle_before_drop):
        """The DELETE sequence, with the re-settle's flush on either side of the drop."""
        from app.auth.database import delete_usage_records

        async with tracker.pause_flush():
            await tracker.flush_pending()
            await delete_usage_records(self.db, "user", "alice")
            await self.db.commit()

            # A request lands between the purge's flush and the end of the block: the
            # window the drop exists to close.
            tracker._usage_buffer[KEY] = 4

            if resettle_before_drop:
                await tracker.flush_pending()   # what settle_pool does internally
                return await tracker.drop_buffered_usage("user", "alice")
            dropped = await tracker.drop_buffered_usage("user", "alice")
            await tracker.flush_pending()
            return dropped

    async def test_the_resettle_flush_runs_after_the_drop(self):
        await self._seed("alice", 11)
        tracker = RequestTracker()

        with self.slow_flush(delay=0):
            dropped = await self._purge(tracker, resettle_before_drop=False)

        self.assertEqual(dropped, 4)
        self.assertEqual(await self._rows(RequestUsage, "alice"), (0, 0))
        self.assertEqual(await self._rows(RequestUsageHourly, "alice"), (0, 0))

    async def test_resettling_before_the_drop_would_resurrect_the_rows(self):
        """Guards the ordering: this is what the sequence does if it is reversed."""
        await self._seed("alice", 11)
        tracker = RequestTracker()

        with self.slow_flush(delay=0):
            dropped = await self._purge(tracker, resettle_before_drop=True)

        self.assertEqual(dropped, 0, "the flush already emptied the buffer")
        self.assertEqual(await self._rows(RequestUsage, "alice"), (1, 4))


class PartialFlushTests(UsageDBTestCase):
    """The two usage tables must be written in one transaction, or not at all.

    The buffer is subtracted only after the write returns, so a flush that committed
    one table and raised on the other would leave the counts in the buffer *and* in
    the DB. Every later cycle would re-add the half that landed, inflating it without
    bound and splitting the 'today' chart (hourly) from the 'today' table (daily).
    """

    def _failing_daily(self):
        """Patch the daily upsert to blow up, leaving the hourly one intact."""
        import app.auth.database as database

        real_upsert = database._usage_upsert

        def upsert(table, rows, index_elements):
            if table is RequestUsage:
                raise RuntimeError("daily upsert failed")
            return real_upsert(table, rows, index_elements)

        return patch("app.auth.database._usage_upsert", upsert)

    async def test_a_failed_daily_write_rolls_the_hourly_write_back(self):
        tracker = RequestTracker()
        tracker._usage_buffer[KEY] = 5

        with patch("app.auth.database.AsyncSessionLocal", self._session_factory), \
             patch("app.auth.database.prune_hourly_usage", _noop), \
             patch("app.auth.database.rollup_to_monthly", _noop):
            with self._failing_daily():
                await tracker.flush_pending()

            # Nothing was written, and the counts are still owed.
            self.assertEqual(await self._rows(RequestUsageHourly, "alice"), (0, 0))
            self.assertEqual(await self._rows(RequestUsage, "alice"), (0, 0))
            self.assertEqual(tracker._usage_buffer[KEY], 5)

            # The retry writes each table exactly once.
            await tracker.flush_pending()

        self.assertEqual(await self._rows(RequestUsageHourly, "alice"), (1, 5))
        self.assertEqual(await self._rows(RequestUsage, "alice"), (1, 5))
        self.assertEqual(tracker._usage_buffer, {})


if __name__ == "__main__":
    unittest.main()

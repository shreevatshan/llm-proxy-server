"""Concurrent RPD cache misses for one key share a single DB read.

Without this, a burst of requests for one user or pool each opened a pooled DB
session when the 5-second cache entry expired.
"""

import asyncio
import unittest
from unittest.mock import patch

from app.rate_limit import RateLimitTracker


class RpdSingleFlightTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tracker = RateLimitTracker()
        from app.request_tracker import request_tracker
        self.request_tracker = request_tracker
        self.calls = 0

    async def _slow_count(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.1)
        return 7

    async def test_overall_count_single_flight(self):
        with patch.object(self.request_tracker, "get_today_count", self._slow_count):
            results = await asyncio.gather(
                *(self.tracker._get_today_count("user:alice", [1]) for _ in range(20))
            )
        self.assertEqual(self.calls, 1)
        self.assertEqual(set(results), {7})
        self.assertEqual(self.tracker._rpd_inflight, {})

    async def test_distinct_keys_are_not_merged(self):
        with patch.object(self.request_tracker, "get_today_count", self._slow_count):
            await asyncio.gather(
                self.tracker._get_today_count("user:alice", [1]),
                self.tracker._get_today_count("user:bob", [2]),
            )
        self.assertEqual(self.calls, 2)

    async def test_group_counts_single_flight(self):
        with patch.object(self.request_tracker, "get_today_group_count", self._slow_count), \
             patch.object(self.request_tracker, "get_today_instance_group_count", self._slow_count):
            await asyncio.gather(
                *(self.tracker._get_today_group_count("user:alice", [1], ["p/m"], 3)
                  for _ in range(10)),
                *(self.tracker._get_today_instance_group_count("user:alice", [1], ["p"], 3)
                  for _ in range(10)),
            )
        # One read per cache (model group, instance group), not per request.
        self.assertEqual(self.calls, 2)

    async def test_failure_serves_stale_and_does_not_cache_zero(self):
        async def boom(*args, **kwargs):
            raise RuntimeError("db down")

        with patch.object(self.request_tracker, "get_today_count", boom):
            self.assertEqual(await self.tracker._get_today_count("user:alice", [1]), 0)
        self.assertNotIn("user:alice", self.tracker._rpd_cache)
        self.assertEqual(self.tracker._rpd_inflight, {})

    async def test_waiters_share_a_failure_instead_of_retrying_serially(self):
        async def slow_boom(*args, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.1)
            raise RuntimeError("db down")

        with patch.object(self.request_tracker, "get_today_count", slow_boom):
            results = await asyncio.gather(
                *(self.tracker._get_today_count("user:alice", [1]) for _ in range(20))
            )
            self.assertEqual(self.calls, 1)
            self.assertEqual(set(results), {0})
            self.assertEqual(self.tracker._rpd_inflight, {})

            # A request arriving after the failure still retries the DB.
            await self.tracker._get_today_count("user:alice", [1])
        self.assertEqual(self.calls, 2)


if __name__ == "__main__":
    unittest.main()

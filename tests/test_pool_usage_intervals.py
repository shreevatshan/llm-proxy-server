"""Tests that pool usage means what the pool consumed, not what its members ever sent.

The usage tables have no pool dimension -- rows are keyed by user identity -- so a pool's
consumption has to be reconstructed at read time. Reconstructing it as "current members x
all their rows" is wrong in both directions:

  * someone who burned 900 requests before joining drags all 900 in with them, so the
    Usage tab reports a number the pool never spent and the Quotas tab disagrees with it;
  * someone who leaves retroactively erases spending that genuinely was the pool's, so
    yesterday's total silently changes when today's roster does.

PoolMembershipInterval records each stint, and every pool usage read is bounded by those
spans. Both ends are inclusive, matching what settlement charges for a mid-day join.
"""

import unittest
from datetime import timedelta

from fastapi import HTTPException

from app.routes import pools as pool_routes
from tests.pool_test_base import DAY, PoolTestCase


class PoolUsageIntervalTests(PoolTestCase):

    async def _total(self, pool, window="30d"):
        payload = await pool_routes.build_pool_usage(self.db, pool.id, window=window)
        return payload["totals"]["requests"]

    async def test_usage_sent_before_the_join_is_not_the_pools(self):
        """The reported case: a heavy user joins and brings their whole history in."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", bob)

        await self.seed_usage("alice", 900, day=DAY - timedelta(days=5))
        await self.seed_usage("alice", 3, day=DAY)
        await self.seed_usage("bob", 10, day=DAY, model="other/model")

        await self.join_pool(pool, alice, joined_on=DAY)

        self.assertEqual(await self._total(pool), 13,
                         "alice's 3 since joining plus bob's 10, not her 900 from before")

    async def test_a_leavers_contribution_stays_in_the_pools_history(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob], joined_on=DAY - timedelta(days=5))

        await self.seed_usage("bob", 50, day=DAY - timedelta(days=3))
        self.assertEqual(await self._total(pool), 50)

        await self.leave_pool_on(pool, bob, left_on=DAY - timedelta(days=1))
        self.assertEqual(await self._total(pool), 50,
                         "the pool really did spend it; walking out cannot unspend it")

    async def test_the_join_day_and_the_leave_day_are_both_counted(self):
        """Usage is day-grained, so a mid-day join cannot be split -- as settlement agrees."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, joined_on=DAY - timedelta(days=9))

        joined = DAY - timedelta(days=4)
        left = DAY - timedelta(days=2)
        await self.join_pool(pool, bob, joined_on=joined)
        await self.seed_usage("bob", 7, day=joined)
        await self.seed_usage("bob", 5, day=left)
        await self.leave_pool_on(pool, bob, left_on=left)

        self.assertEqual(await self._total(pool), 12)

    async def test_two_stints_sum_and_the_gap_between_them_is_excluded(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, joined_on=DAY - timedelta(days=20))

        # In for day -15, out for -12, back in from -8.
        await self.join_pool(pool, bob, joined_on=DAY - timedelta(days=16))
        await self.seed_usage("bob", 4, day=DAY - timedelta(days=15))
        await self.leave_pool_on(pool, bob, left_on=DAY - timedelta(days=14))

        await self.seed_usage("bob", 100, day=DAY - timedelta(days=12))  # the gap

        await self.join_pool(pool, bob, joined_on=DAY - timedelta(days=8))
        await self.seed_usage("bob", 6, day=DAY - timedelta(days=7))

        self.assertEqual(await self._total(pool), 10,
                         "both stints count; the 100 sent between them does not")

    async def test_a_former_member_can_still_be_drilled_into(self):
        """A leaver has a row in the breakdown, so their row must not 403."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob], joined_on=DAY - timedelta(days=5))

        await self.seed_usage("bob", 9, day=DAY - timedelta(days=3))
        await self.leave_pool_on(pool, bob, left_on=DAY - timedelta(days=1))

        payload = await pool_routes.build_pool_usage(
            self.db, pool.id, window="30d", view="user", target="bob",
        )
        self.assertEqual(sum(r["request_count"] for r in payload["breakdown"]), 9)

    async def test_a_stranger_is_still_refused(self):
        alice = await self.make_user("alice", rpd_limit=100)
        pool = await self.make_pool("team", alice)
        await self.make_user("carol", rpd_limit=100)
        await self.seed_usage("carol", 5)

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.build_pool_usage(
                self.db, pool.id, window="30d", view="user", target="carol",
            )
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_a_drill_down_stops_at_the_members_own_stint(self):
        """Not just the total: the per-member view must be bounded too, or the
        breakdown row and the number beside it disagree."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", bob)

        await self.seed_usage("alice", 900, day=DAY - timedelta(days=5))
        await self.seed_usage("alice", 3, day=DAY)
        await self.join_pool(pool, alice, joined_on=DAY)

        payload = await pool_routes.build_pool_usage(
            self.db, pool.id, window="30d", view="user", target="alice",
        )
        self.assertEqual(sum(r["request_count"] for r in payload["breakdown"]), 3)

    async def test_todays_usage_total_agrees_with_the_quotas_tab(self):
        """The two tabs answer the same question for today and must not diverge."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])

        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await self.seed_usage("bob", 2, model="other/model")

        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="today")
        _members, scopes, _detail, _users = await pool_routes._render_pool(self.db, pool)
        overall = next(s for s in scopes if s.scope_kind == "overall")

        self.assertEqual(payload["totals"]["requests"], overall.used)

    async def test_pool_usage_is_empty_before_anyone_has_a_stint(self):
        """No overlapping interval must mean no rows, never every row."""
        alice = await self.make_user("alice", rpd_limit=100)
        pool = await self.make_pool("team", alice, joined_on=DAY)
        await self.seed_usage("alice", 40, day=DAY - timedelta(days=2))

        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="yesterday")
        self.assertEqual(payload["totals"]["requests"], 0)
        self.assertEqual(payload["per_member"], [])


class AdminPoolIntervalTests(PoolTestCase):
    """The admin By Pool views fold the same spans, so they cannot drift from the member view."""

    async def _pool_with_a_leaver(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob], joined_on=DAY - timedelta(days=5))
        await self.seed_usage("alice", 900, day=DAY - timedelta(days=40))  # outside 30d
        await self.seed_usage("alice", 4, day=DAY)
        await self.seed_usage("bob", 6, day=DAY - timedelta(days=3))
        await self.leave_pool_on(pool, bob, left_on=DAY - timedelta(days=1))
        return pool

    async def test_the_admin_drill_down_lists_the_leaver_with_their_contribution(self):
        from types import SimpleNamespace
        from app.routes import admin as admin_routes

        pool = await self._pool_with_a_leaver()
        payload = await admin_routes.get_usage(
            view="pool", id=str(pool.id), window="30d", year=None, month=None,
            current_admin=SimpleNamespace(username="root"), db=self.db,
        )
        counts = {r["user_identity"]: r["request_count"] for r in payload["breakdown"]}
        self.assertEqual(counts, {"alice": 4, "bob": 6})

    async def test_the_admin_route_and_the_member_route_report_the_same_pool(self):
        """Two implementations of the same fold, so they need an assertion holding them
        together -- the admin /admin/usage?view=pool branch is deliberately not a call
        into build_pool_usage."""
        from types import SimpleNamespace
        from app.routes import admin as admin_routes

        pool = await self._pool_with_a_leaver()
        admin_payload = await admin_routes.get_usage(
            view="pool", id=str(pool.id), window="30d", year=None, month=None,
            current_admin=SimpleNamespace(username="root"), db=self.db,
        )
        member_payload = await pool_routes.build_pool_usage(self.db, pool.id, window="30d")

        admin_counts = {
            r["user_identity"]: r["request_count"]
            for r in admin_payload["breakdown"] if r["request_count"]
        }
        member_counts = {
            r["user_identity"]: r["request_count"] for r in member_payload["per_member"]
        }
        self.assertEqual(admin_counts, member_counts)
        self.assertEqual(admin_payload["timeseries"], member_payload["timeseries"])

    async def test_per_pool_counts_only_what_the_pool_consumed(self):
        from types import SimpleNamespace
        from app.routes import admin as admin_routes

        pool = await self._pool_with_a_leaver()
        payload = await admin_routes.get_usage(
            view=None, id=None, window="30d", year=None, month=None,
            current_admin=SimpleNamespace(username="root"), db=self.db,
        )
        entry = next(p for p in payload["per_pool"] if p["pool_id"] == pool.id)
        self.assertEqual(entry["request_count"], 10,
                         "not folded from each member's whole per_user total")


if __name__ == "__main__":
    unittest.main()

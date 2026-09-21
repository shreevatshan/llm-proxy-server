"""Tests for the By Pool axis of /admin/usage.

Pools are a third way of slicing the same traffic the Usage tab already reports by user
and by model, so the admin views for them are folds of the usage tables through one
membership query rather than a separate accounting of their own. Three properties carry
that:

  * the fold is over `(user_identity, user_type)` rows, so a username appearing under two
    types has to accumulate, not overwrite;
  * a pool's member list, not its traffic, decides which rows exist — a member with
    nothing in the window reads as 0, never as absent;
  * purging a pool's usage has to re-settle it, or every member's ledger keeps charging
    them for requests that no longer exist.

The route functions are called directly: they take the db session as a dependency, and
the admin dependency is not exercised by anything these tests assert.
"""

import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from app.auth.models import RequestUsage
from app.routes import admin as admin_routes
from app.routes import pools as pool_routes
from tests.pool_test_base import DAY, PoolTestCase

# The admin dependency is resolved by FastAPI and is only read for the audit log line;
# these tests call the route functions directly, so a name is all it has to supply.
ADMIN = SimpleNamespace(username="root")


class PoolMembershipMapTests(PoolTestCase):

    async def test_maps_every_pool_to_its_members_in_join_order(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        carol = await self.make_user("carol", rpd_limit=100)
        team = await self.make_pool("team", alice, [bob])
        solo = await self.make_pool("solo", carol)

        mapping = await pool_routes.pool_membership_map(self.db)

        self.assertEqual(set(mapping), {team.id, solo.id})
        self.assertEqual(mapping[team.id]["name"], "team")
        # Ordered by joined_at then id, matching _member_rows, so a pool reads the
        # same here as on every other pool surface.
        self.assertEqual(mapping[team.id]["members"], ["alice", "bob"])
        self.assertEqual(mapping[solo.id]["members"], ["carol"])

    async def test_a_pool_with_no_members_is_present_at_zero(self):
        """Shouldn't occur — _remove_member deletes the last member's pool — but the
        outer join must degrade to an empty list rather than dropping the pool."""
        from app.auth.models import RequestPool

        owner = await self.make_user("alice", rpd_limit=100)
        pool = RequestPool(name="ghost", owner_user_id=owner.id)
        self.db.add(pool)
        await self.db.commit()

        mapping = await pool_routes.pool_membership_map(self.db)
        self.assertEqual(mapping[pool.id]["members"], [])

    async def test_no_pools_is_an_empty_map_not_an_error(self):
        await self.make_user("alice", rpd_limit=100)
        self.assertEqual(await pool_routes.pool_membership_map(self.db), {})


class TopLevelPerPoolTests(PoolTestCase):

    async def _usage(self, window="today"):
        return await admin_routes.get_usage(
            view=None, id=None, window=window, year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )

    async def test_per_pool_totals_its_members_requests(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        await self.make_user("carol", rpd_limit=100)  # unpooled
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await self.seed_usage("bob", 2, model="other/model")
        await self.seed_usage("carol", 5)

        result = await self._usage()

        self.assertEqual(len(result["per_pool"]), 1)
        entry = result["per_pool"][0]
        self.assertEqual(entry["pool_id"], pool.id)
        self.assertEqual(entry["name"], "team")
        self.assertEqual(entry["member_count"], 2)
        self.assertEqual(entry["members"], ["alice", "bob"])
        self.assertEqual(entry["request_count"], 10)
        # Carol's traffic is in the totals but in no pool, which is exactly why the
        # By Pool percentages don't sum to 100.
        self.assertEqual(result["totals"]["requests"], 15)

    async def test_a_pool_with_no_traffic_still_has_a_row(self):
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_pool("idle", alice)

        result = await self._usage()
        self.assertEqual(
            [(p["name"], p["request_count"]) for p in result["per_pool"]], [("idle", 0)]
        )

    async def test_one_username_on_two_user_types_is_accumulated(self):
        """get_usage_aggregates groups by (user_identity, user_type), so the same name
        can arrive on two rows. Assigning instead of adding would drop one of them."""
        alice = await self.make_user("alice", rpd_limit=100)
        pool = await self.make_pool("team", alice)
        await self.seed_usage("alice", 8)
        # A different model: request_usage is unique on (date, identity, model, server),
        # so the same name on two user_types has to differ somewhere else.
        self.db.add(RequestUsage(
            date=DAY, user_identity="alice", user_type="api_key",
            model="other/model", server="openai", request_count=3,
        ))
        await self.db.commit()

        result = await self._usage()
        self.assertEqual(result["per_pool"][0]["request_count"], 11)

    async def test_pools_sort_by_requests_then_name(self):
        a = await self.make_user("a", rpd_limit=100)
        b = await self.make_user("b", rpd_limit=100)
        c = await self.make_user("c", rpd_limit=100)
        await self.make_pool("zeta", a)
        await self.make_pool("alpha", b)
        await self.make_pool("busy", c)
        await self.seed_usage("c", 4)

        result = await self._usage()
        self.assertEqual([p["name"] for p in result["per_pool"]], ["busy", "alpha", "zeta"])

    async def test_per_pool_is_absent_from_a_drill_down_payload(self):
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_pool("team", alice)
        await self.seed_usage("alice", 4)

        drill = await admin_routes.get_usage(
            view="user", id="alice", window="today", year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )
        self.assertNotIn("per_pool", drill)


class PoolDrillDownTests(PoolTestCase):

    async def _drill(self, pool_id, window="today"):
        return await admin_routes.get_usage(
            view="pool", id=str(pool_id), window=window, year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )

    async def test_breakdown_is_per_member_and_zero_filled(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        await self.make_user("carol", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await self.seed_usage("alice", 1, model="other/model")
        await self.seed_usage("carol", 5)  # outside the pool, must not appear

        payload = await self._drill(pool.id)

        self.assertEqual(payload["pool"], {"id": pool.id, "name": "team", "member_count": 2})
        self.assertEqual(
            [(r["user_identity"], r["request_count"]) for r in payload["breakdown"]],
            [("alice", 9), ("bob", 0)],
        )
        # user_type rides along from the same select, so the drill-down keeps the Type
        # column the By User table has.
        self.assertEqual(payload["breakdown"][0]["user_type"], "user")
        self.assertIsNone(payload["breakdown"][1]["user_type"])

    async def test_timeseries_is_scoped_to_the_pools_members(self):
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_user("carol", rpd_limit=100)
        pool = await self.make_pool("team", alice)
        await self.seed_usage("alice", 3)
        await self.seed_usage("carol", 40)

        payload = await self._drill(pool.id, window="30d")
        self.assertEqual(sum(b["count"] for b in payload["timeseries"]), 3)

    async def test_an_unknown_pool_is_a_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._drill(9999)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_a_non_numeric_id_is_a_400(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._drill("team")
        self.assertEqual(ctx.exception.status_code, 400)


class PoolPurgeTests(PoolTestCase):
    """DELETE /admin/usage?view=pool — the purge, and the settlement it has to repair."""

    async def _delete(self, view, ident):
        return await admin_routes.delete_usage(
            view=view, id=str(ident), current_admin=ADMIN, db=self.db,
        )

    async def _team(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await self.seed_usage("bob", 2, model="other/model")
        return alice, bob, pool

    async def test_purging_a_pool_clears_every_members_usage(self):
        alice, bob, pool = await self._team()

        result = await self._delete("pool", pool.id)

        self.assertEqual(result["total"], 10)
        self.assertEqual(result["members"], ["alice", "bob"])
        remaining = await admin_routes.get_usage(
            view=None, id=None, window="today", year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )
        self.assertEqual(remaining["totals"]["requests"], 0)
        # The pool itself survives the purge of its usage.
        self.assertEqual([p["name"] for p in remaining["per_pool"]], ["team"])

    async def test_purging_a_pool_leaves_no_member_charged(self):
        """The ledger charges members against today's rows. Deleting the rows without
        re-settling leaves charged > sent on every member row of their own pool tab."""
        from app.auth import pools as pool_settlement

        alice, bob, pool = await self._team()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        # 10 shared requests split evenly between two equal limits.
        self.assertEqual(await self.charged(pool.id, alice.id), 5)

        await self._delete("pool", pool.id)

        self.assertEqual(await self.charged(pool.id, alice.id), 0)
        self.assertEqual(await self.charged(pool.id, bob.id), 0)
        self.assertEqual(await self.effective_used(alice), 0)
        self.assertEqual(await self.effective_used(bob), 0)

    async def test_purging_one_pooled_user_also_re_settles_their_pool(self):
        """The same gap on the pre-existing per-user path: a pooled user's charge has to
        be re-apportioned too, or their pool keeps billing them for deleted traffic.

        Not zero, because the pool is still consuming: what must hold is settlement's
        invariant, sum(charged) == what the pool has actually sent. Without the re-settle
        alice stays charged 5 for requests that no longer exist.
        """
        from app.auth import pools as pool_settlement

        alice, bob, pool = await self._team()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        await self._delete("user", "alice")

        # Bob's 2 remaining requests, split evenly across the two members, and that is
        # the whole of what the ledger now claims.
        self.assertEqual(await self.charged(pool.id, alice.id), 1)
        self.assertEqual(await self.charged(pool.id, bob.id), 1)
        # Carry is absolute — charged minus own rows — so alice carries the 1 she is
        # charged for bob's traffic, and he is credited the 1 she took over.
        self.assertEqual(await self.carry(alice.id), 1)
        self.assertEqual(await self.carry(bob.id), -1)
        # Both read the pool's consumption, which is now just bob's 2.
        self.assertEqual(await self.effective_used(alice), 2)
        self.assertEqual(await self.effective_used(bob), 2)

    async def test_purging_an_unpooled_user_is_unaffected(self):
        carol = await self.make_user("carol", rpd_limit=100)
        await self.seed_usage("carol", 5)

        result = await self._delete("user", "carol")
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["members"], [])
        self.assertEqual(await self.effective_used(carol), 0)

    async def test_an_unknown_pool_is_a_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._delete("pool", 9999)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_a_non_numeric_pool_id_is_a_400(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._delete("pool", "team")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()

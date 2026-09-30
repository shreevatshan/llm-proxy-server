"""Tests for the By Pool axis of /admin/usage.

Pools are a third way of slicing the same traffic the Usage tab already reports by user
and by model, so the admin views for them are folds of the usage tables through one
membership query rather than a separate accounting of their own. Three properties carry
that:

  * the fold is over per-(user, user_type) rows, so a username appearing under two
    types has to accumulate, not overwrite;
  * a pool's member list, not its traffic, decides which rows exist — a member with
    nothing in the window reads as 0, never as absent;
  * purging a pool's usage has to re-settle it, or every member's ledger keeps charging
    them for requests that no longer exist.

The route functions are called directly: they take the db session as a dependency, and
the admin dependency is not exercised by anything these tests assert.
"""

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.auth.models import ADMIN_USAGE_USER_ID, RequestUsage
from app.routes import admin as admin_routes
from app.routes import pools as pool_routes
from tests.pool_test_base import DAY, PoolTestCase

# Seeding a day either side of DAY, for the membership-interval cases.
ONE_DAY = timedelta(days=1)

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
        # A different model: request_usage is unique on (date, user_id, model, server,
        # pool_id), so the same user on two user_types has to differ somewhere else.
        self.db.add(RequestUsage(
            date=DAY, user_id=alice.id, pool_id=pool.id, user_identity="alice",
            user_type="api_key", model="other/model", server="openai", request_count=3,
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

    # -- pool_timeseries: the By Pool chart's own series ------------------
    #
    # `timeseries` is every request in the window, which is the right series for By User
    # and By Model -- those re-partition the same traffic -- but not for By Pool, which
    # covers only what was pooled. These pin the difference; without it the chart drew
    # bars the table underneath could not account for.

    async def test_pool_timeseries_excludes_unpooled_traffic(self):
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_user("carol", rpd_limit=100)  # unpooled
        await self.make_pool("team", alice)
        await self.seed_usage("alice", 3)
        await self.seed_usage("carol", 40)

        result = await self._usage(window="30d")

        self.assertEqual(sum(b["count"] for b in result["pool_timeseries"]), 3)
        # The overall series is untouched, so By User and By Model keep theirs.
        self.assertEqual(sum(b["count"] for b in result["timeseries"]), 43)
        # Same buckets either way, so switching tabs moves the bars, not the axis.
        self.assertEqual(
            [b["label"] for b in result["pool_timeseries"]],
            [b["label"] for b in result["timeseries"]],
        )

    async def test_pool_timeseries_excludes_a_members_pre_join_traffic(self):
        """The same pool_id scoping per_pool's counts use: joining a pool doesn't
        retroactively make everything you ever sent the pool's."""
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_pool("team", alice, joined_on=DAY)
        await self.seed_usage("alice", 7, day=DAY - 5 * ONE_DAY, pool_id=0)
        await self.seed_usage("alice", 3, day=DAY)

        result = await self._usage(window="30d")

        self.assertEqual(sum(b["count"] for b in result["pool_timeseries"]), 3)
        self.assertEqual(sum(b["count"] for b in result["timeseries"]), 10)

    async def test_a_member_who_switched_pools_contributes_to_each_in_turn(self):
        """Every row carries exactly one pool, so a user who left one pool for another
        mid-window has each half of their traffic credited to the pool they were in when
        they sent it, and the chart -- all pooled rows -- carries every request once.
        Nothing is credited twice, not even on the changeover day.

        A schema UNIQUE keeps anyone from being in two pools concurrently, so switching
        is the only way one user reaches the series through two pools at all."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        old_day, switch_day = DAY - 4 * ONE_DAY, DAY - 2 * ONE_DAY

        first = await self.make_pool("first", alice, joined_on=old_day)
        second = await self.make_pool("second", bob, joined_on=old_day)
        await self.seed_usage("alice", 5, day=old_day)                 # while in `first`
        await self.seed_usage("alice", 3, day=switch_day)              # still in `first`
        await self.leave_pool_on(first, alice, left_on=switch_day)
        await self.join_pool(second, alice, joined_on=switch_day)
        await self.seed_usage("alice", 1, day=switch_day, model="p/x")  # now in `second`
        await self.seed_usage("alice", 2, day=DAY)                     # while in `second`

        result = await self._usage(window="30d")

        # 11 requests were made and 11 are on the chart.
        self.assertEqual(sum(b["count"] for b in result["pool_timeseries"]), 11)
        self.assertEqual(
            {p["name"]: p["request_count"] for p in result["per_pool"]},
            {"first": 8, "second": 3},
        )

    async def test_rows_of_a_deleted_pool_are_not_a_phantom_pool(self):
        """Usage keeps the pool_id of a pool that no longer exists. It stays in the
        members' own totals, but the By Pool table lists only pools that still exist."""
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        gone = await self.make_pool("gone", alice)
        await self.make_pool("kept", bob)
        await self.seed_usage("alice", 5)   # stamped with `gone`
        await self.seed_usage("bob", 2)

        await pool_routes.admin_delete_pool(self.db, gone.id)

        result = await self._usage(window="30d")
        self.assertEqual([p["name"] for p in result["per_pool"]], ["kept"])
        self.assertEqual(sum(b["count"] for b in result["pool_timeseries"]), 2)
        self.assertEqual(result["totals"]["requests"], 7, "alice's rows are still hers")

    async def test_no_pools_is_an_empty_series_not_the_overall_one(self):
        """With no stints to match, the restriction has to match nothing. Falling back
        to "unrestricted" would draw every request under a By Pool table with no rows."""
        await self.make_user("carol", rpd_limit=100)
        await self.seed_usage("carol", 9)

        result = await self._usage(window="30d")

        self.assertEqual(result["per_pool"], [])
        self.assertEqual(sum(b["count"] for b in result["pool_timeseries"]), 0)
        # Zero-filled, not empty: the chart keeps its axis instead of collapsing.
        self.assertEqual(
            [b["label"] for b in result["pool_timeseries"]],
            [b["label"] for b in result["timeseries"]],
        )

    async def test_per_pool_is_absent_from_a_drill_down_payload(self):
        alice = await self.make_user("alice", rpd_limit=100)
        await self.make_pool("team", alice)
        await self.seed_usage("alice", 4)

        drill = await admin_routes.get_usage(
            view="user", id="alice", window="today", year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )
        self.assertNotIn("per_pool", drill)
        self.assertNotIn("pool_timeseries", drill)


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


class AdminUsageLabelResolutionTests(PoolTestCase):
    """The config admin is not a users row, so its usage keeps whatever ADMIN_USERNAME
    was in force when each row was written and nothing relabels it afterwards.

    is_reserved_username only recognises the *current* name, so without a fallback a
    rename or a disable strands that history: still listed in the By User table, but
    neither openable nor deletable.
    """

    async def _admin_row(self, label, count=5):
        self.db.add(RequestUsage(
            date=DAY, user_id=ADMIN_USAGE_USER_ID, pool_id=0, user_identity=label,
            user_type="admin", model="p/m", server="openai", request_count=count,
        ))
        await self.db.commit()

    async def _resolve(self, name, *, enabled=True, admin_name="root"):
        with patch("app.auth.admin.is_admin_enabled", return_value=enabled), \
             patch("app.auth.admin.get_admin_username", return_value=admin_name):
            return await admin_routes._resolve_usage_user_id(self.db, name)

    async def test_the_current_admin_name_resolves(self):
        await self._admin_row("root")

        self.assertEqual(await self._resolve("root"), ADMIN_USAGE_USER_ID)

    async def test_a_renamed_admins_old_label_still_resolves(self):
        await self._admin_row("old-admin")

        self.assertEqual(await self._resolve("old-admin"), ADMIN_USAGE_USER_ID)

    async def test_a_disabled_admins_label_still_resolves(self):
        await self._admin_row("root")

        self.assertEqual(await self._resolve("root", enabled=False), ADMIN_USAGE_USER_ID)

    async def test_a_real_account_of_the_same_name_wins(self):
        """`users` is checked first, so an account that legitimately holds a name the
        admin once used keeps its own rows."""
        alice = await self.make_user("alice")
        await self._admin_row("alice")

        self.assertEqual(await self._resolve("alice"), alice.id)

    async def test_a_real_account_holding_the_current_admin_name_wins(self):
        """is_reserved_username only blocks the name while the admin is enabled, so an
        account can hold the admin's current name. It must keep its own rows, as the
        rekey migration decides the same collision."""
        root = await self.make_user("root")
        await self._admin_row("root")

        self.assertEqual(await self._resolve("root"), root.id)

    async def test_an_explicit_user_id_disambiguates_a_shared_label(self):
        root = await self.make_user("root")
        await self._admin_row("root")

        with patch("app.auth.admin.is_admin_enabled", return_value=True), \
             patch("app.auth.admin.get_admin_username", return_value="root"):
            as_admin = await admin_routes._resolve_usage_user_id(
                self.db, "root", ADMIN_USAGE_USER_ID)
            as_user = await admin_routes._resolve_usage_user_id(self.db, "root", root.id)

        self.assertEqual(as_admin, ADMIN_USAGE_USER_ID)
        self.assertEqual(as_user, root.id)

    async def test_an_unknown_user_id_is_a_404(self):
        with self.assertRaises(HTTPException) as caught:
            await admin_routes._resolve_usage_user_id(self.db, "ghost", 999)
        self.assertEqual(caught.exception.status_code, 404)

    async def test_deleting_by_user_id_leaves_the_same_named_account_alone(self):
        root = await self.make_user("root")
        await self._admin_row("root", count=9)
        await self.seed_usage("root", 4)

        with patch("app.auth.admin.is_admin_enabled", return_value=True), \
             patch("app.auth.admin.get_admin_username", return_value="root"):
            result = await admin_routes.delete_usage(
                view="user", id="root", user_id=ADMIN_USAGE_USER_ID,
                current_admin=ADMIN, db=self.db,
            )
            remaining = await admin_routes.get_usage(
                view="user", id="root", user_id=root.id, window="today",
                year=None, month=None, current_admin=ADMIN, db=self.db,
            )

        self.assertEqual(result["total"], 9)
        self.assertEqual(sum(r["request_count"] for r in remaining["breakdown"]), 4)

    async def test_a_name_with_no_account_and_no_admin_rows_is_a_404(self):
        with self.assertRaises(HTTPException) as caught:
            await self._resolve("nobody")
        self.assertEqual(caught.exception.status_code, 404)

    async def test_a_stranded_admin_label_can_be_purged(self):
        await self._admin_row("old-admin", count=9)

        with patch("app.auth.admin.is_admin_enabled", return_value=True), \
             patch("app.auth.admin.get_admin_username", return_value="root"):
            result = await admin_routes.delete_usage(
                view="user", id="old-admin", current_admin=ADMIN, db=self.db,
            )

        self.assertEqual(result["total"], 9)
        remaining = await admin_routes.get_usage(
            view=None, id=None, window="today", year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )
        self.assertEqual(remaining["totals"]["requests"], 0)


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

    async def test_purging_a_pool_clears_what_the_pool_consumed(self):
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

    async def test_purging_a_pool_keeps_what_members_sent_outside_it(self):
        """The number the admin saw in the By Pool table is what goes; a member's
        pre-join history is not the pool's and stays."""
        alice, _bob, pool = await self._team()
        await self.seed_usage("alice", 40, day=DAY - ONE_DAY, pool_id=0)

        result = await self._delete("pool", pool.id)

        self.assertEqual(result["total"], 10)
        remaining = await admin_routes.get_usage(
            view=None, id=None, window="7d", year=None, month=None,
            current_admin=ADMIN, db=self.db,
        )
        self.assertEqual(remaining["totals"]["requests"], 40)
        self.assertEqual(remaining["per_pool"][0]["request_count"], 0)

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

    async def test_purging_a_pool_re_settles_a_member_who_has_since_left(self):
        """The purge takes every row stamped with the pool, including those of someone
        who has moved on since -- so the re-settle has to cover them too.

        Their new pool is charged against what they sent today *wherever they were*
        (_member_row_counts is deliberately not filtered by pool_id), so deleting those
        rows without re-settling leaves that pool billing for traffic that is gone.
        Settling only today's members of the purged pool misses them entirely.
        """
        from app.auth import pools as pool_settlement

        alice, _bob, team = await self._team()
        carol = await self.make_user("carol", rpd_limit=100)
        dave = await self.make_user("dave", rpd_limit=100)
        other = await self.make_pool("other", carol, [dave])
        # Dave spent the morning in `team` and has since joined `other`; his rows still
        # carry the pool he was in when they completed.
        await self.seed_usage("dave", 6, pool_id=team.id)
        await pool_settlement.settle_pool(self.db, other.id)
        await self.db.commit()
        self.assertEqual(await self.charged(other.id, dave.id), 3)

        await self._delete("pool", team.id)

        self.assertEqual(await self.charged(other.id, dave.id), 0)
        self.assertEqual(await self.charged(other.id, carol.id), 0)
        self.assertEqual(await self.effective_used(dave), 0)
        self.assertEqual(await self.effective_used(carol), 0)

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

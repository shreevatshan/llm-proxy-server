"""Tests for the admin's pool policy: max members per pool and the per-member share cap.

Both settings are global defaults a pool may override. The share cap is the subtle one:
a member may consume at most max(own limit, pct x pool limit) of the pool on each tier,
measured by their settled share (rows + carry) so that moving between pools mid-day
never counts traffic the previous pool already absorbed. The pool's own total is still
checked first, so the cap can only ever tighten what the pool allows.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.auth.admin import AdminUser
from app.auth.database import load_pool_settings, upsert_pool_settings
from app.auth.models import (
    MAX_POOL_MEMBERS, AdminPoolLimitsUpdate, PoolCreate, PoolInviteCreate,
    PoolSettingsUpdate, RequestPool, RequestPoolInvitation, RequestPoolMember,
)
from app.rate_limit import _limit_message, _rl_headers
from app.routes import admin as admin_routes
from app.routes import pools as pool_routes
from tests.pool_test_base import PoolTestCase


ADMIN = AdminUser(username="admin", email="admin@example.com")


class PoolSettingsTests(PoolTestCase):

    async def test_defaults_when_never_saved(self):
        settings = await load_pool_settings(self.db)
        self.assertEqual(settings.max_members, MAX_POOL_MEMBERS)
        self.assertIsNone(settings.max_member_share_pct)

    async def test_admin_update_round_trips_and_reaches_the_tracker(self):
        saved = await admin_routes.admin_update_pool_settings(
            PoolSettingsUpdate(max_members=10, max_member_share_pct=40), ADMIN, self.db,
        )
        self.assertEqual((saved.max_members, saved.max_member_share_pct), (10, 40))
        self.assertEqual(saved.updated_by, "admin")

        read = await admin_routes.admin_get_pool_settings(ADMIN, self.db)
        self.assertEqual((read.max_members, read.max_member_share_pct), (10, 40))
        self.assertEqual(self.tracker._pool_share_default, 40)

    def test_bounds_are_validated(self):
        for bad in ({"max_members": 1}, {"max_members": 101},
                    {"max_members": 5, "max_member_share_pct": -1},
                    {"max_members": 5, "max_member_share_pct": 101}):
            with self.subTest(payload=bad), self.assertRaises(ValidationError):
                PoolSettingsUpdate(**bad)
        PoolSettingsUpdate(max_members=2, max_member_share_pct=0)
        PoolSettingsUpdate(max_members=100, max_member_share_pct=None)

    def test_override_payload_tells_omitted_from_null(self):
        self.assertEqual(AdminPoolLimitsUpdate().model_fields_set, set())
        cleared = AdminPoolLimitsUpdate(max_members=None)
        self.assertEqual(cleared.model_fields_set, {"max_members"})


class MaxMembersTests(PoolTestCase):

    async def _pool_with(self, owner, *others):
        await pool_routes.create_pool(PoolCreate(name="team"), owner, self.db)
        pool = (await self.db.execute(select(RequestPool))).scalar_one()
        for user in others:
            self.db.add(RequestPoolMember(pool_id=pool.id, user_id=user.id))
        await self.db.commit()
        return pool

    async def _invite(self, inviter, name):
        return await pool_routes.create_invite(PoolInviteCreate(username=name), inviter, self.db)

    async def test_global_setting_caps_invites(self):
        await upsert_pool_settings(self.db, 3, None, "admin")
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.make_user("dave", rpd_limit=5)
        await self._pool_with(alice, bob, carol)

        with self.assertRaises(HTTPException) as ctx:
            await self._invite(alice, "dave")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("(3 members)", ctx.exception.detail)

    async def test_per_pool_override_wins_and_null_inherits_again(self):
        await upsert_pool_settings(self.db, 3, None, "admin")
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.make_user("dave", rpd_limit=5)
        await self.make_user("erin", rpd_limit=5)
        pool = await self._pool_with(alice, bob, carol)

        await admin_routes.admin_update_pool_limits(
            AdminPoolLimitsUpdate(max_members=5), pool.id, ADMIN, self.db,
        )
        await self._invite(alice, "dave")   # room under the override

        await admin_routes.admin_update_pool_limits(
            AdminPoolLimitsUpdate(max_members=None), pool.id, ADMIN, self.db,
        )
        with self.assertRaises(HTTPException) as ctx:
            await self._invite(alice, "erin")
        self.assertIn("(3 members)", ctx.exception.detail)

    async def test_lowering_below_current_size_removes_nobody(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.make_user("dave", rpd_limit=5)
        pool = await self._pool_with(alice, bob, carol)

        await admin_routes.admin_update_pool_limits(
            AdminPoolLimitsUpdate(max_members=2), pool.id, ADMIN, self.db,
        )
        members = (await self.db.execute(
            select(RequestPoolMember).where(RequestPoolMember.pool_id == pool.id)
        )).scalars().all()
        self.assertEqual(len(members), 3)
        with self.assertRaises(HTTPException):
            await self._invite(alice, "dave")

        listing = (await pool_routes.admin_list_pools(self.db))[0]
        self.assertEqual(listing.member_count, 3)
        self.assertEqual(listing.max_members_override, 2)
        self.assertEqual(listing.effective_max_members, 2)

    async def test_accept_rechecks_the_override_set_after_the_invite(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        pool = await self._pool_with(alice, bob)
        await self._invite(alice, "carol")
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()

        await admin_routes.admin_update_pool_limits(
            AdminPoolLimitsUpdate(max_members=2), pool.id, ADMIN, self.db,
        )
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.accept_invite(invite.id, carol, self.db)
        self.assertIn("(2 members)", ctx.exception.detail)

    async def test_my_pool_reports_the_effective_values(self):
        await upsert_pool_settings(self.db, 7, 30, "admin")
        alice = await self.make_user("alice", rpd_limit=5)
        await self._pool_with(alice)

        mine = await pool_routes.get_my_pool(alice, self.db)
        self.assertEqual((mine.max_members, mine.max_member_share_pct), (7, 30))


class ShareCapTests(PoolTestCase):

    async def _share(self, pct):
        await upsert_pool_settings(self.db, MAX_POOL_MEMBERS, pct, "admin")
        await self.refresh()

    async def _trio(self, limit=10):
        alice = await self.make_user("alice", rpd_limit=limit)
        bob = await self.make_user("bob", rpd_limit=limit)
        carol = await self.make_user("carol", rpd_limit=limit)
        pool = await self.make_pool("team", alice, [bob, carol])
        return pool, alice, bob, carol

    async def test_member_is_stopped_at_their_share_of_the_overall_quota(self):
        await self._share(50)
        _pool, alice, bob, _carol = await self._trio()   # pool 30, cap max(10, 15) = 15
        await self.seed_usage("alice", 15)
        await self.refresh()

        decision = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.limited_by, "rpd")
        self.assertEqual((decision.pool_share_cap, decision.pool_share_pct), (15, 50))
        self.assertIn("your share", _limit_message(decision))
        self.assertEqual(_rl_headers(decision)["X-RateLimit-Limit-Requests"], "15")

        self.assertTrue((await self.tracker.check_and_increment(bob.id, "bob")).allowed,
                        "the pool still has 15 left for everyone else")

    async def test_member_below_their_share_is_allowed(self):
        await self._share(50)
        _pool, alice, _bob, _carol = await self._trio()
        await self.seed_usage("alice", 14)
        await self.refresh()
        self.assertTrue((await self.tracker.check_and_increment(alice.id, "alice")).allowed)

    async def test_floor_is_the_members_own_limit(self):
        await self._share(10)   # 10% of 30 = 3, but alice's own limit is 10
        _pool, alice, _bob, _carol = await self._trio()
        await self.seed_usage("alice", 9)
        await self.refresh()
        self.assertTrue((await self.tracker.check_and_increment(alice.id, "alice")).allowed)

        await self.seed_usage("alice", 1, server="anthropic")   # a distinct usage row
        await self.refresh()
        decision = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.pool_share_cap, 10)

    async def test_zero_percent_means_only_their_own_limit(self):
        await self._share(0)
        _pool, alice, _bob, _carol = await self._trio()
        await self.seed_usage("alice", 10)
        await self.refresh()
        self.assertEqual(
            (await self.tracker.check_and_increment(alice.id, "alice")).pool_share_cap, 10)

    async def test_pool_total_is_still_checked_first(self):
        await self._share(80)
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        await self.make_pool("team", alice, [bob])   # pool 20, cap 16
        await self.seed_usage("bob", 15)
        await self.seed_usage("alice", 5)
        await self.refresh()

        decision = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertFalse(decision.allowed)
        self.assertIsNone(decision.pool_share_cap, "it was the pool that ran dry")

    async def test_no_percent_or_100_percent_means_uncapped(self):
        for pct in (None, 100):
            with self.subTest(pct=pct):
                self.assertIsNone(self.tracker.member_share_cap(None, 10, 30))
        await self._share(100)
        _pool, alice, _bob, _carol = await self._trio()
        await self.seed_usage("alice", 29)
        await self.refresh()
        self.assertTrue((await self.tracker.check_and_increment(alice.id, "alice")).allowed)

    async def test_per_pool_override_beats_the_global_default(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await admin_routes.admin_update_pool_limits(
            AdminPoolLimitsUpdate(max_member_share_pct=50), pool.id, ADMIN, self.db,
        )
        await self.seed_usage("alice", 10)
        await self.refresh()

        decision = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.pool_share_cap, 10)

    async def test_deactivated_member_leaves_the_capped_pool_limit(self):
        await self._share(50)
        _pool, alice, _bob, carol = await self._trio()   # 30 -> 20 once carol is off
        await self.deactivate(carol)
        await self.seed_usage("alice", 10)
        await self.refresh()

        decision = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.pool_share_cap, 10)

    async def test_model_group_share(self):
        group = await self.make_model_group("chat", "p/chat", rpd_default=10)
        await self._share(50)
        _pool, alice, bob, _carol = await self._trio()
        await self.seed_usage("alice", 15, model="p/chat")
        await self.refresh()

        decision = await self.tracker.check_group_limit(alice.id, "alice", "p/chat")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.limited_by, "group_rpd")
        self.assertEqual(decision.pool_share_cap, 15)
        self.assertIn("model group 'chat'", _limit_message(decision))
        self.assertIsNone(await self.tracker.check_group_limit(bob.id, "bob", "p/chat"))
        self.assertEqual(group.name, "chat")

    async def test_instance_group_share(self):
        await self.make_instance_group("fast", "fastprov", rpd_default=10)
        await self._share(50)
        _pool, alice, bob, _carol = await self._trio()
        await self.seed_usage("alice", 15, model="fastprov/m")
        await self.refresh()

        decision = await self.tracker.check_instance_group_limit(alice.id, "alice", "fastprov")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.limited_by, "instance_group_rpd")
        self.assertEqual(decision.pool_share_cap, 15)
        self.assertIsNone(await self.tracker.check_instance_group_limit(bob.id, "bob", "fastprov"))

    async def test_mover_is_measured_by_settled_share_not_raw_rows(self):
        """Heavy use in a pool she has left must not lock Alice out of the next one.

        In P1 her 200 requests are re-split across four equal members on leaving, so she
        carries only 50 of them. Raw rows would deny her in P2 at once (200 >= cap 200).
        """
        await self._share(50)
        alice = await self.make_user("alice", rpd_limit=100)
        p1_others = [await self.make_user(n, rpd_limit=100) for n in ("b1", "c1", "d1")]
        await self.make_pool("one", alice, p1_others)
        await self.seed_usage("alice", 200)
        await pool_routes.leave_pool(alice, self.db)

        erin = await self.make_user("erin", rpd_limit=100)
        p2_others = [await self.make_user(n, rpd_limit=100) for n in ("f2", "g2")]
        await self.make_pool("two", erin, p2_others)
        await pool_routes.create_invite(PoolInviteCreate(username="alice"), erin, self.db)
        invite = (await self.db.execute(
            select(RequestPoolInvitation).where(RequestPoolInvitation.status == "pending")
        )).scalar_one()
        await pool_routes.accept_invite(invite.id, alice, self.db)
        await self.refresh()

        cap, used = await self.tracker.member_share_status(alice.id, "alice", "overall", 0, 400)
        self.assertEqual(cap, 200)
        self.assertLess(used, cap)
        self.assertTrue((await self.tracker.check_and_increment(alice.id, "alice")).allowed)

    async def test_share_figures_are_rendered_for_members_and_admin(self):
        await self._share(50)
        _pool, alice, _bob, _carol = await self._trio()
        await self.seed_usage("alice", 4)
        await self.refresh()

        listing = (await pool_routes.admin_list_pools(self.db))[0]
        self.assertEqual(listing.effective_max_member_share_pct, 50)
        self.assertIsNone(listing.max_member_share_pct_override)
        alice_row = next(m for m in listing.members if m.username == "alice")
        overall = next(s for s in alice_row.scopes if s.scope_kind == "overall")
        self.assertEqual((overall.share_cap, overall.share_used), (15, 4))

        cap, used = await self.tracker.member_share_status(alice.id, "alice", "overall", 0, 30)
        self.assertEqual((cap, used), (15, 4))

    async def test_admin_kick_removes_and_settles(self):
        _pool, alice, bob, _carol = await self._trio()
        await self.seed_usage("bob", 6)

        result = await admin_routes.admin_remove_pool_member(bob.id, ADMIN, self.db)
        self.assertEqual(result["message"], "Removed bob")
        self.assertIsNone(
            (await self.db.execute(
                select(RequestPoolMember).where(RequestPoolMember.user_id == bob.id)
            )).scalar_one_or_none()
        )
        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 20, "bob's limit left with him")


class PoolOverrideMigrationTests(unittest.IsolatedAsyncioTestCase):
    """A pre-existing request_pools table gains the two nullable override columns.

    The old table is built with raw DDL first: create_all skips tables that already
    exist, so it keeps the old shape and the ALTER path is what gets exercised.
    """

    OLD_POOLS = """
    CREATE TABLE request_pools (
        id INTEGER NOT NULL PRIMARY KEY,
        name VARCHAR(64) NOT NULL UNIQUE,
        description VARCHAR(256),
        owner_user_id INTEGER NOT NULL,
        created_at DATETIME,
        updated_at DATETIME
    )"""

    async def asyncSetUp(self):
        from app.auth import database
        from app.auth.models import Base

        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with sqlite3.connect(self.path) as raw:
            raw.execute(self.OLD_POOLS)
            raw.execute("INSERT INTO request_pools (id, name, owner_user_id) VALUES (1, 'legacy', 1)")
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "DATABASE_URL", f"sqlite+aiosqlite:///{self.path}"),
            patch("app.auth.admin.is_admin_enabled", return_value=True),
            patch("app.auth.admin.get_admin_username", return_value="root"),
        ]
        for p in self._patches:
            p.start()

    async def asyncTearDown(self):
        for p in self._patches:
            p.stop()
        await self.engine.dispose()
        for name in os.listdir(os.path.dirname(self.path)):
            full = os.path.join(os.path.dirname(self.path), name)
            if full.startswith(self.path):
                os.unlink(full)

    async def test_override_columns_are_added_and_existing_pools_inherit(self):
        from app.auth import database

        await database._run_auto_migrations()
        with sqlite3.connect(self.path) as raw:
            columns = [row[1] for row in raw.execute("PRAGMA table_info(request_pools)")]
            row = raw.execute(
                "SELECT max_members, max_member_share_pct FROM request_pools WHERE id = 1"
            ).fetchone()
            settings_columns = [row[1] for row in raw.execute("PRAGMA table_info(pool_settings)")]
        self.assertIn("max_members", columns)
        self.assertIn("max_member_share_pct", columns)
        self.assertEqual(row, (None, None), "an existing pool inherits the global values")
        self.assertIn("max_member_share_pct", settings_columns)

        await database._run_auto_migrations()   # a second start finds nothing to do

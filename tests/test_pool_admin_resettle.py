"""Re-apportioning pools after an admin edit, and the limits of what that can fix.

Every admin route that moves a limit or a group's membership now runs the pools it
touches through `resettle_pools`. The ordering is the fragile part and is why one shared
helper exists instead of seventeen bespoke edits: settlement resolves limits out of the
tracker snapshot rather than the database, so it reloads the snapshot *first*, settles,
then reloads again to pull the rewritten carries back in.

What re-settling does and does not do is the other thing worth pinning. Apportionment is
incremental -- it splits each new delta by the weights in force at the time -- so a
limit edit changes how the *next* interval is divided, not how an interval that has
already closed was. Retroactively re-clamping would break SUM(charged) == pool_used and
hand the pool quota its members never earned. What genuinely does go stale is the scope
partition: regrouping a model moves rows between scopes, which is a real delta on both.
"""

from app.auth.admin import AdminUser
from app.auth.models import UserRateLimitUpdate
from app.routes import admin as admin_routes
from app.routes.pools import resettle_pools
from tests.pool_test_base import PoolTestCase


ADMIN = AdminUser(username="root", email="root@example.test")


class AdminResettleTests(PoolTestCase):

    async def test_an_unpooled_user_is_not_settled_at_all(self):
        alice = await self.make_user("alice", rpd_limit=10)
        await self.seed_usage("alice", 3)

        await admin_routes.update_user_rate_limit(
            UserRateLimitUpdate(rpd_limit=30), alice.id, ADMIN, self.db,
        )

        self.assertEqual(await self.carry(alice.id), 0,
                         "an ordinary limit edit must not pay for a settlement pass")
        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual((status.rpd_limit, status.rpd_count), (30, 3))

    async def test_a_closed_interval_is_not_re_apportioned_behind_the_members(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await resettle_pools(self.db, user_ids=[alice.id])
        self.assertEqual(await self.charged(pool.id, alice.id), 4)   # split 10:10

        await admin_routes.update_user_rate_limit(
            UserRateLimitUpdate(rpd_limit=30), alice.id, ADMIN, self.db,
        )

        self.assertEqual(await self.charged(pool.id, alice.id), 4,
                         "those eight were consumed under the old weights and stay split "
                         "under them; re-clamping would invent or destroy quota")
        self.assertEqual(await self.charged(pool.id, bob.id), 4)
        self.assertEqual(
            (await self.charged(pool.id, alice.id)) + (await self.charged(pool.id, bob.id)),
            8, "SUM(charged) still equals what the pool consumed",
        )

    async def test_the_next_interval_uses_the_new_weights(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await resettle_pools(self.db, user_ids=[alice.id])

        await admin_routes.update_user_rate_limit(
            UserRateLimitUpdate(rpd_limit=30), alice.id, ADMIN, self.db,
        )
        await self.seed_usage("bob", 4)
        await resettle_pools(self.db, user_ids=[alice.id])

        # 4 more on 30:10 weights -> 3 to Alice, 1 to Bob, on top of the 4/4 above.
        self.assertEqual(await self.charged(pool.id, alice.id), 7)
        self.assertEqual(await self.charged(pool.id, bob.id), 5)

    async def test_the_limiter_sees_the_edit_without_waiting_for_the_rpd_ttl(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.tracker.get_user_status(alice.id, "alice")   # warm the RPD cache

        await admin_routes.update_user_rate_limit(
            UserRateLimitUpdate(rpd_limit=30), alice.id, ADMIN, self.db,
        )

        # No refresh() here on purpose: the helper's own trailing reload is what has to
        # have landed, or the cache serves the pre-edit numbers for _RPD_TTL.
        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 40)
        self.assertEqual(status.rpd_count, 8)

    async def test_grouping_a_model_moves_the_charge_onto_the_group_scope(self):
        """The edit that genuinely leaves the ledger describing a partition that is gone.

        Each usage row folds into exactly one scope, so putting a model into a group
        moves its rows off 'overall'. Both scopes see a real delta -- a refund on one,
        a charge on the other -- which is what settlement is for.
        """
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await resettle_pools(self.db, user_ids=[alice.id])
        self.assertEqual(await self.charged(pool.id, alice.id), 4, "ungrouped -> overall")

        mg = await self.make_model_group("grouped", "openai/gpt-4o", rpd_default=10)
        await resettle_pools(self.db, all_pools=True)

        scope = ("model_group", mg.id)
        self.assertEqual(await self.charged(pool.id, alice.id), 0,
                         "the overall scope is refunded; those rows are governed "
                         "by the group's own limit now")
        self.assertEqual(await self.charged(pool.id, bob.id), 0)
        self.assertEqual(await self.charged(pool.id, alice.id, scope), 4)
        self.assertEqual(await self.charged(pool.id, bob.id, scope), 4)

    async def test_deactivating_a_member_shrinks_the_pool_and_freezes_their_charge(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await resettle_pools(self.db, user_ids=[alice.id])
        self.assertEqual(await self.charged(pool.id, bob.id), 4)

        await admin_routes.delete_user(bob.id, ADMIN, self.db)   # soft delete

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 10, "Bob's ten is not quota anyone can spend")
        self.assertEqual(status.rpd_count, 8)
        self.assertEqual(status.rpd_remaining, 2)
        self.assertEqual(await self.charged(pool.id, bob.id), 4,
                         "his share of what was already sent stays with him")

        # And he absorbs none of what the pool spends from here on. A second model
        # keeps the seeded row distinct; both are ungrouped, so both land on 'overall'.
        await self.seed_usage("alice", 2, model="p/m2")
        await resettle_pools(self.db, user_ids=[alice.id])
        self.assertEqual(await self.charged(pool.id, bob.id), 4)
        self.assertEqual(await self.charged(pool.id, alice.id), 6)

    async def test_reactivating_a_member_gives_the_pool_its_quota_back(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await admin_routes.delete_user(bob.id, ADMIN, self.db)
        self.assertEqual(
            (await self.tracker.get_user_status(alice.id, "alice")).rpd_limit, 10)

        await admin_routes.activate_user(bob.id, ADMIN, self.db)

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 20)
        self.assertEqual(status.rpd_count, 8)

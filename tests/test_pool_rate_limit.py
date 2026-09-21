"""Tests for the pooled RPD arithmetic in the rate-limit tracker.

A pool's daily limit is the sum of its members' own limits and its count is the sum of
their consumption, on all three tiers. The two things most likely to break silently:
RPM leaking into the pool (it must stay strictly per-user, since a shared per-minute
budget would let one member stall the rest), and the snapshot going stale after a
rename, since usage rows are keyed by the username string rather than the user id.
"""

from app.auth.models import (
    InstanceGroup, InstanceGroupMember, ModelGroup, ModelGroupMember,
    RequestPoolMember, UserRateLimit, UserRpdCarry,
)
from tests.pool_test_base import DAY, PoolTestCase


class PooledLimitTests(PoolTestCase):

    async def test_pool_rpd_limit_is_the_sum_of_member_limits(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=7)
        await self.make_pool("team", alice, [bob])

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 12)
        self.assertEqual(
            (await self.tracker.get_user_status(bob.id, "bob")).rpd_limit, 12,
            "both members see the same shared ceiling",
        )

    async def test_one_unlimited_member_makes_the_pool_unlimited(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=None)   # inherits the null default
        await self.make_pool("team", alice, [bob])

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertIsNone(status.rpd_limit)
        self.assertIsNone(status.rpd_remaining)

    async def test_pooled_rpd_counts_every_members_usage(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        status = await self.tracker.get_user_status(bob.id, "bob")
        self.assertEqual(status.rpd_count, 9, "Bob sees the pool's consumption, not his own")
        self.assertEqual(status.rpd_remaining, 1)

    async def test_an_unpooled_user_is_unaffected(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4)

        status = await self.tracker.get_user_status(carol.id, "carol")
        self.assertEqual((status.rpd_limit, status.rpd_count), (5, 0))

    async def test_rpm_stays_per_user_when_pooled(self):
        # The core guarantee of the design: sharing a day's budget must not share the
        # minute's, or one member's burst would block everyone else's next request.
        alice = await self.make_user("alice")
        bob = await self.make_user("bob")
        self.db.add_all([
            UserRateLimit(user_id=alice.id, rpm_limit=2, rpd_limit=5),
            UserRateLimit(user_id=bob.id, rpm_limit=2, rpd_limit=5),
        ])
        await self.db.commit()
        await self.make_pool("team", alice, [bob])

        self.assertEqual(
            (await self.tracker.get_user_status(alice.id, "alice")).rpm_limit, 2,
            "the pool does not sum RPM",
        )
        # Alice exhausts her own minute.
        for _ in range(2):
            decision = await self.tracker.check_and_increment(alice.id, "alice")
            self.assertTrue(decision.allowed)
        self.assertFalse((await self.tracker.check_and_increment(alice.id, "alice")).allowed)
        # Bob's minute is untouched.
        self.assertTrue((await self.tracker.check_and_increment(bob.id, "bob")).allowed)

    async def test_pooled_rpd_is_enforced_against_the_shared_ceiling(self):
        alice = await self.make_user("alice", rpd_limit=2)
        bob = await self.make_user("bob", rpd_limit=2)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4)

        decision = await self.tracker.check_and_increment(bob.id, "bob")
        self.assertFalse(decision.allowed, "Alice spent the pool's whole day")
        self.assertEqual(decision.limited_by, "rpd")

    async def test_carries_adjust_the_pooled_count(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        self.db.add_all([
            UserRpdCarry(user_id=alice.id, usage_date=DAY,
                         scope_kind="overall", scope_id=0, carry=-3),
            UserRpdCarry(user_id=bob.id, usage_date=DAY,
                         scope_kind="overall", scope_id=0, carry=3),
        ])
        await self.db.commit()
        await self.refresh()

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_count, 8,
                         "carries net to zero across a settled pool's members")

    async def test_a_carry_from_a_previous_day_is_ignored(self):
        from datetime import date

        alice = await self.make_user("alice", rpd_limit=5)
        await self.make_pool("solo", alice)
        await self.seed_usage("alice", 2)
        self.db.add(UserRpdCarry(
            user_id=alice.id, usage_date=date(2026, 4, 6),
            scope_kind="overall", scope_id=0, carry=3,
        ))
        await self.db.commit()
        await self.refresh()

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_count, 2, "yesterday's adjustment never applies")

    async def test_pool_members_share_one_rpd_cache_entry(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 3)

        await self.tracker.get_user_status(alice.id, "alice")
        await self.tracker.get_user_status(bob.id, "bob")

        from app.rate_limit import _user_scope_key

        self.assertIn(f"pool:{pool.id}", self.tracker._rpd_cache)
        self.assertNotIn(_user_scope_key("alice"), self.tracker._rpd_cache)
        self.assertNotIn(_user_scope_key("bob"), self.tracker._rpd_cache,
                         "one DB read per pool per TTL, not one per member")

    async def test_a_user_named_like_a_pool_key_gets_its_own_cache_entry(self):
        """Usernames are validated for length only, so "pool:1" is registrable.

        Both keyspaces share one dict, so without a prefix on the user side that
        user's day count would be served to every member of pool 1 for the length
        of the TTL — and the pool's count back to them.
        """
        from app.rate_limit import _user_scope_key

        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])

        impostor_name = f"pool:{pool.id}"
        impostor = await self.make_user(impostor_name, rpd_limit=5)
        await self.seed_usage(impostor_name, 4)
        await self.seed_usage("alice", 1)
        await self.refresh()

        impostor_status = await self.tracker.get_user_status(impostor.id, impostor_name)
        pool_status = await self.tracker.get_user_status(alice.id, "alice")

        self.assertEqual(impostor_status.rpd_count, 4)
        self.assertEqual(pool_status.rpd_count, 1, "the pool never reads the impostor's count")
        self.assertIn(_user_scope_key(impostor_name), self.tracker._rpd_cache)
        self.assertEqual(self.tracker._rpd_cache[f"pool:{pool.id}"].count, 1)

    async def test_invalidating_a_pool_drops_its_cached_counts(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.tracker.get_user_status(alice.id, "alice")
        self.assertIn(f"pool:{pool.id}", self.tracker._rpd_cache)

        self.tracker.invalidate_pool(pool.id)
        self.assertNotIn(f"pool:{pool.id}", self.tracker._rpd_cache)

    async def test_rename_refreshes_the_pool_snapshot(self):
        # Usage rows are keyed by username, so a stale snapshot would count the pool
        # against a name nobody writes to any more.
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])

        alice.username = "alice2"
        await self.db.commit()
        await self.refresh()

        self.assertEqual(
            {name for _, name in self.tracker._pool_members[pool.id]},
            {"alice2", "bob"},
        )
        self.assertEqual(self.tracker._identity_to_pool.get("alice2"), pool.id)

    async def test_pooled_rpd_overshoot_is_bounded(self):
        # check_and_increment holds a PER-USER lock, so two members can both pass the
        # RPD check concurrently. The overshoot is bounded by the number of members
        # racing — it is the existing single-user TTL window multiplied by pool size,
        # not a new class of error. Documented in the module docstring; asserted here
        # so a future change that widens it fails loudly.
        import asyncio

        alice = await self.make_user("alice", rpd_limit=1)
        bob = await self.make_user("bob", rpd_limit=1)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 2)   # the pool is already at its limit of 2

        results = await asyncio.gather(
            self.tracker.check_and_increment(alice.id, "alice"),
            self.tracker.check_and_increment(bob.id, "bob"),
        )
        allowed = [r for r in results if r.allowed]
        self.assertLessEqual(len(allowed), 2, "overshoot cannot exceed the member count")
        self.assertEqual(len(allowed), 0, "a pool already at its limit admits neither")


class PooledGroupLimitTests(PoolTestCase):
    """The group tiers pool the same way the overall tier does."""

    async def _model_group(self, name, model_id, rpd_default):
        group = ModelGroup(name=name, rpm_default=None, rpd_default=rpd_default)
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupMember(group_id=group.id, model_id=model_id))
        await self.db.commit()
        await self.refresh()
        return group

    async def _instance_group(self, name, provider_key, rpd_default):
        group = InstanceGroup(name=name, rpm_default=None, rpd_default=rpd_default)
        self.db.add(group)
        await self.db.flush()
        self.db.add(InstanceGroupMember(group_id=group.id, provider_key=provider_key))
        await self.db.commit()
        await self.refresh()
        return group

    async def test_model_group_rpd_is_pooled(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        group = await self._model_group("grouped", "openai/gpt-4o", rpd_default=3)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4, model="openai/gpt-4o")

        self.assertEqual(
            self.tracker.pooled_group_rpd_limit(alice.id, group.id, instance=False), 6,
            "3 + 3 across the two members",
        )
        count = await self.tracker.get_group_rpd_count(
            bob.id, "bob", ["openai/gpt-4o"], group.id
        )
        self.assertEqual(count, 4, "Bob's group count includes Alice's requests")

        # Under the pooled ceiling of 6 the fifth request is still allowed...
        self.assertIsNone(await self.tracker.check_group_limit(bob.id, "bob", "openai/gpt-4o"))
        # ...but past it the group gate denies.
        await self.seed_usage("bob", 2, model="openai/gpt-4o")
        self.tracker.invalidate_pool(self.tracker._user_to_pool[bob.id])
        decision = await self.tracker.check_group_limit(bob.id, "bob", "openai/gpt-4o")
        self.assertIsNotNone(decision)
        self.assertFalse(decision.allowed)

    async def test_instance_group_rpd_is_pooled_and_takes_precedence(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        await self._model_group("by-model", "azure/gpt-4o", rpd_default=3)
        ig = await self._instance_group("by-instance", "azure", rpd_default=4)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 5, model="azure/gpt-4o")

        self.assertEqual(
            self.tracker.pooled_group_rpd_limit(alice.id, ig.id, instance=True), 8)
        count = await self.tracker.get_instance_group_rpd_count(
            bob.id, "bob", ["azure"], ig.id
        )
        self.assertEqual(count, 5)
        self.assertEqual(
            self.tracker.scope_for_model("azure/gpt-4o"), ("instance_group", ig.id),
            "the instance group wins, matching enforcement order",
        )

    async def test_grouped_requests_do_not_count_against_the_pooled_overall_quota(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await self._model_group("grouped", "openai/gpt-4o", rpd_default=50)
        await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4, model="openai/gpt-4o")
        await self.seed_usage("bob", 1, model="other/model")

        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_count, 1,
                         "only the ungrouped request reaches the overall tier")

    async def test_joining_mid_day_inherits_todays_consumption_and_limit(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice)
        await self.seed_usage("alice", 4)
        await self.seed_usage("bob", 3)

        before = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual((before.rpd_limit, before.rpd_count), (5, 4))

        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=bob.id))
        await self.db.commit()
        self.tracker.invalidate_pool(pool.id)
        await self.refresh()

        after = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual((after.rpd_limit, after.rpd_count), (10, 7),
                         "limit and usage move together, so joining hands out no free quota")

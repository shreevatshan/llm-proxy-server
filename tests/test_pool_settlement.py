"""Tests for §4 settlement: charging a pool member for what the pool spent while they were in it.

Letting a leaver walk away with their own raw row count is the obvious implementation
and it is exploitable: a user who has spent their whole limit could join a fresh pool,
leave, and arrive at the next one with quota that was never theirs. Settlement closes
the interval on every composition change instead, dividing what the pool consumed among
the members who were actually present for it.

The properties these tests defend:

* a member is never charged for consumption that predates their arrival,
* a member is never credited for consumption after their departure,
* SUM(charged) over current members always equals the pool's used,
* no leave creates or destroys total quota.
"""

from datetime import date

from app.auth import pools as pool_settlement
from app.auth.models import (
    InstanceGroup, InstanceGroupMember, ModelGroup, ModelGroupMember,
    RequestPoolLedger, RequestPoolMember, RequestUsage, UserRpdCarry,
)
from tests.pool_test_base import DAY, PoolTestCase


class SettlementTests(PoolTestCase):
    """Example A and the invariants that follow from it."""

    async def test_leaving_charges_a_share_of_the_pool_not_own_consumption(self):
        # Alice sends 8, Bob sends 1, limits 5 + 5. The pool is at 9/10.
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        # 9 split 5:5 is 4.5 each; the remainder goes to the heavier consumer.
        self.assertEqual(await self.charged(pool.id, alice.id), 5)
        self.assertEqual(await self.charged(pool.id, bob.id), 4)
        # Carry is absolute: effective_used == charged.
        self.assertEqual(await self.carry(alice.id), -3)
        self.assertEqual(await self.carry(bob.id), 3)

    async def test_settlement_conserves_total_remaining_quota(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        charged = (await self.charged(pool.id, alice.id)) + (await self.charged(pool.id, bob.id))
        self.assertEqual(charged, 9, "the pool's 9 requests are all accounted for")
        # Total limit 10, total charged 9, so exactly 1 request of quota survives.
        remaining = (5 - await self.charged(pool.id, alice.id)) + (5 - await self.charged(pool.id, bob.id))
        self.assertEqual(remaining, 1)

    async def test_sum_of_charged_equals_pool_used_after_every_composition_change(self):
        alice = await self.make_user("alice", rpd_limit=500)
        bob = await self.make_user("bob", rpd_limit=500)
        carol = await self.make_user("carol", rpd_limit=500)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 500)

        async def pool_used(member_ids):
            rows = await pool_settlement._member_row_counts(
                self.db, [(u.id, u.username) for u in member_ids]
            )
            carries = await pool_settlement._load_carries(
                self.db, [u.id for u in member_ids], DAY
            )
            return sum(
                rows.get((u.id, "overall", 0), 0) + carries.get((u.id, "overall", 0), 0)
                for u in member_ids
            )

        async def assert_invariant(member_ids):
            ledger = await pool_settlement._load_ledger(self.db, pool.id, DAY)
            total = sum(ledger.get((u.id, "overall", 0), 0) for u in member_ids)
            self.assertEqual(total, await pool_used(member_ids))

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        await assert_invariant([alice, bob])

        # Carol joins.
        await pool_settlement.settle_pool(self.db, pool.id)
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=carol.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, pool.id, carol.id, carol.username)
        await self.db.commit()
        await assert_invariant([alice, bob, carol])

        # Alice leaves.
        await pool_settlement.settle_pool(self.db, pool.id)
        await pool_settlement.clear_member_settlement(self.db, pool.id, alice.id)
        await self.db.execute(
            RequestPoolMember.__table__.delete().where(
                RequestPoolMember.pool_id == pool.id,
                RequestPoolMember.user_id == alice.id,
            )
        )
        await self.db.commit()
        await assert_invariant([bob, carol])

    async def test_joining_a_busy_pool_costs_the_joiner_nothing(self):
        # Example B: the pool has spent 500 before Carol arrives.
        alice = await self.make_user("alice", rpd_limit=500)
        bob = await self.make_user("bob", rpd_limit=500)
        carol = await self.make_user("carol", rpd_limit=500)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 500)

        await pool_settlement.settle_pool(self.db, pool.id)
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=carol.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, pool.id, carol.id, carol.username)
        await self.db.commit()

        self.assertEqual(await self.charged(pool.id, alice.id), 250)
        self.assertEqual(await self.charged(pool.id, bob.id), 250)
        self.assertEqual(await self.charged(pool.id, carol.id), 0,
                         "the 500 spent before Carol arrived was never hers")
        self.assertEqual(await self.carry(carol.id), 0)

    async def test_a_member_is_not_charged_for_usage_that_predates_their_join(self):
        alice = await self.make_user("alice", rpd_limit=500)
        bob = await self.make_user("bob", rpd_limit=500)
        carol = await self.make_user("carol", rpd_limit=500)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 500)

        await pool_settlement.settle_pool(self.db, pool.id)
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=carol.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, pool.id, carol.id, carol.username)
        await self.db.commit()

        # Carol leaves an hour later having sent nothing.
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        self.assertEqual(await self.charged(pool.id, carol.id), 0)
        self.assertEqual(await self.carry(carol.id), 0,
                         "she walks out with her full 500 intact")

    async def test_a_member_is_not_charged_for_usage_after_they_leave(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4)

        # Alice leaves: 4 split evenly, 2 each.
        await pool_settlement.settle_pool(self.db, pool.id)
        frozen = await self.carry(alice.id)
        await pool_settlement.clear_member_settlement(self.db, pool.id, alice.id)
        await self.db.execute(
            RequestPoolMember.__table__.delete().where(
                RequestPoolMember.pool_id == pool.id,
                RequestPoolMember.user_id == alice.id,
            )
        )
        await self.db.commit()

        # Bob spends heavily afterwards and the pool settles again.
        await self.seed_usage("bob", 6)
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        self.assertEqual(await self.carry(alice.id), frozen,
                         "Alice's carry is frozen at the moment she left")

    async def test_depleted_user_cannot_extract_quota_by_joining_and_leaving(self):
        # Example C: Alice is settled out (rows 10, carry -5, effective 5 of 5).
        alice = await self.make_user("alice", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.seed_usage("alice", 10)
        self.db.add(UserRpdCarry(
            user_id=alice.id, usage_date=DAY, scope_kind="overall", scope_id=0, carry=-5,
        ))
        await self.db.commit()

        pool = await self.make_pool("fresh", carol)
        await pool_settlement.settle_pool(self.db, pool.id)
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=alice.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, pool.id, alice.id, alice.username)
        await self.db.commit()

        self.assertEqual(await self.charged(pool.id, alice.id), 5,
                         "Alice enters already owing her whole limit")
        self.assertEqual(await self.charged(pool.id, carol.id), 0,
                         "Carol's 5 is untouched")

        # She leaves immediately having consumed nothing.
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        self.assertEqual(await self.carry(alice.id), -5)
        self.assertEqual(await self.charged(pool.id, carol.id), 0)

    async def test_fully_used_pool_leaves_every_member_with_zero(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 10)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        for user in (alice, bob):
            with self.subTest(user=user.username):
                self.assertEqual(await self.charged(pool.id, user.id), 5)

    async def test_settled_member_contributes_nothing_to_the_next_pool_they_join(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        dave = await self.make_user("dave", rpd_limit=5)
        first = await self.make_pool("first", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        await pool_settlement.settle_pool(self.db, first.id)
        await pool_settlement.clear_member_settlement(self.db, first.id, alice.id)
        await self.db.execute(
            RequestPoolMember.__table__.delete().where(
                RequestPoolMember.pool_id == first.id,
                RequestPoolMember.user_id == alice.id,
            )
        )
        await self.db.commit()

        second = await self.make_pool("second", dave)
        await pool_settlement.settle_pool(self.db, second.id)
        self.db.add(RequestPoolMember(pool_id=second.id, user_id=alice.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, second.id, alice.id, alice.username)
        await self.db.commit()

        # Pool limit 10, and Alice arrives owing her full 5 — not her raw 8.
        used = (await self.charged(second.id, alice.id)) + (await self.charged(second.id, dave.id))
        self.assertEqual(used, 5)
        self.assertEqual(10 - used, 5, "Dave's own 5 is exactly what is left")

    async def test_leaving_an_unlimited_pool_restores_the_full_own_limit(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=None)  # inherits the null default
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        self.assertEqual(await self.charged(pool.id, alice.id), 0,
                         "nothing is enforced on an unlimited pool, so nothing is owed")
        self.assertEqual(await self.carry(alice.id), -8,
                         "her carry cancels her rows: she leaves with her full limit")

    async def test_settlement_never_leaves_a_member_with_negative_remaining(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 40)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        for user in (alice, bob):
            with self.subTest(user=user.username):
                charged = await self.charged(pool.id, user.id)
                self.assertLessEqual(charged, 5)
                self.assertGreaterEqual(charged, 0)

    async def test_admission_clamp_caps_a_joiner_whose_limit_was_lowered_mid_day(self):
        # Alice already sent 20 when an admin drops her limit to 5.
        alice = await self.make_user("alice", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await self.seed_usage("alice", 20)
        pool = await self.make_pool("team", carol)

        await pool_settlement.settle_pool(self.db, pool.id)
        self.db.add(RequestPoolMember(pool_id=pool.id, user_id=alice.id))
        await self.db.flush()
        await pool_settlement.admit_member(self.db, pool.id, alice.id, alice.username)
        await self.db.commit()

        self.assertEqual(await self.charged(pool.id, alice.id), 5,
                         "clamped at her own limit, so she cannot drag the pool negative")
        self.assertEqual(await self.carry(alice.id), 5 - 20)

    async def test_resettling_is_idempotent(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        first = (await self.charged(pool.id, alice.id), await self.charged(pool.id, bob.id),
                 await self.carry(alice.id), await self.carry(bob.id))

        for _ in range(3):
            await pool_settlement.settle_pool(self.db, pool.id)
            await self.db.commit()

        again = (await self.charged(pool.id, alice.id), await self.charged(pool.id, bob.id),
                 await self.carry(alice.id), await self.carry(bob.id))
        self.assertEqual(first, again)

    async def test_first_settlement_of_a_new_day_charges_only_todays_usage(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 99, day=date(2026, 4, 6))  # yesterday
        await self.seed_usage("alice", 2)

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        total = (await self.charged(pool.id, alice.id)) + (await self.charged(pool.id, bob.id))
        self.assertEqual(total, 2, "yesterday's 99 requests are not today's problem")

    async def test_carries_and_ledger_rows_expire_at_local_midnight(self):
        from app.auth.database import purge_stale_pool_rows
        from sqlalchemy import func, select

        alice = await self.make_user("alice", rpd_limit=5)
        pool = await self.make_pool("team", alice)
        yesterday = date(2026, 4, 6)
        self.db.add_all([
            RequestPoolLedger(pool_id=pool.id, user_id=alice.id, usage_date=yesterday,
                              scope_kind="overall", scope_id=0, charged=4),
            UserRpdCarry(user_id=alice.id, usage_date=yesterday,
                         scope_kind="overall", scope_id=0, carry=-4),
            RequestPoolLedger(pool_id=pool.id, user_id=alice.id, usage_date=DAY,
                              scope_kind="overall", scope_id=0, charged=1),
        ])
        await self.db.commit()

        # A read scoped to today never sees yesterday's rows...
        self.assertEqual(await self.carry(alice.id), 0)
        self.assertEqual(await self.charged(pool.id, alice.id), 1)

        # ...and the scheduled purge removes them for good.
        await purge_stale_pool_rows()
        for table in (RequestPoolLedger, UserRpdCarry):
            with self.subTest(table=table.__tablename__):
                stale = (await self.db.execute(
                    select(func.count(table.id)).where(table.usage_date < DAY)
                )).scalar_one()
                self.assertEqual(stale, 0)

    async def test_settlement_flushes_buffered_usage_before_reading(self):
        # The buffer holds up to a minute of traffic, and it is exactly the traffic a
        # departing member most recently made.
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        self.request_tracker._usage_buffer[(DAY, 9, "alice", "user", "p/m", "openai")] = 4

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        total = (await self.charged(pool.id, alice.id)) + (await self.charged(pool.id, bob.id))
        self.assertEqual(total, 4, "the buffered requests were settled, not missed")

    async def test_deleting_usage_rows_under_a_negative_carry_clamps_at_zero(self):
        from sqlalchemy import delete

        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        # An admin purges the usage rows out from under the pool.
        await self.db.execute(delete(RequestUsage))
        await self.db.commit()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        for user in (alice, bob):
            with self.subTest(user=user.username):
                self.assertGreaterEqual(await self.charged(pool.id, user.id), 0)
        # Nothing was consumed and nothing remains charged.
        total = (await self.charged(pool.id, alice.id)) + (await self.charged(pool.id, bob.id))
        self.assertEqual(total, 0)


class GroupScopeSettlementTests(PoolTestCase):
    """Settlement runs on the model-group and instance-group tiers too."""

    async def _make_model_group(self, name, model_id, rpd_default):
        group = ModelGroup(name=name, rpm_default=None, rpd_default=rpd_default)
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupMember(group_id=group.id, model_id=model_id))
        await self.db.commit()
        await self.refresh()
        return group

    async def _make_instance_group(self, name, provider_key, rpd_default):
        group = InstanceGroup(name=name, rpm_default=None, rpd_default=rpd_default)
        self.db.add(group)
        await self.db.flush()
        self.db.add(InstanceGroupMember(group_id=group.id, provider_key=provider_key))
        await self.db.commit()
        await self.refresh()
        return group

    async def test_settlement_applies_to_model_and_instance_group_scopes(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        mg = await self._make_model_group("grouped", "openai/gpt-4o", rpd_default=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 6, model="openai/gpt-4o")

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        scope = ("model_group", mg.id)
        self.assertEqual(await self.charged(pool.id, alice.id, scope), 3)
        self.assertEqual(await self.charged(pool.id, bob.id, scope), 3)
        self.assertEqual(await self.charged(pool.id, alice.id), 0,
                         "grouped rows do not touch the overall scope")

    async def test_instance_group_takes_precedence_over_model_group(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        mg = await self._make_model_group("by-model", "azure/gpt-4o", rpd_default=10)
        ig = await self._make_instance_group("by-instance", "azure", rpd_default=20)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 4, model="azure/gpt-4o")

        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        self.assertEqual(await self.charged(pool.id, alice.id, ("instance_group", ig.id)), 2)
        self.assertEqual(await self.charged(pool.id, alice.id, ("model_group", mg.id)), 0)

"""Tests for splitting a pool's consumption among its members as whole requests.

Every count in this system is an integer, so a pool that consumed 9 requests cannot
charge two equal members 4.5 each. Rounding each share independently would drift the
total by up to N/2 requests on every composition change, quietly creating or destroying
quota; largest-remainder apportionment makes the integer shares sum exactly instead.

The capping rules matter just as much as the arithmetic: a member must never be charged
past their own limit (that would let a pool push someone negative), and when everyone is
capped the leftover is dropped rather than forced onto somebody -- quota is destroyed,
never created.
"""

import unittest

from app.auth.pools import MemberShare, apportion


def member(user_id, limit, charged=0, consumed=0):
    return MemberShare(user_id=user_id, limit=limit, charged=charged, consumed=consumed)


class ApportionmentTests(unittest.TestCase):

    def test_shares_are_whole_numbers_and_sum_exactly_to_delta(self):
        members = [member(1, 300), member(2, 300), member(3, 400)]
        shares = apportion(7, members)
        for value in shares.values():
            self.assertIsInstance(value, int)
        self.assertEqual(sum(shares.values()), 7)

    def test_odd_delta_between_equal_members_gives_the_extra_to_the_heavier_consumer(self):
        # Alice sent 8, Bob sent 1, both limited to 5; the pool spent 9.
        # An exact split is 4.5 each, so one whole request has to land somewhere.
        alice = member(1, 5, consumed=8)
        bob = member(2, 5, consumed=1)
        shares = apportion(9, [alice, bob])
        self.assertEqual(shares[1], 5)
        self.assertEqual(shares[2], 4)
        self.assertEqual(sum(shares.values()), 9)

    def test_shares_are_proportional_to_member_limits(self):
        members = [member(1, 100), member(2, 300)]
        shares = apportion(100, members)
        self.assertEqual(shares[1], 25)
        self.assertEqual(shares[2], 75)

    def test_a_member_is_never_charged_past_their_own_limit(self):
        # Member 1 has already been charged 4 of their 5, so they can absorb 1 more.
        members = [member(1, 5, charged=4), member(2, 100)]
        shares = apportion(50, members)
        self.assertLessEqual(shares[1], 1)
        self.assertEqual(shares[1] + members[0].charged <= members[0].limit, True)

    def test_overflow_from_a_capped_member_is_reapportioned_to_the_rest(self):
        # By weight alone member 1 would take 25 of 50, but only 1 of their limit is
        # left. The other 24 must land on member 2, not vanish.
        members = [member(1, 100, charged=99), member(2, 100)]
        shares = apportion(50, members)
        self.assertEqual(shares[1], 1)
        self.assertEqual(shares[2], 49)
        self.assertEqual(sum(shares.values()), 50)

    def test_remainder_is_dropped_when_every_member_is_capped(self):
        members = [member(1, 5, charged=5), member(2, 5, charged=5)]
        shares = apportion(10, members)
        self.assertEqual(sum(shares.values()), 0)

    def test_negative_delta_refunds_proportionally_and_floors_charged_at_zero(self):
        # Usage rows deleted underneath a live pool drive delta negative.
        members = [member(1, 100, charged=10), member(2, 100, charged=10)]
        shares = apportion(-8, members)
        self.assertEqual(sum(shares.values()), -8)
        for m in members:
            self.assertGreaterEqual(m.charged + shares[m.user_id], 0)

    def test_a_refund_never_takes_a_member_below_zero_charged(self):
        members = [member(1, 100, charged=2), member(2, 100, charged=50)]
        shares = apportion(-40, members)
        self.assertGreaterEqual(2 + shares[1], 0)
        self.assertGreaterEqual(50 + shares[2], 0)
        self.assertEqual(sum(shares.values()), -40)

    def test_zero_delta_charges_nobody(self):
        members = [member(1, 5, consumed=3), member(2, 5)]
        self.assertEqual(apportion(0, members), {1: 0, 2: 0})

    def test_members_with_a_zero_limit_absorb_nothing(self):
        members = [member(1, 0), member(2, 10)]
        shares = apportion(6, members)
        self.assertEqual(shares[1], 0)
        self.assertEqual(shares[2], 6)

    def test_apportionment_never_drifts_over_many_sequential_intervals(self):
        """Sum of charged must track the pool's total exactly, interval after interval.

        This is the property that keeps the pool's arithmetic consistent: if each
        settlement were allowed to be off by a request, a busy day of joins and leaves
        would accumulate a visible discrepancy between what the pool spent and what its
        members were charged.
        """
        limits = {1: 7, 2: 11, 3: 13}
        charged = {uid: 0 for uid in limits}
        consumed = {uid: 0 for uid in limits}
        pool_total = 0

        # Deltas chosen to be awkward: coprime-ish with the weights, so almost every
        # interval has a nonzero remainder to place.
        for step, delta in enumerate([3, 1, 5, 2, 8, 1, 1, 4, 6, 2]):
            if pool_total + delta > sum(limits.values()):
                break
            consumed[(step % 3) + 1] += delta
            shares = apportion(
                delta,
                [MemberShare(uid, limits[uid], charged[uid], consumed[uid]) for uid in limits],
            )
            self.assertEqual(sum(shares.values()), delta, f"drifted at step {step}")
            for uid, s in shares.items():
                charged[uid] += s
            pool_total += delta
            self.assertEqual(sum(charged.values()), pool_total)

        for uid, value in charged.items():
            self.assertLessEqual(value, limits[uid])

    def test_no_members_yields_no_shares(self):
        self.assertEqual(apportion(10, []), {})


if __name__ == "__main__":
    unittest.main()

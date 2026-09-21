"""Tests for what a pool member may see — the one place a user reads another's usage.

Sharing a quota is only tolerable if members can see who is spending it, so the pool
usage endpoint deliberately exposes per-member consumption. That makes the membership
check a security boundary rather than a convenience: without it, any user could name any
username and read their history. Two other properties are load-bearing here — carries
never leak into usage views (attribution stays exactly as recorded), and the admin
payload is built by the same function members get, so the two cannot drift apart.
"""

from fastapi import HTTPException

from app.auth.models import (
    InstanceGroup, InstanceGroupMember, ModelGroup, ModelGroupMember,
    UserModelGroupRateLimit, UserRpdCarry,
)
from app.routes import pools as pool_routes
from tests.pool_test_base import DAY, PoolTestCase


class PoolUsageVisibilityTests(PoolTestCase):

    async def _team(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8, model="openai/gpt-4o")
        await self.seed_usage("bob", 2, model="other/model")
        return alice, bob, pool

    async def test_member_can_read_another_members_usage_in_the_same_pool(self):
        alice, bob, pool = await self._team()

        payload = await pool_routes.build_pool_usage(
            self.db, pool.id, window="today", view="user", target="bob",
        )
        self.assertEqual(payload["id"], "bob")
        self.assertTrue(payload["breakdown"], "Alice sees Bob's per-model split")

    async def test_member_cannot_read_a_non_members_usage(self):
        alice, _bob, pool = await self._team()
        await self.make_user("carol", rpd_limit=100)
        await self.seed_usage("carol", 5)

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.build_pool_usage(
                self.db, pool.id, window="today", view="user", target="carol",
            )
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_unpooled_user_gets_an_empty_pool_view(self):
        carol = await self.make_user("carol", rpd_limit=100)
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.get_pool_usage(
                window="today", view=None, id=None, year=None, month=None,
                current_user=carol, db=self.db,
            )
        self.assertEqual(ctx.exception.status_code, 404)

        me = await pool_routes.get_my_pool(carol, self.db)
        self.assertIsNone(me.pool)
        self.assertEqual(me.members, [])

    async def test_per_member_totals_cover_the_whole_pool(self):
        _alice, _bob, pool = await self._team()
        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="today")

        totals = {r["user_identity"]: r["request_count"] for r in payload["per_member"]}
        self.assertEqual(totals, {"alice": 8, "bob": 2})
        self.assertEqual(payload["totals"]["requests"], 10)
        self.assertEqual(payload["totals"]["unique_members"], 2)

    async def test_per_group_totals_apply_instance_group_precedence(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])

        mg = ModelGroup(name="by-model", rpm_default=None, rpd_default=10)
        ig = InstanceGroup(name="by-instance", rpm_default=None, rpd_default=10)
        self.db.add_all([mg, ig])
        await self.db.flush()
        self.db.add_all([
            ModelGroupMember(group_id=mg.id, model_id="azure/gpt-4o"),
            InstanceGroupMember(group_id=ig.id, provider_key="azure"),
        ])
        await self.db.commit()
        await self.refresh()

        await self.seed_usage("alice", 3, model="azure/gpt-4o")
        await self.seed_usage("bob", 1, model="loose/model")

        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="today")
        groups = {g["name"]: g for g in payload["per_group"]}
        self.assertEqual(groups["by-instance"]["request_count"], 3)
        self.assertNotIn("by-model", groups,
                         "a model in both groups belongs to the instance group")
        self.assertEqual(groups["Other Models"]["request_count"], 1)
        self.assertEqual(
            groups["by-instance"]["per_member"],
            [{"user_identity": "alice", "request_count": 3}],
        )

    async def test_usage_views_never_include_carry_adjustments(self):
        alice, bob, pool = await self._team()
        self.db.add_all([
            UserRpdCarry(user_id=alice.id, usage_date=DAY,
                         scope_kind="overall", scope_id=0, carry=-3),
            UserRpdCarry(user_id=bob.id, usage_date=DAY,
                         scope_kind="overall", scope_id=0, carry=3),
        ])
        await self.db.commit()
        await self.refresh()

        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="today")
        totals = {r["user_identity"]: r["request_count"] for r in payload["per_member"]}
        self.assertEqual(totals, {"alice": 8, "bob": 2},
                         "usage reports what was sent; only quotas reflect settlement")

    async def test_member_rows_show_both_sent_and_charged(self):
        from app.auth import pools as pool_settlement

        alice, bob, pool = await self._team()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        me = await pool_routes.get_my_pool(alice, self.db)
        rows = {m.username: m for m in me.members}
        alice_overall = next(
            s for s in rows["alice"].scopes if s.scope_kind == "overall"
        )
        self.assertEqual(alice_overall.sent, 8, "what she actually sent")
        self.assertEqual(alice_overall.charged, 5, "half of the pool's 10, by equal limits")
        self.assertEqual(
            alice_overall.net_contribution,
            alice_overall.limit - alice_overall.charged,
            "the headroom she still brings to the pool",
        )

    async def test_leave_preview_reports_the_numbers_leaving_would_produce(self):
        from app.auth import pools as pool_settlement

        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 8)
        await self.seed_usage("bob", 1)

        me = await pool_routes.get_my_pool(alice, self.db)
        preview = next(p for p in me.if_i_leave_now if p.scope_kind == "overall")
        self.assertEqual((preview.sent, preview.charged, preview.remaining), (8, 5, 0))

        # The preview is a real settlement rolled back, so it must leave nothing behind.
        self.assertEqual(await self.charged(pool.id, alice.id), 0)
        self.assertEqual(await self.carry(alice.id), 0)

        # And leaving for real produces exactly the previewed numbers.
        await pool_routes.leave_pool(alice, self.db)
        self.assertEqual(await self.carry(alice.id), 5 - 8)

    async def test_admin_can_read_any_pools_usage_without_membership(self):
        _alice, _bob, pool = await self._team()
        payload = await pool_routes.build_pool_usage(
            self.db, pool.id, window="today",
        )
        self.assertEqual(payload["totals"]["requests"], 10)

    async def test_admin_and_member_pool_usage_payloads_are_identical(self):
        alice, _bob, pool = await self._team()
        as_member = await pool_routes.get_pool_usage(
            window="today", view=None, id=None, year=None, month=None,
            current_user=alice, db=self.db,
        )
        as_admin = await pool_routes.build_pool_usage(self.db, pool.id, window="today")
        self.assertEqual(as_member, as_admin, "one implementation, so they cannot drift")

    async def test_admin_pool_list_reports_every_members_charged_and_carry(self):
        from app.auth import pools as pool_settlement

        alice, bob, pool = await self._team()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()

        listing = await pool_routes.admin_list_pools(self.db)
        self.assertEqual(len(listing), 1)
        entry = listing[0]
        self.assertEqual(entry.owner_username, "alice")
        by_name = {m.username: m for m in entry.members}
        self.assertEqual(set(by_name), {"alice", "bob"})
        for name, member in by_name.items():
            with self.subTest(member=name):
                overall = next(s for s in member.scopes if s.scope_kind == "overall")
                self.assertEqual(overall.carry, overall.charged - overall.sent)

    async def test_pool_scope_used_tracks_traffic_since_the_last_settlement(self):
        """The pool row must show what the limiter enforces, not the last ledger write.

        `charged` is only rewritten on a composition change, so a pool that has spent
        half its shared quota since the last join or leave would otherwise render as
        completely idle — while the Quotas tab beside it showed the real number.
        """
        from app.auth import pools as pool_settlement

        alice, _bob, pool = await self._team()
        await pool_settlement.settle_pool(self.db, pool.id)
        await self.db.commit()
        await self.seed_usage("alice", 7, model="openai/gpt-4o-mini")

        _members, scopes, _detail, _users = await pool_routes._render_pool(self.db, pool)
        overall = next(s for s in scopes if s.scope_kind == "overall")

        live = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(overall.used, live.rpd_count, "the pool row agrees with enforcement")
        self.assertEqual(overall.limit, 200)
        self.assertEqual(overall.remaining, 200 - overall.used)

    async def test_idle_group_is_listed_so_a_per_user_override_is_visible(self):
        """A group with no traffic still has to appear, with each member's own limit.

        The Quotas tab lists every group unconditionally, so an admin who lowers one
        member's limit on a quiet group sees it there immediately. While the pool view
        hid idle groups, its only row was Overall — an untouched quota — which read as
        the override not having taken effect.
        """
        alice = await self.make_user("alice", rpd_limit=1500)
        bob = await self.make_user("bob", rpd_limit=1500)
        pool = await self.make_pool("shared-usage", alice, [bob])

        group = ModelGroup(name="opus", rpm_default=None, rpd_default=1500)
        self.db.add(group)
        await self.db.flush()
        self.db.add_all([
            ModelGroupMember(group_id=group.id, model_id="anthropic/opus"),
            UserModelGroupRateLimit(user_id=alice.id, group_id=group.id,
                                    rpm_limit=None, rpd_limit=1),
        ])
        await self.db.commit()
        await self.refresh()

        members, scopes, _detail, _users = await pool_routes._render_pool(self.db, pool)
        opus = next(s for s in scopes if s.name == "opus")
        self.assertEqual(opus.limit, 1501, "alice's 1 plus bob's group default of 1500")
        self.assertEqual(opus.used, 0)

        limits = {
            m.username: next(s.limit for s in m.scopes if s.name == "opus")
            for m in members
        }
        self.assertEqual(limits, {"alice": 1, "bob": 1500})

    async def test_leave_preview_skips_groups_the_member_never_touched(self):
        """The leave dialog writes a paragraph per scope, so it stays activity-scoped.

        Listing every idle group there would bury the one number the dialog exists to
        show: what leaving costs the member right now.
        """
        alice, _bob, pool = await self._team()

        group = ModelGroup(name="opus", rpm_default=None, rpd_default=50)
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupMember(group_id=group.id, model_id="anthropic/opus"))
        await self.db.commit()
        await self.refresh()

        preview = await pool_routes._leave_preview(self.db, pool, alice.id)
        self.assertEqual([p.scope_kind for p in preview], ["overall"])

    async def test_per_group_entries_carry_their_settlement_scope(self):
        """The Pool tab joins this fold to the quota numbers in GET /auth/pools/me.

        Joining on the display name would break the moment two groups shared one, and
        would silently mismatch when a group is renamed between the two reads. The
        (kind, scope_id) pair is the same key settlement writes its ledger under.
        """
        alice = await self.make_user("alice", rpd_limit=100)
        pool = await self.make_pool("team", alice)

        mg = ModelGroup(name="opus", rpm_default=None, rpd_default=10)
        ig = InstanceGroup(name="edge", rpm_default=None, rpd_default=10)
        self.db.add_all([mg, ig])
        await self.db.flush()
        self.db.add_all([
            ModelGroupMember(group_id=mg.id, model_id="anthropic/opus"),
            InstanceGroupMember(group_id=ig.id, provider_key="azure"),
        ])
        await self.db.commit()
        await self.refresh()

        await self.seed_usage("alice", 3, model="anthropic/opus")
        await self.seed_usage("alice", 2, model="azure/gpt-4o")
        await self.seed_usage("alice", 1, model="loose/model")

        payload = await pool_routes.build_pool_usage(self.db, pool.id, window="today")
        scoped = {g["name"]: (g["kind"], g["scope_id"]) for g in payload["per_group"]}
        self.assertEqual(scoped["opus"], ("model_group", mg.id))
        self.assertEqual(scoped["edge"], ("instance_group", ig.id))
        self.assertEqual(scoped["Other Models"], ("other", 0),
                         "ungrouped traffic has no quota scope of its own")

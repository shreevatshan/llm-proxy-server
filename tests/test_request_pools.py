"""Tests for pool membership and the invitation lifecycle.

Pooling is voluntary in both directions: an invite does nothing until the invitee
accepts, and a user belongs to at most one pool so their limit is never counted into
two shared quotas at once. The rules that need defending are the ones that only bite
under concurrency or after a user has moved around — a stale invite letting someone
switch pools silently, a pool outliving its last member, an owner leaving and taking
the pool's identity with them.
"""

from fastapi import HTTPException
from sqlalchemy import select

from app.auth.admin import AdminUser
from app.auth.models import (
    MAX_POOL_MEMBERS, PoolCreate, PoolInviteCreate, PoolUpdate,
    RequestPool, RequestPoolInvitation, RequestPoolMember,
)
from app.routes import pools as pool_routes
from tests.pool_test_base import PoolTestCase


class PoolMembershipTests(PoolTestCase):

    async def _invite_and_accept(self, inviter, invitee):
        await pool_routes.create_invite(
            PoolInviteCreate(username=invitee.username), inviter, self.db
        )
        invite = (await self.db.execute(
            select(RequestPoolInvitation).where(
                RequestPoolInvitation.invitee_user_id == invitee.id,
                RequestPoolInvitation.status == "pending",
            )
        )).scalars().first()
        await pool_routes.accept_invite(invite.id, invitee, self.db)
        return invite

    async def _members(self, pool_id):
        rows = (await self.db.execute(
            select(RequestPoolMember).where(RequestPoolMember.pool_id == pool_id)
        )).scalars().all()
        return {r.user_id for r in rows}

    async def test_creating_a_pool_makes_the_creator_its_first_member(self):
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        pool = (await self.db.execute(select(RequestPool))).scalar_one()
        self.assertEqual(pool.owner_user_id, alice.id)
        self.assertEqual(await self._members(pool.id), {alice.id},
                         "a pool is never memberless")

    async def test_user_cannot_belong_to_two_pools(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="first"), alice, self.db)
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()

        # Bob creates his own pool while that invite is still outstanding, so accepting
        # it would put him in two.
        await pool_routes.create_pool(PoolCreate(name="second"), bob, self.db)
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.accept_invite(invite.id, bob, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

        # And creating a third pool of his own is refused too.
        with self.assertRaises(HTTPException):
            await pool_routes.create_pool(PoolCreate(name="third"), bob, self.db)

    async def test_accepting_an_invite_supersedes_other_pending_invites(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="alices"), alice, self.db)
        await pool_routes.create_pool(PoolCreate(name="carols"), carol, self.db)

        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), carol, self.db)
        accepted = await self._invite_and_accept_existing(bob, "alices")

        rows = (await self.db.execute(
            select(RequestPoolInvitation).where(RequestPoolInvitation.id != accepted)
        )).scalars().all()
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(pool_id=row.pool_id):
                self.assertEqual(row.status, "superseded",
                                 "a stale invite must not let him switch pools later")

    async def _invite_and_accept_existing(self, invitee, pool_name):
        pool = (await self.db.execute(
            select(RequestPool).where(RequestPool.name == pool_name)
        )).scalar_one()
        invite = (await self.db.execute(
            select(RequestPoolInvitation).where(
                RequestPoolInvitation.invitee_user_id == invitee.id,
                RequestPoolInvitation.pool_id == pool.id,
            )
        )).scalar_one()
        await pool_routes.accept_invite(invite.id, invitee, self.db)
        return invite.id

    async def test_declined_invite_does_not_block_a_reinvite(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        first = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()
        await pool_routes.decline_invite(first.id, bob, self.db)

        # The declined row is kept as history, so the partial unique index must not
        # treat it as an outstanding invite.
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        pending = (await self.db.execute(
            select(RequestPoolInvitation).where(RequestPoolInvitation.status == "pending")
        )).scalars().all()
        self.assertEqual(len(pending), 1)

    async def test_a_second_pending_invite_to_the_same_user_is_rejected(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_owner_leaving_transfers_ownership_to_earliest_member(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await self._invite_and_accept(alice, bob)
        await self._invite_and_accept(alice, carol)

        await pool_routes.leave_pool(alice, self.db)

        pool = (await self.db.execute(select(RequestPool))).scalar_one()
        self.assertEqual(pool.owner_user_id, bob.id, "the earliest-joined member takes over")
        self.assertEqual(await self._members(pool.id), {bob.id, carol.id})

    async def test_last_member_leaving_deletes_the_pool(self):
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await pool_routes.leave_pool(alice, self.db)

        self.assertEqual(
            (await self.db.execute(select(RequestPool))).scalars().all(), [],
            "the name is freed for reuse",
        )
        # And it can be taken again immediately.
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

    async def test_invite_to_a_full_pool_is_rejected(self):
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        pool = (await self.db.execute(select(RequestPool))).scalar_one()

        # Fill it to capacity without going through the invite flow.
        for i in range(MAX_POOL_MEMBERS - 1):
            filler = await self.make_user(f"filler{i}", rpd_limit=1)
            self.db.add(RequestPoolMember(pool_id=pool.id, user_id=filler.id))
        await self.db.commit()

        await self.make_user("latecomer", rpd_limit=5)
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_invite(
                PoolInviteCreate(username="latecomer"), alice, self.db
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn(str(MAX_POOL_MEMBERS), ctx.exception.detail)

    async def test_accepting_into_a_full_pool_is_rejected(self):
        """The create_invite check is advisory; this is the one that has to hold.

        Two invites issued while there was still room can both be pending when the pool
        fills up, so the count is re-checked at accept time. It runs under the pool lock
        so a second accept cannot read the membership the first has not committed yet.
        """
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        pool = (await self.db.execute(select(RequestPool))).scalar_one()

        latecomer = await self.make_user("latecomer", rpd_limit=5)
        await pool_routes.create_invite(
            PoolInviteCreate(username="latecomer"), alice, self.db
        )
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()

        # The pool fills up while the invite sits unanswered.
        for i in range(MAX_POOL_MEMBERS - 1):
            filler = await self.make_user(f"filler{i}", rpd_limit=1)
            self.db.add(RequestPoolMember(pool_id=pool.id, user_id=filler.id))
        await self.db.commit()

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.accept_invite(invite.id, latecomer, self.db)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn(str(MAX_POOL_MEMBERS), ctx.exception.detail)

        await self.db.refresh(invite)
        self.assertEqual(invite.status, "pending", "a rejected accept leaves it open")
        self.assertEqual(
            len((await self.db.execute(
                select(RequestPoolMember).where(RequestPoolMember.pool_id == pool.id)
            )).scalars().all()),
            MAX_POOL_MEMBERS,
            "and does not squeeze in a 26th member",
        )

    async def test_a_deactivated_member_stays_in_the_pool_but_stops_donating_quota(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=7)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()
        await pool_routes.accept_invite(invite.id, bob, self.db)

        view = await pool_routes.get_my_pool(alice, self.db)
        self.assertEqual(len(view.members), 2)
        self.assertEqual(
            next(s for s in view.scopes if s.scope_kind == "overall").limit, 12)

        bob.is_active = False
        await self.db.commit()
        await self.refresh()

        view = await pool_routes.get_my_pool(alice, self.db)
        self.assertEqual(len(view.members), 2,
                         "he is still a member -- deactivating is not kicking")
        self.assertFalse(next(m for m in view.members if m.username == "bob").is_active)
        self.assertEqual(
            next(s for s in view.scopes if s.scope_kind == "overall").limit, 5,
            "but his 7 is gone from the shared ceiling",
        )

    async def test_only_the_owner_can_remove_another_member(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await self._invite_and_accept(alice, bob)
        await self._invite_and_accept(alice, carol)
        pool = (await self.db.execute(select(RequestPool))).scalar_one()

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.remove_member(carol.id, bob, self.db)
        self.assertEqual(ctx.exception.status_code, 403)

        await pool_routes.remove_member(carol.id, alice, self.db)
        self.assertEqual(await self._members(pool.id), {alice.id, bob.id})

    async def test_owner_cannot_remove_themselves(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await self._invite_and_accept(alice, bob)

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.remove_member(alice.id, alice, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_only_the_owner_can_rename_or_dissolve_the_pool(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await self._invite_and_accept(alice, bob)

        for call in (
            pool_routes.update_my_pool(PoolUpdate(name="renamed"), bob, self.db),
            pool_routes.delete_my_pool(bob, self.db),
        ):
            with self.subTest(call=call.__qualname__ if hasattr(call, "__qualname__") else call):
                with self.assertRaises(HTTPException) as ctx:
                    await call
                self.assertEqual(ctx.exception.status_code, 403)

    async def test_dissolving_a_pool_removes_its_members_and_invitations(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await self._invite_and_accept(alice, bob)
        await pool_routes.create_invite(PoolInviteCreate(username="carol"), alice, self.db)

        await pool_routes.delete_my_pool(alice, self.db)

        self.assertEqual((await self.db.execute(select(RequestPool))).scalars().all(), [])
        self.assertEqual((await self.db.execute(select(RequestPoolMember))).scalars().all(), [])
        self.assertEqual((await self.db.execute(select(RequestPoolInvitation))).scalars().all(), [])

    async def test_admin_cannot_be_invited_or_pooled(self):
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        # The config-based admin has no integer id, so every handler rejects it before
        # anything touches .id.
        admin = AdminUser(username="admin", email="admin@example.test")
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_pool(PoolCreate(name="admins"), admin, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

        # And there is no user row to invite, so inviting the admin by name 404s.
        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_invite(
                PoolInviteCreate(username=admin.username), alice, self.db
            )
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_an_inactive_user_cannot_be_invited(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        bob.is_active = False
        await self.db.commit()
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        self.assertEqual(ctx.exception.status_code, 404,
                         "a deactivated user reads as no such active user")

    async def test_an_invite_can_only_be_answered_by_its_invitee(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        carol = await self.make_user("carol", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()

        for actor in (carol, alice):
            with self.subTest(actor=actor.username):
                with self.assertRaises(HTTPException) as ctx:
                    await pool_routes.accept_invite(invite.id, actor, self.db)
                self.assertEqual(ctx.exception.status_code, 404)

    async def test_the_inviter_can_cancel_a_pending_invite(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        invite = (await self.db.execute(select(RequestPoolInvitation))).scalar_one()

        await pool_routes.cancel_invite(invite.id, alice, self.db)
        await self.db.refresh(invite)
        self.assertEqual(invite.status, "cancelled")

        with self.assertRaises(HTTPException):
            await pool_routes.accept_invite(invite.id, bob, self.db)


class PoolUserDeletionTests(PoolTestCase):
    """Permanently deleting a pooled user must leave the pool as a leave would.

    The FK cascade drops the membership row without going through any pool code, so
    both halves have to be driven by hand: settle inside the delete's transaction,
    invalidate the tracker once it has committed.
    """

    async def _delete(self, user):
        from app.auth.database import permanently_delete_user

        pool_id, affected = await pool_routes.settle_before_user_delete(self.db, user.id)
        self.assertTrue(await permanently_delete_user(self.db, user.id))
        await pool_routes.invalidate_after_user_delete(pool_id, affected)

    async def test_deleting_a_member_takes_their_limit_share_with_them(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        await self.make_pool("team", alice, [bob])

        before = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(before.rpd_limit, 200, "the pool sums both members' limits")

        await self._delete(bob)

        after = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(after.rpd_limit, 100,
                         "a deleted member left in the snapshot resolves to the default "
                         "RPD, inflating the pool's ceiling until the next refresh")

    async def test_deleting_a_member_drops_the_pools_cached_count(self):
        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 3)
        await self.seed_usage("bob", 4)

        self.assertEqual((await self.tracker.get_user_status(alice.id, "alice")).rpd_count, 7)

        await self._delete(bob)

        self.assertNotIn(f"pool:{pool.id}", self.tracker._rpd_cache,
                         "the composition that produced the cached count is gone")
        self.assertEqual((await self.tracker.get_user_status(alice.id, "alice")).rpd_count, 3)

    async def test_deleting_the_last_member_deletes_the_pool(self):
        alice = await self.make_user("alice", rpd_limit=100)
        pool = await self.make_pool("solo", alice)

        await self._delete(alice)

        self.assertIsNone((await self.db.execute(
            select(RequestPool).where(RequestPool.id == pool.id)
        )).scalar_one_or_none())

    async def test_self_service_delete_settles_the_pool_too(self):
        """The endpoint a user reaches, not just the sequence _delete performs by hand.

        Every test above drives the two steps directly, so they pass whether or not a
        given delete path actually calls them -- which is how DELETE /account came to
        skip settlement entirely. This one goes through the route.
        """
        from app.auth.models import AccountDelete
        from app.routes import auth as auth_routes

        alice = await self.make_user("alice", rpd_limit=100)
        bob = await self.make_user("bob", rpd_limit=100)
        await self.make_pool("team", alice, [bob])
        self.assertEqual(
            (await self.tracker.get_user_status(alice.id, "alice")).rpd_limit, 200,
        )

        await auth_routes.delete_account(
            AccountDelete(confirmation="DELETE"), bob, None, self.db,
        )

        after = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(after.rpd_limit, 100,
                         "bob's share must leave with him, exactly as an admin delete "
                         "or a voluntary leave would take it")


class DirectoryTests(PoolTestCase):
    """The Invite pane's people list.

    It replaced a prefix typeahead, and it keeps that control's one promise while adding
    a second. The promise kept: a row reading `invitable` is a name create_invite will
    accept, so every exclusion here mirrors a rejection it raises — a row that fails on
    click is worse than no row, because it teaches people not to trust the list. The
    promise added: the states are visible before you act, so `member` and `invited`
    appear rather than silently vanishing the way the typeahead hid them.

    There is deliberately no prefix floor and no row cap. Filtering happens in the
    browser, so this endpoint hands over the whole directory — the security trade that
    choice makes is argued in list_directory's own docstring.
    """

    async def _team(self):
        alice = await self.make_user("alice", rpd_limit=10)
        await self.make_pool("team", alice)
        return alice

    async def _directory(self, user):
        body = await pool_routes.list_directory(current_user=user, db=self.db)
        return body["people"]

    async def _states(self, user):
        return {p["username"]: p["state"] for p in await self._directory(user)}

    async def test_each_state_is_reported(self):
        alice = await self._team()
        bob = await self.make_user("bob")
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        await self.make_user("carol")

        self.assertEqual(await self._states(alice), {
            "alice": "member",     # the caller is a picture of the pool, not an omission
            "bob": "invited",
            "carol": "invitable",
        })

    async def test_a_member_of_this_pool_reads_as_member(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        await self.make_pool("team", alice, members=(bob,))

        self.assertEqual((await self._states(alice))["bob"], "member")

    async def test_a_user_in_another_pool_is_absent(self):
        """create_invite 400s them, so a row would be a control that always fails."""
        alice = await self._team()
        carol = await self.make_user("carol")
        await pool_routes.create_pool(PoolCreate(name="other"), carol, self.db)

        self.assertNotIn("carol", await self._states(alice))

    async def test_a_deactivated_user_is_absent(self):
        """create_invite 404s them."""
        alice = await self._team()
        erin = await self.make_user("erin")
        erin.is_active = False
        await self.db.commit()

        self.assertNotIn("erin", await self._states(alice))

    async def test_every_invitable_row_is_one_create_invite_accepts(self):
        """The contract the typeahead held, carried over to the listing."""
        alice = await self._team()
        await self.make_user("carol")
        await self.make_user("dave")
        erin = await self.make_user("erin")
        await pool_routes.create_pool(PoolCreate(name="other"), erin, self.db)

        invitable = [p["username"] for p in await self._directory(alice)
                     if p["state"] == "invitable"]
        self.assertEqual(sorted(invitable), ["carol", "dave"])
        for username in invitable:
            await pool_routes.create_invite(
                PoolInviteCreate(username=username), alice, self.db,
            )

    async def test_rows_are_grouped_by_state_then_alphabetical(self):
        alice = await self._team()
        zoe = await self.make_user("zoe", rpd_limit=10)
        self.db.add(RequestPoolMember(
            pool_id=(await pool_routes._pool_of(self.db, alice.id)).pool_id,
            user_id=zoe.id,
        ))
        await self.db.commit()
        await self.make_user("bob")
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        await self.make_user("adam")
        await self.make_user("carol")

        self.assertEqual(
            [p["username"] for p in await self._directory(alice)],
            ["alice", "zoe",        # members
             "bob",                 # invited
             "adam", "carol"],      # invitable
        )

    async def test_ordering_is_case_insensitive(self):
        """SQLite orders bare strings by byte, which would file Zoe above adam."""
        alice = await self._team()
        await self.make_user("Zoe")
        await self.make_user("adam")

        invitable = [p["username"] for p in await self._directory(alice)
                     if p["state"] == "invitable"]
        self.assertEqual(invitable, ["adam", "Zoe"])

    async def test_a_row_carries_the_user_id(self):
        alice = await self._team()
        carol = await self.make_user("carol")

        row = next(p for p in await self._directory(alice) if p["username"] == "carol")
        self.assertEqual(row["user_id"], carol.id)

    async def test_only_pool_members_can_read_the_directory(self):
        carol = await self.make_user("carol")
        await self.make_user("bobby")

        with self.assertRaises(HTTPException) as ctx:
            await self._directory(carol)
        self.assertEqual(ctx.exception.status_code, 404)

    # -- Cancelling from the row ------------------------------------------- #
    # The Invite pane withdraws an invite on the row that names the person, so an
    # `invited` row has to carry the id to withdraw and whether this reader may.

    async def _row(self, user, username):
        return next(p for p in await self._directory(user) if p["username"] == username)

    async def test_create_invite_returns_the_new_invite_id(self):
        alice = await self._team()
        await self.make_user("bob")

        body = await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)
        self.assertEqual(body["invite_id"], (await self._row(alice, "bob"))["invite_id"])

    async def test_the_sender_may_cancel_from_the_row(self):
        alice = await self._team()
        await self.make_user("bob")
        await pool_routes.create_invite(PoolInviteCreate(username="bob"), alice, self.db)

        self.assertTrue((await self._row(alice, "bob"))["can_cancel"])

    async def test_the_owner_may_cancel_an_invite_someone_else_sent(self):
        alice = await self._team()
        bob = await self.make_user("bob", rpd_limit=10)
        self.db.add(RequestPoolMember(
            pool_id=(await pool_routes._pool_of(self.db, alice.id)).pool_id, user_id=bob.id,
        ))
        await self.db.commit()
        await self.make_user("carol")
        await pool_routes.create_invite(PoolInviteCreate(username="carol"), bob, self.db)

        self.assertTrue((await self._row(alice, "carol"))["can_cancel"])

    async def test_a_member_who_is_neither_sender_nor_owner_sees_a_plain_chip(self):
        """cancel_invite 403s them, so the row must not offer a button that fails."""
        alice = await self._team()
        bob = await self.make_user("bob", rpd_limit=10)
        self.db.add(RequestPoolMember(
            pool_id=(await pool_routes._pool_of(self.db, alice.id)).pool_id, user_id=bob.id,
        ))
        await self.db.commit()
        await self.make_user("carol")
        await pool_routes.create_invite(PoolInviteCreate(username="carol"), alice, self.db)

        self.assertFalse((await self._row(bob, "carol"))["can_cancel"])

    async def test_every_cancellable_row_is_one_cancel_invite_accepts(self):
        """The mirror of the invitable contract: a Cancel button always succeeds."""
        alice = await self._team()
        await self.make_user("bob")
        await self.make_user("carol")
        for username in ("bob", "carol"):
            await pool_routes.create_invite(PoolInviteCreate(username=username), alice, self.db)

        rows = [p for p in await self._directory(alice) if p["state"] == "invited"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row["can_cancel"])
            await pool_routes.cancel_invite(row["invite_id"], alice, self.db)

        self.assertEqual(
            [p["username"] for p in await self._directory(alice) if p["state"] == "invited"], [],
        )

    async def test_an_invitable_row_carries_no_invite_fields(self):
        alice = await self._team()
        await self.make_user("carol")

        row = await self._row(alice, "carol")
        self.assertNotIn("invite_id", row)
        self.assertNotIn("can_cancel", row)


class PoolLockTests(PoolTestCase):
    """The per-pool lock behind every composition change.

    Settlement reads a pool's membership, apportions against it and writes a ledger row
    per member. Two of those interleaving apportion against a membership that is already
    stale by the time they write, and the MAX_POOL_MEMBERS re-check at accept time has
    no database constraint behind it, so the lock is the only thing deciding both.
    """

    async def _overlap(self, first_pool, second_pool):
        """Run two guarded sections concurrently; return True if they overlapped."""
        import asyncio

        state = {"inside": 0, "overlapped": False}

        async def section(pool_id):
            async with pool_routes.pool_guard(pool_id):
                state["inside"] += 1
                if state["inside"] > 1:
                    state["overlapped"] = True
                await asyncio.sleep(0)      # hand control to the other task
                await asyncio.sleep(0)
                state["inside"] -= 1

        await asyncio.gather(section(first_pool), section(second_pool))
        return state["overlapped"]

    async def test_two_changes_to_one_pool_do_not_interleave(self):
        self.assertFalse(await self._overlap(1, 1))

    async def test_changes_to_different_pools_still_run_concurrently(self):
        self.assertTrue(await self._overlap(1, 2),
                        "locking per pool, not globally -- one pool must not stall another")

    async def test_an_unpooled_caller_takes_no_lock(self):
        # delete_user_account guards unconditionally, so None has to be a clean no-op.
        self.assertTrue(await self._overlap(None, None))
        self.assertNotIn(None, self.tracker._pool_locks)

    async def test_the_lock_registry_does_not_grow_without_bound(self):
        async with pool_routes.pool_guard(7):
            pass
        self.assertIn(7, self.tracker._pool_locks)

        self.tracker._evict_stale_buckets()
        self.assertNotIn(7, self.tracker._pool_locks,
                         "an unheld lock is reclaimed; nobody is inside it to be split off")

    async def test_a_held_lock_survives_eviction(self):
        async with pool_routes.pool_guard(7):
            self.tracker._evict_stale_buckets()
            self.assertIn(7, self.tracker._pool_locks,
                          "dropping this one would let a second writer build a new lock "
                          "and walk straight into the critical section")


class PoolLockOrderingTests(PoolTestCase):
    """The pool lock must be outermost, and re-entrant so nesting stays free.

    Two locks are in play on every settlement: the pool lock and request_tracker's flush
    mutex, which settle_pool takes internally. Acquire them in opposite orders in two
    tasks and they deadlock on each other -- the ordering rule in pool_lock's docstring
    is what rules that out, and re-entrancy is what lets a caller hold the pool lock
    across a section that settles again inside it.
    """

    async def test_the_pool_lock_is_reentrant_within_one_task(self):
        """Without this, every outermost acquisition would self-deadlock on the re-take.

        The admin usage-delete endpoint takes the locks before pausing the flush and
        then calls a resettle that takes them again; delete_user_account is the same
        shape. Both are only writable because the second acquisition is free.
        """
        import asyncio

        lock = self.tracker.pool_lock(1)
        async with lock:
            async with lock:                      # would hang on a plain asyncio.Lock
                self.assertTrue(lock.locked())
        self.assertFalse(lock.locked(), "the outermost exit is what actually releases it")

        # ...and it is re-entrant per *task*, not globally: a second task still waits.
        entered = asyncio.Event()

        async def other():
            async with lock:
                entered.set()

        async with lock:
            task = asyncio.ensure_future(other())
            await asyncio.sleep(0)
            self.assertFalse(entered.is_set(), "a different task must not walk in")
        await asyncio.wait_for(task, timeout=1)

    async def test_the_admin_usage_purge_takes_the_pool_lock_before_the_flush_mutex(self):
        """Asserts the acquisition ORDER, because the cycle it prevents needs two tasks.

        Staging a real AB-BA here would hang rather than fail, and a test that hangs on
        regression is worse than no test. So this records the order the endpoint takes
        the two locks in and pins it: pool lock, then flush mutex. Reversed, a task
        holding the flush mutex here would wait on a pool lock held by a settle that is
        itself waiting for the flush mutex.
        """
        from contextlib import asynccontextmanager

        from app.request_tracker import request_tracker
        from app.routes import admin as admin_routes
        from app.routes import pools as pool_routes
        from app.auth.admin import AdminUser

        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=10)
        pool = await self.make_pool("team", alice, [bob])
        await self.seed_usage("alice", 6)

        order = []
        real_guard, real_pause = pool_routes.pool_guard, request_tracker.pause_flush

        @asynccontextmanager
        async def traced_guard(pool_id):
            if pool_id is not None:
                order.append(("pool", pool_id))
            async with real_guard(pool_id):
                yield

        @asynccontextmanager
        async def traced_pause():
            order.append(("flush", None))
            async with real_pause():
                yield

        pool_routes.pool_guard = traced_guard
        request_tracker.pause_flush = traced_pause
        try:
            await admin_routes.delete_usage(
                view="user", id="alice", user_id=alice.id,
                current_admin=AdminUser(username="root", email="root@example.test"),
                db=self.db,
            )
        finally:
            pool_routes.pool_guard = real_guard
            request_tracker.pause_flush = real_pause

        self.assertIn(("pool", pool.id), order, "the purge must lock the affected pool")
        self.assertIn(("flush", None), order)
        self.assertLess(order.index(("pool", pool.id)), order.index(("flush", None)),
                        "pool lock outermost; see pool_lock's ORDERING note")

        self.assertEqual(await self.charged(pool.id, alice.id), 0,
                         "and the purged rows come off the ledger, not just the usage table")


class PoolIdReuseTests(PoolTestCase):
    """A pool id is never handed out twice.

    Usage rows keep a dissolved pool's id, and every pool usage view -- and the admin's
    per-pool purge -- is a filter on that id. A new pool given the old id would inherit
    the old pool's history and could have it deleted along with its own.
    """

    async def test_a_dissolved_pools_id_is_not_reused(self):
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="first"), alice, self.db)
        first_id = (await self.db.execute(select(RequestPool.id))).scalar_one()
        await self.seed_usage("alice", 3)
        await pool_routes.delete_my_pool(alice, self.db)

        await pool_routes.create_pool(PoolCreate(name="second"), alice, self.db)
        second_id = (await self.db.execute(select(RequestPool.id))).scalar_one()
        self.assertGreater(second_id, first_id)

    async def test_an_id_surviving_only_in_usage_is_not_reused(self):
        # A pool dissolved before the high-water mark existed leaves nothing behind
        # but its usage rows; the seed must still count its id as spent.
        alice = await self.make_user("alice", rpd_limit=5)
        await self.seed_usage("alice", 3, pool_id=7)

        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)
        self.assertEqual((await self.db.execute(select(RequestPool.id))).scalar_one(), 8)


class PoolLockEvictionTests(PoolTestCase):

    async def test_a_lock_with_a_waiter_is_not_evicted(self):
        """release() clears locked() before the next waiter has taken the lock.

        Evicting in that gap would let a new caller build a fresh lock and enter
        alongside the woken waiter, which still holds the old one.
        """
        import asyncio

        lock = self.tracker.pool_lock(1)
        waiter_in = asyncio.Event()

        async def waiter():
            async with self.tracker.pool_lock(1):
                waiter_in.set()

        async with lock:
            task = asyncio.ensure_future(waiter())
            await asyncio.sleep(0)             # let it queue on the lock
        # Released, but the waiter has not run yet to take it.
        self.assertFalse(lock.locked())
        self.tracker._evict_stale_buckets()
        self.assertIs(self.tracker.pool_lock(1), lock,
                      "the lock the waiter is queued on must survive the sweep")
        await asyncio.wait_for(task, timeout=1)
        self.assertTrue(waiter_in.is_set())

        self.tracker._evict_stale_buckets()
        self.assertIsNot(self.tracker.pool_lock(1), lock, "an idle lock is still swept")


class AdminResettleFailureTests(PoolTestCase):

    async def test_a_failed_resettle_does_not_fail_a_committed_edit(self):
        from unittest import mock

        from app.auth.models import User
        from app.routes import admin as admin_routes

        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="team"), alice, self.db)

        with mock.patch.object(pool_routes, "resettle_pools",
                               side_effect=RuntimeError("settlement blew up")):
            result = await admin_routes.delete_user(
                user_id=alice.id, current_admin=AdminUser(username="root", email="root@example.test"), db=self.db,
            )

        self.assertIn("deactivated", result["message"])
        is_active = (await self.db.execute(
            select(User.is_active).where(User.id == alice.id)
        )).scalar_one()
        self.assertFalse(is_active, "the deactivation was committed and stands")

"""Tests for the admin dashboard's bulk user actions.

The bulk endpoint has to behave exactly like the single-user endpoints it
mirrors, or the two paths drift and an admin gets different results depending on
whether they clicked one row or three. The cases that matter:

* a bad id in the batch must not sink the users next to it,
* deactivating has to drop cached auth, or the user's API keys keep working,
* activating a *pending* user would report success while the row still reads
  "Pending", so it is refused instead.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth.admin import AdminUser
from app.auth.models import APIKey, Base, BulkUserActionRequest, User, UserRateLimit
from app.routes.admin import MAX_BULK_USER_IDS, bulk_user_action


class BulkUserActionTests(unittest.IsolatedAsyncioTestCase):
    """Exercises bulk_user_action against a throwaway database."""

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        self.db = self._session_factory()
        self.admin = AdminUser(username="admin", email="admin@example.com")

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    async def _add_user(self, username, *, is_active=True, is_pending_approval=False):
        user = User(
            username=username,
            email=f"{username}@example.com",
            hashed_password="x",
            is_active=is_active,
            is_pending_approval=is_pending_approval,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def _call(self, action, user_ids):
        return await bulk_user_action(
            BulkUserActionRequest(user_ids=user_ids, action=action),
            current_admin=self.admin,
            db=self.db,
        )

    async def _reload(self, user_id):
        user = (await self.db.execute(
            select(User).where(User.id == user_id)
        )).scalar_one()
        await self.db.refresh(user)
        return user

    async def _is_active(self, user_id):
        return (await self._reload(user_id)).is_active

    async def _is_pending(self, user_id):
        return (await self._reload(user_id)).is_pending_approval

    # -- deactivate ------------------------------------------------------

    async def test_deactivate_clears_is_active_for_every_id(self):
        alice = await self._add_user("alice")
        bob = await self._add_user("bob")

        result = await self._call("deactivate", [alice.id, bob.id])

        self.assertEqual(result["succeeded_count"], 2)
        self.assertEqual(result["failed"], [])
        self.assertFalse(await self._is_active(alice.id))
        self.assertFalse(await self._is_active(bob.id))

    async def test_deactivate_invalidates_cached_auth(self):
        """Otherwise the deactivated user's API keys keep authenticating."""
        alice = await self._add_user("alice")
        invalidated = []

        from app.auth import cache as cache_module

        with patch.object(
            cache_module.auth_cache, "invalidate_user_api_keys",
            side_effect=lambda uid: invalidated.append(("keys", uid)),
        ), patch.object(
            cache_module.auth_cache, "invalidate_user_by_id",
            side_effect=lambda uid: invalidated.append(("user", uid)),
        ):
            await self._call("deactivate", [alice.id])

        self.assertIn(("keys", alice.id), invalidated)
        self.assertIn(("user", alice.id), invalidated)

    # -- activate --------------------------------------------------------

    async def test_activate_restores_deactivated_users(self):
        alice = await self._add_user("alice", is_active=False)

        result = await self._call("activate", [alice.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertTrue(await self._is_active(alice.id))

    async def test_activate_refuses_pending_users(self):
        """Activating a pending user would leave the row still reading Pending."""
        pending = await self._add_user("carol", is_active=False, is_pending_approval=True)

        result = await self._call("activate", [pending.id])

        self.assertEqual(result["succeeded_count"], 0)
        self.assertEqual(result["failed_count"], 1)
        self.assertIn("Approve", result["failed"][0]["error"])
        # Still inactive, and still pending — nothing was quietly changed.
        self.assertFalse(await self._is_active(pending.id))
        self.assertTrue(await self._is_pending(pending.id))

    async def test_activate_processes_others_alongside_a_pending_user(self):
        alice = await self._add_user("alice", is_active=False)
        pending = await self._add_user("carol", is_active=False, is_pending_approval=True)

        result = await self._call("activate", [alice.id, pending.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertTrue(await self._is_active(alice.id))
        self.assertFalse(await self._is_active(pending.id))

    # -- approve ---------------------------------------------------------

    async def test_approve_clears_pending_and_activates(self):
        carol = await self._add_user("carol", is_active=False, is_pending_approval=True)

        result = await self._call("approve", [carol.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertTrue(await self._is_active(carol.id))
        self.assertFalse(await self._is_pending(carol.id))

    async def test_approve_refuses_users_that_are_not_pending(self):
        """Mirrors the single-user endpoint, which 400s on the same input."""
        alice = await self._add_user("alice")

        result = await self._call("approve", [alice.id])

        self.assertEqual(result["succeeded_count"], 0)
        self.assertEqual(result["failed_count"], 1)
        self.assertIn("not pending approval", result["failed"][0]["error"])

    async def test_approve_of_a_mixed_selection_reports_each_outcome(self):
        """A row's Approve now fires on the whole selection, which is mixed."""
        carol = await self._add_user("carol", is_active=False, is_pending_approval=True)
        alice = await self._add_user("alice")

        result = await self._call("approve", [carol.id, alice.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["succeeded"][0]["username"], "carol")
        self.assertEqual(result["failed"][0]["username"], "alice")
        self.assertFalse(await self._is_pending(carol.id))
        # alice was already active and stays that way — nothing was toggled.
        self.assertTrue(await self._is_active(alice.id))

    # -- delete ----------------------------------------------------------

    async def test_delete_removes_users_and_dependent_rows(self):
        alice = await self._add_user("alice")
        self.db.add(APIKey(user_id=alice.id, api_key="k" * 64, name="key"))
        self.db.add(UserRateLimit(user_id=alice.id))
        await self.db.commit()

        result = await self._call("delete", [alice.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertIsNone((await self.db.execute(
            select(User).where(User.id == alice.id)
        )).scalar_one_or_none())
        self.assertEqual((await self.db.execute(
            select(APIKey).where(APIKey.user_id == alice.id)
        )).scalars().all(), [])
        self.assertEqual((await self.db.execute(
            select(UserRateLimit).where(UserRateLimit.user_id == alice.id)
        )).scalars().all(), [])

    # -- partial failure --------------------------------------------------

    async def test_unknown_id_fails_without_sinking_the_batch(self):
        alice = await self._add_user("alice")

        result = await self._call("deactivate", [alice.id, 9999])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(result["failed"], [
            {"id": 9999, "username": None, "error": "User not found"}
        ])
        self.assertFalse(await self._is_active(alice.id))
        self.assertIn("1 failed", result["message"])

    async def test_unknown_id_does_not_sink_a_delete_batch(self):
        alice = await self._add_user("alice")

        result = await self._call("delete", [9999, alice.id])

        self.assertEqual(result["succeeded_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertIsNone((await self.db.execute(
            select(User).where(User.id == alice.id)
        )).scalar_one_or_none())

    async def test_duplicate_ids_are_processed_once(self):
        alice = await self._add_user("alice")

        result = await self._call("deactivate", [alice.id, alice.id])

        self.assertEqual(result["succeeded_count"], 1)

    # -- request validation ----------------------------------------------

    async def test_empty_selection_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._call("deactivate", [])
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_unknown_action_is_rejected(self):
        alice = await self._add_user("alice")
        with self.assertRaises(HTTPException) as ctx:
            await self._call("banish", [alice.id])
        self.assertEqual(ctx.exception.status_code, 400)
        # The user is untouched.
        self.assertTrue(await self._is_active(alice.id))

    async def test_oversized_batch_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._call("deactivate", list(range(1, MAX_BULK_USER_IDS + 2)))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_endpoint_requires_an_admin(self):
        """Guards against the admin dependency being dropped in a refactor."""
        import inspect

        from app.auth.middleware import get_current_admin

        dependency = inspect.signature(bulk_user_action).parameters["current_admin"].default
        self.assertIs(dependency.dependency, get_current_admin)


if __name__ == "__main__":
    unittest.main()

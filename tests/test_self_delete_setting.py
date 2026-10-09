"""Tests for the admin switch that allows or blocks self-service account deletion.

The setting defaults to allowed so existing deployments keep working after an
upgrade. When an admin blocks it, DELETE /auth/account must refuse with a 403
that names the admin contact, before the confirmation text is even looked at,
and must not touch the account.
"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth.admin import AdminUser
from app.auth.database import get_user_management_settings, is_self_delete_allowed
from app.auth.models import AccountDelete, Base, User, UserManagementSettingsUpdate
from app.routes.admin import (
    get_user_management_settings_endpoint,
    update_user_management_settings_endpoint,
)
from app.routes.auth import delete_account

ADMIN_EMAIL = "owner@example.com"


class SelfDeleteSettingTests(unittest.IsolatedAsyncioTestCase):
    """Exercises the setting endpoints and the delete guard against a throwaway database."""

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

        self.user = User(username="alice", email="alice@example.com", hashed_password="x", is_active=True)
        self.db.add(self.user)
        await self.db.commit()
        await self.db.refresh(self.user)

        self._email_patch = patch("app.routes.auth.get_admin_email", return_value=ADMIN_EMAIL)
        self._email_patch.start()

    async def asyncTearDown(self):
        self._email_patch.stop()
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    async def _set_allowed(self, allowed):
        return await update_user_management_settings_endpoint(
            UserManagementSettingsUpdate(allow_self_delete=allowed),
            current_admin=self.admin,
            db=self.db,
        )

    async def _delete(self, confirmation="DELETE"):
        return await delete_account(
            AccountDelete(confirmation=confirmation),
            current_user=self.user,
            response=None,
            db=self.db,
        )

    # -- tests -----------------------------------------------------------

    async def test_defaults_to_allowed_without_a_row(self):
        self.assertIsNone(await get_user_management_settings(self.db))
        self.assertTrue(await is_self_delete_allowed(self.db))

        result = await get_user_management_settings_endpoint(current_admin=self.admin, db=self.db)
        self.assertTrue(result.allow_self_delete)
        self.assertIsNone(result.updated_by)

    async def test_blocked_refuses_with_admin_contact_and_keeps_account(self):
        result = await self._set_allowed(False)
        self.assertFalse(result.allow_self_delete)
        self.assertEqual(result.updated_by, "admin")
        self.assertIsNotNone(result.updated_at)

        with patch("app.routes.pools.delete_user_account", new=AsyncMock(return_value=True)) as deleter:
            with self.assertRaises(HTTPException) as ctx:
                await self._delete()

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("disabled by the administrator", ctx.exception.detail)
        self.assertIn(ADMIN_EMAIL, ctx.exception.detail)
        self.assertEqual(ctx.exception.headers, {"X-Account-Deletion-Disabled": "true"})
        deleter.assert_not_called()

        row = (await self.db.execute(select(User).where(User.id == self.user.id))).scalar_one_or_none()
        self.assertIsNotNone(row)

    async def test_blocked_wins_over_wrong_confirmation(self):
        await self._set_allowed(False)
        with self.assertRaises(HTTPException) as ctx:
            await self._delete(confirmation="nope")
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_reallowing_lets_deletion_through(self):
        await self._set_allowed(False)
        result = await self._set_allowed(True)
        self.assertTrue(result.allow_self_delete)
        self.assertTrue(await is_self_delete_allowed(self.db))

        with patch("app.routes.pools.delete_user_account", new=AsyncMock(return_value=True)) as deleter:
            response = await self._delete()

        deleter.assert_awaited_once_with(self.db, self.user.id)
        self.assertEqual(response, {"message": "Account deleted successfully"})

    async def test_admin_cannot_use_self_delete_endpoint(self):
        await self._set_allowed(True)
        with self.assertRaises(HTTPException) as ctx:
            await delete_account(
                AccountDelete(confirmation="DELETE"), current_user=self.admin, response=None, db=self.db
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertNotIn("disabled by the administrator", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()

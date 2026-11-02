"""Tests for case-insensitive email matching.

``get_user_by_email`` backs the signup, admin-create, profile-update and OAuth
email checks, so "Alice@x.com" and "alice@x.com" must be the same account there.
Databases created before the check may already hold case-variant duplicates; a
lookup on one of those must pick a row deterministically, not crash.
"""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth.database import get_user_by_email, update_user_profile
from app.auth.models import Base, User, UserCreate
from app.routes.auth import signup


class EmailCaseTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self.db = self._session_factory()

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def _add(self, username, email):
        user = User(username=username, email=email, hashed_password="x", is_active=True)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def test_lookup_ignores_case(self):
        alice = await self._add("alice", "Alice@Example.com")
        for probe in ("alice@example.com", "ALICE@EXAMPLE.COM", "Alice@Example.com"):
            with self.subTest(probe=probe):
                found = await get_user_by_email(self.db, probe)
                self.assertIsNotNone(found)
                self.assertEqual(found.id, alice.id)
        self.assertIsNone(await get_user_by_email(self.db, "bob@example.com"))
        self.assertIsNone(await get_user_by_email(self.db, ""))

    async def test_signup_with_case_variant_email_is_rejected(self):
        await self._add("alice", "alice@example.com")
        webhook = AsyncMock()
        with patch("app.routes.auth.send_signup_webhook", webhook):
            with self.assertRaises(HTTPException) as ctx:
                await signup(UserCreate(username="alice2", email="ALICE@example.com", password="secret1"), db=self.db)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Email already registered")
        self.assertEqual(ctx.exception.headers, {"X-Error-Field": "email"})
        webhook.assert_not_called()

    async def test_signup_field_headers_for_username_and_password(self):
        await self._add("alice", "alice@example.com")
        with patch("app.routes.auth.send_signup_webhook", AsyncMock()):
            with self.assertRaises(HTTPException) as ctx:
                await signup(UserCreate(username="alice", email="new@example.com", password="secret1"), db=self.db)
            self.assertEqual(ctx.exception.headers, {"X-Error-Field": "username"})
            with self.assertRaises(HTTPException) as ctx:
                await signup(UserCreate(username="carol", email="carol@example.com", password="123"), db=self.db)
            self.assertEqual(ctx.exception.headers, {"X-Error-Field": "password"})

    async def test_legacy_duplicates_pick_the_oldest_without_crashing(self):
        older = await self._add("dup1", "Dup@Example.com")
        await self._add("dup2", "dup@example.com")
        with self.assertLogs("app.auth.database", level="WARNING"):
            found = await get_user_by_email(self.db, "DUP@example.com")
        self.assertEqual(found.id, older.id)

    async def test_profile_update_rejects_a_case_variant_of_another_users_email(self):
        await self._add("alice", "alice@example.com")
        bob = await self._add("bob", "bob@example.com")
        with self.assertRaises(ValueError):
            await update_user_profile(self.db, bob.id, email="ALICE@example.com")

    async def test_legacy_duplicate_can_still_save_its_unchanged_email(self):
        await self._add("dup1", "Dup@Example.com")
        newer = await self._add("dup2", "dup@example.com")
        updated = await update_user_profile(self.db, newer.id, email="dup@example.com")
        self.assertEqual(updated.email, "dup@example.com")

    async def test_changing_only_the_case_of_your_own_email_is_allowed(self):
        alice = await self._add("alice", "alice@example.com")
        updated = await update_user_profile(self.db, alice.id, email="Alice@Example.com")
        self.assertEqual(updated.email, "Alice@Example.com")


if __name__ == "__main__":
    unittest.main()

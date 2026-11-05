"""PlatformAI persistence: auto-migration adds the OAuth / BYOK columns and keeps them."""

import os
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth import database
from app.auth.models import Base

NEW_COLUMNS = ("oauth_client_id", "oauth_client_secret", "oauth_refresh_token", "oauth_accounts_url", "pai_token_id")


class PlatformAIMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Simulate a database from before this release.
            for column in NEW_COLUMNS:
                await conn.execute(text(f"ALTER TABLE provider_credentials DROP COLUMN {column}"))
            await conn.execute(text(
                "INSERT INTO provider_credentials (provider_key, provider_type, instance_name, enabled, "
                "provider_name, region, supported_apis) "
                "VALUES ('platformai:main', 'platformai', 'main', 1, 'platformai', NULL, NULL)"
            ))
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
        os.unlink(self.path)

    async def _columns(self):
        async with self.engine.connect() as conn:
            return [r[1] for r in await conn.execute(text("PRAGMA table_info(provider_credentials)"))]

    async def test_columns_added_and_survive_reruns(self):
        await database._run_auto_migrations()
        async with self.engine.begin() as conn:
            await conn.execute(text(
                "UPDATE provider_credentials SET oauth_client_id = 'cid', oauth_refresh_token = 'r' "
                "WHERE provider_key = 'platformai:main'"
            ))
        # A second boot must not drop them (the legacy drop list removes client_id/client_secret).
        await database._run_auto_migrations()
        columns = await self._columns()
        for column in NEW_COLUMNS:
            self.assertIn(column, columns)
        async with self.engine.connect() as conn:
            row = (await conn.execute(text(
                "SELECT oauth_client_id, oauth_refresh_token, supported_apis FROM provider_credentials"
            ))).one()
        self.assertEqual(tuple(row), ("cid", "r", '["openai", "anthropic"]'))


if __name__ == "__main__":
    unittest.main()

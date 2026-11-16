"""POST /admin/providers/sync: re-discover one provider's models on demand.

Saved models keep their switch, new ones follow the provider's setting for new
models, models gone upstream are dropped. An empty or failed fetch keeps the saved
models. Only that provider's slice of the listing changes, and the registry is not
rebuilt.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth.admin import AdminUser
from app.auth.models import Base, ModelConfiguration, ProviderCredentials
from app.openai_models import ModelInfo

ADMIN = AdminUser(username="admin", email="admin@example.com")


def _models(provider_key, names):
    return [
        ModelInfo(id=f"{provider_key}/{n}", created=0, owned_by=provider_key, provider=provider_key)
        for n in names
    ]


class ManualSyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self.db = self._factory()

        from app.auth import database as auth_database
        real = auth_database.AsyncSessionLocal
        auth_database.AsyncSessionLocal = self._factory
        self.addCleanup(setattr, auth_database, "AsyncSessionLocal", real)

        from app.providers import auto_sync
        from app.providers.provider_manager import ProviderManager
        from app.routes import admin as admin_routes
        self.r = admin_routes
        self.pm = ProviderManager()
        self.upstream = AsyncMock(return_value=[])
        self.rebuild = AsyncMock()
        instance = SimpleNamespace(get_available_models=self.upstream)
        for p in (
            patch.object(auto_sync, "provider_manager", self.pm),
            patch.object(self.pm, "refresh_providers_from_database", self.rebuild),
            patch.object(auto_sync, "_create_provider_instance", AsyncMock(return_value=instance)),
            patch.object(auto_sync, "_close_provider_instance", AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def _add_provider(self, key, auto_enable=True, enabled=True, owner=None):
        async with self._factory() as db:
            db.add(ProviderCredentials(
                provider_key=key, provider_type="custom", instance_name=key.split(":", 1)[-1],
                provider_name=key.split(":", 1)[0], base_url="https://api.example.com/v1",
                enabled=enabled, auto_enable_new_models=auto_enable, owner_user_id=owner,
            ))
            await db.commit()

    async def _add_models(self, states):
        async with self._factory() as db:
            for model_id, on in states.items():
                db.add(ModelConfiguration(model_id=model_id, provider_key=model_id.split("/")[0],
                                          model_name=model_id.split("/", 1)[1], is_enabled=on))
            await db.commit()

    async def _states(self, provider_key):
        async with self._factory() as db:
            rows = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.provider_key == provider_key)
            )).scalars()
            return {m.model_id: m.is_enabled for m in rows}

    async def _sync(self, key):
        return await self.r.sync_provider(provider_key=key, current_admin=ADMIN, db=self.db)

    def _listed(self):
        return {m.id for m in self.pm.model_cache.get_models()}

    async def test_sync_keeps_choices_adds_new_off_and_drops_gone(self):
        await self._add_provider("held:a", auto_enable=False)
        await self._add_models({"held:a/keep-on": True, "held:a/keep-off": False, "held:a/gone": True})
        self.pm.model_cache.update_models(_models("other:b", ["x"]) + _models("held:a", ["gone"]))
        self.upstream.return_value = _models("held:a", ["keep-on", "keep-off", "new"])

        result = await self._sync("held:a")

        self.assertEqual(result["added"], ["held:a/new"])
        self.assertEqual(result["removed"], ["held:a/gone"])
        self.assertEqual(result["model_count"], 3)
        self.assertFalse(result["new_models_enabled"])
        self.assertIn("1 new (disabled until you enable them)", result["message"])
        self.assertEqual(
            await self._states("held:a"),
            {"held:a/keep-on": True, "held:a/keep-off": False, "held:a/new": False},
        )
        # Only this provider's slice changed; flags were loaded before listing.
        self.assertEqual(self._listed(), {"other:b/x", "held:a/keep-on", "held:a/keep-off", "held:a/new"})
        self.assertFalse(self.pm.model_cache._passes_global_gate("held:a/new"))
        self.assertTrue(self.pm.model_cache._passes_global_gate("held:a/keep-on"))
        self.assertEqual(self.pm.get_sync_status("held:a")["state"], "ok")
        self.assertIn("held:a", self.pm._recently_synced_providers)
        self.rebuild.assert_not_awaited()   # no registry rebuild for a manual sync

    async def test_new_models_follow_an_open_provider(self):
        await self._add_provider("open:a")
        self.upstream.return_value = _models("open:a", ["x"])
        result = await self._sync("open:a")
        self.assertIn("1 new (enabled)", result["message"])
        self.assertEqual(await self._states("open:a"), {"open:a/x": True})

    async def test_no_changes_message(self):
        await self._add_provider("open:a")
        await self._add_models({"open:a/x": False})
        self.upstream.return_value = _models("open:a", ["x"])
        result = await self._sync("open:a")
        self.assertEqual((result["added"], result["removed"]), ([], []))
        self.assertTrue(result["message"].endswith("no changes"))
        self.assertEqual(await self._states("open:a"), {"open:a/x": False})

    async def test_empty_list_keeps_saved_models(self):
        await self._add_provider("held:a", auto_enable=False)
        await self._add_models({"held:a/x": True, "held:a/y": False})
        self.upstream.return_value = []

        with self.assertRaises(HTTPException) as ctx:
            await self._sync("held:a")
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("saved models were kept", ctx.exception.detail)
        self.assertEqual(await self._states("held:a"), {"held:a/x": True, "held:a/y": False})
        self.assertEqual(self.pm.get_sync_status("held:a")["state"], "failed")

    async def test_upstream_error_keeps_rows_and_hides_listing(self):
        await self._add_provider("held:a")
        await self._add_models({"held:a/x": True})
        self.pm.model_cache.update_models(_models("held:a", ["x"]) + _models("other:b", ["y"]))
        self.upstream.side_effect = RuntimeError("connection refused\nsecret upstream body")

        with self.assertRaises(HTTPException) as ctx:
            await self._sync("held:a")
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("connection refused", ctx.exception.detail)
        self.assertNotIn("secret upstream body", ctx.exception.detail)
        self.assertEqual(await self._states("held:a"), {"held:a/x": True})
        # Same policy as every sync path: a failed provider's models are hidden.
        self.assertEqual(self._listed(), {"other:b/y"})

    async def test_disabled_provider_saves_without_listing(self):
        await self._add_provider("held:a", auto_enable=False, enabled=False)
        self.upstream.return_value = _models("held:a", ["x", "y"])

        result = await self._sync("held:a")
        self.assertFalse(result["provider_enabled"])
        self.assertIn("provider is off", result["message"])
        self.assertEqual(await self._states("held:a"), {"held:a/x": False, "held:a/y": False})
        self.assertEqual(self._listed(), set())
        self.assertIsNone(self.pm.get_sync_status("held:a"))

    async def test_unknown_and_byok_are_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._sync("nope:x")
        self.assertEqual(ctx.exception.status_code, 404)

        await self._add_provider("openai:byok-7-mine", owner=7)
        with self.assertRaises(HTTPException) as ctx:
            await self._sync("openai:byok-7-mine")
        self.assertEqual(ctx.exception.status_code, 404)
        self.upstream.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

"""A provider can hold newly discovered models back (auto_enable_new_models=False).

Every sync path gives a model it sees for the first time the provider's setting,
models already saved keep their own flag, and the serving cache treats a model with
no saved flag yet as off for such a provider, so it is never served in between.
"""

import json
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
from app.auth.models import (
    Base, BulkToggleRequest, ModelConfiguration, ProviderCredentials,
    ProviderCredentialsCreate, ProviderCredentialsUpdate, ProviderSettingsUpdate,
)
from app.cache import ModelCache
from app.openai_models import ModelInfo

ADMIN = AdminUser(username="admin", email="admin@example.com")


def _models(provider_key, names):
    return [
        ModelInfo(id=f"{provider_key}/{n}", created=0, owned_by=provider_key, provider=provider_key)
        for n in names
    ]


class CacheDefaultTests(unittest.TestCase):
    def test_unsaved_model_follows_provider_setting(self):
        cache = ModelCache()
        cache.update_model_configurations(
            {"held:a/kept-on": True, "open:a/off": False}, {"held:a": True, "open:a": True}, {"held:a"},
        )
        self.assertFalse(cache.is_model_enabled("held:a/brand-new"))
        self.assertTrue(cache.is_model_enabled("held:a/kept-on"))   # a saved flag wins
        self.assertTrue(cache.is_model_enabled("open:a/brand-new"))
        self.assertFalse(cache.is_model_enabled("open:a/off"))
        self.assertFalse(cache._passes_global_gate("held:a/brand-new"))

    def test_omitted_set_is_kept(self):
        cache = ModelCache()
        cache.update_model_configurations({}, {}, {"held:a"})
        cache.update_model_configurations({}, {})  # e.g. the DB-failure fallback
        self.assertFalse(cache.is_model_enabled("held:a/x"))
        cache.update_model_configurations({}, {}, set())
        self.assertTrue(cache.is_model_enabled("held:a/x"))


class _DbCase(unittest.IsolatedAsyncioTestCase):
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

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def _add_provider(self, key, auto_enable=True, **fields):
        async with self._factory() as db:
            db.add(ProviderCredentials(
                provider_key=key, provider_type=fields.pop("provider_type", "custom"),
                instance_name=key.split(":", 1)[-1], provider_name=key.split(":", 1)[0],
                auto_enable_new_models=auto_enable, **fields,
            ))
            await db.commit()

    async def _set_enabled(self, model_id, enabled):
        async with self._factory() as db:
            row = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.model_id == model_id)
            )).scalar_one()
            row.is_enabled = enabled
            await db.commit()

    async def _states(self, provider_key=None):
        async with self._factory() as db:
            query = select(ModelConfiguration)
            if provider_key:
                query = query.where(ModelConfiguration.provider_key == provider_key)
            return {m.model_id: m.is_enabled for m in (await db.execute(query)).scalars()}


class SyncPathTests(_DbCase):
    async def test_column_defaults_on(self):
        async with self._factory() as db:
            db.add(ProviderCredentials(provider_key="x:y", provider_type="custom",
                                       instance_name="y", provider_name="x"))
            await db.commit()
            row = (await db.execute(select(ProviderCredentials))).scalar_one()
        self.assertIs(row.auto_enable_new_models, True)

    async def test_background_sync(self):
        from app.providers.provider_manager import ProviderManager
        pm = ProviderManager()
        await self._add_provider("held:a", auto_enable=False)
        await self._add_provider("open:a")

        await pm._sync_provider_to_database("held:a", _models("held:a", ["one", "two"]))
        await pm._sync_provider_to_database("open:a", _models("open:a", ["one"]))
        self.assertEqual(await self._states("held:a"), {"held:a/one": False, "held:a/two": False})
        self.assertEqual(await self._states("open:a"), {"open:a/one": True})

        # The admin picks one; a later sync adds a model, which stays off.
        await self._set_enabled("held:a/one", True)
        await pm._sync_provider_to_database("held:a", _models("held:a", ["one", "two", "three"]))
        self.assertEqual(
            await self._states("held:a"),
            {"held:a/one": True, "held:a/two": False, "held:a/three": False},
        )

    async def test_admin_auto_sync(self):
        from app.providers import auto_sync
        await self._add_provider("held:a", auto_enable=False)

        async def run(names, creds):
            instance = SimpleNamespace(get_available_models=AsyncMock(return_value=_models("held:a", names)))
            with patch.object(auto_sync, "_create_provider_instance", AsyncMock(return_value=instance)), \
                 patch.object(auto_sync, "_close_provider_instance", AsyncMock()):
                async with self._factory() as db:
                    return await auto_sync._sync_dynamic_provider_models(db, creds)

        creds = SimpleNamespace(provider_key="held:a", provider_type="custom", auto_enable_new_models=False)
        self.assertNotIn("error", await run(["one", "two"], creds))
        await self._set_enabled("held:a/two", True)
        self.assertNotIn("error", await run(["one", "two", "three"], creds))
        self.assertEqual(
            await self._states("held:a"),
            {"held:a/one": False, "held:a/two": True, "held:a/three": False},
        )

        # A stand-in without the attribute keeps the old default (enabled).
        legacy = SimpleNamespace(provider_key="held:a", provider_type="custom")
        await run(["one", "two", "three", "four"], legacy)
        self.assertTrue((await self._states("held:a"))["held:a/four"])

    async def test_azure_manual_deployments(self):
        from app.providers import auto_sync
        await self._add_provider("azure:a", auto_enable=False, provider_type="azure")
        creds = SimpleNamespace(
            provider_key="azure:a", provider_type="azure", azure_backend="openai",
            dynamic_discovery=False, deployments_json=json.dumps(["gpt-4o", "gpt-4o-mini"]),
            auto_enable_new_models=False,
        )
        async with self._factory() as db:
            result = await auto_sync._sync_azure_provider_models(db, creds)
        self.assertNotIn("error", result)
        self.assertEqual(await self._states("azure:a"), {"azure:a/gpt-4o": False, "azure:a/gpt-4o-mini": False})

    async def test_config_load_feeds_cache(self):
        from app.providers.provider_manager import ProviderManager
        pm = ProviderManager()
        await self._add_provider("held:a", auto_enable=False)
        await self._add_provider("open:a")
        await pm.refresh_model_configurations()
        self.assertFalse(pm.model_cache.is_model_enabled("held:a/unsaved"))
        self.assertTrue(pm.model_cache.is_model_enabled("open:a/unsaved"))

    async def test_provider_change_loads_configs_before_listing(self):
        """The admin-save path must not list just-synced models before their flags load."""
        from app.providers import auto_sync
        from app.providers.provider_manager import ProviderManager
        pm = ProviderManager()
        await self._add_provider("held:a", auto_enable=False)
        seen = []

        async def sync(db, key):
            # What the real sync writes: the new model saved as off.
            async with self._factory() as s:
                s.add(ModelConfiguration(model_id="held:a/new", provider_key="held:a",
                                         model_name="new", is_enabled=False))
                await s.commit()
            return {"models": ["held:a/new"], "message": "ok"}

        real_refresh = pm.refresh_model_configurations

        async def refresh():
            seen.append({m.id for m in pm.model_cache.get_models()})
            await real_refresh()

        with patch.object(auto_sync, "provider_manager", pm), \
             patch.object(pm, "refresh_providers_from_database", AsyncMock()), \
             patch.object(pm, "refresh_model_configurations", refresh), \
             patch.object(auto_sync, "sync_provider_models", sync):
            async with self._factory() as db:
                await auto_sync.auto_sync_on_provider_change(db, "held:a", "update")

        self.assertEqual(seen, [set()])   # configs loaded while the model was not yet listed
        self.assertIn("held:a/new", {m.id for m in pm.model_cache.get_models()})
        self.assertFalse(pm.model_cache._passes_global_gate("held:a/new"))


class AdminRouteTests(_DbCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        from app.routes import admin as admin_routes
        self.r = admin_routes
        self.refresh = AsyncMock()
        for p in (
            patch.object(admin_routes, "auto_sync_on_provider_change", AsyncMock(return_value={"message": "ok"})),
            patch.object(admin_routes.provider_manager, "refresh_providers_from_database", AsyncMock()),
            patch.object(admin_routes.provider_manager, "refresh_model_configurations", self.refresh),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def _row(self, key):
        async with self._factory() as db:
            return (await db.execute(
                select(ProviderCredentials).where(ProviderCredentials.provider_key == key)
            )).scalar_one()

    async def _create(self, **fields):
        body = {"provider_type": "custom", "provider_name": "held", "instance_name": "a",
                "base_url": "https://api.example.com/v1", **fields}
        return await self.r.create_provider_credential(
            ProviderCredentialsCreate.model_validate(body), upsert=False, current_admin=ADMIN, db=self.db,
        )

    async def test_create_update_detail_and_list(self):
        created = await self._create(auto_enable_new_models=False)
        self.assertFalse(created.auto_enable_new_models)
        self.assertFalse((await self._row("held:a")).auto_enable_new_models)
        listed = {p.provider_key: p.auto_enable_new_models for p in await self.r.list_provider_credentials(ADMIN, self.db)}
        self.assertEqual(listed, {"held:a": False})

        # An edit that does not mention it leaves it alone and skips the reload.
        await self.r.update_provider_credential(
            ProviderCredentialsUpdate(api_key="k2"), provider_key="held:a", current_admin=ADMIN, db=self.db,
        )
        self.assertFalse((await self._row("held:a")).auto_enable_new_models)
        self.refresh.assert_not_awaited()

        await self.r.update_provider_credential(
            ProviderCredentialsUpdate(auto_enable_new_models=True), provider_key="held:a",
            current_admin=ADMIN, db=self.db,
        )
        self.assertTrue((await self._row("held:a")).auto_enable_new_models)
        self.refresh.assert_awaited()
        detail = await self.r.get_provider_credential(provider_key="held:a", current_admin=ADMIN, db=self.db)
        self.assertTrue(detail.auto_enable_new_models)

    async def test_create_defaults_on(self):
        created = await self._create()
        self.assertTrue(created.auto_enable_new_models)

    async def test_settings_endpoints(self):
        await self._add_provider("held:a")
        got = await self.r.get_provider_settings(provider_key="held:a", current_admin=ADMIN, db=self.db)
        self.assertTrue(got.auto_enable_new_models)

        result = await self.r.put_provider_settings(
            ProviderSettingsUpdate(auto_enable_new_models=False), provider_key="held:a",
            current_admin=ADMIN, db=self.db,
        )
        self.assertFalse(result.auto_enable_new_models)
        self.assertFalse((await self._row("held:a")).auto_enable_new_models)
        self.refresh.assert_awaited_once()

        # An empty update changes nothing and skips the cache reload.
        result = await self.r.put_provider_settings(
            ProviderSettingsUpdate(), provider_key="held:a", current_admin=ADMIN, db=self.db,
        )
        self.assertFalse(result.auto_enable_new_models)
        self.refresh.assert_awaited_once()

        for call in (
            self.r.get_provider_settings(provider_key="nope:x", current_admin=ADMIN, db=self.db),
            self.r.put_provider_settings(ProviderSettingsUpdate(auto_enable_new_models=True),
                                         provider_key="nope:x", current_admin=ADMIN, db=self.db),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await call
            self.assertEqual(ctx.exception.status_code, 404)

    def test_settings_body_rejects_unknown_fields(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ProviderSettingsUpdate.model_validate({"api_key": "x"})

    async def test_update_helper_refuses_non_settings(self):
        from app.auth.database import update_provider_settings
        await self._add_provider("held:a")
        with self.assertRaises(ValueError):
            await update_provider_settings(self.db, "held:a", {"api_key": "x"})

    async def test_rename_keeps_setting(self):
        await self._create(auto_enable_new_models=False)
        await self.r.update_provider_credential(
            ProviderCredentialsUpdate(instance_name="b"), provider_key="held:a", current_admin=ADMIN, db=self.db,
        )
        self.assertFalse((await self._row("held:b")).auto_enable_new_models)

    async def test_bulk_toggle_one_provider(self):
        await self._add_provider("held:a")
        await self._add_provider("open:a")
        async with self._factory() as db:
            for mid in ("held:a/x", "held:a/y", "open:a/x"):
                db.add(ModelConfiguration(model_id=mid, provider_key=mid.split("/")[0], model_name="m"))
            await db.commit()

        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            result = await self.r.bulk_toggle_models(
                BulkToggleRequest(action="disable_all", provider_key="held:a"), current_admin=ADMIN, db=self.db,
            )
            self.assertEqual(result["provider_key"], "held:a")
            self.assertEqual(
                await self._states(), {"held:a/x": False, "held:a/y": False, "open:a/x": True},
            )
            with self.assertRaises(HTTPException) as ctx:
                await self.r.bulk_toggle_models(
                    BulkToggleRequest(action="disable_all", provider_key="nope:x"), current_admin=ADMIN, db=self.db,
                )
            self.assertEqual(ctx.exception.status_code, 404)

    async def test_export_import_round_trip(self):
        await self._create(auto_enable_new_models=False)
        response = await self.r.export_provider_config(include_secrets=False, current_admin=ADMIN, db=self.db)
        data = json.loads(b"".join([chunk async for chunk in response.body_iterator]))
        self.assertIs(data["providers"][0]["auto_enable_new_models"], False)

        # Imported onto a fresh row under another name: the setting comes along.
        data["providers"][0].update(provider_key="held:c", instance_name="c")
        data["model_configurations"] = []
        import io
        from fastapi import UploadFile
        upload = UploadFile(file=io.BytesIO(json.dumps(data).encode()), filename="c.json")
        result = await self.r.import_provider_config(
            file=upload, overwrite=False, sync_models=False, current_admin=ADMIN, db=self.db,
        )
        self.assertEqual((result.imported, result.errors), (1, []))
        self.assertFalse((await self._row("held:c")).auto_enable_new_models)

        # An older export without the key creates the provider with the default.
        del data["providers"][0]["auto_enable_new_models"]
        data["providers"][0].update(provider_key="held:d", instance_name="d")
        upload = UploadFile(file=io.BytesIO(json.dumps(data).encode()), filename="d.json")
        await self.r.import_provider_config(
            file=upload, overwrite=False, sync_models=False, current_admin=ADMIN, db=self.db,
        )
        self.assertTrue((await self._row("held:d")).auto_enable_new_models)


if __name__ == "__main__":
    unittest.main()

"""Provider model sync writes in one short transaction per provider.

Regression for "QueuePool limit of size 5 overflow 10 reached": every provider synced
in parallel and committed once per model, so connections piled up waiting on SQLite's
single write lock until the pool ran dry.
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.auth.models import Base, ModelConfiguration, ProviderCredentials
from app.openai_models import ModelInfo


def _models(provider_key, names):
    return [
        ModelInfo(id=f"{provider_key}/{n}", created=0, owned_by=provider_key, provider=provider_key)
        for n in names
    ]


class ProviderDbSyncTestCase(unittest.IsolatedAsyncioTestCase):
    # The production pool shape; tests assert how much of it a sync actually uses.
    POOL = dict(pool_size=5, max_overflow=10, pool_timeout=2)

    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}", **self.POOL)
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

        self.commits = 0

        def _count_commit(conn):
            self.commits += 1

        event.listen(self._engine.sync_engine, "commit", _count_commit)

        # Track how many pooled connections are checked out at once.
        self.in_use = 0
        self.peak_in_use = 0

        def _checkout(*args):
            self.in_use += 1
            self.peak_in_use = max(self.peak_in_use, self.in_use)

        def _checkin(*args):
            self.in_use -= 1

        event.listen(self._engine.sync_engine.pool, "checkout", _checkout)
        event.listen(self._engine.sync_engine.pool, "checkin", _checkin)

        from app.auth import database as auth_database
        self._real_session_local = auth_database.AsyncSessionLocal
        auth_database.AsyncSessionLocal = self._factory

        from app.providers.provider_manager import ProviderManager
        self.pm = ProviderManager()

    async def asyncTearDown(self):
        from app.auth import database as auth_database
        auth_database.AsyncSessionLocal = self._real_session_local
        await self._engine.dispose()
        os.unlink(self._db_path)

    async def _add_provider(self, key):
        async with self._factory() as db:
            db.add(ProviderCredentials(
                provider_key=key, provider_type="ollama",
                instance_name=key.split(":", 1)[-1], provider_name="ollama",
            ))
            await db.commit()

    async def _rows(self, provider_key):
        async with self._factory() as db:
            result = await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.provider_key == provider_key)
            )
            return {m.model_id: m for m in result.scalars().all()}

    # -- background sync (provider_manager) ---------------------------------

    async def test_sync_commits_once_and_skips_when_unchanged(self):
        await self._add_provider("ollama:a")
        models = _models("ollama:a", [f"m{i}" for i in range(50)])

        self.commits = 0
        await self.pm._sync_provider_to_database("ollama:a", models)
        self.assertEqual(self.commits, 1)
        self.assertEqual(len(await self._rows("ollama:a")), 50)

        self.commits = 0
        await self.pm._sync_provider_to_database("ollama:a", models)
        self.assertEqual(self.commits, 0)

    async def test_sync_preserves_enabled_and_removes_stale(self):
        await self._add_provider("ollama:a")
        await self.pm._sync_provider_to_database("ollama:a", _models("ollama:a", ["keep", "gone"]))

        async with self._factory() as db:
            row = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.model_id == "ollama:a/keep")
            )).scalar_one()
            row.is_enabled = False
            await db.commit()

        self.commits = 0
        await self.pm._sync_provider_to_database("ollama:a", _models("ollama:a", ["keep", "new"]))
        self.assertEqual(self.commits, 1)

        rows = await self._rows("ollama:a")
        self.assertEqual(set(rows), {"ollama:a/keep", "ollama:a/new"})
        self.assertFalse(rows["ollama:a/keep"].is_enabled)
        self.assertTrue(rows["ollama:a/new"].is_enabled)

    async def test_stale_detection_skips_excluded_providers(self):
        from app.auth.database import identify_stale_models
        for key in ("ollama:a", "ollama:b"):
            await self._add_provider(key)
            await self.pm._sync_provider_to_database(key, _models(key, ["x", "y"]))

        # ollama:a failed to sync (nothing listed); ollama:b listed only "x".
        async with self._factory() as db:
            stale = await identify_stale_models(
                db, ["ollama:b/x"], exclude_provider_keys={"ollama:a"}
            )
        self.assertEqual({m["model_id"] for m in stale}, {"ollama:b/y"})

    async def test_parallel_provider_syncs_use_one_connection_at_a_time(self):
        keys = [f"ollama:p{i}" for i in range(8)]
        for key in keys:
            await self._add_provider(key)

        self.peak_in_use = 0
        results = await asyncio.gather(
            *(self.pm._sync_provider_to_database(k, _models(k, [f"m{i}" for i in range(30)]))
              for k in keys),
            return_exceptions=True,
        )
        self.assertEqual([r for r in results if isinstance(r, Exception)], [])
        # Writes are serialised, so the whole refresh holds a single connection
        # instead of one per provider parked on SQLite's write lock.
        self.assertEqual(self.peak_in_use, 1)
        for key in keys:
            self.assertEqual(len(await self._rows(key)), 30)

    # -- admin-triggered auto sync -------------------------------------------

    async def _auto_sync(self, key, models):
        from app.providers import auto_sync

        instance = SimpleNamespace(get_available_models=AsyncMock(return_value=models))
        creds = SimpleNamespace(provider_key=key, provider_type="ollama")
        with patch.object(auto_sync, "_create_provider_instance", AsyncMock(return_value=instance)), \
             patch.object(auto_sync, "_close_provider_instance", AsyncMock()):
            async with self._factory() as db:
                return await auto_sync._sync_dynamic_provider_models(db, creds)

    async def test_auto_sync_reinserts_same_ids_in_one_transaction(self):
        await self._add_provider("ollama:a")
        names = [f"m{i}" for i in range(20)]
        result = await self._auto_sync("ollama:a", _models("ollama:a", names))
        self.assertNotIn("error", result)

        async with self._factory() as db:
            row = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.model_id == "ollama:a/m3")
            )).scalar_one()
            row.is_enabled = False
            await db.commit()

        # Clear + re-insert of the same unique model_ids must not collide.
        self.commits = 0
        result = await self._auto_sync("ollama:a", _models("ollama:a", names))
        self.assertNotIn("error", result)
        self.assertEqual(self.commits, 1)

        rows = await self._rows("ollama:a")
        self.assertEqual(len(rows), 20)
        self.assertFalse(rows["ollama:a/m3"].is_enabled)

    async def test_auto_sync_failure_keeps_old_models(self):
        from app.providers import auto_sync

        await self._add_provider("ollama:a")
        await self._auto_sync("ollama:a", _models("ollama:a", ["old1", "old2"]))

        real = auto_sync.create_or_update_model_configuration
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return await real(*args, **kwargs)

        with patch.object(auto_sync, "create_or_update_model_configuration", flaky):
            result = await self._auto_sync("ollama:a", _models("ollama:a", ["n1", "n2", "n3"]))
        self.assertIn("error", result)
        self.assertEqual(set(await self._rows("ollama:a")), {"ollama:a/old1", "ollama:a/old2"})

    async def test_auto_sync_empty_fetch_still_clears(self):
        await self._add_provider("ollama:a")
        await self._auto_sync("ollama:a", _models("ollama:a", ["x"]))

        result = await self._auto_sync("ollama:a", [])
        self.assertEqual(result["cleared"], 1)
        self.assertEqual(await self._rows("ollama:a"), {})

    # -- failed sync hides models from the listing --------------------------

    async def _periodic_sync(self, key, fetched):
        provider = SimpleNamespace(provider_type="ollama")
        with patch.object(self.pm, "_fetch_models_with_timeout", AsyncMock(return_value=fetched)):
            await self.pm._fetch_and_sync_provider_models(key, provider)

    def _cached_ids(self):
        return {m.id for m in self.pm.model_cache.get_models()}

    async def test_failed_fetch_clears_cache_keeps_db(self):
        await self._add_provider("ollama:a")
        a_models = _models("ollama:a", ["keep", "off"])
        b_models = _models("ollama:b", ["other"])
        await self.pm._sync_provider_to_database("ollama:a", a_models)
        self.pm.model_cache.update_models(a_models + b_models)

        async with self._factory() as db:
            row = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.model_id == "ollama:a/off")
            )).scalar_one()
            row.is_enabled = False
            await db.commit()

        await self._periodic_sync("ollama:a", [])

        self.assertEqual(self._cached_ids(), {"ollama:b/other"})
        rows = await self._rows("ollama:a")
        self.assertEqual(set(rows), {"ollama:a/keep", "ollama:a/off"})
        self.assertFalse(rows["ollama:a/off"].is_enabled)

    async def test_models_return_after_recovery(self):
        await self._add_provider("ollama:a")
        a_models = _models("ollama:a", ["keep", "off"])
        await self.pm._sync_provider_to_database("ollama:a", a_models)
        async with self._factory() as db:
            row = (await db.execute(
                select(ModelConfiguration).where(ModelConfiguration.model_id == "ollama:a/off")
            )).scalar_one()
            row.is_enabled = False
            await db.commit()

        await self._periodic_sync("ollama:a", [])
        self.assertEqual(self._cached_ids(), set())

        await self._periodic_sync("ollama:a", a_models)
        self.assertEqual(self._cached_ids(), {"ollama:a/keep", "ollama:a/off"})
        self.assertFalse((await self._rows("ollama:a"))["ollama:a/off"].is_enabled)

    async def test_sync_status_recorded(self):
        provider = SimpleNamespace(provider_type="ollama", get_available_models=AsyncMock())
        self.assertIsNone(self.pm.get_sync_status("ollama:a"))

        provider.get_available_models.return_value = _models("ollama:a", ["x", "y"])
        await self.pm._fetch_models_with_timeout("ollama:a", provider)
        status = self.pm.get_sync_status("ollama:a")
        self.assertEqual((status["state"], status["model_count"], status["error"]), ("ok", 2, None))

        provider.get_available_models.side_effect = RuntimeError("connection refused")
        await self.pm._fetch_models_with_timeout("ollama:a", provider)
        status = self.pm.get_sync_status("ollama:a")
        self.assertEqual((status["state"], status["model_count"]), ("failed", 0))
        self.assertIn("connection refused", status["error"])

        provider.get_available_models.side_effect = None
        provider.get_available_models.return_value = []
        await self.pm._fetch_models_with_timeout("ollama:a", provider)
        self.assertEqual(self.pm.get_sync_status("ollama:a")["state"], "failed")

    async def test_stale_startup_fetch_does_not_clobber_newer_sync(self):
        a_models = _models("ollama:a", ["fresh"])
        self.pm.model_cache.update_models(a_models)
        released = asyncio.Event()

        async def slow_fetch():
            await released.wait()
            raise RuntimeError("timed out upstream")

        provider = SimpleNamespace(provider_type="ollama", get_available_models=slow_fetch)
        startup = asyncio.create_task(self.pm._fetch_and_sync_provider_models("ollama:a", provider))

        # An admin sync starts and finishes while the startup fetch is still in flight.
        await asyncio.sleep(0.01)
        self.assertTrue(self.pm.record_sync_status("ollama:a", True, 1))
        released.set()
        await startup

        self.assertEqual(self._cached_ids(), {"ollama:a/fresh"})
        self.assertEqual(self.pm.get_sync_status("ollama:a")["state"], "ok")

    async def _provider_change(self, key, result):
        from app.providers import auto_sync

        with patch.object(auto_sync, "provider_manager", self.pm), \
             patch.object(self.pm, "refresh_providers_from_database", AsyncMock()), \
             patch.object(self.pm, "refresh_model_configurations", AsyncMock()), \
             patch.object(auto_sync, "sync_provider_models", AsyncMock(return_value=result)):
            async with self._factory() as db:
                return await auto_sync.auto_sync_on_provider_change(db, key, "update")

    async def test_failed_admin_sync_hides_models(self):
        self.pm.model_cache.update_models(_models("ollama:a", ["x"]) + _models("ollama:b", ["y"]))

        await self._provider_change("ollama:a", {"error": "connection refused"})
        self.assertEqual(self._cached_ids(), {"ollama:b/y"})
        self.assertEqual(self.pm.get_sync_status("ollama:a")["state"], "failed")

        await self._provider_change("ollama:a", {"models": ["ollama:a/x"], "message": "ok"})
        self.assertEqual(self._cached_ids(), {"ollama:a/x", "ollama:b/y"})
        self.assertEqual(self.pm.get_sync_status("ollama:a")["state"], "ok")


if __name__ == "__main__":
    unittest.main()

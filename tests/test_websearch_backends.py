"""Tests for multiple web search backends, admin defaults and per-user choices.

The admin keeps any number of SearXNG / 4get backends and picks a default; each
user may pick another selectable backend and override its search parameters and
the result limits. Anything a user leaves unset follows the admin default, and a
pick that becomes unavailable falls back to the default backend.
"""

import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.auth import database
from app.auth.admin import AdminUser
from app.auth.models import (
    ToggleRequest, UserWebSearchBackendParams, UserWebSearchSettings, UserWebSearchUpdate,
    WebSearchBackendCreate, WebSearchSettingsUpdate,
)
from app.routes import admin as admin_routes
from app.routes import user_websearch as user_routes
from app.websearch.settings import websearch_settings_cache
from tests.pool_test_base import PoolTestCase

ADMIN = AdminUser(username="root", email="root@example.test")


class _WebSearchTestCase(PoolTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._saved_snapshot = websearch_settings_cache.snapshot

    async def asyncTearDown(self):
        websearch_settings_cache._snapshot = self._saved_snapshot
        await super().asyncTearDown()

    async def add_backend(self, name, type="searxng", **kw):
        body = WebSearchBackendCreate(name=name, type=type, base_url=f"http://{name.lower()}", **kw)
        return await admin_routes.create_websearch_backend_endpoint(body, ADMIN, self.db)

    async def save_settings(self, default_backend_id, **kw):
        body = WebSearchSettingsUpdate(enabled=True, default_backend_id=default_backend_id,
                                       enabled_providers=["*"], **kw)
        return await admin_routes.update_websearch_settings_endpoint(body, ADMIN, self.db)

    async def update_backend(self, backend, **changes):
        fields = backend.model_dump(include=set(WebSearchBackendCreate.model_fields))
        fields.update(changes)
        return await admin_routes.update_websearch_backend_endpoint(
            WebSearchBackendCreate(**fields), backend.id, ADMIN, self.db,
        )

    async def put_prefs(self, user, **kw):
        return await user_routes.put_my_websearch(UserWebSearchUpdate(**kw), user, self.db)

    def resolve(self, user):
        return websearch_settings_cache.resolve_config(user.id)

    async def setup_three_backends(self):
        """A (default SearXNG), B (SearXNG with an engine list), C (4get)."""
        self.a = await self.add_backend("A", language="en", safesearch=2)
        self.b = await self.add_backend("B", engines="bing")
        self.c = await self.add_backend("C", type="fourget", fourget_scraper="brave", fourget_country="us")
        await self.save_settings(self.a.id, max_results=5, timeout_seconds=7)
        self.alice = await self.make_user("alice")


class ResolutionTests(_WebSearchTestCase):
    async def test_user_without_choices_gets_the_default(self):
        await self.setup_three_backends()
        cfg = self.resolve(self.alice)
        self.assertIs(cfg, websearch_settings_cache.config)
        self.assertEqual(cfg.base_url, "http://a")
        self.assertEqual(cfg.language, "en")
        self.assertEqual(cfg.safesearch, 2)
        self.assertIs(websearch_settings_cache.resolve_config(None), websearch_settings_cache.config)

    async def test_picked_backend_with_its_params_and_the_users(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, time_range="week", max_results=9)
        cfg = self.resolve(self.alice)
        self.assertEqual(cfg.base_url, "http://b")
        self.assertEqual(cfg.backend_name, "B")
        self.assertEqual(cfg.engines, "bing")          # B's own
        self.assertEqual(cfg.language, "auto")         # B left it unset -> global default
        self.assertEqual(cfg.time_range, "week")       # the user's
        self.assertEqual(cfg.max_results, 9)
        self.assertEqual(cfg.max_snippet_chars, 500)   # admin default
        self.assertEqual(cfg.timeout_seconds, 7)       # admin only
        self.assertTrue(cfg.enabled)

    async def test_engines_and_categories_override_together(self):
        await self.setup_three_backends()
        # B sends engines=bing; SearXNG ignores categories when engines is set, so
        # the user's categories must replace the inherited engine list.
        await self.put_prefs(self.alice, backend_id=self.b.id, categories="news")
        cfg = self.resolve(self.alice)
        self.assertEqual(cfg.categories, "news")
        self.assertIsNone(cfg.engines)

    async def test_fourget_locale_follows_the_users_scraper(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.c.id, fourget_scraper="ddg")
        cfg = self.resolve(self.alice)
        self.assertEqual(cfg.provider, "fourget")
        self.assertEqual(cfg.base_url, "http://c")
        self.assertEqual(cfg.fourget_scraper, "ddg")
        self.assertIsNone(cfg.fourget_country)   # C's "us" was for brave
        self.assertIsNone(cfg.engines)           # no SearXNG leftovers

    async def test_params_are_kept_per_backend(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de")
        await self.put_prefs(self.alice, backend_id=None, language="fr")  # for the default, A
        self.assertEqual(self.resolve(self.alice).language, "fr")
        payload = await self.put_prefs(self.alice, backend_id=self.b.id, language="de")
        self.assertEqual(self.resolve(self.alice).language, "de")
        saved = payload["selection"]["params_by_backend"]
        self.assertEqual(saved[str(self.a.id)]["language"], "fr", "A's params survived the switch")
        # A PUT replaces the params of its target backend only: none clears B's.
        payload = await self.put_prefs(self.alice, backend_id=self.b.id)
        self.assertEqual(self.resolve(self.alice).language, "auto")
        self.assertEqual(list(payload["selection"]["params_by_backend"]), [str(self.a.id)])

    async def test_unavailable_pick_falls_back_to_the_default(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de", max_results=9)
        await self.put_prefs(self.alice, backend_id=None, language="fr")
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de", max_results=9)

        for change in ({"user_selectable": False}, {"enabled": False}):
            await self.update_backend(self.b, **change)
            cfg = self.resolve(self.alice)
            self.assertEqual(cfg.base_url, "http://a", change)
            self.assertEqual(cfg.language, "fr", "the user's params for the default backend apply")
            self.assertEqual(cfg.max_results, 9, "limits do not depend on the backend")
            payload = await user_routes.get_my_websearch(self.alice, self.db)
            self.assertTrue(payload["effective"]["fell_back"])
            self.assertEqual(payload["effective"]["chosen_backend_id"], self.b.id)
            await self.update_backend(self.b, enabled=True, user_selectable=True)

        self.assertEqual(self.resolve(self.alice).base_url, "http://b", "restoring B restores the pick")

    async def test_the_default_backend_is_selectable_even_when_hidden(self):
        await self.setup_three_backends()
        await self.update_backend(self.a, user_selectable=False)
        await self.put_prefs(self.alice, backend_id=self.a.id)
        self.assertEqual(self.resolve(self.alice).base_url, "http://a")


class AdminRouteTests(_WebSearchTestCase):
    async def test_first_backend_becomes_the_default(self):
        a = await self.add_backend("A")
        self.assertTrue(a.is_default)
        b = await self.add_backend("B")
        self.assertFalse(b.is_default)
        settings = await admin_routes.get_websearch_settings_endpoint(ADMIN, self.db)
        self.assertEqual(settings.default_backend, {"id": a.id, "name": "A", "type": "searxng"})

    async def test_names_are_unique_ignoring_case(self):
        a = await self.add_backend("Main")
        with self.assertRaises(HTTPException) as ctx:
            await self.add_backend("main")
        self.assertEqual(ctx.exception.status_code, 409)
        b = await self.add_backend("Other")
        with self.assertRaises(HTTPException) as ctx:
            await self.update_backend(b, name="MAIN")
        self.assertEqual(ctx.exception.status_code, 409)
        renamed = await self.update_backend(a, name="main")   # its own name, new case
        self.assertEqual(renamed.name, "main")

    async def test_underscore_in_a_name_is_not_a_wildcard(self):
        await self.add_backend("a-b")
        await self.add_backend("a b")
        created = await self.add_backend("a_b")   # ilike matched both rows: a 500
        self.assertEqual(created.name, "a_b")

    async def test_type_cannot_change(self):
        a = await self.add_backend("A")
        with self.assertRaises(HTTPException) as ctx:
            await self.update_backend(a, type="fourget")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_default_backend_cannot_be_disabled_or_deleted(self):
        a = await self.add_backend("A")
        for action in (
            lambda: admin_routes.toggle_websearch_backend_endpoint(ToggleRequest(enabled=False), a.id, ADMIN, self.db),
            lambda: self.update_backend(a, enabled=False),
            lambda: admin_routes.delete_websearch_backend_endpoint(a.id, ADMIN, self.db),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await action()
            self.assertEqual(ctx.exception.status_code, 409)

    async def test_settings_need_an_enabled_existing_default(self):
        a = await self.add_backend("A")
        b = await self.add_backend("B")
        await admin_routes.toggle_websearch_backend_endpoint(ToggleRequest(enabled=False), b.id, ADMIN, self.db)
        for bad in (b.id, 999):
            with self.assertRaises(HTTPException) as ctx:
                await self.save_settings(bad)
            self.assertEqual(ctx.exception.status_code, 422)
        saved = await self.save_settings(a.id)
        self.assertEqual(saved.default_backend_id, a.id)
        self.assertTrue(websearch_settings_cache.is_active("chat_completions", "custom:x"))

    async def test_deleting_a_backend_returns_its_users_to_the_default(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de")
        listed = {b.name: b for b in await admin_routes.list_websearch_backends_endpoint(ADMIN, self.db)}
        self.assertEqual(listed["B"].user_count, 1)

        await admin_routes.delete_websearch_backend_endpoint(self.b.id, ADMIN, self.db)
        row = (await self.db.execute(
            select(UserWebSearchSettings).where(UserWebSearchSettings.user_id == self.alice.id)
        )).scalar_one()
        self.assertIsNone(row.backend_id)
        params = (await self.db.execute(select(UserWebSearchBackendParams))).scalars().all()
        self.assertEqual(params, [])
        self.assertEqual(self.resolve(self.alice).base_url, "http://a")

    async def test_test_endpoint_uses_an_unsaved_backend(self):
        await self.setup_three_backends()
        seen = []

        async def fake_search(query, cfg):
            seen.append(cfg)
            return SimpleNamespace(ok=True, results=[])

        with patch("app.websearch.backends.search", fake_search):
            body = admin_routes.WebSearchTestRequest(
                query="q", backend_id=self.b.id,
                backend=WebSearchBackendCreate(name="B", type="searxng", base_url="http://new", engines="ddg"),
            )
            await admin_routes.test_websearch_endpoint(body, ADMIN, self.db)
            await admin_routes.test_websearch_endpoint(
                admin_routes.WebSearchTestRequest(query="q", backend_id=self.c.id), ADMIN, self.db,
            )
        self.assertEqual((seen[0].base_url, seen[0].engines), ("http://new", "ddg"))
        self.assertEqual((seen[1].provider, seen[1].base_url), ("fourget", "http://c"))
        self.assertEqual(seen[1].timeout_seconds, 7)

    async def test_engines_without_a_url_use_a_searxng_backend_when_the_default_is_4get(self):
        from app.auth.models import WebSearchEnginesRequest

        await self.setup_three_backends()
        await self.save_settings(self.c.id)
        calls = []

        async def fake_fetch(base_url, timeout):
            calls.append(base_url)
            return {"engines": [], "categories": []}

        with patch("app.websearch.searxng.fetch_engines", fake_fetch):
            await admin_routes.list_websearch_engines(WebSearchEnginesRequest(), ADMIN, self.db)
        self.assertEqual(calls, ["http://a"])


class UserRouteTests(_WebSearchTestCase):
    async def test_payload_hides_urls_and_unpickable_backends(self):
        await self.setup_three_backends()
        await self.update_backend(self.c, user_selectable=False)
        await self.update_backend(self.a, user_selectable=False)   # the default stays listed
        payload = await user_routes.get_my_websearch(self.alice, self.db)
        self.assertEqual([b["name"] for b in payload["backends"]], ["A", "B"])
        self.assertNotIn("http://", repr(payload))
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["defaults"]["backend_id"], self.a.id)
        a = payload["backends"][0]
        self.assertTrue(a["is_default"])
        self.assertEqual(a["params"]["language"], "en")
        self.assertEqual(payload["effective"]["backend_name"], "A")
        self.assertFalse(payload["effective"]["fell_back"])

    async def test_cannot_pick_an_unavailable_backend(self):
        await self.setup_three_backends()
        await self.update_backend(self.c, user_selectable=False)
        with self.assertRaises(HTTPException) as ctx:
            await self.put_prefs(self.alice, backend_id=self.c.id)
        self.assertEqual(ctx.exception.status_code, 422)
        with self.assertRaises(HTTPException) as ctx:
            await self.put_prefs(self.alice, backend_id=999)
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_an_unavailable_pick_can_be_sent_back_unchanged(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id)
        await self.update_backend(self.b, user_selectable=False)
        payload = await user_routes.get_my_websearch(self.alice, self.db)
        self.assertEqual(payload["effective"]["chosen_backend_name"], "B")

        # Saving other fields keeps the pick; the params go to the backend in use.
        payload = await self.put_prefs(self.alice, backend_id=self.b.id, max_results=4, language="it")
        self.assertEqual(payload["selection"]["backend_id"], self.b.id)
        self.assertEqual(list(payload["selection"]["params_by_backend"]), [str(self.a.id)])
        self.assertEqual((self.resolve(self.alice).base_url, self.resolve(self.alice).max_results), ("http://a", 4))

        await self.update_backend(self.b, user_selectable=True)
        self.assertEqual(self.resolve(self.alice).base_url, "http://b", "the kept pick applies again")

    async def test_a_concurrent_first_save_is_retried(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de")
        real = database.get_user_websearch
        calls = []

        async def stale_first_read(db, user_id):
            # The other tab's insert landed after this request read: it saw nothing.
            # Expunging makes the session forget the row too, as a separate one would.
            calls.append(user_id)
            if len(calls) == 1:
                db.expunge_all()
                return None, {}
            return await real(db, user_id)

        with patch.object(database, "get_user_websearch", stale_first_read):
            await self.put_prefs(self.alice, backend_id=self.c.id, fourget_scraper="ddg")
        self.assertEqual(len(calls), 2)
        cfg = self.resolve(self.alice)
        self.assertEqual((cfg.base_url, cfg.fourget_scraper), ("http://c", "ddg"))

    async def test_put_then_reset(self):
        await self.setup_three_backends()
        payload = await self.put_prefs(self.alice, backend_id=self.b.id, categories="news", max_results=3)
        self.assertEqual(payload["selection"]["backend_id"], self.b.id)
        self.assertEqual(payload["selection"]["params_by_backend"][str(self.b.id)]["categories"], "news")
        self.assertEqual(payload["effective"]["params"]["categories"], "news")
        self.assertEqual(payload["effective"]["max_results"], 3)

        payload = await user_routes.reset_my_websearch(self.alice, self.db)
        self.assertIsNone(payload["selection"]["backend_id"])
        self.assertEqual(payload["selection"]["params_by_backend"], {})
        self.assertIs(self.resolve(self.alice), websearch_settings_cache.config)
        self.assertEqual((await self.db.execute(select(UserWebSearchSettings))).scalars().all(), [])

    async def test_putting_nothing_leaves_no_rows(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, max_results=3)
        await self.put_prefs(self.alice)
        self.assertEqual((await self.db.execute(select(UserWebSearchSettings))).scalars().all(), [])

    async def test_admin_is_refused(self):
        with self.assertRaises(HTTPException) as ctx:
            await user_routes.get_my_websearch(ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_engines_only_for_pickable_searxng_backends(self):
        await self.setup_three_backends()
        with self.assertRaises(HTTPException) as ctx:
            await user_routes.get_backend_engines(self.c.id, self.alice, self.db)  # 4get
        self.assertEqual(ctx.exception.status_code, 404)

        calls = []

        async def fake_fetch(base_url, timeout):
            calls.append(base_url)
            return {"engines": [], "categories": ["general"]}

        user_routes._engines_cache.clear()
        with patch("app.websearch.searxng.fetch_engines", fake_fetch):
            await user_routes.get_backend_engines(self.b.id, self.alice, self.db)
            await user_routes.get_backend_engines(self.b.id, self.alice, self.db)
        self.assertEqual(calls, ["http://b"], "the second call is served from the cache")
        user_routes._engines_cache.clear()

    async def test_deleting_the_user_removes_their_choices(self):
        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id, language="de")
        self.assertTrue(await database.permanently_delete_user(self.db, self.alice.id))
        for model in (UserWebSearchSettings, UserWebSearchBackendParams):
            self.assertEqual((await self.db.execute(select(model))).scalars().all(), [])


class InterceptUserTests(_WebSearchTestCase):
    """maybe_intercept_* run with the calling user's config."""

    async def test_chat_and_responses_use_the_users_config(self):
        from app.websearch import maybe_intercept_chat, maybe_intercept_responses

        await self.setup_three_backends()
        await self.put_prefs(self.alice, backend_id=self.b.id)
        chat = SimpleNamespace(model_dump=lambda **_: {
            "model": "custom:x/m", "messages": [{"role": "user", "content": "hi"}], "web_search_options": {},
        })
        loop = maybe_intercept_chat(chat, "custom:x", None, None, user_id=self.alice.id)
        self.assertEqual(loop.cfg.base_url, "http://b")
        self.assertEqual(maybe_intercept_chat(chat, "custom:x", None, None).cfg.base_url, "http://a")

        responses = SimpleNamespace(model_dump=lambda **_: {"input": "hi", "tools": [{"type": "web_search"}]})
        loop = maybe_intercept_responses(responses, "custom:x", None, user_id=self.alice.id)
        self.assertEqual(loop.cfg.base_url, "http://b")


class WebSearchBackendMigrationTests(unittest.IsolatedAsyncioTestCase):
    """The single-backend websearch_settings row becomes backend rows.

    The old table is built with raw DDL first: create_all skips tables that already
    exist, so it keeps the old shape and the migration path is what gets exercised.
    """

    OLD_SETTINGS = """
    CREATE TABLE websearch_settings (
        id INTEGER NOT NULL PRIMARY KEY,
        enabled BOOLEAN NOT NULL,
        provider VARCHAR(20),
        searxng_base_url VARCHAR(500),
        engines VARCHAR(500),
        categories VARCHAR(200),
        language VARCHAR(20),
        safesearch INTEGER,
        time_range VARCHAR(10),
        fourget_base_url VARCHAR(500),
        fourget_scraper VARCHAR(50),
        fourget_lang VARCHAR(20),
        fourget_country VARCHAR(32),
        max_results INTEGER,
        max_snippet_chars INTEGER,
        timeout_seconds INTEGER,
        max_agentic_loops INTEGER,
        max_queries_per_turn INTEGER,
        apply_to TEXT,
        enabled_providers TEXT,
        updated_at DATETIME,
        updated_by VARCHAR(50)
    )"""

    async def _start(self, insert):
        from app.auth.models import Base

        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with sqlite3.connect(self.path) as raw:
            raw.execute(self.OLD_SETTINGS)
            if insert:
                raw.execute(insert)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
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
        for name in os.listdir(os.path.dirname(self.path)):
            full = os.path.join(os.path.dirname(self.path), name)
            if full.startswith(self.path):
                os.unlink(full)

    def _query(self, sql):
        with sqlite3.connect(self.path) as raw:
            return raw.execute(sql).fetchall()

    async def test_both_backends_move_and_the_selected_one_is_default(self):
        await self._start(
            "INSERT INTO websearch_settings (id, enabled, provider, searxng_base_url, engines, safesearch, "
            "time_range, fourget_base_url, fourget_scraper, fourget_country, max_results) VALUES "
            "(1, 1, 'fourget', 'http://s/', 'bing', 2, 'week', 'http://f', 'brave', 'us', 7)"
        )
        await database._run_auto_migrations()
        backends = self._query(
            "SELECT id, name, type, base_url, engines, safesearch, time_range, fourget_scraper, "
            "fourget_country FROM websearch_backends ORDER BY id"
        )
        self.assertEqual([b[1:] for b in backends], [
            ("SearXNG", "searxng", "http://s", "bing", 2, "week", None, None),
            ("4get", "fourget", "http://f", None, 2, "week", "brave", "us"),
        ])
        settings = self._query("SELECT enabled, default_backend_id, max_results FROM websearch_settings")
        self.assertEqual(settings, [(1, backends[1][0], 7)])
        self.assertTrue(any(n.startswith(os.path.basename(self.path) + ".pre-websearch-backends")
                            for n in os.listdir(os.path.dirname(self.path))), "backed up first")

        await database._run_auto_migrations()   # a second start finds nothing to do
        self.assertEqual(len(self._query("SELECT id FROM websearch_backends")), 2)

        # The ORM reads the migrated table, legacy columns and all.
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        from app.websearch.settings import load_snapshot
        async with sessionmaker(self.engine, class_=AsyncSession)() as db:
            snapshot = await load_snapshot(db)
        self.assertEqual(snapshot.default.base_url, "http://f")
        self.assertEqual(snapshot.default.fourget_scraper, "brave")

    async def test_unconfigured_fourget_falls_back_to_searxng(self):
        await self._start(
            "INSERT INTO websearch_settings (id, enabled, provider, searxng_base_url) "
            "VALUES (1, 1, 'fourget', 'http://s')"
        )
        await database._run_auto_migrations()
        backends = self._query("SELECT id, type FROM websearch_backends")
        self.assertEqual([b[1] for b in backends], ["searxng"])
        self.assertEqual(self._query("SELECT enabled, default_backend_id FROM websearch_settings"),
                         [(1, backends[0][0])])

    async def test_existing_backends_are_reused_for_the_default(self):
        await self._start(
            "INSERT INTO websearch_settings (id, enabled, provider, searxng_base_url) "
            "VALUES (1, 1, 'searxng', 'http://s')"
        )
        with sqlite3.connect(self.path) as raw:
            raw.execute(
                "INSERT INTO websearch_backends (name, type, base_url, enabled, user_selectable) "
                "VALUES ('Kept', 'searxng', 'http://kept', 1, 1)"
            )
        await database._run_auto_migrations()
        backends = self._query("SELECT id, name FROM websearch_backends")
        self.assertEqual([b[1] for b in backends], ["Kept"], "not duplicated")
        self.assertEqual(self._query("SELECT enabled, default_backend_id FROM websearch_settings"),
                         [(1, backends[0][0])])

    async def test_never_configured(self):
        await self._start(None)
        await database._run_auto_migrations()
        self.assertEqual(self._query("SELECT id FROM websearch_backends"), [])
        columns = [r[1] for r in self._query("PRAGMA table_info(websearch_settings)")]
        self.assertIn("default_backend_id", columns)


if __name__ == "__main__":
    unittest.main()

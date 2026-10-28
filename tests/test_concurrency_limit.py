"""Per-profile cap on requests in progress at once (max_concurrent_requests).

Covers the tracker's slot counting, enforcement through both charging paths (the route
helper and the auth middleware), slot release over the whole ASGI call however it ends,
the admin API, /quotas, and the column migration.
"""

import asyncio
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.auth.admin import AdminUser
from app.auth.middleware import _enforce_rate_limit
from app.auth.models import ProfileCreate, ProfileOverallLimitsUpdate
from app.concurrency_slots import SLOT_SCOPE_KEY, ConcurrencySlotMiddleware, _SlotHolder
from app.main import _add_request_tracking
from app.profiles import profile_registry
from app.rate_limit import RateLimitExceeded
from app.rate_limit_dep import enforce_group_rate_limit
from app.request_tracker import request_tracker
from app.routes import admin as admin_routes
from tests.pool_test_base import PoolTestCase

ADMIN = AdminUser(username="root", email="root@example.test")


def _scope(path, method="POST"):
    return {
        "type": "http", "method": method, "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [], "scheme": "http", "http_version": "1.1",
        "server": ("test", 80), "client": ("client", 1), "root_path": "",
    }


def _request(path="/v1/chat/completions"):
    """A request carrying a slot holder, as ConcurrencySlotMiddleware would give it."""
    holder = _SlotHolder()
    scope = _scope(path)
    scope[SLOT_SCOPE_KEY] = holder
    return Request(scope), holder


def _auth(user):
    return SimpleNamespace(id=user.id, user_id=user.id, username=user.username)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TrackerSlotTests(PoolTestCase):

    async def test_cap_denies_once_full_and_frees_on_release(self):
        alice = await self.make_user("alice")
        team = await self.make_profile("team", max_concurrent_requests=2)
        await self.set_profile(alice, team.id)

        self.assertIsNone(self.tracker.try_acquire_slot(alice.id))
        self.assertIsNone(self.tracker.try_acquire_slot(alice.id))
        denied = self.tracker.try_acquire_slot(alice.id)
        self.assertEqual((denied.allowed, denied.limited_by, denied.concurrency_limit),
                         (False, "concurrency", 2))
        self.assertEqual(self.tracker.inflight_for(alice.id), 2, "a denial takes no slot")

        self.tracker.release_slot(alice.id)
        self.assertIsNone(self.tracker.try_acquire_slot(alice.id))

    async def test_unlimited_users_are_counted_but_never_denied(self):
        alice = await self.make_user("alice")
        for _ in range(50):
            self.assertIsNone(self.tracker.try_acquire_slot(alice.id))
        self.assertEqual(self.tracker.inflight_for(alice.id), 50)

    async def test_a_cap_set_mid_flight_counts_requests_already_open(self):
        alice = await self.make_user("alice")
        for _ in range(3):
            self.tracker.try_acquire_slot(alice.id)
        team = await self.make_profile("team", max_concurrent_requests=2)
        await self.set_profile(alice, team.id)
        self.assertEqual(self.tracker.try_acquire_slot(alice.id).limited_by, "concurrency")

    async def test_release_never_goes_negative_and_drops_the_key(self):
        alice = await self.make_user("alice")
        self.tracker.release_slot(alice.id)
        self.assertNotIn(alice.id, self.tracker._inflight)
        self.tracker.try_acquire_slot(alice.id)
        self.tracker.release_slot(alice.id)
        self.tracker.release_slot(alice.id)
        self.assertNotIn(alice.id, self.tracker._inflight)
        self.assertEqual(self.tracker.inflight_for(alice.id), 0)

    async def test_moving_profile_changes_the_cap(self):
        alice = await self.make_user("alice")
        one = await self.make_profile("one", max_concurrent_requests=1)
        three = await self.make_profile("three", max_concurrent_requests=3)
        await self.set_profile(alice, one.id)
        self.assertEqual(self.tracker.concurrency_limit_for(alice.id), 1)
        await self.set_profile(alice, three.id)
        self.assertEqual(self.tracker.concurrency_limit_for(alice.id), 3)


# ---------------------------------------------------------------------------
# Enforcement through both charging paths
# ---------------------------------------------------------------------------

class EnforcementTests(PoolTestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.alice = await self.make_user("alice")

    async def _cap(self, n, **limits):
        team = await self.make_profile("team", max_concurrent_requests=n, **limits)
        await self.set_profile(self.alice, team.id)

    async def test_route_helper_denies_without_spending_rpm(self):
        await self._cap(1)
        first, _ = _request()
        await enforce_group_rate_limit(first, _auth(self.alice), "p/m")
        second, _ = _request()
        with self.assertRaises(RateLimitExceeded) as ctx:
            await enforce_group_rate_limit(second, _auth(self.alice), "p/m")
        self.assertIn("Concurrency limit exceeded", ctx.exception.body["error"]["message"])
        self.assertEqual(self.tracker._minute_buckets[self.alice.id].count, 1,
                         "only the admitted request reached the RPM bucket")

    async def test_auth_middleware_path_denies_without_spending_rpm(self):
        await self._cap(1)
        first, _ = _request("/v1/files")
        await _enforce_rate_limit(first, _auth(self.alice))
        second, _ = _request("/v1/files")
        with self.assertRaises(RateLimitExceeded):
            await _enforce_rate_limit(second, _auth(self.alice))
        self.assertEqual(self.tracker._minute_buckets[self.alice.id].count, 1)

    async def test_an_rpm_denial_gives_the_slot_back(self):
        await self._cap(5, rpm_limit=1)
        ok, _ = _request()
        await enforce_group_rate_limit(ok, _auth(self.alice), "p/m")
        for path_fn in (
            lambda r: enforce_group_rate_limit(r, _auth(self.alice), "p/m"),
            lambda r: _enforce_rate_limit(r, _auth(self.alice)),
        ):
            req, holder = _request()
            with self.assertRaises(RateLimitExceeded) as ctx:
                await path_fn(req)
            self.assertIn("requests per minute", ctx.exception.body["error"]["message"])
            self.assertIsNone(holder.user_id)
            self.assertEqual(self.tracker.inflight_for(self.alice.id), 1, "only the admitted one")

    async def test_enforcing_twice_in_one_request_counts_once(self):
        await self._cap(1)
        req, holder = _request()
        await enforce_group_rate_limit(req, _auth(self.alice), "p/m")
        await enforce_group_rate_limit(req, _auth(self.alice), "p/m")
        self.assertEqual(self.tracker.inflight_for(self.alice.id), 1)
        holder.release()
        holder.release()
        self.assertEqual(self.tracker.inflight_for(self.alice.id), 0)

    async def test_a_later_denial_in_the_same_request_keeps_the_earlier_slot(self):
        await self._cap(5, rpm_limit=1)
        req, holder = _request()
        await enforce_group_rate_limit(req, _auth(self.alice), "p/m")
        with self.assertRaises(RateLimitExceeded):
            await enforce_group_rate_limit(req, _auth(self.alice), "p/m")
        self.assertEqual(holder.user_id, self.alice.id, "the first check's slot is still held")
        self.assertEqual(self.tracker.inflight_for(self.alice.id), 1)

    async def test_admins_are_exempt(self):
        req, holder = _request()
        await enforce_group_rate_limit(req, ADMIN, "p/m")
        await _enforce_rate_limit(req, ADMIN)
        self.assertIsNone(holder.user_id)
        self.assertEqual(self.tracker._inflight, {})

    async def test_a_request_without_a_holder_is_not_counted(self):
        await self._cap(1)
        bare = Request(_scope("/v1/chat/completions"))
        for _ in range(3):
            await enforce_group_rate_limit(bare, _auth(self.alice), "p/m")
        self.assertEqual(self.tracker._inflight, {})

    async def test_each_envelope_carries_the_concurrency_body_and_headers(self):
        await self._cap(1)
        hold, _ = _request()
        await enforce_group_rate_limit(hold, _auth(self.alice), "p/m")
        for path, get_msg in (
            ("/v1/chat/completions", lambda b: b["error"]["message"]),
            ("/v1/messages", lambda b: b["error"]["message"]),
            ("/openai/deployments/d/chat/completions", lambda b: b["error"]["message"]),
        ):
            with self.subTest(path=path):
                req, _ = _request(path)
                with self.assertRaises(RateLimitExceeded) as ctx:
                    await enforce_group_rate_limit(req, _auth(self.alice), "p/m")
                exc = ctx.exception
                self.assertEqual(exc.body["error"]["type"], "rate_limit_error")
                self.assertIn("at most 1 request may be in progress", get_msg(exc.body))
                self.assertEqual(exc.headers, {
                    "Retry-After": "1",
                    "X-RateLimit-Limit-Concurrent-Requests": "1",
                    "X-RateLimit-Remaining-Concurrent-Requests": "0",
                })
        anthropic, _ = _request("/v1/messages")
        with self.assertRaises(RateLimitExceeded) as ctx:
            await enforce_group_rate_limit(anthropic, _auth(self.alice), "p/m")
        self.assertEqual(ctx.exception.body["type"], "error")


# ---------------------------------------------------------------------------
# Slot lifetime: the whole ASGI call, however it ends
# ---------------------------------------------------------------------------

class SlotLifetimeTests(PoolTestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.alice = await self.make_user("alice")
        team = await self.make_profile("team", max_concurrent_requests=1)
        await self.set_profile(self.alice, team.id)

        patcher = patch.multiple(request_tracker, start_request=AsyncMock(), end_request=AsyncMock())
        patcher.start()
        self.addCleanup(patcher.stop)

        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        auth = _auth(self.alice)
        app = FastAPI()
        _add_request_tracking(app, "openai")

        @app.exception_handler(RateLimitExceeded)
        async def _rl(request, exc):
            return JSONResponse(status_code=429, content=exc.body, headers=exc.headers)

        @app.get("/v1/stream")
        async def stream(request: Request):
            await enforce_group_rate_limit(request, auth, "p/m")

            async def body():
                self.started.set()
                await self.finish.wait()
                yield b"done"

            return StreamingResponse(body(), media_type="text/plain")

        @app.get("/v1/boom")
        async def boom(request: Request):
            await enforce_group_rate_limit(request, auth, "p/m")
            raise RuntimeError("upstream exploded")

        self.app = app

    async def _call(self, path, send=None):
        """Run one request through the app; returns the messages sent."""
        sent = []
        disconnect = asyncio.Event()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"", "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def default_send(message):
            sent.append(message)

        await self.app(_scope(path, "GET"), receive, send or default_send)
        return sent

    @staticmethod
    def _status(sent):
        return next(m["status"] for m in sent if m["type"] == "http.response.start")

    def _inflight(self):
        return self.tracker.inflight_for(self.alice.id)

    async def test_held_while_streaming_and_freed_when_the_stream_ends(self):
        first = asyncio.create_task(self._call("/v1/stream"))
        await asyncio.wait_for(self.started.wait(), 5)
        self.assertEqual(self._inflight(), 1)

        second = await self._call("/v1/stream")
        self.assertEqual(self._status(second), 429)
        self.assertEqual(self._inflight(), 1)

        self.finish.set()
        self.assertEqual(self._status(await asyncio.wait_for(first, 5)), 200)
        self.assertEqual(self._inflight(), 0)

    async def test_freed_when_the_route_raises(self):
        with self.assertRaises(RuntimeError):
            await self._call("/v1/boom")
        self.assertEqual(self._inflight(), 0)

    async def test_freed_when_send_fails_mid_stream(self):
        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("connection reset")

        self.finish.set()
        with self.assertRaises(OSError):
            await self._call("/v1/stream", send=send)
        self.assertEqual(self._inflight(), 0)

    async def test_freed_when_the_call_is_cancelled(self):
        task = asyncio.create_task(self._call("/v1/stream"))
        await asyncio.wait_for(self.started.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self._inflight(), 0)


def test_every_api_app_wraps_its_routes_in_slots():
    from app import main

    for name in ("create_openai_app", "create_anthropic_app", "create_azure_openai_app"):
        layers = [m.cls for m in getattr(main, name)().user_middleware]
        assert ConcurrencySlotMiddleware in layers, name
        # Just inside tracking, which test_user_id_span_attribute pins at index 0.
        assert layers.index(ConcurrencySlotMiddleware) == 1, (name, layers)


# ---------------------------------------------------------------------------
# Admin API and /quotas
# ---------------------------------------------------------------------------

class AdminApiTests(PoolTestCase):

    async def _team(self):
        return await self.make_profile("team", rpm_limit=4)

    async def test_set_clear_and_leave_unchanged(self):
        team = await self._team()
        resp = await admin_routes.update_profile_limits_endpoint(
            ProfileOverallLimitsUpdate(max_concurrent_requests=3), team.id, ADMIN, self.db)
        self.assertEqual((resp.max_concurrent_requests, resp.rpm_limit), (3, 4))
        self.assertEqual(profile_registry.get(team.id).max_concurrent, 3, "applies at once")

        resp = await admin_routes.update_profile_limits_endpoint(
            ProfileOverallLimitsUpdate(rpm_limit=8), team.id, ADMIN, self.db)
        self.assertEqual((resp.max_concurrent_requests, resp.rpm_limit), (3, 8),
                         "an omitted cap is left alone")

        resp = await admin_routes.update_profile_limits_endpoint(
            ProfileOverallLimitsUpdate(max_concurrent_requests=None), team.id, ADMIN, self.db)
        self.assertIsNone(resp.max_concurrent_requests)
        self.assertIsNone(profile_registry.get(team.id).max_concurrent)

        limits = await admin_routes.get_profile_limits_endpoint(team.id, ADMIN, self.db)
        self.assertIsNone(limits.profile.max_concurrent_requests)

    def test_zero_and_negative_are_rejected(self):
        for bad in (0, -1):
            with self.assertRaises(ValidationError):
                ProfileOverallLimitsUpdate(max_concurrent_requests=bad)

    async def test_only_rpm_rpd_changes_resettle_pools(self):
        team = await self._team()
        with patch.object(admin_routes, "_resettle_profile", AsyncMock()) as resettle, \
             patch.object(self.tracker, "reload_config", AsyncMock()) as reload:
            await admin_routes.update_profile_limits_endpoint(
                ProfileOverallLimitsUpdate(max_concurrent_requests=2), team.id, ADMIN, self.db)
            resettle.assert_not_awaited()
            reload.assert_awaited_once()

            await admin_routes.update_profile_limits_endpoint(
                ProfileOverallLimitsUpdate(rpm_limit=1, max_concurrent_requests=2), team.id, ADMIN, self.db)
            resettle.assert_awaited_once()

    async def test_listed_and_cloned(self):
        team = await self.make_profile("team", max_concurrent_requests=5)
        listed = {p.id: p for p in await admin_routes.list_profiles_endpoint(ADMIN, self.db)}
        self.assertEqual(listed[team.id].max_concurrent_requests, 5)
        clone = await admin_routes.create_profile_endpoint(
            ProfileCreate(name="copy", clone_from_id=team.id), ADMIN, self.db)
        self.assertEqual(clone.max_concurrent_requests, 5)

    async def test_quotas_reports_the_cap_and_current_count(self):
        from app.providers.provider_manager import provider_manager
        from app.routes import auth as auth_routes

        alice = await self.make_user("alice")
        team = await self.make_profile("team", max_concurrent_requests=4)
        await self.set_profile(alice, team.id)
        self.tracker.try_acquire_slot(alice.id)
        with patch.object(provider_manager, "get_all_models", AsyncMock(return_value=[])):
            quotas = await auth_routes.get_my_quotas(alice, self.db)
        self.assertEqual((quotas.max_concurrent_requests, quotas.in_flight), (4, 1))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_adds_the_column_to_an_existing_profiles_table(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.auth import database

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE user_profiles (id INTEGER PRIMARY KEY, name VARCHAR(64) NOT NULL, "
        "description VARCHAR(256), rpm_limit INTEGER, rpd_limit INTEGER, "
        "model_access_mode VARCHAR(16) NOT NULL DEFAULT 'default', created_at DATETIME, "
        "updated_at DATETIME, updated_by VARCHAR(50))"
    )
    conn.execute("INSERT INTO user_profiles (id, name, rpm_limit) VALUES (1, 'Default', 10), (2, 'Team', 5)")
    conn.commit()
    conn.close()

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "_backup_sqlite_file", lambda tag: None)

    async def run():
        for _ in range(2):
            await database.create_tables_async()
            await database._run_auto_migrations()
        await engine.dispose()

    try:
        asyncio.run(run())
        conn = sqlite3.connect(path)
        columns = [r[1] for r in conn.execute("PRAGMA table_info(user_profiles)").fetchall()]
        assert columns.count("max_concurrent_requests") == 1
        assert conn.execute(
            "SELECT id, rpm_limit, max_concurrent_requests FROM user_profiles ORDER BY id"
        ).fetchall() == [(1, 10, None), (2, 5, None)]
        conn.close()
    finally:
        os.unlink(path)

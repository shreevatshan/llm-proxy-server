"""User profiles: limits, model access and mappings come from the user's profile.

Covers the enforcement side (rate limiter, model access, alias resolution against a
pre-auth mapping with Default's list), the admin API, and the migration that moves an
existing database onto profiles.
"""

import asyncio
import os
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.auth.admin import AdminUser
from app.auth.models import (
    DEFAULT_PROFILE_ID, AdminUserCreate, BulkUserActionRequest, ModelAlias, ModelConfiguration,
    ProfileCreate, ProfileLimitsUpdate, ProfileOverallLimitsUpdate, ProviderCredentials, User,
    UserProfile,
)
from app.model_alias import (
    alias_input_name, alias_resolved_name, alias_resolved_scope, alias_seen_names,
    apply_alias, build_rule, current_api_surface, current_profile_id, model_alias_resolver,
    original_model_name,
)
from app.profiles import DEFAULT_PROFILE_ID as DEFAULT
from app.profiles import ProfileSnapshot, profile_id_for_auth, profile_registry
from app.routes import admin as admin_routes
from tests.pool_test_base import PoolTestCase

ADMIN = AdminUser(username="root", email="root@example.test")


# ---------------------------------------------------------------------------
# Registry and model access
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    yield profile_registry
    profile_registry.reset()


def test_unknown_users_and_empty_registry_fall_back_to_default(registry):
    assert registry.for_user(42).id == DEFAULT_PROFILE_ID
    assert registry.for_user(42).rpm is None
    registry.install({1: 5}, [ProfileSnapshot(1, "Default", rpm=3), ProfileSnapshot(5, "Team", rpm=9)])
    assert registry.for_user(1).rpm == 9
    assert registry.for_user(2).rpm == 3, "not yet loaded -> default profile"
    registry.set_users_profile([2], 5)
    assert registry.for_user(2).name == "Team"


def test_config_admin_has_no_profile(registry):
    registry.install({0: 5}, [ProfileSnapshot(5, "Team")])
    assert profile_id_for_auth(ADMIN) is None
    assert profile_id_for_auth(SimpleNamespace(user_id=0, id=3)) == DEFAULT_PROFILE_ID


def test_model_access_follows_the_users_profile(registry):
    from app.cache import ModelCache
    cache = ModelCache()
    cache.update_model_configurations({"p:a/m1": True, "p:a/m2": False}, {"p:a": True})
    registry.install({1: 2, 2: 3, 3: 4}, [
        ProfileSnapshot(2, "deny", access_mode="deny"),
        ProfileSnapshot(3, "allow", access_mode="allow"),
        ProfileSnapshot(4, "custom", access_mode="custom",
                        access_exceptions={"p:a/m1": False, "p:a/m2": True}),
    ])
    assert not cache.is_model_allowed_for_user(1, "p:a/m1")
    assert cache.is_model_allowed_for_user(2, "p:a/m2"), "allow beats the global gate"
    assert not cache.is_model_allowed_for_user(3, "p:a/m1")
    assert cache.is_model_allowed_for_user(3, "p:a/m2")
    # A user in no profile yet follows the global gate.
    assert cache.is_model_allowed_for_user(99, "p:a/m1")
    assert not cache.is_model_allowed_for_user(99, "p:a/m2")


def test_models_with_access_shows_profile_disabled_models(registry):
    from app.cache import ModelCache
    from app.openai_models import ModelInfo
    cache = ModelCache()
    cache.update_models([ModelInfo(id=i, created=0, owned_by="x", provider="p:a") for i in ("p:a/m1", "p:a/m2")])
    cache.update_model_configurations({"p:a/m1": True, "p:a/m2": False}, {"p:a": True})
    registry.install({1: 2, 2: 3, 3: 4}, [
        ProfileSnapshot(2, "deny", access_mode="deny"),
        ProfileSnapshot(3, "allow", access_mode="allow"),
        ProfileSnapshot(4, "custom", access_mode="custom",
                        access_exceptions={"p:a/m1": False, "p:a/m2": True}),
    ])

    def access(user_id):
        return {m.id: allowed for m, allowed in cache.get_models_with_access_for_user(user_id)}

    # Globally-off models stay hidden unless the profile allows them.
    assert access(1) == {"p:a/m1": False}
    assert access(2) == {"p:a/m1": True, "p:a/m2": True}
    assert access(3) == {"p:a/m1": False, "p:a/m2": True}
    assert access(99) == {"p:a/m1": True}


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

@pytest.fixture
def aliases():
    saved = (model_alias_resolver._snapshot, model_alias_resolver._profile_snapshots,
             model_alias_resolver._fallback_snapshot, model_alias_resolver._fallback_profile_snapshots)
    tokens = [(v, v.set(d)) for v, d in (
        (original_model_name, None), (current_api_surface, "openai"), (alias_resolved_name, None),
        (alias_input_name, None), (alias_resolved_scope, None), (alias_seen_names, frozenset()),
        (current_profile_id, None),
    )]
    model_alias_resolver._profile_snapshots = {}
    model_alias_resolver._fallback_profile_snapshots = {}
    # "gpt" is no model, so requests for it are mapped by the fallback lists; the
    # same rules sit in both lists so apply_alias's default (regular) list sees them too.
    for kind in ("alias", "fallback"):
        model_alias_resolver.set_rules([build_rule("gpt", "p/default", rule_id=1)], kind=kind)
        model_alias_resolver.set_rules([build_rule("gpt", "p/team", rule_id=2)], profile_id=7, kind=kind)
    yield model_alias_resolver
    (model_alias_resolver._snapshot, model_alias_resolver._profile_snapshots,
     model_alias_resolver._fallback_snapshot, model_alias_resolver._fallback_profile_snapshots) = saved
    for var, token in reversed(tokens):
        var.reset(token)


def test_each_profile_maps_with_its_own_list_only(aliases):
    assert aliases.resolve("gpt", "openai", 7) == "p/team"
    assert aliases.resolve("gpt", "openai", DEFAULT) == "p/default"
    assert aliases.resolve("gpt", "openai", None) == "p/default", "before auth, and the config admin"
    assert aliases.resolve("gpt", "openai", 8) == "gpt", "no fallback to Default's list"


def test_pre_auth_default_mapping_is_corrected_once_the_profile_is_known(aliases):
    # Middleware, before auth: the profile is unknown, so Default's list applies.
    assert apply_alias("gpt") == "p/default"
    # Auth binds the profile; the handler re-reads the rewritten body ("p/default").
    current_profile_id.set(7)
    assert apply_alias("p/default") == "p/team"
    # Feeding the result back in never chains.
    assert apply_alias("p/team") == "p/team"
    assert original_model_name.get() == "gpt", "the client's name is still echoed"


def test_profile_without_rules_undoes_the_pre_auth_mapping(aliases):
    assert apply_alias("gpt") == "p/default"
    current_profile_id.set(8)
    assert apply_alias("p/default") == "gpt"
    assert apply_alias("gpt") == "gpt"


def test_default_members_keep_the_pre_auth_mapping(aliases):
    assert apply_alias("gpt") == "p/default"
    current_profile_id.set(DEFAULT)
    assert apply_alias("p/default") == "p/default"


def test_profile_mapping_applies_directly_after_auth(aliases):
    current_profile_id.set(7)
    assert apply_alias("gpt") == "p/team"
    assert apply_alias("p/team") == "p/team"


def test_bind_profile_context_retags_the_request(aliases, registry):
    from unittest.mock import AsyncMock, patch
    from app.auth.middleware import _bind_profile_context
    from app.cache import ModelCache
    from app.openai_models import ModelInfo
    from app.providers.provider_manager import provider_manager

    # After auth a fallback is only taken when its target can be served.
    cache = ModelCache()
    cache.update_models([ModelInfo(id="p/team", created=0, owned_by="t", provider="t")])
    stub = SimpleNamespace(supports_api_for_model=lambda model, api: True)
    registry.install({5: 7}, [ProfileSnapshot(7, "team")])
    assert apply_alias("gpt", kind="fallback") == "p/default"    # what the middleware did
    request = SimpleNamespace(state=SimpleNamespace(model="p/default", tracking_request_id="r1"))
    with patch.object(provider_manager, "model_cache", cache), \
         patch.object(provider_manager, "providers", {"p": stub}), \
         patch("app.request_tracker.request_tracker.update_model", new_callable=AsyncMock) as upd:
        asyncio.run(_bind_profile_context(request, SimpleNamespace(user_id=5, id=1)))
    assert request.state.model == "p/team", "the overall-limit check sees the final model"
    upd.assert_awaited_once_with("r1", "p/team")


def test_bind_profile_context_restores_the_name_for_a_profile_without_mappings(aliases, registry):
    from unittest.mock import AsyncMock, patch
    from app.auth.middleware import _bind_profile_context

    registry.install({6: 8}, [ProfileSnapshot(8, "empty")])
    assert apply_alias("gpt", kind="fallback") == "p/default"
    request = SimpleNamespace(state=SimpleNamespace(model="p/default", tracking_request_id="r1"))
    with patch("app.request_tracker.request_tracker.update_model", new_callable=AsyncMock) as upd:
        asyncio.run(_bind_profile_context(request, SimpleNamespace(user_id=6, id=1)))
    assert request.state.model == "gpt"
    upd.assert_awaited_once_with("r1", "gpt")


def test_bind_profile_context_rejudges_default_members_by_their_access(aliases, registry):
    """Before auth only the global gate is known; a Default member whose access denies
    the model gets the fallback list, so the pre-auth regular mapping is undone."""
    from unittest.mock import AsyncMock, patch
    from app.auth.middleware import _bind_profile_context
    from app.cache import ModelCache
    from app.openai_models import ModelInfo
    from app.providers.provider_manager import provider_manager

    cache = ModelCache()
    cache.update_models([ModelInfo(id=m, created=0, owned_by="t", provider="t")
                         for m in ("p/gpt", "p/fallback")])
    stub = SimpleNamespace(supports_api_for_model=lambda model, api: True)
    model_alias_resolver.set_rules([build_rule("gpt", "p/mapped", rule_id=3)])            # regular
    model_alias_resolver.set_rules([build_rule("gpt", "p/fallback", rule_id=4)], kind="fallback")
    registry.install({5: DEFAULT}, [ProfileSnapshot(DEFAULT, "Default", access_mode="custom",
                                                    access_exceptions={"p/gpt": False})])
    with patch.object(provider_manager, "model_cache", cache), \
         patch.object(provider_manager, "providers", {"p": stub}):
        from app.model_resolution import map_model_name
        assert map_model_name("gpt") == "p/mapped", "pre-auth: globally available"
        request = SimpleNamespace(state=SimpleNamespace(model="p/mapped", tracking_request_id="r1"))
        with patch("app.request_tracker.request_tracker.update_model", new_callable=AsyncMock) as upd:
            asyncio.run(_bind_profile_context(request, SimpleNamespace(user_id=5, id=1)))
    assert request.state.model == "p/fallback"
    upd.assert_awaited_once_with("r1", "p/fallback")


# ---------------------------------------------------------------------------
# Rate limits and admin API, against a real tracker and temp DB
# ---------------------------------------------------------------------------

class ProfileRateLimitTests(PoolTestCase):

    async def test_overall_limits_come_from_the_profile_and_blank_is_unlimited(self):
        alice = await self.make_user("alice")
        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual((status.rpm_limit, status.rpd_limit), (None, None))

        team = await self.make_profile("team", rpm_limit=1, rpd_limit=50)
        await self.set_profile(alice, team.id)
        self.assertTrue((await self.tracker.check_and_increment(alice.id, "alice")).allowed)
        denied = await self.tracker.check_and_increment(alice.id, "alice")
        self.assertEqual((denied.allowed, denied.limited_by), (False, "rpm"))

    async def test_moving_profile_keeps_the_minute_count(self):
        alice = await self.make_user("alice")
        loose = await self.make_profile("loose", rpm_limit=5)
        tight = await self.make_profile("tight", rpm_limit=2)
        await self.set_profile(alice, loose.id)
        for _ in range(2):
            await self.tracker.check_and_increment(alice.id, "alice")
        await self.set_profile(alice, tight.id)
        self.assertFalse((await self.tracker.check_and_increment(alice.id, "alice")).allowed)

    async def test_group_limits_come_from_the_profile(self):
        alice = await self.make_user("alice")
        mg = await self.make_model_group("opus", "p/opus", rpd_default=None)
        ig = await self.make_instance_group("edge", "edge", rpd_default=None)
        team = await self.make_profile("team")
        await self.set_group_limit(team.id, mg, rpm=1)
        await self.set_group_limit(team.id, ig, rpm=1)
        self.assertIsNone(await self.tracker.check_group_limit(alice.id, "alice", "p/opus"),
                          "Default has no limit on the group")
        await self.set_profile(alice, team.id)
        self.assertIsNone(await self.tracker.check_group_limit(alice.id, "alice", "p/opus"))
        self.assertEqual(
            (await self.tracker.check_group_limit(alice.id, "alice", "p/opus")).limited_by, "group_rpm")
        # A model on the instance, synced after the group was made, is governed by it.
        self.assertIsNone(await self.tracker.check_group_limit(alice.id, "alice", "edge/m"))
        self.assertEqual(
            (await self.tracker.check_group_limit(alice.id, "alice", "edge/m")).limited_by,
            "group_rpm")

    async def test_pooled_daily_limit_sums_across_profiles(self):
        alice = await self.make_user("alice", rpd_limit=10)
        bob = await self.make_user("bob", rpd_limit=30)
        await self.make_pool("team", alice, [bob])
        status = await self.tracker.get_user_status(alice.id, "alice")
        self.assertEqual(status.rpd_limit, 40)

    # -- admin API ------------------------------------------------------

    async def test_profile_crud_and_clone(self):
        mg = await self.make_model_group("opus", "p/opus", rpd_default=7)
        created = await admin_routes.create_profile_endpoint(
            ProfileCreate(name="Team A", clone_from_id=DEFAULT_PROFILE_ID), ADMIN, self.db)
        limits = await admin_routes.get_profile_limits_endpoint(created.id, ADMIN, self.db)
        self.assertEqual([g.rpd_limit for g in limits.groups if g.group_id == mg.id], [7],
                         "a clone carries the group limits")

        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.create_profile_endpoint(ProfileCreate(name="team a"), ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 409, "names are case-insensitive")

        await admin_routes.update_profile_limits_endpoint(
            ProfileOverallLimitsUpdate(rpm_limit=4), created.id, ADMIN, self.db)
        listed = {p.id: p for p in await admin_routes.list_profiles_endpoint(ADMIN, self.db)}
        self.assertEqual(listed[created.id].rpm_limit, 4)
        self.assertTrue(listed[DEFAULT_PROFILE_ID].is_default)

    async def test_apply_to_all_profiles_leaves_omitted_fields_alone(self):
        mg = await self.make_model_group("opus", "p/opus", rpd_default=None)
        team = await self.make_profile("team")
        await self.set_group_limit(DEFAULT_PROFILE_ID, mg, rpd=500)
        await self.set_group_limit(team.id, mg, rpd=50)

        result = await admin_routes.update_model_group_limit_all_profiles_endpoint(
            ProfileLimitsUpdate(rpm_limit=10), mg.id, ADMIN, self.db)
        self.assertEqual(result, {"group_id": mg.id, "rpm_limit": 10},
                         "the omitted RPD is not reported as null (unlimited)")

        for pid, rpd in ((DEFAULT_PROFILE_ID, 500), (team.id, 50)):
            limits = await admin_routes.get_profile_limits_endpoint(pid, ADMIN, self.db)
            row = next(g for g in limits.groups if g.group_id == mg.id)
            self.assertEqual((row.rpm_limit, row.rpd_limit), (10, rpd),
                             "an omitted RPD keeps each profile's own value")

    async def test_deleting_a_profile_moves_its_members_and_default_is_protected(self):
        alice = await self.make_user("alice")
        team = await self.make_profile("team", rpm_limit=1)
        await self.set_profile(alice, team.id)
        self.db.add(ModelAlias(alias="x", target_model_id="p/x", profile_id=team.id, enabled=True))
        await self.db.commit()

        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.delete_profile_endpoint(DEFAULT_PROFILE_ID, DEFAULT_PROFILE_ID, ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

        result = await admin_routes.delete_profile_endpoint(team.id, DEFAULT_PROFILE_ID, ADMIN, self.db)
        self.assertEqual(result["moved_count"], 1)
        await self.db.refresh(alice)
        self.assertEqual(alice.profile_id, DEFAULT_PROFILE_ID)
        self.assertEqual(self.tracker._own_rpm_limit(alice.id), None, "the move applies at once")
        self.assertEqual((await self.db.execute(select(ModelAlias))).scalars().all(), [],
                         "the profile's mappings go with it")

    async def test_create_user_with_profile_and_validation(self):
        team = await self.make_profile("team", rpm_limit=2)
        user = await admin_routes.create_user_admin(
            AdminUserCreate(username="  carol  ", email="carol@x.test", password="secret1",
                            profile_id=team.id), ADMIN, self.db)
        self.assertEqual((user.username, user.profile_id), ("carol", team.id))
        self.assertEqual(self.tracker._own_rpm_limit(user.id), 2)

        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.create_user_admin(
                AdminUserCreate(username=" ab ", email="ab@x.test", password="secret1"), ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 400)
        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.create_user_admin(
                AdminUserCreate(username="dave", email="d@x.test", password="secret1",
                                profile_id=999), ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_bulk_assign_profile(self):
        alice = await self.make_user("alice")
        bob = await self.make_user("bob")
        team = await self.make_profile("team", rpd_limit=3)
        result = await admin_routes.bulk_user_action(
            BulkUserActionRequest(user_ids=[alice.id, bob.id, 999], action="assign_profile",
                                  profile_id=team.id), ADMIN, self.db)
        self.assertEqual((result["succeeded_count"], result["failed_count"]), (2, 1))
        self.assertEqual(self.tracker._own_rpd_limit(bob.id), 3)

    async def test_profile_model_access_toggle_seeds_custom_from_deny(self):
        self.db.add(ProviderCredentials(provider_key="p:a", provider_type="custom",
                                        instance_name="a", provider_name="p"))
        self.db.add_all([
            ModelConfiguration(model_id="p:a/m1", provider_key="p:a", model_name="m1"),
            ModelConfiguration(model_id="p:a/m2", provider_key="p:a", model_name="m2"),
        ])
        await self.db.commit()
        team = await self.make_profile("team")
        from app.routes.admin import UserModelExceptionUpdate, UserModelPolicyUpdate
        await admin_routes.update_profile_model_policy(UserModelPolicyUpdate(mode="deny"), team.id, ADMIN, self.db)
        view = await admin_routes.upsert_profile_model_access_exception(
            UserModelExceptionUpdate(is_allowed=True), team.id, "p:a/m1", ADMIN, self.db)
        self.assertEqual(view.mode, "custom")
        self.assertEqual({m.model_id: m.effective_allowed for m in view.models},
                         {"p:a/m1": True, "p:a/m2": False})
        snap = profile_registry.get(team.id)
        self.assertEqual((snap.access_mode, dict(snap.access_exceptions)),
                         ("custom", {"p:a/m1": True, "p:a/m2": False}))


# ---------------------------------------------------------------------------
# Alias uniqueness per profile
# ---------------------------------------------------------------------------

class AliasScopeTests(PoolTestCase):

    async def test_same_alias_text_in_two_profiles_but_not_twice_in_one(self):
        team = await self.make_profile("team")
        self.db.add_all([
            ModelAlias(alias="gpt", target_model_id="p/a", enabled=True),
            ModelAlias(alias="gpt", target_model_id="p/b", enabled=True, profile_id=team.id),
        ])
        await self.db.commit()
        rows = (await self.db.execute(select(ModelAlias).order_by(ModelAlias.id))).scalars().all()
        self.assertEqual([r.profile_id for r in rows], [DEFAULT_PROFILE_ID, team.id],
                         "a mapping without a profile is Default's")
        self.db.add(ModelAlias(alias="gpt", target_model_id="p/c", enabled=True))
        with self.assertRaises(IntegrityError):
            await self.db.commit()
        await self.db.rollback()


# ---------------------------------------------------------------------------
# Migration of an existing (pre-profile) database
# ---------------------------------------------------------------------------

_LEGACY_DDL = [
    "CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(50) NOT NULL UNIQUE, "
    "email VARCHAR(100) NOT NULL UNIQUE, hashed_password VARCHAR(255), created_at DATETIME, "
    "is_active BOOLEAN, is_pending_approval BOOLEAN DEFAULT 0)",
    "CREATE TABLE global_rate_limits (id INTEGER PRIMARY KEY, rpm_default INTEGER, rpd_default INTEGER, "
    "updated_at DATETIME, updated_by VARCHAR(50))",
    "CREATE TABLE model_groups (id INTEGER PRIMARY KEY, name VARCHAR(64) NOT NULL UNIQUE, "
    "description VARCHAR(256), rpm_default INTEGER, rpd_default INTEGER, created_at DATETIME, "
    "updated_at DATETIME, updated_by VARCHAR(50))",
    "CREATE TABLE model_aliases (id INTEGER NOT NULL PRIMARY KEY, alias VARCHAR(200) NOT NULL, "
    "target_model_id VARCHAR(200) NOT NULL, enabled BOOLEAN NOT NULL, created_at DATETIME, "
    "updated_at DATETIME, apis TEXT, match_type VARCHAR(16) NOT NULL DEFAULT 'exact', "
    "priority INTEGER NOT NULL DEFAULT 0)",
    "CREATE UNIQUE INDEX ix_model_aliases_alias ON model_aliases (alias)",
    "INSERT INTO users (id, username, email, is_active) VALUES (1, 'alice', 'a@x', 1), (2, 'bob', 'b@x', 1)",
    "INSERT INTO global_rate_limits (id, rpm_default, rpd_default) VALUES (1, 30, 1500)",
    "INSERT INTO model_groups (id, name, rpd_default) VALUES (1, 'opus', 100)",
    "INSERT INTO model_aliases (id, alias, target_model_id, enabled) VALUES (1, 'gpt', 'p/gpt', 1)",
]


def test_migration_seeds_default_from_global_values_and_is_idempotent(monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.auth import database

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    for ddl in _LEGACY_DDL:
        conn.execute(ddl)
    conn.commit()
    conn.close()

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "_backup_sqlite_file", lambda tag: None)

    async def run():
        for _ in range(2):
            await database.create_tables_async()
            await database._run_auto_migrations()
            async with engine.begin() as c:
                from sqlalchemy import text
                await c.execute(text("UPDATE user_profiles SET rpm_limit = 99 WHERE id = 1"))
        await engine.dispose()

    try:
        asyncio.run(run())
        conn = sqlite3.connect(path)
        assert conn.execute("SELECT id, name, rpd_limit FROM user_profiles").fetchall() == [(1, "Default", 1500)]
        assert conn.execute("SELECT rpm_limit FROM user_profiles").fetchone() == (99,), \
            "a second start must not re-seed over an admin's edit"
        assert conn.execute("SELECT DISTINCT profile_id FROM users").fetchall() == [(1,)]
        assert conn.execute(
            "SELECT profile_id, group_id, rpd_limit FROM profile_model_group_limits").fetchall() == [(1, 1, 100)]
        assert conn.execute(
            "SELECT profile_id, alias, kind FROM model_aliases ORDER BY kind").fetchall() == [
                (1, "gpt", "alias"), (1, "gpt", "fallback")], \
            "mappings from before profiles become Default's, in both lists"
        conn.execute("INSERT INTO user_profiles (id, name, model_access_mode) VALUES (2, 'Team', 'default')")
        conn.execute("INSERT INTO model_aliases (alias, target_model_id, enabled, profile_id) "
                     "VALUES ('gpt', 'p/team', 1, 2)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO model_aliases (alias, target_model_id, enabled, profile_id) "
                         "VALUES ('gpt', 'p/x', 1, 1)")
        conn.close()
    finally:
        os.unlink(path)

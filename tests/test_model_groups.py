"""Tests for the single model-group concept and the fixes that came with it.

Groups are managed in one place and hold explicit models and/or whole provider
instances; every model no group claims is in the built-in Default group. Alongside
that: a group stays editable when a member disappears upstream, references follow a
provider rename or removal (but survive a model sync), and the provider and model
enable flags no longer overwrite each other.
"""

import asyncio
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.auth import database
from app.auth.admin import AdminUser
from app.auth.models import (
    ModelAlias, ModelConfiguration, ModelGroupCreate, ModelGroupInstanceMember, ModelGroupMember,
    ModelGroupMembersUpdate, ModelGroupUpdate, ProfileModelAccessException, ProfileModelGroupLimit,
    ProviderCredentials, ToggleRequest,
)
from app.cache import ModelCache
from app.profiles import DEFAULT_GROUP_ID
from app.routes import admin as admin_routes
from tests.pool_test_base import PoolTestCase

ADMIN = AdminUser(username="root", email="root@example.test")


class _GroupTestCase(PoolTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Group edits re-settle pools; nothing here is pooled, so keep it cheap and quiet.
        async def resettle(*_a, **_k):
            await self.tracker.refresh_now()   # what resettle_pools does first
        self._resettle = patch("app.routes.pools.resettle_pools_after_commit", resettle)
        self._resettle.start()

    async def asyncTearDown(self):
        self._resettle.stop()
        await super().asyncTearDown()

    async def provider(self, key, *models, enabled=True):
        ptype, name = key.split(":", 1)
        self.db.add(ProviderCredentials(provider_key=key, provider_type=ptype, instance_name=name,
                                        provider_name=ptype, enabled=enabled))
        for m in models:
            self.db.add(ModelConfiguration(model_id=f"{key}/{m}", provider_key=key, model_name=m))
        await self.db.commit()

    async def groups(self):
        return {g.name: g for g in await admin_routes.get_model_groups(None, ADMIN, self.db)}


class MembershipApiTests(_GroupTestCase):

    async def test_a_group_holds_models_and_instances_and_default_holds_the_rest(self):
        await self.provider("azure:east", "gpt-5", "gpt-4o")
        await self.provider("openai:main", "o3", "o4")
        await admin_routes.create_model_group_endpoint(ModelGroupCreate(
            name="premium", provider_keys=["azure:east"], model_ids=["openai:main/o3"]), ADMIN, self.db)

        groups = await self.groups()
        self.assertEqual(list(groups)[0], "Default", "Default is listed first")
        self.assertEqual(groups["Default"].effective_models, ["openai:main/o4"])
        self.assertEqual(sorted(groups["premium"].effective_models),
                         ["azure:east/gpt-4o", "azure:east/gpt-5", "openai:main/o3"])
        instance = next(m for m in groups["premium"].members if m.kind == "instance")
        self.assertEqual(sorted(instance.covers), ["azure:east/gpt-4o", "azure:east/gpt-5"])

        # Live: a model synced from the instance later joins the group by itself.
        await self.provider_model("azure:east", "gpt-6")
        self.assertIn("azure:east/gpt-6", (await self.groups())["premium"].effective_models)
        self.assertEqual(self.tracker.group_for("azure:east/gpt-6"), groups["premium"].id)

    async def provider_model(self, key, name):
        self.db.add(ModelConfiguration(model_id=f"{key}/{name}", provider_key=key, model_name=name))
        await self.db.commit()

    async def test_a_member_belongs_to_one_group_only(self):
        await self.provider("azure:east", "gpt-5")
        await admin_routes.create_model_group_endpoint(
            ModelGroupCreate(name="a", provider_keys=["azure:east"], model_ids=["azure:east/gpt-5"]),
            ADMIN, self.db)
        for body in (ModelGroupCreate(name="b", provider_keys=["azure:east"]),
                     ModelGroupCreate(name="c", model_ids=["azure:east/gpt-5"])):
            with self.subTest(body=body):
                with self.assertRaises(HTTPException) as ctx:
                    await admin_routes.create_model_group_endpoint(body, ADMIN, self.db)
                self.assertEqual(ctx.exception.status_code, 409)
                self.assertIn("'a'", ctx.exception.detail["message"], "the owning group is named")

    async def test_names_are_unique_ignoring_case_and_default_is_reserved(self):
        await admin_routes.create_model_group_endpoint(ModelGroupCreate(name="Fast"), ADMIN, self.db)
        for name in ("fast", "Default", "default"):
            with self.subTest(name=name):
                with self.assertRaises(HTTPException) as ctx:
                    await admin_routes.create_model_group_endpoint(ModelGroupCreate(name=name), ADMIN, self.db)
                self.assertEqual(ctx.exception.status_code, 409)

    async def test_the_default_group_cannot_be_edited(self):
        for call in (
            admin_routes.update_model_group_endpoint(DEFAULT_GROUP_ID, ModelGroupUpdate(name="x"), ADMIN, self.db),
            admin_routes.delete_model_group_endpoint(DEFAULT_GROUP_ID, ADMIN, self.db),
            admin_routes.set_model_group_members_endpoint(DEFAULT_GROUP_ID, ModelGroupMembersUpdate(), ADMIN, self.db),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await call
            self.assertEqual(ctx.exception.status_code, 400)

    async def test_a_member_deleted_upstream_does_not_lock_the_group(self):
        """Regression: saving a group with a stale member used to fail with 404 forever."""
        await self.provider("azure:east", "gpt-5", "gpt-4o")
        created = await admin_routes.create_model_group_endpoint(
            ModelGroupCreate(name="g", model_ids=["azure:east/gpt-5", "azure:east/gpt-4o"]), ADMIN, self.db)
        model = (await self.db.execute(select(ModelConfiguration).where(
            ModelConfiguration.model_id == "azure:east/gpt-5"))).scalar_one()
        await self.db.delete(model)   # as a model sync that drops it would
        await self.db.commit()

        status = {m.key: m.status for m in (await self.groups())["g"].members}
        self.assertEqual(status["azure:east/gpt-5"], "missing")

        # Saving with it kept, or with it removed, both work.
        await admin_routes.set_model_group_members_endpoint(created.id, ModelGroupMembersUpdate(
            model_ids=["azure:east/gpt-5", "azure:east/gpt-4o"]), ADMIN, self.db)
        await admin_routes.set_model_group_members_endpoint(created.id, ModelGroupMembersUpdate(
            model_ids=["azure:east/gpt-4o"]), ADMIN, self.db)
        # A *new* member must still exist.
        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.set_model_group_members_endpoint(created.id, ModelGroupMembersUpdate(
                model_ids=["azure:east/nope"]), ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_a_new_group_starts_with_each_profiles_default_limits(self):
        await self.provider("azure:east", "gpt-5")
        team = await self.make_profile("team", rpm_limit=5, rpd_limit=50)
        free = await self.make_profile("free")
        created = await admin_routes.create_model_group_endpoint(
            ModelGroupCreate(name="g", model_ids=["azure:east/gpt-5"]), ADMIN, self.db)

        rows = {r.profile_id: (r.rpm_limit, r.rpd_limit) for r in (await self.db.execute(
            select(ProfileModelGroupLimit).where(ProfileModelGroupLimit.group_id == created.id))).scalars()}
        self.assertEqual(rows.get(team.id), (5, 50))
        self.assertNotIn(free.id, rows, "an unlimited Default stays unlimited")

    async def test_a_model_carved_out_of_a_group_keeps_that_groups_limits(self):
        """Regression: it was seeded with Default's limits, so 50/day became 1000/day."""
        await self.provider("azure:east", "gpt-5", "gpt-4o")
        await self.provider("openai:main", "o3")
        team = await self.make_profile("team", rpm_limit=100, rpd_limit=1000)
        edge = await admin_routes.create_model_group_endpoint(
            ModelGroupCreate(name="edge", provider_keys=["azure:east"]), ADMIN, self.db)
        await self.set_group_limit(team.id, edge, rpd=50)

        async def seeded(body):
            created = await admin_routes.create_model_group_endpoint(body, ADMIN, self.db)
            row = (await self.db.execute(select(ProfileModelGroupLimit).where(
                ProfileModelGroupLimit.group_id == created.id,
                ProfileModelGroupLimit.profile_id == team.id))).scalar_one_or_none()
            return row and (row.rpm_limit, row.rpd_limit)

        self.assertEqual(await seeded(ModelGroupCreate(name="carved", model_ids=["azure:east/gpt-5"])),
                         (None, 50))
        # Members from Default and from edge: the strictest of each limit.
        self.assertEqual(await seeded(ModelGroupCreate(
            name="mixed", model_ids=["azure:east/gpt-4o", "openai:main/o3"])), (100, 50))

    async def test_a_rejected_member_list_leaves_the_name_alone(self):
        """Regression: the rename and the members were two requests, so a 409 on the
        members left the group half-saved."""
        await self.provider("azure:east", "gpt-5")
        await admin_routes.create_model_group_endpoint(
            ModelGroupCreate(name="a", provider_keys=["azure:east"]), ADMIN, self.db)
        b = await admin_routes.create_model_group_endpoint(ModelGroupCreate(name="b"), ADMIN, self.db)

        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.update_model_group_endpoint(
                b.id, ModelGroupUpdate(name="renamed", provider_keys=["azure:east"]), ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("b", await self.groups())

        updated = await admin_routes.update_model_group_endpoint(
            b.id, ModelGroupUpdate(name="renamed", model_ids=["azure:east/gpt-5"]), ADMIN, self.db)
        self.assertEqual((updated.name, updated.model_ids), ("renamed", ["azure:east/gpt-5"]))
        self.assertEqual(self.tracker.group_for("azure:east/gpt-5"), b.id, "takes effect at once")
        # Leaving the lists out keeps the members.
        updated = await admin_routes.update_model_group_endpoint(
            b.id, ModelGroupUpdate(description="d"), ADMIN, self.db)
        self.assertEqual(updated.model_ids, ["azure:east/gpt-5"])

    async def test_a_membership_edit_takes_effect_at_once(self):
        await self.provider("azure:east", "gpt-5")
        created = await admin_routes.create_model_group_endpoint(ModelGroupCreate(name="g"), ADMIN, self.db)
        self.assertEqual(self.tracker.group_for("azure:east/gpt-5"), DEFAULT_GROUP_ID)
        await admin_routes.set_model_group_members_endpoint(
            created.id, ModelGroupMembersUpdate(provider_keys=["azure:east"]), ADMIN, self.db)
        self.assertEqual(self.tracker.group_for("azure:east/gpt-5"), created.id)

    async def test_default_limits_are_the_profile_limits(self):
        team = await self.make_profile("team")
        await admin_routes.update_profile_model_group_limit_endpoint(
            admin_routes.ProfileLimitsUpdate(rpm_limit=3), team.id, DEFAULT_GROUP_ID, ADMIN, self.db)
        limits = await admin_routes.get_profile_limits_endpoint(team.id, ADMIN, self.db)
        self.assertEqual((limits.groups[0].name, limits.groups[0].rpm_limit), ("Default", 3))
        self.assertEqual(limits.profile.rpm_limit, 3)


class ToggleTests(_GroupTestCase):

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self._cache = patch("app.auth.database._update_cache_after_database_change", AsyncMock())
        self._cache.start()

    async def asyncTearDown(self):
        self._cache.stop()
        await super().asyncTearDown()

    async def flags(self):
        self.db.expire_all()
        provider = (await self.db.execute(select(ProviderCredentials))).scalar_one()
        models = {m.model_name: m.is_enabled for m in (await self.db.execute(select(ModelConfiguration))).scalars()}
        return provider.enabled, models

    async def test_the_provider_switch_leaves_model_flags_alone(self):
        await self.provider("azure:east", "a", "b")
        await database.toggle_model_configuration(self.db, "azure:east/b", False)
        await database.toggle_provider_configuration(self.db, "azure:east", False)
        self.assertEqual(await self.flags(), (False, {"a": True, "b": False}))
        await database.toggle_provider_configuration(self.db, "azure:east", True)
        self.assertEqual(await self.flags(), (True, {"a": True, "b": False}),
                         "re-enabling the provider restores each model's own setting")

    async def test_enabling_a_model_does_not_enable_its_provider(self):
        await self.provider("azure:east", "a", enabled=False)
        await database.toggle_model_configuration(self.db, "azure:east/a", True)
        self.assertEqual((await self.flags())[0], False)

    async def test_bulk_toggle_only_touches_models(self):
        await self.provider("azure:east", "a", enabled=False)
        await database.bulk_toggle_all_models(self.db, True)
        self.assertEqual(await self.flags(), (False, {"a": True}))


class GroupSwitchTests(_GroupTestCase):
    """A switched-off group blocks its models like a disabled model would."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.provider("azure:east", "gpt-5", "gpt-4o")
        await self.provider("openai:main", "o3", "o4")
        await admin_routes.create_model_group_endpoint(ModelGroupCreate(
            name="premium", provider_keys=["azure:east"], model_ids=["openai:main/o3"]), ADMIN, self.db)
        # Carved out of premium's instance into an enabled group.
        await admin_routes.create_model_group_endpoint(ModelGroupCreate(
            name="cheap", model_ids=["azure:east/gpt-4o"]), ADMIN, self.db)
        self.premium = (await self.groups())["premium"].id
        self.cache = ModelCache()

    async def switch(self, group_id, enabled):
        return await admin_routes.toggle_model_group_endpoint(
            ToggleRequest(enabled=enabled), group_id, ADMIN, self.db)

    def served(self):
        return {m for m in ("azure:east/gpt-5", "azure:east/gpt-4o", "openai:main/o3", "openai:main/o4")
                if self.cache._passes_global_gate(m)}

    async def test_groups_start_enabled_and_the_switch_round_trips(self):
        groups = await self.groups()
        self.assertTrue(all(g.enabled for g in groups.values()))
        self.assertFalse((await self.switch(self.premium, False)).enabled)
        self.assertFalse((await self.groups())["premium"].enabled)
        self.assertTrue((await self.switch(self.premium, True)).enabled)

    async def test_the_default_group_cannot_be_switched_and_unknown_groups_are_404(self):
        for gid, code in ((DEFAULT_GROUP_ID, 400), (9999, 404)):
            with self.subTest(gid=gid):
                with self.assertRaises(HTTPException) as ctx:
                    await self.switch(gid, False)
                self.assertEqual(ctx.exception.status_code, code)

    async def test_a_disabled_group_blocks_its_models_until_enabled(self):
        all_models = {"azure:east/gpt-5", "azure:east/gpt-4o", "openai:main/o3", "openai:main/o4"}
        self.assertEqual(self.served(), all_models)

        await self.switch(self.premium, False)
        self.assertEqual(self.served(), {"azure:east/gpt-4o", "openai:main/o4"},
                         "instance and model members are blocked; a carve-out into an enabled group is not")
        self.assertEqual(self.tracker.blocking_group("azure:east/gpt-5"), "premium")
        self.assertIsNone(self.tracker.blocking_group("openai:main/o4"))

        await self.switch(self.premium, True)
        self.assertEqual(self.served(), all_models)

    async def test_identity_less_listings_hide_a_disabled_groups_models(self):
        """Regression: get_enabled_models (admin /v1/models) checked only the model and
        provider flags, so it listed models a bare name could no longer reach."""
        from app.openai_models import ModelInfo
        ids = ["azure:east/gpt-5", "azure:east/gpt-4o", "openai:main/o3", "openai:main/o4"]
        self.cache.update_models([ModelInfo(id=m, created=0, owned_by="t", provider="t") for m in ids])
        await self.switch(self.premium, False)
        self.assertEqual({m.id for m in self.cache.get_enabled_models()},
                         {"azure:east/gpt-4o", "openai:main/o4"})

    async def test_profiles_that_override_the_global_config_still_get_blocked_models(self):
        await self.switch(self.premium, False)
        rules = lambda mode, exc=None: SimpleNamespace(access_mode=mode, access_exceptions=exc or {})
        self.assertFalse(self.cache.is_model_allowed_for_profile(rules("default"), "azure:east/gpt-5"))
        self.assertTrue(self.cache.is_model_allowed_for_profile(rules("allow"), "azure:east/gpt-5"))
        self.assertTrue(self.cache.is_model_allowed_for_profile(
            rules("custom", {"azure:east/gpt-5": True}), "azure:east/gpt-5"))

    async def test_membership_and_limits_survive_the_switch(self):
        team = await self.make_profile("team")
        await self.set_group_limit(team.id, SimpleNamespace(id=self.premium), rpm=7)
        await self.switch(self.premium, False)

        groups = await self.groups()
        self.assertEqual(groups["premium"].provider_keys, ["azure:east"])
        self.assertEqual(self.tracker.group_for("azure:east/gpt-5"), self.premium,
                         "the group still governs its models, so counts and pools stay put")
        limits = await admin_routes.get_profile_limits_endpoint(team.id, ADMIN, self.db)
        row = next(g for g in limits.groups if g.group_id == self.premium)
        self.assertEqual((row.enabled, row.rpm_limit), (False, 7))

    async def test_the_access_view_names_the_blocking_group(self):
        team = await self.make_profile("team")
        await self.switch(self.premium, False)
        view = await admin_routes._build_profile_model_access(self.db, team.id)
        blocked = {m.model_id: m.blocked_by_group for m in view.models}
        self.assertEqual(blocked["azure:east/gpt-5"], "premium")
        self.assertIsNone(blocked["azure:east/gpt-4o"])


class ProfileGroupAccessTests(_GroupTestCase):
    """Model Access can allow or deny a whole group for one profile."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.provider("azure:east", "gpt-5", "gpt-4o")
        await self.provider("openai:main", "o3", "o4")
        await admin_routes.create_model_group_endpoint(ModelGroupCreate(
            name="premium", provider_keys=["azure:east"], model_ids=["openai:main/o3"]), ADMIN, self.db)
        self.premium = (await self.groups())["premium"].id
        self.team = await self.make_profile("team")

    async def set_group(self, group_id, allowed):
        return await admin_routes.set_profile_group_access(
            admin_routes.UserModelExceptionUpdate(is_allowed=allowed), self.team.id, group_id, ADMIN, self.db)

    @staticmethod
    def allowed(view):
        return {m.model_id for m in view.models if m.effective_allowed}

    async def test_the_view_lists_groups_and_each_models_group(self):
        view = await admin_routes._build_profile_model_access(self.db, self.team.id)
        self.assertEqual([(g.group_id, g.name) for g in view.groups],
                         [(DEFAULT_GROUP_ID, "Default"), (self.premium, "premium")])
        self.assertEqual({m.model_id: m.group_id for m in view.models}, {
            "azure:east/gpt-5": self.premium, "azure:east/gpt-4o": self.premium,
            "openai:main/o3": self.premium, "openai:main/o4": DEFAULT_GROUP_ID})

    async def test_denying_a_group_denies_every_model_it_governs(self):
        view = await self.set_group(self.premium, False)
        self.assertEqual(view.mode, "custom")
        self.assertEqual(self.allowed(view), {"openai:main/o4"})
        view = await self.set_group(self.premium, True)
        self.assertEqual(len(self.allowed(view)), 4)

    async def test_from_deny_all_allowing_a_group_allows_only_it(self):
        await admin_routes.update_profile_model_policy(
            admin_routes.UserModelPolicyUpdate(mode="deny"), self.team.id, ADMIN, self.db)
        view = await self.set_group(DEFAULT_GROUP_ID, True)
        self.assertEqual((view.mode, self.allowed(view)), ("custom", {"openai:main/o4"}))

    async def test_unknown_groups_are_404(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.set_group(9999, True)
        self.assertEqual(ctx.exception.status_code, 404)


class ReferenceCleanupTests(_GroupTestCase):

    async def _references(self, key="azure:east", model="gpt-5"):
        await self.provider(key, model)
        team = await self.make_profile("team")
        created = await admin_routes.create_model_group_endpoint(ModelGroupCreate(
            name="g", provider_keys=[key], model_ids=[f"{key}/{model}"]), ADMIN, self.db)
        self.db.add_all([
            ProfileModelAccessException(profile_id=team.id, model_id=f"{key}/{model}", is_allowed=True),
            ModelAlias(alias="x", target_model_id=f"{key}/{model}", profile_id=team.id, enabled=True),
        ])
        await self.db.commit()
        return created

    async def snapshot(self):
        self.db.expire_all()
        q = lambda col: self.db.execute(select(col))
        return (
            sorted((await q(ModelGroupMember.model_id)).scalars().all()),
            sorted((await q(ModelGroupInstanceMember.provider_key)).scalars().all()),
            sorted((await q(ProfileModelAccessException.model_id)).scalars().all()),
            [(a.target_model_id, a.enabled) for a in (await q(ModelAlias)).scalars().all()],
        )

    async def test_a_rename_carries_every_reference_along(self):
        await self._references(key="az_1:east")
        await database._rename_provider_references(self.db, "az_1:east", "az_1:west")
        await self.db.commit()
        self.assertEqual(await self.snapshot(), (
            ["az_1:west/gpt-5"], ["az_1:west"], ["az_1:west/gpt-5"], [("az_1:west/gpt-5", True)],
        ))

    async def test_deleting_a_provider_drops_references_and_switches_aliases_off(self):
        await self._references()
        self.assertTrue(await database.delete_provider_credentials(self.db, "azure:east"))
        self.assertEqual(await self.snapshot(), ([], [], [], [("azure:east/gpt-5", False)]))

    async def test_removing_stale_models_drops_their_references(self):
        await self._references()
        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            await database.delete_stale_models(self.db, ["azure:east/gpt-5"])
        self.assertEqual(await self.snapshot(), ([], ["azure:east"], [], [("azure:east/gpt-5", False)]),
                         "the instance member stays: only the model went away")

    async def test_a_model_sync_leaves_references_alone(self):
        """A sync deletes and re-creates the same ids; membership must survive it."""
        await self._references()
        before = await self.snapshot()
        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            await database.refresh_models_from_providers(self.db, [
                {"model_id": "azure:east/gpt-5", "provider_key": "azure:east", "model_name": "gpt-5"},
            ])
        self.assertEqual(await self.snapshot(), before)


# ---------------------------------------------------------------------------
# Migration of a database that still has instance groups
# ---------------------------------------------------------------------------

def _run_migrations(path, monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "_backup_sqlite_file", lambda tag: path + ".bak")

    async def run():
        for _ in range(2):   # idempotent
            await database.create_tables_async()
            await database._run_auto_migrations()
        await engine.dispose()

    asyncio.run(run())


def test_existing_model_groups_gain_an_enabled_switch_that_starts_on(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        monkeypatch.setattr(database, "engine", engine)
        asyncio.run(database.create_tables_async())
        asyncio.run(engine.dispose())

        # A database from before the switch existed.
        conn = sqlite3.connect(path)
        conn.execute("ALTER TABLE model_groups DROP COLUMN enabled")
        conn.execute("INSERT INTO model_groups (id, name) VALUES (1, 'edge')")
        conn.commit()
        conn.close()

        _run_migrations(path, monkeypatch)

        conn = sqlite3.connect(path)
        assert conn.execute("SELECT name, enabled FROM model_groups").fetchall() == [("edge", 1)]
        conn.close()
    finally:
        os.unlink(path)


def test_instance_groups_fold_into_model_groups(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        # A current schema plus the old instance-group tables, as a live install has.
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        monkeypatch.setattr(database, "engine", engine)
        asyncio.run(database.create_tables_async())
        asyncio.run(engine.dispose())

        conn = sqlite3.connect(path)
        for stmt in (
            "CREATE TABLE instance_groups (id INTEGER PRIMARY KEY, name VARCHAR(64) NOT NULL UNIQUE, "
            "description VARCHAR(256), rpm_default INTEGER, rpd_default INTEGER, created_at DATETIME, "
            "updated_at DATETIME, updated_by VARCHAR(50))",
            "CREATE TABLE instance_group_members (id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL "
            "REFERENCES instance_groups(id) ON DELETE CASCADE, provider_key VARCHAR(100) NOT NULL UNIQUE, "
            "created_at DATETIME)",
            "CREATE TABLE profile_instance_group_limits (id INTEGER PRIMARY KEY, profile_id INTEGER NOT NULL, "
            "group_id INTEGER NOT NULL REFERENCES instance_groups(id) ON DELETE CASCADE, rpm_limit INTEGER, "
            "rpd_limit INTEGER, updated_at DATETIME, updated_by VARCHAR(50))",
            "CREATE TABLE user_instance_group_rate_limits (id INTEGER PRIMARY KEY, user_id INTEGER, "
            "group_id INTEGER REFERENCES instance_groups(id) ON DELETE CASCADE, rpm_limit INTEGER, "
            "rpd_limit INTEGER, updated_at DATETIME, updated_by VARCHAR(50))",
            "INSERT OR IGNORE INTO user_profiles (id, name, model_access_mode) VALUES (1, 'Default', 'default')",
            "INSERT INTO model_groups (id, name) VALUES (1, 'edge'), (2, 'DEFAULT')",
            "INSERT INTO model_group_members (group_id, model_id) VALUES (1, 'azure:east/gpt-5')",
            "INSERT INTO instance_groups (id, name) VALUES (7, 'Edge'), (8, 'fast')",
            "INSERT INTO instance_group_members (group_id, provider_key) VALUES (7, 'azure:east'), (8, 'openai:main')",
            "INSERT INTO profile_instance_group_limits (profile_id, group_id, rpm_limit, rpd_limit) VALUES (1, 8, 5, 50)",
            "INSERT INTO user_rpd_carries (user_id, usage_date, scope_kind, scope_id, carry) "
            "VALUES (1, '2026-01-01', 'instance_group', 8, 3)",
            # Old toggle: provider off forced all its models off ...
            "INSERT INTO provider_credentials (provider_key, provider_type, instance_name, provider_name, enabled) "
            "VALUES ('off:all', 'off', 'all', 'off', 0), ('off:mixed', 'off', 'mixed', 'off', 0)",
            "INSERT INTO model_configurations (model_id, provider_key, model_name, is_enabled) VALUES "
            "('off:all/a', 'off:all', 'a', 0), ('off:all/b', 'off:all', 'b', 0), "
            # ... while a provider with mixed flags was edited model by model.
            "('off:mixed/a', 'off:mixed', 'a', 1), ('off:mixed/b', 'off:mixed', 'b', 0)",
        ):
            conn.execute(stmt)
        conn.commit()
        conn.close()

        _run_migrations(path, monkeypatch)
        assert database.GROUPS_UNIFIED_THIS_START

        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not tables & {"instance_groups", "instance_group_members",
                             "profile_instance_group_limits", "user_instance_group_rate_limits"}
        names = dict(conn.execute("SELECT name, id FROM model_groups"))
        assert set(names) == {"edge", "DEFAULT (2)", "Edge (instance)", "fast"}, names
        assert dict(conn.execute("SELECT provider_key, group_id FROM model_group_instance_members")) == {
            "azure:east": names["Edge (instance)"], "openai:main": names["fast"]}
        assert conn.execute("SELECT profile_id, group_id, rpm_limit, rpd_limit FROM profile_model_group_limits"
                            ).fetchall() == [(1, names["fast"], 5, 50)]
        assert conn.execute("SELECT scope_kind, scope_id FROM user_rpd_carries").fetchall() == [
            ("model_group", names["fast"])]
        assert dict(conn.execute("SELECT model_id, is_enabled FROM model_configurations")) == {
            "off:all/a": 1, "off:all/b": 1, "off:mixed/a": 1, "off:mixed/b": 0}
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO model_groups (name) VALUES ('FAST')")
        conn.close()
    finally:
        os.unlink(path)


def _database_with_instance_groups(path, monkeypatch, *stmts):
    """A current schema plus the bare old instance-group tables, then `stmts`."""
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    monkeypatch.setattr(database, "engine", engine)
    asyncio.run(database.create_tables_async())
    asyncio.run(engine.dispose())

    conn = sqlite3.connect(path)
    for stmt in (
        "CREATE TABLE instance_groups (id INTEGER PRIMARY KEY, name VARCHAR(64) NOT NULL UNIQUE, "
        "description VARCHAR(256), rpm_default INTEGER, rpd_default INTEGER, created_at DATETIME, "
        "updated_at DATETIME, updated_by VARCHAR(50))",
        "CREATE TABLE instance_group_members (id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL "
        "REFERENCES instance_groups(id) ON DELETE CASCADE, provider_key VARCHAR(100) NOT NULL UNIQUE, "
        "created_at DATETIME)",
        *stmts,
    ):
        conn.execute(stmt)
    conn.commit()
    conn.close()


def test_instance_group_defaults_from_before_profiles_carry_over(monkeypatch):
    """Regression: with no profile_instance_group_limits table to seed, the folded
    groups came out unlimited."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        _database_with_instance_groups(
            path, monkeypatch,
            "INSERT INTO instance_groups (id, name, rpm_default, rpd_default) "
            "VALUES (7, 'edge', 10, 100), (8, 'open', NULL, NULL)",
            "INSERT INTO instance_group_members (group_id, provider_key) VALUES (7, 'azure:east'), (8, 'openai:main')",
        )
        _run_migrations(path, monkeypatch)

        conn = sqlite3.connect(path)
        names = dict(conn.execute("SELECT name, id FROM model_groups"))
        assert conn.execute("SELECT profile_id, group_id, rpm_limit, rpd_limit FROM profile_model_group_limits"
                            ).fetchall() == [(database.DEFAULT_PROFILE_ID, names["edge"], 10, 100)]
        conn.close()
    finally:
        os.unlink(path)


def test_rows_left_by_a_deleted_group_do_not_block_the_fold(monkeypatch):
    """model_groups ids are reused, and ledger and carry rows have no foreign key: rows
    a deleted group left under the reused id must not collide with the ones moved in."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        _database_with_instance_groups(
            path, monkeypatch,
            "INSERT INTO instance_groups (id, name) VALUES (7, 'edge')",
            "INSERT INTO instance_group_members (group_id, provider_key) VALUES (7, 'azure:east')",
            # model_groups is empty, so 'edge' becomes model group 1 -- the id a deleted
            # group used, whose carry is still there.
            "INSERT INTO user_rpd_carries (user_id, usage_date, scope_kind, scope_id, carry) VALUES "
            "(1, '2026-01-01', 'model_group', 1, 9), (1, '2026-01-01', 'instance_group', 7, 3)",
        )
        _run_migrations(path, monkeypatch)

        conn = sqlite3.connect(path)
        assert conn.execute("SELECT id FROM model_groups WHERE name = 'edge'").fetchone() == (1,)
        assert conn.execute("SELECT scope_kind, scope_id, carry FROM user_rpd_carries").fetchall() == [
            ("model_group", 1, 3)]
        conn.close()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()

"""Model alias persistence: priority order, match types, reorder and migration.

The resolver evaluates mappings in stored priority order, so the data layer has
to keep that order stable across creates, edits and renames, and a reorder from a
stale client list must change nothing.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.auth import database
from app.auth.database import (
    get_all_model_aliases,
    reorder_model_aliases,
    upsert_model_alias,
)
from app.auth.models import (
    Base,
    ModelAliasReorder,
    ModelAliasUpsert,
    ModelConfiguration,
    ProviderCredentials,
)
from app.model_alias import model_alias_resolver

ALL = ["openai", "anthropic", "azure_openai"]


class _TempDb(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)

    async def asyncTearDown(self):
        await self._engine.dispose()
        os.unlink(self._db_path)


class ModelAliasDataAccessTests(_TempDb):

    async def _upsert(self, alias, target="p/t", **kw):
        async with self._factory() as db:
            return await upsert_model_alias(db, alias, target, True, ALL, **kw)

    async def _rows(self):
        async with self._factory() as db:
            return [(r.alias, r.match_type, r.priority) for r in await get_all_model_aliases(db)]

    async def test_new_rows_append_at_max_plus_one(self):
        await self._upsert("zeta")
        await self._upsert("alpha", match_type="contains")
        await self._upsert("mid", match_type="regex")
        self.assertEqual(await self._rows(), [
            ("zeta", "exact", 0), ("alpha", "contains", 1), ("mid", "regex", 2),
        ])

    async def test_update_keeps_priority_and_none_keeps_match_type(self):
        await self._upsert("a", match_type="contains")
        await self._upsert("b")
        await self._upsert("a", target="p/other")  # by name, match_type omitted
        self.assertEqual(await self._rows(), [("a", "contains", 0), ("b", "exact", 1)])

    async def test_rename_by_id_keeps_position(self):
        first = await self._upsert("first")
        await self._upsert("second")
        renamed = await self._upsert("renamed", alias_id=first.id, match_type="contains")
        self.assertEqual(renamed.id, first.id)
        self.assertEqual(await self._rows(), [("renamed", "contains", 0), ("second", "exact", 1)])

    async def test_update_by_missing_id_returns_none(self):
        self.assertIsNone(await self._upsert("x", alias_id=999))
        self.assertEqual(await self._rows(), [])

    async def test_reorder_with_full_set(self):
        a, b, c = [await self._upsert(n) for n in ("a", "b", "c")]
        async with self._factory() as db:
            self.assertTrue(await reorder_model_aliases(db, [c.id, a.id, b.id]))
        self.assertEqual([r[0] for r in await self._rows()], ["c", "a", "b"])

    async def test_reorder_rejects_stale_lists(self):
        a, b, c = [await self._upsert(n) for n in ("a", "b", "c")]
        for ids in ([a.id, b.id], [a.id, b.id, c.id, 999], [a.id, b.id, 999], [a.id, a.id, b.id]):
            async with self._factory() as db:
                self.assertFalse(await reorder_model_aliases(db, ids), ids)
        self.assertEqual([r[0] for r in await self._rows()], ["a", "b", "c"])

    async def test_order_ties_break_on_id(self):
        a = await self._upsert("a")
        b = await self._upsert("b")
        async with self._factory() as db:
            await db.execute(text("UPDATE model_aliases SET priority = 5"))
            await db.commit()
        async with self._factory() as db:
            self.assertEqual([r.id for r in await get_all_model_aliases(db)], [a.id, b.id])


class ModelAliasSchemaValidationTests(unittest.TestCase):

    def test_invalid_regex_rejected(self):
        with self.assertRaises(ValidationError):
            ModelAliasUpsert(alias="(", target_model_id="p/t", match_type="regex")

    def test_nested_quantifier_regex_rejected(self):
        for pattern in ("(a+)+", r"(\w*)*$", "(x+)*", "(ab+){2,}",
                        "((a)+)+$", "(a|aa)+$", "(a{1,})+$", "((ab)*c)*$", "(a?)+"):
            with self.assertRaises(ValidationError, msg=pattern):
                ModelAliasUpsert(alias=pattern, target_model_id="p/t", match_type="regex")

    def test_open_ended_regex_rejected(self):
        # Polynomial blow-ups: several wide quantifiers, or a long run of optionals.
        for pattern in (".*.*.*!", "a*a*a*!", "^a*a*a*a*!", ".*a.*a.*!", "^" + "a?" * 22 + "a" * 22):
            with self.assertRaises(ValidationError, msg=pattern):
                ModelAliasUpsert(alias=pattern, target_model_id="p/t", match_type="regex")

    def test_ordinary_regex_accepted(self):
        for pattern in (r"^claude-(opus|sonnet)-4(\.\d+)?$", ".*opus.*", "gpt-4o.*mini",
                        r"^gpt-\d+\.\d+-\w+$", "claude-(opus|sonnet|haiku)", ".*"):
            body = ModelAliasUpsert(alias=pattern, target_model_id="p/t", match_type="regex")
            self.assertEqual(body.match_type, "regex")

    def test_exact_alias_equal_to_target_rejected_but_contains_allowed(self):
        with self.assertRaises(ValidationError):
            ModelAliasUpsert(alias="p/t", target_model_id="p/t", match_type="exact")
        ModelAliasUpsert(alias="p/t", target_model_id="p/t", match_type="contains")

    def test_unknown_match_type_rejected(self):
        with self.assertRaises(ValidationError):
            ModelAliasUpsert(alias="x", target_model_id="p/t", match_type="glob")

    def test_reorder_body_requires_unique_non_empty_ids(self):
        with self.assertRaises(ValidationError):
            ModelAliasReorder(ids=[])
        with self.assertRaises(ValidationError):
            ModelAliasReorder(ids=[1, 1])
        self.assertEqual(ModelAliasReorder(ids=[2, 1]).ids, [2, 1])


class ModelAliasEndpointTests(_TempDb):
    """The upsert endpoint re-validates against the row's effective match type."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        async with self._factory() as db:
            db.add(ProviderCredentials(
                provider_key="p", provider_type="ollama", instance_name="p", provider_name="ollama",
            ))
            db.add(ModelConfiguration(model_id="p/t", provider_key="p", model_name="t"))
            await db.commit()
        self._saved_snapshot = model_alias_resolver._snapshot
        self._patch = patch("app.model_alias.AsyncSessionLocal", self._factory)
        self._patch.start()

    async def asyncTearDown(self):
        self._patch.stop()
        model_alias_resolver._snapshot = self._saved_snapshot
        await super().asyncTearDown()

    async def _post(self, **body):
        from app.routes.admin import upsert_model_alias_endpoint
        async with self._factory() as db:
            return await upsert_model_alias_endpoint(ModelAliasUpsert(**body), current_admin=None, db=db)

    async def test_omitted_match_type_revalidates_existing_regex_row(self):
        row = await self._post(alias="^claude", target_model_id="p/t", match_type="regex")
        with self.assertRaises(HTTPException) as ctx:
            await self._post(id=row.id, alias="(", target_model_id="p/t")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_omitted_match_type_on_new_row_means_exact(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._post(alias="p/t", target_model_id="p/t")
        self.assertEqual(ctx.exception.status_code, 422)

    async def test_rename_onto_existing_alias_conflicts(self):
        await self._post(alias="one", target_model_id="p/t")
        two = await self._post(alias="two", target_model_id="p/t")
        with self.assertRaises(HTTPException) as ctx:
            await self._post(id=two.id, alias="one", target_model_id="p/t")
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_save_reloads_resolver_in_priority_order(self):
        await self._post(alias="opus", target_model_id="p/t", match_type="contains")
        self.assertEqual(model_alias_resolver.resolve("Claude-Opus-4", "openai"), "p/t")


class ModelAliasMigrationTests(unittest.IsolatedAsyncioTestCase):
    """A pre-pattern model_aliases table gains the columns, ordered alphabetically."""

    LEGACY = """
    CREATE TABLE model_aliases (
        id INTEGER NOT NULL PRIMARY KEY,
        alias VARCHAR(200) NOT NULL UNIQUE,
        target_model_id VARCHAR(200) NOT NULL,
        enabled BOOLEAN NOT NULL,
        apis TEXT,
        created_at DATETIME,
        updated_at DATETIME
    )"""

    async def asyncSetUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with sqlite3.connect(self.path) as raw:
            raw.execute(self.LEGACY)
            raw.executemany(
                "INSERT INTO model_aliases (id, alias, target_model_id, enabled) VALUES (?, ?, ?, 1)",
                [(1, "zeta", "p/z"), (2, "alpha", "p/a"), (3, "mid", "p/m")],
            )
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

    async def _rows(self):
        async with self.engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT alias, match_type, priority FROM model_aliases ORDER BY priority, id"
            ))
            return [tuple(r) for r in result]

    async def test_adds_columns_and_backfills_alphabetical_priority(self):
        await database._run_auto_migrations()
        self.assertEqual(await self._rows(), [
            ("alpha", "exact", 0), ("mid", "exact", 1), ("zeta", "exact", 2),
        ])

    async def test_second_run_changes_nothing(self):
        await database._run_auto_migrations()
        async with self.engine.begin() as conn:
            await conn.execute(text("UPDATE model_aliases SET priority = 9 WHERE alias = 'alpha'"))
        await database._run_auto_migrations()
        self.assertEqual((await self._rows())[-1], ("alpha", "exact", 9))

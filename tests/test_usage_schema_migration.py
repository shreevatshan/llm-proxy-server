"""Tests for the startup migration that rekeys the usage tables onto (user_id, pool_id).

Pre-rekey deployments hold usage rows keyed by the username string and carrying no pool.
The rebuild has to map every identity to a user id, merge rows that collapse onto one
key, stamp pool_id from the membership intervals, drop what cannot be attributed, and
do all of that once -- a second start must find nothing to do.

The test builds the OLD schema with raw DDL (the create_all path would already produce
the new one), so it exercises the real rebuild rather than a fresh table.
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.auth import database
from app.auth.models import ADMIN_USAGE_USER_ID, Base

TODAY = date(2026, 4, 7)

OLD_DAILY = """
CREATE TABLE request_usage (
    id INTEGER NOT NULL PRIMARY KEY,
    date DATE NOT NULL,
    user_identity VARCHAR(200) NOT NULL,
    user_type VARCHAR(20) NOT NULL,
    model VARCHAR(200) NOT NULL,
    server VARCHAR(20) NOT NULL,
    request_count INTEGER NOT NULL,
    CONSTRAINT uq_usage_day UNIQUE (date, user_identity, model, server)
)"""
OLD_DAILY_INDEXES = (
    "CREATE INDEX ix_request_usage_date ON request_usage (date)",
    "CREATE INDEX ix_request_usage_user_identity ON request_usage (user_identity)",
    "CREATE INDEX ix_request_usage_model ON request_usage (model)",
    "CREATE INDEX ix_request_usage_id ON request_usage (id)",
)
OLD_HOURLY = """
CREATE TABLE request_usage_hourly (
    id INTEGER NOT NULL PRIMARY KEY,
    date DATE NOT NULL,
    hour INTEGER NOT NULL,
    user_identity VARCHAR(200) NOT NULL,
    user_type VARCHAR(20) NOT NULL,
    model VARCHAR(200) NOT NULL,
    server VARCHAR(20) NOT NULL,
    request_count INTEGER NOT NULL,
    CONSTRAINT uq_usage_hour UNIQUE (date, hour, user_identity, model, server)
)"""
OLD_MONTHLY = """
CREATE TABLE request_usage_monthly (
    id INTEGER NOT NULL PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    user_identity VARCHAR(200) NOT NULL,
    user_type VARCHAR(20) NOT NULL,
    model VARCHAR(200) NOT NULL,
    server VARCHAR(20) NOT NULL,
    request_count INTEGER NOT NULL,
    CONSTRAINT uq_usage_month UNIQUE (year, month, user_identity, model, server)
)"""


class UsageSchemaMigrationTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # Old-schema usage tables first, then everything else from the models. create_all
        # skips tables that already exist, so the three usage tables keep the old shape.
        with sqlite3.connect(self.path) as raw:
            raw.execute(OLD_DAILY)
            for ddl in OLD_DAILY_INDEXES:
                raw.execute(ddl)
            raw.execute(OLD_HOURLY)
            raw.execute(OLD_MONTHLY)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.path}")
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # _run_auto_migrations works on the module-level engine and DATABASE_URL.
        self._patches = [
            patch.object(database, "engine", self.engine),
            patch.object(database, "DATABASE_URL", f"sqlite+aiosqlite:///{self.path}"),
            patch("app.auth.admin.is_admin_enabled", return_value=True),
            patch("app.auth.admin.get_admin_username", return_value="root"),
            patch("app.time_utils.local_today", return_value=TODAY),
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

    # -- seeding the pre-rekey world ----------------------------------------

    async def _exec(self, sql, params=None):
        async with self.engine.begin() as conn:
            return await conn.execute(text(sql), params or {})

    async def _fetch(self, sql, params=None):
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params or {})).fetchall()

    async def _old_row(self, table, identity, count, *, model="p/m", server="openai",
                       day=TODAY, hour=9, user_type="user"):
        if table == "request_usage":
            await self._exec(
                "INSERT INTO request_usage (date, user_identity, user_type, model, server, "
                "request_count) VALUES (:d, :i, :t, :m, :s, :c)",
                {"d": day.isoformat(), "i": identity, "t": user_type, "m": model, "s": server, "c": count},
            )
        elif table == "request_usage_hourly":
            await self._exec(
                "INSERT INTO request_usage_hourly (date, hour, user_identity, user_type, model, "
                "server, request_count) VALUES (:d, :h, :i, :t, :m, :s, :c)",
                {"d": day.isoformat(), "h": hour, "i": identity, "t": user_type, "m": model, "s": server, "c": count},
            )
        else:
            await self._exec(
                "INSERT INTO request_usage_monthly (year, month, user_identity, user_type, model, "
                "server, request_count) VALUES (:y, :mo, :i, :t, :m, :s, :c)",
                {"y": day.year, "mo": day.month, "i": identity, "t": user_type, "m": model, "s": server, "c": count},
            )

    async def _user(self, username):
        await self._exec(
            "INSERT INTO users (username, email, hashed_password, is_active) "
            "VALUES (:u, :e, 'x', 1)", {"u": username, "e": f"{username}@example.test"},
        )
        return (await self._fetch("SELECT id FROM users WHERE username = :u", {"u": username}))[0][0]

    async def _api_key(self, key_id, user_id):
        await self._exec(
            "INSERT INTO api_keys (id, user_id, api_key, name, is_active) "
            "VALUES (:k, :u, :h, 'k', 1)", {"k": key_id, "u": user_id, "h": f"hash{key_id}"},
        )

    async def _pool(self, name, owner_id):
        await self._exec(
            "INSERT INTO request_pools (name, owner_user_id) VALUES (:n, :o)",
            {"n": name, "o": owner_id},
        )
        return (await self._fetch("SELECT id FROM request_pools WHERE name = :n", {"n": name}))[0][0]

    async def _interval(self, pool_id, user_id, joined_on, left_on=None):
        await self._exec(
            "INSERT INTO pool_membership_intervals (pool_id, user_id, joined_on, left_on) "
            "VALUES (:p, :u, :j, :l)",
            {"p": pool_id, "u": user_id, "j": joined_on.isoformat(),
             "l": left_on.isoformat() if left_on else None},
        )

    async def _columns(self, table):
        return [r[1] for r in await self._fetch(f"PRAGMA table_info({table})")]

    async def _rows(self, table, **where):
        clause = " AND ".join(f"{k} = :{k}" for k in where) or "1=1"
        return await self._fetch(
            f"SELECT user_id, pool_id, user_identity, user_type, model, request_count "
            f"FROM {table} WHERE {clause} ORDER BY id", where,
        )

    # -- tests --------------------------------------------------------------

    async def test_old_schema_is_detected_and_rebuilt_with_a_backup(self):
        alice = await self._user("alice")
        await self._old_row("request_usage", "alice", 7)

        await database._run_auto_migrations()

        for table in ("request_usage", "request_usage_hourly", "request_usage_monthly"):
            with self.subTest(table=table):
                cols = await self._columns(table)
                self.assertIn("user_id", cols)
                self.assertIn("pool_id", cols)
        self.assertEqual(await self._rows("request_usage"), [(alice, 0, "alice", "user", "p/m", 7)])

        # The old index names are gone (they would have collided with the new ones).
        names = {r[0] for r in await self._fetch(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'request_usage'"
        )}
        self.assertNotIn("ix_request_usage_user_identity", names)
        self.assertNotIn("ix_request_usage_id", names)
        self.assertIn("ix_usage_user_date", names)

        backups = [n for n in os.listdir(os.path.dirname(self.path))
                   if n.startswith(os.path.basename(self.path) + ".pre-usage-rekey-")]
        self.assertEqual(len(backups), 1, "exactly one backup, taken before the rebuild")
        with sqlite3.connect(os.path.join(os.path.dirname(self.path), backups[0])) as bak:
            self.assertEqual(
                bak.execute("SELECT user_identity, request_count FROM request_usage").fetchall(),
                [("alice", 7)],
            )
            self.assertNotIn("user_id", [r[1] for r in bak.execute("PRAGMA table_info(request_usage)")])

    async def test_identities_map_to_users_keys_to_owners_and_admin_to_zero(self):
        alice = await self._user("alice")
        await self._api_key(7, alice)
        await self._old_row("request_usage", "alice", 4)
        await self._old_row("request_usage", "key:7", 5, user_type="api_key")      # merges into alice
        await self._old_row("request_usage", "key:7", 2, model="p/azure", user_type="api_key")
        await self._old_row("request_usage", "root", 500, user_type="admin")

        await database._run_auto_migrations()

        by_key = {(r[0], r[4]): (r[1], r[2], r[5]) for r in await self._rows("request_usage")}
        self.assertEqual(by_key[(alice, "p/m")], (0, "alice", 9), "key:7 folded onto its owner")
        self.assertEqual(by_key[(alice, "p/azure")], (0, "alice", 2), "relabelled with the owner's name")
        self.assertEqual(by_key[(ADMIN_USAGE_USER_ID, "p/m")], (0, "root", 500))
        self.assertEqual(len(by_key), 3)

    async def test_admin_rows_survive_even_if_the_admin_is_disabled_or_renamed(self):
        """user_type 'admin' is only ever written for the config admin, so it is the
        attribution to trust -- not whatever the environment says at migration time."""
        await self._old_row("request_usage", "old-admin", 500, user_type="admin")
        await self._old_row("request_usage_monthly", "old-admin", 90, user_type="admin")

        with patch("app.auth.admin.is_admin_enabled", return_value=False):
            await database._run_auto_migrations()

        self.assertEqual(await self._rows("request_usage"),
                         [(ADMIN_USAGE_USER_ID, 0, "old-admin", "admin", "p/m", 500)])
        self.assertEqual(await self._rows("request_usage_monthly"),
                         [(ADMIN_USAGE_USER_ID, 0, "old-admin", "admin", "p/m", 90)])

    async def test_a_real_account_holding_the_admin_name_keeps_its_own_history(self):
        """is_reserved_username only rejects the admin's name while the admin is
        *enabled*, so an account registered during a disabled stint -- or under an
        earlier ADMIN_USERNAME -- owns it legitimately. The rekey must not hand that
        account's history to ADMIN_USAGE_USER_ID: the migration is one-shot.
        """
        root = await self._user("root")  # the patched ADMIN_USERNAME
        await self._old_row("request_usage", "root", 12)

        await database._run_auto_migrations()

        self.assertEqual(await self._rows("request_usage"),
                         [(root, 0, "root", "user", "p/m", 12)])
        self.assertNotEqual(root, ADMIN_USAGE_USER_ID)

    async def test_unattributable_rows_are_dropped_and_totals_otherwise_preserved(self):
        alice = await self._user("alice")
        for table in ("request_usage", "request_usage_hourly", "request_usage_monthly"):
            await self._old_row(table, "alice", 10)
            await self._old_row(table, "deleted.user", 3)
            await self._old_row(table, "key:99", 1, user_type="api_key")  # no such key

        with self.assertLogs("app.auth.database", level="INFO") as logs:
            await database._run_auto_migrations()

        for table in ("request_usage", "request_usage_hourly", "request_usage_monthly"):
            with self.subTest(table=table):
                rows = await self._rows(table)
                self.assertEqual([(r[0], r[5]) for r in rows], [(alice, 10)])
        self.assertTrue(any("'orphan_rows_dropped': 2" in line for line in logs.output))

    async def test_pool_id_is_back_filled_from_membership_intervals(self):
        alice = await self._user("alice")
        bob = await self._user("bob")
        team = await self._pool("team", alice)
        other = await self._pool("other", bob)
        d = lambda n: date(2026, 4, n)
        # alice: in `team` from the 3rd to the 5th, then in `other` from the 5th (open).
        await self._interval(team, alice, d(3), d(5))
        await self._interval(other, alice, d(5))
        await self._old_row("request_usage", "alice", 1, day=d(2))   # before any pool
        await self._old_row("request_usage", "alice", 2, day=d(4))   # team
        await self._old_row("request_usage", "alice", 3, day=d(5))   # changeover: earlier stint wins
        await self._old_row("request_usage", "alice", 4, day=d(6))   # other
        await self._old_row("request_usage_hourly", "alice", 2, day=d(4))
        # Monthly rows are month-granular: April overlaps both stints; March neither.
        await self._old_row("request_usage_monthly", "alice", 30, day=d(1))
        await self._old_row("request_usage_monthly", "alice", 9, day=date(2026, 3, 1))
        await self._old_row("request_usage", "bob", 5, day=d(6))     # never pooled

        await database._run_auto_migrations()

        daily = {r[5]: r[1] for r in await self._rows("request_usage", user_id=alice)}
        self.assertEqual(daily, {1: 0, 2: team, 3: team, 4: other})
        self.assertEqual([r[1] for r in await self._rows("request_usage_hourly")], [team])
        monthly = {r[5]: r[1] for r in await self._rows("request_usage_monthly")}
        self.assertEqual(monthly, {30: team, 9: 0})
        self.assertEqual([r[1] for r in await self._rows("request_usage", user_id=bob)], [0])

    async def test_a_second_start_is_a_noop(self):
        alice = await self._user("alice")
        await self._old_row("request_usage", "alice", 7)

        await database._run_auto_migrations()
        first = await self._rows("request_usage")
        await database._run_auto_migrations()

        self.assertEqual(await self._rows("request_usage"), first)
        backups = [n for n in os.listdir(os.path.dirname(self.path))
                   if n.startswith(os.path.basename(self.path) + ".pre-usage-rekey-")]
        self.assertEqual(len(backups), 1, "no second backup: nothing needed migrating")
        self.assertEqual(first, [(alice, 0, "alice", "user", "p/m", 7)])

    async def test_the_bucketing_timezone_is_recorded_once(self):
        from app.config import config

        await database._run_auto_migrations()
        rows = await self._fetch("SELECT value FROM usage_meta WHERE key = 'timezone'")
        self.assertEqual(rows, [(config.server.timezone,)])

        # A changed TIMEZONE is reported, loudly, and the recorded value is kept.
        with patch.object(config.server, "timezone", "Pacific/Kiritimati"), \
             self.assertLogs("app.auth.database", level="ERROR") as logs:
            await database._run_auto_migrations()
        self.assertTrue(any("Pacific/Kiritimati" in line for line in logs.output))
        self.assertEqual(
            await self._fetch("SELECT value FROM usage_meta WHERE key = 'timezone'"), rows,
        )

    async def test_new_rows_upsert_onto_the_rebuilt_key(self):
        """The rebuilt table must accept the tracker's increment-on-conflict writes."""
        alice = await self._user("alice")
        await self._old_row("request_usage", "alice", 7)
        await database._run_auto_migrations()

        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker
        factory = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        row = {"date": TODAY, "user_id": alice, "pool_id": 0, "user_identity": "alice",
               "user_type": "api_key", "model": "p/m", "server": "openai", "request_count": 3}
        with patch.object(database, "AsyncSessionLocal", factory):
            await database.flush_usage_rows([dict(row, hour=9)], [row])

        self.assertEqual(await self._rows("request_usage"), [(alice, 0, "alice", "api_key", "p/m", 10)])


if __name__ == "__main__":
    unittest.main()

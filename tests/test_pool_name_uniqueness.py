"""Pool names are unique case-insensitively, and the database agrees with the routes.

The routes compare names with ilike, so "Alpha" and "alpha" are the same pool name as
far as a user is concerned. A pre-check is not a constraint, though: two concurrent
creates can both read "no clash" before either commits. These tests pin the schema-level
backstop that decides the race, and the handlers that turn losing it into a 409.
"""

import sqlite3

from fastapi import HTTPException
from sqlalchemy import select, text

from app.auth.models import PoolCreate, PoolUpdate, RequestPool
from app.routes import pools as pool_routes
from tests.pool_test_base import PoolTestCase


class PoolNameUniquenessTests(PoolTestCase):

    async def test_routes_reject_a_name_differing_only_by_case(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="Alpha"), alice, self.db)

        with self.assertRaises(HTTPException) as ctx:
            await pool_routes.create_pool(PoolCreate(name="alpha"), bob, self.db)
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_schema_rejects_case_duplicates_the_precheck_raced_past(self):
        """The create pre-check and the DB have to agree, or a race leaves two pools.

        Writing the row directly is what a create that lost the race does: its ilike
        check passed against a database that no longer looks that way by commit time.
        """
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="Alpha"), alice, self.db)

        with self.assertRaises(Exception) as ctx:
            await self.db.execute(text(
                "INSERT INTO request_pools (name, owner_user_id) VALUES ('alpha', :o)"
            ), {"o": alice.id})
            await self.db.commit()
        self.assertIn("uq_request_pools_name_lower", str(ctx.exception))
        await self.db.rollback()

        names = (await self.db.execute(select(RequestPool.name))).scalars().all()
        self.assertEqual(names, ["Alpha"], "the losing create must not leave a pool")

    async def test_rename_race_answers_409_not_500(self):
        alice = await self.make_user("alice", rpd_limit=5)
        bob = await self.make_user("bob", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="alices"), alice, self.db)
        await pool_routes.create_pool(PoolCreate(name="bobs"), bob, self.db)

        # Bob renames to a free name, then Alice takes it before he commits. The
        # steal is hung off the clash pre-check itself (the only ilike -> LIKE query
        # in this handler), because the window being tested opens the moment that
        # SELECT returns "free" and closes when his UPDATE lands.
        real_execute = self.db.execute
        state = {"stolen": False}

        async def steal_then_execute(stmt, *a, **kw):
            result = await real_execute(stmt, *a, **kw)
            if not state["stolen"] and "like" in str(stmt).lower():
                state["stolen"] = True
                self.assertIsNone(result.scalar_one_or_none(),
                                  "the pre-check must see the name as free")
                alices = (await real_execute(
                    select(RequestPool).where(RequestPool.name == "alices")
                )).scalar_one()
                alices.name = "shared"
                await self.db.flush()
                return await real_execute(stmt.where(RequestPool.id < 0))
            return result

        self.db.execute = steal_then_execute
        try:
            with self.assertRaises(HTTPException) as ctx:
                await pool_routes.update_my_pool(PoolUpdate(name="shared"), bob, self.db)
        finally:
            self.db.execute = real_execute

        self.assertEqual(ctx.exception.status_code, 409,
                         "a lost rename race must not surface as a 500")
        self.assertIn("shared", ctx.exception.detail)

    async def test_description_only_update_still_works(self):
        """The 409 handler reads the attempted name, which is None on this path."""
        alice = await self.make_user("alice", rpd_limit=5)
        await pool_routes.create_pool(PoolCreate(name="alices"), alice, self.db)

        await pool_routes.update_my_pool(PoolUpdate(description="hi"), alice, self.db)
        pool = (await self.db.execute(select(RequestPool))).scalar_one()
        self.assertEqual(pool.description, "hi")
        self.assertEqual(pool.name, "alices")

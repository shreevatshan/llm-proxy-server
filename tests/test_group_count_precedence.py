"""Tests that every model is counted against exactly one group.

A group claims models explicitly and/or whole provider instances; an explicit model
beats an instance, and a model no group claims is in the built-in Default group
(rate_limit_tracker.group_for). The same rule is applied by enforcement, by
get_today_count (Default), by get_today_group_count, and by settlement's fold. If any
of them disagreed, a quota page would show traffic its own gate never sees, and the
scopes would stop partitioning the day's usage.
"""

import unittest

from app.auth.models import ModelGroup, ModelGroupInstanceMember, ModelGroupMember
from app.profiles import DEFAULT_GROUP_ID
from tests.pool_test_base import DAY, PoolTestCase


class GroupCountPrecedenceTests(PoolTestCase):

    async def _groups(self):
        """azure/gpt-4o and anthropic/opus are explicit members of 'opus'; the whole
        azure instance is in 'edge'."""
        mg = ModelGroup(name="opus")
        ig = ModelGroup(name="edge")
        self.db.add_all([mg, ig])
        await self.db.flush()
        self.db.add_all([
            ModelGroupMember(group_id=mg.id, model_id="azure/gpt-4o"),
            ModelGroupMember(group_id=mg.id, model_id="anthropic/opus"),
            ModelGroupInstanceMember(group_id=ig.id, provider_key="azure"),
        ])
        await self.db.commit()
        await self.refresh()
        return mg, ig

    async def _count(self, user, group):
        snap = self.tracker._groups[group.id]
        return await self.request_tracker.get_today_group_count(user.id, snap.model_ids, snap.provider_keys)

    async def test_group_for_resolves_explicit_then_instance_then_default(self):
        mg, ig = await self._groups()
        self.assertEqual(self.tracker.group_for("azure/gpt-4o"), mg.id, "an explicit model wins")
        self.assertEqual(self.tracker.group_for("azure/gpt-5"), ig.id,
                         "any other model on the instance -- including one synced later")
        self.assertEqual(self.tracker.group_for("loose/model"), DEFAULT_GROUP_ID)
        self.assertEqual(self.tracker.group_for("azure"), ig.id, "a bare provider_key is its own instance")

    async def test_an_explicit_member_is_carved_out_of_the_instance_group(self):
        alice = await self.make_user("alice", rpd_limit=1000)
        mg, ig = await self._groups()
        await self.seed_usage("alice", 5, model="azure/gpt-4o")

        self.assertEqual(await self._count(alice, mg), 5)
        self.assertEqual(await self._count(alice, ig), 0, "the explicit group owns it")

    async def test_buffered_counts_obey_the_same_precedence(self):
        """The buffer scan and the SQL half must agree, or the number jumps on flush."""
        alice = await self.make_user("alice", rpd_limit=1000)
        mg, ig = await self._groups()
        self.request_tracker._usage_buffer[(DAY, 9, alice.id, "alice", "user", "azure/gpt-4o", "azure", 0)] = 4
        self.request_tracker._usage_buffer[(DAY, 9, alice.id, "alice", "user", "azure/gpt-5", "azure", 0)] = 2

        self.assertEqual(await self._count(alice, mg), 4)
        self.assertEqual(await self._count(alice, ig), 2)

    async def test_an_underscore_in_a_provider_key_is_not_a_wildcard(self):
        """SQL LIKE would read 'a_b/%' as matching 'axb/...'; the buffer would not."""
        alice = await self.make_user("alice", rpd_limit=1000)
        group = ModelGroup(name="under")
        self.db.add(group)
        await self.db.flush()
        self.db.add(ModelGroupInstanceMember(group_id=group.id, provider_key="a_b"))
        await self.db.commit()
        await self.refresh()
        await self.seed_usage("alice", 3, model="a_b/m")
        await self.seed_usage("alice", 7, model="axb/m")

        self.assertEqual(await self._count(alice, group), 3)
        self.assertEqual(await self.request_tracker.get_today_count(alice.id), 7)

    async def test_the_scopes_partition_the_days_usage(self):
        """Every request belongs to exactly one scope, so the counts must sum to it."""
        alice = await self.make_user("alice", rpd_limit=1000)
        mg, ig = await self._groups()
        await self.seed_usage("alice", 5, model="azure/gpt-4o")     # explicit: opus
        await self.seed_usage("alice", 4, model="azure/gpt-5")      # instance: edge
        await self.seed_usage("alice", 3, model="anthropic/opus")   # explicit: opus
        await self.seed_usage("alice", 2, model="loose/model")      # Default

        default = await self.request_tracker.get_today_count(alice.id)
        opus = await self._count(alice, mg)
        edge = await self._count(alice, ig)

        self.assertEqual((default, opus, edge), (2, 8, 4))
        self.assertEqual(default + opus + edge, 14)
        for model, scope in (("azure/gpt-4o", ("model_group", mg.id)),
                             ("azure/gpt-5", ("model_group", ig.id)),
                             ("loose/model", None)):
            self.assertEqual(self.tracker.scope_for_model(model), scope, "settlement folds the same way")


if __name__ == "__main__":
    unittest.main()

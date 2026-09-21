"""Tests that a model in both a model group and an instance group is counted once.

Scope precedence is instance group > model group > overall, and it is applied in five
places: scope_for_model, the auth dependency's choice of gate, get_today_count's
exclusion from the overall number, settlement's fold, and -- last to learn it --
get_today_group_count. While that last one counted by model id alone, a model served by
a grouped provider was charged against its model group *and* its instance group, so the
number /auth/quotas showed for the model group was inflated by traffic its own gate
never sees, and the scopes stopped partitioning the day's usage.
"""

import unittest

from app.auth.models import (
    InstanceGroup, InstanceGroupMember, ModelGroup, ModelGroupMember,
)
from tests.pool_test_base import DAY, PoolTestCase


class GroupCountPrecedenceTests(PoolTestCase):

    async def _groups(self):
        """azure/gpt-4o is in model group 'opus'; provider azure is in instance group 'edge'."""
        mg = ModelGroup(name="opus", rpm_default=None, rpd_default=100)
        ig = InstanceGroup(name="edge", rpm_default=None, rpd_default=100)
        self.db.add_all([mg, ig])
        await self.db.flush()
        self.db.add_all([
            ModelGroupMember(group_id=mg.id, model_id="azure/gpt-4o"),
            ModelGroupMember(group_id=mg.id, model_id="anthropic/opus"),
            InstanceGroupMember(group_id=ig.id, provider_key="azure"),
        ])
        await self.db.commit()
        await self.refresh()
        return mg, ig

    async def test_an_instance_grouped_model_does_not_count_against_its_model_group(self):
        await self.make_user("alice", rpd_limit=1000)
        await self._groups()
        await self.seed_usage("alice", 5, model="azure/gpt-4o")

        self.assertEqual(
            await self.request_tracker.get_today_group_count("alice", ["azure/gpt-4o"]), 0,
            "the instance group owns it, and only the instance-group gate will see it",
        )
        self.assertEqual(
            await self.request_tracker.get_today_instance_group_count("alice", ["azure"]), 5,
        )

    async def test_buffered_counts_obey_the_same_precedence(self):
        """The buffer scan and the SQL half must agree, or the number jumps on flush."""
        await self.make_user("alice", rpd_limit=1000)
        await self._groups()
        self.request_tracker._usage_buffer[(DAY, 9, "alice", "user", "azure/gpt-4o", "azure")] = 4

        self.assertEqual(
            await self.request_tracker.get_today_group_count("alice", ["azure/gpt-4o"]), 0)
        self.assertEqual(
            await self.request_tracker.get_today_instance_group_count("alice", ["azure"]), 4)

    async def test_an_ungrouped_provider_still_counts_for_its_model_group(self):
        await self.make_user("alice", rpd_limit=1000)
        await self._groups()
        await self.seed_usage("alice", 3, model="anthropic/opus")

        self.assertEqual(
            await self.request_tracker.get_today_group_count(
                "alice", ["azure/gpt-4o", "anthropic/opus"]), 3)

    async def test_the_scopes_partition_the_days_usage(self):
        """Every request belongs to exactly one scope, so the three counts must sum to it."""
        await self.make_user("alice", rpd_limit=1000)
        mg, ig = await self._groups()
        await self.seed_usage("alice", 5, model="azure/gpt-4o")     # instance group
        await self.seed_usage("alice", 3, model="anthropic/opus")   # model group
        await self.seed_usage("alice", 2, model="loose/model")      # overall

        overall = await self.request_tracker.get_today_count("alice")
        grouped = await self.request_tracker.get_today_group_count(
            "alice", ["azure/gpt-4o", "anthropic/opus"])
        instanced = await self.request_tracker.get_today_instance_group_count("alice", ["azure"])

        self.assertEqual((overall, grouped, instanced), (2, 3, 5))
        self.assertEqual(overall + grouped + instanced, 10)


if __name__ == "__main__":
    unittest.main()

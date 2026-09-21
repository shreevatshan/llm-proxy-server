"""Unit tests for the per-model parameter capability table.

Covers the table resolution itself (which params each model family rejects, per
surface), the in-place scrub helper, the thinking-config normalization, and the
unparseable-deployment-name warning.
"""

import logging
import unittest

from app.model_capabilities import (
    CONVERSE_SPELLINGS,
    SURFACE_CONVERSE,
    SURFACE_NATIVE,
    THINKING_ADAPTIVE,
    THINKING_BUDGET,
    is_claude_at_least,
    is_claude_model,
    normalize_thinking,
    rejected_params,
    scrub,
    thinking_style,
    warn_if_version_unparseable,
)


class RejectedParamsTests(unittest.TestCase):
    def test_claude_5_rejects_all_sampling_params_on_both_surfaces(self):
        # The gap that made Claude 5 fail on the native paths: temperature was
        # only ever stripped on Converse.
        for surface in (SURFACE_NATIVE, SURFACE_CONVERSE):
            with self.subTest(surface=surface):
                self.assertEqual(
                    rejected_params("claude-opus-5", surface),
                    frozenset({"temperature", "top_p", "top_k"}),
                )

    def test_claude_5_variants_all_match(self):
        for model in (
            "claude-opus-5",
            "claude-sonnet-5",
            "global.anthropic.claude-opus-5-v1:0",
            "claude-opus-5-20260101",
        ):
            with self.subTest(model=model):
                self.assertIn("temperature", rejected_params(model, SURFACE_NATIVE))

    def test_claude_4_7_and_4_8_reject_sampling_on_both_surfaces(self):
        # 4.7 removed temperature/top_p/top_k outright — they 400, they are not
        # merely deprecated. Forwarding temperature natively (as the earlier
        # Converse-only rule did) is an upstream error, not a determinism win.
        for model in ("claude-opus-4-7", "claude-opus-4-8"):
            for surface in (SURFACE_NATIVE, SURFACE_CONVERSE):
                with self.subTest(model=model, surface=surface):
                    self.assertEqual(
                        rejected_params(model, surface),
                        frozenset({"temperature", "top_p", "top_k"}),
                    )

    def test_fable_and_mythos_families_are_covered(self):
        # Post-trio families reject the same params; before the family
        # alternation included them they matched no rule and every request 400'd.
        for model in ("claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5"):
            with self.subTest(model=model):
                self.assertEqual(
                    rejected_params(model, SURFACE_NATIVE),
                    frozenset({"temperature", "top_p", "top_k"}),
                )

    def test_claude_4_6_and_below_unrestricted(self):
        # 4.6 is the last version that accepts all three; the Claude 3 shape
        # ("claude-3-5-sonnet-...") is version-first but must still parse.
        for model in (
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ):
            with self.subTest(model=model):
                self.assertEqual(rejected_params(model, SURFACE_NATIVE), frozenset())

    def test_grok_scoped_to_converse(self):
        self.assertEqual(
            rejected_params("us.xai.grok-4.6", SURFACE_CONVERSE),
            frozenset({"temperature", "top_p", "stop_sequences"}),
        )
        self.assertEqual(rejected_params("us.xai.grok-4.6", SURFACE_NATIVE), frozenset())

    def test_unknown_and_unparseable_models_unrestricted(self):
        # A deployment name that hides the version matches no rule — the known
        # limitation the startup warning exists to surface.
        for model in ("gpt-4o", "opus5-prod", "us.amazon.nova-pro-v1:0", ""):
            with self.subTest(model=model):
                self.assertEqual(rejected_params(model, SURFACE_NATIVE), frozenset())


class ScrubTests(unittest.TestCase):
    def test_removes_rejected_and_reports_canonical_names(self):
        body = {"temperature": 0.5, "top_p": 0.9, "top_k": 5, "max_tokens": 64}
        removed = scrub(body, "claude-opus-5", SURFACE_NATIVE)
        self.assertEqual(removed, ["temperature", "top_k", "top_p"])
        self.assertEqual(body, {"max_tokens": 64})

    def test_keeps_allowed_params(self):
        body = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 64}
        removed = scrub(body, "claude-sonnet-4-6", SURFACE_NATIVE)
        self.assertEqual(removed, [])
        self.assertEqual(body, {"temperature": 0.5, "top_p": 0.9, "max_tokens": 64})

    def test_converse_spellings(self):
        config = {"temperature": 0.5, "topP": 0.9, "stopSequences": ["x"], "maxTokens": 64}
        removed = scrub(
            config, "us.xai.grok-4.6", SURFACE_CONVERSE, spellings=CONVERSE_SPELLINGS
        )
        self.assertEqual(removed, ["stop_sequences", "temperature", "top_p"])
        self.assertEqual(config, {"maxTokens": 64})

    def test_only_restricts_the_scrub(self):
        body = {"temperature": 0.5, "top_p": 0.9, "top_k": 5}
        removed = scrub(
            body, "claude-opus-5", SURFACE_NATIVE, only=frozenset({"top_k"})
        )
        self.assertEqual(removed, ["top_k"])
        self.assertEqual(body, {"temperature": 0.5, "top_p": 0.9})

    def test_absent_params_are_not_reported(self):
        body = {"max_tokens": 64}
        self.assertEqual(scrub(body, "claude-opus-5", SURFACE_NATIVE), [])
        self.assertEqual(body, {"max_tokens": 64})

    def test_none_mapping_is_safe(self):
        self.assertEqual(scrub(None, "claude-opus-5", SURFACE_NATIVE), [])


class ThinkingStyleTests(unittest.TestCase):
    def test_adaptive_from_4_7_onwards(self):
        # budget_tokens was removed in 4.7, not in 5.0 — scoping adaptive to
        # >= 5 left 4.7/4.8 emitting a budget that Bedrock and the native APIs
        # both reject.
        for model in (
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5-1",
        ):
            with self.subTest(model=model):
                self.assertEqual(thinking_style(model), THINKING_ADAPTIVE)

    def test_older_models_use_budgets(self):
        # 4.6 deprecates budget_tokens but still honours it; Haiku 4.5 requires it.
        for model in ("claude-sonnet-4-6", "claude-haiku-4-5", "gpt-4o", ""):
            with self.subTest(model=model):
                self.assertEqual(thinking_style(model), THINKING_BUDGET)

    def test_budget_converted_to_adaptive(self):
        thinking, dropped = normalize_thinking(
            {"type": "enabled", "budget_tokens": 4096}, "claude-opus-5"
        )
        self.assertEqual(thinking, {"type": "adaptive"})
        self.assertEqual(dropped, ["thinking.budget_tokens"])

    def test_sibling_keys_survive_normalization(self):
        # Only budget_tokens is rejected. Dropping display too would silently
        # give the client empty thinking blocks instead of the summary it asked
        # for, with nothing reported in dropped.
        thinking, dropped = normalize_thinking(
            {"type": "enabled", "budget_tokens": 4096, "display": "summarized"},
            "claude-opus-5",
        )
        self.assertEqual(thinking, {"type": "adaptive", "display": "summarized"})
        self.assertEqual(dropped, ["thinking.budget_tokens"])

    def test_enabled_without_budget_still_becomes_adaptive(self):
        thinking, dropped = normalize_thinking({"type": "enabled"}, "claude-opus-5")
        self.assertEqual(thinking, {"type": "adaptive"})
        self.assertEqual(dropped, [])

    def test_disabled_and_adaptive_pass_through(self):
        for config in ({"type": "disabled"}, {"type": "adaptive"}):
            with self.subTest(config=config):
                thinking, dropped = normalize_thinking(config, "claude-opus-5")
                self.assertEqual(thinking, config)
                self.assertEqual(dropped, [])

    def test_budget_preserved_on_older_models(self):
        config = {"type": "enabled", "budget_tokens": 4096}
        thinking, dropped = normalize_thinking(config, "claude-sonnet-4-6")
        self.assertEqual(thinking, config)
        self.assertEqual(dropped, [])

    def test_non_dict_passes_through(self):
        self.assertEqual(normalize_thinking(None, "claude-opus-5"), (None, []))


class VersionParsingTests(unittest.TestCase):
    def test_is_claude_model(self):
        self.assertTrue(is_claude_model("claude-opus-5"))
        self.assertTrue(is_claude_model("global.anthropic.claude-sonnet-4-6-v1:0"))
        self.assertTrue(is_claude_model("claude-fable-5-1"))
        self.assertTrue(is_claude_model("claude-3-5-sonnet-20241022"))
        self.assertFalse(is_claude_model("opus5-prod"))
        self.assertFalse(is_claude_model("gpt-4o"))

    def test_version_first_naming_parses(self):
        # "claude-3-5-sonnet-*" puts the version before the family. It needs no
        # scrubbing, but it must parse so the startup warning stays quiet.
        self.assertTrue(is_claude_at_least("claude-3-5-sonnet-20241022", 3, 0))
        self.assertTrue(is_claude_at_least("claude-3-5-sonnet-20241022", 3, 5))
        self.assertFalse(is_claude_at_least("claude-3-5-sonnet-20241022", 4, 7))
        self.assertFalse(is_claude_at_least("claude-3-opus-20240229", 3, 5))

    def test_fable_and_mythos_versions_parse(self):
        self.assertTrue(is_claude_at_least("claude-fable-5-1", 5, 0))
        self.assertTrue(is_claude_at_least("claude-mythos-5-1", 4, 7))
        self.assertFalse(is_claude_at_least("claude-fable-5-1", 6, 0))

    def test_is_claude_at_least_moved_intact(self):
        # Same behaviour as before the move out of app.anthropic_models.
        self.assertTrue(is_claude_at_least("claude-sonnet-5", 4, 7))
        self.assertTrue(is_claude_at_least("claude-sonnet-5-20250101", 5, 0))
        self.assertFalse(is_claude_at_least("claude-sonnet-4-6", 4, 7))
        self.assertFalse(is_claude_at_least("gpt-4o", 4, 7))


class DeploymentNameWarningTests(unittest.TestCase):
    def test_warns_on_claude_looking_name_without_version(self):
        with self.assertLogs("app.model_capabilities", level=logging.WARNING) as cm:
            self.assertTrue(warn_if_version_unparseable("opus5-prod", "test"))
        self.assertIn("opus5-prod", cm.output[0])

    def test_warns_for_every_family_not_just_the_original_trio(self):
        """The marker list is derived from _CLAUDE_FAMILIES, so none is missed.

        A deployment named for a newer family gets no scrub (the version does not
        parse) and, before this, no warning either — leaving nothing to point the
        operator at the deployment name when the model rejected the request.
        """
        for name in ("fable5-prod", "mythos-1", "sonnet5-prod", "haiku45"):
            with self.subTest(name=name):
                with self.assertLogs("app.model_capabilities", level=logging.WARNING):
                    self.assertTrue(warn_if_version_unparseable(name, "test"))

    def test_silent_for_parseable_and_unrelated_names(self):
        # claude-3-* and the fable/mythos families are correctly-named
        # deployments — warning about them told operators to rename something
        # that needs no scrubbing at all.
        for model in (
            "claude-opus-5",
            "claude-3-5-sonnet-20241022",
            "claude-fable-5-1",
            "gpt-4o",
            "us.amazon.nova-pro-v1:0",
        ):
            with self.subTest(model=model):
                self.assertFalse(warn_if_version_unparseable(model, "test"))


if __name__ == "__main__":
    unittest.main()

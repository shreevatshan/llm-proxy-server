import os
import unittest
from unittest.mock import patch

from app import anthropic_models


class AnthropicSdkTimeoutEnvTests(unittest.TestCase):
    def test_missing_env_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                anthropic_models._get_positive_float_env("ANTHROPIC_SDK_TIMEOUT_SECONDS", 900.0),
                900.0,
            )

    def test_valid_env_returns_value(self):
        with patch.dict(os.environ, {"ANTHROPIC_SDK_TIMEOUT_SECONDS": "123.5"}, clear=False):
            self.assertEqual(
                anthropic_models._get_positive_float_env("ANTHROPIC_SDK_TIMEOUT_SECONDS", 900.0),
                123.5,
            )

    def test_invalid_env_returns_default_and_logs(self):
        with patch.dict(os.environ, {"ANTHROPIC_SDK_TIMEOUT_SECONDS": "900s"}, clear=False):
            with self.assertLogs("app.anthropic_models", level="WARNING") as logs:
                value = anthropic_models._get_positive_float_env("ANTHROPIC_SDK_TIMEOUT_SECONDS", 900.0)

        self.assertEqual(value, 900.0)
        self.assertIn("Invalid float", "\n".join(logs.output))

    def test_non_positive_env_returns_default_and_logs(self):
        for raw in ("0", "-1"):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"ANTHROPIC_SDK_TIMEOUT_SECONDS": raw}, clear=False):
                    with self.assertLogs("app.anthropic_models", level="WARNING") as logs:
                        value = anthropic_models._get_positive_float_env("ANTHROPIC_SDK_TIMEOUT_SECONDS", 900.0)

                self.assertEqual(value, 900.0)
                self.assertIn("Non-positive or non-finite float", "\n".join(logs.output))

    def test_non_finite_env_returns_default_and_logs(self):
        for raw in ("nan", "inf", "-inf"):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {"ANTHROPIC_SDK_TIMEOUT_SECONDS": raw}, clear=False):
                    with self.assertLogs("app.anthropic_models", level="WARNING") as logs:
                        value = anthropic_models._get_positive_float_env("ANTHROPIC_SDK_TIMEOUT_SECONDS", 900.0)

                self.assertEqual(value, 900.0)
                self.assertIn("Non-positive or non-finite float", "\n".join(logs.output))


class IsClaudeAtLeastTests(unittest.TestCase):
    def test_major_minor_naming(self):
        self.assertTrue(anthropic_models.is_claude_at_least("claude-sonnet-4-7", 4, 7))
        self.assertFalse(anthropic_models.is_claude_at_least("claude-sonnet-4-6", 4, 7))
        self.assertFalse(anthropic_models.is_claude_at_least("claude-sonnet-4-5", 4, 7))

    def test_major_only_naming(self):
        # Newer models drop the minor entirely; must count as >= 4.7.
        self.assertTrue(anthropic_models.is_claude_at_least("claude-sonnet-5", 4, 7))
        self.assertTrue(anthropic_models.is_claude_at_least("us.anthropic.claude-sonnet-5", 4, 7))

    def test_date_snapshot_not_treated_as_minor(self):
        self.assertTrue(anthropic_models.is_claude_at_least("claude-sonnet-5-20250101", 4, 7))

    def test_non_claude_and_empty(self):
        self.assertFalse(anthropic_models.is_claude_at_least("gpt-4o", 4, 7))
        self.assertFalse(anthropic_models.is_claude_at_least("", 4, 7))
        self.assertFalse(anthropic_models.is_claude_at_least("claude-3-5-sonnet-20241022", 4, 7))


class BuildAnthropicSdkKwargsTopPTests(unittest.TestCase):
    def _request(self, model, top_p=0.9):
        return anthropic_models.AnthropicMessagesRequest(
            model=model,
            max_tokens=16,
            top_p=top_p,
            messages=[{"role": "user", "content": "hi"}],
        )

    def test_top_p_dropped_for_deprecated_model(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request("claude-sonnet-5"), "claude-sonnet-5"
        )
        self.assertNotIn("top_p", kwargs)

    def test_top_p_kept_for_older_model(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request("claude-sonnet-4-5"), "claude-sonnet-4-5"
        )
        self.assertEqual(kwargs.get("top_p"), 0.9)


class BuildAnthropicSdkKwargsCapabilityTests(unittest.TestCase):
    """Every Anthropic-SDK provider (Foundry, direct Anthropic, custom) goes
    through build_anthropic_sdk_kwargs, so the capability scrub lands here once."""

    def _request(self, model, **kwargs):
        kwargs.setdefault("max_tokens", 16)
        return anthropic_models.AnthropicMessagesRequest(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            **kwargs,
        )

    def test_claude_5_drops_every_sampling_param(self):
        # The Copilot failure: temperature reached Claude 5 and came back a 400.
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request("claude-opus-5", temperature=0.1, top_p=0.9, top_k=5),
            "claude-opus-5",
        )
        for param in ("temperature", "top_p", "top_k"):
            self.assertNotIn(param, kwargs)

    def test_claude_4_7_and_4_8_drop_every_sampling_param(self):
        # 4.7 removed all three, natively as well as on Converse. The table
        # previously forwarded temperature here, which is a 400 upstream.
        for model in ("claude-opus-4-7", "claude-opus-4-8"):
            with self.subTest(model=model):
                kwargs = anthropic_models.build_anthropic_sdk_kwargs(
                    self._request(model, temperature=0.1, top_p=0.9, top_k=5),
                    model,
                )
                for param in ("temperature", "top_p", "top_k"):
                    self.assertNotIn(param, kwargs)

    def test_older_model_keeps_everything(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request("claude-sonnet-4-6", temperature=0.1, top_p=0.9, top_k=5),
            "claude-sonnet-4-6",
        )
        self.assertEqual(kwargs.get("temperature"), 0.1)
        self.assertEqual(kwargs.get("top_p"), 0.9)
        self.assertEqual(kwargs.get("top_k"), 5)

    def test_stop_sequences_never_scrubbed(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request("claude-opus-5", stop_sequences=["END"]), "claude-opus-5"
        )
        self.assertEqual(kwargs.get("stop_sequences"), ["END"])

    def test_thinking_budget_becomes_adaptive_on_claude_5(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request(
                "claude-opus-5",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            ),
            "claude-opus-5",
        )
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})

    def test_thinking_budget_becomes_adaptive_on_claude_4_8(self):
        # budget_tokens was removed in 4.7, not 5.0 — it 400s here too.
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request(
                "claude-opus-4-8",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            ),
            "claude-opus-4-8",
        )
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})

    def test_thinking_budget_preserved_on_older_model(self):
        kwargs = anthropic_models.build_anthropic_sdk_kwargs(
            self._request(
                "claude-sonnet-4-6",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            ),
            "claude-sonnet-4-6",
        )
        self.assertEqual(kwargs["thinking"], {"type": "enabled", "budget_tokens": 4096})


if __name__ == "__main__":
    unittest.main()

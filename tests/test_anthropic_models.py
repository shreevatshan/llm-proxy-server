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


if __name__ == "__main__":
    unittest.main()

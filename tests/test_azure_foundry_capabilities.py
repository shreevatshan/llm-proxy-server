"""Unit tests for capability scrubbing on the Azure Foundry native path.

_prepare_foundry_native_request is the third consumer of app.model_capabilities
(alongside the Anthropic SDK kwargs builder and Bedrock's native builder), so it
has to reshape a request identically to the other two.
"""

import unittest

from app.anthropic_models import AnthropicMessage, AnthropicMessagesRequest
from app.providers.azure_provider import AzureProvider


def _provider():
    return AzureProvider({
        "name": "test",
        "endpoint": "https://example.openai.azure.com/",
        "api_key": "k",
        "azure_backend": "foundry",
        "dynamic_discovery": False,
    })


class FoundryNativeCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def _request(self, model, **kwargs):
        kwargs.setdefault("max_tokens", 64)
        return AnthropicMessagesRequest(
            model=model,
            messages=[AnthropicMessage(role="user", content="hi")],
            **kwargs,
        )

    def test_sampling_params_scrubbed(self):
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request("claude-opus-5", temperature=0.1, top_p=0.9, top_k=5),
            stream=False,
        )
        for param in ("temperature", "top_p", "top_k"):
            self.assertNotIn(param, payload)
        self.assertEqual(
            sorted(dropped), ["temperature", "top_k", "top_p"]
        )

    def test_thinking_budget_converted_and_reported(self):
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request(
                "claude-opus-5",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            ),
            stream=False,
        )
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertIn("thinking.budget_tokens", dropped)

    def test_enabled_without_budget_is_still_rewritten(self):
        # The write-back used to be gated on something having been dropped, so
        # this config kept type="enabled" and 400'd upstream.
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request("claude-opus-5", thinking={"type": "enabled"}),
            stream=False,
        )
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(dropped, [])

    def test_thinking_display_survives(self):
        payload, _ = self.p._prepare_foundry_native_request(
            self._request(
                "claude-opus-5",
                max_tokens=32000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4096,
                    "display": "summarized",
                },
            ),
            stream=False,
        )
        self.assertEqual(
            payload["thinking"], {"type": "adaptive", "display": "summarized"}
        )

    def test_older_model_untouched(self):
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request(
                "claude-sonnet-4-6",
                max_tokens=32000,
                temperature=0.1,
                thinking={"type": "enabled", "budget_tokens": 4096},
            ),
            stream=False,
        )
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(
            payload["thinking"], {"type": "enabled", "budget_tokens": 4096}
        )
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()

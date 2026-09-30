"""Unit tests for capability scrubbing on the Azure Foundry native path.

_prepare_foundry_native_request is the third consumer of app.model_capabilities
(alongside the Anthropic SDK kwargs builder and Bedrock's native builder), so it
has to reshape a request identically to the other two.
"""

import json
import unittest

from app.anthropic_models import (
    ANTHROPIC_FORWARDABLE_EXTRAS,
    AnthropicMessage,
    AnthropicMessagesRequest,
    build_anthropic_sdk_kwargs,
)
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


class FoundryNativeExtraFieldTests(unittest.TestCase):
    """Foundry's Anthropic surface validates with a closed schema: an unknown
    top-level field comes back as "<name>: Extra inputs are not permitted" and
    fails the whole request. The wire path and the dropped_fields reporting path
    are separate implementations, so both are pinned here."""

    def setUp(self):
        self.p = _provider()

    def _request(self, **kwargs):
        kwargs.setdefault("max_tokens", 64)
        return AnthropicMessagesRequest(
            model="claude-opus-5",
            messages=[AnthropicMessage(role="user", content="hi")],
            **kwargs,
        )

    def test_unknown_field_stripped_from_payload_and_reported(self):
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request(safeguards={"enabled": True}), stream=True
        )
        self.assertNotIn("safeguards", payload)
        self.assertIn("safeguards", dropped)

    def test_allowlisted_extra_survives(self):
        payload, dropped = self.p._prepare_foundry_native_request(
            self._request(context_management={"ttl": 1}), stream=True
        )
        self.assertEqual(payload.get("context_management"), {"ttl": 1})
        self.assertEqual(dropped, [])

    def test_metadata_surfaces_dropped_field_for_the_response_header(self):
        meta = self.p.get_anthropic_request_metadata(
            self._request(safeguards={"enabled": True})
        )
        self.assertEqual(meta.mode, "native")
        self.assertIn("safeguards", meta.dropped_fields)

    def test_unknown_field_never_reaches_the_wire(self):
        kwargs = build_anthropic_sdk_kwargs(
            self._request(safeguards={"enabled": True}),
            "claude-opus-5",
            forwardable_extras=ANTHROPIC_FORWARDABLE_EXTRAS,
        )
        self.assertNotIn("safeguards", json.dumps(kwargs, default=str))

    def test_reporting_and_wire_paths_agree(self):
        request = self._request(
            safeguards={"enabled": True},
            context_management={"ttl": 1},
            another_new_field={"x": 1},
        )
        _, dropped = self.p._prepare_foundry_native_request(request, stream=True)
        wire_dropped = []
        kwargs = build_anthropic_sdk_kwargs(
            request,
            "claude-opus-5",
            forwardable_extras=ANTHROPIC_FORWARDABLE_EXTRAS,
            dropped_fields=wire_dropped,
        )
        self.assertEqual(sorted(wire_dropped), ["another_new_field", "safeguards"])
        for name in wire_dropped:
            self.assertIn(name, dropped)
        self.assertEqual(kwargs.get("extra_body"), {"context_management": {"ttl": 1}})


if __name__ == "__main__":
    unittest.main()

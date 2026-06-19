"""Unit tests for the Bedrock provider.

These cover the pure/synchronous logic that can be tested without live AWS:
A2 (stream tool-call index), A4 (cache tokens), A5 (reasoning_content),
A6 (tool pairing), A7 (empty content), A9 (finish reason), C6 (budget clamp),
and the _map_bedrock_error table.

Plus the AsyncAnthropicBedrock native-path migration (Part 2): _build_native_sdk_kwargs
shaping, lazy client init, messages.create / messages.stream behavior, and SDK
error translation to ProviderHTTPError.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from app.providers.bedrock_provider import (
    BedrockProvider,
    _map_bedrock_error,
)
from app.providers.base import ProviderHTTPError


def _provider() -> BedrockProvider:
    # Bypass __init__ (which needs real AWS config); we only exercise pure methods.
    p = BedrockProvider.__new__(BedrockProvider)
    p.debug = False
    return p


class FinishReasonTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_known_reasons_map(self):
        self.assertEqual(self.p._convert_finish_reason("tool_use"), "tool_calls")
        self.assertEqual(self.p._convert_finish_reason("end_turn"), "stop")
        self.assertEqual(self.p._convert_finish_reason("max_tokens"), "length")
        self.assertEqual(self.p._convert_finish_reason("guardrail_intervened"), "content_filter")

    def test_unknown_reason_defaults_to_stop(self):
        # A9: never leak a non-enum value through verbatim.
        self.assertEqual(self.p._convert_finish_reason("some_new_reason"), "stop")

    def test_empty_is_none(self):
        self.assertIsNone(self.p._convert_finish_reason(""))


class BudgetTokenClampTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_low_effort_small_max_clamped_to_floor(self):
        # C6: 0.3 * 2048 = 614 < 1024 -> clamp up to 1024.
        self.assertEqual(self.p._calc_budget_tokens(2048, "low"), 1024)

    def test_high_effort_clamped_below_max(self):
        self.assertEqual(self.p._calc_budget_tokens(4096, "high"), 4095)

    def test_never_exceeds_max_minus_one(self):
        # medium = 0.6 * 100000 = 60000, well under max-1.
        self.assertEqual(self.p._calc_budget_tokens(100000, "medium"), 60000)


class ToolPairingTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_cross_turn_pairing_preserved(self):
        # A6: toolUse and its toolResult are NOT adjacent — must be kept.
        messages = [
            {"role": "assistant", "content": [
                {"text": "let me check"},
                {"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}},
            ]},
            {"role": "assistant", "content": [{"text": "still working"}]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}},
            ]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        ids = [
            item["toolUse"]["toolUseId"]
            for m in out for item in m["content"]
            if isinstance(item, dict) and "toolUse" in item
        ]
        self.assertIn("t1", ids)

    def test_orphan_tooluse_dropped(self):
        messages = [
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "orphan", "name": "f", "input": {}}},
            ]},
            {"role": "user", "content": [{"text": "no result here"}]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        ids = [
            item["toolUse"]["toolUseId"]
            for m in out for item in m["content"]
            if isinstance(item, dict) and "toolUse" in item
        ]
        self.assertNotIn("orphan", ids)


class MapBedrockErrorTests(unittest.TestCase):
    def test_throttling_maps_to_429(self):
        m = _map_bedrock_error("ThrottlingException", "slow down")
        self.assertEqual(m["status"], 429)
        self.assertEqual(m["body"]["error"]["type"], "rate_limit_error")

    def test_validation_maps_to_400(self):
        m = _map_bedrock_error("ValidationException", "bad")
        self.assertEqual(m["status"], 400)
        self.assertEqual(m["body"]["error"]["type"], "invalid_request_error")

    def test_unknown_maps_to_500(self):
        m = _map_bedrock_error("SomethingNew", "x")
        self.assertEqual(m["status"], 500)


class StreamChunkTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()
        self.state = {"tool_call_index": -1, "block_index_map": {}}

    def test_message_start_omits_content(self):
        # A3: first delta carries role only, no content key.
        resp, _ = self.p._parse_stream_chunk(
            {"messageStart": {"role": "assistant"}}, "id", "m", self.state
        )
        self.assertEqual(resp["choices"][0]["delta"], {"role": "assistant"})

    def test_tool_only_response_index_zero(self):
        # A2: first tool block at contentBlockIndex 0 -> OpenAI index 0, not -1.
        chunk = {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "t1", "name": "f"}},
            }
        }
        resp, state = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        self.assertEqual(resp["choices"][0]["delta"]["tool_calls"][0].index, 0)
        # The matching delta resolves to the same ordinal.
        delta_chunk = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"a":1}'}},
            }
        }
        resp2, _ = self.p._parse_stream_chunk(delta_chunk, "id", "m", state)
        self.assertEqual(resp2["choices"][0]["delta"]["tool_calls"][0].index, 0)

    def test_reasoning_emits_reasoning_content(self):
        # A5: reasoning text on reasoning_content field, no <think> literals.
        chunk = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "hmm"}}}}
        resp, _ = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        self.assertEqual(resp["choices"][0]["delta"], {"reasoning_content": "hmm"})

    def test_usage_includes_cache_tokens(self):
        # A4: cache token counts preserved in the streaming usage chunk.
        chunk = {"metadata": {"usage": {
            "inputTokens": 10, "outputTokens": 5, "totalTokens": 18,
            "cacheReadInputTokens": 3, "cacheWriteInputTokens": 0,
        }}}
        resp, _ = self.p._parse_stream_chunk(chunk, "id", "m", self.state)
        usage = resp["usage"]
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(usage["cache_read_input_tokens"], 3)


class BuildNativeSdkKwargsTests(unittest.TestCase):
    """Part 2: _build_native_sdk_kwargs adapts the Bedrock transform output to
    the AsyncAnthropicBedrock messages.create/stream call contract."""

    def setUp(self):
        from app.anthropic_models import AnthropicMessagesRequest

        self.p = _provider()
        self.Request = AnthropicMessagesRequest

    def _req(self, **extra):
        base = {
            "model": "claude-opus-4-8",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        base.update(extra)
        return self.Request(**base)

    def test_anthropic_version_stripped_and_model_set(self):
        kwargs, beta = self._build("us.anthropic.claude-opus-4-8")
        self.assertNotIn("anthropic_version", kwargs)
        self.assertEqual(kwargs["model"], "us.anthropic.claude-opus-4-8")

    def _build(self, model_id, request=None, anthropic_beta=None):
        request = request or self._req()
        return self.p._build_native_sdk_kwargs(request, model_id, anthropic_beta)

    def test_beta_moved_to_header_not_body(self):
        # interleaved-thinking maps to a real Bedrock beta flag.
        kwargs, beta = self._build(
            "claude-opus-4-8", anthropic_beta="interleaved-thinking-2025-05-14"
        )
        self.assertNotIn("anthropic_beta", kwargs)
        self.assertIsNotNone(beta)
        self.assertIn("interleaved-thinking-2025-05-14", beta)

    def test_no_beta_returns_none_header(self):
        _, beta = self._build("claude-opus-4-8")
        self.assertIsNone(beta)

    def test_cache_scope_stripped(self):
        req = self._req(
            system=[{
                "type": "text",
                "text": "sys",
                "cache_control": {"type": "ephemeral", "scope": {"foo": "bar"}},
            }]
        )
        kwargs, _ = self._build("claude-opus-4-8", request=req)
        for block in kwargs["system"]:
            cc = block.get("cache_control")
            if isinstance(cc, dict):
                self.assertNotIn("scope", cc)

    def test_caller_stripped_from_tool_use(self):
        req = self._req(messages=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {},
                 "caller": "should_be_removed"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ]},
        ])
        kwargs, _ = self._build("claude-opus-4-8", request=req)
        for msg in kwargs["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        self.assertNotIn("caller", block)

    def test_web_search_tool_result_converted(self):
        req = self._req(messages=[
            {"role": "user", "content": [
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                 "content": [{"type": "web_search_result", "title": "T",
                              "url": "http://x", "encrypted_content": "E"}]},
            ]},
        ])
        kwargs, _ = self._build("claude-opus-4-8", request=req)
        types = [
            b.get("type")
            for m in kwargs["messages"] for b in (m.get("content") or [])
            if isinstance(b, dict)
        ]
        self.assertIn("tool_result", types)
        self.assertNotIn("web_search_tool_result", types)

    def test_skip_types_dropped(self):
        req = self._req(tools=[
            {"type": "web_search_20250305", "name": "ws"},
            {"name": "real", "description": "d",
             "input_schema": {"type": "object", "properties": {}}},
        ])
        kwargs, _ = self._build("claude-opus-4-8", request=req)
        tool_names = [t.get("name") for t in kwargs.get("tools", [])]
        self.assertIn("real", tool_names)
        self.assertNotIn("ws", tool_names)

    def test_context_management_routed_to_extra_body(self):
        req = self._req(context_management={"edits": []})
        kwargs, _ = self._build("claude-opus-4-8", request=req)
        self.assertNotIn("context_management", kwargs)
        self.assertIn("extra_body", kwargs)
        self.assertIn("context_management", kwargs["extra_body"])


class InitClientTests(unittest.TestCase):
    """Part 2: lazy AsyncAnthropicBedrock init wiring."""

    def _bare_provider(self):
        p = BedrockProvider.__new__(BedrockProvider)
        p.aws_region = "us-west-2"
        p.aws_access_key = "AKIA_TEST"
        p.aws_secret_key = "secret_test"
        p._anthropic_client = None
        return p

    def test_builds_client_with_region_and_keys(self):
        p = self._bare_provider()
        fake_ctor = mock.MagicMock(return_value="CLIENT")
        with mock.patch("anthropic.AsyncAnthropicBedrock", fake_ctor):
            p._init_bedrock_anthropic_client()
        self.assertEqual(p._anthropic_client, "CLIENT")
        kwargs = fake_ctor.call_args.kwargs
        self.assertEqual(kwargs["aws_region"], "us-west-2")
        self.assertEqual(kwargs["aws_access_key"], "AKIA_TEST")
        self.assertEqual(kwargs["aws_secret_key"], "secret_test")
        self.assertIn("timeout", kwargs)
        self.assertIn("max_retries", kwargs)

    def test_omits_creds_when_absent(self):
        p = self._bare_provider()
        p.aws_access_key = None
        p.aws_secret_key = None
        fake_ctor = mock.MagicMock(return_value="CLIENT")
        with mock.patch("anthropic.AsyncAnthropicBedrock", fake_ctor):
            p._init_bedrock_anthropic_client()
        kwargs = fake_ctor.call_args.kwargs
        self.assertNotIn("aws_access_key", kwargs)
        self.assertNotIn("aws_secret_key", kwargs)

    def test_idempotent(self):
        p = self._bare_provider()
        p._anthropic_client = "EXISTING"
        fake_ctor = mock.MagicMock()
        with mock.patch("anthropic.AsyncAnthropicBedrock", fake_ctor):
            p._init_bedrock_anthropic_client()
        fake_ctor.assert_not_called()
        self.assertEqual(p._anthropic_client, "EXISTING")


class TranslateSdkErrorTests(unittest.TestCase):
    """Part 2 / C8: SDK exceptions become ProviderHTTPError with correct status."""

    def setUp(self):
        self.p = _provider()
        self.p.full_provider_name = "bedrock:test"

    def _make_anthropic_error(self, cls_name, status):
        import anthropic
        cls = getattr(anthropic, cls_name)
        response = SimpleNamespace(
            status_code=status,
            headers={},
            request=None,
        )
        body = {"type": "error", "error": {"type": "x", "message": "boom"}}
        # APIStatusError subclasses take (message, *, response, body)
        return cls("boom", response=response, body=body)

    def test_bad_request_maps_to_400(self):
        err = self._make_anthropic_error("BadRequestError", 400)
        translated = self.p._translate_bedrock_sdk_error(err)
        self.assertIsInstance(translated, ProviderHTTPError)
        self.assertEqual(translated.status_code, 400)

    def test_rate_limit_maps_to_429(self):
        err = self._make_anthropic_error("RateLimitError", 429)
        translated = self.p._translate_bedrock_sdk_error(err)
        self.assertIsInstance(translated, ProviderHTTPError)
        self.assertEqual(translated.status_code, 429)

    def test_non_anthropic_error_passthrough(self):
        err = ValueError("not an sdk error")
        translated = self.p._translate_bedrock_sdk_error(err)
        self.assertIs(translated, err)


class NativeMessagesTests(unittest.IsolatedAsyncioTestCase):
    """Part 2: anthropic_messages / anthropic_messages_stream native branches
    drive the AsyncAnthropicBedrock SDK."""

    def setUp(self):
        from app.anthropic_models import AnthropicMessagesRequest

        self.p = _provider()
        self.p.aws_region = "us-west-2"
        self.p.bedrock_model_list = {}
        self.p.full_provider_name = "bedrock:test"
        self.request = AnthropicMessagesRequest(
            model="claude-opus-4-8",
            max_tokens=50,
            messages=[{"role": "user", "content": "hi"}],
        )

    async def test_non_stream_returns_dump_with_model_override(self):
        dump = {
            "id": "msg_bdrk_real",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-opus-4-8",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        fake_resp = mock.MagicMock()
        fake_resp.model_dump_json.return_value = json.dumps(dump)
        client = mock.MagicMock()
        client.messages.create = mock.AsyncMock(return_value=fake_resp)
        self.p._anthropic_client = client

        result = await self.p.anthropic_messages(self.request)
        # real upstream id preserved; model echoes the client-facing name
        self.assertEqual(result["id"], "msg_bdrk_real")
        self.assertEqual(result["model"], "claude-opus-4-8")
        self.assertTrue(client.messages.create.await_count == 1)

    async def test_non_stream_error_translated(self):
        client = mock.MagicMock()
        import anthropic
        response = SimpleNamespace(status_code=400, headers={}, request=None)
        err = anthropic.BadRequestError(
            "bad", response=response,
            body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )
        client.messages.create = mock.AsyncMock(side_effect=err)
        self.p._anthropic_client = client

        with self.assertRaises(ProviderHTTPError) as ctx:
            await self.p.anthropic_messages(self.request)
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_stream_emits_sse_and_stops_at_terminal(self):
        def _event(payload):
            e = mock.MagicMock()
            e.model_dump_json.return_value = json.dumps(payload)
            return e

        events = [
            _event({"type": "message_start", "message": {}}),
            _event({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}),
            _event({"type": "message_stop"}),
            _event({"type": "should_not_appear"}),
        ]

        class _FakeStream:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *a):
                return False

            async def __aiter__(self_inner):
                for e in events:
                    yield e

        client = mock.MagicMock()
        client.messages.stream = mock.MagicMock(return_value=_FakeStream())
        self.p._anthropic_client = client

        chunks = []
        async for sse in self.p.anthropic_messages_stream(self.request):
            chunks.append(sse)

        joined = "".join(chunks)
        self.assertIn("event: message_start", joined)
        self.assertIn("event: message_stop", joined)
        # post-terminal event within drain budget is consumed but not emitted
        self.assertNotIn("should_not_appear", joined)


if __name__ == "__main__":
    unittest.main()

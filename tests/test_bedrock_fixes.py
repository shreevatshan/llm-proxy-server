"""Unit tests for the Bedrock provider.

These cover the pure/synchronous logic that can be tested without live AWS:
A2 (stream tool-call index), A4 (cache tokens), A5 (reasoning_content),
A6 (tool pairing), A7 (empty content), A9 (finish reason), C6 (budget clamp),
and the _map_bedrock_error table.

Plus the AsyncAnthropicBedrock native-path migration (Part 2): _build_native_sdk_kwargs
shaping, lazy client init, messages.create / messages.stream behavior, and SDK
error translation to ProviderHTTPError.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from app.providers.bedrock_provider import (
    BedrockProvider,
    _map_bedrock_error,
)
from app.providers.base import ProviderHTTPError
from app.anthropic_models import AnthropicMessage, AnthropicMessagesRequest
from app.openai_models import ChatCompletionRequest, ChatMessage


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

    @staticmethod
    def _shape(messages):
        """[(role, [block kind:id ...]), ...] for compact assertions."""
        out = []
        for m in messages:
            blocks = []
            for item in m["content"]:
                if "toolUse" in item:
                    blocks.append(f"use:{item['toolUse']['toolUseId']}")
                elif "toolResult" in item:
                    blocks.append(f"result:{item['toolResult']['toolUseId']}")
                else:
                    blocks.append(f"text:{item.get('text')}")
            out.append((m["role"], blocks))
        return out

    def test_cross_turn_pairing_preserved(self):
        # A6: toolUse and its toolResult are NOT adjacent — the pair is kept,
        # and the result is moved to directly follow its call (Converse
        # rejects a result that isn't in the very next user message).
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
        self.assertEqual(self._shape(out), [
            ("assistant", ["text:let me check", "use:t1"]),
            ("user", ["result:t1"]),
            ("assistant", ["text:still working"]),
        ])

    def test_valid_history_unchanged(self):
        messages = [
            {"role": "user", "content": [{"text": "q"}]},
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "a", "name": "f", "input": {}}},
                {"toolUse": {"toolUseId": "b", "name": "g", "input": {}}},
            ]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "a", "content": [{"text": "1"}]}},
                {"toolResult": {"toolUseId": "b", "content": [{"text": "2"}]}},
            ]},
            {"role": "assistant", "content": [{"text": "done"}]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        self.assertEqual(out, messages)

    def test_result_after_interjected_user_turn_moves_up(self):
        messages = [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}}]},
            {"role": "user", "content": [{"text": "hurry up"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        self.assertEqual(self._shape(out), [
            ("assistant", ["use:t1"]),
            ("user", ["result:t1"]),
            ("user", ["text:hurry up"]),
        ])

    def test_results_split_across_turns_are_grouped_after_their_call(self):
        messages = [
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "a", "name": "f", "input": {}}},
                {"toolUse": {"toolUseId": "b", "name": "f", "input": {}}},
            ]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "a", "content": [{"text": "1"}]}}]},
            {"role": "assistant", "content": [{"text": "waiting on b"}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": "b", "content": [{"text": "2"}]}}]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        self.assertEqual(self._shape(out), [
            ("assistant", ["use:a", "use:b"]),
            ("user", ["result:a", "result:b"]),
            ("assistant", ["text:waiting on b"]),
        ])

    def test_result_before_its_call_is_an_orphan(self):
        messages = [
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "early"}]}}]},
            {"role": "assistant", "content": [
                {"text": "calling"},
                {"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}},
            ]},
        ]
        out = self.p._validate_tool_use_result_pairing(messages)
        self.assertEqual(self._shape(out), [("assistant", ["text:calling"])])

    def test_openai_parallel_calls_form_one_group(self):
        # Two tool_calls on one assistant message + their tool messages become
        # one assistant toolUse group followed by one user toolResult group.
        from app.openai_models import ChatCompletionRequest
        request = ChatCompletionRequest(model="amazon.nova-pro-v1:0", messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "calling", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "1"},
            {"role": "tool", "tool_call_id": "b", "content": "2"},
        ])
        self.p.bedrock_model_list = {}
        out = self.p._parse_messages(request)
        self.assertEqual(self._shape(out), [
            ("user", ["text:go"]),
            ("assistant", ["text:calling", "use:a", "use:b"]),
            ("user", ["result:a", "result:b"]),
        ])

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
        self.assertEqual(resp["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        # The matching delta resolves to the same ordinal.
        delta_chunk = {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"a":1}'}},
            }
        }
        resp2, _ = self.p._parse_stream_chunk(delta_chunk, "id", "m", state)
        self.assertEqual(resp2["choices"][0]["delta"]["tool_calls"][0]["index"], 0)

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
        # web_search_tool_result is converted to a plain text block (not an
        # orphan tool_result that would have no matching tool_use → API 400).
        self.assertNotIn("web_search_tool_result", types)
        self.assertNotIn("tool_result", types)
        self.assertIn("text", types)

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


class GrokConverseConstraintTests(unittest.TestCase):
    """Bedrock's xAI Grok models reject temperature/topP/stopSequences and take
    reasoning as an effort string (verified against Converse in us-west-2)."""

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = []

    def _chat_request(self, **kwargs):
        kwargs.setdefault("model", "us.xai.grok-4.6")
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            **kwargs,
        )

    def _anthropic_request(self, **kwargs):
        kwargs.setdefault("model", "global.xai.grok-4.6")
        kwargs.setdefault("max_tokens", 64)
        return AnthropicMessagesRequest(
            messages=[AnthropicMessage(role="user", content="hi")],
            **kwargs,
        )

    def test_sampling_fields_stripped_on_chat_path(self):
        args = self.p._parse_bedrock_request(
            self._chat_request(temperature=0.7, top_p=0.9, stop=["END"], max_tokens=64)
        )
        self.assertEqual(args["inferenceConfig"], {"maxTokens": 64})

    def test_sampling_fields_stripped_on_anthropic_path(self):
        args = self.p._build_bedrock_anthropic_args(
            self._anthropic_request(temperature=0.7, top_p=0.9, stop_sequences=["END"])
        )
        self.assertEqual(args["inferenceConfig"], {"maxTokens": 64})

    def test_reasoning_effort_passed_as_string(self):
        args = self.p._parse_bedrock_request(
            self._chat_request(max_tokens=4096, reasoning_effort="high")
        )
        self.assertEqual(args["additionalModelRequestFields"], {"reasoning_effort": "high"})

    def test_unsupported_effort_is_dropped(self):
        # Grok's enum is none/low/medium/high/xhigh/max; anything else would 400.
        args = self.p._parse_bedrock_request(
            self._chat_request(max_tokens=4096, reasoning_effort="turbo")
        )
        self.assertNotIn("additionalModelRequestFields", args)

    def test_minimal_effort_aliased_to_low(self):
        args = self.p._parse_bedrock_request(
            self._chat_request(max_tokens=4096, reasoning_effort="minimal")
        )
        self.assertEqual(args["additionalModelRequestFields"]["reasoning_effort"], "low")

    def test_thinking_block_becomes_effort_string(self):
        args = self.p._build_bedrock_anthropic_args(
            self._anthropic_request(
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 16000},
            )
        )
        self.assertEqual(args["additionalModelRequestFields"]["reasoning_effort"], "high")
        self.assertNotIn("thinking", args["additionalModelRequestFields"])

    def test_extra_thinking_field_becomes_effort_string(self):
        # A client passing Anthropic-style thinking through the OpenAI surface
        # must not reach Grok as a reasoning_config object.
        args = self.p._parse_bedrock_request(
            self._chat_request(
                max_tokens=32000,
                temperature=0.5,
                thinking={"type": "enabled", "budget_tokens": 500},
            )
        )
        self.assertEqual(args["additionalModelRequestFields"], {"reasoning_effort": "low"})
        self.assertNotIn("temperature", args["inferenceConfig"])

    def test_redacted_reasoning_block_is_skipped(self):
        # Grok returns reasoningContent with redactedContent only (no reasoningText).
        response = {
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"redactedContent": b"rsn_encrypted"}},
                {"text": "Hello there friend."},
            ]}},
            "usage": {"inputTokens": 35, "outputTokens": 203, "totalTokens": 238},
            "stopReason": "end_turn",
        }
        with mock.patch.object(
            BedrockProvider, "_invoke_bedrock", new=mock.AsyncMock(return_value=(response, {}))
        ):
            result = asyncio.run(self.p.chat_completion(self._chat_request(max_tokens=64)))
        self.assertEqual(result.choices[0].message.content, "Hello there friend.")
        self.assertIsNone(result.choices[0].message.reasoning_content)

    def test_non_grok_model_keeps_sampling_fields(self):
        args = self.p._parse_bedrock_request(
            self._chat_request(model="us.amazon.nova-pro-v1:0", temperature=0.7, top_p=0.9, max_tokens=64)
        )
        self.assertEqual(args["inferenceConfig"]["temperature"], 0.7)
        self.assertEqual(args["inferenceConfig"]["topP"], 0.9)


class ClaudeCapabilityScrubTests(unittest.TestCase):
    """Both Bedrock surfaces now consult app.model_capabilities, so a model's
    unsupported sampling params are stripped identically on each."""

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = []

    def _chat_request(self, model, **kwargs):
        kwargs.setdefault("max_tokens", 64)
        return ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content="hi")],
            **kwargs,
        )

    def _anthropic_request(self, model, **kwargs):
        kwargs.setdefault("max_tokens", 64)
        return AnthropicMessagesRequest(
            model=model,
            messages=[AnthropicMessage(role="user", content="hi")],
            **kwargs,
        )

    # --- native (/v1/messages -> InvokeModel) -------------------------------

    def test_native_drops_temperature_for_claude_5(self):
        # The reported Copilot failure: temperature reached Opus 5 -> 400 ->
        # a lone SSE error frame -> "Response contained no choices".
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "global.anthropic.claude-opus-5", temperature=0.1, top_p=0.9, top_k=5
            )
        )
        for param in ("temperature", "top_p", "top_k"):
            self.assertNotIn(param, native)

    def test_native_drops_sampling_params_for_claude_4_8(self):
        # 4.7 removed temperature along with top_p/top_k — forwarding it here
        # was an upstream 400, not a determinism win.
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "global.anthropic.claude-opus-4-8", temperature=0.1, top_p=0.9, top_k=5
            )
        )
        for param in ("temperature", "top_p", "top_k"):
            self.assertNotIn(param, native)

    def test_native_keeps_everything_for_older_claude(self):
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "anthropic.claude-sonnet-4-6", temperature=0.1, top_p=0.9, top_k=5
            )
        )
        self.assertEqual(native["temperature"], 0.1)
        self.assertEqual(native["top_p"], 0.9)
        self.assertEqual(native["top_k"], 5)

    def test_native_stop_sequences_survive(self):
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request("claude-opus-5", stop_sequences=["END"])
        )
        self.assertEqual(native["stop_sequences"], ["END"])

    def test_native_thinking_budget_becomes_adaptive_on_claude_5(self):
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "claude-opus-5",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            )
        )
        self.assertEqual(native["thinking"], {"type": "adaptive"})

    def test_native_thinking_budget_becomes_adaptive_on_claude_4_8(self):
        # budget_tokens was removed in 4.7, not 5.0.
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "claude-opus-4-8",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            )
        )
        self.assertEqual(native["thinking"], {"type": "adaptive"})

    def test_native_thinking_budget_preserved_on_claude_4_6(self):
        native = self.p._convert_to_anthropic_native_request(
            self._anthropic_request(
                "claude-sonnet-4-6",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            )
        )
        self.assertEqual(native["thinking"]["type"], "enabled")
        self.assertEqual(native["thinking"]["budget_tokens"], 4096)

    # --- converse (/v1/chat/completions) ------------------------------------

    def test_converse_drops_sampling_params_for_claude_5(self):
        args = self.p._parse_bedrock_request(
            self._chat_request(
                "global.anthropic.claude-opus-5", temperature=0.1, top_p=0.9
            )
        )
        self.assertEqual(args["inferenceConfig"], {"maxTokens": 64})

    def test_converse_drops_top_k_for_claude_5(self):
        args = self.p._parse_bedrock_request(
            self._chat_request("global.anthropic.claude-opus-5", top_k=5)
        )
        self.assertNotIn("top_k", args.get("additionalModelRequestFields") or {})

    def test_converse_drops_temperature_for_claude_4_8(self):
        # Same rule as the native surface now — 4.7 removed it everywhere.
        args = self.p._parse_bedrock_request(
            self._chat_request("global.anthropic.claude-opus-4-8", temperature=0.1, top_p=0.9)
        )
        self.assertEqual(args["inferenceConfig"], {"maxTokens": 64})

    def test_converse_temperature_topp_conflict_rule_still_applies(self):
        # Claude >= 4.5 (but < 4.7) reject both at once; temperature wins.
        args = self.p._parse_bedrock_request(
            self._chat_request("anthropic.claude-sonnet-4-5", temperature=0.1, top_p=0.9)
        )
        self.assertEqual(args["inferenceConfig"]["temperature"], 0.1)
        self.assertNotIn("topP", args["inferenceConfig"])

    def test_converse_omits_reasoning_config_for_claude_5(self):
        # budget_tokens is rejected on adaptive-thinking models; thinking is on
        # by default there, so omitting the field preserves the behaviour.
        args = self.p._parse_bedrock_request(
            self._chat_request(
                "global.anthropic.claude-opus-5", max_tokens=32000, reasoning_effort="high"
            )
        )
        self.assertNotIn("additionalModelRequestFields", args)

    def test_converse_omits_reasoning_config_for_claude_4_8(self):
        # 4.7/4.8 are adaptive-thinking too, so no budget_tokens can be emitted.
        args = self.p._parse_bedrock_request(
            self._chat_request(
                "global.anthropic.claude-opus-4-8", max_tokens=32000, reasoning_effort="high"
            )
        )
        self.assertNotIn("additionalModelRequestFields", args)

    def test_converse_emits_reasoning_config_for_claude_4_6(self):
        # 4.6 still takes an explicit budget.
        args = self.p._parse_bedrock_request(
            self._chat_request(
                "anthropic.claude-sonnet-4-6", max_tokens=32000, reasoning_effort="high"
            )
        )
        self.assertEqual(
            args["additionalModelRequestFields"]["reasoning_config"]["type"], "enabled"
        )

    def test_converse_extra_body_thinking_is_normalized(self):
        # A client-supplied thinking config in the extra body used to bypass
        # normalize_thinking entirely and reach Bedrock with budget_tokens.
        args = self.p._parse_bedrock_request(
            self._chat_request(
                "global.anthropic.claude-opus-5",
                max_tokens=32000,
                thinking={"type": "enabled", "budget_tokens": 4096},
            )
        )
        self.assertEqual(
            args["additionalModelRequestFields"]["thinking"], {"type": "adaptive"}
        )


def _summary(model_id, inference=("ON_DEMAND",), status="ACTIVE"):
    return {
        "modelId": model_id,
        "modelLifecycle": {"status": status},
        "inferenceTypesSupported": list(inference),
        "inputModalities": ["TEXT"],
    }


class ModelListRefreshTests(unittest.IsolatedAsyncioTestCase):
    """A failed model listing must raise, not pass a fallback/partial list off as success."""

    def setUp(self):
        import threading
        p = _provider()
        p.aws_region = "us-west-2"
        p.enable_cross_region = True
        p.enable_app_profiles = False
        p.default_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        p.bedrock_model_list = {}
        p._model_list_refresh_lock = threading.Lock()
        p.full_provider_name = "bedrock:primary"
        p.bedrock_client = mock.Mock()
        p.bedrock_client.list_foundation_models.return_value = {
            "modelSummaries": [_summary("anthropic.claude-x"), _summary("amazon.nova-lite")]
        }
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {"inferenceProfileSummaries": [
                {"inferenceProfileId": "us.anthropic.claude-x"},
                {"inferenceProfileId": "global.anthropic.claude-x"},
            ]}
        ]
        p.bedrock_client.get_paginator.return_value = paginator
        self.p = p
        self.paginator = paginator

    async def _ids(self):
        return {m.id.split("/", 1)[-1] for m in await self.p.get_available_models()}

    async def test_happy_path_lists_on_demand_and_profiles(self):
        self.assertEqual(
            await self._ids(),
            {"anthropic.claude-x", "amazon.nova-lite", "us.anthropic.claude-x", "global.anthropic.claude-x"},
        )

    async def test_foundation_listing_failure_raises(self):
        from botocore.exceptions import EndpointConnectionError
        self.p.bedrock_client.list_foundation_models.side_effect = EndpointConnectionError(
            endpoint_url="https://bedrock.us-west-2.amazonaws.com/foundation-models"
        )
        with self.assertRaises(EndpointConnectionError):
            await self.p.get_available_models()
        # No fake default-model fallback is installed.
        self.assertEqual(self.p.bedrock_model_list, {})

    async def test_failed_refresh_keeps_last_good_list(self):
        await self.p.get_available_models()
        good = dict(self.p.bedrock_model_list)
        fetched_at = self.p._model_list_fetched_at

        self.p.bedrock_client.list_foundation_models.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            await self.p.refresh_models()
        self.assertEqual(self.p.bedrock_model_list, good)
        self.assertEqual(self.p._model_list_fetched_at, fetched_at)

    async def test_profile_listing_failure_raises(self):
        from botocore.exceptions import EndpointConnectionError
        self.paginator.paginate.side_effect = EndpointConnectionError(
            endpoint_url="https://bedrock.us-west-2.amazonaws.com/inference-profiles"
        )
        with self.assertRaises(EndpointConnectionError):
            await self.p.get_available_models()

    async def test_profile_access_denied_falls_back_to_on_demand(self):
        from botocore.exceptions import ClientError
        self.paginator.paginate.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "nope"}}, "ListInferenceProfiles"
        )
        self.assertEqual(await self._ids(), {"anthropic.claude-x", "amazon.nova-lite"})

    async def test_no_usable_models_raises(self):
        self.p.bedrock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                _summary("old.model", status="EOL"),
                _summary("provisioned.only", inference=("PROVISIONED",)),
            ]
        }
        self.paginator.paginate.return_value = [{"inferenceProfileSummaries": []}]
        with self.assertRaises(RuntimeError):
            await self.p.get_available_models()

    async def test_application_profile_records_its_base_model(self):
        self.p.enable_app_profiles = True
        arn = "arn:aws:bedrock:us-west-2:123:application-inference-profile/abc"

        def _paginate(**kwargs):
            if kwargs.get("typeEquals") == "APPLICATION":
                return [{"inferenceProfileSummaries": [{
                    "inferenceProfileArn": arn,
                    "models": [{"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-x"}],
                }]}]
            return [{"inferenceProfileSummaries": []}]

        self.paginator.paginate.side_effect = _paginate
        await self.p.get_available_models()
        self.assertEqual(self.p.bedrock_model_list[arn]["base_model_id"], "anthropic.claude-x")
        self.assertEqual(self.p._capability_model_id(arn), "anthropic.claude-x")
        # The routed name keeps the whole ARN (only the provider prefix is cut).
        self.assertEqual(self.p.get_model_id(f"bedrock:primary/{arn}"), arn)


class ClientConfigTests(unittest.TestCase):
    """Data-plane vs control-plane boto3 budgets."""

    def test_runtime_and_control_plane_budgets(self):
        p = BedrockProvider({"region": "us-east-1", "access_key_id": "AKIAX", "secret_access_key": "s"})
        runtime, control = p.bedrock_runtime.meta.config, p.bedrock_client.meta.config
        self.assertEqual(runtime.read_timeout, 900)
        self.assertEqual(runtime.connect_timeout, 10)
        self.assertEqual(runtime.retries, {"mode": "adaptive", "total_max_attempts": 3})
        self.assertEqual(control.read_timeout, 30)
        self.assertEqual(control.connect_timeout, 5)
        self.assertEqual(control.retries, {"mode": "standard", "total_max_attempts": 3})

    def test_budgets_follow_env(self):
        env = {"BEDROCK_READ_TIMEOUT_SECONDS": "120", "BEDROCK_MAX_ATTEMPTS": "5"}
        with mock.patch.dict("os.environ", env):
            p = BedrockProvider({"region": "us-east-1", "access_key_id": "AKIAX", "secret_access_key": "s"})
        self.assertEqual(p.bedrock_runtime.meta.config.read_timeout, 120)
        self.assertEqual(p.bedrock_runtime.meta.config.retries["total_max_attempts"], 5)


class ModelDiscoverySharingTests(unittest.TestCase):
    def _bedrock(self, region="us-east-1", key="AKIAX"):
        return BedrockProvider({"region": region, "access_key_id": key, "secret_access_key": "s"})

    def test_adopts_list_from_identically_configured_instance(self):
        live, discovery = self._bedrock(), self._bedrock()
        discovery.bedrock_model_list = {"m": {"modalities": ["TEXT", "IMAGE"]}}
        discovery._model_list_fetched_at = 123.0
        self.assertTrue(live.adopt_model_discovery(discovery))
        self.assertTrue(live._is_supported_modality("m", "IMAGE"))
        self.assertEqual(live._model_list_fetched_at, 123.0)

    def test_does_not_adopt_from_other_account_or_region(self):
        discovery = self._bedrock(region="eu-west-1")
        discovery.bedrock_model_list = {"m": {"modalities": ["TEXT"]}}
        discovery._model_list_fetched_at = 1.0
        live = self._bedrock()
        self.assertFalse(live.adopt_model_discovery(discovery))
        self.assertEqual(live.bedrock_model_list, {})
        other_key = self._bedrock(key="AKIAOTHER")
        self.assertFalse(other_key.adopt_model_discovery(self._with_list(self._bedrock())))

    def test_does_not_replace_a_newer_list(self):
        live = self._with_list(self._bedrock(), fetched_at=50.0)
        older = self._with_list(self._bedrock(), fetched_at=10.0, models={"old": {}})
        self.assertFalse(live.adopt_model_discovery(older))
        self.assertIn("m", live.bedrock_model_list)

    @staticmethod
    def _with_list(p, fetched_at=1.0, models=None):
        p.bedrock_model_list = models or {"m": {"modalities": ["TEXT"]}}
        p._model_list_fetched_at = fetched_at
        return p

    def test_registry_refresh_carries_lists_over(self):
        from app.providers.provider_manager import ProviderManager
        old = self._with_list(self._bedrock())
        new = self._bedrock()
        ProviderManager._carry_over_model_discovery({"bedrock:a": old}, {"bedrock:a": new, "x": object()})
        self.assertEqual(new.bedrock_model_list, old.bedrock_model_list)

    def test_sync_shares_discovery_with_live_instance(self):
        from app.providers import auto_sync
        live = self._bedrock()
        discovery = self._with_list(self._bedrock())
        with mock.patch.object(auto_sync.provider_manager, "providers", {"bedrock:a": live}):
            auto_sync._share_discovery_with_live_instance("bedrock:a", discovery)
        self.assertEqual(live.bedrock_model_list, discovery.bedrock_model_list)


class ReasoningGatingTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = {}

    def _args(self, model, **kwargs):
        kwargs.setdefault("max_tokens", 4000)
        return self.p._parse_bedrock_request(ChatCompletionRequest(
            model=model, messages=[ChatMessage(role="user", content="hi")], **kwargs,
        ))

    def test_budget_reasoning_for_claude_3_7_and_later(self):
        for model in ("anthropic.claude-3-7-sonnet-20250219-v1:0", "us.anthropic.claude-sonnet-4-20250514-v1:0"):
            amrf = self._args(model, reasoning_effort="high").get("additionalModelRequestFields", {})
            self.assertEqual(amrf.get("reasoning_config", {}).get("type"), "enabled", model)

    def test_no_reasoning_config_for_models_without_extended_thinking(self):
        for model in (
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "meta.llama3-70b-instruct-v1:0",
            "amazon.nova-pro-v1:0",
            "mistral.mistral-large-2407-v1:0",
        ):
            with self.assertLogs("app.providers.bedrock_provider", level="WARNING"):
                args = self._args(model, reasoning_effort="high", temperature=0.5)
            self.assertNotIn("reasoning_config", args.get("additionalModelRequestFields", {}), model)
            # Nothing was enabled, so the sampling params are left alone.
            self.assertEqual(args["inferenceConfig"].get("temperature"), 0.5, model)

    def test_application_profile_uses_its_base_model(self):
        arn = "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc"
        self.p.bedrock_model_list = {arn: {"modalities": ["TEXT"], "base_model_id": "anthropic.claude-sonnet-4-20250514-v1:0"}}
        amrf = self._args(arn, reasoning_effort="medium").get("additionalModelRequestFields", {})
        self.assertIn("reasoning_config", amrf)


class MaxTokensMappingTests(unittest.TestCase):
    """OpenAI's max_completion_tokens must reach Converse's maxTokens even
    without reasoning_effort (it used to silently fall back to 2048)."""

    MODEL = "us.openai.gpt-oss-120b-1:0"

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = {}

    def _max_tokens(self, **kwargs):
        args = self.p._parse_bedrock_request(ChatCompletionRequest(
            model=self.MODEL, messages=[ChatMessage(role="user", content="hi")], **kwargs,
        ))
        return args["inferenceConfig"]["maxTokens"]

    def test_max_completion_tokens_alone(self):
        self.assertEqual(self._max_tokens(max_completion_tokens=128000), 128000)

    def test_max_completion_tokens_wins_over_max_tokens(self):
        self.assertEqual(self._max_tokens(max_completion_tokens=5000, max_tokens=100), 5000)

    def test_max_tokens_alone(self):
        self.assertEqual(self._max_tokens(max_tokens=777), 777)

    def test_default_when_neither_sent(self):
        self.assertEqual(self._max_tokens(), 2048)


class OpenAIToolChoiceTests(unittest.TestCase):
    TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = {}

    def _args(self, messages=None, **kwargs):
        return self.p._parse_bedrock_request(ChatCompletionRequest(
            model="amazon.nova-pro-v1:0",
            messages=messages or [{"role": "user", "content": "hi"}],
            **kwargs,
        ), {})

    def test_none_withholds_tools(self):
        args = self._args(tools=[self.TOOL], tool_choice="none")
        self.assertNotIn("toolConfig", args)

    def test_none_with_tool_history_keeps_only_placeholder(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
        args = self._args(messages=history, tools=[self.TOOL], tool_choice="none")
        self.assertEqual(args["toolConfig"]["tools"], [BedrockProvider._PLACEHOLDER_TOOL_SPEC])
        self.assertNotIn("toolChoice", args["toolConfig"])

    def test_auto_and_required(self):
        self.assertEqual(self._args(tools=[self.TOOL], tool_choice="auto")["toolConfig"]["toolChoice"], {"auto": {}})
        self.assertEqual(self._args(tools=[self.TOOL], tool_choice="required")["toolConfig"]["toolChoice"], {"any": {}})

    def test_named_tool_uses_truncated_name(self):
        long_name = "x" * 80
        tool = {"type": "function", "function": {"name": long_name, "parameters": {"type": "object"}}}
        args = self._args(tools=[tool], tool_choice={"type": "function", "function": {"name": long_name}})
        spec_name = args["toolConfig"]["tools"][0]["toolSpec"]["name"]
        self.assertEqual(args["toolConfig"]["toolChoice"], {"tool": {"name": spec_name}})
        self.assertLessEqual(len(spec_name), 64)


class NonStreamReasoningTests(unittest.TestCase):
    def test_multiple_reasoning_blocks_are_all_kept(self):
        p = _provider()
        response = {
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "first", "signature": "s1"}}},
                {"text": "Hello"},
                {"reasoningContent": {"reasoningText": {"text": "second", "signature": "s2"}}},
                {"text": " world"},
            ]}},
            "usage": {"inputTokens": 1, "outputTokens": 2, "totalTokens": 3},
            "stopReason": "end_turn",
        }
        request = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])
        with mock.patch.object(BedrockProvider, "_invoke_bedrock", new=mock.AsyncMock(return_value=(response, {}))):
            result = asyncio.run(p.chat_completion(request))
        self.assertEqual(result.choices[0].message.reasoning_content, "first\n\nsecond")
        self.assertEqual(result.choices[0].message.content, "Hello world")


class AnthropicAdapterTests(unittest.TestCase):
    """Converse adapter (/v1/messages for non-Claude models)."""

    PNG = base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = {}
        self.p.full_provider_name = "bedrock:test"

    def _request(self, **kwargs):
        kwargs.setdefault("model", "bedrock:test/amazon.nova-pro-v1:0")
        kwargs.setdefault("max_tokens", 64)
        kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
        return AnthropicMessagesRequest.model_validate(kwargs)

    def test_system_role_messages_become_system_prompt(self):
        args = self.p._build_bedrock_anthropic_args(self._request(
            system="top",
            messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
        ))
        self.assertEqual(args["system"], [{"text": "top"}, {"text": "be terse"}])
        self.assertEqual([m["role"] for m in args["messages"]], ["user"])

    def test_tool_result_image_stays_an_image(self):
        args = self.p._build_bedrock_anthropic_args(self._request(messages=[
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "shot", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "text", "text": "screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": self.PNG}},
            ]}]},
        ]))
        result = args["messages"][2]["content"][0]["toolResult"]["content"]
        self.assertEqual(result[0], {"text": "screenshot:"})
        self.assertEqual(result[1]["image"]["format"], "png")  # bytes win over the declared type
        self.assertTrue(result[1]["image"]["source"]["bytes"].startswith(b"\x89PNG"))

    def test_tool_choice_none_withholds_tools(self):
        tool = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
        args = self.p._build_bedrock_anthropic_args(self._request(tools=[tool], tool_choice={"type": "none"}))
        self.assertNotIn("toolConfig", args)

    def test_unknown_tool_choice_is_rejected(self):
        tool = {"name": "f", "description": "d", "input_schema": {"type": "object"}}
        with self.assertRaises(ValueError):
            self.p._build_bedrock_anthropic_args(self._request(tools=[tool], tool_choice={"type": "sometimes"}))
        with self.assertRaises(ValueError):
            self.p._build_bedrock_anthropic_args(self._request(tools=[tool], tool_choice={"type": "tool"}))

    def test_thinking_not_forwarded_to_model_without_reasoning_and_reported(self):
        request = self._request(
            model="bedrock:test/meta.llama3-70b-instruct-v1:0",
            thinking={"type": "enabled", "budget_tokens": 2000},
            metadata={"user_id": "u1"},
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            tools=[{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
        )
        args = self.p._build_bedrock_anthropic_args(request)
        self.assertNotIn("thinking", args.get("additionalModelRequestFields", {}))
        metadata = self.p.get_anthropic_request_metadata(request)
        self.assertEqual(metadata.mode, "adapter")
        for field in ("thinking", "metadata", "tool_choice.disable_parallel_tool_use"):
            self.assertIn(field, metadata.dropped_fields)

    def test_disabled_thinking_enables_nothing(self):
        args = self.p._build_bedrock_anthropic_args(self._request(
            model="bedrock:test/us.amazon.nova-2-lite-v1:0",
            thinking={"type": "disabled"},
        ))
        self.assertNotIn("reasoningConfig", args.get("additionalModelRequestFields", {}))

    def test_claude_application_profile_thinking_is_normalized(self):
        arn = "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc"
        self.p.bedrock_model_list = {arn: {"modalities": ["TEXT"], "base_model_id": "anthropic.claude-opus-4-8-v1:0"}}
        request = self._request(model=f"bedrock:test/{arn}", thinking={"type": "enabled", "budget_tokens": 2000})
        args = self.p._build_bedrock_anthropic_args(request)
        self.assertEqual(args["additionalModelRequestFields"]["thinking"], {"type": "adaptive"})
        self.assertIn("thinking.budget_tokens", self.p.get_anthropic_request_metadata(request).dropped_fields)


class _FakeEventStream:
    """Stands in for a botocore EventStream: yields events, then ends or raises."""

    def __init__(self, events, error=None):
        self.events, self.error, self.closed = events, error, False

    def __iter__(self):
        yield from self.events
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class AnthropicConverseStreamTerminalTests(unittest.IsolatedAsyncioTestCase):
    """A truncated upstream response must not be reported as a complete message."""

    START = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
    ]
    STOP = [
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    METADATA = [{"metadata": {"usage": {"inputTokens": 3, "outputTokens": 2}}}]

    def setUp(self):
        self.p = _provider()
        self.p.bedrock_model_list = {}
        self.p.full_provider_name = "bedrock:test"
        self.p._converse_semaphore = None
        self.p.bedrock_runtime = mock.Mock()
        self.request = AnthropicMessagesRequest.model_validate({
            "model": "bedrock:test/amazon.nova-pro-v1:0", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}], "stream": True,
        })

    async def _events(self, stream):
        self.p.bedrock_runtime.converse_stream.return_value = {"stream": stream}
        out = []
        async for sse in self.p.anthropic_messages_stream(self.request):
            event = sse.split("\n", 1)[0].removeprefix("event: ")
            payload = json.loads(sse.split("data: ", 1)[1])
            out.append((event, payload))
        return out

    async def test_complete_stream(self):
        events = await self._events(_FakeEventStream(self.START + self.STOP + self.METADATA))
        self.assertEqual([e for e, _ in events][-2:], ["message_delta", "message_stop"])
        self.assertEqual(events[-2][1]["usage"]["output_tokens"], 2)

    async def test_error_before_message_stop_is_an_error_not_end_turn(self):
        from botocore.exceptions import EventStreamError
        err = EventStreamError(
            {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "ConverseStream"
        )
        stream = _FakeEventStream(self.START, error=err)
        events = await self._events(stream)
        names = [e for e, _ in events]
        self.assertNotIn("message_stop", names)
        self.assertNotIn("message_delta", names)
        self.assertEqual(names[-1], "error")
        self.assertEqual(events[-1][1]["error"]["type"], "rate_limit_error")
        self.assertTrue(stream.closed)

    async def test_stream_ending_before_message_stop_is_an_error(self):
        events = await self._events(_FakeEventStream(self.START))
        names = [e for e, _ in events]
        self.assertNotIn("message_stop", names)
        self.assertEqual(names[-1], "error")

    async def test_error_after_message_stop_still_completes(self):
        # messageStop arrived, so the content is complete; only usage was lost.
        events = await self._events(_FakeEventStream(self.START + self.STOP, error=RuntimeError("reset")))
        names = [e for e, _ in events]
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        self.assertNotIn("error", names)
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")

    async def test_open_error_maps_to_http_status(self):
        from botocore.exceptions import ClientError
        from fastapi import HTTPException
        self.p.bedrock_runtime.converse_stream.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "ConverseStream"
        )
        with self.assertRaises(HTTPException) as ctx:
            async for _ in self.p.anthropic_messages_stream(self.request):
                pass
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(self.p._get_converse_semaphore()._value, 15)


class ConverseStreamReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_permit_is_held_until_the_reader_thread_exits(self):
        import threading
        release_read = threading.Event()

        class _BlockingStream:
            closed = False

            def __iter__(self):
                yield {"messageStart": {"role": "assistant"}}
                release_read.wait(5)  # a read the socket shutdown can't reach
                yield {"messageStop": {"stopReason": "end_turn"}}

            def close(self):
                self.closed = True

        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        stream = _BlockingStream()
        from app.providers.bedrock_provider import _ConverseStreamReader
        reader = _ConverseStreamReader(lambda: stream, on_exit=semaphore.release)
        reader.start()
        await reader.wait_opened()
        self.assertEqual(await reader.next_event(), {"messageStart": {"role": "assistant"}})

        with mock.patch("app.providers.bedrock_provider._STREAM_CLOSE_WAIT_SECONDS", 0.1):
            await reader.close()
        # The thread is still in its read: the permit must still be held.
        self.assertTrue(semaphore.locked())
        release_read.set()
        await asyncio.wait_for(semaphore.acquire(), 2)
        self.assertTrue(stream.closed)  # closed by the reader thread itself

    async def test_backpressure_stops_reading_ahead(self):
        reads = []

        def _events():
            for i in range(100):
                reads.append(i)
                yield {"contentBlockDelta": {"delta": {"text": str(i)}}}

        from app.providers.bedrock_provider import _ConverseStreamReader
        reader = _ConverseStreamReader(lambda: _events(), max_buffered=4)
        reader.start()
        await reader.wait_opened()
        await asyncio.sleep(0.3)
        self.assertLessEqual(len(reads), 6)  # 4 buffered + the read waiting for a slot
        await reader.close()


class EmbeddingInputTests(unittest.TestCase):
    def test_token_id_arrays_are_rejected(self):
        from app.openai_models import EmbeddingRequest
        p = _provider()
        p.bedrock_runtime = mock.Mock()
        for bad in ([1, 2, 3], [[1, 2], [3]], []):
            with self.assertRaises(ProviderHTTPError) as ctx:
                asyncio.run(p.embeddings(EmbeddingRequest(model="bedrock:x/cohere.embed-english-v3", input=bad)))
            self.assertEqual(ctx.exception.status_code, 400)
        p.bedrock_runtime.invoke_model.assert_not_called()


class ImageInputTests(unittest.TestCase):
    def setUp(self):
        self.p = _provider()

    def test_base64_images_share_the_download_size_cap(self):
        oversized = "data:image/png;base64," + "A" * (BedrockProvider._IMAGE_MAX_BYTES * 4 // 3 + 16)
        with self.assertRaises(ValueError):
            self.p._parse_image_sync(oversized)
        with self.assertRaises(ValueError):
            self.p._convert_anthropic_image_block(
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": oversized[22:]}}
            )

    def test_formats_are_normalized_or_rejected(self):
        from app.providers.bedrock_provider import _converse_image_format
        self.assertEqual(_converse_image_format(b"", "image/jpg"), "jpeg")
        self.assertEqual(_converse_image_format(b"", "image/PNG; charset=binary"), "png")
        self.assertEqual(_converse_image_format(b"\xff\xd8\xff\xe0", "image/png"), "jpeg")
        for bad in ("image/svg+xml", "application/octet-stream", None):
            with self.assertRaises(ValueError):
                _converse_image_format(b"<svg/>", bad)

    def test_private_addresses_are_refused(self):
        for url in ("http://169.254.169.254/latest", "http://127.0.0.1/x.png", "http://[::1]/x.png"):
            with self.assertRaises(ValueError):
                self.p._parse_image_sync(url)

    def test_fetch_connects_to_the_validated_address(self):
        # One DNS lookup; the connection goes to the address that was checked,
        # so a rebinding resolver can't swap in an internal one afterwards.
        import socket
        lookups = []

        def _getaddrinfo(host, *a, **k):
            lookups.append(host)
            ip = "93.184.215.14" if len(lookups) == 1 else "169.254.169.254"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]

        pools = []

        class _Pool:
            def __init__(self, host, port, **kwargs):
                pools.append((host, port))

            def urlopen(self, *a, **k):
                raise ValueError("stop")

            def close(self):
                pass

        with mock.patch("socket.getaddrinfo", _getaddrinfo), \
             mock.patch("urllib3.HTTPConnectionPool", _Pool):
            with self.assertRaises(ValueError):
                self.p._parse_image_sync("http://rebind.example/a.png")
        self.assertEqual(lookups, ["rebind.example"])
        self.assertEqual(pools, [("93.184.215.14", 80)])


class ChainCredentialTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_chain_credentials_resolved_off_loop_and_refreshed(self):
        import threading
        p = _provider()
        p.aws_region = "us-east-1"
        p._anthropic_client = SimpleNamespace(aws_access_key=None, aws_secret_key=None, aws_session_token=None)
        p._anthropic_uses_default_chain = True
        p._chain_credentials = None
        p._chain_credentials_lock = None

        loop_thread = threading.get_ident()
        state = {"refresh": False, "threads": []}

        class _Creds:
            def refresh_needed(self):
                return state["refresh"]

            def get_frozen_credentials(self):
                state["threads"].append(threading.get_ident())
                n = len(state["threads"])
                return SimpleNamespace(access_key=f"AK{n}", secret_key=f"SK{n}", token=f"T{n}")

        session = mock.Mock()
        session.get_credentials.return_value = _Creds()
        with mock.patch("boto3.session.Session", return_value=session):
            await p._ensure_anthropic_client()
            self.assertEqual(p._anthropic_client.aws_access_key, "AK1")
            await p._ensure_anthropic_client()  # fresh: no second resolution
            self.assertEqual(len(state["threads"]), 1)
            state["refresh"] = True
            await p._ensure_anthropic_client()
        self.assertEqual(p._anthropic_client.aws_session_token, "T2")
        self.assertNotIn(loop_thread, state["threads"])

    async def test_explicit_keys_untouched(self):
        p = _provider()
        p._anthropic_client = SimpleNamespace(aws_access_key="AK", aws_secret_key="SK", aws_session_token=None)
        p._anthropic_uses_default_chain = False
        with mock.patch("boto3.session.Session") as session:
            await p._ensure_anthropic_client()
        session.assert_not_called()


class ProviderBuildOffLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_bedrock_provider_is_built_on_a_worker_thread(self):
        import threading
        from app.providers.provider_manager import ProviderManager
        threads = []
        manager = ProviderManager.__new__(ProviderManager)
        manager._build_provider = lambda cred, check_urls=True: threads.append(threading.get_ident()) or "built"
        cred = SimpleNamespace(provider_type="bedrock", user_id=None, owner_user_id=None)
        with mock.patch("app.providers.provider_manager._row_owner", return_value=None):
            self.assertEqual(await manager.build_provider(cred), "built")
            cred.provider_type = "custom"
            await manager.build_provider(cred)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertEqual(threads[1], threading.get_ident())


if __name__ == "__main__":
    unittest.main()

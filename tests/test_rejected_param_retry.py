"""Optional request params an upstream refuses are dropped and the call retried.

Claude Code's auto-mode classifier always sends ``stop_sequences: ["</block>"]``.
A model that refuses the field made every classifier call fail, so auto mode
blocked the action ("<model> is temporarily unavailable, so auto mode cannot
determine the safety of <tool>"). Covers the Bedrock Converse path ("This
model doesn't support the stopSequences field") and the native /v1/messages
path that custom providers use for their own Anthropic endpoint.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from botocore.exceptions import ClientError

from app.anthropic_models import AnthropicMessagesRequest
from app.providers import anthropic_adapter
from app.providers.anthropic_adapter import known_rejected_params, unsupported_param
from app.providers.base import ProviderHTTPError
from app.providers.bedrock_provider import BedrockProvider
from app.providers.custom_providers import create_custom_provider

KIMI = "global.moonshotai.kimi-k3.1"
BEDROCK_STOP_MSG = "This model doesn't support the stopSequences field. Remove stopSequences and try again."
CLASSIFIER_BODY = {
    "max_tokens": 64,
    "temperature": 0,
    "stop_sequences": ["</block>"],
    "system": "You are a security monitor.",
    "messages": [{"role": "user", "content": "<transcript>...</transcript>"}],
}
CONVERSE_OK = {
    "output": {"message": {"role": "assistant", "content": [{"text": "<block>no"}]}},
    "stopReason": "stop_sequence",
    "usage": {"inputTokens": 10, "outputTokens": 3},
}


def _validation_error(message, operation="Converse"):
    return ClientError({"Error": {"Code": "ValidationException", "Message": message}}, operation)


def _bedrock_provider():
    p = BedrockProvider.__new__(BedrockProvider)
    p.debug = False
    p.bedrock_model_list = {}
    p.full_provider_name = "bedrock:test"
    p._converse_semaphore = None
    p.bedrock_runtime = mock.Mock()
    return p


def _request(model, **extra):
    return AnthropicMessagesRequest.model_validate({"model": model, **CLASSIFIER_BODY, **extra})


class _FakeEventStream:
    def __init__(self, events):
        self.events, self.closed = events, False

    def __iter__(self):
        yield from self.events

    def close(self):
        self.closed = True


class _CacheReset(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        anthropic_adapter._rejected_params.clear()
        self.addCleanup(anthropic_adapter._rejected_params.clear)


class UnsupportedParamDetectionTests(unittest.TestCase):
    def _detect(self, message, sent, candidates):
        return unsupported_param(400, message, {name: 1 for name in sent}, candidates)

    def test_bedrock_field_messages(self):
        converse = ("stopsequences", "temperature", "topp")
        self.assertEqual(self._detect(BEDROCK_STOP_MSG, converse, converse), "stopsequences")
        self.assertEqual(
            self._detect("This model doesn't support the temperature field.", converse, converse),
            "temperature",
        )
        self.assertEqual(
            self._detect("This model doesn't support the topP field. Remove topP and try again.", converse, converse),
            "topp",
        )

    def test_value_errors_are_not_rejections(self):
        converse = ("stopsequences", "temperature", "topp")
        for message in (
            "1 validation error detected: Value at 'inferenceConfig.stopSequences' failed to satisfy constraint",
            "temperature must be between 0 and 1",
            "The model returned the following errors: roles must alternate between user and assistant",
        ):
            with self.subTest(message=message):
                self.assertIsNone(self._detect(message, converse, converse))

    def test_unsent_param_is_never_blamed(self):
        self.assertIsNone(self._detect(BEDROCK_STOP_MSG, ("temperature",), ("stopsequences", "temperature")))

    def test_ambiguous_thinking_needs_a_parameter_reading(self):
        candidates = anthropic_adapter.RETRYABLE_PARAMS_ANTHROPIC
        self.assertEqual(self._detect("thinking is not supported for this model", candidates, candidates), "thinking")
        self.assertIsNone(self._detect("thinking blocks are not supported in this position", candidates, candidates))
        self.assertIsNone(
            self._detect("Extended thinking is not supported when tool_choice forces tool use", candidates, candidates)
        )


class BedrockConverseRetryTests(_CacheReset):
    async def test_rejected_stop_sequences_dropped_and_remembered(self):
        p = _bedrock_provider()
        sent = []
        outcomes = iter([_validation_error(BEDROCK_STOP_MSG), CONVERSE_OK])

        def converse(**args):
            # The retry reuses (and edits) the same args dict, so copy it now.
            sent.append(dict(args["inferenceConfig"]))
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        p.bedrock_runtime.converse.side_effect = converse

        result = await p.anthropic_messages(_request(f"bedrock:test/{KIMI}"))

        self.assertEqual(result["content"], [{"type": "text", "text": "<block>no"}])
        first, second = sent
        self.assertEqual(first["stopSequences"], ["</block>"])
        self.assertNotIn("stopSequences", second)
        self.assertEqual(second["maxTokens"], 64)
        self.assertEqual(known_rejected_params(p, KIMI, "converse"), {"stopSequences"})

        # The next request skips the field instead of paying for another 400.
        p.bedrock_runtime.converse.reset_mock(side_effect=True)
        p.bedrock_runtime.converse.return_value = CONVERSE_OK
        await p.anthropic_messages(_request(f"bedrock:test/{KIMI}"))
        self.assertEqual(p.bedrock_runtime.converse.call_count, 1)
        self.assertNotIn("stopSequences", p.bedrock_runtime.converse.call_args.kwargs["inferenceConfig"])
        self.assertEqual(p._get_converse_semaphore()._value, 15)

    async def test_two_rejected_fields_in_turn(self):
        p = _bedrock_provider()
        p.bedrock_runtime.converse.side_effect = [
            _validation_error(BEDROCK_STOP_MSG),
            _validation_error("This model doesn't support the temperature field. Remove temperature and try again."),
            CONVERSE_OK,
        ]
        await p.anthropic_messages(_request(f"bedrock:test/{KIMI}"))
        last = p.bedrock_runtime.converse.call_args.kwargs["inferenceConfig"]
        self.assertEqual(set(last), {"maxTokens"})
        self.assertEqual(known_rejected_params(p, KIMI, "converse"), {"stopSequences", "temperature"})

    async def test_unrelated_validation_error_is_not_retried(self):
        p = _bedrock_provider()
        p.bedrock_runtime.converse.side_effect = _validation_error("Malformed input request: messages roles must alternate")
        with self.assertRaises(ValueError):
            await p.anthropic_messages(_request(f"bedrock:test/{KIMI}"))
        self.assertEqual(p.bedrock_runtime.converse.call_count, 1)
        self.assertEqual(known_rejected_params(p, KIMI, "converse"), set())

    async def test_other_models_unaffected(self):
        p = _bedrock_provider()
        p.bedrock_runtime.converse.side_effect = [_validation_error(BEDROCK_STOP_MSG), CONVERSE_OK]
        await p.anthropic_messages(_request(f"bedrock:test/{KIMI}"))
        p.bedrock_runtime.converse.reset_mock(side_effect=True)
        p.bedrock_runtime.converse.return_value = CONVERSE_OK
        await p.anthropic_messages(_request("bedrock:test/amazon.nova-pro-v1:0"))
        self.assertEqual(
            p.bedrock_runtime.converse.call_args.kwargs["inferenceConfig"]["stopSequences"], ["</block>"]
        )

    async def test_stream_open_retried_without_field(self):
        p = _bedrock_provider()
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "<block>no"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "stop_sequence"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 3}}},
        ]
        p.bedrock_runtime.converse_stream.side_effect = [
            _validation_error(BEDROCK_STOP_MSG, "ConverseStream"),
            {"stream": _FakeEventStream(events)},
        ]

        names = []
        async for sse in p.anthropic_messages_stream(_request(f"bedrock:test/{KIMI}", stream=True)):
            names.append(sse.split("\n", 1)[0].removeprefix("event: "))

        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertNotIn("error", names)
        second = p.bedrock_runtime.converse_stream.call_args_list[1].kwargs
        self.assertNotIn("stopSequences", second["inferenceConfig"])
        self.assertEqual(p._get_converse_semaphore()._value, 15)

    async def test_gives_up_after_the_retry_budget(self):
        p = _bedrock_provider()
        calls = []

        async def call(args):
            calls.append(dict(args["inferenceConfig"]))
            raise _validation_error(BEDROCK_STOP_MSG)

        # Every attempt names a field that is still present: refill it so the
        # loop cannot terminate by running out of fields.
        original = p._drop_rejected_converse_field

        def drop_and_refill(error, args):
            dropped = original(error, args)
            args["inferenceConfig"]["stopSequences"] = ["</block>"]
            return dropped

        p._drop_rejected_converse_field = drop_and_refill
        with self.assertRaises(ClientError):
            await p._call_converse(
                {"modelId": KIMI, "inferenceConfig": {"maxTokens": 8, "stopSequences": ["x"]}}, call
            )
        self.assertEqual(len(calls), anthropic_adapter.MAX_PARAM_RETRIES + 1)


def _native_custom_provider():
    provider = create_custom_provider({
        "provider_type": "custom",
        "provider_name": "lmstudio",
        "name": "t",
        "base_url": "http://upstream.test/v1",
        "api_key": "k",
        "supported_apis": ["openai", "anthropic"],
    })
    provider._anthropic_client = mock.MagicMock()
    return provider


def _bad_request(message, status=400, code=None):
    import anthropic
    error = {"type": "invalid_request_error", "message": message}
    if code:
        error["code"] = code
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    cls = anthropic.BadRequestError if status == 400 else anthropic.UnprocessableEntityError
    return cls(message, response=response, body={"type": "error", "error": error})


def _sdk_message(text="<block>no"):
    return SimpleNamespace(model_dump_json=lambda **_: json.dumps({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "stop_sequence", "stop_sequence": "</block>",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }))


class _FakeSdkStream:
    def __init__(self, events, error_at_open=None):
        self.events, self.error_at_open = events, error_at_open

    async def __aenter__(self):
        if self.error_at_open is not None:
            raise self.error_at_open
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


STREAM_EVENTS = [
    {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [],
                                          "model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "<block>no"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
    {"type": "message_stop"},
]


class NativeAnthropicRetryTests(_CacheReset):
    async def test_rejected_stop_sequences_dropped_and_remembered(self):
        provider = _native_custom_provider()
        create = provider._anthropic_client.messages.create = mock.AsyncMock(side_effect=[
            _bad_request("stop_sequences is not supported for this model"),
            _sdk_message(),
        ])

        result = await provider.anthropic_messages(_request("custom:t/m"))

        self.assertEqual(result["content"][0]["text"], "<block>no")
        first, second = (c.kwargs for c in create.call_args_list)
        self.assertEqual(first["stop_sequences"], ["</block>"])
        self.assertNotIn("stop_sequences", second)
        self.assertEqual(second["temperature"], 0)
        self.assertEqual(known_rejected_params(provider, "m", "anthropic"), {"stop_sequences"})

        create.reset_mock(side_effect=True)
        create.return_value = _sdk_message()
        await provider.anthropic_messages(_request("custom:t/m"))
        self.assertEqual(create.await_count, 1)
        self.assertNotIn("stop_sequences", create.call_args.kwargs)

    async def test_structured_unsupported_parameter_code(self):
        provider = _native_custom_provider()
        error = _bad_request("Unsupported value", code="unsupported_parameter")
        error.body["error"]["param"] = "temperature"
        create = provider._anthropic_client.messages.create = mock.AsyncMock(side_effect=[error, _sdk_message()])
        await provider.anthropic_messages(_request("custom:t/m"))
        self.assertNotIn("temperature", create.call_args.kwargs)
        self.assertIn("stop_sequences", create.call_args.kwargs)

    async def test_unrelated_400_is_not_retried(self):
        provider = _native_custom_provider()
        create = provider._anthropic_client.messages.create = mock.AsyncMock(
            side_effect=_bad_request("messages: roles must alternate between user and assistant")
        )
        with self.assertRaises(ProviderHTTPError) as ctx:
            await provider.anthropic_messages(_request("custom:t/m"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(create.await_count, 1)

    async def test_stream_retried_when_open_is_rejected(self):
        provider = _native_custom_provider()
        stream = provider._anthropic_client.messages.stream = mock.MagicMock(side_effect=[
            _FakeSdkStream([], error_at_open=_bad_request("Unsupported parameter: 'stop_sequences'")),
            _FakeSdkStream(STREAM_EVENTS),
        ])

        chunks = [sse async for sse in provider.anthropic_messages_stream(_request("custom:t/m", stream=True))]

        self.assertTrue(chunks[0].startswith("event: message_start"))
        self.assertTrue(chunks[-1].startswith("event: message_stop"))
        self.assertNotIn("stop_sequences", stream.call_args_list[1].kwargs)

    async def test_stream_error_after_first_event_is_not_retried(self):
        provider = _native_custom_provider()
        stream = provider._anthropic_client.messages.stream = mock.MagicMock(return_value=_FakeSdkStream(
            STREAM_EVENTS[:1] + [_bad_request("stop_sequences is not supported for this model")]
        ))
        chunks = []
        with self.assertRaises(ProviderHTTPError):
            async for sse in provider.anthropic_messages_stream(_request("custom:t/m", stream=True)):
                chunks.append(sse)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stream.call_count, 1)


if __name__ == "__main__":
    unittest.main()

"""Anthropic adapter → real OpenAI-compatible provider methods.

A fake SDK client stands in for the upstream, so these tests exercise the
provider's own wire preparation (model-id normalization, the include_usage
default, Google's parameter filter) rather than mocked provider methods.
"""

import copy
import unittest
from types import SimpleNamespace

import httpx
import openai

from app.openai_models import ChatCompletionRequest
from app.providers import anthropic_adapter
from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    anthropic_adapter_metadata,
    known_rejected_params,
)
from app.providers.base import ProviderHTTPError
from app.providers.google_provider import GoogleProvider
from tests import test_anthropic_compat as compat
from tests import test_anthropic_messages as route_tests

ROUTED = "t/org/model"
UPSTREAM = "org/model"


class _Obj:
    """An SDK response object: attribute access plus model_dump()."""

    def __init__(self, data):
        self._data = data
        self.type = data.get("type")

    def model_dump(self, **_):
        return copy.deepcopy(self._data)


async def _aiter(items):
    for item in items:
        yield item


def _chat_response():
    return _Obj({
        "id": "c1", "object": "chat.completion", "created": 1, "model": "upstream-name",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def _chat_chunks():
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "upstream-name"}
    return _aiter([
        _Obj({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]}),
        _Obj({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ])


def _responses_response():
    return _Obj({
        "id": "r1", "object": "response", "status": "completed", "model": "upstream-name",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
    })


def _responses_events():
    return _aiter([
        _Obj({"type": "response.created", "response": {"id": "r1", "status": "in_progress", "model": "upstream-name"}}),
        _Obj({"type": "response.output_text.delta", "delta": "ok"}),
        _Obj({"type": "response.completed", "response": {"id": "r1", "status": "completed"}}),
    ])


def _bad_request(message, *, param=None, code=None):
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://upstream.test/v1")),
        body={"message": message, "type": "invalid_request_error", "param": param, "code": code},
    )


class FakeClient:
    """Captures every create() call; ``reject(kwargs, endpoint)`` may return
    an exception to raise instead of a successful response."""

    def __init__(self, reject=None):
        self.calls = []
        self._reject = reject or (lambda kwargs, endpoint: None)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)

    def endpoints(self):
        return [endpoint for endpoint, _ in self.calls]

    def kwargs(self, endpoint=None):
        return [kwargs for e, kwargs in self.calls if endpoint is None or e == endpoint]

    async def _create(self, endpoint, kwargs, success):
        self.calls.append((endpoint, copy.deepcopy(kwargs)))
        error = self._reject(kwargs, endpoint)
        if error is not None:
            raise error
        return success()

    async def _chat(self, **kwargs):
        return await self._create("chat", kwargs, _chat_chunks if kwargs.get("stream") else _chat_response)

    async def _responses(self, **kwargs):
        return await self._create("responses", kwargs, _responses_events if kwargs.get("stream") else _responses_response)


def _provider(reject=None):
    provider = compat._custom_provider()
    provider.client = FakeClient(reject)
    return provider


def _request(stream=False, **extra):
    return compat._req(model=ROUTED, stream=stream, **extra)


def _run(provider, request, *, transport="chat", allow_thinking=True):
    """Execute ``request`` and return (result, sse_events_or_None)."""
    if request.stream:
        chunks = compat.run(compat._collect(anthropic_adapter_messages_stream(
            provider, request, transport=transport, allow_thinking=allow_thinking,
        )))
        events = compat._parse_sse(chunks)
        return None, events
    return compat.run(anthropic_adapter_messages(
        provider, request, transport=transport, allow_thinking=allow_thinking,
    )), None


def _assert_successful_stream(test, events):
    types = [t for t, _ in events]
    test.assertEqual(types.count("message_start"), 1)
    test.assertNotIn("error", types)
    test.assertEqual(types[-1], "message_stop")


def _assert_stream_error(test, events):
    types = [t for t, _ in events]
    test.assertEqual(types.count("message_start"), 1)
    test.assertEqual(types[-1], "error")


class _AdapterTestCase(unittest.TestCase):
    def setUp(self):
        for cache in (anthropic_adapter._rejected_params, anthropic_adapter._responses_models):
            cache.clear()
            self.addCleanup(cache.clear)


class ModelIdOwnershipTests(_AdapterTestCase):
    def test_routed_model_reaches_upstream_with_its_namespace(self):
        for transport in ("chat", "responses"):
            for stream in (False, True):
                with self.subTest(transport=transport, stream=stream):
                    provider = _provider()
                    result, events = _run(provider, _request(stream=stream), transport=transport)
                    self.assertEqual([k["model"] for k in provider.client.kwargs()], [UPSTREAM])
                    if stream:
                        _assert_successful_stream(self, events)
                        self.assertEqual(events[0][1]["message"]["model"], ROUTED)
                    else:
                        self.assertEqual(result["model"], ROUTED)

    def test_simple_model_name_unchanged(self):
        provider = _provider()
        _run(provider, compat._req(model="plain"))
        self.assertEqual(provider.client.kwargs()[0]["model"], "plain")

    def test_chat_to_responses_fallback_and_cached_transport(self):
        def reject(kwargs, endpoint):
            if endpoint == "chat":
                return _bad_request("This model is only supported in v1/responses; use /v1/responses instead.")

        for stream in (False, True):
            with self.subTest(stream=stream):
                anthropic_adapter._responses_models.clear()
                provider = _provider(reject)
                _run(provider, _request(stream=stream))
                self.assertEqual(provider.client.endpoints(), ["chat", "responses"])
                # The model is now known to need Responses: no chat round trip.
                _run(provider, _request(stream=stream))
                self.assertEqual(provider.client.endpoints(), ["chat", "responses", "responses"])
                self.assertEqual({k["model"] for k in provider.client.kwargs()}, {UPSTREAM})


class RetryClassificationTests(_AdapterTestCase):
    SCHEMA_FORMAT = {"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"a": {"type": "string"}}, "patternProperties": {"^x": {}},
    }}}

    # (description, request extras, upstream message, structured param, structured code, the field
    # that must still be forwarded by a later request)
    NEGATIVE = [
        ("malformed schema", {"output_config": SCHEMA_FORMAT},
         "Invalid schema for response_format 'output': schema must be a JSON object.", None, None, "response_format"),
        ("unsupported schema keyword", {"output_config": SCHEMA_FORMAT},
         "Invalid schema for response_format 'output': unsupported keyword 'patternProperties'.",
         "response_format", "invalid_json_schema", "response_format"),
        ("top_k value", {"top_k": 5}, "'top_k' must be less than or equal to 1.", None, None, "top_k"),
        ("top_k typed value error", {"top_k": 5}, "Invalid value.", "top_k", "invalid_value", "top_k"),
        ("conflicting options", {"temperature": 0.5, "top_p": 0.5},
         "'top_p' is not supported together with 'temperature'; set only one.", None, None, "top_p"),
        ("value restriction", {"temperature": 0.5},
         "This model does not support temperature values other than 1.", None, None, "temperature"),
        ("incidental prose", {"metadata": {"user_id": "u1"}},
         "The user message is not supported: roles must alternate between user and assistant.",
         None, None, "user"),
    ]

    @staticmethod
    def _sent(kwargs, field):
        return field in kwargs or field in (kwargs.get("extra_body") or {})

    def test_value_and_schema_errors_are_not_retried_or_cached(self):
        for description, extras, message, param, code, field in self.NEGATIVE:
            for stream in (False, True):
                with self.subTest(description, stream=stream):
                    anthropic_adapter._rejected_params.clear()
                    failing = [True]

                    def reject(kwargs, endpoint):
                        if failing[0]:
                            return _bad_request(message, param=param, code=code)

                    provider = _provider(reject)
                    if stream:
                        _, events = _run(provider, _request(stream=True, **extras))
                        _assert_stream_error(self, events)
                        self.assertIn(message, events[-1][1]["error"]["message"])
                    else:
                        with self.assertRaises(ProviderHTTPError) as caught:
                            _run(provider, _request(**extras))
                        self.assertEqual(caught.exception.status_code, 400)
                        self.assertIn(message, str(caught.exception.body))
                    self.assertEqual(len(provider.client.calls), 1)
                    self.assertTrue(self._sent(provider.client.kwargs()[0], field))
                    self.assertEqual(known_rejected_params(provider, UPSTREAM, "chat"), set())

                    failing[0] = False
                    _run(provider, _request(stream=stream, **extras))
                    self.assertTrue(self._sent(provider.client.kwargs()[-1], field))

    def test_unrecognized_argument_is_retried_and_remembered(self):
        def reject(kwargs, endpoint):
            if "top_k" in (kwargs.get("extra_body") or {}):
                return _bad_request("Unrecognized request argument supplied: top_k")

        for stream in (False, True):
            with self.subTest(stream=stream):
                anthropic_adapter._rejected_params.clear()
                provider = _provider(reject)
                _, events = _run(provider, _request(stream=stream, top_k=5))
                if stream:
                    _assert_successful_stream(self, events)
                self.assertEqual(len(provider.client.calls), 2)
                self.assertNotIn("extra_body", provider.client.kwargs()[1])
                self.assertEqual(known_rejected_params(provider, UPSTREAM, "chat"), {"top_k"})

                # Later requests skip the round trip and report the drop.
                _run(provider, _request(stream=stream, top_k=5))
                self.assertEqual(len(provider.client.calls), 3)
                self.assertNotIn("extra_body", provider.client.kwargs()[2])
                metadata = anthropic_adapter_metadata(provider, _request(top_k=5), transport="chat", allow_thinking=True)
                self.assertIn("top_k", metadata.dropped_fields)
                metadata = anthropic_adapter_metadata(provider, _request(), transport="chat", allow_thinking=True)
                self.assertNotIn("top_k", metadata.dropped_fields)

    def test_structured_unsupported_parameter(self):
        def reject(kwargs, endpoint):
            if "temperature" in kwargs:
                return _bad_request("Unsupported parameter.", param="temperature", code="unsupported_parameter")

        provider = _provider(reject)
        result, _ = _run(provider, _request(temperature=0.5))
        self.assertEqual(result["content"][0]["text"], "ok")
        self.assertEqual(len(provider.client.calls), 2)
        self.assertNotIn("temperature", provider.client.kwargs()[1])
        self.assertEqual(known_rejected_params(provider, UPSTREAM, "chat"), {"temperature"})

    def test_nested_reasoning_content_rejection(self):
        def reject(kwargs, endpoint):
            if any("reasoning_content" in m for m in kwargs["messages"]):
                return _bad_request(
                    "Unknown field.", param="messages[1].reasoning_content", code="unknown_parameter",
                )

        provider = _provider(reject)
        request = compat._req(model=ROUTED, messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "s"},
                {"type": "text", "text": "hello"},
            ]},
            {"role": "user", "content": "again"},
        ])
        compat.run(anthropic_adapter_messages(
            provider, request, transport="chat", allow_thinking=True, echo_reasoning=True,
        ))
        self.assertEqual(len(provider.client.calls), 2)
        self.assertTrue(any("reasoning_content" in m for m in provider.client.kwargs()[0]["messages"]))
        self.assertFalse(any("reasoning_content" in m for m in provider.client.kwargs()[1]["messages"]))
        self.assertEqual(known_rejected_params(provider, UPSTREAM, "chat"), {"reasoning_content"})

    def test_rejection_is_scoped_to_its_transport(self):
        def reject(kwargs, endpoint):
            if endpoint == "chat" and "temperature" in kwargs:
                return _bad_request("Unrecognized request argument supplied: temperature")

        provider = _provider(reject)
        _run(provider, _request(temperature=0.5))
        self.assertEqual(known_rejected_params(provider, UPSTREAM, "chat"), {"temperature"})
        self.assertEqual(known_rejected_params(provider, UPSTREAM, "responses"), set())
        _run(provider, _request(temperature=0.5), transport="responses")
        self.assertEqual(provider.client.kwargs("responses")[0]["temperature"], 0.5)

    def test_organization_verification_drops_summary_for_this_request_only(self):
        def reject(kwargs, endpoint):
            if (kwargs.get("reasoning") or {}).get("summary"):
                return _bad_request(
                    "Your organization must be verified to generate reasoning summaries.",
                    param="reasoning.summary", code="unsupported_value",
                )

        thinking = {"type": "enabled", "budget_tokens": 2048}
        for stream in (False, True):
            with self.subTest(stream=stream):
                provider = _provider(reject)
                _run(provider, _request(stream=stream, max_tokens=4096, thinking=thinking), transport="responses")
                first, retry = provider.client.kwargs()
                self.assertEqual(first["reasoning"]["summary"], "auto")
                self.assertNotIn("summary", retry["reasoning"])
                self.assertEqual(retry["reasoning"]["effort"], first["reasoning"]["effort"])
                self.assertEqual(known_rejected_params(provider, UPSTREAM, "responses"), set())
                # No permanent ban: the next request asks for the summary again.
                _run(provider, _request(stream=stream, max_tokens=4096, thinking=thinking), transport="responses")
                self.assertEqual(provider.client.kwargs()[2]["reasoning"]["summary"], "auto")

    def test_retry_bound(self):
        order = ("temperature", "top_p", "stop", "top_k")

        def reject(kwargs, endpoint):
            present = [p for p in order if p in kwargs or p in (kwargs.get("extra_body") or {})]
            if present:
                return _bad_request(f"Unrecognized request argument supplied: {present[0]}")

        extras = {"temperature": 0.5, "top_p": 0.5, "stop_sequences": ["END"], "top_k": 5}
        provider = _provider(reject)
        with self.assertRaises(ProviderHTTPError):
            _run(provider, _request(**extras))
        self.assertEqual(len(provider.client.calls), anthropic_adapter._MAX_PARAM_RETRIES + 1)

        provider = _provider(reject)
        _, events = _run(provider, _request(stream=True, **extras))
        _assert_stream_error(self, events)
        self.assertEqual(len(provider.client.calls), anthropic_adapter._MAX_PARAM_RETRIES + 1)


class StreamOptionsSuppressionTests(_AdapterTestCase):
    @staticmethod
    def _reject_stream_options(kwargs, endpoint):
        if "stream_options" in kwargs:
            return _bad_request("Unrecognized request argument supplied: stream_options")

    def test_rejected_stream_options_stays_omitted(self):
        provider = _provider(self._reject_stream_options)
        _, events = _run(provider, _request(stream=True))
        _assert_successful_stream(self, events)
        first, retry = provider.client.kwargs()
        self.assertEqual(first["stream_options"], {"include_usage": True})
        self.assertNotIn("stream_options", retry)
        for kwargs in provider.client.kwargs():
            self.assertFalse(any("omit" in key for key in kwargs))

        # Remembered: the next stream omits it without a failed attempt.
        _, events = _run(provider, _request(stream=True))
        _assert_successful_stream(self, events)
        self.assertEqual(len(provider.client.calls), 3)
        self.assertNotIn("stream_options", provider.client.kwargs()[2])
        metadata = anthropic_adapter_metadata(provider, _request(stream=True), transport="chat", allow_thinking=True)
        self.assertNotIn("stream_options", metadata.dropped_fields)

    def test_suppression_survives_a_second_parameter_retry(self):
        def reject(kwargs, endpoint):
            return self._reject_stream_options(kwargs, endpoint) or (
                _bad_request("Unrecognized request argument supplied: top_k")
                if "top_k" in (kwargs.get("extra_body") or {}) else None
            )

        provider = _provider(reject)
        _, events = _run(provider, _request(stream=True, top_k=5))
        _assert_successful_stream(self, events)
        self.assertEqual(len(provider.client.calls), 3)
        last = provider.client.kwargs()[-1]
        self.assertNotIn("stream_options", last)
        self.assertNotIn("extra_body", last)

    def test_usage_default_kept_elsewhere(self):
        provider = _provider(self._reject_stream_options)
        _run(provider, _request(stream=True))
        # Another model on the same provider still gets the default first.
        _run(provider, compat._req(model="t/other", stream=True))
        self.assertEqual(provider.client.kwargs()[2]["stream_options"], {"include_usage": True})

        other_provider = _provider()
        _run(other_provider, _request(stream=True))
        self.assertEqual(other_provider.client.kwargs()[0]["stream_options"], {"include_usage": True})

        direct = ChatCompletionRequest(model=ROUTED, messages=[{"role": "user", "content": "hi"}], stream=True)
        compat.run(compat._collect(provider.chat_completion_stream(direct)))
        self.assertEqual(provider.client.kwargs()[-1]["stream_options"], {"include_usage": True})


def _google_provider():
    provider = GoogleProvider({"api_key": "k", "name": "test"})
    provider.client = FakeClient()
    return provider


GOOGLE_EXTRAS = {
    "output_config": {"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"a": {"type": "string"}},
    }}},
    "metadata": {"user_id": "u1"},
    "top_k": 5,
    "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
    "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": True},
}
GOOGLE_DROPS = {"output_config.format", "metadata.user_id", "top_k", "tool_choice.disable_parallel_tool_use"}


class GoogleDroppedFieldTests(_AdapterTestCase):
    @staticmethod
    def _google_request(stream=False, **extras):
        return compat._req(model="google:test/gemini-2.5-flash", stream=stream, **extras)

    def test_reported_drops_match_the_wire_request(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                provider = _google_provider()
                request = self._google_request(stream=stream, **GOOGLE_EXTRAS)
                metadata = provider.get_anthropic_request_metadata(request)
                self.assertTrue(GOOGLE_DROPS <= set(metadata.dropped_fields), metadata.dropped_fields)
                _, events = _run(provider, request, allow_thinking=False)
                if stream:
                    _assert_successful_stream(self, events)
                (kwargs,) = provider.client.kwargs()
                for field in ("response_format", "user", "parallel_tool_calls", "extra_body", "stream_options"):
                    self.assertNotIn(field, kwargs)
                # Dropping only the parallelism setting keeps the forced tool.
                self.assertEqual(kwargs["tool_choice"], {"type": "function", "function": {"name": "f"}})

    def test_absent_fields_are_not_reported(self):
        provider = _google_provider()
        metadata = provider.get_anthropic_request_metadata(self._google_request())
        self.assertFalse(GOOGLE_DROPS & set(metadata.dropped_fields))

        allowed_parallel = dict(GOOGLE_EXTRAS, tool_choice={"type": "auto", "disable_parallel_tool_use": False})
        metadata = provider.get_anthropic_request_metadata(self._google_request(**allowed_parallel))
        self.assertNotIn("tool_choice.disable_parallel_tool_use", metadata.dropped_fields)

    def test_custom_provider_still_forwards_the_same_fields(self):
        provider = _provider()
        request = _request(**GOOGLE_EXTRAS)
        metadata = provider.get_anthropic_request_metadata(request)
        self.assertFalse(GOOGLE_DROPS & set(metadata.dropped_fields))
        _run(provider, request)
        (kwargs,) = provider.client.kwargs()
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(kwargs["user"], "u1")
        self.assertIs(kwargs["parallel_tool_calls"], False)
        self.assertEqual(kwargs["extra_body"], {"top_k": 5})


class GoogleRouteHeaderTests(_AdapterTestCase):
    """The route's x-llmproxy-dropped-anthropic-fields header for a real
    GoogleProvider, streaming and not."""

    _route = route_tests.AnthropicMessagesRouteTests
    _invoke_create_message = _route._invoke_create_message
    _start_tracking_request = _route._start_tracking_request
    _make_request_obj = _route.__dict__["_make_request_obj"]
    _collect_response_body = _route.__dict__["_collect_response_body"]
    _payload = _route.__dict__["_payload"]

    def setUp(self):
        super().setUp()
        self._route.setUp(self)

    def tearDown(self):
        self._route.tearDown(self)

    def test_header_lists_the_static_drops(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                provider = _google_provider()
                provider._is_chat_capable_model = lambda model_name: True
                response, body, _ = self._invoke_create_message(
                    self._payload(model="google:test/gemini-2.5-flash", stream=stream, **GOOGLE_EXTRAS),
                    provider,
                )
                self.assertEqual(response.status_code, 200, body)
                self.assertEqual(response.headers.get("x-llmproxy-anthropic-mode"), "adapter")
                header = set(response.headers.get("x-llmproxy-dropped-anthropic-fields", "").split(","))
                self.assertTrue(GOOGLE_DROPS <= {h.strip() for h in header}, header)
                self.assertEqual(len(provider.client.calls), 1)
                self.assertNotIn("response_format", provider.client.kwargs()[0])


if __name__ == "__main__":
    unittest.main()

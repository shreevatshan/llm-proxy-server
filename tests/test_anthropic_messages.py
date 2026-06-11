import asyncio
import itertools
import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.anthropic_models import AnthropicMessagesRequest
from app.openai_models import ModelInfo
from app.providers import anthropic_compatible
from app.request_tracker import request_tracker
from app.routes import anthropic_messages
from app.providers.base import AnthropicRequestMetadata
from app.providers.bedrock_provider import BedrockProvider
from app.providers.custom_providers import CustomProvider


def _make_model_info(model_id: str) -> ModelInfo:
    return ModelInfo(id=model_id, created=0, owned_by="test", provider=model_id.split("/")[0])


def _fake_model_cache(model_id: str):
    """Return a SimpleNamespace that looks like ModelCache for the given model id."""
    return SimpleNamespace(
        get_enabled_models=lambda: [_make_model_info(model_id)],
        update_models=lambda models: None,
    )


class FakeCustomProvider(CustomProvider):
    def __init__(self):
        self.full_provider_name = "custom:test"
        self._supported_apis = ["anthropic"]
        self.stream_calls = []
        self.nonstream_calls = []

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        self.stream_calls.append({
            "request_stream": request.stream,
            "beta": anthropic_beta,
        })
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

    async def anthropic_messages(self, request, anthropic_beta=None):
        self.nonstream_calls.append({
            "request_stream": request.stream,
            "beta": anthropic_beta,
        })
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": request.model,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class FakeNonCustomProvider:
    def __init__(self):
        self.full_provider_name = "bedrock:test"
        self.stream_calls = []
        self.nonstream_calls = []
        self.metadata = AnthropicRequestMetadata(mode="native", transport="messages")

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        self.stream_calls.append({
            "request_stream": request.stream,
            "beta": anthropic_beta,
        })
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'

    async def anthropic_messages(self, request, anthropic_beta=None):
        self.nonstream_calls.append({
            "request_stream": request.stream,
            "beta": anthropic_beta,
        })
        return {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": request.model,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def get_anthropic_request_metadata(self, request, anthropic_beta=None):
        return self.metadata


class FakeAdapterProvider(FakeNonCustomProvider):
    def __init__(self):
        super().__init__()
        self.full_provider_name = "google:test"
        self.metadata = AnthropicRequestMetadata(
            mode="adapter",
            transport="chat",
            dropped_fields=["thinking", "anthropic-beta"],
        )


class FakeAsyncAnthropicEventStream:
    def __init__(self, events):
        self._events = list(events)
        self._index = 0
        self._terminal_seen = False
        self.post_terminal_pulls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        if self._terminal_seen:
            self.post_terminal_pulls += 1
        event = self._events[self._index]
        self._index += 1
        event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
        if event_type in {"message_stop", "error"}:
            self._terminal_seen = True
        return event


class FakeAsyncAnthropicStreamContext:
    def __init__(self, event_stream):
        self.event_stream = event_stream
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.event_stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True


class FakeAnthropicSdkClient:
    def __init__(self, stream_context):
        self.stream_context = stream_context
        self.last_stream_kwargs = None
        self.messages = SimpleNamespace(stream=self.stream)

    def stream(self, **kwargs):
        self.last_stream_kwargs = kwargs
        return self.stream_context


class FakeSdkBackedCustomProvider(CustomProvider):
    def __init__(self, sdk_client):
        self.full_provider_name = "custom:test"
        self.custom_provider_name = "test"
        self._supported_apis = ["anthropic"]
        self._anthropic_client = sdk_client
        self.base_url = "http://example.com/v1"
        self.api_key = None


class FakeSyncBedrockEventStream:
    def __init__(self, events):
        self._events = list(events)
        self._index = 0
        self._terminal_seen = False
        self.post_terminal_pulls = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._events):
            raise StopIteration
        if self._terminal_seen:
            self.post_terminal_pulls += 1
        event = self._events[self._index]
        self._index += 1
        if event.get("type") in {"message_stop", "error"}:
            self._terminal_seen = True
        return {
            "chunk": {
                "bytes": json.dumps(event).encode("utf-8"),
            }
        }

    def close(self):
        self.closed = True


class FakeBedrockRuntime:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []

    def invoke_model_with_response_stream(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": self.stream}


class AnthropicMessagesRouteTests(unittest.TestCase):
    def setUp(self):
        asyncio.run(request_tracker.stop())
        asyncio.run(request_tracker.start())
        self._request_ids = itertools.count(1)

    def tearDown(self):
        asyncio.run(request_tracker.stop())

    @staticmethod
    def _payload(**extra):
        payload = {
            "model": "custom:test/claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _active_request():
        active = request_tracker.get_active_requests()
        assert len(active) == 1, active
        return active[0]

    @staticmethod
    async def _collect_async_chunks(generator):
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)
        return chunks

    @staticmethod
    async def _collect_response_body(response):
        if hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                chunks.append(chunk)
            return "".join(chunks)
        body = getattr(response, "body", b"")
        if isinstance(body, bytes):
            return body.decode("utf-8")
        return str(body)

    async def _start_tracking_request(self, request_id: str, *, endpoint: str = "/v1/messages"):
        await request_tracker.start_request(
            request_id=request_id,
            server="anthropic",
            endpoint=endpoint,
            method="POST",
            model=None,
            user_identity="test-user",
            user_type="user",
            is_streaming=False,
        )

    @staticmethod
    def _make_request_obj(request_id: str):
        async def is_disconnected():
            return False

        return SimpleNamespace(
            state=SimpleNamespace(tracking_request_id=request_id),
            is_disconnected=is_disconnected,
        )

    def _invoke_create_message(self, payload, provider, *, anthropic_beta=None, span_attrs=None):
        request_id = f"req-{next(self._request_ids)}"
        request_obj = self._make_request_obj(request_id)
        asyncio.run(self._start_tracking_request(request_id))

        patchers = [
            patch.object(
                anthropic_messages.provider_manager,
                "get_anthropic_provider_for_model",
                AsyncMock(return_value=provider),
            ),
            patch.object(
                anthropic_messages.provider_manager,
                "model_cache",
                _fake_model_cache(payload.get("model", "custom:test/model")),
            ),
        ]
        if span_attrs is not None:
            patchers.append(
                patch.object(
                    anthropic_messages,
                    "add_span_attributes",
                    side_effect=lambda span, attrs: span_attrs.append(dict(attrs)),
                )
            )

        with patchers[0], patchers[1]:
            if len(patchers) == 3:
                with patchers[2]:
                    response = asyncio.run(
                        anthropic_messages.create_message(
                            request_obj=request_obj,
                            request=AnthropicMessagesRequest(**payload),
                            anthropic_beta=anthropic_beta,
                            auth=object(),
                        )
                    )
            else:
                response = asyncio.run(
                    anthropic_messages.create_message(
                        request_obj=request_obj,
                        request=AnthropicMessagesRequest(**payload),
                        anthropic_beta=anthropic_beta,
                        auth=object(),
                    )
                )

        body = asyncio.run(self._collect_response_body(response))
        return response, body, request_obj

    def test_omitted_stream_uses_streaming_for_custom_provider(self):
        provider = FakeCustomProvider()
        span_attrs = []
        response, body, _ = self._invoke_create_message(
            self._payload(),
            provider,
            anthropic_beta="beta-stream",
            span_attrs=span_attrs,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.media_type.startswith("text/event-stream"))
        self.assertIn("event: message_start", body)
        self.assertEqual(len(provider.stream_calls), 1)
        self.assertEqual(provider.stream_calls[0]["beta"], "beta-stream")
        self.assertEqual(provider.nonstream_calls, [])
        self.assertTrue(self._active_request()["is_streaming"])
        self.assertTrue(any(attrs.get("anthropic.stream") is True for attrs in span_attrs))

    def test_explicit_false_stays_non_streaming_for_custom_provider(self):
        provider = FakeCustomProvider()
        span_attrs = []
        response, body, _ = self._invoke_create_message(
            self._payload(stream=False),
            provider,
            anthropic_beta="beta-json",
            span_attrs=span_attrs,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.media_type.startswith("application/json"))
        self.assertEqual(json.loads(body)["id"], "msg_123")
        self.assertEqual(len(provider.nonstream_calls), 1)
        self.assertEqual(provider.nonstream_calls[0]["beta"], "beta-json")
        self.assertEqual(provider.stream_calls, [])
        self.assertFalse(self._active_request()["is_streaming"])
        self.assertTrue(any(attrs.get("anthropic.stream") is False for attrs in span_attrs))

    def test_explicit_true_stays_streaming_for_custom_provider(self):
        provider = FakeCustomProvider()
        response, _, _ = self._invoke_create_message(
            self._payload(stream=True),
            provider,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.media_type.startswith("text/event-stream"))
        self.assertEqual(len(provider.stream_calls), 1)
        self.assertEqual(provider.nonstream_calls, [])
        self.assertTrue(self._active_request()["is_streaming"])

    def test_omitted_stream_stays_non_streaming_for_non_custom_provider(self):
        provider = FakeNonCustomProvider()
        span_attrs = []
        response, _, _ = self._invoke_create_message(
            self._payload(model="bedrock:test/claude-sonnet"),
            provider,
            span_attrs=span_attrs,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.media_type.startswith("application/json"))
        self.assertEqual(len(provider.nonstream_calls), 1)
        self.assertEqual(provider.stream_calls, [])
        self.assertFalse(self._active_request()["is_streaming"])
        self.assertTrue(any(attrs.get("anthropic.stream") is False for attrs in span_attrs))

    def test_count_tokens_returns_estimate_even_without_provider_resolution(self):
        span_attrs = []

        with patch.object(
            anthropic_messages.provider_manager,
            "get_anthropic_provider_for_model",
            AsyncMock(return_value=None),
        ), patch.object(
            anthropic_messages,
            "add_span_attributes",
            side_effect=lambda span, attrs: span_attrs.append(dict(attrs)),
        ):
            response = asyncio.run(
                anthropic_messages.count_message_tokens(
                    payload={
                    "model": "claude-sonnet-without-prefix",
                    "messages": [{"role": "user", "content": "hello world"}],
                    },
                    auth=object(),
                )
            )

        self.assertIn("input_tokens", response.model_dump())
        self.assertGreater(response.input_tokens, 0)
        self.assertTrue(any(attrs.get("anthropic.provider_resolved") is False for attrs in span_attrs))

    def test_count_tokens_uses_provider_hook_when_available(self):
        provider = FakeNonCustomProvider()
        provider.anthropic_count_tokens = AsyncMock(return_value={"input_tokens": 77})

        with patch.object(
            anthropic_messages.provider_manager,
            "get_anthropic_provider_for_model",
            AsyncMock(return_value=provider),
        ):
            response = asyncio.run(
                anthropic_messages.count_message_tokens(
                    payload={
                        "model": "bedrock:test/claude-sonnet",
                        "messages": [{"role": "user", "content": "hello world"}],
                    },
                    auth=object(),
                )
            )

        self.assertEqual(response.input_tokens, 77)

    def test_nonstream_response_includes_adapter_headers(self):
        provider = FakeAdapterProvider()
        response, _, _ = self._invoke_create_message(
            self._payload(model="google:test/gemini-2.5-flash", stream=False),
            provider,
            anthropic_beta="beta-a",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("x-llmproxy-anthropic-mode"), "adapter")
        self.assertIn("thinking", response.headers.get("x-llmproxy-dropped-anthropic-fields", ""))

    def test_sdk_backed_custom_provider_stops_after_message_stop(self):
        event_stream = FakeAsyncAnthropicEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "message_stop"},
            {"type": "ping"},
        ])
        stream_context = FakeAsyncAnthropicStreamContext(event_stream)
        provider = FakeSdkBackedCustomProvider(FakeAnthropicSdkClient(stream_context))
        request = AnthropicMessagesRequest(**self._payload(stream=True))

        chunks = asyncio.run(self._collect_async_chunks(provider.anthropic_messages_stream(request)))

        self.assertEqual(len(chunks), 2)
        self.assertIn("event: message_start", chunks[0])
        self.assertIn("event: message_stop", chunks[1])
        self.assertTrue(stream_context.entered)
        self.assertTrue(stream_context.exited)
        self.assertEqual(event_stream.post_terminal_pulls, 1)
        self.assertNotIn("event: ping", "".join(chunks))

    def test_sdk_backed_custom_provider_treats_error_as_terminal(self):
        event_stream = FakeAsyncAnthropicEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "error", "error": {"type": "api_error", "message": "boom"}},
            {"type": "ping"},
        ])
        stream_context = FakeAsyncAnthropicStreamContext(event_stream)
        provider = FakeSdkBackedCustomProvider(FakeAnthropicSdkClient(stream_context))
        request = AnthropicMessagesRequest(**self._payload(stream=True))

        chunks = asyncio.run(self._collect_async_chunks(provider.anthropic_messages_stream(request)))

        self.assertEqual(len(chunks), 2)
        self.assertIn("event: error", chunks[1])
        self.assertEqual(event_stream.post_terminal_pulls, 1)
        self.assertNotIn("event: ping", "".join(chunks))

    def test_sdk_backed_custom_provider_bounds_post_terminal_drain(self):
        event_stream = FakeAsyncAnthropicEventStream(
            [
                {"type": "message_start", "message": {"id": "msg_1"}},
                {"type": "message_stop"},
                {"type": "ping"},
                {"type": "ping"},
                {"type": "ping"},
            ]
        )
        stream_context = FakeAsyncAnthropicStreamContext(event_stream)
        provider = FakeSdkBackedCustomProvider(FakeAnthropicSdkClient(stream_context))
        request = AnthropicMessagesRequest(**self._payload(stream=True))

        with patch.object(anthropic_compatible, "ANTHROPIC_POST_TERMINAL_DRAIN_MAX_EVENTS", 2), patch.object(
            anthropic_compatible,
            "ANTHROPIC_POST_TERMINAL_DRAIN_MAX_SECONDS",
            10.0,
        ):
            chunks = asyncio.run(self._collect_async_chunks(provider.anthropic_messages_stream(request)))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(event_stream.post_terminal_pulls, 2)
        self.assertNotIn("event: ping", "".join(chunks))

    def test_post_terminal_drain_stops_when_time_budget_expires(self):
        terminal_seen_at = time.monotonic() - 1.0

        with patch.object(anthropic_compatible, "ANTHROPIC_POST_TERMINAL_DRAIN_MAX_SECONDS", 0.25), patch.object(
            anthropic_compatible,
            "ANTHROPIC_POST_TERMINAL_DRAIN_MAX_EVENTS",
            100,
        ):
            stop_reason = anthropic_compatible.get_anthropic_post_terminal_drain_stop_reason(
                terminal_seen_at=terminal_seen_at,
                drained_event_count=1,
            )

        self.assertIsNotNone(stop_reason)
        self.assertTrue(stop_reason.startswith("time_budget_exceeded("))

    def test_route_stream_finishes_after_message_stop_and_clears_active_request(self):
        event_stream = FakeAsyncAnthropicEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "message_stop"},
            {"type": "ping"},
        ])
        stream_context = FakeAsyncAnthropicStreamContext(event_stream)
        provider = FakeSdkBackedCustomProvider(FakeAnthropicSdkClient(stream_context))
        response, body, request_obj = self._invoke_create_message(
            self._payload(),
            provider,
        )

        tracking_final = getattr(request_obj.state, "tracking_final", None)
        if isinstance(tracking_final, dict):
            asyncio.run(
                request_tracker.end_request(
                    request_obj.state.tracking_request_id,
                    status=tracking_final.get("status", "completed"),
                    termination_reason=tracking_final.get("termination_reason"),
                    error=tracking_final.get("error"),
                )
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message_stop", body)
        self.assertNotIn("event: ping", body)
        self.assertEqual(event_stream.post_terminal_pulls, 1)
        self.assertEqual(request_tracker.get_active_requests(), [])

    def test_route_does_not_mark_late_error_after_message_stop_as_failed(self):
        class _LateErrorProvider(FakeNonCustomProvider):
            async def anthropic_messages_stream(self, request, anthropic_beta=None):
                yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
                yield 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"late boom"}}\n\n'

        provider = _LateErrorProvider()
        response, body, request_obj = self._invoke_create_message(
            self._payload(model="bedrock:test/claude-sonnet", stream=True),
            provider,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: message_stop", body)
        self.assertIn("late boom", body)
        self.assertEqual(
            getattr(request_obj.state, "tracking_final", None),
            {
                "status": "completed",
                "termination_reason": "completed",
                "error": None,
            },
        )

    def test_bedrock_native_worker_stops_after_message_stop_and_closes_stream(self):
        event_stream = FakeSyncBedrockEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "message_stop"},
            {"type": "ping"},
        ])
        provider = BedrockProvider.__new__(BedrockProvider)
        provider.bedrock_runtime_native_stream = FakeBedrockRuntime(event_stream)
        event_queue = queue.Queue()

        provider._stream_worker_native("anthropic.claude-sonnet-4-5", {"messages": []}, event_queue)

        queued_items = []
        while not event_queue.empty():
            queued_items.append(event_queue.get_nowait())

        self.assertEqual([item[0] for item in queued_items], ["upstream_event", "upstream_event", "done"])
        self.assertIn("event: message_stop", queued_items[1][1]["sse"])
        self.assertEqual(queued_items[2][1]["terminal_event_seen"], True)
        self.assertEqual(queued_items[2][1]["terminal_event_type"], "message_stop")
        self.assertEqual(event_stream.post_terminal_pulls, 0)
        self.assertTrue(event_stream.closed)

    def test_bedrock_native_worker_reads_to_eof_without_terminal_event(self):
        event_stream = FakeSyncBedrockEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "ping"},
        ])
        provider = BedrockProvider.__new__(BedrockProvider)
        provider.bedrock_runtime_native_stream = FakeBedrockRuntime(event_stream)
        event_queue = queue.Queue()

        provider._stream_worker_native("anthropic.claude-haiku-4-5", {"messages": []}, event_queue)

        queued_items = []
        while not event_queue.empty():
            queued_items.append(event_queue.get_nowait())

        self.assertEqual([item[0] for item in queued_items], ["upstream_event", "upstream_event", "done"])
        self.assertIn("event: ping", queued_items[1][1]["sse"])
        self.assertEqual(queued_items[2][1]["terminal_event_seen"], False)
        self.assertEqual(queued_items[2][1]["transport_eof_observed"], True)
        self.assertTrue(event_stream.closed)

    def test_bedrock_native_worker_honors_pre_set_stop_signal(self):
        event_stream = FakeSyncBedrockEventStream([
            {"type": "message_start", "message": {"id": "msg_1"}},
        ])
        provider = BedrockProvider.__new__(BedrockProvider)
        provider.bedrock_runtime_native_stream = FakeBedrockRuntime(event_stream)
        event_queue = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()

        provider._stream_worker_native(
            "anthropic.claude-sonnet-4-5",
            {"messages": []},
            event_queue,
            stop_event,
        )

        queued_items = []
        while not event_queue.empty():
            queued_items.append(event_queue.get_nowait())

        self.assertEqual([item[0] for item in queued_items], ["done"])
        self.assertTrue(queued_items[0][1]["worker_stopped_early"])
        self.assertEqual(queued_items[0][1]["event_count"], 0)


class BedrockNativeSystemNormalizationTests(unittest.TestCase):
    """Tests for role='system' normalization in _convert_to_anthropic_native_request."""

    def _provider(self):
        p = BedrockProvider.__new__(BedrockProvider)
        p.full_provider_name = "bedrock:test"
        return p

    def _make_request(self, messages, system=None):
        data = {
            "model": "anthropic.claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": messages,
        }
        if system is not None:
            data["system"] = system
        return AnthropicMessagesRequest.model_validate(data)

    def test_system_in_messages_no_top_level_system(self):
        request = self._make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "sys A"},
            ]
        )
        native = self._provider()._convert_to_anthropic_native_request(request)
        self.assertEqual([m["role"] for m in native["messages"]], ["user"])
        system = native.get("system")
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["text"], "sys A")

    def test_system_in_messages_list_content_preserves_cache_control(self):
        request = self._make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": [{"type": "text", "text": "X", "cache_control": {"type": "ephemeral"}}]},
            ]
        )
        native = self._provider()._convert_to_anthropic_native_request(request)
        system = native.get("system")
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})

    def test_system_in_messages_alongside_top_level_list_system(self):
        request = self._make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "extra"},
            ],
            system=[{"type": "text", "text": "hdr", "cache_control": {"type": "ephemeral"}}],
        )
        native = self._provider()._convert_to_anthropic_native_request(request)
        system = native.get("system")
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["text"], "hdr")
        self.assertEqual(system[0].get("cache_control"), {"type": "ephemeral"})
        self.assertEqual(system[1]["text"], "extra")

    def test_unknown_role_coerced_to_user(self):
        request = self._make_request(
            messages=[{"role": "tool", "content": "result"}]
        )
        with self.assertLogs("app.anthropic_models", level="WARNING"):
            native = self._provider()._convert_to_anthropic_native_request(request)
        self.assertEqual(native["messages"][0]["role"], "user")

    def test_happy_path_no_system_messages_passthrough(self):
        request = self._make_request(
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            system="base system",
        )
        native = self._provider()._convert_to_anthropic_native_request(request)
        self.assertEqual([m["role"] for m in native["messages"]], ["user", "assistant"])
        self.assertEqual(native.get("system"), "base system")


class BuildAnthropicSdkKwargsTests(unittest.TestCase):
    """Unit tests for role="system" normalization in build_anthropic_sdk_kwargs."""

    def _make_request(self, messages, system=None):
        data = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": messages,
        }
        if system is not None:
            data["system"] = system
        return AnthropicMessagesRequest.model_validate(data)

    def _kwargs(self, messages, system=None):
        from app.anthropic_models import build_anthropic_sdk_kwargs
        return build_anthropic_sdk_kwargs(self._make_request(messages, system), "claude-sonnet-4-6")

    def test_system_in_messages_string_content_no_top_level_system(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "sys A"},
            ]
        )
        self.assertEqual([m["role"] for m in kwargs["messages"]], ["user"])
        system = kwargs["system"]
        self.assertIsInstance(system, list)
        self.assertEqual(system[0]["type"], "text")
        self.assertEqual(system[0]["text"], "sys A")

    def test_system_in_messages_with_top_level_system_string(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "EXTRA"},
            ],
            system="MAIN",
        )
        system = kwargs["system"]
        self.assertIsInstance(system, list)
        self.assertEqual(system[0], {"type": "text", "text": "MAIN"})
        self.assertEqual(system[1]["text"], "EXTRA")

    def test_system_in_messages_list_content_preserves_cache_control(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": [{"type": "text", "text": "X", "cache_control": {"type": "ephemeral"}}]},
            ]
        )
        system = kwargs["system"]
        self.assertEqual(len(system), 1)
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(system[0]["text"], "X")

    def test_multiple_system_messages_concatenated_in_order(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "q1"},
                {"role": "system", "content": "first"},
                {"role": "assistant", "content": "a1"},
                {"role": "system", "content": "second"},
            ]
        )
        roles = [m["role"] for m in kwargs["messages"]]
        self.assertEqual(roles, ["user", "assistant"])
        texts = [b["text"] for b in kwargs["system"]]
        self.assertEqual(texts, ["first", "second"])

    def test_system_in_messages_alongside_top_level_list_system(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "extra"},
            ],
            system=[{"type": "text", "text": "hdr", "cache_control": {"type": "ephemeral"}}],
        )
        system = kwargs["system"]
        self.assertEqual(system[0]["text"], "hdr")
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(system[1]["text"], "extra")

    def test_unknown_role_coerced_to_user_with_warning(self):
        with self.assertLogs("app.anthropic_models", level="WARNING") as cm:
            kwargs = self._kwargs(
                messages=[{"role": "tool", "content": "result"}]
            )
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertTrue(any("tool" in line for line in cm.output))

    def test_valid_messages_passthrough_unchanged(self):
        kwargs = self._kwargs(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            system="base",
        )
        self.assertEqual([m["role"] for m in kwargs["messages"]], ["user", "assistant"])
        self.assertEqual(kwargs["system"], "base")

    def test_adapter_prepare_strips_system_messages(self):
        from app.providers.anthropic_adapter import prepare_anthropic_adapter_request
        request = self._make_request(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "adapter sys"},
            ]
        )
        sanitized, metadata = prepare_anthropic_adapter_request(request, transport="chat")
        roles = [m.role for m in sanitized.messages]
        self.assertNotIn("system", roles)
        self.assertIsNotNone(sanitized.system)
        self.assertIn("messages.system", metadata.dropped_fields)


class PreflightAndSdkErrorTests(unittest.TestCase):
    """Tests for pre-flight model validation and Anthropic SDK error translation."""

    def setUp(self):
        asyncio.run(request_tracker.stop())
        asyncio.run(request_tracker.start())
        self._request_ids = itertools.count(100)

    def tearDown(self):
        asyncio.run(request_tracker.stop())

    @staticmethod
    def _payload(**extra):
        payload = {
            "model": "lmstudio:msi-ai-test-01/test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "stream": False,
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _make_request_obj(request_id: str):
        async def is_disconnected():
            return False
        return SimpleNamespace(
            state=SimpleNamespace(tracking_request_id=request_id),
            is_disconnected=is_disconnected,
        )

    @staticmethod
    async def _collect_response_body(response):
        if hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                chunks.append(chunk)
            return "".join(chunks)
        body = getattr(response, "body", b"")
        if isinstance(body, bytes):
            return body.decode("utf-8")
        return str(body)

    def _invoke(self, payload, provider, *, cache_models=None):
        """Invoke create_message with patched provider resolution and model cache."""
        request_id = f"req-{next(self._request_ids)}"
        request_obj = self._make_request_obj(request_id)
        asyncio.run(
            request_tracker.start_request(
                request_id=request_id,
                server="anthropic",
                endpoint="/v1/messages",
                method="POST",
                model=None,
                user_identity="test-user",
                user_type="user",
                is_streaming=False,
            )
        )

        # Default cache: model is present
        if cache_models is None:
            cache_models = [_make_model_info(payload["model"])]

        fake_cache = SimpleNamespace(
            get_enabled_models=lambda: cache_models,
            update_models=lambda models: None,
        )

        with patch.object(
            anthropic_messages.provider_manager,
            "get_anthropic_provider_for_model",
            AsyncMock(return_value=provider),
        ), patch.object(
            anthropic_messages.provider_manager,
            "model_cache",
            fake_cache,
        ):
            response = asyncio.run(
                anthropic_messages.create_message(
                    request_obj=request_obj,
                    request=AnthropicMessagesRequest(**payload),
                    anthropic_beta=None,
                    auth=object(),
                )
            )

        body = asyncio.run(self._collect_response_body(response))
        return response, body

    # ------------------------------------------------------------------
    # Pre-flight validation tests
    # ------------------------------------------------------------------

    def test_preflight_404_when_model_not_in_cache_and_live_refresh_empty(self):
        provider = FakeCustomProvider()
        provider.supports_api_for_model = lambda model, api: True
        provider.get_available_models = AsyncMock(return_value=[])

        response, body = self._invoke(self._payload(), provider, cache_models=[])

        self.assertEqual(response.status_code, 404)
        data = json.loads(body)
        self.assertEqual(data["error"]["type"], "not_found_error")
        self.assertIn("lmstudio:msi-ai-test-01", data["error"]["message"])
        # anthropic_messages was never called
        self.assertEqual(provider.nonstream_calls, [])
        self.assertEqual(provider.stream_calls, [])

    def test_preflight_proceeds_on_cache_hit(self):
        provider = FakeCustomProvider()
        provider.supports_api_for_model = lambda model, api: True

        response, body = self._invoke(self._payload(), provider)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(provider.nonstream_calls), 1)

    def test_preflight_live_refresh_hit_allows_request(self):
        provider = FakeCustomProvider()
        provider.supports_api_for_model = lambda model, api: True
        model_id = self._payload()["model"]
        provider.get_available_models = AsyncMock(return_value=[_make_model_info(model_id)])

        updated = []
        fake_cache = SimpleNamespace(
            get_enabled_models=lambda: [],
            update_models=lambda models: updated.extend(models),
        )

        request_id = f"req-{next(self._request_ids)}"
        request_obj = self._make_request_obj(request_id)
        asyncio.run(
            request_tracker.start_request(
                request_id=request_id,
                server="anthropic",
                endpoint="/v1/messages",
                method="POST",
                model=None,
                user_identity="test-user",
                user_type="user",
                is_streaming=False,
            )
        )

        with patch.object(
            anthropic_messages.provider_manager,
            "get_anthropic_provider_for_model",
            AsyncMock(return_value=provider),
        ), patch.object(
            anthropic_messages.provider_manager,
            "model_cache",
            fake_cache,
        ):
            response = asyncio.run(
                anthropic_messages.create_message(
                    request_obj=request_obj,
                    request=AnthropicMessagesRequest(**self._payload()),
                    anthropic_beta=None,
                    auth=object(),
                )
            )

        body = asyncio.run(self._collect_response_body(response))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(provider.nonstream_calls), 1)
        # Cache was updated with the refreshed model list
        self.assertTrue(any(m.id == model_id for m in updated))

    # ------------------------------------------------------------------
    # SDK error translation tests (via AnthropicCompatibleProvider directly)
    # ------------------------------------------------------------------

    def test_translate_api_response_validation_error_returns_502(self):
        import anthropic as anthropic_sdk
        from app.providers.anthropic_compatible import _translate_anthropic_sdk_error
        from app.providers.base import ProviderHTTPError

        fake_response = SimpleNamespace(
            status_code=200,
            headers={},
            request=SimpleNamespace(method="POST", url="http://example.com"),
        )
        exc = anthropic_sdk.APIResponseValidationError(
            response=fake_response,
            body=None,
            message="API returned an empty or malformed response (HTTP 200)",
        )
        result = _translate_anthropic_sdk_error(exc, "lmstudio:msi-ai-test-01")

        self.assertIsInstance(result, ProviderHTTPError)
        self.assertEqual(result.status_code, 502)
        self.assertIn("lmstudio:msi-ai-test-01", result.message)
        self.assertIn("unparseable", result.message)
        self.assertEqual(result.body["error"]["type"], "api_error")

    def test_translate_api_status_error_preserves_upstream_status(self):
        import anthropic as anthropic_sdk
        from app.providers.anthropic_compatible import _translate_anthropic_sdk_error
        from app.providers.base import ProviderHTTPError

        fake_response = SimpleNamespace(
            status_code=404,
            headers={},
            request=SimpleNamespace(method="POST", url="http://example.com"),
        )
        exc = anthropic_sdk.NotFoundError(
            response=fake_response,
            body={"type": "error", "error": {"type": "not_found_error", "message": "model not found"}},
            message="model not found",
        )
        result = _translate_anthropic_sdk_error(exc, "lmstudio:msi-ai-test-01")

        self.assertIsInstance(result, ProviderHTTPError)
        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.body["error"]["type"], "not_found_error")

    def test_translate_api_timeout_error_returns_504(self):
        import anthropic as anthropic_sdk
        from app.providers.anthropic_compatible import _translate_anthropic_sdk_error
        from app.providers.base import ProviderHTTPError

        exc = anthropic_sdk.APITimeoutError(request=SimpleNamespace(method="POST", url="http://example.com"))
        result = _translate_anthropic_sdk_error(exc, "lmstudio:msi-ai-test-01")

        self.assertIsInstance(result, ProviderHTTPError)
        self.assertEqual(result.status_code, 504)

    def test_nonstream_api_response_validation_error_surfaces_as_502(self):
        import anthropic as anthropic_sdk

        fake_response = SimpleNamespace(status_code=200, headers={}, request=SimpleNamespace(method="POST", url="http://example.com"))
        exc = anthropic_sdk.APIResponseValidationError(
            response=fake_response,
            body=None,
            message="API returned an empty or malformed response (HTTP 200)",
        )

        # Use FakeSdkBackedCustomProvider so the real AnthropicCompatibleProvider.anthropic_messages
        # runs and the SDK exception passes through _translate_anthropic_sdk_error.
        fake_sdk_client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(side_effect=exc))
        )
        provider = FakeSdkBackedCustomProvider(fake_sdk_client)

        response, body = self._invoke(self._payload(), provider)

        self.assertEqual(response.status_code, 502)
        data = json.loads(body)
        self.assertEqual(data["error"]["type"], "api_error")
        self.assertIn("unparseable", data["error"]["message"])

    def test_streaming_api_response_validation_error_surfaces_as_sse_error(self):
        import anthropic as anthropic_sdk

        fake_response = SimpleNamespace(status_code=200, headers={}, request=SimpleNamespace(method="POST", url="http://example.com"))
        exc = anthropic_sdk.APIResponseValidationError(
            response=fake_response,
            body=None,
            message="API returned an empty or malformed response (HTTP 200)",
        )

        async def _raising_stream(request, anthropic_beta=None):
            raise exc
            yield  # make it an async generator

        provider = FakeCustomProvider()
        provider.supports_api_for_model = lambda model, api: True
        provider.anthropic_messages_stream = _raising_stream

        response, body = self._invoke(self._payload(stream=True), provider)

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        data_line = next(l for l in body.splitlines() if l.startswith("data:"))
        payload = json.loads(data_line[len("data:"):].strip())
        self.assertEqual(payload["error"]["type"], "api_error")


if __name__ == "__main__":
    unittest.main()

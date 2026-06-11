import json
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.providers.bedrock_provider import (
    BedrockNativeIdleTimeout,
    BedrockNativePrematureEOF,
    BedrockNativeProviderError,
    BedrockProvider,
)


def _sse(event_type: str, payload: dict | None = None) -> str:
    payload = payload or {"type": event_type}
    return f"event: {event_type}\ndata: {payload}\n\n"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ScheduledEventQueue:
    def __init__(self, clock: _FakeClock, scheduled_items: list[tuple[float, tuple]]) -> None:
        self.clock = clock
        self.scheduled_items = list(scheduled_items)

    def get_nowait(self):
        if self.scheduled_items and self.scheduled_items[0][0] <= self.clock.now:
            _, item = self.scheduled_items.pop(0)
            return item
        raise queue.Empty


class _ScheduledFuture:
    def __init__(self, clock: _FakeClock, done_at: float | None = None, result=None, exc: Exception | None = None) -> None:
        self.clock = clock
        self.done_at = done_at
        self._result = result
        self._exc = exc

    def done(self) -> bool:
        return self.done_at is not None and self.clock.now >= self.done_at

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class BedrockNativeStreamConsumerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.provider = BedrockProvider.__new__(BedrockProvider)
        self.provider.native_stream_ping_interval = 1
        self.provider.full_provider_name = "bedrock:test"

    async def _collect(self, request, scheduled_items, future=None):
        clock = _FakeClock()
        event_queue = _ScheduledEventQueue(clock, scheduled_items)
        future = future or _ScheduledFuture(clock)
        chunks = []
        exc = None

        with patch.multiple(
            "app.providers.bedrock_provider",
            BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS=3,
            BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS=5,
            BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS=2,
            BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS=4,
        ), patch("app.providers.bedrock_provider.time.monotonic", side_effect=clock.monotonic), patch(
            "app.providers.bedrock_provider.asyncio.sleep",
            side_effect=clock.sleep,
        ):
            generator = self.provider._consume_native_stream_events(
                request,
                request.model,
                event_queue,
                future,
            )
            try:
                async for chunk in generator:
                    chunks.append(chunk)
            except Exception as caught:
                exc = caught

        return chunks, exc

    async def test_completes_cleanly_when_progress_events_continue(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, ("upstream_event", {"sse": _sse("message_start"), "event_type": "message_start"})),
            (0.6, ("upstream_event", {"sse": _sse("content_block_delta"), "event_type": "content_block_delta"})),
            (1.2, ("upstream_event", {"sse": _sse("message_stop"), "event_type": "message_stop"})),
            (1.2, ("done", {"terminal_event_seen": True, "terminal_event_type": "message_stop", "transport_eof_observed": False})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsNone(exc)
        self.assertEqual(
            chunks,
            [
                _sse("message_start"),
                _sse("content_block_delta"),
                "event: ping\ndata: {}\n\n",
                _sse("message_stop"),
            ],
        )

    async def test_times_out_when_only_proxy_pings_continue(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)

        chunks, exc = await self._collect(request, [])

        self.assertEqual(chunks, ["event: ping\ndata: {}\n\n", "event: ping\ndata: {}\n\n"])
        self.assertIsInstance(exc, BedrockNativeIdleTimeout)
        self.assertEqual(exc.phase, "initial")
        self.assertEqual(exc.progress_event_count, 0)
        self.assertEqual(exc.proxy_ping_count, 2)
        self.assertEqual(exc.threshold_seconds, 3)

    async def test_thinking_request_allows_longer_initial_idle_before_first_event(self):
        request = SimpleNamespace(
            model="bedrock:test/claude-sonnet",
            thinking={"type": "enabled"},
            tools=None,
        )
        scheduled_items = [
            (4.2, ("upstream_event", {"sse": _sse("message_start"), "event_type": "message_start"})),
            (4.4, ("upstream_event", {"sse": _sse("message_stop"), "event_type": "message_stop"})),
            (4.4, ("done", {"terminal_event_seen": True, "terminal_event_type": "message_stop", "transport_eof_observed": False})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsNone(exc)
        self.assertIn(_sse("message_start"), chunks)
        self.assertIn(_sse("message_stop"), chunks)

    async def test_non_thinking_request_times_out_during_same_initial_silence(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (4.2, ("upstream_event", {"sse": _sse("message_start"), "event_type": "message_start"})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertEqual(chunks, ["event: ping\ndata: {}\n\n", "event: ping\ndata: {}\n\n"])
        self.assertIsInstance(exc, BedrockNativeIdleTimeout)
        self.assertEqual(exc.phase, "initial")
        self.assertEqual(exc.threshold_seconds, 3)

    async def test_tools_request_uses_extended_midstream_idle_threshold(self):
        request = SimpleNamespace(
            model="bedrock:test/claude-sonnet",
            thinking=None,
            tools=[{"name": "lookup"}],
        )
        scheduled_items = [
            (0.0, ("upstream_event", {"sse": _sse("message_start"), "event_type": "message_start"})),
            (3.5, ("upstream_event", {"sse": _sse("message_stop"), "event_type": "message_stop"})),
            (3.5, ("done", {"terminal_event_seen": True, "terminal_event_type": "message_stop", "transport_eof_observed": False})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsNone(exc)
        self.assertEqual(chunks.count("event: ping\ndata: {}\n\n"), 3)
        self.assertEqual(chunks[-1], _sse("message_stop"))

    async def test_done_without_terminal_event_raises_premature_eof(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, ("done", {"terminal_event_seen": False, "terminal_event_type": None, "transport_eof_observed": True})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertEqual(chunks, [])
        self.assertIsInstance(exc, BedrockNativePrematureEOF)
        self.assertTrue(exc.transport_eof_observed)

    async def test_worker_error_raises_provider_error_with_mapped_classification(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, ("error", {"code": "ValidationException", "message": "Bad request"})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertEqual(chunks, [])
        self.assertIsInstance(exc, BedrockNativeProviderError)
        self.assertEqual(exc.error_code, "ValidationException")
        self.assertEqual(exc.status_code, 400)
        self.assertEqual(exc.body["error"]["type"], "invalid_request_error")

    async def test_worker_stopped_early_is_treated_as_clean_completion(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (
                0.0,
                (
                    "done",
                    {
                        "terminal_event_seen": False,
                        "terminal_event_type": None,
                        "transport_eof_observed": False,
                        "event_count": 0,
                        "worker_stopped_early": True,
                    },
                ),
            ),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertEqual(chunks, [])
        self.assertIsNone(exc)

    # ------------------------------------------------------------------
    # Helpers for building worker queue entries
    # ------------------------------------------------------------------

    def _upstream(self, event_type: str, event_data: dict) -> tuple:
        sse = f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False, separators=(',', ':'))}\n\n"
        return ("upstream_event", {"sse": sse, "event_type": event_type, "event_data": event_data})

    # ------------------------------------------------------------------
    # Graceful finalization on mid-stream errors
    # ------------------------------------------------------------------

    async def test_provider_error_mid_stream_emits_synthetic_terminal_events(self):
        """Regression: ProtocolError mid-stream should close open blocks + message_stop."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
            (0.1, self._upstream("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            (0.2, self._upstream("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}})),
            (0.3, ("error", {"code": "internal_error", "message": "Response ended prematurely"})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsInstance(exc, BedrockNativeProviderError)
        self.assertEqual(exc.error_code, "internal_error")
        body = "".join(chunks)
        self.assertIn("event: content_block_stop", body)
        self.assertIn('"index":0', body)
        self.assertIn("event: message_delta", body)
        self.assertIn('"stop_reason":"end_turn"', body)
        self.assertIn("event: message_stop", body)
        # Synthetic events must precede the raise (all in chunks, not after exc)
        self.assertGreater(len(chunks), 3)

    async def test_premature_eof_after_partial_blocks_emits_synthetic_terminal_events(self):
        """Done entry without terminal event should finalize open blocks."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
            (0.1, self._upstream("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            (0.2, ("done", {"terminal_event_seen": False, "terminal_event_type": None, "transport_eof_observed": True})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsInstance(exc, BedrockNativePrematureEOF)
        body = "".join(chunks)
        self.assertIn("event: content_block_stop", body)
        self.assertIn("event: message_delta", body)
        self.assertIn("event: message_stop", body)

    async def test_error_before_message_start_does_not_synthesize(self):
        """If message_start was never sent, no synthetic terminal events should be emitted."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, ("error", {"code": "internal_error", "message": "Pre-stream failure"})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertEqual(chunks, [])
        self.assertIsInstance(exc, BedrockNativeProviderError)

    async def test_multiple_open_blocks_emit_in_index_order(self):
        """Open blocks at indices 0, 2 (1 already closed) should emit content_block_stop in order."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
            (0.1, self._upstream("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            (0.2, self._upstream("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})),
            (0.3, self._upstream("content_block_stop", {"type": "content_block_stop", "index": 1})),
            (0.4, self._upstream("content_block_start", {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}})),
            (0.5, ("error", {"code": "internal_error", "message": "transport drop"})),
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsInstance(exc, BedrockNativeProviderError)
        stops_0 = [c for c in chunks if '"content_block_stop"' in c and '"index":0' in c]
        stops_1 = [c for c in chunks if '"content_block_stop"' in c and '"index":1' in c]
        stops_2 = [c for c in chunks if '"content_block_stop"' in c and '"index":2' in c]
        # Indices 0 and 2 synthesized exactly once; index 1 from upstream only (not duplicated)
        self.assertEqual(len(stops_0), 1)
        self.assertEqual(len(stops_1), 1)
        self.assertEqual(len(stops_2), 1)

    async def test_idle_timeout_emits_synthetic_terminal_events_when_blocks_open(self):
        """Idle timeout after partial content should finalize open blocks before raising."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        scheduled_items = [
            (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
            (0.1, self._upstream("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            # No more events — midstream idle timeout fires at 2.1s
        ]

        chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsInstance(exc, BedrockNativeIdleTimeout)
        body = "".join(chunks)
        self.assertIn("event: content_block_stop", body)
        self.assertIn('"index":0', body)
        self.assertIn("event: message_delta", body)
        self.assertIn("event: message_stop", body)

    # ------------------------------------------------------------------
    # Diagnostic drop log + pre-output retry behaviour
    # ------------------------------------------------------------------

    async def test_drop_log_includes_aws_request_id_and_byte_counts(self):
        """Worker error queue entries should produce a structured DROP record."""
        request = SimpleNamespace(
            model="bedrock:test/claude-sonnet",
            thinking=None,
            tools=None,
            max_tokens=4096,
        )
        scheduled_items = [
            (
                0.0,
                self._upstream(
                    "message_start",
                    {"type": "message_start", "message": {}},
                ),
            ),
            (
                0.1,
                (
                    "error",
                    {
                        "code": "internal_error",
                        "message": "Response ended prematurely",
                        "error_class": "protocol_error",
                        "aws_request_id": "REQ-ABC-123",
                        "output_bytes_received": 4096,
                    },
                ),
            ),
        ]

        with self.assertLogs("app.providers.bedrock_provider", level="WARNING") as cm:
            chunks, exc = await self._collect(request, scheduled_items)

        self.assertIsInstance(exc, BedrockNativeProviderError)
        drop_lines = [
            record.getMessage()
            for record in cm.records
            if "[BEDROCK STREAM NATIVE DROP]" in record.getMessage()
        ]
        self.assertTrue(drop_lines, "expected a [BEDROCK STREAM NATIVE DROP] log line")
        drop = drop_lines[0]
        self.assertIn("aws_request_id=REQ-ABC-123", drop)
        self.assertIn("error_class=protocol_error", drop)
        self.assertIn("output_bytes_received=4096", drop)
        self.assertIn("max_tokens=4096", drop)
        self.assertIn("progress_event_count=1", drop)

    async def test_pre_output_drop_triggers_single_retry(self):
        """A drop with progress_event_count == 0 should restart the worker once."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        clock = _FakeClock()

        first_queue = _ScheduledEventQueue(
            clock,
            [
                (
                    0.0,
                    (
                        "error",
                        {
                            "code": "internal_error",
                            "message": "Response ended prematurely",
                            "error_class": "protocol_error",
                            "aws_request_id": "REQ-FIRST",
                            "output_bytes_received": 0,
                        },
                    ),
                ),
            ],
        )
        first_future = _ScheduledFuture(clock)

        second_queue = _ScheduledEventQueue(
            clock,
            [
                (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
                (0.1, self._upstream("message_stop", {"type": "message_stop"})),
                (
                    0.1,
                    (
                        "done",
                        {
                            "terminal_event_seen": True,
                            "terminal_event_type": "message_stop",
                            "transport_eof_observed": False,
                            "aws_request_id": "REQ-SECOND",
                            "output_bytes_received": 64,
                        },
                    ),
                ),
            ],
        )
        second_future = _ScheduledFuture(clock)

        restart_calls = []

        def restart_worker():
            restart_calls.append(1)
            return second_queue, second_future, None

        chunks = []
        exc = None
        with patch.multiple(
            "app.providers.bedrock_provider",
            BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS=3,
            BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS=5,
            BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS=2,
            BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS=4,
            BEDROCK_NATIVE_RETRY_ON_PRE_OUTPUT_DROP=True,
        ), patch("app.providers.bedrock_provider.time.monotonic", side_effect=clock.monotonic), patch(
            "app.providers.bedrock_provider.asyncio.sleep",
            side_effect=clock.sleep,
        ):
            generator = self.provider._consume_native_stream_events(
                request,
                request.model,
                first_queue,
                first_future,
                restart_worker=restart_worker,
            )
            try:
                async for chunk in generator:
                    chunks.append(chunk)
            except Exception as caught:
                exc = caught

        self.assertIsNone(exc)
        self.assertEqual(len(restart_calls), 1)
        body = "".join(chunks)
        self.assertIn("event: message_start", body)
        self.assertIn("event: message_stop", body)

    async def test_post_output_drop_does_not_retry(self):
        """A drop after message_start has been forwarded must not retry."""
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        clock = _FakeClock()

        event_queue = _ScheduledEventQueue(
            clock,
            [
                (0.0, self._upstream("message_start", {"type": "message_start", "message": {}})),
                (
                    0.1,
                    self._upstream(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "hi"},
                        },
                    ),
                ),
                (
                    0.2,
                    (
                        "error",
                        {
                            "code": "internal_error",
                            "message": "Response ended prematurely",
                            "error_class": "protocol_error",
                            "aws_request_id": "REQ-POST",
                            "output_bytes_received": 1024,
                        },
                    ),
                ),
            ],
        )
        future = _ScheduledFuture(clock)

        restart_calls = []

        def restart_worker():
            restart_calls.append(1)
            return event_queue, future, None

        chunks = []
        exc = None
        with patch.multiple(
            "app.providers.bedrock_provider",
            BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS=3,
            BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS=5,
            BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS=2,
            BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS=4,
            BEDROCK_NATIVE_RETRY_ON_PRE_OUTPUT_DROP=True,
        ), patch("app.providers.bedrock_provider.time.monotonic", side_effect=clock.monotonic), patch(
            "app.providers.bedrock_provider.asyncio.sleep",
            side_effect=clock.sleep,
        ):
            generator = self.provider._consume_native_stream_events(
                request,
                request.model,
                event_queue,
                future,
                restart_worker=restart_worker,
            )
            try:
                async for chunk in generator:
                    chunks.append(chunk)
            except Exception as caught:
                exc = caught

        self.assertIsInstance(exc, BedrockNativeProviderError)
        self.assertEqual(restart_calls, [])

    async def test_future_done_requires_grace_window_before_premature_eof(self):
        request = SimpleNamespace(model="bedrock:test/claude-sonnet", thinking=None, tools=None)
        future = _ScheduledFuture(_FakeClock(), done_at=0.0)

        clock = future.clock
        event_queue = _ScheduledEventQueue(
            clock,
            [
                (0.05, ("done", {"terminal_event_seen": True, "terminal_event_type": "message_stop", "transport_eof_observed": False})),
            ],
        )
        chunks = []
        exc = None

        with patch.multiple(
            "app.providers.bedrock_provider",
            BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS=3,
            BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS=5,
            BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS=2,
            BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS=4,
        ), patch("app.providers.bedrock_provider.time.monotonic", side_effect=clock.monotonic), patch(
            "app.providers.bedrock_provider.asyncio.sleep",
            side_effect=clock.sleep,
        ):
            generator = self.provider._consume_native_stream_events(
                request,
                request.model,
                event_queue,
                future,
            )
            try:
                async for chunk in generator:
                    chunks.append(chunk)
            except Exception as caught:
                exc = caught

        self.assertEqual(chunks, [])
        self.assertIsNone(exc)


class BedrockConverseStreamTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, provider, request):
        chunks = []
        async for chunk in provider.anthropic_messages_stream(request):
            chunks.append(chunk)
        return chunks

    async def test_suppresses_late_transport_error_after_message_stop(self):
        provider = BedrockProvider.__new__(BedrockProvider)
        provider.debug = False
        provider.bedrock_runtime = SimpleNamespace(
            converse_stream=lambda **kwargs: {"stream": object()}
        )
        provider._resolve_bedrock_anthropic_model_id = lambda model: "meta.llama3-1-8b-instruct-v1:0"
        provider._build_bedrock_anthropic_args = lambda request, anthropic_beta: {}

        async def fake_async_iterate_catching(stream):
            yield {"messageStart": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {"_bedrock_stream_error": RuntimeError("Response ended prematurely")}

        provider._async_iterate_catching = fake_async_iterate_catching
        request = SimpleNamespace(model="bedrock:test/llama", thinking=None, tools=None)

        chunks = await self._collect(provider, request)
        body = "".join(chunks)

        self.assertIn("event: message_start", body)
        self.assertIn("event: message_stop", body)
        self.assertNotIn("event: error", body)

    async def test_preserves_error_when_terminal_message_stop_was_never_seen(self):
        provider = BedrockProvider.__new__(BedrockProvider)
        provider.debug = False
        provider.bedrock_runtime = SimpleNamespace(
            converse_stream=lambda **kwargs: {"stream": object()}
        )
        provider._resolve_bedrock_anthropic_model_id = lambda model: "meta.llama3-1-8b-instruct-v1:0"
        provider._build_bedrock_anthropic_args = lambda request, anthropic_beta: {}

        async def fake_async_iterate_catching(stream):
            yield {"messageStart": {}}
            yield {"_bedrock_stream_error": RuntimeError("Response ended prematurely")}

        provider._async_iterate_catching = fake_async_iterate_catching
        request = SimpleNamespace(model="bedrock:test/llama", thinking=None, tools=None)

        chunks = await self._collect(provider, request)
        body = "".join(chunks)

        self.assertIn("event: error", body)


if __name__ == "__main__":
    unittest.main()

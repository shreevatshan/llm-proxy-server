import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.context import get_current

from app.routes import stream_utils


class _FakeRequest:
    def __init__(self, disconnect_after: int | None = None) -> None:
        self.state = SimpleNamespace()
        self._disconnect_after = disconnect_after
        self._disconnect_checks = 0

    async def is_disconnected(self) -> bool:
        self._disconnect_checks += 1
        if self._disconnect_after is None:
            return False
        return self._disconnect_checks >= self._disconnect_after


class _PendingAsyncIterator:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Future()

    async def aclose(self):
        self.closed = True


async def _collect_chunks(generator):
    chunks = []
    async for chunk in generator:
        chunks.append(chunk)
    return chunks


class ChunkBudgetTests(unittest.TestCase):
    def test_first_chunk_uses_ttft_budget(self):
        with patch.object(stream_utils, "STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", 45), patch.object(
            stream_utils, "STREAM_CHUNK_TIMEOUT_SECONDS", 120
        ):
            self.assertEqual(stream_utils._chunk_budget_seconds(0), 45)
            self.assertEqual(stream_utils._chunk_budget_seconds(1), 120)
            self.assertEqual(stream_utils._chunk_budget_seconds(99), 120)

    def test_ttft_budget_capped_at_per_chunk_budget(self):
        # A tiny per-chunk timeout (e.g. in tests) bounds the first chunk too.
        with patch.object(stream_utils, "STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", 45), patch.object(
            stream_utils, "STREAM_CHUNK_TIMEOUT_SECONDS", 1
        ):
            self.assertEqual(stream_utils._chunk_budget_seconds(0), 1)


class StreamUtilsTests(unittest.IsolatedAsyncioTestCase):
    async def test_set_request_tracking_outcome_preserves_error_over_completed(self):
        request = _FakeRequest()

        stream_utils.set_request_tracking_outcome(
            request,
            status="errored",
            termination_reason="bedrock_native_idle_timeout",
            error="boom",
        )
        stream_utils.set_request_tracking_outcome(
            request,
            status="completed",
            termination_reason="completed",
        )

        self.assertEqual(request.state.tracking_final["status"], "errored")
        self.assertEqual(
            request.state.tracking_final["termination_reason"],
            "bedrock_native_idle_timeout",
        )

    async def test_anthropic_wrapper_sets_completed_on_normal_finish(self):
        request = _FakeRequest()

        async def generator():
            yield stream_utils.format_anthropic_sse_event("message_start", {"type": "message_start"})

        chunks = await _collect_chunks(
            stream_utils.anthropic_stream_with_context_and_timeout(
                generator(),
                get_current(),
                request,
                timeout=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            request.state.tracking_final,
            {
                "status": "completed",
                "termination_reason": "completed",
                "error": None,
            },
        )

    async def test_anthropic_wrapper_detects_client_disconnect_while_chunk_pending(self):
        request = _FakeRequest(disconnect_after=2)
        generator = _PendingAsyncIterator()

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.01), patch.object(
            stream_utils,
            "STREAM_CHUNK_TIMEOUT_SECONDS",
            1,
        ):
            chunks = await _collect_chunks(
                stream_utils.anthropic_stream_with_context_and_timeout(
                    generator,
                    get_current(),
                    request,
                    timeout=1,
                )
            )

        self.assertEqual(chunks, [])
        self.assertTrue(generator.closed)
        self.assertEqual(
            request.state.tracking_final,
            {
                "status": "cancelled",
                "termination_reason": "client_disconnect",
                "error": None,
            },
        )

    async def test_anthropic_wrapper_emits_first_chunk_timeout_error(self):
        # A provider that never emits a first event hits the TTFT (first-chunk)
        # budget and is reported as `first_chunk_timeout`, distinct from a
        # mid-stream `chunk_timeout`.
        request = _FakeRequest()
        generator = _PendingAsyncIterator()

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.01), patch.object(
            stream_utils,
            "STREAM_FIRST_CHUNK_TIMEOUT_SECONDS",
            0.03,
        ), patch.object(
            stream_utils,
            "STREAM_CHUNK_TIMEOUT_SECONDS",
            10,
        ):
            chunks = await _collect_chunks(
                stream_utils.anthropic_stream_with_context_and_timeout(
                    generator,
                    get_current(),
                    request,
                    timeout=1,
                )
            )

        self.assertEqual(len(chunks), 1)
        self.assertIn("event: error", chunks[0])
        self.assertIn("Provider response timeout (0.03s)", chunks[0])
        self.assertTrue(generator.closed)
        self.assertEqual(request.state.tracking_final["status"], "errored")
        self.assertEqual(
            request.state.tracking_final["termination_reason"], "first_chunk_timeout"
        )

    async def test_anthropic_wrapper_emits_chunk_timeout_after_first_chunk(self):
        # Once a first chunk has been delivered, a subsequent stall uses the full
        # per-chunk budget and is reported as `chunk_timeout`.
        request = _FakeRequest()

        class _OneChunkThenStall:
            def __init__(self):
                self.closed = False
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._yielded:
                    self._yielded = True
                    return stream_utils.format_anthropic_sse_event(
                        "message_start", {"type": "message_start"}
                    )
                await asyncio.Future()  # stall forever on the second chunk

            async def aclose(self):
                self.closed = True

        generator = _OneChunkThenStall()

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.01), patch.object(
            stream_utils,
            "STREAM_FIRST_CHUNK_TIMEOUT_SECONDS",
            10,
        ), patch.object(
            stream_utils,
            "STREAM_CHUNK_TIMEOUT_SECONDS",
            0.03,
        ):
            chunks = await _collect_chunks(
                stream_utils.anthropic_stream_with_context_and_timeout(
                    generator,
                    get_current(),
                    request,
                    timeout=1,
                )
            )

        self.assertEqual(len(chunks), 2)
        self.assertIn("event: message_start", chunks[0])
        self.assertIn("event: error", chunks[1])
        self.assertIn("Provider response timeout (0.03s)", chunks[1])
        self.assertTrue(generator.closed)
        self.assertEqual(request.state.tracking_final["status"], "errored")
        self.assertEqual(request.state.tracking_final["termination_reason"], "chunk_timeout")

    async def test_anthropic_wrapper_emits_stream_timeout_error(self):
        request = _FakeRequest()
        generator = _PendingAsyncIterator()

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.01), patch.object(
            stream_utils,
            "STREAM_CHUNK_TIMEOUT_SECONDS",
            10,
        ):
            chunks = await _collect_chunks(
                stream_utils.anthropic_stream_with_context_and_timeout(
                    generator,
                    get_current(),
                    request,
                    timeout=0.03,
                )
            )

        self.assertEqual(len(chunks), 1)
        self.assertIn("event: error", chunks[0])
        self.assertIn("Stream timeout exceeded (0.03s)", chunks[0])
        self.assertTrue(generator.closed)
        self.assertEqual(request.state.tracking_final["status"], "errored")
        self.assertEqual(request.state.tracking_final["termination_reason"], "stream_timeout")

    async def test_prior_error_outcome_is_not_overwritten_by_wrapper_cleanup(self):
        request = _FakeRequest()
        stream_utils.set_request_tracking_outcome(
            request,
            status="errored",
            termination_reason="bedrock_native_idle_timeout",
            error="boom",
        )

        async def generator():
            yield stream_utils.format_anthropic_sse_event("message_stop", {"type": "message_stop"})

        chunks = await _collect_chunks(
            stream_utils.anthropic_stream_with_context_and_timeout(
                generator(),
                get_current(),
                request,
                timeout=1,
            )
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(request.state.tracking_final["status"], "errored")
        self.assertEqual(
            request.state.tracking_final["termination_reason"],
            "bedrock_native_idle_timeout",
        )

    async def test_anthropic_wrapper_arms_and_disarms_stall_watchdog(self):
        # On a normal finish the wrapper must arm the stall watchdog on entry
        # and disarm it in finally, so a completed stream leaves no live deadline.
        request = _FakeRequest()

        async def generator():
            yield stream_utils.format_anthropic_sse_event("message_start", {"type": "message_start"})

        with patch.object(stream_utils.diagnostics, "arm") as arm, \
                patch.object(stream_utils.diagnostics, "disarm") as disarm:
            await _collect_chunks(
                stream_utils.anthropic_stream_with_context_and_timeout(
                    generator(),
                    get_current(),
                    request,
                    timeout=1,
                )
            )

        self.assertEqual(arm.call_count, 1)
        self.assertEqual(disarm.call_count, 1)
        # Same key armed and disarmed.
        self.assertEqual(arm.call_args.args[0], disarm.call_args.args[0])

    async def test_chat_wrapper_arms_and_disarms_stall_watchdog(self):
        # Mirror of the anthropic test for the OpenAI-style timeout wrapper: it
        # must arm the watchdog on entry and disarm it in finally on a clean finish.
        request = _FakeRequest()

        async def generator():
            yield 'data: {"id":"chunk-1"}\n\n'

        with patch.object(stream_utils.diagnostics, "arm") as arm, \
                patch.object(stream_utils.diagnostics, "disarm") as disarm:
            await _collect_chunks(
                stream_utils.stream_with_context_and_timeout(
                    generator(),
                    get_current(),
                    request,
                    timeout=1,
                )
            )

        self.assertEqual(arm.call_count, 1)
        self.assertEqual(disarm.call_count, 1)
        # Same key armed and disarmed.
        self.assertEqual(arm.call_args.args[0], disarm.call_args.args[0])

    async def test_openai_wrapper_preserves_final_chunk_when_timeout_boundary_is_crossed(self):
        request = _FakeRequest()

        async def generator():
            yield 'data: {"id":"chunk-1"}\n\n'

        chunks = await _collect_chunks(
            stream_utils.stream_with_context_and_timeout(
                generator(),
                get_current(),
                request,
                timeout=0.01,
                request_started_at=time.monotonic() - 1,
            )
        )

        self.assertEqual(chunks[0], 'data: {"id":"chunk-1"}\n\n')
        self.assertIn("Stream timeout exceeded (0.01s)", chunks[1])
        self.assertEqual(chunks[2], "data: [DONE]\n\n")

    async def test_openai_wrapper_cancel_mid_chunk_closes_generator_without_error(self):
        # Regression: cancelling while blocked in __anext__ must cancel the
        # in-flight chunk task BEFORE aclose(), otherwise aclose() raises
        # "asynchronous generator is already running" and the socket leaks.
        request = _FakeRequest()
        closed = {"value": False}
        started = asyncio.Event()

        async def generator():
            try:
                yield 'data: {"id":"chunk-1"}\n\n'
                started.set()
                await asyncio.sleep(3600)  # park inside the second __anext__
                yield 'data: {"id":"never"}\n\n'
            finally:
                closed["value"] = True

        agen = stream_utils._stream_with_timeout_and_disconnect(
            generator(), request, timeout=600
        )

        async def consume():
            async for _ in agen:
                pass

        task = asyncio.create_task(consume())
        await started.wait()
        await asyncio.sleep(0)  # let the wrapper enter the wait for chunk 2

        with self.assertNoLogs(stream_utils.logger, level="WARNING"):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(closed["value"], "generator should be closed on cancel")

    async def test_openai_wrapper_detects_client_disconnect_while_chunk_pending(self):
        # Regression: the chat path must poll for client disconnect DURING the
        # per-chunk wait, not only after a chunk arrives. Otherwise a request
        # whose first/next chunk is slow stays "active" with no output until the
        # full STREAM_CHUNK_TIMEOUT_SECONDS budget elapses. The Anthropic path
        # already did this; the chat path was brought to parity.
        request = _FakeRequest(disconnect_after=1)
        closed = {"value": False}

        async def generator():
            try:
                # Never yields — emulates a provider that opened the stream
                # (headers sent, 200 OK logged) but stalls before the first event.
                await asyncio.Future()
                yield 'data: {"id":"never"}\n\n'  # pragma: no cover
            finally:
                closed["value"] = True

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.05), \
                patch.object(stream_utils, "STREAM_CHUNK_TIMEOUT_SECONDS", 600), \
                patch.object(stream_utils, "STREAM_EARLY_DISCONNECT_CHECK_CHUNKS", 0):
            chunks = await asyncio.wait_for(
                _collect_chunks(
                    stream_utils._stream_with_timeout_and_disconnect(
                        generator(), request, timeout=600
                    )
                ),
                timeout=5,
            )

        # No chunks emitted, and the wrapper exited promptly (well under the
        # 600s chunk budget) because it noticed the disconnect mid-wait.
        self.assertEqual(chunks, [])
        self.assertTrue(closed["value"], "generator should be closed on disconnect")


if __name__ == "__main__":
    unittest.main()

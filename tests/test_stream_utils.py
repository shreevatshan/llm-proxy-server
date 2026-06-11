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

    async def test_anthropic_wrapper_emits_chunk_timeout_error(self):
        request = _FakeRequest()
        generator = _PendingAsyncIterator()

        with patch.object(stream_utils, "STREAM_DISCONNECT_POLL_SECONDS", 0.01), patch.object(
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

        self.assertEqual(len(chunks), 1)
        self.assertIn("event: error", chunks[0])
        self.assertIn("Provider response timeout (0.03s)", chunks[0])
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


if __name__ == "__main__":
    unittest.main()

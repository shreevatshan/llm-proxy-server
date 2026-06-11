import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.main import _add_request_tracking
from app.request_tracker import ActiveRequest, RequestTracker, request_tracker


class RequestTrackerTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_request_broadcast_includes_termination_reason_and_error(self):
        tracker = RequestTracker()
        tracker._broadcast_raw = AsyncMock()
        tracker._active["req-1"] = ActiveRequest(
            request_id="req-1",
            server="anthropic",
            endpoint="/v1/messages",
            method="POST",
            model="bedrock:test/claude-sonnet",
            user_identity="user",
            user_type="user",
            is_streaming=True,
            start_time=0.0,
        )

        await tracker.end_request(
            "req-1",
            status="errored",
            termination_reason="bedrock_native_idle_timeout",
            error="upstream stalled",
        )

        tracker._broadcast_raw.assert_awaited_once()
        event_type, payload = tracker._broadcast_raw.await_args.args
        self.assertEqual(event_type, "request_errored")
        self.assertEqual(payload["termination_reason"], "bedrock_native_idle_timeout")
        self.assertEqual(payload["error"], "upstream stalled")


class RequestTrackingMiddlewareTests(unittest.TestCase):
    def test_streaming_response_uses_tracking_final_error_outcome(self):
        start_request = AsyncMock()
        end_request = AsyncMock()

        app = FastAPI()
        with patch.object(request_tracker, "start_request", start_request), patch.object(
            request_tracker,
            "end_request",
            end_request,
        ):
            _add_request_tracking(app, "anthropic")

            @app.get("/v1/messages")
            async def route(request: Request):
                request.state.tracking_final = {
                    "status": "errored",
                    "termination_reason": "bedrock_native_idle_timeout",
                    "error": "stalled upstream",
                }

                async def body():
                    yield b"hello"

                return StreamingResponse(body(), media_type="text/plain")

            with TestClient(app) as client:
                response = client.get("/v1/messages")

        self.assertEqual(response.status_code, 200)
        end_request.assert_awaited_once()
        kwargs = end_request.await_args.kwargs
        self.assertEqual(kwargs["status"], "errored")
        self.assertEqual(kwargs["termination_reason"], "bedrock_native_idle_timeout")
        self.assertEqual(kwargs["error"], "stalled upstream")

    def test_streaming_response_without_tracking_final_falls_back_to_completed(self):
        start_request = AsyncMock()
        end_request = AsyncMock()

        app = FastAPI()
        with patch.object(request_tracker, "start_request", start_request), patch.object(
            request_tracker,
            "end_request",
            end_request,
        ):
            _add_request_tracking(app, "anthropic")

            @app.get("/v1/messages")
            async def route():
                async def body():
                    yield b"hello"

                return StreamingResponse(body(), media_type="text/plain")

            with TestClient(app) as client:
                response = client.get("/v1/messages")

        self.assertEqual(response.status_code, 200)
        end_request.assert_awaited_once()
        kwargs = end_request.await_args.kwargs
        self.assertEqual(kwargs["status"], "completed")
        self.assertNotIn("termination_reason", kwargs)


if __name__ == "__main__":
    unittest.main()

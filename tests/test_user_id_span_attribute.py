"""Tests for the ``user.id`` span attribute.

The attribute has to land on the spans traceloop's instrumentation creates — the
ones already carrying the gen_ai token usage — which route code cannot reach. It
gets there via a SpanProcessor.on_start hook reading a request-scoped holder, so
these tests pin the two things that make that work: the processor itself, and the
contextvars propagation across the ASGI task boundaries the holder has to cross.
"""

import asyncio
import unittest

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.testclient import TestClient

from app.auth.admin import AdminUser
from app.auth.cache import CachedAPIKey, CachedUser
from app.auth.middleware import _resolve_identity
from app.auth.models import APIKey, User
from app.tracing import (
    UserIdSpanProcessor,
    begin_trace_identity,
    set_trace_user,
    trace_identity,
)


def _local_provider():
    """A throwaway provider wired exactly like production: exporter first, ours after.

    Registration order matters enough to encode it here — in the real server the
    processor is added by init_tracing() *after* Traceloop.init() has already
    installed its exporting processor.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.add_span_processor(UserIdSpanProcessor())
    return provider, exporter


class UserIdSpanProcessorTests(unittest.TestCase):
    def setUp(self):
        self.provider, self.exporter = _local_provider()
        self.tracer = self.provider.get_tracer("test")
        self._token = trace_identity.set(None)

    def tearDown(self):
        trace_identity.reset(self._token)

    def _attrs(self, name):
        for span in self.exporter.get_finished_spans():
            if span.name == name:
                return dict(span.attributes)
        self.fail(f"span {name!r} was never exported")

    def test_stamps_username_alongside_usage_attributes(self):
        """The attribute and the gen_ai usage must end up on the same span."""
        begin_trace_identity(None)
        set_trace_user("alice")

        with self.tracer.start_as_current_span("openai.chat") as span:
            # traceloop fills these in after on_start has already run
            span.set_attribute("gen_ai.usage.input_tokens", 10)

        attrs = self._attrs("openai.chat")
        self.assertEqual(attrs["user.id"], "alice")
        self.assertEqual(attrs["gen_ai.usage.input_tokens"], 10)

    def test_absent_when_unauthenticated(self):
        with self.tracer.start_as_current_span("startup.work"):
            pass
        self.assertNotIn("user.id", self._attrs("startup.work"))

    def test_holder_without_username_is_ignored(self):
        """begin_trace_identity runs before auth; spans until then carry nothing."""
        begin_trace_identity(None)
        with self.tracer.start_as_current_span("pre-auth"):
            pass
        self.assertNotIn("user.id", self._attrs("pre-auth"))

    def test_on_start_never_propagates(self):
        """A raising processor would fail the LLM call that is creating the span.

        SynchronousMultiSpanProcessor.on_start fans out with no try/except of its
        own, so an exception here escapes Span.start().
        """
        class Exploding:
            def get(self, _key):
                raise RuntimeError("boom")

            def __bool__(self):
                return True

        trace_identity.set(Exploding())
        with self.tracer.start_as_current_span("survives"):
            pass
        self.assertNotIn("user.id", self._attrs("survives"))

    def test_backfills_the_open_span(self):
        """The server and auth spans start before the username is known."""
        with self.tracer.start_as_current_span("auth.authenticate") as span:
            begin_trace_identity(span)
            set_trace_user("alice")
        self.assertEqual(self._attrs("auth.authenticate")["user.id"], "alice")

    def test_backfills_server_span_once(self):
        """Azure has no auth span, so the captured and current span coincide."""
        with self.tracer.start_as_current_span("server") as server:
            begin_trace_identity(server)
            set_trace_user("alice")
        self.assertEqual(self._attrs("server")["user.id"], "alice")

    def test_empty_username_is_not_recorded(self):
        begin_trace_identity(None)
        set_trace_user("")
        with self.tracer.start_as_current_span("blank"):
            pass
        self.assertNotIn("user.id", self._attrs("blank"))


class ResolveIdentityTests(unittest.TestCase):
    """Pins the strings the tracker and the span attribute both report."""

    def test_all_auth_types(self):
        cases = [
            (AdminUser("root", "root@example.com"), ("root", "admin")),
            (User(id=3, username="alice"), ("alice", "user")),
            (
                CachedUser(id=3, username="alice", email="a@example.com", is_active=True),
                ("alice", "user"),
            ),
            (APIKey(id=7, user_id=3, name="key"), ("key:7", "api_key")),
            (
                CachedAPIKey(
                    id=7, user_id=3, api_key="k", name="key", is_active=True,
                    username="alice",
                ),
                ("alice", "api_key"),
            ),
            (
                CachedAPIKey(id=7, user_id=3, api_key="k", name="key", is_active=True),
                ("key:7", "api_key"),
            ),
            (object(), (None, None)),
        ]
        for auth_result, expected in cases:
            with self.subTest(type=type(auth_result).__name__, expected=expected):
                self.assertEqual(_resolve_identity(auth_result), expected)

    def test_admin_is_matched_before_api_key(self):
        """AdminUser.id is None, so the APIKey branch would render it "key:None"."""
        identity, kind = _resolve_identity(AdminUser("root", "root@example.com"))
        self.assertEqual(identity, "root")
        self.assertNotEqual(identity, "key:None")

    def test_empty_username_is_preserved_not_coerced(self):
        """The tracker is called with "" today; the refactor must not change that."""
        self.assertEqual(_resolve_identity(User(id=3, username="")), ("", "user"))


class ContextPropagationTests(unittest.TestCase):
    """The holder has to survive the task boundaries a real request crosses.

    BaseHTTPMiddleware runs routing, the dependencies, the endpoint and the body
    iteration in one *inner* task whose contextvars are a copy, and streaming pulls
    each chunk in a further asyncio.create_task. A plain string ContextVar set in
    the auth dependency is invisible back in the outer task, which is why the
    identity is carried in a dict that is mutated in place.
    """

    def test_username_reaches_streamed_provider_call_and_outer_task(self):
        seen = {}
        app = FastAPI()

        @app.middleware("http")
        async def tracking(request, call_next):
            begin_trace_identity(None)          # outer task
            response = await call_next(request)
            seen["outer_after_call_next"] = (trace_identity.get() or {}).get("id")

            original = response.body_iterator

            async def wrapped():
                async for chunk in original:
                    yield chunk
                # where end_request() and its DB spans run
                seen["outer_body_iter_end"] = (trace_identity.get() or {}).get("id")

            response.body_iterator = wrapped()
            return response

        async def auth_dep():
            set_trace_user("alice")             # inner task
            return "alice"

        async def provider_stream():
            seen["provider_call"] = (trace_identity.get() or {}).get("id")
            yield b"chunk"

        async def chunked(gen):
            iterator = gen.__aiter__()
            while True:
                task = asyncio.create_task(iterator.__anext__())
                try:
                    yield await task
                except StopAsyncIteration:
                    break

        @app.get("/stream")
        async def stream(_user=Depends(auth_dep)):
            seen["handler"] = (trace_identity.get() or {}).get("id")
            return StreamingResponse(chunked(provider_stream()))

        response = TestClient(app).get("/stream")
        self.assertEqual(response.status_code, 200)

        # The LLM span is created here — this is the one that must not regress.
        self.assertEqual(seen["provider_call"], "alice")
        self.assertEqual(seen["handler"], "alice")
        # Outer-task spans (ASGI "http send", end_request's DB work) see it too.
        self.assertEqual(seen["outer_after_call_next"], "alice")
        self.assertEqual(seen["outer_body_iter_end"], "alice")

    def test_holder_is_optional(self):
        """The management app has no tracking middleware, so there is no holder."""
        trace_identity.set(None)
        set_trace_user("alice")
        self.assertEqual((trace_identity.get() or {}).get("id"), "alice")


if __name__ == "__main__":
    unittest.main()


class MiddlewareOrderTests(unittest.TestCase):
    """The nesting the holder depends on, pinned per app.

    Each BaseHTTPMiddleware runs its downstream in a task whose contextvars are a
    copy, so a middleware registered *outside* the tracking one never sees the
    username written into the holder — its spans, including the ASGI "http send"
    spans, would lose user.id.

    Starlette's add_middleware inserts at index 0 and the stack is built so index 0
    is outermost, meaning user_middleware reads outermost-first. The invariant is
    therefore: track_requests at index 0, every other BaseHTTPMiddleware after it.

    OpenTelemetryMiddleware is deliberately not in this list — instrument_app patches
    build_middleware_stack rather than registering a middleware, so the server span
    wraps the whole user stack whatever order these were added in. That is what lets
    begin_trace_identity() capture it as the current span.
    """

    APPS = ("create_openai_app", "create_anthropic_app", "create_azure_openai_app")

    def _layers(self, app):
        return [
            (m.cls.__name__, getattr(m.kwargs.get("dispatch"), "__name__", None))
            for m in app.user_middleware
        ]

    def test_tracking_is_the_outermost_user_middleware(self):
        from app import main

        for name in self.APPS:
            with self.subTest(app=name):
                layers = self._layers(getattr(main, name)())
                self.assertEqual(layers[0], ("BaseHTTPMiddleware", "track_requests"), layers)

    def test_otel_server_span_wraps_the_whole_stack(self):
        """begin_trace_identity() captures get_current_span() as the server span."""
        from app import main

        for name in self.APPS:
            with self.subTest(app=name):
                app = getattr(main, name)()
                # instrument_app swaps in its own build_middleware_stack; without it
                # there is no server span to capture and the backfill is a no-op.
                self.assertTrue(getattr(app, "_is_instrumented_by_opentelemetry", False))
                self.assertTrue(hasattr(app, "_original_build_middleware_stack"))

    def test_azure_v1_middleware_is_inside_tracking(self):
        """The regression this ordering fixes: v1_api_middleware used to be last.

        Registered last it became the outermost layer, and the "http send" spans
        emitted from its task lost user.id.
        """
        from app import main

        layers = self._layers(main.create_azure_openai_app())
        names = [dispatch for cls, dispatch in layers if cls == "BaseHTTPMiddleware"]
        self.assertEqual(names, ["track_requests", "v1_api_middleware"], layers)

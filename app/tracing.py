"""
OpenTelemetry tracing configuration for LLM Proxy Server.

This module provides OpenTelemetry tracing utilities to work alongside OpenLit.
OpenLit handles instrumentation of FastAPI, HTTP clients, and LLM providers,
while this module provides helper functions for custom span creation and management.

NOTE: The context.detach() patch is now in app/otel_patch.py and must be
imported BEFORE any OpenTelemetry instrumentation libraries. See run.py.
"""

import os
from contextvars import ContextVar
from typing import Optional, Union
from opentelemetry import trace
from pydantic import BaseModel
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
try:
    from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
except ImportError:  # _incubating moves between 0.xxbN releases; this module
    USER_ID = "user.id"  # is imported unconditionally, even with tracing off
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.semconv.attributes import http_attributes
from opentelemetry.semconv.attributes import service_attributes
from opentelemetry.semconv.attributes import url_attributes
from opentelemetry import context as otel_context
import logging

logger = logging.getLogger(__name__)

# Global tracer instance
tracer: Optional[trace.Tracer] = None


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global tracer
    if tracer is None:
        tracer = trace.get_tracer(__name__)
    return tracer


# ---------------------------------------------------------------------------
# user.id span attribute
# ---------------------------------------------------------------------------
#
# Traceloop's instrumentation creates the LLM spans (openai.chat,
# anthropic.messages, bedrock) deep inside the provider call and fills them with
# the gen_ai.* token usage. Nothing in the route layer can reach those spans --
# chat_completion_request is their *parent*, and OTel attributes do not inherit
# downward. The only hook is SpanProcessor.on_start, which runs as each span is
# created and can still mutate it (a span refuses set_attribute only after
# _end_time is set).

# The identity is carried in a *mutable dict*, not a plain string, on purpose.
# Starlette's BaseHTTPMiddleware runs routing, the dependencies, the endpoint and
# the response body iteration in one inner task spawned via task_group.start_soon(),
# which COPIES the contextvars context. A plain ContextVar.set() inside the auth
# dependency therefore never flows back out to the middleware task, and the spans
# created there -- the ASGI "http send" spans and end_request()'s DB work in
# tracking_iterator (app/main.py) -- would miss the attribute. Handing the inner
# task a dict that the outer task created and mutating it in place makes the value
# visible in both directions, because the object is shared by reference across the
# context copy.
trace_identity: ContextVar[Optional[dict]] = ContextVar("trace_identity", default=None)


def begin_trace_identity(server_span: Optional[trace.Span] = None) -> None:
    """Start a fresh identity holder for this request.

    Called from the outermost request middleware, before authentication has run.
    ``server_span`` is the FastAPI HTTP span, kept so set_trace_user() can backfill
    it: it was started before the username was known, so on_start already fired for
    it with an empty holder.
    """
    trace_identity.set({"server_span": server_span})


def set_trace_user(username: str) -> None:
    """Record the authenticated username for the rest of this request's spans.

    Every span started from here on gets ``user.id`` via UserIdSpanProcessor. The
    two spans that were already open when authentication finished -- the HTTP
    server span and the auth span -- are backfilled directly.

    Note the deliberate divergence from the semantic conventions: ``user.id`` is
    specified as the *unique* identifier and ``user.name`` as the login name, but
    this proxy reports the username in ``user.id``. The numeric id remains
    available as ``auth.user_id`` on the auth span, so the two attributes carry
    different values by design.
    """
    try:
        if not username:
            return

        holder = trace_identity.get()
        if holder is None:
            # No tracking middleware on this app (e.g. the management server).
            # A fresh holder still covers everything started inward from here.
            holder = {}
            trace_identity.set(holder)
        holder["id"] = username

        # Backfill the spans that started before the username was known.
        backfilled = set()
        for span in (holder.get("server_span"), trace.get_current_span()):
            # On the Azure path _authenticate_azure creates no auth span, so
            # get_current_span() *is* the server span -- don't set it twice.
            if span is None or id(span) in backfilled:
                continue
            backfilled.add(id(span))
            if span.is_recording():
                span.set_attribute(USER_ID, username)
    except Exception:  # noqa: BLE001 - tracing must never fail a request
        logger.debug("set_trace_user failed", exc_info=True)


class UserIdSpanProcessor(SpanProcessor):
    """Stamps ``user.id`` onto every span in an authenticated request.

    on_start is the only place this can happen: it runs while the span is still
    mutable, so traceloop fills in the gen_ai.* usage attributes afterwards and
    both land on the same span.

    The body MUST NOT raise. SynchronousMultiSpanProcessor.on_start fans out to
    the registered processors with no try/except of its own, so an exception here
    propagates out of Span.start() and fails the call that was creating the span --
    i.e. the LLM request itself.

    Re-entering the span's lock is safe: Span.start() releases self._lock before
    dispatching to on_start (opentelemetry/sdk/trace/__init__.py:967-977), and
    set_attribute re-acquires it. If upstream ever moved that dispatch inside the
    `with self._lock` block this would deadlock.

    This runs off the event loop thread too -- Bedrock's boto3 calls go through
    run_in_threadpool and SQLAlchemy's async work runs in a greenlet, both of
    which carry the context across. Keep it allocation-light and free of I/O.
    """

    def on_start(self, span, parent_context=None) -> None:
        try:
            holder = trace_identity.get()
            if holder:
                username = holder.get("id")
                if username is not None:
                    span.set_attribute(USER_ID, username)
        except Exception:  # noqa: BLE001 - see the docstring; must never raise
            pass


_user_id_processor_registered = False


def _register_user_id_processor() -> None:
    """Attach UserIdSpanProcessor to the TracerProvider OpenLit/Traceloop set up.

    Must run *after* Traceloop.init(). Passing our processor to Traceloop.init via
    its ``processor=`` argument would replace its default exporting processor
    instead of joining it, which would silently stop OTLP export.
    """
    global _user_id_processor_registered
    if _user_id_processor_registered:
        return

    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        # Traceloop.init() never ran (no OTEL_EXPORTER_OTLP_ENDPOINT) or was
        # short-circuited, so the global provider is still a ProxyTracerProvider.
        # Warn rather than pass silently: otherwise the feature disappears with
        # no signal at all.
        logger.warning(
            "TracerProvider does not support span processors; user.id attribute disabled"
        )
        return

    provider.add_span_processor(UserIdSpanProcessor())
    _user_id_processor_registered = True
    logger.info("user.id span processor registered")


def init_tracing() -> None:
    """
    Initialize OpenTelemetry tracing utilities.
    
    Note:
        TracerProvider, exporters, and instrumentation are handled by OpenLit.
        This function only sets up trace context propagation and gets the tracer
        from the existing TracerProvider that OpenLit has configured.
    """
    
    # Set up W3C trace context propagation (if not already set by OpenLit)
    try:
        set_global_textmap(TraceContextTextMapPropagator())
    except Exception as e:
        logger.debug(f"Trace context propagation already configured: {e}")
    
    # Get tracer from the existing TracerProvider (set up by OpenLit)
    global tracer
    tracer = trace.get_tracer(__name__)

    # Stamp user.id onto every span, including the LLM spans that traceloop's
    # instrumentation creates and fills with gen_ai.* usage.
    _register_user_id_processor()

    logger.info("OpenTelemetry tracing utilities initialized (using OpenLit TracerProvider)")


def instrument_database(engine):
    """Instrument SQLAlchemy database with OpenTelemetry."""
    try:
        # For async engines, instrument the sync_engine instead
        if hasattr(engine, 'sync_engine'):
            SQLAlchemyInstrumentor().instrument(
                engine=engine.sync_engine,
                tracer_provider=trace.get_tracer_provider()
            )
            logger.info("SQLAlchemy async engine instrumented via sync_engine")
        else:
            SQLAlchemyInstrumentor().instrument(
                engine=engine,
                tracer_provider=trace.get_tracer_provider()
            )
            logger.info("SQLAlchemy instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument SQLAlchemy: {e}")


def create_span(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: Optional[dict] = None
):
    """
    Create a new span with the given name and attributes as a context manager.
    
    This function uses start_as_current_span() to ensure proper parent-child
    relationships and trace context propagation.
    
    Args:
        name: Name of the span
        kind: Kind of span (INTERNAL, SERVER, CLIENT, etc.)
        attributes: Dictionary of attributes to add to the span
    
    Returns:
        A context manager that yields the created span
    
    Example:
        with create_span("my_operation") as span:
            add_span_attributes(span, {"key": "value"})
            # ... do work ...
    """
    # Use start_as_current_span to automatically set parent-child relationships
    span = get_tracer().start_as_current_span(name, kind=kind)
    
    # The context manager is returned by start_as_current_span
    # We need to wrap it to add attributes
    class SpanContextManager:
        def __init__(self, span_cm, attrs):
            self._span_cm = span_cm
            self._attrs = attrs
            self._span = None
        
        def __enter__(self):
            self._span = self._span_cm.__enter__()
            if self._attrs:
                add_span_attributes(self._span, self._attrs)
            return self._span
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._span_cm.__exit__(exc_type, exc_val, exc_tb)
    
    return SpanContextManager(span, attributes)


def add_span_attributes(span: trace.Span, attributes: dict) -> None:
    """
    Add attributes to an existing span, safely handling Pydantic models.
    
    This function will automatically serialize Pydantic models to dictionaries
    before adding them as span attributes.
    """
    for key, value in attributes.items():
        if value is not None:
            # Convert Pydantic models to dict for serialization
            if isinstance(value, BaseModel):
                if hasattr(value, 'model_dump'):
                    value = str(value.model_dump())
                elif hasattr(value, 'dict'):
                    value = str(value.dict())
                else:
                    value = str(value)
            # OpenTelemetry only supports certain types for attributes
            # Convert complex types to strings
            elif not isinstance(value, (str, int, float, bool)):
                value = str(value)
            
            span.set_attribute(key, value)


def set_span_error(span: trace.Span, error: Union[str, Exception]) -> None:
    """
    Set span status to error and record the exception.
    
    Args:
        span: The span to set error status on
        error: Either a string error message or an Exception object
    """
    # Set span status with error message
    error_message = str(error)
    span.set_status(Status(StatusCode.ERROR, error_message))
    
    # Only record exception if it's actually an Exception object
    if isinstance(error, Exception):
        span.record_exception(error)


def get_current_span() -> Optional[trace.Span]:
    """Get the current active span."""
    return trace.get_current_span()


def get_trace_id() -> Optional[str]:
    """Get the current trace ID as a string."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_span_id() -> Optional[str]:
    """Get the current span ID as a string."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, '016x')
    return None


def get_w3c_traceparent() -> Optional[str]:
    """
    Get the current W3C Trace Context traceparent header value.
    
    Format: 00-{trace_id}-{span_id}-{trace_flags}
    
    Returns:
        W3C traceparent header value or None if no valid span context
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')
        trace_flags = format(span_context.trace_flags, '02x')
        return f"00-{trace_id}-{span_id}-{trace_flags}"
    return None


# Semantic Convention Helper Functions


class AuthAttributes:
    """Authentication-related attributes."""
    
    METHOD = "auth.method"
    RESULT = "auth.result"
    API_KEY_PREFIX = "auth.api_key_prefix"
    API_KEY_ID = "auth.api_key_id"
    API_KEY_NAME = "auth.api_key_name"
    USER_ID = "auth.user_id"
    TIMESTAMP_UPDATE_ERROR = "auth.timestamp_update_error"


def create_http_attributes(method: str, url: str, status_code: Optional[int] = None) -> dict:
    """Create standard HTTP attributes using semantic conventions."""
    attributes = {
        http_attributes.HTTP_REQUEST_METHOD: method,
        url_attributes.URL_FULL: url,
    }
    
    if status_code is not None:
        attributes[http_attributes.HTTP_RESPONSE_STATUS_CODE] = status_code
    
    return attributes


def safe_detach(token) -> None:
    """
    Safely detach an OpenTelemetry context token.
    
    In async Python applications, context tokens may become invalid when the 
    execution crosses async boundaries (e.g., after yielding to the event loop
    via await or async for). This is because contextvars work per-task, and 
    the token may have been created in a different task context.
    
    The error "was created in a different Context" is benign in these cases -
    the context cleanup happens automatically when the async task completes.
    
    This function wraps otel_context.detach() to catch and log this expected
    error, preventing noisy error logs in production.
    
    Args:
        token: The token returned by otel_context.attach()
    """
    try:
        otel_context.detach(token)
    except ValueError as e:
        # This is expected in async contexts when the token was created
        # in a different contextvars context (e.g., different async task)
        # The context will be cleaned up automatically by the event loop
        logger.debug(f"Context detach skipped (expected in async): {e}")




def create_auth_attributes(
    method: str,
    result: str,
    api_key_prefix: Optional[str] = None,
    api_key_id: Optional[str] = None,
    user_id: Optional[str] = None,
    api_key_name: Optional[str] = None
) -> dict:
    """Create authentication attributes."""
    attributes = {
        AuthAttributes.METHOD: method,
        AuthAttributes.RESULT: result
    }
    
    if api_key_prefix is not None:
        attributes[AuthAttributes.API_KEY_PREFIX] = api_key_prefix
    if api_key_id is not None:
        attributes[AuthAttributes.API_KEY_ID] = api_key_id
    if user_id is not None:
        attributes[AuthAttributes.USER_ID] = user_id
    if api_key_name is not None:
        attributes[AuthAttributes.API_KEY_NAME] = api_key_name
    
    return attributes

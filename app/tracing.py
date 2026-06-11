"""
OpenTelemetry tracing configuration for LLM Proxy Server.

This module provides OpenTelemetry tracing utilities to work alongside OpenLit.
OpenLit handles instrumentation of FastAPI, HTTP clients, and LLM providers,
while this module provides helper functions for custom span creation and management.

NOTE: The context.detach() patch is now in app/otel_patch.py and must be
imported BEFORE any OpenTelemetry instrumentation libraries. See run.py.
"""

import os
from typing import Optional, Union
from opentelemetry import trace
from pydantic import BaseModel
from opentelemetry.sdk.trace import TracerProvider
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

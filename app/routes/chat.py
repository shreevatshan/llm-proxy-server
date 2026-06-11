import os
import asyncio
import json
import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.openai_models import ChatCompletionRequest, ChatCompletionResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from typing import Union
from app.transformation import get_transformation_manager
from app.tracing import (
    get_w3c_traceparent,
    create_span,
    set_span_error,
    get_current_span,
    add_span_attributes,
    safe_detach
)
from app.routes.stream_utils import (
    stream_with_context_and_timeout,
    stream_with_context,
    STREAM_TIMEOUT_SECONDS,
    STREAM_CHUNK_TIMEOUT_SECONDS,
    DISCONNECT_CHECK_INTERVAL
)
from opentelemetry import trace
from opentelemetry.context import attach

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request_obj: Request,  # FastAPI request for disconnect detection
    request: ChatCompletionRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key),
    x_transformation_context: Optional[str] = Header(None, alias="X-Transformation-Context")
):
    """Handle chat completion requests with transformation support."""
    request_started_at = time.monotonic()
    # Check if there's an active parent span from Traceloop
    parent_span = get_current_span()
    if parent_span and parent_span.get_span_context().is_valid:
        parent_span_id = format(parent_span.get_span_context().span_id, '016x')
        logger.debug(f"[TRACE] chat_completions has parent span: {parent_span_id}")
    else:
        logger.debug(f"[TRACE] chat_completions has NO parent span - context may be lost!")
    
    # Create span for chat completion logic (child of FastAPI's HTTP request span)
    with create_span(
        "chat_completion_request",
        kind=trace.SpanKind.INTERNAL
    ) as span:
        try:
            # Apply request transformations
            transformation_manager = get_transformation_manager()
            if transformation_manager:
                # Build transformation context from headers
                context = {}
                if x_transformation_context:
                    # Parse JSON context if provided
                    try:
                        import json
                        additional_context = json.loads(x_transformation_context)
                        context.update(additional_context)
                    except json.JSONDecodeError:
                        # Ignore invalid JSON
                        pass
                
                # Preprocess the request
                request = await transformation_manager.preprocess_request(request, context)
            
            if request.stream:
                # For streaming responses, we need to preserve the trace context
                # Capture the current context before exiting the span
                from opentelemetry import context as otel_context
                current_context = otel_context.get_current()
                
                # Get W3C traceparent header for trace correlation
                traceparent = get_w3c_traceparent()
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
                
                # Add W3C Trace Context header if available
                if traceparent:
                    headers["traceparent"] = traceparent
                
                # Add streaming timeout info to span
                add_span_attributes(span, {
                    "stream.timeout_seconds": STREAM_TIMEOUT_SECONDS,
                    "stream.chunk_timeout_seconds": STREAM_CHUNK_TIMEOUT_SECONDS,
                    "stream.enabled": True
                })
                
                return StreamingResponse(
                    stream_with_context_and_timeout(
                        provider_manager.chat_completion_stream(request),
                        current_context,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                # For non-streaming responses, Langtrace will handle LLM instrumentation
                response = await provider_manager.chat_completion(request)

                # Apply response transformations
                if transformation_manager:
                    # Get API key ID if auth is an APIKey instance
                    api_key_id = auth.id if isinstance(auth, APIKey) else None
                    context = {"api_key_id": api_key_id}
                    response = await transformation_manager.postprocess_response(response, context)
                
                # Use exclude_unset to preserve upstream response fidelity:
                # only include fields actually returned by the provider
                if hasattr(response, 'model_dump'):
                    return response.model_dump(exclude_unset=True)
                return response
                
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

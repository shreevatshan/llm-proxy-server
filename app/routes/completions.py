import logging
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.openai_models import CompletionRequest, CompletionResponse
from app.providers.provider_manager import provider_manager
from app.providers.base import ProviderHTTPError
from app.routes._errors import openai_provider_error_response
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.rate_limit_dep import enforce_group_rate_limit
from app.model_access_dep import enforce_model_access, ModelAccessDenied
from app.model_resolution import resolve_model_for_request, ModelUnavailable
from app.rate_limit import RateLimitExceeded
from typing import Union
from app.routes.stream_utils import (
    stream_with_context_and_timeout,
    STREAM_TIMEOUT_SECONDS,
)
from app.tracing import (
    get_w3c_traceparent,
    create_span,
    add_span_attributes,
    set_span_error
)
from opentelemetry import trace
from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/completions", tags=["completions"])
async def completions(
    request_obj: Request,
    request: CompletionRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Handle text completion requests."""
    request_started_at = time.monotonic()
    # Create parent span for the entire completion request
    with create_span(
        "completion_request",
        kind=trace.SpanKind.SERVER
    ) as span:
        try:
            request.model = await resolve_model_for_request(request_obj, auth, request.model)
            # Group rate limit check (request-level limits already handled in middleware)
            await enforce_group_rate_limit(request_obj, auth, request.model)
            await enforce_model_access(request_obj, auth, request.model)
            if request.stream:
                # Validate the model up front so a bad/unknown model name yields
                # a clean 400 (caught below) instead of an SSE error chunk inside
                # a 200 response — the latter would be mis-counted as a completed
                # request in usage stats. Mirrors the non-streaming path.
                provider_manager.get_provider_for_model(request.model)

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

                # Preserve trace context and apply streaming timeout / disconnect
                # detection / cleanup, mirroring the chat completions handler.
                current_context = otel_context.get_current()

                return StreamingResponse(
                    stream_with_context_and_timeout(
                        provider_manager.completion_stream(request),
                        current_context,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                response = await provider_manager.completion(request)
                if hasattr(response, 'model_dump'):
                    return response.model_dump(exclude_unset=True)
                return response
        except (HTTPException, RateLimitExceeded, ModelAccessDenied, ModelUnavailable):
            raise
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Completion error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Completion error")

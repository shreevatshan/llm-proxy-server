from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.openai_models import CompletionRequest, CompletionResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.rate_limit_dep import enforce_group_rate_limit
from app.rate_limit import RateLimitExceeded
from typing import Union
from app.tracing import (
    get_w3c_traceparent,
    create_span,
    add_span_attributes,
    set_span_error
)
from opentelemetry import trace

router = APIRouter()


@router.post("/v1/completions", tags=["completions"])
async def completions(
    request_obj: Request,
    request: CompletionRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Handle text completion requests."""
    # Create parent span for the entire completion request
    with create_span(
        "completion_request",
        kind=trace.SpanKind.SERVER
    ) as span:
        try:
            # Group rate limit check (request-level limits already handled in middleware)
            await enforce_group_rate_limit(request_obj, auth, request.model)
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
                
                return StreamingResponse(
                    provider_manager.completion_stream(request),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                response = await provider_manager.completion(request)
                if hasattr(response, 'model_dump'):
                    return response.model_dump(exclude_unset=True)
                return response
        except (HTTPException, RateLimitExceeded):
            raise
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")

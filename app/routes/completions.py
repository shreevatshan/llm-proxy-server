from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.openai_models import CompletionRequest, CompletionResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
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
            if request.stream:
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
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")

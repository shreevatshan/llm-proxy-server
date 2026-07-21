"""Routes for the OpenAI Responses API.

Provides endpoints for creating, retrieving, deleting, cancelling responses,
listing input items, counting input tokens, and compacting conversations.
Supported by OpenAI-compatible providers and Azure. Not supported by Google or Bedrock.
Note: Azure does not currently support /responses/compact or /responses/input_tokens.
"""

import logging
import time
from typing import Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.openai_models import (
    ResponsesCreateRequest,
    ResponsesCompactRequest,
    ResponsesInputTokensRequest,
    ResponseObject,
    ResponseDeletedObject,
    ResponseInputTokensResult,
    CompactedResponseObject,
    ResponseItemList
)
from app.providers.provider_manager import provider_manager
from app.providers.base import ProviderHTTPError
from app.routes._errors import openai_provider_error_response
from app.auth.middleware import (
    authenticate_jwt_or_api_key,
    get_owner_user_id,
    verify_response_ownership,
)
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.rate_limit_dep import enforce_group_rate_limit
from app.model_access_dep import enforce_model_access, ModelAccessDenied
from app.rate_limit import RateLimitExceeded
from typing import Union
from app.routes.stream_utils import (
    stream_with_context_and_timeout,
    STREAM_TIMEOUT_SECONDS
)
from app.tracing import (
    get_w3c_traceparent,
    create_span,
    set_span_error,
    get_current_span,
    add_span_attributes
)
from opentelemetry import trace
from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== POST /v1/responses/input_tokens ====================
# NOTE: This must be registered BEFORE the {response_id} catch-all routes

@router.post("/v1/responses/input_tokens", tags=["responses"])
async def responses_input_tokens(
    request_obj: Request,
    request: ResponsesInputTokensRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Count input tokens for a Responses API request."""
    with create_span("responses_input_tokens", kind=trace.SpanKind.INTERNAL) as span:
        try:
            # Group rate limit + per-user model access enforcement.
            await enforce_group_rate_limit(request_obj, auth, request.model)
            await enforce_model_access(request_obj, auth, request.model)

            response = await provider_manager.responses_input_tokens(request)
            return response
        except (HTTPException, RateLimitExceeded, ModelAccessDenied):
            raise
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses input tokens error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses input tokens error")


# ==================== POST /v1/responses/compact ====================

@router.post("/v1/responses/compact", tags=["responses"])
async def responses_compact(
    request_obj: Request,
    request: ResponsesCompactRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Compact a conversation to reduce token usage."""
    with create_span("responses_compact", kind=trace.SpanKind.INTERNAL) as span:
        try:
            # Group rate limit + per-user model access enforcement. compact
            # invokes the upstream model, so it must be gated like inference.
            await enforce_group_rate_limit(request_obj, auth, request.model)
            await enforce_model_access(request_obj, auth, request.model)

            response = await provider_manager.responses_compact(request)
            return response
        except (HTTPException, RateLimitExceeded, ModelAccessDenied):
            raise
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses compact error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses compact error")


# ==================== POST /v1/responses ====================

@router.post("/v1/responses", tags=["responses"])
async def responses_create(
    request_obj: Request,  # FastAPI request for disconnect detection
    request: ResponsesCreateRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Create a model response. Supports both streaming and non-streaming."""
    request_started_at = time.monotonic()
    with create_span("responses_create", kind=trace.SpanKind.INTERNAL) as span:
        try:
            # Group rate limit check (request-level limits already handled in middleware)
            await enforce_group_rate_limit(request_obj, auth, request.model)
            await enforce_model_access(request_obj, auth, request.model)

            if request.stream:
                # Validate the model up front so a bad/unknown model name yields
                # a clean 400 (caught below) instead of an SSE error chunk inside
                # a 200 response — the latter would be mis-counted as a completed
                # request in usage stats. Mirrors the non-streaming path.
                provider_manager.get_provider_for_model(request.model)

                # Capture current trace context for streaming
                current_context = otel_context.get_current()
                
                traceparent = get_w3c_traceparent()
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
                if traceparent:
                    headers["traceparent"] = traceparent
                
                add_span_attributes(span, {
                    "stream.timeout_seconds": STREAM_TIMEOUT_SECONDS,
                    "stream.enabled": True,
                    "responses_api": True
                })
                
                return StreamingResponse(
                    stream_with_context_and_timeout(
                        provider_manager.responses_create_stream(
                            request, user_id=get_owner_user_id(auth)
                        ),
                        current_context,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                response = await provider_manager.responses_create(
                    request, user_id=get_owner_user_id(auth)
                )
                return response

        except (HTTPException, RateLimitExceeded, ModelAccessDenied):
            raise
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=400, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses create error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses create error")


# ==================== GET /v1/responses/{response_id} ====================

@router.get("/v1/responses/{response_id}", tags=["responses"])
async def responses_retrieve(
    response_id: str,
    include: Optional[List[str]] = Query(None),
    stream: Optional[bool] = Query(None),
    starting_after: Optional[int] = Query(None),
    include_obfuscation: Optional[bool] = Query(None),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Retrieve a model response by ID."""
    # Ownership check (IDOR guard): only the creating user (or an admin) may
    # access a stored response.
    await verify_response_ownership(response_id, auth)
    with create_span("responses_retrieve", kind=trace.SpanKind.INTERNAL) as span:
        try:
            kwargs = {}
            if include is not None:
                kwargs["include"] = include
            # Pass through additional query params as the SDK supports them
            
            response = await provider_manager.responses_retrieve(response_id, **kwargs)
            return response
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=404, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses retrieve error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses retrieve error")


# ==================== DELETE /v1/responses/{response_id} ====================

@router.delete("/v1/responses/{response_id}", tags=["responses"])
async def responses_delete(
    response_id: str,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Delete a stored model response."""
    # Ownership check (IDOR guard): see responses_retrieve.
    await verify_response_ownership(response_id, auth)
    with create_span("responses_delete", kind=trace.SpanKind.INTERNAL) as span:
        try:
            response = await provider_manager.responses_delete(response_id)
            return response
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=404, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses delete error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses delete error")


# ==================== POST /v1/responses/{response_id}/cancel ====================

@router.post("/v1/responses/{response_id}/cancel", tags=["responses"])
async def responses_cancel(
    response_id: str,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """Cancel a background response."""
    # Ownership check (IDOR guard): see responses_retrieve.
    await verify_response_ownership(response_id, auth)
    with create_span("responses_cancel", kind=trace.SpanKind.INTERNAL) as span:
        try:
            response = await provider_manager.responses_cancel(response_id)
            return response
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=404, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses cancel error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses cancel error")


# ==================== GET /v1/responses/{response_id}/input_items ====================

@router.get("/v1/responses/{response_id}/input_items", tags=["responses"])
async def responses_list_input_items(
    response_id: str,
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query(None),
    include: Optional[List[str]] = Query(None),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
):
    """List input items for a response."""
    # Ownership check (IDOR guard): see responses_retrieve.
    await verify_response_ownership(response_id, auth)
    with create_span("responses_list_input_items", kind=trace.SpanKind.INTERNAL) as span:
        try:
            kwargs = {}
            if after is not None:
                kwargs["after"] = after
            if limit is not None:
                kwargs["limit"] = limit
            if order is not None:
                kwargs["order"] = order
            if include is not None:
                kwargs["include"] = include
            
            response = await provider_manager.responses_list_input_items(response_id, **kwargs)
            return response
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except ValueError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=404, detail=str(e))
        except NotImplementedError as e:
            set_span_error(span, e)
            raise HTTPException(status_code=501, detail=str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("Responses list input items error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Responses list input items error")

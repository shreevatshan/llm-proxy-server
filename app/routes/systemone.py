"""Route for the TypeSafe System One API (POST /v1/systemone).

Typed answers (noul, choice, score, box) to named questions about a state.
Native passthrough: the body is forwarded as sent (only ``model`` mapped to the
upstream id) and the upstream ``{model, answers, usage}`` object is relayed
unchanged except that ``model`` echoes the client's name. Non-streaming.

Mounted only on the TypeSafe app (port 11366, and /typesafe on the management
port), whose exception handlers render every error in TypeSafe's
``{"detail": {"error_type", "message"}}`` shape. Served by custom providers
with a System One base URL and by Azure deployments whose row selects
/systemone (Foundry's Microsoft-Decision-1); any other model answers 400.
"""

import logging
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from app.auth.admin import AdminUser
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.model_access_dep import ModelAccessDenied
from app.model_resolution import ModelUnavailable, resolve_model_for_request
from app.providers.base import SYSTEMONE_API, ProviderHTTPError
from app.providers.openai_compatible import encode_passthrough_body
from app.providers.provider_manager import provider_manager
from app.rate_limit import RateLimitExceeded
from app.rate_limit_dep import enforce_access_and_limits, enforces_limits
from app.routes._errors import (
    TYPESAFE_RELAYED_HEADERS,
    TYPESAFE_REQUEST_ID_HEADER,
    relay_headers,
    typesafe_error_response,
    typesafe_json_response,
    typesafe_provider_error_response,
    typesafe_request_id,
)
from app.tracing import add_span_attributes, create_span, set_span_error

logger = logging.getLogger(__name__)

router = APIRouter()

# The only request headers forwarded upstream (exact names, not a prefix).
SYSTEMONE_FORWARDED_HEADERS = (
    "TypeSafe-Danger-Zone-Images",
    "TypeSafe-Danger-Zone-Long-Context",
)


class SystemOneRequest(BaseModel):
    """POST /v1/systemone body.

    Locally checked: ``model`` is a string and ``questions`` a non-empty object
    (TypeSafe's own 422 rule, same FastAPI detail[] shape). ``state``, ``assets``
    and every other field are forwarded exactly as sent and validated upstream.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    questions: Dict[str, Any] = Field(min_length=1)
    state: Any = None


def _forwarded_headers(request: Request) -> Dict[str, str]:
    return {name: request.headers[name] for name in SYSTEMONE_FORWARDED_HEADERS if name in request.headers}


def _reject_multipart(request: Request) -> None:
    """Route-level dependency: runs before auth and before body parsing."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type.startswith("multipart/"):
        raise HTTPException(
            status_code=415,
            detail={
                "error_type": "api_usage_error",
                "message": "multipart/form-data System One requests (SDK image uploads) are not "
                           "supported by this proxy; send application/json",
            },
        )


def _usage_attributes(body: Dict[str, Any]) -> Dict[str, Any]:
    """Token counts for the route span (no gen_ai span exists for client.post)."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        "systemone.input_tokens": usage.get("input_tokens"),
        "systemone.output_tokens": usage.get("output_tokens"),
    }


def _success_headers(request: Request, upstream_headers: Dict[str, str]) -> Dict[str, str]:
    headers = relay_headers(upstream_headers, TYPESAFE_RELAYED_HEADERS)
    if not any(k.lower() == TYPESAFE_REQUEST_ID_HEADER for k in headers):
        headers[TYPESAFE_REQUEST_ID_HEADER] = typesafe_request_id(request)
    return headers


@router.post(
    "/v1/systemone",
    tags=["systemone"],
    summary="System One",
    dependencies=[Depends(_reject_multipart)],
)
@enforces_limits
async def systemone_create(
    request_obj: Request,
    request: SystemOneRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key),
):
    """Answer typed questions about the state with a System One model."""
    with create_span("systemone_create", kind=trace.SpanKind.INTERNAL) as span:
        try:
            if (request.model_extra or {}).get("stream"):
                # Before resolution and charging: a stream request spends nothing.
                return typesafe_error_response(
                    request_obj, 400, "The System One API does not stream; remove 'stream'"
                )

            request.model = await resolve_model_for_request(request_obj, auth, request.model)
            # Bare names only resolve to System One upstreams (the surface filter
            # in model_resolution); a canonical or provider-prefixed id is refused
            # here, before any quota is spent.
            provider = provider_manager.get_provider_for_model(request.model)
            if not provider.supports_api_for_model(request.model, SYSTEMONE_API):
                return typesafe_error_response(
                    request_obj,
                    400,
                    f"The model '{request.model}' is not served over the System One API; only custom "
                    "providers with a System One base URL and Azure deployments with /systemone "
                    "(Microsoft-Decision-1) serve /v1/systemone",
                )
            # NaN / Infinity / lone surrogates are client errors: refused here,
            # before any quota is spent (the provider re-checks before any I/O).
            encode_passthrough_body(request.model_dump(exclude_unset=True), surrogate_message=True)
            # Access first, so a 403 spends no quota; then the model's group limit.
            await enforce_access_and_limits(request_obj, auth, request.model)
            add_span_attributes(span, {
                "model": request.model,
                "systemone.question_count": len(request.questions),
            })

            body, upstream_headers = await provider_manager.systemone_create(
                request, _forwarded_headers(request_obj)
            )
        except (HTTPException, RateLimitExceeded, ModelAccessDenied, ModelUnavailable):
            raise  # rendered by the TypeSafe app's exception handlers
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return typesafe_provider_error_response(request_obj, e)
        except NotImplementedError as e:
            set_span_error(span, e)
            return typesafe_error_response(request_obj, 400, str(e))
        except ValueError as e:
            # NaN / unpaired surrogate in the body, or an ambiguous provider prefix.
            set_span_error(span, e)
            return typesafe_error_response(request_obj, 400, str(e))
        except Exception as e:
            set_span_error(span, e)
            logger.error("System One create error: %s", e, exc_info=True)
            return typesafe_error_response(request_obj, 500, "System One create error")

        # Built outside the try: nothing after a billed upstream answer may turn
        # it into an error.
        add_span_attributes(span, _usage_attributes(body))
        return typesafe_json_response(200, body, _success_headers(request_obj, upstream_headers))

"""Route for the OpenAI Decisions API (POST /v1/decisions).

Typed answers (predicate probabilities, choices, scores) about shared input.
Non-streaming, with no retrieve/list endpoints. The request body is forwarded
as sent and the upstream ``{model, answers, usage}`` object is relayed
unchanged. Served by OpenAI-compatible custom providers and by Azure
deployments whose row selects /decisions; other providers answer 400
``unsupported_api``. Also mounted at /openai/v1/decisions on the Azure port.
"""

import logging
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from opentelemetry import trace

from app.auth.admin import AdminUser
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.model_access_dep import ModelAccessDenied
from app.model_resolution import ModelUnavailable, resolve_model_for_request
from app.openai_models import DecisionCreateRequest
from app.providers.base import ProviderHTTPError
from app.providers.provider_manager import provider_manager
from app.rate_limit import RateLimitExceeded
from app.rate_limit_dep import enforce_access_and_limits, enforces_limits
from app.routes._errors import openai_error_response, openai_provider_error_response
from app.tracing import add_span_attributes, create_span, set_span_error

logger = logging.getLogger(__name__)

router = APIRouter()


def _usage_attributes(response: Dict[str, Any]) -> Dict[str, Any]:
    """Token counts for the route span (no gen_ai span exists for client.post)."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    details = usage.get("input_tokens_details")
    return {
        "decisions.input_tokens": usage.get("input_tokens"),
        "decisions.cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None,
        "decisions.total_tokens": usage.get("total_tokens"),
    }


@router.post("/v1/decisions", tags=["decisions"])
@enforces_limits
async def decisions_create(
    request_obj: Request,
    request: DecisionCreateRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key),
):
    """Answer typed questions about the input with a decision model."""
    with create_span("decisions_create", kind=trace.SpanKind.INTERNAL) as span:
        try:
            request.model = await resolve_model_for_request(request_obj, auth, request.model)
            # Access first, so a 403 spends no quota; then the model's group limit.
            await enforce_access_and_limits(request_obj, auth, request.model)

            questions = request.questions
            add_span_attributes(span, {
                "model": request.model,
                "decisions.question_count": len(questions) if isinstance(questions, list) else None,
            })

            response = await provider_manager.decisions_create(request)
            add_span_attributes(span, _usage_attributes(response))
            return response

        except (HTTPException, RateLimitExceeded, ModelAccessDenied, ModelUnavailable):
            raise
        except ProviderHTTPError as e:
            set_span_error(span, e)
            return openai_provider_error_response(e)
        except NotImplementedError as e:
            # The resolved model's provider cannot serve /decisions (Google,
            # Bedrock, PlatformAI, Anthropic-only custom providers, Foundry Claude).
            # Same code as Azure's per-deployment gating error.
            set_span_error(span, e)
            return openai_error_response(
                400, str(e), type="invalid_request_error", param="model", code="unsupported_api"
            )
        except ValueError as e:
            set_span_error(span, e)
            return openai_error_response(400, str(e), type="invalid_request_error")
        except Exception as e:
            set_span_error(span, e)
            logger.error("Decisions create error: %s", e, exc_info=True)
            return openai_error_response(500, "Decisions create error", type="server_error")

"""Anthropic Models listing route.

Handles GET /v1/models for the Anthropic API server (port 2027).
Lists only models from providers that support the Anthropic API format.
"""

import logging
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.anthropic_models import AnthropicModelInfo, AnthropicModelListResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_anthropic_request
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.tracing import create_span, add_span_attributes, set_span_error

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/models", tags=["anthropic"])
async def list_anthropic_models(
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_anthropic_request),
):
    """List all available models that support the Anthropic API format."""
    with create_span("api.anthropic.models.list") as span:
        try:
            models = await provider_manager.get_all_models(api_filter="anthropic")

            anthropic_models = []
            for model in models:
                # Determine which APIs this model supports
                provider = provider_manager.providers.get(model.provider)
                supported_apis = provider.get_supported_apis_for_model(model.id) if provider else ["openai"]

                anthropic_models.append(AnthropicModelInfo(
                    id=model.id,
                    display_name=model.id,
                    created_at=None,
                    provider=model.provider,
                    supported_apis=supported_apis,
                ))

            add_span_attributes(span, {
                "models.count": len(anthropic_models),
                "models.source": "cache",
            })

            response = AnthropicModelListResponse(
                data=anthropic_models,
                has_more=False,
                first_id=anthropic_models[0].id if anthropic_models else None,
                last_id=anthropic_models[-1].id if anthropic_models else None,
            )
            return response

        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

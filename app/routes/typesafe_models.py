"""GET /v1/models on the TypeSafe app, in TypeSafe's own shape.

``{"models": [{"name", "description", "release_date"}]}`` (all three required by
TypeSafe's OpenAPI), so the TypeSafe SDK's ``models.list()`` works through the
proxy. Lists the System One models the caller may use, with the same per-caller
visibility as the other ports' model lists. ``name`` is the canonical id, which
POST /v1/systemone always accepts for a permitted caller.
"""

from typing import Union

from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth.admin import AdminUser
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.providers.base import SYSTEMONE_API
from app.providers.custom_providers import systemone_model_meta
from app.providers.provider_manager import provider_manager
from app.routes._errors import TYPESAFE_REQUEST_ID_HEADER, typesafe_json_response, typesafe_request_id
from app.routes.models import _caller_user_id
from app.tracing import add_span_attributes, create_span, set_span_error

router = APIRouter()

# TypeSafe requires a YYYY-MM-DD release_date; used when the upstream gave none.
_UNKNOWN_RELEASE_DATE = "1970-01-01"


@router.get("/v1/models", tags=["models"], summary="List Models")
async def list_models(
    request: Request,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key),
):
    """List the System One models available to the caller."""
    with create_span("api.typesafe_models.list") as span:
        try:
            models = await provider_manager.get_all_models(
                api_filter=SYSTEMONE_API, user_id=_caller_user_id(auth)
            )
        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(
                status_code=500,
                detail={"error_type": "api_error", "message": "Error fetching models"},
            )
        out = []
        for model in sorted(models, key=lambda m: m.id):
            description, release_date = systemone_model_meta(model.id)
            out.append({
                "name": model.id,
                "description": description or model.description or "",
                "release_date": release_date or _UNKNOWN_RELEASE_DATE,
            })
        add_span_attributes(span, {"models.count": len(out), "models.source": "cache"})
        return typesafe_json_response(
            200, {"models": out}, {TYPESAFE_REQUEST_ID_HEADER: typesafe_request_id(request)}
        )

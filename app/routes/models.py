from fastapi import APIRouter, HTTPException, Depends
from app.openai_models import ModelsResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.tracing import create_span, add_span_attributes, set_span_error
from typing import Union, Optional

router = APIRouter()


def _caller_user_id(auth: Union[User, AdminUser, APIKey]) -> Optional[int]:
    """User id for per-user model filtering; None for admins (see full list)."""
    if isinstance(auth, AdminUser):
        return None
    return getattr(auth, "user_id", None) or getattr(auth, "id", None)


@router.get("/v1/models", response_model=ModelsResponse, tags=["models"])
async def list_models(auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)):
    """List all available models from all providers."""
    with create_span("api.models.list") as span:
        try:
            models = await provider_manager.get_all_models(
                api_filter="openai", user_id=_caller_user_id(auth)
            )
            
            add_span_attributes(span, {
                "models.count": len(models),
                "models.source": "cache"
            })
            
            return ModelsResponse(data=models)
        except Exception as e:
            set_span_error(span, e)
            raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

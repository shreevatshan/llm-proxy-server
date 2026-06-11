from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from typing import Dict, Any
from app.openai_models import EmbeddingRequest, EmbeddingResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from typing import Union
from app.tracing import create_span, add_span_attributes, set_span_error
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
) -> EmbeddingResponse:
    """Create embeddings for the given input."""
    
    with create_span("embeddings_request") as span:
        try:
            # Add span attributes
            add_span_attributes(span, {
                "model": request.model,
                "input_type": type(request.input).__name__,
                "encoding_format": request.encoding_format,
                "dimensions": request.dimensions,
                "user_id": getattr(auth, 'id', None) if auth else None
            })
            
            # Get provider for the model
            provider = provider_manager.get_provider_for_model(request.model)
            if not provider:
                error_msg = f"No provider found for model: {request.model}"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Check if provider supports embeddings
            if not hasattr(provider, 'embeddings') or not callable(getattr(provider, 'embeddings')):
                error_msg = f"Provider {provider.__class__.__name__} does not support embeddings"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Call provider's embeddings method
            response = await provider.embeddings(request)
            
            # Add response attributes to span
            add_span_attributes(span, {
                "response_data_count": len(response.data),
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens
            })
            
            return response
            
        except HTTPException:
            raise
        except NotImplementedError as e:
            error_msg = f"Embeddings not supported: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Error creating embeddings: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

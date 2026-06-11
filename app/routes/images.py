from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from fastapi.security import HTTPBearer
from typing import Dict, Any, Optional
from app.openai_models import ImageGenerationRequest, ImageEditRequest, ImageVariationRequest, ImageResponse
from app.providers.provider_manager import provider_manager
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from typing import Union
from app.tracing import create_span, add_span_attributes, set_span_error
import logging
import base64

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/v1/images/generations", response_model=ImageResponse)
async def create_image(
    request: ImageGenerationRequest,
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
) -> ImageResponse:
    """Generate images from text prompts."""
    
    with create_span("image_generation_request") as span:
        try:
            # Add span attributes
            add_span_attributes(span, {
                "model": request.model,
                "prompt_length": len(request.prompt),
                "n": request.n,
                "size": request.size,
                "quality": request.quality,
                "style": request.style,
                "response_format": request.response_format,
                "user_id": getattr(auth, 'id', None) if auth else None
            })
            
            # Get provider for the model
            provider = provider_manager.get_provider_for_model(request.model)
            if not provider:
                error_msg = f"No provider found for model: {request.model}"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Check if provider supports image generation
            if not hasattr(provider, 'image_generation') or not callable(getattr(provider, 'image_generation')):
                error_msg = f"Provider {provider.__class__.__name__} does not support image generation"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Call provider's image generation method
            response = await provider.image_generation(request)
            
            # Add response attributes to span
            add_span_attributes(span, {
                "response_data_count": len(response.data),
                "created": response.created
            })
            
            return response
            
        except HTTPException:
            raise
        except NotImplementedError as e:
            error_msg = f"Image generation not supported: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


@router.post("/v1/images/edits", response_model=ImageResponse)
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mask: Optional[UploadFile] = File(None),
    model: Optional[str] = Form("dall-e-2"),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("url"),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
) -> ImageResponse:
    """Edit images with text prompts."""
    
    with create_span("image_edit_request") as span:
        try:
            # Read and encode image file
            image_content = await image.read()
            image_b64 = base64.b64encode(image_content).decode('utf-8')
            
            # Read and encode mask file if provided
            mask_b64 = None
            if mask:
                mask_content = await mask.read()
                mask_b64 = base64.b64encode(mask_content).decode('utf-8')
            
            # Create request object
            request = ImageEditRequest(
                image=image_b64,
                prompt=prompt,
                mask=mask_b64,
                model=model,
                n=n,
                response_format=response_format,
                size=size,
                user=user
            )
            
            # Add span attributes
            add_span_attributes(span, {
                "model": request.model,
                "prompt_length": len(request.prompt),
                "n": request.n,
                "size": request.size,
                "response_format": request.response_format,
                "has_mask": mask is not None,
                "user_id": getattr(auth, 'id', None) if auth else None
            })
            
            # Get provider for the model
            provider = provider_manager.get_provider_for_model(request.model)
            if not provider:
                error_msg = f"No provider found for model: {request.model}"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Check if provider supports image editing
            if not hasattr(provider, 'image_edit') or not callable(getattr(provider, 'image_edit')):
                error_msg = f"Provider {provider.__class__.__name__} does not support image editing"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Call provider's image edit method
            response = await provider.image_edit(request)
            
            # Add response attributes to span
            add_span_attributes(span, {
                "response_data_count": len(response.data),
                "created": response.created
            })
            
            return response
            
        except HTTPException:
            raise
        except NotImplementedError as e:
            error_msg = f"Image editing not supported: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Error editing image: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)


@router.post("/v1/images/variations", response_model=ImageResponse)
async def create_image_variation(
    image: UploadFile = File(...),
    model: Optional[str] = Form("dall-e-2"),
    n: Optional[int] = Form(1),
    response_format: Optional[str] = Form("url"),
    size: Optional[str] = Form("1024x1024"),
    user: Optional[str] = Form(None),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_jwt_or_api_key)
) -> ImageResponse:
    """Create variations of images."""
    
    with create_span("image_variation_request") as span:
        try:
            # Read and encode image file
            image_content = await image.read()
            image_b64 = base64.b64encode(image_content).decode('utf-8')
            
            # Create request object
            request = ImageVariationRequest(
                image=image_b64,
                model=model,
                n=n,
                response_format=response_format,
                size=size,
                user=user
            )
            
            # Add span attributes
            add_span_attributes(span, {
                "model": request.model,
                "n": request.n,
                "size": request.size,
                "response_format": request.response_format,
                "user_id": getattr(auth, 'id', None) if auth else None
            })
            
            # Get provider for the model
            provider = provider_manager.get_provider_for_model(request.model)
            if not provider:
                error_msg = f"No provider found for model: {request.model}"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Check if provider supports image variations
            if not hasattr(provider, 'image_variation') or not callable(getattr(provider, 'image_variation')):
                error_msg = f"Provider {provider.__class__.__name__} does not support image variations"
                logger.error(error_msg)
                set_span_error(span, error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Call provider's image variation method
            response = await provider.image_variation(request)
            
            # Add response attributes to span
            add_span_attributes(span, {
                "response_data_count": len(response.data),
                "created": response.created
            })
            
            return response
            
        except HTTPException:
            raise
        except NotImplementedError as e:
            error_msg = f"Image variations not supported: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            error_msg = f"Error creating image variation: {str(e)}"
            logger.error(error_msg)
            set_span_error(span, error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

"""Routes for the Azure OpenAI direct API.

Provides Azure OpenAI-compatible endpoints on a dedicated port (default 11439).
Accepts native Azure OpenAI REST API URL patterns with provider name in the path:

    POST /openai/deployments/{provider_name}/{deployment}/chat/completions?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/completions?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/embeddings?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/images/generations?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/audio/speech?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/audio/transcriptions?api-version=...
    POST /openai/deployments/{provider_name}/{deployment}/audio/translations?api-version=...
    POST /openai/deployments/{provider_name}/responses?api-version=...
    GET  /openai/deployments/{provider_name}  — list deployments for a provider
    GET  /openai/models  — list all models across all Azure providers

Clients (e.g. Azure OpenAI SDK) should set api-key header for auth.
"""

import logging
import base64
import json
import time
from typing import Optional, List, Union

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response

from app.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageResponse,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    AudioTranscriptionResponse,
    AudioTranslationResponse,
    ResponsesCreateRequest,
    ResponseObject,
    ModelInfo,
)
from app.providers.provider_manager import provider_manager
from app.providers.openai_compatible import preserve_upstream_model
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.auth.database import get_api_key, AsyncSessionLocal
from app.auth.cache import auth_cache
from app.routes.stream_utils import (
    stream_with_context_and_timeout,
    STREAM_TIMEOUT_SECONDS,
)
from app.rate_limit_dep import enforce_group_rate_limit
from app.tracing import (
    get_w3c_traceparent,
    create_span,
    set_span_error,
    get_current_span,
    add_span_attributes,
)
from opentelemetry import trace
from opentelemetry import context as otel_context

logger = logging.getLogger(__name__)


async def _use_upstream_model_names():
    """Router-level dependency: tells the provider layer to keep the
    upstream model name (e.g. ``gpt-4.1-2025-04-14``) instead of
    rewriting it to the proxy's internal routing name."""
    preserve_upstream_model.set(True)


router = APIRouter(dependencies=[Depends(_use_upstream_model_names)])


# ==================== Helpers ====================

def _azure_error(status_code: int, code: str, message: str):
    """Return an Azure-style error response."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


async def _authenticate_azure(
    request: Request,
    api_key_header: Optional[str] = Header(None, alias="api-key"),
):
    """Authenticate via the ``api-key`` header – the native Azure OpenAI format.

    The Azure OpenAI SDK always sends credentials as ``api-key: <value>``.
    We validate that value directly against the proxy's API-key database
    (the same keys generated from the management panel).
    """
    token = api_key_header
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "401", "message": "api-key header is required"}},
        )

    # Fast-path: check the in-memory cache first
    cached = auth_cache.get_cached_api_key(token)
    if cached and cached.is_active:
        auth_cache.mark_api_key_used(token)
        from app.auth.middleware import _update_tracking_identity, _enforce_rate_limit
        await _update_tracking_identity(request, cached)
        await _enforce_rate_limit(request, cached, envelope_override="azure")
        return cached

    # Slow-path: look up in the database
    async with AsyncSessionLocal() as db:
        api_key_obj = await get_api_key(db, token)
    if api_key_obj:
        cached_key = auth_cache.cache_api_key(token, api_key_obj)
        auth_cache.mark_api_key_used(token)
        from app.auth.middleware import _update_tracking_identity, _enforce_rate_limit
        await _update_tracking_identity(request, cached_key)
        await _enforce_rate_limit(request, cached_key, envelope_override="azure")
        return api_key_obj

    raise HTTPException(
        status_code=401,
        detail={"error": {"code": "401", "message": "Invalid api-key"}},
    )


def _get_azure_provider(provider_name: str):
    """Resolve *provider_name* and verify it is an Azure provider."""
    from app.providers.azure_provider import AzureProvider

    try:
        provider = provider_manager._get_provider(provider_name)
    except (ValueError, Exception) as exc:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "ProviderNotFound", "message": str(exc)}},
        )

    if not isinstance(provider, AzureProvider):
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "InvalidProvider",
                    "message": f"Provider '{provider_name}' is not an Azure OpenAI provider",
                }
            },
        )
    return provider


def _build_model_name(provider_name: str, deployment: str) -> str:
    """Construct the canonical model reference used by the provider manager."""
    return f"{provider_name}/{deployment}"


# ==================== Models / Deployments ====================

@router.get("/openai/models", tags=["azure_openai"])
async def list_azure_models(
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """List all models across all Azure providers."""
    models = await provider_manager.get_all_models(api_filter="azure_openai")
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": m.object,
                "created": m.created,
                "owned_by": m.owned_by,
            }
            for m in models
        ],
    }


@router.get("/openai/deployments/{provider_name}", tags=["azure_openai"])
async def list_deployments(
    provider_name: str,
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """List deployments for a specific Azure provider."""
    provider = _get_azure_provider(provider_name)
    models: List[ModelInfo] = await provider.get_available_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": m.object,
                "created": m.created,
                "owned_by": m.owned_by,
            }
            for m in models
        ],
    }


# ==================== Chat Completions ====================

@router.post(
    "/openai/deployments/{provider_name}/{deployment}/chat/completions",
    tags=["azure_openai"],
)
async def azure_chat_completions(
    request_obj: Request,
    provider_name: str,
    deployment: str,
    request: ChatCompletionRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI chat completions endpoint."""
    request_started_at = time.monotonic()
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)
    request.model = model_name
    await enforce_group_rate_limit(request_obj, auth, model_name, envelope_override="azure")

    with create_span("azure_chat_completion", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
            "azure.api_version": api_version or "",
        })
        try:
            if request.stream:
                current_context = otel_context.get_current()
                traceparent = get_w3c_traceparent()
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                if traceparent:
                    headers["traceparent"] = traceparent

                return StreamingResponse(
                    stream_with_context_and_timeout(
                        provider.chat_completion_stream(request),
                        current_context,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )
            else:
                response = await provider.chat_completion(request)
                if hasattr(response, 'model_dump'):
                    return response.model_dump(exclude_unset=True)
                return response
        except ValueError as e:
            set_span_error(span, e)
            return _azure_error(400, "InvalidRequest", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Chat completion error: {str(e)}")


# ==================== Text Completions ====================

@router.post(
    "/openai/deployments/{provider_name}/{deployment}/completions",
    tags=["azure_openai"],
)
async def azure_completions(
    request_obj: Request,
    provider_name: str,
    deployment: str,
    request: CompletionRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI text completions endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)
    request.model = model_name
    await enforce_group_rate_limit(request_obj, auth, model_name, envelope_override="azure")

    with create_span("azure_completion", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            if request.stream:
                traceparent = get_w3c_traceparent()
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                if traceparent:
                    headers["traceparent"] = traceparent

                return StreamingResponse(
                    provider.completion_stream(request),
                    media_type="text/event-stream",
                    headers=headers,
                )
            else:
                response = await provider.completion(request)
                if hasattr(response, 'model_dump'):
                    return response.model_dump(exclude_unset=True)
                return response
        except ValueError as e:
            set_span_error(span, e)
            return _azure_error(400, "InvalidRequest", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Completion error: {str(e)}")


# ==================== Embeddings ====================

@router.post(
    "/openai/deployments/{provider_name}/{deployment}/embeddings",
    tags=["azure_openai"],
)
async def azure_embeddings(
    request_obj: Request,
    provider_name: str,
    deployment: str,
    request: EmbeddingRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI embeddings endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)
    request.model = model_name
    await enforce_group_rate_limit(request_obj, auth, model_name, envelope_override="azure")

    with create_span("azure_embeddings", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            response = await provider.embeddings(request)
            return response
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(400, "OperationNotSupported", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Embeddings error: {str(e)}")


# ==================== Images ====================

@router.post(
    "/openai/deployments/{provider_name}/{deployment}/images/generations",
    tags=["azure_openai"],
)
async def azure_image_generation(
    request_obj: Request,
    provider_name: str,
    deployment: str,
    request: ImageGenerationRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI image generation endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)
    request.model = model_name
    await enforce_group_rate_limit(request_obj, auth, model_name, envelope_override="azure")

    with create_span("azure_image_generation", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            response = await provider.image_generation(request)
            return response
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(400, "OperationNotSupported", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Image generation error: {str(e)}")


# ==================== Audio ====================

@router.post(
    "/openai/deployments/{provider_name}/{deployment}/audio/speech",
    tags=["azure_openai"],
)
async def azure_audio_speech(
    provider_name: str,
    deployment: str,
    request: AudioSpeechRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI text-to-speech endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)
    request.model = model_name

    with create_span("azure_audio_speech", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            audio_data = await provider.audio_speech(request)
            content_type_map = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav",
                "pcm": "audio/pcm",
            }
            content_type = content_type_map.get(request.response_format, "audio/mpeg")
            return Response(
                content=audio_data,
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"},
            )
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(501, "OperationNotSupported", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Audio speech error: {str(e)}")


@router.post(
    "/openai/deployments/{provider_name}/{deployment}/audio/transcriptions",
    tags=["azure_openai"],
)
async def azure_audio_transcription(
    provider_name: str,
    deployment: str,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[str] = Form(None),
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI audio transcription endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)

    with create_span("azure_audio_transcription", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            file_content = await file.read()
            file_b64 = base64.b64encode(file_content).decode("utf-8")

            granularities = None
            if timestamp_granularities:
                try:
                    granularities = timestamp_granularities.split(",")
                except Exception:
                    granularities = [timestamp_granularities]

            req = AudioTranscriptionRequest(
                file=file_b64,
                model=model_name,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=granularities,
            )
            result = await provider.audio_transcription(req)

            if response_format == "text":
                return Response(content=result.text, media_type="text/plain")
            return result
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(501, "OperationNotSupported", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Audio transcription error: {str(e)}")


@router.post(
    "/openai/deployments/{provider_name}/{deployment}/audio/translations",
    tags=["azure_openai"],
)
async def azure_audio_translation(
    provider_name: str,
    deployment: str,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI audio translation endpoint."""
    provider = _get_azure_provider(provider_name)
    model_name = _build_model_name(provider_name, deployment)

    with create_span("azure_audio_translation", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.deployment": deployment,
        })
        try:
            file_content = await file.read()
            file_b64 = base64.b64encode(file_content).decode("utf-8")

            req = AudioTranslationRequest(
                file=file_b64,
                model=model_name,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
            result = await provider.audio_translation(req)

            if response_format == "text":
                return Response(content=result.text, media_type="text/plain")
            return result
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(501, "OperationNotSupported", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Audio translation error: {str(e)}")


# ==================== Responses API ====================

@router.post(
    "/openai/deployments/{provider_name}/responses",
    tags=["azure_openai"],
)
async def azure_responses_create(
    request_obj: Request,
    provider_name: str,
    request: ResponsesCreateRequest,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI Responses API — create a response."""
    request_started_at = time.monotonic()
    provider = _get_azure_provider(provider_name)

    # For Responses API, model is in the request body — prefix it with provider
    if request.model and "/" not in request.model:
        request.model = _build_model_name(provider_name, request.model)

    with create_span("azure_responses_create", kind=trace.SpanKind.INTERNAL) as span:
        add_span_attributes(span, {
            "azure.provider": provider_name,
            "azure.api_version": api_version or "",
        })
        try:
            if request.stream:
                current_context = otel_context.get_current()
                traceparent = get_w3c_traceparent()
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                if traceparent:
                    headers["traceparent"] = traceparent

                return StreamingResponse(
                    stream_with_context_and_timeout(
                        provider.responses_create_stream(request),
                        current_context,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers=headers,
                )
            else:
                response = await provider.responses_create(request)
                return response
        except NotImplementedError as e:
            set_span_error(span, e)
            return _azure_error(501, "OperationNotSupported", str(e))
        except ValueError as e:
            set_span_error(span, e)
            return _azure_error(400, "InvalidRequest", str(e))
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Responses create error: {str(e)}")


@router.get(
    "/openai/deployments/{provider_name}/responses/{response_id}",
    tags=["azure_openai"],
)
async def azure_responses_retrieve(
    provider_name: str,
    response_id: str,
    include: Optional[List[str]] = Query(None),
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI Responses API — retrieve a response."""
    _get_azure_provider(provider_name)  # validate provider

    with create_span("azure_responses_retrieve", kind=trace.SpanKind.INTERNAL) as span:
        try:
            kwargs = {}
            if include is not None:
                kwargs["include"] = include
            response = await provider_manager.responses_retrieve(response_id, **kwargs)
            return response
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Responses retrieve error: {str(e)}")


@router.delete(
    "/openai/deployments/{provider_name}/responses/{response_id}",
    tags=["azure_openai"],
)
async def azure_responses_delete(
    provider_name: str,
    response_id: str,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI Responses API — delete a response."""
    _get_azure_provider(provider_name)  # validate provider

    with create_span("azure_responses_delete", kind=trace.SpanKind.INTERNAL) as span:
        try:
            response = await provider_manager.responses_delete(response_id)
            return response
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Responses delete error: {str(e)}")


@router.post(
    "/openai/deployments/{provider_name}/responses/{response_id}/cancel",
    tags=["azure_openai"],
)
async def azure_responses_cancel(
    provider_name: str,
    response_id: str,
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI Responses API — cancel a response."""
    _get_azure_provider(provider_name)  # validate provider

    with create_span("azure_responses_cancel", kind=trace.SpanKind.INTERNAL) as span:
        try:
            response = await provider_manager.responses_cancel(response_id)
            return response
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Responses cancel error: {str(e)}")


@router.get(
    "/openai/deployments/{provider_name}/responses/{response_id}/input_items",
    tags=["azure_openai"],
)
async def azure_responses_list_input_items(
    provider_name: str,
    response_id: str,
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query(None),
    include: Optional[List[str]] = Query(None),
    api_version: Optional[str] = Query(None, alias="api-version"),
    auth: Union[User, AdminUser, APIKey] = Depends(_authenticate_azure),
):
    """Azure OpenAI Responses API — list input items for a response."""
    _get_azure_provider(provider_name)  # validate provider

    with create_span("azure_responses_list_input_items", kind=trace.SpanKind.INTERNAL) as span:
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
        except Exception as e:
            set_span_error(span, e)
            return _azure_error(500, "InternalServerError", f"Responses list input items error: {str(e)}")

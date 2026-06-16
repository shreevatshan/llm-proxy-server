"""Anthropic Messages API route.

Handles POST /v1/messages for the Anthropic API server (port 2027).
Supports both streaming and non-streaming responses.
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse

from app.anthropic_models import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicErrorResponse,
    AnthropicErrorDetail,
    is_anthropic_terminal_stream_event,
)
from app.providers.provider_manager import provider_manager
from app.providers.bedrock_provider import (
    BedrockNativeIdleTimeout,
    BedrockNativePrematureEOF,
    BedrockNativeProviderError,
)
from app.providers.custom_providers import CustomProvider
from app.providers.base import AnthropicRequestMetadata, ProviderHTTPError
from app.auth.middleware import authenticate_anthropic_request
from app.auth.models import APIKey, User
from app.auth.admin import AdminUser
from app.tracing import (
    create_span,
    set_span_error,
    get_current_span,
    add_span_attributes,
)
from app.routes.stream_utils import (
    anthropic_stream_with_context_and_timeout,
    STREAM_TIMEOUT_SECONDS,
    format_anthropic_sse_event,
    set_request_tracking_outcome,
)
from app.rate_limit_dep import enforce_group_rate_limit
from opentelemetry import trace
from opentelemetry.context import get_current

logger = logging.getLogger(__name__)

router = APIRouter()


async def _check_model_exists(provider: object, model_name: str) -> str:
    """Check whether *model_name* is known to *provider* via the model cache.

    Returns one of:
      "cache_hit"        — model found in the in-memory cache
      "live_refresh_hit" — cache miss, but a live discovery call found it (cache updated)
      "missing"          — not in cache, not found after a live refresh
    """
    cache = provider_manager.model_cache
    cached_models = cache.get_enabled_models()
    known_ids = {m.id for m in cached_models}
    if model_name in known_ids:
        return "cache_hit"

    # Cache miss — try a single live refresh to handle the staleness window
    try:
        fresh_models: List = await provider.get_available_models()
        if fresh_models:
            # Merge fresh models into the existing cache snapshot (replace provider's slice)
            provider_prefix = getattr(provider, "full_provider_name", "")
            other_models = [m for m in cached_models if not m.id.startswith(f"{provider_prefix}/")]
            merged = other_models + fresh_models
            cache.update_models(merged)
            if any(m.id == model_name for m in fresh_models):
                return "live_refresh_hit"
    except Exception:
        logger.debug("Live model refresh failed for provider '%s'", getattr(provider, "full_provider_name", "unknown"), exc_info=True)

    return "missing"


def _anthropic_error(status_code: int, error_type: str, message: str):
    """Return a JSONResponse with Anthropic error format."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        }
    )


def _anthropic_headers_from_metadata(metadata: AnthropicRequestMetadata) -> dict[str, str]:
    headers = {
        "x-llmproxy-anthropic-mode": metadata.mode,
    }
    if metadata.dropped_fields:
        headers["x-llmproxy-dropped-anthropic-fields"] = ", ".join(metadata.dropped_fields)
    return headers


def _get_anthropic_request_metadata(
    provider: object,
    request: AnthropicMessagesRequest,
    anthropic_beta: Optional[str],
) -> AnthropicRequestMetadata:
    metadata_fn = getattr(provider, "get_anthropic_request_metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn(request, anthropic_beta=anthropic_beta)
        if isinstance(metadata, AnthropicRequestMetadata):
            return metadata
    mode_fn = getattr(provider, "get_anthropic_mode_for_model", None)
    if callable(mode_fn):
        mode = mode_fn(request.model)
        return AnthropicRequestMetadata(mode=mode or "unsupported")
    return AnthropicRequestMetadata(mode="unsupported")


_SAFE_UPSTREAM_HEADERS = {
    "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
    "content-type",
}


def _provider_http_error_response(error: ProviderHTTPError) -> JSONResponse:
    body = error.body
    if not isinstance(body, dict):
        body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": error.message,
            },
        }
    headers = {
        k: v for k, v in (error.headers or {}).items()
        if isinstance(k, str) and isinstance(v, str) and k.lower() in _SAFE_UPSTREAM_HEADERS
    }
    return JSONResponse(status_code=error.status_code, content=body, headers=headers)


def _parse_sse_chunk(chunk: str) -> tuple[Optional[str], Optional[dict]]:
    """Extract SSE event type and JSON payload from a formatted chunk."""
    event_type = None
    payload = None
    for line in chunk.splitlines():
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = None
    return event_type, payload


def _is_custom_anthropic_provider(provider: object) -> bool:
    """Return True for custom providers serving the Anthropic-compatible API."""
    return (
        isinstance(provider, CustomProvider)
        and "anthropic" in getattr(provider, "_supported_apis", [])
    )


def _get_effective_stream(
    request: AnthropicMessagesRequest,
    provider: object,
) -> bool:
    """Default omitted ``stream`` to True for custom Anthropic providers only."""
    if "stream" not in getattr(request, "model_fields_set", set()) and _is_custom_anthropic_provider(provider):
        return True
    return bool(request.stream)


async def _sync_request_tracking_streaming_mode(
    request_obj: Request,
    is_streaming: bool,
) -> None:
    """Keep request tracking aligned with the route's effective stream mode."""
    request_id = getattr(getattr(request_obj, "state", None), "tracking_request_id", None)
    if not request_id:
        return
    try:
        from app.request_tracker import request_tracker

        await request_tracker.update_streaming(request_id, is_streaming)
    except Exception:
        logger.debug("Unable to update request tracker streaming mode", exc_info=True)


@router.post("/v1/messages", tags=["anthropic"])
async def create_message(
    request_obj: Request,
    request: AnthropicMessagesRequest,
    anthropic_beta: Optional[str] = Header(default=None, alias="anthropic-beta"),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_anthropic_request),
):
    """Handle Anthropic Messages API requests (streaming and non-streaming)."""
    request_started_at = time.monotonic()
    with create_span(
        "anthropic_messages_request",
        kind=trace.SpanKind.INTERNAL
    ) as span:
        try:
            model_name = request.model
            add_span_attributes(span, {
                "anthropic.model": model_name,
                "anthropic.max_tokens": request.max_tokens,
                "anthropic.message_count": len(request.messages),
            })

            # Route to the appropriate provider (Anthropic API)
            provider = await provider_manager.get_anthropic_provider_for_model(model_name)
            if not provider:
                return _anthropic_error(
                    404,
                    "not_found_error",
                    f"Model '{model_name}' not found or not available via Anthropic API"
                )

            add_span_attributes(span, {
                "anthropic.provider": getattr(provider, 'full_provider_name', 'unknown'),
            })

            request_metadata = _get_anthropic_request_metadata(provider, request, anthropic_beta)
            if request_metadata.mode == "unsupported":
                return _anthropic_error(
                    404,
                    "not_found_error",
                    f"Model '{model_name}' not found or not available via Anthropic API"
                )

            # Group rate limit check (request-level limits already handled in middleware)
            await enforce_group_rate_limit(request_obj, auth, model_name, envelope_override="anthropic")

            # Pre-flight: confirm the specific model exists in the cached model list.
            preflight_result = await _check_model_exists(provider, model_name)
            if preflight_result == "missing":
                add_span_attributes(span, {"anthropic.model_preflight": "missing"})
                return _anthropic_error(
                    404,
                    "not_found_error",
                    f"Model '{model_name}' not found on provider "
                    f"'{getattr(provider, 'full_provider_name', 'unknown')}'"
                )
            add_span_attributes(span, {"anthropic.model_preflight": preflight_result})

            add_span_attributes(span, {
                "anthropic.mode": request_metadata.mode,
                "anthropic.transport": request_metadata.transport or "unknown",
                "anthropic.dropped_fields": ",".join(request_metadata.dropped_fields),
            })
            anthropic_headers = _anthropic_headers_from_metadata(request_metadata)

            effective_stream = _get_effective_stream(request, provider)
            await _sync_request_tracking_streaming_mode(request_obj, effective_stream)
            add_span_attributes(span, {
                "anthropic.stream": effective_stream,
            })

            if effective_stream:
                # Streaming response
                context_token = get_current()
                request_id = getattr(getattr(request_obj, "state", None), "tracking_request_id", None)
                stream_log_context = {
                    "request_id": request_id,
                    "model": model_name,
                    "stream": effective_stream,
                    "provider": getattr(provider, "full_provider_name", "unknown"),
                }

                async def generate():
                    terminal_event_seen = False
                    try:
                        async for chunk in provider.anthropic_messages_stream(request, anthropic_beta=anthropic_beta):
                            event_type, payload = _parse_sse_chunk(chunk)
                            # The first terminal error is still surfaced as the
                            # request outcome. Only error events that arrive
                            # after an earlier terminal event are ignored as
                            # post-terminal drain noise.
                            late_terminal_error = terminal_event_seen and event_type == "error"
                            if is_anthropic_terminal_stream_event(event_type=event_type, event_data=payload):
                                terminal_event_seen = True
                            if event_type == "error":
                                error_message = None
                                if isinstance(payload, dict):
                                    error_message = (
                                        ((payload.get("error") or {}).get("message"))
                                        or payload.get("message")
                                    )
                                if not late_terminal_error:
                                    set_request_tracking_outcome(
                                        request_obj,
                                        status="errored",
                                        termination_reason="stream_error",
                                        error=error_message,
                                    )
                                else:
                                    logger.warning(
                                        "Ignoring late Anthropic stream error after terminal event: %s",
                                        error_message,
                                    )
                            yield chunk
                    except BedrockNativeIdleTimeout as e:
                        logger.warning("Bedrock native idle timeout: %s", e)
                        current_span = get_current_span()
                        if current_span:
                            add_span_attributes(
                                current_span,
                                {
                                    "stream.termination_reason": "bedrock_native_idle_timeout",
                                    "bedrock.stream.phase": e.phase,
                                },
                            )
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="bedrock_native_idle_timeout",
                            error=str(e),
                        )
                        yield format_anthropic_sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "timeout_error",
                                "message": str(e),
                            }
                        })
                    except BedrockNativePrematureEOF as e:
                        logger.warning("Bedrock native premature EOF: %s", e)
                        current_span = get_current_span()
                        if current_span:
                            add_span_attributes(
                                current_span,
                                {
                                    "stream.termination_reason": "bedrock_native_premature_eof",
                                },
                            )
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="bedrock_native_premature_eof",
                            error=str(e),
                        )
                        yield format_anthropic_sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e),
                            }
                        })
                    except BedrockNativeProviderError as e:
                        logger.warning("Bedrock native provider error: %s", e)
                        current_span = get_current_span()
                        if current_span:
                            add_span_attributes(
                                current_span,
                                {
                                    "stream.termination_reason": "bedrock_native_provider_error",
                                    "bedrock.error_code": e.error_code,
                                },
                            )
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="bedrock_native_provider_error",
                            error=e.error_message,
                        )
                        yield format_anthropic_sse_event("error", e.body)
                    except NotImplementedError:
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="provider_not_supported",
                            error="Anthropic streaming not supported by provider",
                        )
                        yield format_anthropic_sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"Anthropic streaming not supported by provider"
                            }
                        })
                    except ProviderHTTPError as e:
                        logger.warning("Anthropic stream provider HTTP error: %s", e.message)
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="provider_http_error",
                            error=e.message,
                        )
                        body = e.body if isinstance(e.body, dict) else {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": e.message,
                            }
                        }
                        yield format_anthropic_sse_event("error", body)
                    except ValueError as e:
                        logger.warning(f"Anthropic stream validation error: {e}")
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="invalid_request",
                            error=str(e),
                        )
                        yield format_anthropic_sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": str(e)
                            }
                        })
                    except Exception as e:
                        logger.error(f"Anthropic stream error: {e}")
                        set_request_tracking_outcome(
                            request_obj,
                            status="errored",
                            termination_reason="stream_error",
                            error=str(e),
                        )
                        yield format_anthropic_sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e)
                            }
                        })

                return StreamingResponse(
                    anthropic_stream_with_context_and_timeout(
                        generate(),
                        context_token,
                        request_obj,
                        timeout=STREAM_TIMEOUT_SECONDS,
                        log_context=stream_log_context,
                        request_started_at=request_started_at,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        **anthropic_headers,
                    },
                )
            else:
                # Non-streaming response
                try:
                    response = await provider.anthropic_messages(request, anthropic_beta=anthropic_beta)
                    if hasattr(response, "model_dump"):
                        response = response.model_dump(exclude_none=True)
                    return JSONResponse(content=response, headers=anthropic_headers)
                except NotImplementedError:
                    return _anthropic_error(
                        501,
                        "api_error",
                        f"Anthropic Messages API not supported by provider"
                    )
                except ProviderHTTPError as e:
                    response = _provider_http_error_response(e)
                    for key, value in anthropic_headers.items():
                        response.headers[key] = value
                    return response
                except ValueError as e:
                    logger.warning(f"Anthropic request validation error: {e}")
                    return _anthropic_error(
                        400,
                        "invalid_request_error",
                        str(e)
                    )

        except HTTPException:
            raise
        except ValueError as e:
            logger.warning(f"Anthropic request validation error: {e}")
            return _anthropic_error(
                400,
                "invalid_request_error",
                str(e)
            )
        except Exception as e:
            logger.error(f"Anthropic messages error: {e}", exc_info=True)
            set_span_error(span, e)
            return _anthropic_error(
                500,
                "api_error",
                f"Internal server error: {str(e)}"
            )


def _estimate_text_tokens(text: str) -> int:
    """Rough local estimate: ~4 chars per token. Under-counts CJK, over-counts short texts."""
    if not text:
        return 0
    return (len(text) + 3) // 4


def _extract_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    if not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if hasattr(block, "model_dump"):
            data = block.model_dump(exclude_none=True)
        elif isinstance(block, dict):
            data = block
        else:
            parts.append(str(block))
            continue

        block_type = data.get("type")
        if block_type == "text":
            parts.append(data.get("text", ""))
        elif block_type == "image":
            source = data.get("source", {}) or {}
            parts.append(source.get("data", ""))
            parts.append(source.get("url", ""))
        elif block_type == "tool_use":
            parts.append(data.get("id", ""))
            parts.append(data.get("name", ""))
            parts.append(json.dumps(data.get("input", {}), ensure_ascii=False, sort_keys=True))
        elif block_type == "tool_result":
            parts.append(data.get("tool_use_id", ""))
            value = data.get("content")
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        elif block_type == "thinking":
            parts.append(data.get("thinking", ""))
            parts.append(data.get("signature", ""))
        elif block_type == "redacted_thinking":
            parts.append(data.get("data", ""))
        else:
            parts.append(json.dumps(data, ensure_ascii=False, sort_keys=True))

    return "".join(parts)


@router.post("/v1/messages/count_tokens", response_model=AnthropicCountTokensResponse, tags=["anthropic"])
async def count_message_tokens(
    payload: Dict[str, Any] = Body(...),
    auth: Union[User, AdminUser, APIKey] = Depends(authenticate_anthropic_request),
):
    """Estimate input tokens for Anthropic Messages payload locally (~4 chars/token heuristic)."""
    with create_span("anthropic_messages_count_tokens", kind=trace.SpanKind.INTERNAL) as span:
        try:
            if not isinstance(payload, dict):
                return _anthropic_error(
                    400,
                    "invalid_request_error",
                    "Request body must be a JSON object"
                )

            model_name = payload.get("model", "")
            provider = None
            if isinstance(model_name, str) and model_name:
                provider = await provider_manager.get_anthropic_provider_for_model(model_name)

            add_span_attributes(span, {
                "anthropic.model": model_name,
                "anthropic.provider_resolved": bool(provider),
            })

            if provider:
                try:
                    count_request = AnthropicCountTokensRequest.model_validate(payload)
                    provider_response = await provider.anthropic_count_tokens(count_request)
                    if hasattr(provider_response, "model_dump"):
                        provider_response = provider_response.model_dump(exclude_none=True)
                    if isinstance(provider_response, dict) and provider_response.get("input_tokens") is not None:
                        add_span_attributes(span, {
                            "anthropic.count_tokens_mode": "provider",
                        })
                        return AnthropicCountTokensResponse(input_tokens=int(provider_response["input_tokens"]))
                except NotImplementedError:
                    logger.debug("Provider %s does not implement anthropic_count_tokens", getattr(provider, "full_provider_name", "unknown"))
                except Exception as e:
                    logger.warning(
                        "Provider-backed Anthropic count_tokens failed for model '%s': %s",
                        model_name,
                        e,
                    )

            total_text_parts = []

            system = payload.get("system")
            if system is not None:
                if isinstance(system, str):
                    total_text_parts.append(system)
                else:
                    total_text_parts.append(_extract_content_text(system))

            messages = payload.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        total_text_parts.append(str(msg.get("role", "")))
                        total_text_parts.append(_extract_content_text(msg.get("content")))
                    else:
                        total_text_parts.append(str(msg))

            tools = payload.get("tools")
            if tools is not None:
                total_text_parts.append(json.dumps(tools, ensure_ascii=False, sort_keys=True))

            if payload.get("tool_choice") is not None:
                total_text_parts.append(json.dumps(payload.get("tool_choice"), ensure_ascii=False, sort_keys=True))

            if payload.get("thinking") is not None:
                total_text_parts.append(json.dumps(payload.get("thinking"), ensure_ascii=False, sort_keys=True))

            extra_payload = {
                k: v for k, v in payload.items()
                if k not in {"model", "messages", "system", "tools", "tool_choice", "thinking"}
            }
            if extra_payload:
                total_text_parts.append(json.dumps(extra_payload, ensure_ascii=False, sort_keys=True))

            combined = "\n".join(total_text_parts)
            input_tokens = _estimate_text_tokens(combined)
            add_span_attributes(span, {
                "anthropic.count_tokens_mode": "local_estimate",
            })
            if not provider:
                logger.info(
                    "Returning local Anthropic count_tokens estimate without provider resolution for model '%s'",
                    model_name,
                )
            return AnthropicCountTokensResponse(input_tokens=input_tokens)
        except Exception as e:
            logger.error(f"Anthropic count_tokens error: {e}", exc_info=True)
            set_span_error(span, e)
            return _anthropic_error(
                500,
                "api_error",
                f"Internal server error: {str(e)}"
            )

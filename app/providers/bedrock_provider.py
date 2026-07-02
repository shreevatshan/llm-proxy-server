"""
AWS Bedrock Provider Implementation for OpenAI-compatible Proxy Server

This provider implements AWS Bedrock integration with support for:
- Chat completions (streaming and non-streaming)
- Text completions
- Embeddings
- Multi-modal inputs (text and images)
- Tool/function calling
- Cross-region inference profiles
- Application inference profiles
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Literal, Optional
from collections import defaultdict
import boto3
import numpy as np
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel


# ---------------------------------------------------------------------------
#  Module-level concurrency primitive for the Converse path.
#  The Claude (native) path now runs on the AsyncAnthropicBedrock SDK, which
#  owns its own transport/concurrency; only the Converse / converse_stream
#  path (OpenAI chat + non-Claude Anthropic) still needs a bound, since it
#  runs on Starlette's default threadpool and is otherwise ungoverned.
# ---------------------------------------------------------------------------
_CONVERSE_SEMAPHORE_LIMIT = 15

# Governs the Converse / converse_stream path (run via Starlette's default
# threadpool, which is otherwise unbounded relative to Bedrock concurrency).
_converse_semaphore: Optional[asyncio.Semaphore] = None


def _get_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer env var with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "[BEDROCK] Invalid integer for %s=%r, using default %s",
            name,
            raw,
            default,
        )
        return default

    if value <= 0:
        logging.getLogger(__name__).warning(
            "[BEDROCK] Non-positive integer for %s=%r, using default %s",
            name,
            raw,
            default,
        )
        return default

    return value


def _get_converse_semaphore() -> asyncio.Semaphore:
    """Get or create the semaphore bounding the Converse / converse_stream path.

    Without this, the OpenAI chat path and the non-Claude Anthropic path call
    converse/converse_stream on Starlette's default threadpool with no Bedrock-
    side concurrency bound — the one ungoverned path relative to the native
    InvokeModel paths. Limit is configurable via BEDROCK_CONVERSE_SEMAPHORE_LIMIT.
    """
    global _converse_semaphore
    if _converse_semaphore is None:
        limit = _get_positive_int_env(
            "BEDROCK_CONVERSE_SEMAPHORE_LIMIT", _CONVERSE_SEMAPHORE_LIMIT
        )
        _converse_semaphore = asyncio.Semaphore(limit)
        logger.info("[BEDROCK] Created converse semaphore with limit %d", limit)
    return _converse_semaphore


def _map_bedrock_error(error_code: str, error_message: str) -> Dict[str, Any]:
    """Map a Bedrock ClientError to an Anthropic-format error dict + HTTP status.

    Returns ``{"status": int, "body": {...}}`` where ``body`` is a well-formed
    Anthropic error envelope.
    """
    _mapping = {
        "ThrottlingException": (429, "rate_limit_error"),
        "TooManyRequestsException": (429, "rate_limit_error"),
        "ServiceQuotaExceededException": (429, "rate_limit_error"),
        "ValidationException": (400, "invalid_request_error"),
        "AccessDeniedException": (403, "permission_error"),
        "ResourceNotFoundException": (404, "not_found_error"),
        "ModelNotReadyException": (503, "api_error"),
        "ServiceUnavailableException": (503, "api_error"),
        "ModelStreamErrorException": (500, "api_error"),
        "ModelErrorException": (500, "api_error"),
        "ModelTimeoutException": (408, "timeout_error"),
        "InternalServerException": (500, "api_error"),
    }
    http_status, error_type = _mapping.get(error_code, (500, "api_error"))
    return {
        "status": http_status,
        "body": {
            "type": "error",
            "error": {
                "type": error_type,
                "message": error_message,
            },
        },
    }


from app.providers.base import AnthropicRequestMetadata, BaseProvider
from app.anthropic_models import (
    ANTHROPIC_SDK_TIMEOUT_SECONDS,
    _extract_system_messages_from_messages,
    _merge_system_fields,
    is_anthropic_terminal_stream_event,
    is_claude_at_least,
)
from app.providers.anthropic_compatible import (
    get_anthropic_post_terminal_drain_stop_reason,
)
from app.openai_models import (
    ModelInfo,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
    ChatCompletionChoice,
    ChatCompletionStreamChoice,
    CompletionChoice,
    CompletionStreamChoice,
    ChatMessage,
    ChatMessageDelta,
    Usage,
    ChatResponseMessage,
    ToolCall,
    ResponseFunction,
    TextContent,
    ImageContent,
    Tool,
    Function,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)


logger = logging.getLogger(__name__)


def _raise_anthropic_api_error(e: Exception) -> None:
    """Re-raise Anthropic SDK errors as HTTPException with proper status codes.

    Forwards the upstream error body as-is so the client receives the exact
    error response from the provider instead of a synthetic one.
    Only raises if ``e`` is an ``anthropic.APIStatusError``; otherwise returns
    so the caller can re-raise the original exception.
    """
    import anthropic
    if isinstance(e, anthropic.APIStatusError):
        from fastapi import HTTPException
        # Forward the upstream error body directly — avoid re-constructing it
        detail = e.body if hasattr(e, "body") and e.body is not None else {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        raise HTTPException(
            status_code=e.status_code,
            detail=detail,
        )


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to ensure it contains no Pydantic models.
    Converts all Pydantic models to dictionaries.
    """
    if isinstance(obj, BaseModel):
        # Convert Pydantic model to dict
        if hasattr(obj, 'model_dump'):
            obj = obj.model_dump()
        elif hasattr(obj, 'dict'):
            obj = obj.dict()
        else:
            return str(obj)
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    else:
        return obj


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider implementation."""

    # Supported embedding models
    SUPPORTED_EMBEDDING_MODELS = {
        "cohere.embed-multilingual-v3": "Cohere Embed Multilingual",
        "cohere.embed-english-v3": "Cohere Embed English",
        "amazon.titan-embed-text-v1": "Titan Embeddings G1 - Text",
        "amazon.titan-embed-text-v2:0": "Titan Embeddings G2 - Text",
    }

    ANTHROPIC_TO_BEDROCK_MODEL_MAP = {
        "claude-3-5-sonnet-latest": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-sonnet-20241022": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku-latest": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-5-haiku-20241022": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-7-sonnet-latest": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3-7-sonnet-20250219": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3-opus-latest": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
    }

    ANTHROPIC_BETA_MAP = {
        "advanced-tool-use-2025-11-20": [
            "tool-examples-2025-10-29",
            "tool-search-tool-2025-10-19",
        ],
    }

    ANTHROPIC_BETA_PASSTHROUGH = {
        "fine-grained-tool-streaming-2025-05-14",
        "interleaved-thinking-2025-05-14",
        "context-management-2025-06-27",
        "compact-2026-01-12",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # AWS Configuration - matches database schema
        # region, access_key_id, secret_access_key from database
        self.aws_region = config.get('region', 'us-east-1')
        self.aws_access_key = config.get('access_key_id')
        self.aws_secret_key = config.get('secret_access_key')
        
        # Advanced features - enabled by default as per provider_manager
        self.enable_cross_region = config.get('enable_cross_region_inference', True)
        self.enable_app_profiles = config.get('enable_application_inference_profiles', True)
        
        # Default model configuration
        self.default_model = config.get('default_model', 'anthropic.claude-3-sonnet-20240229-v1:0')
        self.debug = config.get('debug', False)
        
        # Configure boto3 client with custom config
        boto_config = Config(
            connect_timeout=60,
            read_timeout=900,
            retries={'max_attempts': 8, 'mode': 'adaptive'},
            max_pool_connections=50
        )

        # Initialize boto3 clients
        client_kwargs = {
            'service_name': 'bedrock-runtime',
            'region_name': self.aws_region,
            'config': boto_config
        }

        if self.aws_access_key and self.aws_secret_key:
            client_kwargs['aws_access_key_id'] = self.aws_access_key
            client_kwargs['aws_secret_access_key'] = self.aws_secret_key

        self.bedrock_runtime = boto3.client(**client_kwargs)

        # Client for listing models
        control_client_kwargs = {
            'service_name': 'bedrock',
            'region_name': self.aws_region,
            'config': boto_config
        }
        
        if self.aws_access_key and self.aws_secret_key:
            control_client_kwargs['aws_access_key_id'] = self.aws_access_key
            control_client_kwargs['aws_secret_access_key'] = self.aws_secret_key
        
        self.bedrock_client = boto3.client(**control_client_kwargs)
        
        # Cache for model list
        self.bedrock_model_list = {}

        self._anthropic_client = None

    def get_model_id(self, model_name: str) -> str:
        """
        Extract actual Bedrock model ID from provider-prefixed model name.
        
        Examples:
            "bedrock:default/anthropic.claude-3-sonnet-20240229-v1:0" 
              -> "anthropic.claude-3-sonnet-20240229-v1:0"
            
            "anthropic.claude-3-sonnet-20240229-v1:0" 
              -> "anthropic.claude-3-sonnet-20240229-v1:0"
        """
        original_model = model_name
        
        # Remove provider prefix if present (e.g., "bedrock:default/model-id" -> "model-id")
        if '/' in model_name:
            model_name = model_name.split('/', 1)[1]
        
        #if self.debug:
        #    logger.info(f"[BEDROCK] Model name transformation: '{original_model}' -> '{model_name}'")
        
        return model_name

    def _get_inference_region_prefix(self) -> str:
        """Get inference region prefix for cross-region inference."""
        if self.aws_region.startswith("ap-"):
            return "apac"
        return self.aws_region[:2]

    # Model-list cache TTL: control-plane calls (list_inference_profiles x2 +
    # list_foundation_models) are slow and rate-limit-prone, so serve a cached
    # list on the request path and only refresh past this interval.
    _MODEL_LIST_TTL_SECONDS = 600  # 10 minutes

    def _refresh_model_list(self, force: bool = False):
        """Refresh the list of available Bedrock models.

        Skips the (blocking, paginated) control-plane calls when a populated
        list was fetched within the TTL, unless ``force`` is set. The final
        assignment to ``self.bedrock_model_list`` is a single reference swap,
        so concurrent readers always see either the old or new map whole —
        never a half-built one.
        """
        if not force:
            last = getattr(self, "_model_list_fetched_at", None)
            if last is not None and self.bedrock_model_list and (time.monotonic() - last) < self._MODEL_LIST_TTL_SECONDS:
                return
        try:
            model_list = {}
            profile_list = []
            app_profiles_by_model = defaultdict(set)
            
            # Get cross-region inference profiles
            if self.enable_cross_region:
                try:
                    paginator = self.bedrock_client.get_paginator('list_inference_profiles')
                    for page in paginator.paginate(maxResults=1000, typeEquals="SYSTEM_DEFINED"):
                        profile_list.extend([p["inferenceProfileId"] for p in page["inferenceProfileSummaries"]])
                except Exception as e:
                    logger.warning(f"Error listing cross-region inference profiles: {e}")
            
            # Get application inference profiles
            if self.enable_app_profiles:
                try:
                    paginator = self.bedrock_client.get_paginator('list_inference_profiles')
                    for page in paginator.paginate(maxResults=1000, typeEquals="APPLICATION"):
                        for profile in page["inferenceProfileSummaries"]:
                            try:
                                profile_arn = profile.get("inferenceProfileArn")
                                if not profile_arn:
                                    continue
                                
                                models = profile.get("models", [])
                                for model in models:
                                    model_arn = model.get("modelArn", "")
                                    if model_arn:
                                        model_id = model_arn.split('/')[-1] if '/' in model_arn else model_arn
                                        if model_id:
                                            app_profiles_by_model[model_id].add(profile_arn)
                            except Exception as e:
                                logger.warning(f"Error processing application profile: {e}")
                except Exception as e:
                    logger.warning(f"Error listing application inference profiles: {e}")
            
            # List foundation models - removed byOutputModality filter
            #response = self.bedrock_client.list_foundation_models(byOutputModality="TEXT")
            response = self.bedrock_client.list_foundation_models()
            cr_inference_prefix = self._get_inference_region_prefix()
            
            for model in response["modelSummaries"]:
                model_id = model.get("modelId", "N/A")
                status = model["modelLifecycle"].get("status", "ACTIVE")
                
                # Filter only by status
                if status not in ["ACTIVE", "LEGACY"]:
                    continue
                
                inference_types = model.get("inferenceTypesSupported", [])
                input_modalities = model["inputModalities"]
                
                # Add on-demand model
                if "ON_DEMAND" in inference_types:
                    model_list[model_id] = {"modalities": input_modalities}
                
                # Add cross-region inference model
                profile_id = cr_inference_prefix + "." + model_id
                if profile_id in profile_list:
                    model_list[profile_id] = {"modalities": input_modalities}
                
                # Add global cross-region inference profiles
                global_profile_id = "global." + model_id
                if global_profile_id in profile_list:
                    model_list[global_profile_id] = {"modalities": input_modalities}
                
                # Add application inference profiles
                if model_id in app_profiles_by_model:
                    for profile_arn in app_profiles_by_model[model_id]:
                        model_list[profile_arn] = {"modalities": input_modalities}
            
            if not model_list:
                # Fallback to default model
                model_list[self.default_model] = {"modalities": ["TEXT", "IMAGE"]}
            
            # Atomic swap — readers see old or new map whole, never partial.
            self.bedrock_model_list = model_list
            self._model_list_fetched_at = time.monotonic()
            logger.info(f"Loaded {len(model_list)} Bedrock models")

        except Exception as e:
            logger.error(f"Error listing Bedrock models: {e}")
            # Set a default model only if we have nothing cached; don't clobber a
            # previously-good list on a transient control-plane failure.
            if not self.bedrock_model_list:
                self.bedrock_model_list = {self.default_model: {"modalities": ["TEXT", "IMAGE"]}}

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from Bedrock."""
        await run_in_threadpool(self._refresh_model_list)

        models = []
        for model_id in self.bedrock_model_list.keys():
            models.append(self.create_model_info(
                model_id=model_id,
                owned_by="bedrock"
            ))
        
        return models
    
    async def refresh_models(self) -> None:
        """Explicitly refresh the model list from Bedrock (bypasses the TTL)."""
        await run_in_threadpool(self._refresh_model_list, force=True)

    def _truncate_tool_name(self, tool_name: str, max_length: int = 64, tool_name_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Truncate tool name to fit Bedrock's 64 character limit.
        
        Uses a hash suffix to preserve uniqueness for truncated names.
        If tool_name_mapping dict is provided, stores the mapping for
        restoring original names in responses.
        """
        if not tool_name or len(tool_name) <= max_length:
            return tool_name

        # Create a short hash of the full name for uniqueness
        name_hash = hashlib.md5(tool_name.encode()).hexdigest()[:8]
        # Truncate name to (max_length - 9) chars + underscore + 8 char hash
        truncated_name = tool_name[:max_length - 9] + "_" + name_hash
        
        # Store mapping for reverse lookup in responses
        if tool_name_mapping is not None:
            tool_name_mapping[truncated_name] = tool_name
        
        logger.debug(
            f"[BEDROCK] Tool name truncated from {len(tool_name)} to {max_length} chars: "
            f"'{tool_name}' -> '{truncated_name}'"
        )
        return truncated_name

    def _restore_tool_name(self, truncated_name: str, tool_name_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Restore original tool name from truncated name.
        
        Returns the original name if a mapping exists, otherwise returns the input unchanged.
        """
        if tool_name_mapping is not None:
            return tool_name_mapping.get(truncated_name, truncated_name)
        return truncated_name

    def _is_supported_modality(self, model_id: str, modality: str = "IMAGE") -> bool:
        """Check if model supports a specific modality."""
        model = self.bedrock_model_list.get(model_id, {})
        modalities = model.get("modalities", [])
        return modality in modalities

    def _parse_system_prompts(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Extract system prompts from messages."""
        system_prompts = []
        for message in messages:
            if message.role == "system":
                if isinstance(message.content, str):
                    system_prompts.append({"text": message.content})
        return system_prompts

    # Image download guard rails (SSRF + resource exhaustion).
    _IMAGE_DOWNLOAD_TIMEOUT = 10  # seconds
    _IMAGE_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

    def _assert_safe_image_url(self, image_url: str) -> None:
        """Reject URLs that could be used for SSRF before fetching them.

        Blocks non-http(s) schemes and hosts that resolve to private,
        loopback, link-local, or otherwise non-global IP ranges (e.g. cloud
        metadata endpoints at 169.254.169.254).
        """
        import ipaddress
        import socket as _socket
        from urllib.parse import urlparse

        parsed = urlparse(image_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r}")
        host = parsed.hostname
        if not host:
            raise ValueError("Image URL has no host")

        try:
            addrinfos = _socket.getaddrinfo(host, None)
        except _socket.gaierror as e:
            raise ValueError(f"Could not resolve image host: {host}") from e

        for info in addrinfos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
                raise ValueError(f"Image URL resolves to a disallowed address: {ip}")

    def _parse_image_sync(self, image_url: str) -> tuple:
        """Parse image from URL or base64 data (synchronous - for use in threadpool)."""
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)

        # Check if already base64 encoded
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)

        # SSRF guard before any network access.
        self._assert_safe_image_url(image_url)

        # Download from URL (blocking - must be called from threadpool), with a
        # size cap so a large/slow image can't pin a worker or exhaust memory.
        response = requests.get(image_url, timeout=self._IMAGE_DOWNLOAD_TIMEOUT, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Unable to access image URL: {image_url}")

        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > self._IMAGE_MAX_BYTES:
                response.close()
                raise ValueError(f"Image exceeds maximum size of {self._IMAGE_MAX_BYTES} bytes")
            chunks.append(chunk)

        content_type = response.headers.get("Content-Type", "image/jpeg")
        if not content_type.startswith("image"):
            content_type = "image/jpeg"
        return b"".join(chunks), content_type

    def _parse_content_parts(self, message: ChatMessage, model_id: str) -> List[Dict]:
        """Parse message content into Bedrock format."""
        if isinstance(message.content, str):
            return [{"text": message.content}]
        
        content_parts = []
        
        for part in message.content:
            if isinstance(part, TextContent):
                # Only add non-empty text content
                if part.text and part.text.strip():
                    content_parts.append({"text": part.text})
                else:
                    logger.warning(f"[BEDROCK] Skipping empty TextContent: '{part.text}'")
            elif isinstance(part, ImageContent):
                if not self._is_supported_modality(model_id, "IMAGE"):
                    raise ValueError(f"Model {model_id} does not support images")
                
                image_data, content_type = self._parse_image_sync(part.image_url.url)
                content_parts.append({
                    "image": {
                        "format": content_type.replace("image/", ""),
                        "source": {"bytes": image_data}
                    }
                })

        # Converse rejects empty content arrays with a ValidationException. If a
        # message had only empty/whitespace text parts and nothing else, emit a
        # single-space placeholder so the message remains valid.
        if not content_parts:
            logger.warning("[BEDROCK] Message content was empty after parsing; inserting placeholder text")
            content_parts.append({"text": " "})

        return content_parts

    def _extract_tool_content(self, content) -> str:
        """Extract text content from tool message."""
        try:
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            item_text = item["text"]
                            if isinstance(item_text, str):
                                # Try to parse as JSON if it looks like JSON
                                if item_text.strip().startswith('{') and item_text.strip().endswith('}'):
                                    try:
                                        parsed_json = json.loads(item_text)
                                        text_parts.append(json.dumps(parsed_json, indent=2))
                                    except json.JSONDecodeError:
                                        text_parts.append(item_text)
                                else:
                                    text_parts.append(item_text)
                            else:
                                text_parts.append(str(item_text))
                        else:
                            text_parts.append(json.dumps(item, indent=2))
                    elif hasattr(item, 'text'):
                        text_parts.append(item.text)
                    else:
                        text_parts.append(str(item))
                return "\n".join(text_parts)
            
            return str(content)
        except Exception as e:
            logger.warning(f"Tool content extraction failed: {e}")
            return str(content) if content is not None else ""

    def _reframe_multi_payload(self, messages: List[Dict]) -> List[Dict]:
        """
        Combine consecutive messages from same role, but keep toolResult blocks separate.
        
        Bedrock requires that after an assistant message with toolUse, the immediately 
        following user message must contain ONLY the toolResult(s), not mixed with other text.
        """
        reformatted_messages = []
        current_role = None
        current_content = []
        current_has_tool_result = False
        
        def flush_current():
            nonlocal current_role, current_content, current_has_tool_result
            if current_content:
                reformatted_messages.append({
                    "role": current_role,
                    "content": current_content
                })
            current_content = []
            current_has_tool_result = False
        
        for message in messages:
            next_role = message["role"]
            next_content = message["content"]

            # Check if this message contains toolResult
            has_tool_result = False
            if isinstance(next_content, list):
                for item in next_content:
                    if isinstance(item, dict):
                        if "toolResult" in item:
                            has_tool_result = True
                            break
            
            # Decide whether to flush current content
            if next_role != current_role:
                # Role changed - flush
                flush_current()
                current_role = next_role
            elif next_role == "user":
                # Same role (user) - check if we need to separate toolResult from regular content
                if has_tool_result and not current_has_tool_result and current_content:
                    # Current has regular content, new has toolResult - flush first
                    flush_current()
                    current_role = next_role
                elif not has_tool_result and current_has_tool_result:
                    # Current has toolResult, new has regular content - flush first
                    flush_current()
                    current_role = next_role
            # Add content
            if isinstance(next_content, str):
                current_content.append({"text": next_content})
            elif isinstance(next_content, list):
                current_content.extend(next_content)
            
            # Track if current accumulated content has toolResult
            if has_tool_result:
                current_has_tool_result = True
        
        # Flush remaining
        flush_current()
        
        # # Debug log: messages before validation
        # logger.warning(f"[BEDROCK DEBUG] Messages BEFORE validation ({len(reformatted_messages)} messages):")
        # for idx, msg in enumerate(reformatted_messages):
        #     role = msg.get("role")
        #     content = msg.get("content", [])
        #     tool_uses = []
        #     tool_results = []
        #     text_count = 0
        #     if isinstance(content, list):
        #         for item in content:
        #             if isinstance(item, dict):
        #                 if "toolUse" in item:
        #                     tool_uses.append(item["toolUse"].get("toolUseId", "?"))
        #                 elif "toolResult" in item:
        #                     tool_results.append(item["toolResult"].get("toolUseId", "?"))
        #                 elif "text" in item:
        #                     text_count += 1
        #     logger.warning(f"  [{idx}] {role}: texts={text_count}, toolUses={tool_uses}, toolResults={tool_results}")
        
        validated = self._validate_tool_use_result_pairing(reformatted_messages)
        
        # # Debug log: messages after validation
        # logger.warning(f"[BEDROCK DEBUG] Messages AFTER validation ({len(validated)} messages):")
        # for idx, msg in enumerate(validated):
        #     role = msg.get("role")
        #     content = msg.get("content", [])
        #     tool_uses = []
        #     tool_results = []
        #     text_count = 0
        #     if isinstance(content, list):
        #         for item in content:
        #             if isinstance(item, dict):
        #                 if "toolUse" in item:
        #                     tool_uses.append(item["toolUse"].get("toolUseId", "?"))
        #                 elif "toolResult" in item:
        #                     tool_results.append(item["toolResult"].get("toolUseId", "?"))
        #                 elif "text" in item:
        #                     text_count += 1
        #     logger.warning(f"  [{idx}] {role}: texts={text_count}, toolUses={tool_uses}, toolResults={tool_results}")
        
        return validated

    def _validate_tool_use_result_pairing(self, messages: List[Dict]) -> List[Dict]:
        """
        Validate and repair tool_use/tool_result pairing in messages.

        AWS Bedrock (Converse) requires that every toolUse block has a matching
        toolResult block and vice-versa. The real Anthropic Messages API is more
        lenient about where they sit, so a toolUse and its toolResult can legally
        end up non-adjacent after reframing (e.g. assistant text + toolUse split,
        or tool results answered across several user turns).

        This method pairs toolUse<->toolResult by toolUseId across the WHOLE
        message list (not just immediately-adjacent messages), and only drops a
        block when its partner is missing ANYWHERE. This preserves the orphan
        cleanup Converse needs without deleting valid tool calls.
        """
        if not messages:
            return messages

        # First pass: collect every toolUse id and every toolResult id across
        # all messages. A tool id is valid (kept) iff it appears on BOTH sides.
        all_tool_use_ids = set()
        all_tool_result_ids = set()

        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict):
                    if "toolUse" in item:
                        tid = item["toolUse"].get("toolUseId")
                        if tid:
                            all_tool_use_ids.add(tid)
                    elif "toolResult" in item:
                        tid = item["toolResult"].get("toolUseId")
                        if tid:
                            all_tool_result_ids.add(tid)

        valid_tool_ids = all_tool_use_ids & all_tool_result_ids

        if self.debug:
            logger.info(f"[BEDROCK] Valid tool IDs (paired across all messages): {valid_tool_ids}")
        
        # Second pass: filter messages to only keep valid tool_use and tool_result blocks
        validated_messages = []
        
        for i, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", [])
            
            if isinstance(content, list):
                filtered_content = []
                removed_items = []
                
                for item in content:
                    if isinstance(item, dict):
                        if "toolUse" in item:
                            tool_use_id = item["toolUse"].get("toolUseId")
                            if tool_use_id not in valid_tool_ids:
                                removed_items.append(f"toolUse:{tool_use_id}")
                                continue  # Skip orphaned toolUse
                        elif "toolResult" in item:
                            tool_result_id = item["toolResult"].get("toolUseId")
                            if tool_result_id not in valid_tool_ids:
                                removed_items.append(f"toolResult:{tool_result_id}")
                                continue  # Skip orphaned toolResult
                    filtered_content.append(item)
                
                if removed_items:
                    logger.warning(
                        f"[BEDROCK] Removed orphaned items at message index {i} ({role}): {removed_items}"
                    )
                
                # Only add message if there's remaining content
                if filtered_content:
                    validated_messages.append({
                        "role": role,
                        "content": filtered_content
                    })
                else:
                    logger.warning(
                        f"[BEDROCK] Skipping {role} message at index {i} as all content was orphaned"
                    )
            else:
                validated_messages.append(message)
        
        return validated_messages

    def _parse_messages(self, request: ChatCompletionRequest, tool_name_mapping: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Convert OpenAI messages to Bedrock format."""
        messages = []
        
        # # Debug log: incoming OpenAI-format messages
        # logger.warning(f"[BEDROCK DEBUG] Incoming OpenAI messages ({len(request.messages)} messages):")
        # for idx, msg in enumerate(request.messages):
        #     role = msg.role
        #     has_content = bool(msg.content)
        #     tool_calls = []
        #     tool_call_id = getattr(msg, 'tool_call_id', None)
        #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #         for tc in msg.tool_calls:
        #             if hasattr(tc, 'id'):
        #                 tool_calls.append(tc.id)
        #             elif isinstance(tc, dict):
        #                 tool_calls.append(tc.get('id', '?'))
        #     logger.warning(f"  [{idx}] {role}: has_content={has_content}, tool_calls={tool_calls}, tool_call_id={tool_call_id}")
        
        for message in request.messages:
            if message.role == "system":
                continue  # Handled separately
            
            elif message.role == "user":
                messages.append({
                    "role": "user",
                    "content": self._parse_content_parts(message, request.model)
                })
            
            elif message.role == "assistant":
                # Check if message has content
                has_content = False
                if isinstance(message.content, str):
                    has_content = message.content.strip() != ""
                elif isinstance(message.content, list):
                    has_content = len(message.content) > 0
                elif message.content is not None:
                    has_content = True
                
                if has_content:
                    messages.append({
                        "role": "assistant",
                        "content": self._parse_content_parts(message, request.model)
                    })
                
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # Handle both Pydantic models and plain dicts
                        if hasattr(tool_call, 'model_dump'):
                            tool_call_dict = tool_call.model_dump()
                        elif hasattr(tool_call, 'dict'):
                            tool_call_dict = tool_call.dict()
                        else:
                            tool_call_dict = tool_call if isinstance(tool_call, dict) else {}
                        
                        # Extract function arguments
                        function_data = tool_call_dict.get('function', {}) if isinstance(tool_call_dict, dict) else {}
                        arguments_str = function_data.get('arguments', '{}')
                        
                        # Parse arguments if they're a string
                        if isinstance(arguments_str, str):
                            try:
                                tool_input = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                tool_input = {}
                        else:
                            tool_input = arguments_str
                        
                        # Truncate tool name to fit Bedrock's 64 char limit
                        tool_name = self._truncate_tool_name(function_data.get('name', ''), tool_name_mapping=tool_name_mapping)
                        
                        messages.append({
                            "role": "assistant",
                            "content": [{
                                "toolUse": {
                                    "toolUseId": tool_call_dict.get('id', ''),
                                    "name": tool_name,
                                    "input": tool_input
                                }
                            }]
                        })
            
            elif message.role == "tool":
                tool_content = self._extract_tool_content(message.content)
                messages.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": message.tool_call_id,
                            "content": [{"text": tool_content}]
                        }
                    }]
                })
        
        return self._reframe_multi_payload(messages)

    def _convert_tool_spec(self, func: Function, tool_name_mapping: Optional[Dict[str, str]] = None) -> Dict:
        """Convert OpenAI tool spec to Bedrock format."""
        # Ensure func is converted to dict if it's a Pydantic model
        if hasattr(func, 'model_dump'):
            func_dict = func.model_dump()
        elif hasattr(func, 'dict'):
            func_dict = func.dict()
        else:
            func_dict = {
                "name": func.name if hasattr(func, 'name') else str(func),
                "description": func.description if hasattr(func, 'description') else None,
                "parameters": func.parameters if hasattr(func, 'parameters') else {}
            }
        
        # Bedrock requires description to have minimum length of 1
        # Use a default description if none provided or empty
        description = func_dict.get("description")
        if not description or len(description.strip()) == 0:
            description = func_dict.get("name", "Tool function")
        
        # Bedrock requires tool name to be <= 64 characters
        tool_name = self._truncate_tool_name(func_dict.get("name", ""), tool_name_mapping=tool_name_mapping)
        
        return {
            "toolSpec": {
                "name": tool_name,
                "description": description,
                "inputSchema": {
                    "json": func_dict.get("parameters", {})
                }
            }
        }

    def _messages_contain_tool_usage(self, messages: List[Dict]) -> bool:
        """
        Check if any message in the conversation contains toolUse or toolResult blocks.
        
        AWS Bedrock requires toolConfig to be present whenever messages contain
        tool-related content, even if no new tools are being provided in the current request.
        """
        return any(
            isinstance(item, dict) and ("toolUse" in item or "toolResult" in item)
            for message in messages
            for content in [message.get("content", [])]
            if isinstance(content, list)
            for item in content
        )

    def _parse_bedrock_request(self, request: ChatCompletionRequest, tool_name_mapping: Optional[Dict[str, str]] = None) -> Dict:
        """Build Bedrock converse API request."""
        messages = self._parse_messages(request, tool_name_mapping=tool_name_mapping)
        system_prompts = self._parse_system_prompts(request.messages)
        
        # Base inference parameters
        inference_config = {
            "maxTokens": request.max_tokens or 2048,
        }
        
        # Only include optional parameters when specified
        if request.temperature is not None:
            inference_config["temperature"] = request.temperature
        if request.top_p is not None:
            inference_config["topP"] = request.top_p
        
        # Claude >= 4.7 don't support temperature or topP in inferenceConfig.
        # (topK was never placed in inferenceConfig — it lives in
        # additionalModelRequestFields and is scrubbed separately below.)
        if self._is_claude_at_least(request.model, 4, 7):
            inference_config.pop("temperature", None)
            inference_config.pop("topP", None)
            if self.debug:
                logger.info(f"Removed temperature and topP for {request.model} (not supported in Claude >= 4.7)")
        # Claude >= 4.5 don't support both temperature and topP simultaneously
        # When both are provided, temperature takes precedence and topP is removed
        elif "temperature" in inference_config and "topP" in inference_config:
            if self._is_claude_at_least(request.model, 4, 5):
                inference_config.pop("topP", None)
                if self.debug:
                    logger.info(f"Removed topP for {request.model} (conflicts with temperature)")
        
        if request.stop:
            stop = request.stop if isinstance(request.stop, list) else [request.stop]
            inference_config["stopSequences"] = stop
        
        args = {
            "modelId": request.model,
            "messages": messages,
            "system": system_prompts,
            "inferenceConfig": inference_config
        }
        
        # Handle reasoning effort
        if request.reasoning_effort:
            max_tokens = request.max_completion_tokens or request.max_tokens or 2048
            inference_config["maxTokens"] = max_tokens
            # Reasoning requires budget_tokens >= 1024 AND < max_tokens. If
            # max_tokens is too small to satisfy the floor, skip reasoning
            # rather than emit an invalid budget that Bedrock would reject.
            if max_tokens <= self._MIN_BUDGET_TOKENS:
                logger.warning(
                    f"[BEDROCK] max_tokens={max_tokens} too small for reasoning "
                    f"(needs > {self._MIN_BUDGET_TOKENS}); skipping reasoning_config"
                )
            else:
                budget_tokens = self._calc_budget_tokens(max_tokens, request.reasoning_effort)
                # unset topP - Not supported
                inference_config.pop("topP", None)
                args["additionalModelRequestFields"] = {
                    "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
                }
        
        # Check if messages contain tool usage (toolUse or toolResult blocks)
        # AWS Bedrock requires toolConfig to be present when any message contains these blocks
        messages_have_tools = self._messages_contain_tool_usage(messages)
        
        # Handle tools
        if request.tools or messages_have_tools:
            # Build tool config from current request tools, or use empty tools list if none provided
            # but message history contains tool usage
            tool_config = {
                "tools": [self._convert_tool_spec(t.function, tool_name_mapping=tool_name_mapping) for t in request.tools] if request.tools else []
            }
            
            if request.tool_choice and request.tools and not request.model.startswith("meta.llama3-1-"):
                if isinstance(request.tool_choice, str):
                    if request.tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    if isinstance(request.tool_choice, dict) and "function" in request.tool_choice:
                        tool_config["toolChoice"] = {
                            "tool": {"name": request.tool_choice["function"].get("name", "")}
                        }
            
            args["toolConfig"] = tool_config
            
            if self.debug and messages_have_tools and not request.tools:
                logger.info(f"Added empty toolConfig for {request.model} due to tool usage in message history")
        
        # add Additional fields to enable extend thinking
        # Access extra fields from Pydantic model
        extra_fields = {}
        if hasattr(request, 'model_extra') and request.model_extra:
            extra_fields = request.model_extra
        elif hasattr(request, '__pydantic_extra__') and request.__pydantic_extra__:
            extra_fields = request.__pydantic_extra__
        
        if extra_fields:
            # Filter out OpenAI-specific parameters that Bedrock doesn't support
            bedrock_unsupported_params = {
                'parallel_tool_calls',  # OpenAI-specific tool calling parameter
                'n',  # Number of completions (OpenAI-specific)
                'logprobs',  # Log probabilities (OpenAI-specific)
                'top_logprobs',  # Top log probabilities (OpenAI-specific)
                'logit_bias',  # Logit bias (OpenAI-specific)
                'response_format',  # Response format (handled differently in Bedrock)
                'seed',  # Seed (OpenAI-specific)
                'user',  # User identifier (OpenAI-specific)
                'frequency_penalty',  # Frequency penalty (not in Bedrock)
                'presence_penalty',  # Presence penalty (not in Bedrock)
                'stream_options',  # Stream options (handled separately)
                'verbosity',  # Not supported by Bedrock Converse API
            }
            
            # Filter out unsupported parameters
            filtered_extra_fields = {
                k: v for k, v in extra_fields.items()
                if k not in bedrock_unsupported_params
            }

            # Strip effort from output_config — Converse path is non-Claude only
            if "output_config" in filtered_extra_fields:
                oc = filtered_extra_fields["output_config"]
                if isinstance(oc, dict) and "effort" in oc:
                    oc = {k: v for k, v in oc.items() if k != "effort"}
                    if oc:
                        filtered_extra_fields["output_config"] = oc
                    else:
                        del filtered_extra_fields["output_config"]
            
            if filtered_extra_fields:
                # Merge into (not overwrite) additionalModelRequestFields so a
                # reasoning_config set from request.reasoning_effort above is
                # preserved. Explicit extra fields win on key conflicts.
                existing_amrf = args.get("additionalModelRequestFields", {})
                existing_amrf.update(filtered_extra_fields)
                args["additionalModelRequestFields"] = existing_amrf
                # Extended thinking doesn't support both temperature and topP
                # Remove topP to avoid validation error
                if "thinking" in filtered_extra_fields:
                    inference_config.pop("topP", None)

        # Claude >= 4.7 doesn't support top_k in additionalModelRequestFields either
        if self._is_claude_at_least(request.model, 4, 7):
            amrf = args.get("additionalModelRequestFields")
            if amrf and "top_k" in amrf:
                amrf.pop("top_k")
                if self.debug:
                    logger.info(f"Removed top_k from additionalModelRequestFields for {request.model}")

        return args

    # Anthropic requires 1024 <= budget_tokens < max_tokens.
    _MIN_BUDGET_TOKENS = 1024

    def _calc_budget_tokens(self, max_tokens: int, reasoning_effort: str) -> int:
        """Calculate budget tokens for reasoning, clamped to Anthropic's bounds.

        Anthropic requires budget_tokens >= 1024 and < max_tokens. Without
        clamping, small max_tokens (e.g. 256) produce sub-1024 budgets that
        Bedrock rejects with a ValidationException.
        """
        if reasoning_effort == "low":
            computed = int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            computed = int(max_tokens * 0.6)
        else:
            computed = max_tokens - 1
        # Clamp into [1024, max_tokens - 1].
        return max(self._MIN_BUDGET_TOKENS, min(computed, max_tokens - 1))

    def _build_usage(self, usage_raw: Dict, input_tokens: int, output_tokens: int, total_tokens: int) -> Usage:
        """Build a Usage object, preserving Bedrock cache token counts when present."""
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        if usage_raw.get("cacheWriteInputTokens") is not None:
            usage.cache_creation_input_tokens = usage_raw["cacheWriteInputTokens"]
        if usage_raw.get("cacheReadInputTokens") is not None:
            usage.cache_read_input_tokens = usage_raw["cacheReadInputTokens"]
        return usage

    # OpenAI's finish_reason enum is closed; unmapped Bedrock reasons must not
    # leak through verbatim (clients that validate the enum would reject them).
    _FINISH_REASON_MAP = {
        "tool_use": "tool_calls",
        "finished": "stop",
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "complete": "stop",
        "content_filtered": "content_filter",
        "content_filter": "content_filter",
        "guardrail_intervened": "content_filter",
    }

    def _convert_finish_reason(self, finish_reason: str) -> Optional[str]:
        """Convert Bedrock finish reason to OpenAI format."""
        if not finish_reason:
            return None

        key = finish_reason.lower()
        if key in self._FINISH_REASON_MAP:
            return self._FINISH_REASON_MAP[key]
        # Unknown reason: default to a valid enum value rather than leaking it.
        logger.warning(f"[BEDROCK] Unknown finish reason '{finish_reason}', defaulting to 'stop'")
        return "stop"

    async def _invoke_bedrock(self, request: ChatCompletionRequest, stream: bool = False):
        """Invoke Bedrock model.
        
        Returns:
            tuple: (response, tool_name_mapping) where tool_name_mapping is a
                   request-scoped dict mapping truncated tool names to originals.
        """
        # Request-scoped tool name mapping to avoid race conditions between concurrent requests
        tool_name_mapping: Dict[str, str] = {}

        # Pre-populate the mapping from every tool the client declared in THIS
        # request. Truncation is deterministic, so re-truncating each declared
        # name reproduces the truncated->original pair the model will echo —
        # even for a tool first defined in an earlier turn — as long as the
        # client re-sends the tool spec (the standard pattern). Without this,
        # restoration only worked when toolConfig happened to be built.
        if request.tools:
            for t in request.tools:
                original = getattr(t.function, "name", None) if hasattr(t, "function") else None
                if original:
                    self._truncate_tool_name(original, tool_name_mapping=tool_name_mapping)

        # Extract clean model ID from prefixed model name
        model_id = self.get_model_id(request.model)
        
        if self.debug:
            logger.info(f"Bedrock request for model: {request.model} (clean model_id: {model_id})")
        
        # Create a modified request with clean model ID for parsing.
        # Copy the original request wholesale (preserving ALL fields — declared
        # ones like reasoning_effort/max_completion_tokens AND any extra fields)
        # and only override the model. Rebuilding field-by-field silently drops
        # declared fields that aren't in the explicit list (e.g. reasoning_effort,
        # max_completion_tokens), so model_copy is both correct and future-proof.
        clean_request = request.model_copy(update={"model": model_id})
        
        # Parse request in threadpool since _parse_bedrock_request may download images
        # which would block the event loop
        args = await run_in_threadpool(self._parse_bedrock_request, clean_request, tool_name_mapping)
        
        # Sanitize args to ensure no Pydantic models remain
        # This is critical for boto3 which uses json.dumps internally
        args = _sanitize_for_json(args)
        
        if self.debug:
            try:
                # Create a safe version for logging (convert non-serializable objects)
                safe_args = json.dumps(args, default=str, indent=2)
                logger.info(f"Bedrock args: {safe_args}")
            except Exception as e:
                logger.warning(f"Failed to serialize args for logging: {e}")
        
        # Bound Converse concurrency. For converse_stream this covers opening
        # the stream (boto3 returns the EventStream once headers arrive, not
        # when the body is drained), so the permit is released before the
        # client reads the body — a slow consumer never holds a Converse permit.
        converse_semaphore = _get_converse_semaphore()
        try:
            async with converse_semaphore:
                if stream:
                    response = await run_in_threadpool(
                        self.bedrock_runtime.converse_stream, **args
                    )
                else:
                    response = await run_in_threadpool(
                        self.bedrock_runtime.converse, **args
                    )
            return response, tool_name_mapping
        except ClientError as e:
            # Map Bedrock errors to correct HTTP status codes (429/400/403/...)
            # instead of letting them propagate to a generic 500. SDK-based
            # providers get this for free from typed exceptions.
            err = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            code = err.get("Code", "ClientError")
            msg = err.get("Message", str(e))
            logger.error(f"Bedrock invocation failed: {code}: {msg}")
            mapped = _map_bedrock_error(code, msg)
            from fastapi import HTTPException
            raise HTTPException(status_code=mapped["status"], detail=mapped["body"])
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request."""
        response, tool_name_mapping = await self._invoke_bedrock(request, stream=False)
        
        output_message = response["output"]["message"]
        usage_raw = response["usage"]
        input_tokens = usage_raw["inputTokens"]
        output_tokens = usage_raw["outputTokens"]
        # Prefer Bedrock's authoritative total (differs from input+output when
        # cache tokens are present) so stream and non-stream report the same total.
        total_tokens = usage_raw.get("totalTokens", input_tokens + output_tokens)
        finish_reason = response["stopReason"]

        # A10: if a guardrail intervened, surface it as content_filter rather
        # than reporting a normal stop with no signal.
        trace = response.get("trace")
        if trace and isinstance(trace, dict) and trace.get("guardrail"):
            logger.warning(f"[BEDROCK] Guardrail trace present on response: {trace.get('guardrail')}")
            if finish_reason not in ("content_filtered", "guardrail_intervened"):
                finish_reason = "guardrail_intervened"

        if self.debug:
            logger.info(f"[BEDROCK] Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
        
        # Parse response message
        message = ChatMessage(role="assistant", content="")
        
        if finish_reason == "tool_use":
            tool_calls = []
            for part in output_message["content"]:
                if "toolUse" in part:
                    tool = part["toolUse"]
                    # Ensure we have all required fields
                    tool_id = tool.get("toolUseId", "")
                    tool_name = tool.get("name", "")
                    tool_input = tool.get("input")
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    
                    # Restore original tool name if it was truncated
                    original_tool_name = self._restore_tool_name(tool_name, tool_name_mapping)
                    
                    tool_calls.append(ToolCall(
                        id=tool_id,
                        type="function",
                        function=ResponseFunction(
                            name=original_tool_name,
                            arguments=json.dumps(tool_input)
                        )
                    ))
            message.tool_calls = tool_calls
            # Keep content as empty string for tool calls (required field)
        else:
            content = ""
            for c in output_message["content"]:
                if "reasoningContent" in c:
                    message.reasoning_content = c["reasoningContent"]["reasoningText"].get("text", "")
                elif "text" in c:
                    content = c["text"]

            # Emit reasoning on the dedicated reasoning_content field; keep
            # content as the visible text only (no <think> injection).
            message.content = content
        
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=self._convert_finish_reason(finish_reason)
            )],
            usage=self._build_usage(usage_raw, input_tokens, output_tokens, total_tokens)
        )

        if self.debug:
            logger.info(f"[BEDROCK] Created response with usage: {chat_response.usage}")
            logger.info(f"[BEDROCK] Response dict: {chat_response.model_dump()}")
        
        return chat_response

    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle text completion request."""
        # Convert to chat completion format
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=request.prompt)],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            stream=False
        )
        
        chat_response = await self.chat_completion(chat_request)
        
        # Convert to completion format
        return CompletionResponse(
            id=chat_response.id.replace("chatcmpl", "cmpl"),
            object="text_completion",
            created=chat_response.created,
            model=chat_response.model,
            choices=[CompletionChoice(
                index=0,
                text=chat_response.choices[0].message.content or "",
                finish_reason=chat_response.choices[0].finish_reason
            )],
            usage=chat_response.usage
        )

    async def _async_iterate(self, stream):
        """Convert sync iterator to async iterator by running iteration in threadpool.
        
        This implementation ensures both iter() and next() calls happen in the threadpool
        to avoid blocking the event loop when dealing with blocking generators.
        """
        async def _close_stream(state):
            stream_obj = state.get('stream')
            close_stream = getattr(stream_obj, 'close', None)
            if not callable(close_stream):
                return
            try:
                await run_in_threadpool(close_stream)
            except Exception as exc:
                logger.warning(
                    "[BEDROCK STREAM CONVERSE] Failed to close stream: %s",
                    exc,
                    exc_info=True,
                )
        
        def _get_next(state):
            """Get next item from iterator in threadpool.
            
            Uses a state dict to lazily initialize the iterator inside the threadpool,
            ensuring iter(stream) doesn't block the event loop.
            """
            if state.get('iterator') is None:
                state['iterator'] = iter(state['stream'])
            try:
                return next(state['iterator']), False
            except StopIteration:
                return None, True
        
        # Store stream in state dict so iter() happens in threadpool
        state = {'stream': stream, 'iterator': None}
        try:
            while True:
                chunk, done = await run_in_threadpool(_get_next, state)
                if done:
                    break
                yield chunk
                # Yield control to allow other tasks to run
                await asyncio.sleep(0)
        finally:
            await _close_stream(state)
            state.clear()

    async def _async_iterate_catching(self, stream):
        """Like _async_iterate but converts mid-stream exceptions into a sentinel chunk.

        When the Bedrock event stream raises an error during iteration (e.g.
        EventStreamError "Response ended prematurely"), the error is surfaced as a
        synthetic ``{"_bedrock_stream_error": <exc>}`` chunk instead of propagating
        as an exception.  This lets the caller emit proper SSE terminal events before
        reporting the error to the client.
        """
        async def _close_stream(state):
            stream_obj = state.get('stream')
            close_stream = getattr(stream_obj, 'close', None)
            if not callable(close_stream):
                return
            try:
                await run_in_threadpool(close_stream)
            except Exception as exc:
                logger.warning(
                    "[BEDROCK STREAM CONVERSE] Failed to close error-handling stream: %s",
                    exc,
                    exc_info=True,
                )

        def _get_next(state):
            if state.get('iterator') is None:
                state['iterator'] = iter(state['stream'])
            try:
                chunk = next(state['iterator'])
                state['chunk_count'] = state.get('chunk_count', 0) + 1
                return chunk, False, None
            except StopIteration:
                return None, True, None
            except Exception as exc:
                logger.warning(
                    "[BEDROCK STREAM CONVERSE] Mid-stream error after %d chunks: %s",
                    state.get('chunk_count', 0),
                    exc,
                    exc_info=True,
                )
                return None, True, exc  # surface as sentinel

        state = {'stream': stream, 'iterator': None, 'chunk_count': 0}
        try:
            while True:
                chunk, done, exc = await run_in_threadpool(_get_next, state)
                if exc is not None:
                    yield {"_bedrock_stream_error": exc}
                    return
                if done:
                    break
                yield chunk
                await asyncio.sleep(0)
        finally:
            await _close_stream(state)
            state.clear()

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request."""
        try:
            response, tool_name_mapping = await self._invoke_bedrock(request, stream=True)
            message_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            stream = response.get("stream")

            # Per-stream parser state (local to this stream to avoid cross-request races):
            #   tool_call_index   — last OpenAI tool-call ordinal assigned
            #   block_index_map   — Bedrock contentBlockIndex -> OpenAI tool-call ordinal
            stream_state = {"tool_call_index": -1, "block_index_map": {}}

            # Default include_usage to True if not specified
            include_usage = True
            if request.stream_options is not None:
                include_usage = request.stream_options.include_usage if request.stream_options.include_usage is not None else True

            async for chunk in self._async_iterate(stream):
                # Thread parser state through each chunk
                stream_response, stream_state = self._parse_stream_chunk(chunk, message_id, request.model, stream_state, tool_name_mapping)
                
                if not stream_response:
                    continue
                
                if self.debug:
                    logger.info(f"Stream response: {stream_response}")
                
                # Emit if has choices or usage (when include_usage is true)
                if stream_response.get("choices") or (
                    include_usage and 
                    stream_response.get("usage")
                ):
                    yield self.format_sse_data(stream_response)
            
            # Send [DONE] message
            yield self.format_sse_done()
            
        except ClientError as e:
            # Preserve the real Bedrock error type/status in the SSE error chunk
            # rather than collapsing everything to a generic server_error.
            err = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            code = err.get("Code", "ClientError")
            msg = err.get("Message", str(e))
            logger.error(f"Stream error: {code}: {msg}")
            mapped = _map_bedrock_error(code, msg)
            error_body = mapped["body"]["error"]
            yield self.format_sse_data({
                "error": {
                    "message": error_body["message"],
                    "type": error_body["type"],
                    "code": mapped["status"],
                }
            })
        except Exception as e:
            # HTTPException from _invoke_bedrock already carries a mapped status/type.
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                detail = e.detail if isinstance(e.detail, dict) else {}
                err_obj = detail.get("error", {}) if isinstance(detail, dict) else {}
                logger.error(f"Stream error: {e.status_code}: {err_obj.get('message', str(e))}")
                yield self.format_sse_data({
                    "error": {
                        "message": err_obj.get("message", str(e.detail)),
                        "type": err_obj.get("type", "api_error"),
                        "code": e.status_code,
                    }
                })
            else:
                logger.error(f"Stream error: {e}")
                yield self.format_sse_data({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                    }
                })

    def _parse_stream_chunk(self, chunk: Dict, message_id: str, model_id: str, stream_state: Dict, tool_name_mapping: Optional[Dict[str, str]] = None) -> tuple:
        """Parse Bedrock stream chunk into OpenAI format.

        Args:
            stream_state: per-stream mutable state carrying:
                tool_call_index (int) — last OpenAI tool-call ordinal assigned
                block_index_map (dict) — Bedrock contentBlockIndex -> ordinal

        Returns:
            tuple: (response_dict, stream_state)
        """
        if self.debug:
            logger.info(f"Bedrock chunk: {chunk}")

        delta = {}
        finish_reason = None

        if "messageStart" in chunk:
            # Spec: the first chunk carries only {"role": "assistant"} — no
            # content key (emitting content:"" makes strict clients render an
            # empty assistant turn).
            delta = {"role": chunk["messageStart"]["role"]}

        elif "contentBlockStart" in chunk:
            start = chunk["contentBlockStart"]["start"]
            if "toolUse" in start:
                # Assign a fresh OpenAI tool-call ordinal by emission order,
                # not by arithmetic on Bedrock's block index (which yields -1
                # for tool-call-only responses). Map this Bedrock block index
                # to the ordinal so the matching deltas resolve to it.
                stream_state["tool_call_index"] += 1
                index = stream_state["tool_call_index"]
                block_idx = chunk["contentBlockStart"]["contentBlockIndex"]
                stream_state["block_index_map"][block_idx] = index
                # Restore original tool name if it was truncated
                tool_name = self._restore_tool_name(start["toolUse"]["name"], tool_name_mapping)
                delta = {
                    "tool_calls": [ToolCall(
                        index=index,
                        type="function",
                        id=start["toolUse"]["toolUseId"],
                        function=ResponseFunction(
                            name=tool_name,
                            arguments=""
                        )
                    )]
                }

        elif "contentBlockDelta" in chunk:
            block_delta = chunk["contentBlockDelta"]["delta"]
            if "text" in block_delta:
                delta = {"content": block_delta["text"]}
            elif "reasoningContent" in block_delta:
                # Emit reasoning on the dedicated reasoning_content field rather
                # than injecting literal <think>...</think> into content (which
                # is lossy and non-standard on the OpenAI surface).
                if "text" in block_delta["reasoningContent"]:
                    delta = {"reasoning_content": block_delta["reasoningContent"]["text"]}
                else:
                    # signature / redactedContent carry no client-visible text
                    return None, stream_state
            elif "toolUse" in block_delta:
                block_idx = chunk["contentBlockDelta"]["contentBlockIndex"]
                # Resolve to the ordinal assigned at contentBlockStart; fall back
                # to the running counter if no start was seen for this block.
                index = stream_state["block_index_map"].get(
                    block_idx, stream_state["tool_call_index"]
                )
                # Bedrock sends tool input as a JSON string fragment (already serialized),
                # not a dict — pass through directly; only json.dumps if unexpectedly a dict.
                tool_input = block_delta["toolUse"].get("input", "")
                arguments_str = json.dumps(tool_input) if isinstance(tool_input, dict) else tool_input

                delta = {
                    "tool_calls": [ToolCall(
                        index=index,
                        id="",  # Empty ID for delta updates
                        function=ResponseFunction(
                            name="",  # Empty name for delta updates
                            arguments=arguments_str
                        )
                    )]
                }

        elif "messageStop" in chunk:
            finish_reason = chunk["messageStop"]["stopReason"]
            delta = {}

        elif "metadata" in chunk:
            metadata = chunk["metadata"]
            if "usage" in metadata:
                u = metadata["usage"]
                usage = Usage(
                    prompt_tokens=u["inputTokens"],
                    completion_tokens=u["outputTokens"],
                    total_tokens=u.get("totalTokens", u["inputTokens"] + u["outputTokens"])
                )
                # Preserve cache token counts when present (matches the Anthropic
                # path; dropping them under-reports cached usage for accounting).
                if u.get("cacheWriteInputTokens") is not None:
                    usage.cache_creation_input_tokens = u["cacheWriteInputTokens"]
                if u.get("cacheReadInputTokens") is not None:
                    usage.cache_read_input_tokens = u["cacheReadInputTokens"]
                # Return usage chunk
                return ({
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [],
                    "usage": usage.model_dump()
                }, stream_state)

        if delta or finish_reason:
            return ({
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": self._convert_finish_reason(finish_reason)
                }]
            }, stream_state)

        return None, stream_state

    async def completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming text completion request."""
        # Convert to chat completion format
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=request.prompt)],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            stream=True
        )
        
        async for chunk in self.chat_completion_stream(chat_request):
            # Convert chat chunk to completion chunk
            if chunk == self.format_sse_done():
                yield chunk
            else:
                # Parse the SSE data
                if chunk.startswith("data: "):
                    data_str = chunk[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                # Convert to completion format
                                completion_chunk = {
                                    "id": data["id"].replace("chatcmpl", "cmpl"),
                                    "object": "text_completion",
                                    "created": data["created"],
                                    "model": data["model"],
                                    "choices": [{
                                        "index": 0,
                                        "text": choice.get("delta", {}).get("content", ""),
                                        "finish_reason": choice.get("finish_reason")
                                    }]
                                }
                                yield self.format_sse_data(completion_chunk)
                        except json.JSONDecodeError:
                            pass

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Handle embeddings request."""
        # Extract clean model ID from prefixed model name
        model_id = self.get_model_id(request.model)
        
        if model_id not in self.SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported embedding model: {model_id}")
        
        model_name = self.SUPPORTED_EMBEDDING_MODELS[model_id]
        
        # Prepare input
        texts = []
        if isinstance(request.input, str):
            texts = [request.input]
        elif isinstance(request.input, list):
            texts = request.input
        
        # Handle Cohere models
        if "Cohere" in model_name:
            # Honor a client-supplied input_type (search_query vs search_document
            # materially affects retrieval quality); default to search_document.
            extra = getattr(request, "model_extra", None) or {}
            input_type = extra.get("input_type") or "search_document"
            if input_type not in ("search_document", "search_query", "classification", "clustering"):
                input_type = "search_document"
            args = {
                "texts": texts,
                "input_type": input_type,
                "truncate": "END"
            }
            response = await run_in_threadpool(
                self.bedrock_runtime.invoke_model,
                body=json.dumps(args),
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            embeddings = response_body["embeddings"]
            # Cohere returns billed/token info under meta.billed_units when present;
            # fall back to a whitespace-token estimate rather than hardcoding 0.
            meta = response_body.get("meta", {}) or {}
            billed = meta.get("billed_units", {}) or {}
            input_tokens = billed.get("input_tokens")
            if input_tokens is None:
                input_tokens = sum(len(t.split()) for t in texts)

        # Handle Titan models
        elif "Titan" in model_name:
            # Titan's invoke_model accepts a single inputText; loop over the batch
            # so OpenAI-style array inputs work instead of being rejected.
            embeddings = []
            input_tokens = 0
            for text in texts:
                args = {"inputText": text}
                response = await run_in_threadpool(
                    self.bedrock_runtime.invoke_model,
                    body=json.dumps(args),
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                response_body = json.loads(response.get("body").read())
                embeddings.append(response_body["embedding"])
                input_tokens += response_body.get("inputTextTokenCount", 0)

        else:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        # Format response
        data = []
        for i, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded = base64.b64encode(arr_bytes).decode()
                data.append(EmbeddingData(index=i, embedding=encoded, object="embedding"))
            else:
                data.append(EmbeddingData(index=i, embedding=embedding, object="embedding"))
        
        return EmbeddingResponse(
            object="list",
            data=data,
            model=model_id,
            usage=EmbeddingUsage(
                prompt_tokens=input_tokens,
                total_tokens=input_tokens
            )
        )

    # ==================== LIFECYCLE ====================

    async def close(self):
        """Cleanup resources, including the Anthropic Bedrock client."""
        if self._anthropic_client:
            await self._anthropic_client.close()
            self._anthropic_client = None

    # ==================== ANTHROPIC MESSAGES API ====================

    def get_supported_apis(self) -> List[str]:
        """Bedrock supports both OpenAI and Anthropic API formats."""
        return ["openai", "anthropic"]

    def get_supported_endpoints(self) -> List[str]:
        """Bedrock supports chat, completions, and embeddings. No images/audio/responses."""
        endpoints = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]
        endpoints.append("/v1/messages")
        return endpoints

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        if not self._is_chat_capable_model(model_name):
            return "unsupported"
        model_id = self._resolve_bedrock_anthropic_model_id(model_name)
        return "native" if self._is_claude_model(model_id) else "adapter"

    def get_anthropic_request_metadata(self, request, anthropic_beta: Optional[str] = None) -> AnthropicRequestMetadata:
        mode = self.get_anthropic_mode_for_model(request.model)
        dropped_fields: List[str] = []

        if mode == "adapter" and anthropic_beta:
            dropped_fields.append("anthropic-beta")

        if isinstance(getattr(request, "system", None), list):
            for block in request.system:
                block_data = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
                cache_control = block_data.get("cache_control") if isinstance(block_data, dict) else None
                if isinstance(cache_control, dict) and cache_control.get("scope") is not None:
                    dropped_fields.append("system.cache_control.scope")

        for message in getattr(request, "messages", []) or []:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                block_data = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
                cache_control = block_data.get("cache_control") if isinstance(block_data, dict) else None
                if isinstance(cache_control, dict) and cache_control.get("scope") is not None:
                    dropped_fields.append("messages.cache_control.scope")

        for tool in getattr(request, "tools", []) or []:
            tool_data = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
            if not isinstance(tool_data, dict):
                continue
            cache_control = tool_data.get("cache_control")
            if isinstance(cache_control, dict) and cache_control.get("scope") is not None:
                dropped_fields.append("tools.cache_control.scope")
            input_schema = tool_data.get("input_schema")
            if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
                dropped_fields.append("tools.input_schema.type")

        return AnthropicRequestMetadata(
            mode=mode,
            transport="messages" if mode == "native" else "chat",
            dropped_fields=sorted(set(dropped_fields)),
        )

    def _get_anthropic_model_id(self, model_name: str) -> str:
        """Extract a clean model ID for the Anthropic SDK from a provider-prefixed name.
        
        The Anthropic Bedrock client understands cross-region inference profile IDs
        (e.g., 'us.anthropic.claude-sonnet-4-5') directly, so we only strip our
        internal provider prefix.
        """
        # Remove our internal provider prefix (e.g., "bedrock:default/us.anthropic.claude-sonnet-4-5")
        if '/' in model_name:
            model_name = model_name.split('/', 1)[1]
        return model_name

    def _resolve_bedrock_anthropic_model_id(self, model_name: str) -> str:
        normalized = self._get_anthropic_model_id(model_name)
        if normalized in self.bedrock_model_list:
            return normalized
        mapped = self.ANTHROPIC_TO_BEDROCK_MODEL_MAP.get(normalized)
        if mapped:
            return mapped
        # Fallback: if the model looks like a short Anthropic name (e.g.
        # "claude-sonnet-4-20250514") that isn't in the static map, try
        # prefixing with "anthropic." so it resolves on Bedrock.
        if normalized.startswith("claude-") and "anthropic." not in normalized:
            prefixed = f"anthropic.{normalized}"
            if prefixed in self.bedrock_model_list:
                return prefixed
        return normalized

    def _is_claude_model(self, model_id: str) -> bool:
        """True if model_id refers to an Anthropic/Claude model on Bedrock."""
        lower = model_id.lower()
        return "anthropic" in lower or "claude" in lower

    def _is_nova2_model(self, model_id: str) -> bool:
        """True if model_id refers to an Amazon Nova 2 model."""
        lower = model_id.lower()
        return "amazon.nova" in lower and "-2" in lower

    def _is_kimi_model(self, model_id: str) -> bool:
        """True if model_id refers to a Moonshot Kimi model."""
        lower = model_id.lower()
        return "kimi" in lower or "moonshotai" in lower

    @staticmethod
    def _is_claude_at_least(model_id: str, min_major: int, min_minor: int) -> bool:
        return is_claude_at_least(model_id, min_major, min_minor)

    @staticmethod
    def _supports_output_config_effort(model_id: str) -> bool:
        return BedrockProvider._is_claude_at_least(model_id, 4, 6)

    def _parse_anthropic_beta_values(self, anthropic_beta: Optional[str]) -> List[str]:
        if not anthropic_beta:
            return []
        return [value.strip() for value in anthropic_beta.split(',') if value.strip()]

    def _map_anthropic_beta_values(self, beta_values: List[str]) -> List[str]:
        """Map Anthropic beta flags to Bedrock-supported values.

        Only flags in ANTHROPIC_BETA_MAP or ANTHROPIC_BETA_PASSTHROUGH are
        forwarded. Everything else is silently dropped so Bedrock never
        receives an unrecognised flag ("invalid beta flag" error).
        """
        mapped: List[str] = []
        seen = set()
        for value in beta_values:
            if value in self.ANTHROPIC_BETA_MAP:
                expanded = self.ANTHROPIC_BETA_MAP[value]
            elif value in self.ANTHROPIC_BETA_PASSTHROUGH:
                expanded = [value]
            else:
                logger.info(f"[BEDROCK] Filtering unsupported beta header: {value}")
                continue

            for item in expanded:
                if item not in seen:
                    seen.add(item)
                    mapped.append(item)
        return mapped

    # ==================== InvokeModel (native Anthropic) path ====================

    def _convert_to_anthropic_native_request(
        self, request, anthropic_beta: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a native Anthropic Messages API request body for InvokeModel.

        The Bedrock InvokeModel API accepts native Anthropic JSON and returns
        native Anthropic JSON — no Converse-format conversion needed.
        """
        self._validate_anthropic_thinking_blocks(request)

        native: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "messages": [],
        }

        # --- messages ----------------------------------------------------------
        for msg in request.messages:
            message_dict: Dict[str, Any] = {"role": msg.role}

            if isinstance(msg.content, str):
                message_dict["content"] = msg.content
            else:
                content_list: list = []
                for block in msg.content:
                    if hasattr(block, "model_dump"):
                        bd = block.model_dump(exclude_none=True)
                    elif isinstance(block, dict):
                        bd = dict(block)
                    else:
                        continue

                    bt = bd.get("type", "")

                    # Strip `caller` from tool_use — Bedrock doesn't accept it (PTC ext)
                    if bt == "tool_use" and "caller" in bd:
                        bd = {k: v for k, v in bd.items() if k != "caller"}

                    # Skip server/web-search content in assistant messages
                    if msg.role == "assistant" and bt in (
                        "server_tool_use",
                        "web_search_tool_result",
                        "bash_code_execution_tool_result",
                    ):
                        continue

                    # Convert web_search_tool_result in user messages → tool_result
                    if bt == "web_search_tool_result":
                        ws_id = bd.get("tool_use_id", "")
                        bedrock_id = (
                            ws_id.replace("srvtoolu_", "toolu_", 1)
                            if ws_id.startswith("srvtoolu_")
                            else ws_id
                        )
                        ws_content = bd.get("content", [])
                        if isinstance(ws_content, list):
                            parts = []
                            for sr in ws_content:
                                if isinstance(sr, dict) and sr.get("type") == "web_search_result":
                                    title = sr.get("title", "")
                                    url = sr.get("url", "")
                                    enc = sr.get("encrypted_content", "")
                                    parts.append(f"Title: {title}\nURL: {url}\nContent: {enc}")
                            result_text = "\n\n---\n\n".join(parts) if parts else "No results"
                        elif isinstance(ws_content, dict):
                            result_text = f"Error: {ws_content.get('error_code', 'unknown')}"
                        else:
                            result_text = str(ws_content)
                        bd = {
                            "type": "tool_result",
                            "tool_use_id": bedrock_id,
                            "content": result_text,
                        }

                    # Strip citations from text blocks — Bedrock doesn't support them
                    if bt == "text" and "citations" in bd:
                        bd = {k: v for k, v in bd.items() if k != "citations"}

                    content_list.append(bd)
                message_dict["content"] = content_list

            native["messages"].append(message_dict)

        # Lift role=="system" entries out of messages[] into the top-level
        # system field. Bedrock InvokeModel (native Anthropic) rejects any
        # role other than user/assistant, mirroring the direct Anthropic API.
        native["messages"], _extracted_system = _extract_system_messages_from_messages(
            native["messages"]
        )

        # --- system ------------------------------------------------------------
        existing_system: Any = None
        if request.system is not None:
            if isinstance(request.system, str):
                existing_system = request.system
            else:
                system_parts: list = []
                for sys_msg in request.system:
                    if hasattr(sys_msg, "model_dump"):
                        system_parts.append(sys_msg.model_dump(exclude_none=True))
                    elif isinstance(sys_msg, dict):
                        system_parts.append(sys_msg)
                    elif hasattr(sys_msg, "text"):
                        sd: Dict[str, Any] = {"type": "text", "text": sys_msg.text}
                        if hasattr(sys_msg, "cache_control") and sys_msg.cache_control:
                            cc = sys_msg.cache_control
                            sd["cache_control"] = (
                                cc.model_dump(exclude_none=True)
                                if hasattr(cc, "model_dump")
                                else cc
                            )
                        system_parts.append(sd)
                existing_system = system_parts

        merged_system = _merge_system_fields(existing_system, _extracted_system)
        if merged_system is not None:
            native["system"] = merged_system

        # --- scalar params -----------------------------------------------------
        if request.temperature is not None:
            native["temperature"] = request.temperature
        # Claude >= 4.7 deprecated top_p — passing it triggers a
        # "top_p is deprecated for this model" error. Drop it for those models.
        if request.top_p is not None and not self._is_claude_at_least(request.model, 4, 7):
            native["top_p"] = request.top_p
        if request.top_k is not None:
            native["top_k"] = request.top_k
        if request.stop_sequences:
            native["stop_sequences"] = request.stop_sequences

        # --- tools -------------------------------------------------------------
        if request.tools:
            tools_list: list = []
            # Tool types that map between Anthropic SDK versioned names and Bedrock names
            _tool_type_mapping = {
                "tool_search_tool_regex_20251119": "tool_search_tool_regex",
                "tool_search_tool_20251119": "tool_search_tool",
            }
            _special_tool_types = {"tool_search_tool_regex", "tool_search_tool"}
            _skip_types = {"code_execution_20250825", "web_search_20250305", "web_search"}

            for index, tool in enumerate(request.tools):
                td = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else (tool if isinstance(tool, dict) else {})
                tool_type = td.get("type")

                # Skip unsupported tools — but never silently. These server-side
                # tools aren't honored on this Bedrock path; log so the client
                # can tell the tool was dropped rather than ignored at runtime.
                if tool_type in _skip_types:
                    logger.warning(
                        f"[BEDROCK NATIVE] Tool type '{tool_type}' is not supported on this path; dropping it"
                    )
                    continue

                mapped_type = _tool_type_mapping.get(tool_type, tool_type)
                if mapped_type in _special_tool_types:
                    tool_copy = dict(td)
                    if mapped_type != tool_type:
                        tool_copy["type"] = mapped_type
                    tools_list.append(tool_copy)
                    continue

                # Special built-in tool types (bash, computer, text_editor) pass through natively
                if tool_type and tool_type in {
                    "computer_20250124", "text_editor_20250124", "bash_20250124",
                }:
                    if not td.get("name"):
                        td["name"] = f"tool_{index + 1}"
                    tools_list.append(td)
                    continue

                # Regular tool
                input_schema = td.get("input_schema") or {}
                if not isinstance(input_schema, dict):
                    raise ValueError(
                        f"Tool '{td.get('name') or f'tool_{index + 1}'}' input_schema must be an object"
                    )
                # Anthropic has two different `type` fields here:
                # `input_schema.type` is JSON Schema and must be an object type,
                # while the top-level tool `type` is the Anthropic tool kind.
                if not input_schema.get("type") or input_schema.get("type") == "custom":
                    input_schema = {**input_schema, "type": "object"}
                tool_dict: Dict[str, Any] = {
                    "name": td.get("name") or f"tool_{index + 1}",
                    "description": td.get("description", ""),
                    "input_schema": input_schema,
                }
                # Preserve tool-level type (e.g., "custom") for the Anthropic API
                if tool_type:
                    tool_dict["type"] = tool_type
                if td.get("input_examples"):
                    tool_dict["input_examples"] = td["input_examples"]
                if td.get("defer_loading") is not None:
                    tool_dict["defer_loading"] = td["defer_loading"]
                if td.get("cache_control"):
                    tool_dict["cache_control"] = td["cache_control"]
                tools_list.append(tool_dict)

            if tools_list:
                native["tools"] = tools_list

        # --- tool_choice -------------------------------------------------------
        if request.tool_choice is not None:
            tc = (
                request.tool_choice.model_dump(exclude_none=True)
                if hasattr(request.tool_choice, "model_dump")
                else request.tool_choice
            )
            native["tool_choice"] = tc

        # --- thinking / extended thinking --------------------------------------
        if request.thinking is not None:
            native["thinking"] = (
                request.thinking.model_dump(exclude_none=True)
                if hasattr(request.thinking, "model_dump")
                else request.thinking
            )

        # --- metadata ----------------------------------------------------------
        if request.metadata is not None:
            native["metadata"] = (
                request.metadata.model_dump(exclude_none=True)
                if hasattr(request.metadata, "model_dump")
                else request.metadata
            )

        # --- forward-compat extra fields (output_config, context_management) ---
        if hasattr(request, "model_extra") and request.model_extra:
            _forward = {"output_config", "context_management"}
            for key in _forward:
                if key in request.model_extra:
                    val = request.model_extra[key]
                    if (
                        key == "output_config"
                        and isinstance(val, dict)
                        and not self._supports_output_config_effort(request.model)
                    ):
                        val = {k: v for k, v in val.items() if k != "effort"}
                        if not val:
                            continue
                    native[key] = val

        # --- auto-inject advanced-tool-use beta when defer_loading detected ----
        TOOL_SEARCH_BETA = "advanced-tool-use-2025-11-20"
        if request.tools:
            has_defer = any(
                (isinstance(t, dict) and t.get("defer_loading") is not None)
                or (hasattr(t, "defer_loading") and getattr(t, "defer_loading", None) is not None)
                for t in request.tools
            )
            if has_defer:
                beta_str = anthropic_beta or ""
                if TOOL_SEARCH_BETA not in beta_str:
                    anthropic_beta = f"{beta_str},{TOOL_SEARCH_BETA}".strip(",")

        # --- beta headers ------------------------------------------------------
        bedrock_beta = self._map_anthropic_beta_values(
            self._parse_anthropic_beta_values(anthropic_beta)
        )
        if bedrock_beta:
            native["anthropic_beta"] = bedrock_beta

        return native

    # --- cache TTL helpers (operate on native request dicts) ---

    def _apply_native_cache_ttl(self, body: dict, api_key_cache_ttl: Optional[str] = None) -> None:
        """Apply cache TTL to all cache_control blocks in a native Anthropic request.

        Priority: api_key_cache_ttl > existing client TTL > (no default for now).
        """
        def _update(block: dict) -> None:
            cc = block.get("cache_control")
            if not cc or not isinstance(cc, dict):
                return
            if api_key_cache_ttl:
                cc["ttl"] = api_key_cache_ttl

        system = body.get("system")
        if isinstance(system, list):
            for part in system:
                if isinstance(part, dict):
                    _update(part)
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _update(block)
        for tool in body.get("tools", []):
            if isinstance(tool, dict):
                _update(tool)

    def _strip_native_cache_scope(self, body: dict) -> None:
        """Remove ``scope`` from every cache_control block — Bedrock doesn't support it."""
        def _strip(block: dict) -> None:
            cc = block.get("cache_control")
            if isinstance(cc, dict) and "scope" in cc:
                del cc["scope"]

        system = body.get("system")
        if isinstance(system, list):
            for part in system:
                if isinstance(part, dict):
                    _strip(part)
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        _strip(block)
        for tool in body.get("tools", []):
            if isinstance(tool, dict):
                _strip(tool)

    # --- AsyncAnthropicBedrock SDK helpers ---

    def _init_bedrock_anthropic_client(self) -> None:
        """Lazily build the AsyncAnthropicBedrock client for the native Claude path.

        Mirrors ``anthropic_compatible._init_anthropic_client``. The SDK owns the
        transport, retries, SSE parsing and typed error mapping that the old
        hand-rolled native worker/queue/socket machinery reimplemented.
        """
        if self._anthropic_client is not None:
            return
        try:
            from anthropic import AsyncAnthropicBedrock  # function-scoped: optional dep
        except ImportError:
            logger.warning(
                "[BEDROCK] anthropic SDK not installed — native Claude path unavailable"
            )
            return

        kwargs: Dict[str, Any] = {
            "aws_region": self.aws_region,
            # Generous ceiling so long extended-thinking responses aren't
            # truncated (old native socket ceiling was ~900s). Shares the same
            # operator knob as the other Anthropic-SDK providers; an explicit
            # timeout also disables the SDK's client-side non-streaming guard.
            "timeout": ANTHROPIC_SDK_TIMEOUT_SECONDS,
            "max_retries": 2,
        }
        if self.aws_access_key and self.aws_secret_key:
            kwargs["aws_access_key"] = self.aws_access_key
            kwargs["aws_secret_key"] = self.aws_secret_key
        # else: omit creds → SDK uses the default boto3 credential chain, matching
        # how self.bedrock_runtime is built without explicit creds.
        try:
            self._anthropic_client = AsyncAnthropicBedrock(**kwargs)
            logger.info(
                "[BEDROCK] AsyncAnthropicBedrock client initialized for region=%s",
                self.aws_region,
            )
        except Exception as e:
            logger.warning("[BEDROCK] Failed to init AsyncAnthropicBedrock client: %s", e)

    def _build_native_sdk_kwargs(
        self, request, model_id: str, anthropic_beta: Optional[str]
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Build messages.create/stream kwargs from a request via the Bedrock transform.

        Reuses ``_convert_to_anthropic_native_request`` (which encodes every
        Bedrock-Claude quirk) then adapts the resulting body to the SDK call
        contract: drop the body-level ``anthropic_version`` (SDK injects it),
        lift ``anthropic_beta`` to the header, set ``model``, and route the
        non-native ``context_management`` field into ``extra_body``.
        """
        native = self._convert_to_anthropic_native_request(request, anthropic_beta)
        self._apply_native_cache_ttl(native)
        self._strip_native_cache_scope(native)  # Bedrock rejects cache_control.scope

        native.pop("anthropic_version", None)  # SDK injects bedrock-2023-05-31 itself
        betas = native.pop("anthropic_beta", None)  # SDK takes betas as a header
        native["model"] = model_id  # transform doesn't set model

        # context_management is NOT a native messages.create param (the SDK would
        # raise a client-side TypeError). Route it through extra_body so it reaches
        # Bedrock. output_config IS a native param and stays top-level.
        if "context_management" in native:
            native.setdefault("extra_body", {})["context_management"] = native.pop(
                "context_management"
            )

        beta_header = ",".join(betas) if betas else None
        return native, beta_header

    def _translate_bedrock_sdk_error(self, e: Exception):
        """Map an AsyncAnthropicBedrock SDK error to a ProviderHTTPError.

        Reuses the canonical translator so SDK exceptions become correct HTTP
        statuses (429/400/403/404/5xx) with Anthropic-shaped bodies. Non-Anthropic
        exceptions are returned unchanged for the caller to re-raise.
        """
        try:
            import anthropic
        except ImportError:
            return e
        if isinstance(e, anthropic.AnthropicError):
            from app.providers.anthropic_compatible import _translate_anthropic_sdk_error

            return _translate_anthropic_sdk_error(
                e, getattr(self, "full_provider_name", "bedrock")
            )
        return e

    def _cache_control_to_cache_point_block(self, cache_control: Any) -> Optional[Dict[str, Any]]:
        if cache_control is None:
            return None
        if hasattr(cache_control, 'model_dump'):
            cache_control = cache_control.model_dump(exclude_none=True)
        if not isinstance(cache_control, dict):
            return None

        cache_point = {
            "type": "default"
        }

        ttl = cache_control.get("ttl")
        if isinstance(ttl, int) and ttl > 0:
            cache_point["ttl"] = ttl
        return {"cachePoint": cache_point}

    # Special Anthropic tool types that Bedrock Converse API does not support.
    # These are skipped when building toolConfig and optionally forwarded via
    # additionalModelRequestFields.
    _SPECIAL_TOOL_TYPES = {
        "computer_20250124", "text_editor_20250124", "bash_20250124",
        "code_execution_20250825",
        "web_search_20250305", "web_search",
    }

    def _convert_anthropic_tool_schema(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert an Anthropic tool definition to Bedrock toolSpec format.

        Returns None for special tool types that the Converse API cannot
        represent (computer use, code execution, web search, etc.).
        """
        tool_type = tool.get("type")
        if tool_type and tool_type in self._SPECIAL_TOOL_TYPES:
            return None

        input_schema = tool.get("input_schema") or {"type": "object", "properties": {}}
        if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
            input_schema = {**input_schema, "type": "object"}
        bedrock_tool: Dict[str, Any] = {
            "toolSpec": {
                "name": tool.get("name") or "tool_1",
                "description": tool.get("description", ""),
                "inputSchema": {
                    "json": input_schema,
                },
            }
        }
        return bedrock_tool

    def _convert_anthropic_content_to_bedrock(self, content: Any, supports_caching: bool = True) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]

        if not isinstance(content, list):
            return [{"text": str(content)}]

        converted: List[Dict[str, Any]] = []
        for block in content:
            if hasattr(block, 'model_dump'):
                block = block.model_dump(exclude_none=True)
            if not isinstance(block, dict):
                converted.append({"text": str(block)})
                continue

            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if text:
                    converted.append({"text": text})
            elif block_type == "image":
                source = block.get("source") or {}
                media_type = source.get("media_type", "image/jpeg")
                image_format = media_type.replace("image/", "") if isinstance(media_type, str) else "jpeg"
                image_bytes = None

                if source.get("type") == "base64" and source.get("data"):
                    image_bytes = base64.b64decode(source["data"])
                elif source.get("type") == "url" and source.get("url"):
                    image_bytes, parsed_media_type = self._parse_image_sync(source["url"])
                    image_format = parsed_media_type.replace("image/", "")

                if image_bytes is not None:
                    converted.append({
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes}
                        }
                    })
            elif block_type == "tool_use":
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    tool_input = {}
                converted.append({
                    "toolUse": {
                        "toolUseId": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": tool_input,
                    }
                })
            elif block_type == "tool_result":
                tool_result_content = block.get("content")
                converted_tool_result_content: List[Dict[str, Any]] = []

                if isinstance(tool_result_content, str):
                    converted_tool_result_content.append({"text": tool_result_content})
                elif isinstance(tool_result_content, list):
                    for result_block in tool_result_content:
                        if hasattr(result_block, 'model_dump'):
                            result_block = result_block.model_dump(exclude_none=True)
                        if isinstance(result_block, dict):
                            if result_block.get("type") == "text":
                                converted_tool_result_content.append({"text": result_block.get("text", "")})
                            else:
                                converted_tool_result_content.append({"text": json.dumps(result_block, ensure_ascii=False)})
                        else:
                            converted_tool_result_content.append({"text": str(result_block)})

                if not converted_tool_result_content:
                    converted_tool_result_content.append({"text": ""})

                tool_result = {
                    "toolUseId": block.get("tool_use_id", ""),
                    "content": converted_tool_result_content,
                }
                if block.get("is_error") is True:
                    tool_result["status"] = "error"

                converted.append({"toolResult": tool_result})
            elif block_type == "thinking":
                thinking_block = {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": block.get("thinking", "")
                        }
                    }
                }
                signature = block.get("signature")
                if signature:
                    thinking_block["reasoningContent"]["reasoningText"]["signature"] = signature
                converted.append(thinking_block)
            elif block_type == "redacted_thinking":
                converted.append({
                    "reasoningContent": {
                        "redactedContent": block.get("data", "")
                    }
                })
            else:
                converted.append({"text": json.dumps(block, ensure_ascii=False)})

            if supports_caching:
                cache_point_block = self._cache_control_to_cache_point_block(block.get("cache_control"))
                if cache_point_block:
                    converted.append(cache_point_block)

        return converted

    def _validate_anthropic_thinking_blocks(self, request: Any) -> None:
        """Strip thinking/redacted_thinking blocks that lack required signatures.

        Both the upstream Anthropic API and Bedrock require a `signature` field
        on every `thinking` block in multi-turn assistant messages.  If the
        client did not preserve the signature (e.g. because an earlier streaming
        response was malformed), we strip those blocks so the request can still
        succeed — the model simply loses that prior thinking context.
        """
        for message in getattr(request, "messages", []) or []:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue

            indices_to_remove: list[int] = []
            for i, block in enumerate(content):
                block_data = block.model_dump(exclude_none=True) if hasattr(block, 'model_dump') else block
                if not isinstance(block_data, dict):
                    continue
                block_type = block_data.get("type")
                if block_type == "thinking":
                    sig = block_data.get("signature")
                    if not isinstance(sig, str) or not sig.strip():
                        indices_to_remove.append(i)
                elif block_type == "redacted_thinking":
                    if not block_data.get("data"):
                        indices_to_remove.append(i)

            for idx in reversed(indices_to_remove):
                content.pop(idx)

    def _build_bedrock_anthropic_args(self, request, anthropic_beta: Optional[str] = None) -> Dict[str, Any]:
        self._validate_anthropic_thinking_blocks(request)
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)
        messages = []
        system_blocks: List[Dict[str, Any]] = []

        if request.system is not None:
            if isinstance(request.system, str):
                system_blocks.append({"text": request.system})
            else:
                for block in request.system:
                    block_data = block.model_dump(exclude_none=True) if hasattr(block, 'model_dump') else block
                    if isinstance(block_data, dict) and block_data.get("text"):
                        system_blocks.append({"text": block_data["text"]})
                        if self._is_claude_model(model_id):
                            cache_point_block = self._cache_control_to_cache_point_block(block_data.get("cache_control"))
                            if cache_point_block:
                                system_blocks.append(cache_point_block)

        for message in request.messages:
            role = "assistant" if message.role == "assistant" else "user"
            messages.append({
                "role": role,
                "content": self._convert_anthropic_content_to_bedrock(message.content, supports_caching=self._is_claude_model(model_id)),
            })

        inference_config: Dict[str, Any] = {"maxTokens": request.max_tokens}
        if request.temperature is not None:
            inference_config["temperature"] = request.temperature
        if request.top_p is not None:
            inference_config["topP"] = request.top_p
        if request.stop_sequences:
            inference_config["stopSequences"] = request.stop_sequences

        args: Dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system_blocks:
            args["system"] = system_blocks

        additional_fields: Dict[str, Any] = {}
        if request.top_k is not None:
            additional_fields["top_k"] = request.top_k
        if request.thinking is not None:
            if self._is_nova2_model(model_id):
                # Nova 2 uses reasoningConfig with effort level; temperature and
                # maxTokens must be removed from inferenceConfig when reasoning is on.
                thinking_data = request.thinking.model_dump(exclude_none=True)
                budget = thinking_data.get("budget_tokens", 0)
                effort = "high" if budget > 10000 else ("low" if budget < 1000 else "medium")
                additional_fields["reasoningConfig"] = {"type": "enabled", "maxReasoningEffort": effort}
                inference_config.pop("temperature", None)
                inference_config.pop("maxTokens", None)
            elif self._is_kimi_model(model_id):
                # Kimi only supports reasoning_effort="high"
                additional_fields["reasoning_effort"] = "high"
            else:
                # Claude and others: pass thinking config directly
                additional_fields["thinking"] = request.thinking.model_dump(exclude_none=True)

        # Beta headers are only meaningful for Claude/Anthropic models on Bedrock
        if self._is_claude_model(model_id):
            beta_values = self._map_anthropic_beta_values(self._parse_anthropic_beta_values(anthropic_beta))
            if beta_values:
                additional_fields["anthropic_beta"] = beta_values

        if additional_fields:
            args["additionalModelRequestFields"] = additional_fields

        messages_have_tools = self._messages_contain_tool_usage(messages)
        if request.tools or messages_have_tools:
            bedrock_tools = []
            special_tools: List[Dict[str, Any]] = []
            if request.tools:
                for index, tool in enumerate(request.tools):
                    tool_data = tool.model_dump(exclude_none=True) if hasattr(tool, 'model_dump') else tool
                    if isinstance(tool_data, dict) and not tool_data.get("name") and tool_data.get("type") not in self._SPECIAL_TOOL_TYPES:
                        tool_data = {**tool_data, "name": f"tool_{index + 1}"}
                    converted = self._convert_anthropic_tool_schema(tool_data)
                    if converted is not None:
                        bedrock_tools.append(converted)
                        if self._is_claude_model(model_id):
                            cache_point_block = self._cache_control_to_cache_point_block(tool_data.get("cache_control"))
                            if cache_point_block:
                                bedrock_tools.append(cache_point_block)
                    else:
                        # Special tools (text_editor, bash, computer_use, etc.)
                        # must be forwarded in Anthropic-native format via
                        # additionalModelRequestFields so the model knows about
                        # them even though the Converse API toolConfig cannot
                        # represent them.
                        special_tools.append(tool_data)
            if special_tools:
                if "additionalModelRequestFields" not in args:
                    args["additionalModelRequestFields"] = additional_fields
                args["additionalModelRequestFields"]["tools"] = special_tools
                # Converse may not honor special tools forwarded this way; log
                # so behavior differences vs the native path are observable.
                logger.warning(
                    "[BEDROCK] Forwarding %d special tool(s) via additionalModelRequestFields "
                    "(Converse may not honor: %s)",
                    len(special_tools),
                    [t.get("type") for t in special_tools],
                )
            tool_config: Dict[str, Any] = {
                "tools": bedrock_tools,
            }

            if request.tool_choice is not None:
                tool_choice_data = (
                    request.tool_choice.model_dump(exclude_none=True)
                    if hasattr(request.tool_choice, 'model_dump')
                    else request.tool_choice
                )
                if isinstance(tool_choice_data, dict):
                    tool_choice_type = tool_choice_data.get("type")
                    if tool_choice_type == "auto":
                        tool_config["toolChoice"] = {"auto": {}}
                    elif tool_choice_type == "any":
                        tool_config["toolChoice"] = {"any": {}}
                    elif tool_choice_type == "tool" and tool_choice_data.get("name"):
                        tool_config["toolChoice"] = {"tool": {"name": tool_choice_data["name"]}}

            args["toolConfig"] = tool_config

        return _sanitize_for_json(args)

    def _convert_bedrock_stop_reason(self, stop_reason: Optional[str]) -> Optional[str]:
        if not stop_reason:
            return None
        mapping = {
            "end_turn": "end_turn",
            "stop_sequence": "stop_sequence",
            "max_tokens": "max_tokens",
            "tool_use": "tool_use",
            "content_filtered": "end_turn",
        }
        return mapping.get(stop_reason, stop_reason)

    def _convert_bedrock_content_to_anthropic(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anthropic_content: List[Dict[str, Any]] = []
        for block in content:
            if "text" in block:
                text = block["text"]
                if text:  # Skip empty text blocks (common before tool_use in Bedrock)
                    anthropic_content.append({"type": "text", "text": text})
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_input = tool_use.get("input")
                if not isinstance(tool_input, dict):
                    tool_input = {}
                anthropic_content.append({
                    "type": "tool_use",
                    "id": tool_use.get("toolUseId", ""),
                    "name": tool_use.get("name", ""),
                    "input": tool_input,
                })
            elif "reasoningContent" in block:
                reasoning = block["reasoningContent"]
                # Reconstruct redacted_thinking on the non-stream path too
                # (the stream path already does this); otherwise redacted
                # reasoning is silently dropped.
                if "redactedContent" in reasoning:
                    redacted = reasoning["redactedContent"]
                    # Bedrock may return bytes; Anthropic expects a string blob.
                    if isinstance(redacted, (bytes, bytearray)):
                        redacted = base64.b64encode(redacted).decode("utf-8")
                    anthropic_content.append({
                        "type": "redacted_thinking",
                        "data": redacted,
                    })
                    continue
                reasoning_text = reasoning.get("reasoningText", {})
                anthropic_thinking = {
                    "type": "thinking",
                    "thinking": reasoning_text.get("text", ""),
                }
                signature = reasoning_text.get("signature") or reasoning.get("signature")
                if signature:
                    anthropic_thinking["signature"] = signature
                anthropic_content.append(anthropic_thinking)
            else:
                # Don't silently drop unrecognized block types.
                logger.warning(f"[BEDROCK] Unrecognized content block type, dropping: {list(block.keys())}")
        return anthropic_content

    def _format_anthropic_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"

    async def anthropic_messages(self, request, anthropic_beta: Optional[str] = None) -> Any:
        """Handle Anthropic Messages API request.

        Routes Claude models through the AsyncAnthropicBedrock SDK (native
        Anthropic JSON) and non-Claude models through the Converse API.
        """
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)

        # ── Native path for Claude models (AsyncAnthropicBedrock SDK) ──────
        if self._is_claude_model(model_id):
            self._init_bedrock_anthropic_client()
            if self._anthropic_client is None:
                raise NotImplementedError(
                    "anthropic SDK not available for Bedrock native path"
                )

            kwargs, beta_header = self._build_native_sdk_kwargs(
                request, model_id, anthropic_beta
            )
            if beta_header:
                kwargs["extra_headers"] = {"anthropic-beta": beta_header}

            if self.debug:
                logger.info(
                    "[BEDROCK NATIVE] messages.create: model=%s, msgs=%d, tools=%s, thinking=%s",
                    model_id,
                    len(kwargs.get("messages", [])),
                    bool(kwargs.get("tools")),
                    bool(kwargs.get("thinking")),
                )

            try:
                response = await self._anthropic_client.messages.create(**kwargs)
            except Exception as e:
                raise self._translate_bedrock_sdk_error(e) from e

            result = json.loads(response.model_dump_json(warnings="none"))
            # Echo the client-facing model name; keep the real upstream message id.
            result["model"] = request.model
            return result

        # ── Converse path for non-Claude models ────────────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
            async with _get_converse_semaphore():
                response = await run_in_threadpool(self.bedrock_runtime.converse, **args)
        except ClientError as e:
            error = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            if error.get("Code") == "ValidationException":
                raise ValueError(error.get("Message") or str(e)) from e
            raise

        output_message = response.get("output", {}).get("message", {})
        usage = response.get("usage", {})

        usage_data = {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }
        if usage.get("cacheWriteInputTokens") is not None:
            usage_data["cache_creation_input_tokens"] = usage.get("cacheWriteInputTokens")
        if usage.get("cacheReadInputTokens") is not None:
            usage_data["cache_read_input_tokens"] = usage.get("cacheReadInputTokens")

        # Surface the actual stop sequence when the model stopped on one,
        # instead of hardcoding None. Converse returns it as `stopSequence`.
        stop_reason_raw = response.get("stopReason")
        stop_sequence = None
        if stop_reason_raw == "stop_sequence":
            stop_sequence = response.get("stopSequence") or output_message.get("stopSequence")

        # A10: if a guardrail intervened, don't report a normal stop — log the
        # trace so the intervention isn't silently invisible.
        trace = response.get("trace")
        if trace and isinstance(trace, dict) and trace.get("guardrail"):
            logger.warning(f"[BEDROCK] Guardrail trace present on response: {trace.get('guardrail')}")

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": self._convert_bedrock_content_to_anthropic(output_message.get("content", [])),
            "model": request.model,
            "stop_reason": self._convert_bedrock_stop_reason(stop_reason_raw),
            "stop_sequence": stop_sequence,
            "usage": usage_data,
        }

    async def anthropic_messages_stream(self, request, anthropic_beta: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic Messages API request.

        Routes Claude models through the AsyncAnthropicBedrock SDK
        (``messages.stream`` → native Anthropic SSE events), and non-Claude
        models through the Converse Stream API.
        """
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)

        # ── Native streaming for Claude models (AsyncAnthropicBedrock SDK) ──
        if self._is_claude_model(model_id):
            self._init_bedrock_anthropic_client()
            if self._anthropic_client is None:
                raise NotImplementedError(
                    "anthropic SDK not available for Bedrock native streaming path"
                )

            kwargs, beta_header = self._build_native_sdk_kwargs(
                request, model_id, anthropic_beta
            )
            if beta_header:
                kwargs["extra_headers"] = {"anthropic-beta": beta_header}

            if self.debug:
                logger.info(
                    "[BEDROCK STREAM NATIVE] messages.stream: model=%s, msgs=%d, tools=%s, thinking=%s",
                    model_id,
                    len(kwargs.get("messages", [])),
                    bool(kwargs.get("tools")),
                    bool(kwargs.get("thinking")),
                )

            full_provider_name = getattr(self, "full_provider_name", "bedrock")
            terminal_event_type = None
            terminal_seen_at: Optional[float] = None
            drained_event_count = 0

            try:
                async with self._anthropic_client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if terminal_event_type is not None:
                            drained_event_count += 1
                            drain_stop_reason = get_anthropic_post_terminal_drain_stop_reason(
                                terminal_seen_at=terminal_seen_at,
                                drained_event_count=drained_event_count,
                            )
                            if drain_stop_reason is not None:
                                logger.warning(
                                    "[BEDROCK STREAM NATIVE] post-terminal drain budget reached "
                                    "provider=%s model=%s terminal_event=%s drained_event_count=%s stop_reason=%s",
                                    full_provider_name,
                                    request.model,
                                    terminal_event_type,
                                    drained_event_count,
                                    drain_stop_reason,
                                )
                                break
                            continue

                        json_str = event.model_dump_json(exclude_none=True, warnings="none")
                        event_data = json.loads(json_str)
                        event_type = (
                            event_data.get("type", "unknown")
                            if isinstance(event_data, dict)
                            else "unknown"
                        )
                        yield f"event: {event_type}\ndata: {json_str}\n\n"

                        if is_anthropic_terminal_stream_event(
                            event_type=event_type, event_data=event_data
                        ):
                            terminal_event_type = event_type
                            terminal_seen_at = time.monotonic()
                            logger.info(
                                "[BEDROCK STREAM NATIVE] terminal event provider=%s model=%s event_type=%s",
                                full_provider_name,
                                request.model,
                                event_type,
                            )
            except Exception as e:
                raise self._translate_bedrock_sdk_error(e) from e
            return

        # ── Converse Stream path for non-Claude models ─────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
            # Hold the Converse permit only while opening the stream (boto3
            # returns once headers arrive), not while the client drains it.
            async with _get_converse_semaphore():
                response = await run_in_threadpool(self.bedrock_runtime.converse_stream, **args)
        except ClientError as e:
            error = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            if error.get("Code") == "ValidationException":
                raise ValueError(error.get("Message") or str(e)) from e
            raise
        stream = response.get("stream")

        message_id = f"msg_{uuid.uuid4().hex}"
        content_block_started = set()
        content_block_stopped = set()
        # Bedrock event order: ... contentBlockStop → messageStop → metadata
        # We must buffer the messageStop payload and only emit message_delta
        # once we have real token counts from the trailing metadata event.
        buffered_stop_reason: Optional[str] = None
        buffered_stop_sequence: Optional[str] = None
        message_stop_seen = False
        message_stop_emitted = False

        async for chunk in self._async_iterate_catching(stream):
            if "messageStart" in chunk:
                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": request.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                yield self._format_anthropic_sse_event("message_start", message_start)
            elif "contentBlockStart" in chunk:
                start = chunk["contentBlockStart"]
                index = start.get("contentBlockIndex", 0)
                start_body = start.get("start", {})
                if "toolUse" in start_body:
                    tool_use = start_body["toolUse"]
                    block = {
                        "type": "tool_use",
                        "id": tool_use.get("toolUseId", ""),
                        "name": tool_use.get("name", ""),
                        # Anthropic spec: content_block_start always has empty input;
                        # actual input arrives via input_json_delta events.
                        "input": {},
                    }
                elif "reasoningContent" in start_body:
                    reasoning_content = start_body["reasoningContent"]
                    if "redactedContent" in reasoning_content:
                        block = {"type": "redacted_thinking", "data": ""}
                    else:
                        block = {"type": "thinking", "thinking": ""}
                else:
                    block = {"type": "text", "text": ""}
                content_block_started.add(index)
                yield self._format_anthropic_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": block,
                })
            elif "contentBlockDelta" in chunk:
                delta_data = chunk["contentBlockDelta"]
                index = delta_data.get("contentBlockIndex", 0)
                delta = delta_data.get("delta", {})

                if index not in content_block_started:
                    content_block_started.add(index)
                    # Detect block type from the delta content
                    if "reasoningContent" in delta:
                        synth_block = {"type": "thinking", "thinking": ""}
                    else:
                        synth_block = {"type": "text", "text": ""}
                    yield self._format_anthropic_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": synth_block,
                    })

                if "text" in delta:
                    yield self._format_anthropic_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "text_delta",
                            "text": delta.get("text", ""),
                        },
                    })
                elif "toolUse" in delta:
                    # Bedrock sends tool input as a JSON string fragment (already
                    # serialized), but occasionally delivers it as a dict.  The
                    # Anthropic SSE format requires ``partial_json`` to be a
                    # *string*, so we json.dumps dicts to keep downstream
                    # clients (Claude Code, Cline, etc.) from seeing an empty
                    # tool input.
                    tool_input = delta["toolUse"].get("input", "")
                    if isinstance(tool_input, dict):
                        tool_input = json.dumps(tool_input, ensure_ascii=False)
                    elif tool_input is None:
                        tool_input = ""
                    yield self._format_anthropic_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_input,
                        },
                    })
                elif "reasoningContent" in delta:
                    reasoning_delta = delta["reasoningContent"]
                    if "text" in reasoning_delta:
                        yield self._format_anthropic_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": reasoning_delta.get("text", ""),
                            },
                        })
                    elif "signature" in reasoning_delta:
                        yield self._format_anthropic_sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": reasoning_delta.get("signature", ""),
                            },
                        })
            elif "contentBlockStop" in chunk:
                stop = chunk["contentBlockStop"]
                blk_idx = stop.get("contentBlockIndex", 0)
                content_block_stopped.add(blk_idx)
                yield self._format_anthropic_sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": blk_idx,
                })
            elif "metadata" in chunk and "usage" in chunk["metadata"]:
                usage = chunk["metadata"]["usage"]
                # metadata is the last event — emit message_delta now with real
                # token counts merged with the buffered stop_reason.
                usage_data: Dict[str, Any] = {
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                }
                if usage.get("cacheWriteInputTokens") is not None:
                    usage_data["cache_creation_input_tokens"] = usage.get("cacheWriteInputTokens")
                if usage.get("cacheReadInputTokens") is not None:
                    usage_data["cache_read_input_tokens"] = usage.get("cacheReadInputTokens")
                yield self._format_anthropic_sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": buffered_stop_reason,
                        "stop_sequence": buffered_stop_sequence,
                    },
                    "usage": usage_data,
                })
                yield self._format_anthropic_sse_event("message_stop", {"type": "message_stop"})
                message_stop_emitted = True
            elif "_bedrock_stream_error" in chunk:
                # Bedrock raised an error mid-stream (e.g. EventStreamError
                # "Response ended prematurely").  Close any content blocks that
                # Bedrock never closed, emit the terminal Anthropic SSE events,
                # and then surface the error — so clients receive a well-formed
                # stream instead of a truncated one.
                terminal_already_seen = message_stop_seen or message_stop_emitted
                for open_idx in sorted(content_block_started - content_block_stopped):
                    yield self._format_anthropic_sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": open_idx,
                    })
                if not message_stop_emitted:
                    yield self._format_anthropic_sse_event("message_delta", {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": buffered_stop_reason or "end_turn",
                            "stop_sequence": buffered_stop_sequence,
                        },
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    })
                    yield self._format_anthropic_sse_event("message_stop", {"type": "message_stop"})
                    message_stop_emitted = True
                stream_exc = chunk["_bedrock_stream_error"]
                if terminal_already_seen:
                    logger.info(
                        "Bedrock converse stream ended after terminal messageStop; suppressing late transport error: %s",
                        stream_exc,
                    )
                    return
                logger.warning(f"Bedrock stream error mid-response: {stream_exc}")
                yield self._format_anthropic_sse_event("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": str(stream_exc),
                    },
                })
                return  # stream fully terminated; no post-loop fallback needed
            elif "messageStop" in chunk:
                message_stop_seen = True
                stop_reason = chunk["messageStop"].get("stopReason")
                # Buffer stop info — will be emitted once the trailing metadata arrives.
                buffered_stop_reason = self._convert_bedrock_stop_reason(stop_reason)
                buffered_stop_sequence = chunk["messageStop"].get("stopSequence")

        # Guard: if Bedrock ended the stream without sending a metadata event
        # (or without any messageStop/metadata at all), emit the terminal events
        # now so the client gets a well-formed Anthropic SSE stream and does not
        # report "Response ended prematurely".
        if not message_stop_emitted:
            yield self._format_anthropic_sse_event("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": buffered_stop_reason or "end_turn",
                    "stop_sequence": buffered_stop_sequence,
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
            yield self._format_anthropic_sse_event("message_stop", {"type": "message_stop"})

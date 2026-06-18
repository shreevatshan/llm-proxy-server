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
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
#  Module-level concurrency primitives for the InvokeModel (native) path.
#  Shared across BedrockProvider instances.  Thread-pool workers run the
#  synchronous boto3 calls; the semaphore prevents stampede overload.
#
#  Two separate pool/semaphore pairs are used:
#    • "native stream"  — InvokeModelWithResponseStream (Claude streaming).
#      Workers are long-lived (hold a thread for the full response duration).
#    • "native"         — InvokeModel (Claude non-stream / embeddings).
#      Workers are short-lived (return as soon as the response body is read).
#
#  Keeping them separate prevents a burst of long-running streams from
#  starving synchronous calls and vice versa.  Pool sizes are tunable via
#  env vars so operators can right-size them for their workload.
# ---------------------------------------------------------------------------
_NATIVE_POOL_SIZE = 15
_NATIVE_SEMAPHORE_LIMIT = 15
_NATIVE_STREAM_POOL_SIZE = 15   # dedicated to InvokeModelWithResponseStream workers
_NATIVE_STREAM_SEMAPHORE_LIMIT = 15
_BEDROCK_NATIVE_WARNING_THRESHOLD_RATIO = 0.6
_BEDROCK_NATIVE_FUTURE_DONE_EMPTY_CHECKS = 2
_BEDROCK_NATIVE_FUTURE_DONE_GRACE_SECONDS = 0.1
_BEDROCK_NATIVE_SEMAPHORE_WAIT_WARNING_SECONDS = 1.0
_BEDROCK_NATIVE_PROGRESS_EVENT_TYPES = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "error",
    }
)

_native_executor: Optional[ThreadPoolExecutor] = None
_native_semaphore: Optional[asyncio.Semaphore] = None
_native_executor_lock = threading.Lock()

# Dedicated to the native streaming (InvokeModelWithResponseStream) path.
_native_stream_executor: Optional[ThreadPoolExecutor] = None
_native_stream_semaphore: Optional[asyncio.Semaphore] = None
_native_stream_executor_lock = threading.Lock()


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


BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS", 180
)
BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS", 300
)
BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS", 90
)
BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS", 900
)
BEDROCK_NATIVE_PING_INTERVAL_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_PING_INTERVAL_SECONDS", 30
)
BEDROCK_NATIVE_SOCKET_READ_TIMEOUT_SECONDS = _get_positive_int_env(
    "BEDROCK_NATIVE_SOCKET_READ_TIMEOUT_SECONDS", 940
)
BEDROCK_NATIVE_RETRY_ON_PRE_OUTPUT_DROP = os.environ.get(
    "BEDROCK_NATIVE_RETRY_ON_PRE_OUTPUT_DROP", "1"
) not in ("0", "false", "False", "")


def _get_native_executor() -> ThreadPoolExecutor:
    """Get or create the global thread-pool executor for InvokeModel calls."""
    global _native_executor
    if _native_executor is None:
        with _native_executor_lock:
            if _native_executor is None:
                _native_executor = ThreadPoolExecutor(
                    max_workers=_NATIVE_POOL_SIZE,
                    thread_name_prefix="bedrock-native-",
                )
                logger.info(
                    "[BEDROCK] Created native thread pool with %d workers",
                    _NATIVE_POOL_SIZE,
                )
    return _native_executor


def _get_native_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for InvokeModel (non-stream) concurrency."""
    global _native_semaphore
    if _native_semaphore is None:
        _native_semaphore = asyncio.Semaphore(_NATIVE_SEMAPHORE_LIMIT)
        logger.info(
            "[BEDROCK] Created native semaphore with limit %d",
            _NATIVE_SEMAPHORE_LIMIT,
        )
    return _native_semaphore


def _get_native_stream_executor() -> ThreadPoolExecutor:
    """Get or create the dedicated thread-pool for InvokeModelWithResponseStream workers.

    Kept separate from the non-stream executor so long-running stream workers
    cannot starve short synchronous InvokeModel calls, and vice versa.
    Pool size is configurable via BEDROCK_NATIVE_STREAM_POOL_SIZE env var.
    """
    global _native_stream_executor
    if _native_stream_executor is None:
        with _native_stream_executor_lock:
            if _native_stream_executor is None:
                pool_size = _get_positive_int_env(
                    "BEDROCK_NATIVE_STREAM_POOL_SIZE", _NATIVE_STREAM_POOL_SIZE
                )
                _native_stream_executor = ThreadPoolExecutor(
                    max_workers=pool_size,
                    thread_name_prefix="bedrock-native-stream-",
                )
                logger.info(
                    "[BEDROCK] Created native-stream thread pool with %d workers",
                    pool_size,
                )
    return _native_stream_executor


def _get_native_stream_semaphore() -> asyncio.Semaphore:
    """Get or create the dedicated semaphore for InvokeModelWithResponseStream concurrency.

    Kept separate from the non-stream semaphore so streaming and synchronous
    native calls cannot deplete each other's permit budgets.
    Limit is configurable via BEDROCK_NATIVE_STREAM_SEMAPHORE_LIMIT env var.
    """
    global _native_stream_semaphore
    if _native_stream_semaphore is None:
        limit = _get_positive_int_env(
            "BEDROCK_NATIVE_STREAM_SEMAPHORE_LIMIT", _NATIVE_STREAM_SEMAPHORE_LIMIT
        )
        _native_stream_semaphore = asyncio.Semaphore(limit)
        logger.info(
            "[BEDROCK] Created native-stream semaphore with limit %d",
            limit,
        )
    return _native_stream_semaphore


def _close_stream_box(stream_box: Optional[list]) -> None:
    """Close the boto3 event-stream held in *stream_box[0]* if present.

    Called from the async consumer when it exits early so the worker thread
    blocked in ``next(stream)`` is unblocked immediately rather than waiting
    for the ~940 s socket read_timeout.  Safe to call from any thread/coroutine;
    idempotent (no-op when stream_box is None or already cleared).
    """
    if stream_box is None:
        return
    stream = stream_box[0]
    if stream is None:
        return
    close_fn = getattr(stream, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            logger.debug(
                "[BEDROCK STREAM NATIVE] _close_stream_box: ignoring close error",
                exc_info=True,
            )


def _classify_native_stream_exception(exc: Exception) -> str:
    """Bucket native-stream worker exceptions by type-name so operators can grep.

    Returns one of: "protocol_error" (urllib3 connection truncation),
    "event_stream_error" (botocore EventStreamError),
    "read_timeout" (urllib3 ReadTimeoutError — socket read deadline hit),
    "connection_closed" (socket fp set to None mid-stream),
    "other".
    """
    name = type(exc).__name__
    if name == "ProtocolError":
        return "protocol_error"
    if name == "EventStreamError":
        return "event_stream_error"
    if name == "ReadTimeoutError":
        return "read_timeout"
    # http.client sets self.fp = None when the connection is closed; urllib3
    # then propagates this as AttributeError: 'NoneType' object has no attribute 'read'.
    if name == "AttributeError" and "'NoneType' object has no attribute 'read'" in str(exc):
        return "connection_closed"
    return "other"


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
    _extract_system_messages_from_messages,
    _merge_system_fields,
    is_anthropic_terminal_stream_event,
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


class BedrockNativeStreamError(Exception):
    """Base class for Bedrock native Claude streaming failures."""


class BedrockNativeIdleTimeout(BedrockNativeStreamError):
    """Raised when Bedrock native Claude streaming goes idle without progress."""

    def __init__(
        self,
        *,
        model: str,
        phase: str,
        idle_seconds: float,
        threshold_seconds: int,
        last_progress_event_type: Optional[str],
        progress_event_count: int,
        proxy_ping_count: int,
    ):
        self.model = model
        self.phase = phase
        self.idle_seconds = idle_seconds
        self.threshold_seconds = threshold_seconds
        self.last_progress_event_type = last_progress_event_type
        self.progress_event_count = progress_event_count
        self.proxy_ping_count = proxy_ping_count
        super().__init__(
            f"Bedrock native Claude stream stalled during {phase} phase after "
            f"{idle_seconds:.1f}s without upstream progress (threshold {threshold_seconds}s)."
        )


class BedrockNativePrematureEOF(BedrockNativeStreamError):
    """Raised when Bedrock closes the stream without a terminal Anthropic event."""

    def __init__(
        self,
        *,
        model: str,
        transport_eof_observed: bool,
        last_progress_event_type: Optional[str],
        progress_event_count: int,
        proxy_ping_count: int,
    ):
        self.model = model
        self.transport_eof_observed = transport_eof_observed
        self.last_progress_event_type = last_progress_event_type
        self.progress_event_count = progress_event_count
        self.proxy_ping_count = proxy_ping_count
        super().__init__(
            "Bedrock native Claude stream ended before a terminal Anthropic event was received."
        )


class BedrockNativeProviderError(BedrockNativeStreamError):
    """Raised when the Bedrock worker reports a provider or transport failure."""

    def __init__(self, error_code: str, error_message: str):
        self.error_code = error_code
        self.error_message = error_message
        mapped = _map_bedrock_error(error_code, error_message)
        self.status_code = mapped["status"]
        self.body = mapped["body"]
        super().__init__(f"{error_code}: {error_message}")

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

        required_min_native_socket_timeout = (
            max(
                BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS,
                BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS,
            )
            + 30
        )
        configured_native_socket_timeout = BEDROCK_NATIVE_SOCKET_READ_TIMEOUT_SECONDS
        if configured_native_socket_timeout < required_min_native_socket_timeout:
            logger.warning(
                "[BEDROCK] BEDROCK_NATIVE_SOCKET_READ_TIMEOUT_SECONDS=%s is too low; "
                "using %s instead so native stream cleanup outlives the idle watchdog",
                configured_native_socket_timeout,
                required_min_native_socket_timeout,
            )
        self.native_stream_socket_read_timeout = max(
            configured_native_socket_timeout,
            required_min_native_socket_timeout,
        )
        self.native_stream_ping_interval = BEDROCK_NATIVE_PING_INTERVAL_SECONDS

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

        native_stream_client_kwargs = dict(client_kwargs)
        native_stream_client_kwargs["config"] = Config(
            connect_timeout=60,
            read_timeout=self.native_stream_socket_read_timeout,
            # 3 attempts (initial + 2 retries) is sufficient for transient
            # connection errors; the idle watchdog handles genuine slow starts.
            # The original value of 8 adaptive retries prolonged thread holding
            # on Bedrock throttling, contributing to pool exhaustion.
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50,
        )
        self.bedrock_runtime_native_stream = boto3.client(**native_stream_client_kwargs)
        
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

    def _is_native_stream_thinking_enabled(self, request: Any) -> bool:
        """Return True when the Anthropic request enables extended thinking."""
        thinking = getattr(request, "thinking", None)
        if thinking is None:
            return False

        if hasattr(thinking, "model_dump"):
            thinking = thinking.model_dump(exclude_none=True)

        if isinstance(thinking, dict):
            return thinking.get("type", "disabled") != "disabled"

        return getattr(thinking, "type", "disabled") != "disabled"

    def _log_native_stream_drop(
        self,
        *,
        request: Any,
        model_id: str,
        aws_request_id: Optional[str],
        error_class: str,
        elapsed_seconds: float,
        progress_event_count: int,
        last_progress_event_type: Optional[str],
        proxy_ping_count: int,
        output_bytes_received: int,
    ) -> None:
        """Emit one structured drop record. Grep [BEDROCK STREAM NATIVE DROP]."""
        logger.warning(
            "[BEDROCK STREAM NATIVE DROP] aws_request_id=%s model_id=%s error_class=%s "
            "elapsed_seconds=%.1f progress_event_count=%d last_progress_event=%s "
            "proxy_ping_count=%d output_bytes_received=%d thinking_enabled=%s "
            "tools_present=%s max_tokens=%s",
            aws_request_id or "unknown",
            model_id,
            error_class,
            elapsed_seconds,
            progress_event_count,
            last_progress_event_type or "none",
            proxy_ping_count,
            output_bytes_received,
            self._is_native_stream_thinking_enabled(request),
            bool(getattr(request, "tools", None)),
            getattr(request, "max_tokens", None),
        )

    def _get_native_stream_idle_timeout(
        self,
        request: Any,
        saw_first_progress_event: bool,
    ) -> tuple[int, str]:
        """Return the effective Bedrock native idle threshold and phase name."""
        thinking_enabled = self._is_native_stream_thinking_enabled(request)
        if not saw_first_progress_event:
            if thinking_enabled:
                return BEDROCK_NATIVE_THINKING_INITIAL_IDLE_TIMEOUT_SECONDS, "initial"
            return BEDROCK_NATIVE_INITIAL_IDLE_TIMEOUT_SECONDS, "initial"

        if thinking_enabled or bool(getattr(request, "tools", None)):
            return BEDROCK_NATIVE_EXTENDED_MIDSTREAM_IDLE_TIMEOUT_SECONDS, "midstream"
        return BEDROCK_NATIVE_MIDSTREAM_IDLE_TIMEOUT_SECONDS, "midstream"

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

    def _refresh_model_list(self):
        """Refresh the list of available Bedrock models."""
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
            
            self.bedrock_model_list = model_list
            logger.info(f"Loaded {len(model_list)} Bedrock models")
            
        except Exception as e:
            logger.error(f"Error listing Bedrock models: {e}")
            # Set a default model
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
        """Explicitly refresh the model list from Bedrock."""
        await run_in_threadpool(self._refresh_model_list)

    def _truncate_tool_name(self, tool_name: str, max_length: int = 64, tool_name_mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Truncate tool name to fit Bedrock's 64 character limit.
        
        Uses a hash suffix to preserve uniqueness for truncated names.
        If tool_name_mapping dict is provided, stores the mapping for
        restoring original names in responses.
        """
        if not tool_name or len(tool_name) <= max_length:
            return tool_name
        
        import hashlib
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

    def _parse_image_sync(self, image_url: str) -> tuple:
        """Parse image from URL or base64 data (synchronous - for use in threadpool)."""
        pattern = r"^data:(image/[a-z]*);base64,\s*"
        content_type = re.search(pattern, image_url)
        
        # Check if already base64 encoded
        if content_type:
            image_data = re.sub(pattern, "", image_url)
            return base64.b64decode(image_data), content_type.group(1)
        
        # Download from URL (blocking - must be called from threadpool)
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "image/jpeg")
            if not content_type.startswith("image"):
                content_type = "image/jpeg"
            return response.content, content_type
        else:
            raise ValueError(f"Unable to access image URL: {image_url}")

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
        
        AWS Bedrock requires that every toolUse block in an assistant message
        must be IMMEDIATELY followed by a user message containing a corresponding
        toolResult block with matching toolUseId.
        
        This method:
        1. For each assistant message with toolUse blocks, checks if the immediately
           next message is a user message with matching toolResult blocks
        2. Removes orphaned toolUse blocks that don't have their toolResult in the
           immediately following user message
        3. Also removes orphaned toolResult blocks that don't have a preceding toolUse
        4. Logs warnings for removed orphaned items
        """
        if not messages:
            return messages
        
        # First pass: identify all valid tool_use/tool_result pairs
        # A valid pair requires: assistant message with toolUse immediately followed by user message with matching toolResult
        valid_tool_ids = set()
        
        for i in range(len(messages) - 1):
            curr_msg = messages[i]
            next_msg = messages[i + 1]
            
            if curr_msg.get("role") == "assistant" and next_msg.get("role") == "user":
                # Get toolUse IDs from current assistant message
                curr_content = curr_msg.get("content", [])
                if isinstance(curr_content, list):
                    tool_use_ids = set()
                    for item in curr_content:
                        if isinstance(item, dict) and "toolUse" in item:
                            tool_use_id = item["toolUse"].get("toolUseId")
                            if tool_use_id:
                                tool_use_ids.add(tool_use_id)
                    
                    # Get toolResult IDs from next user message
                    next_content = next_msg.get("content", [])
                    if isinstance(next_content, list):
                        for item in next_content:
                            if isinstance(item, dict) and "toolResult" in item:
                                tool_result_id = item["toolResult"].get("toolUseId")
                                # Only valid if there's a matching toolUse in the immediately preceding assistant message
                                if tool_result_id and tool_result_id in tool_use_ids:
                                    valid_tool_ids.add(tool_result_id)
        
        if self.debug:
            logger.info(f"[BEDROCK] Valid tool IDs (paired): {valid_tool_ids}")
        
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
        
        # Claude >= 4.7 don't support temperature, topP, or topK
        if self._is_claude_at_least(request.model, 4, 7):
            inference_config.pop("temperature", None)
            inference_config.pop("topP", None)
            inference_config.pop("topK", None)
            if self.debug:
                logger.info(f"Removed temperature, topP, and topK for {request.model} (not supported in Claude >= 4.7)")
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
            budget_tokens = self._calc_budget_tokens(max_tokens, request.reasoning_effort)
            inference_config["maxTokens"] = max_tokens
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
                # reasoning_config will not be used
                args["additionalModelRequestFields"] = filtered_extra_fields
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

    def _calc_budget_tokens(self, max_tokens: int, reasoning_effort: str) -> int:
        """Calculate budget tokens for reasoning."""
        if reasoning_effort == "low":
            return int(max_tokens * 0.3)
        elif reasoning_effort == "medium":
            return int(max_tokens * 0.6)
        else:
            return max_tokens - 1

    def _convert_finish_reason(self, finish_reason: str) -> Optional[str]:
        """Convert Bedrock finish reason to OpenAI format."""
        if not finish_reason:
            return None
        
        mapping = {
            "tool_use": "tool_calls",
            "finished": "stop",
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "complete": "stop",
            "content_filtered": "content_filter",
        }
        return mapping.get(finish_reason.lower(), finish_reason.lower())

    async def _invoke_bedrock(self, request: ChatCompletionRequest, stream: bool = False):
        """Invoke Bedrock model.
        
        Returns:
            tuple: (response, tool_name_mapping) where tool_name_mapping is a
                   request-scoped dict mapping truncated tool names to originals.
        """
        # Request-scoped tool name mapping to avoid race conditions between concurrent requests
        tool_name_mapping: Dict[str, str] = {}
        
        # Extract clean model ID from prefixed model name
        model_id = self.get_model_id(request.model)
        
        if self.debug:
            logger.info(f"Bedrock request for model: {request.model} (clean model_id: {model_id})")
        
        # Create a modified request with clean model ID for parsing
        clean_request = ChatCompletionRequest(
            model=model_id,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            stream=request.stream,
            tools=request.tools,
            tool_choice=request.tool_choice,
            **request.model_extra if hasattr(request, 'model_extra') else {}
        )
        
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
        
        try:
            if stream:
                response = await run_in_threadpool(
                    self.bedrock_runtime.converse_stream, **args
                )
            else:
                response = await run_in_threadpool(
                    self.bedrock_runtime.converse, **args
                )
            return response, tool_name_mapping
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request."""
        response, tool_name_mapping = await self._invoke_bedrock(request, stream=False)
        
        output_message = response["output"]["message"]
        input_tokens = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]
        finish_reason = response["stopReason"]
        
        if self.debug:
            logger.info(f"[BEDROCK] Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
        
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
            
            # Combine reasoning content with main content
            if message.reasoning_content:
                message.content = f"<think>{message.reasoning_content}</think>{content}"
                message.reasoning_content = None
            else:
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
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
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
        import asyncio

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
        import asyncio

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
            
            # Track think tag emission for this specific stream (local variable to avoid race conditions)
            think_emitted = False
            
            # Default include_usage to True if not specified
            include_usage = True
            if request.stream_options is not None:
                include_usage = request.stream_options.include_usage if request.stream_options.include_usage is not None else True
            
            async for chunk in self._async_iterate(stream):
                # Pass think_emitted state to parser and get it back
                stream_response, think_emitted = self._parse_stream_chunk(chunk, message_id, request.model, think_emitted, tool_name_mapping)
                
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
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_data = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield self.format_sse_data(error_data)

    def _parse_stream_chunk(self, chunk: Dict, message_id: str, model_id: str, think_emitted: bool, tool_name_mapping: Optional[Dict[str, str]] = None) -> tuple:
        """Parse Bedrock stream chunk into OpenAI format.
        
        Returns:
            tuple: (response_dict, think_emitted_state)
        """
        if self.debug:
            logger.info(f"Bedrock chunk: {chunk}")
        
        delta = {}
        finish_reason = None
        usage = None
        
        if "messageStart" in chunk:
            delta = {"role": chunk["messageStart"]["role"], "content": ""}
        
        elif "contentBlockStart" in chunk:
            start = chunk["contentBlockStart"]["start"]
            if "toolUse" in start:
                index = chunk["contentBlockStart"]["contentBlockIndex"] - 1
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
                if "text" in block_delta["reasoningContent"]:
                    content = block_delta["reasoningContent"]["text"]
                    if not think_emitted:
                        # Port of "content_block_start" with "thinking"
                        content = "<think>" + content
                        think_emitted = True
                    delta = {"content": content}
                elif "signature" in block_delta["reasoningContent"]:
                    # Port of "signature_delta"
                    if think_emitted:
                        delta = {"content": "\n</think>\n\n"}
                    else:
                        return None, think_emitted  # Ignore signature if no <think> started
            elif "toolUse" in block_delta:
                index = chunk["contentBlockDelta"]["contentBlockIndex"] - 1
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
                usage = Usage(
                    prompt_tokens=metadata["usage"]["inputTokens"],
                    completion_tokens=metadata["usage"]["outputTokens"],
                    total_tokens=metadata["usage"]["totalTokens"]
                )
                # Return usage chunk
                return ({
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [],
                    "usage": usage.model_dump()
                }, think_emitted)
        
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
            }, think_emitted)
        
        return None, think_emitted

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
            args = {
                "texts": texts,
                "input_type": "search_document",
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
            input_tokens = 0  # Cohere doesn't return token count
        
        # Handle Titan models
        elif "Titan" in model_name:
            if len(texts) != 1:
                raise ValueError("Titan models only support single input")
            
            args = {"inputText": texts[0]}
            response = await run_in_threadpool(
                self.bedrock_runtime.invoke_model,
                body=json.dumps(args),
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            embeddings = [response_body["embedding"]]
            input_tokens = response_body.get("inputTextTokenCount", 0)
        
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
        lower = model_id.lower()
        if "claude" not in lower:
            return False
        import re
        match = re.search(r'claude-(?:sonnet|opus|haiku)-(\d+)-(\d+)', lower)
        if not match:
            return False
        major, minor = int(match.group(1)), int(match.group(2))
        return (major, minor) >= (min_major, min_minor)

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
        if request.top_p is not None:
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

                # Skip unsupported tools
                if tool_type in _skip_types:
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

    # --- native response conversion ---

    def _convert_native_response(
        self, response_body: Dict[str, Any], original_model: str
    ) -> Dict[str, Any]:
        """Wrap a native InvokeModel response into the dict we return from
        ``anthropic_messages``.  The response is already Anthropic JSON —
        we just stamp our own ``id`` and echo the client-facing model name.
        """
        usage = response_body.get("usage", {})
        usage_data: Dict[str, Any] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
        for optional_key in (
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "iterations",
        ):
            if usage.get(optional_key) is not None:
                usage_data[optional_key] = usage[optional_key]

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": response_body.get("content", []),
            "model": original_model,
            "stop_reason": response_body.get("stop_reason"),
            "stop_sequence": response_body.get("stop_sequence"),
            "usage": usage_data,
        }

    # --- native stream worker (runs in thread pool) ---

    def _start_native_stream_worker(
        self,
        model_id: str,
        native_request: Dict[str, Any],
    ) -> tuple["queue.Queue", asyncio.Future, threading.Event, list]:
        """Launch a fresh worker for the native Claude streaming path.

        Returns the (event_queue, future, stop_event, stream_box) 4-tuple
        driving one invocation of ``_stream_worker_native``. Used both at
        stream entry and on the pre-output retry path.

        ``stream_box`` is a single-element list that the worker populates with
        the boto3 event-stream object once it has been opened.  The consumer
        can call ``stream_box[0].close()`` from the async side to immediately
        unblock a worker thread parked inside ``next(stream)`` — this prevents
        zombie threads from exhausting the thread pool when the consumer exits
        early (client disconnect, idle timeout, outer stream wrapper timeout).
        """
        event_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        stream_box: list = [None]  # filled by worker after stream is opened
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            _get_native_stream_executor(),
            self._stream_worker_native,
            model_id,
            native_request,
            event_queue,
            stop_event,
            stream_box,
        )
        return event_queue, future, stop_event, stream_box

    def _stream_worker_native(
        self,
        bedrock_model_id: str,
        native_request: Dict[str, Any],
        event_queue: "queue.Queue[tuple]",
        stop_event: Optional[threading.Event] = None,
        stream_box: Optional[list] = None,
    ) -> None:
        """Thread-pool worker: call ``invoke_model_with_response_stream`` and
        push native Anthropic SSE events into *event_queue*.

        Queue protocol:
            ("upstream_event", {"sse": str, "event_type": str})
            ("done", {...})         — worker finished and reports terminal state
            ("error", {...})        — an error occurred

        ``stream_box`` is a single-element list shared with the consumer.  The
        worker stores the open boto3 event-stream in ``stream_box[0]`` as soon
        as it is available so the async consumer can close it from the event
        loop thread, unblocking this thread if it is parked in ``next(stream)``.
        """
        stream = None
        terminal_event_type = None
        transport_eof_observed = False
        worker_stopped_early = False
        event_count = 0
        output_bytes_received = 0
        aws_request_id: Optional[str] = None
        stream_start = time.monotonic()
        try:
            if stop_event is not None and stop_event.is_set():
                worker_stopped_early = True
                logger.info(
                    "[BEDROCK STREAM NATIVE] Stop requested before invoke for model=%s",
                    bedrock_model_id,
                )
                event_queue.put(
                    (
                        "done",
                        {
                            "terminal_event_seen": False,
                            "terminal_event_type": None,
                            "transport_eof_observed": False,
                            "event_count": 0,
                            "worker_stopped_early": True,
                            "aws_request_id": None,
                            "output_bytes_received": 0,
                        },
                    )
                )
                return

            response = self.bedrock_runtime_native_stream.invoke_model_with_response_stream(
                modelId=bedrock_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(native_request),
            )
            response_metadata = response.get("ResponseMetadata", {}) or {}
            aws_request_id = response_metadata.get("RequestId")
            logger.info(
                "[BEDROCK STREAM NATIVE] Stream opened: model=%s aws_request_id=%s",
                bedrock_model_id,
                aws_request_id or "unknown",
            )
            stream = response.get("body")
            if not stream:
                event_queue.put(
                    (
                        "error",
                        {
                            "code": "no_stream",
                            "message": "No stream body returned from Bedrock",
                            "error_class": "other",
                            "aws_request_id": aws_request_id,
                            "output_bytes_received": output_bytes_received,
                        },
                    )
                )
                return
            # Publish the stream object so the async consumer can close it
            # from the event loop if it needs to exit early, which unblocks
            # this thread parked in next(stream) without waiting for the
            # ~940s socket read_timeout.
            if stream_box is not None:
                stream_box[0] = stream

            for event in stream:
                if stop_event is not None and stop_event.is_set():
                    worker_stopped_early = True
                    logger.info(
                        "[BEDROCK STREAM NATIVE] Stop requested during stream for model=%s events_received=%d elapsed=%.1fs",
                        bedrock_model_id,
                        event_count,
                        time.monotonic() - stream_start,
                    )
                    break
                if terminal_event_type is not None:
                    # Stop consuming once the upstream sent a terminal event.
                    # The stream is closed in the finally block below.
                    break
                chunk = event.get("chunk")
                if chunk:
                    chunk_bytes = chunk.get("bytes")
                    if chunk_bytes:
                        event_count += 1
                        output_bytes_received += len(chunk_bytes)
                        event_data = json.loads(chunk_bytes.decode("utf-8"))
                        event_type = event_data.get("type", "unknown")
                        sse = f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False, separators=(',', ':'))}\n\n"
                        event_queue.put(
                            (
                                "upstream_event",
                                {
                                    "sse": sse,
                                    "event_type": event_type,
                                    "event_data": event_data,
                                },
                            )
                        )
                        if is_anthropic_terminal_stream_event(
                            event_type=event_type,
                            event_data=event_data,
                        ):
                            terminal_event_type = event_type
                            logger.info(
                                "[BEDROCK STREAM NATIVE] Terminal event for model=%s event_type=%s",
                                bedrock_model_id,
                                event_type,
                            )
                            break

            if terminal_event_type is None:
                transport_eof_observed = True
                logger.warning(
                    "[BEDROCK STREAM NATIVE] Transport EOF without terminal event: model=%s events_received=%d elapsed=%.1fs",
                    bedrock_model_id,
                    event_count,
                    time.monotonic() - stream_start,
                )

            event_queue.put(
                (
                    "done",
                    {
                        "terminal_event_seen": terminal_event_type is not None,
                        "terminal_event_type": terminal_event_type,
                        "transport_eof_observed": transport_eof_observed,
                        "event_count": event_count,
                        "worker_stopped_early": worker_stopped_early,
                        "aws_request_id": aws_request_id,
                        "output_bytes_received": output_bytes_received,
                    },
                )
            )

        except ClientError as e:
            err = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            logger.warning(
                "[BEDROCK STREAM NATIVE] ClientError after %d events, elapsed=%.1fs aws_request_id=%s: %s",
                event_count,
                time.monotonic() - stream_start,
                aws_request_id or "unknown",
                e,
                exc_info=True,
            )
            event_queue.put(
                (
                    "error",
                    {
                        "code": err.get("Code", "ClientError"),
                        "message": err.get("Message", str(e)),
                        "error_class": "client_error",
                        "aws_request_id": aws_request_id,
                        "output_bytes_received": output_bytes_received,
                    },
                )
            )
        except Exception as e:
            error_class = _classify_native_stream_exception(e)
            # When the async consumer asked us to stop (client disconnect or a
            # pre-output retry), it set stop_event before closing the socket from
            # its side. The resulting mid-read failure here is self-inflicted and
            # the consumer is no longer reading this queue, so log it quietly and
            # skip the error tuple instead of emitting a misleading WARNING+traceback.
            if stop_event is not None and stop_event.is_set():
                logger.info(
                    "[BEDROCK STREAM NATIVE] Worker stopped during requested teardown "
                    "after %d events, elapsed=%.1fs aws_request_id=%s error_class=%s: %s",
                    event_count,
                    time.monotonic() - stream_start,
                    aws_request_id or "unknown",
                    error_class,
                    e,
                )
            else:
                logger.warning(
                    "[BEDROCK STREAM NATIVE] Stream error after %d events, elapsed=%.1fs "
                    "aws_request_id=%s error_class=%s: %s",
                    event_count,
                    time.monotonic() - stream_start,
                    aws_request_id or "unknown",
                    error_class,
                    e,
                    exc_info=True,
                )
                event_queue.put(
                    (
                        "error",
                        {
                            "code": "internal_error",
                            "message": str(e),
                            "error_class": error_class,
                            "aws_request_id": aws_request_id,
                            "output_bytes_received": output_bytes_received,
                        },
                    )
                )
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    logger.debug(
                        "[BEDROCK STREAM NATIVE] Failed to close stream for model=%s terminal_event=%s transport_eof_observed=%s",
                        bedrock_model_id,
                        terminal_event_type,
                        transport_eof_observed,
                        exc_info=True,
                    )
            # Clear stream_box so a stale reference cannot be used after the worker exits.
            if stream_box is not None:
                stream_box[0] = None

    def _synthesize_native_finalization_events(
        self,
        *,
        cb_started: set,
        cb_stopped: set,
        buffered_stop_reason: Optional[str],
        buffered_stop_sequence: Optional[str],
        message_delta_emitted: bool,
        message_stop_emitted: bool,
    ):
        """Yield synthetic terminal SSE events to close an incomplete native stream.

        Emits content_block_stop for every still-open block, then message_delta
        (with stop_reason="end_turn" if none was buffered) and message_stop if
        the message was never cleanly terminated.  Called before raising on every
        error path in _consume_native_stream_events so that clients receive a
        well-formed Anthropic SSE stream even when the upstream connection drops.
        """
        for idx in sorted(cb_started - cb_stopped):
            yield self._format_anthropic_sse_event("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })
        if not message_stop_emitted:
            if not message_delta_emitted:
                yield self._format_anthropic_sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": buffered_stop_reason or "end_turn",
                        "stop_sequence": buffered_stop_sequence,
                    },
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                })
            yield self._format_anthropic_sse_event("message_stop", {"type": "message_stop"})

    async def _consume_native_stream_events(
        self,
        request: Any,
        model_id: str,
        event_queue: "queue.Queue[tuple]",
        future: asyncio.Future,
        stop_event: Optional[threading.Event] = None,
        restart_worker: Optional[Any] = None,
        stream_box: Optional[list] = None,
    ) -> AsyncGenerator[str, None]:
        """Consume worker events for Bedrock native Claude streaming.

        ``stream_box`` is a single-element list populated by the worker with
        the open boto3 event-stream object.  On early consumer exit (client
        disconnect, idle timeout, outer stream-wrapper timeout) the ``finally``
        block calls ``stream_box[0].close()`` to immediately unblock a worker
        thread parked inside ``next(stream)``, preventing zombie threads that
        would otherwise hold thread-pool slots for up to ~940 s.
        """
        stream_started_at = time.monotonic()
        last_progress_event_at = stream_started_at
        last_progress_event_type: Optional[str] = None
        progress_event_count = 0
        proxy_ping_count = 0
        saw_first_progress_event = False
        warning_logged = False
        future_done_empty_checks = 0
        future_done_since: Optional[float] = None
        last_ping_sent_at = stream_started_at
        # One retry permitted when the upstream drops before any progress event
        # has been forwarded to the client. Resetting it requires a fresh worker.
        retries_remaining = 1 if (BEDROCK_NATIVE_RETRY_ON_PRE_OUTPUT_DROP and restart_worker is not None) else 0
        # Stream-state tracking for graceful finalization on mid-stream errors.
        _message_start_seen = False
        _cb_started: set = set()
        _cb_stopped: set = set()
        _buffered_stop_reason: Optional[str] = None
        _buffered_stop_sequence: Optional[str] = None
        _message_delta_emitted = False
        _message_stop_emitted = False
        # Drop-diagnostic state, populated from worker queue payloads.
        _aws_request_id: Optional[str] = None
        _output_bytes_received = 0

        initial_timeout, initial_phase = self._get_native_stream_idle_timeout(
            request,
            saw_first_progress_event=False,
        )
        logger.info(
            "[BEDROCK STREAM NATIVE] Start: provider=%s model=%s phase=%s "
            "idle_timeout_seconds=%s thinking_enabled=%s tools_present=%s",
            getattr(self, "full_provider_name", "bedrock"),
            model_id,
            initial_phase,
            initial_timeout,
            self._is_native_stream_thinking_enabled(request),
            bool(getattr(request, "tools", None)),
        )

        try:
            while True:
                try:
                    msg_type, data = event_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    now = time.monotonic()

                    active_timeout, phase = self._get_native_stream_idle_timeout(
                        request,
                        saw_first_progress_event,
                    )
                    idle_seconds = now - last_progress_event_at
                    warning_threshold = active_timeout * _BEDROCK_NATIVE_WARNING_THRESHOLD_RATIO
                    if not warning_logged and idle_seconds >= warning_threshold:
                        logger.warning(
                            "[BEDROCK STREAM NATIVE] Upstream idle warning: model=%s phase=%s "
                            "idle_seconds=%.1f threshold_seconds=%s last_progress_event=%s "
                            "progress_event_count=%s proxy_ping_count=%s",
                            model_id,
                            phase,
                            idle_seconds,
                            active_timeout,
                            last_progress_event_type or "none",
                            progress_event_count,
                            proxy_ping_count,
                        )
                        warning_logged = True

                    if idle_seconds >= active_timeout:
                        logger.warning(
                            "[BEDROCK STREAM NATIVE] Upstream idle timeout: model=%s phase=%s "
                            "idle_seconds=%.1f threshold_seconds=%s last_progress_event=%s "
                            "progress_event_count=%s proxy_ping_count=%s",
                            model_id,
                            phase,
                            idle_seconds,
                            active_timeout,
                            last_progress_event_type or "none",
                            progress_event_count,
                            proxy_ping_count,
                        )
                        self._log_native_stream_drop(
                            request=request,
                            model_id=model_id,
                            aws_request_id=_aws_request_id,
                            error_class="idle_timeout",
                            elapsed_seconds=now - stream_started_at,
                            progress_event_count=progress_event_count,
                            last_progress_event_type=last_progress_event_type,
                            proxy_ping_count=proxy_ping_count,
                            output_bytes_received=_output_bytes_received,
                        )
                        if _message_start_seen:
                            for _sse in self._synthesize_native_finalization_events(
                                cb_started=_cb_started,
                                cb_stopped=_cb_stopped,
                                buffered_stop_reason=_buffered_stop_reason,
                                buffered_stop_sequence=_buffered_stop_sequence,
                                message_delta_emitted=_message_delta_emitted,
                                message_stop_emitted=_message_stop_emitted,
                            ):
                                yield _sse
                        raise BedrockNativeIdleTimeout(
                            model=request.model,
                            phase=phase,
                            idle_seconds=idle_seconds,
                            threshold_seconds=active_timeout,
                            last_progress_event_type=last_progress_event_type,
                            progress_event_count=progress_event_count,
                            proxy_ping_count=proxy_ping_count,
                        )

                    if now - last_ping_sent_at >= self.native_stream_ping_interval:
                        yield "event: ping\ndata: {}\n\n"
                        proxy_ping_count += 1
                        last_ping_sent_at = now

                    # Detect premature worker thread death. We require the queue to
                    # be empty on 2 consecutive loop iterations after future.done()
                    # returns True before declaring a premature EOF. This guards
                    # against the following race: the worker puts the "done" message
                    # into the queue and then the Future is resolved; if future.done()
                    # fired first (before the "done" message was consumed), a single
                    # check could misfire. Two consecutive empty checks are enough
                    # because the queue-put happens in the worker thread before the
                    # concurrent.futures executor marks the Future done, and the
                    # asyncio event loop has always yielded at least once between
                    # the two checks (via asyncio.sleep(0.01)).
                    if future.done():
                        if future_done_since is None:
                            future_done_since = now
                        future_done_empty_checks += 1
                        if (
                            future_done_empty_checks >= _BEDROCK_NATIVE_FUTURE_DONE_EMPTY_CHECKS
                            and (now - future_done_since) >= _BEDROCK_NATIVE_FUTURE_DONE_GRACE_SECONDS
                        ):
                            try:
                                future.result()
                            except Exception as exc:
                                self._log_native_stream_drop(
                                    request=request,
                                    model_id=model_id,
                                    aws_request_id=_aws_request_id,
                                    error_class=_classify_native_stream_exception(exc),
                                    elapsed_seconds=now - stream_started_at,
                                    progress_event_count=progress_event_count,
                                    last_progress_event_type=last_progress_event_type,
                                    proxy_ping_count=proxy_ping_count,
                                    output_bytes_received=_output_bytes_received,
                                )
                                if _message_start_seen:
                                    for _sse in self._synthesize_native_finalization_events(
                                        cb_started=_cb_started,
                                        cb_stopped=_cb_stopped,
                                        buffered_stop_reason=_buffered_stop_reason,
                                        buffered_stop_sequence=_buffered_stop_sequence,
                                        message_delta_emitted=_message_delta_emitted,
                                        message_stop_emitted=_message_stop_emitted,
                                    ):
                                        yield _sse
                                raise BedrockNativeProviderError(
                                    "internal_error",
                                    str(exc),
                                ) from exc
                            logger.warning(
                                "[BEDROCK STREAM NATIVE] Future resolved without terminal queue event: "
                                "model=%s empty_checks=%s future_done_elapsed=%.3fs",
                                model_id,
                                future_done_empty_checks,
                                now - future_done_since,
                            )
                            self._log_native_stream_drop(
                                request=request,
                                model_id=model_id,
                                aws_request_id=_aws_request_id,
                                error_class="premature_eof",
                                elapsed_seconds=now - stream_started_at,
                                progress_event_count=progress_event_count,
                                last_progress_event_type=last_progress_event_type,
                                proxy_ping_count=proxy_ping_count,
                                output_bytes_received=_output_bytes_received,
                            )
                            if _message_start_seen:
                                for _sse in self._synthesize_native_finalization_events(
                                    cb_started=_cb_started,
                                    cb_stopped=_cb_stopped,
                                    buffered_stop_reason=_buffered_stop_reason,
                                    buffered_stop_sequence=_buffered_stop_sequence,
                                    message_delta_emitted=_message_delta_emitted,
                                    message_stop_emitted=_message_stop_emitted,
                                ):
                                    yield _sse
                            raise BedrockNativePrematureEOF(
                                model=request.model,
                                transport_eof_observed=False,
                                last_progress_event_type=last_progress_event_type,
                                progress_event_count=progress_event_count,
                                proxy_ping_count=proxy_ping_count,
                            )
                    else:
                        future_done_empty_checks = 0
                        future_done_since = None
                    continue

                future_done_empty_checks = 0
                future_done_since = None

                if msg_type == "upstream_event":
                    event_type = data.get("event_type", "unknown")
                    yield data.get("sse", "")

                    event_data_fwd = data.get("event_data") or {}
                    if event_type == "message_start":
                        _message_start_seen = True
                    elif event_type == "content_block_start":
                        idx = event_data_fwd.get("index")
                        if idx is not None:
                            _cb_started.add(idx)
                    elif event_type == "content_block_stop":
                        idx = event_data_fwd.get("index")
                        if idx is not None:
                            _cb_stopped.add(idx)
                    elif event_type == "message_delta":
                        _message_delta_emitted = True
                        delta = event_data_fwd.get("delta") or {}
                        _buffered_stop_reason = delta.get("stop_reason") or _buffered_stop_reason
                        _buffered_stop_sequence = delta.get("stop_sequence") or _buffered_stop_sequence
                    elif event_type == "message_stop":
                        _message_stop_emitted = True

                    if event_type in _BEDROCK_NATIVE_PROGRESS_EVENT_TYPES:
                        last_progress_event_at = time.monotonic()
                        last_progress_event_type = event_type
                        progress_event_count += 1
                        warning_logged = False

                        if not saw_first_progress_event:
                            saw_first_progress_event = True
                            active_timeout, phase = self._get_native_stream_idle_timeout(
                                request,
                                saw_first_progress_event,
                            )
                            logger.info(
                                "[BEDROCK STREAM NATIVE] First upstream progress event: model=%s "
                                "event_type=%s phase=%s next_idle_threshold_seconds=%s",
                                model_id,
                                event_type,
                                phase,
                                active_timeout,
                            )
                    continue

                if msg_type == "error":
                    _aws_request_id = data.get("aws_request_id") or _aws_request_id
                    _output_bytes_received = max(
                        _output_bytes_received,
                        int(data.get("output_bytes_received") or 0),
                    )
                    logger.warning(
                        "[BEDROCK STREAM NATIVE] Worker reported provider error: provider=%s "
                        "model=%s last_progress_event=%s progress_event_count=%s "
                        "proxy_ping_count=%s code=%s",
                        getattr(self, "full_provider_name", "bedrock"),
                        model_id,
                        last_progress_event_type or "none",
                        progress_event_count,
                        proxy_ping_count,
                        data.get("code", "internal_error"),
                    )
                    self._log_native_stream_drop(
                        request=request,
                        model_id=model_id,
                        aws_request_id=_aws_request_id,
                        error_class=data.get("error_class") or "other",
                        elapsed_seconds=time.monotonic() - stream_started_at,
                        progress_event_count=progress_event_count,
                        last_progress_event_type=last_progress_event_type,
                        proxy_ping_count=proxy_ping_count,
                        output_bytes_received=_output_bytes_received,
                    )
                    _NON_RETRYABLE_CODES = {
                        "ValidationException",
                        "AccessDeniedException",
                        "ResourceNotFoundException",
                        "ModelNotReadyException",
                    }
                    if (
                        progress_event_count == 0
                        and retries_remaining > 0
                        and restart_worker is not None
                        and data.get("code") not in _NON_RETRYABLE_CODES
                    ):
                        retries_remaining -= 1
                        logger.warning(
                            "[BEDROCK STREAM NATIVE] Retrying after pre-output drop: model=%s "
                            "error_class=%s aws_request_id=%s",
                            model_id,
                            data.get("error_class") or "other",
                            _aws_request_id or "unknown",
                        )
                        if stop_event is not None and not stop_event.is_set():
                            stop_event.set()
                        # Close any still-open stream on the old worker so its
                        # thread exits without waiting for the socket timeout.
                        _close_stream_box(stream_box)
                        event_queue, future, stop_event, stream_box = restart_worker()
                        # Reset per-attempt diagnostic state; client has seen
                        # at most proxy pings, so any synthesis state is moot.
                        stream_started_at = time.monotonic()
                        last_progress_event_at = stream_started_at
                        last_ping_sent_at = stream_started_at
                        future_done_empty_checks = 0
                        future_done_since = None
                        warning_logged = False
                        _aws_request_id = None
                        _output_bytes_received = 0
                        continue
                    if _message_start_seen:
                        for _sse in self._synthesize_native_finalization_events(
                            cb_started=_cb_started,
                            cb_stopped=_cb_stopped,
                            buffered_stop_reason=_buffered_stop_reason,
                            buffered_stop_sequence=_buffered_stop_sequence,
                            message_delta_emitted=_message_delta_emitted,
                            message_stop_emitted=_message_stop_emitted,
                        ):
                            yield _sse
                    raise BedrockNativeProviderError(
                        data.get("code", "internal_error"),
                        data.get("message", "Bedrock native streaming failed"),
                    )

                if msg_type == "done":
                    _aws_request_id = data.get("aws_request_id") or _aws_request_id
                    _output_bytes_received = max(
                        _output_bytes_received,
                        int(data.get("output_bytes_received") or 0),
                    )
                    terminal_event_seen = bool(data.get("terminal_event_seen"))
                    terminal_event_type = data.get("terminal_event_type")
                    transport_eof_observed = bool(data.get("transport_eof_observed"))
                    event_count = int(data.get("event_count", 0))
                    worker_stopped_early = bool(data.get("worker_stopped_early"))
                    if terminal_event_seen or worker_stopped_early:
                        logger.info(
                            "[BEDROCK STREAM NATIVE] Completed: model=%s terminal_event_type=%s "
                            "worker_stopped_early=%s event_count=%s progress_event_count=%s "
                            "proxy_ping_count=%s duration_seconds=%.2f aws_request_id=%s "
                            "output_bytes_received=%d",
                            model_id,
                            terminal_event_type,
                            worker_stopped_early,
                            event_count,
                            progress_event_count,
                            proxy_ping_count,
                            time.monotonic() - stream_started_at,
                            _aws_request_id or "unknown",
                            _output_bytes_received,
                        )
                        break

                    logger.warning(
                        "[BEDROCK STREAM NATIVE] Premature EOF: provider=%s model=%s "
                        "transport_eof_observed=%s last_progress_event=%s "
                        "progress_event_count=%s proxy_ping_count=%s event_count=%s",
                        getattr(self, "full_provider_name", "bedrock"),
                        model_id,
                        transport_eof_observed,
                        last_progress_event_type or "none",
                        progress_event_count,
                        proxy_ping_count,
                        event_count,
                    )
                    self._log_native_stream_drop(
                        request=request,
                        model_id=model_id,
                        aws_request_id=_aws_request_id,
                        error_class="premature_eof",
                        elapsed_seconds=time.monotonic() - stream_started_at,
                        progress_event_count=progress_event_count,
                        last_progress_event_type=last_progress_event_type,
                        proxy_ping_count=proxy_ping_count,
                        output_bytes_received=_output_bytes_received,
                    )
                    if _message_start_seen:
                        for _sse in self._synthesize_native_finalization_events(
                            cb_started=_cb_started,
                            cb_stopped=_cb_stopped,
                            buffered_stop_reason=_buffered_stop_reason,
                            buffered_stop_sequence=_buffered_stop_sequence,
                            message_delta_emitted=_message_delta_emitted,
                            message_stop_emitted=_message_stop_emitted,
                        ):
                            yield _sse
                    raise BedrockNativePrematureEOF(
                        model=request.model,
                        transport_eof_observed=transport_eof_observed,
                        last_progress_event_type=last_progress_event_type,
                        progress_event_count=progress_event_count,
                        proxy_ping_count=proxy_ping_count,
                    )

                logger.warning(
                    "[BEDROCK STREAM NATIVE] Ignoring unknown worker queue message type=%s for model=%s",
                    msg_type,
                    model_id,
                )
        finally:
            if stop_event is not None and not stop_event.is_set() and not future.done():
                stop_event.set()
                logger.debug(
                    "[BEDROCK STREAM NATIVE] Signalled worker stop for model=%s future_done=%s",
                    model_id,
                    future.done(),
                )
            # Close the boto3 event-stream (if still open) so the worker thread
            # unblocks out of next(stream) immediately instead of waiting up to
            # ~940 s for the socket read_timeout.  Safe to call from the async
            # side because boto3's EventStream.close() acquires no Python-side
            # lock that the worker holds; it only closes the underlying socket.
            _close_stream_box(stream_box)

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
                reasoning_text = reasoning.get("reasoningText", {})
                anthropic_thinking = {
                    "type": "thinking",
                    "thinking": reasoning_text.get("text", ""),
                }
                signature = reasoning_text.get("signature") or reasoning.get("signature")
                if signature:
                    anthropic_thinking["signature"] = signature
                anthropic_content.append(anthropic_thinking)
        return anthropic_content

    def _format_anthropic_sse_event(self, event_type: str, data: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"

    async def anthropic_messages(self, request, anthropic_beta: Optional[str] = None) -> Any:
        """Handle Anthropic Messages API request.

        Routes Claude models through the InvokeModel API (native Anthropic JSON)
        and non-Claude models through the Converse API.
        """
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)

        # ── Native InvokeModel path for Claude models ──────────────────────
        if self._is_claude_model(model_id):
            native_request = self._convert_to_anthropic_native_request(request, anthropic_beta)
            self._apply_native_cache_ttl(native_request)
            self._strip_native_cache_scope(native_request)

            if self.debug:
                logger.info(
                    "[BEDROCK NATIVE] InvokeModel request: model=%s, msgs=%d, tools=%s, thinking=%s",
                    model_id,
                    len(native_request.get("messages", [])),
                    bool(native_request.get("tools")),
                    bool(native_request.get("thinking")),
                )

            def _invoke() -> Dict[str, Any]:
                resp = self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(native_request),
                )
                return json.loads(resp["body"].read())

            semaphore = _get_native_semaphore()
            async with semaphore:
                loop = asyncio.get_event_loop()
                try:
                    response_body = await loop.run_in_executor(
                        _get_native_executor(), _invoke
                    )
                except ClientError as e:
                    err = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
                    code = err.get("Code", "ClientError")
                    msg = err.get("Message", str(e))
                    mapped = _map_bedrock_error(code, msg)
                    from fastapi import HTTPException
                    raise HTTPException(status_code=mapped["status"], detail=mapped["body"])

            return self._convert_native_response(response_body, request.model)

        # ── Converse path for non-Claude models ────────────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
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

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": self._convert_bedrock_content_to_anthropic(output_message.get("content", [])),
            "model": request.model,
            "stop_reason": self._convert_bedrock_stop_reason(response.get("stopReason")),
            "stop_sequence": None,
            "usage": usage_data,
        }

    async def anthropic_messages_stream(self, request, anthropic_beta: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic Messages API request.

        Routes Claude models through InvokeModelWithResponseStream (native
        Anthropic SSE events via a queue+thread worker), and non-Claude models
        through the Converse Stream API.
        """
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)

        # ── Native InvokeModelWithResponseStream for Claude models ─────────
        if self._is_claude_model(model_id):
            native_request = self._convert_to_anthropic_native_request(request, anthropic_beta)
            self._apply_native_cache_ttl(native_request)
            self._strip_native_cache_scope(native_request)

            if self.debug:
                logger.info(
                    "[BEDROCK STREAM NATIVE] model=%s, msgs=%d, tools=%s, thinking=%s",
                    model_id,
                    len(native_request.get("messages", [])),
                    bool(native_request.get("tools")),
                    bool(native_request.get("thinking")),
                )

            # Use the dedicated streaming semaphore (separate from the non-stream
            # InvokeModel semaphore) so long-running stream workers cannot starve
            # synchronous calls and vice versa.
            semaphore = _get_native_stream_semaphore()
            semaphore_wait_started_at = time.monotonic()
            acquired_semaphore = False

            try:
                await semaphore.acquire()
                acquired_semaphore = True
                semaphore_wait_seconds = time.monotonic() - semaphore_wait_started_at
                if semaphore_wait_seconds >= _BEDROCK_NATIVE_SEMAPHORE_WAIT_WARNING_SECONDS:
                    logger.warning(
                        "[BEDROCK STREAM NATIVE] Waited %.2fs for native stream semaphore: model=%s",
                        semaphore_wait_seconds,
                        model_id,
                    )
                else:
                    logger.debug(
                        "[BEDROCK STREAM NATIVE] Acquired native stream semaphore in %.2fs: model=%s",
                        semaphore_wait_seconds,
                        model_id,
                    )

                event_queue, future, stop_event, stream_box = self._start_native_stream_worker(
                    model_id, native_request,
                )

                async for sse_chunk in self._consume_native_stream_events(
                    request,
                    model_id,
                    event_queue,
                    future,
                    stop_event,
                    restart_worker=lambda: self._start_native_stream_worker(model_id, native_request),
                    stream_box=stream_box,
                ):
                    yield sse_chunk
            finally:
                if acquired_semaphore:
                    semaphore.release()
            return

        # ── Converse Stream path for non-Claude models ─────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
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

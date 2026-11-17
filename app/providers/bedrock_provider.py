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
import concurrent.futures
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from typing import List, Dict, Any, AsyncGenerator, Callable, Literal, Optional, Tuple
from collections import defaultdict
from urllib.parse import urlparse
import boto3
import numpy as np
import urllib3
from botocore.config import Config
from botocore.exceptions import ClientError
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.model_alias import echo_model_name


# ---------------------------------------------------------------------------
#  Module-level concurrency primitive for the Converse path.
#  The Claude (native) path now runs on the AsyncAnthropicBedrock SDK, which
#  owns its own transport/concurrency; only the Converse / converse_stream
#  path (OpenAI chat + non-Claude Anthropic) still needs a bound, since its
#  blocking boto3 calls run on threads and are otherwise ungoverned. A stream
#  holds its permit until its reader thread exits (see _ConverseStreamReader).
# ---------------------------------------------------------------------------
_CONVERSE_SEMAPHORE_LIMIT = 15

# boto3.client() builds on the shared default session, whose client
# construction is not thread-safe, and Bedrock providers are built on worker
# threads (ProviderManager.build_provider). Build one client at a time.
_BOTO_CLIENT_LOCK = threading.Lock()

# Converse stream reader tuning: undelivered events buffered per stream before
# the reader stops reading (backpressure), how often a reader blocked on that
# buffer re-checks for a stop request, and how long close() waits for the
# reader thread to exit once its socket has been shut down.
_STREAM_BUFFER_EVENTS = 64
_STREAM_STOP_POLL_SECONDS = 0.25
_STREAM_CLOSE_WAIT_SECONDS = 2.0


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


def _is_access_denied(error: Exception) -> bool:
    """True if *error* is an AWS AccessDenied ClientError (missing IAM permission)."""
    if not isinstance(error, ClientError):
        return False
    return error.response.get("Error", {}).get("Code") in ("AccessDeniedException", "AccessDenied")


# Optional inferenceConfig fields a Converse model may refuse with a
# ValidationException ("This model doesn't support the stopSequences field.
# Remove stopSequences and try again."), keyed by the lower-cased name the
# message is matched on. maxTokens is required and never dropped.
_CONVERSE_OPTIONAL_FIELDS = {
    "stopsequences": "stopSequences",
    "temperature": "temperature",
    "topp": "topP",
}
# Key for remembered Converse field rejections (see known_rejected_params).
_CONVERSE_TRANSPORT = "converse"


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


from app.providers.base import AnthropicRequestMetadata, BaseProvider, ProviderHTTPError
from app.anthropic_models import (
    ANTHROPIC_SDK_TIMEOUT_SECONDS,
    _extract_system_messages_from_messages,
    _merge_system_fields,
    is_anthropic_terminal_stream_event,
)
from app.model_capabilities import (
    CONVERSE_SPELLINGS,
    SURFACE_CONVERSE,
    SURFACE_NATIVE,
    THINKING_ADAPTIVE,
    is_claude_at_least,
    normalize_thinking,
    scrub as scrub_unsupported_params,
    thinking_style,
)
from app.providers.anthropic_compatible import (
    # Bedrock native Anthropic streams share the same bounded post-terminal
    # drain policy and event serialization as the other Anthropic-SDK paths.
    stream_anthropic_sdk_events,
)
from app.providers.anthropic_adapter import (
    MAX_PARAM_RETRIES,
    known_rejected_params,
    remember_rejected_param,
    unsupported_param,
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


def _event_stream_socket(stream: Any) -> Optional[socket.socket]:
    """The socket under a botocore EventStream, or None if it can't be found.

    botocore wraps the urllib3 response as ``EventStream._raw_stream``; urllib3
    keeps the connection (and its socket) on the response while the body is
    being streamed.
    """
    raw = getattr(stream, "_raw_stream", None)
    if raw is None:
        return None
    for path in (("_connection", "sock"), ("_fp", "fp", "raw", "_sock")):
        obj = raw
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, socket.socket):
            return obj
    return None


def _interrupt_event_stream(stream: Any) -> None:
    """Make a read blocked on ``stream``'s socket return now.

    ``shutdown()`` (unlike ``close()``) wakes a thread blocked in ``recv`` on
    the same socket. The base-class method is called directly so an SSL
    socket's state is left for its owning thread to tear down.
    """
    sock = _event_stream_socket(stream)
    if sock is None:
        return
    try:
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass  # already closed or never connected


class _ConverseStreamReader:
    """Open and read one Bedrock ``converse_stream`` on a dedicated thread.

    The blocking boto3 reads never touch the shared AnyIO threadpool, and a
    cancelled consumer can actually stop them: ``close()`` shuts the socket
    down, which makes a blocked read return at once, and the reader thread then
    closes the stream itself, so no other thread ever closes it mid-read.

    ``on_exit`` runs on the event loop once the thread has exited. Callers pass
    the Converse semaphore's ``release`` there, so the concurrency bound covers
    the whole stream (open, every read and cleanup), not just opening it.

    Delivered-but-unconsumed events are capped at ``max_buffered``; past that
    the reader stops reading until the consumer catches up.
    """

    _OPENED = "opened"
    _EVENT = "event"
    _DONE = "done"
    _ERROR = "error"

    def __init__(
        self,
        open_stream: Callable[[], Any],
        *,
        on_exit: Optional[Callable[[], None]] = None,
        max_buffered: int = _STREAM_BUFFER_EVENTS,
    ):
        self._open_stream = open_stream
        self._on_exit = on_exit
        self._loop = asyncio.get_running_loop()
        self._items: asyncio.Queue = asyncio.Queue()
        self._slots = threading.Semaphore(max_buffered)
        self._stop = threading.Event()
        self._stream: Any = None
        self._stream_lock = threading.Lock()
        self._exited = self._loop.create_future()
        self._finished = False
        self._started = False
        self._thread = threading.Thread(
            target=self._run, name="bedrock-converse-stream", daemon=True
        )

    # --- reader thread ---------------------------------------------------

    def _post(self, kind: str, payload: Any = None) -> None:
        try:
            self._loop.call_soon_threadsafe(self._items.put_nowait, (kind, payload))
        except RuntimeError:
            pass  # loop already closed (shutdown); nobody is listening

    def _wait_for_slot(self) -> bool:
        while not self._slots.acquire(timeout=_STREAM_STOP_POLL_SECONDS):
            if self._stop.is_set():
                return False
        return not self._stop.is_set()

    def _run(self) -> None:
        stream = None
        try:
            try:
                stream = self._open_stream()
            except BaseException as exc:
                self._post(self._ERROR, exc)
                return
            with self._stream_lock:
                self._stream = stream
            if self._stop.is_set():  # cancelled while the stream was opening
                return
            self._post(self._OPENED)
            for event in stream:
                if not self._wait_for_slot():
                    return
                self._post(self._EVENT, event)
            self._post(self._DONE)
        except BaseException as exc:
            # After a stop request this is the read our own shutdown() broke.
            if not self._stop.is_set():
                self._post(self._ERROR, exc)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.debug("[BEDROCK STREAM] Error closing event stream: %s", exc)
            try:
                self._loop.call_soon_threadsafe(self._finish)
            except RuntimeError:
                pass

    # --- event loop ------------------------------------------------------

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._exited.done():
            self._exited.set_result(None)
        if self._on_exit is not None:
            self._on_exit()

    def start(self) -> None:
        self._thread.start()
        self._started = True

    async def wait_opened(self) -> None:
        """Wait until the stream is open; raises the error that opening raised."""
        kind, payload = await self._items.get()
        if kind == self._OPENED:
            return
        if kind == self._ERROR:
            raise payload
        raise RuntimeError(f"Unexpected stream state while opening: {kind}")

    async def next_event(self) -> Optional[Dict[str, Any]]:
        """The next event, or None at the end; raises a mid-stream read error."""
        while True:
            kind, payload = await self._items.get()
            if kind == self._EVENT:
                self._slots.release()
                return payload
            if kind == self._DONE:
                return None
            if kind == self._ERROR:
                raise payload
            # _OPENED when the caller skipped wait_opened(): nothing to deliver.

    async def close(self) -> None:
        """Stop the reader: interrupt any blocked read, then wait for the thread.

        Safe to call more than once. The wait is bounded; a reader that has
        not exited by then still releases its permit when it does.
        """
        if not self._started:
            self._finish()
            return
        self._stop.set()
        self._slots.release()  # wake a reader waiting for buffer space
        with self._stream_lock:
            stream = self._stream
        if stream is not None:
            _interrupt_event_stream(stream)
        if self._exited.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._exited), _STREAM_CLOSE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "[BEDROCK STREAM] Reader thread still running %ss after close; "
                "it exits (and releases its Converse permit) when its read returns",
                _STREAM_CLOSE_WAIT_SECONDS,
            )


_SUPPORTED_IMAGE_FORMATS = {
    "png": "png",
    "jpeg": "jpeg",
    "jpg": "jpeg",
    "pjpeg": "jpeg",
    "gif": "gif",
    "webp": "webp",
}


def _sniff_image_format(data: bytes) -> Optional[str]:
    """The Converse image format of ``data`` from its magic bytes, if recognised."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _converse_image_format(data: bytes, declared_type: Optional[str]) -> str:
    """Pick the Converse ``image.format`` for ``data``.

    The bytes decide when they're recognisable: a declared type can be wrong
    (a ``.png`` URL served as ``image/jpeg``) and Bedrock rejects a format that
    doesn't match the data. Otherwise the declared media type is normalised
    (parameters dropped, ``jpg`` -> ``jpeg``). Anything else is rejected rather
    than sent as a format Converse doesn't accept.
    """
    sniffed = _sniff_image_format(data)
    if sniffed:
        return sniffed
    subtype = (declared_type or "").split(";", 1)[0].strip().lower()
    if subtype.startswith("image/"):
        subtype = subtype[len("image/"):]
    fmt = _SUPPORTED_IMAGE_FORMATS.get(subtype)
    if fmt is None:
        raise ValueError(
            f"Unsupported image format {declared_type!r}; Bedrock accepts png, jpeg, gif and webp"
        )
    return fmt


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

    # Bedrock Converse rejects a toolConfig whose tools list is empty (it
    # requires >= 1 entry). When message history carries tool blocks but the
    # request declares no usable tools, we inject this harmless placeholder so
    # the required toolConfig stays valid.
    _PLACEHOLDER_TOOL_SPEC = {
        "toolSpec": {
            "name": "noop",
            "description": "Placeholder tool. Do not call.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
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
        
        # Data-plane (bedrock-runtime) budgets. The read timeout stays long:
        # a non-streaming Converse call only returns once generation finishes.
        # Connect timeout and attempt count are tighter (previously 60s and 9
        # attempts), since every attempt can wait out the full read timeout.
        # 3 attempts in total (first try included) matches the native Claude
        # path (AsyncAnthropicBedrock max_retries=2). Streams are additionally
        # torn down by the route's TTFT/per-chunk budgets via
        # _ConverseStreamReader.close().
        boto_config = Config(
            connect_timeout=_get_positive_int_env("BEDROCK_CONNECT_TIMEOUT_SECONDS", 10),
            read_timeout=_get_positive_int_env("BEDROCK_READ_TIMEOUT_SECONDS", 900),
            retries={
                'total_max_attempts': _get_positive_int_env("BEDROCK_MAX_ATTEMPTS", 3),
                'mode': 'adaptive',
            },
            max_pool_connections=50
        )
        # Control-plane (model listing) budgets: these are small, fast calls,
        # and an Anthropic cache-miss preflight can wait on them, so they get
        # short timeouts instead of the runtime's generation-sized ones.
        control_config = Config(
            connect_timeout=_get_positive_int_env("BEDROCK_CONTROL_CONNECT_TIMEOUT_SECONDS", 5),
            read_timeout=_get_positive_int_env("BEDROCK_CONTROL_READ_TIMEOUT_SECONDS", 30),
            retries={
                'total_max_attempts': _get_positive_int_env("BEDROCK_CONTROL_MAX_ATTEMPTS", 3),
                'mode': 'standard',
            },
            max_pool_connections=10,
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

        # Client for listing models
        control_client_kwargs = {
            'service_name': 'bedrock',
            'region_name': self.aws_region,
            'config': control_config
        }

        if self.aws_access_key and self.aws_secret_key:
            control_client_kwargs['aws_access_key_id'] = self.aws_access_key
            control_client_kwargs['aws_secret_access_key'] = self.aws_secret_key

        # Client creation loads service models and, without explicit keys,
        # resolves the credential chain (possibly over the network), so the
        # provider is constructed off the event loop (see
        # ProviderManager.build_provider). Serialized: see _BOTO_CLIENT_LOCK.
        with _BOTO_CLIENT_LOCK:
            self.bedrock_runtime = boto3.client(**client_kwargs)
            self.bedrock_client = boto3.client(**control_client_kwargs)
        
        # Cache for model list
        self.bedrock_model_list = {}

        self._anthropic_client = None
        # Set when the native client was built without explicit keys: the
        # credentials then come from the default chain and are resolved (and
        # refreshed) off the event loop by _ensure_anthropic_client().
        self._anthropic_uses_default_chain = False
        self._chain_credentials = None
        self._chain_credentials_lock: Optional[asyncio.Lock] = None

        # Per-instance Converse concurrency bound, created lazily on the running
        # event loop (a module-level Semaphore would be shared across accounts/
        # regions and pinned to whichever loop first touched it).
        self._converse_semaphore: Optional[asyncio.Semaphore] = None
        # Single-flight guard so concurrent model-list refreshes don't all run
        # the slow paginated control-plane sweep at once (losers serve stale).
        self._model_list_refresh_lock = threading.Lock()

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

    def _get_converse_semaphore(self) -> asyncio.Semaphore:
        """Get or create this instance's Converse / converse_stream bound.

        The OpenAI chat path and the non-Claude Anthropic path make blocking
        converse/converse_stream calls on threads with no Bedrock-side
        concurrency bound. A non-streaming call holds a permit for the whole
        call; a stream holds one until its reader thread has exited. The
        Semaphore is created lazily so it
        binds to the running event loop, and is per-instance so one busy AWS
        account/region can't starve another. Limit is configurable via
        BEDROCK_CONVERSE_SEMAPHORE_LIMIT.
        """
        if self._converse_semaphore is None:
            limit = _get_positive_int_env(
                "BEDROCK_CONVERSE_SEMAPHORE_LIMIT", _CONVERSE_SEMAPHORE_LIMIT
            )
            self._converse_semaphore = asyncio.Semaphore(limit)
            logger.info("[BEDROCK] Created converse semaphore with limit %d", limit)
        return self._converse_semaphore

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

        # Single-flight: only one refresh runs the slow paginated control-plane
        # sweep at a time. On the non-force path, losers return immediately and
        # serve the existing (possibly stale) list instead of stampeding the
        # control plane. force=True (explicit refresh) waits for the in-flight
        # refresh rather than skipping.
        acquired = self._model_list_refresh_lock.acquire(blocking=force)
        if not acquired:
            if self.bedrock_model_list:
                # Serve the existing (possibly stale) list.
                return
            # Cold start: no list has ever been populated, so returning early
            # would leave callers with an empty model map. Wait for the
            # in-flight refresh instead. Blocking here is loop-safe — every
            # caller runs this method via run_in_threadpool.
            self._model_list_refresh_lock.acquire()
        try:
            # Re-check the TTL now that we hold the lock — another thread may
            # have refreshed while we were waiting.
            if not force:
                last = getattr(self, "_model_list_fetched_at", None)
                if last is not None and self.bedrock_model_list and (time.monotonic() - last) < self._MODEL_LIST_TTL_SECONDS:
                    return
            self._refresh_model_list_locked()
        finally:
            self._model_list_refresh_lock.release()

    def _refresh_model_list_locked(self):
        """Run the actual (blocking) control-plane model-list sweep.

        Must be called while holding ``self._model_list_refresh_lock``.
        """
        try:
            model_list = {}

            # The three listings are independent, so run them concurrently: a
            # cold refresh then costs the slowest call, not the sum of all three.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="bedrock-model-list"
            ) as pool:
                system_profiles_future = pool.submit(self._list_system_inference_profiles)
                app_profiles_future = pool.submit(self._list_application_inference_profiles)
                # List foundation models - removed byOutputModality filter
                foundation_future = pool.submit(self.bedrock_client.list_foundation_models)
                response = foundation_future.result()
                profile_list = system_profiles_future.result()
                app_profiles_by_model = app_profiles_future.result()

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
                
                # Add application inference profiles. An ARN says nothing about
                # the model behind it, so record that for capability checks.
                if model_id in app_profiles_by_model:
                    for profile_arn in app_profiles_by_model[model_id]:
                        model_list[profile_arn] = {
                            "modalities": input_modalities,
                            "base_model_id": model_id,
                        }

            if not model_list:
                # Raise rather than return []: an empty fetch tells auto-sync to
                # clear this provider's saved models.
                raise RuntimeError("Bedrock returned no usable models")

            # Atomic swap — readers see old or new map whole, never partial.
            self.bedrock_model_list = model_list
            self._model_list_fetched_at = time.monotonic()
            logger.info(f"Loaded {len(model_list)} Bedrock models")

        except Exception as e:
            logger.error(f"Error listing Bedrock models: {e}")
            # Re-raise so callers record the sync as failed instead of treating a
            # fallback/partial list as success. The previously-good list (if any)
            # is left in place for the request-path helpers, and
            # _model_list_fetched_at is not bumped, so the next call retries.
            raise

    def _list_system_inference_profiles(self) -> List[str]:
        """Cross-region (system-defined) inference profile ids."""
        profile_list: List[str] = []
        if not self.enable_cross_region:
            return profile_list
        try:
            paginator = self.bedrock_client.get_paginator('list_inference_profiles')
            for page in paginator.paginate(maxResults=1000, typeEquals="SYSTEM_DEFINED"):
                profile_list.extend([p["inferenceProfileId"] for p in page["inferenceProfileSummaries"]])
        except Exception as e:
            # A missing IAM permission is a stable config choice and yields the
            # same list every time. Anything else (e.g. endpoint unreachable)
            # would silently drop every cross-region model, so fail the refresh.
            if not _is_access_denied(e):
                raise
            logger.warning(f"Not permitted to list cross-region inference profiles: {e}")
        return profile_list

    def _list_application_inference_profiles(self) -> Dict[str, set]:
        """Application inference profile ARNs keyed by the foundation model they serve."""
        app_profiles_by_model: Dict[str, set] = defaultdict(set)
        if not self.enable_app_profiles:
            return app_profiles_by_model
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
            if not _is_access_denied(e):
                raise
            logger.warning(f"Not permitted to list application inference profiles: {e}")
        return app_profiles_by_model

    def _discovery_signature(self) -> tuple:
        """What determines this instance's model list (account, region, listing flags)."""
        return (
            getattr(self, "aws_region", None),
            getattr(self, "aws_access_key", None),
            getattr(self, "aws_secret_key", None),
            getattr(self, "enable_cross_region", None),
            getattr(self, "enable_app_profiles", None),
        )

    def adopt_model_discovery(self, other: Any) -> bool:
        """Take over ``other``'s already-fetched model list when it is equivalent.

        A provider save rebuilds every live instance (empty model list) and runs
        discovery on a separate throwaway instance, so without this the live
        instances start cold: image-modality checks fail and the next request
        that needs the list pays for a full control-plane sweep. Only adopted
        from an instance configured for the same account, region and listing
        flags. Returns True if the list was adopted.
        """
        if other is self or not isinstance(other, BedrockProvider):
            return False
        model_list = getattr(other, "bedrock_model_list", None)
        fetched_at = getattr(other, "_model_list_fetched_at", None)
        if not model_list or fetched_at is None:
            return False
        if other._discovery_signature() != self._discovery_signature():
            return False
        own_fetched_at = getattr(self, "_model_list_fetched_at", None)
        if own_fetched_at is not None and own_fetched_at >= fetched_at:
            return False
        # Same single-reference swap as a refresh (readers never see a mix).
        self.bedrock_model_list = model_list
        self._model_list_fetched_at = fetched_at
        return True

    def _capability_model_id(self, model_id: str) -> str:
        """The foundation model id to judge ``model_id``'s capabilities by.

        Application inference profile ARNs don't name their model, so they
        resolve through the base model recorded at discovery; any other id is
        its own answer.
        """
        entry = self.bedrock_model_list.get(model_id) if getattr(self, "bedrock_model_list", None) else None
        if isinstance(entry, dict) and entry.get("base_model_id"):
            return entry["base_model_id"]
        return model_id

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
        """Extract system prompts from messages.

        Handles both plain-string system content and OpenAI's array form
        (``[{"type": "text", "text": ...}]``); the latter was previously dropped.
        """
        system_prompts = []
        for message in messages:
            if message.role != "system":
                continue
            if isinstance(message.content, str):
                if message.content:
                    system_prompts.append({"text": message.content})
            elif isinstance(message.content, list):
                for part in message.content:
                    text = None
                    if isinstance(part, TextContent):
                        text = part.text
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                    elif hasattr(part, "text"):
                        text = part.text
                    if text and text.strip():
                        system_prompts.append({"text": text})
        return system_prompts

    # Image download guard rails (SSRF + resource exhaustion).
    _IMAGE_DOWNLOAD_TIMEOUT = 10  # seconds
    _IMAGE_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

    _DATA_URL_RE = re.compile(r"^data:(?P<type>[^;,]*)(?P<params>[^,]*?);base64,", re.IGNORECASE)

    @staticmethod
    def _resolve_safe_image_address(host: str, port: int) -> str:
        """Resolve ``host`` once; return an address to connect to, or raise.

        Every address the name resolves to must be public: private, loopback,
        link-local, reserved and multicast ranges (e.g. the cloud metadata
        endpoint 169.254.169.254) are refused. The caller connects to the
        returned address itself, so a second lookup can't hand back a
        different (internal) one (DNS rebinding).
        """
        try:
            addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as e:
            raise ValueError(f"Could not resolve image host: {host}") from e

        addresses = []
        for info in addrinfos:
            ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
            if (
                not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_reserved or ip.is_multicast
            ):
                raise ValueError(f"Image URL resolves to a disallowed address: {ip}")
            addresses.append(str(ip))
        if not addresses:
            raise ValueError(f"Could not resolve image host: {host}")
        return addresses[0]

    def _decode_base64_image(self, data: str) -> bytes:
        """Decode base64 image data, enforcing the same size cap as downloads."""
        # Base64 is 4 chars per 3 bytes; check before decoding so an oversized
        # payload is refused without materialising it a second time.
        if len(data) * 3 // 4 > self._IMAGE_MAX_BYTES + 3:
            raise ValueError(f"Image exceeds maximum size of {self._IMAGE_MAX_BYTES} bytes")
        image_bytes = base64.b64decode(data)
        if len(image_bytes) > self._IMAGE_MAX_BYTES:
            raise ValueError(f"Image exceeds maximum size of {self._IMAGE_MAX_BYTES} bytes")
        return image_bytes

    def _fetch_image_url(self, image_url: str) -> tuple:
        """Download an image over http(s) with SSRF and size guards (blocking)."""
        parsed = urlparse(image_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r}")
        host = parsed.hostname
        if not host:
            raise ValueError("Image URL has no host")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # Connect to the address that was validated, never re-resolving the
        # name. TLS still verifies the certificate against the hostname (SNI +
        # hostname check), and Host carries the name the server expects.
        address = self._resolve_safe_image_address(host, port)
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        headers = {
            "Host": parsed.netloc.rsplit("@", 1)[-1],
            "Accept": "image/*",
            "User-Agent": "llm-proxy-image-fetch",
        }
        timeout = urllib3.Timeout(
            connect=self._IMAGE_DOWNLOAD_TIMEOUT, read=self._IMAGE_DOWNLOAD_TIMEOUT
        )
        if parsed.scheme == "https":
            try:
                import certifi  # the trust store requests used for this fetch
                ca_certs = certifi.where()
            except ImportError:
                ca_certs = None  # urllib3 falls back to the system store
            pool = urllib3.HTTPSConnectionPool(
                address, port, timeout=timeout, retries=False, maxsize=1,
                cert_reqs="CERT_REQUIRED", ca_certs=ca_certs, server_hostname=host,
            )
        else:
            pool = urllib3.HTTPConnectionPool(address, port, timeout=timeout, retries=False, maxsize=1)

        try:
            # redirect=False: a vetted public URL must not 3xx to an internal
            # address and bypass the guard above.
            response = pool.urlopen(
                "GET", target, headers=headers, redirect=False, retries=False,
                preload_content=False, assert_same_host=False,
            )
            try:
                if response.status != 200:
                    raise ValueError(f"Unable to access image URL: {image_url}")
                declared_length = response.headers.get("Content-Length")
                if declared_length and declared_length.isdigit() and int(declared_length) > self._IMAGE_MAX_BYTES:
                    raise ValueError(f"Image exceeds maximum size of {self._IMAGE_MAX_BYTES} bytes")

                # Size cap so a large/slow image can't pin a worker or exhaust memory.
                chunks = []
                total = 0
                for chunk in response.stream(65536):
                    total += len(chunk)
                    if total > self._IMAGE_MAX_BYTES:
                        raise ValueError(f"Image exceeds maximum size of {self._IMAGE_MAX_BYTES} bytes")
                    chunks.append(chunk)
                return b"".join(chunks), response.headers.get("Content-Type")
            finally:
                response.release_conn()
        except urllib3.exceptions.HTTPError as e:
            raise ValueError(f"Unable to access image URL: {image_url} ({e})") from e
        finally:
            pool.close()

    def _parse_image_sync(self, image_url: str) -> tuple:
        """Parse an image from a URL or base64 data URL (blocking; use a threadpool).

        Returns ``(bytes, format)`` where ``format`` is a Converse image format
        (png, jpeg, gif or webp).
        """
        data_url = self._DATA_URL_RE.match(image_url)
        if data_url:
            image_bytes = self._decode_base64_image(image_url[data_url.end():].strip())
            return image_bytes, _converse_image_format(image_bytes, data_url.group("type"))

        image_bytes, content_type = self._fetch_image_url(image_url)
        return image_bytes, _converse_image_format(image_bytes, content_type)

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
                
                image_data, image_format = self._parse_image_sync(part.image_url.url)
                content_parts.append({
                    "image": {
                        "format": image_format,
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

        Converse (like the Anthropic API behind it) requires each toolResult to
        sit in the user message immediately after the assistant message whose
        toolUse it answers. History doesn't always arrive that way (a result
        answered a turn late, a user message interjected between a call and its
        result), and a pair that merely exists somewhere in the list is still
        rejected. So:

        * each assistant message's toolUse blocks claim the first unclaimed
          toolResult with the same id found later in the conversation; the
          claimed results are moved into a user message placed directly after
          that assistant message, in call order;
        * a toolUse with no later result, and a toolResult no earlier toolUse
          claimed, are dropped as orphans;
        * a message left with no content is dropped.

        Already-valid histories come out unchanged.
        """
        if not messages:
            return messages

        def _tool_id(item: Any, key: str) -> Optional[str]:
            block = item.get(key) if isinstance(item, dict) else None
            return block.get("toolUseId") if isinstance(block, dict) else None

        def _is_block(item: Any, key: str) -> bool:
            return isinstance(item, dict) and key in item

        # Every toolResult in conversation order, as [message index, block].
        unclaimed = [
            [j, item]
            for j, msg in enumerate(messages)
            if isinstance(msg.get("content"), list)
            for item in msg["content"]
            if _is_block(item, "toolResult")
        ]

        # Pair each toolUse with the first unclaimed result for its id that
        # comes after it. paired_after[i] holds (origin index, block) for the
        # results that must directly follow assistant message i.
        paired_after: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        kept_ids: set = set()  # id() of every toolUse/toolResult block kept
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if msg.get("role") != "assistant" or not isinstance(content, list):
                continue
            for item in content:
                use_id = _tool_id(item, "toolUse") if _is_block(item, "toolUse") else None
                if not use_id:
                    continue
                for pos, (j, result) in enumerate(unclaimed):
                    if j > i and _tool_id(result, "toolResult") == use_id:
                        del unclaimed[pos]
                        kept_ids.update((id(item), id(result)))
                        paired_after.setdefault(i, []).append((j, result))
                        break

        validated_messages = []
        for i, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", [])

            if isinstance(content, list):
                filtered_content = []
                removed_items = []
                for item in content:
                    key = next((k for k in ("toolUse", "toolResult") if _is_block(item, k)), None)
                    if key is None:
                        filtered_content.append(item)
                    elif id(item) not in kept_ids:
                        removed_items.append(f"{key}:{_tool_id(item, key)}")
                    elif key == "toolUse":
                        filtered_content.append(item)
                    # A kept toolResult is re-emitted right after its call below.

                if removed_items:
                    logger.warning(
                        f"[BEDROCK] Removed orphaned items at message index {i} ({role}): {removed_items}"
                    )

                if filtered_content:
                    validated_messages.append({
                        "role": role,
                        "content": filtered_content
                    })
                elif removed_items:
                    logger.warning(
                        f"[BEDROCK] Skipping {role} message at index {i} as all content was orphaned"
                    )
            else:
                validated_messages.append(message)

            paired = paired_after.get(i)
            if paired:
                if any(j != i + 1 for j, _ in paired):
                    logger.warning(
                        f"[BEDROCK] Moved tool result(s) to directly follow their call at message index {i}"
                    )
                validated_messages.append({"role": "user", "content": [result for _, result in paired]})

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
        
        # Base inference parameters. max_completion_tokens is OpenAI's current
        # name for the output cap (max_tokens is the deprecated alias), so it
        # wins when both are sent.
        max_tokens = request.max_completion_tokens or request.max_tokens or 2048
        inference_config = {
            "maxTokens": max_tokens,
        }
        
        # Only include optional parameters when specified
        if request.temperature is not None:
            inference_config["temperature"] = request.temperature
        if request.top_p is not None:
            inference_config["topP"] = request.top_p
        
        # Drop whichever sampling params this model rejects. (topK is never
        # placed in inferenceConfig — it lives in additionalModelRequestFields
        # and is scrubbed separately below.)
        removed = scrub_unsupported_params(
            inference_config,
            request.model,
            SURFACE_CONVERSE,
            spellings=CONVERSE_SPELLINGS,
            only=frozenset({"temperature", "top_p"}),
        )
        if removed:
            if self.debug:
                logger.info(
                    f"Removed {', '.join(removed)} for {request.model} (not supported by this model)"
                )
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
        
        # Handle reasoning effort. Judged by the foundation model, so an
        # application inference profile ARN gets its model's capabilities.
        capability_model = self._capability_model_id(request.model)
        if request.reasoning_effort:
            if self._is_grok_model(capability_model):
                # Grok wants an effort string, not a token budget;
                # _apply_grok_constraints validates/normalizes it below.
                args.setdefault("additionalModelRequestFields", {})["reasoning_effort"] = request.reasoning_effort
            # Adaptive-thinking models (Claude >= 4.7) reject an explicit token
            # budget, so no reasoning_config can be emitted for them. Thinking is
            # on by default there, so omitting it cannot trigger a 400 — but the
            # requested effort has nowhere to go on this surface and is lost.
            # Warn rather than swallow it silently; on Converse the only
            # candidate carrier is additionalModelRequestFields["output_config"],
            # which is unverified against Bedrock (see note in the module docs).
            elif thinking_style(capability_model) == THINKING_ADAPTIVE:
                logger.warning(
                    "[BEDROCK] reasoning_effort=%r cannot be forwarded for %s on the "
                    "Converse path (adaptive thinking rejects budget_tokens); the "
                    "request will run at the model's default effort",
                    request.reasoning_effort,
                    request.model,
                )
            # reasoning_config is Claude's extended-thinking switch (Claude 3.7
            # and later). Other models (older Claude, Llama, Nova, Mistral, ...)
            # either reject the unknown field or have no budget-style reasoning,
            # so it is not sent to them.
            elif not self._supports_budget_thinking(capability_model):
                logger.warning(
                    "[BEDROCK] reasoning_effort=%r is not supported for %s on the "
                    "Converse path; ignoring it",
                    request.reasoning_effort,
                    request.model,
                )
            # Reasoning requires budget_tokens >= 1024 AND < max_tokens. If
            # max_tokens is too small to satisfy the floor, skip reasoning
            # rather than emit an invalid budget that Bedrock would reject.
            elif max_tokens <= self._MIN_BUDGET_TOKENS:
                logger.warning(
                    f"[BEDROCK] max_tokens={max_tokens} too small for reasoning "
                    f"(needs > {self._MIN_BUDGET_TOKENS}); skipping reasoning_config"
                )
            else:
                budget_tokens = self._calc_budget_tokens(max_tokens, request.reasoning_effort)
                # Extended thinking rejects topP and any temperature != 1, so
                # drop both from inferenceConfig when reasoning is enabled.
                inference_config.pop("topP", None)
                inference_config.pop("temperature", None)
                args["additionalModelRequestFields"] = {
                    "reasoning_config": {"type": "enabled", "budget_tokens": budget_tokens}
                }
        
        # Check if messages contain tool usage (toolUse or toolResult blocks)
        # AWS Bedrock requires toolConfig to be present when any message contains these blocks
        messages_have_tools = self._messages_contain_tool_usage(messages)

        # tool_choice "none" means the model must not call a tool. Converse has
        # no "none" choice, so the declared tools are withheld instead (only
        # the placeholder remains when history forces a toolConfig).
        tool_choice = request.tool_choice
        tools_disabled = tool_choice == "none" or (
            isinstance(tool_choice, dict) and tool_choice.get("type") == "none"
        )
        declared_tools = request.tools if request.tools and not tools_disabled else None

        # Handle tools
        if declared_tools or messages_have_tools:
            # Build tool config from current request tools, or use empty tools list if none provided
            # but message history contains tool usage
            tool_config = {
                "tools": [self._convert_tool_spec(t.function, tool_name_mapping=tool_name_mapping) for t in declared_tools] if declared_tools else []
            }

            # Converse requires >= 1 tool when toolConfig is present; if history
            # has tool blocks but no tools are declared, inject a placeholder.
            if not tool_config["tools"]:
                tool_config["tools"] = [self._PLACEHOLDER_TOOL_SPEC]

            if tool_choice and declared_tools and not request.model.startswith("meta.llama3-1-"):
                if isinstance(tool_choice, str):
                    if tool_choice == "required":
                        tool_config["toolChoice"] = {"any": {}}
                    else:
                        tool_config["toolChoice"] = {"auto": {}}
                else:
                    # Specific tool to use
                    if isinstance(tool_choice, dict) and "function" in tool_choice:
                        # Named the way the toolSpec was (truncated past 64 chars).
                        tool_config["toolChoice"] = {
                            "tool": {"name": self._truncate_tool_name(
                                (tool_choice.get("function") or {}).get("name", ""),
                                tool_name_mapping=tool_name_mapping,
                            )}
                        }

            args["toolConfig"] = tool_config

            if self.debug and messages_have_tools and not declared_tools:
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

            # Strip effort from output_config. Claude *does* reach this builder
            # (/v1/chat/completions routes every model through Converse; only
            # /v1/messages has a Claude-native branch), so the old "non-Claude
            # only" reasoning was wrong — but Converse has no verified carrier
            # for output_config, so effort still cannot be forwarded here.
            if "output_config" in filtered_extra_fields:
                oc = filtered_extra_fields["output_config"]
                if isinstance(oc, dict) and "effort" in oc:
                    oc = {k: v for k, v in oc.items() if k != "effort"}
                    if oc:
                        filtered_extra_fields["output_config"] = oc
                    else:
                        del filtered_extra_fields["output_config"]
            
            # A client-supplied thinking config in the extra body has to go
            # through the same normalization as every other request builder —
            # otherwise budget_tokens reaches an adaptive-thinking model verbatim
            # and 400s. This was the one builder the capability table missed.
            if isinstance(filtered_extra_fields.get("thinking"), dict):
                normalized_thinking, thinking_dropped = normalize_thinking(
                    filtered_extra_fields["thinking"], request.model
                )
                filtered_extra_fields["thinking"] = normalized_thinking
                if thinking_dropped and self.debug:
                    logger.info(
                        f"Removed {', '.join(thinking_dropped)} from extra fields "
                        f"for {request.model} (adaptive thinking)"
                    )

            if filtered_extra_fields:
                # Merge into (not overwrite) additionalModelRequestFields so a
                # reasoning_config set from request.reasoning_effort above is
                # preserved. Explicit extra fields win on key conflicts.
                existing_amrf = args.get("additionalModelRequestFields", {})
                existing_amrf.update(filtered_extra_fields)
                args["additionalModelRequestFields"] = existing_amrf
                # Extended thinking rejects topP, temperature != 1, and top_k;
                # scrub all three to avoid a Bedrock validation error.
                if "thinking" in filtered_extra_fields:
                    inference_config.pop("topP", None)
                    inference_config.pop("temperature", None)
                    existing_amrf.pop("top_k", None)

        # top_k rides in additionalModelRequestFields rather than inferenceConfig,
        # so it needs its own scrub against the same table.
        amrf = args.get("additionalModelRequestFields")
        if amrf:
            removed_amrf = scrub_unsupported_params(
                amrf,
                request.model,
                SURFACE_CONVERSE,
                only=frozenset({"top_k"}),
            )
            if removed_amrf and self.debug:
                logger.info(
                    f"Removed {', '.join(removed_amrf)} from additionalModelRequestFields "
                    f"for {request.model}"
                )

        # Grok rejects the sampling knobs entirely; scrub last so nothing above
        # (extra fields, reasoning) can reintroduce them.
        self._apply_grok_constraints(args)

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

    async def _prepare_converse_args(self, request: ChatCompletionRequest) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Build the Converse args for an OpenAI chat request.

        Returns:
            tuple: (args, tool_name_mapping) where tool_name_mapping is a
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

        return args, tool_name_mapping

    @staticmethod
    def _http_exception_for_client_error(e: ClientError):
        """Map a Bedrock ClientError to an HTTPException with the right status.

        Throttling/quota/validation/permission errors keep their real HTTP
        status (429/400/403/...) instead of collapsing to a generic 500.
        """
        err = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
        code = err.get("Code", "ClientError")
        msg = err.get("Message", str(e))
        logger.error(f"Bedrock invocation failed: {code}: {msg}")
        mapped = _map_bedrock_error(code, msg)
        from fastapi import HTTPException
        return HTTPException(status_code=mapped["status"], detail=mapped["body"])

    async def _open_converse_stream(self, args: Dict[str, Any]) -> _ConverseStreamReader:
        """Open ``converse_stream`` on a dedicated reader thread.

        Takes a Converse permit first and hands it to the reader, which
        releases it when its thread exits, so active streams (not just
        stream opens) count against the bound. Raises whatever opening raised
        (e.g. a ClientError) before any event is read.
        """
        semaphore = self._get_converse_semaphore()
        await semaphore.acquire()
        try:
            reader = _ConverseStreamReader(
                lambda: self.bedrock_runtime.converse_stream(**args).get("stream"),
                on_exit=semaphore.release,
            )
            reader.start()
        except BaseException:
            semaphore.release()
            raise
        try:
            await reader.wait_opened()
        except BaseException:
            await reader.close()
            raise
        return reader

    async def _converse_once(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """One blocking ``converse`` call, holding a Converse permit throughout."""
        async with self._get_converse_semaphore():
            return await run_in_threadpool(self.bedrock_runtime.converse, **args)

    def _drop_rejected_converse_field(self, error: ClientError, args: Dict[str, Any]) -> bool:
        """Drop the optional inferenceConfig field a ValidationException refuses.

        Bedrock's static capability rules only know some models (Grok); any
        other model that refuses an optional field ("This model doesn't
        support the stopSequences field") would otherwise fail every request
        that carries it, e.g. each Claude Code auto-mode classifier call.
        Returns True when a field was dropped (and remembered for the model)
        so the caller can retry.
        """
        err = (getattr(error, "response", None) or {}).get("Error", {})
        if err.get("Code") != "ValidationException":
            return False
        inference_config = args.get("inferenceConfig")
        if not isinstance(inference_config, dict):
            return False
        present = {
            name: inference_config[field]
            for name, field in _CONVERSE_OPTIONAL_FIELDS.items()
            if inference_config.get(field) is not None
        }
        name = unsupported_param(400, err.get("Message") or "", present, tuple(_CONVERSE_OPTIONAL_FIELDS))
        if name is None:
            return False
        field = _CONVERSE_OPTIONAL_FIELDS[name]
        inference_config.pop(field, None)
        remember_rejected_param(self, args.get("modelId"), _CONVERSE_TRANSPORT, field)
        return True

    async def _call_converse(self, args: Dict[str, Any], call):
        """Run ``call(args)`` (a converse call or stream open), dropping and
        retrying optional fields the model refuses.

        Fields this model refused before are dropped up front. Only the
        opening is retried: a stream is returned once it has opened, before
        any event has been read.
        """
        inference_config = args.get("inferenceConfig")
        if isinstance(inference_config, dict):
            for field in known_rejected_params(self, args.get("modelId"), _CONVERSE_TRANSPORT):
                inference_config.pop(field, None)
        retries = 0
        while True:
            try:
                return await call(args)
            except ClientError as e:
                if retries >= MAX_PARAM_RETRIES or not self._drop_rejected_converse_field(e, args):
                    raise
                retries += 1

    async def _invoke_bedrock(self, request: ChatCompletionRequest):
        """Invoke Bedrock Converse (non-streaming).

        Returns:
            tuple: (response, tool_name_mapping) where tool_name_mapping is a
                   request-scoped dict mapping truncated tool names to originals.
        """
        args, tool_name_mapping = await self._prepare_converse_args(request)

        try:
            # Bound Converse concurrency for the whole (blocking) call.
            response = await self._call_converse(args, self._converse_once)
            return response, tool_name_mapping
        except ClientError as e:
            raise self._http_exception_for_client_error(e)
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise

    async def _open_bedrock_stream(self, request: ChatCompletionRequest) -> Tuple[_ConverseStreamReader, Dict[str, str]]:
        """Open a Converse stream for an OpenAI chat request.

        Returns:
            tuple: (reader, tool_name_mapping). The caller must ``close()`` the reader.
        """
        args, tool_name_mapping = await self._prepare_converse_args(request)
        try:
            reader = await self._call_converse(args, self._open_converse_stream)
        except ClientError as e:
            raise self._http_exception_for_client_error(e)
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            raise
        return reader, tool_name_mapping

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle chat completion request."""
        response, tool_name_mapping = await self._invoke_bedrock(request)
        
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
        
        # Parse response message. Accumulate ALL blocks regardless of stop
        # reason: a tool_use response can still carry preceding text/reasoning
        # blocks, and multiple text blocks must be concatenated (not overwritten).
        message = ChatMessage(role="assistant", content="")
        content = ""
        reasoning_parts: List[str] = []
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
            elif "reasoningContent" in part:
                # Emit reasoning on the dedicated reasoning_content field; keep
                # content as the visible text only (no <think> injection).
                # Encrypted reasoning (Grok returns redactedContent only) has no
                # client-visible text, so skip it rather than KeyError.
                # Interleaved thinking yields several reasoning blocks; keep
                # them all, in order.
                reasoning_text = part["reasoningContent"].get("reasoningText")
                if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                    reasoning_parts.append(reasoning_text["text"])
            elif "text" in part:
                content += part["text"]

        if tool_calls:
            message.tool_calls = tool_calls
        if reasoning_parts:
            message.reasoning_content = "\n\n".join(reasoning_parts)
        message.content = content
        
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=echo_model_name(request),
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

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion request."""
        reader: Optional[_ConverseStreamReader] = None
        try:
            reader, tool_name_mapping = await self._open_bedrock_stream(request)
            message_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            # Per-stream parser state (local to this stream to avoid cross-request races):
            #   tool_call_index   — last OpenAI tool-call ordinal assigned
            #   block_index_map   — Bedrock contentBlockIndex -> OpenAI tool-call ordinal
            #   created           — one timestamp for every chunk of the stream
            stream_state = {
                "tool_call_index": -1,
                "block_index_map": {},
                "created": int(time.time()),
            }

            # Default include_usage to True if not specified
            include_usage = True
            if request.stream_options is not None:
                include_usage = request.stream_options.include_usage if request.stream_options.include_usage is not None else True

            while True:
                chunk = await reader.next_event()
                if chunk is None:
                    break
                # Thread parser state through each chunk
                stream_response, stream_state = self._parse_stream_chunk(chunk, message_id, echo_model_name(request), stream_state, tool_name_mapping)
                
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
        finally:
            # Runs on completion, error, cancellation and client disconnect
            # (aclose): stops the reader thread and frees its Converse permit.
            if reader is not None:
                await reader.close()

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
                # Emit a plain dict (not the Pydantic ToolCall) so format_sse_data's
                # json.dumps can serialize the delta — a ToolCall object here raises
                # "Object of type ToolCall is not JSON serializable".
                delta = {
                    "tool_calls": [ToolCall(
                        index=index,
                        type="function",
                        id=start["toolUse"]["toolUseId"],
                        function=ResponseFunction(
                            name=tool_name,
                            arguments=""
                        )
                    ).model_dump(exclude_none=True)]
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
                    ).model_dump(exclude_none=True)]
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
                    "created": stream_state.get("created") or int(time.time()),
                    "model": model_id,
                    "choices": [],
                    "usage": usage.model_dump()
                }, stream_state)

        if delta or finish_reason:
            return ({
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": stream_state.get("created") or int(time.time()),
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
                        except json.JSONDecodeError:
                            continue

                        # Pass error chunks straight through so a failed stream
                        # surfaces to the client instead of silently ending.
                        if "error" in data:
                            yield self.format_sse_data(data)
                            continue

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
                            # Forward the usage chunk (choices:[] carries usage).
                            if data.get("usage") is not None:
                                completion_chunk["usage"] = data["usage"]
                            yield self.format_sse_data(completion_chunk)
                        elif data.get("usage") is not None:
                            # Usage-only chunk (empty choices) — forward it with an
                            # empty choices list so clients that read usage see it.
                            yield self.format_sse_data({
                                "id": data.get("id", "").replace("chatcmpl", "cmpl"),
                                "object": "text_completion",
                                "created": data.get("created", int(time.time())),
                                "model": data.get("model", request.model),
                                "choices": [],
                                "usage": data["usage"],
                            })

    async def embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Handle embeddings request."""
        # Extract clean model ID from prefixed model name
        model_id = self.get_model_id(request.model)
        
        if model_id not in self.SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported embedding model: {model_id}")
        
        model_name = self.SUPPORTED_EMBEDDING_MODELS[model_id]

        # Prepare input. The OpenAI schema also allows token-ID arrays, but
        # Cohere and Titan on Bedrock only embed text, so those are refused
        # here instead of being sent upstream (or tokenized with str.split).
        if isinstance(request.input, str):
            texts = [request.input]
        elif (
            isinstance(request.input, list)
            and request.input
            and all(isinstance(item, str) for item in request.input)
        ):
            texts = list(request.input)
        else:
            message = (
                "Bedrock embeddings accept a string or a non-empty list of strings; "
                "token-ID arrays are not supported"
            )
            raise ProviderHTTPError(
                400,
                message,
                body={"error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": None,
                }},
            )

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
            model=echo_model_name(request),
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

        if mode == "adapter":
            model_id = self._resolve_bedrock_anthropic_model_id(request.model)
            dropped_fields.extend(self._adapter_dropped_fields(request, model_id))

        return AnthropicRequestMetadata(
            mode=mode,
            transport="messages" if mode == "native" else "chat",
            dropped_fields=sorted(set(dropped_fields)),
        )

    # ---- Converse adapter (non-Claude /v1/messages) request semantics ----

    _ADAPTER_TOOL_CHOICE_TYPES = frozenset({"auto", "any", "tool", "none"})

    @staticmethod
    def _dump_optional(value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return value if isinstance(value, dict) else None

    def _adapter_reasoning_family(self, model_id: str) -> Optional[str]:
        """How the Converse adapter carries a thinking config for ``model_id``.

        None when the model has no known reasoning control on Converse, in
        which case an Anthropic ``thinking`` block is not forwarded (Bedrock
        rejects unknown additionalModelRequestFields for those models).
        """
        capability_model = self._capability_model_id(model_id)
        if self._is_nova2_model(capability_model):
            return "nova2"
        if self._is_kimi_model(capability_model):
            return "kimi"
        if self._is_grok_model(capability_model):
            return "grok"
        if self._is_claude_model(capability_model):
            return "claude"
        return None

    def _adapter_thinking(self, request, model_id: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """The thinking config the adapter honours, and the fields it cannot.

        Returns ``(thinking_data, dropped)``. ``thinking_data`` is None when the
        request asked for no thinking (or only ``disabled``) or the model has no
        reasoning control on Converse; for Claude it has been through the same
        normalize_thinking as the native path (adaptive models reject a budget).
        """
        thinking_data = self._dump_optional(getattr(request, "thinking", None))
        if not thinking_data or thinking_data.get("type") == "disabled":
            return None, []
        family = self._adapter_reasoning_family(model_id)
        if family is None:
            return None, ["thinking"]
        if family == "claude":
            normalized, dropped = normalize_thinking(thinking_data, self._capability_model_id(model_id))
            return normalized, list(dropped)
        return thinking_data, []

    def _adapter_dropped_fields(self, request, model_id: str) -> List[str]:
        """Request fields the Converse adapter cannot honour for ``model_id``."""
        dropped: List[str] = []
        metadata = self._dump_optional(getattr(request, "metadata", None))
        if metadata:
            # Converse has no counterpart to Anthropic's metadata.user_id.
            dropped.append("metadata")
        tool_choice = self._dump_optional(getattr(request, "tool_choice", None))
        if tool_choice and tool_choice.get("disable_parallel_tool_use"):
            # Converse's toolChoice has no parallel-tool-use switch.
            dropped.append("tool_choice.disable_parallel_tool_use")
        _, thinking_dropped = self._adapter_thinking(request, model_id)
        dropped.extend(thinking_dropped)
        return dropped

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

    def _is_grok_model(self, model_id: str) -> bool:
        """True if model_id refers to an xAI Grok model on Bedrock."""
        lower = model_id.lower()
        return "xai." in lower or "grok" in lower

    # Grok takes reasoning as an effort string in additionalModelRequestFields.
    # The Claude-style reasoning_config/thinking objects map onto reasoning.effort
    # and are rejected ("Invalid type for 'reasoning.effort'").
    _GROK_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
    _GROK_REASONING_EFFORT_ALIASES = {"minimal": "low"}

    @staticmethod
    def _budget_to_effort(budget_tokens: int) -> str:
        """Map an Anthropic thinking budget onto a coarse reasoning effort."""
        if budget_tokens > 10000:
            return "high"
        return "low" if budget_tokens < 1000 else "medium"

    @classmethod
    def _grok_reasoning_effort(cls, value: str) -> Optional[str]:
        """Normalize an effort string to one Grok accepts, or None if unusable."""
        normalized = value.strip().lower()
        normalized = cls._GROK_REASONING_EFFORT_ALIASES.get(normalized, normalized)
        return normalized if normalized in cls._GROK_REASONING_EFFORTS else None

    def _apply_grok_constraints(self, args: Dict[str, Any]) -> None:
        """Scrub Converse fields that Bedrock's xAI Grok models reject (in place)."""
        model_id = args.get("modelId", "")
        if not self._is_grok_model(model_id):
            return

        # Grok accepts only maxTokens in inferenceConfig; the rejected field list
        # lives with every other model's in app.model_capabilities.
        removed = scrub_unsupported_params(
            args.get("inferenceConfig") or {},
            model_id,
            SURFACE_CONVERSE,
            spellings=CONVERSE_SPELLINGS,
        )

        additional_fields = args.get("additionalModelRequestFields")
        if isinstance(additional_fields, dict):
            # Collapse every reasoning spelling down to a single effort string.
            effort = additional_fields.pop("reasoning_effort", None)
            for key in ("reasoning_config", "thinking"):
                value = additional_fields.pop(key, None)
                if effort is not None:
                    continue
                if isinstance(value, dict) and value.get("type") != "disabled":
                    effort = self._budget_to_effort(value.get("budget_tokens") or 0)
                elif isinstance(value, str):
                    effort = value
            normalized = self._grok_reasoning_effort(str(effort)) if effort is not None else None
            if normalized:
                additional_fields["reasoning_effort"] = normalized
            if not additional_fields:
                args.pop("additionalModelRequestFields", None)

        if removed and self.debug:
            logger.info(
                f"Removed {', '.join(removed)} for {args.get('modelId')} (not supported by Grok on Bedrock)"
            )

    @staticmethod
    def _is_claude_at_least(model_id: str, min_major: int, min_minor: int) -> bool:
        return is_claude_at_least(model_id, min_major, min_minor)

    @staticmethod
    def _supports_output_config_effort(model_id: str) -> bool:
        return BedrockProvider._is_claude_at_least(model_id, 4, 6)

    @staticmethod
    def _supports_budget_thinking(model_id: str) -> bool:
        """Claude models with extended thinking (3.7 and later).

        Adaptive-thinking models (see thinking_style) are a subset that rejects
        an explicit budget; callers check for those first.
        """
        return BedrockProvider._is_claude_at_least(model_id, 3, 7)

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

                    # Convert web_search_tool_result in user messages → plain
                    # text. The matching server_tool_use block is dropped from
                    # the assistant message above, so emitting a tool_result here
                    # would leave an orphan (no matching tool_use) that the
                    # Anthropic/Bedrock API rejects with a 400.
                    if bt == "web_search_tool_result":
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
                            "type": "text",
                            "text": f"[Web search results]\n{result_text}",
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

        # Drop whichever of those this model rejects ("top_p is deprecated for
        # this model", "This model doesn't support the temperature field", ...).
        removed = scrub_unsupported_params(native, request.model, SURFACE_NATIVE)
        if removed and self.debug:
            logger.info(
                f"Removed {', '.join(removed)} for {request.model} (not supported by this model)"
            )

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
            thinking = (
                request.thinking.model_dump(exclude_none=True)
                if hasattr(request.thinking, "model_dump")
                else request.thinking
            )
            # Adaptive-thinking models reject an explicit token budget.
            thinking, thinking_dropped = normalize_thinking(thinking, request.model)
            native["thinking"] = thinking
            if thinking_dropped and self.debug:
                logger.info(
                    f"Removed {', '.join(thinking_dropped)} for {request.model} "
                    f"(adaptive thinking)"
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
        uses_default_chain = not (self.aws_access_key and self.aws_secret_key)
        if not uses_default_chain:
            kwargs["aws_access_key"] = self.aws_access_key
            kwargs["aws_secret_key"] = self.aws_secret_key
        # else: omit creds → the default boto3 credential chain, matching how
        # self.bedrock_runtime is built without explicit creds. The SDK would
        # resolve (and refresh) that chain synchronously inside its async
        # request preparation, i.e. on the event loop, so
        # _ensure_anthropic_client() resolves it off-loop and hands the SDK
        # static credentials instead.
        try:
            self._anthropic_client = AsyncAnthropicBedrock(**kwargs)
            self._anthropic_uses_default_chain = uses_default_chain
            logger.info(
                "[BEDROCK] AsyncAnthropicBedrock client initialized for region=%s",
                self.aws_region,
            )
        except Exception as e:
            logger.warning("[BEDROCK] Failed to init AsyncAnthropicBedrock client: %s", e)

    async def _ensure_anthropic_client(self) -> None:
        """Build the native client and make sure its credentials are current.

        With explicit keys the SDK signs with static credentials and never
        blocks. With the default chain, credentials are resolved on a worker
        thread (instance metadata, SSO, credential_process, ... can each block
        on I/O) and refreshed there before they expire; the SDK client gets the
        resulting static key/secret/token, so signing on the loop stays cheap.
        """
        self._init_bedrock_anthropic_client()
        client = self._anthropic_client
        if client is None or not getattr(self, "_anthropic_uses_default_chain", False):
            return
        if not self._chain_credentials_stale():
            return
        if getattr(self, "_chain_credentials_lock", None) is None:
            self._chain_credentials_lock = asyncio.Lock()
        async with self._chain_credentials_lock:
            if not self._chain_credentials_stale():
                return  # another request refreshed them while we waited
            credentials, frozen = await asyncio.to_thread(
                self._resolve_chain_credentials, self._chain_credentials
            )
            # Plain attribute swap on the loop thread: no request can observe
            # a half-updated key/secret/token triple.
            client.aws_access_key = frozen.access_key
            client.aws_secret_key = frozen.secret_key
            client.aws_session_token = frozen.token
            self._chain_credentials = credentials

    def _chain_credentials_stale(self) -> bool:
        credentials = getattr(self, "_chain_credentials", None)
        if credentials is None:
            return True
        refresh_needed = getattr(credentials, "refresh_needed", None)
        # Static credentials (e.g. from env vars) never need a refresh;
        # refreshable ones report it ahead of expiry (botocore's advisory window).
        return bool(callable(refresh_needed) and refresh_needed())

    def _resolve_chain_credentials(self, credentials: Any) -> tuple:
        """Resolve/refresh default-chain credentials (blocking; run off-loop)."""
        if credentials is None:
            credentials = boto3.session.Session(region_name=self.aws_region).get_credentials()
            if credentials is None:
                raise RuntimeError("could not resolve AWS credentials for Bedrock")
        # Refreshes in place when expired or about to expire.
        return credentials, credentials.get_frozen_credentials()

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

    def _convert_anthropic_image_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert an Anthropic image block to a Converse image block.

        Used for message content and tool_result content alike. Returns None
        for a source type Converse can't take (e.g. a Files API reference).
        """
        source = block.get("source") or {}
        if source.get("type") == "base64" and source.get("data"):
            image_bytes = self._decode_base64_image(source["data"])
            image_format = _converse_image_format(image_bytes, source.get("media_type"))
        elif source.get("type") == "url" and source.get("url"):
            image_bytes, image_format = self._parse_image_sync(source["url"])
        else:
            return None
        return {"image": {"format": image_format, "source": {"bytes": image_bytes}}}

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
                image_block = self._convert_anthropic_image_block(block)
                if image_block is not None:
                    converted.append(image_block)
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
                            result_type = result_block.get("type")
                            image_block = (
                                self._convert_anthropic_image_block(result_block)
                                if result_type == "image" else None
                            )
                            if result_type == "text":
                                converted_tool_result_content.append({"text": result_block.get("text", "")})
                            elif image_block is not None:
                                # Converse tool results carry images natively.
                                converted_tool_result_content.append(image_block)
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

        # role="system" entries inside messages[] are instructions, not user
        # turns: lift them into the system prompt after request.system, exactly
        # as the native path does (same helper), instead of sending them as
        # user messages. Unknown roles are coerced to "user" by the helper.
        raw_messages = []
        for message in request.messages:
            content = message.content
            if isinstance(content, list):
                content = [
                    block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
                    for block in content
                ]
            raw_messages.append({"role": message.role, "content": content})
        raw_messages, extracted_system = _extract_system_messages_from_messages(raw_messages)
        for block in extracted_system:
            if block.get("text"):
                system_blocks.append({"text": block["text"]})
                if self._is_claude_model(model_id):
                    cache_point_block = self._cache_control_to_cache_point_block(block.get("cache_control"))
                    if cache_point_block:
                        system_blocks.append(cache_point_block)

        for message in raw_messages:
            role = "assistant" if message["role"] == "assistant" else "user"
            messages.append({
                "role": role,
                "content": self._convert_anthropic_content_to_bedrock(message["content"], supports_caching=self._is_claude_model(model_id)),
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

        # Thinking goes out in the model family's own spelling, or not at all
        # for a model with no reasoning control on Converse (reported through
        # get_anthropic_request_metadata). A disabled block enables nothing.
        thinking_data, thinking_dropped = self._adapter_thinking(request, model_id)
        if thinking_dropped and self.debug:
            logger.info(f"Not forwarding {', '.join(thinking_dropped)} for {model_id} (Converse adapter)")
        reasoning_family = self._adapter_reasoning_family(model_id) if thinking_data else None
        if reasoning_family == "nova2":
            # Nova 2 uses reasoningConfig with effort level; temperature and
            # maxTokens must be removed from inferenceConfig when reasoning is on.
            effort = self._budget_to_effort(thinking_data.get("budget_tokens", 0))
            additional_fields["reasoningConfig"] = {"type": "enabled", "maxReasoningEffort": effort}
            inference_config.pop("temperature", None)
            inference_config.pop("maxTokens", None)
        elif reasoning_family == "kimi":
            # Kimi only supports reasoning_effort="high"
            additional_fields["reasoning_effort"] = "high"
        elif reasoning_family == "grok":
            # Grok takes an effort string instead of a thinking block.
            additional_fields["reasoning_effort"] = self._budget_to_effort(
                thinking_data.get("budget_tokens", 0)
            )
        elif reasoning_family == "claude":
            # Already normalized (adaptive models reject budget_tokens).
            additional_fields["thinking"] = thinking_data

        # Beta headers are only meaningful for Claude/Anthropic models on Bedrock
        if self._is_claude_model(model_id):
            beta_values = self._map_anthropic_beta_values(self._parse_anthropic_beta_values(anthropic_beta))
            if beta_values:
                additional_fields["anthropic_beta"] = beta_values

        if additional_fields:
            args["additionalModelRequestFields"] = additional_fields

        # tool_choice: auto/any/tool map onto Converse; "none" (no tool calls)
        # has no Converse choice, so the declared tools are withheld instead.
        # Anything else is refused rather than silently run as "auto".
        tool_choice_data = self._dump_optional(request.tool_choice)
        tool_choice_type = tool_choice_data.get("type") if tool_choice_data else None
        if tool_choice_data is not None:
            if tool_choice_type not in self._ADAPTER_TOOL_CHOICE_TYPES:
                raise ValueError(
                    f"Unsupported tool_choice type {tool_choice_type!r}; "
                    "expected one of auto, any, tool, none"
                )
            if tool_choice_type == "tool" and not tool_choice_data.get("name"):
                raise ValueError("tool_choice of type 'tool' requires a name")
        declared_tools = request.tools if request.tools and tool_choice_type != "none" else None

        messages_have_tools = self._messages_contain_tool_usage(messages)
        if declared_tools or messages_have_tools:
            bedrock_tools = []
            special_tools: List[Dict[str, Any]] = []
            if declared_tools:
                for index, tool in enumerate(declared_tools):
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
            # Converse requires >= 1 tool when toolConfig is present. If every
            # declared tool was a special type (forwarded via
            # additionalModelRequestFields) or history has tool blocks but no
            # tools remain, inject a placeholder to keep toolConfig valid.
            placeholder_only = False
            if not bedrock_tools:
                bedrock_tools.append(self._PLACEHOLDER_TOOL_SPEC)
                placeholder_only = True

            tool_config: Dict[str, Any] = {
                "tools": bedrock_tools,
            }

            if tool_choice_type in ("any", "tool") and placeholder_only:
                # The only tool in toolConfig is the injected placeholder; a
                # forced choice would make the model call the fake noop tool
                # (or fail validation for a named tool that isn't in
                # toolConfig). Downgrade to auto instead.
                tool_config["toolChoice"] = {"auto": {}}
            elif tool_choice_type == "auto":
                tool_config["toolChoice"] = {"auto": {}}
            elif tool_choice_type == "any":
                tool_config["toolChoice"] = {"any": {}}
            elif tool_choice_type == "tool":
                tool_config["toolChoice"] = {"tool": {"name": tool_choice_data["name"]}}
            # "none": no real tools were declared above; history-only toolConfig
            # keeps just the placeholder with Converse's default choice.

            args["toolConfig"] = tool_config

        removed = scrub_unsupported_params(
            inference_config,
            model_id,
            SURFACE_CONVERSE,
            spellings=CONVERSE_SPELLINGS,
        )
        if removed and self.debug:
            logger.info(f"Removed {', '.join(removed)} for {model_id} (not supported by this model)")

        self._apply_grok_constraints(args)

        return _sanitize_for_json(args)

    # Anthropic's stop_reason enum is closed; unmapped Bedrock values (e.g.
    # "guardrail_intervened") must not leak through as non-enum values.
    _ANTHROPIC_STOP_REASON_MAP = {
        "end_turn": "end_turn",
        "stop_sequence": "stop_sequence",
        "max_tokens": "max_tokens",
        "tool_use": "tool_use",
        "content_filtered": "end_turn",
    }
    # Valid Anthropic-native stop reasons that may already be in Anthropic form.
    _ANTHROPIC_STOP_REASON_PASSTHROUGH = {
        "end_turn",
        "stop_sequence",
        "max_tokens",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    }

    def _convert_bedrock_stop_reason(self, stop_reason: Optional[str]) -> Optional[str]:
        if not stop_reason:
            return None
        if stop_reason in self._ANTHROPIC_STOP_REASON_MAP:
            return self._ANTHROPIC_STOP_REASON_MAP[stop_reason]
        if stop_reason in self._ANTHROPIC_STOP_REASON_PASSTHROUGH:
            return stop_reason
        logger.warning(f"[BEDROCK] Unknown stop reason '{stop_reason}', defaulting to 'end_turn'")
        return "end_turn"

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

    def _anthropic_stop_events(self, stop_reason: Optional[str], stop_sequence: Optional[str]) -> List[str]:
        """message_delta + message_stop for a complete message whose usage is unknown."""
        return [
            self._format_anthropic_sse_event("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }),
            self._format_anthropic_sse_event("message_stop", {"type": "message_stop"}),
        ]

    def _anthropic_stream_error_event(self, exc: BaseException) -> str:
        """An Anthropic ``error`` SSE event for a failed Converse stream.

        A Bedrock error (EventStreamError is a ClientError) keeps its mapped
        type, e.g. throttling -> rate_limit_error; anything else is api_error.
        """
        if isinstance(exc, ClientError):
            err = (exc.response or {}).get("Error", {})
            body = _map_bedrock_error(err.get("Code") or "ClientError", err.get("Message") or str(exc))["body"]
        else:
            body = {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
        return self._format_anthropic_sse_event("error", body)

    async def anthropic_messages(self, request, anthropic_beta: Optional[str] = None) -> Any:
        """Handle Anthropic Messages API request.

        Routes Claude models through the AsyncAnthropicBedrock SDK (native
        Anthropic JSON) and non-Claude models through the Converse API.
        """
        model_id = self._resolve_bedrock_anthropic_model_id(request.model)

        # ── Native path for Claude models (AsyncAnthropicBedrock SDK) ──────
        if self._is_claude_model(model_id):
            await self._ensure_anthropic_client()
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
            result["model"] = echo_model_name(request)
            return result

        # ── Converse path for non-Claude models ────────────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
            response = await self._call_converse(args, self._converse_once)
        except ClientError as e:
            error = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            code = error.get("Code")
            if code == "ValidationException":
                raise ValueError(error.get("Message") or str(e)) from e
            # Map throttling/quota/permission errors to their real HTTP status
            # (429/403/...) with an Anthropic-shaped body instead of a 500.
            mapped = _map_bedrock_error(code or "ClientError", error.get("Message", str(e)))
            from fastapi import HTTPException
            raise HTTPException(status_code=mapped["status"], detail=mapped["body"]) from e

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
            "model": echo_model_name(request),
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
            await self._ensure_anthropic_client()
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

            try:
                # messages.stream(...) yields the SDK's *processed* event union
                # (synthesized TextEvent/InputJsonEvent, snapshot-enriched stops),
                # which the shared generator serializes verbatim. The bounded
                # post-terminal drain policy and event serialization are shared
                # with the Azure Foundry / anthropic_compatible paths via
                # stream_anthropic_sdk_events so cleanup semantics stay aligned
                # across all three providers.
                async with self._anthropic_client.messages.stream(**kwargs) as stream:
                    async for sse in stream_anthropic_sdk_events(
                        stream,
                        provider_label=full_provider_name,
                        model=echo_model_name(request),
                        log=logger,
                    ):
                        yield sse
            except Exception as e:
                raise self._translate_bedrock_sdk_error(e) from e
            return

        # ── Converse Stream path for non-Claude models ─────────────────────
        args = await run_in_threadpool(self._build_bedrock_anthropic_args, request, anthropic_beta)
        try:
            # The reader holds a Converse permit until its thread exits, so an
            # active stream counts against the bound, not just its opening.
            reader = await self._call_converse(args, self._open_converse_stream)
        except ClientError as e:
            error = (e.response or {}).get("Error", {}) if hasattr(e, "response") else {}
            code = error.get("Code")
            if code == "ValidationException":
                raise ValueError(error.get("Message") or str(e)) from e
            # Map throttling/quota/permission errors to their real HTTP status
            # (429/403/...) with an Anthropic-shaped body instead of a 500.
            mapped = _map_bedrock_error(code or "ClientError", error.get("Message", str(e)))
            from fastapi import HTTPException
            raise HTTPException(status_code=mapped["status"], detail=mapped["body"]) from e

        try:
            message_id = f"msg_{uuid.uuid4().hex}"
            content_block_started = set()
            # Bedrock event order: ... contentBlockStop → messageStop → metadata
            # We must buffer the messageStop payload and only emit message_delta
            # once we have real token counts from the trailing metadata event.
            buffered_stop_reason: Optional[str] = None
            buffered_stop_sequence: Optional[str] = None
            message_stop_seen = False
            message_stop_emitted = False

            while True:
                try:
                    chunk = await reader.next_event()
                except Exception as exc:
                    # A read failed mid-stream (e.g. EventStreamError "Response
                    # ended prematurely"); handled by the branch below.
                    chunk = {"_bedrock_stream_error": exc}
                if chunk is None:
                    break
                if "messageStart" in chunk:
                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": echo_model_name(request),
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
                    stream_exc = chunk["_bedrock_stream_error"]
                    if message_stop_emitted:
                        logger.info(
                            "Bedrock converse stream errored after message_stop; "
                            "ignoring late transport error: %s",
                            stream_exc,
                        )
                        return
                    if message_stop_seen:
                        # Bedrock's messageStop arrived, so the response itself
                        # is complete; only the trailing usage metadata was
                        # lost. Finish the message normally (usage unknown).
                        logger.warning(
                            "Bedrock converse stream failed after messageStop; "
                            "usage metadata unavailable: %s",
                            stream_exc,
                        )
                        for sse in self._anthropic_stop_events(buffered_stop_reason, buffered_stop_sequence):
                            yield sse
                        message_stop_emitted = True
                        return
                    # The upstream response is incomplete. Report the failure
                    # instead of closing the message with end_turn/message_stop,
                    # which clients (and request tracking) read as a complete,
                    # successful response.
                    logger.warning(f"Bedrock stream error mid-response: {stream_exc}")
                    yield self._anthropic_stream_error_event(stream_exc)
                    return  # stream fully terminated; no post-loop fallback needed
                elif "messageStop" in chunk:
                    message_stop_seen = True
                    stop_reason = chunk["messageStop"].get("stopReason")
                    # Buffer stop info — will be emitted once the trailing metadata arrives.
                    buffered_stop_reason = self._convert_bedrock_stop_reason(stop_reason)
                    buffered_stop_sequence = chunk["messageStop"].get("stopSequence")

            if not message_stop_emitted:
                if message_stop_seen:
                    # Complete response whose trailing metadata event never
                    # came: finish it normally (usage unknown).
                    logger.warning(
                        "Bedrock converse stream ended without usage metadata after messageStop"
                    )
                    for sse in self._anthropic_stop_events(buffered_stop_reason, buffered_stop_sequence):
                        yield sse
                else:
                    # Ended before messageStop: the response is truncated, so
                    # don't fabricate a normal end_turn for it.
                    logger.warning("Bedrock converse stream ended before messageStop")
                    yield self._anthropic_stream_error_event(
                        RuntimeError("Bedrock stream ended before the response was complete")
                    )
        finally:
            # Completion, error, cancellation or client disconnect: stop the
            # reader thread (interrupting a blocked read) and free its permit.
            await reader.close()

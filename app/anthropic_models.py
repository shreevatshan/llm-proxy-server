"""Pydantic models for Anthropic Messages API request/response formats.

These models mirror the Anthropic API specification for:
- POST /v1/messages (create a message)
- GET /v1/models (list models)
- SSE streaming events
"""

import logging
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, Tuple
from pydantic import BaseModel, Field


# ==================== Content Block Types ====================

class AnthropicCacheControl(BaseModel):
    """Cache control directive for prompt caching."""
    type: str = "ephemeral"
    ttl: Optional[int] = None


class AnthropicTextBlock(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


class AnthropicImageSource(BaseModel):
    """Image source for image content blocks."""
    type: str = "base64"  # "base64" or "url"
    media_type: Optional[str] = None  # e.g., "image/jpeg", "image/png"
    data: Optional[str] = None  # base64-encoded image data
    url: Optional[str] = None  # URL for url type


class AnthropicImageBlock(BaseModel):
    """Image content block."""
    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):
    """Tool use content block (in assistant responses)."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


class AnthropicToolResultBlock(BaseModel):
    """Tool result content block (in user messages)."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List[Union["AnthropicTextBlock", "AnthropicImageBlock", "AnthropicGenericBlock"]]]] = None
    is_error: Optional[bool] = None
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


class AnthropicThinkingBlock(BaseModel):
    """Thinking content block (extended thinking).
    
    The signature field is required when sending back thinking blocks from
    previous assistant turns in multi-turn conversations. It is returned by
    the API and must be preserved as-is.
    """
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


class AnthropicRedactedThinkingBlock(BaseModel):
    """Redacted thinking content block.
    
    Returned by the API when thinking content is redacted for safety.
    Must be preserved in multi-turn conversations.
    """
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


class AnthropicGenericBlock(BaseModel):
    """Catch-all for unrecognized content block types (e.g., tool_reference).

    Preserves unknown blocks so they pass through to the upstream provider
    without data loss. Must be placed last in any Union so that specific
    typed blocks match first.
    """
    type: str

    class Config:
        extra = "allow"


# AnthropicGenericBlock must be last so specific Literal-typed blocks match first.
AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
    AnthropicGenericBlock,
]


# ==================== Messages ====================

class AnthropicMessage(BaseModel):
    """A message in the Anthropic Messages API format."""
    role: str  # "user" or "assistant"
    content: Union[str, List[AnthropicContentBlock]]


# ==================== Tool Definitions ====================

class AnthropicToolInputSchema(BaseModel):
    """JSON Schema for tool input parameters."""
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    class Config:
        extra = "allow"


class AnthropicTool(BaseModel):
    """Tool definition for the Anthropic API.

    Regular tools have name + description + input_schema.
    Special tools (computer_20250124, code_execution_20250825,
    text_editor_20250124, web_search, etc.) may omit input_schema
    and carry additional fields — extra="allow" lets those pass through.
    """
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[AnthropicToolInputSchema] = None
    type: Optional[str] = None
    cache_control: Optional[AnthropicCacheControl] = None

    class Config:
        extra = "allow"


# ==================== Request ====================

class AnthropicToolChoice(BaseModel):
    """Tool choice configuration."""
    type: str  # "auto", "any", "tool", "none"
    name: Optional[str] = None  # Required when type is "tool"
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicThinkingConfig(BaseModel):
    """Extended thinking configuration."""
    type: str = "enabled"  # "enabled" or "disabled"
    budget_tokens: Optional[int] = None


class AnthropicMetadata(BaseModel):
    """Request metadata."""
    user_id: Optional[str] = None


class AnthropicMessagesRequest(BaseModel):
    """Request body for POST /v1/messages."""
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Any]] = None  # List[Any] to accept all tool types (regular, bash, text_editor, etc.)
    tool_choice: Optional[Union[AnthropicToolChoice, Dict[str, Any]]] = None
    metadata: Optional[AnthropicMetadata] = None
    thinking: Optional[AnthropicThinkingConfig] = None

    class Config:
        extra = "allow"  # Allow extra fields for forward compat


class AnthropicCountTokensRequest(BaseModel):
    """Request body for POST /v1/messages/count_tokens."""
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    tools: Optional[List[Any]] = None  # List[Any] to accept all tool types
    tool_choice: Optional[Union[AnthropicToolChoice, Dict[str, Any]]] = None
    thinking: Optional[AnthropicThinkingConfig] = None

    class Config:
        extra = "allow"


class AnthropicCountTokensResponse(BaseModel):
    """Response body for POST /v1/messages/count_tokens."""
    input_tokens: int


# ==================== Response ====================

class AnthropicUsage(BaseModel):
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessagesResponse(BaseModel):
    """Response body from POST /v1/messages (non-streaming)."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[str] = None  # "end_turn", "max_tokens", "stop_sequence", "tool_use"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

    class Config:
        extra = "allow"


# ==================== Streaming Events ====================

class AnthropicStreamMessageStart(BaseModel):
    """event: message_start"""
    type: str = "message_start"
    message: AnthropicMessagesResponse


class AnthropicStreamContentBlockStart(BaseModel):
    """event: content_block_start"""
    type: str = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class AnthropicStreamTextDelta(BaseModel):
    """Delta for text content blocks."""
    type: Literal["text_delta"] = "text_delta"
    text: str


class AnthropicStreamToolInputDelta(BaseModel):
    """Delta for tool input JSON."""
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class AnthropicStreamThinkingDelta(BaseModel):
    """Delta for thinking content blocks."""
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class AnthropicStreamSignatureDelta(BaseModel):
    """Delta for thinking block signature (sent at end of thinking block)."""
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


AnthropicStreamDelta = Annotated[
    Union[
        AnthropicStreamTextDelta,
        AnthropicStreamToolInputDelta,
        AnthropicStreamThinkingDelta,
        AnthropicStreamSignatureDelta,
    ],
    Field(discriminator="type"),
]


class AnthropicStreamContentBlockDelta(BaseModel):
    """event: content_block_delta"""
    type: str = "content_block_delta"
    index: int
    delta: AnthropicStreamDelta


class AnthropicStreamContentBlockStop(BaseModel):
    """event: content_block_stop"""
    type: str = "content_block_stop"
    index: int


class AnthropicStreamMessageDeltaBody(BaseModel):
    """The delta body within message_delta event."""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamMessageDelta(BaseModel):
    """event: message_delta"""
    type: str = "message_delta"
    delta: AnthropicStreamMessageDeltaBody
    usage: Optional[AnthropicUsage] = None


class AnthropicStreamMessageStop(BaseModel):
    """event: message_stop"""
    type: str = "message_stop"


class AnthropicStreamPing(BaseModel):
    """event: ping"""
    type: str = "ping"


class AnthropicStreamError(BaseModel):
    """event: error"""
    type: str = "error"
    error: Dict[str, Any]


_ANTHROPIC_TERMINAL_STREAM_EVENT_TYPES = frozenset({"message_stop", "error"})


def get_anthropic_stream_event_type(
    event_type: Optional[str] = None,
    event_data: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the Anthropic SSE event type from explicit or payload data."""
    if event_type:
        return event_type
    if isinstance(event_data, dict):
        payload_type = event_data.get("type")
        if isinstance(payload_type, str):
            return payload_type
    return None


def is_anthropic_terminal_stream_event(
    event_type: Optional[str] = None,
    event_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True when an Anthropic SSE event should terminate the stream."""
    return (
        get_anthropic_stream_event_type(event_type=event_type, event_data=event_data)
        in _ANTHROPIC_TERMINAL_STREAM_EVENT_TYPES
    )


# ==================== Model Listing ====================

class AnthropicModelInfo(BaseModel):
    """Model information in Anthropic format."""
    id: str
    type: str = "model"
    display_name: Optional[str] = None
    created_at: Optional[str] = None  # ISO 8601
    provider: Optional[str] = None  # Extension: which provider serves this model
    supported_apis: Optional[List[str]] = None  # Extension: ["openai", "anthropic"]


class AnthropicModelListResponse(BaseModel):
    """Response for GET /v1/models in Anthropic format."""
    data: List[AnthropicModelInfo]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


# ==================== Error Response ====================

class AnthropicErrorDetail(BaseModel):
    """Error detail object."""
    type: str  # "invalid_request_error", "authentication_error", "permission_error", etc.
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response format for Anthropic API."""
    type: str = "error"
    error: AnthropicErrorDetail


# ==================== Shared Helpers ====================


def _strip_signatureless_thinking_blocks(messages: list) -> list:
    """Return serialised messages with signature-less thinking blocks removed.

    The Anthropic API requires a ``signature`` field on every ``thinking``
    block in multi-turn assistant messages.  If the client did not preserve
    the signature (e.g. broken SSE, client bug), we strip those blocks so
    the request can still succeed.
    """
    cleaned: list = []
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            cleaned.append(msg)
            continue
        new_content = []
        for block in msg["content"]:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            btype = block.get("type")
            if btype == "thinking":
                sig = block.get("signature")
                if not isinstance(sig, str) or not sig.strip():
                    continue  # drop this block
            elif btype == "redacted_thinking":
                if not block.get("data"):
                    continue
            new_content.append(block)
        cleaned.append({**msg, "content": new_content})
    return cleaned


_logger = logging.getLogger(__name__)

_VALID_MESSAGE_ROLES = {"user", "assistant"}


def _extract_system_messages_from_messages(
    messages: list,
) -> Tuple[list, List[Dict[str, Any]]]:
    """Return (cleaned_messages, extracted_system_blocks).

    Clients sometimes place role="system" entries inside messages[] (e.g.
    older Claude Code builds that mix OpenAI and Anthropic conventions).
    The Anthropic API rejects these with a 400.  We lift them out into a
    list of Anthropic text-block dicts so the caller can fold them into the
    top-level system field, matching what bedrock_provider's Converse path
    does at line 3219 of that file.

    - role == "system": removed from messages; content converted to text
      blocks preserving cache_control when present.  String content yields
      one synthesized block; list content keeps each text block individually
      (non-text blocks are dropped).
    - role not in {"user","assistant","system"}: coerced to "user" with a
      warning (mirrors Bedrock Converse permissiveness).
    - Other roles pass through unchanged.
    """
    cleaned: list = []
    extracted: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue

        role = msg.get("role")

        if role == "system":
            content = msg.get("content")
            if isinstance(content, str):
                block: Dict[str, Any] = {"type": "text", "text": content}
                if msg.get("cache_control"):
                    block["cache_control"] = msg["cache_control"]
                extracted.append(block)
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") != "text":
                        _logger.debug(
                            "Dropping non-text block from role='system' message during normalization: type=%s",
                            blk.get("type"),
                        )
                        continue
                    extracted_blk: Dict[str, Any] = {"type": "text", "text": blk.get("text", "")}
                    if blk.get("cache_control"):
                        extracted_blk["cache_control"] = blk["cache_control"]
                    extracted.append(extracted_blk)
            # Do not append to cleaned — drop the system-role entry entirely.
            continue

        if role not in _VALID_MESSAGE_ROLES:
            _logger.warning(
                "Unexpected message role %r; coercing to 'user' for Anthropic API compatibility",
                role,
            )
            msg = {**msg, "role": "user"}

        cleaned.append(msg)

    return cleaned, extracted


def _merge_system_fields(
    existing: Any,
    extracted_blocks: List[Dict[str, Any]],
) -> Any:
    """Merge extracted system blocks into the existing top-level system value.

    - None  + []          → None
    - None  + non-empty   → list of extracted blocks
    - str   + []          → str unchanged
    - str   + non-empty   → [text-block(existing), *extracted]
    - list  + []          → list unchanged
    - list  + non-empty   → [*existing, *extracted]
    """
    if not extracted_blocks:
        return existing
    if existing is None:
        return extracted_blocks
    if isinstance(existing, str):
        return [{"type": "text", "text": existing}] + extracted_blocks
    # existing is a list of serialised block dicts
    return list(existing) + extracted_blocks


def build_anthropic_sdk_kwargs(
    request: "AnthropicMessagesRequest",
    model_id: str,
) -> Dict[str, Any]:
    """Build kwargs dict for the Anthropic SDK from a request model.

    Shared by all providers that call the Anthropic SDK (custom providers,
    Bedrock, etc.) to avoid duplicating the serialisation logic.

    Args:
        request: The validated Anthropic Messages request.
        model_id: The provider-specific model ID (already stripped of internal prefix).

    Returns:
        Dict ready to be passed as **kwargs to client.messages.create().
    """
    raw_messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
    raw_messages = _strip_signatureless_thinking_blocks(raw_messages)
    raw_messages, extracted_system = _extract_system_messages_from_messages(raw_messages)

    kwargs: Dict[str, Any] = {
        "model": model_id,
        "max_tokens": request.max_tokens,
        "messages": raw_messages,
    }

    existing_system: Any = None
    if request.system is not None:
        if isinstance(request.system, str):
            existing_system = request.system
        else:
            existing_system = [block.model_dump(exclude_none=True) for block in request.system]

    merged_system = _merge_system_fields(existing_system, extracted_system)
    if merged_system is not None:
        kwargs["system"] = merged_system

    # Simple scalar parameters
    for param in ("temperature", "top_p", "top_k", "stop_sequences"):
        value = getattr(request, param, None)
        if value is not None:
            kwargs[param] = value

    # Complex model parameters
    if request.tools:
        kwargs["tools"] = [
            tool.model_dump(exclude_none=True) if hasattr(tool, 'model_dump') else tool
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = (
            request.tool_choice.model_dump(exclude_none=True)
            if hasattr(request.tool_choice, 'model_dump')
            else request.tool_choice
        )
    if request.metadata is not None:
        kwargs["metadata"] = request.metadata.model_dump(exclude_none=True)
    if request.thinking is not None:
        kwargs["thinking"] = request.thinking.model_dump(exclude_none=True)

    # Forward any extra/unknown fields via extra_body so the Anthropic SDK
    # does not reject them as unexpected kwargs. This preserves forward
    # compatibility for newer fields and third-party gateways.
    extra_fields = getattr(request, "model_extra", None) or {}
    if extra_fields:
        # Avoid duplicating known fields already included in kwargs
        filtered = {k: v for k, v in extra_fields.items() if k not in kwargs and v is not None}
        if filtered:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **filtered}

    return kwargs

"""Shared Anthropic adapter helpers for chat/responses backends."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.anthropic_models import (
    AnthropicMessagesRequest,
    _extract_system_messages_from_messages,
    _merge_system_fields,
)
from app.conversion.anthropic_openai import (
    AnthropicToOpenAIConverter,
    AnthropicToResponsesConverter,
    OpenAIToAnthropicConverter,
    ResponsesToAnthropicConverter,
    StreamConversionState,
    _is_server_tool,
)
from app.providers.base import AnthropicRequestMetadata

_SPECIAL_ADAPTER_TOOL_TYPES = (
    "computer_",
    "text_editor",
    "web_search",
    "code_execution",
    "bash",
)
_SUPPORTED_CONTENT_TYPES = {
    "text",
    "image",
    "tool_use",
    "tool_result",
}
_KNOWN_ANTHROPIC_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "system",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "stream",
    "tools",
    "tool_choice",
    "metadata",
    "thinking",
}
_TOP_LEVEL_ADAPTER_ONLY_FIELDS = {
    "container",
    "context_management",
    "mcp_servers",
    "output_config",
    "service_tier",
}


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _to_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _format_anthropic_error_sse(message: str) -> str:
    payload = {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        },
    }
    return f"event: error\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def prepare_anthropic_adapter_request(
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    anthropic_beta: Optional[str] = None,
) -> Tuple[AnthropicMessagesRequest, AnthropicRequestMetadata]:
    """Normalize an Anthropic request for adapter-backed execution."""
    raw = request.model_dump(exclude_none=True)
    dropped_fields: List[str] = []

    for key in list(raw.keys()):
        if key in _TOP_LEVEL_ADAPTER_ONLY_FIELDS:
            raw.pop(key, None)
            dropped_fields.append(key)
        elif key not in _KNOWN_ANTHROPIC_FIELDS and key not in {"metadata"}:
            raw.pop(key, None)
            dropped_fields.append(key)

    if anthropic_beta:
        dropped_fields.append("anthropic-beta")

    if not allow_thinking and raw.pop("thinking", None) is not None:
        dropped_fields.append("thinking")

    system = raw.get("system")
    if isinstance(system, list):
        normalized_system = []
        for block in system:
            block_data = _to_dict(block)
            if block_data.pop("cache_control", None) is not None:
                dropped_fields.append("system.cache_control")
            normalized_system.append(block_data)
        raw["system"] = normalized_system

    normalized_messages = []
    for message in raw.get("messages", []) or []:
        message_data = _to_dict(message)
        content = message_data.get("content")
        if isinstance(content, list):
            normalized_blocks = []
            for block in content:
                block_data = _to_dict(block)
                block_type = block_data.get("type")

                if block_data.pop("cache_control", None) is not None:
                    dropped_fields.append("messages.cache_control")

                if block_type in {"thinking", "redacted_thinking"}:
                    dropped_fields.append("messages.thinking")
                    continue

                if block_type and block_type not in _SUPPORTED_CONTENT_TYPES:
                    dropped_fields.append(f"messages.{block_type}")
                    continue

                normalized_blocks.append(block_data)
            message_data["content"] = normalized_blocks
        normalized_messages.append(message_data)
    raw["messages"] = normalized_messages

    normalized_tools = []
    for index, tool in enumerate(raw.get("tools", []) or []):
        tool_data = _to_dict(tool)
        tool_type = str(tool_data.get("type") or "")
        tool_name = str(tool_data.get("name") or "")

        if tool_data.pop("cache_control", None) is not None:
            dropped_fields.append("tools.cache_control")

        if _is_server_tool(tool_data) or (
            tool_type and any(tool_type.startswith(prefix) for prefix in _SPECIAL_ADAPTER_TOOL_TYPES)
        ):
            dropped_fields.append(f"tools.{tool_type or tool_name or index}")
            continue

        if not tool_name:
            tool_data["name"] = f"tool_{index + 1}"

        input_schema = tool_data.get("input_schema")
        if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
            input_schema["type"] = "object"
            dropped_fields.append("tools.input_schema.type")

        normalized_tools.append(tool_data)

    if normalized_tools:
        raw["tools"] = normalized_tools
    else:
        raw.pop("tools", None)
        if raw.pop("tool_choice", None) is not None:
            dropped_fields.append("tool_choice")

    # Lift role="system" entries from messages[] into the top-level system field.
    # Some clients mix OpenAI-style system messages into messages[]; the
    # adapter's downstream OpenAI converter only handles user/assistant roles.
    if raw.get("messages"):
        cleaned_messages, extracted_system = _extract_system_messages_from_messages(raw["messages"])
        if extracted_system:
            raw["messages"] = cleaned_messages
            raw["system"] = _merge_system_fields(raw.get("system"), extracted_system)
            dropped_fields.append("messages.system")

    sanitized_request = AnthropicMessagesRequest.model_validate(raw)
    metadata = AnthropicRequestMetadata(
        mode="adapter",
        transport=transport,
        dropped_fields=_dedupe(dropped_fields),
    )
    return sanitized_request, metadata


async def anthropic_adapter_messages(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    anthropic_beta: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a non-streaming adapter-backed Anthropic request."""
    sanitized_request, _ = prepare_anthropic_adapter_request(
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        anthropic_beta=anthropic_beta,
    )
    model_id = provider.get_model_id(sanitized_request.model)

    if transport == "responses":
        converter_in = AnthropicToResponsesConverter()
        converter_out = ResponsesToAnthropicConverter()
        adapted_request = converter_in.convert_request(sanitized_request, model_id)
        adapted_request.stream = False
        response_obj = await provider.responses_create(adapted_request)
        if hasattr(response_obj, "model_dump"):
            response_obj = response_obj.model_dump(exclude_unset=True)
        return converter_out.convert_response(response_obj, request.model)

    converter_in = AnthropicToOpenAIConverter()
    converter_out = OpenAIToAnthropicConverter()
    adapted_request = converter_in.convert_request(sanitized_request, model_id)
    adapted_request.stream = False
    response_obj = await provider.chat_completion(adapted_request)
    if hasattr(response_obj, "model_dump"):
        response_obj = response_obj.model_dump(exclude_unset=True)
    return converter_out.convert_response(response_obj, request.model)


async def anthropic_adapter_messages_stream(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    anthropic_beta: Optional[str] = None,
):
    """Execute a streaming adapter-backed Anthropic request."""
    sanitized_request, _ = prepare_anthropic_adapter_request(
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        anthropic_beta=anthropic_beta,
    )
    model_id = provider.get_model_id(sanitized_request.model)
    state = StreamConversionState(
        message_id=f"msg_{uuid.uuid4().hex}",
        model=request.model,
    )

    if transport == "responses":
        converter_in = AnthropicToResponsesConverter()
        converter_out = ResponsesToAnthropicConverter()
        adapted_request = converter_in.convert_request(sanitized_request, model_id)
        adapted_request.stream = True

        yield converter_out.convert_stream_start(state.message_id, state.model)

        async for sse_chunk in provider.responses_create_stream(adapted_request):
            event_type = None
            event_data = None
            for line in sse_chunk.splitlines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:].strip())
                    except (json.JSONDecodeError, ValueError):
                        event_data = None

            if event_type == "error":
                message = None
                if isinstance(event_data, dict):
                    message = ((event_data.get("error") or {}).get("message")) or event_data.get("message")
                yield _format_anthropic_error_sse(message or "Responses API stream failed")
                return

            if event_type and event_data is not None:
                anthropic_events = converter_out.convert_stream_event(event_type, event_data, state)
                for event in anthropic_events:
                    yield event

        for event in converter_out.convert_stream_end(state):
            yield event
        return

    converter_in = AnthropicToOpenAIConverter()
    converter_out = OpenAIToAnthropicConverter()
    adapted_request = converter_in.convert_request(sanitized_request, model_id)
    adapted_request.stream = True

    yield converter_out.convert_stream_start(state.message_id, state.model)

    async for sse_chunk in provider.chat_completion_stream(adapted_request):
        if sse_chunk.startswith("data: [DONE]"):
            break
        if not sse_chunk.startswith("data: "):
            continue
        try:
            chunk_data = json.loads(sse_chunk[6:].strip())
        except (json.JSONDecodeError, ValueError):
            continue

        if isinstance(chunk_data, dict) and chunk_data.get("error"):
            message = ((chunk_data.get("error") or {}).get("message")) or "Chat completions stream failed"
            yield _format_anthropic_error_sse(message)
            return

        anthropic_events = converter_out.convert_stream_chunk(chunk_data, state)
        for event in anthropic_events:
            yield event

    for event in converter_out.convert_stream_end(state):
        yield event

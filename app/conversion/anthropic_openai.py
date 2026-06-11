"""
Converters between Anthropic Messages API and OpenAI Chat Completions API formats.

Enables any OpenAI-compatible provider (Azure, custom, etc.) to accept
Anthropic-format requests by converting them internally to OpenAI format,
calling the existing chat_completion / chat_completion_stream methods,
and converting the responses back.
"""
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.anthropic_models import (
    AnthropicContentBlock,
    AnthropicGenericBlock,
    AnthropicImageBlock,
    AnthropicMessagesRequest,
    AnthropicRedactedThinkingBlock,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
)
from app.openai_models import ChatCompletionRequest, ResponsesCreateRequest

logger = logging.getLogger(__name__)

# Server-side tool types that should be filtered out (not sent to OpenAI backends)
SERVER_TOOL_NAMES = {"code_execution"}
SERVER_TOOL_NAME_PREFIXES = ("web_search_", "web_fetch_")
SERVER_TOOL_TYPE_PREFIXES = ("computer_",)


def _is_server_tool(tool_dict: Dict[str, Any]) -> bool:
    name = tool_dict.get("name", "")
    tool_type = tool_dict.get("type", "")
    if name in SERVER_TOOL_NAMES:
        return True
    for prefix in SERVER_TOOL_NAME_PREFIXES:
        if name.startswith(prefix):
            return True
    for prefix in SERVER_TOOL_TYPE_PREFIXES:
        if tool_type.startswith(prefix):
            return True
    return False


def _format_sse(event_type: str, data: dict) -> str:
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {json_str}\n\n"


# ---------------------------------------------------------------------------
# Anthropic → OpenAI request conversion
# ---------------------------------------------------------------------------

class AnthropicToOpenAIConverter:
    """Converts an AnthropicMessagesRequest into a ChatCompletionRequest."""

    def convert_request(
        self,
        request: AnthropicMessagesRequest,
        model_id: str,
    ) -> ChatCompletionRequest:
        messages: List[Dict[str, Any]] = []

        # System prompt → system message
        if request.system:
            system_text = self._convert_system(request.system)
            if system_text:
                messages.append({"role": "system", "content": system_text})

        # Conversation messages
        for msg in request.messages:
            converted = self._convert_message(msg.role, msg.content)
            messages.extend(converted)

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": request.stream or False,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        # Tools
        if request.tools:
            openai_tools = self._convert_tools(request.tools)
            if openai_tools:
                kwargs["tools"] = openai_tools

        # Tool choice
        if request.tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        # Thinking → reasoning_effort
        if request.thinking and getattr(request.thinking, "type", None) == "enabled":
            kwargs["reasoning_effort"] = "high"

        # Metadata user_id → OpenAI user (max 64 chars per OpenAI API)
        if request.metadata and getattr(request.metadata, "user_id", None):
            kwargs["user"] = request.metadata.user_id[:64]

        return ChatCompletionRequest(**kwargs)

    # -- System --

    def _convert_system(self, system: Any) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, AnthropicTextBlock):
                    parts.append(block.text)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return ""

    # -- Messages --

    def _convert_message(self, role: str, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": role, "content": content}]
        if not isinstance(content, list):
            return [{"role": role, "content": str(content)}]
        if role == "user":
            return self._convert_user_blocks(content)
        elif role == "assistant":
            return self._convert_assistant_blocks(content)
        return [{"role": role, "content": str(content)}]

    def _convert_user_blocks(self, blocks: list) -> List[Dict[str, Any]]:
        content_parts: List[Dict[str, Any]] = []
        tool_messages: List[Dict[str, Any]] = []

        for block in blocks:
            block_dict = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            block_type = block_dict.get("type", "") if isinstance(block_dict, dict) else getattr(block, "type", "")

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block_dict.get("text", "")
                content_parts.append({"type": "text", "text": text})

            elif block_type == "image":
                content_parts.append(self._convert_image_block(block, block_dict))

            elif block_type == "tool_result":
                tool_messages.append(self._convert_tool_result(block, block_dict))

            elif block_type in ("thinking", "redacted_thinking"):
                # Thinking blocks in user messages (from multi-turn) — drop
                pass

            else:
                logger.warning("Dropping unknown user content block type: %s", block_type)

        result: List[Dict[str, Any]] = []
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                result.append({"role": "user", "content": content_parts[0]["text"]})
            else:
                result.append({"role": "user", "content": content_parts})
        result.extend(tool_messages)
        return result

    def _convert_image_block(self, block: Any, block_dict: dict) -> Dict[str, Any]:
        if isinstance(block, AnthropicImageBlock):
            source = block.source
            source_type = source.type
            media_type = source.media_type or "image/png"
            if source_type == "url" and source.url:
                url = source.url
            else:
                url = f"data:{media_type};base64,{source.data or ''}"
        else:
            source = block_dict.get("source", {})
            source_type = source.get("type", "base64")
            media_type = source.get("media_type", "image/png")
            if source_type == "url" and source.get("url"):
                url = source["url"]
            else:
                url = f"data:{media_type};base64,{source.get('data', '')}"

        return {"type": "image_url", "image_url": {"url": url}}

    def _convert_tool_result(self, block: Any, block_dict: dict) -> Dict[str, Any]:
        if isinstance(block, AnthropicToolResultBlock):
            tool_call_id = block.tool_use_id
            content = block.content
        else:
            tool_call_id = block_dict.get("tool_use_id", "")
            content = block_dict.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, AnthropicTextBlock):
                    parts.append(item.text)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(item.text)
            text = "\n".join(parts) if parts else ""
        else:
            text = str(content) if content else ""

        return {"role": "tool", "tool_call_id": tool_call_id, "content": text}

    def _convert_assistant_blocks(self, blocks: list) -> List[Dict[str, Any]]:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in blocks:
            block_dict = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            block_type = block_dict.get("type", "") if isinstance(block_dict, dict) else getattr(block, "type", "")

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block_dict.get("text", "")
                if text:
                    text_parts.append(text)

            elif block_type == "tool_use":
                tc_id = block.id if hasattr(block, "id") else block_dict.get("id", "")
                tc_name = block.name if hasattr(block, "name") else block_dict.get("name", "")
                tc_input = block.input if hasattr(block, "input") else block_dict.get("input", {})
                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": json.dumps(tc_input),
                    },
                })

            elif block_type in ("thinking", "redacted_thinking"):
                # Thinking blocks from previous assistant turns — drop for OpenAI
                pass

            else:
                logger.warning("Dropping unknown assistant content block type: %s", block_type)

        msg: Dict[str, Any] = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    # -- Tools --

    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        openai_tools: List[Dict[str, Any]] = []
        for tool in tools:
            tool_dict = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else (
                tool if isinstance(tool, dict) else {}
            )
            if _is_server_tool(tool_dict):
                continue
            # Special tools without input_schema are server-side, skip
            if not tool_dict.get("input_schema") and tool_dict.get("type"):
                continue

            name = tool_dict.get("name", "")
            description = tool_dict.get("description", "")
            input_schema = tool_dict.get("input_schema", {})
            if isinstance(input_schema, dict):
                parameters = input_schema
            elif hasattr(input_schema, "model_dump"):
                parameters = input_schema.model_dump(exclude_none=True)
            else:
                parameters = {}

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return openai_tools

    # -- Tool choice --

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        if isinstance(tool_choice, str):
            return "required" if tool_choice == "any" else tool_choice

        tc_dict = tool_choice.model_dump(exclude_none=True) if hasattr(tool_choice, "model_dump") else (
            tool_choice if isinstance(tool_choice, dict) else {}
        )
        tc_type = tc_dict.get("type", "")
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any":
            return "required"
        elif tc_type == "none":
            return "none"
        elif tc_type == "tool":
            return {"type": "function", "function": {"name": tc_dict.get("name", "")}}
        return "auto"


# ---------------------------------------------------------------------------
# Streaming state
# ---------------------------------------------------------------------------

@dataclass
class StreamConversionState:
    """Mutable state tracked across streaming chunks for OpenAI → Anthropic conversion."""
    message_id: str
    model: str
    content_block_index: int = -1  # -1 means no block opened yet
    current_block_type: Optional[str] = None  # "text", "tool_use", "thinking"
    tool_call_indices: Dict[int, int] = field(default_factory=dict)  # OpenAI tc index → our block index
    finish_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    had_tool_use: bool = False  # True if any tool_use block was opened during the stream
    error_message: Optional[str] = None  # Error message from upstream API


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response conversion
# ---------------------------------------------------------------------------

STOP_REASON_MAP: Dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


class OpenAIToAnthropicConverter:
    """Converts OpenAI ChatCompletion responses to Anthropic Messages format."""

    # -- Non-streaming --

    def convert_response(self, openai_response: dict, original_model: str) -> dict:
        choices = openai_response.get("choices", [{}])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})

        content: List[Dict[str, Any]] = []

        # Reasoning/thinking content
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning, "signature": ""})

        # Text content
        text = message.get("content")
        if text:
            content.append({"type": "text", "text": text})

        # Tool calls
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                arguments_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": arguments,
                })

        if not content:
            content.append({"type": "text", "text": ""})

        # Stop reason
        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        # Usage
        openai_usage = openai_response.get("usage", {})
        usage = {
            "input_tokens": openai_usage.get("prompt_tokens", 0),
            "output_tokens": openai_usage.get("completion_tokens", 0),
        }

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    # -- Streaming helpers --

    def convert_stream_start(self, message_id: str, model: str) -> str:
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return _format_sse("message_start", message_start) + _format_sse("ping", {"type": "ping"})

    def convert_stream_chunk(
        self, chunk_data: dict, state: StreamConversionState
    ) -> List[str]:
        events: List[str] = []
        choices = chunk_data.get("choices", [])
        if not choices:
            # May contain only usage
            usage = chunk_data.get("usage")
            if usage:
                state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
                state.output_tokens = usage.get("completion_tokens", state.output_tokens)
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Reasoning/thinking content
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            if state.current_block_type != "thinking":
                events.extend(self._open_block(state, "thinking", {
                    "type": "thinking", "thinking": "",
                }))
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            }))

        # Text content
        text = delta.get("content")
        if text:
            if state.current_block_type != "text":
                events.extend(self._open_block(state, "text", {
                    "type": "text", "text": "",
                }))
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {"type": "text_delta", "text": text},
            }))

        # Tool calls
        tc_list = delta.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id")
                tc_name = func.get("name")

                if tc_id and tc_name:
                    # New tool call — close previous block, open tool_use block
                    events.extend(self._open_block(state, "tool_use", {
                        "type": "tool_use",
                        "id": tc_id,
                        "name": tc_name,
                        "input": {},
                    }))
                    state.tool_call_indices[tc_index] = state.content_block_index

                # Argument fragments
                args_chunk = func.get("arguments")
                if args_chunk:
                    block_idx = state.tool_call_indices.get(tc_index, state.content_block_index)
                    events.append(_format_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "input_json_delta", "partial_json": args_chunk},
                    }))

        # Finish reason
        if finish_reason:
            state.finish_reason = finish_reason

        # Usage (may come in the same or a later chunk)
        usage = choice.get("usage") or chunk_data.get("usage")
        if usage:
            state.input_tokens = usage.get("prompt_tokens", state.input_tokens)
            state.output_tokens = usage.get("completion_tokens", state.output_tokens)

        return events

    def convert_stream_end(self, state: StreamConversionState) -> List[str]:
        events: List[str] = []

        # Close any open content block
        if state.content_block_index >= 0:
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": state.content_block_index,
            }))

        # If no blocks were ever opened, emit an empty text block
        if state.content_block_index < 0:
            state.content_block_index = 0
            events.append(_format_sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }))
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }))

        stop_reason = STOP_REASON_MAP.get(state.finish_reason or "stop", "end_turn")
        events.append(_format_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.output_tokens},
        }))
        events.append(_format_sse("message_stop", {"type": "message_stop"}))

        return events

    # -- Internal helpers --

    def _open_block(
        self, state: StreamConversionState, block_type: str, content_block: dict
    ) -> List[str]:
        events: List[str] = []
        # Close previous block if one is open
        if state.content_block_index >= 0:
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": state.content_block_index,
            }))
        state.content_block_index += 1
        state.current_block_type = block_type
        events.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": state.content_block_index,
            "content_block": content_block,
        }))
        return events


def _is_codex_model(model_name: str) -> bool:
    """Check if a model name indicates a codex/responses-api model."""
    return "codex" in model_name.lower()


# ---------------------------------------------------------------------------
# Anthropic → OpenAI Responses API conversion (for codex models)
# ---------------------------------------------------------------------------

class AnthropicToResponsesConverter:
    """Converts an AnthropicMessagesRequest into a ResponsesCreateRequest.

    The Responses API uses:
    - ``input``: list of message items (``{"type": "message", "role": ..., "content": ...}``)
    - ``instructions``: system prompt
    - ``tools``: list of ``{"type": "function", "name": ..., "parameters": ..., "description": ...}``
    - ``max_output_tokens`` instead of ``max_tokens``
    """

    def __init__(self):
        # Mapping from original tool-use IDs (call_*, toolu_*) to fc_* IDs
        # required by Azure codex models via the Responses API.
        self._id_map: Dict[str, str] = {}

    def _ensure_fc_id(self, original_id: str) -> str:
        """Ensure an ID starts with 'fc_' for Responses API compatibility."""
        if original_id.startswith("fc"):
            return original_id
        if original_id in self._id_map:
            return self._id_map[original_id]
        new_id = f"fc_{uuid.uuid4().hex[:24]}"
        self._id_map[original_id] = new_id
        return new_id

    def convert_request(
        self,
        request: AnthropicMessagesRequest,
        model_id: str,
    ) -> ResponsesCreateRequest:
        # System prompt → instructions
        instructions = None
        if request.system:
            instructions = self._convert_system(request.system)

        # Build input items
        input_items: List[Dict[str, Any]] = []
        for msg in request.messages:
            items = self._convert_message(msg.role, msg.content)
            input_items.extend(items)

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "input": input_items,
            "max_output_tokens": request.max_tokens,
            "stream": request.stream or False,
        }

        if instructions:
            kwargs["instructions"] = instructions
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Tools (Responses API format)
        if request.tools:
            resp_tools = self._convert_tools(request.tools)
            if resp_tools:
                kwargs["tools"] = resp_tools

        # Tool choice
        if request.tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice(request.tool_choice)

        # Thinking → reasoning
        if request.thinking and getattr(request.thinking, "type", None) == "enabled":
            kwargs["reasoning"] = {"effort": "high"}

        # Metadata user_id → user (max 64 chars per OpenAI API)
        if request.metadata and getattr(request.metadata, "user_id", None):
            kwargs["user"] = request.metadata.user_id[:64]

        return ResponsesCreateRequest(**kwargs)

    def _convert_system(self, system: Any) -> str:
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, AnthropicTextBlock):
                    parts.append(block.text)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return ""

    def _convert_message(self, role: str, content: Any) -> List[Dict[str, Any]]:
        """Convert a single Anthropic message into Responses API input items.

        Each input item is: {"type": "message", "role": ..., "content": ...}
        Tool results become: {"type": "function_call_output", "call_id": ..., "output": ...}
        """
        if isinstance(content, str):
            return [{"type": "message", "role": role, "content": content}]

        if not isinstance(content, list):
            return [{"type": "message", "role": role, "content": str(content)}]

        if role == "user":
            return self._convert_user_blocks(content)
        elif role == "assistant":
            return self._convert_assistant_blocks(content)
        return [{"type": "message", "role": role, "content": str(content)}]

    def _convert_user_blocks(self, blocks: list) -> List[Dict[str, Any]]:
        content_parts: List[Any] = []
        extra_items: List[Dict[str, Any]] = []

        for block in blocks:
            block_dict = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            block_type = block_dict.get("type", "") if isinstance(block_dict, dict) else getattr(block, "type", "")

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block_dict.get("text", "")
                content_parts.append({"type": "input_text", "text": text})

            elif block_type == "image":
                content_parts.append(self._convert_image_block(block, block_dict))

            elif block_type == "tool_result":
                extra_items.append(self._convert_tool_result(block, block_dict))

            elif block_type in ("thinking", "redacted_thinking"):
                pass

        result: List[Dict[str, Any]] = []
        if content_parts:
            if len(content_parts) == 1 and content_parts[0].get("type") == "input_text":
                result.append({"type": "message", "role": "user", "content": content_parts[0]["text"]})
            else:
                result.append({"type": "message", "role": "user", "content": content_parts})
        result.extend(extra_items)
        return result

    def _convert_image_block(self, block: Any, block_dict: dict) -> Dict[str, Any]:
        if isinstance(block, AnthropicImageBlock):
            source = block.source
            media_type = source.media_type or "image/png"
            if source.type == "url" and source.url:
                return {"type": "input_image", "image_url": source.url}
            else:
                url = f"data:{media_type};base64,{source.data or ''}"
                return {"type": "input_image", "image_url": url}
        else:
            source = block_dict.get("source", {})
            media_type = source.get("media_type", "image/png")
            if source.get("type") == "url" and source.get("url"):
                return {"type": "input_image", "image_url": source["url"]}
            else:
                url = f"data:{media_type};base64,{source.get('data', '')}"
                return {"type": "input_image", "image_url": url}

    def _convert_tool_result(self, block: Any, block_dict: dict) -> Dict[str, Any]:
        if isinstance(block, AnthropicToolResultBlock):
            call_id = block.tool_use_id
            content = block.content
        else:
            call_id = block_dict.get("tool_use_id", "")
            content = block_dict.get("content", "")

        # Remap to fc_ ID (must match the function_call it references)
        fc_id = self._ensure_fc_id(call_id)

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, AnthropicTextBlock):
                    parts.append(item.text)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif hasattr(item, "text"):
                    parts.append(item.text)
            text = "\n".join(parts) if parts else ""
        else:
            text = str(content) if content else ""

        return {"type": "function_call_output", "call_id": fc_id, "output": text}

    def _convert_assistant_blocks(self, blocks: list) -> List[Dict[str, Any]]:
        """Convert assistant blocks into Responses API input items.

        Text → message item, tool_use → function_call item.
        """
        items: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        for block in blocks:
            block_dict = block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            block_type = block_dict.get("type", "") if isinstance(block_dict, dict) else getattr(block, "type", "")

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block_dict.get("text", "")
                if text:
                    text_parts.append(text)

            elif block_type == "tool_use":
                # Flush accumulated text first
                if text_parts:
                    items.append({"type": "message", "role": "assistant", "content": "\n".join(text_parts)})
                    text_parts = []
                tc_id = block.id if hasattr(block, "id") else block_dict.get("id", "")
                tc_name = block.name if hasattr(block, "name") else block_dict.get("name", "")
                tc_input = block.input if hasattr(block, "input") else block_dict.get("input", {})
                fc_id = self._ensure_fc_id(tc_id)
                items.append({
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": fc_id,
                    "name": tc_name,
                    "arguments": json.dumps(tc_input),
                })

            elif block_type in ("thinking", "redacted_thinking"):
                pass

        if text_parts:
            items.append({"type": "message", "role": "assistant", "content": "\n".join(text_parts)})

        return items

    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        resp_tools: List[Dict[str, Any]] = []
        for tool in tools:
            tool_dict = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else (
                tool if isinstance(tool, dict) else {}
            )
            if _is_server_tool(tool_dict):
                continue
            if not tool_dict.get("input_schema") and tool_dict.get("type"):
                continue

            name = tool_dict.get("name", "")
            description = tool_dict.get("description", "")
            input_schema = tool_dict.get("input_schema", {})
            if isinstance(input_schema, dict):
                parameters = input_schema
            elif hasattr(input_schema, "model_dump"):
                parameters = input_schema.model_dump(exclude_none=True)
            else:
                parameters = {}

            resp_tools.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            })
        return resp_tools

    def _convert_tool_choice(self, tool_choice: Any) -> Any:
        tc_dict = tool_choice.model_dump(exclude_none=True) if hasattr(tool_choice, "model_dump") else (
            tool_choice if isinstance(tool_choice, dict) else {}
        )
        tc_type = tc_dict.get("type", "") if isinstance(tc_dict, dict) else str(tool_choice)
        if tc_type == "auto":
            return "auto"
        elif tc_type == "any":
            return "required"
        elif tc_type == "none":
            return "none"
        elif tc_type == "tool":
            return {"type": "function", "name": tc_dict.get("name", "")}
        return "auto"


# ---------------------------------------------------------------------------
# Responses API → Anthropic response conversion
# ---------------------------------------------------------------------------

RESPONSES_STATUS_TO_STOP_REASON: Dict[str, str] = {
    "completed": "end_turn",
    "incomplete": "max_tokens",
    "failed": "end_turn",
}


def _format_anthropic_error_sse(error_type: str, message: str) -> str:
    """Emit a single Anthropic SSE error event."""
    return _format_sse("error", {
        "type": "error",
        "error": {"type": error_type, "message": message},
    })


class ResponsesToAnthropicConverter:
    """Converts OpenAI Responses API output to Anthropic Messages format."""

    # -- Stream start (mirrors OpenAIToAnthropicConverter.convert_stream_start) --

    def convert_stream_start(self, message_id: str, model: str) -> str:
        message_start = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return _format_sse("message_start", message_start) + _format_sse("ping", {"type": "ping"})

    # -- Non-streaming --

    def convert_response(self, response_obj: dict, original_model: str) -> dict:
        content: List[Dict[str, Any]] = []

        # Extract output items
        output = response_obj.get("output", []) or []
        for item in output:
            item_type = item.get("type", "")

            if item_type == "message":
                # Message output item — extract text content
                msg_content = item.get("content", [])
                if isinstance(msg_content, str):
                    content.append({"type": "text", "text": msg_content})
                elif isinstance(msg_content, list):
                    for part in msg_content:
                        if isinstance(part, dict):
                            if part.get("type") == "output_text":
                                content.append({"type": "text", "text": part.get("text", "")})
                            elif part.get("type") == "refusal":
                                content.append({"type": "text", "text": part.get("refusal", "")})
                        elif isinstance(part, str):
                            content.append({"type": "text", "text": part})

            elif item_type == "function_call":
                arguments_str = item.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                content.append({
                    "type": "tool_use",
                    "id": item.get("call_id", item.get("id", f"toolu_{uuid.uuid4().hex[:24]}")),
                    "name": item.get("name", ""),
                    "input": arguments,
                })

            elif item_type == "reasoning":
                # Reasoning summary items
                summaries = item.get("summary", [])
                if isinstance(summaries, list):
                    for s in summaries:
                        if isinstance(s, dict) and s.get("type") == "summary_text":
                            content.append({"type": "thinking", "thinking": s.get("text", ""), "signature": ""})

        # Also check output_text shorthand
        if not content:
            output_text = response_obj.get("output_text")
            if output_text:
                content.append({"type": "text", "text": output_text})

        if not content:
            content.append({"type": "text", "text": ""})

        # Stop reason from status
        status = response_obj.get("status", "completed")
        # Check if any output items are function_calls
        has_tool_use = any(c.get("type") == "tool_use" for c in content)
        if has_tool_use and status == "completed":
            stop_reason = "tool_use"
        else:
            stop_reason = RESPONSES_STATUS_TO_STOP_REASON.get(status, "end_turn")

        # Usage
        resp_usage = response_obj.get("usage", {}) or {}
        usage = {
            "input_tokens": resp_usage.get("input_tokens", 0),
            "output_tokens": resp_usage.get("output_tokens", 0),
        }

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    # -- Streaming --

    def convert_stream_event(
        self, event_type: str, event_data: dict, state: StreamConversionState
    ) -> List[str]:
        """Convert a single Responses API SSE event to Anthropic SSE events."""
        events: List[str] = []

        if event_type == "response.output_text.delta":
            delta_text = event_data.get("delta", "")
            if delta_text:
                if state.current_block_type != "text":
                    events.extend(self._open_block(state, "text", {"type": "text", "text": ""}))
                events.append(_format_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state.content_block_index,
                    "delta": {"type": "text_delta", "text": delta_text},
                }))

        elif event_type == "response.function_call_arguments.delta":
            delta_args = event_data.get("delta", "")
            if delta_args:
                # Should already have an open tool_use block
                events.append(_format_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": state.content_block_index,
                    "delta": {"type": "input_json_delta", "partial_json": delta_args},
                }))

        elif event_type == "response.output_item.added":
            item = event_data.get("item", {})
            item_type = item.get("type", "")
            if item_type == "function_call":
                state.had_tool_use = True
                events.extend(self._open_block(state, "tool_use", {
                    "type": "tool_use",
                    "id": item.get("call_id", item.get("id", f"toolu_{uuid.uuid4().hex[:24]}")),
                    "name": item.get("name", ""),
                    "input": {},
                }))
            elif item_type == "message":
                # Text output message — we'll get deltas for the content
                pass

        elif event_type == "response.output_text.done":
            # Text block finished — will be closed when next block opens or at stream end
            pass

        elif event_type == "response.function_call_arguments.done":
            # Tool call args finished
            pass

        elif event_type == "response.output_item.done":
            # Item done — nothing to emit, close happens at next block or end
            pass

        elif event_type == "response.completed":
            resp = event_data.get("response", {})
            resp_usage = resp.get("usage", {})
            state.input_tokens = resp_usage.get("input_tokens", state.input_tokens)
            state.output_tokens = resp_usage.get("output_tokens", state.output_tokens)
            # Determine stop reason
            status = resp.get("status", "completed")
            state.finish_reason = status

        elif event_type == "response.failed":
            state.finish_reason = "failed"
            # Extract error details from the response payload
            resp = event_data.get("response", {})
            err = resp.get("error", {})
            if err:
                state.error_message = err.get("message") or json.dumps(err, ensure_ascii=False)
            else:
                state.error_message = "Upstream API request failed"

        elif event_type == "response.incomplete":
            state.finish_reason = "incomplete"

        elif event_type == "error":
            # Error event from the provider's responses_create_stream error handler
            err = event_data.get("error", {})
            msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
            state.error_message = msg
            state.finish_reason = "failed"
            events.append(_format_anthropic_error_sse("api_error", msg))

        return events

    def convert_stream_end(self, state: StreamConversionState) -> List[str]:
        events: List[str] = []

        if state.content_block_index >= 0:
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": state.content_block_index,
            }))

        if state.content_block_index < 0:
            state.content_block_index = 0
            events.append(_format_sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }))
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }))

        status = state.finish_reason or "completed"
        has_tool_use = (
            state.had_tool_use
            or state.current_block_type == "tool_use"
            or any(v for v in state.tool_call_indices.values())
        )
        if has_tool_use and status == "completed":
            stop_reason = "tool_use"
        else:
            stop_reason = RESPONSES_STATUS_TO_STOP_REASON.get(status, "end_turn")

        # If the stream failed, emit an error event before closing
        if status == "failed" and state.error_message:
            events.append(_format_anthropic_error_sse("api_error", state.error_message))

        events.append(_format_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": state.output_tokens},
        }))
        events.append(_format_sse("message_stop", {"type": "message_stop"}))

        return events

    def _open_block(
        self, state: StreamConversionState, block_type: str, content_block: dict
    ) -> List[str]:
        events: List[str] = []
        if state.content_block_index >= 0:
            events.append(_format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": state.content_block_index,
            }))
        state.content_block_index += 1
        state.current_block_type = block_type
        events.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": state.content_block_index,
            "content_block": content_block,
        }))
        return events

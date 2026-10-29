"""
Converters between Anthropic Messages API and OpenAI Chat Completions API formats.

Enables any OpenAI-compatible provider (Azure, custom, etc.) to accept
Anthropic-format requests by converting them internally to OpenAI format,
calling the existing chat_completion / chat_completion_stream methods,
and converting the responses back.
"""
import base64
import binascii
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
from app.conversion.anthropic_helpers import (
    MAX_STOP_SEQUENCES,
    ThinkTagSplitter,
    ToolNameMap,
    anthropic_usage_from_openai,
    citations_from_annotations,
    matched_stop_sequence,
    parse_tool_arguments,
    reasoning_effort_for,
    reasoning_text_from_message,
    response_format_for,
    restore_tool_id,
    sanitize_tool_id,
    split_think_tags,
    strip_billing_header,
    tool_id_signature,
    tool_id_with_signature,
)
from app.openai_models import ChatCompletionRequest, ResponsesCreateRequest
from app.websearch.tools import INTERNAL_TOOL_NAMES

logger = logging.getLogger(__name__)

# Server-side tool types that should be filtered out (not sent to OpenAI backends)
SERVER_TOOL_NAMES = {"code_execution"}
SERVER_TOOL_NAME_PREFIXES = ("web_search_", "web_fetch_")
SERVER_TOOL_TYPE_PREFIXES = ("computer_",)

# Google's documented stand-in for tool calls it did not sign itself (history
# from another model, or ids minted before signatures were carried).
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def _is_server_tool(tool_dict: Dict[str, Any]) -> bool:
    name = tool_dict.get("name", "")
    tool_type = tool_dict.get("type", "")
    # The web search interceptor's own function tool ("web_search_proxy")
    # shares the "web_search_" prefix but is a plain client-side function.
    if name in INTERNAL_TOOL_NAMES:
        return False
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
# Block helpers
# ---------------------------------------------------------------------------

_FOREIGN_SERVER_RESULT_TYPES = {
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "tool_search_tool_result",
}


def _block_dict(block: Any) -> Dict[str, Any]:
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if isinstance(block, dict):
        return block
    return {}


def _thought_signature(tc: Dict[str, Any]) -> Optional[str]:
    """Gemini's ``extra_content.google.thought_signature`` on a tool call."""
    google = (tc.get("extra_content") or {}).get("google") or {}
    signature = google.get("thought_signature") if isinstance(google, dict) else None
    return signature if isinstance(signature, str) and signature else None


def _signature_extra(signature: str) -> Dict[str, Any]:
    return {"google": {"thought_signature": signature}}


def _text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _image_part_from_source(source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_type = source.get("type", "base64")
    if source_type == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    if source_type == "base64" and source.get("data"):
        media_type = source.get("media_type") or "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{source['data']}"}}
    return None


def _search_result_text(block: Dict[str, Any]) -> str:
    lines = []
    if block.get("title"):
        lines.append(f"Search result: {block['title']}")
    if block.get("source"):
        lines.append(f"Source: {block['source']}")
    content = block.get("content")
    if isinstance(content, str):
        lines.append(content)
    elif isinstance(content, list):
        lines.extend(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return "\n".join(line for line in lines if line)


def _flatten_server_tool_turn(use: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]]) -> str:
    """Summarize a foreign server-tool call/result pair as plain text.

    Encrypted search content can't be decoded, so only titles/URLs survive.
    """
    name = (use or {}).get("name") or "server_tool"
    query = ((use or {}).get("input") or {}).get("query")
    lines = [f"[{name}: {query}]" if query else f"[{name}]"]
    content = (result or {}).get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or ""
            url = item.get("url") or ""
            if title or url:
                lines.append(f"- {title} ({url})" if title and url else f"- {title or url}")
    elif isinstance(content, dict) and content.get("error_code"):
        lines.append(f"(failed: {content['error_code']})")
    return "\n".join(lines)


def _merge_consecutive_user_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent user messages; some chat templates require alternation."""
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        prev = merged[-1] if merged else None
        if prev is not None and prev.get("role") == "user" and msg.get("role") == "user":
            a, b = prev.get("content"), msg.get("content")
            if isinstance(a, str) and isinstance(b, str):
                prev["content"] = f"{a}\n\n{b}" if a and b else (a or b)
            else:
                parts_a = [_text_part(a)] if isinstance(a, str) else list(a or [])
                parts_b = [_text_part(b)] if isinstance(b, str) else list(b or [])
                prev["content"] = [p for p in parts_a + parts_b if not (p.get("type") == "text" and not p.get("text"))]
            continue
        merged.append(msg)
    return merged


# ---------------------------------------------------------------------------
# Anthropic → OpenAI request conversion
# ---------------------------------------------------------------------------

class AnthropicToOpenAIConverter:
    """Converts an AnthropicMessagesRequest into a ChatCompletionRequest.

    After ``convert_request`` the instance exposes ``tool_names`` (the
    Anthropic <-> OpenAI tool-name map the response side must reverse).

    Options:
        echo_reasoning: send prior assistant ``thinking`` back as
            ``reasoning_content`` (DeepSeek/vLLM/Kimi-style backends need it;
            strict OpenAI/Azure/Google endpoints reject the field).
        supports_top_k: forward ``top_k`` via ``extra_body``.
        supports_file_parts: send binary documents (PDF etc.) as OpenAI
            ``file`` content parts; otherwise they become a text placeholder
            (Gemini's OpenAI layer, vLLM, Ollama reject the part type).
        thought_signatures: send Gemini thought signatures back on tool
            calls as ``extra_content`` (Gemini rejects calls without one).
    """

    def __init__(
        self,
        *,
        echo_reasoning: bool = False,
        supports_top_k: bool = True,
        supports_file_parts: bool = True,
        thought_signatures: bool = False,
    ):
        self.echo_reasoning = echo_reasoning
        self.supports_top_k = supports_top_k
        self.supports_file_parts = supports_file_parts
        self.thought_signatures = thought_signatures
        self.tool_names = ToolNameMap()

    def convert_request(
        self,
        request: AnthropicMessagesRequest,
        model_id: str,
    ) -> ChatCompletionRequest:
        self.tool_names = ToolNameMap()
        messages: List[Dict[str, Any]] = []

        # System prompt → system message
        if request.system:
            system_text = strip_billing_header(self._convert_system(request.system))
            if system_text.strip():
                messages.append({"role": "system", "content": system_text})

        # Conversation messages
        for msg in request.messages:
            converted = self._convert_message(msg.role, msg.content)
            messages.extend(converted)
        messages = _merge_consecutive_user_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": request.stream or False,
        }

        # Ask OpenAI backends to emit a final usage chunk when streaming;
        # without this the converted Anthropic stream reports 0 output tokens.
        if kwargs["stream"]:
            kwargs["stream_options"] = {"include_usage": True}

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.top_k is not None and self.supports_top_k:
            # The OpenAI SDK has no top_k kwarg; most OpenAI-compatible
            # servers (vLLM, Ollama, llama.cpp, LM Studio, OpenRouter) read it
            # from the body.
            kwargs["extra_body"] = {"top_k": request.top_k}
        if request.stop_sequences:
            kwargs["stop"] = list(request.stop_sequences)[:MAX_STOP_SEQUENCES]

        # Tools
        openai_tools: List[Dict[str, Any]] = []
        if request.tools:
            openai_tools = self._convert_tools(request.tools)
            if openai_tools:
                kwargs["tools"] = openai_tools

        # Tool choice
        if request.tool_choice is not None and openai_tools:
            tool_choice, parallel = self._convert_tool_choice(request.tool_choice)
            declared = {t["function"]["name"] for t in openai_tools}
            if isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name") not in declared:
                # The forced tool was dropped (e.g. an unemulated server tool).
                tool_choice = "auto"
            kwargs["tool_choice"] = tool_choice
            if parallel is False:
                kwargs["parallel_tool_calls"] = False

        # Thinking / output_config.effort → reasoning_effort
        output_config = getattr(request, "output_config", None)
        effort = reasoning_effort_for(request.thinking, output_config)
        if effort:
            kwargs["reasoning_effort"] = effort

        # output_config.format → response_format (json_schema)
        response_format = response_format_for(output_config)
        if response_format:
            kwargs["response_format"] = response_format

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
        blocks = [_block_dict(block) for block in content]
        if role == "user":
            return self._convert_user_blocks(blocks)
        elif role == "assistant":
            return self._convert_assistant_blocks(blocks)
        return [{"role": role, "content": str(content)}]

    def _content_parts(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert one user-side content block into OpenAI content parts."""
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "")
            return [_text_part(text)] if text else []
        if block_type == "image":
            part = _image_part_from_source(block.get("source") or {})
            if part is not None:
                return [part]
            file_id = (block.get("source") or {}).get("file_id")
            return [_text_part(f"[image {file_id} is not available to this model]" if file_id
                               else "[image is not available to this model]")]
        if block_type == "document":
            return self._document_parts(block)
        if block_type == "search_result":
            text = _search_result_text(block)
            return [_text_part(text)] if text else []
        if block_type in ("thinking", "redacted_thinking"):
            return []
        if block_type == "tool_reference":
            # A tool-search hit: the adapter has already declared the tool.
            name = block.get("tool_name") or block.get("name")
            return [_text_part(f"Tool loaded: {self.tool_names.to_openai(name)}")] if name else []
        logger.warning("Dropping unsupported user content block type: %s", block_type)
        return []

    def _document_parts(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        source = block.get("source") or {}
        source_type = source.get("type", "base64")
        title = block.get("title")
        header_lines = [f"Document: {title}" if title else "", block.get("context") or ""]
        header = "\n".join(line for line in header_lines if line)
        parts: List[Dict[str, Any]] = []

        if source_type == "base64" and source.get("data"):
            media_type = source.get("media_type") or "application/pdf"
            if media_type.startswith("image/"):
                parts.append(_image_part_from_source({**source, "type": "base64"}))
            elif media_type.startswith("text/"):
                try:
                    decoded = base64.b64decode(source["data"]).decode("utf-8", errors="replace")
                except (binascii.Error, ValueError):
                    decoded = ""
                parts.append(_text_part(decoded))
            elif not self.supports_file_parts:
                label = f"{title} ({media_type})" if title else media_type
                parts.append(_text_part(f"[document {label} is not available to this model]"))
            else:
                filename = title or "document"
                if media_type == "application/pdf" and not filename.lower().endswith(".pdf"):
                    filename = f"{filename}.pdf"
                parts.append({
                    "type": "file",
                    "file": {"filename": filename, "file_data": f"data:{media_type};base64,{source['data']}"},
                })
        elif source_type == "text":
            parts.append(_text_part(source.get("data", "")))
        elif source_type == "content":
            inner = source.get("content")
            if isinstance(inner, str):
                parts.append(_text_part(inner))
            elif isinstance(inner, list):
                for item in inner:
                    parts.extend(self._content_parts(_block_dict(item)))
        elif source_type == "url" and source.get("url"):
            parts.append(_text_part(f"[Document available at {source['url']}]"))
        else:
            file_id = source.get("file_id")
            parts.append(_text_part(f"[document {file_id} is not available to this model]" if file_id
                                    else "[document is not available to this model]"))

        parts = [p for p in parts if p and not (p.get("type") == "text" and not p.get("text"))]
        if header and parts:
            parts.insert(0, _text_part(header))
        return parts

    def _convert_user_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content_parts: List[Dict[str, Any]] = []
        tool_messages: List[Dict[str, Any]] = []
        tool_media: List[Dict[str, Any]] = []

        for block in blocks:
            if block.get("type") == "tool_result":
                tool_message, media = self._convert_tool_result(block)
                tool_messages.append(tool_message)
                if media:
                    tool_media.append(_text_part(f"[Content returned by tool call {tool_message['tool_call_id']}]"))
                    tool_media.extend(media)
            else:
                content_parts.extend(self._content_parts(block))

        # Emit tool messages BEFORE the user content message so that an
        # assistant(tool_calls) turn is immediately followed by the matching
        # tool messages (OpenAI rejects assistant tool_calls not immediately
        # followed by tool messages with a 400). Media returned by tools goes
        # in a follow-up user message: most backends only accept text in
        # role="tool" messages.
        result: List[Dict[str, Any]] = []
        result.extend(tool_messages)
        if tool_media:
            result.append({"role": "user", "content": tool_media})
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                result.append({"role": "user", "content": content_parts[0]["text"]})
            else:
                result.append({"role": "user", "content": content_parts})
        return result

    def _convert_tool_result(self, block: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        tool_call_id = restore_tool_id(block.get("tool_use_id", ""))
        content = block.get("content")
        texts: List[str] = []
        media: List[Dict[str, Any]] = []

        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for item in content:
                for part in self._content_parts(_block_dict(item)):
                    if part.get("type") == "text":
                        texts.append(part["text"])
                    else:
                        media.append(part)
        elif content:
            texts.append(str(content))

        text = "\n".join(t for t in texts if t)
        if not text and media:
            text = "[see attached content]"
        if block.get("is_error"):
            text = f"Error: {text}" if text else "Error"
        return {"role": "tool", "tool_call_id": tool_call_id, "content": text}, media

    def _convert_assistant_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []
        server_uses: Dict[str, Dict[str, Any]] = {}

        for block in blocks:
            block_type = block.get("type", "")

            if block_type == "text":
                if block.get("text"):
                    text_parts.append(block["text"])

            elif block_type == "tool_use":
                tool_call: Dict[str, Any] = {
                    "id": restore_tool_id(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": self.tool_names.to_openai(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
                signature = tool_id_signature(block.get("id", ""))
                if self.thought_signatures and signature:
                    tool_call["extra_content"] = _signature_extra(signature)
                tool_calls.append(tool_call)

            elif block_type == "thinking":
                if self.echo_reasoning and block.get("thinking"):
                    reasoning_parts.append(block["thinking"])

            elif block_type == "redacted_thinking":
                pass

            elif block_type == "server_tool_use":
                server_uses[block.get("id", "")] = block

            elif block_type in _FOREIGN_SERVER_RESULT_TYPES:
                use = server_uses.pop(block.get("tool_use_id", ""), None)
                text_parts.append(_flatten_server_tool_turn(use, block))

            else:
                logger.warning("Dropping unknown assistant content block type: %s", block_type)

        for use in server_uses.values():
            text_parts.append(_flatten_server_tool_turn(use, None))

        # Gemini signs only the first call of a step; a step with none signed
        # came from elsewhere and needs the stand-in to pass validation.
        if self.thought_signatures and tool_calls and not any("extra_content" in tc for tc in tool_calls):
            tool_calls[0]["extra_content"] = _signature_extra(SKIP_THOUGHT_SIGNATURE)

        if not text_parts and not tool_calls and not reasoning_parts:
            # An assistant turn with nothing left to send; several chat
            # templates reject empty assistant messages.
            return []

        msg: Dict[str, Any] = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else (None if tool_calls else "")
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_parts:
            msg["reasoning_content"] = "\n".join(reasoning_parts)
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
            if not tool_dict.get("input_schema") and tool_dict.get("type") and tool_dict.get("type") != "custom":
                continue

            name = tool_dict.get("name", "")
            description = tool_dict.get("description", "")
            input_schema = tool_dict.get("input_schema", {})
            if isinstance(input_schema, dict):
                parameters = dict(input_schema)
            elif hasattr(input_schema, "model_dump"):
                parameters = input_schema.model_dump(exclude_none=True)
            else:
                parameters = {}
            parameters.setdefault("type", "object")
            if parameters.get("type") == "object":
                parameters.setdefault("properties", {})

            function: Dict[str, Any] = {
                "name": self.tool_names.to_openai(name),
                "description": description,
                "parameters": parameters,
            }
            if isinstance(tool_dict.get("strict"), bool):
                function["strict"] = tool_dict["strict"]
            openai_tools.append({"type": "function", "function": function})
        return openai_tools

    # -- Tool choice --

    def _convert_tool_choice(self, tool_choice: Any) -> Tuple[Any, Optional[bool]]:
        """Return ``(openai_tool_choice, parallel_tool_calls_or_None)``."""
        if isinstance(tool_choice, str):
            return ("required" if tool_choice == "any" else tool_choice), None

        tc_dict = tool_choice.model_dump(exclude_none=True) if hasattr(tool_choice, "model_dump") else (
            tool_choice if isinstance(tool_choice, dict) else {}
        )
        parallel = False if tc_dict.get("disable_parallel_tool_use") else None
        tc_type = tc_dict.get("type", "")
        if tc_type == "auto":
            return "auto", parallel
        elif tc_type == "any":
            return "required", parallel
        elif tc_type == "none":
            return "none", None
        elif tc_type == "tool":
            name = self.tool_names.to_openai(tc_dict.get("name", ""))
            return {"type": "function", "function": {"name": name}}, parallel
        return "auto", parallel


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
    block_open: bool = False
    tool_call_indices: Dict[int, int] = field(default_factory=dict)  # tool-call key → our block index
    # Buffers id/name/args per tool call, keyed 0, 1, ... in arrival order.
    # Only the first tool call streams live; later ones are emitted whole at
    # the end so argument deltas never target an already-closed block,
    # however the upstream interleaves.
    pending_tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # Upstream id / upstream index → tool-call key. Calls are found by id
    # first; id-less fragments go to the last call seen at their index.
    # Gemini's OpenAI layer sends parallel calls under one index (or none),
    # told apart only by a new id, so a new id with a name starts a new call.
    tool_call_ids: Dict[str, int] = field(default_factory=dict)
    tool_call_keys: Dict[int, int] = field(default_factory=dict)
    live_tool_index: Optional[int] = None
    # Text/thinking that arrived while a tool block was live; emitted after.
    deferred_content: List[Tuple[str, str]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    had_tool_use: bool = False  # True if any tool_use block was opened during the stream
    error_message: Optional[str] = None  # Error message from upstream API
    tool_names: Optional[ToolNameMap] = None
    stop_sequences: Optional[List[str]] = None
    matched_stop_sequence: Optional[str] = None
    think_splitter: Optional[ThinkTagSplitter] = None
    full_text: str = ""
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    citations_emitted: bool = False


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response conversion
# ---------------------------------------------------------------------------

STOP_REASON_MAP: Dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
    "refusal": "refusal",
}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "output_text")
        )
    return ""


def _final_stop_reason(finish_reason: Optional[str], has_tool_use: bool) -> str:
    stop_reason = STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
    # A backend may omit finish_reason or report "stop" even when it emitted
    # tool calls; fall back to "tool_use" so the client executes the tools.
    if has_tool_use and stop_reason not in ("max_tokens", "refusal"):
        stop_reason = "tool_use"
    return stop_reason


class OpenAIToAnthropicConverter:
    """Converts OpenAI ChatCompletion responses to Anthropic Messages format.

    Args:
        tool_names: map produced by ``AnthropicToOpenAIConverter`` so tool
            names shortened for OpenAI are restored.
        stop_sequences: the request's stop sequences, to report
            ``stop_sequence`` when the upstream says which one matched.
        split_think_tags: route ``<think>...</think>`` content to thinking.
    """

    def __init__(
        self,
        *,
        tool_names: Optional[ToolNameMap] = None,
        stop_sequences: Optional[List[str]] = None,
        split_think_tags: bool = False,
    ):
        self.tool_names = tool_names or ToolNameMap()
        self.stop_sequences = list(stop_sequences or [])
        self.split_think_tags = split_think_tags

    # -- Non-streaming --

    def convert_response(self, openai_response: dict, original_model: str) -> dict:
        choices = openai_response.get("choices") or [{}]
        choice = choices[0] or {}
        message = choice.get("message") or {}

        content: List[Dict[str, Any]] = []

        reasoning = reasoning_text_from_message(message)
        text = _message_text(message.get("content"))
        if self.split_think_tags and text:
            inline_thinking, text = split_think_tags(text)
            reasoning = reasoning or inline_thinking

        # Reasoning/thinking content (no signature: not a Claude model)
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning, "signature": ""})

        # Text content
        if text:
            text_block: Dict[str, Any] = {"type": "text", "text": text}
            citations = citations_from_annotations(text, message.get("annotations"))
            if citations:
                text_block["citations"] = citations
            content.append(text_block)
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal and not text:
            content.append({"type": "text", "text": refusal})

        # Tool calls
        for tc in message.get("tool_calls") or []:
            func = tc.get("function") or {}
            name = func.get("name")
            if not name:
                continue
            content.append({
                "type": "tool_use",
                "id": tool_id_with_signature(tc.get("id"), _thought_signature(tc)),
                "name": self.tool_names.to_anthropic(name),
                "input": parse_tool_arguments(func.get("arguments")),
            })

        if not content:
            content.append({"type": "text", "text": ""})

        has_tool_use = any(block["type"] == "tool_use" for block in content)
        stop_reason = _final_stop_reason(choice.get("finish_reason"), has_tool_use)
        stop_sequence = None
        if stop_reason == "end_turn":
            stop_sequence = matched_stop_sequence(choice, self.stop_sequences)
            if stop_sequence:
                stop_reason = "stop_sequence"

        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": original_model,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "usage": anthropic_usage_from_openai(openai_response.get("usage")),
        }

    # -- Streaming helpers --

    def new_stream_state(self, message_id: str, model: str, input_tokens: int = 0) -> StreamConversionState:
        return StreamConversionState(
            message_id=message_id,
            model=model,
            input_tokens=input_tokens,
            tool_names=self.tool_names,
            stop_sequences=self.stop_sequences,
            think_splitter=ThinkTagSplitter() if self.split_think_tags else None,
        )

    def convert_stream_start(self, message_id: str, model: str, input_tokens: int = 0) -> str:
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
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }
        return _format_sse("message_start", message_start) + _format_sse("ping", {"type": "ping"})

    def convert_stream_chunk(
        self, chunk_data: dict, state: StreamConversionState
    ) -> List[str]:
        events: List[str] = []
        self._record_usage(chunk_data.get("usage"), state)
        choices = chunk_data.get("choices") or []
        if not choices:
            # May contain only usage
            return events

        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        pieces: List[Tuple[str, str]] = []
        reasoning = reasoning_text_from_message(delta)
        if reasoning:
            pieces.append(("thinking", reasoning))
        text = _message_text(delta.get("content"))
        if text:
            if state.think_splitter is not None:
                pieces.extend(state.think_splitter.feed(text))
            else:
                pieces.append(("text", text))
        for kind, piece in pieces:
            events.extend(self._emit_content(state, kind, piece))

        for tc in delta.get("tool_calls") or []:
            events.extend(self._handle_tool_delta(state, tc))

        annotations = delta.get("annotations") or choice.get("annotations")
        if isinstance(annotations, list):
            state.annotations.extend(a for a in annotations if isinstance(a, dict))

        if finish_reason:
            state.finish_reason = finish_reason
            matched = matched_stop_sequence(choice, state.stop_sequences)
            if matched:
                state.matched_stop_sequence = matched

        # Usage (may come in the same or a later chunk)
        self._record_usage(choice.get("usage"), state)
        return events

    def convert_stream_end(self, state: StreamConversionState) -> List[str]:
        events: List[str] = []

        if state.think_splitter is not None:
            for kind, piece in state.think_splitter.flush():
                events.extend(self._emit_content(state, kind, piece))

        # Close the live tool block, then emit buffered tool calls whole.
        if state.live_tool_index is not None:
            events.extend(self._close_block(state))
            state.live_tool_index = None
        for tc_index, pending in state.pending_tool_calls.items():
            if tc_index in state.tool_call_indices:
                continue
            if not pending.get("name"):
                continue
            events.extend(self._open_tool_block(state, tc_index))
            if pending["buffered_args"]:
                events.append(self._input_json_delta(state.content_block_index, pending["buffered_args"]))
                pending["buffered_args"] = ""
            events.extend(self._close_block(state))

        # Text/thinking that arrived while a tool block was live.
        deferred, state.deferred_content = state.deferred_content, []
        for kind, piece in deferred:
            events.extend(self._emit_content(state, kind, piece))

        events.extend(self._close_block(state))

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

        stop_reason = _final_stop_reason(state.finish_reason, state.had_tool_use)
        stop_sequence = None
        if stop_reason == "end_turn" and state.matched_stop_sequence:
            stop_reason = "stop_sequence"
            stop_sequence = state.matched_stop_sequence

        usage: Dict[str, int] = {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
        }
        if state.cache_read_input_tokens:
            usage["cache_read_input_tokens"] = state.cache_read_input_tokens
        events.append(_format_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": usage,
        }))
        events.append(_format_sse("message_stop", {"type": "message_stop"}))

        return events

    # -- Internal helpers --

    @staticmethod
    def _record_usage(usage: Any, state: StreamConversionState) -> None:
        if not isinstance(usage, dict):
            return
        mapped = anthropic_usage_from_openai(usage)
        if "prompt_tokens" in usage:
            state.input_tokens = mapped["input_tokens"]
            state.cache_read_input_tokens = mapped.get("cache_read_input_tokens", 0)
        if "completion_tokens" in usage:
            state.output_tokens = mapped["output_tokens"]

    @staticmethod
    def _input_json_delta(index: int, partial_json: str) -> str:
        return _format_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })

    def _emit_content(self, state: StreamConversionState, kind: str, piece: str) -> List[str]:
        if not piece:
            return []
        if state.live_tool_index is not None:
            # Never interleave text into an open tool block; emit it after.
            if state.deferred_content and state.deferred_content[-1][0] == kind:
                state.deferred_content[-1] = (kind, state.deferred_content[-1][1] + piece)
            else:
                state.deferred_content.append((kind, piece))
            return []

        events: List[str] = []
        if kind == "thinking":
            if not (state.block_open and state.current_block_type == "thinking"):
                events.extend(self._open_block(state, "thinking", {
                    "type": "thinking", "thinking": "", "signature": "",
                }))
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {"type": "thinking_delta", "thinking": piece},
            }))
        else:
            if not (state.block_open and state.current_block_type == "text"):
                events.extend(self._open_block(state, "text", {
                    "type": "text", "text": "",
                }))
            state.full_text += piece
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {"type": "text_delta", "text": piece},
            }))
        return events

    def _handle_tool_delta(self, state: StreamConversionState, tc: Dict[str, Any]) -> List[str]:
        events: List[str] = []
        func = tc.get("function") or {}
        args_chunk = func.get("arguments")

        upstream_index = tc.get("index")
        if upstream_index is None:
            upstream_index = 0
        tc_id = tc.get("id")
        tc_index = state.tool_call_ids.get(tc_id) if tc_id else None
        if tc_index is None:
            tc_index = state.tool_call_keys.get(upstream_index)
            # An unseen id without a name is a continuation (some upstreams
            # regenerate ids per delta); with a name it is a new call.
            if tc_index is None or (
                tc_id and func.get("name") and state.pending_tool_calls[tc_index]["id"]
            ):
                tc_index = len(state.pending_tool_calls)
                state.pending_tool_calls[tc_index] = {
                    "id": None, "name": None, "signature": None, "buffered_args": "",
                }
        state.tool_call_keys[upstream_index] = tc_index
        pending = state.pending_tool_calls[tc_index]
        if tc_id:
            state.tool_call_ids[tc_id] = tc_index
            if not pending["id"]:
                pending["id"] = tc_id
        if func.get("name"):
            pending["name"] = func["name"]
        signature = _thought_signature(tc)
        if signature:
            pending["signature"] = signature

        if tc_index == state.live_tool_index:
            if args_chunk:
                events.append(self._input_json_delta(state.tool_call_indices[tc_index], args_chunk))
            return events

        # Only the first tool call streams live. Open it once both id and
        # name are known (or once arguments start without an id).
        should_open = (
            tc_index == 0
            and state.live_tool_index is None
            and tc_index not in state.tool_call_indices
            and pending["name"]
            and (pending["id"] or pending["buffered_args"] or args_chunk)
        )
        if not should_open:
            if args_chunk:
                pending["buffered_args"] += args_chunk
            return events

        events.extend(self._open_tool_block(state, tc_index))
        state.live_tool_index = tc_index
        # Flush any argument fragments buffered before the block opened.
        if pending["buffered_args"]:
            events.append(self._input_json_delta(state.content_block_index, pending["buffered_args"]))
            pending["buffered_args"] = ""
        if args_chunk:
            events.append(self._input_json_delta(state.content_block_index, args_chunk))
        return events

    def _open_tool_block(self, state: StreamConversionState, tc_index: int) -> List[str]:
        pending = state.pending_tool_calls[tc_index]
        state.had_tool_use = True
        names = state.tool_names or self.tool_names
        events = self._open_block(state, "tool_use", {
            "type": "tool_use",
            "id": tool_id_with_signature(pending.get("id"), pending.get("signature")),
            "name": names.to_anthropic(pending["name"]),
            "input": {},
        })
        state.tool_call_indices[tc_index] = state.content_block_index
        return events

    def _close_block(self, state: StreamConversionState) -> List[str]:
        if not state.block_open:
            return []
        events: List[str] = []
        index = state.content_block_index
        if state.current_block_type == "thinking":
            # Non-Claude reasoning has no signature; send an empty one so
            # clients that expect signature_delta see a well-formed block.
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": ""},
            }))
        elif state.current_block_type == "text" and state.annotations and not state.citations_emitted:
            for citation in citations_from_annotations(state.full_text, state.annotations):
                events.append(_format_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "citations_delta", "citation": citation},
                }))
            state.citations_emitted = True
        events.append(_format_sse("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        }))
        state.block_open = False
        return events

    def _open_block(
        self, state: StreamConversionState, block_type: str, content_block: dict
    ) -> List[str]:
        # Close previous block if one is open
        events = self._close_block(state)
        state.content_block_index += 1
        state.current_block_type = block_type
        state.block_open = True
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

MIN_RESPONSES_OUTPUT_TOKENS = 16

class AnthropicToResponsesConverter:
    """Converts an AnthropicMessagesRequest into a ResponsesCreateRequest.

    The Responses API uses:
    - ``input``: list of message items (``{"type": "message", "role": ..., "content": ...}``)
    - ``instructions``: system prompt
    - ``tools``: list of ``{"type": "function", "name": ..., "parameters": ..., "description": ...}``
    - ``max_output_tokens`` instead of ``max_tokens``

    Content blocks are converted with the chat converter's helpers and then
    re-typed for the Responses API, so both transports stay in sync.
    """

    def __init__(self):
        # Mapping from original tool-use IDs (call_*, toolu_*) to fc_* IDs
        # required by Azure codex models via the Responses API.
        self._id_map: Dict[str, str] = {}
        self._chat = AnthropicToOpenAIConverter()
        self.tool_names = self._chat.tool_names

    def _ensure_fc_id(self, original_id: str) -> str:
        """Ensure an ID starts with 'fc_' for Responses API compatibility.

        Derived from the original id rather than random, so the same history
        converts to the same input on every turn and upstream prompt caching
        keeps hitting.
        """
        if original_id.startswith("fc_"):
            return original_id
        if original_id in self._id_map:
            return self._id_map[original_id]
        digest = hashlib.sha256(original_id.encode("utf-8")).hexdigest()[:24] if original_id else uuid.uuid4().hex[:24]
        new_id = f"fc_{digest}"
        self._id_map[original_id] = new_id
        return new_id

    def convert_request(
        self,
        request: AnthropicMessagesRequest,
        model_id: str,
    ) -> ResponsesCreateRequest:
        self._chat.tool_names = ToolNameMap()
        self.tool_names = self._chat.tool_names

        # System prompt → instructions
        instructions = None
        if request.system:
            instructions = strip_billing_header(self._convert_system(request.system))

        # Build input items
        input_items: List[Dict[str, Any]] = []
        for msg in request.messages:
            items = self._convert_message(msg.role, msg.content)
            input_items.extend(items)

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "input": input_items,
            # The Responses API rejects max_output_tokens below 16; clients
            # such as Claude Code probe with max_tokens=1.
            "max_output_tokens": max(request.max_tokens, MIN_RESPONSES_OUTPUT_TOKENS),
            "stream": request.stream or False,
        }

        if instructions and instructions.strip():
            kwargs["instructions"] = instructions
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Tools (Responses API format)
        resp_tools: List[Dict[str, Any]] = []
        if request.tools:
            resp_tools = self._convert_tools(request.tools)
            if resp_tools:
                kwargs["tools"] = resp_tools

        # Tool choice
        if request.tool_choice is not None and resp_tools:
            tool_choice, parallel = self._convert_tool_choice(request.tool_choice)
            declared = {t["name"] for t in resp_tools}
            if isinstance(tool_choice, dict) and tool_choice.get("name") not in declared:
                tool_choice = "auto"
            kwargs["tool_choice"] = tool_choice
            if parallel is False:
                kwargs["parallel_tool_calls"] = False

        # Thinking / output_config.effort → reasoning
        output_config = getattr(request, "output_config", None)
        effort = reasoning_effort_for(request.thinking, output_config)
        if effort:
            kwargs["reasoning"] = {"effort": effort}
            # Without a summary the Responses API returns no reasoning text,
            # so a client that asked for thinking would never see any.
            thinking = request.thinking
            if (
                thinking is not None
                and thinking.type in ("enabled", "adaptive")
                and (thinking.model_extra or {}).get("display") != "omitted"
            ):
                kwargs["reasoning"]["summary"] = "auto"

        # output_config.format → text.format (json_schema)
        response_format = response_format_for(output_config)
        if response_format:
            schema = response_format["json_schema"]
            kwargs["text"] = {"format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema["strict"],
            }}

        # Metadata user_id → user (max 64 chars per OpenAI API)
        if request.metadata and getattr(request.metadata, "user_id", None):
            kwargs["user"] = request.metadata.user_id[:64]

        return ResponsesCreateRequest(**kwargs)

    def _convert_system(self, system: Any) -> str:
        return self._chat._convert_system(system)

    def _convert_message(self, role: str, content: Any) -> List[Dict[str, Any]]:
        """Convert a single Anthropic message into Responses API input items.

        Each input item is: {"type": "message", "role": ..., "content": ...}
        Tool results become: {"type": "function_call_output", "call_id": ..., "output": ...}
        """
        if isinstance(content, str):
            return [{"type": "message", "role": role, "content": content}]

        if not isinstance(content, list):
            return [{"type": "message", "role": role, "content": str(content)}]

        blocks = [_block_dict(block) for block in content]
        if role == "user":
            return self._convert_user_blocks(blocks)
        elif role == "assistant":
            return self._convert_assistant_blocks(blocks)
        return [{"type": "message", "role": role, "content": str(content)}]

    @staticmethod
    def _responses_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Re-type a chat content part for the Responses API."""
        part_type = part.get("type")
        if part_type == "text":
            return {"type": "input_text", "text": part.get("text", "")}
        if part_type == "image_url":
            return {"type": "input_image", "image_url": (part.get("image_url") or {}).get("url", "")}
        if part_type == "file":
            file_info = part.get("file") or {}
            return {"type": "input_file", "filename": file_info.get("filename"), "file_data": file_info.get("file_data")}
        return None

    def _user_message_item(self, parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        converted = [p for p in (self._responses_part(part) for part in parts) if p]
        if not converted:
            return None
        if len(converted) == 1 and converted[0]["type"] == "input_text":
            return {"type": "message", "role": "user", "content": converted[0]["text"]}
        return {"type": "message", "role": "user", "content": converted}

    def _convert_user_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content_parts: List[Dict[str, Any]] = []
        extra_items: List[Dict[str, Any]] = []
        tool_media: List[Dict[str, Any]] = []

        for block in blocks:
            if block.get("type") == "tool_result":
                tool_message, media = self._chat._convert_tool_result(block)
                fc_id = self._ensure_fc_id(tool_message["tool_call_id"])
                extra_items.append({"type": "function_call_output", "call_id": fc_id, "output": tool_message["content"]})
                if media:
                    tool_media.append(_text_part(f"[Content returned by tool call {fc_id}]"))
                    tool_media.extend(media)
            else:
                content_parts.extend(self._chat._content_parts(block))

        # Emit function_call_output items BEFORE the user content message so a
        # preceding assistant function_call turn is immediately followed by its
        # matching outputs.
        result: List[Dict[str, Any]] = []
        result.extend(extra_items)
        for parts in (tool_media, content_parts):
            item = self._user_message_item(parts)
            if item:
                result.append(item)
        return result

    def _convert_assistant_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert assistant blocks into Responses API input items.

        Text → message item, tool_use → function_call item.
        """
        items: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        server_uses: Dict[str, Dict[str, Any]] = {}

        for block in blocks:
            block_type = block.get("type", "")

            if block_type == "text":
                if block.get("text"):
                    text_parts.append(block["text"])

            elif block_type == "tool_use":
                # Flush accumulated text first
                if text_parts:
                    items.append({"type": "message", "role": "assistant", "content": "\n".join(text_parts)})
                    text_parts = []
                fc_id = self._ensure_fc_id(restore_tool_id(block.get("id", "")))
                items.append({
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": fc_id,
                    "name": self.tool_names.to_openai(block.get("name", "")),
                    "arguments": json.dumps(block.get("input") or {}),
                })

            elif block_type == "server_tool_use":
                server_uses[block.get("id", "")] = block

            elif block_type in _FOREIGN_SERVER_RESULT_TYPES:
                use = server_uses.pop(block.get("tool_use_id", ""), None)
                text_parts.append(_flatten_server_tool_turn(use, block))

        for use in server_uses.values():
            text_parts.append(_flatten_server_tool_turn(use, None))

        if text_parts:
            items.append({"type": "message", "role": "assistant", "content": "\n".join(text_parts)})

        return items

    def _convert_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        resp_tools: List[Dict[str, Any]] = []
        for tool in self._chat._convert_tools(tools):
            function = tool["function"]
            resp_tool = {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
            if "strict" in function:
                resp_tool["strict"] = function["strict"]
            resp_tools.append(resp_tool)
        return resp_tools

    def _convert_tool_choice(self, tool_choice: Any) -> Tuple[Any, Optional[bool]]:
        choice, parallel = self._chat._convert_tool_choice(tool_choice)
        if isinstance(choice, dict):
            return {"type": "function", "name": choice["function"]["name"]}, parallel
        return choice, parallel


# ---------------------------------------------------------------------------
# Responses API → Anthropic response conversion
# ---------------------------------------------------------------------------

RESPONSES_STATUS_TO_STOP_REASON: Dict[str, str] = {
    "completed": "end_turn",
    "incomplete": "max_tokens",
    "failed": "end_turn",
}

# incomplete_details.reason → Anthropic stop_reason
RESPONSES_INCOMPLETE_TO_STOP_REASON: Dict[str, str] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "refusal",
}

# Responses error.code values that are not server faults. Everything else
# (server_error, vector_store_timeout, unknown) is a 500 api_error. The
# remaining documented codes are all input problems (invalid_prompt,
# invalid_image*, image_*, unsupported_image_media_type, ...).
_RESPONSES_CLIENT_ERROR_PREFIXES = ("invalid_", "image_", "unsupported_", "empty_", "failed_to_download")


def responses_error_status(error: Any) -> Tuple[int, str, str]:
    """``(http_status, anthropic_error_type, message)`` for a Responses ``error``."""
    error = error if isinstance(error, dict) else {}
    code = str(error.get("code") or "")
    if code == "rate_limit_exceeded":
        status, error_type = 429, "rate_limit_error"
    elif code.startswith(_RESPONSES_CLIENT_ERROR_PREFIXES):
        status, error_type = 400, "invalid_request_error"
    else:
        status, error_type = 500, "api_error"
    message = error.get("message") or (json.dumps(error, ensure_ascii=False) if error else "Upstream API request failed")
    return status, error_type, message


def _responses_stop_reason(status: Optional[str], incomplete_details: Any, has_tool_use: bool, refused: bool) -> str:
    status = status or "completed"
    if status == "incomplete":
        reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
        return RESPONSES_INCOMPLETE_TO_STOP_REASON.get(reason, "max_tokens")
    if refused and not has_tool_use:
        return "refusal"
    if has_tool_use and status == "completed":
        return "tool_use"
    return RESPONSES_STATUS_TO_STOP_REASON.get(status, "end_turn")


def _format_anthropic_error_sse(error_type: str, message: str) -> str:
    """Emit a single Anthropic SSE error event."""
    return _format_sse("error", {
        "type": "error",
        "error": {"type": error_type, "message": message},
    })


def _usage_from_responses(resp_usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    resp_usage = resp_usage or {}
    details = resp_usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    usage = {
        "input_tokens": max(int(resp_usage.get("input_tokens") or 0) - cached, 0),
        "output_tokens": int(resp_usage.get("output_tokens") or 0),
    }
    if cached:
        usage["cache_read_input_tokens"] = cached
    return usage


def _same_json(a: str, b: str) -> bool:
    """True when two argument strings are complete, equal JSON values."""
    try:
        return json.loads(a) == json.loads(b)
    except ValueError:
        return False


class ResponsesToAnthropicConverter:
    """Converts OpenAI Responses API output to Anthropic Messages format."""

    def __init__(self, *, tool_names: Optional[ToolNameMap] = None):
        self.tool_names = tool_names or ToolNameMap()
        # Streamed function calls in first-seen order, each tracking its
        # id/name, received arguments, block index once opened, and done flag.
        # Only the first call streams live (as the Chat converter does); later
        # calls and conflicting content are buffered and emitted at the end,
        # so no delta ever targets a block that is already closed.
        self._calls: List[Dict[str, Any]] = []
        # ("output", output_index) / ("item", item id) / ("call", call_id)
        # → position in ``_calls``; upstream events name a call by any of them.
        self._call_aliases: Dict[Tuple[str, Any], int] = {}
        self._live_call: Optional[Dict[str, Any]] = None
        self._deferred: List[Tuple[str, str]] = []
        self._last_content_kind: Optional[str] = None
        # Set by a later reasoning summary part; joins it to the previous
        # part's thinking, wherever that text ended up (live or deferred).
        self._pending_separator = False
        self._refused = False
        self._incomplete_details: Optional[Dict[str, Any]] = None
        self._error: Optional[Dict[str, Any]] = None

    # -- Stream start (mirrors OpenAIToAnthropicConverter.convert_stream_start) --

    def convert_stream_start(self, message_id: str, model: str, input_tokens: int = 0) -> str:
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
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }
        return _format_sse("message_start", message_start) + _format_sse("ping", {"type": "ping"})

    # -- Non-streaming --

    def convert_response(self, response_obj: dict, original_model: str) -> dict:
        content: List[Dict[str, Any]] = []
        refused = False

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
                                text = part.get("text", "")
                                text_block: Dict[str, Any] = {"type": "text", "text": text}
                                citations = citations_from_annotations(text, part.get("annotations"))
                                if citations:
                                    text_block["citations"] = citations
                                content.append(text_block)
                            elif part.get("type") == "refusal":
                                refused = True
                                content.append({"type": "text", "text": part.get("refusal", "")})
                        elif isinstance(part, str):
                            content.append({"type": "text", "text": part})

            elif item_type == "function_call":
                content.append({
                    "type": "tool_use",
                    "id": sanitize_tool_id(item.get("call_id") or item.get("id")),
                    "name": self.tool_names.to_anthropic(item.get("name", "")),
                    "input": parse_tool_arguments(item.get("arguments")),
                })

            elif item_type == "reasoning":
                # One thinking block per reasoning item: its summary parts, or
                # the raw reasoning_text content (gpt-oss on vLLM and others).
                texts = [
                    s.get("text", "") for s in item.get("summary") or []
                    if isinstance(s, dict) and s.get("type") == "summary_text"
                ]
                if not any(texts):
                    texts = [
                        c.get("text", "") for c in item.get("content") or []
                        if isinstance(c, dict) and c.get("type") == "reasoning_text"
                    ]
                thinking = "\n\n".join(t for t in texts if t)
                if thinking:
                    content.append({"type": "thinking", "thinking": thinking, "signature": ""})

        # Also check output_text shorthand
        if not content:
            output_text = response_obj.get("output_text")
            if output_text:
                content.append({"type": "text", "text": output_text})

        if not content:
            content.append({"type": "text", "text": ""})

        has_tool_use = any(c.get("type") == "tool_use" for c in content)
        stop_reason = _responses_stop_reason(
            response_obj.get("status"), response_obj.get("incomplete_details"), has_tool_use, refused,
        )

        usage = _usage_from_responses(response_obj.get("usage"))

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
        if self._error is not None:
            # A failure is final: later events (even response.completed)
            # neither add output nor replace the error that will be reported.
            return []
        events: List[str] = []

        if event_type in ("response.output_text.delta", "response.refusal.delta"):
            if event_type == "response.refusal.delta":
                self._refused = True
            events.extend(self._emit_content(state, "text", event_data.get("delta", "")))

        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            events.extend(self._emit_content(state, "thinking", event_data.get("delta", "")))

        elif event_type == "response.reasoning_summary_part.added":
            # Separate successive summary parts, as the non-streaming path does.
            if (event_data.get("summary_index") or 0) > 0:
                self._pending_separator = True

        elif event_type == "response.output_item.added":
            item = event_data.get("item") or {}
            if item.get("type") == "function_call":
                call = self._resolve_call(event_data, item)
                if call is not None:
                    self._update_call_identity(call, item)
                    events.extend(self._maybe_open_live_call(state))

        elif event_type == "response.function_call_arguments.delta":
            call = self._resolve_call(event_data)
            delta_args = event_data.get("delta", "")
            if call is not None and delta_args and not call["done"]:
                call["arguments"] += delta_args
                if call is self._live_call:
                    events.append(self._input_json_delta(call["block"], delta_args))
                else:
                    events.extend(self._maybe_open_live_call(state))

        elif event_type in ("response.function_call_arguments.done", "response.output_item.done"):
            # Some upstreams send a call's arguments only whole, in the done
            # events, with no deltas; reconcile them with what was received.
            if event_type == "response.function_call_arguments.done":
                item = {"name": event_data.get("name")}
                call = self._resolve_call(event_data)
                arguments = event_data.get("arguments")
            else:
                item = event_data.get("item") or {}
                is_call = item.get("type") == "function_call"
                call = self._resolve_call(event_data, item) if is_call else None
                arguments = item.get("arguments")
            if call is not None:
                self._update_call_identity(call, item)
                events.extend(self._apply_arguments_snapshot(state, call, arguments))
                call["done"] = True
                if self._error is None:
                    events.extend(self._maybe_open_live_call(state))

        elif event_type in ("response.completed", "response.incomplete", "response.failed"):
            resp = event_data.get("response") or {}
            if resp.get("usage"):
                usage = _usage_from_responses(resp["usage"])
                state.input_tokens = usage["input_tokens"]
                state.output_tokens = usage["output_tokens"]
                state.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
            status = resp.get("status") or event_type.split(".", 1)[1]
            if status == "failed":
                self._fail(state, resp.get("error") or {})
                return events
            # Complete calls already seen from the terminal snapshot; its
            # text/reasoning was streamed already and is not replayed.
            for position, item in enumerate(resp.get("output") or []):
                if not (isinstance(item, dict) and item.get("type") == "function_call"):
                    continue
                by_index = {} if (item.get("id") or item.get("call_id")) else {"output_index": position}
                call = self._resolve_call(by_index, item, create=False)
                if call is not None:
                    events.extend(self._apply_arguments_snapshot(state, call, item.get("arguments")))
            if self._error is None:
                state.finish_reason = status
                if isinstance(resp.get("incomplete_details"), dict):
                    self._incomplete_details = resp["incomplete_details"]

        elif event_type == "error":
            # Error event from the provider's responses_create_stream error
            # handler or the upstream; reported once, by convert_stream_end.
            err = event_data.get("error")
            self._fail(state, err if isinstance(err, dict) else {
                "code": event_data.get("code"),
                "message": event_data.get("message") or (str(err) if err else "Unknown error"),
            })

        return events

    def convert_stream_end(self, state: StreamConversionState) -> List[str]:
        events: List[str] = []

        status = state.finish_reason or "completed"
        if status == "failed":
            # An Anthropic stream ends at its error event; clients raise on
            # it, so no message_delta / message_stop follows, and buffered
            # calls/content are not flushed after the failure.
            events.extend(self._close_block(state))
            _, error_type, message = responses_error_status(self._error)
            events.append(_format_anthropic_error_sse(error_type, state.error_message or message))
            return events

        # Close the live call, emit buffered calls whole in first-seen order,
        # then the content that arrived while a call was live.
        if self._live_call is not None:
            events.extend(self._close_block(state))
            self._live_call = None
        for call in self._calls:
            if call["block"] is not None or not call["name"]:
                continue
            events.extend(self._open_tool_block(state, call))
            if call["arguments"]:
                events.append(self._input_json_delta(call["block"], call["arguments"]))
            events.extend(self._close_block(state))
        self._pending_separator = False
        deferred, self._deferred = self._deferred, []
        for kind, piece in deferred:
            events.extend(self._emit_content(state, kind, piece))
        events.extend(self._close_block(state))

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

        stop_reason = _responses_stop_reason(status, self._incomplete_details, state.had_tool_use, self._refused)

        usage: Dict[str, int] = {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
        }
        if state.cache_read_input_tokens:
            usage["cache_read_input_tokens"] = state.cache_read_input_tokens
        events.append(_format_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        }))
        events.append(_format_sse("message_stop", {"type": "message_stop"}))

        return events

    # -- Streaming helpers --

    def _fail(self, state: StreamConversionState, error: Dict[str, Any]) -> None:
        self._error = error
        state.error_message = responses_error_status(error)[2]
        state.finish_reason = "failed"

    def _resolve_call(
        self, event_data: dict, item: Optional[dict] = None, *, create: bool = True
    ) -> Optional[Dict[str, Any]]:
        """The call an event refers to, by item id, call_id, or output_index."""
        item = item or {}
        keys = [key for key in (
            ("item", item.get("id")),
            ("call", item.get("call_id")),
            ("item", event_data.get("item_id")),
            ("output", event_data.get("output_index")),
        ) if key[1] is not None and key[1] != ""]
        position = next((self._call_aliases[key] for key in keys if key in self._call_aliases), None)
        if position is None:
            if not keys:
                # An event naming no call can only belong to a sole call.
                return self._calls[0] if len(self._calls) == 1 else None
            if not create:
                return None
            position = len(self._calls)
            self._calls.append({"id": None, "name": None, "arguments": "", "block": None, "done": False})
        for key in keys:
            self._call_aliases.setdefault(key, position)
        return self._calls[position]

    @staticmethod
    def _update_call_identity(call: Dict[str, Any], item: dict) -> None:
        if call["block"] is None:  # fixed once its block has started
            call["id"] = item.get("call_id") or call["id"] or item.get("id")
        if item.get("name") and not call["name"]:
            call["name"] = item["name"]

    def _maybe_open_live_call(self, state: StreamConversionState) -> List[str]:
        """Open the first-seen call live once its name is known."""
        if self._live_call is not None or not self._calls:
            return []
        call = self._calls[0]
        if call["block"] is not None or not call["name"]:
            return []
        events = self._open_tool_block(state, call)
        self._live_call = call
        # Flush argument fragments buffered before the name arrived.
        if call["arguments"]:
            events.append(self._input_json_delta(call["block"], call["arguments"]))
        return events

    def _open_tool_block(self, state: StreamConversionState, call: Dict[str, Any]) -> List[str]:
        state.had_tool_use = True
        events = self._open_block(state, "tool_use", {
            "type": "tool_use",
            "id": sanitize_tool_id(call["id"]),
            "name": self.tool_names.to_anthropic(call["name"]),
            "input": {},
        })
        call["block"] = state.content_block_index
        return events

    def _apply_arguments_snapshot(
        self, state: StreamConversionState, call: Dict[str, Any], arguments: Any
    ) -> List[str]:
        """Reconcile a call's whole arguments with the fragments received."""
        if not isinstance(arguments, str) or not arguments:
            return []
        received = call["arguments"]
        if call["block"] is None:
            # Nothing emitted yet: the whole value is authoritative.
            call["arguments"] = arguments
            return []
        if arguments == received or _same_json(received, arguments):
            return []
        if arguments.startswith(received):
            call["arguments"] = arguments
            return [self._input_json_delta(call["block"], arguments[len(received):])]
        # Already-emitted input cannot be retracted; fail rather than
        # concatenate incompatible JSON.
        self._fail(state, {
            "code": "tool_arguments_conflict",
            "message": f"Upstream sent conflicting arguments for tool call {call['name']!r}",
        })
        return []

    def _emit_content(self, state: StreamConversionState, kind: str, piece: str) -> List[str]:
        if not piece:
            return []
        if self._pending_separator:
            self._pending_separator = False
            if self._live_call is not None:
                continues = bool(self._deferred) and self._deferred[-1][0] == "thinking"
            else:
                continues = state.block_open and state.current_block_type == "thinking"
            if kind == "thinking" and continues:
                piece = "\n\n" + piece
        if self._live_call is not None:
            # Never interleave content into the live tool block; emit it after.
            if self._deferred and self._deferred[-1][0] == kind:
                self._deferred[-1] = (kind, self._deferred[-1][1] + piece)
            else:
                self._deferred.append((kind, piece))
            return []

        events: List[str] = []
        if not (state.block_open and state.current_block_type == kind):
            content_block = (
                {"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking"
                else {"type": "text", "text": ""}
            )
            events.extend(self._open_block(state, kind, content_block))
        delta = {"type": "thinking_delta", "thinking": piece} if kind == "thinking" else {"type": "text_delta", "text": piece}
        events.append(_format_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": state.content_block_index,
            "delta": delta,
        }))
        return events

    @staticmethod
    def _input_json_delta(index: int, partial_json: str) -> str:
        return _format_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })

    @staticmethod
    def _close_block(state: StreamConversionState) -> List[str]:
        if not state.block_open:
            return []
        events: List[str] = []
        if state.current_block_type == "thinking":
            events.append(_format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": state.content_block_index,
                "delta": {"type": "signature_delta", "signature": ""},
            }))
        events.append(_format_sse("content_block_stop", {
            "type": "content_block_stop",
            "index": state.content_block_index,
        }))
        state.block_open = False
        return events

    def _open_block(
        self, state: StreamConversionState, block_type: str, content_block: dict
    ) -> List[str]:
        events = self._close_block(state)
        state.content_block_index += 1
        state.current_block_type = block_type
        state.block_open = True
        events.append(_format_sse("content_block_start", {
            "type": "content_block_start",
            "index": state.content_block_index,
            "content_block": content_block,
        }))
        return events

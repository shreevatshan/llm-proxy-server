"""Shared Anthropic adapter helpers for chat/responses backends."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
import weakref
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple

from app.anthropic_models import (
    AnthropicMessagesRequest,
    _extract_system_messages_from_messages,
    _merge_system_fields,
)
from app.conversion.anthropic_helpers import MAX_STOP_SEQUENCES, response_format_for, sanitize_tool_id
from app.conversion.anthropic_openai import (
    AnthropicToOpenAIConverter,
    AnthropicToResponsesConverter,
    OpenAIToAnthropicConverter,
    ResponsesToAnthropicConverter,
    StreamConversionState,
    _is_server_tool,
    responses_error_status,
)
from app.conversion.token_estimate import estimate_anthropic_input_tokens
from app.openai_models import ChatCompletionRequest
from app.providers.base import AnthropicRequestMetadata, BaseProvider, ProviderHTTPError
from app.model_alias import echo_model_name

logger = logging.getLogger(__name__)

_SPECIAL_ADAPTER_TOOL_TYPES = (
    "computer_",
    "text_editor",
    "web_search",
    "web_fetch",
    "code_execution",
    "bash",
    "memory",
    "tool_search_tool",
)
_SUPPORTED_CONTENT_TYPES = {
    "text",
    "image",
    "document",
    "search_result",
    "tool_use",
    "tool_result",
    # Foreign server-tool history is flattened to text by the converter.
    "server_tool_use",
    "web_search_tool_result",
    "web_fetch_tool_result",
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
    "output_config",
}
_TOP_LEVEL_ADAPTER_ONLY_FIELDS = {
    "container",
    "context_management",
    "mcp_servers",
    "service_tier",
}

# Anthropic error type per upstream HTTP status.
_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}

# Optional request parameters an upstream may reject with a 400/422. When the
# error says the parameter itself is unsupported, the request is retried
# without it and the rejection is remembered per (provider, model, transport)
# so later requests skip the round trip.
_RETRYABLE_PARAMS_CHAT = (
    "reasoning_effort",
    "parallel_tool_calls",
    "response_format",
    "stream_options",
    "top_k",
    "top_p",
    "temperature",
    "user",
    "stop",
    # Prior thinking echoed on assistant messages: DeepSeek/Kimi-style
    # backends require it, strict OpenAI-compatible ones reject it.
    "reasoning_content",
)
_RETRYABLE_PARAMS_RESPONSES = (
    # Before "reasoning": an organization not verified for reasoning
    # summaries 400s on the summary alone, and the effort should survive.
    "reasoning.summary",
    "reasoning",
    "parallel_tool_calls",
    "top_p",
    "temperature",
    "user",
)
# Parameter names that are also ordinary words in error text ("roles must
# alternate user/assistant", "stop sequence"); these only count when the
# message clearly names them as a parameter.
_AMBIGUOUS_PARAMS = {"user", "stop", "temperature", "reasoning"}
_PARAM_CONTEXT = r"(?:param(?:eter)?|argument|field|property|key|kwarg)s?"
_MAX_PARAM_RETRIES = 3
# Structured upstream error codes meaning "this parameter is not accepted".
_UNSUPPORTED_PARAM_CODES = {"unsupported_parameter", "unknown_parameter", "unrecognized_parameter", "extra_forbidden"}
# Structured codes for a bad value, schema or combination: never a reason to
# drop the parameter.
_VALUE_ERROR_CODE = re.compile(r"value|schema|type|range|missing|required|conflict|json")
_UNSUPPORTED_PREDICATE = r"(?:not supported|unsupported|not permitted|not allowed|unrecognized|unrecognised|not recognized)"
# "unsupported keyword/value ..." describes something inside the parameter.
_NOT_ADJECTIVE = r"(?!\s+(?:keyword|value|type|format|schema|enum|combination|option|mode|content|character)s?\b)"
# "... when tools are absent", "... with top_p": a conflict, not a capability.
_NOT_CONDITIONAL = (
    r"(?!\s+(?:when|while|if|unless|to|together|alongside|in combination|combined"
    r"|with\s+(?!(?:this|the|current|that)\s+(?:model|deployment|endpoint|api)\b)))"
)
# "does not support temperature values other than 1": a value restriction.
_NOT_VALUE_TAIL = r"(?!\s+(?:values?|other|of|=|set|greater|less|above|below|outside|when|with|together)\b)"
_REASONING_SUMMARY = re.compile(r"reasoning[._ ]summar")
_ORG_VERIFICATION = re.compile(r"\borgani[sz]ation\b.*\bverif", re.S)
# provider instance -> (model id, transport) -> rejected params. Keyed weakly
# by instance so a provider rebuilt after a config change starts with a clean
# slate; per transport because Chat and Responses accept different fields.
_rejected_params: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str], Set[str]]]" = weakref.WeakKeyDictionary()
# Some models serve function tools only over the Responses API: chat
# completions 400s with "... not supported ... in /v1/chat/completions. To use
# function tools, use /v1/responses ..." whatever reasoning_effort is sent.
_RESPONSES_ONLY_HINT = re.compile(r"\buse\s+/v1/responses\b")
# provider instance -> model ids moved from the chat to the Responses transport.
_responses_models: "weakref.WeakKeyDictionary[Any, Set[str]]" = weakref.WeakKeyDictionary()


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


def _loaded_tool_names(value: Any, names: Optional[Set[str]] = None) -> Set[str]:
    """Tools the history has loaded: tool-search references and tools already called."""
    names = set() if names is None else names
    if isinstance(value, list):
        for item in value:
            _loaded_tool_names(item, names)
    elif isinstance(value, dict):
        if value.get("type") == "tool_reference" and value.get("tool_name"):
            names.add(value["tool_name"])
        elif value.get("type") == "tool_use" and value.get("name"):
            names.add(value["name"])
        for item in value.values():
            if isinstance(item, (dict, list)):
                _loaded_tool_names(item, names)
    return names


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def anthropic_error_type(status: Any) -> str:
    try:
        status = int(status)
    except (TypeError, ValueError):
        return "api_error"
    if status in _ERROR_TYPE_BY_STATUS:
        return _ERROR_TYPE_BY_STATUS[status]
    if 400 <= status < 500:
        return "invalid_request_error"
    return "api_error"


def anthropic_error_body(status: Any, message: str) -> Dict[str, Any]:
    return {"type": "error", "error": {"type": anthropic_error_type(status), "message": message}}


def _format_anthropic_error_sse(message: str, status: Any = None) -> str:
    payload = anthropic_error_body(status, message)
    return f"event: error\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _provider_error_message(error: ProviderHTTPError) -> str:
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error")
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"]
        if isinstance(body.get("message"), str):
            return body["message"]
    return error.message or "Upstream provider error"


def to_anthropic_http_error(error: ProviderHTTPError) -> ProviderHTTPError:
    """Re-shape an (OpenAI-shaped) provider error into an Anthropic error."""
    body = error.body
    if isinstance(body, dict) and body.get("type") == "error" and isinstance(body.get("error"), dict):
        return error
    message = _provider_error_message(error)
    return ProviderHTTPError(
        error.status_code,
        message,
        body=anthropic_error_body(error.status_code, message),
        headers=error.headers,
    )


# ---------------------------------------------------------------------------
# Rejected-parameter retry
# ---------------------------------------------------------------------------

def known_rejected_params(provider: Any, model_id: str, transport: str) -> Set[str]:
    try:
        return set(_rejected_params.get(provider, {}).get((model_id, transport), ()))
    except TypeError:  # not weak-referenceable
        return set()


def resolve_adapter_transport(provider: Any, model_id: str, transport: str) -> str:
    """The transport to use for ``model_id``: chat unless the upstream has
    already told us this model needs the Responses API."""
    if transport != "chat":
        return transport
    try:
        if model_id in _responses_models.get(provider, ()):
            return "responses"
    except TypeError:  # not weak-referenceable
        pass
    return transport


def _supports_responses(provider: Any) -> bool:
    impl = getattr(type(provider), "responses_create", None)
    return impl is not None and impl is not BaseProvider.responses_create


def _switch_to_responses(provider: Any, model_id: str, transport: str, message: str) -> bool:
    """Remember and report whether a chat-transport error means the model must
    be served over the Responses API instead."""
    if transport != "chat" or not message or not _RESPONSES_ONLY_HINT.search(message.lower()):
        return False
    if not _supports_responses(provider):
        return False
    try:
        _responses_models.setdefault(provider, set()).add(model_id)
    except TypeError:  # not weak-referenceable; this request still switches
        pass
    logger.info(
        "Upstream %s requires the Responses API for model %s; switching Anthropic adapter transport",
        getattr(provider, "full_provider_name", type(provider).__name__), model_id,
    )
    return True


def _remember_rejected(provider: Any, model_id: str, transport: str, param: str) -> None:
    try:
        _rejected_params.setdefault(provider, {}).setdefault((model_id, transport), set()).add(param)
    except TypeError:  # not weak-referenceable; the retry still happens
        return
    logger.info(
        "Upstream %s rejected '%s' for model %s over %s; dropping it for future Anthropic adapter requests",
        getattr(provider, "full_provider_name", type(provider).__name__), param, model_id, transport,
    )


def _request_params(request_dict: Dict[str, Any]) -> Set[str]:
    present = {k for k, v in request_dict.items() if v is not None}
    extra_body = request_dict.get("extra_body")
    if isinstance(extra_body, dict):
        present.update(extra_body.keys())
    for message in request_dict.get("messages") or ():
        if isinstance(message, dict) and message.get("reasoning_content"):
            present.add("reasoning_content")
            break
    reasoning = request_dict.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("summary"):
        present.add("reasoning.summary")
    return present


def _names_param(lowered: str, param: str) -> bool:
    """True when the error message names ``param`` as a whole token; for
    ambiguous everyday words it must also be quoted or labelled a parameter."""
    name = re.escape(param)
    if not re.search(rf"(?<![\w.-]){name}(?![\w-])", lowered):
        return False
    if param not in _AMBIGUOUS_PARAMS:
        return True
    return bool(
        re.search(rf"[\'\"`]{name}[\'\"`]", lowered)
        or re.search(rf"{_PARAM_CONTEXT}(?:\s+supplied)?\W{{0,5}}{name}(?![\w-])", lowered)
        or re.search(rf"(?<![\w.-]){name}\W{{0,5}}(?:is|are)?\s*(?:not supported|unsupported|not permitted|not allowed|unrecognized)", lowered)
    )


def _rejects_param(lowered: str, param: str) -> bool:
    """True when the error message says ``param`` itself is unsupported, not
    merely that it carries a bad value, schema or combination."""
    if not _names_param(lowered, param):
        return False
    name = rf"[\'\"`]?{re.escape(param)}[\'\"`]?"
    patterns = (
        # "Unrecognized request argument supplied: reasoning_effort",
        # "Unknown parameter: 'top_k'", "unknown field `top_k`"
        rf"(?:unsupported|unrecognized|unrecognised|unknown|not supported|not allowed|not permitted|extra)"
        rf"\s+(?:request\s+|body\s+)?{_PARAM_CONTEXT}(?:\s+supplied|\s+provided)?\W{{0,5}}{name}(?![\w-])",
        # "'top_p' is not supported with this model", "property 'x' is unsupported"
        rf"(?<![\w.-]){name}\W{{0,3}}(?:(?:is|are)\s+)?{_UNSUPPORTED_PREDICATE}{_NOT_ADJECTIVE}{_NOT_CONDITIONAL}",
        # "This model does not support parameter top_k"
        rf"(?:does not|doesn't|do not|don't)\s+(?:support|accept|recognize)\s+(?:the\s+)?"
        rf"(?:{_PARAM_CONTEXT}\s+)?{name}(?![\w-]){_NOT_VALUE_TAIL}",
        # pydantic-style servers (vLLM): ('body', 'top_k') ... extra inputs are not permitted
        rf"[\'\"`]{re.escape(param)}[\'\"`]\s*\).*extra(?:_forbidden| inputs are not permitted)",
    )
    return any(re.search(pattern, lowered, re.S) for pattern in patterns)


def _structured_param(error_param: str, candidates: Tuple[str, ...], present: Set[str]) -> Optional[str]:
    """Map an upstream ``param`` path ("messages[1].reasoning_content",
    "reasoning.effort") onto the retryable parameter it belongs to."""
    path = error_param.strip().lower()
    for param in candidates:
        if param in present and (path == param or path.endswith(f".{param}") or path.startswith(f"{param}.")):
            return param
    return None


def _retryable_param(
    status: Any,
    message: str,
    request_dict: Dict[str, Any],
    candidates: Tuple[str, ...],
    *,
    error_param: Any = None,
    error_code: Any = None,
) -> Optional[Tuple[str, bool]]:
    """``(param, remember)`` when the upstream rejected a sent ``param`` as
    unsupported, else None. ``remember`` is False for a request-local
    fallback that says nothing about what the model accepts."""
    try:
        status = int(status)
    except (TypeError, ValueError):
        return None
    if status not in (400, 422) or not message:
        return None
    lowered = message.lower()
    present = _request_params(request_dict)
    if (
        "reasoning.summary" in candidates
        and "reasoning.summary" in present
        and _REASONING_SUMMARY.search(lowered)
        and _ORG_VERIFICATION.search(lowered)
    ):
        # An unverified organization: drop the summary for this request only.
        return "reasoning.summary", False
    code = error_code.lower() if isinstance(error_code, str) else None
    if code in _UNSUPPORTED_PARAM_CODES and isinstance(error_param, str) and error_param:
        param = _structured_param(error_param, candidates, present)
        return (param, True) if param else None
    if code and _VALUE_ERROR_CODE.search(code):
        return None
    for param in candidates:
        if param in present and _rejects_param(lowered, param):
            return param, True
    return None


def _strip_params(adapted: Any, params: Iterable[str]) -> Any:
    params = set(params)
    if not params:
        return adapted
    data = adapted.model_dump(exclude_none=True)
    for param in params:
        if param == "top_k":
            extra_body = data.get("extra_body")
            if isinstance(extra_body, dict):
                extra_body.pop("top_k", None)
                if not extra_body:
                    data.pop("extra_body", None)
        elif param == "reasoning_content":
            for message in data.get("messages") or ():
                if isinstance(message, dict):
                    message.pop("reasoning_content", None)
        elif param == "reasoning.summary":
            reasoning = data.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning.pop("summary", None)
                if not reasoning:
                    data.pop("reasoning", None)
        else:
            data.pop(param, None)
    stripped = type(adapted)(**data)
    if isinstance(stripped, ChatCompletionRequest) and (
        "stream_options" in params or getattr(adapted, "_omit_stream_options", False)
    ):
        # Keep the provider from re-adding its include_usage default.
        stripped._omit_stream_options = True
    return stripped


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------

def _sanitize_block_ids(block_data: Dict[str, Any]) -> None:
    block_type = block_data.get("type")
    if block_type in ("tool_use", "server_tool_use") and "id" in block_data:
        block_data["id"] = sanitize_tool_id(block_data.get("id"))
    if block_type in ("tool_result", "web_search_tool_result", "web_fetch_tool_result") and "tool_use_id" in block_data:
        block_data["tool_use_id"] = sanitize_tool_id(block_data.get("tool_use_id"))


def prepare_anthropic_adapter_request(
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    echo_reasoning: bool = False,
    anthropic_beta: Optional[str] = None,
    supports_top_k: bool = True,
    unforwarded_params: Iterable[str] = (),
) -> Tuple[AnthropicMessagesRequest, AnthropicRequestMetadata]:
    """Normalize an Anthropic request for adapter-backed execution.

    Every field the adapter will not forward is recorded in
    ``metadata.dropped_fields`` here, because the route builds the
    ``x-llmproxy-dropped-anthropic-fields`` header before the upstream call.
    ``supports_top_k`` and ``unforwarded_params`` (chat parameters the
    provider's wire preparation filters out) describe the provider, so those
    drops happen here too, under their Anthropic source names.
    """
    raw = request.model_dump(exclude_none=True)
    dropped_fields: List[str] = []
    # Read before unsupported blocks (e.g. tool_search_tool_result) are dropped.
    loaded_tools = _loaded_tool_names(raw.get("messages"))

    for key in list(raw.keys()):
        if key in _TOP_LEVEL_ADAPTER_ONLY_FIELDS:
            raw.pop(key, None)
            dropped_fields.append(key)
        elif key not in _KNOWN_ANTHROPIC_FIELDS:
            # "metadata" is already in _KNOWN_ANTHROPIC_FIELDS, so no separate
            # exclusion is needed here.
            raw.pop(key, None)
            dropped_fields.append(key)

    if anthropic_beta:
        dropped_fields.append("anthropic-beta")

    if not allow_thinking:
        if raw.pop("thinking", None) is not None:
            dropped_fields.append("thinking")
        output_config = raw.get("output_config")
        if isinstance(output_config, dict) and output_config.pop("effort", None) is not None:
            dropped_fields.append("output_config.effort")

    output_config = raw.get("output_config")
    if isinstance(output_config, dict):
        for key in list(output_config.keys()):
            if key not in ("effort", "format"):
                output_config.pop(key)
                dropped_fields.append(f"output_config.{key}")
        if not output_config:
            raw.pop("output_config", None)

    stop_sequences = raw.get("stop_sequences")
    if transport == "responses":
        # The Responses API has no stop or top_k parameter.
        for key in ("stop_sequences", "top_k"):
            if raw.pop(key, None) is not None:
                dropped_fields.append(key)
    elif isinstance(stop_sequences, list) and len(stop_sequences) > MAX_STOP_SEQUENCES:
        raw["stop_sequences"] = stop_sequences[:MAX_STOP_SEQUENCES]
        dropped_fields.append("stop_sequences")

    unforwarded = set(unforwarded_params or ()) if transport == "chat" else set()
    if transport == "chat" and not supports_top_k and raw.pop("top_k", None) is not None:
        dropped_fields.append("top_k")
    output_config = raw.get("output_config")
    if "response_format" in unforwarded and isinstance(output_config, dict) and response_format_for(output_config):
        output_config.pop("format", None)
        dropped_fields.append("output_config.format")
        if not output_config:
            raw.pop("output_config", None)
    request_metadata = raw.get("metadata")
    if "user" in unforwarded and isinstance(request_metadata, dict) and request_metadata.get("user_id"):
        request_metadata.pop("user_id", None)
        dropped_fields.append("metadata.user_id")
        if not request_metadata:
            raw.pop("metadata", None)

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

                if block_type == "thinking" and echo_reasoning:
                    normalized_blocks.append(block_data)
                    continue
                if block_type in {"thinking", "redacted_thinking"}:
                    dropped_fields.append("messages.thinking")
                    continue

                if block_type and block_type not in _SUPPORTED_CONTENT_TYPES:
                    dropped_fields.append(f"messages.{block_type}")
                    continue

                if block_type == "text" and not str(block_data.get("text") or "").strip():
                    # Anthropic clients occasionally send empty text blocks;
                    # several OpenAI-compatible backends reject them.
                    continue

                source = block_data.get("source") if block_type in ("image", "document") else None
                if isinstance(source, dict) and source.get("type") in ("file", "url") and block_type == "document":
                    dropped_fields.append(f"messages.document.{source['type']}")
                elif isinstance(source, dict) and source.get("type") == "file":
                    dropped_fields.append("messages.image.file")

                _sanitize_block_ids(block_data)
                normalized_blocks.append(block_data)
            if not normalized_blocks:
                continue
            message_data["content"] = normalized_blocks
        elif isinstance(content, str) and not content and message_data.get("role") == "assistant":
            continue
        normalized_messages.append(message_data)
    raw["messages"] = normalized_messages

    # Emulate tool search: deferred tools are declared once the history loads
    # them. Not when the client relies on the server-side search tool (the
    # adapter can't run it) or defers everything, as nothing could load them.
    raw_tools = [_to_dict(tool) for tool in raw.get("tools", []) or []]
    defer_tools = not any(
        str(t.get("type") or "").startswith("tool_search_tool") for t in raw_tools
    ) and any(not t.get("defer_loading") and not _is_server_tool(t) for t in raw_tools)
    withheld = 0

    normalized_tools = []
    for index, tool_data in enumerate(raw_tools):
        tool_type = str(tool_data.get("type") or "")
        tool_name = str(tool_data.get("name") or "")

        if tool_data.pop("cache_control", None) is not None:
            dropped_fields.append("tools.cache_control")

        if _is_server_tool(tool_data) or (
            tool_type and any(tool_type.startswith(prefix) for prefix in _SPECIAL_ADAPTER_TOOL_TYPES)
        ):
            dropped_fields.append(f"tools.{tool_type or tool_name or index}")
            continue

        # Withheld on purpose, so not reported as dropped.
        if tool_data.pop("defer_loading", None) and defer_tools and tool_name not in loaded_tools:
            withheld += 1
            continue

        if not tool_name:
            tool_data["name"] = f"tool_{index + 1}"

        input_schema = tool_data.get("input_schema")
        if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
            input_schema["type"] = "object"
            dropped_fields.append("tools.input_schema.type")

        normalized_tools.append(tool_data)

    if withheld:
        logger.debug("Withholding %d deferred tool(s) not yet loaded by tool search", withheld)

    if normalized_tools:
        raw["tools"] = normalized_tools
        tool_choice = raw.get("tool_choice")
        if (
            "parallel_tool_calls" in unforwarded
            and isinstance(tool_choice, dict)
            and tool_choice.get("type") != "none"
            and tool_choice.get("disable_parallel_tool_use")
        ):
            # Only a true value becomes parallel_tool_calls=False upstream.
            tool_choice.pop("disable_parallel_tool_use")
            dropped_fields.append("tool_choice.disable_parallel_tool_use")
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


def _input_converter(provider: Any, *, transport: str, echo_reasoning: bool) -> Any:
    if transport == "responses":
        return AnthropicToResponsesConverter()
    return AnthropicToOpenAIConverter(
        echo_reasoning=echo_reasoning,
        supports_top_k=bool(getattr(provider, "anthropic_adapter_supports_top_k", True)),
        supports_file_parts=bool(getattr(provider, "anthropic_adapter_supports_file_parts", False)),
        thought_signatures=bool(getattr(provider, "anthropic_adapter_thought_signatures", False)),
    )


def _prepare_for_provider(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool,
    echo_reasoning: bool,
    anthropic_beta: Optional[str],
    stream: bool,
) -> Tuple[str, str, AnthropicMessagesRequest, AnthropicRequestMetadata, Any, Any]:
    """Prepare and convert ``request`` exactly as execution will send it.

    Returns ``(model_id, transport, sanitized_request, metadata, converter_in,
    adapted_request)``; metadata and execution share this so the reported
    drops match the wire request.
    """
    model_id = provider.get_model_id(request.model)
    transport = resolve_adapter_transport(provider, model_id, transport)
    sanitized_request, metadata = prepare_anthropic_adapter_request(
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        echo_reasoning=echo_reasoning,
        anthropic_beta=anthropic_beta,
        supports_top_k=bool(getattr(provider, "anthropic_adapter_supports_top_k", True)),
        unforwarded_params=getattr(provider, "anthropic_adapter_unforwarded_params", ()),
    )
    converter_in = _input_converter(provider, transport=transport, echo_reasoning=echo_reasoning)
    # The routed name goes upstream unchanged: the provider's own call path
    # strips its prefix, and doing it here too would strip twice.
    adapted_request = converter_in.convert_request(sanitized_request, sanitized_request.model)
    adapted_request.stream = stream
    rejected = known_rejected_params(provider, model_id, transport)
    rejected &= _request_params(adapted_request.model_dump(exclude_none=True))
    adapted_request = _strip_params(adapted_request, rejected)
    # stream_options is generated by the adapter, not sent by the client.
    metadata.dropped_fields = _dedupe([*metadata.dropped_fields, *sorted(rejected - {"stream_options"})])
    return model_id, transport, sanitized_request, metadata, converter_in, adapted_request


def anthropic_adapter_metadata(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    echo_reasoning: bool = False,
    anthropic_beta: Optional[str] = None,
) -> AnthropicRequestMetadata:
    """Routing metadata for an adapter-served request (drops included).

    Rejections first learned while this request runs cannot be reported: the
    route sends these headers before the upstream call.
    """
    _, _, _, metadata, _, _ = _prepare_for_provider(
        provider,
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        echo_reasoning=echo_reasoning,
        anthropic_beta=anthropic_beta,
        stream=bool(request.stream),
    )
    return metadata


def _output_converter(provider: Any, request: AnthropicMessagesRequest, converter_in: Any, transport: str) -> Any:
    if transport == "responses":
        return ResponsesToAnthropicConverter(tool_names=converter_in.tool_names)
    return OpenAIToAnthropicConverter(
        tool_names=converter_in.tool_names,
        stop_sequences=request.stop_sequences,
        split_think_tags=bool(getattr(provider, "split_think_tags", False)),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def anthropic_adapter_messages(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    echo_reasoning: bool = False,
    anthropic_beta: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a non-streaming adapter-backed Anthropic request."""
    model_id, transport, sanitized_request, _, converter_in, adapted_request = _prepare_for_provider(
        provider,
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        echo_reasoning=echo_reasoning,
        anthropic_beta=anthropic_beta,
        stream=False,
    )
    converter_out = _output_converter(provider, sanitized_request, converter_in, transport)

    call = provider.responses_create if transport == "responses" else provider.chat_completion
    candidates = _RETRYABLE_PARAMS_RESPONSES if transport == "responses" else _RETRYABLE_PARAMS_CHAT

    retries = 0
    while True:
        try:
            response_obj = await call(adapted_request)
            break
        except ProviderHTTPError as error:
            message = _provider_error_message(error)
            if _switch_to_responses(provider, model_id, transport, message):
                return await anthropic_adapter_messages(
                    provider,
                    request,
                    transport="responses",
                    allow_thinking=allow_thinking,
                    echo_reasoning=echo_reasoning,
                    anthropic_beta=anthropic_beta,
                )
            inner = error.body.get("error") if isinstance(error.body, dict) else None
            inner = inner if isinstance(inner, dict) else {}
            rejection = _retryable_param(
                error.status_code,
                message,
                adapted_request.model_dump(exclude_none=True),
                candidates,
                error_param=inner.get("param"),
                error_code=inner.get("code"),
            )
            if rejection is None or retries >= _MAX_PARAM_RETRIES:
                raise to_anthropic_http_error(error) from error
            param, remember = rejection
            retries += 1
            if remember:
                _remember_rejected(provider, model_id, transport, param)
            adapted_request = _strip_params(adapted_request, {param})

    if hasattr(response_obj, "model_dump"):
        response_obj = response_obj.model_dump(exclude_unset=True)
    if transport == "responses" and response_obj.get("status") == "failed":
        # A failed response arrives as a 200; surface it as the error it is.
        status, _, message = responses_error_status(response_obj.get("error"))
        raise ProviderHTTPError(status, message, body=anthropic_error_body(status, message))
    return converter_out.convert_response(response_obj, echo_model_name(request))


def _parse_chat_sse(sse_chunk: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Return ``(is_done, payload)`` for one OpenAI chat SSE chunk."""
    if sse_chunk.startswith("data: [DONE]"):
        return True, None
    if not sse_chunk.startswith("data: "):
        return False, None
    try:
        payload = json.loads(sse_chunk[6:].strip())
    except (json.JSONDecodeError, ValueError):
        return False, None
    return False, payload if isinstance(payload, dict) else None


def _parse_responses_sse(sse_chunk: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
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
    if event_type is None and isinstance(event_data, dict) and isinstance(event_data.get("type"), str):
        # Upstreams that send bare ``data:`` lines carry the type in the payload.
        event_type = event_data["type"]
    return event_type, event_data


async def _close_stream(stream: Any) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:  # pragma: no cover - best effort
            pass


async def anthropic_adapter_messages_stream(
    provider: Any,
    request: AnthropicMessagesRequest,
    *,
    transport: str,
    allow_thinking: bool = False,
    echo_reasoning: bool = False,
    anthropic_beta: Optional[str] = None,
) -> AsyncIterator[str]:
    """Execute a streaming adapter-backed Anthropic request.

    ``message_start`` is held back until the upstream produces its first
    chunk, so a 400 caused by an unsupported optional parameter can be retried
    without that parameter before anything reaches the client.
    """
    model_id, transport, sanitized_request, _, converter_in, adapted_request = _prepare_for_provider(
        provider,
        request,
        transport=transport,
        allow_thinking=allow_thinking,
        echo_reasoning=echo_reasoning,
        anthropic_beta=anthropic_beta,
        stream=True,
    )
    converter_out = _output_converter(provider, sanitized_request, converter_in, transport)

    message_id = f"msg_{uuid.uuid4().hex}"
    model_name = echo_model_name(request)
    # tiktoken releases the GIL, and its first use may download the encoding;
    # keep both off the event loop.
    input_estimate = await asyncio.to_thread(estimate_anthropic_input_tokens, sanitized_request)
    if transport == "responses":
        state = StreamConversionState(message_id=message_id, model=model_name, input_tokens=input_estimate)
        open_stream = provider.responses_create_stream
        candidates = _RETRYABLE_PARAMS_RESPONSES
    else:
        state = converter_out.new_stream_state(message_id, model_name, input_estimate)
        open_stream = provider.chat_completion_stream
        candidates = _RETRYABLE_PARAMS_CHAT
    start_event = converter_out.convert_stream_start(message_id, model_name, input_estimate)

    started = False
    retries = 0
    switch = False
    while True:
        retry = False
        upstream = open_stream(adapted_request)
        try:
            async for sse_chunk in upstream:
                if transport == "responses":
                    event_type, payload = _parse_responses_sse(sse_chunk)
                    is_error = event_type == "error"
                else:
                    is_done, payload = _parse_chat_sse(sse_chunk)
                    if is_done:
                        break
                    event_type = None
                    is_error = bool(payload and payload.get("error"))

                if is_error:
                    error = (payload or {}).get("error") if isinstance((payload or {}).get("error"), dict) else {}
                    message = error.get("message") or (payload or {}).get("message") or "Upstream stream failed"
                    status = error.get("code")
                    if not started and _switch_to_responses(provider, model_id, transport, message):
                        switch = True
                        break
                    if not started and retries < _MAX_PARAM_RETRIES:
                        # The provider's SSE wrapper puts the HTTP status in
                        # "code" and drops "param", so this rests on the message.
                        rejection = _retryable_param(
                            status, message, adapted_request.model_dump(exclude_none=True), candidates,
                        )
                        if rejection is not None:
                            param, remember = rejection
                            retries += 1
                            if remember:
                                _remember_rejected(provider, model_id, transport, param)
                            adapted_request = _strip_params(adapted_request, {param})
                            retry = True
                            break
                    if not started:
                        yield start_event
                    yield _format_anthropic_error_sse(message, status)
                    return

                if payload is None:
                    continue
                if not started:
                    yield start_event
                    started = True

                if transport == "responses":
                    if event_type:
                        for event in converter_out.convert_stream_event(event_type, payload, state):
                            yield event
                else:
                    for event in converter_out.convert_stream_chunk(payload, state):
                        yield event
        finally:
            await _close_stream(upstream)
        if switch:
            async for event in anthropic_adapter_messages_stream(
                provider,
                request,
                transport="responses",
                allow_thinking=allow_thinking,
                echo_reasoning=echo_reasoning,
                anthropic_beta=anthropic_beta,
            ):
                yield event
            return
        if not retry:
            break

    if not started:
        yield start_event
    for event in converter_out.convert_stream_end(state):
        yield event

"""Shared helpers for the Anthropic <-> OpenAI adapter.

Pure functions and small state objects used by the request/response/stream
converters in ``app.conversion.anthropic_openai``: tool name/id sanitizing,
tool-argument repair, thinking -> reasoning_effort mapping, strict JSON
schemas, usage and stop-sequence mapping, url_citation annotations and a
``<think>`` tag splitter for models that inline their reasoning.
"""

import base64
import binascii
import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# OpenAI function names: ^[a-zA-Z0-9_-]{1,64}$
MAX_TOOL_NAME_LENGTH = 64
_TOOL_NAME_HASH_LENGTH = 8
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_VALID_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Anthropic tool_use ids: ^[a-zA-Z0-9_-]+$
_INVALID_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Claude Code stamps a per-request billing line into the system prompt; it
# defeats upstream prefix caching on OpenAI-compatible servers.
_BILLING_HEADER_LINE = re.compile(r"^x-anthropic-billing-header:.*(?:\r?\n|$)", re.MULTILINE)

# OpenAI rejects more than 4 stop sequences.
MAX_STOP_SEQUENCES = 4

_EFFORT_ALIASES = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


# ---------------------------------------------------------------------------
# Tool names and ids
# ---------------------------------------------------------------------------

def _shorten_tool_name(name: str) -> str:
    sanitized = _INVALID_NAME_CHARS.sub("_", name) or "tool"
    if len(sanitized) <= MAX_TOOL_NAME_LENGTH and sanitized == name:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_TOOL_NAME_HASH_LENGTH]
    prefix_len = MAX_TOOL_NAME_LENGTH - _TOOL_NAME_HASH_LENGTH - 1
    return f"{sanitized[:prefix_len]}_{digest}"


class ToolNameMap:
    """Deterministic Anthropic <-> OpenAI tool-name mapping.

    Names that already satisfy OpenAI's function-name rules map to themselves.
    Others are sanitized and, when needed, truncated with a hash suffix. The
    mapping is a pure function of the name, so a name seen only in history
    (or only in ``tool_choice``) maps the same way as its tool definition.
    """

    def __init__(self) -> None:
        self._forward: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}

    def to_openai(self, name: Optional[str]) -> str:
        name = name or ""
        if not name:
            return name
        mapped = self._forward.get(name)
        if mapped is None:
            mapped = name if _VALID_NAME.match(name) else _shorten_tool_name(name)
            self._forward[name] = mapped
            self._reverse.setdefault(mapped, name)
        return mapped

    def to_anthropic(self, name: Optional[str]) -> str:
        name = name or ""
        return self._reverse.get(name, name)

    @property
    def reverse(self) -> Dict[str, str]:
        return {k: v for k, v in self._reverse.items() if k != v}


_ENCODED_ID_PREFIX = "toolu_enc_"


def sanitize_tool_id(raw_id: Any, prefix: str = "toolu_") -> str:
    """Return an id matching ``^[a-zA-Z0-9_-]+$``; generate one if missing.

    Valid ids pass through untouched. Others are base64url-encoded rather
    than rewritten, so :func:`restore_tool_id` can hand the upstream its
    exact id back: Gemini thought signatures ride in ids such as
    ``call_x__thought__<base64 sig>`` and must round-trip byte for byte.
    """
    if raw_id is None or raw_id == "":
        return f"{prefix}{uuid.uuid4().hex[:24]}"
    text = str(raw_id)
    if not _INVALID_ID_CHARS.search(text):
        return text
    encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_ENCODED_ID_PREFIX}{encoded}"


def restore_tool_id(tool_id: Any) -> str:
    """Invert :func:`sanitize_tool_id` for ids sent back upstream."""
    text = "" if tool_id is None else str(tool_id)
    if not text.startswith(_ENCODED_ID_PREFIX):
        return text
    payload = text[len(_ENCODED_ID_PREFIX):]
    try:
        return base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
    except (binascii.Error, ValueError):
        return text


# ---------------------------------------------------------------------------
# Tool arguments
# ---------------------------------------------------------------------------

def _close_json(text: str) -> str:
    """Append the closers needed to balance a truncated JSON object/array."""
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    suffix = '"' if in_string else ""
    return text + suffix + "".join(reversed(stack))


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Parse OpenAI tool-call arguments into an Anthropic ``input`` object.

    Empty -> ``{}``. Truncated JSON is repaired by closing open strings and
    brackets. Anything still unparseable, or a non-object value, is wrapped
    as ``{"_raw_arguments": <original>}`` rather than silently discarded.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    text = str(arguments).strip()
    if not text:
        return {}
    for candidate in (text, _close_json(text)):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        break
    return {"_raw_arguments": str(arguments)}


# ---------------------------------------------------------------------------
# Thinking / reasoning
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    if value is default and hasattr(obj, "model_extra") and obj.model_extra:
        return obj.model_extra.get(key, default)
    return value


def normalize_effort(effort: Any) -> Optional[str]:
    if not isinstance(effort, str):
        return None
    return _EFFORT_ALIASES.get(effort.strip().lower())


def effort_from_budget(budget_tokens: Optional[int]) -> str:
    """Map an Anthropic thinking budget onto an OpenAI reasoning effort."""
    if budget_tokens is None:
        return "medium"
    if budget_tokens >= 4096:
        return "high"
    if budget_tokens >= 2048:
        return "medium"
    return "low"


def reasoning_effort_for(thinking: Any, output_config: Any) -> Optional[str]:
    """Translate Anthropic ``thinking`` + ``output_config.effort``.

    - ``disabled``           -> None (omit the parameter)
    - ``adaptive``           -> output_config.effort, default "medium"
    - ``enabled``            -> output_config.effort, else budget buckets
    - no thinking but effort -> effort
    """
    effort = normalize_effort(_get(output_config, "effort"))
    thinking_type = _get(thinking, "type")
    if thinking is None:
        return effort
    if thinking_type == "disabled":
        return None
    if thinking_type == "adaptive":
        return effort or "medium"
    if thinking_type == "enabled":
        return effort or effort_from_budget(_get(thinking, "budget_tokens"))
    return effort


def reasoning_text_from_message(message: Dict[str, Any]) -> str:
    """Collect reasoning text from the various OpenAI-compatible fields."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts = [d.get("text") or d.get("summary") or "" for d in details if isinstance(d, dict)]
        return "".join(p for p in parts if isinstance(p, str))
    return ""


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

def make_strict_schema(schema: Any) -> Optional[Any]:
    """Return a copy of ``schema`` usable with OpenAI strict mode, or None.

    Strict mode needs every property required and ``additionalProperties:
    false`` on each object. The latter is filled in when absent; a schema
    with optional properties or open objects can't be made strict without
    changing its meaning, so it returns None.
    """
    schema = copy.deepcopy(schema)

    def _walk(node: Any) -> bool:
        if isinstance(node, list):
            return all(_walk(item) for item in node)
        if not isinstance(node, dict):
            return True
        props = node.get("properties")
        if node.get("type") == "object" or isinstance(props, dict):
            if node.get("additionalProperties", False) is not False:
                return False
            if isinstance(props, dict):
                if set(props) - set(node.get("required") or ()):
                    return False
                node["additionalProperties"] = False
                if not all(_walk(sub) for sub in props.values()):
                    return False
        return all(_walk(value) for key, value in node.items() if key != "properties")

    return schema if _walk(schema) else None


def response_format_for(output_config: Any) -> Optional[Dict[str, Any]]:
    fmt = _get(output_config, "format")
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        return None
    strict_schema = make_strict_schema(fmt["schema"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "output",
            "schema": strict_schema if strict_schema is not None else fmt["schema"],
            "strict": strict_schema is not None,
        },
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def strip_billing_header(text: str) -> str:
    if "x-anthropic-billing-header" not in text:
        return text
    return _BILLING_HEADER_LINE.sub("", text)


# ---------------------------------------------------------------------------
# Usage, stop reason
# ---------------------------------------------------------------------------

def anthropic_usage_from_openai(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Map OpenAI usage to Anthropic usage (cache fields only when non-zero)."""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = 0
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    if not cached:
        # DeepSeek reports cache hits at the top level.
        cached = int(usage.get("prompt_cache_hit_tokens") or 0)
    result = {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": completion,
    }
    if cached:
        result["cache_read_input_tokens"] = cached
    return result


def matched_stop_sequence(choice: Dict[str, Any], stop_sequences: Optional[Iterable[str]]) -> Optional[str]:
    """Return the stop sequence the upstream reports as matched, if any.

    vLLM puts the matched string in ``choice.stop_reason``; llama.cpp uses
    ``stopping_word``. Only values the client actually asked for count.
    """
    if not stop_sequences:
        return None
    wanted = set(stop_sequences)
    for key in ("stop_reason", "stopping_word", "matched_stop"):
        value = choice.get(key)
        if isinstance(value, str) and value in wanted:
            return value
    return None


# ---------------------------------------------------------------------------
# url_citation annotations -> Anthropic citations
# ---------------------------------------------------------------------------

def citations_from_annotations(text: str, annotations: Any) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    if not isinstance(annotations, list):
        return citations
    for ann in annotations:
        if not isinstance(ann, dict) or ann.get("type") != "url_citation":
            continue
        info = ann.get("url_citation") if isinstance(ann.get("url_citation"), dict) else ann
        url = info.get("url")
        if not url:
            continue
        start, end = info.get("start_index"), info.get("end_index")
        cited = ""
        if isinstance(start, int) and isinstance(end, int) and text:
            cited = text[max(start, 0):max(end, 0)]
        citations.append({
            "type": "web_search_result_location",
            "url": url,
            "title": info.get("title") or url,
            "encrypted_index": "",
            "cited_text": cited[:150],
        })
    return citations


# ---------------------------------------------------------------------------
# <think> tag splitting
# ---------------------------------------------------------------------------

@dataclass
class ThinkTagSplitter:
    """Route ``<think>...</think>`` text in content to thinking output.

    Streaming-safe: a tag split across chunks is held back until it can be
    classified. ``feed`` returns ``[(kind, text), ...]`` with kind in
    ``{"thinking", "text"}``; ``flush`` releases anything held at the end.
    """

    open_tag: str = "<think>"
    close_tag: str = "</think>"
    in_think: bool = False
    seen_text: bool = False
    _buffer: str = ""

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        self._buffer += chunk
        out: List[Tuple[str, str]] = []
        while self._buffer:
            tag = self.close_tag if self.in_think else self.open_tag
            idx = self._buffer.find(tag)
            if idx >= 0:
                before = self._buffer[:idx]
                if before:
                    self._emit(out, before)
                self._buffer = self._buffer[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            # Hold back a possible partial tag at the end of the buffer.
            hold = 0
            for size in range(min(len(tag) - 1, len(self._buffer)), 0, -1):
                if tag.startswith(self._buffer[-size:]):
                    hold = size
                    break
            emit = self._buffer[: len(self._buffer) - hold]
            if emit:
                self._emit(out, emit)
            self._buffer = self._buffer[len(self._buffer) - hold:]
            break
        return out

    def flush(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if self._buffer:
            self._emit(out, self._buffer)
            self._buffer = ""
        return out

    def _emit(self, out: List[Tuple[str, str]], text: str) -> None:
        kind = "thinking" if self.in_think else "text"
        if kind == "text":
            if not self.seen_text:
                text = text.lstrip("\n")
                if not text:
                    return
            self.seen_text = True
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + text)
        else:
            out.append((kind, text))


def split_think_tags(text: str) -> Tuple[str, str]:
    """Non-streaming split: return ``(thinking, text)``."""
    splitter = ThinkTagSplitter()
    parts = splitter.feed(text) + splitter.flush()
    thinking = "".join(t for k, t in parts if k == "thinking")
    visible = "".join(t for k, t in parts if k == "text")
    return thinking, visible

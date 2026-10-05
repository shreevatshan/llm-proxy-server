"""Web search tool detection, rewriting and history rehydration.

Port of LiteLLM's integrations/websearch_interception/tools.py, adapted to
the three API surfaces this proxy serves. Everything here works on plain
dicts so it stays independent of the request models.
"""

import base64
import binascii
import copy
import json
import secrets
from typing import Any, Dict, List, Optional, Tuple

from app.websearch.searxng import SearchOutcome, format_outcome_text

INTERNAL_TOOL_NAME = "web_search_proxy"
# Also accepted from clients (LiteLLM compatibility).
INTERNAL_TOOL_NAMES = frozenset({INTERNAL_TOOL_NAME, "litellm_web_search"})

TOOL_DESCRIPTION = (
    "Search the web for information. Use this when you need current information "
    "or answers to questions that require up-to-date data."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query"},
    },
    "required": ["query"],
}

# Ids of the blocks/items we synthesize. The prefix tells our own replayed
# blocks apart from genuine upstream server-tool blocks.
SERVER_TOOL_ID_PREFIX = "srvtoolu_wsp"
RESPONSES_CALL_ID_PREFIX = "ws_wsp"
_ENCRYPTED_PREFIX = "wsp1:"


def new_server_tool_id() -> str:
    return f"{SERVER_TOOL_ID_PREFIX}{secrets.token_hex(10)}"


def new_responses_call_id() -> str:
    return f"{RESPONSES_CALL_ID_PREFIX}{secrets.token_hex(10)}"


def _as_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return obj


def query_from_input(tool_input: Any) -> str:
    """Extract the query from a tool call's input (dict or JSON string)."""
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input) if tool_input.strip() else {}
        except ValueError:
            return tool_input.strip()
    if isinstance(tool_input, dict):
        query = tool_input.get("query")
        if isinstance(query, str):
            return query.strip()
    return ""


# ==================== Anthropic Messages ====================

def anthropic_internal_tool() -> dict:
    return {
        "name": INTERNAL_TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "input_schema": copy.deepcopy(TOOL_INPUT_SCHEMA),
    }


def is_web_search_tool_anthropic(tool: Any) -> bool:
    tool = _as_dict(tool)
    if not isinstance(tool, dict):
        return False
    name = tool.get("name")
    tool_type = tool.get("type")
    if name in INTERNAL_TOOL_NAMES:
        return True
    if isinstance(tool_type, str) and tool_type.startswith("web_search_"):
        return True
    # Claude Code style: {"type": "...", "name": "web_search"} with no schema.
    if name == "web_search" and tool_type and "input_schema" not in tool:
        return True
    return False


def is_native_web_search_tool_anthropic(tool: Any) -> bool:
    tool = _as_dict(tool)
    tool_type = tool.get("type") if isinstance(tool, dict) else None
    return isinstance(tool_type, str) and tool_type.startswith("web_search_")


def rewrite_anthropic_tools(tools: Optional[List[Any]]) -> Tuple[List[Any], bool, bool]:
    """Replace web search tools with the internal function tool.

    Returns ``(tools, found, had_native)``.
    """
    if not tools:
        return list(tools or []), False, False
    found = False
    had_native = False
    out: List[Any] = []
    for tool in tools:
        if is_web_search_tool_anthropic(tool):
            found = True
            had_native = had_native or is_native_web_search_tool_anthropic(tool)
            continue
        out.append(tool)
    if found:
        out.append(anthropic_internal_tool())
    return out, found, had_native


def ensure_anthropic_internal_tool(tools: List[Any]) -> List[Any]:
    if any(isinstance(_as_dict(t), dict) and _as_dict(t).get("name") == INTERNAL_TOOL_NAME for t in tools):
        return tools
    return [*tools, anthropic_internal_tool()]


def remap_anthropic_tool_choice(tool_choice: Any) -> Any:
    tc = _as_dict(tool_choice)
    if isinstance(tc, dict) and tc.get("type") == "tool" and tc.get("name") in (
        "web_search", *INTERNAL_TOOL_NAMES
    ):
        return {**tc, "name": INTERNAL_TOOL_NAME}
    return tool_choice


def _encode_text(text: str) -> str:
    return _ENCRYPTED_PREFIX + base64.b64encode(text.encode("utf-8")).decode("ascii")


def _decode_text(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.startswith(_ENCRYPTED_PREFIX):
        return None
    try:
        return base64.b64decode(value[len(_ENCRYPTED_PREFIX):]).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def build_native_blocks(tool_id: str, outcome: SearchOutcome) -> List[dict]:
    """``server_tool_use`` + ``web_search_tool_result`` blocks for one search.

    The full tool-result text rides along in the first result's
    ``encrypted_content`` (opaque to clients) so a replayed history can be
    turned back into the exact tool_result the model saw.
    """
    server_tool_use = {
        "type": "server_tool_use",
        "id": tool_id,
        "name": "web_search",
        "input": {"query": outcome.query},
    }
    if not outcome.ok:
        content: Any = {"type": "web_search_tool_result_error", "error_code": outcome.error_code}
    else:
        content = []
        for i, r in enumerate(outcome.results):
            content.append({
                "type": "web_search_result",
                "url": r.url,
                "title": r.title,
                "encrypted_content": _encode_text(format_outcome_text(outcome)) if i == 0 else "",
                "page_age": r.date,
            })
    return [
        server_tool_use,
        {"type": "web_search_tool_result", "tool_use_id": tool_id, "content": content},
    ]


def _replayed_result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, dict):
        code = content.get("error_code") or "unavailable"
        return f"Search failed: {code}"
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = _decode_text(item.get("encrypted_content"))
                if text is not None:
                    return text
        if not content:
            return "No results found."
        # Ours, but the text did not survive: rebuild a minimal listing.
        return "\n\n".join(
            f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}"
            for item in content
            if isinstance(item, dict)
        )
    return "No results found."


def _is_ours(block: Any, key: str) -> bool:
    return (
        isinstance(block, dict)
        and isinstance(block.get(key), str)
        and block[key].startswith(SERVER_TOOL_ID_PREFIX)
    )


def history_has_replayed_blocks(messages: List[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "server_tool_use" and _is_ours(block, "id"):
                    return True
    return False


def _content_list(content: Any) -> List[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content or [])


def _merge_consecutive(messages: List[dict]) -> List[dict]:
    """Merge adjacent same-role messages; user tool_results stay first."""
    merged: List[dict] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            prev["content"] = _content_list(prev["content"]) + _content_list(msg["content"])
        else:
            merged.append(dict(msg))
    return merged


def rehydrate_anthropic_messages(messages: List[dict]) -> List[dict]:
    """Turn our replayed native blocks back into tool_use/tool_result turns.

    One final assistant message carries every search round::

        [thinking, text, server_tool_use, web_search_tool_result, text]

    which becomes::

        assistant[thinking, text, tool_use]  user[tool_result]  assistant[text]
    """
    if not history_has_replayed_blocks(messages):
        return messages
    out: List[dict] = []
    for msg in messages:
        content = msg.get("content")
        if msg.get("role") != "assistant" or not isinstance(content, list):
            out.append(msg)
            continue
        segment: List[Any] = []
        pending_results: List[dict] = []
        for block in content:
            btype = block.get("type") if isinstance(block, dict) else None
            if btype == "server_tool_use" and _is_ours(block, "id"):
                if pending_results:
                    out.append({"role": "assistant", "content": segment})
                    out.append({"role": "user", "content": pending_results})
                    segment, pending_results = [], []
                segment.append({
                    "type": "tool_use",
                    "id": block["id"],
                    "name": INTERNAL_TOOL_NAME,
                    "input": {"query": query_from_input(block.get("input"))},
                })
            elif btype == "web_search_tool_result" and _is_ours(block, "tool_use_id"):
                pending_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["tool_use_id"],
                    "content": _replayed_result_text(block),
                })
            else:
                if pending_results:
                    out.append({"role": "assistant", "content": segment})
                    out.append({"role": "user", "content": pending_results})
                    segment, pending_results = [], []
                segment.append(block)
        if pending_results:
            out.append({"role": "assistant", "content": segment})
            out.append({"role": "user", "content": pending_results})
        elif segment:
            out.append({**msg, "content": segment})
    return _merge_consecutive(out)


# ==================== Chat Completions ====================

def chat_internal_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": INTERNAL_TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": copy.deepcopy(TOOL_INPUT_SCHEMA),
        },
    }


def _chat_tool_name(tool: Any) -> Optional[str]:
    tool = _as_dict(tool)
    if isinstance(tool, dict):
        fn = tool.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
    return None


def detect_chat(request_dict: dict) -> bool:
    if "web_search_options" in request_dict:
        return True
    return any(_chat_tool_name(t) in INTERNAL_TOOL_NAMES for t in request_dict.get("tools") or [])


def rewrite_chat_request(request_dict: dict) -> dict:
    """Drop ``web_search_options`` and declare the internal function tool once."""
    out = dict(request_dict)
    out.pop("web_search_options", None)
    tools = [t for t in (out.get("tools") or []) if _chat_tool_name(t) not in INTERNAL_TOOL_NAMES]
    tools.append(chat_internal_tool())
    out["tools"] = tools
    tc = out.get("tool_choice")
    if isinstance(tc, dict) and (tc.get("function") or {}).get("name") in INTERNAL_TOOL_NAMES:
        out["tool_choice"] = {"type": "function", "function": {"name": INTERNAL_TOOL_NAME}}
    return out


# ==================== Responses ====================

def responses_internal_tool() -> dict:
    return {
        "type": "function",
        "name": INTERNAL_TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": copy.deepcopy(TOOL_INPUT_SCHEMA),
    }


def is_web_search_tool_responses(tool: Any) -> bool:
    tool = _as_dict(tool)
    if not isinstance(tool, dict):
        return False
    tool_type = tool.get("type")
    if isinstance(tool_type, str) and tool_type.startswith("web_search"):
        return True
    return tool_type == "function" and tool.get("name") in INTERNAL_TOOL_NAMES


def detect_responses(request_dict: dict) -> bool:
    return any(is_web_search_tool_responses(t) for t in request_dict.get("tools") or [])


def rewrite_responses_request(request_dict: dict) -> dict:
    out = dict(request_dict)
    tools = [t for t in (out.get("tools") or []) if not is_web_search_tool_responses(t)]
    tools.append(responses_internal_tool())
    out["tools"] = tools
    tc = out.get("tool_choice")
    if isinstance(tc, dict) and (
        str(tc.get("type", "")).startswith("web_search")
        or (tc.get("type") == "function" and tc.get("name") in INTERNAL_TOOL_NAMES)
    ):
        out["tool_choice"] = {"type": "function", "name": INTERNAL_TOOL_NAME}
    include = out.get("include")
    if isinstance(include, list):
        out["include"] = [i for i in include if not str(i).startswith("web_search_call")] or None
    inp = out.get("input")
    if isinstance(inp, list):
        # Upstream cannot run web search (that is why we intercept), so it
        # would reject replayed web_search_call items.
        out["input"] = [
            item for item in inp
            if not (isinstance(_as_dict(item), dict) and _as_dict(item).get("type") == "web_search_call")
        ]
    return out

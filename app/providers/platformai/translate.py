"""OpenAI chat request -> PlatformAI v3 native chat body; gateway usage -> OpenAI.

Wire shapes (docs, confirmed against the live gateway):

- every turn is ``{"role", "contents": [Chat Content]}``; text is
  ``{"type":"text","text"}``, images ``{"type":"image","image":{"type":"url",
  "url"}}`` or ``{"type":"image","image":{"type":"base64","mime_type",
  "base64": "<data URI>"}}``. No file or audio blocks exist;
- system prompts go in the top-level ``context`` array;
- assistant tool calls ride on ``tool_calls`` (the *response* names them
  ``tools``); tool output is a ``{"role":"tool","tool_results":[{id, name,
  type, content}]}`` turn;
- ``tool_choice`` is ``{"function": {"type": "auto"|"none"|"any", "names"?}}``;
- ``reasoning`` is ``{"effort": minimal|low|medium|high}`` or ``{"enabled": false}``;
- ``response_format`` json_schema carries the raw schema under ``parameters``;
- ``prompt_caching: {"name": "ephemeral", "ttl"}`` (undocumented) turns on
  Anthropic prompt caching; without it Claude models never cache;
- limits: messages/context 1-100, tools 1-128, stop_sequences 1-4.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from typing import AbstractSet, Any, Dict, FrozenSet, List, Optional, Tuple

from app.model_capabilities import SURFACE_NATIVE, rejected_params
from app.openai_models import ChatCompletionRequest, Usage
from app.providers.base import ProviderHTTPError
from app.providers.platformai.schema import sanitize_tool_schema

logger = logging.getLogger(__name__)

MAX_MESSAGES = 100
MAX_CONTEXT_PARTS = 100
MAX_TOOLS = 128
MAX_STOP_SEQUENCES = 4
MAX_VARIATIONS = 5

# Accepted reasoning efforts -> the gateway's ``reasoning.effort`` (None = turn reasoning off).
EFFORT_MAP = {
    "none": None,
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
EFFORT_LADDER = ["minimal", "low", "medium", "high"]

# OpenAI chat params the gateway has no equivalent for; dropped, not rejected.
IGNORED_PARAMS = (
    "logprobs", "top_logprobs", "presence_penalty", "frequency_penalty", "user", "logit_bias",
    "parallel_tool_calls",
)

_IMAGE_SUBTYPES = {"png", "jpg", "jpeg", "gif", "webp"}
_DATA_URI = re.compile(r"^data:([a-z]+/[a-z0-9.+-]+);base64,(.+)$", re.I | re.S)
_WEB_SEARCH_KEYS = ("search_context_size", "allowed_domains", "blocked_domains", "max_uses", "user_location")


# Prompt caching the gateway accepts for Claude models (verified live). Other
# vendors are left alone: their caching names are unverified on v3.
CLAUDE_PROMPT_CACHING = {"name": "ephemeral", "ttl": "1h"}


def prompt_caching_for(model: str) -> Optional[Dict[str, str]]:
    return dict(CLAUDE_PROMPT_CACHING) if model.lower().startswith("claude") else None


# app.model_capabilities uses the OpenAI/Anthropic spellings; these are the gateway's.
_CAPABILITY_SPELLINGS = {"stop": "stop_sequences"}


def model_rejected_params(model: str) -> FrozenSet[str]:
    """Body keys ``model`` rejects per the shared table in app.model_capabilities
    (e.g. temperature/top_p/top_k on Claude >= 4.7), in the gateway's spelling."""
    return frozenset(_CAPABILITY_SPELLINGS.get(p, p) for p in rejected_params(model, SURFACE_NATIVE))


def bad_request(message: str) -> ProviderHTTPError:
    return ProviderHTTPError(
        400, message, body={"error": {"message": message, "type": "invalid_request_error", "code": 400}}
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return {}


def _extra(request: ChatCompletionRequest, name: str) -> Any:
    """A field the OpenAI model does not declare (``extra="allow"``)."""
    extra = getattr(request, "model_extra", None) or {}
    if name in extra:
        return extra[name]
    extra_body = extra.get("extra_body")
    if isinstance(extra_body, dict):
        return extra_body.get(name)
    return None


# ---------------------------------------------------------------------------
# Effort / max_tokens
# ---------------------------------------------------------------------------

_OFF = "none"


def map_effort(value: Optional[str], ceiling: Optional[str] = None) -> Optional[str]:
    """Map a client effort onto the gateway's levels, capped at ``ceiling``.

    Returns a level, ``"none"`` (reasoning explicitly off) or None (unset).
    """
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in EFFORT_MAP:
        raise bad_request(f'"reasoning_effort" must be one of: {", ".join(EFFORT_MAP)}.')
    effort = EFFORT_MAP[key]
    if effort is None:
        return _OFF
    if ceiling is None:
        return effort
    if ceiling == _OFF:
        return None
    if EFFORT_LADDER.index(effort) > EFFORT_LADDER.index(ceiling):
        return ceiling
    return effort


def step_down_effort(effort: str) -> Optional[str]:
    idx = EFFORT_LADDER.index(effort)
    return EFFORT_LADDER[idx - 1] if idx > 0 else None


def resolve_max_tokens(requested: Optional[int], ceiling: Optional[int]) -> Optional[int]:
    if requested is not None:
        if requested <= 0:
            raise bad_request('"max_tokens" must be a positive integer.')
        return min(requested, ceiling) if ceiling else requested
    return ceiling


# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------

def _image_from_data_uri(media_type: str, data_uri: str) -> Dict[str, Any]:
    mime = media_type.lower()
    subtype = mime.split("/", 1)[1] if "/" in mime else ""
    if mime == "application/pdf":
        raise bad_request("PlatformAI chat does not accept PDF or other file attachments; only images.")
    if not mime.startswith("image/") or subtype not in _IMAGE_SUBTYPES:
        raise bad_request(
            f'Unsupported attachment type "{media_type}". PlatformAI accepts '
            f'{", ".join(sorted(_IMAGE_SUBTYPES))} images.'
        )
    return {"type": "image", "image": {"type": "base64", "mime_type": mime, "base64": data_uri}}


def _image_part(url: Optional[str]) -> Dict[str, Any]:
    if not url:
        raise bad_request("An attached image carried no URL or data.")
    match = _DATA_URI.match(url)
    if match:
        return _image_from_data_uri(match.group(1), url)
    if not re.match(r"^https?://", url, re.I):
        raise bad_request("Image URLs must be http(s) or a base64 data URI.")
    image: Dict[str, Any] = {"type": "url", "url": url}
    guessed, _ = mimetypes.guess_type(url.split("?", 1)[0])
    if guessed and guessed.startswith("image/"):
        image["mime_type"] = guessed
    return {"type": "image", "image": image}


def _file_part(file: Dict[str, Any]) -> Dict[str, Any]:
    data = file.get("file_data") or ""
    match = _DATA_URI.match(data)
    if match and match.group(1).lower().startswith("image/"):
        return _image_from_data_uri(match.group(1), data)
    raise bad_request("PlatformAI chat does not accept PDF or other file attachments; only images.")


def _text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _parts(message: Dict[str, Any], *, allow_media: bool) -> List[Dict[str, Any]]:
    """``message.content`` as Chat Content parts.

    With ``allow_media`` False (system/assistant) non-text parts are dropped.
    """
    content = message.get("content")
    if content is None:
        return []
    if isinstance(content, str):
        return [_text(content)]
    if not isinstance(content, list):
        raise bad_request(f'Unsupported "content" type on a {message.get("role")} message.')
    parts: List[Dict[str, Any]] = []
    for raw in content:
        part = _as_dict(raw)
        ptype = part.get("type")
        if ptype in ("text", "tool", "input_text", "output_text"):
            parts.append(_text(part.get("text") or ""))
        elif ptype in ("image_url", "input_image", "file", "input_file"):
            if not allow_media:
                logger.debug("PlatformAI: dropping %s part on a %s message", ptype, message.get("role"))
                continue
            if ptype in ("file", "input_file"):
                parts.append(_file_part(part.get("file") or part))
            else:
                image_url = part.get("image_url")
                parts.append(_image_part(image_url.get("url") if isinstance(image_url, dict) else image_url))
        elif ptype == "input_audio":
            raise bad_request('Content part type "input_audio" is not supported by PlatformAI chat.')
        else:
            raise bad_request(f'Unknown content part type "{ptype}".')
    return parts


def _joined_text(parts: List[Dict[str, Any]]) -> str:
    return "".join(p.get("text") or "" for p in parts if p.get("type") == "text")


def _non_empty(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in parts if p.get("type") != "text" or p.get("text")]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _tools(tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """Function tools in gateway shape. Other tool types are dropped: the
    gateway runs only function tools (web search is ``web_search_options``)."""
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for i, raw in enumerate(tools):
        tool = _as_dict(raw)
        ttype = tool.get("type", "function")
        if ttype != "function":
            dropped.append(f"tools[{i}]:{ttype}")
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            raise bad_request(f'tools[{i}] is missing "function.name".')
        entry: Dict[str, Any] = {"name": name}
        if fn.get("description"):
            entry["description"] = fn["description"]
        entry["parameters"] = sanitize_tool_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        out.append({"type": "function", "function": entry})
    if dropped:
        logger.info("PlatformAI: dropped tools the gateway cannot run: %s", ", ".join(dropped))
    if len(out) > MAX_TOOLS:
        raise bad_request(f"PlatformAI accepts at most {MAX_TOOLS} tools; got {len(out)}.")
    return out or None


def _tool_choice(choice: Any, names: List[str]) -> Optional[Dict[str, Any]]:
    if choice is None:
        return None

    def any_of(selected: List[str]) -> Dict[str, Any]:
        if not selected:
            raise bad_request('"tool_choice" requires a tool to choose from, but no "tools" were provided.')
        return {"function": {"type": "any", "names": selected}}

    if choice == "auto":
        return {"function": {"type": "auto"}} if names else None
    if choice == "none":
        return {"function": {"type": "none"}} if names else None
    if choice == "required":
        return any_of(names)
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "auto":
            return {"function": {"type": "auto"}} if names else None
        if ctype == "none":
            return {"function": {"type": "none"}} if names else None
        if ctype in ("any", "required"):
            return any_of(names)
        if ctype in ("function", "tool"):
            name = (choice.get("function") or {}).get("name") or choice.get("name")
            if not name:
                raise bad_request('tool_choice.function.name is required when type is "function".')
            if name not in names:
                raise bad_request(f'tool_choice names "{name}", which is not among the provided tools.')
            return any_of([name])
    raise bad_request(f'Unsupported "tool_choice" value: {json.dumps(choice, default=str)}')


def _web_search_options(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    out = {k: raw[k] for k in _WEB_SEARCH_KEYS if raw.get(k) is not None}
    location = out.get("user_location")
    if isinstance(location, dict):
        # OpenAI wraps it as {"type": "approximate", "approximate": {...}}.
        inner = location.get("approximate") if isinstance(location.get("approximate"), dict) else location
        cleaned = {k: v for k, v in inner.items() if k != "type" and v is not None}
        for coord in ("latitude", "longitude"):
            if isinstance(cleaned.get(coord), (int, float)) and not isinstance(cleaned[coord], bool):
                cleaned[coord] = int(round(cleaned[coord]))  # gateway takes whole numbers only
        if cleaned:
            out["user_location"] = cleaned
        else:
            out.pop("user_location")
    return out


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

def _response_format(response_format: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(native response_format, fallback instruction)`` for a request."""
    rf = _as_dict(response_format)
    rtype = rf.get("type")
    if rtype == "json_object":
        return {"type": "json_object"}, (
            "Respond with a single valid JSON object and nothing else. "
            "Do not wrap it in markdown code fences and do not add commentary."
        )
    if rtype != "json_schema":
        return None, None
    spec = rf.get("json_schema") or {}
    # OpenAI wraps the schema as {name, schema, strict}; the gateway wants the raw schema.
    schema = spec.get("schema") if isinstance(spec.get("schema"), dict) else spec.get("parameters")
    if not isinstance(schema, dict):
        return {"type": "json_object"}, None
    try:
        rendered = json.dumps(schema, indent=2)
    except (TypeError, ValueError):
        rendered = str(schema)
    name = spec.get("name") if isinstance(spec.get("name"), str) else None
    named = f' (named "{name}")' if name else ""
    instruction = (f"Respond with a single valid JSON value conforming to this JSON Schema{named}:\n"
                   f"{rendered}\n\nOutput only the JSON. Do not wrap it in markdown code fences and do not add commentary.")
    return {"type": "json_schema", "json_schema": {"parameters": schema}}, instruction


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------

def build_chat_body(
    request: ChatCompletionRequest,
    *,
    upstream_model: str,
    max_tokens_ceiling: Optional[int] = None,
    effort_ceiling: Optional[str] = None,
    unsupported: AbstractSet[str] = frozenset(),
    stream: bool,
) -> Dict[str, Any]:
    """Translate an OpenAI chat request into the gateway's native chat body.

    ``unsupported`` names top-level body keys the model has refused before;
    they are left out (``response_format`` becomes a prompt instruction), as
    are the keys ``app.model_capabilities`` says the model rejects.
    """
    unsupported = frozenset(unsupported) | model_rejected_params(upstream_model)
    n = getattr(request, "n", None) or _extra(request, "n")
    variations = int(n) if isinstance(n, int) and not isinstance(n, bool) and n > 1 else None
    if variations is not None:
        if variations > MAX_VARIATIONS:
            raise bad_request(f'"n" must be at most {MAX_VARIATIONS} for PlatformAI.')
        if stream:
            raise bad_request('"n" greater than 1 is not supported with streaming on PlatformAI.')
        if "variations" in unsupported:
            raise bad_request(f'"n" greater than 1 is not supported by the PlatformAI model "{upstream_model}".')
    if not request.messages:
        raise bad_request('"messages" must be a non-empty array.')

    context: List[Dict[str, Any]] = []
    turns: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    tool_attachments: List[Tuple[str, List[Dict[str, Any]]]] = []
    call_names: Dict[str, str] = {}

    def flush_tools() -> None:
        if tool_results:
            turns.append({"role": "tool", "tool_results": list(tool_results)})
            tool_results.clear()
        for call_id, parts in tool_attachments:
            turns.append({"role": "user", "contents": [_text(f"Image returned by tool call {call_id}:"), *parts]})
        tool_attachments.clear()

    for raw in request.messages:
        message = _as_dict(raw)
        role = message.get("role")
        if role in ("system", "developer"):
            flush_tools()
            context.extend(_non_empty(_parts(message, allow_media=False)))
            continue
        if role in ("tool", "function"):
            call_id = message.get("tool_call_id")
            if not call_id:
                raise bad_request('A "tool" message must carry a "tool_call_id".')
            parts = _parts(message, allow_media=True)
            result: Dict[str, Any] = {"id": call_id, "type": "function", "content": _joined_text(parts)}
            name = call_names.get(call_id) or message.get("name")
            if name:
                result["name"] = name
            tool_results.append(result)
            media = [p for p in parts if p.get("type") != "text"]
            if media:
                tool_attachments.append((call_id, media))
            continue
        flush_tools()
        if role == "assistant":
            calls = []
            for i, raw_call in enumerate(message.get("tool_calls") or []):
                call = _as_dict(raw_call)
                fn = call.get("function") or {}
                if not call.get("id") or not fn.get("name"):
                    raise bad_request(f'assistant tool_calls[{i}] must have both "id" and "function.name".')
                call_names[call["id"]] = fn["name"]
                calls.append({"id": call["id"], "type": "function",
                              "function": {"name": fn["name"], "arguments": fn.get("arguments") or "{}"}})
            text = _joined_text(_parts(message, allow_media=False))
            if not text and not calls:
                continue
            turn: Dict[str, Any] = {"role": "assistant"}
            if text:
                turn["contents"] = [_text(text)]
            if calls:
                turn["tool_calls"] = calls
            turns.append(turn)
            continue
        if role == "user":
            parts = _non_empty(_parts(message, allow_media=True)) or [_text("")]
            turns.append({"role": "user", "contents": parts})
            continue
        raise bad_request(f'Unsupported message role "{role}".')
    flush_tools()

    if not turns:
        raise bad_request('"messages" contained no forwardable turns — a system prompt alone is not a request.')
    if len(turns) > MAX_MESSAGES:
        raise bad_request(
            f"PlatformAI accepts at most {MAX_MESSAGES} messages per request; this conversation has {len(turns)} "
            "after merging consecutive tool results. Shorten or compact the conversation."
        )

    response_format, format_instruction = _response_format(request.response_format)
    if response_format is not None and "response_format" in unsupported:
        response_format = None
        if format_instruction:
            context.append(_text(format_instruction))
    if len(context) > MAX_CONTEXT_PARTS:
        raise bad_request(f"PlatformAI accepts at most {MAX_CONTEXT_PARTS} system/context parts.")

    tools = _tools(request.tools)
    choice = _tool_choice(request.tool_choice, [t["function"]["name"] for t in tools or []])
    web_search_options = _web_search_options(_extra(request, "web_search_options"))

    body: Dict[str, Any] = {"model": upstream_model}
    if context:
        body["context"] = context
    body["messages"] = turns
    body["stream"] = bool(stream)

    requested_max = request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens
    optional: Dict[str, Any] = {
        "max_tokens": resolve_max_tokens(requested_max, max_tokens_ceiling),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": _extra(request, "top_k"),
        "seed": request.seed,
        "stop_sequences": _stop_sequences(request.stop),
        "variations": variations,
        "tools": tools,
        "tool_choice": choice,
        "response_format": response_format,
        "web_search_options": web_search_options,
        "prompt_caching": prompt_caching_for(upstream_model),
    }
    effort = map_effort(request.reasoning_effort, effort_ceiling)
    if effort == _OFF:
        optional["reasoning"] = {"enabled": False}
    elif effort:
        optional["reasoning"] = {"effort": effort}

    skipped = []
    for key, value in optional.items():
        if value is None:
            continue
        if key in unsupported:
            skipped.append(key)
            continue
        body[key] = value
    if "tools" not in body:
        body.pop("tool_choice", None)
    if skipped:
        logger.debug("PlatformAI: %s does not support %s; left out", upstream_model, ", ".join(skipped))

    ignored = [p for p in IGNORED_PARAMS if getattr(request, p, None) is not None]
    if ignored:
        logger.debug("PlatformAI: ignoring unsupported params: %s", ", ".join(ignored))
    return body


def _stop_sequences(stop: Any) -> Optional[List[str]]:
    if stop is None:
        return None
    values = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]
    values = [s for s in values if s]
    if not values:
        return None
    if len(values) > MAX_STOP_SEQUENCES:
        raise bad_request(f'PlatformAI accepts at most {MAX_STOP_SEQUENCES} "stop" sequences.')
    return values


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def normalize_finish_reason(reason: Optional[str], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    key = re.sub(r"[_-]", "", str(reason or "")).lower()
    if key in ("maxtokens", "length"):
        return "length"
    if key in ("toolcalls", "tooluse"):
        return "tool_calls"
    if key in ("contentfilter", "guardrail", "refusal", "safety"):
        return "content_filter"
    return "stop"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_usage(raw: Optional[Dict[str, Any]]) -> Usage:
    """OpenAI usage from the gateway's ``usage`` object.

    The gateway's ``prompt_tokens`` (and ``total_tokens``) exclude cache reads
    (``cached_tokens``) and cache writes (``cache_write_tokens``), Anthropic
    style. OpenAI's ``prompt_tokens`` includes them, so they are added back,
    with the read share in ``prompt_tokens_details.cached_tokens``.
    """
    raw = raw if isinstance(raw, dict) else {}
    cache_read = _int(raw.get("cached_tokens"))
    cache_write = _int(raw.get("cache_write_tokens")) or (
        _int(raw.get("cache_write_5m_tokens")) + _int(raw.get("cache_write_1h_tokens"))
    )
    prompt = _int(raw.get("prompt_tokens")) + cache_read + cache_write
    completion = _int(raw.get("completion_tokens"))
    usage = Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
    usage.prompt_tokens_details = {"cached_tokens": cache_read}
    reasoning = _int(raw.get("reasoning_token"))
    if reasoning:
        usage.completion_tokens_details = {"reasoning_tokens": reasoning}
    if cache_read:
        usage.cache_read_input_tokens = cache_read
    if cache_write:
        usage.cache_creation_input_tokens = cache_write
    return usage


def merge_usage(current: Optional[Dict[str, Any]], update: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fold a streamed usage frame in, keeping the largest value per counter.

    Streamed frames report usage repeatedly and the final frame can carry a
    stale (smaller) completion count, so "last wins" would under-report.
    """
    if not isinstance(update, dict):
        return current
    merged = dict(current or {})
    for key, value in update.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            prior = merged.get(key)
            merged[key] = max(value, prior) if isinstance(prior, (int, float)) else value
        elif value is not None:
            merged[key] = value
    return merged

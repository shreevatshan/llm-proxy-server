"""Server-side web search loops for the three API surfaces.

Each loop takes a request whose web search tool was already rewritten to the
internal function tool (see app.websearch.tools), calls the upstream model,
runs any internal tool calls against the configured search backend, feeds the results back and
repeats until the model answers or the round limit is reached. The client
sees one request and one response.

Modelled on LiteLLM's websearch_interception handler (agentic loop with a
max-loops limit and a duplicate-query fingerprint check).
"""

import asyncio
import copy
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Iterator, List, Optional, Tuple

from app.tracing import add_span_attributes, create_span
from app.websearch.backends import run_searches
from app.websearch.results import SearchFailed, SearchOutcome, format_outcome_text
from app.websearch.settings import WebSearchConfig
from app.websearch.tools import (
    INTERNAL_TOOL_NAME,
    INTERNAL_TOOL_NAMES,
    build_native_blocks,
    new_responses_call_id,
    new_server_tool_id,
    query_from_input,
)

logger = logging.getLogger(__name__)

LIMIT_NUDGE = (
    "\n\n[Search limit reached. Answer now using the results above; "
    f"do not call {INTERNAL_TOOL_NAME} again.]"
)
LIMIT_REACHED_TEXT = "(web search limit reached)"
HEADER_NAME = "x-llmproxy-websearch"


@dataclass
class LoopStats:
    rounds: int = 0
    queries: int = 0
    failures: int = 0

    def header_value(self) -> str:
        return f"rounds={self.rounds};queries={self.queries}"


def add_usage(total: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> None:
    """Sum numeric usage fields (recursively for nested detail dicts)."""
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = (total.get(key) or 0) + value
        elif isinstance(value, dict):
            sub = total.get(key)
            if not isinstance(sub, dict):
                sub = {}
                total[key] = sub
            add_usage(sub, value)
        elif key not in total and value is not None:
            total[key] = value


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(obj or {})


class _QueryGuard:
    """Round limit + duplicate-query fingerprint check shared by all loops."""

    def __init__(self, cfg: WebSearchConfig, stats: LoopStats):
        self.cfg = cfg
        self.stats = stats
        self._fingerprints = set()

    def allowed(self, queries: List[str]) -> bool:
        if self.stats.rounds >= self.cfg.max_agentic_loops:
            return False
        fingerprint = tuple(sorted(q.lower() for q in queries))
        if fingerprint in self._fingerprints:
            return False
        self._fingerprints.add(fingerprint)
        return True

    @property
    def is_last_round(self) -> bool:
        """True when the round about to run is the final permitted one."""
        return self.stats.rounds + 1 >= self.cfg.max_agentic_loops

    async def search(self, queries: List[str]) -> List[SearchOutcome]:
        limit = self.cfg.max_queries_per_turn
        with create_span("websearch.round") as span:
            started = time.monotonic()
            outcomes = await run_searches(queries[:limit], self.cfg)
            outcomes += [
                SearchFailed(q, "too_many_requests", f"At most {limit} searches are run per turn")
                for q in queries[limit:]
            ]
            self.stats.rounds += 1
            self.stats.queries += len(queries)
            self.stats.failures += sum(1 for o in outcomes if not o.ok)
            add_span_attributes(span, {
                "websearch.round": self.stats.rounds,
                "websearch.queries": len(queries),
                "websearch.failures": sum(1 for o in outcomes if not o.ok),
                "websearch.latency_ms": int((time.monotonic() - started) * 1000),
            })
        return outcomes

    def result_text(self, outcome: SearchOutcome, last_round: bool) -> str:
        text = format_outcome_text(outcome)
        return text + LIMIT_NUDGE if last_round else text


# ==================== Anthropic Messages ====================

def _is_internal_tool_use(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "tool_use"
        and block.get("name") in INTERNAL_TOOL_NAMES
    )


def _relax_anthropic_tool_choice(tool_choice: Any, has_client_tools: bool) -> Any:
    """After a search round, stop forcing the search tool (it would loop)."""
    tc = _to_dict(tool_choice) if tool_choice is not None else None
    if not isinstance(tc, dict):
        return tool_choice
    forced_search = tc.get("type") == "tool" and tc.get("name") in INTERNAL_TOOL_NAMES
    forced_any = tc.get("type") == "any" and not has_client_tools
    if forced_search or forced_any:
        relaxed = {"type": "auto"}
        if tc.get("disable_parallel_tool_use") is not None:
            relaxed["disable_parallel_tool_use"] = tc["disable_parallel_tool_use"]
        return relaxed
    return tool_choice


def iter_sse_events(chunk: str) -> Iterator[Tuple[Optional[str], Optional[dict], str]]:
    """Split a chunk into (event_type, json_payload, raw_event) triples."""
    for raw in chunk.split("\n\n"):
        if not raw.strip():
            continue
        event_type = None
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        payload = None
        if data_lines:
            try:
                payload = json.loads("\n".join(data_lines))
            except ValueError:
                payload = None
        yield event_type, payload, raw + "\n\n"


def _anthropic_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


@dataclass
class _StreamBlock:
    block: Dict[str, Any]
    internal: bool
    out_index: Optional[int]
    json_buf: str = ""

    def finish(self) -> Dict[str, Any]:
        if self.block.get("type") in ("tool_use", "server_tool_use"):
            if self.json_buf:
                try:
                    self.block["input"] = json.loads(self.json_buf)
                except ValueError:
                    self.block["input"] = {}
            elif not isinstance(self.block.get("input"), dict):
                self.block["input"] = {}
        return self.block


def _apply_delta(block: Dict[str, Any], state: _StreamBlock, delta: Dict[str, Any]) -> None:
    dtype = delta.get("type")
    if dtype == "text_delta":
        block["text"] = block.get("text", "") + delta.get("text", "")
    elif dtype == "thinking_delta":
        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
    elif dtype == "signature_delta":
        block["signature"] = delta.get("signature", "")
    elif dtype == "input_json_delta":
        state.json_buf += delta.get("partial_json", "")
    elif dtype == "citations_delta":
        block.setdefault("citations", []).append(delta.get("citation"))


class AnthropicWebSearchLoop:
    def __init__(
        self,
        request,
        provider,
        cfg: WebSearchConfig,
        emit_native: bool,
        anthropic_beta: Optional[str] = None,
    ):
        self.request = request
        self.provider = provider
        self.cfg = cfg
        self.emit_native = emit_native
        self.anthropic_beta = anthropic_beta
        self.stats = LoopStats()
        self._guard = _QueryGuard(cfg, self.stats)
        self._has_client_tools = any(
            not (isinstance(_to_dict(t), dict) and _to_dict(t).get("name") in INTERNAL_TOOL_NAMES)
            for t in (request.tools or [])
        )

    # -- request building --

    def _initial_messages(self) -> List[dict]:
        return [_to_dict(m) for m in self.request.messages]

    def _round_request(self, messages: List[dict], round_index: int):
        from app.anthropic_models import AnthropicMessage

        update: Dict[str, Any] = {
            "messages": [AnthropicMessage.model_validate(m) for m in messages],
        }
        if round_index > 0:
            update["tool_choice"] = _relax_anthropic_tool_choice(
                self.request.tool_choice, self._has_client_tools
            )
        return self.request.model_copy(update=update)

    def _tool_results(self, calls: List[dict], outcomes: List[SearchOutcome], last: bool) -> List[dict]:
        return [
            {
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": self._guard.result_text(outcome, last),
                **({"is_error": True} if not outcome.ok else {}),
            }
            for call, outcome in zip(calls, outcomes)
        ]

    def _final_usage(self, usage_total: Dict[str, Any]) -> Dict[str, Any]:
        usage = dict(usage_total)
        usage.setdefault("input_tokens", 0)
        usage.setdefault("output_tokens", 0)
        if self.emit_native and self.stats.queries:
            server = dict(usage.get("server_tool_use") or {})
            server["web_search_requests"] = self.stats.queries
            usage["server_tool_use"] = server
        return usage

    # -- non-streaming --

    async def run(self) -> Dict[str, Any]:
        messages = self._initial_messages()
        usage_total: Dict[str, Any] = {}
        emitted: List[dict] = []  # visible blocks of earlier rounds
        round_index = 0
        while True:
            response = _to_dict(
                await self.provider.anthropic_messages(
                    self._round_request(messages, round_index), anthropic_beta=self.anthropic_beta
                )
            )
            round_index += 1
            add_usage(usage_total, response.get("usage"))
            content = list(response.get("content") or [])
            internal = [b for b in content if _is_internal_tool_use(b)]
            visible = [b for b in content if not _is_internal_tool_use(b)]
            stop_reason = response.get("stop_reason")

            if not internal or stop_reason != "tool_use":
                return self._finish(response, emitted + visible, stop_reason, usage_total)
            if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in visible):
                # Mixed turn: hand the client its own tool calls; the search
                # request is dropped (the model re-asks next turn).
                return self._finish(response, emitted + visible, "tool_use", usage_total)

            queries = [query_from_input(b.get("input")) for b in internal]
            if not self._guard.allowed(queries):
                blocks = emitted + visible
                if not any(b.get("type") == "text" for b in blocks if isinstance(b, dict)):
                    blocks.append({"type": "text", "text": LIMIT_REACHED_TEXT})
                return self._finish(response, blocks, "end_turn", usage_total)

            last = self._guard.is_last_round
            outcomes = await self._guard.search(queries)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": self._tool_results(internal, outcomes, last)})

            if self.emit_native:
                emitted.extend(visible)
                for outcome in outcomes:
                    emitted.extend(build_native_blocks(new_server_tool_id(), outcome))
            else:
                # No native blocks to carry the search, so earlier-round
                # thinking would be orphaned; keep only the text.
                emitted.extend(b for b in visible if b.get("type") == "text")

    def _finish(self, response: Dict[str, Any], blocks: List[dict], stop_reason, usage_total) -> Dict[str, Any]:
        final = dict(response)
        final["content"] = blocks
        final["stop_reason"] = stop_reason
        final["usage"] = self._final_usage(usage_total)
        return final

    # -- streaming --

    async def stream(self) -> AsyncGenerator[str, None]:
        messages = self._initial_messages()
        usage_total: Dict[str, Any] = {}
        started = False
        next_out_index = 0
        visible_text_seen = False
        round_index = 0
        while True:
            states: Dict[int, _StreamBlock] = {}
            order: List[int] = []
            round_usage: Dict[str, Any] = {}
            stop_reason = None
            stop_sequence = None

            upstream = self.provider.anthropic_messages_stream(
                self._round_request(messages, round_index), anthropic_beta=self.anthropic_beta
            )
            round_index += 1
            async for chunk in upstream:
                for event_type, payload, raw in iter_sse_events(chunk):
                    etype = (payload or {}).get("type") or event_type
                    if not isinstance(payload, dict):
                        yield raw
                        continue
                    if etype == "message_start":
                        message = payload.get("message") or {}
                        round_usage.update(message.get("usage") or {})
                        if not started:
                            started = True
                            yield raw
                    elif etype == "content_block_start":
                        idx = payload.get("index", 0)
                        block = copy.deepcopy(payload.get("content_block") or {})
                        internal = _is_internal_tool_use(block)
                        state = _StreamBlock(block, internal, None if internal else next_out_index)
                        states[idx] = state
                        order.append(idx)
                        if not internal:
                            next_out_index += 1
                            if block.get("type") == "text":
                                visible_text_seen = True
                            yield _anthropic_sse("content_block_start", {**payload, "index": state.out_index})
                    elif etype == "content_block_delta":
                        state = states.get(payload.get("index", 0))
                        if state is None:
                            continue
                        _apply_delta(state.block, state, payload.get("delta") or {})
                        if not state.internal:
                            yield _anthropic_sse("content_block_delta", {**payload, "index": state.out_index})
                    elif etype == "content_block_stop":
                        state = states.get(payload.get("index", 0))
                        if state is None:
                            continue
                        state.finish()
                        if not state.internal:
                            yield _anthropic_sse("content_block_stop", {**payload, "index": state.out_index})
                    elif etype == "message_delta":
                        delta = payload.get("delta") or {}
                        stop_reason = delta.get("stop_reason", stop_reason)
                        stop_sequence = delta.get("stop_sequence", stop_sequence)
                        round_usage.update(payload.get("usage") or {})
                    elif etype == "message_stop":
                        pass
                    elif etype == "error":
                        yield raw
                        return
                    else:
                        yield raw  # ping and unknown events

            add_usage(usage_total, round_usage)
            blocks = [states[i].finish() for i in order]
            internal = [b for b in blocks if _is_internal_tool_use(b)]
            client_tool = any(
                b.get("type") == "tool_use" and not _is_internal_tool_use(b) for b in blocks
            )

            if not started:
                return  # upstream produced nothing; the route reports it
            if not internal or stop_reason != "tool_use":
                for event in self._stream_finish(stop_reason, stop_sequence, usage_total):
                    yield event
                return
            if client_tool:
                for event in self._stream_finish("tool_use", None, usage_total):
                    yield event
                return

            queries = [query_from_input(b.get("input")) for b in internal]
            if not self._guard.allowed(queries):
                if not visible_text_seen:
                    for event in self._text_block_events(next_out_index, LIMIT_REACHED_TEXT):
                        yield event
                    next_out_index += 1
                for event in self._stream_finish("end_turn", None, usage_total):
                    yield event
                return

            last = self._guard.is_last_round
            outcomes = await self._guard.search(queries)
            messages.append({"role": "assistant", "content": blocks})
            messages.append({"role": "user", "content": self._tool_results(internal, outcomes, last)})

            if self.emit_native:
                for call, outcome in zip(internal, outcomes):
                    server_use, result = build_native_blocks(new_server_tool_id(), outcome)
                    yield _anthropic_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": next_out_index,
                        "content_block": {**server_use, "input": {}},
                    })
                    yield _anthropic_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": next_out_index,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(server_use["input"])},
                    })
                    yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": next_out_index})
                    next_out_index += 1
                    yield _anthropic_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": next_out_index,
                        "content_block": result,
                    })
                    yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": next_out_index})
                    next_out_index += 1

    def _text_block_events(self, index: int, text: str) -> List[str]:
        return [
            _anthropic_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""},
            }),
            _anthropic_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": text},
            }),
            _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]

    def _stream_finish(self, stop_reason, stop_sequence, usage_total) -> List[str]:
        return [
            _anthropic_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": self._final_usage(usage_total),
            }),
            _anthropic_sse("message_stop", {"type": "message_stop"}),
        ]


# ==================== Chat Completions ====================

def _relax_chat_tool_choice(tool_choice: Any, has_client_tools: bool) -> Any:
    if isinstance(tool_choice, dict) and (tool_choice.get("function") or {}).get("name") in INTERNAL_TOOL_NAMES:
        return "auto"
    if tool_choice == "required" and not has_client_tools:
        return "auto"
    return tool_choice


def _chat_internal_calls(tool_calls: List[dict]) -> Tuple[List[dict], List[dict]]:
    internal, client = [], []
    for tc in tool_calls or []:
        name = (tc.get("function") or {}).get("name")
        (internal if name in INTERNAL_TOOL_NAMES else client).append(tc)
    return internal, client


def _chat_history_turn(message: Dict[str, Any], internal: List[dict], outcomes, guard, last) -> List[dict]:
    assistant: Dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments") or "{}",
                },
            }
            for tc in internal
        ],
    }
    if message.get("reasoning_content"):
        assistant["reasoning_content"] = message["reasoning_content"]
    tool_messages = [
        {"role": "tool", "tool_call_id": tc["id"], "content": guard.result_text(outcome, last)}
        for tc, outcome in zip(internal, outcomes)
    ]
    return [assistant, *tool_messages]


class ChatWebSearchLoop:
    """``call`` / ``call_stream`` take a ChatCompletionRequest."""

    def __init__(
        self,
        request_dict: Dict[str, Any],
        cfg: WebSearchConfig,
        call: Callable[[Any], Awaitable[Any]],
        call_stream: Callable[[Any], AsyncGenerator[str, None]],
    ):
        self.request_dict = request_dict
        self.cfg = cfg
        self.call = call
        self.call_stream = call_stream
        self.stats = LoopStats()
        self._guard = _QueryGuard(cfg, self.stats)
        self._has_client_tools = any(
            ((t.get("function") or {}).get("name") not in INTERNAL_TOOL_NAMES)
            for t in request_dict.get("tools") or []
        )

    def _round_request(self, messages: List[dict], round_index: int, stream: bool):
        from app.openai_models import ChatCompletionRequest

        body = {**self.request_dict, "messages": messages, "stream": stream}
        if round_index > 0 and "tool_choice" in body:
            body["tool_choice"] = _relax_chat_tool_choice(body["tool_choice"], self._has_client_tools)
        return ChatCompletionRequest.model_validate(body)

    async def run(self):
        from app.openai_models import ChatCompletionResponse

        messages = list(self.request_dict.get("messages") or [])
        usage_total: Dict[str, Any] = {}
        content_prefix = ""
        round_index = 0
        while True:
            response = await self.call(self._round_request(messages, round_index, stream=False))
            round_index += 1
            data = response.model_dump(exclude_unset=True) if hasattr(response, "model_dump") else dict(response)
            add_usage(usage_total, data.get("usage"))
            choices = data.get("choices") or []
            if not choices:
                return self._finish_chat(data, usage_total, ChatCompletionResponse)
            choice = choices[0]
            message = choice.get("message") or {}
            internal, client = _chat_internal_calls(message.get("tool_calls") or [])
            finish = choice.get("finish_reason")

            def finalize(finish_reason, fallback_text=None):
                text = (content_prefix + (message.get("content") or "")) or fallback_text
                message["content"] = text
                if client:
                    message["tool_calls"] = client
                else:
                    message.pop("tool_calls", None)
                choice["message"] = message
                choice["finish_reason"] = finish_reason
                return self._finish_chat(data, usage_total, ChatCompletionResponse)

            if not internal or finish == "length":
                return finalize(finish if not internal else "length")
            if client:
                return finalize("tool_calls")

            queries = [query_from_input((tc.get("function") or {}).get("arguments")) for tc in internal]
            if not self._guard.allowed(queries):
                return finalize("stop", LIMIT_REACHED_TEXT)

            last = self._guard.is_last_round
            outcomes = await self._guard.search(queries)
            messages.extend(_chat_history_turn(message, internal, outcomes, self._guard, last))
            if message.get("content"):
                content_prefix += message["content"]

    def _finish_chat(self, data, usage_total, response_cls):
        if usage_total:
            data["usage"] = usage_total
        return response_cls(**data)

    async def stream(self) -> AsyncGenerator[str, None]:
        messages = list(self.request_dict.get("messages") or [])
        usage_total: Dict[str, Any] = {}
        meta: Optional[Dict[str, Any]] = None
        next_client_index = 0
        round_index = 0
        while True:
            calls: Dict[int, Dict[str, Any]] = {}
            client_index_map: Dict[int, int] = {}
            finish = None
            content = ""
            reasoning = ""
            async for chunk in self.call_stream(self._round_request(messages, round_index, stream=True)):
                for line in chunk.split("\n"):
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if "error" in obj and "choices" not in obj:
                        yield f"data: {data_str}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    if meta is None:
                        meta = {k: obj.get(k) for k in ("id", "created", "model", "system_fingerprint") if obj.get(k) is not None}
                    if obj.get("usage"):
                        add_usage(usage_total, obj["usage"])
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = dict(choice.get("delta") or {})
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or ""

                    forwarded_calls = []
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        state = calls.get(idx)
                        fn = tc.get("function") or {}
                        if state is None:
                            state = {"id": tc.get("id"), "name": fn.get("name"), "arguments": ""}
                            calls[idx] = state
                            if state["name"] not in INTERNAL_TOOL_NAMES:
                                client_index_map[idx] = next_client_index
                                next_client_index += 1
                        if tc.get("id") and not state.get("id"):
                            state["id"] = tc["id"]
                        if fn.get("name") and not state.get("name"):
                            state["name"] = fn["name"]
                        state["arguments"] += fn.get("arguments") or ""
                        if idx in client_index_map:
                            forwarded_calls.append({**tc, "index": client_index_map[idx]})
                    if "tool_calls" in delta:
                        if forwarded_calls:
                            delta["tool_calls"] = forwarded_calls
                        else:
                            delta.pop("tool_calls")

                    has_payload = any(
                        v not in (None, "", []) for k, v in delta.items() if k != "role"
                    ) or (round_index == 0 and delta.get("role"))
                    if not has_payload:
                        continue
                    out = {**obj, **(meta or {}), "choices": [{**choice, "delta": delta, "finish_reason": None}]}
                    out.pop("usage", None)
                    yield f"data: {json.dumps(out, ensure_ascii=False, separators=(',', ':'))}\n\n"

            round_index += 1
            internal = [
                {"id": s["id"] or f"call_{i}", "type": "function", "function": {"name": s["name"], "arguments": s["arguments"]}}
                for i, s in sorted(calls.items())
                if s["name"] in INTERNAL_TOOL_NAMES
            ]
            has_client = bool(client_index_map)

            if not internal or finish == "length":
                yield self._final_chunk(meta, finish or "stop")
            elif has_client:
                yield self._final_chunk(meta, "tool_calls")
            else:
                queries = [query_from_input(tc["function"]["arguments"]) for tc in internal]
                if self._guard.allowed(queries):
                    last = self._guard.is_last_round
                    outcomes = await self._guard.search(queries)
                    message = {"content": content}
                    if reasoning:
                        message["reasoning_content"] = reasoning
                    messages.extend(_chat_history_turn(message, internal, outcomes, self._guard, last))
                    continue
                if not content:
                    yield self._chunk(meta, {"content": LIMIT_REACHED_TEXT}, None)
                yield self._final_chunk(meta, "stop")

            if usage_total:
                usage_chunk = {**self._base(meta), "choices": [], "usage": usage_total}
                yield f"data: {json.dumps(usage_chunk, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
            return

    @staticmethod
    def _base(meta) -> Dict[str, Any]:
        return {"object": "chat.completion.chunk", **(meta or {"id": "chatcmpl-websearch", "created": int(time.time()), "model": ""})}

    def _chunk(self, meta, delta, finish_reason) -> str:
        chunk = {**self._base(meta), "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}\n\n"

    def _final_chunk(self, meta, finish_reason) -> str:
        return self._chunk(meta, {}, finish_reason)


# ==================== Responses ====================

def _responses_text(output: List[dict]) -> str:
    parts = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    parts.append(part.get("text") or "")
    return "".join(parts)


def _is_internal_function_call(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") in INTERNAL_TOOL_NAMES
    )


def _input_as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    return list(value)


# Internal function_call ids left unanswered in a stored upstream response,
# keyed by that response's id. Some exits hand the response back while an
# internal call is still pending: a mixed turn (search + client tool) or a hit
# round/duplicate limit. The client's next turn chains onto it with
# previous_response_id, and the upstream rejects that turn unless every call in
# it gets an output, so the next turn injects a "not run" output for each.
# Process-local, like the rate limiter's buckets (single-worker deployment).
_PENDING_TTL_SECONDS = 24 * 3600
_PENDING_MAX_ENTRIES = 10000
_pending_internal_calls: "OrderedDict[str, Tuple[float, List[str]]]" = OrderedDict()
SKIPPED_SEARCH_OUTPUT = "[Web search was not run for this call.]"


def _remember_pending_calls(response_id: str, call_ids: List[str]) -> None:
    now = time.monotonic()
    _pending_internal_calls[response_id] = (now + _PENDING_TTL_SECONDS, call_ids)
    _pending_internal_calls.move_to_end(response_id)
    while _pending_internal_calls:
        oldest_id, (expires, _) = next(iter(_pending_internal_calls.items()))
        if expires > now and len(_pending_internal_calls) <= _PENDING_MAX_ENTRIES:
            break
        _pending_internal_calls.pop(oldest_id, None)


def pending_call_outputs(previous_response_id: Optional[str], input_items: List[Any]) -> List[dict]:
    """``function_call_output`` items for internal calls still open in the
    stored response ``previous_response_id`` that ``input_items`` leaves out.

    Kept (not popped) until expiry so a retried turn is settled again.
    """
    if not previous_response_id:
        return []
    entry = _pending_internal_calls.get(previous_response_id)
    if entry is None:
        return []
    expires, call_ids = entry
    if expires <= time.monotonic():
        _pending_internal_calls.pop(previous_response_id, None)
        return []
    answered = set()
    for item in input_items:
        item = _to_dict(item) if hasattr(item, "model_dump") else item
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            answered.add(item.get("call_id"))
    return [
        {"type": "function_call_output", "call_id": cid, "output": SKIPPED_SEARCH_OUTPUT}
        for cid in call_ids
        if cid not in answered
    ]


def _replayable_output(output: List[dict]) -> List[dict]:
    """Output items safe to resend as input when the upstream does not store."""
    keep = []
    for item in output:
        if item.get("type") == "reasoning" and not item.get("encrypted_content"):
            continue  # unverifiable without encrypted content
        keep.append({k: v for k, v in item.items() if k != "status" or item.get("type") == "message"})
    return keep


class ResponsesWebSearchLoop:
    """``call`` takes a ResponsesCreateRequest and returns a ResponseObject."""

    def __init__(
        self,
        request_dict: Dict[str, Any],
        cfg: WebSearchConfig,
        call: Callable[[Any], Awaitable[Any]],
        original_tools: Optional[List[Any]] = None,
    ):
        self.request_dict = request_dict
        self.cfg = cfg
        self.call = call
        self.original_tools = original_tools
        self.stats = LoopStats()
        self._guard = _QueryGuard(cfg, self.stats)
        # Set after each search round so a streaming caller can report progress.
        self.round_completed = asyncio.Event()
        self._has_client_tools = any(
            not (isinstance(t, dict) and t.get("type") == "function" and t.get("name") in INTERNAL_TOOL_NAMES)
            for t in request_dict.get("tools") or []
        )

    def _request(self, body: Dict[str, Any]):
        from app.openai_models import ResponsesCreateRequest

        body = {**body, "stream": False}
        body.pop("stream_options", None)
        return ResponsesCreateRequest.model_validate(body)

    async def run(self):
        from app.openai_models import ResponseObject

        body = dict(self.request_dict)
        stateless = body.get("store") is False
        history = _input_as_list(body.get("input"))
        usage_total: Dict[str, Any] = {}
        emitted: List[dict] = []
        while True:
            response = await self.call(self._request(body))
            data = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else dict(response)
            add_usage(usage_total, data.get("usage"))
            output = list(data.get("output") or [])
            internal = [i for i in output if _is_internal_function_call(i)]
            visible = [i for i in output if not _is_internal_function_call(i)]
            client_calls = any(i.get("type") == "function_call" for i in visible)

            if not internal:
                return self._finish(data, emitted + visible, usage_total, ResponseObject)
            if client_calls:
                if not stateless and data.get("id"):
                    _remember_pending_calls(data["id"], [i.get("call_id") for i in internal])
                return self._finish(data, emitted + visible, usage_total, ResponseObject)

            queries = [query_from_input(i.get("arguments")) for i in internal]
            if not self._guard.allowed(queries):
                if not stateless and data.get("id"):
                    _remember_pending_calls(data["id"], [i.get("call_id") for i in internal])
                blocks = emitted + visible
                if not any(i.get("type") == "message" for i in blocks):
                    blocks.append(_message_item(LIMIT_REACHED_TEXT))
                return self._finish(data, blocks, usage_total, ResponseObject)

            last = self._guard.is_last_round
            outcomes = await self._guard.search(queries)
            self.round_completed.set()
            outputs = [
                {"type": "function_call_output", "call_id": item.get("call_id"), "output": self._guard.result_text(o, last)}
                for item, o in zip(internal, outcomes)
            ]
            emitted.extend(visible)
            emitted.extend(_web_search_call_item(o) for o in outcomes)

            if not stateless and data.get("id"):
                body = {**self.request_dict, "previous_response_id": data["id"], "input": outputs}
            else:
                history = history + _replayable_output(output) + outputs
                body = {**self.request_dict, "input": history}
            tc = body.get("tool_choice")
            if isinstance(tc, dict) and tc.get("type") == "function" and tc.get("name") in INTERNAL_TOOL_NAMES:
                body["tool_choice"] = "auto"
            elif tc == "required" and not self._has_client_tools:
                body["tool_choice"] = "auto"

    def _finish(self, data, output, usage_total, response_cls):
        data = dict(data)
        data["output"] = output
        data["output_text"] = _responses_text(output) or data.get("output_text")
        if usage_total:
            data["usage"] = usage_total
        if self.original_tools is not None:
            data["tools"] = self.original_tools  # echo what the client declared
        return response_cls(**data)


def _message_item(text: str) -> dict:
    return {
        "type": "message",
        "id": f"msg_{new_responses_call_id()}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _web_search_call_item(outcome: SearchOutcome) -> dict:
    action: Dict[str, Any] = {"type": "search", "query": outcome.query}
    if outcome.ok:
        action["sources"] = [{"type": "url", "url": r.url} for r in outcome.results]
    return {
        "type": "web_search_call",
        "id": new_responses_call_id(),
        "status": "completed" if outcome.ok else "failed",
        "action": action,
    }


def _responses_sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def synthesize_responses_stream(response: Dict[str, Any], text_chunk_size: int = 200) -> Iterator[str]:
    """Typed Responses API SSE events reconstructed from a final response."""
    seq = 0

    def event(event_type: str, payload: dict) -> str:
        nonlocal seq
        out = _responses_sse(event_type, {"type": event_type, "sequence_number": seq, **payload})
        seq += 1
        return out

    pending = {**response, "status": "in_progress", "output": []}
    pending.pop("output_text", None)
    yield event("response.created", {"response": pending})
    yield event("response.in_progress", {"response": pending})

    for output_index, item in enumerate(response.get("output") or []):
        item_id = item.get("id")
        if item.get("type") == "message":
            yield event("response.output_item.added", {
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            })
            for content_index, part in enumerate(item.get("content") or []):
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "output_text":
                    yield event("response.content_part.added", {
                        "item_id": item_id, "output_index": output_index,
                        "content_index": content_index, "part": part,
                    })
                    yield event("response.content_part.done", {
                        "item_id": item_id, "output_index": output_index,
                        "content_index": content_index, "part": part,
                    })
                    continue
                text = part.get("text") or ""
                yield event("response.content_part.added", {
                    "item_id": item_id, "output_index": output_index, "content_index": content_index,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                for start in range(0, len(text), text_chunk_size):
                    yield event("response.output_text.delta", {
                        "item_id": item_id, "output_index": output_index,
                        "content_index": content_index, "delta": text[start:start + text_chunk_size],
                    })
                yield event("response.output_text.done", {
                    "item_id": item_id, "output_index": output_index,
                    "content_index": content_index, "text": text,
                })
                yield event("response.content_part.done", {
                    "item_id": item_id, "output_index": output_index,
                    "content_index": content_index, "part": part,
                })
        else:
            yield event("response.output_item.added", {"output_index": output_index, "item": item})
        yield event("response.output_item.done", {"output_index": output_index, "item": item})

    yield event("response.completed", {"response": response})

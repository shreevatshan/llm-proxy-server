"""Accumulate PlatformAI v3 chat replies into assistant turns.

Reply shape: ``messages[]`` of ``{role, index?, contents[], tools[],
finish_reason}``. Observed streaming behaviour:

- non-final frames are deltas; many carry only ``model`` or an empty message;
- text parts flagged ``"reasoning": true`` are the model's (signed, empty)
  thinking blocks, not reply text;
- tool calls stream as ``tools[]`` deltas without ``index``: the first delta
  of a call carries its ``id`` and name, later ones only argument fragments;
- the ``is_final_chunk`` frame repeats the whole text but carries no
  ``finish_reason`` and no tool calls, and its usage can be stale;
- a stream whose reply ends in a tool call is terminated with an
  ``event: error`` (HTTP-500 body) instead of a final frame, after every
  argument fragment was sent (see ``tool_calls_complete``).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.providers.platformai.translate import merge_usage, normalize_finish_reason


def _is_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        json.loads(stripped)
        return True
    except ValueError:
        return False


def message_text(message: Dict[str, Any]) -> str:
    """The reply text of a gateway message (reasoning blocks excluded)."""
    parts = message.get("contents")
    if not isinstance(parts, list):
        return ""
    return "".join(
        str(p.get("text") or "")
        for p in parts
        if isinstance(p, dict) and p.get("type", "text") == "text" and not p.get("reasoning")
    )


@dataclass
class _ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class GatewayTurn:
    """One assistant turn (one variation) being read from the gateway."""

    text: str = ""
    model: Optional[str] = None
    id: Optional[str] = None
    created: Optional[int] = None
    gateway_finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    citations: List[Any] = field(default_factory=list)
    search_results: List[Any] = field(default_factory=list)
    finished: bool = False
    _calls: List[_ToolCall] = field(default_factory=list)

    # -- frame metadata -------------------------------------------------

    def _apply_envelope(self, frame: Dict[str, Any]) -> None:
        if frame.get("model"):
            self.model = str(frame["model"])
        if frame.get("id"):
            self.id = str(frame["id"])
        if isinstance(frame.get("created"), int):
            self.created = frame["created"]
        if isinstance(frame.get("usage"), dict):
            self.usage = merge_usage(self.usage, frame["usage"])

    def _apply_message_extras(self, message: Dict[str, Any]) -> None:
        if message.get("finish_reason"):
            self.gateway_finish_reason = str(message["finish_reason"])
        if isinstance(message.get("citations"), list):
            self.citations = message["citations"]
        if isinstance(message.get("search_results"), list):
            self.search_results = message["search_results"]

    # -- streaming ------------------------------------------------------

    def apply_chunk(self, chunk: Dict[str, Any]) -> str:
        """Fold one streamed frame in; return the new text it carried."""
        if not isinstance(chunk, dict):
            return ""
        self._apply_envelope(chunk)
        final = bool(chunk.get("is_final_chunk"))
        delta = ""
        for message in chunk.get("messages") or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") not in (None, "", "assistant"):
                continue
            if isinstance(message.get("index"), int) and message["index"] != 0:
                continue
            self._apply_message_extras(message)
            text = message_text(message)
            if final:
                # The final frame repeats the whole text; emit only the unsent
                # tail (all of it when nothing was streamed before).
                sent = self.text + delta
                if text.startswith(sent):
                    delta += text[len(sent):]
                if message.get("tools"):
                    self._replace_tool_calls(message["tools"])
            else:
                delta += text
                self._merge_tool_deltas(message.get("tools"))
        if final:
            self.finished = True
        self.text += delta
        return delta

    def _merge_tool_deltas(self, calls: Any) -> None:
        if not isinstance(calls, list):
            return
        for call in calls:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            call_id = str(call.get("id") or "")
            if isinstance(index, int):
                while len(self._calls) <= index:
                    self._calls.append(_ToolCall())
                entry = self._calls[index]
            elif call_id and not any(c.id == call_id for c in self._calls):
                entry = _ToolCall()
                self._calls.append(entry)
            elif call_id:
                entry = next(c for c in self._calls if c.id == call_id)
            elif self._calls:
                entry = self._calls[-1]
            else:
                entry = _ToolCall()
                self._calls.append(entry)
            if call_id:
                entry.id = call_id
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            if fn.get("name"):
                entry.name = str(fn["name"])
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                entry.arguments += arguments
            elif arguments is not None:
                entry.arguments = json.dumps(arguments)

    def _replace_tool_calls(self, calls: Any) -> None:
        if not isinstance(calls, list):
            return
        self._calls = []
        self._merge_tool_deltas(calls)

    # -- non-streaming --------------------------------------------------

    def apply_message(self, message: Dict[str, Any]) -> None:
        self.text = message_text(message)
        self._apply_message_extras(message)
        self._replace_tool_calls(message.get("tools"))
        self.finished = True

    # -- results --------------------------------------------------------

    def tool_calls(self) -> List[Dict[str, Any]]:
        """Completed tool calls in OpenAI shape (with ``index``)."""
        out: List[Dict[str, Any]] = []
        for entry in self._calls:
            if not entry.name:
                continue
            if not entry.id:
                entry.id = f"call_{uuid.uuid4().hex[:24]}"
            out.append({
                "index": len(out),
                "id": entry.id,
                "type": "function",
                "function": {"name": entry.name, "arguments": entry.arguments.strip() or "{}"},
            })
        return out

    def tool_calls_complete(self) -> bool:
        """Whether streamed tool calls are whole: every call named, with JSON arguments."""
        return bool(self._calls) and all(c.name and _is_json(c.arguments or "{}") for c in self._calls)

    def finish_reason(self) -> str:
        return normalize_finish_reason(self.gateway_finish_reason, bool(self.tool_calls()))


def turns_from_response(data: Dict[str, Any]) -> List[GatewayTurn]:
    """One turn per reply message (variations), ordered by ``index``."""
    messages = [m for m in (data.get("messages") or []) if isinstance(m, dict)
                and m.get("role") in (None, "", "assistant")]
    messages.sort(key=lambda m: m.get("index") if isinstance(m.get("index"), int) else 0)
    turns: List[GatewayTurn] = []
    for message in messages or [{}]:
        turn = GatewayTurn()
        turn._apply_envelope(data)
        turn.apply_message(message)
        turns.append(turn)
    return turns

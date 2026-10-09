"""Server-side web search interception (SearXNG or 4get).

Routes call the ``maybe_intercept_*`` helpers; each returns None when the
feature is off for that surface/provider or the request carries no web search
tool, and otherwise a loop object that drives the upstream calls.
"""

from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional

from app.websearch.loop import (
    HEADER_NAME,
    AnthropicWebSearchLoop,
    ChatWebSearchLoop,
    ResponsesWebSearchLoop,
    pending_call_outputs,
    synthesize_responses_stream,
)
from app.websearch.settings import (
    SURFACE_ANTHROPIC_MESSAGES,
    SURFACE_CHAT_COMPLETIONS,
    SURFACE_RESPONSES,
    websearch_settings_cache,
)
from app.websearch import tools as _tools

__all__ = [
    "HEADER_NAME",
    "maybe_intercept_anthropic",
    "maybe_intercept_chat",
    "maybe_intercept_responses",
    "rewrite_anthropic_count_tokens_payload",
    "settle_pending_responses_calls",
    "synthesize_responses_stream",
    "websearch_settings_cache",
]


def _provider_key(provider: Any) -> Optional[str]:
    return getattr(provider, "full_provider_name", None)


def _anthropic_messages_dicts(messages) -> list:
    return [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else dict(m) for m in messages or []]


def maybe_intercept_anthropic(request, provider, anthropic_beta: Optional[str] = None) -> Optional[AnthropicWebSearchLoop]:
    """Rewrite an Anthropic Messages request for interception, or return None."""
    if not websearch_settings_cache.is_active(SURFACE_ANTHROPIC_MESSAGES, _provider_key(provider)):
        return None
    tools, found, had_native = _tools.rewrite_anthropic_tools(request.tools)
    messages = _anthropic_messages_dicts(request.messages)
    replayed = _tools.history_has_replayed_blocks(messages)
    if not found and not replayed:
        return None

    from app.anthropic_models import AnthropicMessage

    update: Dict[str, Any] = {"tools": tools}
    if replayed:
        messages = _tools.rehydrate_anthropic_messages(messages)
        update["tools"] = _tools.ensure_anthropic_internal_tool(tools)
        update["messages"] = [AnthropicMessage.model_validate(m) for m in messages]
    if request.tool_choice is not None:
        update["tool_choice"] = _tools.remap_anthropic_tool_choice(request.tool_choice)
    rewritten = request.model_copy(update=update)
    return AnthropicWebSearchLoop(
        rewritten,
        provider,
        websearch_settings_cache.config,
        emit_native=had_native or replayed,
        anthropic_beta=anthropic_beta,
    )


def rewrite_anthropic_count_tokens_payload(payload: Dict[str, Any], provider) -> Dict[str, Any]:
    """Apply the same tool rewrite to a count_tokens payload (no search runs)."""
    if not websearch_settings_cache.is_active(SURFACE_ANTHROPIC_MESSAGES, _provider_key(provider)):
        return payload
    tools, found, _ = _tools.rewrite_anthropic_tools(payload.get("tools"))
    messages = payload.get("messages")
    replayed = isinstance(messages, list) and _tools.history_has_replayed_blocks(messages)
    if not found and not replayed:
        return payload
    out = dict(payload)
    out["tools"] = _tools.ensure_anthropic_internal_tool(tools) if replayed else tools
    if replayed:
        out["messages"] = _tools.rehydrate_anthropic_messages(messages)
    if out.get("tool_choice") is not None:
        out["tool_choice"] = _tools.remap_anthropic_tool_choice(out["tool_choice"])
    return out


def maybe_intercept_chat(
    request,
    provider_key: Optional[str],
    call: Callable[[Any], Awaitable[Any]],
    call_stream: Callable[[Any], AsyncGenerator[str, None]],
) -> Optional[ChatWebSearchLoop]:
    """Rewrite a ChatCompletionRequest for interception, or return None."""
    if not websearch_settings_cache.is_active(SURFACE_CHAT_COMPLETIONS, provider_key):
        return None
    request_dict = request.model_dump(exclude_unset=True)
    if not _tools.detect_chat(request_dict):
        return None
    return ChatWebSearchLoop(
        _tools.rewrite_chat_request(request_dict),
        websearch_settings_cache.config,
        call,
        call_stream,
    )


def settle_pending_responses_calls(request):
    """Answer internal calls a previous intercepted turn left open.

    Runs before (and independently of) interception: the stored response named
    by ``previous_response_id`` may hold an internal ``function_call`` the client
    never saw, and the upstream rejects a follow-up that leaves it unanswered.
    """
    input_items = request.input if isinstance(request.input, list) else (
        [] if request.input is None else [{"role": "user", "content": request.input}]
    )
    outputs = pending_call_outputs(request.previous_response_id, input_items)
    if not outputs:
        return request
    return request.model_copy(update={"input": outputs + input_items})


def maybe_intercept_responses(
    request,
    provider_key: Optional[str],
    call: Callable[[Any], Awaitable[Any]],
) -> Optional[ResponsesWebSearchLoop]:
    """Rewrite a ResponsesCreateRequest for interception, or return None."""
    if not websearch_settings_cache.is_active(SURFACE_RESPONSES, provider_key):
        return None
    request_dict = request.model_dump(exclude_unset=True)
    if not _tools.detect_responses(request_dict):
        return None
    return ResponsesWebSearchLoop(
        _tools.rewrite_responses_request(request_dict),
        websearch_settings_cache.config,
        call,
        original_tools=request_dict.get("tools"),
    )


def provider_key_for_model(model_name: str) -> Optional[str]:
    """Provider key (``type:instance``) for a prefixed model name."""
    from app.providers.provider_manager import provider_manager

    try:
        provider = provider_manager.get_provider_for_model(model_name)
    except Exception:
        return None
    return _provider_key(provider)


# An SSE comment the stream wrapper counts as progress (unlike its own
# keepalives), so each round gets a fresh chunk budget.
_ROUND_PROGRESS_CHUNK = ": websearch round completed\n\n"


async def responses_websearch_stream(loop: ResponsesWebSearchLoop) -> AsyncGenerator[str, None]:
    """Run the Responses loop to completion, then replay it as typed SSE events.

    The stream wrapper's keepalives cover the wait while each upstream call
    runs; a progress comment after every search round keeps its first-chunk
    budget per round instead of for the whole loop.
    """
    import asyncio
    import json
    import logging

    from app.providers.base import ProviderHTTPError

    def error_event(code: str, message: str) -> str:
        payload = {"type": "error", "code": code, "message": message, "param": None, "sequence_number": 0}
        return f"event: error\ndata: {json.dumps(payload)}\n\n"

    task = asyncio.ensure_future(loop.run())
    try:
        while not task.done():
            waiter = asyncio.ensure_future(loop.round_completed.wait())
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
            if loop.round_completed.is_set():
                loop.round_completed.clear()
                yield _ROUND_PROGRESS_CHUNK
        try:
            response = task.result()
        except ProviderHTTPError as e:
            err = e.body.get("error") if isinstance(e.body, dict) else None
            if not isinstance(err, dict):
                err = {"message": err} if isinstance(err, str) else {}
            yield error_event(str(err.get("code") or e.status_code), err.get("message") or e.message)
            return
        except ValueError as e:
            yield error_event("invalid_request_error", str(e))
            return
        except Exception as e:  # never leak upstream details
            logging.getLogger(__name__).error("Web search responses loop failed: %s", e, exc_info=True)
            yield error_event("server_error", "Responses create error")
            return
    finally:
        if not task.done():
            task.cancel()
    data = response.model_dump(exclude_none=True) if hasattr(response, "model_dump") else dict(response)
    for event in synthesize_responses_stream(data):
        yield event

"""Zoho PlatformAI provider (native v3 ``/ai/chat``).

Translates OpenAI chat/completions onto PlatformAI's native chat API and
serves Anthropic ``/v1/messages`` through the generic chat adapter. Auth is a
PlatformAI API key or a Zoho OAuth refresh token. The gateway has no model
listing endpoint, so the admin lists the models this provider serves.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.model_alias import echo_model_name
from app.openai_models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ResponseFunction,
    ToolCall,
)
from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    anthropic_adapter_metadata,
)
from app.providers.base import AnthropicRequestMetadata, BaseProvider, ProviderHTTPError
from app.providers.platformai import limits
from app.providers.platformai.auth import PlatformAITokenManager
from app.providers.platformai.client import (
    ApiKeyAuth,
    OAuthAuth,
    PlatformAIClient,
    gateway_error,
    metadata_header,
    pai_code,
    parse_error,
)
from app.providers.platformai.stream import GatewayTurn, turns_from_response
from app.providers.platformai.translate import (
    EFFORT_LADDER,
    bad_request,
    build_chat_body,
    build_usage,
    step_down_effort,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://platformai.zoho.in"
DEFAULT_ACCOUNTS_URL = "https://accounts.zoho.in"

# Refusals of a request we adjusted (dropped params / effort / max_tokens) are
# retried at most this often.
_MAX_ADJUSTMENTS = 4

_REFUSAL_CODES = {"PAI-MODEL-002", "PAI-MODEL-003", "PAI-MODEL-004", "PAI-PROVIDER-002"}


def _normalize_models(raw: Any) -> List[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [p for p in raw.replace("\n", ",").split(",")]
    if isinstance(raw, dict):
        raw = raw.get("openai") or raw.get("deployments") or []
    names: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        name = str(item).strip() if item is not None else ""
        if name and name not in names:
            names.append(name)
    return names


class PlatformAIProvider(BaseProvider):
    """Provider for Zoho's PlatformAI gateway."""

    anthropic_adapter_supports_top_k = True
    anthropic_adapter_supports_file_parts = False
    anthropic_adapter_unforwarded_params = ("user", "parallel_tool_calls")

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = (config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.accounts_url = (config.get("oauth_accounts_url") or DEFAULT_ACCOUNTS_URL).rstrip("/")
        self.provider_key = config.get("provider_key") or self.full_provider_name
        self.models = _normalize_models(config.get("models"))
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        if config.get("byok"):
            # BYOK: no redirects, every request's destination re-checked (url_guard).
            from app.providers.url_guard import guarded_http_client
            self.client = guarded_http_client(timeout=timeout)
        else:
            self.client = httpx.AsyncClient(timeout=timeout)
        api_key = (config.get("api_key") or "").strip()
        self.tokens: Optional[PlatformAITokenManager] = None
        if api_key:
            auth = ApiKeyAuth(api_key)
        elif config.get("oauth_refresh_token"):
            self.tokens = PlatformAITokenManager(
                provider_key=self.provider_key,
                accounts_url=self.accounts_url,
                client_id=(config.get("oauth_client_id") or "").strip(),
                client_secret=config.get("oauth_client_secret"),
                refresh_token=config.get("oauth_refresh_token") or "",
                http_client=self.client,
            )
            auth = OAuthAuth(self.tokens)
        else:
            raise ValueError("PlatformAI needs an API key, or an OAuth client ID and refresh token")
        self.auth_mode = "api_key" if api_key else "oauth"
        self.gateway = PlatformAIClient(
            base_url=self.base_url,
            auth=auth,
            token_id=config.get("pai_token_id"),
            http_client=self.client,
        )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_supported_apis(self) -> List[str]:
        return ["openai", "anthropic"]

    def get_supported_endpoints(self) -> List[str]:
        return ["/v1/chat/completions", "/v1/completions", "/v1/messages"]

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        return "adapter"

    def get_anthropic_request_metadata(self, request, anthropic_beta=None) -> AnthropicRequestMetadata:
        return anthropic_adapter_metadata(
            self, request, transport="chat", allow_thinking=True, anthropic_beta=anthropic_beta
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def get_available_models(self) -> List[ModelInfo]:
        return [self.create_model_info(name, "platformai") for name in self.models]

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata(request: ChatCompletionRequest) -> Optional[str]:
        extra = getattr(request, "model_extra", None) or {}
        return metadata_header(extra.get("metadata"))

    async def _call_chat(self, request: ChatCompletionRequest, *, stream: bool):
        """Send a chat request, adapting to a learned limit on each refusal.

        Returns the response JSON or, when streaming, the open ``httpx.Response``.
        Refusals arrive as an HTTP 400 before the stream starts, so retrying
        here never re-sends output the client already has.
        """
        model = self.get_model_id(request.model)
        key = self.provider_key
        metadata = self._metadata(request)
        for _ in range(_MAX_ADJUSTMENTS + 1):
            body = build_chat_body(
                request,
                upstream_model=model,
                max_tokens_ceiling=limits.max_tokens_ceiling(key, model),
                effort_ceiling=limits.effort_ceiling(key, model),
                unsupported=limits.unsupported_params(key, model),
                stream=stream,
            )
            try:
                if stream:
                    return await self.gateway.open_stream(body, metadata=metadata)
                return await self.gateway.chat(body, metadata=metadata)
            except ProviderHTTPError as err:
                if err.status_code != 400 or not self._learn_from_refusal(model, body, err):
                    raise
        raise bad_request("PlatformAI kept refusing the request after adjusting its parameters.")

    def _learn_from_refusal(self, model: str, body: Dict[str, Any], err: ProviderHTTPError) -> bool:
        """Record the limit a 400 names; True when the retry will differ."""
        code = pai_code(err)
        if code not in _REFUSAL_CODES:
            return False
        message = getattr(err, "upstream_message", None) or err.message
        if code == "PAI-MODEL-002":
            param = limits.parse_unsupported_param(message)
            if param and param in body:
                limits.learn_unsupported_param(self.provider_key, model, param)
                return True
            return False
        effort = (body.get("reasoning") or {}).get("effort")
        if effort in EFFORT_LADDER and limits.is_effort_refusal(message):
            limits.learn_effort_ceiling(self.provider_key, model, step_down_effort(effort))
            return True
        limit = limits.parse_max_tokens_limit(message, int(body.get("max_tokens") or 0))
        if limit:
            limits.learn_max_tokens_ceiling(self.provider_key, model, limit)
            return True
        return False

    def _note_served_model(self, requested: str, turn: GatewayTurn) -> None:
        if turn.model and turn.model != requested:
            logger.warning("PlatformAI: requested model %s but the gateway served %s", requested, turn.model)

    @staticmethod
    def _choice_message(turn: GatewayTurn) -> ChatMessage:
        calls = turn.tool_calls()
        message = ChatMessage(role="assistant", content=turn.text if (turn.text or not calls) else None)
        if calls:
            message.tool_calls = [
                ToolCall(id=c["id"], type="function",
                         function=ResponseFunction(name=c["function"]["name"], arguments=c["function"]["arguments"]))
                for c in calls
            ]
        if turn.citations:
            setattr(message, "citations", turn.citations)
        if turn.search_results:
            setattr(message, "search_results", turn.search_results)
        return message

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        payload = await self._call_chat(request, stream=False)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        turns = turns_from_response(data)
        self._note_served_model(self.get_model_id(request.model), turns[0])
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=echo_model_name(request),
            choices=[
                ChatCompletionChoice(index=i, message=self._choice_message(turn), finish_reason=turn.finish_reason())
                for i, turn in enumerate(turns)
            ],
            usage=build_usage(turns[0].usage),
        )

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        include_usage = True
        if request.stream_options is not None and request.stream_options.include_usage is not None:
            include_usage = bool(request.stream_options.include_usage)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model = echo_model_name(request)

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> str:
            return self.format_sse_data({
                "id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })

        response = None
        try:
            response = await self._call_chat(request, stream=True)
            turn = GatewayTurn()
            yield chunk({"role": "assistant", "content": ""})
            async for event in self.gateway.events(response):
                if event.event == "error":
                    if turn.tool_calls_complete():
                        # Gateway bug: a reply ending in a tool call is closed
                        # with an error event instead of a final frame, after
                        # every argument fragment was sent.
                        logger.info("PlatformAI: stream ended with an error after complete tool calls; "
                                    "treating it as the end of the turn (%s)", event.data[:200])
                        break
                    raise gateway_error(502, parse_error(event.data))
                if event.event not in ("", "generate"):
                    logger.warning("PlatformAI: ignoring stream event %r: %s", event.event, event.data[:300])
                    continue
                try:
                    parsed = json.loads(event.data)
                except ValueError:
                    logger.warning("PlatformAI: unparseable stream chunk: %s", event.data[:300])
                    continue
                payload = parsed.get("data", parsed) if isinstance(parsed, dict) else None
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("error"), (dict, str)):
                    raise gateway_error(502, parse_error(json.dumps(payload)))
                delta = turn.apply_chunk(payload)
                if delta:
                    yield chunk({"content": delta})
                if turn.finished:
                    break
            self._note_served_model(self.get_model_id(request.model), turn)
            calls = turn.tool_calls()
            if calls:
                yield chunk({"tool_calls": calls})
            yield chunk({}, turn.finish_reason())
            if include_usage:
                yield self.format_sse_data({
                    "id": chunk_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [], "usage": build_usage(turn.usage).model_dump(),
                })
            yield self.format_sse_done()
        except ProviderHTTPError as err:
            logger.error("PlatformAI stream error: %s", err.message)
            yield self._sse_error(err)
            yield self.format_sse_done()
        except Exception as exc:  # never leak an exception mid-stream
            logger.exception("PlatformAI stream failed")
            yield self.format_sse_data({"error": {"message": f"PlatformAI stream failed: {type(exc).__name__}",
                                                  "type": "server_error", "code": 500}})
            yield self.format_sse_done()
        finally:
            if response is not None:
                await response.aclose()

    def _sse_error(self, err: ProviderHTTPError) -> str:
        body = err.body.get("error", {}) if isinstance(err.body, dict) else {}
        return self.format_sse_data({"error": {
            "message": body.get("message", err.message),
            "type": body.get("type", "upstream_error"),
            "code": err.status_code,
        }})

    # ------------------------------------------------------------------
    # Legacy completions (built on chat)
    # ------------------------------------------------------------------

    @staticmethod
    def _chat_from_completion(request: CompletionRequest, stream: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=request.model,
            messages=[ChatMessage(role="user", content=request.prompt)],
            max_tokens=request.max_tokens,
            stream=stream,
        )

    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        chat = await self.chat_completion(self._chat_from_completion(request, False))
        return CompletionResponse(
            id=chat.id.replace("chatcmpl", "cmpl"),
            created=chat.created,
            model=chat.model,
            choices=[CompletionChoice(index=0, text=chat.choices[0].message.content or "",
                                      finish_reason=chat.choices[0].finish_reason)],
            usage=chat.usage,
        )

    async def completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        async for sse in self.chat_completion_stream(self._chat_from_completion(request, True)):
            if not sse.startswith("data: ") or sse.strip() == "data: [DONE]":
                yield sse
                continue
            try:
                data = json.loads(sse[6:])
            except ValueError:
                continue
            if "error" in data:
                yield sse
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            text = (choice.get("delta") or {}).get("content") or ""
            if not text and choice.get("finish_reason") is None:
                continue
            yield self.format_sse_data({
                "id": data["id"].replace("chatcmpl", "cmpl"), "object": "text_completion",
                "created": data["created"], "model": data["model"],
                "choices": [{"index": 0, "text": text, "finish_reason": choice.get("finish_reason")}],
            })

    # ------------------------------------------------------------------
    # Anthropic Messages (adapter over chat)
    # ------------------------------------------------------------------

    async def anthropic_messages(self, request, anthropic_beta=None) -> dict:
        return await anthropic_adapter_messages(
            self, request, transport="chat", allow_thinking=True, anthropic_beta=anthropic_beta
        )

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        async for chunk in anthropic_adapter_messages_stream(
            self, request, transport="chat", allow_thinking=True, anthropic_beta=anthropic_beta
        ):
            yield chunk

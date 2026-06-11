"""
Anthropic Compatible Provider
Shared Anthropic Messages API handling for providers that can speak the
Anthropic-compatible protocol (e.g. direct Anthropic gateways, custom proxies).
"""

import json
import logging
import os
import time
from typing import List, Any, AsyncGenerator, Optional

from app.anthropic_models import (
    AnthropicMessagesRequest,
    build_anthropic_sdk_kwargs,
    is_anthropic_terminal_stream_event,
)
from app.openai_models import ModelInfo
from app.providers.base import BaseProvider, ProviderHTTPError

logger = logging.getLogger(__name__)

_DEFAULT_API_KEY = "not-required"


def _translate_anthropic_sdk_error(e: Exception, provider_name: str) -> ProviderHTTPError:
    """Convert an Anthropic SDK exception into a ProviderHTTPError.

    Maps the SDK's exception hierarchy onto HTTP status codes and Anthropic-shaped
    error envelopes so the route's existing ProviderHTTPError branches handle them
    consistently instead of falling through to the generic 500/stream_error path.
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        return ProviderHTTPError(
            status_code=502,
            message=str(e),
            body={"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )

    headers = None
    try:
        if hasattr(e, "response") and e.response is not None:
            headers = dict(e.response.headers)
    except Exception:
        pass

    if isinstance(e, _anthropic.APIStatusError):
        body = e.body if isinstance(e.body, dict) else {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        return ProviderHTTPError(
            status_code=e.status_code,
            message=str(e),
            body=body,
            headers=headers,
        )

    if isinstance(e, _anthropic.APIResponseValidationError):
        message = (
            f"Upstream provider '{provider_name}' returned an unparseable response: {e.message}"
        )
        return ProviderHTTPError(
            status_code=502,
            message=message,
            body={"type": "error", "error": {"type": "api_error", "message": message}},
            headers=headers,
        )

    if isinstance(e, _anthropic.APITimeoutError):
        message = f"Request to upstream provider '{provider_name}' timed out: {e}"
        return ProviderHTTPError(
            status_code=504,
            message=message,
            body={"type": "error", "error": {"type": "api_error", "message": message}},
        )

    if isinstance(e, (_anthropic.APIConnectionError, _anthropic.APIError)):
        message = f"Connection to upstream provider '{provider_name}' failed: {e}"
        return ProviderHTTPError(
            status_code=502,
            message=message,
            body={"type": "error", "error": {"type": "api_error", "message": message}},
        )

    # Fallback for any other AnthropicError subclass
    message = f"Upstream provider '{provider_name}' error: {e}"
    return ProviderHTTPError(
        status_code=502,
        message=message,
        body={"type": "error", "error": {"type": "api_error", "message": message}},
    )


ANTHROPIC_POST_TERMINAL_DRAIN_MAX_EVENTS = int(
    os.getenv("ANTHROPIC_POST_TERMINAL_DRAIN_MAX_EVENTS", "4")
)
ANTHROPIC_POST_TERMINAL_DRAIN_MAX_SECONDS = float(
    os.getenv("ANTHROPIC_POST_TERMINAL_DRAIN_MAX_SECONDS", "0.25")
)


def get_anthropic_post_terminal_drain_stop_reason(
    *,
    terminal_seen_at: Optional[float],
    drained_event_count: int,
) -> Optional[str]:
    """Return the stop reason when post-terminal drain exceeds its budget."""
    if terminal_seen_at is None:
        return None
    if drained_event_count >= ANTHROPIC_POST_TERMINAL_DRAIN_MAX_EVENTS:
        return f"event_budget_exceeded({drained_event_count})"
    elapsed = time.monotonic() - terminal_seen_at
    if elapsed >= ANTHROPIC_POST_TERMINAL_DRAIN_MAX_SECONDS:
        return f"time_budget_exceeded({elapsed:.3f}s)"
    return None


class AnthropicCompatibleProvider(BaseProvider):
    """
    Base class for providers that use the Anthropic SDK for API calls.
    Provides shared Anthropic-compatible request/response handling.

    Expected attributes on subclasses:
    - self.base_url
    - self.api_key
    - self._supported_apis (list of strings)
    - self.custom_provider_name
    - self.get_model_id(model_name: str) -> str
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._anthropic_client = None

    def _init_anthropic_client(self) -> None:
        """Initialize AsyncAnthropic client when Anthropic API is supported."""
        if "anthropic" not in getattr(self, "_supported_apis", []):
            return

        try:
            from anthropic import AsyncAnthropic

            # Strip /v1 suffix — the Anthropic SDK appends its own path
            base_url = (self.base_url or "").rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]

            self._anthropic_client = AsyncAnthropic(
                base_url=base_url,
                api_key=self.api_key or _DEFAULT_API_KEY,
            )
            logger.info(f"Anthropic client initialized for {self.custom_provider_name}")
        except ImportError:
            logger.warning(
                f"anthropic SDK not installed — Anthropic API unavailable for {self.custom_provider_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to init Anthropic client for {self.custom_provider_name}: {e}")

    async def _fetch_anthropic_models(self) -> List[ModelInfo] | None:
        """Fetch models via Anthropic models.list endpoint."""
        if "anthropic" not in getattr(self, "_supported_apis", []) or not getattr(self, "_anthropic_client", None):
            return None
        try:
            models_page = await self._anthropic_client.models.list()
            return [
                self.create_model_info(m.id, self.custom_provider_name)
                for m in models_page.data
            ]
        except Exception as e:
            logger.debug(f"Error fetching models via Anthropic API from {self.custom_provider_name}: {e}")
            return None

    async def anthropic_messages(
        self,
        request: AnthropicMessagesRequest,
        anthropic_beta: str | None = None,
    ) -> Any:
        """Handle Anthropic Messages API request (non-streaming)."""
        if not getattr(self, "_anthropic_client", None):
            raise NotImplementedError(f"Anthropic API not available for {self.custom_provider_name}")

        kwargs = build_anthropic_sdk_kwargs(request, self.get_model_id(request.model))
        if anthropic_beta:
            kwargs["extra_headers"] = {
                **kwargs.get("extra_headers", {}),
                "anthropic-beta": anthropic_beta,
            }
        try:
            response = await self._anthropic_client.messages.create(**kwargs)
        except Exception as e:
            try:
                import anthropic as _anthropic
                if isinstance(e, _anthropic.AnthropicError):
                    raise _translate_anthropic_sdk_error(e, self.custom_provider_name) from e
            except ImportError:
                pass
            raise
        return json.loads(response.model_dump_json(warnings="none"))

    async def anthropic_messages_stream(
        self,
        request: AnthropicMessagesRequest,
        anthropic_beta: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Handle streaming Anthropic Messages API request."""
        if not getattr(self, "_anthropic_client", None):
            raise NotImplementedError(f"Anthropic streaming not available for {self.custom_provider_name}")

        kwargs = build_anthropic_sdk_kwargs(request, self.get_model_id(request.model))
        if anthropic_beta:
            kwargs["extra_headers"] = {
                **kwargs.get("extra_headers", {}),
                "anthropic-beta": anthropic_beta,
            }

        terminal_event_type = None
        terminal_seen_at: Optional[float] = None
        drained_event_count = 0
        transport_eof_observed = False

        try:
            async with self._anthropic_client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if terminal_event_type is not None:
                        drained_event_count += 1
                        drain_stop_reason = get_anthropic_post_terminal_drain_stop_reason(
                            terminal_seen_at=terminal_seen_at,
                            drained_event_count=drained_event_count,
                        )
                        if drain_stop_reason is not None:
                            logger.warning(
                                "Anthropic post-terminal drain budget reached for provider=%s model=%s terminal_event=%s drained_event_count=%s stop_reason=%s",
                                getattr(self, "full_provider_name", self.custom_provider_name),
                                request.model,
                                terminal_event_type,
                                drained_event_count,
                                drain_stop_reason,
                            )
                            break
                        continue
                    if hasattr(event, "model_dump_json"):
                        json_str = event.model_dump_json(exclude_none=True, warnings="none")
                        event_data = json.loads(json_str)
                    else:
                        event_data = event
                        json_str = json.dumps(event_data, ensure_ascii=False, separators=(",", ":"))
                    event_type = (
                        event_data.get("type", "unknown") if isinstance(event_data, dict) else "unknown"
                    )
                    sse = f"event: {event_type}\ndata: {json_str}\n\n"
                    yield sse

                    if is_anthropic_terminal_stream_event(event_type=event_type, event_data=event_data):
                        terminal_event_type = event_type
                        terminal_seen_at = time.monotonic()
                        logger.info(
                            "Anthropic upstream stream reached terminal event for provider=%s model=%s event_type=%s",
                            getattr(self, "full_provider_name", self.custom_provider_name),
                            request.model,
                            event_type,
                        )
        except Exception as e:
            try:
                import anthropic as _anthropic
                if isinstance(e, _anthropic.AnthropicError):
                    raise _translate_anthropic_sdk_error(e, self.custom_provider_name) from e
            except ImportError:
                pass
            raise

        if terminal_event_type is not None:
            drain_ms = 0.0
            if terminal_seen_at is not None:
                drain_ms = (time.monotonic() - terminal_seen_at) * 1000
            logger.info(
                "Anthropic stream completed after terminal event for provider=%s model=%s event_type=%s drained_event_count=%s post_terminal_drain_ms=%.1f",
                getattr(self, "full_provider_name", self.custom_provider_name),
                request.model,
                terminal_event_type,
                drained_event_count,
                drain_ms,
            )

        if terminal_event_type is None:
            transport_eof_observed = True
            logger.debug(
                "Anthropic upstream stream ended without terminal event for provider=%s model=%s transport_eof_observed=%s",
                getattr(self, "full_provider_name", self.custom_provider_name),
                request.model,
                transport_eof_observed,
            )

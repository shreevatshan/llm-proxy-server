"""
Custom Providers
Unified implementation for all custom providers (OpenAI, Ollama, LlamaCpp, etc.)
Supports both OpenAI and Anthropic API compatibility when configured.
"""

import json
from typing import List, Dict, Any

from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    anthropic_adapter_metadata,
)
from app.providers.anthropic_compatible import AnthropicCompatibleProvider
from app.providers.base import AnthropicRequestMetadata, BaseProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.openai_models import ModelInfo


class CustomProvider(AnthropicCompatibleProvider, OpenAICompatibleProvider):
    """
    Unified custom provider with support for both OpenAI and Anthropic APIs.

    Handles any server that exposes an OpenAI-compatible and/or Anthropic-compatible
    API (e.g. Ollama, LlamaCpp, vLLM, custom gateways).

    Configuration:
        custom_provider_name: Identifier for the provider (required)
        base_url:             OpenAI-compatible server URL including /v1 path
                              (required unless anthropic_base_url is set)
        api_key:              API key for authentication (optional)
        supported_apis:       List of API formats, e.g. ["openai", "anthropic"].
                              "anthropic" means the upstream serves a native
                              /v1/messages; without it, OpenAI-capable models
                              are served on /v1/messages through the adapter.
        anthropic_base_url:   Native Anthropic base URL when it differs from
                              base_url (e.g. https://api.deepseek.com/anthropic)
        anthropic_auth_style: "x-api-key" (default) or "bearer"
        split_think_tags:     Route <think>...</think> content to thinking
                              blocks in adapter mode (default off)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.custom_provider_name = config.get('custom_provider_name', 'custom').lower()

        # An Anthropic-only provider has no OpenAI base URL; fall back to the
        # Anthropic one so base_url is always a usable upstream address.
        self.base_url = config.get('base_url') or config.get('endpoint') or config.get('anthropic_base_url')
        if not self.base_url:
            raise ValueError(
                f"base_url or anthropic_base_url is required for custom provider '{self.custom_provider_name}'"
            )

        # Reject non-http(s) base URLs up front (SSRF hardening): the URL is
        # admin-supplied and used for outbound requests.
        from urllib.parse import urlparse
        _parsed = urlparse(self.base_url)
        if _parsed.scheme not in ("http", "https") or not _parsed.hostname:
            raise ValueError(
                f"Invalid base_url for custom provider '{self.custom_provider_name}' "
                f"(must be http(s) with a host): {self.base_url!r}"
            )

        self.api_key = config.get('api_key')
        # The configured list: "anthropic" here means a native upstream
        # endpoint. get_supported_apis() additionally advertises "anthropic"
        # for OpenAI-only providers, which are served through the adapter.
        self._supported_apis = self._parse_supported_apis(config.get('supported_apis', ['openai']))
        self.split_think_tags = bool(config.get('split_think_tags', False))
        # Keep provider_type as the class-derived type ("custom"); the
        # user-chosen instance name is exposed via instance_name /
        # full_provider_name (set by BaseProvider.__init__). Overwriting
        # provider_type with the instance name previously let an instance named
        # "azure" trip Azure-only code paths that branched on provider_type.

        self._assert_anthropic_method_resolution()
        self._init_openai_client()
        self._init_anthropic_client()

    # ==================== Initialization helpers ====================

    def _assert_anthropic_method_resolution(self) -> None:
        """Fail fast if Anthropic methods resolve to BaseProvider stubs."""
        cls = type(self)
        invalid_resolution = (
            cls.anthropic_messages is BaseProvider.anthropic_messages
            or cls.anthropic_messages_stream is BaseProvider.anthropic_messages_stream
        )
        if invalid_resolution:
            mro = " -> ".join(base.__name__ for base in cls.__mro__)
            raise RuntimeError(
                f"Invalid CustomProvider MRO for Anthropic API ({mro}): "
                "anthropic_messages resolved to BaseProvider."
            )

    @staticmethod
    def _parse_supported_apis(raw: Any) -> List[str]:
        """Normalize supported_apis config value to a list of strings."""
        if isinstance(raw, list) and raw:
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return ['openai']

    # ==================== Public API ====================

    def get_supported_apis(self) -> List[str]:
        """Return the configured API formats, plus "anthropic" (via the
        adapter) for OpenAI-capable providers."""
        apis = list(self._supported_apis)
        if "openai" in apis and "anthropic" not in apis:
            apis.append("anthropic")
        return apis

    def _has_native_anthropic(self) -> bool:
        return "anthropic" in self._supported_apis

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        if not self._is_chat_capable_model(model_name):
            return "unsupported"
        if self._has_native_anthropic():
            return "native"
        if "openai" in self._supported_apis:
            return "adapter"
        return "unsupported"

    def get_anthropic_request_metadata(self, request, anthropic_beta=None) -> AnthropicRequestMetadata:
        if self.get_anthropic_mode_for_model(request.model) == "adapter":
            return anthropic_adapter_metadata(
                self,
                request,
                transport="chat",
                allow_thinking=True,
                echo_reasoning=True,
                anthropic_beta=anthropic_beta,
            )
        return super().get_anthropic_request_metadata(request, anthropic_beta=anthropic_beta)

    async def anthropic_messages(self, request, anthropic_beta=None):
        """Native passthrough, or the chat-completions adapter for OpenAI-only upstreams."""
        if self.get_anthropic_mode_for_model(request.model) == "adapter":
            return await anthropic_adapter_messages(
                self,
                request,
                transport="chat",
                allow_thinking=True,
                echo_reasoning=True,
                anthropic_beta=anthropic_beta,
            )
        return await super().anthropic_messages(request, anthropic_beta=anthropic_beta)

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        """Streaming variant of :meth:`anthropic_messages`."""
        if self.get_anthropic_mode_for_model(request.model) == "adapter":
            stream = anthropic_adapter_messages_stream(
                self,
                request,
                transport="chat",
                allow_thinking=True,
                echo_reasoning=True,
                anthropic_beta=anthropic_beta,
            )
        else:
            stream = super().anthropic_messages_stream(request, anthropic_beta=anthropic_beta)
        async for chunk in stream:
            yield chunk

    def get_supported_endpoints(self) -> List[str]:
        """Custom providers support endpoints based on their configured APIs."""
        endpoints = []
        if "openai" in self._supported_apis:
            endpoints.extend([
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/images/generations",
                "/v1/audio/speech",
                "/v1/audio/transcriptions",
                "/v1/responses",
            ])
        if ("anthropic" in self._supported_apis and getattr(self, "_anthropic_client", None)) or "openai" in self._supported_apis:
            endpoints.append("/v1/messages")
        return endpoints

    def get_model_id(self, model_name: str) -> str:
        """Extract model ID from provider-prefixed model name (e.g. 'ollama/llama3' → 'llama3')."""
        if '/' in model_name:
            return model_name.split('/', 1)[1]
        return model_name

    async def get_available_models(self) -> List[ModelInfo]:
        """Discover models via OpenAI API, falling back to Anthropic API."""
        models = await self._fetch_openai_models()
        if models is not None:
            return models

        models = await self._fetch_anthropic_models()
        if models is not None:
            return models

        return []

    # ==================== Model discovery helpers ====================


def create_custom_provider(config: Dict[str, Any]) -> "CustomProvider":
    """
    Factory function to create a CustomProvider instance.

    Args:
        config: Provider configuration dict with keys:
            custom_provider_name, base_url, api_key, supported_apis

    Returns:
        Configured CustomProvider instance.
    """
    if not config.get('custom_provider_name') and config.get('provider_type'):
        config['custom_provider_name'] = config['provider_type'].lower()

    return CustomProvider(config)

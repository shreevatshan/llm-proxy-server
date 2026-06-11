"""
Custom Providers
Unified implementation for all custom providers (OpenAI, Ollama, LlamaCpp, etc.)
Supports both OpenAI and Anthropic API compatibility when configured.
"""

import json
from typing import List, Dict, Any

from app.providers.anthropic_compatible import AnthropicCompatibleProvider
from app.providers.base import BaseProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.openai_models import ModelInfo


class CustomProvider(AnthropicCompatibleProvider, OpenAICompatibleProvider):
    """
    Unified custom provider with support for both OpenAI and Anthropic APIs.

    Handles any server that exposes an OpenAI-compatible and/or Anthropic-compatible
    API (e.g. Ollama, LlamaCpp, vLLM, custom gateways).

    Configuration:
        custom_provider_name: Identifier for the provider (required)
        base_url:             Server URL including /v1 path (required)
        api_key:              API key for authentication (optional)
        supported_apis:       List of API formats, e.g. ["openai", "anthropic"]
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.custom_provider_name = config.get('custom_provider_name', 'custom').lower()

        self.base_url = config.get('base_url') or config.get('endpoint')
        if not self.base_url:
            raise ValueError(f"base_url is required for custom provider '{self.custom_provider_name}'")

        self.api_key = config.get('api_key')
        self._supported_apis = self._parse_supported_apis(config.get('supported_apis', ['openai']))
        self.provider_type = config.get('name', self.custom_provider_name)

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
        """Return the configured supported API formats."""
        return self._supported_apis

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        if "anthropic" in self._supported_apis and self._is_chat_capable_model(model_name):
            return "native"
        return "unsupported"

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
        if "anthropic" in self._supported_apis and self._anthropic_client:
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

    # ==================== Lifecycle ====================

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup API clients."""
        if getattr(self, 'client', None):
            await self.client.close()
        if self._anthropic_client:
            await self._anthropic_client.close()


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

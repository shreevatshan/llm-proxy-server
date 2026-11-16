"""
Custom Providers
Unified implementation for all custom providers (OpenAI, Ollama, LlamaCpp, etc.)
Supports OpenAI, Anthropic and TypeSafe System One API compatibility when configured.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    anthropic_adapter_metadata,
)
from app.providers.anthropic_compatible import AnthropicCompatibleProvider
from app.providers.base import (
    SYSTEMONE_API,
    AnthropicRequestMetadata,
    BaseProvider,
    ProviderHTTPError,
)
from app.providers.openai_compatible import (
    _DEFAULT_API_KEY,
    OpenAICompatibleProvider,
    _translate_openai_sdk_error,
)
from app.model_alias import echo_model_name
from app.openai_models import ModelInfo

logger = logging.getLogger(__name__)

# {provider_key: {model_id: (description, release_date)}} from each System One
# provider's latest successful GET {systemone_base_url}/models (base_url for rows
# without it), for the TypeSafe-shaped /v1/models and per-model capability. Kept outside ModelInfo: auto-sync rebuilds ModelInfo from the DB
# and the OpenAI listing would show new fields. Each listing replaces its
# provider's entry, so models that disappeared upstream drop out; the strings
# are upstream-controlled (BYOK too), so they are bounded.
_SYSTEMONE_META: Dict[str, Dict[str, Tuple[Optional[str], Optional[str]]]] = {}
_RELEASE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_META_MODELS = 5000
_MAX_DESCRIPTION_CHARS = 1024


def systemone_model_meta(model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """(description, release_date) recorded by discovery, or (None, None)."""
    provider_key = model_id.split("/", 1)[0]
    return _SYSTEMONE_META.get(provider_key, {}).get(model_id, (None, None))


# {provider_key: frozenset(model_id)} from the latest successful OpenAI (or
# Anthropic) listing of a provider whose System One URL is a different one, so
# each model is offered only on the API whose listing named it.
_CHAT_LISTED: Dict[str, frozenset] = {}


def forget_systemone_meta(provider_key: str) -> None:
    """Drop a removed provider's discovery metadata."""
    _SYSTEMONE_META.pop(provider_key, None)
    _CHAT_LISTED.pop(provider_key, None)


def _release_date(item: Dict[str, Any]) -> Optional[str]:
    value = item.get("release_date")
    if isinstance(value, str) and _RELEASE_DATE.match(value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            pass
    created = item.get("created")
    if isinstance(created, (int, float)) and not isinstance(created, bool) and created > 0:
        try:
            return datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            pass
    return None


def _listed_models(listing: Any) -> Optional[List[Dict[str, Any]]]:
    """Items of an OpenAI ``{"data":[{"id"}]}`` or TypeSafe ``{"models":[{"name"}]}``
    listing, normalized to dicts with an ``id``; de-duplicated, order kept."""
    if not isinstance(listing, dict):
        return None
    items = listing.get("data") if isinstance(listing.get("data"), list) else listing.get("models")
    if not isinstance(items, list):
        return None
    seen, out = set(), []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("name")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append({**item, "id": model_id})
    return out


class CustomProvider(AnthropicCompatibleProvider, OpenAICompatibleProvider):
    """
    Unified custom provider with support for both OpenAI and Anthropic APIs.

    Handles any server that exposes an OpenAI-compatible and/or Anthropic-compatible
    API (e.g. Ollama, LlamaCpp, vLLM, custom gateways).

    Configuration:
        custom_provider_name: Identifier for the provider (required)
        base_url:             OpenAI-compatible server URL including /v1 path
                              (required unless anthropic_base_url or
                              systemone_base_url is set)
        api_key:              API key for authentication (optional)
        supported_apis:       List of API formats, e.g. ["openai", "anthropic"].
                              "anthropic" means the upstream serves a native
                              /v1/messages; without it, OpenAI-capable models
                              are served on /v1/messages through the adapter.
                              "systemone" means the upstream serves the TypeSafe
                              System One API (POST {systemone_base_url}/systemone,
                              GET {systemone_base_url}/models); with neither
                              "openai" nor "anthropic" the provider serves only
                              the TypeSafe app.
        anthropic_base_url:   Native Anthropic base URL when it differs from
                              base_url (e.g. https://api.deepseek.com/anthropic)
        systemone_base_url:   TypeSafe System One base URL including /v1
                              (e.g. https://api.typesafe.ai/v1); rows that list
                              "systemone" without it use base_url
        anthropic_auth_style: "x-api-key" (default) or "bearer"
        split_think_tags:     Route <think>...</think> content to thinking
                              blocks in adapter mode (default off)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.custom_provider_name = config.get('custom_provider_name', 'custom').lower()

        # An Anthropic-only provider has no OpenAI base URL; fall back to the
        # Anthropic one so base_url is always a usable upstream address. The
        # System One URL only stands in when there is no chat URL at all, and
        # then the OpenAI and Anthropic APIs are not served (below): System One
        # is never reached through them, nor they through its URL.
        chat_url = config.get('base_url') or config.get('endpoint') or config.get('anthropic_base_url')
        self.base_url = chat_url or config.get('systemone_base_url')
        if not self.base_url:
            raise ValueError(
                "base_url, anthropic_base_url or systemone_base_url is required for custom provider "
                f"'{self.custom_provider_name}'"
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
        if not chat_url and any(api in self._supported_apis for api in ("openai", "anthropic")):
            logger.warning(
                "Custom provider '%s': OpenAI/Anthropic APIs need an OpenAI or Anthropic base URL; not served",
                self.custom_provider_name,
            )
            self._supported_apis = [a for a in self._supported_apis if a not in ("openai", "anthropic")]
        self.split_think_tags = bool(config.get('split_think_tags', False))
        # Keep provider_type as the class-derived type ("custom"); the
        # user-chosen instance name is exposed via instance_name /
        # full_provider_name (set by BaseProvider.__init__). Overwriting
        # provider_type with the instance name previously let an instance named
        # "azure" trip Azure-only code paths that branched on provider_type.

        self._assert_anthropic_method_resolution()
        self._init_openai_client()
        self._init_anthropic_client()
        self._init_systemone_client()

    # Class default: tests subclass CustomProvider without running __init__.
    _systemone_client: Optional[AsyncOpenAI] = None

    # ==================== Initialization helpers ====================

    def _systemone_base_url(self) -> Optional[str]:
        """Where the System One API is served: systemone_base_url, else (rows
        written before the separate field) base_url. Never the Anthropic URL."""
        return (self.config.get("systemone_base_url") or self.config.get("base_url")
                or self.config.get("endpoint") or None)

    def _init_systemone_client(self) -> None:
        """A client for POST {systemone_base_url}/systemone when "systemone" is configured.

        Reuses the OpenAI client when it points at the same URL (same key and
        BYOK-guarded transport). Without any URL to serve it from the capability
        is dropped, so "systemone" in supported_apis always means a usable client.
        """
        self._systemone_client = None
        if SYSTEMONE_API not in self._supported_apis:
            return
        base = self._systemone_base_url()
        if not base:
            logger.warning(
                "Custom provider '%s': the System One API needs a System One base URL; not served",
                self.custom_provider_name,
            )
            self._supported_apis = [a for a in self._supported_apis if a != SYSTEMONE_API]
            return
        from urllib.parse import urlparse
        parsed = urlparse(base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                f"Invalid System One base URL for custom provider '{self.custom_provider_name}' "
                f"(must be http(s) with a host): {base!r}"
            )
        client = getattr(self, "client", None)
        if client is not None and base.rstrip("/") == str(self.base_url or "").rstrip("/"):
            self._systemone_client = client
            return
        from app.providers.url_guard import http_client_for
        self._systemone_client = AsyncOpenAI(
            base_url=base,
            api_key=self.api_key or _DEFAULT_API_KEY,
            # BYOK: no redirects, every request's destination re-checked (url_guard).
            http_client=http_client_for(self.config),
        )

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
        if self._listed_only_on(model_name) == SYSTEMONE_API:
            return "unsupported"  # a System One model is never sent to the Anthropic URL
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

    def _listed_only_on(self, model_name: str) -> Optional[str]:
        """"systemone" or "chat" when, with a separate System One URL, only that
        listing named the model; None when both or neither did, or before both
        listings have succeeded (e.g. after a restart), so a full model id always
        remains reachable."""
        if not self._systemone_url_is_separate():
            return None
        systemone_listed = _SYSTEMONE_META.get(self.full_provider_name)
        chat_listed = _CHAT_LISTED.get(self.full_provider_name)
        if systemone_listed is None or chat_listed is None:
            return None
        prefix = f"{self.full_provider_name}/"
        model_id = model_name if model_name.startswith(prefix) else prefix + model_name
        in_systemone, in_chat = model_id in systemone_listed, model_id in chat_listed
        if in_systemone != in_chat:
            return SYSTEMONE_API if in_systemone else "chat"
        return None

    def get_supported_apis_for_model(self, model_name: str) -> List[str]:
        """Per-model APIs: with a separate System One URL, each model is offered
        on the API whose listing named it (see _listed_only_on)."""
        apis = super().get_supported_apis_for_model(model_name)
        side = self._listed_only_on(model_name)
        if side == SYSTEMONE_API:
            return [api for api in apis if api == SYSTEMONE_API]
        if side == "chat":
            return [api for api in apis if api != SYSTEMONE_API]
        return apis

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
                "/v1/decisions",
            ])
        if ("anthropic" in self._supported_apis and getattr(self, "_anthropic_client", None)) or "openai" in self._supported_apis:
            endpoints.append("/v1/messages")
        if self._systemone_client is not None:
            endpoints.append("/v1/systemone")
        return endpoints

    def get_model_id(self, model_name: str) -> str:
        """Extract model ID from provider-prefixed model name (e.g. 'ollama/llama3' → 'llama3')."""
        if '/' in model_name:
            return model_name.split('/', 1)[1]
        return model_name

    def _systemone_url_is_separate(self) -> bool:
        return self._systemone_client is not None and self._systemone_client is not getattr(self, "client", None)

    async def get_available_models(self) -> List[ModelInfo]:
        """Discover models via the OpenAI and System One APIs, falling back to Anthropic.

        System One models come from {systemone_base_url}/models (TypeSafe or
        OpenAI shape). When it is the OpenAI URL too, that one GET serves both
        (a TypeSafe-shaped listing has no OpenAI ``data`` array). At a separate
        URL, the chat listing (OpenAI, else native Anthropic) and the System One
        listing are both read and merged, chat models first; which listing named
        a model then decides the API it is offered on (get_supported_apis_for_model).
        """
        if self._systemone_url_is_separate():
            chat_models = await self._fetch_openai_models()
            if chat_models is None:
                chat_models = await self._fetch_anthropic_models()
            systemone_models = await self._fetch_systemone_models()
            if chat_models is not None:
                _CHAT_LISTED[self.full_provider_name] = frozenset(m.id for m in chat_models)
            seen, models = set(), []
            for model in (chat_models or []) + (systemone_models or []):
                if model.id not in seen:
                    seen.add(model.id)
                    models.append(model)
            return models
        _CHAT_LISTED.pop(self.full_provider_name, None)
        if self._systemone_client is None:
            models = await self._fetch_openai_models()
        else:
            models = await self._fetch_systemone_models()
        if models is not None:
            return models

        models = await self._fetch_anthropic_models()
        if models is not None:
            return models

        return []

    # ==================== Model discovery helpers ====================

    async def _fetch_systemone_models(self) -> Optional[List[ModelInfo]]:
        """GET {systemone_base_url}/models (legacy rows: base_url) in TypeSafe or OpenAI shape; records metadata."""
        if self._systemone_client is None:
            return None
        try:
            listing = await self._systemone_client.get("/models", cast_to=object)
        except Exception as e:
            logger.warning(
                "Error fetching System One models from %s: %s",
                self.custom_provider_name, type(e).__name__,
            )
            return None
        items = _listed_models(listing)
        if items is None:
            logger.warning(
                "System One models from %s: unrecognized listing shape", self.custom_provider_name
            )
            return None
        if len(items) > _MAX_META_MODELS:
            logger.warning(
                "System One models from %s: keeping the first %d of %d",
                self.custom_provider_name, _MAX_META_MODELS, len(items),
            )
            items = items[:_MAX_META_MODELS]
        models, meta = [], {}
        for item in items:
            info = self.create_model_info(item["id"], self.custom_provider_name)
            description = item.get("description")
            meta[info.id] = (
                description[:_MAX_DESCRIPTION_CHARS] if isinstance(description, str) else None,
                _release_date(item),
            )
            models.append(info)
        _SYSTEMONE_META[self.full_provider_name] = meta  # one step: stale ids drop out
        return models

    # ==================== SYSTEM ONE API ====================

    async def systemone_create(
        self, request: Any, headers: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Relay POST /v1/systemone to {systemone_base_url}/systemone (legacy rows:
        base_url); see OpenAICompatibleProvider._relay_systemone."""
        client = self._systemone_client
        if client is None:
            raise NotImplementedError(
                "This provider is not configured for the System One API; "
                "set its System One base URL"
            )
        return await self._relay_systemone(client, "/systemone", request, headers)


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

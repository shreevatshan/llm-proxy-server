import time
import json
import logging
from contextvars import ContextVar
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Per-request selection of the upstream call style for AzureProvider inference methods.
#   "v1"         → AsyncOpenAI at {endpoint}/openai/v1/  (no api-version required)
#   "deployment" → legacy AsyncAzureOpenAI /openai/deployments/{dep}/...?api-version=
# Default "v1" — standard /v1/... and /openai/v1/... routes never set this.
azure_call_style: ContextVar[str] = ContextVar("azure_call_style", default="v1")

# The api-version forwarded from the inbound /openai/deployments/... request.
azure_api_version: ContextVar[Optional[str]] = ContextVar("azure_api_version", default=None)

from app.providers.openai_compatible import OpenAICompatibleProvider
from app.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ResponsesCreateRequest,
)
from app.providers.anthropic_adapter import (
    anthropic_adapter_messages,
    anthropic_adapter_messages_stream,
    prepare_anthropic_adapter_request,
)
from app.providers.anthropic_compatible import (
    # Azure Foundry native Anthropic streams intentionally share the same
    # bounded post-terminal drain policy as direct Anthropic-compatible
    # providers so cleanup semantics stay aligned.
    get_anthropic_post_terminal_drain_stop_reason,
    _translate_anthropic_sdk_error,
)
from app.anthropic_models import (
    ANTHROPIC_SDK_TIMEOUT_SECONDS,
    build_anthropic_sdk_kwargs,
    is_anthropic_terminal_stream_event,
)
from app.providers.base import AnthropicRequestMetadata, ProviderHTTPError
from app.conversion.anthropic_openai import _is_codex_model
from app.providers.azure_deployments import merge_azure_deployments

logger = logging.getLogger(__name__)

_AZURE_FOUNDRY_ANTHROPIC_BETA_SUPPORTED = {
    "fine-grained-tool-streaming-2025-05-14",
    "interleaved-thinking-2025-05-14",
    "context-management-2025-06-27",
}


class AzureProvider(OpenAICompatibleProvider):
    """
    Azure OpenAI / Azure AI Foundry provider implementation.

    Dynamic discovery uses GET {endpoint}/openai/models?api-version={discovery_api_version}
    which is the standard Azure OpenAI models-list endpoint (no Azure AD service principal
    required — only the resource API key is needed).

    Required Configuration Parameters:
    - endpoint: Azure OpenAI endpoint URL
    - api_key: Azure OpenAI API key

    Optional:
    - discovery_api_version: api-version string used for the models-list call
      (required when dynamic_discovery is True, e.g. "2024-10-21")
    - azure_backend: "openai" (default) or "foundry"
    - dynamic_discovery: True (default) to discover models at runtime; False to
      use the manually supplied deployment lists

    Configuration Example:
    ```yaml
    azure:
      - name: "production"
        enabled: true
        endpoint: "https://your-resource.openai.azure.com/"
        api_key: "your-api-key"
        discovery_api_version: "2024-10-21"
    ```
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', '').rstrip('/')
        self.api_key = config.get('api_key')
        self.discovery_api_version = config.get('discovery_api_version')
        self.azure_backend = (config.get('azure_backend') or 'openai').lower()
        self.deployments = config.get('deployments', [])
        self.openai_deployments = config.get('openai_deployments', [])
        self.anthropic_deployments = config.get('anthropic_deployments', [])
        self.dynamic_discovery = config.get('dynamic_discovery', True)
        self._anthropic_client = None

        # Validate required configuration parameters
        if not self.endpoint:
            raise ValueError("Azure OpenAI provider requires 'endpoint' to be configured")

        if not self.api_key:
            raise ValueError("Azure OpenAI provider requires 'api_key' to be configured")

        # Initialize the v1 API client (standard OpenAI SDK).
        # Azure's v1 API (GA Aug 2025) uses /openai/v1/ base path with standard OpenAI()
        # client.  This replaces the need for api-version and AzureOpenAI() for v1 callers.
        # See: https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle
        self._v1_client = AsyncOpenAI(
            base_url=f"{self.endpoint}/openai/v1/",
            api_key=self.api_key
        )
        # Backward-compat alias used by OpenAICompatibleProvider._get_responses_client()
        self._responses_client = self._v1_client

        # Cache of AsyncAzureOpenAI clients keyed by api-version string.
        # Populated lazily by _get_deployment_client() per inbound request api-version.
        self._deployment_clients: Dict[str, AsyncAzureOpenAI] = {}

        # All inference goes through the v1 client; deployment-style callers get a
        # per-request AsyncAzureOpenAI built from the inbound ?api-version= param.
        self.client = self._v1_client
        if self.azure_backend == "foundry":
            self._init_foundry_anthropic_client()

    def _init_foundry_anthropic_client(self) -> None:
        try:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(
                base_url=f"{self.endpoint}/anthropic",
                api_key=self.api_key,
                # Explicit timeout disables the SDK's client-side non-streaming
                # guard (anthropic/_base_client.py::_calculate_nonstreaming_timeout),
                # which otherwise rejects any non-streaming call whose max_tokens
                # *could* take >10 min — even when generation actually finishes
                # fast. Shares the same operator knob as Bedrock/direct SDK providers.
                timeout=ANTHROPIC_SDK_TIMEOUT_SECONDS,
            )
            logger.info("Anthropic SDK client initialized for Azure Foundry")
        except ImportError:
            logger.warning("anthropic SDK not installed — Anthropic native API unavailable for Azure Foundry")
        except Exception as e:
            logger.warning(f"Failed to init Anthropic client for Azure Foundry: {e}")

    def get_model_id(self, model_name: str) -> str:
        """Extract deployment name from provider-prefixed model name."""
        # Remove provider prefix (e.g., "azure:primary/my-deployment" -> "my-deployment")
        if '/' in model_name:
            deployment_name = model_name.split('/', 1)[1]
        else:
            deployment_name = model_name

        return deployment_name

    def _is_foundry_backend(self) -> bool:
        return self.azure_backend == "foundry"

    def _get_inference_client(self):
        """Return the upstream client to use for chat/completions/embeddings/audio/images.

        Reads the azure_call_style ContextVar set by the route handler:
          "deployment" → legacy AsyncAzureOpenAI using the deployment URL format
          "v1" (default) → AsyncOpenAI at /openai/v1/
        """
        if azure_call_style.get("v1") == "deployment":
            return self._get_deployment_client(azure_api_version.get(None))
        return self._v1_client

    def _get_deployment_client(self, api_version: Optional[str]) -> AsyncAzureOpenAI:
        """Return (or lazily build) an AsyncAzureOpenAI client for the given api-version.

        Raises ValueError if no usable api-version can be determined or if this
        is a Foundry backend (which has no legacy deployment URL surface).
        The deployment route handlers catch ValueError and return a 400 response.
        """
        eff = api_version
        if self._is_foundry_backend() or not eff:
            raise ValueError(
                "api-version is required for deployment-style calls to this Azure provider"
            )
        client = self._deployment_clients.get(eff)
        if client is None:
            client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=eff,
            )
            self._deployment_clients[eff] = client
        return client

    def _is_claude_model(self, model_name: str) -> bool:
        return "claude" in self.get_model_id(model_name).lower()

    def _is_anthropic_deployment(self, model_name: str) -> bool:
        """Decide whether this Foundry deployment serves the Anthropic API.

        Prefers the explicit configuration (anthropic_deployments /
        openai_deployments set when the provider was added). Falls back to a
        name-based heuristic only when the deployment is not listed in either
        bucket — typically for entries discovered dynamically via
        _fetch_deployments without classification config.
        """
        deployment = self.get_model_id(model_name)
        if deployment in (self.anthropic_deployments or []):
            return True
        if deployment in (self.openai_deployments or []):
            return False
        return self._is_claude_model(model_name)

    def _get_adapter_transport(self, model_name: str) -> str:
        if not self._is_foundry_backend() and _is_codex_model(model_name):
            return "responses"
        return "chat"

    def _get_manual_model_names(self) -> List[str]:
        if self._is_foundry_backend():
            return merge_azure_deployments(
                {
                    "openai": self.openai_deployments,
                    "anthropic": self.anthropic_deployments,
                },
                include_anthropic=True,
            )
        if self.openai_deployments:
            return list(self.openai_deployments)
        return list(self.deployments)

    def _strip_cache_control_scope(self, blocks: Optional[List[Any]]) -> bool:
        if not isinstance(blocks, list):
            return False
        removed = False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            cache_control = block.get("cache_control")
            if isinstance(cache_control, dict) and "scope" in cache_control:
                cache_control.pop("scope", None)
                removed = True
        return removed

    def _prepare_foundry_native_request(self, request, *, stream: bool) -> tuple[Dict[str, Any], List[str]]:
        """Build payload and compute dropped_fields for metadata reporting."""
        payload = request.model_dump(exclude_none=True)
        payload["model"] = self.get_model_id(request.model)
        payload["stream"] = stream
        dropped_fields: List[str] = []

        if self._strip_cache_control_scope(payload.get("system")):
            dropped_fields.append("system.cache_control.scope")

        for message in payload.get("messages", []) or []:
            if self._strip_cache_control_scope(message.get("content")):
                dropped_fields.append("messages.cache_control.scope")

        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    cache_control = tool.get("cache_control")
                    if isinstance(cache_control, dict) and "scope" in cache_control:
                        cache_control.pop("scope", None)
                        dropped_fields.append("tools.cache_control.scope")
                    input_schema = tool.get("input_schema")
                    if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
                        input_schema["type"] = "object"
                        dropped_fields.append("tools.input_schema.type")

        return payload, sorted(set(dropped_fields))

    def _apply_foundry_anthropic_fixups(self, kwargs: Dict[str, Any]) -> None:
        """Apply Azure Foundry-specific fixups to Anthropic SDK kwargs in-place.

        Strips cache_control.scope (unsupported on Foundry) and normalises
        input_schema.type 'custom' → 'object'.
        """
        self._strip_cache_control_scope(kwargs.get("system"))

        for message in kwargs.get("messages", []) or []:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    self._strip_cache_control_scope(content)

        tools = kwargs.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                cache_control = tool.get("cache_control")
                if isinstance(cache_control, dict) and "scope" in cache_control:
                    cache_control.pop("scope", None)
                input_schema = tool.get("input_schema")
                if isinstance(input_schema, dict) and input_schema.get("type") == "custom":
                    input_schema["type"] = "object"

    def _filter_foundry_anthropic_beta(self, anthropic_beta: Optional[str]) -> Optional[str]:
        """Return only the beta values supported by Azure Foundry, or None."""
        if not anthropic_beta:
            return None
        filtered = ",".join(
            v for v in (s.strip() for s in anthropic_beta.split(","))
            if v in _AZURE_FOUNDRY_ANTHROPIC_BETA_SUPPORTED
        )
        return filtered or None

    async def _fetch_deployments(self) -> List[str]:
        """Fetch available model names from Azure.

        When dynamic_discovery is False the manually configured deployment lists
        are returned as-is.  When enabled, both the ``openai`` and ``foundry``
        backends are queried via:

            GET {endpoint}/openai/models?api-version={discovery_api_version}

        This endpoint is the standard Azure OpenAI models-list API and returns
        all deployed models without requiring Azure AD service-principal
        credentials.
        """
        try:
            if not self.dynamic_discovery:
                manual_models = self._get_manual_model_names()
                if not manual_models:
                    raise ValueError(
                        "Azure provider requires manual model entries when dynamic_discovery is False"
                    )
                print(f"Azure provider using configured models: {manual_models}")
                return manual_models

            if not self.discovery_api_version:
                raise ValueError(
                    "discovery_api_version is required for Azure dynamic discovery"
                )

            url = f"{self.endpoint}/openai/models"
            params = {"api-version": self.discovery_api_version}
            headers = {
                "api-key": self.api_key,
            }

            import aiohttp  # already in requirements; local import keeps the module usable
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [
                            item["id"]
                            for item in data.get("data", [])
                            if item.get("id")
                        ]
                        print(
                            f"Azure provider ({self.azure_backend}) discovered "
                            f"{len(models)} models via models API: {models}"
                        )
                        return models
                    else:
                        body = await response.text()
                        raise Exception(
                            f"Azure models API returned HTTP {response.status}: {body}"
                        )

        except Exception as e:
            print(f"Azure deployment fetch error: {e}")
            raise Exception(f"Failed to fetch deployments: {e}")

    async def responses_input_tokens(self, request, **kwargs):
        """Azure OpenAI does not currently support the input_tokens endpoint."""
        raise NotImplementedError(
            "Azure OpenAI does not currently support POST /v1/responses/input_tokens. "
            "This feature is only available with direct OpenAI API providers."
        )

    async def responses_compact(self, request, **kwargs):
        """Azure OpenAI does not currently support the compact endpoint."""
        raise NotImplementedError(
            "Azure OpenAI does not currently support POST /v1/responses/compact. "
            "This feature is only available with direct OpenAI API providers."
        )

    async def get_available_models(self) -> List[ModelInfo]:
        """Get available models from Azure by fetching deployments or Foundry models dynamically."""
        try:
            deployments = await self._fetch_deployments()
            models = []

            # Create ModelInfo for each deployment
            for deployment_name in deployments:
                models.append(self.create_model_info(deployment_name, "azure"))

            return models

        except Exception as e:
            print(f"Error getting Azure models: {e}")
            return []

    def get_supported_apis(self) -> List[str]:
        """Azure providers support OpenAI, Azure OpenAI, and Anthropic API formats."""
        return ["openai", "azure_openai", "anthropic"]

    def get_supported_apis_for_model(self, model_name: str) -> List[str]:
        """On Foundry, enforce strict separation using the configured deployment
        buckets: Anthropic deployments are Anthropic-only; everything else is
        OpenAI / Azure OpenAI only. Non-Foundry defers to base behavior."""
        if not self._is_foundry_backend():
            return super().get_supported_apis_for_model(model_name)
        if self._is_anthropic_deployment(model_name):
            if self._is_chat_capable_model(model_name):
                return ["anthropic"]
            return []
        return ["openai", "azure_openai"]

    def get_supported_endpoints(self) -> List[str]:
        """Azure supports chat, completions, embeddings, images, audio, responses, and anthropic messages."""
        endpoints = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/images/generations",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
            "/v1/messages",
        ]
        # v1 API endpoints (always available when endpoint + api_key are configured)
        endpoints.extend([
            "/openai/v1/chat/completions",
            "/openai/v1/completions",
            "/openai/v1/embeddings",
            "/openai/v1/models",
            "/openai/v1/images/generations",
            "/openai/v1/audio/speech",
            "/openai/v1/audio/transcriptions",
        ])
        if not self._is_foundry_backend():
            endpoints.append("/openai/v1/responses")
        if not self._is_foundry_backend() and (self._responses_client is not None or self.client is not None):
            endpoints.append("/v1/responses")
        return endpoints

    def get_anthropic_mode_for_model(self, model_name: str) -> str:
        if not self._is_chat_capable_model(model_name):
            return "unsupported"
        if self._is_foundry_backend():
            return "native" if self._is_anthropic_deployment(model_name) else "unsupported"
        return "adapter"

    def get_anthropic_request_metadata(self, request, anthropic_beta=None) -> AnthropicRequestMetadata:
        mode = self.get_anthropic_mode_for_model(request.model)
        if mode == "unsupported":
            return AnthropicRequestMetadata(mode="unsupported", transport=None)
        if mode == "native":
            _, dropped_fields = self._prepare_foundry_native_request(
                request,
                stream=bool(getattr(request, "stream", False)),
            )
            if anthropic_beta:
                for v in (s.strip() for s in anthropic_beta.split(",")):
                    if v and v not in _AZURE_FOUNDRY_ANTHROPIC_BETA_SUPPORTED:
                        dropped_fields.append(f"anthropic-beta:{v}")
                dropped_fields = sorted(set(dropped_fields))
            return AnthropicRequestMetadata(
                mode="native",
                transport="messages",
                dropped_fields=dropped_fields,
            )

        transport = self._get_adapter_transport(request.model)
        _, metadata = prepare_anthropic_adapter_request(
            request,
            transport=transport,
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        )
        return metadata

    # ==================== Anthropic Messages API ====================

    async def anthropic_messages(self, request, anthropic_beta=None) -> dict:
        """Handle Anthropic Messages API using native SDK or adapter mode per model."""
        mode = self.get_anthropic_mode_for_model(request.model)
        if mode == "unsupported":
            raise NotImplementedError(
                f"Model {request.model} does not support the Anthropic Messages API on this provider"
            )
        if mode == "native":
            if not self._anthropic_client:
                raise NotImplementedError("Anthropic SDK not available for Azure Foundry native mode")

            model_id = self.get_model_id(request.model)
            kwargs = build_anthropic_sdk_kwargs(request, model_id)
            self._apply_foundry_anthropic_fixups(kwargs)

            filtered_beta = self._filter_foundry_anthropic_beta(anthropic_beta)
            if filtered_beta:
                kwargs["extra_headers"] = {
                    **kwargs.get("extra_headers", {}),
                    "anthropic-beta": filtered_beta,
                }

            try:
                response = await self._anthropic_client.messages.create(**kwargs)
            except Exception as e:
                raise _translate_anthropic_sdk_error(e, "azure-foundry") from e
            response_data = json.loads(response.model_dump_json(warnings="none"))
            response_data["model"] = request.model
            return response_data

        transport = self._get_adapter_transport(request.model)
        return await anthropic_adapter_messages(
            self,
            request,
            transport=transport,
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        )

    async def anthropic_messages_stream(self, request, anthropic_beta=None):
        """Handle streaming Anthropic Messages API using native SDK or adapter mode per model."""
        mode = self.get_anthropic_mode_for_model(request.model)
        if mode == "unsupported":
            raise NotImplementedError(
                f"Model {request.model} does not support the Anthropic Messages API on this provider"
            )
        if mode == "native":
            if not self._anthropic_client:
                raise NotImplementedError("Anthropic SDK not available for Azure Foundry native mode")

            model_id = self.get_model_id(request.model)
            kwargs = build_anthropic_sdk_kwargs(request, model_id)
            self._apply_foundry_anthropic_fixups(kwargs)

            filtered_beta = self._filter_foundry_anthropic_beta(anthropic_beta)
            if filtered_beta:
                kwargs["extra_headers"] = {
                    **kwargs.get("extra_headers", {}),
                    "anthropic-beta": filtered_beta,
                }

            try:
                async with self._anthropic_client.messages.stream(**kwargs) as stream:
                    terminal_event_type = None
                    terminal_seen_at: Optional[float] = None
                    drained_event_count = 0
                    async for event in stream:
                        if terminal_event_type is not None:
                            drained_event_count += 1
                            drain_stop_reason = get_anthropic_post_terminal_drain_stop_reason(
                                terminal_seen_at=terminal_seen_at,
                                drained_event_count=drained_event_count,
                            )
                            if drain_stop_reason is not None:
                                logger.warning(
                                    "Anthropic post-terminal drain budget reached for provider=azure-foundry model=%s terminal_event=%s drained_event_count=%s stop_reason=%s",
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
                                "Anthropic upstream stream reached terminal event for provider=azure-foundry model=%s event_type=%s",
                                request.model,
                                event_type,
                            )

                    if terminal_event_type is not None:
                        drain_ms = 0.0
                        if terminal_seen_at is not None:
                            drain_ms = (time.monotonic() - terminal_seen_at) * 1000
                        logger.info(
                            "Anthropic stream completed after terminal event for provider=azure-foundry model=%s event_type=%s drained_event_count=%s post_terminal_drain_ms=%.1f",
                            request.model,
                            terminal_event_type,
                            drained_event_count,
                            drain_ms,
                        )
            except ProviderHTTPError:
                raise
            except Exception as e:
                raise _translate_anthropic_sdk_error(e, "azure-foundry") from e

            return

        transport = self._get_adapter_transport(request.model)
        async for chunk in anthropic_adapter_messages_stream(
            self,
            request,
            transport=transport,
            allow_thinking=False,
            anthropic_beta=anthropic_beta,
        ):
            yield chunk

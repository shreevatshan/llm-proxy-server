import time
import json
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncAzureOpenAI, AsyncOpenAI
import aiohttp
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
    Azure OpenAI provider implementation with dynamic deployment discovery using Azure Management API.

    This provider uses the Azure Management API to automatically discover deployments based on:
    - Azure subscription ID
    - Resource group name
    - Cognitive Services account name
    - Azure AD service principal credentials

    Required Configuration Parameters:
    - endpoint: Azure OpenAI endpoint URL (for API calls)
    - api_key: Azure OpenAI API key (for API calls)
    - api_version: Azure OpenAI API version
    - subscription_id: Azure subscription ID
    - resource_group: Resource group containing the Cognitive Services resource
    - account_name: Name of the Cognitive Services resource
    - client_id: Azure AD service principal client ID
    - client_secret: Azure AD service principal client secret
    - tenant_id: Azure AD tenant ID

    Setup Instructions:

    1. Create an Azure AD App Registration:
       - Go to Azure Portal > Azure Active Directory > App registrations
       - Click "New registration"
       - Name: "LLM-Proxy-Server" (or your preferred name)
       - Account types: "Accounts in this organizational directory only"
       - Click "Register"

    2. Create a Client Secret:
       - In your app registration, go to "Certificates & secrets"
       - Click "New client secret"
       - Description: "LLM Proxy Server Secret"
       - Expires: Choose appropriate duration
       - Click "Add" and copy the secret value (client_secret)

    3. Note the Application Details:
       - Application (client) ID = client_id
       - Directory (tenant) ID = tenant_id

    4. Assign Permissions:
       - Go to your Cognitive Services resource in Azure Portal
       - Click "Access control (IAM)"
       - Click "Add role assignment"
       - Role: "Cognitive Services Contributor" or "Reader"
       - Assign access to: "User, group, or service principal"
       - Select your app registration
       - Click "Save"

    5. Configure the Provider:
       - subscription_id: Your Azure subscription ID
       - resource_group: Resource group containing your Cognitive Services resource
       - account_name: Name of your Cognitive Services resource
       - client_id: Application (client) ID from step 3
       - client_secret: Client secret value from step 2
       - tenant_id: Directory (tenant) ID from step 3

    Configuration Example:
    ```yaml
    azure:
      - name: "production"
        enabled: true
        endpoint: "https://your-resource.openai.azure.com/"
        api_key: "your-api-key"
        api_version: "2024-12-01-preview"
        subscription_id: "12345678-1234-1234-1234-123456789012"
        resource_group: "my-resource-group"
        account_name: "my-openai-resource"
        client_id: "87654321-4321-4321-4321-210987654321"
        client_secret: "your-client-secret"
        tenant_id: "11111111-2222-3333-4444-555555555555"
    ```
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', '').rstrip('/')
        self.api_key = config.get('api_key')
        self.api_version = config.get('api_version')
        self.azure_backend = (config.get('azure_backend') or 'openai').lower()
        self.subscription_id = config.get('subscription_id')
        self.resource_group = config.get('resource_group')
        self.account_name = config.get('account_name')
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        self.tenant_id = config.get('tenant_id')
        self.deployments = config.get('deployments', [])
        self.openai_deployments = config.get('openai_deployments', [])
        self.anthropic_deployments = config.get('anthropic_deployments', [])
        self.dynamic_discovery = config.get('dynamic_discovery', True)
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None
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

        # Initialize legacy Azure OpenAI client (for deployment-based endpoints).
        # api_version is optional — when absent, only v1 endpoints are available.
        if self.azure_backend == "foundry":
            self.client = self._v1_client
            self._responses_client = self._v1_client
            self._init_foundry_anthropic_client()
        elif self.api_version:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        else:
            # No api_version: fall back to v1 client for all SDK calls.
            # Legacy deployment-based routes may not work without api_version,
            # but v1 routes (/openai/v1/*) will work fine.
            self.client = self._v1_client

        # Warn if api_version is set but may be too old for Responses API
        if self.api_version and self.api_version < "2025-03-01-preview":
            print(
                f"Warning: Azure api_version '{self.api_version}' may not support the Responses API. "
                f"The Responses API requires the v1 endpoint which is used automatically, but "
                f"consider updating api_version to '2025-03-01-preview' or later for full compatibility."
            )

    def _init_foundry_anthropic_client(self) -> None:
        try:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(
                base_url=f"{self.endpoint}/anthropic",
                api_key=self.api_key,
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
        """Fetch available deployments from Azure OpenAI using configured method."""
        try:
            # If dynamic discovery is disabled, use the configured deployments list
            if not self.dynamic_discovery:
                manual_models = self._get_manual_model_names()
                if not manual_models:
                    raise ValueError("Azure provider requires manual model entries when dynamic_discovery is False")
                print(f"Azure provider using configured models: {manual_models}")
                return manual_models

            if self._is_foundry_backend():
                response = await self._v1_client.models.list()
                models = [m.id for m in response.data if m.id]
                print(f"Azure Foundry provider discovered {len(models)} models via SDK: {models}")
                return models

            # Use Azure Management API for dynamic discovery (always fetch fresh - no individual caching)
            if not all([self.subscription_id, self.resource_group, self.account_name]):
                raise ValueError("Azure provider requires subscription_id, resource_group, and account_name for dynamic deployment discovery")

            deployments = await self._fetch_deployments_via_management_api()
            return deployments

        except Exception as e:
            print(f"Azure deployment fetch error: {e}")
            raise Exception(f"Failed to fetch deployments: {e}")

    async def _fetch_deployments_via_management_api(self) -> List[str]:
        """Fetch deployments using Azure Management API."""
        try:
            # Check if we have the required Azure AD credentials
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                print("Azure Management API requires client_id, client_secret, and tenant_id - skipping")
                return []

            # Get Azure AD access token
            access_token = await self._get_azure_ad_token()
            if not access_token:
                print("Failed to acquire Azure AD token - skipping Management API")
                return []

            # Azure Management API endpoint for listing deployments
            management_api_url = (
                f"https://management.azure.com/subscriptions/{self.subscription_id}/"
                f"resourceGroups/{self.resource_group}/providers/Microsoft.CognitiveServices/"
                f"accounts/{self.account_name}/deployments"
            )

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            params = {"api-version": "2023-05-01"}

            async with aiohttp.ClientSession() as session:
                async with session.get(management_api_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        deployments = []

                        # Extract deployment names from the response
                        if "value" in data:
                            for deployment in data["value"]:
                                if "name" in deployment:
                                    deployments.append(deployment["name"])

                        print(f"Azure provider discovered {len(deployments)} deployments via Management API: {deployments}")
                        return deployments
                    else:
                        print(f"Azure Management API returned HTTP {response.status}")
                        response_text = await response.text()
                        print(f"Response: {response_text}")
                        return []

        except Exception as e:
            print(f"Error fetching deployments via Management API: {e}")
            return []

    async def _get_azure_ad_token(self) -> Optional[str]:
        """Get Azure AD access token using client credentials flow."""
        try:
            # Check if we have a valid cached token
            if (self._access_token and self._token_expires_at and
                time.time() < self._token_expires_at - 300):  # 5 minutes buffer
                return self._access_token

            # Azure AD token endpoint
            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

            # Prepare the request data for client credentials flow
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://management.azure.com/.default"
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(token_url, data=data, headers=headers) as response:
                    if response.status == 200:
                        token_data = await response.json()

                        # Extract access token and expiration
                        access_token = token_data.get("access_token")
                        expires_in = token_data.get("expires_in", 3600)  # Default to 1 hour

                        if access_token:
                            # Cache the token
                            self._access_token = access_token
                            self._token_expires_at = time.time() + expires_in

                            print("Successfully acquired Azure AD access token")
                            return access_token
                        else:
                            print("No access token in response")
                            return None
                    else:
                        print(f"Azure AD token request failed with status {response.status}")
                        response_text = await response.text()
                        print(f"Response: {response_text}")
                        return None

        except Exception as e:
            print(f"Error acquiring Azure AD token: {e}")
            return None

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

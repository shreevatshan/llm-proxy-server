import asyncio
import unittest
from types import SimpleNamespace

from app.openai_models import ModelInfo
from app.providers.azure_provider import AzureProvider
from app.providers.provider_manager import ProviderManager


class _ModelAwareProvider:
    full_provider_name = "azure:primary"

    def supports_api_for_model(self, model_name: str, api_name: str) -> bool:
        return api_name == "anthropic" and model_name.endswith("/chat-model")

    def get_supported_apis(self):
        return ["openai", "anthropic"]


class ProviderRoutingTests(unittest.TestCase):
    def test_create_provider_config_defaults_azure_backend_to_openai(self):
        manager = ProviderManager()
        cred = SimpleNamespace(
            provider_type="azure",
            instance_name="primary",
            enabled=True,
            provider_name="azure",
            endpoint="https://example.openai.azure.com",
            api_key="secret",
            discovery_api_version="2023-05-01",
            azure_backend=None,
            deployments_json='["gpt-4o"]',
            dynamic_discovery=False,
        )

        config = manager._create_provider_config(cred)

        self.assertEqual(config["azure_backend"], "openai")

    def test_create_provider_config_preserves_explicit_azure_backend(self):
        manager = ProviderManager()
        cred = SimpleNamespace(
            provider_type="azure",
            instance_name="primary",
            enabled=True,
            provider_name="azure",
            endpoint="https://example.services.ai.azure.com",
            api_key="secret",
            discovery_api_version="2023-05-01",
            azure_backend="foundry",
            deployments_json='["claude-3-7-sonnet"]',
            dynamic_discovery=False,
        )

        config = manager._create_provider_config(cred)

        self.assertEqual(config["azure_backend"], "foundry")

    def test_create_provider_config_parses_split_azure_deployments(self):
        manager = ProviderManager()
        cred = SimpleNamespace(
            provider_type="azure",
            instance_name="primary",
            enabled=True,
            provider_name="azure",
            endpoint="https://example.services.ai.azure.com",
            api_key="secret",
            discovery_api_version="2023-05-01",
            azure_backend="foundry",
            deployments_json='{"openai":["gpt-4.1"],"anthropic":["claude-3-7-sonnet"]}',
            dynamic_discovery=False,
        )

        config = manager._create_provider_config(cred)

        self.assertEqual(config["openai_deployments"], ["gpt-4.1"])
        self.assertEqual(config["anthropic_deployments"], ["claude-3-7-sonnet"])
        self.assertEqual(config["deployments"], ["gpt-4.1", "claude-3-7-sonnet"])

    def test_get_all_models_uses_model_aware_api_filtering(self):
        manager = ProviderManager()
        provider = _ModelAwareProvider()
        manager.providers = {"azure:primary": provider}
        manager.model_cache = SimpleNamespace(
            get_enabled_models=lambda: [
                ModelInfo(
                    id="azure:primary/chat-model",
                    created=1,
                    owned_by="azure:primary",
                    provider="azure:primary",
                ),
                ModelInfo(
                    id="azure:primary/text-embedding-3-large",
                    created=1,
                    owned_by="azure:primary",
                    provider="azure:primary",
                ),
            ]
        )

        models = asyncio.run(manager.get_all_models(api_filter="anthropic"))

        self.assertEqual([model.id for model in models], ["azure:primary/chat-model"])

    def test_get_anthropic_provider_for_model_respects_model_specific_support(self):
        manager = ProviderManager()
        provider = _ModelAwareProvider()
        manager.providers = {"azure:primary": provider}

        supported = asyncio.run(manager.get_anthropic_provider_for_model("azure:primary/chat-model"))
        unsupported = asyncio.run(manager.get_anthropic_provider_for_model("azure:primary/text-embedding-3-large"))

        self.assertIs(supported, provider)
        self.assertIsNone(unsupported)

    def test_azure_provider_manual_models_use_backend_specific_lists(self):
        provider = AzureProvider(
            {
                "name": "primary",
                "enabled": True,
                "endpoint": "https://example.services.ai.azure.com",
                "api_key": "secret",
                "discovery_api_version": "2023-05-01",
                "azure_backend": "foundry",
                "dynamic_discovery": False,
                "openai_deployments": ["gpt-4.1"],
                "anthropic_deployments": ["claude-sonnet-4-5"],
                "deployments": ["gpt-4.1", "claude-sonnet-4-5"],
            }
        )

        self.assertEqual(
            provider._get_manual_model_names(),
            ["gpt-4.1", "claude-sonnet-4-5"],
        )

    def test_foundry_anthropic_non_chat_model_is_not_advertised_under_openai(self):
        provider = AzureProvider(
            {
                "name": "primary",
                "enabled": True,
                "endpoint": "https://example.services.ai.azure.com",
                "api_key": "secret",
                "discovery_api_version": "2023-05-01",
                "azure_backend": "foundry",
                "dynamic_discovery": False,
                "openai_deployments": ["gpt-4.1"],
                "anthropic_deployments": ["text-embedding-3-large"],
                "deployments": ["gpt-4.1", "text-embedding-3-large"],
            }
        )

        self.assertEqual(
            provider.get_supported_apis_for_model("azure:primary/text-embedding-3-large"),
            [],
        )


if __name__ == "__main__":
    unittest.main()

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


class ModelNameParsingTests(unittest.TestCase):
    """_parse_model_name is the defensive net under the dispatch sites.

    Route handlers canonicalise the name first (app/model_resolution.py); this
    keeps the direct callers (audio.py) working on a prefix-less name too.
    """

    def _manager(self, model_ids=(), providers=()):
        manager = ProviderManager()
        manager.providers = {key: _ModelAwareProvider() for key in providers}
        manager.model_cache.update_models([
            ModelInfo(id=m, created=0, owned_by="t", provider="t") for m in model_ids
        ])
        return manager

    def test_bare_name_resolves_via_the_cache(self):
        manager = self._manager(["azure:primary/gpt-5.4"], ["azure:primary"])
        self.assertEqual(manager._parse_model_name("gpt-5.4"),
                         ("azure:primary", "gpt-5.4"))

    def test_live_prefix_is_honoured_without_consulting_the_cache(self):
        manager = self._manager([], ["azure:primary"])
        self.assertEqual(manager._parse_model_name("azure:primary/anything"),
                         ("azure:primary", "anything"))

    def test_dead_prefix_falls_back_to_a_provider_that_has_the_model(self):
        manager = self._manager(["azure:primary/gpt-5.4"], ["azure:primary"])
        self.assertEqual(manager._parse_model_name("azure:gone/gpt-5.4"),
                         ("azure:primary", "gpt-5.4"))

    def test_bare_name_containing_a_slash_is_tried_whole_first(self):
        manager = self._manager(["lmstudio:box/meta-llama/Llama-3.1-8B"], ["lmstudio:box"])
        self.assertEqual(
            manager._parse_model_name("meta-llama/Llama-3.1-8B"),
            ("lmstudio:box", "meta-llama/Llama-3.1-8B"),
        )

    def test_ambiguous_prefix_is_handed_on_for_disambiguation(self):
        """An explicit prefix must never be traded for some other provider.

        'azure' matches two instances, so the name cannot be resolved here --
        but falling back to the cache would route the request to whichever
        provider serves the bare name, which need not be an azure one at all.
        """
        manager = self._manager(
            ["azure:primary/gpt-5.4", "openai:main/gpt-5.4"],
            ["azure:primary", "azure:foundry", "openai:main"],
        )
        self.assertEqual(manager._parse_model_name("azure/gpt-5.4"), ("azure", "gpt-5.4"))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            manager._get_provider("azure")

    def test_unresolvable_names_raise(self):
        manager = self._manager(["azure:primary/gpt-5.4"], ["azure:primary"])
        for name in ("", "nope", "azure:gone/nope"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    manager._parse_model_name(name)


class FindProviderKeyTests(unittest.TestCase):
    """The non-raising counterpart to _get_provider's lookup."""

    def _manager(self, *keys):
        manager = ProviderManager()
        manager.providers = {key: _ModelAwareProvider() for key in keys}
        return manager

    def test_exact_key(self):
        manager = self._manager("azure:primary")
        self.assertEqual(manager.find_provider_key("azure:primary"), "azure:primary")

    def test_unique_bare_prefix(self):
        manager = self._manager("azure:primary")
        self.assertEqual(manager.find_provider_key("azure"), "azure:primary")

    def test_ambiguous_prefix_is_not_a_match(self):
        manager = self._manager("azure:primary", "azure:foundry")
        self.assertIsNone(manager.find_provider_key("azure"))

    def test_unknown_and_empty(self):
        manager = self._manager("azure:primary")
        self.assertIsNone(manager.find_provider_key("openai"))
        self.assertIsNone(manager.find_provider_key(""))
        self.assertIsNone(manager.find_provider_key(None))

    def test_has_provider_prefix_separates_ambiguous_from_unknown(self):
        manager = self._manager("azure:primary", "azure:foundry")
        self.assertTrue(manager.has_provider_prefix("azure:primary"))
        self.assertTrue(manager.has_provider_prefix("azure"))  # ambiguous, still ours
        self.assertFalse(manager.has_provider_prefix("openai"))
        self.assertFalse(manager.has_provider_prefix("azure:gone"))
        self.assertFalse(manager.has_provider_prefix(""))
        self.assertFalse(manager.has_provider_prefix(None))


if __name__ == "__main__":
    unittest.main()

"""Azure deployment configuration must keep legacy semantics on round trips.

The per-bucket lists that predate deployment groups classify deployments
(OpenAI / Responses-only / Anthropic) but never selected APIs. Reading them
back as groups must not turn that classification into an API allowlist, or an
unchanged save of an old provider starts refusing its embeddings, image and
audio deployments. Explicit API selections keep restricting as before.
"""

import json
import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from app.auth.models import AzureDeploymentGroup, ProviderCredentialsCreate, ProviderCredentialsUpdate
from app.providers.azure_deployments import (
    API_CHAT,
    API_COMPLETIONS,
    API_EMBEDDINGS,
    API_IMAGES,
    API_RESPONSES,
    API_SPEECH,
    API_TRANSCRIPTION,
    build_azure_config_fields,
    deployment_apis,
    normalize_azure_deployments,
    normalize_deployment_groups,
    serialize_azure_deployments,
)
from app.providers.azure_provider import AzureProvider

BACKENDS = ("openai", "foundry")
NON_CHAT_APIS = (API_EMBEDDINGS, API_IMAGES, API_SPEECH, API_TRANSCRIPTION, API_COMPLETIONS)

LEGACY_DICT = {
    "openai": ["gpt-4o", "codex-mini", "text-embedding-3-large"],
    "anthropic": ["claude-sonnet-4"],
    "responses": ["codex-mini", "not-listed"],
}
LEGACY_DICT_ROWS = [
    {"deployments": ["gpt-4o", "text-embedding-3-large"], "apis": [], "legacy_mode": "openai"},
    {"deployments": ["codex-mini"], "apis": [], "legacy_mode": "responses"},
    {"deployments": ["claude-sonnet-4"], "apis": [], "legacy_mode": "anthropic"},
]


def _provider(deployments_json, backend="openai"):
    creds = SimpleNamespace(
        endpoint="https://example.openai.azure.com",
        api_key="k",
        discovery_api_version=None,
        azure_backend=backend,
        dynamic_discovery=False,
        deployments_json=deployments_json if deployments_json is None or isinstance(deployments_json, str)
        else json.dumps(deployments_json),
    )
    return AzureProvider({"name": "test", **build_azure_config_fields(creds)})


class LegacyNormalizationTests(unittest.TestCase):
    def test_legacy_formats_become_marked_rows(self):
        cases = {
            "array": (["a", "b"], [{"deployments": ["a", "b"], "apis": [], "legacy_mode": "openai"}]),
            "deployments key": (
                {"deployments": ["old-embedding"]},
                [{"deployments": ["old-embedding"], "apis": [], "legacy_mode": "openai"}],
            ),
            "buckets": (LEGACY_DICT, LEGACY_DICT_ROWS),
        }
        for label, (raw, rows) in cases.items():
            with self.subTest(label):
                groups = normalize_azure_deployments(json.dumps(raw))
                self.assertEqual(groups["groups"], rows)
                self.assertEqual(deployment_apis(groups["groups"]), {})

    def test_responses_hint_marks_without_adding(self):
        groups = normalize_azure_deployments(LEGACY_DICT)
        self.assertEqual(groups["responses"], ["codex-mini"])
        self.assertEqual(groups["openai"], ["gpt-4o", "codex-mini", "text-embedding-3-large"])
        self.assertNotIn("not-listed", groups["openai"] + groups["anthropic"])

    def test_legacy_rows_survive_serialization(self):
        rows = normalize_azure_deployments(LEGACY_DICT)["groups"]
        stored = serialize_azure_deployments(deployment_groups=rows)
        again = normalize_azure_deployments(stored)
        self.assertEqual(again["groups"], LEGACY_DICT_ROWS)
        self.assertEqual(again["responses"], ["codex-mini"])
        self.assertEqual(again["anthropic"], ["claude-sonnet-4"])
        # Older readers still find the derived lists.
        self.assertEqual(json.loads(stored)["responses"], ["codex-mini"])

    def test_explicit_groups_restrict_and_union(self):
        rows = normalize_deployment_groups([
            {"deployments": ["gpt-4o"], "apis": ["chat"]},
            {"deployments": ["gpt-4o", "emb"], "apis": ["embeddings"]},
        ], strict=True)
        self.assertEqual(deployment_apis(rows), {"gpt-4o": ["chat", "embeddings"], "emb": ["embeddings"]})

    def test_mixed_legacy_and_explicit(self):
        rows = normalize_deployment_groups([
            {"deployments": ["old"], "apis": [], "legacy_mode": "openai"},
            {"deployments": ["new"], "apis": ["chat"]},
        ], strict=True)
        self.assertEqual(deployment_apis(rows), {"new": ["chat"]})
        self.assertEqual(normalize_azure_deployments({"groups": rows})["openai"], ["old", "new"])

    def test_legacy_buckets_may_overlap(self):
        rows = normalize_deployment_groups([
            {"deployments": ["dual"], "apis": [], "legacy_mode": "openai"},
            {"deployments": ["dual"], "apis": [], "legacy_mode": "anthropic"},
        ], strict=True)
        groups = normalize_azure_deployments({"groups": rows})
        self.assertEqual((groups["openai"], groups["anthropic"]), (["dual"], ["dual"]))

    def test_strict_rejections(self):
        cases = {
            "marker with apis": [{"deployments": ["x"], "apis": ["chat"], "legacy_mode": "openai"}],
            "neither": [{"deployments": ["x"], "apis": []}],
            "unknown mode": [{"deployments": ["x"], "apis": [], "legacy_mode": "embeddings"}],
            "legacy and explicit": [
                {"deployments": ["x"], "apis": [], "legacy_mode": "openai"},
                {"deployments": ["x"], "apis": ["chat"]},
            ],
        }
        for label, raw in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    normalize_deployment_groups(raw, strict=True)
        with self.assertRaises(ValidationError):
            AzureDeploymentGroup(deployments=["x"], apis=["chat"], legacy_mode="openai")
        with self.assertRaises(ValidationError):
            AzureDeploymentGroup(deployments=["x"], apis=[])
        AzureDeploymentGroup(deployments=["x"], apis=[], legacy_mode="responses")

    def test_lenient_read_prefers_selected_apis_over_a_stale_marker(self):
        rows = normalize_deployment_groups([{"deployments": ["x"], "apis": ["chat"], "legacy_mode": "openai"}])
        self.assertEqual(rows, [{"deployments": ["x"], "apis": ["chat"]}])


class LegacyProviderBehaviorTests(unittest.TestCase):
    def test_legacy_deployments_serve_every_api(self):
        for backend in BACKENDS:
            for raw in (["text-embedding-3-large"], {"deployments": ["text-embedding-3-large"]}, LEGACY_DICT):
                with self.subTest(backend=backend, raw=raw):
                    provider = _provider(raw, backend)
                    self.assertEqual(provider.deployment_apis, {})
                    for api in (API_CHAT, API_RESPONSES, *NON_CHAT_APIS):
                        self.assertIsNone(provider._deployment_api_error("text-embedding-3-large", api))
                    # Name fallback still keeps the embedding model off /v1/messages.
                    self.assertFalse(provider._is_chat_capable_model("text-embedding-3-large"))
                    self.assertEqual(provider.get_anthropic_mode_for_model("text-embedding-3-large"), "unsupported")

    def test_legacy_transport_hints(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                provider = _provider({"openai": ["gpt-4o", "resp-only", "gpt-5-codex"], "responses": ["resp-only"]}, backend)
                self.assertEqual(provider._get_adapter_transport("azure:test/gpt-4o"), "chat")
                self.assertEqual(provider._get_adapter_transport("azure:test/resp-only"), "responses")
                self.assertEqual(provider._get_adapter_transport("azure:test/gpt-5-codex"), "responses")
                self.assertTrue(provider._is_chat_capable_model("gpt-4o"))

    def test_foundry_legacy_anthropic_classification(self):
        provider = _provider(LEGACY_DICT, "foundry")
        self.assertEqual(provider.get_anthropic_mode_for_model("claude-sonnet-4"), "native")
        self.assertEqual(provider.get_anthropic_mode_for_model("gpt-4o"), "adapter")

    def test_explicit_groups_still_restrict(self):
        rows = [
            {"deployments": ["gpt-4o"], "apis": ["chat"]},
            {"deployments": ["gpt-5-codex"], "apis": ["chat", "responses"]},
            {"deployments": ["resp"], "apis": ["responses"]},
            {"deployments": ["legacy-emb"], "apis": [], "legacy_mode": "openai"},
        ]
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                provider = _provider(serialize_azure_deployments(deployment_groups=rows), backend)
                self.assertIsNone(provider._deployment_api_error("gpt-4o", API_CHAT))
                self.assertIsNotNone(provider._deployment_api_error("gpt-4o", API_EMBEDDINGS))
                self.assertIsNone(provider._deployment_api_error("legacy-emb", API_EMBEDDINGS))
                # Explicit selection wins over the Codex name fallback.
                self.assertEqual(provider._get_adapter_transport("gpt-5-codex"), "chat")
                self.assertEqual(provider._get_adapter_transport("resp"), "responses")


class RequestModelPrecedenceTests(unittest.TestCase):
    STALE = {"deployments": ["stale"], "openai_deployments": ["stale"], "anthropic_deployments": ["stale"]}

    def test_empty_groups_clear_every_compatibility_list(self):
        for model, extra in (
            (ProviderCredentialsUpdate, {}),
            (ProviderCredentialsCreate, {"provider_type": "azure", "instance_name": "i", "provider_name": "azure"}),
        ):
            with self.subTest(model=model.__name__):
                data = model.model_validate({
                    **extra, **self.STALE,
                    "deployment_groups": [],
                    "dynamic_discovery": True,
                    "discovery_api_version": "2024-10-21",
                })
                self.assertEqual((data.deployments, data.openai_deployments, data.anthropic_deployments), ([], [], []))
                self.assertEqual(data.deployment_groups, [])

    def test_groups_replace_stale_lists(self):
        data = ProviderCredentialsUpdate.model_validate({
            **self.STALE, "deployment_groups": [{"deployments": ["new"], "apis": ["anthropic"]}],
        })
        self.assertEqual(data.deployments, ["new"])
        self.assertEqual((data.openai_deployments, data.anthropic_deployments), ([], ["new"]))

    def test_invalid_stale_list_is_ignored_when_groups_are_given(self):
        data = ProviderCredentialsUpdate.model_validate({
            "openai_deployments": ["bad name!"], "deployment_groups": [],
        })
        self.assertEqual(data.openai_deployments, [])

    def test_lists_alone_keep_legacy_behavior(self):
        data = ProviderCredentialsUpdate.model_validate({"openai_deployments": ["a"]})
        self.assertIsNone(data.deployment_groups)
        self.assertEqual(data.openai_deployments, ["a"])
        self.assertNotIn("deployment_groups", data.model_dump(exclude_unset=True))


if __name__ == "__main__":
    unittest.main()

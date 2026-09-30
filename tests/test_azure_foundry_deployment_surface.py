"""Serving the Azure OpenAI deployment surface from a Foundry-backed provider.

A Foundry resource does not host the dated deployment data plane
(/openai/deployments/{dep}/x?api-version=) at all; its only OpenAI-shaped surface
is /openai/v1/.  So an inbound deployment-style request is *translated* onto the
v1 client rather than forwarded, and the inbound api-version is accepted and
ignored.  Claude deployments are the exception: they live behind
{endpoint}/anthropic with no OpenAI-shaped upstream, so they are refused.

The provider-level tests pin the transport seam (_get_inference_client); the
route-level tests pin the guard, including the regressions for the classic
`openai` backend, which still requires api-version.
"""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from openai import AsyncAzureOpenAI

from app.auth.admin import AdminUser
from app.main import create_azure_openai_app
from app.openai_models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    Usage,
)
from app.providers.azure_provider import (
    AzureProvider,
    azure_api_version,
    azure_call_style,
)
from app.providers.provider_manager import provider_manager
from app.routes import azure_openai

FOUNDRY_KEY = "azure:foundry"
CLASSIC_KEY = "azure:classic"
OPENAI_DEPLOYMENT = "grok-3"
CLAUDE_DEPLOYMENT = "claude-sonnet-4"


def _provider(backend):
    return AzureProvider({
        "name": backend,
        "endpoint": "https://example.openai.azure.com/",
        "api_key": "k",
        "azure_backend": backend,
        "dynamic_discovery": False,
        "openai_deployments": [OPENAI_DEPLOYMENT],
        "anthropic_deployments": [CLAUDE_DEPLOYMENT],
        "deployments": [OPENAI_DEPLOYMENT, CLAUDE_DEPLOYMENT],
    })


class _CallStyle:
    """Set the deployment-style ContextVars, then restore them."""

    def __init__(self, style, api_version=None):
        self._style = style
        self._api_version = api_version

    def __enter__(self):
        self._tokens = (
            azure_call_style.set(self._style),
            azure_api_version.set(self._api_version),
        )
        return self

    def __exit__(self, *exc):
        azure_call_style.reset(self._tokens[0])
        azure_api_version.reset(self._tokens[1])


class InferenceClientTransportTests(unittest.TestCase):
    """_get_inference_client is the single seam the translation hangs off."""

    def test_foundry_ignores_deployment_call_style(self):
        p = _provider("foundry")
        with _CallStyle("deployment", "2024-10-21"):
            self.assertIs(p._get_inference_client(), p._v1_client)

    def test_foundry_ignores_deployment_call_style_without_api_version(self):
        p = _provider("foundry")
        with _CallStyle("deployment", None):
            self.assertIs(p._get_inference_client(), p._v1_client)

    def test_foundry_v1_call_style_is_unchanged(self):
        p = _provider("foundry")
        with _CallStyle("v1"):
            self.assertIs(p._get_inference_client(), p._v1_client)

    def test_classic_backend_still_uses_the_deployment_client(self):
        p = _provider("openai")
        with _CallStyle("deployment", "2024-10-21"):
            client = p._get_inference_client()
        self.assertIsInstance(client, AsyncAzureOpenAI)
        self.assertIsNot(client, p._v1_client)

    def test_classic_backend_still_requires_an_api_version(self):
        p = _provider("openai")
        with _CallStyle("deployment", None):
            with self.assertRaises(ValueError) as ctx:
                p._get_inference_client()
        self.assertIn("api-version is required", str(ctx.exception))

    def test_direct_deployment_client_call_on_foundry_is_still_guarded(self):
        """_get_inference_client no longer routes here; the raise guards callers."""
        p = _provider("foundry")
        with self.assertRaises(ValueError) as ctx:
            p._get_deployment_client("2024-10-21")
        self.assertIn("no deployment-style upstream", str(ctx.exception))


class FoundryResponsesCapabilityTests(unittest.TestCase):
    """Foundry serves /openai/v1/responses; the advertised contract must say so."""

    def test_foundry_advertises_the_responses_endpoints(self):
        endpoints = _provider("foundry").get_supported_endpoints()
        self.assertIn("/openai/v1/responses", endpoints)
        self.assertIn("/v1/responses", endpoints)

    def test_classic_backend_still_advertises_them(self):
        endpoints = _provider("openai").get_supported_endpoints()
        self.assertIn("/openai/v1/responses", endpoints)
        self.assertIn("/v1/responses", endpoints)


async def _fake_chat_completion(request):
    return ChatCompletionResponse(
        id="chatcmpl-1",
        created=0,
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="hi"),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


async def _fake_embeddings(request):
    return EmbeddingResponse(
        data=[EmbeddingData(index=0, embedding=[0.0])],
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
    )


class _Registry:
    """Swap the process-wide provider registry for real AzureProvider instances.

    _get_azure_provider does an isinstance(provider, AzureProvider) check, so a
    duck-typed stub would yield a 400 from the resolver rather than reaching the
    handler under test.
    """

    def __enter__(self):
        self.foundry = _provider("foundry")
        self.classic = _provider("openai")
        self._saved = provider_manager.providers
        provider_manager.providers = {
            FOUNDRY_KEY: self.foundry,
            CLASSIC_KEY: self.classic,
        }
        return self

    def __exit__(self, *exc):
        provider_manager.providers = self._saved


class DeploymentRouteGuardTests(unittest.TestCase):
    """The seven deployment routes; chat / embeddings / speech cover all shapes."""

    def setUp(self):
        self.app = create_azure_openai_app()
        self.app.dependency_overrides[azure_openai._authenticate_azure] = \
            lambda: AdminUser(username="admin", email="admin@example.com")

    def _call(self, path, patches=(), **post_kw):
        """POST *path*, with tracking and the provider's upstream call stubbed."""
        with _Registry() as registry, \
             patch("app.request_tracker.request_tracker.start_request",
                   new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request",
                   new_callable=AsyncMock):
            for name, impl in patches:
                setattr(registry.foundry, name, impl)
                setattr(registry.classic, name, impl)
            client = TestClient(self.app)
            return client.post(path, **post_kw)

    def _url(self, provider_key, deployment, suffix, api_version=None):
        url = f"/openai/deployments/{provider_key}/{deployment}/{suffix}"
        if api_version:
            url += f"?api-version={api_version}"
        return url

    # ---- chat/completions: the JSON + streaming shape -------------------

    def _chat_body(self):
        # `model` is required by the Pydantic body even on the deployment
        # surface, where the URL is authoritative; the handler overwrites it.
        return {"model": "ignored", "messages": [{"role": "user", "content": "hi"}],
                "stream": False}

    def test_foundry_openai_deployment_without_api_version_reaches_the_handler(self):
        response = self._call(
            self._url(FOUNDRY_KEY, OPENAI_DEPLOYMENT, "chat/completions"),
            patches=[("chat_completion", _fake_chat_completion)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], f"{FOUNDRY_KEY}/{OPENAI_DEPLOYMENT}")

    def test_foundry_openai_deployment_accepts_and_ignores_api_version(self):
        response = self._call(
            self._url(FOUNDRY_KEY, OPENAI_DEPLOYMENT, "chat/completions", "2024-10-21"),
            patches=[("chat_completion", _fake_chat_completion)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 200)

    def test_foundry_claude_deployment_is_refused(self):
        response = self._call(
            self._url(FOUNDRY_KEY, CLAUDE_DEPLOYMENT, "chat/completions", "2024-10-21"),
            patches=[("chat_completion", _fake_chat_completion)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 400)
        message = response.json()["error"]["message"]
        self.assertIn("/v1/messages", message)

    def test_classic_backend_still_requires_api_version(self):
        response = self._call(
            self._url(CLASSIC_KEY, OPENAI_DEPLOYMENT, "chat/completions"),
            patches=[("chat_completion", _fake_chat_completion)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("api-version is required",
                      response.json()["error"]["message"])

    def test_classic_backend_claude_deployment_is_not_refused(self):
        """The Claude exclusion is Foundry-specific; classic Azure adapts them."""
        response = self._call(
            self._url(CLASSIC_KEY, CLAUDE_DEPLOYMENT, "chat/completions", "2024-10-21"),
            patches=[("chat_completion", _fake_chat_completion)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 200)

    def test_foundry_request_runs_with_the_deployment_call_style_set(self):
        """The ContextVars are still set; the provider is what ignores them.

        TestClient runs the app in a worker thread, so the ContextVars are
        invisible to this thread -- read them inside the patched call.
        """
        seen = {}

        async def _record(request):
            seen["style"] = azure_call_style.get()
            seen["api_version"] = azure_api_version.get()
            return await _fake_chat_completion(request)

        response = self._call(
            self._url(FOUNDRY_KEY, OPENAI_DEPLOYMENT, "chat/completions", "2024-10-21"),
            patches=[("chat_completion", _record)],
            json=self._chat_body(),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, {"style": "deployment", "api_version": "2024-10-21"})

    # ---- embeddings: JSON, but no ValueError handler --------------------

    def test_foundry_embeddings_without_api_version_reaches_the_handler(self):
        response = self._call(
            self._url(FOUNDRY_KEY, OPENAI_DEPLOYMENT, "embeddings"),
            patches=[("embeddings", _fake_embeddings)],
            json={"model": "ignored", "input": "hi"},
        )
        self.assertEqual(response.status_code, 200)

    def test_foundry_embeddings_claude_deployment_is_refused_as_400_not_500(self):
        """The guard returns a response; raising would be a generic 500 here."""
        response = self._call(
            self._url(FOUNDRY_KEY, CLAUDE_DEPLOYMENT, "embeddings"),
            patches=[("embeddings", _fake_embeddings)],
            json={"model": "ignored", "input": "hi"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("/v1/messages", response.json()["error"]["message"])

    def test_classic_embeddings_still_requires_api_version(self):
        response = self._call(
            self._url(CLASSIC_KEY, OPENAI_DEPLOYMENT, "embeddings"),
            patches=[("embeddings", _fake_embeddings)],
            json={"model": "ignored", "input": "hi"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("api-version is required",
                      response.json()["error"]["message"])

    # ---- audio/speech: the multipart-adjacent shape ---------------------

    async def _fake_speech(self, request):  # pragma: no cover - replaced per test
        raise NotImplementedError

    def test_foundry_audio_speech_without_api_version_reaches_the_handler(self):
        async def _speech(request):
            return b"audio"

        response = self._call(
            self._url(FOUNDRY_KEY, OPENAI_DEPLOYMENT, "audio/speech"),
            patches=[("audio_speech", _speech)],
            json={"model": "ignored", "input": "hi", "voice": "alloy"},
        )
        self.assertEqual(response.status_code, 200)

    def test_foundry_audio_speech_claude_deployment_is_refused(self):
        async def _speech(request):
            return b"audio"

        response = self._call(
            self._url(FOUNDRY_KEY, CLAUDE_DEPLOYMENT, "audio/speech"),
            patches=[("audio_speech", _speech)],
            json={"model": "ignored", "input": "hi", "voice": "alloy"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("/v1/messages", response.json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()

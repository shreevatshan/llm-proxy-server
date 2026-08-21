"""End-to-end prefix-less model naming, through the real sub-apps.

The unit tests in test_model_resolution.py cover the resolver itself; these drive
a real request all the way through routing, the per-app exception handlers, and
the response echo -- the wiring the unit tests cannot see. Only the upstream
provider call is stubbed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.auth.middleware import authenticate_jwt_or_api_key, authenticate_anthropic_request
from app.cache import ModelCache
from app.main import create_anthropic_app, create_azure_openai_app, create_openai_app
from app.model_alias import echo_model_name
from app.openai_models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    Usage,
)
from app.providers.provider_manager import provider_manager


class _Caller:
    """A non-admin principal with no per-user restrictions."""

    id = 1
    user_id = 1
    username = "alice"


class _StubProvider:
    full_provider_name = "azure:foundry"

    def supports_api_for_model(self, model_name, api_name):
        return True


def _cache(*model_ids):
    cache = ModelCache()
    cache.update_models([
        ModelInfo(id=m, created=0, owned_by="t", provider="t") for m in model_ids
    ])
    return cache


class _Inventory:
    """Swap the process-wide provider registry and model cache."""

    def __init__(self, model_ids, provider_keys):
        self.cache = _cache(*model_ids)
        self.providers = {k: _StubProvider() for k in provider_keys}

    def __enter__(self):
        self._saved = (provider_manager.model_cache, provider_manager.providers)
        provider_manager.model_cache = self.cache
        provider_manager.providers = self.providers
        return self

    def __exit__(self, *exc):
        provider_manager.model_cache, provider_manager.providers = self._saved


async def _fake_chat_completion(request):
    """Stand in for a provider: echo the client's name, as every real one does."""
    return ChatCompletionResponse(
        id="chatcmpl-1",
        created=0,
        model=echo_model_name(request),
        choices=[ChatCompletionChoice(
            index=0, message=ChatMessage(role="assistant", content="hi"),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class PrefixlessOpenAIEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.app = create_openai_app()
        self.app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()

    def _post(self, body, model_ids, provider_keys=("azure:foundry",)):
        with _Inventory(model_ids, provider_keys), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock), \
             patch.object(provider_manager, "chat_completion", _fake_chat_completion), \
             patch.object(provider_manager, "get_provider_for_model", lambda m: _StubProvider()):
            client = TestClient(self.app)
            response = client.post("/v1/chat/completions", json=body)
        return response

    def _body(self, model):
        return {"model": model, "messages": [{"role": "user", "content": "hi"}],
                "stream": False}

    def test_bare_name_is_accepted_and_echoed_back(self):
        response = self._post(self._body("gpt-5.4"), ["azure:foundry/gpt-5.4"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "gpt-5.4")

    def test_canonical_id_is_unchanged(self):
        response = self._post(self._body("azure:foundry/gpt-5.4"), ["azure:foundry/gpt-5.4"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "azure:foundry/gpt-5.4")

    def test_dead_provider_prefix_is_served_by_a_live_one(self):
        response = self._post(self._body("azure:gone/gpt-5.4"), ["azure:foundry/gpt-5.4"])
        self.assertEqual(response.status_code, 200)
        # The client still sees the name it sent.
        self.assertEqual(response.json()["model"], "azure:gone/gpt-5.4")

    def test_unknown_model_is_404_model_not_found(self):
        response = self._post(self._body("gpt-9-nonexistent"), ["azure:foundry/gpt-5.4"])
        self.assertEqual(response.status_code, 404)
        error = response.json()["error"]
        self.assertEqual(error["code"], "model_not_found")
        self.assertEqual(error["type"], "invalid_request_error")

    def test_a_globally_disabled_model_is_not_reachable_by_bare_name(self):
        with _Inventory(["azure:foundry/gpt-5.4"], ["azure:foundry"]) as inv, \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
            inv.cache._model_configs = {"azure:foundry/gpt-5.4": False}
            client = TestClient(self.app)
            response = client.post("/v1/chat/completions", json=self._body("gpt-5.4"))
        self.assertEqual(response.status_code, 404)


class RoundRobinAttributionTests(unittest.TestCase):
    """Usage must be attributed to the canonical id the request actually used.

    Tracking starts in middleware, before resolution, so the entry begins life
    holding the bare name; the route retags it. Without that, one logical model
    would split across a bare usage row and a prefixed one.
    """

    def setUp(self):
        self.app = create_openai_app()
        self.app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()

    def test_repeated_bare_requests_alternate_and_retag(self):
        pool = ["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]
        retagged = []

        async def _record(request_id, model):
            retagged.append(model)

        with _Inventory(pool, ["azure:foundry", "openai:main"]), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.update_model", _record), \
             patch.object(provider_manager, "chat_completion", _fake_chat_completion), \
             patch.object(provider_manager, "get_provider_for_model", lambda m: _StubProvider()):
            client = TestClient(self.app)
            body = {"model": "gpt-5.4", "stream": False,
                    "messages": [{"role": "user", "content": "hi"}]}
            responses = [client.post("/v1/chat/completions", json=body) for _ in range(4)]

        self.assertTrue(all(r.status_code == 200 for r in responses))
        # Every request was retagged to a canonical id, and the two providers
        # alternate rather than one absorbing the whole load.
        self.assertEqual(set(retagged), set(pool))
        self.assertEqual(len(retagged), 4)
        self.assertNotEqual(retagged[0], retagged[1])

    def test_a_user_denied_one_provider_is_routed_to_the_other(self):
        pool = ["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]
        with _Inventory(pool, ["azure:foundry", "openai:main"]) as inv, \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock), \
             patch.object(provider_manager, "chat_completion", _fake_chat_completion), \
             patch.object(provider_manager, "get_provider_for_model", lambda m: _StubProvider()):
            inv.cache._user_model_policies = {1: "custom"}
            inv.cache._user_model_exceptions = {1: {"azure:foundry/gpt-5.4": False}}
            client = TestClient(self.app)
            body = {"model": "gpt-5.4", "stream": False,
                    "messages": [{"role": "user", "content": "hi"}]}
            responses = [client.post("/v1/chat/completions", json=body) for _ in range(3)]
        self.assertEqual([r.status_code for r in responses], [200, 200, 200])

    def test_a_user_denied_every_provider_gets_403_not_404(self):
        pool = ["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]
        with _Inventory(pool, ["azure:foundry", "openai:main"]) as inv, \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
            inv.cache._user_model_policies = {1: "deny"}
            client = TestClient(self.app)
            response = client.post("/v1/chat/completions", json={
                "model": "gpt-5.4", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "model_access_denied")


class PrefixlessAnthropicEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.app = create_anthropic_app()
        self.app.dependency_overrides[authenticate_anthropic_request] = lambda: _Caller()

    def test_unknown_model_uses_the_anthropic_error_envelope(self):
        with _Inventory([], ["azure:foundry"]), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
            client = TestClient(self.app)
            response = client.post("/v1/messages", json={
                "model": "gpt-9-nonexistent",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            })
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "not_found_error")

    def test_count_tokens_degrades_instead_of_404ing(self):
        """count_tokens deliberately estimates locally for an unknown model."""
        with _Inventory([], ["azure:foundry"]), \
             patch.object(provider_manager, "get_anthropic_provider_for_model",
                          AsyncMock(return_value=None)):
            client = TestClient(self.app)
            response = client.post("/v1/messages/count_tokens", json={
                "model": "gpt-9-nonexistent",
                "messages": [{"role": "user", "content": "hello world"}],
            })
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["input_tokens"], 0)


class AzureV1SurfaceTests(unittest.TestCase):
    """/openai/v1/* is not covered by the tracking middleware (its paths do not
    match _TRACKED_PREFIXES), so route-level resolution is the only mechanism
    that reaches it -- and it does, because it lives in the handler."""

    def setUp(self):
        self.app = create_azure_openai_app()
        self.app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()

    def test_bare_name_resolves_on_the_azure_v1_surface(self):
        with _Inventory(["azure:foundry/gpt-5.4"], ["azure:foundry"]), \
             patch.object(provider_manager, "chat_completion", _fake_chat_completion), \
             patch.object(provider_manager, "get_provider_for_model", lambda m: _StubProvider()):
            client = TestClient(self.app)
            response = client.post("/openai/v1/chat/completions", json={
                "model": "gpt-5.4", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(response.status_code, 200)

    def test_unknown_model_uses_the_openai_error_envelope(self):
        with _Inventory([], ["azure:foundry"]):
            client = TestClient(self.app)
            response = client.post("/openai/v1/chat/completions", json={
                "model": "gpt-9-nonexistent", "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            })
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")


if __name__ == "__main__":
    unittest.main()

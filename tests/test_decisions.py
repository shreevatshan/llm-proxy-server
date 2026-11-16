"""POST /v1/decisions (OpenAI Decisions API) end to end.

Route tests drive real sub-apps through the real route -> ProviderManager ->
provider dispatch; only the provider object is a stub. Provider tests run the
real OpenAI-compatible and Azure providers against an httpx MockTransport, so the
URL, headers and JSON body the upstream would see are asserted directly.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.auth.admin import AdminUser
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import AzureDeploymentGroup
from app.byok import byok_registry
from app.cache import ModelCache
from app.main import create_azure_openai_app, create_management_app, create_openai_app
from app.model_alias import echo_model_name
from app.openai_models import DecisionCreateRequest, ModelInfo
from app.providers.azure_deployments import (
    API_CHAT,
    API_DECISIONS,
    build_azure_config_fields,
    normalize_deployment_groups,
    serialize_azure_deployments,
)
from app.providers.azure_provider import AzureProvider, azure_call_style
from app.providers.base import BaseProvider, ProviderHTTPError
from app.providers.bedrock_provider import BedrockProvider
from app.providers.custom_providers import create_custom_provider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_compatible import extra_request_headers, preserve_upstream_model
from app.providers.platformai import PlatformAIProvider
from app.providers.provider_manager import provider_manager

OPENAI_KEY = "openai:primary"
AZURE_KEY = "azure:primary"
CANONICAL = f"{OPENAI_KEY}/gpt-6-luna"
AZURE_CANONICAL = f"{AZURE_KEY}/gpt-6-luna"
UPSTREAM_MODEL = "gpt-6-luna-2026-10-22"

BODY = {
    "model": "gpt-6-luna",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "My screen arrived cracked."}]}],
    "questions": [
        {"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"},
        {"type": "choice", "name": None, "instructions": "Refund?",
         "choices": [{"value": True}, {"value": "true", "description": None}]},
        {"type": "score", "name": "urgency", "instructions": "How urgent?",
         "levels": [{"label": "low"}, {"label": "high"}]},
    ],
    "safety_identifier": None,
    "prompt_cache_options": {"mode": "implicit"},
    "future_field": {"kept": None},
}

UPSTREAM = {
    "model": UPSTREAM_MODEL,
    "answers": [
        {"type": "predicate", "name": "damaged", "probability": 0.97},
        {"type": "choice", "name": None, "choice": True, "confidence": 0.8,
         "probabilities": [{"value": True, "probability": 0.8}, {"value": "true", "probability": 0.2}]},
        {"type": "refusal", "name": "urgency"},
        {"type": "brand_new_answer", "name": "x", "extra": [1, 2]},
    ],
    "usage": {
        "input_tokens": 42,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 42,
        "brand_new_usage": 7,
    },
}


# ==================== Route helpers ====================

class _Caller:
    """A non-admin principal with no per-user restrictions."""

    def __init__(self, uid=1):
        self.id = self.user_id = uid
        self.username = f"user{uid}"


async def _echo_upstream(request):
    """What a real provider returns: the upstream object, client name echoed."""
    return {**json.loads(json.dumps(UPSTREAM)), "model": echo_model_name(request)}


class _DecisionsStub:
    """A provider whose decisions_create each test scripts. It sits in
    provider_manager.providers, so ProviderManager.decisions_create is real."""

    def __init__(self, key, behaviour=_echo_upstream):
        self.full_provider_name = key
        self.behaviour = behaviour
        self.calls = []
        self.preserve_seen = []

    def supports_api_for_model(self, model_name, api_name):
        return True

    async def decisions_create(self, request):
        self.calls.append(request)
        self.preserve_seen.append(preserve_upstream_model.get())
        return await self.behaviour(request)


class _Inventory:
    """Swap the process-wide provider registry and model cache."""

    def __init__(self, providers, model_ids):
        self.cache = ModelCache()
        self.cache.update_models([ModelInfo(id=m, created=0, owned_by="t", provider="t") for m in model_ids])
        self.providers = providers

    def __enter__(self):
        self._saved = (provider_manager.model_cache, provider_manager.providers)
        provider_manager.model_cache = self.cache
        provider_manager.providers = self.providers
        return self

    def __exit__(self, *exc):
        provider_manager.model_cache, provider_manager.providers = self._saved


def _post(app, path, body, providers, model_ids, headers=None):
    """POST through *app* with tracking stubbed; returns (response, tracker start kwargs)."""
    with _Inventory(providers, model_ids), \
         patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock) as start, \
         patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
        response = TestClient(app).post(path, json=body, headers=headers or {})
    return response, (start.await_args.kwargs if start.await_args else None)


def _raising(exc):
    async def behaviour(request):
        raise exc
    return behaviour


# ==================== Route: /v1/decisions on the OpenAI port ====================

class DecisionsRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = create_openai_app()
        self.app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()

    def _post(self, body=BODY, behaviour=_echo_upstream, model_ids=(CANONICAL,)):
        stub = _DecisionsStub(OPENAI_KEY, behaviour)
        response, start = _post(self.app, "/v1/decisions", body, {OPENAI_KEY: stub}, model_ids)
        return response, stub, start

    def test_bare_name_resolves_and_the_upstream_object_is_relayed_unchanged(self):
        response, stub, start = self._post()
        self.assertEqual(response.status_code, 200)
        # Unknown answer types, refusals and usage fields pass through; only
        # the model is the client's own name.
        self.assertEqual(response.json(), {**UPSTREAM, "model": "gpt-6-luna"})
        self.assertEqual(stub.calls[0].model, CANONICAL)
        self.assertEqual((start["endpoint"], start["server"]), ("/v1/decisions", "openai"))
        self.assertEqual(stub.preserve_seen, [False])

    def test_the_client_body_reaches_the_provider_as_sent(self):
        _, stub, _ = self._post()
        forwarded = stub.calls[0].model_dump(exclude_unset=True)
        self.assertEqual(forwarded, {**BODY, "model": CANONICAL})

    def test_the_route_charges_the_resolved_model(self):
        enforce = AsyncMock(return_value=None)
        with patch("app.routes.decisions.enforce_access_and_limits", enforce):
            response, stub, _ = self._post()
        self.assertEqual(response.status_code, 200)
        enforce.assert_awaited_once()
        self.assertEqual(enforce.await_args.args[2], CANONICAL)

    def test_a_provider_without_decisions_is_a_400_unsupported_api(self):
        response, _, _ = self._post(
            behaviour=_raising(NotImplementedError("The Decisions API is not supported by this provider"))
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": {
            "message": "The Decisions API is not supported by this provider",
            "type": "invalid_request_error", "param": "model", "code": "unsupported_api",
        }})

    def test_upstream_errors_keep_their_status_body_and_retry_after(self):
        body = {"error": {"message": "slow down", "type": "rate_limit_error", "param": None, "code": None}}
        response, _, _ = self._post(behaviour=_raising(ProviderHTTPError(
            429, "rate limited", body=body, headers={"retry-after": "3", "x-upstream-secret": "s"},
        )))
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json(), body)
        self.assertEqual(response.headers.get("retry-after"), "3")
        self.assertNotIn("x-upstream-secret", response.headers)

    def test_azure_gating_error_passes_through_unchanged(self):
        body = {"error": {"message": "Deployment 'gpt-6-luna' is not configured for /decisions",
                          "type": "invalid_request_error", "param": "model", "code": "unsupported_api"}}
        response, _, _ = self._post(behaviour=_raising(ProviderHTTPError(400, "gated", body)))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), body)

    def test_unexpected_errors_are_a_generic_500(self):
        response, _, _ = self._post(behaviour=_raising(RuntimeError("https://secret.internal/v1 leaked")))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["type"], "server_error")
        self.assertNotIn("secret", response.text)

    def test_unknown_model_is_a_404_and_never_reaches_a_provider(self):
        response, stub, _ = self._post(body={**BODY, "model": "nope-9"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "model_not_found")
        self.assertEqual(stub.calls, [])

    def test_missing_model_is_rejected_before_any_provider(self):
        body = {k: v for k, v in BODY.items() if k != "model"}
        response, stub, _ = self._post(body=body)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(stub.calls, [])

    def test_canonical_ids_pin_requests_to_one_backend(self):
        """The documented workaround for a bare name served by mixed backends."""
        gated = {"error": {"message": "gated", "type": "invalid_request_error",
                           "param": "model", "code": "unsupported_api"}}
        providers = {
            OPENAI_KEY: _DecisionsStub(OPENAI_KEY),
            AZURE_KEY: _DecisionsStub(AZURE_KEY, _raising(ProviderHTTPError(400, "gated", gated))),
        }
        models = (CANONICAL, AZURE_CANONICAL)
        ok, _ = _post(self.app, "/v1/decisions", {**BODY, "model": CANONICAL}, providers, models)
        refused, _ = _post(self.app, "/v1/decisions", {**BODY, "model": AZURE_CANONICAL}, providers, models)
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["model"], CANONICAL)
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(refused.json()["error"]["code"], "unsupported_api")


    def test_nan_in_the_body_is_a_client_error_and_never_reaches_upstream(self):
        upstream = _Upstream()
        provider = _custom_provider()
        provider.client = _mock(provider, upstream)
        raw = b'{"model": "custom:t/gpt-6-luna", "input": "x", "questions": [{"type": "predicate", "w": NaN}]}'
        with _Inventory({"custom:t": provider}, ("custom:t/gpt-6-luna",)), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
            response = TestClient(self.app).post(
                "/v1/decisions", content=raw, headers={"content-type": "application/json"}
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(upstream.requests, [])


class DecisionsAccessTests(unittest.TestCase):
    """Auth and BYOK ownership come from the shared helpers the route calls."""

    def test_unauthenticated_callers_are_rejected_on_both_ports(self):
        for factory, path in ((create_openai_app, "/v1/decisions"),
                              (create_azure_openai_app, "/openai/v1/decisions")):
            with self.subTest(path=path):
                stub = _DecisionsStub(OPENAI_KEY)
                response, _ = _post(factory(), path, BODY, {OPENAI_KEY: stub}, (CANONICAL,))
                self.assertEqual(response.status_code, 401)
                self.assertEqual(stub.calls, [])

    def test_only_the_owner_reaches_a_byok_instance(self):
        owner, other = 42, 7
        key = f"openai:byok-{owner}-personal"
        model = f"{key}/gpt-6-luna"
        saved_owners = dict(byok_registry._owners)
        byok_registry.replace({key: owner})
        self.addCleanup(byok_registry.replace, saved_owners)
        for factory, path in ((create_openai_app, "/v1/decisions"),
                              (create_azure_openai_app, "/openai/v1/decisions")):
            for label, principal, expected in (
                ("other user", _Caller(other), 404),
                ("admin", AdminUser(username="admin", email="admin@example.com"), 404),
                ("owner", _Caller(owner), 200),
            ):
                with self.subTest(path=path, caller=label):
                    app = factory()
                    app.dependency_overrides[authenticate_jwt_or_api_key] = lambda p=principal: p
                    stub = _DecisionsStub(key)
                    with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)):
                        response, _ = _post(app, path, {**BODY, "model": model}, {key: stub}, (model,))
                    self.assertEqual(response.status_code, expected)
                    self.assertEqual(len(stub.calls), 1 if expected == 200 else 0)


# ==================== Route: Azure port and management mounts ====================

class DecisionsSurfaceTests(unittest.TestCase):
    def _app(self, factory):
        app = factory()
        app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()
        return app

    def test_azure_port_serves_openai_v1_decisions(self):
        stub = _DecisionsStub(OPENAI_KEY)
        response, start = _post(self._app(create_azure_openai_app), "/openai/v1/decisions",
                                BODY, {OPENAI_KEY: stub}, (CANONICAL,))
        self.assertEqual(response.status_code, 200)
        self.assertEqual((start["endpoint"], start["server"]), ("/openai/v1/decisions", "azure_openai"))
        # The Azure v1 surface keeps the upstream model string.
        self.assertEqual(stub.preserve_seen, [True])

    def test_management_mounts_match_their_dedicated_ports(self):
        mgmt = create_management_app(
            self._app(create_openai_app), FastAPI(), self._app(create_azure_openai_app), FastAPI()
        )
        for path, endpoint, server, preserve in (
            ("/openai/v1/decisions", "/v1/decisions", "openai", False),
            ("/azure-openai/openai/v1/decisions", "/openai/v1/decisions", "azure_openai", True),
        ):
            with self.subTest(path=path):
                stub = _DecisionsStub(OPENAI_KEY)
                response, start = _post(mgmt, path, BODY, {OPENAI_KEY: stub}, (CANONICAL,))
                self.assertEqual(response.status_code, 200)
                self.assertEqual((start["endpoint"], start["server"]), (endpoint, server))
                self.assertEqual(stub.preserve_seen, [preserve])


# ==================== Provider wire tests ====================

class _Upstream:
    """An httpx MockTransport handler that records requests."""

    def __init__(self, status=200, json_body=None, content=None, content_type="application/json"):
        self.status = status
        self.json_body = UPSTREAM if json_body is None and content is None else json_body
        self.content = content
        self.content_type = content_type
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.content is not None:
            return httpx.Response(self.status, content=self.content, headers={"content-type": self.content_type})
        return httpx.Response(self.status, json=self.json_body)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


def _mock(provider, upstream):
    """The provider's own client (so its base_url stays under test) on a mock transport."""
    return provider._get_responses_client().with_options(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)), max_retries=0,
    )


def _custom_provider(supported_apis=None):
    return create_custom_provider({
        "provider_type": "custom",
        "provider_name": "custom",
        "name": "t",
        "base_url": "http://upstream.test/v1",
        "api_key": "k",
        "supported_apis": supported_apis or ["openai"],
    })


def _request(model):
    request = DecisionCreateRequest.model_validate(BODY)
    request.model = model
    return request


class _ContextVar:
    def __init__(self, var, value):
        self.var, self.value = var, value

    def __enter__(self):
        self.token = self.var.set(self.value)

    def __exit__(self, *exc):
        self.var.reset(self.token)


class OpenAICompatiblePassthroughTests(unittest.IsolatedAsyncioTestCase):
    async def _call(self, upstream, request=None):
        provider = _custom_provider()
        provider.client = _mock(provider, upstream)
        return await provider.decisions_create(request or _request("custom:t/gpt-6-luna"))

    async def test_posts_the_client_body_to_base_url_decisions(self):
        upstream = _Upstream()
        result = await self._call(upstream)
        sent = upstream.requests[0]
        self.assertEqual((sent.method, str(sent.url)), ("POST", "http://upstream.test/v1/decisions"))
        self.assertEqual(sent.headers["authorization"], "Bearer k")
        self.assertEqual(sent.headers["content-type"], "application/json")
        # Exactly what the client sent: nulls, bool-vs-string choices and unknown
        # fields kept; only the model is the upstream id.
        self.assertEqual(upstream.body, {**BODY, "model": "gpt-6-luna"})
        self.assertEqual(result["answers"], UPSTREAM["answers"])
        self.assertEqual(result["usage"], UPSTREAM["usage"])
        self.assertEqual(result["model"], "custom:t/gpt-6-luna")

    async def test_omitted_fields_are_not_sent_as_null(self):
        upstream = _Upstream()
        request = DecisionCreateRequest.model_validate({"model": "gpt-6-luna", "questions": BODY["questions"]})
        request.model = "custom:t/gpt-6-luna"
        await self._call(upstream, request)
        self.assertEqual(upstream.body, {"model": "gpt-6-luna", "questions": BODY["questions"]})

    async def test_nan_is_rejected_before_any_call(self):
        upstream = _Upstream()
        request = DecisionCreateRequest.model_validate({"model": "gpt-6-luna", "input": float("nan")})
        request.model = "custom:t/gpt-6-luna"
        with self.assertRaises(ValueError):
            await self._call(upstream, request)
        self.assertEqual(upstream.requests, [])

    async def test_preview_headers_travel_as_headers_not_in_the_body(self):
        upstream = _Upstream()
        with _ContextVar(extra_request_headers, {"aoai-evals": "preview"}):
            await self._call(upstream)
        self.assertEqual(upstream.requests[0].headers["aoai-evals"], "preview")
        self.assertNotIn("extra_headers", upstream.body)

    async def test_upstream_model_is_kept_when_preserve_is_set(self):
        with _ContextVar(preserve_upstream_model, True):
            result = await self._call(_Upstream())
        self.assertEqual(result["model"], UPSTREAM_MODEL)

    async def test_upstream_4xx_keeps_its_status_and_body(self):
        error = {"error": {"message": "questions: too many", "type": "invalid_request_error",
                           "param": "questions", "code": None}}
        with self.assertRaises(ProviderHTTPError) as ctx:
            await self._call(_Upstream(status=400, json_body=error))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.body["error"]["param"], "questions")

    async def test_malformed_or_non_object_responses_are_a_502(self):
        cases = (
            (b"", "application/json"),
            (b"<html>oops</html>", "application/json"),
            (b"null", "application/json"),
            (b"[1, 2]", "application/json"),
            (b"ok", "text/plain"),
        )
        for content, content_type in cases:
            with self.subTest(content=content, content_type=content_type):
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await self._call(_Upstream(content=content, content_type=content_type))
                self.assertEqual(ctx.exception.status_code, 502)
                self.assertEqual(ctx.exception.body["error"]["type"], "upstream_error")

    async def test_provider_without_the_openai_api_refuses_without_a_call(self):
        provider = _custom_provider(supported_apis=["anthropic"])
        self.assertIsNone(provider._get_responses_client())
        with self.assertRaises(NotImplementedError):
            await provider.decisions_create(_request("custom:t/gpt-6-luna"))


# ==================== Azure ====================

LUNA = "luna"
CHAT_DEP = "chat-dep"
ROWS = [
    {"deployments": [LUNA], "apis": [API_DECISIONS]},
    {"deployments": [CHAT_DEP], "apis": [API_CHAT]},
]


def _azure(rows=ROWS, backend="openai"):
    creds = SimpleNamespace(
        endpoint="https://example.openai.azure.com",
        api_key="k",
        discovery_api_version=None,
        azure_backend=backend,
        dynamic_discovery=False,
        deployments_json=serialize_azure_deployments(deployment_groups=rows),
    )
    return AzureProvider({"name": "test", **build_azure_config_fields(creds)})


def _wire_azure(provider, upstream):
    client = _mock(provider, upstream)
    provider._v1_client = provider._responses_client = provider.client = client
    return provider


class AzureDecisionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_decisions_row_posts_to_the_v1_surface(self):
        upstream = _Upstream()
        provider = _wire_azure(_azure(), upstream)
        await provider.decisions_create(_request(f"azure:test/{LUNA}"))
        sent = upstream.requests[0]
        self.assertEqual(str(sent.url), "https://example.openai.azure.com/openai/v1/decisions")
        self.assertEqual(sent.headers["authorization"], "Bearer k")
        self.assertEqual(upstream.body["model"], LUNA)

    async def test_azure_uses_the_v1_client_even_in_a_deployment_style_context(self):
        upstream = _Upstream()
        provider = _wire_azure(_azure(), upstream)
        with _ContextVar(azure_call_style, "deployment"):
            await provider.decisions_create(_request(f"azure:test/{LUNA}"))
        self.assertEqual(str(upstream.requests[0].url), "https://example.openai.azure.com/openai/v1/decisions")

    async def test_a_row_without_decisions_is_refused_before_any_call(self):
        upstream = _Upstream()
        provider = _wire_azure(_azure(), upstream)
        with self.assertRaises(ProviderHTTPError) as ctx:
            await provider.decisions_create(_request(f"azure:test/{CHAT_DEP}"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.body["error"]["code"], "unsupported_api")
        self.assertEqual(upstream.requests, [])

    def test_a_decisions_only_row_stays_off_chat_and_messages(self):
        provider = _azure()
        self.assertEqual(provider._deployment_api_error(LUNA, API_CHAT).body["error"]["code"], "unsupported_api")
        self.assertEqual(provider.get_supported_apis_for_model(f"azure:test/{LUNA}"), ["openai", "azure_openai"])
        self.assertFalse(provider._is_chat_capable_model(LUNA))
        self.assertEqual(provider.get_anthropic_mode_for_model(LUNA), "unsupported")

    def test_deployment_rows_accept_decisions(self):
        rows = normalize_deployment_groups([{"deployments": [LUNA], "apis": ["decisions"]}], strict=True)
        self.assertEqual(rows[0]["apis"], ["decisions"])
        self.assertEqual(AzureDeploymentGroup(deployments=[LUNA], apis=["decisions"]).apis, ["decisions"])
        with self.assertRaises(ValidationError):
            AzureDeploymentGroup(deployments=[LUNA], apis=["decide"])

    async def test_foundry_claude_deployments_are_refused_without_a_call(self):
        provider = AzureProvider({
            "name": "foundry",
            "endpoint": "https://example.openai.azure.com/",
            "api_key": "k",
            "azure_backend": "foundry",
            "dynamic_discovery": False,
            "openai_deployments": ["grok-3"],
            "anthropic_deployments": ["claude-sonnet-4"],
            "deployments": ["grok-3", "claude-sonnet-4"],
        })
        upstream = _Upstream()
        _wire_azure(provider, upstream)
        with self.assertRaises(NotImplementedError):
            await provider.decisions_create(_request("azure:foundry/claude-sonnet-4"))
        self.assertEqual(upstream.requests, [])
        await provider.decisions_create(_request("azure:foundry/grok-3"))
        self.assertEqual(upstream.body["model"], "grok-3")

    def test_supported_endpoints_list_decisions(self):
        for backend in ("openai", "foundry"):
            with self.subTest(backend=backend):
                endpoints = _azure(backend=backend).get_supported_endpoints()
                self.assertIn("/openai/v1/decisions", endpoints)
                self.assertIn("/v1/decisions", endpoints)


class AzurePortIntegrationTests(unittest.TestCase):
    """Real AzureProvider behind the real Azure app: the v1 middleware's preview
    headers and preserve flag must reach the upstream call."""

    def setUp(self):
        self.upstream = _Upstream()
        self.provider = _wire_azure(_azure(), self.upstream)
        self.app = create_azure_openai_app()
        self.app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()

    def _post(self, deployment):
        key = self.provider.full_provider_name
        models = (f"{key}/{LUNA}", f"{key}/{CHAT_DEP}")
        response, _ = _post(self.app, "/openai/v1/decisions", {**BODY, "model": f"{key}/{deployment}"},
                            {key: self.provider}, models, headers={"aoai-foo": "1"})
        return response

    def test_decisions_deployment_round_trip(self):
        response = self._post(LUNA)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], UPSTREAM_MODEL)
        sent = self.upstream.requests[0]
        self.assertEqual(sent.url.path, "/openai/v1/decisions")
        self.assertEqual(sent.headers["aoai-foo"], "1")
        self.assertEqual(self.upstream.body, {**BODY, "model": LUNA})

    def test_chat_only_deployment_is_a_400_unsupported_api(self):
        response = self._post(CHAT_DEP)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "unsupported_api")
        self.assertEqual(self.upstream.requests, [])


# ==================== Other providers and the chat-capable heuristic ====================

class UnsupportedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_google_refuses(self):
        provider = GoogleProvider({"api_key": "k", "name": "test"})
        with self.assertRaises(NotImplementedError):
            await provider.decisions_create(_request("google:test/gemini-3.5-flash"))

    async def test_bedrock_and_platformai_use_the_refusing_default(self):
        for cls in (BedrockProvider, PlatformAIProvider):
            with self.subTest(provider=cls.__name__):
                self.assertIs(cls.decisions_create, BaseProvider.decisions_create)
        with self.assertRaises(NotImplementedError):
            await BaseProvider.decisions_create(object(), _request("x:y/gpt-6-luna"))


class DecisionModelHeuristicTests(unittest.TestCase):
    DECISION_ONLY = ("gpt-6-luna-decisions", "openai/gpt-6-luna-decisions",
                     "microsoft-decision-1", "gemini-decision-1")
    CHAT = ("gpt-6-luna", "loan-decision-gpt4o", "decision-router", "decisions-prod",
            "loan-decision", "credit.decision", "claims_decisions", "decision")

    def test_decision_only_models_are_not_chat_capable(self):
        custom = _custom_provider()
        azure = _azure()  # these names are not in its rows, so the name heuristic decides
        for provider in (custom, azure):
            for name in self.DECISION_ONLY:
                with self.subTest(provider=type(provider).__name__, name=name):
                    self.assertFalse(provider._is_chat_capable_model(f"x:y/{name}"))
            for name in self.CHAT:
                with self.subTest(provider=type(provider).__name__, name=name):
                    self.assertTrue(provider._is_chat_capable_model(f"x:y/{name}"))


if __name__ == "__main__":
    unittest.main()

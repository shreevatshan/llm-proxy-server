"""The TypeSafe app (port 11366, /typesafe on the management port): System One.

Route tests drive the real TypeSafe app (directly and through the management
mount) via the real route -> ProviderManager -> provider dispatch; only the
provider object is a stub. Provider tests run the real CustomProvider against an
httpx MockTransport, so the URL, headers and body the upstream sees are asserted
directly. Decoding tests send real gzip / zstd bodies.
"""

import gzip
import json
import logging
import os
import re
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
try:  # requirements.txt installs pyzstd below Python 3.14; 3.14+ has compression.zstd
    import pyzstd as _zstd_lib

    def zstd_compress(data, window_log=None):
        if window_log is None:
            return _zstd_lib.compress(data)
        return _zstd_lib.compress(data, level_or_option={_zstd_lib.CParameter.windowLog: window_log})
except ImportError:
    from compression import zstd as _zstd_lib

    def zstd_compress(data, window_log=None):
        if window_log is None:
            return _zstd_lib.compress(data)
        return _zstd_lib.compress(data, options={_zstd_lib.CompressionParameter.window_log: window_log})
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth.admin import AdminUser
from app.auth.middleware import authenticate_jwt_or_api_key
from app.auth.models import MODEL_ALIAS_DEFAULT_SURFACES, ModelAliasResponse, ModelAliasUpsert
from app.byok import byok_registry
from app.cache import ModelCache
from app.main import (
    create_azure_openai_app,
    create_management_app,
    create_openai_app,
    create_typesafe_app,
)
from app.model_access_dep import ModelAccessDenied
from app.model_alias import _ALL_SURFACES, _parse_apis, build_rule, echo_model_name
from app.model_resolution import ModelUnavailable, _filter_pool
from app.openai_models import ModelInfo
from app.providers import custom_providers
from app.providers.base import SYSTEMONE_API, BaseProvider, ProviderHTTPError
from app.providers.bedrock_provider import BedrockProvider
from app.providers.custom_providers import CustomProvider, create_custom_provider
from app.providers.google_provider import GoogleProvider
from app.providers.openai_compatible import _translate_openai_sdk_error, extra_request_headers
from app.providers.platformai import PlatformAIProvider
from app.providers.provider_manager import provider_manager
from app.rate_limit import RateLimitExceeded
from app.routes import _errors
from app.routes.systemone import SystemOneRequest
from tests.test_decisions import _ContextVar, _azure, _wire_azure
from tests.test_mount_parity import _AliasSnapshot

KEY = "custom:ts"
CANONICAL = f"{KEY}/jev-latest"
REQUEST_ID = re.compile(r"^req_[0-9a-f]{32}$")

BODY = {
    "model": "jev-latest",
    "state": {"ticket": "My screen arrived cracked.", "order": 1234},
    "questions": {
        "damaged": {"type": "noul", "instructions": "Is the item damaged?", "criteria": None},
        "refund": {"type": "choice", "criteria": {"yes": "issue a refund", "no": "no refund"}, "weight": 2},
        "urgency": {"type": "score", "criteria": [{"label": "low"}, {"label": "high", "level": {"x": 1}}]},
    },
    "assets": {"photo": {"data_base64": "aGVsbG8="}},
    "future_field": {"kept": None},
}

UPSTREAM = {
    "model": "jev-1.14.0",
    "answers": {
        "damaged": {"type": "noul", "noul": 0.97},
        "refund": {"type": "choice", "choice": "yes", "confidence": 0.8,
                   "probabilities": {"yes": 0.8, "no": 0.2}},
        "urgency": {"type": "score", "score": 1, "confidence": 0.6,
                    "legend": {"0": "low", "1": "high"}, "probabilities": {"0": 0.4, "1": 0.6}},
    },
    "usage": {"input_tokens": 367, "output_tokens": 3},
}


# ==================== Route helpers ====================

class _Caller:
    """A non-admin principal with no per-user restrictions."""

    def __init__(self, uid=1):
        self.id = self.user_id = uid
        self.username = f"user{uid}"


async def _echo_upstream(request):
    return {**json.loads(json.dumps(UPSTREAM)), "model": echo_model_name(request)}


def _raising(exc):
    async def behaviour(request):
        raise exc
    return behaviour


class _SystemOneStub:
    """A provider whose systemone_create each test scripts. It sits in
    provider_manager.providers, so ProviderManager.systemone_create is real."""

    def __init__(self, key, behaviour=_echo_upstream, surfaces=(SYSTEMONE_API,),
                 upstream_headers=None):
        self.full_provider_name = key
        self.behaviour = behaviour
        self.surfaces = surfaces
        self.upstream_headers = {"x-typesafe-request-id": "req_1", "x-upstream-secret": "s"} \
            if upstream_headers is None else upstream_headers
        self.calls = []
        self.headers_seen = []

    def supports_api_for_model(self, model_name, api_name):
        return api_name in self.surfaces

    async def systemone_create(self, request, headers=None):
        self.calls.append(request)
        self.headers_seen.append(headers)
        return await self.behaviour(request), dict(self.upstream_headers)


class _Inventory:
    """Swap the process-wide provider registry and model cache."""

    def __init__(self, providers, model_ids):
        self.cache = ModelCache()
        self.cache.update_models([
            ModelInfo(id=m, created=0, owned_by="t", provider=m.split("/", 1)[0]) for m in model_ids
        ])
        self.providers = providers

    def __enter__(self):
        self._saved = (provider_manager.model_cache, provider_manager.providers)
        provider_manager.model_cache = self.cache
        provider_manager.providers = self.providers
        return self

    def __exit__(self, *exc):
        provider_manager.model_cache, provider_manager.providers = self._saved


def _send(app, method, path, providers=None, model_ids=(), raise_server_exceptions=True, **kwargs):
    """One request through *app* with tracking stubbed: (response, start kwargs or None).

    TestClient is used without ``with`` so no lifespan (DB, provider init) runs.
    """
    with _Inventory(providers or {}, model_ids), \
         patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock) as start, \
         patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
        client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
        response = client.request(method, path, **kwargs)
    return response, (start.await_args.kwargs if start.await_args else None)


def _typesafe_app(principal=None):
    app = create_typesafe_app()
    if principal is not False:
        app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: principal or _Caller()
    return app


def _targets(principal=None):
    """(label, app, path prefix): the dedicated port and the management mount."""
    direct = _typesafe_app(principal)
    mounted_sub = _typesafe_app(principal)
    mgmt = create_management_app(FastAPI(), FastAPI(), FastAPI(), mounted_sub)
    return (("direct", direct, ""), ("mounted", mgmt, "/typesafe"))


def _ts_error(response, status, error_type):
    """Assert a TypeSafe {"detail": {...}} error and return its message."""
    assert response.status_code == status, (response.status_code, response.text)
    body = response.json()
    assert set(body) == {"detail"}, body
    assert body["detail"]["error_type"] == error_type, body
    assert REQUEST_ID.match(response.headers["x-typesafe-request-id"]), response.headers
    return body["detail"]["message"]


# ==================== Route: POST /v1/systemone ====================

class SystemOneRouteTests(unittest.TestCase):
    def _post(self, body=BODY, stub=None, model_ids=(CANONICAL,), headers=None, prefix="", app=None):
        stub = stub or _SystemOneStub(KEY)
        response, start = _send(app or _typesafe_app(), "POST", f"{prefix}/v1/systemone",
                                {KEY: stub}, model_ids, json=body, headers=headers or {})
        return response, stub, start

    def test_upstream_object_is_relayed_unchanged_on_both_entry_points(self):
        for label, app, prefix in _targets():
            with self.subTest(label):
                response, stub, start = self._post(app=app, prefix=prefix)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {**UPSTREAM, "model": "jev-latest"})
                self.assertEqual(response.headers["x-typesafe-request-id"], "req_1")
                self.assertNotIn("x-upstream-secret", response.headers)
                self.assertEqual(stub.calls[0].model, CANONICAL)
                self.assertEqual((start["endpoint"], start["server"]), ("/v1/systemone", "systemone"))
                self.assertEqual(start["model"], "jev-latest")

    def test_the_client_body_reaches_the_provider_as_sent(self):
        _, stub, _ = self._post()
        self.assertEqual(stub.calls[0].model_dump(exclude_unset=True), {**BODY, "model": CANONICAL})

    def test_only_the_danger_zone_headers_are_forwarded(self):
        _, stub, _ = self._post(headers={
            "TypeSafe-Danger-Zone-Images": "true",
            "TypeSafe-Danger-Zone-Long-Context": "true",
            "TypeSafe-Danger-Zone-Future": "true",
            "X-TypeSafe-SDK": "typesafe-sdk/0.7.5",
            "aoai-foo": "1",
            "X-Other": "1",
        })
        self.assertEqual(stub.headers_seen, [{
            "TypeSafe-Danger-Zone-Images": "true",
            "TypeSafe-Danger-Zone-Long-Context": "true",
        }])
        _, stub, _ = self._post()
        self.assertEqual(stub.headers_seen, [{}])

    def test_a_local_request_id_when_the_upstream_sends_none(self):
        response, _, _ = self._post(stub=_SystemOneStub(KEY, upstream_headers={"retry-after-ms": "5"}))
        self.assertEqual(response.status_code, 200)
        self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)
        self.assertEqual(response.headers["retry-after-ms"], "5")

    def test_the_route_charges_the_resolved_model_once(self):
        enforce = AsyncMock(return_value=None)
        with patch("app.routes.systemone.enforce_access_and_limits", enforce):
            response, _, _ = self._post()
        self.assertEqual(response.status_code, 200)
        enforce.assert_awaited_once()
        self.assertEqual(enforce.await_args.args[2], CANONICAL)

    def test_stream_is_a_400_before_resolution_or_charging(self):
        enforce = AsyncMock(return_value=None)
        for model in ("jev-latest", "nope-9"):
            with self.subTest(model=model), patch("app.routes.systemone.enforce_access_and_limits", enforce):
                response, stub, _ = self._post(body={**BODY, "model": model, "stream": True})
                self.assertIn("stream", _ts_error(response, 400, "invalid_request_error"))
                self.assertEqual(stub.calls, [])
        enforce.assert_not_awaited()
        response, stub, _ = self._post(body={**BODY, "stream": False})
        self.assertEqual(response.status_code, 200)
        self.assertIs(stub.calls[0].model_dump(exclude_unset=True)["stream"], False)

    def test_questions_must_be_a_non_empty_object(self):
        for questions in ({}, [], "x", None):
            with self.subTest(questions=questions):
                body = {**BODY, "questions": questions} if questions is not None else \
                    {k: v for k, v in BODY.items() if k != "questions"}
                response, stub, _ = self._post(body=body)
                self.assertEqual(response.status_code, 422)
                self.assertIsInstance(response.json()["detail"], list)
                self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)
                self.assertEqual(stub.calls, [])

    def test_missing_model_is_a_422(self):
        response, stub, _ = self._post(body={k: v for k, v in BODY.items() if k != "model"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "model"])
        self.assertEqual(stub.calls, [])

    def test_unknown_model_is_a_typesafe_404(self):
        response, stub, _ = self._post(body={**BODY, "model": "nope-9"})
        self.assertIn("nope-9", _ts_error(response, 404, "not_found_error"))
        self.assertEqual(stub.calls, [])

    def test_a_canonical_id_on_an_incapable_provider_is_a_400_without_charging(self):
        enforce = AsyncMock(return_value=None)
        stub = _SystemOneStub(KEY, surfaces=("openai",))
        with patch("app.routes.systemone.enforce_access_and_limits", enforce):
            response, _, _ = self._post(body={**BODY, "model": CANONICAL}, stub=stub)
        self.assertIn("not served over the System One API", _ts_error(response, 400, "invalid_request_error"))
        enforce.assert_not_awaited()
        self.assertEqual(stub.calls, [])

    def test_an_ambiguous_provider_prefix_is_a_400(self):
        providers = {"custom:a": _SystemOneStub("custom:a"), "custom:b": _SystemOneStub("custom:b")}
        response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", providers,
                            ("custom:a/jev-latest", "custom:b/jev-latest"),
                            json={**BODY, "model": "custom/jev-latest"})
        self.assertIn("ambiguous", _ts_error(response, 400, "invalid_request_error"))

    def test_a_provider_refusal_is_a_400(self):
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(NotImplementedError("not configured"))))
        self.assertEqual(_ts_error(response, 400, "invalid_request_error"), "not configured")

    def test_typesafe_upstream_errors_are_relayed_verbatim(self):
        detail = {"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}
        error = ProviderHTTPError(422, "x", body=detail, headers={
            "retry-after-ms": "250", "x-typesafe-request-id": "req_err", "x-upstream-secret": "s",
        })
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(error)))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), detail)
        self.assertEqual(response.headers["x-typesafe-request-id"], "req_err")
        self.assertEqual(response.headers["retry-after-ms"], "250")
        self.assertNotIn("x-upstream-secret", response.headers)
        overloaded = {"detail": {"error_type": "overloaded_error", "message": "busy"}}
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(
            ProviderHTTPError(529, "x", body=overloaded, headers={"retry-after": "1"}))))
        self.assertEqual((response.status_code, response.json()), (529, overloaded))
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)

    def test_other_upstream_error_shapes_are_converted(self):
        for status, body, error_type, message in (
            (429, {"error": {"code": 429, "message": "slow down", "metadata": {}}, "user_id": "u"},
             "rate_limit_error", "slow down"),
            (500, {"error": "boom"}, "api_error", "boom"),
            (404, {"error": {"message": "Upstream provider returned status 404",
                             "type": "upstream_error", "code": 404}},
             "not_found_error", "Upstream provider returned status 404"),
            (504, {"error": {"message": "timed out", "type": "timeout_error"}}, "timeout_error", "timed out"),
        ):
            with self.subTest(status=status):
                response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(
                    ProviderHTTPError(status, "fallback message", body=body))))
                self.assertEqual(_ts_error(response, status, error_type), message)

    def test_an_upstream_nan_or_lone_surrogate_in_a_success_is_still_a_200(self):
        async def odd(request):
            return {"model": "m", "answers": {"a": {"type": "noul", "noul": float("nan"), "note": "\ud800"}},
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        response, _, _ = self._post(stub=_SystemOneStub(KEY, odd))
        self.assertEqual(response.status_code, 200)
        self.assertIn("NaN", response.text)
        self.assertIn("\\ud800", response.text)

    def test_an_upstream_nan_in_an_error_body_is_relayed_not_a_500(self):
        error = ProviderHTTPError(400, "x", body={"detail": {"error_type": "x", "message": "m", "v": float("nan")}})
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(error)))
        self.assertEqual(response.status_code, 400)
        self.assertIn("NaN", response.text)

    def test_nan_or_surrogates_in_the_request_are_refused_before_charging(self):
        enforce = AsyncMock(return_value=None)
        for raw, needle in (
            (b'{"model": "jev-latest", "state": NaN, "questions": {"q": {"type": "noul"}}}', "NaN"),
            (b'{"model": "jev-latest", "state": "\\ud800", "questions": {"q": {"type": "noul"}}}', "surrogate"),
        ):
            with self.subTest(needle), patch("app.routes.systemone.enforce_access_and_limits", enforce):
                stub = _SystemOneStub(KEY)
                response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", {KEY: stub}, (CANONICAL,),
                                    content=raw, headers={"content-type": "application/json"})
                self.assertIn(needle, _ts_error(response, 400, "invalid_request_error"))
                self.assertEqual(stub.calls, [])
        enforce.assert_not_awaited()

    def test_errors_that_echo_odd_input_keep_their_status(self):
        for raw, status in (
            (b'{"model": "\\ud800", "state": "s", "questions": {"q": {"type": "noul"}}}', 404),
            (b'{"model": NaN, "state": "s", "questions": {"q": {"type": "noul"}}}', 422),
            (b'{"model": "jev-latest", "state": "s", "questions": ["\\ud800"]}', 422),
        ):
            with self.subTest(raw=raw):
                response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", {KEY: _SystemOneStub(KEY)},
                                    (CANONICAL,), content=raw, headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, status, response.text)
                self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)

    def test_unsuffixed_rate_limit_headers_are_relayed(self):
        error = ProviderHTTPError(429, "x", body={"detail": {"error_type": "rate_limit_error", "message": "m"}},
                                  headers={"x-ratelimit-limit": "10", "x-ratelimit-remaining": "0",
                                           "x-ratelimit-reset": "5", "x-ratelimit-used": "10"})
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(error)))
        self.assertEqual((response.headers["x-ratelimit-limit"], response.headers["x-ratelimit-reset"]), ("10", "5"))
        self.assertNotIn("x-ratelimit-used", response.headers)

    def test_unexpected_errors_are_a_generic_500_without_leaks(self):
        response, _, _ = self._post(stub=_SystemOneStub(KEY, _raising(RuntimeError("https://secret.internal/v1"))))
        self.assertEqual(_ts_error(response, 500, "api_error"), "System One create error")
        self.assertNotIn("secret", response.text)

    def test_multipart_is_a_415_before_auth(self):
        for label, app, prefix in _targets(principal=False):  # real auth, no credentials
            with self.subTest(label):
                response, _ = _send(app, "POST", f"{prefix}/v1/systemone",
                                    files={"request": ("r.json", b"{}", "application/json")})
                self.assertIn("multipart", _ts_error(response, 415, "api_usage_error"))


class SystemOneErrorEnvelopeTests(unittest.TestCase):
    """Errors raised by shared code are rendered in TypeSafe's shape."""

    def _post_with_enforce_raising(self, exc):
        with patch("app.routes.systemone.enforce_access_and_limits", AsyncMock(side_effect=exc)):
            response, _ = _send(_typesafe_app(), "POST", "/v1/systemone",
                                {KEY: _SystemOneStub(KEY)}, (CANONICAL,), json=BODY)
        return response

    def test_rate_limit_is_a_typesafe_429_with_its_headers(self):
        exc = RateLimitExceeded(
            body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error",
                            "code": "rate_limit_exceeded", "param": None}},
            headers={"Retry-After": "7", "X-RateLimit-Limit-Requests": "5"},
        )
        response = self._post_with_enforce_raising(exc)
        self.assertEqual(_ts_error(response, 429, "rate_limit_error"), "Rate limit exceeded")
        self.assertEqual(response.headers["retry-after"], "7")
        self.assertEqual(response.headers["x-ratelimit-limit-requests"], "5")

    def test_access_denied_is_a_typesafe_403(self):
        response = self._post_with_enforce_raising(ModelAccessDenied.openai(CANONICAL))
        self.assertIn(CANONICAL, _ts_error(response, 403, "permission_error"))

    def test_an_outage_is_a_typesafe_503(self):
        response = self._post_with_enforce_raising(ModelUnavailable.outage_openai(CANONICAL, "maintenance"))
        self.assertIn("maintenance", _ts_error(response, 503, "overloaded_error"))

    def test_missing_credentials_are_a_403_and_an_invalid_key_a_401(self):
        app = _typesafe_app(principal=False)
        response, _ = _send(app, "POST", "/v1/systemone", json=BODY)
        self.assertIn("API key", _ts_error(response, 403, "authentication_error"))
        self.assertNotIn("www-authenticate", response.headers)
        with patch("app.auth.middleware.get_api_key", AsyncMock(return_value=None)), \
             patch("app.auth.middleware.auth_cache.get_cached_api_key", return_value=None):
            response, _ = _send(app, "POST", "/v1/systemone", json=BODY,
                                headers={"Authorization": "Bearer bogus-key"})
        _ts_error(response, 401, "authentication_error")
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    def test_models_without_credentials_is_a_403(self):
        for label, app, prefix in _targets(principal=False):
            with self.subTest(label):
                response, _ = _send(app, "GET", f"{prefix}/v1/models")
                _ts_error(response, 403, "authentication_error")

    def test_an_auth_infrastructure_error_is_a_typesafe_500(self):
        with patch("app.auth.middleware.get_api_key", AsyncMock(side_effect=RuntimeError("db down"))), \
             patch("app.auth.middleware.auth_cache.get_cached_api_key", return_value=None):
            response, _ = _send(_typesafe_app(principal=False), "POST", "/v1/systemone", json=BODY,
                                headers={"Authorization": "Bearer some-key"})
        self.assertEqual(_ts_error(response, 500, "api_error"), "Authentication error")

    def test_an_unhandled_exception_is_a_typesafe_500(self):
        with patch("app.request_tracker.request_tracker.start_request",
                   AsyncMock(side_effect=RuntimeError("tracker broke"))):
            client = TestClient(_typesafe_app(), raise_server_exceptions=False)
            with _Inventory({KEY: _SystemOneStub(KEY)}, (CANONICAL,)):
                response = client.post("/v1/systemone", json=BODY)
        self.assertEqual(_ts_error(response, 500, "api_error"), "Internal server error")
        self.assertNotIn("tracker broke", response.text)

    def test_router_404_and_405_keep_fastapis_shape(self):
        app = _typesafe_app()
        response, _ = _send(app, "POST", "/v1/chat/completions", json={"model": "x"})
        self.assertEqual((response.status_code, response.json()), (404, {"detail": "Not Found"}))
        self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)
        response, _ = _send(app, "GET", "/v1/systemone")
        self.assertEqual((response.status_code, response.json()), (405, {"detail": "Method Not Allowed"}))

    def test_malformed_json_is_a_422_list(self):
        response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", content=b"{not json",
                            headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["type"], "json_invalid")

    def test_invalid_utf8_is_a_typesafe_400(self):
        response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", content=b'{"model": "\xff"}',
                            headers={"content-type": "application/json"})
        self.assertIn("parsing the body", _ts_error(response, 400, "invalid_request_error"))


class SystemOneAccessTests(unittest.TestCase):
    def test_only_the_owner_reaches_a_byok_instance(self):
        owner, other = 42, 7
        key = f"custom:byok-{owner}-ts"
        model = f"{key}/jev-latest"
        saved_owners = dict(byok_registry._owners)
        byok_registry.replace({key: owner})
        self.addCleanup(byok_registry.replace, saved_owners)
        for label, principal, expected in (
            ("other user", _Caller(other), 404),
            ("admin", AdminUser(username="admin", email="admin@example.com"), 404),
            ("owner", _Caller(owner), 200),
        ):
            with self.subTest(caller=label):
                stub = _SystemOneStub(key)
                with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)):
                    response, _ = _send(_typesafe_app(principal), "POST", "/v1/systemone", {key: stub},
                                        (model,), json={**BODY, "model": model})
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(len(stub.calls), 1 if expected == 200 else 0)


# ==================== Routing on the systemone surface ====================

class SystemOneRoutingTests(unittest.TestCase):
    def _post_many(self, providers, model_ids, body=BODY, n=1, principal=None):
        out = []
        for _ in range(n):
            with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)):
                response, _ = _send(_typesafe_app(principal), "POST", "/v1/systemone", providers,
                                    model_ids, json=body)
            out.append(response)
        return out

    def test_bare_names_only_rotate_over_system_one_providers(self):
        able = _SystemOneStub("custom:a")
        unable = _SystemOneStub("custom:b", surfaces=("openai", "anthropic"))
        responses = self._post_many({"custom:a": able, "custom:b": unable},
                                    ("custom:a/jev-latest", "custom:b/jev-latest"), n=6)
        self.assertEqual([r.status_code for r in responses], [200] * 6)
        self.assertEqual((len(able.calls), len(unable.calls)), (6, 0))

    def test_two_system_one_providers_share_the_load(self):
        a, b = _SystemOneStub("custom:a"), _SystemOneStub("custom:b")
        self._post_many({"custom:a": a, "custom:b": b}, ("custom:a/jev-latest", "custom:b/jev-latest"), n=4)
        self.assertEqual((len(a.calls), len(b.calls)), (2, 2))

    def test_a_bare_name_only_incapable_providers_serve_is_a_404(self):
        unable = _SystemOneStub("custom:b", surfaces=("openai",))
        [response] = self._post_many({"custom:b": unable}, ("custom:b/jev-latest",))
        _ts_error(response, 404, "not_found_error")
        self.assertEqual(unable.calls, [])

    def test_fallback_aliases_fire_on_the_systemone_surface(self):
        ts = _SystemOneStub(KEY)
        oai = _SystemOneStub("custom:oai", surfaces=("openai", "anthropic"))
        providers = {KEY: ts, "custom:oai": oai}
        models = (CANONICAL, "custom:oai/gpt-4o")
        with _AliasSnapshot({"gpt-4o": (CANONICAL, frozenset({SYSTEMONE_API}))}, kind="fallback"):
            [response] = self._post_many(providers, models, body={**BODY, "model": "gpt-4o"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(ts.calls[0].model, CANONICAL)

    def test_aliases_are_opt_in_for_the_systemone_surface(self):
        ts = _SystemOneStub(KEY)
        models = (CANONICAL, f"{KEY}/jev-old")
        with _AliasSnapshot({"jev-old": (CANONICAL, _ALL_SURFACES)}):
            self._post_many({KEY: ts}, models, body={**BODY, "model": "jev-old"})
        self.assertEqual(ts.calls[-1].model, f"{KEY}/jev-old")  # a default-surface rule does not apply
        with _AliasSnapshot({"jev-old": (CANONICAL, frozenset({SYSTEMONE_API}))}):
            self._post_many({KEY: ts}, models, body={**BODY, "model": "jev-old"})
        self.assertEqual(ts.calls[-1].model, CANONICAL)

    def test_another_users_byok_instance_never_changes_the_answer(self):
        owner = 7
        byok_key = f"custom:byok-{owner}-ts"
        saved_owners = dict(byok_registry._owners)
        byok_registry.replace({byok_key: owner})
        self.addCleanup(byok_registry.replace, saved_owners)
        admin_stub = _SystemOneStub("custom:adm", surfaces=("openai",))
        byok_stub = _SystemOneStub(byok_key)
        both = ({"custom:adm": admin_stub, byok_key: byok_stub},
                ("custom:adm/jev-latest", f"{byok_key}/jev-latest"))
        only_admin = ({"custom:adm": admin_stub}, ("custom:adm/jev-latest",))
        for label, (providers, models) in (("with the BYOK", both), ("without", only_admin)):
            with self.subTest(label):
                [response] = self._post_many(providers, models, principal=_Caller(42))
                self.assertEqual(response.status_code, 404)
        [response] = self._post_many(*both, principal=_Caller(owner))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(byok_stub.calls), 1)

    def test_other_surfaces_keep_the_never_empty_backstop(self):
        unable = SimpleNamespace(supports_api_for_model=lambda m, api: False)
        with _Inventory({"custom:b": unable}, ("custom:b/m",)):
            self.assertEqual(_filter_pool(["custom:b/m"], "openai"), ["custom:b/m"])
            self.assertEqual(_filter_pool(["custom:b/m"], SYSTEMONE_API), [])

    def test_the_backstop_never_keeps_a_system_one_providers_models(self):
        system_one_only = _so_provider()
        unable = SimpleNamespace(supports_api_for_model=lambda m, api: False)
        with _Inventory({KEY: system_one_only, "custom:b": unable}, (CANONICAL, "custom:b/jev-latest")):
            self.assertEqual(_filter_pool([CANONICAL], "openai"), [])
            self.assertEqual(_filter_pool([CANONICAL, "custom:b/jev-latest"], "openai"), ["custom:b/jev-latest"])
            self.assertEqual(_filter_pool([CANONICAL], SYSTEMONE_API), [CANONICAL])

    def test_a_system_one_only_model_is_a_404_on_chat(self):
        app = create_openai_app()
        app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()
        with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)) as charge:
            response, _ = _send(app, "POST", "/v1/chat/completions", {KEY: _so_provider()}, (CANONICAL,),
                                json={"model": "jev-latest", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status_code, 404, response.text)
        charge.assert_not_awaited()


# ==================== GET /v1/models (TypeSafe shape) ====================

class TypeSafeModelsTests(unittest.TestCase):
    def setUp(self):
        for store in (custom_providers._SYSTEMONE_META, custom_providers._CHAT_LISTED):
            saved = dict(store)
            self.addCleanup(lambda store=store, saved=saved: (store.clear(), store.update(saved)))

    def test_lists_system_one_models_in_typesafe_shape(self):
        custom_providers._SYSTEMONE_META[KEY] = {CANONICAL: ("General-purpose system one model.", "2026-09-15")}
        providers = {KEY: _SystemOneStub(KEY), "custom:oai": _SystemOneStub("custom:oai", surfaces=("openai",))}
        models = (CANONICAL, f"{KEY}/jev-1.14.0", "custom:oai/gpt-4o")
        for label, app, prefix in _targets():
            with self.subTest(label):
                response, _ = _send(app, "GET", f"{prefix}/v1/models", providers, models)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"models": [
                    {"name": f"{KEY}/jev-1.14.0", "description": "", "release_date": "1970-01-01"},
                    {"name": CANONICAL, "description": "General-purpose system one model.",
                     "release_date": "2026-09-15"},
                ]})
                self.assertRegex(response.headers["x-typesafe-request-id"], REQUEST_ID)

    def test_byok_models_are_listed_only_for_their_owner(self):
        owner = 42
        key = f"custom:byok-{owner}-ts"
        saved_owners = dict(byok_registry._owners)
        byok_registry.replace({key: owner})
        self.addCleanup(byok_registry.replace, saved_owners)
        providers = {KEY: _SystemOneStub(KEY), key: _SystemOneStub(key)}
        models = (CANONICAL, f"{key}/jev-latest")
        for label, principal, expected in (
            ("owner", _Caller(owner), [CANONICAL, f"{key}/jev-latest"]),
            ("other", _Caller(7), [CANONICAL]),
            ("admin", AdminUser(username="admin", email="admin@example.com"), [CANONICAL]),
        ):
            with self.subTest(label):
                response, _ = _send(_typesafe_app(principal), "GET", "/v1/models", providers, models)
                self.assertEqual(sorted(m["name"] for m in response.json()["models"]), sorted(expected))


# ==================== Request-body decoding (TypeSafe app only) ====================

def _big_window_zstd():
    return zstd_compress(os.urandom(1 << 20) * 12, window_log=24)  # 12 MiB; repeats keep the frame ~1 MiB


def _valid_credentials(valid=True):
    """Decoding runs before auth and validates the credentials itself; route
    tests override auth, so they stub that pre-check too."""
    return patch("app.auth.middleware.credentials_are_valid", AsyncMock(return_value=valid))


class RequestDecodingTests(unittest.TestCase):
    RAW = json.dumps(BODY).encode()

    def _post(self, content, encoding=None, headers=None, app=None, prefix=""):
        stub = _SystemOneStub(KEY)
        # Decoding runs before auth and needs *some* credentials to be present
        # (auth itself is overridden in these tests).
        hdrs = {"content-type": "application/json", "authorization": "Bearer test-key", **(headers or {})}
        if encoding is not None:
            hdrs["content-encoding"] = encoding
        with _valid_credentials():
            response, start = _send(app or _typesafe_app(), "POST", f"{prefix}/v1/systemone", {KEY: stub},
                                    (CANONICAL,), content=content, headers=hdrs)
        return response, stub, start

    def test_gzip_x_gzip_and_zstd_are_decoded_before_tracking_and_parsing(self):
        for label, app, prefix in _targets():
            for encoding, content in (("gzip", gzip.compress(self.RAW)), ("x-gzip", gzip.compress(self.RAW)),
                                      ("zstd", zstd_compress(self.RAW))):
                with self.subTest(label, encoding=encoding):
                    response, stub, start = self._post(content, encoding, app=app, prefix=prefix)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(stub.calls[0].model_dump(exclude_unset=True), {**BODY, "model": CANONICAL})
                    self.assertEqual(start["model"], "jev-latest")

    def test_identity_case_and_whitespace(self):
        for encoding, content in (("identity", self.RAW), (" GZIP ", gzip.compress(self.RAW)),
                                  ("identity, gzip", gzip.compress(self.RAW)), ("Zstd", zstd_compress(self.RAW))):
            with self.subTest(encoding=encoding):
                response, _, _ = self._post(content, encoding)
                self.assertEqual(response.status_code, 200, response.text)

    def test_undecodable_or_stacked_codings_are_a_415_before_tracking(self):
        for encoding in ("br", "deflate", "gzip, zstd", "gzip, gzip"):
            with self.subTest(encoding=encoding):
                response, stub, start = self._post(self.RAW, encoding)
                _ts_error(response, 415, "api_usage_error")
                self.assertIsNone(start)
                self.assertEqual(stub.calls, [])
        with _Inventory({}, ()), _valid_credentials(), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock):
            response = TestClient(_typesafe_app()).post(
                "/v1/systemone", content=gzip.compress(self.RAW),
                headers=[("content-type", "application/json"), ("authorization", "Bearer test-key"),
                         ("content-encoding", "gzip"), ("content-encoding", "zstd")])
        self.assertIn("one Content-Encoding layer", _ts_error(response, 415, "api_usage_error"))

    def test_br_names_only_what_can_be_decoded(self):
        response, _, _ = self._post(self.RAW, "br")
        self.assertIn("supported: gzip, x-gzip, zstd", response.json()["detail"]["message"])
        with patch("app.request_encoding._zstd", None):
            response, _, _ = self._post(zstd_compress(self.RAW), "zstd")
        self.assertIn("supported: gzip, x-gzip", _ts_error(response, 415, "api_usage_error"))
        self.assertNotIn("zstd", response.json()["detail"]["message"].split("supported:")[1])

    def test_corrupt_truncated_and_trailing_bodies_are_a_400(self):
        gz, zs = gzip.compress(self.RAW), zstd_compress(self.RAW)
        for encoding, content in (
            ("gzip", b"not gzip at all"), ("gzip", gz[:-8]), ("gzip", gz + b"xx"), ("gzip", gz + gz),
            ("gzip", zlib.compress(self.RAW)),
            ("zstd", b"not zstd at all"), ("zstd", zs[:-5]), ("zstd", zs + b"xx"), ("zstd", zs + zs),
        ):
            with self.subTest(encoding=encoding, size=len(content)):
                response, stub, start = self._post(content, encoding)
                _ts_error(response, 400, "api_usage_error")
                self.assertIsNone(start)
                self.assertEqual(stub.calls, [])

    def test_a_zstd_window_over_8_mib_is_a_400(self):
        response, _, _ = self._post(_big_window_zstd(), "zstd")
        self.assertIn("window", _ts_error(response, 400, "api_usage_error"))

    def test_decompression_bombs_are_a_413(self):
        zeros = b"\0" * ((32 << 20) + 1)
        for encoding, content in (("gzip", gzip.compress(zeros)), ("zstd", zstd_compress(zeros))):
            with self.subTest(encoding=encoding):
                response, _, _ = self._post(content, encoding)
                self.assertIn("Decoded", _ts_error(response, 413, "request_too_large"))

    def test_an_oversized_declared_length_is_a_413_before_reading(self):
        decode = patch("app.request_encoding.decode_body")
        with patch("app.request_encoding.MAX_ENCODED_BODY_BYTES", 100), decode as mocked:
            response, _, _ = self._post(gzip.compress(b"{}"), "gzip", headers={"content-length": "101"})
        _ts_error(response, 413, "request_too_large")
        mocked.assert_not_called()

    def test_an_oversized_streamed_body_is_a_413(self):
        with patch("app.request_encoding.MAX_ENCODED_BODY_BYTES", 100):
            response, _, _ = self._post(iter([b"x" * 60, b"y" * 60]), "gzip")
        _ts_error(response, 413, "request_too_large")

    def test_no_credentials_is_a_403_before_any_decoding(self):
        with patch("app.request_encoding.decode_body") as mocked:
            response, _, _ = self._post(gzip.compress(self.RAW), "gzip", app=_typesafe_app(principal=False),
                                        headers={"authorization": ""})
        self.assertIn("API key", _ts_error(response, 403, "authentication_error"))
        mocked.assert_not_called()

    def test_decoding_runs_in_the_threadpool(self):
        from starlette.concurrency import run_in_threadpool as real
        spy = AsyncMock(side_effect=real)
        with patch("app.request_encoding.run_in_threadpool", spy):
            response, _, _ = self._post(gzip.compress(self.RAW), "gzip")
        self.assertEqual(response.status_code, 200)
        spy.assert_awaited_once()

    def test_an_empty_encoded_body_is_passed_on_empty(self):
        response, stub, _ = self._post(b"", "gzip")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(stub.calls, [])

    def test_the_other_ports_leave_encoded_bodies_alone(self):
        app = create_openai_app()
        app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()
        response, _ = _send(app, "POST", "/v1/decisions", content=gzip.compress(b'{"model": "m"}'),
                            headers={"content-type": "application/json", "content-encoding": "gzip"})
        self.assertEqual((response.status_code, response.json()),
                         (400, {"detail": "There was an error parsing the body"}))
        response, _ = _send(app, "POST", "/v1/decisions", content=b'{"model": "m"}',
                            headers={"content-type": "application/json", "content-encoding": "br"})
        self.assertNotEqual(response.status_code, 415)

    def test_invalid_credentials_are_a_401_before_any_decoding(self):
        app = _typesafe_app(principal=False)
        with patch("app.request_encoding.decode_body") as decoded, \
             patch("app.auth.middleware.get_api_key", AsyncMock(return_value=None)), \
             patch("app.auth.middleware.auth_cache.get_cached_api_key", return_value=None):
            response, _ = _send(app, "POST", "/v1/systemone", {KEY: _SystemOneStub(KEY)}, (CANONICAL,),
                                content=gzip.compress(self.RAW),
                                headers={"content-type": "application/json", "content-encoding": "gzip",
                                         "authorization": "Bearer forged-key"})
        _ts_error(response, 401, "authentication_error")
        decoded.assert_not_called()

    def test_a_credential_check_failure_is_a_500_without_decoding(self):
        with patch("app.request_encoding.decode_body") as decoded, \
             patch("app.auth.middleware.credentials_are_valid", AsyncMock(side_effect=RuntimeError("db down"))):
            response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", content=gzip.compress(self.RAW),
                                headers={"content-type": "application/json", "content-encoding": "gzip",
                                         "authorization": "Bearer k"})
        self.assertEqual(_ts_error(response, 500, "api_error"), "Authentication error")
        decoded.assert_not_called()

    def test_uncompressed_bodies_are_not_capped_here(self):
        with patch("app.request_encoding.MAX_ENCODED_BODY_BYTES", 100), \
             patch("app.request_encoding.MAX_DECODED_BODY_BYTES", 100):
            response, stub, _ = self._post(self.RAW)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(len(self.RAW), 100)

    def test_a_missing_zstd_library_is_logged_at_startup(self):
        with patch("app.request_encoding._zstd", None), self.assertLogs("app.main", level="ERROR") as logs:
            create_typesafe_app()
        self.assertIn("no zstd library", logs.output[0])

    def test_requested_coding_unit_cases(self):
        from app.request_encoding import BodyEncodingError, requested_coding
        self.assertIsNone(requested_coding([]))
        self.assertIsNone(requested_coding(["identity"]))
        self.assertEqual(requested_coding(["identity, X-GZIP"]), "x-gzip")
        for values in (["gzip, zstd"], ["gzip", "zstd"], ["br"], ["compress"]):
            with self.subTest(values=values), self.assertRaises(BodyEncodingError) as ctx:
                requested_coding(values)
            self.assertEqual(ctx.exception.status_code, 415)


# ==================== Wire: real CustomProvider through the TypeSafe app ====================

class _SOUpstream:
    """An httpx MockTransport handler for {base}/systemone and {base}/models."""

    def __init__(self, status=200, json_body=None, content=None, content_type="application/json",
                 headers=None, exc=None, models=None):
        self.status = status
        self.json_body = UPSTREAM if json_body is None and content is None else json_body
        self.content = content
        self.content_type = content_type
        self.headers = {"x-typesafe-request-id": "req_up", **(headers or {})}
        self.exc = exc
        self.models = models
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc
        if request.url.path.endswith("/models") and self.models is not None:
            return httpx.Response(200, json=self.models)
        if self.content is not None:
            return httpx.Response(self.status, content=self.content,
                                  headers={"content-type": self.content_type, **self.headers})
        return httpx.Response(self.status, json=self.json_body, headers=self.headers)

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


def _so_provider(apis=(SYSTEMONE_API,), **extra):
    """A custom provider; each listed API gets its own base URL field (OpenAI's
    base_url, systemone_base_url), all http://upstream.test/v1 unless overridden."""
    config = {"provider_type": "custom", "provider_name": "custom", "name": "ts",
              "api_key": "k", "supported_apis": list(apis)}
    if "openai" in apis:
        config["base_url"] = "http://upstream.test/v1"
    if SYSTEMONE_API in apis:
        config["systemone_base_url"] = "http://upstream.test/v1"
    return create_custom_provider({**config, **extra})


def _mock_so(provider, upstream):
    """The provider's own System One client on a mock transport. max_retries is
    deliberately left at the SDK default, so the per-call 0 is what is tested."""
    shared = provider.client is not None and provider.client is provider._systemone_client
    client = provider._systemone_client.with_options(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    provider._systemone_client = client
    if shared:
        provider.client = client
    return provider


class SystemOneWireTests(unittest.TestCase):
    def test_a_zstd_request_reaches_the_upstream_as_plain_json(self):
        for label, app, prefix in _targets():
            with self.subTest(label):
                upstream = _SOUpstream()
                provider = _mock_so(_so_provider(), upstream)
                with _valid_credentials():
                    response, _ = _send(app, "POST", f"{prefix}/v1/systemone", {KEY: provider}, (CANONICAL,),
                                        content=zstd_compress(json.dumps(BODY).encode()),
                                        headers={"content-type": "application/json", "content-encoding": "zstd",
                                                 "authorization": "Bearer test-key",
                                                 "TypeSafe-Danger-Zone-Images": "true", "aoai-foo": "1"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), {**UPSTREAM, "model": "jev-latest"})
                self.assertEqual(response.headers["x-typesafe-request-id"], "req_up")
                sent = upstream.requests[0]
                self.assertEqual(str(sent.url), "http://upstream.test/v1/systemone")
                self.assertNotIn("content-encoding", sent.headers)
                self.assertNotIn("aoai-foo", sent.headers)
                self.assertEqual(sent.headers["typesafe-danger-zone-images"], "true")
                self.assertEqual(upstream.body, {**BODY, "model": "jev-latest"})

    def test_nan_in_the_request_is_a_400_and_never_reaches_upstream(self):
        upstream = _SOUpstream()
        provider = _mock_so(_so_provider(), upstream)
        raw = b'{"model": "jev-latest", "state": "s", "questions": {"q": {"type": "noul", "w": NaN}}}'
        response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", {KEY: provider}, (CANONICAL,),
                            content=raw, headers={"content-type": "application/json"})
        self.assertIn("NaN", _ts_error(response, 400, "invalid_request_error"))
        self.assertEqual(upstream.requests, [])

    def test_unsupported_providers_are_refused_without_an_upstream_call(self):
        upstream = _SOUpstream()
        azure = _wire_azure(_azure(), upstream)
        openai_only = _so_provider(apis=("openai",))
        for key, provider, model in (
            ("azure:test", azure, "azure:test/luna"),
            ("custom:ts", openai_only, CANONICAL),
            ("google:test", GoogleProvider({"api_key": "k", "name": "test"}), "google:test/gemini-3.5-flash"),
        ):
            with self.subTest(key):
                response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", {key: provider}, (model,),
                                    json={**BODY, "model": model})
                self.assertIn("not served over the System One API",
                              _ts_error(response, 400, "invalid_request_error"))
        self.assertEqual(upstream.requests, [])


# ==================== CustomProvider: System One ====================

def _request(model=CANONICAL, **overrides):
    request = SystemOneRequest.model_validate({**BODY, **overrides})
    request.model = model
    return request


class CustomSystemOneProviderTests(unittest.IsolatedAsyncioTestCase):
    async def _call(self, upstream, request=None, headers=None, apis=(SYSTEMONE_API,)):
        provider = _mock_so(_so_provider(apis), upstream)
        return await provider.systemone_create(request or _request(), headers)

    async def test_posts_the_client_body_to_base_url_systemone(self):
        upstream = _SOUpstream()
        data, headers = await self._call(upstream)
        sent = upstream.requests[0]
        self.assertEqual((sent.method, str(sent.url)), ("POST", "http://upstream.test/v1/systemone"))
        self.assertEqual(sent.headers["authorization"], "Bearer k")
        self.assertEqual(sent.headers["content-type"], "application/json")
        self.assertEqual(upstream.body, {**BODY, "model": "jev-latest"})
        self.assertEqual(data, {**UPSTREAM, "model": echo_model_name(_request())})
        self.assertEqual(headers["x-typesafe-request-id"], "req_up")

    async def test_posts_to_the_system_one_url_not_the_openai_one(self):
        upstream = _SOUpstream()
        provider = _mock_so(_so_provider(apis=("openai", SYSTEMONE_API), base_url="http://openai.test/v1",
                                         systemone_base_url="http://typesafe.test/v1"), upstream)
        await provider.systemone_create(_request())
        self.assertEqual(str(upstream.requests[0].url), "http://typesafe.test/v1/systemone")

    async def test_only_the_given_headers_travel(self):
        upstream = _SOUpstream()
        with _ContextVar(extra_request_headers, {"aoai-foo": "1"}):
            await self._call(upstream, headers={"TypeSafe-Danger-Zone-Long-Context": "true"})
        sent = upstream.requests[0]
        self.assertEqual(sent.headers["typesafe-danger-zone-long-context"], "true")
        self.assertNotIn("aoai-foo", sent.headers)

    async def test_the_model_is_echoed_and_filled_when_missing(self):
        upstream = _SOUpstream(json_body={"answers": {}, "usage": {}})
        data, _ = await self._call(upstream)
        self.assertEqual(data["model"], echo_model_name(_request()))

    async def test_the_proxy_does_not_retry(self):
        upstream = _SOUpstream(status=503, json_body={"detail": {"error_type": "overloaded_error", "message": "x"}})
        with self.assertRaises(ProviderHTTPError) as ctx:
            await self._call(upstream)
        self.assertEqual(len(upstream.requests), 1)
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_json_object_error_bodies_are_kept_verbatim(self):
        for status, body in (
            (403, {"detail": {"error_type": "authentication_error", "message": "bad key"}}),
            (422, {"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}),
            (429, {"error": {"code": 429, "message": "slow", "metadata": {}}, "user_id": "u"}),
            (500, {"error": "x"}),
        ):
            with self.subTest(status=status):
                upstream = _SOUpstream(status=status, json_body=body, headers={"retry-after-ms": "9"})
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await self._call(upstream)
                self.assertEqual((ctx.exception.status_code, ctx.exception.body), (status, body))
                self.assertEqual(ctx.exception.headers["x-typesafe-request-id"], "req_up")
                self.assertEqual(ctx.exception.headers["retry-after-ms"], "9")

    async def test_non_json_error_bodies_get_the_generic_envelope(self):
        upstream = _SOUpstream(status=502, content=b"<html>bad gateway</html>", content_type="text/html")
        with self.assertRaises(ProviderHTTPError) as ctx:
            await self._call(upstream)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(ctx.exception.body["error"]["type"], "upstream_error")

    async def test_malformed_successes_are_a_502(self):
        for content, content_type in (
            (b"", "application/json"), (b"<html>", "text/html"), (b"null", "application/json"),
            (b"[1]", "application/json"), (b"ok", "text/plain"),
            (b'{"model": "m", "usage": {}}', "application/json"), (b'{"answers": []}', "application/json"),
        ):
            with self.subTest(content=content):
                upstream = _SOUpstream(content=content, content_type=content_type)
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await self._call(upstream)
                self.assertEqual(ctx.exception.status_code, 502)
                self.assertEqual(ctx.exception.body["detail"]["error_type"], "api_error")
                self.assertEqual(ctx.exception.headers["x-typesafe-request-id"], "req_up")

    async def test_nan_and_unpaired_surrogates_are_rejected_before_any_call(self):
        for overrides, needle in (({"state": float("nan")}, "NaN"), ({"state": "\ud800"}, "surrogate")):
            with self.subTest(needle):
                upstream = _SOUpstream()
                with self.assertRaises(ValueError) as ctx:
                    await self._call(upstream, request=_request(**overrides))
                self.assertIn(needle, str(ctx.exception))
                self.assertEqual(upstream.requests, [])

    async def test_a_timeout_is_a_504(self):
        upstream = _SOUpstream(exc=httpx.ReadTimeout("slow"))
        with self.assertRaises(ProviderHTTPError) as ctx:
            await self._call(upstream)
        self.assertEqual(ctx.exception.status_code, 504)
        self.assertEqual(len(upstream.requests), 1)


class CustomSystemOneConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_system_one_only_provider_serves_nothing_else(self):
        provider = _so_provider()
        self.assertIsNone(provider.client)
        self.assertIsNotNone(provider._systemone_client)
        self.assertEqual(provider.get_supported_apis(), [SYSTEMONE_API])
        self.assertTrue(provider.supports_api_for_model(CANONICAL, SYSTEMONE_API))
        self.assertFalse(provider.supports_api_for_model(CANONICAL, "openai"))
        self.assertFalse(provider.supports_api_for_model(CANONICAL, "anthropic"))
        self.assertEqual(provider.get_anthropic_mode_for_model(CANONICAL), "unsupported")
        self.assertEqual(provider.get_supported_endpoints(), ["/v1/systemone"])
        with self.assertRaises(NotImplementedError):
            await provider.decisions_create(SimpleNamespace(model=CANONICAL))

    async def test_system_one_has_its_own_base_url(self):
        provider = _so_provider(apis=("openai", SYSTEMONE_API), base_url="http://openai.test/v1",
                                systemone_base_url="http://typesafe.test/v1")
        self.assertIsNot(provider.client, provider._systemone_client)
        self.assertEqual(str(provider._systemone_client.base_url), "http://typesafe.test/v1/")
        self.assertEqual(str(provider.client.base_url), "http://openai.test/v1/")
        only = _so_provider()
        self.assertEqual(only.base_url, "http://upstream.test/v1")  # the System One URL stands in for base_url

    async def test_rows_without_a_system_one_url_fall_back_to_base_url(self):
        provider = create_custom_provider({"provider_type": "custom", "name": "ts", "api_key": "k",
                                           "base_url": "http://legacy.test/v1", "supported_apis": [SYSTEMONE_API]})
        self.assertEqual(str(provider._systemone_client.base_url), "http://legacy.test/v1/")
        upstream = _SOUpstream()
        await _mock_so(provider, upstream).systemone_create(_request())
        self.assertEqual(str(upstream.requests[0].url), "http://legacy.test/v1/systemone")

    async def test_chat_apis_never_use_the_system_one_url(self):
        for apis in (("openai",), ("openai", SYSTEMONE_API), ("anthropic", SYSTEMONE_API)):
            with self.subTest(apis=apis):
                provider = create_custom_provider({
                    "provider_type": "custom", "name": "ts", "api_key": "k",
                    "systemone_base_url": "https://typesafe.test/v1", "supported_apis": list(apis),
                })
                self.assertIsNone(provider.client)
                self.assertIsNone(provider._anthropic_client)
                self.assertEqual(provider.get_supported_apis(), [a for a in apis if a == SYSTEMONE_API])

    async def test_supported_apis_default_to_the_filled_urls(self):
        from app.providers.provider_service import custom_supported_apis_from_urls, infer_custom_supported_apis
        self.assertEqual(custom_supported_apis_from_urls({"systemone_base_url": "https://t/v1"}), [SYSTEMONE_API])
        self.assertEqual(custom_supported_apis_from_urls(
            {"base_url": "https://o/v1", "anthropic_base_url": "https://a", "systemone_base_url": "https://t/v1"}),
            ["openai", "anthropic", SYSTEMONE_API])
        self.assertEqual(custom_supported_apis_from_urls({}), ["openai"])
        body = SimpleNamespace(provider_type="custom", supported_apis=None, systemone_base_url="https://t/v1")
        infer_custom_supported_apis(body)
        self.assertEqual(body.supported_apis, [SYSTEMONE_API])
        kept = SimpleNamespace(provider_type="custom", supported_apis=["openai"], systemone_base_url="https://t/v1")
        infer_custom_supported_apis(kept)
        self.assertEqual(kept.supported_apis, ["openai"])

    async def test_an_invalid_system_one_url_fails_the_build(self):
        with self.assertRaises(ValueError):
            _so_provider(apis=("openai", SYSTEMONE_API), systemone_base_url="ftp://typesafe.test/v1")

    async def test_openai_and_system_one_share_one_client(self):
        provider = _so_provider(apis=("openai", SYSTEMONE_API))
        self.assertIs(provider.client, provider._systemone_client)
        self.assertIn("/v1/systemone", provider.get_supported_endpoints())
        self.assertIn("/v1/decisions", provider.get_supported_endpoints())

    async def test_system_one_needs_the_base_url(self):
        provider = create_custom_provider({
            "provider_type": "custom", "name": "ts", "anthropic_base_url": "https://up.test/anthropic",
            "supported_apis": [SYSTEMONE_API, "anthropic"],
        })
        self.assertIsNone(provider._systemone_client)
        self.assertFalse(provider.supports_api_for_model(CANONICAL, SYSTEMONE_API))

    async def test_providers_without_system_one_refuse_without_a_call(self):
        for apis in (("openai",), ("anthropic",)):
            with self.subTest(apis=apis):
                provider = _so_provider(apis=apis, anthropic_base_url="https://up.test/anthropic")
                self.assertFalse(provider.supports_api_for_model(CANONICAL, SYSTEMONE_API))
                with self.assertRaises(NotImplementedError):
                    await provider.systemone_create(_request())

    async def test_fake_subclasses_without_init_are_not_capable(self):
        class Fake(CustomProvider):
            def __init__(self):
                pass
        self.assertIsNone(Fake()._systemone_client)

    async def test_close_provider_clients_and_remove_provider_close_the_client(self):
        provider = _so_provider()
        self.assertFalse(provider._systemone_client.is_closed())
        await provider_manager.close_provider_clients({KEY: provider})
        self.assertTrue(provider._systemone_client.is_closed())

        shared = _so_provider(apis=("openai", SYSTEMONE_API))
        await provider_manager.close_provider_clients({KEY: shared})
        self.assertTrue(shared._systemone_client.is_closed())

        for apis in ((SYSTEMONE_API,), ("openai", SYSTEMONE_API)):
            with self.subTest(apis=apis):
                provider = _so_provider(apis=apis)
                with _Inventory({KEY: provider}, (CANONICAL,)):
                    self.assertTrue(await provider_manager.remove_provider(KEY))
                self.assertTrue(provider._systemone_client.is_closed())

    async def test_byok_updates_keep_a_base_url_for_system_one(self):
        from app.routes.user_providers import _check_systemone_base_url
        stored = SimpleNamespace(supported_apis='["anthropic"]', base_url=None)
        with self.assertRaises(HTTPException) as ctx:
            _check_systemone_base_url(SimpleNamespace(supported_apis=["systemone", "anthropic"]), stored, {})
        self.assertEqual(ctx.exception.status_code, 422)
        with self.assertRaises(HTTPException):
            _check_systemone_base_url(SimpleNamespace(supported_apis=None),
                                      SimpleNamespace(supported_apis='["systemone"]', base_url="https://up/v1"),
                                      {"base_url": ""})
        _check_systemone_base_url(SimpleNamespace(supported_apis=None),
                                  SimpleNamespace(supported_apis='["systemone"]', base_url="https://up/v1"), {})
        _check_systemone_base_url(SimpleNamespace(supported_apis=["anthropic"]), stored, {})
        _check_systemone_base_url(SimpleNamespace(supported_apis=["systemone"]), stored,
                                  {"systemone_base_url": "https://typesafe.test/v1"})
        with self.assertRaises(HTTPException):
            _check_systemone_base_url(
                SimpleNamespace(supported_apis=None),
                SimpleNamespace(supported_apis='["systemone"]', base_url=None, systemone_base_url="https://t/v1"),
                {"systemone_base_url": ""})

    async def test_byok_system_one_needs_a_base_url(self):
        from app.routes.user_providers import _check_required_credentials
        with self.assertRaises(HTTPException) as ctx:
            _check_required_credentials("custom", {
                "base_url": "", "anthropic_base_url": "https://up.test/anthropic",
                "supported_apis": [SYSTEMONE_API, "anthropic"],
            })
        self.assertEqual(ctx.exception.status_code, 422)
        _check_required_credentials("custom", {"base_url": "https://up.test/v1", "supported_apis": [SYSTEMONE_API]})
        _check_required_credentials("custom", {"systemone_base_url": "https://up.test/v1",
                                               "supported_apis": [SYSTEMONE_API]})
        _check_required_credentials("custom", {"anthropic_base_url": "https://up.test/a", "supported_apis": ["anthropic"]})
        with self.assertRaises(HTTPException):
            _check_required_credentials("custom", {"supported_apis": [SYSTEMONE_API]})


class SystemOneDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for store in (custom_providers._SYSTEMONE_META, custom_providers._CHAT_LISTED):
            saved = dict(store)
            self.addCleanup(lambda store=store, saved=saved: (store.clear(), store.update(saved)))

    async def test_a_typesafe_shaped_listing_is_discovered_with_its_metadata(self):
        upstream = _SOUpstream(models={"models": [
            {"name": "jev-1.14.0", "description": "Pinned.", "release_date": "2026-10-14"},
            {"name": "jev-latest", "description": "General-purpose system one model.", "release_date": "bad"},
            {"name": "jev-latest", "description": "dup", "release_date": "2026-01-01"},
        ]})
        models = await _mock_so(_so_provider(), upstream).get_available_models()
        self.assertEqual([m.id for m in models], [f"{KEY}/jev-1.14.0", CANONICAL])
        self.assertEqual(len(upstream.requests), 1)
        sent = upstream.requests[0]
        self.assertEqual((sent.method, str(sent.url)), ("GET", "http://upstream.test/v1/models"))
        self.assertEqual(sent.headers["authorization"], "Bearer k")
        self.assertEqual(custom_providers.systemone_model_meta(f"{KEY}/jev-1.14.0"), ("Pinned.", "2026-10-14"))
        self.assertEqual(custom_providers.systemone_model_meta(CANONICAL),
                         ("General-purpose system one model.", None))

    async def test_metadata_is_replaced_per_listing_bounded_and_forgotten(self):
        upstream = _SOUpstream(models={"models": [
            {"name": "jev-old", "description": "x" * 5000, "release_date": "2026-01-01"},
        ]})
        provider = _mock_so(_so_provider(), upstream)
        await provider.get_available_models()
        description, _ = custom_providers.systemone_model_meta(f"{KEY}/jev-old")
        self.assertEqual(len(description), 1024)
        upstream.models = {"models": [{"name": "jev-new", "description": "d", "release_date": "2026-02-02"}]}
        await provider.get_available_models()
        self.assertEqual(custom_providers.systemone_model_meta(f"{KEY}/jev-old"), (None, None))
        self.assertEqual(custom_providers.systemone_model_meta(f"{KEY}/jev-new"), ("d", "2026-02-02"))
        upstream.exc = httpx.ConnectError("down")  # a failed listing keeps the last good one
        with self.assertLogs("app.providers.custom_providers", level="WARNING"):
            await provider.get_available_models()
        self.assertEqual(custom_providers.systemone_model_meta(f"{KEY}/jev-new"), ("d", "2026-02-02"))
        with _Inventory({KEY: provider}, (f"{KEY}/jev-new",)):
            await provider_manager.remove_provider(KEY)
        self.assertEqual(custom_providers.systemone_model_meta(f"{KEY}/jev-new"), (None, None))
        with patch.object(custom_providers, "_MAX_META_MODELS", 2):
            upstream.exc = None
            upstream.models = {"models": [{"name": f"m{i}"} for i in range(5)]}
            with self.assertLogs("app.providers.custom_providers", level="WARNING"):
                models = await _mock_so(_so_provider(), upstream).get_available_models()
        self.assertEqual(len(models), 2)

    async def test_an_openai_shaped_listing_works_too(self):
        upstream = _SOUpstream(models={"object": "list", "data": [{"id": "jev-latest", "created": 1760400000}]})
        models = await _mock_so(_so_provider(), upstream).get_available_models()
        self.assertEqual([m.id for m in models], [CANONICAL])
        self.assertEqual(custom_providers.systemone_model_meta(CANONICAL), (None, "2025-10-14"))

    async def test_a_provider_ticked_for_both_apis_lists_once_without_warnings(self):
        upstream = _SOUpstream(models={"models": [{"name": "jev-latest", "description": "d",
                                                   "release_date": "2026-09-15"}]})
        provider = _mock_so(_so_provider(apis=("openai", SYSTEMONE_API)), upstream)
        with self.assertNoLogs("app.providers.openai_compatible", level="WARNING"):
            models = await provider.get_available_models()
        self.assertEqual([m.id for m in models], [CANONICAL])
        self.assertEqual(len(upstream.requests), 1)

    async def test_separate_urls_are_both_listed_and_merged(self):
        openai_up = _SOUpstream(models={"object": "list", "data": [{"id": "gpt-x"}, {"id": "shared"}]})
        ts_up = _SOUpstream(models={"models": [{"name": "jev-latest", "description": "d", "release_date": "2026-09-15"},
                                               {"name": "shared", "description": "s", "release_date": "2026-01-01"}]})
        provider = _so_provider(apis=("openai", SYSTEMONE_API), base_url="http://openai.test/v1",
                                systemone_base_url="http://typesafe.test/v1")
        provider.client = provider.client.with_options(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(openai_up)))
        provider._systemone_client = provider._systemone_client.with_options(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(ts_up)))
        models = await provider.get_available_models()
        self.assertEqual([m.id for m in models], [f"{KEY}/gpt-x", f"{KEY}/shared", CANONICAL])
        self.assertEqual((str(openai_up.requests[0].url), str(ts_up.requests[0].url)),
                         ("http://openai.test/v1/models", "http://typesafe.test/v1/models"))
        self.assertEqual(custom_providers.systemone_model_meta(CANONICAL), ("d", "2026-09-15"))
        # Each model is offered on the API whose listing named it.
        self.assertEqual(provider.get_supported_apis_for_model(CANONICAL), [SYSTEMONE_API])
        self.assertNotIn(SYSTEMONE_API, provider.get_supported_apis_for_model(f"{KEY}/gpt-x"))
        self.assertIn("openai", provider.get_supported_apis_for_model(f"{KEY}/gpt-x"))
        self.assertEqual(set(provider.get_supported_apis_for_model(f"{KEY}/shared")),
                         {"openai", "anthropic", SYSTEMONE_API})
        self.assertEqual(set(provider.get_supported_apis_for_model(f"{KEY}/unlisted")),
                         {"openai", "anthropic", SYSTEMONE_API})

    def _stub_listings(self, provider, openai=None, anthropic=None, systemone=None):
        calls = []

        def fetcher(name, ids):
            async def fetch():
                calls.append(name)
                if ids is None:
                    return None
                models = [provider.create_model_info(i, "custom") for i in ids]
                if name == "systemone":
                    custom_providers._SYSTEMONE_META[provider.full_provider_name] = {
                        m.id: (None, None) for m in models}
                return models
            return fetch
        provider._fetch_openai_models = fetcher("openai", openai)
        provider._fetch_anthropic_models = fetcher("anthropic", anthropic)
        provider._fetch_systemone_models = fetcher("systemone", systemone)
        return calls

    async def test_an_anthropic_url_next_to_a_system_one_url_keeps_its_models(self):
        for systemone in (["jev-latest"], []):
            with self.subTest(systemone=systemone):
                provider = _so_provider(apis=("anthropic", SYSTEMONE_API),
                                        anthropic_base_url="https://vendor.test/anthropic",
                                        systemone_base_url="https://typesafe.test/v1")
                calls = self._stub_listings(provider, openai=None, anthropic=["claude-x"], systemone=systemone)
                models = await provider.get_available_models()
                self.assertEqual([m.id for m in models], [f"{KEY}/claude-x"] + [f"{KEY}/{i}" for i in systemone])
                self.assertEqual(calls, ["openai", "anthropic", "systemone"])
                self.assertEqual(provider.get_supported_apis_for_model(f"{KEY}/claude-x"), ["anthropic"])
                if systemone:
                    self.assertEqual(provider.get_supported_apis_for_model(CANONICAL), [SYSTEMONE_API])
                    self.assertEqual(provider.get_anthropic_mode_for_model(CANONICAL), "unsupported")

    async def test_a_failed_openai_listing_still_falls_back_to_anthropic(self):
        provider = _so_provider(apis=("openai", "anthropic", SYSTEMONE_API), base_url="https://openai.test/v1",
                                anthropic_base_url="https://vendor.test/anthropic",
                                systemone_base_url="https://typesafe.test/v1")
        self._stub_listings(provider, openai=None, anthropic=["claude-x"], systemone=["jev-latest"])
        models = await provider.get_available_models()
        self.assertEqual([m.id for m in models], [f"{KEY}/claude-x", CANONICAL])

    async def test_capability_stays_provider_wide_until_both_listings_are_known(self):
        provider = _so_provider(apis=("openai", SYSTEMONE_API), base_url="https://openai.test/v1",
                                systemone_base_url="https://typesafe.test/v1")
        custom_providers.forget_systemone_meta(provider.full_provider_name)
        self.assertIn(SYSTEMONE_API, provider.get_supported_apis_for_model(f"{KEY}/gpt-x"))
        self.assertIn("openai", provider.get_supported_apis_for_model(CANONICAL))

    async def test_a_failing_or_garbage_listing_is_empty(self):
        for upstream in (_SOUpstream(models=["nope"]), _SOUpstream(exc=httpx.ConnectError("down"))):
            with self.subTest(upstream=upstream):
                with self.assertLogs("app.providers.custom_providers", level="WARNING"):
                    models = await _mock_so(_so_provider(), upstream).get_available_models()
                self.assertEqual(models, [])


class SystemOneBaseUrlPlumbingTests(unittest.TestCase):
    """systemone_base_url is stored, validated and guarded like anthropic_base_url."""

    def test_models_validate_and_carry_the_field(self):
        from pydantic import ValidationError
        from app.auth.models import (
            ProviderCredentials, ProviderCredentialsCreate, ProviderCredentialsResponse,
            ProviderCredentialsUpdate,
        )
        self.assertIn("systemone_base_url", ProviderCredentials.__table__.columns)
        self.assertIn("systemone_base_url", ProviderCredentialsResponse.model_fields)
        for model in (ProviderCredentialsCreate, ProviderCredentialsUpdate):
            with self.subTest(model.__name__):
                fields = {"provider_type": "custom", "instance_name": "ts", "provider_name": "custom"} \
                    if model is ProviderCredentialsCreate else {}
                ok = model(**fields, systemone_base_url=" https://api.typesafe.ai/v1 ")
                self.assertEqual(ok.systemone_base_url, "https://api.typesafe.ai/v1")
                with self.assertRaises(ValidationError):
                    model(**fields, systemone_base_url="file:///etc/passwd")

    def test_the_url_is_ssrf_guarded_for_byok_and_clearable(self):
        from app.auth.database import _CLEARABLE_PROVIDER_FIELDS
        from app.providers.url_guard import URL_FIELDS
        self.assertIn("systemone_base_url", URL_FIELDS)
        self.assertIn("systemone_base_url", _CLEARABLE_PROVIDER_FIELDS)

    def test_the_provider_config_and_response_carry_it(self):
        cred = SimpleNamespace(
            provider_key=KEY, provider_type="custom", instance_name="ts", provider_name="custom",
            enabled=True, owner_user_id=None, api_key="k", endpoint=None, base_url=None,
            supported_apis='["systemone"]', anthropic_base_url=None, anthropic_auth_style=None,
            systemone_base_url="https://api.typesafe.ai/v1",
        )
        config = provider_manager._create_provider_config(cred)
        self.assertEqual(config["systemone_base_url"], "https://api.typesafe.ai/v1")
        provider = create_custom_provider({**config, "custom_provider_name": "custom", "name": "ts"})
        self.assertEqual(str(provider._systemone_client.base_url), "https://api.typesafe.ai/v1/")


# ==================== Azure Foundry: Microsoft-Decision-1 ====================

FOUNDRY_ROWS = [
    {"deployments": ["Decision-1"], "apis": ["systemone"]},
    {"deployments": ["gpt-x"], "apis": ["chat"]},
    {"deployments": ["both"], "apis": ["chat", "systemone"]},
]
FOUNDRY_KEY = "azure:foundry"
DECISION_ONE = f"{FOUNDRY_KEY}/Decision-1"


def _foundry(rows=FOUNDRY_ROWS, endpoint="https://res.services.ai.azure.com/api/projects/p1"):
    from app.providers.azure_deployments import build_azure_config_fields, serialize_azure_deployments
    from app.providers.azure_provider import AzureProvider
    creds = SimpleNamespace(
        endpoint=endpoint, api_key="azure-key", discovery_api_version=None, azure_backend="foundry",
        dynamic_discovery=False, deployments_json=serialize_azure_deployments(deployment_groups=rows),
    )
    return AzureProvider({"name": "foundry", **build_azure_config_fields(creds)})


def _wire_foundry(provider, upstream):
    """The provider's own v1 client on a mock transport (max_retries left at the
    SDK default, so the relay's per-call 0 is what is tested)."""
    client = provider._v1_client.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    provider._v1_client = provider._responses_client = provider.client = client
    return provider


DECISION_ONE_UPSTREAM = {
    "model": "microsoft-decision-1",
    "answers": {"team": {"type": "choice", "choice": "engineering", "confidence": 0.9,
                         "probabilities": {"billing": 0.05, "engineering": 0.9, "support": 0.05}}},
    "usage": {"input_tokens": 40, "output_tokens": 1},
}


class AzureDecisionOneTests(unittest.IsolatedAsyncioTestCase):
    def test_systemone_is_a_deployment_api(self):
        from app.providers.azure_deployments import DEPLOYMENT_APIS, normalize_deployment_groups
        self.assertIn("systemone", DEPLOYMENT_APIS)
        self.assertEqual(normalize_deployment_groups(FOUNDRY_ROWS[:1])[0]["apis"], ["systemone"])

    def test_rows_decide_where_a_deployment_is_offered(self):
        provider = _foundry()
        self.assertEqual(provider.get_supported_apis_for_model(DECISION_ONE), [SYSTEMONE_API])
        self.assertNotIn(SYSTEMONE_API, provider.get_supported_apis_for_model(f"{FOUNDRY_KEY}/gpt-x"))
        self.assertIn(SYSTEMONE_API, provider.get_supported_apis_for_model(f"{FOUNDRY_KEY}/both"))
        self.assertIn("openai", provider.get_supported_apis_for_model(f"{FOUNDRY_KEY}/both"))
        self.assertNotIn(SYSTEMONE_API, provider.get_supported_apis_for_model(f"{FOUNDRY_KEY}/unlisted"))
        self.assertIn("/v1/systemone", provider.get_supported_endpoints())
        self.assertNotIn("/v1/systemone", _foundry(rows=FOUNDRY_ROWS[1:2]).get_supported_endpoints())

    async def test_posts_to_the_foundry_path_with_the_api_key_header(self):
        upstream = _SOUpstream(json_body=DECISION_ONE_UPSTREAM)
        provider = _wire_foundry(_foundry(), upstream)
        request = _request(DECISION_ONE)
        data, headers = await provider.systemone_create(request, {"TypeSafe-Danger-Zone-Images": "true"})
        sent = upstream.requests[0]
        self.assertEqual((sent.method, str(sent.url)),
                         ("POST", "https://res.services.ai.azure.com/providers/microsoft/v1/systemone"))
        self.assertEqual(sent.headers["api-key"], "azure-key")
        self.assertNotIn("authorization", sent.headers)
        self.assertEqual(upstream.body, {**BODY, "model": "Decision-1"})  # the deployment name
        self.assertEqual(data, {**DECISION_ONE_UPSTREAM, "model": echo_model_name(request)})
        self.assertEqual(headers["x-typesafe-request-id"], "req_up")

    async def test_errors_are_relayed_and_never_retried(self):
        body = {"error": {"code": "429", "message": "Rate limit exceeded"}}
        upstream = _SOUpstream(status=429, json_body=body, headers={"retry-after": "2"})
        provider = _wire_foundry(_foundry(), upstream)
        with self.assertRaises(ProviderHTTPError) as ctx:
            await provider.systemone_create(_request(DECISION_ONE))
        self.assertEqual((ctx.exception.status_code, ctx.exception.body), (429, body))
        self.assertEqual(len(upstream.requests), 1)

    async def test_deployments_without_the_row_are_refused_without_a_call(self):
        upstream = _SOUpstream()
        provider = _wire_foundry(_foundry(), upstream)
        for model in (f"{FOUNDRY_KEY}/gpt-x", f"{FOUNDRY_KEY}/unlisted"):
            with self.subTest(model=model), self.assertRaises(NotImplementedError):
                await provider.systemone_create(_request(model))
        self.assertEqual(upstream.requests, [])


class AzureDecisionOneConfigTests(unittest.TestCase):
    def test_the_url_keeps_path_prefixes_and_trims_project_scopes(self):
        for endpoint, url in (
            ("https://res.services.ai.azure.com", "https://res.services.ai.azure.com/providers/microsoft/v1/systemone"),
            ("https://res.services.ai.azure.com/api/projects/p1/",
             "https://res.services.ai.azure.com/providers/microsoft/v1/systemone"),
            ("https://apim.azure-api.net/foundry-gw", "https://apim.azure-api.net/foundry-gw/providers/microsoft/v1/systemone"),
            ("https://apim.azure-api.net/gw/api/projects/p1", "https://apim.azure-api.net/gw/providers/microsoft/v1/systemone"),
        ):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(_foundry(endpoint=endpoint)._systemone_url(), url)

    def test_system_one_is_foundry_only(self):
        from app.providers.azure_deployments import build_azure_config_fields, serialize_azure_deployments
        from app.providers.azure_provider import AzureProvider
        rows = [{"deployments": ["Decision-1"], "apis": ["systemone"]},
                {"deployments": ["mixed"], "apis": ["anthropic", "systemone"]},
                {"deployments": ["gpt-x"], "apis": ["chat"]}]
        creds = SimpleNamespace(endpoint="https://res.openai.azure.com", api_key="k", discovery_api_version=None,
                                azure_backend="openai", dynamic_discovery=False,
                                deployments_json=serialize_azure_deployments(deployment_groups=rows))
        provider = AzureProvider({"name": "aoai", **build_azure_config_fields(creds)})
        self.assertEqual(provider._get_manual_model_names(), ["gpt-x"])  # as a /messages-only row on this backend
        for model in ("azure:aoai/Decision-1", "azure:aoai/mixed"):
            with self.subTest(model=model):
                self.assertNotIn(SYSTEMONE_API, provider.get_supported_apis_for_model(model))
        self.assertNotIn("/v1/systemone", provider.get_supported_endpoints())

    def test_rows_validate_strictly_and_infer_the_badge(self):
        from app.auth.models import AzureDeploymentGroup
        from app.providers.azure_deployments import normalize_deployment_groups
        from app.providers.provider_service import infer_azure_supported_apis
        self.assertEqual(normalize_deployment_groups(FOUNDRY_ROWS, strict=True)[0]["apis"], ["systemone"])
        self.assertEqual(AzureDeploymentGroup(deployments=["Decision-1"], apis=["systemone"]).apis, ["systemone"])
        data = SimpleNamespace(provider_type="azure", supported_apis=None, azure_backend="foundry",
                               anthropic_deployments=[], deployment_groups=FOUNDRY_ROWS)
        infer_azure_supported_apis(data)
        self.assertIn(SYSTEMONE_API, data.supported_apis)


class AzureDecisionOneRouteTests(unittest.TestCase):
    def test_the_typesafe_port_serves_a_decision_one_row(self):
        for label, app, prefix in _targets():
            with self.subTest(label):
                upstream = _SOUpstream(json_body=DECISION_ONE_UPSTREAM)
                provider = _wire_foundry(_foundry(), upstream)
                response, _ = _send(app, "POST", f"{prefix}/v1/systemone", {FOUNDRY_KEY: provider},
                                    (DECISION_ONE, f"{FOUNDRY_KEY}/gpt-x"), json={**BODY, "model": "Decision-1"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["answers"], DECISION_ONE_UPSTREAM["answers"])
                self.assertEqual(response.json()["model"], "Decision-1")
                self.assertEqual(upstream.body["model"], "Decision-1")

    def test_a_chat_row_is_refused_and_not_listed(self):
        upstream = _SOUpstream()
        provider = _wire_foundry(_foundry(), upstream)
        models = (DECISION_ONE, f"{FOUNDRY_KEY}/gpt-x")
        response, _ = _send(_typesafe_app(), "POST", "/v1/systemone", {FOUNDRY_KEY: provider}, models,
                            json={**BODY, "model": f"{FOUNDRY_KEY}/gpt-x"})
        self.assertIn("not served over the System One API", _ts_error(response, 400, "invalid_request_error"))
        response, _ = _send(_typesafe_app(), "GET", "/v1/models", {FOUNDRY_KEY: provider}, models)
        self.assertEqual([m["name"] for m in response.json()["models"]], [DECISION_ONE])
        self.assertEqual(upstream.requests, [])

    def test_a_decision_one_row_stays_off_the_other_apis(self):
        upstream = _SOUpstream()
        provider = _wire_foundry(_foundry(), upstream)
        app = create_openai_app()
        app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()
        with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)):
            chat, _ = _send(app, "POST", "/v1/chat/completions", {FOUNDRY_KEY: provider}, (DECISION_ONE,),
                            json={"model": "Decision-1", "messages": [{"role": "user", "content": "hi"}]})
            decisions, _ = _send(app, "POST", "/v1/decisions", {FOUNDRY_KEY: provider}, (DECISION_ONE,),
                                 json={"model": DECISION_ONE, "input": "x", "questions": []})
            listing, _ = _send(app, "GET", "/v1/models", {FOUNDRY_KEY: provider}, (DECISION_ONE,))
        self.assertEqual(chat.status_code, 404, chat.text)
        self.assertEqual(decisions.status_code, 400, decisions.text)
        self.assertEqual(listing.json()["data"], [])
        self.assertEqual(upstream.requests, [])

    def test_a_decision_one_row_is_not_on_the_anthropic_or_azure_ports(self):
        from app.auth.middleware import authenticate_anthropic_request
        from app.main import create_anthropic_app
        from app.routes.azure_openai import _authenticate_azure
        upstream = _SOUpstream()
        provider = _wire_foundry(_foundry(), upstream)
        anthropic = create_anthropic_app()
        anthropic.dependency_overrides[authenticate_anthropic_request] = lambda: _Caller()
        azure = create_azure_openai_app()
        azure.dependency_overrides[_authenticate_azure] = lambda: _Caller()
        with patch("app.rate_limit_dep.charge_with_slot", AsyncMock(return_value=None)):
            models, _ = _send(anthropic, "GET", "/v1/models", {FOUNDRY_KEY: provider}, (DECISION_ONE,))
            messages, _ = _send(anthropic, "POST", "/v1/messages", {FOUNDRY_KEY: provider}, (DECISION_ONE,),
                                json={"model": "Decision-1", "max_tokens": 5,
                                      "messages": [{"role": "user", "content": "hi"}]})
            azure_models, _ = _send(azure, "GET", "/openai/models", {FOUNDRY_KEY: provider}, (DECISION_ONE,))
        self.assertEqual(models.json()["data"], [])
        self.assertEqual(messages.status_code, 404, messages.text)
        self.assertEqual(azure_models.json()["data"], [])
        self.assertEqual(upstream.requests, [])


class UnsupportedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_other_provider_types_refuse_system_one(self):
        for cls in (BedrockProvider, PlatformAIProvider, GoogleProvider):
            with self.subTest(provider=cls.__name__):
                self.assertIs(cls.systemone_create, BaseProvider.systemone_create)
        google = GoogleProvider({"api_key": "k", "name": "test"})
        self.assertFalse(google.supports_api_for_model("google:test/gemini-3.5-flash", SYSTEMONE_API))
        self.assertFalse(_azure().supports_api_for_model("azure:test/luna", SYSTEMONE_API))
        with self.assertRaises(NotImplementedError):
            await google.systemone_create(_request("google:test/gemini-3.5-flash"))


# ==================== Shared helpers ====================

class TranslatorTests(unittest.TestCase):
    def _status_error(self, status, content):
        import openai
        request = httpx.Request("POST", "http://upstream.test/v1/systemone")
        response = httpx.Response(status, content=content, request=request,
                                  headers={"content-type": "application/json"})
        body = None
        try:
            body = json.loads(content)
        except ValueError:
            pass
        if isinstance(body, dict):
            body = body.get("error", body)
        return openai.APIStatusError("x", response=response, body=body)

    def test_relay_keeps_any_json_object_and_the_default_path_does_not(self):
        detail = {"detail": {"error_type": "authentication_error", "message": "bad key"}}
        for content, expected in ((json.dumps(detail).encode(), detail),
                                  (b'{"error": "plain string"}', {"error": "plain string"})):
            with self.subTest(content=content):
                error = _translate_openai_sdk_error(self._status_error(401, content), "custom",
                                                    relay_upstream_body=True)
                self.assertEqual((error.status_code, error.body), (401, expected))
        default = _translate_openai_sdk_error(self._status_error(401, json.dumps(detail).encode()), "custom")
        self.assertEqual(default.body["error"]["type"], "upstream_error")

    def test_the_decisions_surrogate_message_is_unchanged(self):
        from app.providers.openai_compatible import encode_passthrough_body
        with self.assertRaises(ValueError) as default:
            encode_passthrough_body({"input": "\ud800"})
        self.assertIn("NaN or Infinity", str(default.exception))
        with self.assertRaises(ValueError) as system_one:
            encode_passthrough_body({"input": "\ud800"}, surrogate_message=True)
        self.assertIn("unpaired Unicode surrogate", str(system_one.exception))

    def test_typesafe_body_conversion(self):
        self.assertEqual(_errors.to_typesafe_body({"detail": "x"}, 400), {"detail": "x"})
        self.assertEqual(_errors.to_typesafe_body({"type": "error", "error": {"type": "x", "message": "m"}}, 404),
                         {"detail": {"error_type": "not_found_error", "message": "m"}})
        self.assertEqual(_errors.to_typesafe_body(None, 502, "fallback"),
                         {"detail": {"error_type": "api_error", "message": "fallback"}})
        self.assertEqual(_errors.typesafe_error_type(599), "api_error")
        self.assertEqual(_errors.typesafe_error_type(418), "invalid_request_error")
        state = SimpleNamespace(state=SimpleNamespace(tracking_request_id="ab" * 16))
        self.assertEqual(_errors.typesafe_request_id(state), "req_" + "ab" * 16)
        self.assertRegex(_errors.typesafe_request_id(None), REQUEST_ID)


class CredentialPrecheckTests(unittest.IsolatedAsyncioTestCase):
    """app.auth.middleware.credentials_are_valid: auth's checks, none of its side effects."""

    def _request(self, headers=None, cookies=None):
        from starlette.requests import Request
        raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        if cookies:
            raw.append((b"cookie", "; ".join(f"{k}={v}" for k, v in cookies.items()).encode()))
        return Request({"type": "http", "method": "POST", "path": "/v1/systemone", "headers": raw})

    async def _valid(self, request, cached_key=None, db_key=None, cached_user=None):
        from app.auth.middleware import credentials_are_valid
        get_api_key = AsyncMock(return_value=db_key)
        with patch("app.auth.middleware.get_api_key", get_api_key), \
             patch("app.auth.middleware.auth_cache.get_cached_api_key", return_value=cached_key), \
             patch("app.auth.middleware.auth_cache.get_cached_user", return_value=cached_user), \
             patch("app.auth.middleware.get_user_by_username", AsyncMock(return_value=None)):
            return await credentials_are_valid(request), get_api_key

    async def test_no_or_unknown_credentials_are_invalid(self):
        self.assertFalse((await self._valid(self._request()))[0])
        self.assertFalse((await self._valid(self._request({"Authorization": "Bearer forged"})))[0])
        self.assertFalse((await self._valid(self._request({"Authorization": "Basic Zm9vOmJhcg=="})))[0])

    async def test_active_api_keys_are_valid_from_the_cache_or_the_db(self):
        active = SimpleNamespace(is_active=True, user_is_active=True)
        inactive = SimpleNamespace(is_active=True, user_is_active=False)
        request = self._request({"Authorization": "Bearer sk-test"})
        self.assertTrue((await self._valid(request, cached_key=active))[0])
        self.assertTrue((await self._valid(request, db_key=object()))[0])
        self.assertFalse((await self._valid(request, cached_key=inactive))[0])

    async def test_api_keys_are_not_accepted_from_the_cookie(self):
        valid, get_api_key = await self._valid(self._request(cookies={"access_token": "sk-test"}), db_key=object())
        self.assertFalse(valid)
        get_api_key.assert_not_awaited()

    async def test_user_jwts_are_valid_for_active_users(self):
        from app.auth.auth import create_access_token
        token = create_access_token({"sub": "alice"})
        for request in (self._request({"Authorization": f"Bearer {token}"}),
                        self._request(cookies={"access_token": token})):
            with self.subTest(request.headers):
                self.assertTrue((await self._valid(request, cached_user=SimpleNamespace(is_active=True)))[0])
                self.assertFalse((await self._valid(request, cached_user=SimpleNamespace(is_active=False)))[0])


class AliasSurfaceTests(unittest.TestCase):
    def test_systemone_is_an_opt_in_alias_surface(self):
        self.assertNotIn(SYSTEMONE_API, MODEL_ALIAS_DEFAULT_SURFACES)
        self.assertNotIn(SYSTEMONE_API, _parse_apis(None))
        self.assertEqual(_parse_apis('["systemone"]'), frozenset({SYSTEMONE_API}))
        self.assertNotIn(SYSTEMONE_API, ModelAliasUpsert(alias="a", target_model_id="b").apis)
        self.assertEqual(ModelAliasUpsert(alias="a", target_model_id="b", apis=["systemone", "openai"]).apis,
                         ["openai", "systemone"])
        self.assertNotIn(SYSTEMONE_API, ModelAliasResponse._decode_apis(None))
        self.assertEqual(ModelAliasResponse._decode_apis('["systemone"]'), ["systemone"])
        self.assertNotIn(SYSTEMONE_API, build_rule("a", "b").apis)


# ==================== Wiring ====================

class WiringTests(unittest.TestCase):
    def test_config_port_default_and_override(self):
        from app.config import load_config_from_env
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TYPESAFE_SERVER_PORT", None)
            self.assertEqual(load_config_from_env().server.typesafe_port, 11366)
        with patch.dict(os.environ, {"TYPESAFE_SERVER_PORT": "21366"}):
            self.assertEqual(load_config_from_env().server.typesafe_port, 21366)

    def test_management_mounts_the_typesafe_app(self):
        from app.asgi_mount import MOUNTED_API_PREFIXES
        self.assertEqual(MOUNTED_API_PREFIXES[-1], "/typesafe")
        ts = _typesafe_app()
        mgmt = create_management_app(FastAPI(), FastAPI(), FastAPI(), ts)
        for path in ("/health", "/api"):
            with self.subTest(path=path):
                direct, _ = _send(ts, "GET", path)
                mounted, _ = _send(mgmt, "GET", f"/typesafe{path}")
                self.assertEqual((direct.status_code, direct.json()), (mounted.status_code, mounted.json()))
        self.assertEqual(_send(ts, "GET", "/api")[0].json()["endpoints"], ["/v1/systemone", "/v1/models"])

    def test_system_one_is_served_only_on_the_typesafe_app(self):
        openai_app, azure_app = create_openai_app(), create_azure_openai_app()
        for app in (openai_app, azure_app):
            app.dependency_overrides[authenticate_jwt_or_api_key] = lambda: _Caller()
        mgmt = create_management_app(openai_app, FastAPI(), azure_app, FastAPI())
        for app, path in ((openai_app, "/v1/systemone"), (azure_app, "/openai/v1/systemone"),
                          (mgmt, "/openai/v1/systemone"), (mgmt, "/azure-openai/openai/v1/systemone")):
            with self.subTest(path=path):
                response, _ = _send(app, "POST", path, json=BODY)
                self.assertEqual(response.status_code, 404)

    def test_outbound_clients_do_not_offer_zstd(self):
        import requests
        from aiohttp.client_reqrep import _gen_default_accept_encoding
        self.assertEqual(httpx.AsyncClient().headers["accept-encoding"], "gzip, deflate")
        self.assertEqual(requests.utils.default_headers()["Accept-Encoding"], "gzip, deflate")
        self.assertEqual(_gen_default_accept_encoding(), "gzip, deflate")


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()

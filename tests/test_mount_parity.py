"""Parity between a provider API's dedicated port and its management-port mount.

Every test here asserts the *same* observable value for the same logical request
issued two ways: directly against the sub-app (as its own uvicorn server serves
it) and through the ``MountedApp`` mount on the management app. They exist
because Starlette's ``Mount`` leaves the prefix on ``scope["path"]``, so without
the wrapper each path-based decision silently diverges — see app/asgi_mount.py.
"""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.testclient import TestClient

from app.api_envelope import envelope_for
from app.asgi_mount import MOUNTED_API_PREFIXES, CORSExceptPrefixes, MountedApp, _under
from app.main import _add_request_tracking
from app.model_alias import model_alias_resolver
from app.request_tracker import RequestTracker

ALL_SURFACES = frozenset({"openai", "anthropic", "azure_openai"})


def mount(sub_app, prefix):
    """Wrap *sub_app* the way create_management_app does."""
    parent = FastAPI()
    parent.mount(prefix, MountedApp(sub_app, prefix))
    return parent


class _AliasSnapshot:
    """Swap in an alias table for the duration of a test."""

    def __init__(self, aliases):
        self.aliases = aliases

    def __enter__(self):
        self.saved = model_alias_resolver._aliases
        model_alias_resolver._aliases = self.aliases

    def __exit__(self, *exc):
        model_alias_resolver._aliases = self.saved


def build_openai_sub():
    sub = FastAPI()
    _add_request_tracking(sub, "openai")

    @sub.post("/v1/chat/completions")
    async def chat(request: Request, body: dict):
        return {
            "model_seen": body.get("model"),
            "state_model": getattr(request.state, "model", None),
            "has_tracking_id": hasattr(request.state, "tracking_request_id"),
            "url_path": request.url.path,
        }

    return sub


def build_azure_sub():
    """Azure sub-app with the real v1_api_middleware behavior under test."""
    from app.providers.openai_compatible import extra_request_headers, preserve_upstream_model

    sub = FastAPI()
    _add_request_tracking(sub, "azure_openai")

    @sub.middleware("http")
    async def v1_api_middleware(request: Request, call_next):
        if request.url.path.startswith("/openai/v1/"):
            preserve_upstream_model.set(True)
            preview = {
                k: v for k, v in request.headers.items() if k.lower().startswith("aoai-")
            }
            if preview:
                extra_request_headers.set(preview)
        return await call_next(request)

    @sub.post("/openai/v1/chat/completions")
    async def v1_chat(body: dict):
        return {
            "preserve": preserve_upstream_model.get(),
            "extra_headers": extra_request_headers.get(),
        }

    @sub.post("/openai/deployments/{provider}/{deployment}/chat/completions")
    async def deployment_chat(request: Request, provider: str, deployment: str, body: dict):
        return {"state_model": getattr(request.state, "model", None)}

    @sub.post("/openai/deployments/{provider}/responses")
    async def deployment_responses(request: Request, provider: str, body: dict):
        return {"state_model": getattr(request.state, "model", None)}

    return sub


def post(app, url, **kw):
    """POST with start/end_request mocked; returns (json_body, start kwargs)."""
    with patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock) as start, \
         patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
        with TestClient(app) as client:
            response = client.post(url, **kw)
        called = start.await_args.kwargs if start.await_args else None
    return response, called


class TrackingParityTests(unittest.TestCase):
    """Divergences 1, 2, 3 and 6 — the three reported symptoms and their knock-ons."""

    def _both(self, **post_kw):
        direct, direct_call = post(build_openai_sub(), "/v1/chat/completions", **post_kw)
        mounted, mounted_call = post(
            mount(build_openai_sub(), "/openai"), "/openai/v1/chat/completions", **post_kw
        )
        return (direct, direct_call), (mounted, mounted_call)

    def test_request_is_tracked_with_identical_attribution(self):
        body = {"model": "gpt-4o", "stream": False}
        (direct, direct_call), (mounted, mounted_call) = self._both(json=body)

        self.assertEqual(direct.status_code, 200)
        self.assertEqual(mounted.status_code, 200)
        for call in (direct_call, mounted_call):
            self.assertIsNotNone(call, "start_request was never called")
        # Attribution is indistinguishable between the two ports: same endpoint
        # string, same server label -> one usage row, one Active Requests entry.
        self.assertEqual(direct_call["endpoint"], "/v1/chat/completions")
        self.assertEqual(mounted_call["endpoint"], direct_call["endpoint"])
        self.assertEqual(mounted_call["server"], direct_call["server"])
        self.assertEqual(mounted_call["server"], "openai")

    def test_sub_app_sees_its_standalone_path(self):
        (direct, _), (mounted, _) = self._both(json={"model": "gpt-4o"})
        self.assertEqual(direct.json()["url_path"], "/v1/chat/completions")
        self.assertEqual(mounted.json()["url_path"], direct.json()["url_path"])

    def test_request_state_is_populated(self):
        """Divergences 2 and 3: rate-limit group precedence and tracking identity."""
        (direct, _), (mounted, _) = self._both(json={"model": "gpt-4o"})
        for payload in (direct.json(), mounted.json()):
            self.assertTrue(payload["has_tracking_id"])
            self.assertEqual(payload["state_model"], "gpt-4o")

    def test_alias_mapping_applies(self):
        aliases = {"friendly": ("prov/real-model", ALL_SURFACES)}
        with _AliasSnapshot(aliases):
            (direct, direct_call), (mounted, mounted_call) = self._both(
                json={"model": "friendly", "stream": False}
            )
        # The handler (and thus the upstream call) sees the rewritten model...
        self.assertEqual(direct.json()["model_seen"], "prov/real-model")
        self.assertEqual(mounted.json()["model_seen"], direct.json()["model_seen"])
        # ...and it is the mapped name that gets recorded against usage.
        self.assertEqual(mounted_call["model"], direct_call["model"])
        self.assertEqual(mounted_call["model"], "prov/real-model")

    def test_alias_retains_client_facing_name_for_echo(self):
        """echo_model_name must still return what the client asked for."""
        from app.model_alias import echo_model_name, original_model_name

        aliases = {"friendly": ("prov/real-model", ALL_SURFACES)}
        seen = {}

        sub = FastAPI()
        _add_request_tracking(sub, "openai")

        @sub.post("/v1/chat/completions")
        async def chat(body: dict):
            seen["echoed"] = original_model_name.get()
            return {}

        with _AliasSnapshot(aliases):
            post(mount(sub, "/openai"), "/openai/v1/chat/completions",
                 json={"model": "friendly"})

        # apply_alias stashed the client's name, so the response echoes "friendly"
        # even though "prov/real-model" was called upstream.
        self.assertEqual(seen["echoed"], "friendly")
        original_model_name.set(seen["echoed"])
        routed = type("Req", (), {"model": "prov/real-model"})()
        self.assertEqual(echo_model_name(routed), "friendly")

    def test_bare_model_name_is_left_untouched_by_the_middleware(self):
        """Resolution belongs to the route, not to tracking.

        Canonicalising a prefix-less name here would be a quota bypass: the
        middleware's provisional pick and the route's round-robin pick differ by
        design on a multi-candidate pool, so _enforce_rate_limit could skip the
        overall gate on a grouped provisional id while the route landed on an
        ungrouped one -- leaving the request governed by nothing. The middleware
        must therefore hand the route, and the tracker, the raw name.
        """
        with _AliasSnapshot({}):
            (direct, direct_call), (mounted, mounted_call) = self._both(
                json={"model": "gpt-5.4", "stream": False}
            )
        for payload in (direct.json(), mounted.json()):
            self.assertEqual(payload["model_seen"], "gpt-5.4")
            self.assertEqual(payload["state_model"], "gpt-5.4")
        self.assertEqual(direct_call["model"], "gpt-5.4")
        self.assertEqual(mounted_call["model"], "gpt-5.4")

    def test_untracked_path_is_still_untracked(self):
        """The gate must keep excluding what it excluded before."""
        sub = FastAPI()
        _add_request_tracking(sub, "anthropic")

        @sub.post("/v1/messages/count_tokens")
        async def count(body: dict):
            return {"input_tokens": 1}

        _, call = post(mount(sub, "/anthropic"),
                       "/anthropic/v1/messages/count_tokens", json={"model": "x"})
        self.assertIsNone(call, "count_tokens must not be tracked or counted")


class UsageBufferParityTests(unittest.IsolatedAsyncioTestCase):
    """The usage row a request lands on must be identical on both ports.

    Drives the *real* tracking middleware on each port, captures the arguments it
    hands the tracker, then replays them through a real ``RequestTracker`` and
    compares the resulting ``_usage_buffer`` key. If the two ports produced
    different endpoint/server/model values the counts would split across two rows.
    """

    async def _usage_keys_from(self, kwargs):
        tracker = RequestTracker()
        tracker._broadcast_raw = AsyncMock()
        await tracker.start_request(**kwargs)
        # Mirrors _update_tracking_identity once auth resolves the caller.
        await tracker.update_identity(kwargs["request_id"], "alice", "user")
        await tracker.end_request(kwargs["request_id"], status="completed")
        return list(tracker._usage_buffer.keys())

    async def test_same_usage_key_for_both_ports(self):
        body = {"model": "gpt-4o", "stream": False}
        _, direct_call = post(build_openai_sub(), "/v1/chat/completions", json=body)
        _, mounted_call = post(
            mount(build_openai_sub(), "/openai"), "/openai/v1/chat/completions", json=body
        )
        self.assertIsNotNone(mounted_call, "mounted request was never tracked")

        direct = await self._usage_keys_from(direct_call)
        mounted = await self._usage_keys_from(mounted_call)

        self.assertEqual(len(direct), 1)
        self.assertEqual(mounted, direct)
        # The key carries the model and the server label; both must match.
        self.assertIn("gpt-4o", direct[0])
        self.assertIn("openai", direct[0])

    def test_streaming_response_is_tracked_and_completes(self):
        """SSE is the main provider path; BaseHTTPMiddleware wrapping is delicate."""
        import asyncio

        def build():
            sub = FastAPI()
            _add_request_tracking(sub, "openai")

            @sub.post("/v1/chat/completions")
            async def chat(body: dict):
                async def gen():
                    for i in range(3):
                        yield f"data: chunk{i}\n\n".encode()
                        await asyncio.sleep(0)
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(gen(), media_type="text/event-stream")

            return sub

        results = []
        for app, url in ((build(), "/v1/chat/completions"),
                         (mount(build(), "/openai"), "/openai/v1/chat/completions")):
            with patch("app.request_tracker.request_tracker.start_request",
                       new_callable=AsyncMock) as start, \
                 patch("app.request_tracker.request_tracker.end_request",
                       new_callable=AsyncMock) as end:
                with TestClient(app) as client:
                    response = client.post(url, json={"model": "gpt-4o", "stream": True})
                results.append((
                    response.text,
                    start.await_args.kwargs if start.await_args else None,
                    end.await_args.kwargs.get("status") if end.await_args else None,
                ))

        direct, mounted = results
        self.assertEqual(mounted[0], direct[0])
        self.assertIn("data: [DONE]", mounted[0])
        self.assertTrue(mounted[1]["is_streaming"])
        self.assertEqual(mounted[1]["endpoint"], direct[1]["endpoint"])
        self.assertEqual(mounted[2], "completed")

    async def test_request_is_completed_through_the_mount(self):
        """end_request must fire, or the request would hang in Active Requests."""
        with patch("app.request_tracker.request_tracker.start_request",
                   new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request",
                   new_callable=AsyncMock) as end:
            with TestClient(mount(build_openai_sub(), "/openai")) as client:
                client.post("/openai/v1/chat/completions", json={"model": "gpt-4o"})
        end.assert_awaited()
        self.assertEqual(end.await_args.kwargs.get("status"), "completed")


class AzureParityTests(unittest.TestCase):
    """Divergences 4, 5 and 7 — Azure path arithmetic and the v1 middleware."""

    def test_v1_middleware_sets_preserve_and_forwards_preview_headers(self):
        headers = {"aoai-evt-stream": "on"}
        direct, _ = post(build_azure_sub(), "/openai/v1/chat/completions",
                         json={"model": "gpt-4o"}, headers=headers)
        mounted, _ = post(mount(build_azure_sub(), "/azure-openai"),
                          "/azure-openai/openai/v1/chat/completions",
                          json={"model": "gpt-4o"}, headers=headers)
        self.assertEqual(direct.json()["preserve"], True)
        self.assertEqual(mounted.json(), direct.json())
        self.assertEqual(mounted.json()["extra_headers"], {"aoai-evt-stream": "on"})

    def test_deployment_path_reconstructs_model(self):
        path = "/openai/deployments/prov/my-deployment/chat/completions"
        direct, direct_call = post(build_azure_sub(), path, json={})
        mounted, mounted_call = post(
            mount(build_azure_sub(), "/azure-openai"), "/azure-openai" + path, json={}
        )
        self.assertEqual(direct_call["model"], "prov/my-deployment")
        self.assertEqual(mounted_call["model"], direct_call["model"])
        self.assertEqual(mounted.json(), direct.json())

    def test_responses_path_is_excluded_from_reconstruction(self):
        """"{provider}/responses" is not a model id and must not be invented."""
        path = "/openai/deployments/prov/responses"
        _, direct_call = post(build_azure_sub(), path, json={})
        _, mounted_call = post(
            mount(build_azure_sub(), "/azure-openai"), "/azure-openai" + path, json={}
        )
        self.assertIsNone(direct_call["model"])
        self.assertEqual(mounted_call["model"], direct_call["model"])

    def test_responses_alias_prefixes_with_provider(self):
        """The bare model in a deployment-scoped Responses body gets provider-qualified."""
        aliases = {"prov/friendly": ("prov/real-model", ALL_SURFACES)}
        path = "/openai/deployments/prov/responses"
        with _AliasSnapshot(aliases):
            _, direct_call = post(build_azure_sub(), path, json={"model": "friendly"})
            _, mounted_call = post(mount(build_azure_sub(), "/azure-openai"),
                                   "/azure-openai" + path, json={"model": "friendly"})
        self.assertEqual(direct_call["model"], "prov/real-model")
        self.assertEqual(mounted_call["model"], direct_call["model"])


class _StubAuth:
    """Minimal non-admin principal that gets past the identity checks."""

    user_id = 1
    username = "alice"


class RateLimitSkipParityTests(unittest.IsolatedAsyncioTestCase):
    """Divergence 8 — count_tokens must never be rate limited on either port."""

    async def test_count_tokens_short_circuits_through_the_mount(self):
        from app.auth.middleware import RATE_LIMIT_SKIP_PATHS, _enforce_rate_limit

        sub = FastAPI()
        seen = {}

        @sub.post("/v1/messages/count_tokens")
        async def count(request: Request, body: dict):
            seen["path"] = request.url.path.rstrip("/") or "/"
            return {}

        with TestClient(mount(sub, "/anthropic")) as client:
            client.post("/anthropic/v1/messages/count_tokens", json={})

        # The path the skip-list is checked against is the standalone one.
        self.assertIn(seen["path"], RATE_LIMIT_SKIP_PATHS)

        # And _enforce_rate_limit really does return without consulting limits.
        request = type("R", (), {})()
        request.url = type("U", (), {"path": seen["path"]})()
        request.state = type("S", (), {})()
        with patch("app.rate_limit.rate_limit_tracker.check_and_increment",
                   new_callable=AsyncMock) as check:
            await _enforce_rate_limit(request, object())
        check.assert_not_awaited()

        # Control: a path that is *not* in the skip list does consult the limiter.
        request.url = type("U", (), {"path": "/v1/messages"})()
        with patch("app.rate_limit.rate_limit_tracker.check_and_increment",
                   new_callable=AsyncMock) as check:
            check.return_value = type("D", (), {"allowed": True})()
            await _enforce_rate_limit(request, _StubAuth())
        check.assert_awaited()


class EnvelopeParityTests(unittest.TestCase):
    """Divergence 9 — 429/403 bodies must match the surface the caller used."""

    def test_envelope_matches_surface_on_standalone_paths(self):
        self.assertEqual(envelope_for("/v1/messages", None), "anthropic")
        self.assertEqual(envelope_for("/v1/chat/completions", None), "openai")
        self.assertEqual(envelope_for("/openai/deployments/p/d/chat/completions", None), "azure")
        self.assertEqual(envelope_for("/openai/v1/chat/completions", None), "azure")

    def test_override_still_wins(self):
        self.assertEqual(envelope_for("/v1/messages", "openai"), "openai")

    def test_mounted_request_selects_the_same_envelope(self):
        seen = {}

        def sub_for(surface, route):
            sub = FastAPI()

            @sub.post(route)
            async def handler(request: Request):
                seen[surface] = envelope_for(request.url.path, None)
                return {}

            return sub

        cases = [
            ("anthropic", "/v1/messages", "/anthropic"),
            ("openai", "/v1/chat/completions", "/openai"),
            ("azure", "/openai/v1/chat/completions", "/azure-openai"),
        ]
        for surface, route, prefix in cases:
            with TestClient(mount(sub_for(surface, route), prefix)) as client:
                client.post(prefix + route, json={})

        self.assertEqual(seen["anthropic"], "anthropic")
        self.assertEqual(seen["openai"], "openai")
        self.assertEqual(seen["azure"], "azure")


class RedirectParityTests(unittest.TestCase):
    """Redirects generated by the sub-app must stay reachable through the mount."""

    def setUp(self):
        sub = FastAPI(redirect_slashes=True)

        @sub.get("/things/")
        async def things():
            return {}

        @sub.get("/relative")
        async def relative():
            return RedirectResponse(url="/v1/models", status_code=302)

        @sub.get("/to-management")
        async def to_management():
            # Mirrors the provider apps' root redirect to the management origin.
            return RedirectResponse(url="http://testserver/", status_code=302)

        self.client = TestClient(mount(sub, "/openai"), follow_redirects=False)

    def test_slash_redirect_keeps_the_mount_prefix(self):
        response = self.client.get("/openai/things")
        self.assertIn(response.status_code, (307, 308))
        self.assertTrue(
            response.headers["location"].endswith("/openai/things/"),
            response.headers["location"],
        )

    def test_relative_redirect_is_prefixed(self):
        response = self.client.get("/openai/relative")
        self.assertEqual(response.headers["location"], "/openai/v1/models")

    def test_absolute_redirect_to_management_is_left_alone(self):
        """A deliberate 302 to the management origin must not be rewritten."""
        response = self.client.get("/openai/to-management")
        self.assertEqual(response.headers["location"], "http://testserver/")

    def test_unknown_path_still_404s(self):
        self.assertEqual(self.client.get("/openai/nope").status_code, 404)


class CORSParityTests(unittest.TestCase):
    """Divergence 10 — browser clients must work on both ports."""

    def setUp(self):
        sub = FastAPI()
        sub.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=False,
            allow_methods=["*"], allow_headers=["*"],
        )

        @sub.post("/v1/chat/completions")
        async def chat():
            return {}

        parent = FastAPI()

        @parent.post("/admin/users")
        async def admin_users():
            return {}

        parent.mount("/openai", MountedApp(sub, "/openai"))
        parent.add_middleware(
            CORSExceptPrefixes,
            cors_app_factory=lambda inner: CORSMiddleware(
                inner, allow_origins=["http://localhost:8765"], allow_credentials=True,
                allow_methods=["*"], allow_headers=["*"],
            ),
            exempt_prefixes=MOUNTED_API_PREFIXES,
        )
        self.client = TestClient(parent)

    def _preflight(self, path, origin):
        return self.client.options(path, headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        })

    def test_proxy_preflight_from_third_party_origin_is_allowed(self):
        response = self._preflight("/openai/v1/chat/completions", "http://evil.example")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_proxy_post_carries_a_single_wildcard_header(self):
        response = self.client.post(
            "/openai/v1/chat/completions",
            json={}, headers={"Origin": "http://evil.example"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "*")
        self.assertNotIn("access-control-allow-credentials", response.headers)

    def test_management_route_still_rejects_third_party_origins(self):
        response = self._preflight("/admin/users", "http://evil.example")
        self.assertEqual(response.status_code, 400)

    def test_management_route_allows_its_own_origin_with_credentials(self):
        response = self._preflight("/admin/users", "http://localhost:8765")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")


class RootPathParityTests(unittest.TestCase):
    """The mount must survive a reverse-proxy --root-path.

    The ASGI path carries root_path, and Mount appends the mount prefix to
    root_path without touching the path, so the prefix sits *after* the parent's
    root. Adjusting root_path without also stripping the path there would leave
    every mounted API 404ing behind `uvicorn --root-path`.
    """

    ROOT = "/proxy"

    def setUp(self):
        sub = FastAPI(redirect_slashes=True)

        @sub.get("/v1/models")
        async def models(request: Request):
            return {"url_path": request.url.path, "root_path": request.scope["root_path"]}

        @sub.get("/things/")
        async def things():
            return {}

        self.client = TestClient(
            mount(sub, "/openai"), root_path=self.ROOT, follow_redirects=False
        )

    def test_mounted_route_is_reachable(self):
        response = self.client.get("/proxy/openai/v1/models")
        self.assertEqual(response.status_code, 200)

    def test_sub_app_sees_its_standalone_scope(self):
        body = self.client.get("/proxy/openai/v1/models").json()
        # Exactly what the sub-app would see served directly behind the same
        # --root-path: the root stays, the mount prefix is gone.
        self.assertEqual(body["url_path"], "/proxy/v1/models")
        self.assertEqual(body["root_path"], self.ROOT)

    def test_slash_redirect_keeps_root_and_mount_prefix(self):
        response = self.client.get("/proxy/openai/things")
        self.assertIn(response.status_code, (307, 308))
        self.assertTrue(
            response.headers["location"].endswith("/proxy/openai/things/"),
            response.headers["location"],
        )


class PrefixBoundaryTests(unittest.TestCase):
    """A lookalike prefix must not be treated as a mounted API."""

    def test_under_is_segment_aware(self):
        self.assertTrue(_under("/openai", "/openai"))
        self.assertTrue(_under("/openai/v1/models", "/openai"))
        self.assertFalse(_under("/openai-evil/v1/models", "/openai"))
        self.assertFalse(_under("/openaix", "/openai"))

    def test_lookalike_path_is_not_stripped_or_exempted(self):
        sub = FastAPI()

        @sub.get("/{full_path:path}")
        async def catch_all(request: Request):
            return {"url_path": request.url.path}

        wrapped = MountedApp(sub, "/openai")
        parent = FastAPI()
        parent.mount("/openai-evil", wrapped)  # deliberately mismatched

        with TestClient(parent) as client:
            response = client.get("/openai-evil/v1/models")
        # Prefix doesn't match on a segment boundary, so nothing is stripped.
        self.assertEqual(response.json()["url_path"], "/openai-evil/v1/models")


if __name__ == "__main__":
    unittest.main()

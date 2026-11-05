"""PlatformAI: Zoho refresh-token handling (cache, single flight, back-off, 401 retry)."""

import asyncio
import json
import unittest
from unittest import mock

import httpx

from app.providers.base import ProviderHTTPError
from app.providers.platformai import auth
from app.providers.platformai.auth import PlatformAITokenManager, reset_token_cache
from app.providers.platformai.client import ApiKeyAuth, OAuthAuth, PlatformAIClient


def _manager(handler, *, key="platformai:main", token="1000.refresh", secret=None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PlatformAITokenManager(provider_key=key, accounts_url="https://accounts.test",
                                  client_id="cid", client_secret=secret, refresh_token=token, http_client=http)


class TokenManagerTests(unittest.TestCase):
    def setUp(self):
        reset_token_cache()
        self.addCleanup(reset_token_cache)
        self.calls = []

    def _ok(self, request):
        self.calls.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={"access_token": f"tok{len(self.calls)}", "expires_in": 3600})

    def test_refresh_request_and_cache_reuse(self):
        async def run():
            mgr = _manager(self._ok)
            self.assertEqual(await mgr.get_token(), "tok1")
            self.assertEqual(await mgr.get_token(), "tok1")
            # A rebuilt provider instance with the same credentials shares the cache.
            self.assertEqual(await _manager(self._ok).get_token(), "tok1")
        asyncio.run(run())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], {"refresh_token": "1000.refresh", "client_id": "cid",
                                         "grant_type": "refresh_token"})

    def test_client_secret_sent_when_configured(self):
        asyncio.run(_manager(self._ok, secret="s3").get_token())
        self.assertEqual(self.calls[0]["client_secret"], "s3")

    def test_expiry_skew_triggers_refresh(self):
        async def run():
            mgr = _manager(self._ok)
            await mgr.get_token()
            mgr._state.expires_at = auth.time.time() + 30  # inside the 60s skew
            return await mgr.get_token()
        self.assertEqual(asyncio.run(run()), "tok2")

    def test_single_flight(self):
        async def slow(request):
            await asyncio.sleep(0.05)
            return self._ok(request)

        async def run():
            mgr = _manager(slow)
            return await asyncio.gather(*(mgr.get_token() for _ in range(5)))
        self.assertEqual(asyncio.run(run()), ["tok1"] * 5)
        self.assertEqual(len(self.calls), 1)

    def test_invalid_grant_is_502_and_backs_off(self):
        def dead(request):
            self.calls.append(1)
            return httpx.Response(200, json={"error": "invalid_code"})

        async def run():
            mgr = _manager(dead)
            with self.assertRaises(ProviderHTTPError) as first:
                await mgr.get_token()
            with self.assertRaises(ProviderHTTPError):
                await mgr.get_token()  # paused: no second network call
            return first.exception
        err = asyncio.run(run())
        self.assertEqual(err.status_code, 502)
        self.assertIn("refresh token rejected", err.message)
        self.assertEqual(len(self.calls), 1)

    def test_rate_limited(self):
        def limited(request):
            return httpx.Response(200, json={"error": "Access Denied", "error_description": "too many requests"})

        async def run():
            with self.assertRaises(ProviderHTTPError) as ctx:
                await _manager(limited).get_token()
            return ctx.exception
        self.assertEqual(asyncio.run(run()).status_code, 429)

    def test_token_never_logged(self):
        with self.assertLogs("app.providers.platformai.auth", level="WARNING") as logs:
            def dead(request):
                return httpx.Response(400, json={"error": "invalid_grant"})

            async def run():
                with self.assertRaises(ProviderHTTPError):
                    await _manager(dead, token="SECRET-REFRESH").get_token()
            asyncio.run(run())
        self.assertNotIn("SECRET-REFRESH", "\n".join(logs.output))


class GatewayRetryTests(unittest.TestCase):
    def setUp(self):
        reset_token_cache()
        self.addCleanup(reset_token_cache)

    def test_401_refreshes_once_and_retries(self):
        seen = []

        def handler(request):
            if request.url.path == "/oauth/v2/token":
                seen.append("refresh")
                return httpx.Response(200, json={"access_token": f"t{seen.count('refresh')}"})
            seen.append(request.headers["authorization"])
            if request.headers["authorization"] == "Zoho-oauthtoken t1":
                return httpx.Response(401, json={"error": {"message": "expired"}})
            assert request.url.path == "/pai/api/v3/ai/chat"
            assert "portal_id" not in request.headers
            return httpx.Response(200, json={"data": {"ok": True}})

        async def run():
            http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            tokens = PlatformAITokenManager(provider_key="k", accounts_url="https://a.test", client_id="c",
                                            refresh_token="r", http_client=http)
            await tokens.get_token()
            tokens._state.minted_at -= auth.MIN_TOKEN_AGE_FOR_REMINT_SECONDS + 1  # t1 is old enough to expire
            gateway = PlatformAIClient(base_url="https://gw.test", auth=OAuthAuth(tokens), http_client=http)
            return await gateway.chat({})
        self.assertEqual(asyncio.run(run()), {"data": {"ok": True}})
        self.assertEqual(seen, ["refresh", "Zoho-oauthtoken t1", "refresh", "Zoho-oauthtoken t2"])

    def test_401_on_fresh_token_does_not_remint(self):
        seen = []

        def handler(request):
            if request.url.path == "/oauth/v2/token":
                seen.append("refresh")
                return httpx.Response(200, json={"access_token": "t1"})
            seen.append("call")
            return httpx.Response(401, json={"error": {"message": "wrong portal"}})

        async def run():
            http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            tokens = PlatformAITokenManager(provider_key="k", accounts_url="https://a.test", client_id="c",
                                            refresh_token="r", http_client=http)
            gateway = PlatformAIClient(base_url="https://gw.test", auth=OAuthAuth(tokens), http_client=http)
            for _ in range(3):
                with self.assertRaises(ProviderHTTPError) as ctx:
                    await gateway.chat({})
                self.assertEqual(ctx.exception.status_code, 502)
        asyncio.run(run())
        self.assertEqual(seen, ["refresh", "call", "call", "call"])

    def test_api_key_uses_bearer_on_key_auth_path_and_never_retries(self):
        seen = []

        def handler(request):
            seen.append((request.url.path, request.headers["authorization"], request.headers.get("z-token-id")))
            return httpx.Response(401, json={"error": {"code": "API-invalid_oauth_token",
                                                       "message": "invalid OAuth token"}})

        async def run():
            http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            gateway = PlatformAIClient(base_url="https://gw.test", auth=ApiKeyAuth("193-k"),
                                       token_id="5434566543234", http_client=http)
            with self.assertRaises(ProviderHTTPError) as ctx:
                await gateway.chat({})
            return ctx.exception
        err = asyncio.run(run())
        self.assertEqual(err.status_code, 502)
        self.assertEqual(err.body["error"]["type"], "upstream_auth_error")
        self.assertEqual(seen, [("/pai/key_auth/v3/ai/chat", "Bearer 193-k", "5434566543234")])

    def test_401_on_superseded_token_keeps_the_new_one(self):
        async def run():
            mgr = _manager(lambda r: httpx.Response(200, json={"access_token": "new"}))
            await mgr.get_token()
            # A 401 for a token another request already replaced: retry with "new", don't drop it.
            self.assertTrue(mgr.invalidate("old"))
            self.assertEqual(mgr._state.token, "new")
        asyncio.run(run())

    def test_credential_change_resets_backoff(self):
        def reject(request):
            return httpx.Response(400, json={"error": "invalid_client"})

        async def run():
            http = httpx.AsyncClient(transport=httpx.MockTransport(reject))
            bad = PlatformAITokenManager(provider_key="p", accounts_url="https://a.test", client_id="wrong",
                                         refresh_token="r", http_client=http)
            with self.assertRaises(ProviderHTTPError):
                await bad.get_token()
            fixed = PlatformAITokenManager(
                provider_key="p", accounts_url="https://a.test", client_id="right", refresh_token="r",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json={"access_token": "ok"}))),
            )
            self.assertEqual(await fixed.get_token(), "ok")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

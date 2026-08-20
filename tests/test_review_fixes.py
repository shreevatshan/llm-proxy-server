"""Tests for the O1–O5 review-fix items.

- O1: Responses API ownership (IDOR) — owner-id derivation, mapping user_id
  persistence, and verify_response_ownership enforcement.
- O2: locked per-provider cache update (no clobber under concurrency).
- O5: ProviderHTTPError -> response helpers preserve upstream status + headers.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.openai_models import ModelInfo
from app.cache import ModelCache
from app.providers.base import ProviderHTTPError
from app.routes._errors import (
    openai_provider_error_response,
    azure_provider_error_response,
)
from app.auth.middleware import get_owner_user_id, verify_response_ownership
from app.auth.admin import AdminUser
from app.auth.models import Base, ResponseProviderMapping
from app.auth.database import (
    store_response_provider_mapping,
    get_response_provider_mapping,
)


def _model(model_id: str) -> ModelInfo:
    return ModelInfo(id=model_id, created=0, owned_by="t", provider=model_id.split("/")[0])


# ---------------------------------------------------------------------------
# O2 — locked per-provider cache update
# ---------------------------------------------------------------------------
class UpdateProviderModelsTests(unittest.TestCase):
    def test_replaces_only_target_provider_slice(self):
        cache = ModelCache()
        cache.update_models([_model("openai/a"), _model("anthropic/b")])

        asyncio.run(cache.update_provider_models("anthropic", [_model("anthropic/c")]))

        ids = {m.id for m in cache.get_enabled_models()}
        self.assertEqual(ids, {"openai/a", "anthropic/c"})

    def test_concurrent_updates_do_not_clobber(self):
        cache = ModelCache()
        cache.update_models([_model("openai/a"), _model("anthropic/b")])

        async def drive():
            # Two providers update their own slices concurrently. Neither may
            # lose the other's models (the bug O2 fixes).
            await asyncio.gather(
                cache.update_provider_models("openai", [_model("openai/x")]),
                cache.update_provider_models("anthropic", [_model("anthropic/y")]),
            )

        asyncio.run(drive())
        ids = {m.id for m in cache.get_enabled_models()}
        self.assertEqual(ids, {"openai/x", "anthropic/y"})


# ---------------------------------------------------------------------------
# O5 — ProviderHTTPError response helpers
# ---------------------------------------------------------------------------
class ProviderErrorResponseTests(unittest.TestCase):
    def test_openai_preserves_status_and_retry_after(self):
        err = ProviderHTTPError(
            status_code=429,
            message="slow down",
            body={"error": {"message": "slow down", "type": "rate_limit_error"}},
            headers={"Retry-After": "5", "x-secret": "leak-me"},
        )
        resp = openai_provider_error_response(err)
        self.assertEqual(resp.status_code, 429)
        # Safe header relayed, unsafe header dropped.
        self.assertEqual(resp.headers.get("retry-after"), "5")
        self.assertIsNone(resp.headers.get("x-secret"))

    def test_openai_wraps_non_dict_body(self):
        err = ProviderHTTPError(status_code=503, message="upstream down")
        resp = openai_provider_error_response(err)
        self.assertEqual(resp.status_code, 503)
        payload = json.loads(resp.body)
        self.assertEqual(payload["error"]["message"], "upstream down")

    def test_openai_sdk_error_wraps_bare_upstream_error_body(self):
        """Azure Foundry's /openai/v1/ surface returns the error object unwrapped.

        The real message must survive instead of being replaced by a generic
        "Upstream provider returned status 400".
        """
        import httpx, openai
        from app.providers.openai_compatible import _translate_openai_sdk_error

        bare = {
            "message": "Unsupported parameter: 'top_p' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "top_p",
            "code": "unsupported_parameter",
        }
        request = httpx.Request("POST", "https://example.test/openai/v1/chat/completions")
        response = httpx.Response(400, json=bare, request=request)
        translated = _translate_openai_sdk_error(
            openai.APIStatusError("400", response=response, body=bare), "azure"
        )
        self.assertEqual(translated.status_code, 400)
        self.assertEqual(translated.body["error"], bare)

    def test_openai_sdk_error_falls_back_for_non_dict_body(self):
        import httpx, openai
        from app.providers.openai_compatible import _translate_openai_sdk_error

        request = httpx.Request("POST", "https://example.test/openai/v1/chat/completions")
        response = httpx.Response(400, text="Bad Request", request=request)
        translated = _translate_openai_sdk_error(
            openai.APIStatusError("400", response=response, body="Bad Request"), "azure"
        )
        self.assertEqual(translated.body["error"]["type"], "upstream_error")
        self.assertEqual(
            translated.body["error"]["message"], "Upstream provider returned status 400"
        )

    def test_azure_preserves_status(self):
        err = ProviderHTTPError(status_code=429, message="slow down", headers={"Retry-After": "3"})
        resp = azure_provider_error_response(err)
        self.assertEqual(resp.status_code, 429)
        self.assertEqual(resp.headers.get("retry-after"), "3")


# ---------------------------------------------------------------------------
# O1 — ownership (IDOR)
# ---------------------------------------------------------------------------
class OwnerIdTests(unittest.TestCase):
    def test_user_owner_is_id(self):
        user = SimpleNamespace(id=7, username="u")
        self.assertEqual(get_owner_user_id(user), 7)

    def test_api_key_owner_is_user_id(self):
        key = SimpleNamespace(id=99, user_id=7, username="k")  # id is key id, not owner
        self.assertEqual(get_owner_user_id(key), 7)

    def test_admin_bypasses(self):
        admin = AdminUser(username="admin", email="a@b.c")
        self.assertIsNone(get_owner_user_id(admin))


class ResponseOwnershipTests(unittest.TestCase):
    def _fresh_db(self):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async def _init():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(_init())
        return engine, Session

    def test_mapping_persists_user_id_and_enforces(self):
        engine, Session = self._fresh_db()

        async def scenario():
            async with Session() as db:
                await store_response_provider_mapping(
                    db, "resp_abc", "openai:primary", "openai/gpt", user_id=7
                )
                mapping = await get_response_provider_mapping(db, "resp_abc")
                self.assertEqual(mapping.user_id, 7)

            # verify_response_ownership uses the module-level session factory.
            with patch("app.auth.database.AsyncSessionLocal", Session):
                # Owner: allowed (no exception).
                await verify_response_ownership("resp_abc", SimpleNamespace(id=7))
                # Admin: bypass.
                await verify_response_ownership("resp_abc", AdminUser(username="admin", email="a@b.c"))
                # Different user: rejected with 404.
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as ctx:
                    await verify_response_ownership("resp_abc", SimpleNamespace(id=8))
                self.assertEqual(ctx.exception.status_code, 404)
                # Unknown response id (no mapping): allowed (provider yields 404).
                await verify_response_ownership("resp_missing", SimpleNamespace(id=8))

        asyncio.run(scenario())
        asyncio.run(engine.dispose())

    def test_legacy_null_owner_is_allowed(self):
        engine, Session = self._fresh_db()

        async def scenario():
            async with Session() as db:
                await store_response_provider_mapping(
                    db, "resp_legacy", "openai:primary", "openai/gpt"
                )  # no user_id -> NULL
            with patch("app.auth.database.AsyncSessionLocal", Session):
                # Any user may access an unowned (legacy) mapping.
                await verify_response_ownership("resp_legacy", SimpleNamespace(id=123))

        asyncio.run(scenario())
        asyncio.run(engine.dispose())


if __name__ == "__main__":
    unittest.main()

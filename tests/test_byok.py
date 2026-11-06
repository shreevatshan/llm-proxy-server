"""BYOK (bring your own key): user-owned provider instances.

The invariant under test: a BYOK instance is reached only by its owner. Admins,
other users, allow-mode profiles and identity-less callers can never resolve,
list, dispatch to or probe it; the admin's generic tooling never edits or sweeps
it; its traffic is quota-exempt but still recorded as usage. See app/byok.py.
"""

import asyncio
import os
import socket
import tempfile
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app import byok
from app.auth import secret_box
from app.auth.admin import AdminUser
from app.auth.models import (
    Base, ModelConfiguration, ProviderCredentials, ProviderCredentialsCreate, ProviderCredentialsUpdate,
    RequestUsage, UserProviderCreate, UserProviderModelToggle, UserProviderUpdate,
)
from app.byok import byok_registry
from app.cache import ModelCache
from app.model_access_dep import ModelAccessDenied, caller_may_use
from app.model_alias import current_api_surface, original_model_name
from app.model_resolution import ModelUnavailable, resolve_model_for_request
from app.openai_models import ModelInfo
from app.providers import url_guard
from app.providers.provider_manager import provider_manager

OWNER, OTHER = 42, 7
BYOK_KEY = f"openai:byok-{OWNER}-personal"
BYOK_MODEL = f"{BYOK_KEY}/gpt-4o"
ADMIN_KEY = "openai:prod"
ADMIN_MODEL = f"{ADMIN_KEY}/gpt-4o"
ADMIN = AdminUser(username="admin", email="admin@example.com")

PUBLIC_ADDR = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def _model(model_id):
    return ModelInfo(id=model_id, created=0, owned_by="test", provider=model_id.split("/", 1)[0])


class _Stub:
    def __init__(self, key, apis=("openai", "anthropic")):
        self.full_provider_name = key
        self._apis = set(apis)

    def supports_api_for_model(self, model_name, api):
        return api in self._apis


class _User:
    def __init__(self, uid, username=None):
        self.id = uid
        self.username = username or f"user{uid}"
        self.is_active = True


class _ApiKey:
    """An API key principal: user_id is the owner, id is the key's own id."""

    def __init__(self, user_id, key_id=999):
        self.user_id = user_id
        self.id = key_id
        self.username = f"user{user_id}"


def _request(path="/v1/chat/completions"):
    return SimpleNamespace(url=SimpleNamespace(path=path), state=SimpleNamespace())


class _Env:
    """A provider registry + cache + owner map for one test, restored afterwards."""

    def __init__(self, model_ids, owners=None, disabled_models=(), disabled_providers=()):
        self.model_ids = list(model_ids)
        self.owners = dict(owners or {})
        self.disabled_models = set(disabled_models)
        self.disabled_providers = set(disabled_providers)

    def __enter__(self):
        self.cache = ModelCache()
        self.cache.set_provider_manager(provider_manager)
        self.cache.update_models([_model(m) for m in self.model_ids])
        self.cache._model_configs = {m: False for m in self.disabled_models}
        self.cache._provider_configs = {p: False for p in self.disabled_providers}
        keys = {m.split("/", 1)[0] for m in self.model_ids}
        self._saved = (provider_manager.model_cache, provider_manager.providers, dict(byok_registry._owners))
        provider_manager.model_cache = self.cache
        provider_manager.providers = {k: _Stub(k) for k in keys}
        byok_registry.replace(self.owners)
        self._tokens = (current_api_surface.set(None), original_model_name.set(None))
        return self

    def __exit__(self, *exc):
        provider_manager.model_cache, provider_manager.providers, owners = self._saved
        byok_registry.replace(owners)
        current_api_surface.reset(self._tokens[0])
        original_model_name.reset(self._tokens[1])


def _resolve(auth, model):
    return asyncio.run(resolve_model_for_request(_request(), auth, model))


def _default_profile(test):
    from app.profiles import profile_registry
    test.addCleanup(profile_registry.reset)


def _grant(test, user_id, mode):
    from app.profiles import ProfileSnapshot, profile_registry
    profile_registry.install({user_id: 9}, [ProfileSnapshot(9, "t", access_mode=mode, access_exceptions={})])
    test.addCleanup(profile_registry.reset)


# ======================================================================
# Naming
# ======================================================================

class NamingTests(unittest.TestCase):
    def test_key_helpers(self):
        self.assertEqual(byok.byok_instance_name(42, "personal"), "byok-42-personal")
        self.assertTrue(byok.is_byok_provider_key("openai:byok-42-personal"))
        self.assertTrue(byok.is_byok_model("openai:byok-42-personal/gpt-4o"))
        self.assertFalse(byok.is_byok_provider_key("openai:prod"))
        self.assertFalse(byok.is_byok_model("openai:prod/byok-model"))
        self.assertFalse(byok.is_byok_model("openai:prod/x:byok-1"))
        self.assertFalse(byok.is_byok_model("byok-1-x"))  # no ':' -> not an instance key
        self.assertEqual(byok.slug_of("byok-42-personal", 42), "personal")

    def test_validators(self):
        self.assertEqual(byok.validate_slug(" Personal "), "personal")
        for bad in ("", "-x", "a" * 41, "has space", "a/b", "a:b"):
            with self.assertRaises(ValueError):
                byok.validate_slug(bad)
        for bad in ("a:b", "a/b", "", "-a"):
            with self.assertRaises(ValueError):
                byok.validate_provider_name(bad)
        with self.assertRaises(ValueError):
            byok.validate_provider_type("openai_compatible")
        self.assertEqual(byok.validate_bedrock_region("us-west-2"), "us-west-2")
        with self.assertRaises(ValueError):
            byok.validate_bedrock_region("http://169.254.169.254")

    def test_user_create_schema_fixes_identity(self):
        m = UserProviderCreate(provider_type="google", slug="Mine", provider_name="evil", api_key="k")
        self.assertEqual((m.slug, m.provider_name, m.instance_name), ("mine", "google", None))
        with self.assertRaises(ValueError):
            UserProviderCreate(provider_type="custom", slug="a", provider_name="x:byok-1-y")

    def test_registry_fails_closed_for_unknown_byok_keys(self):
        reg = byok.ByokRegistry()
        reg.replace({BYOK_KEY: OWNER})
        self.assertEqual(reg.owner_of(BYOK_KEY), OWNER)
        self.assertEqual(reg.owner_of("openai:byok-9-ghost"), byok.UNKNOWN_OWNER)
        self.assertIsNone(reg.owner_of(ADMIN_KEY))
        self.assertFalse(byok.is_owner(_User(OWNER), byok.UNKNOWN_OWNER))
        self.assertFalse(byok.is_owner(ADMIN, OWNER))
        self.assertTrue(byok.is_owner(_ApiKey(OWNER), OWNER))
        self.assertFalse(byok.is_owner(_ApiKey(OTHER, key_id=OWNER), OWNER))


class SqlMirrorTests(unittest.IsolatedAsyncioTestCase):
    """_byok_match (SQL) must agree with is_byok_model (Python) on every edge."""

    async def test_sql_and_python_agree(self):
        from app.request_tracker import _byok_match
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        samples = [
            "openai:byok-42-personal/gpt-4o", "openai:byok-42-personal", "azure:byok-1-x/a/b",
            "openai:prod/gpt-4o", "openai:prod/byok-thing", "openai:prod/x:byok-1",
            "byok-1-x", "byok-1-x/model", "custom:mybyok-1/x", "", "plain",
        ]
        factory = sessionmaker(engine, class_=AsyncSession)
        async with factory() as db:
            for i, m in enumerate(samples):
                db.add(RequestUsage(date=__import__("datetime").date(2026, 1, 1), user_id=i, pool_id=0,
                                    user_identity="u", user_type="user", model=m, server="openai",
                                    request_count=1))
            await db.commit()
            rows = (await db.execute(select(RequestUsage.model).where(_byok_match(RequestUsage.model)))).scalars().all()
        await engine.dispose()
        self.assertEqual(set(rows), {m for m in samples if byok.is_byok_model(m)})


# ======================================================================
# Isolation: resolution, access, listing
# ======================================================================

class IsolationTests(unittest.TestCase):
    def setUp(self):
        _default_profile(self)

    def test_owner_resolves_canonical_id(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            self.assertEqual(_resolve(_User(OWNER), BYOK_MODEL), BYOK_MODEL)
            self.assertEqual(_resolve(_ApiKey(OWNER), BYOK_MODEL), BYOK_MODEL)

    def test_everyone_else_gets_404_not_403(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            for auth in (_User(OTHER), ADMIN, _ApiKey(OTHER, key_id=OWNER)):
                with self.assertRaises(ModelUnavailable, msg=repr(auth)):
                    _resolve(auth, BYOK_MODEL)
                self.assertFalse(caller_may_use(auth, BYOK_MODEL))
            # identity-less principals too
            self.assertFalse(caller_may_use(SimpleNamespace(), BYOK_MODEL))

    def test_bare_name_never_reaches_someone_elses_instance(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            with self.assertRaises(ModelUnavailable):
                _resolve(_User(OTHER), "gpt-4o")
            self.assertEqual(_resolve(_User(OWNER), "gpt-4o"), BYOK_MODEL)

    def test_allow_mode_profile_does_not_reach_byok(self):
        _grant(self, OTHER, "allow")
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            self.assertFalse(caller_may_use(_User(OTHER), BYOK_MODEL))

    def test_deny_mode_owner_still_uses_own_instance(self):
        _grant(self, OWNER, "deny")
        with _Env([BYOK_MODEL, ADMIN_MODEL], {BYOK_KEY: OWNER}):
            self.assertTrue(caller_may_use(_User(OWNER), BYOK_MODEL))
            self.assertFalse(caller_may_use(_User(OWNER), ADMIN_MODEL))

    def test_disabled_byok_model_or_instance_is_off_for_the_owner(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}, disabled_models=[BYOK_MODEL]):
            self.assertFalse(caller_may_use(_User(OWNER), BYOK_MODEL))
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}, disabled_providers=[BYOK_KEY]):
            self.assertFalse(caller_may_use(_User(OWNER), BYOK_MODEL))

    def test_orphan_byok_key_is_denied_to_everyone(self):
        with _Env([BYOK_MODEL], owners={}):
            for auth in (_User(OWNER), ADMIN):
                self.assertFalse(caller_may_use(auth, BYOK_MODEL))

    def test_mixed_round_robin_for_the_owner_only(self):
        with _Env([BYOK_MODEL, ADMIN_MODEL], {BYOK_KEY: OWNER}):
            owner_picks = {_resolve(_User(OWNER), "gpt-4o") for _ in range(6)}
            self.assertEqual(owner_picks, {BYOK_MODEL, ADMIN_MODEL})
            other_picks = {_resolve(_User(OTHER), "gpt-4o") for _ in range(6)}
            self.assertEqual(other_picks, {ADMIN_MODEL})
            admin_picks = {_resolve(ADMIN, "gpt-4o") for _ in range(6)}
            self.assertEqual(admin_picks, {ADMIN_MODEL})

    def test_family_prefix_ignores_byok_instances(self):
        with _Env([BYOK_MODEL, ADMIN_MODEL], {BYOK_KEY: OWNER}):
            self.assertEqual(provider_manager.find_provider_key("openai"), ADMIN_KEY)
            self.assertEqual(provider_manager._get_provider("openai").full_provider_name, ADMIN_KEY)
            self.assertEqual(_resolve(_User(OTHER), "openai/gpt-4o"), ADMIN_MODEL)
            # exact key still resolves for the owner
            self.assertEqual(provider_manager._get_provider(BYOK_KEY).full_provider_name, BYOK_KEY)

    def test_family_of_only_byok_is_not_a_provider_prefix(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            self.assertFalse(provider_manager.has_provider_prefix("openai"))
            with self.assertRaises(ValueError) as ctx:
                provider_manager._get_provider("openai")
            self.assertNotIn("byok", str(ctx.exception))

    def test_parse_model_name_never_falls_back_to_byok(self):
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            with self.assertRaises(ValueError):
                provider_manager._parse_model_name("gpt-4o")
            self.assertEqual(provider_manager._parse_model_name(BYOK_MODEL), (BYOK_KEY, "gpt-4o"))

    def test_listings(self):
        with _Env([BYOK_MODEL, ADMIN_MODEL], {BYOK_KEY: OWNER}):
            admin_ids = {m.id for m in asyncio.run(provider_manager.get_all_models(user_id=None))}
            self.assertEqual(admin_ids, {ADMIN_MODEL})
            owner_ids = {m.id for m in asyncio.run(provider_manager.get_all_models(user_id=OWNER))}
            self.assertEqual(owner_ids, {ADMIN_MODEL, BYOK_MODEL})
            other_ids = {m.id for m in asyncio.run(provider_manager.get_all_models(user_id=OTHER))}
            self.assertEqual(other_ids, {ADMIN_MODEL})
            owner_access = dict((m.id, a) for m, a in provider_manager.model_cache.get_models_with_access_for_user(OWNER))
            self.assertTrue(owner_access[BYOK_MODEL])
            other_access = {m.id for m, _ in provider_manager.model_cache.get_models_with_access_for_user(OTHER)}
            self.assertNotIn(BYOK_MODEL, other_access)

    def test_enforce_model_access_reports_404_for_foreign_byok(self):
        from app.model_access_dep import enforce_model_access
        with _Env([BYOK_MODEL], {BYOK_KEY: OWNER}):
            with self.assertRaises(ModelUnavailable):
                asyncio.run(enforce_model_access(_request(), _User(OTHER), BYOK_MODEL))
            asyncio.run(enforce_model_access(_request(), _User(OWNER), BYOK_MODEL))


class DispatchHoleTests(unittest.IsolatedAsyncioTestCase):
    async def test_azure_deployment_routes_404_for_non_owners(self):
        from app.routes.azure_openai import _get_azure_provider
        from app.providers.azure_provider import AzureProvider
        key = f"azure:byok-{OWNER}-mine"
        stub = AzureProvider.__new__(AzureProvider)
        stub.full_provider_name = key
        saved = (provider_manager.providers, dict(byok_registry._owners))
        provider_manager.providers = {key: stub}
        byok_registry.replace({key: OWNER})
        try:
            for auth in (_User(OTHER), ADMIN, None):
                with self.assertRaises(HTTPException) as ctx:
                    _get_azure_provider(key, auth)
                self.assertEqual(ctx.exception.status_code, 404)
            self.assertIs(_get_azure_provider(key, _ApiKey(OWNER)), stub)
        finally:
            provider_manager.providers, owners = saved
            byok_registry.replace(owners)

    async def test_stored_byok_response_is_owner_only(self):
        from app.auth import middleware
        mapping = SimpleNamespace(provider_key=BYOK_KEY, user_id=OWNER)

        @asynccontextmanager
        async def _session():
            yield None

        saved = dict(byok_registry._owners)
        byok_registry.replace({BYOK_KEY: OWNER})
        try:
            with patch("app.auth.database.AsyncSessionLocal", _session), \
                    patch("app.auth.database.get_response_provider_mapping", AsyncMock(return_value=mapping)):
                for auth in (ADMIN, _User(OTHER)):
                    with self.assertRaises(HTTPException):
                        await middleware.verify_response_ownership("resp_1", auth)
                await middleware.verify_response_ownership("resp_1", _User(OWNER))
        finally:
            byok_registry.replace(saved)

    async def test_websearch_never_intercepts_byok(self):
        from app.websearch.settings import websearch_settings_cache
        cfg = SimpleNamespace(enabled=True, base_url="http://s", apply_to=("openai",), enabled_providers=("*",))
        saved = websearch_settings_cache._snapshot
        websearch_settings_cache._snapshot = SimpleNamespace(default=cfg)
        reg = dict(byok_registry._owners)
        byok_registry.replace({BYOK_KEY: OWNER})
        try:
            self.assertTrue(websearch_settings_cache.is_active("openai", ADMIN_KEY))
            self.assertFalse(websearch_settings_cache.is_active("openai", BYOK_KEY))
        finally:
            websearch_settings_cache._snapshot = saved
            byok_registry.replace(reg)


# ======================================================================
# Quotas
# ======================================================================

class QuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_is_not_charged_but_slot_is_taken(self):
        from app import rate_limit_dep
        seen = {}

        async def fake_slot(request, user_id, charge):
            seen["user_id"] = user_id
            seen["decision"] = await charge()
            return None

        saved = dict(byok_registry._owners)
        byok_registry.replace({BYOK_KEY: OWNER})
        try:
            with patch.object(rate_limit_dep, "charge_with_slot", fake_slot), \
                    patch.object(rate_limit_dep.rate_limit_tracker, "check_and_increment",
                                 AsyncMock(side_effect=AssertionError("charged"))):
                await rate_limit_dep.enforce_group_rate_limit(_request(), _User(OWNER), BYOK_MODEL)
            self.assertEqual(seen, {"user_id": OWNER, "decision": None})
        finally:
            byok_registry.replace(saved)

    async def test_non_owner_path_still_charges(self):
        from app import rate_limit_dep
        charged = AsyncMock(return_value=None)

        async def fake_slot(request, user_id, charge):
            return await charge()

        with patch.object(rate_limit_dep, "charge_with_slot", fake_slot), \
                patch.object(rate_limit_dep.rate_limit_tracker, "check_and_increment", charged):
            await rate_limit_dep.enforce_group_rate_limit(_request(), _User(OWNER), ADMIN_MODEL)
        charged.assert_awaited_once()

    def test_pool_settlement_ignores_byok_rows(self):
        from app.auth.pools import _fold_models_into_scopes
        folded = _fold_models_into_scopes([
            {"user_id": 1, "model": BYOK_MODEL, "request_count": 50},
            {"user_id": 1, "model": ADMIN_MODEL, "request_count": 3},
        ])
        self.assertEqual(sum(folded.values()), 3)

    async def test_byok_usage_rows_carry_no_pool(self):
        from app.request_tracker import RequestTracker
        tracker = RequestTracker()
        with patch.object(RequestTracker, "_pool_id_for", staticmethod(lambda uid: 5)):
            for model in (BYOK_MODEL, ADMIN_MODEL):
                rid = f"r-{model}"
                await tracker.start_request(rid, "openai", "/v1/chat/completions", "POST", model, "alice", "user", False)
                await tracker.update_identity(rid, "alice", "user", user_id=OWNER)
                await tracker.end_request(rid, status="completed")
        pools = {key[5]: key[7] for key in tracker._usage_buffer}
        self.assertEqual(pools[BYOK_MODEL], 0)
        self.assertEqual(pools[ADMIN_MODEL], 5)


# ======================================================================
# Encryption and the SSRF guard
# ======================================================================

class SecretBoxTests(unittest.TestCase):
    def test_round_trip_and_rotation(self):
        enc = secret_box.encrypt("sk-live-123")
        self.assertTrue(enc.startswith(secret_box.PREFIX))
        self.assertNotIn("sk-live-123", enc)
        self.assertEqual(secret_box.decrypt(enc), "sk-live-123")
        self.assertEqual(secret_box.encrypt(enc), enc)  # idempotent
        self.assertEqual(secret_box.decrypt("plain"), "plain")
        with patch.dict(os.environ, {"LLMPROXY_BYOK_ENCRYPTION_KEY": "another-key"}):
            with self.assertRaises(secret_box.SecretDecryptError):
                secret_box.decrypt(enc)

    def test_provider_config_gets_plaintext_and_requires_credentials(self):
        cred = SimpleNamespace(
            provider_key=BYOK_KEY, provider_type="custom", instance_name=f"byok-{OWNER}-personal",
            provider_name="openai", enabled=True, owner_user_id=OWNER, admin_disabled=False,
            api_key=secret_box.encrypt("sk-real"), secret_access_key=None, oauth_client_secret=None,
            oauth_refresh_token=None, endpoint=None, base_url="https://api.example.com/v1",
            anthropic_base_url=None, oauth_accounts_url=None, supported_apis='["openai"]',
            anthropic_auth_style=None,
        )
        with patch.object(url_guard.socket, "getaddrinfo", return_value=PUBLIC_ADDR):
            config = provider_manager._create_provider_config(cred)
        self.assertEqual(config["api_key"], "sk-real")
        self.assertTrue(config["byok"])
        self.assertTrue(cred.api_key.startswith(secret_box.PREFIX))  # row untouched

        bedrock = SimpleNamespace(**{**vars(cred), "provider_type": "bedrock", "access_key_id": None})
        with self.assertRaises(ValueError):
            provider_manager._create_provider_config(bedrock)

    def test_global_rows_are_untouched(self):
        cred = SimpleNamespace(provider_key=ADMIN_KEY, provider_type="google", instance_name="prod",
                               provider_name="google", enabled=True, owner_user_id=None,
                               api_key="plain-admin-key", base_url=None)
        config = provider_manager._create_provider_config(cred)
        self.assertEqual(config["api_key"], "plain-admin-key")
        self.assertNotIn("byok", config)


class UrlGuardTests(unittest.TestCase):
    def _resolve_to(self, ip):
        return patch.object(url_guard.socket, "getaddrinfo",
                            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))])

    def test_rejects_internal_and_plain_http(self):
        for ip in ("127.0.0.1", "169.254.169.254", "10.0.0.5", "192.168.1.1", "100.64.0.1", "0.0.0.0"):
            with self._resolve_to(ip), self.assertRaises(url_guard.UnsafeUpstreamURL, msg=ip):
                url_guard.validate_user_upstream_url("https://internal.example.com/v1")
        with self.assertRaises(url_guard.UnsafeUpstreamURL):
            url_guard.validate_user_upstream_url("http://api.example.com/v1")
        with self.assertRaises(url_guard.UnsafeUpstreamURL):
            url_guard.validate_user_upstream_url("https://user:pw@api.example.com")

    def test_accepts_public_https(self):
        with self._resolve_to("93.184.216.34"):
            self.assertEqual(url_guard.validate_user_upstream_url("https://api.example.com/v1"),
                             "https://api.example.com/v1")

    def test_guarded_client_does_not_follow_redirects(self):
        client = url_guard.guarded_http_client()
        self.assertFalse(client.follow_redirects)
        asyncio.run(client.aclose())


# ======================================================================
# Routes: the user's API and the admin's protections (temp database)
# ======================================================================

class _DbCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self._path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.db = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)()
        from app.routes import user_providers
        self.routes = user_providers
        user_providers._last_action.clear()
        self.sync = AsyncMock(return_value={"models": []})
        self._saved_owners = dict(byok_registry._owners)
        for p in (
            patch.object(user_providers, "sync_byok_provider", self.sync),
            patch.object(provider_manager, "remove_byok_provider", AsyncMock()),
            patch.object(url_guard.socket, "getaddrinfo", return_value=PUBLIC_ADDR),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def asyncTearDown(self):
        byok_registry.replace(self._saved_owners)
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._path)

    async def _create(self, user, **kw):
        body = {"provider_type": "custom", "slug": "personal", "provider_name": "openai",
                "base_url": "https://api.example.com/v1", "api_key": "sk-user-secret"}
        body.update(kw)
        return await self.routes.create_my_provider(UserProviderCreate(**body), user, self.db)


class UserRouteTests(_DbCase):
    async def test_create_builds_key_sets_owner_and_encrypts(self):
        alice = _User(OWNER, "alice")
        resp = await self._create(alice)
        self.assertEqual(resp.provider_key, BYOK_KEY)
        self.assertEqual(resp.slug, "personal")
        self.assertEqual(resp.api_key, "********")
        row = (await self.db.execute(select(ProviderCredentials))).scalar_one()
        self.assertEqual(row.owner_user_id, OWNER)
        self.assertTrue(row.api_key.startswith(secret_box.PREFIX))
        self.sync.assert_awaited_once()

    async def test_admin_is_refused(self):
        with self.assertRaises(HTTPException) as ctx:
            await self.routes.list_my_providers(ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_other_users_see_nothing_and_get_404(self):
        await self._create(_User(OWNER))
        bob = _User(OTHER)
        listing = await self.routes.list_my_providers(bob, self.db)
        self.assertEqual(listing["providers"], [])
        calls = [
            self.routes.get_my_provider(BYOK_KEY, bob, self.db),
            self.routes.update_my_provider(UserProviderUpdate(api_key="x"), BYOK_KEY, bob, self.db),
            self.routes.toggle_my_provider(False, BYOK_KEY, bob, self.db),
            self.routes.sync_my_provider(BYOK_KEY, bob, self.db),
            self.routes.list_my_provider_models(BYOK_KEY, bob, self.db),
            self.routes.toggle_my_provider_model(UserProviderModelToggle(model_id=BYOK_MODEL, enabled=False), bob, self.db),
            self.routes.delete_my_provider(BYOK_KEY, bob, self.db),
        ]
        for call in calls:
            with self.assertRaises(HTTPException) as ctx:
                await call
            self.assertEqual(ctx.exception.status_code, 404)

    async def test_identity_cannot_change(self):
        alice = _User(OWNER)
        await self._create(alice)
        self.routes._last_action.clear()
        with self.assertRaises(HTTPException) as ctx:
            await self.routes.update_my_provider(UserProviderUpdate(provider_name="other"), BYOK_KEY, alice, self.db)
        self.assertEqual(ctx.exception.status_code, 422)
        # and the database layer refuses a re-key too
        from app.auth.database import update_provider_credentials
        with self.assertRaises(ValueError):
            await update_provider_credentials(self.db, BYOK_KEY, provider_name="other")

    async def test_rename_helper_keeps_owner(self):
        from app.auth.database import _rename_provider_with_models
        await self._create(_User(OWNER))
        new = await _rename_provider_with_models(self.db, BYOK_KEY, "openai:byok-42-renamed")
        self.assertEqual(new.owner_user_id, OWNER)

    async def test_cap_and_cooldown(self):
        alice = _User(OWNER)
        with patch.object(self.routes, "MAX_BYOK_INSTANCES_PER_USER", 1):
            await self._create(alice)
            self.routes._last_action.clear()
            with self.assertRaises(HTTPException) as ctx:
                await self._create(alice, slug="second")
            self.assertEqual(ctx.exception.status_code, 409)
        with self.assertRaises(HTTPException) as ctx:
            await self._create(alice, slug="third")
        self.assertEqual(ctx.exception.status_code, 429)

    async def test_concurrent_creates_respect_the_cap(self):
        alice = _User(OWNER)
        with patch.object(self.routes, "MAX_BYOK_INSTANCES_PER_USER", 1), \
                patch.object(self.routes, "_cooldown", lambda *a: None):
            results = await asyncio.gather(
                self._create(alice, slug="one"), self._create(alice, slug="two"), return_exceptions=True,
            )
        self.assertEqual(sum(1 for r in results if not isinstance(r, Exception)), 1)

    async def test_internal_urls_are_refused(self):
        with patch.object(url_guard.socket, "getaddrinfo",
                          return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))]):
            with self.assertRaises(HTTPException) as ctx:
                await self._create(_User(OWNER))
        self.assertEqual(ctx.exception.status_code, 422)
        with self.assertRaises(HTTPException):
            await self._create(_User(OWNER), slug="b", base_url="http://api.example.com/v1")

    async def test_bedrock_and_google_need_their_own_credentials(self):
        with self.assertRaises(HTTPException) as ctx:
            await self._create(_User(OWNER), provider_type="bedrock", slug="aws", base_url=None, api_key=None)
        self.assertEqual(ctx.exception.status_code, 422)
        self.routes._last_action.clear()
        with self.assertRaises(HTTPException):
            await self._create(_User(OWNER), provider_type="google", slug="g", base_url=None, api_key=None)

    async def test_toggle_refused_while_admin_disabled(self):
        alice = _User(OWNER)
        await self._create(alice)
        row = (await self.db.execute(select(ProviderCredentials))).scalar_one()
        row.admin_disabled = True
        await self.db.commit()
        with self.assertRaises(HTTPException) as ctx:
            await self.routes.toggle_my_provider(True, BYOK_KEY, alice, self.db)
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_user_delete_takes_byok_rows(self):
        from app.auth.database import permanently_delete_user
        from app.auth.models import User
        self.db.add(User(id=OWNER, username="alice", email="a@example.com", hashed_password="x"))
        await self.db.commit()
        await self._create(_User(OWNER, "alice"))
        self.db.add(ModelConfiguration(model_id=BYOK_MODEL, provider_key=BYOK_KEY, model_name="gpt-4o"))
        await self.db.commit()
        with patch("app.auth.database.purge_user_usage", AsyncMock()):
            self.assertTrue(await permanently_delete_user(self.db, OWNER))
        self.assertEqual((await self.db.execute(select(ProviderCredentials))).scalars().all(), [])
        self.assertEqual((await self.db.execute(select(ModelConfiguration))).scalars().all(), [])


class AdminProtectionTests(_DbCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        from app.routes import admin as admin_routes
        self.admin_routes = admin_routes
        for p in (
            patch.object(admin_routes, "auto_sync_on_provider_change", AsyncMock(return_value={"message": "ok"})),
            patch.object(admin_routes.provider_manager, "refresh_providers_from_database", AsyncMock()),
            patch.object(admin_routes.provider_manager, "remove_provider", AsyncMock()),
        ):
            p.start()
            self.addCleanup(p.stop)
        await self._create(_User(OWNER))
        self.db.add(ModelConfiguration(model_id=BYOK_MODEL, provider_key=BYOK_KEY, model_name="gpt-4o"))
        self.db.add(ProviderCredentials(provider_key=ADMIN_KEY, provider_type="custom", instance_name="prod",
                                        provider_name="openai", base_url="https://a.example.com", api_key="k"))
        self.db.add(ModelConfiguration(model_id=ADMIN_MODEL, provider_key=ADMIN_KEY, model_name="gpt-4o"))
        await self.db.commit()
        byok_registry.replace({BYOK_KEY: OWNER})

    async def _expect_404(self, coro):
        with self.assertRaises(HTTPException) as ctx:
            await coro
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_generic_routes_treat_byok_as_absent(self):
        a, r = self.admin, self.admin_routes
        await self._expect_404(r.get_provider_credential(BYOK_KEY, a, self.db))
        await self._expect_404(r.update_provider_credential(ProviderCredentialsUpdate(api_key="x"), BYOK_KEY, a, self.db))
        await self._expect_404(r.delete_provider_credential(BYOK_KEY, a, self.db))
        await self._expect_404(r.get_provider_models(BYOK_KEY, a, self.db))
        from app.auth.models import ToggleRequest
        await self._expect_404(r.toggle_provider(ToggleRequest(enabled=False), BYOK_KEY, a, self.db))
        await self._expect_404(r.toggle_model(ToggleRequest(enabled=False), BYOK_MODEL, a, self.db))

    @property
    def admin(self):
        return ADMIN

    async def test_listings_and_bulk_toggle_skip_byok(self):
        r = self.admin_routes
        listed = {p.provider_key for p in await r.list_provider_credentials(ADMIN, self.db)}
        self.assertEqual(listed, {ADMIN_KEY})
        tree = await r.get_models_management(ADMIN, self.db)
        self.assertEqual({p.provider_key for p in tree.providers}, {ADMIN_KEY})
        from app.auth.database import bulk_toggle_all_models
        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            await bulk_toggle_all_models(self.db, False)
        rows = {m.model_id: m.is_enabled for m in (await self.db.execute(select(ModelConfiguration))).scalars()}
        self.assertTrue(rows[BYOK_MODEL])
        self.assertFalse(rows[ADMIN_MODEL])

    async def test_stale_sweep_and_reinit_keep_byok(self):
        from app.auth.database import identify_stale_models
        stale = await identify_stale_models(self.db, [])
        self.assertNotIn(BYOK_MODEL, {s["model_id"] for s in stale})
        r = self.admin_routes
        with patch.object(r, "reload_reference_caches", AsyncMock()):
            await r.reinit_system(ADMIN, self.db)
        keys = {p.provider_key for p in (await self.db.execute(select(ProviderCredentials))).scalars()}
        self.assertEqual(keys, {BYOK_KEY})

    async def test_create_and_import_reject_reserved_names(self):
        r = self.admin_routes
        with self.assertRaises(HTTPException) as ctx:
            await r.create_provider_credential(
                ProviderCredentialsCreate(provider_type="custom", instance_name="byok-1-x", provider_name="openai",
                                          base_url="https://x.example.com"),
                upsert=False, current_admin=ADMIN, db=self.db,
            )
        self.assertEqual(ctx.exception.status_code, 422)

        import io
        import json
        from fastapi import UploadFile
        payload = {"providers": [
            {"provider_key": BYOK_KEY, "provider_type": "custom", "instance_name": "hijack",
             "provider_name": "openai", "base_url": "https://evil.example.com"},
            {"provider_key": "openai:x", "provider_type": "custom", "instance_name": "byok-9-x",
             "provider_name": "openai", "base_url": "https://evil.example.com"},
        ], "model_configurations": [
            {"model_id": BYOK_MODEL, "provider_key": BYOK_KEY, "model_name": "gpt-4o", "is_enabled": False},
        ]}
        upload = UploadFile(filename="c.json", file=io.BytesIO(json.dumps(payload).encode()))
        result = await r.import_provider_config(upload, overwrite=True, sync_models=False,
                                                current_admin=ADMIN, db=self.db)
        self.assertEqual(result.imported + result.overwritten, 0)
        self.assertEqual(len(result.errors), 2)
        row = (await self.db.execute(select(ProviderCredentials).where(
            ProviderCredentials.provider_key == BYOK_KEY))).scalar_one()
        self.assertEqual(row.owner_user_id, OWNER)
        model = (await self.db.execute(select(ModelConfiguration).where(
            ModelConfiguration.model_id == BYOK_MODEL))).scalar_one()
        self.assertTrue(model.is_enabled)

    async def test_alias_and_outage_targets_reject_byok(self):
        targets = await self.admin_routes._outage_targets(self.db)
        self.assertNotIn(BYOK_MODEL, targets[1])
        self.assertNotIn(BYOK_KEY, targets[0])

    async def test_admin_byok_list_disable_delete(self):
        r = self.admin_routes
        rows = await r.list_byok_providers(ADMIN, self.db)
        self.assertEqual([(x.provider_key, x.owner_user_id, x.model_count) for x in rows], [(BYOK_KEY, OWNER, 1)])
        self.assertTrue(rows[0].has_secret)
        self.assertNotIn("sk-user-secret", rows[0].model_dump_json())

        out = await r.set_byok_provider_disabled(BYOK_KEY, True, ADMIN, self.db)
        self.assertTrue(out.admin_disabled)
        provider_manager.remove_byok_provider.assert_awaited()

        await r.delete_byok_provider(BYOK_KEY, ADMIN, self.db)
        self.assertIsNone((await self.db.execute(select(ProviderCredentials).where(
            ProviderCredentials.provider_key == BYOK_KEY))).scalar_one_or_none())
        await self._expect_404(r.delete_byok_provider(ADMIN_KEY, ADMIN, self.db))


# ======================================================================
# Usage: BYOK traffic is tracked and attributable in the admin Usage tab
# ======================================================================

class UsageTests(unittest.TestCase):
    def test_per_instance_fold(self):
        from app.auth.database import fold_usage_by_instance
        per_model = [
            {"model": BYOK_MODEL, "request_count": 5},
            {"model": f"{BYOK_KEY}/o3", "request_count": 2},
            {"model": ADMIN_MODEL, "request_count": 4},
            {"model": "bare", "request_count": 1},
        ]
        per_user = [{"user_id": OWNER, "user_identity": "alice"}]
        rows = {r["provider_key"]: r for r in fold_usage_by_instance(per_model, per_user)}
        self.assertEqual(rows[BYOK_KEY]["request_count"], 7)
        self.assertTrue(rows[BYOK_KEY]["is_byok"])
        self.assertEqual((rows[BYOK_KEY]["owner_user_id"], rows[BYOK_KEY]["owner_identity"]), (OWNER, "alice"))
        self.assertFalse(rows[ADMIN_KEY]["is_byok"])
        self.assertEqual(rows["(unattributed)"]["request_count"], 1)


class UsageAggregateTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_usage_payload_has_instances_and_drilldown(self):
        import datetime
        from app import time_utils
        from app.auth.database import get_usage_aggregates
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        today = time_utils.local_today()
        async with sessionmaker(engine, class_=AsyncSession)() as db:
            for uid, model, n in ((OWNER, BYOK_MODEL, 5), (OWNER, ADMIN_MODEL, 2), (OTHER, ADMIN_MODEL, 3)):
                db.add(RequestUsage(date=today, user_id=uid, pool_id=0, user_identity=f"u{uid}",
                                    user_type="user", model=model, server="openai", request_count=n))
            await db.commit()
            top = await get_usage_aggregates(db, window="today")
            self.assertEqual(top["totals"]["byok_requests"], 5)
            inst = {r["provider_key"]: r for r in top["per_instance"]}
            self.assertEqual(inst[BYOK_KEY]["request_count"], 5)
            self.assertEqual(inst[ADMIN_KEY]["request_count"], 5)
            self.assertTrue({r["model"]: r["is_byok"] for r in top["per_model"]}[BYOK_MODEL])
            drill = await get_usage_aggregates(db, window="today", filter_instance=BYOK_KEY)
            self.assertEqual(drill["breakdown"], [{"model": BYOK_MODEL, "request_count": 5}])
            self.assertEqual([u["user_id"] for u in drill["breakdown_users"]], [OWNER])
            self.assertTrue(drill["instance"]["is_byok"])
        await engine.dispose()


if __name__ == "__main__":
    unittest.main()


class RegistryLoadTests(_DbCase):
    """The provider manager's loader and single-instance edits, against a real DB."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self._saved_registry = (provider_manager.providers, provider_manager.model_cache,
                                dict(provider_manager.byok_load_errors))
        provider_manager.providers = {}
        provider_manager.model_cache = ModelCache()
        p = patch("app.auth.database.AsyncSessionLocal", self.factory)
        p.start()
        self.addCleanup(p.stop)

    async def asyncTearDown(self):
        provider_manager.providers, provider_manager.model_cache, errors = self._saved_registry
        provider_manager.byok_load_errors = errors
        await super().asyncTearDown()

    async def test_upsert_disable_and_remove(self):
        from app.providers.provider_manager import ProviderManager
        await self._create(_User(OWNER))
        provider_manager.model_cache.update_models([_model(BYOK_MODEL)])

        with patch.object(ProviderManager, "_deferred_close_providers", AsyncMock()):
            error = await ProviderManager.upsert_byok_provider(provider_manager, BYOK_KEY)
            self.assertIsNone(error)
            self.assertIn(BYOK_KEY, provider_manager.providers)
            self.assertEqual(byok_registry.owner_of(BYOK_KEY), OWNER)

            row = (await self.db.execute(select(ProviderCredentials))).scalar_one()
            row.admin_disabled = True
            await self.db.commit()
            await ProviderManager.upsert_byok_provider(provider_manager, BYOK_KEY)
            self.assertNotIn(BYOK_KEY, provider_manager.providers)
            self.assertFalse(provider_manager.model_cache.has_model_id(BYOK_MODEL))
            self.assertEqual(byok_registry.owner_of(BYOK_KEY), OWNER)  # still denied to others

            await ProviderManager.remove_byok_provider(provider_manager, BYOK_KEY, forget_owner=True)
            self.assertEqual(byok_registry.owner_of(BYOK_KEY), byok.UNKNOWN_OWNER)

    async def test_full_load_maps_owners_and_refuses_reserved_admin_rows(self):
        await self._create(_User(OWNER))
        row = (await self.db.execute(select(ProviderCredentials))).scalar_one()
        row.enabled = False  # disabled rows are still in the owner map
        self.db.add(ProviderCredentials(provider_key="openai:byok-1-legacy", provider_type="custom",
                                        instance_name="byok-1-legacy", provider_name="openai",
                                        base_url="https://a.example.com", api_key="k"))
        await self.db.commit()
        target, owners = {}, {}
        await provider_manager._load_providers_from_database(target=target, owners_target=owners)
        self.assertEqual(owners, {BYOK_KEY: OWNER})
        self.assertNotIn(BYOK_KEY, target)
        self.assertNotIn("openai:byok-1-legacy", target)

    async def test_undecryptable_secret_is_reported_not_fatal(self):
        await self._create(_User(OWNER))
        with patch.dict(os.environ, {"LLMPROXY_BYOK_ENCRYPTION_KEY": "rotated"}):
            target = {}
            await provider_manager._load_providers_from_database(target=target, owners_target={})
            self.assertNotIn(BYOK_KEY, target)
            self.assertIn("decrypt", provider_manager._pending_byok_load_errors[BYOK_KEY])
            listing = await self.routes.list_my_providers(_User(OWNER), self.db)
            self.assertIsNotNone(listing["providers"][0].load_error)

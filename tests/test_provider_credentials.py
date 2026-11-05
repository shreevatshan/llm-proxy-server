"""Provider credential routes: logging, storage encoding and clearing.

* Creating or upserting a provider must not write its secrets to the log.
* ``supported_apis`` is a JSON text column; every write path must encode the
  list, and upsert must keep validator-derived fields and Azure inference.
* An authoritative empty ``deployment_groups`` (switching to discovery) must
  clear the stored deployments, including on rename, upsert and import, while
  omitted secrets on a renaming import are kept.
* Legacy Azure deployment rows survive detail -> unchanged save and
  export -> import without becoming an API allowlist.
"""

import io
import json
import logging
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from app.auth.admin import AdminUser
from app.auth.models import Base, ProviderCredentialsCreate, ProviderCredentialsUpdate
from app.providers.azure_deployments import build_azure_config_fields, normalize_azure_deployments
from app.routes import admin as admin_routes

API_KEY = "sk-sentinel-api-key-7f3a9c"
SECRET = "aws-sentinel-secret-41be2d"
MASK = admin_routes._SECRET_MASK

GROUPS = [
    {"deployments": ["gpt-4o"], "apis": ["chat"]},
    {"deployments": ["claude-sonnet-4"], "apis": ["anthropic"]},
]
LEGACY_JSON = json.dumps({
    "openai": ["gpt-4o", "codex-mini", "text-embedding-3-large"],
    "anthropic": ["claude-sonnet-4"],
    "responses": ["codex-mini"],
})


class _ProviderRouteTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}")
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._session_factory = sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self.db = self._session_factory()
        self.admin = AdminUser(username="admin", email="admin@example.com")
        self.sync = AsyncMock(return_value={"message": "ok"})
        patches = [
            patch.object(admin_routes, "auto_sync_on_provider_change", self.sync),
            patch.object(admin_routes.provider_manager, "refresh_providers_from_database", AsyncMock()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    async def asyncTearDown(self):
        await self.db.close()
        await self._engine.dispose()
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    async def _create(self, upsert=False, **fields):
        return await admin_routes.create_provider_credential(
            ProviderCredentialsCreate.model_validate(fields), upsert=upsert, current_admin=self.admin, db=self.db,
        )

    async def _update(self, key, **fields):
        return await admin_routes.update_provider_credential(
            ProviderCredentialsUpdate.model_validate(fields), provider_key=key, current_admin=self.admin, db=self.db,
        )

    async def _detail(self, key):
        return await admin_routes.get_provider_credential(provider_key=key, current_admin=self.admin, db=self.db)

    async def _row(self, key):
        self.db.expire_all()
        result = await self.db.execute(
            text("SELECT * FROM provider_credentials WHERE provider_key = :k"), {"k": key}
        )
        row = result.mappings().first()
        return dict(row) if row is not None else None

    async def _set_column(self, key, column, value):
        await self.db.execute(
            text(f"UPDATE provider_credentials SET {column} = :v WHERE provider_key = :k"), {"v": value, "k": key}
        )
        await self.db.commit()

    async def _export(self, include_secrets=False):
        response = await admin_routes.export_provider_config(
            include_secrets=include_secrets, current_admin=self.admin, db=self.db,
        )
        body = b"".join([chunk async for chunk in response.body_iterator])
        return json.loads(body)

    async def _import(self, data, overwrite=True):
        upload = UploadFile(file=io.BytesIO(json.dumps(data).encode()), filename="config.json")
        return await admin_routes.import_provider_config(
            file=upload, overwrite=overwrite, sync_models=False, current_admin=self.admin, db=self.db,
        )

    async def _azure(self, instance="primary", **fields):
        return await self._create(**{
            "provider_type": "azure", "provider_name": "azure", "instance_name": instance,
            "endpoint": "https://example.openai.azure.com", "api_key": API_KEY,
            "azure_backend": "foundry", "dynamic_discovery": False, "deployment_groups": GROUPS,
            **fields,
        })

    def _groups(self, row):
        return normalize_azure_deployments(row["deployments_json"])


class CredentialLoggingTests(_ProviderRouteTestCase):
    async def test_create_duplicate_and_upsert_do_not_log_secrets(self):
        bedrock = {
            "provider_type": "bedrock", "provider_name": "bedrock", "instance_name": "main",
            "region": "us-east-1", "access_key_id": "AKIAEXAMPLE", "secret_access_key": SECRET,
        }
        custom = {
            "provider_type": "openai_compatible", "provider_name": "custom", "instance_name": "main",
            "base_url": "https://llm.example.com/v1", "api_key": API_KEY,
        }
        with self.assertLogs(level="INFO") as logs:
            await self._create(**bedrock)
            await self._create(**custom)
            with self.assertRaises(HTTPException) as ctx:
                await self._create(**custom)
            self.assertEqual(ctx.exception.status_code, 400)
            await self._create(upsert=True, **{**custom, "api_key": API_KEY + "-2"})
            await self._create(upsert=True, **{**bedrock, "secret_access_key": SECRET + "-2"})
        output = "\n".join(logs.output)
        self.assertNotIn(API_KEY, output)
        self.assertNotIn(SECRET, output)
        self.assertNotIn("llm.example.com", output)
        self.assertIn("custom:main", output)
        self.assertIn("bedrock:main", output)
        self.assertEqual((await self._row("custom:main"))["api_key"], API_KEY + "-2")
        self.assertEqual((await self._row("bedrock:main"))["secret_access_key"], SECRET + "-2")


class SupportedApisStorageTests(_ProviderRouteTestCase):
    CUSTOM = {
        "provider_type": "openai_compatible", "provider_name": "custom", "instance_name": "main",
        "base_url": "https://llm.example.com/v1", "api_key": API_KEY,
    }

    async def test_upsert_encodes_custom_list(self):
        await self._create(**self.CUSTOM, supported_apis=["openai"])
        response = await self._create(upsert=True, **self.CUSTOM, supported_apis=["openai", "anthropic"])
        self.assertEqual((await self._row("custom:main"))["supported_apis"], '["openai", "anthropic"]')
        self.assertEqual(response.supported_apis, ["openai", "anthropic"])
        self.assertEqual((await self._detail("custom:main")).supported_apis, ["openai", "anthropic"])

    async def test_upsert_keeps_azure_inference_and_derived_deployments(self):
        await self._azure(deployment_groups=[{"deployments": ["gpt-4o"], "apis": ["chat"]}])
        self.assertEqual((await self._row("azure:primary"))["supported_apis"], '["openai"]')
        # The anthropic list only exists as a validator-derived field here.
        response = await self._azure(upsert=True)
        row = await self._row("azure:primary")
        self.assertEqual(row["supported_apis"], '["openai", "anthropic"]')
        self.assertEqual(response.anthropic_deployments, ["claude-sonnet-4"])
        self.assertEqual(self._groups(row)["groups"], GROUPS)

    async def test_put_encodes_lists_and_keeps_masked_secrets(self):
        await self._create(**self.CUSTOM)
        cases = [
            ({"supported_apis": ["anthropic"]}, '["anthropic"]', ["anthropic"]),
            ({"enabled": True}, '["anthropic"]', ["anthropic"]),  # omitted: unchanged
            ({"supported_apis": []}, "[]", []),  # PUT stores an explicit empty list as given
        ]
        for fields, stored, returned in cases:
            with self.subTest(fields=fields):
                response = await self._update("custom:main", api_key=MASK, **fields)
                row = await self._row("custom:main")
                self.assertEqual(row["supported_apis"], stored)
                self.assertEqual(row["api_key"], API_KEY)
                self.assertEqual(response.supported_apis, returned)
                self.assertEqual(response.api_key, MASK)

    async def test_put_rename_encodes_list(self):
        await self._create(**self.CUSTOM)
        response = await self._update(
            "custom:main", instance_name="renamed", supported_apis=["openai", "anthropic"], api_key=MASK,
        )
        self.assertEqual(response.provider_key, "custom:renamed")
        self.assertIsNone(await self._row("custom:main"))
        row = await self._row("custom:renamed")
        self.assertEqual(row["supported_apis"], '["openai", "anthropic"]')
        self.assertEqual(row["api_key"], API_KEY)


class DeploymentClearingTests(_ProviderRouteTestCase):
    STALE = {"deployments": ["gpt-4o"], "openai_deployments": ["gpt-4o"], "anthropic_deployments": ["claude-sonnet-4"]}
    DISCOVERY = {"dynamic_discovery": True, "discovery_api_version": "2024-10-21"}

    async def _assert_cleared(self, key):
        row = await self._row(key)
        self.assertIsNone(row["deployments_json"])
        detail = await self._detail(key)
        self.assertEqual((detail.deployment_groups, detail.deployments), ([], []))
        config = build_azure_config_fields(type("Creds", (), row)())
        self.assertEqual((config["deployments"], config["deployment_apis"]), ([], {}))
        return row

    async def test_empty_groups_clear_on_put(self):
        await self._azure()
        await self._update("azure:primary", deployment_groups=[], **self.STALE, **self.DISCOVERY)
        row = await self._assert_cleared("azure:primary")
        self.assertTrue(row["dynamic_discovery"])
        self.assertEqual(row["api_key"], API_KEY)

    async def test_omitted_or_null_deployments_leave_them_unchanged(self):
        await self._azure()
        before = (await self._row("azure:primary"))["deployments_json"]
        for fields in ({}, {"deployment_groups": None}, {"openai_deployments": None, "deployments": None}):
            with self.subTest(fields=fields):
                await self._update("azure:primary", enabled=True, **fields)
                self.assertEqual((await self._row("azure:primary"))["deployments_json"], before)

    async def test_explicit_false_is_written(self):
        await self._azure(**{"deployment_groups": [], **self.DISCOVERY})
        await self._update("azure:primary", dynamic_discovery=False)
        self.assertIs(bool((await self._row("azure:primary"))["dynamic_discovery"]), False)

    async def test_empty_legacy_lists_clear(self):
        await self._azure()
        await self._update("azure:primary", openai_deployments=[], anthropic_deployments=[], **self.DISCOVERY)
        await self._assert_cleared("azure:primary")

    async def test_empty_groups_clear_on_rename(self):
        await self._azure()
        await self._update("azure:primary", instance_name="renamed", deployment_groups=[], **self.DISCOVERY)
        self.assertIsNone(await self._row("azure:primary"))
        row = await self._assert_cleared("azure:renamed")
        self.assertEqual(row["api_key"], API_KEY)

    async def test_empty_groups_clear_on_upsert(self):
        await self._azure()
        await self._azure(upsert=True, deployment_groups=[], **self.STALE, **self.DISCOVERY)
        await self._assert_cleared("azure:primary")

    async def test_switching_back_to_manual_does_not_resurrect_groups(self):
        await self._azure()
        await self._update("azure:primary", deployment_groups=[], **self.DISCOVERY)
        await self._update("azure:primary", dynamic_discovery=False)
        await self._assert_cleared("azure:primary")

    async def test_renaming_import_keeps_omitted_secrets_and_clears_groups(self):
        await self._azure()
        await self._create(
            provider_type="bedrock", provider_name="bedrock", instance_name="main",
            region="us-east-1", access_key_id="AKIAEXAMPLE", secret_access_key=SECRET,
        )
        result = await self._import({"providers": [
            {
                "provider_key": "azure:primary", "provider_type": "azure", "provider_name": "azure",
                "instance_name": "renamed", "endpoint": "https://example.openai.azure.com",
                "api_key": None, "azure_backend": "foundry", "deployment_groups": [],
                **self.STALE, **self.DISCOVERY,
            },
            {
                "provider_key": "bedrock:main", "provider_type": "bedrock", "provider_name": "bedrock",
                "instance_name": "moved", "region": "us-west-2", "access_key_id": "AKIAEXAMPLE",
                "secret_access_key": None,
            },
        ]})
        self.assertEqual((result.overwritten, result.errors), (2, []))
        self.assertIsNone(await self._row("azure:primary"))
        row = await self._assert_cleared("azure:renamed")
        self.assertEqual(row["api_key"], API_KEY)
        bedrock = await self._row("bedrock:moved")
        self.assertEqual((bedrock["secret_access_key"], bedrock["region"]), (SECRET, "us-west-2"))


class LegacyRoundTripTests(_ProviderRouteTestCase):
    async def _legacy_provider(self, instance="legacy"):
        await self._azure(instance=instance, deployment_groups=[{"deployments": ["placeholder"], "apis": ["chat"]}])
        await self._set_column(f"azure:{instance}", "deployments_json", LEGACY_JSON)

    def _assert_legacy(self, row):
        groups = self._groups(row)
        self.assertEqual([g.get("legacy_mode") for g in groups["groups"]], ["openai", "responses", "anthropic"])
        self.assertEqual(groups["responses"], ["codex-mini"])
        self.assertEqual(build_azure_config_fields(type("Creds", (), row)())["deployment_apis"], {})

    async def test_detail_then_unchanged_save(self):
        await self._legacy_provider()
        detail = await self._detail("azure:legacy")
        self.assertTrue(all(not g.apis and g.legacy_mode for g in detail.deployment_groups))
        # What the form sends back when nothing was edited.
        await self._update(
            "azure:legacy", api_key=detail.api_key, dynamic_discovery=False,
            deployment_groups=[g.model_dump() for g in detail.deployment_groups],
            openai_deployments=detail.openai_deployments, anthropic_deployments=detail.anthropic_deployments,
        )
        self._assert_legacy(await self._row("azure:legacy"))

        # Selecting APIs deliberately turns a row explicit.
        rows = [g.model_dump() for g in detail.deployment_groups]
        rows[0] = {"deployments": ["gpt-4o"], "apis": ["chat"]}
        rows.insert(1, {"deployments": ["text-embedding-3-large"], "apis": [], "legacy_mode": "openai"})
        await self._update("azure:legacy", deployment_groups=rows)
        config = build_azure_config_fields(type("Creds", (), await self._row("azure:legacy"))())
        self.assertEqual(config["deployment_apis"], {"gpt-4o": ["chat"]})
        self.assertEqual(config["responses_deployments"], ["codex-mini"])

    async def test_export_then_import(self):
        await self._legacy_provider()
        await self._azure(instance="explicit")
        await self._azure(instance="discovery", **{"deployment_groups": [], **DeploymentClearingTests.DISCOVERY})
        before = {k: await self._row(f"azure:{k}") for k in ("legacy", "explicit", "discovery")}

        exported = await self._export()
        self.assertTrue(all(p["api_key"] is None for p in exported["providers"]))
        result = await self._import(exported)
        self.assertEqual((result.overwritten, result.errors), (3, []))

        for key, old in before.items():
            with self.subTest(key=key):
                row = await self._row(f"azure:{key}")
                self.assertEqual(row["api_key"], API_KEY)
                self.assertEqual(self._groups(row)["groups"], self._groups(old)["groups"])
        self._assert_legacy(await self._row("azure:legacy"))
        self.assertIsNone((await self._row("azure:discovery"))["deployments_json"])



REFRESH = "1000.sentinel-refresh-token-9d2e"
CLIENT_SECRET = "sentinel-client-secret-5c1a"
PAI_API_KEY = "193-sentinel-api-key-77ab"


class PlatformAICredentialTests(_ProviderRouteTestCase):
    PLATFORMAI = {
        "provider_type": "platformai", "provider_name": "platformai", "instance_name": "main",
        "oauth_client_id": "1000.CLIENT", "oauth_client_secret": CLIENT_SECRET,
        "oauth_refresh_token": REFRESH, "deployments": ["claude-haiku-5-5", " gpt-5.5 ", "claude-haiku-5-5"],
    }
    API_KEY_PROVIDER = {
        "provider_type": "platformai", "provider_name": "platformai", "instance_name": "key",
        "api_key": PAI_API_KEY, "deployments": ["claude-haiku-5-5"],
    }

    async def test_create_stores_and_masks_secrets_without_logging_them(self):
        with self.assertLogs(level="INFO") as logs:
            response = await self._create(**self.PLATFORMAI)
        output = "\n".join(logs.output)
        self.assertNotIn(REFRESH, output)
        self.assertNotIn(CLIENT_SECRET, output)
        row = await self._row("platformai:main")
        self.assertEqual((row["oauth_refresh_token"], row["oauth_client_secret"]), (REFRESH, CLIENT_SECRET))
        self.assertIsNone(row["region"])
        self.assertEqual(row["supported_apis"], '["openai", "anthropic"]')
        self.assertEqual(row["deployments_json"], '["claude-haiku-5-5", "gpt-5.5"]')
        self.assertEqual((response.oauth_refresh_token, response.oauth_client_secret), (MASK, MASK))
        self.assertEqual(response.deployments, ["claude-haiku-5-5", "gpt-5.5"])
        detail = await self._detail("platformai:main")
        self.assertEqual(detail.oauth_refresh_token, MASK)
        self.assertEqual(detail.oauth_client_id, "1000.CLIENT")

    async def test_create_with_api_key(self):
        with self.assertLogs(level="INFO") as logs:
            response = await self._create(**self.API_KEY_PROVIDER, pai_token_id="5434566543234")
        self.assertNotIn(PAI_API_KEY, "\n".join(logs.output))
        row = await self._row("platformai:key")
        self.assertEqual((row["api_key"], row["pai_token_id"]), (PAI_API_KEY, "5434566543234"))
        self.assertIsNone(row["oauth_refresh_token"])
        self.assertEqual((response.api_key, response.pai_token_id), (MASK, "5434566543234"))

    async def test_create_requires_api_key_or_oauth_pair(self):
        for field in ("oauth_refresh_token", "oauth_client_id"):
            with self.subTest(field=field):
                with self.assertRaises(HTTPException) as ctx:
                    await self._create(**{**self.PLATFORMAI, field: " "})
                self.assertEqual(ctx.exception.status_code, 422)
                self.assertIn(field, ctx.exception.detail)
        with self.assertRaises(HTTPException):
            await self._create(**{**self.PLATFORMAI, "oauth_refresh_token": MASK})
        with self.assertRaises(HTTPException):
            await self._create(**{**self.API_KEY_PROVIDER, "api_key": MASK})

    def test_model_names_validated(self):
        with self.assertRaises(ValueError):
            ProviderCredentialsCreate.model_validate({**self.API_KEY_PROVIDER, "deployments": ["bad name"]})

    async def test_upsert_keeps_stored_secrets_when_omitted(self):
        await self._create(**self.PLATFORMAI)
        fields = {k: v for k, v in self.PLATFORMAI.items() if k not in ("oauth_refresh_token", "oauth_client_secret")}
        await self._create(upsert=True, **fields, pai_token_id="42")
        row = await self._row("platformai:main")
        self.assertEqual((row["oauth_refresh_token"], row["pai_token_id"]), (REFRESH, "42"))

    async def test_put_keeps_masked_secrets_and_clears_overrides(self):
        await self._create(**self.PLATFORMAI)
        await self._update("platformai:main", oauth_refresh_token=MASK, oauth_client_secret=MASK,
                           pai_token_id="42", deployments=["gpt-5.5", ""])
        row = await self._row("platformai:main")
        self.assertEqual((row["oauth_refresh_token"], row["oauth_client_secret"]), (REFRESH, CLIENT_SECRET))
        self.assertEqual((row["pai_token_id"], row["deployments_json"]), ("42", '["gpt-5.5"]'))
        await self._update("platformai:main", pai_token_id="")  # empty clears the override
        self.assertIsNone((await self._row("platformai:main"))["pai_token_id"])
        with self.assertRaises(HTTPException) as ctx:
            await self._update("platformai:main", deployments=["bad name"])
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_switching_api_key_to_oauth_clears_the_key(self):
        await self._create(**self.API_KEY_PROVIDER)
        await self._update("platformai:key", api_key="", oauth_client_id="1000.C", oauth_refresh_token=REFRESH)
        row = await self._row("platformai:key")
        self.assertIsNone(row["api_key"])
        self.assertEqual(row["oauth_refresh_token"], REFRESH)

    async def test_rename_keeps_oauth_columns(self):
        await self._create(**{**self.PLATFORMAI, "oauth_accounts_url": "https://accounts.example",
                              "pai_token_id": "42"})
        await self._update("platformai:main", instance_name="renamed", oauth_refresh_token=MASK)
        row = await self._row("platformai:renamed")
        self.assertEqual(row["oauth_refresh_token"], REFRESH)
        self.assertEqual(row["oauth_accounts_url"], "https://accounts.example")
        self.assertEqual(row["oauth_client_id"], "1000.CLIENT")
        self.assertEqual(row["pai_token_id"], "42")

    async def test_export_gates_secrets_and_import_round_trips(self):
        await self._create(**{**self.PLATFORMAI, "pai_token_id": "42"})
        redacted = (await self._export())["providers"][0]
        self.assertIsNone(redacted["oauth_refresh_token"])
        self.assertIsNone(redacted["oauth_client_secret"])
        self.assertEqual((redacted["oauth_client_id"], redacted["pai_token_id"]), ("1000.CLIENT", "42"))
        full = await self._export(include_secrets=True)
        self.assertEqual(full["providers"][0]["oauth_refresh_token"], REFRESH)
        await self._set_column("platformai:main", "oauth_refresh_token", "stale")
        await self._import(full)
        self.assertEqual((await self._row("platformai:main"))["oauth_refresh_token"], REFRESH)


if __name__ == "__main__":
    unittest.main()

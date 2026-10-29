"""Admin-declared outages: a provider instance or single models taken out of service.

A model in outage is unavailable to every caller, admins included, so requests go to
the fallback mappings; with no fallback they fail with 503 quoting the admin's
description. Access is judged first, so a caller who may not use a model still gets
403. The deployment-style Azure routes, which dispatch to the URL's provider whatever
the mapping says, refuse an outage rather than reach it. See app/outages.py.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from app.auth import database
from app.auth.admin import AdminUser
from app.auth.models import ModelOutage, OutageBulkResolve, OutageCreate, OutageUpdate
from app.model_access_dep import ModelAccessDenied
from app.model_resolution import (
    _evaluate, _may_use_for, candidate_ids, is_model_available, map_model_name, ModelUnavailable,
)
from app.outages import Outage, outage_registry
from app.rate_limit_dep import enforce_access_and_limits
from app.routes import admin as admin_routes
from app.routes import anthropic_messages
from tests.test_model_groups import ADMIN, _GroupTestCase
from tests.test_model_resolution import (
    _Env, _fresh_alias_context, _grant, _request, _resolve_fresh, _Rules, _User,
)

MODELS = ["p:a/gpt-5.4", "p:a/gpt-5.5", "q:b/gpt-5.4"]


def _outage(target, description="upstream down", id=None):
    scope = "model" if "/" in target else "provider"
    return Outage(id=id or abs(hash(target)) % 10_000, scope=scope, target=target, description=description)


class _Outages:
    """Declare these outages for one test, restoring the registry after."""

    def __init__(self, *outages):
        self.outages = outages

    def __enter__(self):
        self.saved = outage_registry._state
        outage_registry.install(self.outages)
        return self

    def __exit__(self, *exc):
        outage_registry._state = self.saved


def _status(name, auth=None, api=None):
    return _evaluate(name, _may_use_for(auth), api).status


# ======================================================================
# The registry
# ======================================================================

class RegistryTests(unittest.TestCase):

    def test_a_model_outage_beats_its_instances(self):
        with _Outages(_outage("p:a", "instance down"), _outage("p:a/gpt-5.4", "model down")):
            self.assertEqual(outage_registry.outage_for("p:a/gpt-5.4").description, "model down")
            self.assertEqual(outage_registry.outage_for("p:a/gpt-5.5").description, "instance down")

    def test_an_instance_outage_covers_models_synced_later(self):
        with _Outages(_outage("p:a")):
            self.assertTrue(outage_registry.is_in_outage("p:a/brand-new"))
            self.assertFalse(outage_registry.is_in_outage("q:b/gpt-5.4"))
            self.assertFalse(outage_registry.is_in_outage("gpt-5.4"))
            self.assertFalse(outage_registry.is_in_outage(None))


# ======================================================================
# Evaluation
# ======================================================================

class EvaluationTests(unittest.TestCase):

    def test_an_exact_id_in_outage(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")):
            self.assertEqual(_status("p:a/gpt-5.4"), "outage")
            self.assertEqual(_status("p:a/gpt-5.5"), "ok")

    def test_an_uncached_deployment_behind_a_provider_prefix(self):
        with _Env(MODELS), _Outages(_outage("p:a")):
            self.assertEqual(_status("p:a/some-deployment"), "outage")

    def test_a_bare_name_keeps_its_healthy_candidates(self):
        with _Env(MODELS), _Outages(_outage("p:a")):
            ev = _evaluate("gpt-5.4", _may_use_for(None), None)
            self.assertEqual((ev.status, ev.candidates), ("ok", ("q:b/gpt-5.4",)))
            self.assertEqual(candidate_ids("gpt-5.4"), ["q:b/gpt-5.4"])

    def test_a_bare_name_with_every_candidate_down_quotes_each_description_once(self):
        with _Env(MODELS), _Outages(_outage("p:a", "same"), _outage("q:b", "same")):
            ev = _evaluate("gpt-5.4", _may_use_for(None), None)
            self.assertEqual(ev.status, "outage")
            self.assertEqual(len(ev.outages), 2)
            self.assertEqual(candidate_ids("gpt-5.4"), [])

    def test_admins_and_allow_mode_profiles_get_no_bypass(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")):
            _grant(self, 7, "allow")
            self.assertEqual(_status("p:a/gpt-5.4", admin), "outage")
            self.assertEqual(_status("p:a/gpt-5.4", _User(7)), "outage")

    def test_a_denied_caller_is_told_so_not_about_the_outage(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")):
            _grant(self, 7, "custom", {"p:a/gpt-5.4": False})
            self.assertEqual(_status("p:a/gpt-5.4", _User(7)), "denied")

    def test_availability_can_ignore_outages_for_the_mapping_list_warning(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.5")):
            may_use = _may_use_for(None)
            self.assertFalse(is_model_available("p:a/gpt-5.5", may_use, None))
            self.assertTrue(is_model_available("p:a/gpt-5.5", may_use, None, ignore_outage=True))


# ======================================================================
# Resolution: fallback, and the 503
# ======================================================================

class ResolutionTests(unittest.TestCase):

    def test_a_model_in_outage_uses_the_fallback_list(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")), \
             _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "p:a/gpt-5.4"), ("p:a/gpt-5.5", "p:a/gpt-5.4"))

    def test_admins_go_to_the_fallback_too(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")), \
             _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), admin, "p:a/gpt-5.4")[0], "p:a/gpt-5.5")

    def test_without_a_fallback_the_caller_gets_503_with_the_description(self):
        cases = {
            "/v1/chat/completions": lambda body: (body["error"]["code"], body["error"]["message"]),
            "/v1/messages": lambda body: (body["error"]["type"], body["error"]["message"]),
            "/openai/v1/chat/completions": lambda body: (body["error"]["code"], body["error"]["message"]),
        }
        expected_kind = {"/v1/messages": "api_error"}
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4", "Azure East 5xx since 10:30")), _Rules():
            for path, read in cases.items():
                with self.subTest(path=path):
                    with self.assertRaises(ModelUnavailable) as ctx:
                        _resolve_fresh(_request(path), _User(7), "p:a/gpt-5.4")
                    self.assertEqual(ctx.exception.status_code, 503)
                    kind, message = read(ctx.exception.body)
                    self.assertEqual(kind, expected_kind.get(path, "model_outage"))
                    self.assertIn("Azure East 5xx since 10:30", message)
                    self.assertIn("'p:a/gpt-5.4'", message)

    def test_a_fallback_target_in_outage_too_keeps_the_original_503(self):
        with _Env(MODELS), _Outages(_outage("p:a")), _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve_fresh(_request(), _User(7), "p:a/gpt-5.4")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("'p:a/gpt-5.4'", ctx.exception.body["error"]["message"])

    def test_a_regular_target_in_outage_is_rescued_by_the_fallback_list(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.5")), \
             _Rules(regular={"gpt-5.4": "p:a/gpt-5.5"}, fallback={"gpt-5.4": "q:b/gpt-5.4"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-5.4")[0], "q:b/gpt-5.4")

    def test_a_regular_target_in_outage_without_a_fallback_is_503(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.5")), _Rules(regular={"gpt-5.4": "p:a/gpt-5.5"}):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve_fresh(_request(), _User(7), "gpt-5.4")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("'p:a/gpt-5.5'", ctx.exception.body["error"]["message"])

    def test_a_denied_caller_still_gets_403(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")), _Rules(profile_id=9):
            _grant(self, 7, "custom", {"p:a/gpt-5.4": False})
            with self.assertRaises(ModelAccessDenied):
                _resolve_fresh(_request(), _User(7), "p:a/gpt-5.4")

    def test_map_model_name_applies_the_fallback_for_an_outage(self):
        async def run(name, auth):
            _fresh_alias_context()
            return map_model_name(name, auth)

        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")), _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}):
            self.assertEqual(asyncio.run(run("p:a/gpt-5.4", _User(7))), "p:a/gpt-5.5")


class BackstopTests(unittest.TestCase):
    """enforce_access_and_limits: 403 before 503, and 503 for a model in outage."""

    def test_a_model_in_outage_is_refused(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")):
            with self.assertRaises(ModelUnavailable) as ctx:
                asyncio.run(enforce_access_and_limits(
                    _request(), None, "p:a/gpt-5.4", envelope_override="azure", consume_quota=False))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_access_is_judged_first(self):
        with _Env(MODELS), _Outages(_outage("p:a/gpt-5.4")):
            _grant(self, 7, "custom", {"p:a/gpt-5.4": False})
            with self.assertRaises(ModelAccessDenied):
                asyncio.run(enforce_access_and_limits(
                    _request(), _User(7), "p:a/gpt-5.4", consume_quota=False))

    def test_count_tokens_does_not_degrade_onto_a_provider_in_outage(self):
        provider = SimpleNamespace(anthropic_count_tokens=AsyncMock(return_value={"input_tokens": 1}))
        with _Env(["p:a/claude-x"]), _Outages(_outage("p:a")), _Rules(), \
             patch.object(anthropic_messages.provider_manager, "get_anthropic_provider_for_model",
                          AsyncMock(return_value=provider)):
            with self.assertRaises(ModelUnavailable) as ctx:
                asyncio.run(anthropic_messages.count_message_tokens(
                    request_obj=_request("/v1/messages/count_tokens"),
                    payload={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
                    auth=None,
                ))
        self.assertEqual(ctx.exception.status_code, 503)
        provider.anthropic_count_tokens.assert_not_awaited()


# ======================================================================
# Azure deployment routes: the URL's provider is called whatever the mapping says
# ======================================================================

class AzureDeploymentOutageTests(unittest.TestCase):

    def setUp(self):
        from app.main import create_azure_openai_app
        from app.routes import azure_openai
        self.app = create_azure_openai_app()
        self.app.dependency_overrides[azure_openai._authenticate_azure] = \
            lambda: AdminUser(username="admin", email="admin@example.com")

    def _chat(self, provider_key, deployment, rules=None):
        from tests.test_azure_foundry_deployment_surface import _fake_chat_completion, _Registry
        calls = AsyncMock(side_effect=_fake_chat_completion)
        with _Registry() as registry, _Rules(**(rules or {})), \
             patch("app.request_tracker.request_tracker.start_request", new_callable=AsyncMock), \
             patch("app.request_tracker.request_tracker.end_request", new_callable=AsyncMock):
            registry.foundry.chat_completion = calls
            registry.classic.chat_completion = calls
            response = TestClient(self.app).post(
                f"/openai/deployments/{provider_key}/{deployment}/chat/completions?api-version=2024-10-21",
                json={"model": "ignored", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            )
        return response, calls

    def test_an_instance_in_outage_is_not_reached_through_a_cross_provider_fallback(self):
        with _Outages(_outage("azure:foundry", "foundry is down")):
            response, calls = self._chat(
                "azure:foundry", "grok-3", {"fallback": {"azure:foundry/grok-3": "azure:classic/grok-3"}})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "model_outage")
        self.assertIn("foundry is down", response.json()["error"]["message"])
        calls.assert_not_awaited()

    def test_a_model_in_outage_without_a_fallback_is_refused(self):
        with _Outages(_outage("azure:classic/grok-3")):
            response, calls = self._chat("azure:classic", "grok-3")
        self.assertEqual(response.status_code, 503)
        calls.assert_not_awaited()

    def test_a_fallback_on_the_same_healthy_instance_is_served(self):
        with _Outages(_outage("azure:classic/grok-3")):
            response, calls = self._chat(
                "azure:classic", "grok-3", {"fallback": {"azure:classic/grok-3": "azure:classic/claude-sonnet-4"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "azure:classic/claude-sonnet-4")


# ======================================================================
# Storage, reference cleanups and the admin API
# ======================================================================

class OutageStorageTests(_GroupTestCase):

    async def rows(self):
        self.db.expire_all()
        return sorted((r.scope, r.target, r.description)
                      for r in (await self.db.execute(select(ModelOutage))).scalars().all())

    async def test_reporting_twice_keeps_one_row_and_takes_the_new_description(self):
        await self.provider("azure:east", "gpt-5")
        first = await database.create_outages(self.db, ["azure:east"], [], "first")
        again = await database.create_outages(self.db, ["azure:east"], ["azure:east/gpt-5"], "second")
        self.assertEqual(first[0].id, again[0].id)
        self.assertEqual(await self.rows(), [
            ("model", "azure:east/gpt-5", "second"), ("provider", "azure:east", "second"),
        ])

    async def test_edit_and_resolve(self):
        await self.provider("azure:east", "gpt-5")
        row = (await database.create_outages(self.db, [], ["azure:east/gpt-5"], "down"))[0]
        await database.update_outage_description(self.db, row.id, "still down")
        self.assertEqual(await self.rows(), [("model", "azure:east/gpt-5", "still down")])
        self.assertEqual(await database.delete_outages(self.db, [row.id, 999]), [row.id])
        self.assertEqual(await self.rows(), [])

    async def test_a_rename_carries_outages_along(self):
        await self.provider("az_1:east", "gpt-5")
        await database.create_outages(self.db, ["az_1:east"], ["az_1:east/gpt-5"], "down")
        await database._rename_provider_references(self.db, "az_1:east", "az_1:west")
        await self.db.commit()
        self.assertEqual(await self.rows(), [
            ("model", "az_1:west/gpt-5", "down"), ("provider", "az_1:west", "down"),
        ])

    async def test_deleting_a_provider_ends_its_outages_and_reloads_the_registry(self):
        await self.provider("azure:east", "gpt-5")
        await self.provider("azure:west", "gpt-5")
        await database.create_outages(self.db, ["azure:east", "azure:west"], ["azure:east/gpt-5"], "down")
        await outage_registry.load_from_database()
        self.assertTrue(outage_registry.is_in_outage("azure:east/gpt-5"))
        self.assertTrue(await database.delete_provider_credentials(self.db, "azure:east"))
        self.assertEqual(await self.rows(), [("provider", "azure:west", "down")])
        await database.reload_reference_caches()
        self.assertFalse(outage_registry.is_in_outage("azure:east/gpt-5"))
        self.assertTrue(outage_registry.is_in_outage("azure:west/gpt-5"))

    async def test_removing_a_stale_model_ends_only_its_own_outage(self):
        await self.provider("azure:east", "gpt-5")
        await database.create_outages(self.db, ["azure:east"], ["azure:east/gpt-5"], "down")
        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            await database.delete_stale_models(self.db, ["azure:east/gpt-5"])
        self.assertEqual(await self.rows(), [("provider", "azure:east", "down")])

    async def test_a_model_sync_leaves_outages_alone(self):
        await self.provider("azure:east", "gpt-5")
        await database.create_outages(self.db, [], ["azure:east/gpt-5"], "down")
        with patch("app.auth.database._update_cache_after_database_change", AsyncMock()):
            await database.refresh_models_from_providers(self.db, [
                {"model_id": "azure:east/gpt-5", "provider_key": "azure:east", "model_name": "gpt-5"},
            ])
        self.assertEqual(await self.rows(), [("model", "azure:east/gpt-5", "down")])

    async def test_clear_outages(self):
        await self.provider("azure:east", "gpt-5")
        await database.create_outages(self.db, ["azure:east"], [], "down")
        await database.clear_outages(self.db)
        await self.db.commit()
        self.assertEqual(await self.rows(), [])


class OutageApiTests(_GroupTestCase):

    async def test_report_list_edit_and_resolve(self):
        await self.provider("azure:east", "gpt-5", "gpt-6")
        created = await admin_routes.create_outages_endpoint(
            OutageCreate(description="  East is down  ", provider_keys=["azure:east"]), ADMIN, self.db)
        self.assertEqual([(o.scope, o.target, o.description, o.created_by) for o in created],
                         [("provider", "azure:east", "East is down", "root")])
        self.assertIsNotNone(created[0].created_at)
        self.assertTrue(outage_registry.is_in_outage("azure:east/gpt-6"), "reloaded right away")

        listed = await admin_routes.list_outages_endpoint(ADMIN, self.db)
        self.assertEqual([o.target for o in listed], ["azure:east"])
        self.assertTrue(listed[0].target_exists)

        edited = await admin_routes.update_outage_endpoint(OutageUpdate(description="better"), created[0].id, ADMIN, self.db)
        self.assertEqual(edited.description, "better")
        self.assertEqual(outage_registry.outage_for("azure:east/gpt-5").description, "better")

        await admin_routes.resolve_outage_endpoint(created[0].id, ADMIN, self.db)
        self.assertFalse(outage_registry.is_in_outage("azure:east/gpt-5"))
        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.resolve_outage_endpoint(created[0].id, ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_bulk_resolve_reports_what_was_already_gone(self):
        await self.provider("azure:east", "gpt-5")
        created = await admin_routes.create_outages_endpoint(
            OutageCreate(description="down", provider_keys=["azure:east"], model_ids=["azure:east/gpt-5"]),
            ADMIN, self.db)
        ids = [o.id for o in created]
        result = await admin_routes.bulk_resolve_outages_endpoint(OutageBulkResolve(ids=ids + [999]), ADMIN, self.db)
        self.assertEqual((sorted(result["deleted"]), result["missing"]), (sorted(ids), [999]))
        self.assertEqual(outage_registry.all(), [])

    async def test_unknown_targets_are_refused(self):
        await self.provider("azure:east", "gpt-5")
        with self.assertRaises(HTTPException) as ctx:
            await admin_routes.create_outages_endpoint(
                OutageCreate(description="down", provider_keys=["azure:nope"], model_ids=["azure:east/gpt-9"]),
                ADMIN, self.db)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("azure:nope", ctx.exception.detail)
        self.assertIn("azure:east/gpt-9", ctx.exception.detail)

    async def test_reinit_ends_every_outage(self):
        await self.provider("azure:east", "gpt-5")
        await database.create_outages(self.db, ["azure:east"], [], "down")
        with patch.object(admin_routes.provider_manager, "refresh_providers_from_database", AsyncMock()):
            await admin_routes.reinit_system(ADMIN, self.db)
        self.assertEqual((await self.db.execute(select(ModelOutage))).scalars().all(), [])


class OutageValidationTests(unittest.TestCase):

    def test_a_description_is_required(self):
        for description in ("", "   "):
            with self.subTest(description=description), self.assertRaises(ValidationError):
                OutageCreate(description=description, provider_keys=["azure:east"])

    def test_the_description_is_short(self):
        with self.assertRaises(ValidationError):
            OutageCreate(description="x" * 201, provider_keys=["azure:east"])
        self.assertEqual(len(OutageCreate(description="x" * 200, provider_keys=["a:b"]).description), 200)

    def test_something_must_be_selected(self):
        with self.assertRaises(ValidationError):
            OutageCreate(description="down")

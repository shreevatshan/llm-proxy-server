"""Prefix-less model resolution: the bare-name index, the pool filters, the
round-robin pick, and the two failure statuses.

The canonical model id is ``{provider_key}/{bare_name}``; a bare name may itself
contain '/' (``lmstudio:box/meta-llama/Llama-3.1-8B``), which is why the
resolution order tries the whole string before stripping a head. See
app/model_resolution.py.
"""

import asyncio
import time
import unittest
from types import SimpleNamespace

from app import model_resolution
from app.auth.admin import AdminUser
from app.auth.middleware import _enforce_rate_limit
from app.cache import ModelCache
from app.model_access_dep import ModelAccessDenied
from app.model_alias import current_api_surface, original_model_name
from app.model_resolution import (
    ModelUnavailable,
    candidate_ids,
    resolve_model_for_request,
)
from app.openai_models import ModelInfo
from app.providers.provider_manager import provider_manager


def _model(model_id: str) -> ModelInfo:
    return ModelInfo(id=model_id, created=0, owned_by="test", provider="test")


class _StubProvider:
    """Minimal provider double: only the surface _filter_pool touches."""

    def __init__(self, full_provider_name: str, apis=("openai", "anthropic")):
        self.full_provider_name = full_provider_name
        self._apis = set(apis)

    def supports_api_for_model(self, model_name: str, api_name: str) -> bool:
        return api_name in self._apis


class _ExplodingProvider(_StubProvider):
    def supports_api_for_model(self, model_name, api_name):
        raise RuntimeError("heuristic blew up")


class _Env:
    """Swap provider_manager's cache and provider registry for a test.

    Restores both plus the API-surface ContextVar, so nothing leaks into the
    process-wide singletons the rest of the suite shares.
    """

    def __init__(self, model_ids=(), providers=None, disabled_models=(),
                 disabled_providers=(), api=None):
        self.model_ids = list(model_ids)
        self.providers = providers
        self.disabled_models = set(disabled_models)
        self.disabled_providers = set(disabled_providers)
        self.api = api

    def __enter__(self):
        self.cache = ModelCache()
        self.cache.update_models([_model(m) for m in self.model_ids])
        self.cache._model_configs = {m: False for m in self.disabled_models}
        self.cache._provider_configs = {p: False for p in self.disabled_providers}

        if self.providers is None:
            keys = {m.split('/', 1)[0] for m in self.model_ids if '/' in m}
            self.providers = {k: _StubProvider(k) for k in keys}

        self._saved_cache = provider_manager.model_cache
        self._saved_providers = provider_manager.providers
        provider_manager.model_cache = self.cache
        provider_manager.providers = self.providers
        self._api_token = current_api_surface.set(self.api)
        self._name_token = original_model_name.set(None)
        return self

    def __exit__(self, *exc):
        provider_manager.model_cache = self._saved_cache
        provider_manager.providers = self._saved_providers
        current_api_surface.reset(self._api_token)
        original_model_name.reset(self._name_token)


def _request(path="/v1/chat/completions"):
    """A request object with only what resolution reads: url.path and state."""
    return SimpleNamespace(url=SimpleNamespace(path=path), state=SimpleNamespace())


def _resolve(request_obj, auth, model, **kw):
    return asyncio.run(resolve_model_for_request(request_obj, auth, model, **kw))


def _resolve_with_echo(request_obj, auth, model, **kw):
    """Resolve and read the echo ContextVar *inside* the same context.

    asyncio.run() copies the context, so a ContextVar set by the coroutine is
    invisible to the caller here -- unlike production, where the handler awaits
    resolution within its own task.
    """
    async def run():
        chosen = await resolve_model_for_request(request_obj, auth, model, **kw)
        return chosen, original_model_name.get()

    return asyncio.run(run())


class _User:
    def __init__(self, user_id):
        self.id = user_id


# ======================================================================
# The bare-name index on ModelCache
# ======================================================================

class BareNameIndexTests(unittest.TestCase):
    def test_exact_id_hit_and_miss(self):
        cache = ModelCache()
        cache.update_models([_model("azure:foundry/gpt-5.4")])
        self.assertTrue(cache.has_model_id("azure:foundry/gpt-5.4"))
        self.assertFalse(cache.has_model_id("gpt-5.4"))
        self.assertFalse(cache.has_model_id(""))
        self.assertFalse(cache.has_model_id(None))

    def test_two_providers_for_one_bare_name_are_sorted(self):
        cache = ModelCache()
        cache.update_models([
            _model("openai:main/gpt-5.4"),
            _model("azure:foundry/gpt-5.4"),
        ])
        self.assertEqual(
            cache.bare_model_candidates("gpt-5.4"),
            ["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"],
        )
        self.assertEqual(cache.bare_model_candidates("nope"), [])
        self.assertEqual(cache.bare_model_candidates(""), [])

    def test_bare_name_may_itself_contain_a_slash(self):
        """Only the first '/' separates provider key from bare name."""
        cache = ModelCache()
        cache.update_models([_model("lmstudio:box/openai/gpt-oss-120b")])
        self.assertEqual(
            cache.bare_model_candidates("openai/gpt-oss-120b"),
            ["lmstudio:box/openai/gpt-oss-120b"],
        )
        # Not indexed under the tail alone.
        self.assertEqual(cache.bare_model_candidates("gpt-oss-120b"), [])

    def test_ids_without_a_prefix_are_addressable_but_not_indexed(self):
        cache = ModelCache()
        cache.update_models([_model("bare-id")])
        self.assertTrue(cache.has_model_id("bare-id"))
        self.assertEqual(cache.bare_model_candidates("bare-id"), [])

    def test_index_rebuilds_after_every_write_path(self):
        cache = ModelCache()
        cache.update_models([_model("azure:foundry/gpt-5.4")])
        self.assertEqual(cache.bare_model_candidates("gpt-5.4"),
                         ["azure:foundry/gpt-5.4"])

        cache.update_models([_model("openai:main/gpt-5.4")])
        self.assertEqual(cache.bare_model_candidates("gpt-5.4"),
                         ["openai:main/gpt-5.4"])

        cache.invalidate_model("openai:main/gpt-5.4")
        self.assertEqual(cache.bare_model_candidates("gpt-5.4"), [])

        cache.update_models([_model("openai:main/gpt-5.4"), _model("openai:main/o3")])
        cache.invalidate_provider("openai:main")
        self.assertEqual(cache.bare_model_candidates("gpt-5.4"), [])
        self.assertFalse(cache.has_model_id("openai:main/o3"))

        asyncio.run(cache.update_provider_models(
            "azure:foundry", [_model("azure:foundry/gpt-5.4")]
        ))
        self.assertEqual(cache.bare_model_candidates("gpt-5.4"),
                         ["azure:foundry/gpt-5.4"])

    def test_resubmitting_the_same_list_object_still_rebuilds(self):
        """Identity alone would miss this; the _last_updated stamp catches it."""
        cache = ModelCache()
        models = [_model("azure:foundry/gpt-5.4")]
        cache.update_models(models)
        self.assertEqual(cache.bare_model_candidates("o3"), [])

        models.append(_model("azure:foundry/o3"))
        time.sleep(0.001)  # guarantee a distinct time.time() stamp
        cache.update_models(models)
        self.assertEqual(cache.bare_model_candidates("o3"), ["azure:foundry/o3"])


# ======================================================================
# Resolution order
# ======================================================================

class ResolutionOrderTests(unittest.TestCase):
    def test_exact_canonical_id_passes_through(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(
                _resolve(_request(), None, "azure:foundry/gpt-5.4"),
                "azure:foundry/gpt-5.4",
            )

    def test_bare_name_resolves_to_the_canonical_id(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(_resolve(_request(), None, "gpt-5.4"),
                             "azure:foundry/gpt-5.4")

    def test_live_provider_prefix_is_honoured_even_when_model_is_uncached(self):
        """Azure deployments are often undiscovered; today's behaviour must hold."""
        with _Env([], providers={"azure:primary": _StubProvider("azure:primary")}):
            self.assertEqual(_resolve(_request(), None, "azure:primary/whatever"),
                             "azure:primary/whatever")

    def test_dead_provider_prefix_falls_back_to_a_live_one(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(_resolve(_request(), None, "azure:gone/gpt-5.4"),
                             "azure:foundry/gpt-5.4")

    def test_whole_string_is_tried_as_a_bare_name_before_stripping(self):
        with _Env(["lmstudio:box/meta-llama/Llama-3.1-8B"]):
            self.assertEqual(
                _resolve(_request(), None, "meta-llama/Llama-3.1-8B"),
                "lmstudio:box/meta-llama/Llama-3.1-8B",
            )

    def test_head_is_stripped_exactly_once(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            # "someprovider" is not live, so the head is dropped and "gpt-5.4" hits.
            self.assertEqual(_resolve(_request(), None, "someprovider/gpt-5.4"),
                             "azure:foundry/gpt-5.4")

    def test_falsy_names_pass_through_untouched(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertIsNone(_resolve(_request(), None, None))
            self.assertEqual(_resolve(_request(), None, ""), "")

    def test_no_trimming_and_no_case_folding(self):
        """A silent mutation would desync the usage row from what the client sent."""
        with _Env(["azure:foundry/gpt-5.4"]):
            for name in (" gpt-5.4", "GPT-5.4", "azure:foundry/"):
                with self.subTest(name=name):
                    with self.assertRaises(ModelUnavailable):
                        _resolve(_request(), None, name)


# ======================================================================
# Candidate pool filters
# ======================================================================

class PoolFilterTests(unittest.TestCase):
    def test_unregistered_provider_is_dropped(self):
        with _Env(["azure:gone/gpt-5.4", "azure:foundry/gpt-5.4"],
                  providers={"azure:foundry": _StubProvider("azure:foundry")}):
            self.assertEqual(candidate_ids("gpt-5.4"), ["azure:foundry/gpt-5.4"])

    def test_wrong_api_surface_is_dropped(self):
        providers = {
            "azure:foundry": _StubProvider("azure:foundry", apis=("openai",)),
            "anthropic:main": _StubProvider("anthropic:main", apis=("anthropic",)),
        }
        with _Env(["azure:foundry/gpt-5.4", "anthropic:main/gpt-5.4"],
                  providers=providers):
            self.assertEqual(candidate_ids("gpt-5.4", api="anthropic"),
                             ["anthropic:main/gpt-5.4"])

    def test_api_filter_has_a_non_empty_backstop(self):
        """supports_api_for_model is best-effort (it substring-matches names like
        "speech"), so it must never be the sole reason a pool empties -- a name
        reachable by explicit canonical id must not 404 when written bare."""
        providers = {"azure:foundry": _StubProvider("azure:foundry", apis=("openai",))}
        with _Env(["azure:foundry/llama-3-speech-tuned"], providers=providers):
            self.assertEqual(
                candidate_ids("llama-3-speech-tuned", api="anthropic"),
                ["azure:foundry/llama-3-speech-tuned"],
            )

    def test_api_filter_failure_keeps_the_candidate(self):
        providers = {"azure:foundry": _ExplodingProvider("azure:foundry")}
        with _Env(["azure:foundry/gpt-5.4"], providers=providers):
            self.assertEqual(candidate_ids("gpt-5.4", api="openai"),
                             ["azure:foundry/gpt-5.4"])

    def test_globally_disabled_models_are_dropped_with_no_backstop(self):
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"],
                  disabled_models=["azure:foundry/gpt-5.4"]):
            self.assertEqual(candidate_ids("gpt-5.4"), ["openai:main/gpt-5.4"])

        with _Env(["azure:foundry/gpt-5.4"], disabled_models=["azure:foundry/gpt-5.4"]):
            self.assertEqual(candidate_ids("gpt-5.4"), [])

    def test_disabled_provider_drops_its_models(self):
        with _Env(["azure:foundry/gpt-5.4"], disabled_providers=["azure:foundry"]):
            self.assertEqual(candidate_ids("gpt-5.4"), [])

    def test_candidate_ids_is_empty_for_names_needing_no_resolution(self):
        """It answers "what could this bare name become", not "what serves this"."""
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(candidate_ids("azure:foundry/gpt-5.4"), [])
            self.assertEqual(candidate_ids(None), [])
            self.assertEqual(candidate_ids("unknown-model"), [])

    def test_prefix_strip_can_be_disabled(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(candidate_ids("someprovider/gpt-5.4"),
                             ["azure:foundry/gpt-5.4"])
            self.assertEqual(
                candidate_ids("someprovider/gpt-5.4", allow_prefix_strip=False), [])


class CallerFilterTests(unittest.TestCase):
    def test_denied_provider_is_dropped_for_that_user(self):
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]) as env:
            env.cache._user_model_policies = {7: "custom"}
            env.cache._user_model_exceptions = {7: {"azure:foundry/gpt-5.4": False}}
            self.assertEqual(_resolve(_request(), _User(7), "gpt-5.4"),
                             "openai:main/gpt-5.4")

    def test_admin_bypasses_the_per_user_filter(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(["azure:foundry/gpt-5.4"]) as env:
            env.cache._user_model_policies = {None: "deny"}
            self.assertEqual(_resolve(_request(), admin, "gpt-5.4"),
                             "azure:foundry/gpt-5.4")

    def test_admin_still_cannot_reach_a_globally_disabled_model(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(["azure:foundry/gpt-5.4"], disabled_models=["azure:foundry/gpt-5.4"]):
            with self.assertRaises(ModelUnavailable):
                _resolve(_request(), admin, "gpt-5.4")

    def test_allow_mode_does_not_resurrect_a_globally_disabled_model(self):
        """Documented consequence: a user in "allow" mode can still reach a
        disabled model by explicit canonical id, but never by bare name."""
        with _Env(["azure:foundry/gpt-5.4"],
                  disabled_models=["azure:foundry/gpt-5.4"]) as env:
            env.cache._user_model_policies = {7: "allow"}
            with self.assertRaises(ModelUnavailable):
                _resolve(_request(), _User(7), "gpt-5.4")


# ======================================================================
# Round-robin
# ======================================================================

class RoundRobinTests(unittest.TestCase):
    def setUp(self):
        self.rr = model_resolution._RoundRobin()

    def test_rotates_and_wraps(self):
        pool = ["a/m", "b/m", "c/m"]
        picks = [self.rr.pick(pool) for _ in range(7)]
        self.assertEqual(picks, ["a/m", "b/m", "c/m", "a/m", "b/m", "c/m", "a/m"])

    def test_single_candidate_consumes_no_state(self):
        self.assertEqual(self.rr.pick(["only/m"]), "only/m")
        self.assertEqual(self.rr._cursors, {})

    def test_a_changed_pool_starts_a_fresh_rotation(self):
        self.rr.pick(["a/m", "b/m", "c/m"])
        self.rr.pick(["a/m", "b/m", "c/m"])          # cursor now at 2
        # A different pool is a different key, so it starts at 0...
        self.assertEqual(self.rr.pick(["a/m", "b/m"]), "a/m")
        # ...and the original rotation is untouched.
        self.assertEqual(self.rr.pick(["a/m", "b/m", "c/m"]), "c/m")

    def test_a_stale_cursor_is_never_out_of_range(self):
        key = ("a/m", "b/m")
        self.rr._cursors[key] = 99
        self.assertEqual(self.rr.pick(["a/m", "b/m"]), "b/m")

    def test_cursor_table_is_bounded(self):
        self.rr._MAX_KEYS = 4
        for i in range(10):
            self.rr.pick([f"a{i}/m", f"b{i}/m"])
        self.assertLessEqual(len(self.rr._cursors), 4)

    def test_resolution_alternates_between_two_providers(self):
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]):
            picks = [_resolve(_request(), None, "gpt-5.4") for _ in range(4)]
        self.assertEqual(set(picks), {"azure:foundry/gpt-5.4", "openai:main/gpt-5.4"})
        self.assertNotEqual(picks[0], picks[1])
        self.assertEqual(picks[0], picks[2])


# ======================================================================
# Failure statuses and error envelopes
# ======================================================================

class FailureStatusTests(unittest.TestCase):
    def test_nothing_serves_the_model_is_404(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve(_request(), None, "gpt-9-nonexistent")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.body["error"]["code"], "model_not_found")
        self.assertEqual(ctx.exception.body["error"]["type"], "invalid_request_error")
        self.assertIn("gpt-9-nonexistent", ctx.exception.body["error"]["message"])

    def test_all_candidates_denied_is_403_not_404(self):
        """Otherwise the same user gets "does not exist" for the bare name and
        "not enabled for your account" for the canonical id -- same cause,
        contradictory answers."""
        with _Env(["azure:foundry/gpt-5.4"]) as env:
            env.cache._user_model_policies = {7: "deny"}
            with self.assertRaises(ModelAccessDenied) as ctx:
                _resolve(_request(), _User(7), "gpt-5.4")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.body["error"]["code"], "model_access_denied")

    def test_anthropic_envelope_by_path(self):
        with _Env([]):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve(_request("/v1/messages"), None, "nope")
        body = ctx.exception.body
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "not_found_error")

    def test_anthropic_envelope_by_override(self):
        with _Env([]):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve(_request("/v1/chat/completions"), None, "nope",
                         envelope_override="anthropic")
        self.assertEqual(ctx.exception.body["error"]["type"], "not_found_error")

    def test_azure_surface_uses_the_openai_envelope(self):
        with _Env([]):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve(_request("/openai/v1/chat/completions"), None, "nope")
        self.assertEqual(ctx.exception.body["error"]["code"], "model_not_found")

    def test_denial_envelope_follows_the_surface_too(self):
        with _Env(["azure:foundry/gpt-5.4"]) as env:
            env.cache._user_model_policies = {7: "deny"}
            with self.assertRaises(ModelAccessDenied) as ctx:
                _resolve(_request("/v1/messages"), _User(7), "gpt-5.4")
        self.assertEqual(ctx.exception.body["error"]["type"], "permission_error")


# ======================================================================
# Side effects: response echo, request state, tracker retag
# ======================================================================

class SideEffectTests(unittest.TestCase):
    def test_client_name_is_recorded_for_the_response_echo(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            _, echoed = _resolve_with_echo(_request(), None, "gpt-5.4")
        self.assertEqual(echoed, "gpt-5.4")

    def test_an_alias_already_recorded_keeps_priority(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            original_model_name.set("friendly")
            _, echoed = _resolve_with_echo(_request(), None, "gpt-5.4")
        self.assertEqual(echoed, "friendly")

    def test_pass_through_does_not_touch_the_echo_name(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            _, echoed = _resolve_with_echo(_request(), None, "azure:foundry/gpt-5.4")
        self.assertIsNone(echoed)

    def test_resolved_id_is_written_to_request_state(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            req = _request()
            _resolve(req, None, "gpt-5.4")
            self.assertEqual(req.state.model, "azure:foundry/gpt-5.4")

    def test_a_request_object_without_state_does_not_break_resolution(self):
        """count_tokens hands in a bare object(); resolution must still answer."""
        with _Env(["azure:foundry/gpt-5.4"]):
            self.assertEqual(_resolve(object(), None, "gpt-5.4"),
                             "azure:foundry/gpt-5.4")


# ======================================================================
# The overall-quota bypass in the auth middleware
# ======================================================================

class _StubAuth:
    user_id = 1
    username = "alice"


class OverallQuotaBypassTests(unittest.IsolatedAsyncioTestCase):
    """A prefix-less name must never end up governed by *no* limit.

    _enforce_rate_limit skips the overall RPM/RPD gate when the model or its
    instance is grouped, on the premise that a group limit governs at the route.
    The canonical id is not chosen until the route resolves it, so the bypass has
    to hold for every candidate: if it were decided from one provisional pick and
    the route landed on an ungrouped candidate, check_group_limit would return
    None and nothing would govern the request.
    """

    def _request(self, model):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/chat/completions"),
            state=SimpleNamespace(model=model),
        )
        return request

    async def _overall_limiter_consulted(self, model, grouped_models=(), grouped_instances=()):
        from unittest.mock import AsyncMock, patch
        from app.rate_limit import rate_limit_tracker

        grouped_models = set(grouped_models)
        grouped_instances = set(grouped_instances)
        with patch.object(rate_limit_tracker, "model_belongs_to_group",
                          lambda m: m in grouped_models), \
             patch.object(rate_limit_tracker, "instance_belongs_to_group",
                          lambda p: p in grouped_instances), \
             patch.object(rate_limit_tracker, "check_and_increment",
                          new_callable=AsyncMock) as check:
            check.return_value = SimpleNamespace(allowed=True)
            await _enforce_rate_limit(self._request(model), _StubAuth())
        return check.await_count > 0

    async def test_all_candidates_grouped_skips_the_overall_gate(self):
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]):
            consulted = await self._overall_limiter_consulted(
                "gpt-5.4",
                grouped_models={"azure:foundry/gpt-5.4", "openai:main/gpt-5.4"},
            )
        self.assertFalse(consulted)

    async def test_all_candidates_grouped_by_instance_skips_the_overall_gate(self):
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]):
            consulted = await self._overall_limiter_consulted(
                "gpt-5.4", grouped_instances={"azure:foundry", "openai:main"},
            )
        self.assertFalse(consulted)

    async def test_a_mixed_pool_still_hits_the_overall_gate(self):
        """The quota hole: one grouped candidate must not exempt the request."""
        with _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]):
            consulted = await self._overall_limiter_consulted(
                "gpt-5.4", grouped_models={"azure:foundry/gpt-5.4"},
            )
        self.assertTrue(consulted)

    async def test_an_unresolvable_name_hits_the_overall_gate(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            consulted = await self._overall_limiter_consulted("nothing-serves-this")
        self.assertTrue(consulted)

    async def test_canonical_ids_are_unaffected(self):
        with _Env(["azure:foundry/gpt-5.4"]):
            grouped = await self._overall_limiter_consulted(
                "azure:foundry/gpt-5.4", grouped_models={"azure:foundry/gpt-5.4"},
            )
            ungrouped = await self._overall_limiter_consulted("azure:foundry/gpt-5.4")
        self.assertFalse(grouped)
        self.assertTrue(ungrouped)


if __name__ == "__main__":
    unittest.main()

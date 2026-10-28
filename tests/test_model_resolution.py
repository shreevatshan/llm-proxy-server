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
from unittest.mock import patch

from app import model_resolution
from app.auth.admin import AdminUser
from app.auth.middleware import _enforce_rate_limit
from app.cache import ModelCache
from app.model_access_dep import ModelAccessDenied
from app.model_alias import current_api_surface, original_model_name
from app.model_resolution import (
    PRE_AUTH,
    ModelUnavailable,
    candidate_ids,
    resolve_model_for_request,
)
from app.openai_models import ModelInfo
from app.providers.provider_manager import provider_manager


def _model(model_id: str) -> ModelInfo:
    return ModelInfo(id=model_id, created=0, owned_by="test", provider="test")



def _grant(test, user_id, mode, exceptions=None):
    """Put `user_id` in a profile with this model-access mode, for one test."""
    from app.profiles import ProfileSnapshot, profile_registry
    profile_registry.install(
        {user_id: 9}, [ProfileSnapshot(9, "test", access_mode=mode, access_exceptions=exceptions or {})]
    )
    test.addCleanup(profile_registry.reset)

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

    def test_provider_family_prefix_becomes_the_canonical_id(self):
        """'azure/gpt-4' with one azure instance is that instance's model, so groups,
        access rules and the enable flags all see the id they know."""
        with _Env(["azure:primary/gpt-4"]):
            request = _request()
            self.assertEqual(_resolve(request, None, "azure/gpt-4"), "azure:primary/gpt-4")
            self.assertEqual(request.state.model, "azure:primary/gpt-4")

    def test_ambiguous_provider_family_passes_through(self):
        with _Env(["azure:a/gpt-4", "azure:b/gpt-4"]):
            self.assertEqual(_resolve(_request(), None, "azure/gpt-4"), "azure/gpt-4")

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

    def test_disabled_model_group_drops_its_models(self):
        from app.rate_limit import _ModelGroupSnapshot, rate_limit_tracker
        off = {7: _ModelGroupSnapshot(group_id=7, name="premium", model_ids=["azure:foundry/gpt-5.4"],
                                      enabled=False)}
        with patch.object(rate_limit_tracker, "_groups", off), \
                patch.object(rate_limit_tracker, "_model_to_group", {"azure:foundry/gpt-5.4": 7}), \
                _Env(["azure:foundry/gpt-5.4", "openai:main/gpt-5.4"]):
            self.assertEqual(candidate_ids("gpt-5.4"), ["openai:main/gpt-5.4"])

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
            _grant(self, 7, "custom", {"azure:foundry/gpt-5.4": False})
            self.assertEqual(_resolve(_request(), _User(7), "gpt-5.4"),
                             "openai:main/gpt-5.4")

    def test_admin_bypasses_the_per_user_filter(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(["azure:foundry/gpt-5.4"]) as env:
            _grant(self, 0, "deny")
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
            _grant(self, 7, "allow")
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
            _grant(self, 7, "deny")
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
            _grant(self, 7, "deny")
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
# Regular vs fallback mappings: the client's name picks the list
# ======================================================================

class _Rules:
    """Install one profile's (Default's unless given) regular and fallback rules
    ({alias: target}) for one test. _grant puts its user in profile 9."""

    def __init__(self, regular=None, fallback=None, apis=frozenset({"openai", "anthropic", "azure_openai"}),
                 profile_id=1):
        self.lists = {"alias": regular or {}, "fallback": fallback or {}}
        self.apis = apis
        self.profile_id = profile_id

    def __enter__(self):
        from app.model_alias import build_rule, model_alias_resolver as r
        self.saved = (r._snapshot, r._profile_snapshots, r._fallback_snapshot, r._fallback_profile_snapshots)
        for kind, mapping in self.lists.items():
            r.set_rules([build_rule(a, t, self.apis, rule_id=i) for i, (a, t) in enumerate(mapping.items(), 1)],
                        profile_id=self.profile_id, kind=kind)
        return self

    def __exit__(self, *exc):
        from app.model_alias import model_alias_resolver as r
        (r._snapshot, r._profile_snapshots, r._fallback_snapshot, r._fallback_profile_snapshots) = self.saved


def _fresh_alias_context():
    """Inside asyncio.run: start from a request with nothing mapped, as the middleware does."""
    from app.model_alias import reset_alias_context
    reset_alias_context()
    original_model_name.set(None)


def _resolve_fresh(request_obj, auth, model, **kw):
    async def run():
        _fresh_alias_context()
        chosen = await resolve_model_for_request(request_obj, auth, model, **kw)
        return chosen, original_model_name.get()
    return asyncio.run(run())


MODELS = ["p:a/gpt-5.4", "p:a/gpt-5.5"]


class FallbackMappingTests(unittest.TestCase):

    def test_available_name_uses_the_regular_list(self):
        with _Env(MODELS), _Rules(regular={"gpt-5.4": "p:a/gpt-5.5"}, fallback={"gpt-5.4": "nope"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-5.4"), ("p:a/gpt-5.5", "gpt-5.4"))

    def test_fallback_list_is_ignored_for_an_available_name(self):
        with _Env(MODELS), _Rules(fallback={"gpt-5.4": "p:a/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-5.4")[0], "p:a/gpt-5.4")

    def test_unknown_name_uses_the_fallback_list(self):
        with _Env(MODELS), _Rules(fallback={"gpt-9": "p:a/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-9"), ("p:a/gpt-5.5", "gpt-9"))

    def test_regular_list_never_applies_to_an_unknown_name(self):
        with _Env(MODELS), _Rules(regular={"gpt-9": "p:a/gpt-5.5"}):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve_fresh(_request(), _User(7), "gpt-9")
        self.assertIn("'gpt-9'", ctx.exception.body["error"]["message"])

    def test_pattern_fallback_rules_work(self):
        from app.model_alias import build_rule, model_alias_resolver as r
        with _Env(MODELS), _Rules():
            r.set_rules([build_rule(".", "p:a/gpt-5.5", match_type="regex", rule_id=1)], kind="fallback")
            self.assertEqual(_resolve_fresh(_request(), _User(7), "anything-at-all")[0], "p:a/gpt-5.5")

    def test_globally_disabled_model_uses_the_fallback_list(self):
        with _Env(MODELS, disabled_models=["p:a/gpt-5.4"]), \
             _Rules(fallback={"gpt-5.4": "p:a/gpt-5.5", "p:a/gpt-5.4": "p:a/gpt-5.5"}):
            for name in ("gpt-5.4", "p:a/gpt-5.4"):
                with self.subTest(name=name):
                    self.assertEqual(_resolve_fresh(_request(), _User(7), name), ("p:a/gpt-5.5", name))

    def test_disabled_provider_uses_the_fallback_list(self):
        with _Env(["p:a/gpt-5.4", "q:b/gpt-5.5"], disabled_providers=["p:a"]), \
             _Rules(fallback={"gpt-5.4": "q:b/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-5.4")[0], "q:b/gpt-5.5")

    def test_model_the_profile_denies_uses_the_fallback_list(self):
        with _Env(MODELS), _Rules(fallback={"gpt-5.4": "p:a/gpt-5.5", "p:a/gpt-5.4": "p:a/gpt-5.5"},
                                  profile_id=9):
            _grant(self, 7, "custom", {"p:a/gpt-5.4": False})
            self.assertEqual(_resolve_fresh(_request(), _User(7), "gpt-5.4")[0], "p:a/gpt-5.5")
            self.assertEqual(_resolve_fresh(_request(), _User(7), "p:a/gpt-5.4")[0], "p:a/gpt-5.5")

    def test_denied_fallback_target_keeps_the_original_403(self):
        with _Env(MODELS), _Rules(fallback={"gpt-5.4": "p:a/gpt-5.5"}, profile_id=9):
            _grant(self, 7, "deny")
            with self.assertRaises(ModelAccessDenied) as ctx:
                _resolve_fresh(_request(), _User(7), "gpt-5.4")
        self.assertIn("'gpt-5.4'", ctx.exception.body["error"]["message"])

    def test_unavailable_fallback_target_keeps_the_original_404(self):
        # (A target under a live provider prefix would pass through as an uncached
        # deployment, so this one names no provider.)
        with _Env(MODELS), _Rules(fallback={"gpt-9": "gone:x/gone"}):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve_fresh(_request(), _User(7), "gpt-9")
        self.assertIn("'gpt-9'", ctx.exception.body["error"]["message"])

    def test_unavailable_regular_target_is_reported_not_rescued(self):
        with _Env(MODELS), _Rules(regular={"gpt-5.4": "gone:x/gone"}, fallback={"gpt-5.4": "p:a/gpt-5.5"}):
            with self.assertRaises(ModelUnavailable) as ctx:
                _resolve_fresh(_request(), _User(7), "gpt-5.4")
        self.assertIn("'gone:x/gone'", ctx.exception.body["error"]["message"])

    def test_admin_may_call_a_disabled_model_by_id_so_no_fallback(self):
        admin = AdminUser("root", "root@example.com")
        with _Env(MODELS, disabled_models=["p:a/gpt-5.4"]), \
             _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}):
            self.assertEqual(_resolve_fresh(_request(), admin, "p:a/gpt-5.4")[0], "p:a/gpt-5.4")

    def test_fallback_rules_respect_their_api_surfaces(self):
        with _Env(MODELS, api="openai"), _Rules(fallback={"gpt-9": "p:a/gpt-5.5"}, apis={"anthropic"}):
            with self.assertRaises(ModelUnavailable):
                _resolve_fresh(_request(), _User(7), "gpt-9")

    def test_denied_deployment_behind_a_provider_prefix_uses_the_fallback_list(self):
        providers = {"azure:primary": _StubProvider("azure:primary"), "p:a": _StubProvider("p:a")}
        with _Env(MODELS, providers=providers), \
             _Rules(fallback={"azure:primary/dep": "p:a/gpt-5.5"}, profile_id=9):
            _grant(self, 7, "custom", {"azure:primary/dep": False})
            self.assertEqual(_resolve_fresh(_request(), _User(7), "azure:primary/dep")[0], "p:a/gpt-5.5")

    def test_caller_access_overrides_the_pre_auth_choice_and_retags(self):
        """Before auth a disabled model looks unavailable and the fallback is applied;
        an allow-mode caller may use it, so it is served after all and usage follows."""
        from unittest.mock import AsyncMock
        from app.model_resolution import map_model_name
        request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"),
                                  state=SimpleNamespace(tracking_request_id="r1"))

        async def run():
            _fresh_alias_context()
            request.state.model = map_model_name("p:a/gpt-5.4")      # tracking middleware
            pre_auth = request.state.model
            chosen = await resolve_model_for_request(request, _User(7), request.state.model)
            return pre_auth, chosen

        with _Env(MODELS, disabled_models=["p:a/gpt-5.4"]), \
             _Rules(fallback={"p:a/gpt-5.4": "p:a/gpt-5.5"}), \
             _Rules(profile_id=9), \
             patch("app.request_tracker.request_tracker.update_model", new_callable=AsyncMock) as upd:
            _grant(self, 7, "allow")
            pre_auth, chosen = asyncio.run(run())
        self.assertEqual(pre_auth, "p:a/gpt-5.5")
        self.assertEqual(chosen, "p:a/gpt-5.4")
        self.assertEqual(request.state.model, "p:a/gpt-5.4")
        upd.assert_awaited_once_with("r1", "p:a/gpt-5.4")

    def test_map_model_name_picks_the_list_without_raising(self):
        from app.model_resolution import map_model_name

        async def run(name):
            _fresh_alias_context()
            return map_model_name(name, _User(7))

        with _Env(MODELS), _Rules(regular={"gpt-5.4": "p:a/gpt-5.5"}, fallback={"gpt-9": "p:a/gpt-5.4"}):
            self.assertEqual(asyncio.run(run("gpt-5.4")), "p:a/gpt-5.5")
            self.assertEqual(asyncio.run(run("gpt-9")), "p:a/gpt-5.4")
            self.assertEqual(asyncio.run(run("gpt-10")), "gpt-10")

    def test_map_model_name_keeps_the_clients_name_when_the_fallback_target_fails_too(self):
        """As resolve_model_for_request does, so the Azure deployment routes report the
        failure against the name the client sent, not a target it never asked for."""
        from app.model_resolution import map_model_name

        async def run(name, auth):
            _fresh_alias_context()
            return map_model_name(name, auth), original_model_name.get()

        with _Env(MODELS, disabled_models=["p:a/gpt-5.5"]), \
             _Rules(fallback={"gpt-9": "p:a/gpt-5.5", "gpt-10": "gone:x/gone"}):
            self.assertEqual(asyncio.run(run("gpt-9", _User(7))), ("gpt-9", None))
            self.assertEqual(asyncio.run(run("gpt-10", _User(7))), ("gpt-10", None))
            # Before auth the choice is provisional: the route re-judges it.
            self.assertEqual(asyncio.run(run("gpt-9", PRE_AUTH)), ("p:a/gpt-5.5", "gpt-9"))


# ======================================================================
# Who charges a request: the route (marked endpoints) or the auth middleware
# ======================================================================

class _StubAuth:
    user_id = 1
    username = "alice"


class DefaultGroupChargeTests(unittest.IsolatedAsyncioTestCase):
    """A request is charged exactly once.

    A route that names a model charges it after resolution and the access check
    (enforce_access_and_limits) and is marked @enforces_limits; the middleware
    leaves marked endpoints alone and charges everything else to the Default group.
    The decision is by endpoint, never by request.state.model, which the tracking
    middleware does not set on /openai/v1, large uploads or count_tokens.
    """

    def _request(self, endpoint, model=None):
        return SimpleNamespace(
            url=SimpleNamespace(path="/v1/chat/completions"),
            state=SimpleNamespace(model=model),
            scope={"endpoint": endpoint},
        )

    async def _middleware_charges(self, endpoint, model=None):
        from unittest.mock import AsyncMock, patch
        from app.rate_limit import rate_limit_tracker

        with patch.object(rate_limit_tracker, "check_and_increment", new_callable=AsyncMock) as check:
            check.return_value = SimpleNamespace(allowed=True)
            await _enforce_rate_limit(self._request(endpoint, model), _StubAuth())
        return check.await_count

    async def test_marked_endpoint_is_left_to_the_route(self):
        from app.routes import chat
        self.assertEqual(await self._middleware_charges(chat.chat_completions, "gpt-5.4"), 0)
        self.assertEqual(await self._middleware_charges(chat.chat_completions, None), 0,
                         "no sniffed model (/openai/v1, large upload) is still the route's to charge")

    async def test_unmarked_endpoint_is_charged_to_default(self):
        async def cancel():
            pass
        self.assertEqual(await self._middleware_charges(cancel), 1)
        self.assertEqual(await self._middleware_charges(None), 1)

    async def test_every_model_route_is_marked(self):
        from fastapi.routing import APIRoute
        from app.main import create_anthropic_app, create_azure_openai_app, create_openai_app

        unmarked = []
        for make in (create_openai_app, create_anthropic_app, create_azure_openai_app):
            for route in make().routes:
                if not isinstance(route, APIRoute) or "POST" not in route.methods:
                    continue
                if route.path.endswith("/cancel"):
                    continue  # names no model: the middleware charges it to Default
                if getattr(route.endpoint, "_route_enforces_limits", False) is not True:
                    unmarked.append(route.path)
        self.assertEqual(unmarked, [])


class EnforceAccessAndLimitsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from app.rate_limit import rate_limit_tracker
        self.tracker = rate_limit_tracker
        self._saved = (dict(self.tracker._model_to_group), dict(self.tracker._provider_to_group))
        self.tracker._model_to_group = {"azure:foundry/gpt-5.4": 7}
        self.tracker._provider_to_group = {}

    def tearDown(self):
        self.tracker._model_to_group, self.tracker._provider_to_group = self._saved

    async def _run(self, model, *, allowed=True, consume_quota=True):
        from unittest.mock import AsyncMock, patch
        from app import rate_limit_dep
        from app.model_access_dep import ModelAccessDenied

        async def access(*_a, **_k):
            if not allowed:
                raise ModelAccessDenied.openai(model)

        with patch.object(rate_limit_dep, "enforce_model_access", access), \
             patch.object(self.tracker, "check_and_increment", new_callable=AsyncMock) as default, \
             patch.object(self.tracker, "check_group_limit", new_callable=AsyncMock) as group:
            default.return_value = SimpleNamespace(allowed=True)
            group.return_value = None
            request = SimpleNamespace(url=SimpleNamespace(path="/v1/chat/completions"))
            try:
                await rate_limit_dep.enforce_access_and_limits(
                    request, _StubAuth(), model, consume_quota=consume_quota)
            except ModelAccessDenied:
                pass
        return default.await_count, group.await_count

    async def test_grouped_model_charges_its_group_only(self):
        self.assertEqual(await self._run("azure:foundry/gpt-5.4"), (0, 1))

    async def test_ungrouped_model_charges_default_only(self):
        self.assertEqual(await self._run("openai:main/gpt-5.4"), (1, 0))

    async def test_a_denied_request_spends_no_quota(self):
        self.assertEqual(await self._run("azure:foundry/gpt-5.4", allowed=False), (0, 0))
        self.assertEqual(await self._run("openai:main/gpt-5.4", allowed=False), (0, 0))

    async def test_access_only_spends_no_quota(self):
        self.assertEqual(await self._run("openai:main/gpt-5.4", consume_quota=False), (0, 0))


if __name__ == "__main__":
    unittest.main()

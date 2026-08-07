from types import SimpleNamespace

from app.model_alias import (
    apply_alias,
    current_api_surface,
    echo_model_name,
    model_alias_resolver,
    original_model_name,
    _parse_apis,
    _ALL_SURFACES,
)


def test_resolver_is_single_level_and_unknown_names_pass_through():
    previous = model_alias_resolver._aliases
    try:
        model_alias_resolver._aliases = {
            "friendly": ("provider/model", _ALL_SURFACES),
            "provider/model": ("another/target", _ALL_SURFACES),
        }
        assert model_alias_resolver.resolve("friendly") == "provider/model"
        assert model_alias_resolver.resolve("unknown") == "unknown"
        assert model_alias_resolver.resolve(None) is None
    finally:
        model_alias_resolver._aliases = previous


def test_apply_alias_retains_client_model_for_response_echo():
    previous_aliases = model_alias_resolver._aliases
    token = original_model_name.set(None)
    surface_token = current_api_surface.set(None)
    try:
        model_alias_resolver._aliases = {"friendly": ("provider/model", _ALL_SURFACES)}
        assert apply_alias("friendly") == "provider/model"
        assert echo_model_name(SimpleNamespace(model="provider/model")) == "friendly"
        assert apply_alias("unknown") == "unknown"
    finally:
        model_alias_resolver._aliases = previous_aliases
        original_model_name.reset(token)
        current_api_surface.reset(surface_token)


def test_resolve_scoped_to_selected_surface():
    previous = model_alias_resolver._aliases
    try:
        model_alias_resolver._aliases = {
            "friendly": ("provider/model", frozenset({"openai"})),
        }
        # Applies on the selected surface.
        assert model_alias_resolver.resolve("friendly", "openai") == "provider/model"
        # Passes through unchanged on an unselected surface.
        assert model_alias_resolver.resolve("friendly", "anthropic") == "friendly"
        assert model_alias_resolver.resolve("friendly", "azure_openai") == "friendly"
    finally:
        model_alias_resolver._aliases = previous


def test_resolve_unscoped_caller_always_applies():
    previous = model_alias_resolver._aliases
    try:
        model_alias_resolver._aliases = {
            "friendly": ("provider/model", frozenset({"openai"})),
        }
        # api=None means "unscoped" — apply the mapping regardless of its scope.
        assert model_alias_resolver.resolve("friendly", None) == "provider/model"
        assert model_alias_resolver.resolve("friendly") == "provider/model"
    finally:
        model_alias_resolver._aliases = previous


def test_parse_apis_legacy_values_mean_all_surfaces():
    assert _parse_apis(None) == _ALL_SURFACES
    assert _parse_apis("") == _ALL_SURFACES
    assert _parse_apis("[]") == _ALL_SURFACES
    assert _parse_apis("not json") == _ALL_SURFACES
    assert _parse_apis('{"a": 1}') == _ALL_SURFACES
    assert _parse_apis('["openai"]') == frozenset({"openai"})
    assert _parse_apis('["openai", "bogus"]') == frozenset({"openai"})


def test_apply_alias_reads_current_api_surface():
    previous_aliases = model_alias_resolver._aliases
    token = original_model_name.set(None)
    surface_token = current_api_surface.set("anthropic")
    try:
        model_alias_resolver._aliases = {
            "friendly": ("provider/model", frozenset({"openai"})),
        }
        # Surface excluded: pass through, original_model_name untouched.
        assert apply_alias("friendly") == "friendly"
        assert original_model_name.get() is None
    finally:
        model_alias_resolver._aliases = previous_aliases
        original_model_name.reset(token)
        current_api_surface.reset(surface_token)


def test_apply_alias_applies_on_included_surface():
    previous_aliases = model_alias_resolver._aliases
    token = original_model_name.set(None)
    surface_token = current_api_surface.set("openai")
    try:
        model_alias_resolver._aliases = {
            "friendly": ("provider/model", frozenset({"openai"})),
        }
        assert apply_alias("friendly") == "provider/model"
        assert original_model_name.get() == "friendly"
    finally:
        model_alias_resolver._aliases = previous_aliases
        original_model_name.reset(token)
        current_api_surface.reset(surface_token)

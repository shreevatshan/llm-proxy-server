import asyncio
from types import SimpleNamespace

import pytest

from app.model_alias import (
    alias_resolved_name,
    apply_alias,
    build_rule,
    current_api_surface,
    echo_model_name,
    model_alias_resolver,
    original_model_name,
    rules_from_mapping,
    _parse_apis,
    _ALL_SURFACES,
)


@pytest.fixture(autouse=True)
def _isolated_resolver():
    """Restore the live snapshot and per-request context after every test."""
    previous = model_alias_resolver._snapshot
    tokens = [
        (original_model_name, original_model_name.set(None)),
        (current_api_surface, current_api_surface.set(None)),
        (alias_resolved_name, alias_resolved_name.set(None)),
    ]
    try:
        yield
    finally:
        model_alias_resolver._snapshot = previous
        for var, token in reversed(tokens):
            var.reset(token)


def test_resolver_is_single_level_and_unknown_names_pass_through():
    model_alias_resolver.set_rules(rules_from_mapping({
        "friendly": ("provider/model", _ALL_SURFACES),
        "provider/model": ("another/target", _ALL_SURFACES),
    }))
    assert model_alias_resolver.resolve("friendly") == "provider/model"
    assert model_alias_resolver.resolve("unknown") == "unknown"
    assert model_alias_resolver.resolve(None) is None


def test_apply_alias_retains_client_model_for_response_echo():
    model_alias_resolver.set_rules(rules_from_mapping({"friendly": ("provider/model", _ALL_SURFACES)}))
    assert apply_alias("friendly") == "provider/model"
    assert echo_model_name(SimpleNamespace(model="provider/model")) == "friendly"
    assert apply_alias("unknown") == "unknown"


def test_resolve_scoped_to_selected_surface():
    model_alias_resolver.set_rules(rules_from_mapping({
        "friendly": ("provider/model", frozenset({"openai"})),
    }))
    # Applies on the selected surface.
    assert model_alias_resolver.resolve("friendly", "openai") == "provider/model"
    # Passes through unchanged on an unselected surface.
    assert model_alias_resolver.resolve("friendly", "anthropic") == "friendly"
    assert model_alias_resolver.resolve("friendly", "azure_openai") == "friendly"


def test_resolve_unscoped_caller_always_applies():
    model_alias_resolver.set_rules(rules_from_mapping({
        "friendly": ("provider/model", frozenset({"openai"})),
    }))
    # api=None means "unscoped" — apply the mapping regardless of its scope.
    assert model_alias_resolver.resolve("friendly", None) == "provider/model"
    assert model_alias_resolver.resolve("friendly") == "provider/model"


def test_parse_apis_legacy_values_mean_all_surfaces():
    assert _parse_apis(None) == _ALL_SURFACES
    assert _parse_apis("") == _ALL_SURFACES
    assert _parse_apis("[]") == _ALL_SURFACES
    assert _parse_apis("not json") == _ALL_SURFACES
    assert _parse_apis('{"a": 1}') == _ALL_SURFACES
    assert _parse_apis('["openai"]') == frozenset({"openai"})
    assert _parse_apis('["openai", "bogus"]') == frozenset({"openai"})


def test_apply_alias_reads_current_api_surface():
    current_api_surface.set("anthropic")
    model_alias_resolver.set_rules(rules_from_mapping({
        "friendly": ("provider/model", frozenset({"openai"})),
    }))
    # Surface excluded: pass through, original_model_name untouched.
    assert apply_alias("friendly") == "friendly"
    assert original_model_name.get() is None


def test_apply_alias_applies_on_included_surface():
    current_api_surface.set("openai")
    model_alias_resolver.set_rules(rules_from_mapping({
        "friendly": ("provider/model", frozenset({"openai"})),
    }))
    assert apply_alias("friendly") == "provider/model"
    assert original_model_name.get() == "friendly"


# --- pattern rules -----------------------------------------------------------

def test_exact_is_case_sensitive():
    model_alias_resolver.set_rules([build_rule("friendly", "p/x")])
    assert model_alias_resolver.resolve("friendly") == "p/x"
    assert model_alias_resolver.resolve("Friendly") == "Friendly"


def test_contains_is_case_insensitive_and_matches_qualified_names():
    model_alias_resolver.set_rules([build_rule("opus", "p/opus-target", match_type="contains")])
    assert model_alias_resolver.resolve("claude-opus-4-1") == "p/opus-target"
    assert model_alias_resolver.resolve("Claude-OPUS-4") == "p/opus-target"
    assert model_alias_resolver.resolve("anthropic/claude-opus-4-1") == "p/opus-target"
    assert model_alias_resolver.resolve("claude-sonnet-4") == "claude-sonnet-4"


def test_regex_uses_search_and_ignores_case():
    model_alias_resolver.set_rules([build_rule(r"sonnet-4(\.\d)?$", "p/s", match_type="regex")])
    assert model_alias_resolver.resolve("claude-SONNET-4.5") == "p/s"
    assert model_alias_resolver.resolve("claude-sonnet-4") == "p/s"
    assert model_alias_resolver.resolve("claude-sonnet-4-latest") == "claude-sonnet-4-latest"


def test_anchored_regex_requires_full_match():
    model_alias_resolver.set_rules([build_rule(r"^gpt-4o$", "p/g", match_type="regex")])
    assert model_alias_resolver.resolve("gpt-4o") == "p/g"
    assert model_alias_resolver.resolve("gpt-4o-mini") == "gpt-4o-mini"


def test_priority_contains_above_exact_wins():
    model_alias_resolver.set_rules([
        build_rule("opus", "p/broad", match_type="contains", priority=0, rule_id=1),
        build_rule("claude-opus-4", "p/exact", priority=1, rule_id=2),
    ])
    assert model_alias_resolver.resolve("claude-opus-4") == "p/broad"


def test_priority_exact_above_contains_wins():
    model_alias_resolver.set_rules([
        build_rule("opus", "p/broad", match_type="contains", priority=1, rule_id=1),
        build_rule("claude-opus-4", "p/exact", priority=0, rule_id=2),
    ])
    assert model_alias_resolver.resolve("claude-opus-4") == "p/exact"
    assert model_alias_resolver.resolve("claude-opus-3") == "p/broad"


def test_priority_ties_break_on_id():
    model_alias_resolver.set_rules([
        build_rule("opus", "p/second", match_type="contains", priority=0, rule_id=7),
        build_rule("opus", "p/first", match_type="regex", priority=0, rule_id=3),
    ])
    assert model_alias_resolver.resolve("opus") == "p/first"


def test_rule_scoped_to_other_surface_does_not_block_later_rule():
    model_alias_resolver.set_rules([
        build_rule("claude-opus-4", "p/openai-only", apis={"openai"}, priority=0, rule_id=1),
        build_rule("opus", "p/broad", match_type="contains", priority=1, rule_id=2),
    ])
    assert model_alias_resolver.resolve("claude-opus-4", "openai") == "p/openai-only"
    assert model_alias_resolver.resolve("claude-opus-4", "anthropic") == "p/broad"


def test_name_equal_to_target_is_a_noop():
    current_api_surface.set("openai")
    model_alias_resolver.set_rules([build_rule("opus", "p/claude-opus-4", match_type="contains")])
    assert apply_alias("p/claude-opus-4") == "p/claude-opus-4"
    assert original_model_name.get() is None
    assert alias_resolved_name.get() is None


def test_set_rules_invalidates_cache():
    model_alias_resolver.set_rules([build_rule("opus", "p/a", match_type="contains")])
    assert model_alias_resolver.resolve("claude-opus") == "p/a"
    model_alias_resolver.set_rules([build_rule("opus", "p/b", match_type="contains")])
    assert model_alias_resolver.resolve("claude-opus") == "p/b"
    model_alias_resolver.set_rules([])
    assert model_alias_resolver.resolve("claude-opus") == "claude-opus"


def test_long_names_skip_patterns_but_not_exact_rules():
    long_name = "opus-" + "x" * 300
    model_alias_resolver.set_rules([
        build_rule("opus", "p/broad", match_type="contains", priority=0),
        build_rule(".*", "p/any", match_type="regex", priority=1),
    ])
    assert model_alias_resolver.resolve(long_name) == long_name

    exact_long = "e" * 300
    model_alias_resolver.set_rules([
        build_rule("e", "p/broad", match_type="contains", priority=0),
        build_rule(exact_long, "p/exact", priority=1),
    ])
    assert model_alias_resolver.resolve(exact_long) == "p/exact"


def test_long_names_are_not_cached():
    model_alias_resolver.set_rules([build_rule("opus", "p/a", match_type="contains")])
    snapshot = model_alias_resolver._snapshot
    model_alias_resolver.resolve("x" * 10_000, "openai")
    assert snapshot._cached_match.cache_info().currsize == 0
    model_alias_resolver.resolve("claude-opus", "openai")
    assert snapshot._cached_match.cache_info().currsize == 1


def test_build_rule_rejects_bad_input():
    with pytest.raises(ValueError):
        build_rule("x", "p/x", match_type="glob")
    with pytest.raises(ValueError):
        build_rule("(", "p/x", match_type="regex")
    # Rows saved before the backtracking check existed are refused at load too.
    with pytest.raises(ValueError):
        build_rule("(a|aa)+$", "p/x", match_type="regex")


def test_overlapping_reloads_install_the_latest_rows(monkeypatch):
    """A reload whose SELECT predates a later commit must not install last."""
    import app.model_alias as model_alias_module

    db_rows = [SimpleNamespace(id=1, alias="opus", target_model_id="p/old", enabled=True,
                               apis=None, match_type="contains", priority=0)]
    first_select_started = asyncio.Event()
    release_first_select = asyncio.Event()

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def _fake_get_all(db):
        snapshot = list(db_rows)
        if not first_select_started.is_set():
            first_select_started.set()
            await release_first_select.wait()
        return snapshot

    monkeypatch.setattr(model_alias_module, "AsyncSessionLocal", lambda: _Session())
    monkeypatch.setattr(model_alias_module, "get_all_model_aliases", _fake_get_all)

    async def scenario():
        first = asyncio.create_task(model_alias_resolver.load_from_database())
        await first_select_started.wait()
        db_rows[0] = SimpleNamespace(**{**vars(db_rows[0]), "target_model_id": "p/new"})
        second = asyncio.create_task(model_alias_resolver.load_from_database())
        await asyncio.sleep(0)
        release_first_select.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())
    assert model_alias_resolver.resolve("claude-opus", "openai") == "p/new"


def test_apply_alias_does_not_chain():
    current_api_surface.set("openai")
    model_alias_resolver.set_rules([
        build_rule("mini", "p/gpt-4o", match_type="contains", priority=0, rule_id=1),
        build_rule("gpt-4o", "p/gpt-4.1", match_type="contains", priority=1, rule_id=2),
    ])
    first = apply_alias("gpt-4o-mini")
    assert first == "p/gpt-4o"
    # A later layer feeding the mapped name back in must not re-map it.
    assert apply_alias(first) == "p/gpt-4o"
    assert original_model_name.get() == "gpt-4o-mini"
    assert echo_model_name(SimpleNamespace(model="p/gpt-4o")) == "gpt-4o-mini"
    # The client's original name re-resolves to the same answer.
    assert apply_alias("gpt-4o-mini") == "p/gpt-4o"


def test_load_from_database_skips_disabled_and_invalid_rows(monkeypatch):
    import app.model_alias as model_alias_module

    rows = [
        SimpleNamespace(id=1, alias="(", target_model_id="p/bad", enabled=True, apis=None, match_type="regex", priority=0),
        SimpleNamespace(id=2, alias="x", target_model_id="p/glob", enabled=True, apis=None, match_type="glob", priority=1),
        SimpleNamespace(id=3, alias="opus", target_model_id="p/off", enabled=False, apis=None, match_type="contains", priority=2),
        SimpleNamespace(id=4, alias="OPUS", target_model_id="p/on", enabled=True, apis='["anthropic"]', match_type="contains", priority=3),
    ]

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def _fake_get_all(db):
        return rows

    monkeypatch.setattr(model_alias_module, "AsyncSessionLocal", lambda: _Session())
    monkeypatch.setattr(model_alias_module, "get_all_model_aliases", _fake_get_all)

    asyncio.run(model_alias_resolver.load_from_database())
    assert [r.id for r in model_alias_resolver._snapshot.rules] == [4]
    assert model_alias_resolver.resolve("claude-opus-4", "anthropic") == "p/on"
    assert model_alias_resolver.resolve("claude-opus-4", "openai") == "claude-opus-4"

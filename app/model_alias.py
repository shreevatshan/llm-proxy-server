"""Fast, request-safe resolution of admin-managed model aliases.

Each alias is a rule: an exact name, a case-insensitive substring, or a
case-insensitive regex (re.search), evaluated in admin-defined priority order.
The first enabled rule that matches and is scoped to the request's API surface
wins. Results are memoised per immutable rules snapshot, so after the first
request for a name the lookup is a dict hit regardless of the rule types.
"""

import asyncio
import functools
import json
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from app.auth.database import AsyncSessionLocal, get_all_model_aliases
from app.auth.models import MODEL_ALIAS_API_SURFACES, validate_alias_pattern

logger = logging.getLogger(__name__)


original_model_name: ContextVar[Optional[str]] = ContextVar(
    "original_model_name", default=None
)

# The API surface (openai / anthropic / azure_openai) the current request
# arrived on. Set once per request by the tracking middleware. None means
# "unscoped" — callers without request context apply every mapping (fail-open).
current_api_surface: ContextVar[Optional[str]] = ContextVar(
    "current_api_surface", default=None
)

# The name apply_alias() last rewrote to in this request. Several layers call
# apply_alias (middleware, handlers, resolve_model_for_request); feeding an
# already-mapped name back in returns it untouched, so pattern rules can never
# chain (rule A's target matching rule B) and the client's name stays echoed.
alias_resolved_name: ContextVar[Optional[str]] = ContextVar(
    "alias_resolved_name", default=None
)

_ALL_SURFACES = frozenset(MODEL_ALIAS_API_SURFACES)

# Names longer than this skip contains/regex rules: client-controlled input must
# not be able to drive arbitrarily expensive regex scans. Aliases are at most
# 200 characters, so exact rules are unaffected.
_MAX_PATTERN_NAME_LEN = 256


def _parse_apis(raw) -> frozenset[str]:
    """Decode the stored JSON array; NULL/empty/malformed ⇒ every surface."""
    if not raw:
        return _ALL_SURFACES
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        return _ALL_SURFACES
    if not isinstance(decoded, list):
        return _ALL_SURFACES
    surfaces = frozenset(decoded) & _ALL_SURFACES
    return surfaces or _ALL_SURFACES


@dataclass(frozen=True)
class AliasRule:
    id: Optional[int]
    pattern: str
    target: str
    apis: frozenset
    match_type: str
    priority: int
    # (name, casefolded name) -> bool
    matches: Callable[[str, str], bool]


def build_rule(
    pattern: str,
    target: str,
    apis: Iterable[str] = _ALL_SURFACES,
    match_type: str = "exact",
    priority: int = 0,
    rule_id: Optional[int] = None,
) -> AliasRule:
    """Build a rule; raises ValueError for anything validate_alias_pattern refuses.

    Rows are validated on save, but this re-checks them so a row written before a
    rule was tightened (or edited by hand) can never reach the request path.
    """
    validate_alias_pattern(pattern, target, match_type)
    if match_type == "exact":
        matches = lambda name, folded: name == pattern
    elif match_type == "contains":
        needle = pattern.casefold()
        matches = lambda name, folded: needle in folded
    else:
        search = re.compile(pattern, re.IGNORECASE).search
        matches = lambda name, folded: search(name) is not None
    return AliasRule(rule_id, pattern, target, frozenset(apis), match_type, priority, matches)


def rules_from_mapping(mapping: dict) -> list[AliasRule]:
    """Exact rules from {alias: (target, apis)}, in insertion order (tests/helpers)."""
    return [
        build_rule(alias, target, apis, priority=index)
        for index, (alias, (target, apis)) in enumerate(mapping.items())
    ]


class _RuleSnapshot:
    """Immutable, priority-ordered rules with a per-snapshot match cache."""

    def __init__(self, rules: Iterable[AliasRule] = ()) -> None:
        self.rules = tuple(sorted(rules, key=lambda r: (r.priority, r.id or 0)))
        self._cached_match = functools.lru_cache(maxsize=4096)(self._match)

    def match(self, name: str, api: Optional[str]) -> Optional[AliasRule]:
        # Long names only meet exact rules (a cheap scan), and are not cached:
        # names arrive before auth, so caching them would let any client pin up
        # to 4096 request-sized strings in memory.
        if len(name) > _MAX_PATTERN_NAME_LEN:
            return self._match(name, api)
        return self._cached_match(name, api)

    def _match(self, name: str, api: Optional[str]) -> Optional[AliasRule]:
        exact_only = len(name) > _MAX_PATTERN_NAME_LEN
        folded = "" if exact_only else name.casefold()
        for rule in self.rules:
            if exact_only and rule.match_type != "exact":
                continue
            if api is not None and api not in rule.apis:
                continue          # this surface is unaffected; try the next rule
            if rule.matches(name, folded):
                return rule
        return None


class ModelAliasResolver:
    """Immutable-snapshot alias resolver; reads never touch the database."""

    def __init__(self) -> None:
        self._snapshot = _RuleSnapshot()
        # Serialises reloads. Each admin edit commits and then reloads; without
        # this, an earlier reload whose SELECT predates a later commit could
        # install its snapshot last and leave stale rules live.
        self._reload_lock = asyncio.Lock()

    async def load_from_database(self) -> None:
        async with self._reload_lock:
            async with AsyncSessionLocal() as db:
                aliases = await get_all_model_aliases(db)
            rules = []
            for row in aliases:
                if not row.enabled:
                    continue
                try:
                    rules.append(build_rule(
                        row.alias, row.target_model_id, _parse_apis(row.apis),
                        row.match_type or "exact", row.priority or 0, row.id,
                    ))
                except ValueError as e:
                    # Never fail startup over one bad row; it simply never matches.
                    logger.warning("Skipping model alias %r: %s", row.alias, e)
            self.set_rules(rules)

    def set_rules(self, rules: Iterable[AliasRule]) -> None:
        self._snapshot = _RuleSnapshot(rules)

    def match(self, name: Optional[str], api: Optional[str] = None) -> Optional[AliasRule]:
        """The rule *name* hits on *api* (None = unscoped), or None."""
        if name is None:
            return None
        return self._snapshot.match(name, api)

    def resolve(self, name: Optional[str], api: Optional[str] = None) -> Optional[str]:
        rule = self.match(name, api)
        return name if rule is None else rule.target


model_alias_resolver = ModelAliasResolver()


def apply_alias(name: Optional[str], api: Optional[str] = None) -> Optional[str]:
    """Resolve *name* and retain its client-facing value when it changes."""
    if name is None or name == alias_resolved_name.get():
        return name
    if api is None:
        api = current_api_surface.get()
    mapped = model_alias_resolver.resolve(name, api)
    if mapped != name:
        original_model_name.set(name)
        alias_resolved_name.set(mapped)
    return mapped


def echo_model_name(request) -> Optional[str]:
    """Return the name supplied by the client, falling back to routed model."""
    return original_model_name.get() or request.model

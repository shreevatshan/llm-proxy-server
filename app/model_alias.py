"""Fast, request-safe resolution of admin-managed model aliases.

Each alias is a rule: an exact name, a case-insensitive substring, or a
case-insensitive regex (re.search), evaluated in admin-defined priority order.
The first enabled rule that matches and is scoped to the request's API surface
wins. Results are memoised per immutable rules snapshot, so after the first
request for a name the lookup is a dict hit regardless of the rule types.

Every profile has two lists: the regular one ("alias"), applied when the
requested model is available to the caller, and the fallback one ("fallback"),
applied when it is not. Which list applies is decided by app.model_resolution
(map_model_name / resolve_model_for_request); this module only matches.
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
from app.auth.models import (
    DEFAULT_PROFILE_ID, MODEL_ALIAS_API_SURFACES, MODEL_ALIAS_KINDS, validate_alias_pattern,
)

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

# The caller's profile, bound after authentication. None before auth and for the
# config admin, who has no profile: both map with Default's list.
current_profile_id: ContextVar[Optional[int]] = ContextVar(
    "current_profile_id", default=None
)

# Bookkeeping that corrects a mapping applied before auth. The tracking middleware
# maps the client's name with Default's list (the profile is not known yet), picking
# the regular or fallback list from the global gate alone, and rewrites the body, so
# handlers re-read that target. Once the caller is known, apply_alias recognises any
# name this request already produced and, if the scope -- the profile, or the list --
# differs, re-resolves the client's original input against the new scope's rules,
# which gives the input back unchanged when nothing there matches it.
#   alias_input_name     -- the name the current mapping was computed from
#   alias_resolved_scope -- the (profile, kind) whose list that mapping used
#   alias_seen_names     -- every name apply_alias has produced in this request
alias_input_name: ContextVar[Optional[str]] = ContextVar("alias_input_name", default=None)
alias_resolved_scope: ContextVar[Optional[tuple]] = ContextVar("alias_resolved_scope", default=None)
alias_seen_names: ContextVar[frozenset] = ContextVar("alias_seen_names", default=frozenset())


def reset_alias_context() -> None:
    """Start a request with no mapping applied and no profile bound."""
    alias_resolved_name.set(None)
    alias_input_name.set(None)
    alias_resolved_scope.set(None)
    alias_seen_names.set(frozenset())
    current_profile_id.set(None)

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
        if not self.rules:
            return None
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


# The list of a profile without mappings of its own.
_NO_RULES = _RuleSnapshot()


class ModelAliasResolver:
    """Immutable-snapshot alias resolver; reads never touch the database.

    One snapshot per profile and list; a profile maps with its own lists alone.
    Default's are also the lists requests map with before authentication. Other
    profiles' snapshots are only consulted after authentication, so unauthenticated
    clients cannot fill their match caches.
    """

    def __init__(self) -> None:
        self._snapshot = _RuleSnapshot()                # Default's regular list
        self._profile_snapshots: dict = {}              # every other profile's
        self._fallback_snapshot = _RuleSnapshot()       # Default's fallback list
        self._fallback_profile_snapshots: dict = {}     # every other profile's
        # Serialises reloads. Each admin edit commits and then reloads; without
        # this, an earlier reload whose SELECT predates a later commit could
        # install its snapshot last and leave stale rules live.
        self._reload_lock = asyncio.Lock()

    async def load_from_database(self) -> None:
        async with self._reload_lock:
            async with AsyncSessionLocal() as db:
                aliases = await get_all_model_aliases(db, all_profiles=True, kind=None)
            rules_by_kind = {kind: {} for kind in MODEL_ALIAS_KINDS}
            for row in aliases:
                if not row.enabled:
                    continue
                try:
                    rule = build_rule(
                        row.alias, row.target_model_id, _parse_apis(row.apis),
                        row.match_type or "exact", row.priority or 0, row.id,
                    )
                except ValueError as e:
                    # Never fail startup over one bad row; it simply never matches.
                    logger.warning("Skipping model alias %r: %s", row.alias, e)
                    continue
                # NULL reads as Default, as for users.profile_id; a missing kind as regular.
                pid = getattr(row, "profile_id", None) or DEFAULT_PROFILE_ID
                kind = getattr(row, "kind", None) or "alias"
                if kind not in rules_by_kind:
                    logger.warning("Skipping model alias %r: unknown kind %r", row.alias, kind)
                    continue
                rules_by_kind[kind].setdefault(pid, []).append(rule)
            regular, fallback = rules_by_kind["alias"], rules_by_kind["fallback"]
            self._snapshot = _RuleSnapshot(regular.pop(DEFAULT_PROFILE_ID, ()))
            self._profile_snapshots = {pid: _RuleSnapshot(rules) for pid, rules in regular.items()}
            self._fallback_snapshot = _RuleSnapshot(fallback.pop(DEFAULT_PROFILE_ID, ()))
            self._fallback_profile_snapshots = {
                pid: _RuleSnapshot(rules) for pid, rules in fallback.items()
            }

    def set_rules(
        self, rules: Iterable[AliasRule], profile_id: int = DEFAULT_PROFILE_ID, kind: str = "alias",
    ) -> None:
        """Replace one list of one profile, Default's regular list unless given (tests and helpers)."""
        if kind == "fallback":
            if profile_id == DEFAULT_PROFILE_ID:
                self._fallback_snapshot = _RuleSnapshot(rules)
            else:
                self._fallback_profile_snapshots = {
                    **self._fallback_profile_snapshots, profile_id: _RuleSnapshot(rules),
                }
        elif profile_id == DEFAULT_PROFILE_ID:
            self._snapshot = _RuleSnapshot(rules)
        else:
            self._profile_snapshots = {**self._profile_snapshots, profile_id: _RuleSnapshot(rules)}

    def _snapshot_for(self, profile_id: Optional[int], kind: str = "alias") -> _RuleSnapshot:
        if kind == "fallback":
            default, others = self._fallback_snapshot, self._fallback_profile_snapshots
        else:
            default, others = self._snapshot, self._profile_snapshots
        if profile_id is None or profile_id == DEFAULT_PROFILE_ID:
            return default
        return others.get(profile_id, _NO_RULES)

    def match(
        self, name: Optional[str], api: Optional[str] = None, profile_id: Optional[int] = None,
        kind: str = "alias",
    ) -> Optional[AliasRule]:
        """The rule *name* hits on *api* (None = unscoped) in *profile_id*'s *kind* list
        (None = Default's), or None."""
        if name is None:
            return None
        return self._snapshot_for(profile_id, kind).match(name, api)

    def resolve(
        self, name: Optional[str], api: Optional[str] = None, profile_id: Optional[int] = None,
        kind: str = "alias",
    ) -> Optional[str]:
        rule = self.match(name, api, profile_id, kind)
        return name if rule is None else rule.target


model_alias_resolver = ModelAliasResolver()


def _alias_scope() -> int:
    """The profile whose list this request maps with (Default's until one is bound)."""
    pid = current_profile_id.get()
    return DEFAULT_PROFILE_ID if pid is None else pid


def _produced_by_this_request(name: str) -> bool:
    """True if *name* is the input or an output of a mapping this request applied."""
    return alias_resolved_name.get() is not None and (
        name == alias_resolved_name.get() or name in alias_seen_names.get()
        or name == alias_input_name.get()
    )


def alias_source_name(name: Optional[str]) -> Optional[str]:
    """The client's name behind *name*: the mapping's input if this request produced
    *name* by mapping, else *name* itself. Availability is judged on this name."""
    if name is not None and _produced_by_this_request(name):
        return alias_input_name.get()
    return name


def apply_alias(name: Optional[str], api: Optional[str] = None, kind: str = "alias") -> Optional[str]:
    """Resolve *name* with the *kind* list and retain its client-facing value when it changes.

    Never chains: a name this request already produced comes back unchanged -- unless
    the scope changed since it was produced (the caller's profile became known, or
    the other list now applies), in which case the client's original name is
    re-resolved against the new scope's rules.
    """
    if name is None:
        return name
    if api is None:
        api = current_api_surface.get()
    profile_id = _alias_scope()
    scope = (profile_id, kind)
    if _produced_by_this_request(name):
        resolved = alias_resolved_name.get()
        if alias_resolved_scope.get() == scope:
            return resolved
        source = alias_input_name.get()
        mapped = model_alias_resolver.resolve(source, api, profile_id, kind)
        alias_resolved_name.set(mapped)
        alias_resolved_scope.set(scope)
        alias_seen_names.set(alias_seen_names.get() | {mapped})
        return mapped
    mapped = model_alias_resolver.resolve(name, api, profile_id, kind)
    if mapped != name:
        original_model_name.set(name)
        alias_input_name.set(name)
        alias_resolved_name.set(mapped)
        alias_resolved_scope.set(scope)
        # A fresh input starts a new chain; names produced for an earlier input no
        # longer stand for the current mapping.
        alias_seen_names.set(frozenset({mapped}))
    return mapped


def echo_model_name(request) -> Optional[str]:
    """Return the name supplied by the client, falling back to routed model."""
    return original_model_name.get() or request.model

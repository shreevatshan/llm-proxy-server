"""Database models for authentication."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, Date, UniqueConstraint, Index, func, text, event, DDL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, date
import re
import json
import math
from re import _constants as _sre, _parser as _sre_parse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional, List, Literal

VALID_AZURE_BACKENDS = {"openai", "foundry"}
DEPLOYMENT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')

# Width of the users.username column. SQLite does not enforce VARCHAR limits, so
# signup validates against this explicitly to keep names portable to backends
# (Postgres/MySQL) that would reject an over-long value at INSERT time.
USERNAME_MAX_LENGTH = 50

# Shortest accepted username. Mirrored by the signup form's client-side check;
# enforced server-side so direct API calls cannot bypass it.
USERNAME_MIN_LENGTH = 3

Base = declarative_base()

# The built-in profile every user starts in. Like global_rate_limits' id=1 singleton,
# being the default is a property of the id, not a stored flag: it cannot be deleted,
# and every NULL or dangling users.profile_id reads as this one.
DEFAULT_PROFILE_ID = 1

MODEL_ACCESS_MODES = ("default", "allow", "deny", "custom")


class UserProfile(Base):
    """A named bundle of rate limits, model access and model mappings.

    Every user belongs to exactly one profile; the limits and access rules a user is
    held to are their profile's. NULL limits mean unlimited.
    """
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    description = Column(String(256), nullable=True)
    rpm_limit = Column(Integer, nullable=True)  # null = unlimited
    rpd_limit = Column(Integer, nullable=True)  # null = unlimited
    # Requests a member may have in progress at once, across all their keys; null = unlimited.
    max_concurrent_requests = Column(Integer, nullable=True)
    # One of MODEL_ACCESS_MODES; same semantics as the per-user policy it replaced.
    model_access_mode = Column(String(16), nullable=False, default="default", server_default="default")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    __table_args__ = (
        # Case-insensitive, as for request pools: "Team" and "team" are one name.
        Index("uq_user_profiles_name_lower", func.lower(name), unique=True),
    )


# Seed the Default profile whenever the table is created, so databases built by
# create_all alone (fresh installs, tests) satisfy users.profile_id's foreign key.
# Existing databases get the same row from _run_auto_migrations.
event.listen(UserProfile.__table__, "after_create", DDL(
    "INSERT INTO user_profiles (id, name, description, model_access_mode, created_at, updated_at) "
    "VALUES (1, 'Default', 'Built-in profile for new users', 'default', "
    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
))


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(USERNAME_MAX_LENGTH), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth users
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_pending_approval = Column(Boolean, default=False)
    # Nullable only because SQLite cannot ADD a REFERENCES column with a non-NULL
    # default; NULL reads as DEFAULT_PROFILE_ID everywhere. No ON DELETE: a profile
    # with members cannot be deleted until they are reassigned.
    profile_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=True,
                        default=DEFAULT_PROFILE_ID, index=True)

    # OAuth fields
    oauth_provider = Column(String(50), nullable=True)  # e.g., "zoho", "google"
    oauth_sub = Column(String(255), nullable=True)      # OAuth subject identifier
    oauth_data = Column(Text, nullable=True)            # JSON data from OAuth provider
    
    # Relationship to API keys
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Relationship to OAuth accounts
    oauth_accounts = relationship("OAuthUser", back_populates="user", cascade="all, delete-orphan")

    # Relationship to rate limit override (one-to-one, optional)
    rate_limit = relationship("UserRateLimit", uselist=False, cascade="all, delete-orphan")


class OAuthUser(Base):
    """OAuth user information for external authentication providers."""
    __tablename__ = "oauth_users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)  # e.g., "zoho", "google"
    provider_user_id = Column(String(255), nullable=False)  # OAuth sub
    email = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    picture = Column(String(500), nullable=True)
    raw_data = Column(Text, nullable=True)  # JSON data from OAuth provider
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user - changed from backref to back_populates for consistency
    user = relationship("User", back_populates="oauth_accounts")
    
    # Unique constraint on provider + provider_user_id
    __table_args__ = (
        UniqueConstraint('provider', 'provider_user_id', name='uq_oauth_provider_user'),
    )


class APIKey(Base):
    """API Key model for authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=False)  # SHA-256 hex digest of the secret
    key_prefix = Column(String(16), nullable=True)  # first chars of the plaintext key, for UI display
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationship to user
    user = relationship("User", back_populates="api_keys")


class ModelConfiguration(Base):
    """Model configuration model for enable/disable individual models."""
    __tablename__ = "model_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(200), unique=True, index=True, nullable=False)  # e.g., "ollama:msi-ai-test-01/llama2"
    provider_key = Column(String(100), ForeignKey("provider_credentials.provider_key"), nullable=False)
    model_name = Column(String(100), nullable=False)  # e.g., "llama2"
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to provider
    provider = relationship("ProviderCredentials", back_populates="models")


# API surfaces a model alias can be scoped to. These mirror the server_name
# strings passed to _add_request_tracking in app/main.py.
MODEL_ALIAS_API_SURFACES = ("openai", "anthropic", "azure_openai")

# How a model alias's `alias` text is compared with the requested model name.
# exact is case-sensitive; contains and regex (re.search) are case-insensitive.
MODEL_ALIAS_MATCH_TYPES = ("exact", "contains", "regex")

# Which list a mapping belongs to. "alias" mappings apply when the requested model
# is available to the caller; "fallback" mappings apply only when it is not (an
# unknown name, a disabled model/provider/group, or one the profile does not allow).
MODEL_ALIAS_KINDS = ("alias", "fallback")

# Python's re has no timeout and regex aliases run on the event loop against
# client-supplied model names (up to 256 characters, see app.model_alias), so
# patterns whose worst-case backtracking could stall the server are refused.
# The check works on the parsed pattern, not its text, so groups, alternation
# and {n,} are all seen. Two shapes are refused:
#  - a repeated group that can itself match in more than one way, e.g. (a+)+,
#    ((a)+)+, (a|aa)+, (a{1,})+: exponential in the name length;
#  - too many choices overall, estimated in bits (log2 of tries): a wide
#    quantifier (*, +, {0,50}) costs about 7 bits, a narrow one (?, {1,3}) log2
#    of its choices, an alternation log2 of its branches, and a pattern not
#    anchored with ^ another 7 bits for the start positions re.search tries.
#    21 bits admits ".*opus.*" or "^a*b*c*" and stays near 0.1 s worst case.
_REGEX_WIDE_BITS = 7.0
_REGEX_BUDGET_BITS = 21.0
_REPEATS = (_sre.MAX_REPEAT, _sre.MIN_REPEAT, _sre.POSSESSIVE_REPEAT)


def _regex_children(op, av) -> list:
    """The nested sub-patterns of one parsed regex item."""
    if op in _REPEATS:
        return [av[2]]
    if op is _sre.SUBPATTERN:
        return [av[3]]
    if op in (_sre.ASSERT, _sre.ASSERT_NOT):
        return [av[1]]
    if op is _sre.ATOMIC_GROUP:
        return [av]
    if op is _sre.BRANCH:
        return list(av[1])
    if op is _sre.GROUPREF_EXISTS:
        return [p for p in av[1:] if p is not None]
    return []


def _regex_has_choice(parsed) -> bool:
    """True if *parsed* can match the same text in more than one way."""
    for op, av in parsed:
        if op in (_sre.BRANCH, _sre.GROUPREF_EXISTS):
            return True
        if op in _REPEATS and av[0] != av[1]:
            return True
        if any(_regex_has_choice(p) for p in _regex_children(op, av)):
            return True
    return False


def _regex_cost_bits(parsed) -> float:
    """Estimated log2 of backtracking tries from one start position."""
    bits = 0.0
    for op, av in parsed:
        if op in _REPEATS:
            low, high, body = av
            if high > 1 and _regex_has_choice(body):
                raise ValueError(
                    "regex repeats a group that itself contains a quantifier or alternation "
                    "(e.g. (a+)+ or (a|aa)+), which can hang the server; use a character "
                    "class such as [a-z0-9.-]+ instead"
                )
            if high > low:
                bits += min(math.log2(high - low + 1), _REGEX_WIDE_BITS)
            bits += _regex_cost_bits(body)
        elif op in (_sre.BRANCH, _sre.GROUPREF_EXISTS):
            branches = [2 ** _regex_cost_bits(p) for p in _regex_children(op, av)]
            if op is _sre.GROUPREF_EXISTS and av[2] is None:
                branches.append(1)   # the implicit empty "no" branch
            bits += math.log2(sum(branches))
        else:
            bits += sum(_regex_cost_bits(p) for p in _regex_children(op, av))
    return bits


def _check_regex_cost(alias: str) -> None:
    parsed = _sre_parse.parse(alias, re.IGNORECASE)
    bits = _regex_cost_bits(parsed)
    anchored = bool(parsed.data) and parsed.data[0] in (
        (_sre.AT, _sre.AT_BEGINNING), (_sre.AT, _sre.AT_BEGINNING_STRING),
    )
    if not anchored:
        bits += _REGEX_WIDE_BITS
    if bits > _REGEX_BUDGET_BITS:
        raise ValueError(
            "regex is too open-ended to run safely on every request: use fewer "
            "quantifiers, anchor it with ^, or drop a leading or trailing .* "
            "(patterns already match anywhere in the name)"
        )


def validate_alias_pattern(alias: str, target_model_id: str, match_type: str) -> None:
    """Raise ValueError if *alias* is not a usable pattern for *match_type*."""
    if match_type not in MODEL_ALIAS_MATCH_TYPES:
        raise ValueError(f"unknown match type: {match_type}")
    if match_type == "exact" and alias == target_model_id:
        raise ValueError("alias must differ from target_model_id")
    if match_type == "regex":
        try:
            re.compile(alias, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}")
        _check_regex_cost(alias)


class ModelAlias(Base):
    """Admin-managed client-facing alias for a canonical model identifier."""
    __tablename__ = "model_aliases"

    id = Column(Integer, primary_key=True, index=True)
    alias = Column(String(200), index=True, nullable=False)
    # The profile whose members this mapping applies to. Each profile has its own
    # list; there is no list shared across profiles.
    profile_id = Column(Integer, ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False,
                        default=DEFAULT_PROFILE_ID)
    target_model_id = Column(String(200), nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
    # JSON array of API surfaces this alias applies to. NULL (legacy rows) means
    # "all surfaces". Mirrors ProviderCredentials.supported_apis.
    apis = Column(Text, nullable=True, default='["openai", "anthropic", "azure_openai"]')
    # One of MODEL_ALIAS_MATCH_TYPES.
    match_type = Column(String(16), nullable=False, default="exact", server_default="exact")
    # Evaluation order: lowest first, ties broken by id. Not contiguous (deletes
    # leave gaps); the admin UI shows row position instead.
    priority = Column(Integer, nullable=False, default=0, server_default="0")
    # One of MODEL_ALIAS_KINDS. Each kind is its own list with its own priority order.
    kind = Column(String(16), nullable=False, default="alias", server_default="alias")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # One alias text per list of a profile.
        Index("uq_model_aliases_scope_kind_alias", profile_id, kind, alias, unique=True),
    )


class ResponseProviderMapping(Base):
    """Maps Responses API response IDs to the provider that created them.
    Used to route retrieve/delete/cancel/input_items requests to the correct upstream."""
    __tablename__ = "response_provider_mappings"

    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(String(200), unique=True, index=True, nullable=False)
    provider_key = Column(String(100), nullable=False)  # e.g., "openai:primary"
    model_name = Column(String(200), nullable=True)      # Original model string from request
    # Owning user id, for the Responses API ownership (IDOR) check on
    # retrieve/delete/cancel/input_items. Nullable so pre-migration rows (and
    # admin-created responses) keep NULL; a NULL owner is treated as unowned.
    user_id = Column(Integer, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserRateLimit(Base):
    """Per-user rate limit overrides. Absent row = inherit global defaults."""
    __tablename__ = "user_rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    rpm_limit = Column(Integer, nullable=True)  # null = inherit global default
    rpd_limit = Column(Integer, nullable=True)  # null = inherit global default
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class GlobalRateLimit(Base):
    """Global rate limit defaults. Single row (id=1). null = unlimited."""
    __tablename__ = "global_rate_limits"

    id = Column(Integer, primary_key=True)  # always 1
    rpm_default = Column(Integer, nullable=True)
    rpd_default = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class UserManagementSettings(Base):
    """Admin-managed user account policies. Single row (id=1); absent row = defaults."""
    __tablename__ = "user_management_settings"

    id = Column(Integer, primary_key=True)  # always 1
    allow_self_delete = Column(Boolean, default=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class PoolSettings(Base):
    """Admin-managed request-pool policy. Single row (id=1); absent row = defaults.

    Both values are defaults every pool inherits; RequestPool carries a nullable
    override of each.
    """
    __tablename__ = "pool_settings"

    id = Column(Integer, primary_key=True)  # always 1
    max_members = Column(Integer, default=25, nullable=False)  # MAX_POOL_MEMBERS, defined below
    # Most of a pool's daily limit, in percent, one member may consume on any scope.
    # NULL means no per-member cap -- any member may spend the whole pool.
    max_member_share_pct = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class WebSearchSettings(Base):
    """Web search interception settings. Single row (id=1).

    Holds the admin-only knobs and the defaults every user inherits: the default
    backend and the result limits. Search parameters live on each backend
    (WebSearchBackend). Databases from before multiple backends still carry the
    old per-backend columns; only _migrate_websearch_backends reads them.
    """
    __tablename__ = "websearch_settings"

    id = Column(Integer, primary_key=True)  # always 1
    enabled = Column(Boolean, default=False, nullable=False)
    default_backend_id = Column(Integer, nullable=True)  # NULL -> interception inactive
    # Defaults users may override
    max_results = Column(Integer, nullable=True)
    max_snippet_chars = Column(Integer, nullable=True)
    # Admin only
    timeout_seconds = Column(Integer, nullable=True)
    max_agentic_loops = Column(Integer, nullable=True)
    max_queries_per_turn = Column(Integer, nullable=True)
    apply_to = Column(Text, nullable=True)           # JSON array of API surfaces
    enabled_providers = Column(Text, nullable=True)  # JSON array of provider keys; "*" = all
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class WebSearchBackend(Base):
    """A SearXNG or 4get instance that searches can be sent to.

    Parameter columns are named after WebSearchConfig's fields, and only the
    type's own ones (WEBSEARCH_BACKEND_PARAMS) are read. NULL uses
    WEBSEARCH_DEFAULTS.
    """
    __tablename__ = "websearch_backends"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False)
    type = Column(String(20), nullable=False)        # "searxng" | "fourget"
    base_url = Column(String(500), nullable=False)
    # Off = no request uses it; users who picked it fall back to the default.
    enabled = Column(Boolean, default=True, nullable=False, server_default="1")
    # Off = users can no longer pick it (it can still be the default).
    user_selectable = Column(Boolean, default=True, nullable=False, server_default="1")
    # SearXNG
    engines = Column(String(500), nullable=True)
    categories = Column(String(200), nullable=True)
    language = Column(String(20), nullable=True)
    # 4get
    fourget_scraper = Column(String(50), nullable=True)
    fourget_lang = Column(String(20), nullable=True)
    fourget_country = Column(String(32), nullable=True)
    # Both
    safesearch = Column(Integer, nullable=True)
    time_range = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    __table_args__ = (
        # Same reason as uq_request_pools_name_lower: the routes reject names that
        # differ only by case, and this makes a concurrent create lose with a 409.
        Index("uq_websearch_backends_name_lower", func.lower(name), unique=True),
    )


class UserWebSearchSettings(Base):
    """A user's web search choices. NULL inherits the admin default.

    Search parameters are kept per backend (UserWebSearchBackendParams), since
    engines or a scraper only mean something on one instance.
    """
    __tablename__ = "user_websearch_settings"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    # NULL = the admin default. A backend that is later disabled, deleted or made
    # unselectable is ignored at request time rather than cleared here, so the
    # choice comes back if the admin restores it (deleting does clear it).
    backend_id = Column(Integer, ForeignKey("websearch_backends.id", ondelete="SET NULL"), nullable=True)
    max_results = Column(Integer, nullable=True)
    max_snippet_chars = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserWebSearchBackendParams(Base):
    """A user's search parameters for one backend. NULL inherits the backend's own."""
    __tablename__ = "user_websearch_backend_params"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    backend_id = Column(Integer, ForeignKey("websearch_backends.id", ondelete="CASCADE"), nullable=False)
    engines = Column(String(500), nullable=True)
    categories = Column(String(200), nullable=True)
    language = Column(String(20), nullable=True)
    fourget_scraper = Column(String(50), nullable=True)
    fourget_lang = Column(String(20), nullable=True)
    fourget_country = Column(String(32), nullable=True)
    safesearch = Column(Integer, nullable=True)
    time_range = Column(String(10), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "backend_id", name="uq_user_websearch_backend_params"),
    )


# The config-based admin is not a row in `users`, so its usage is recorded under this
# reserved id. Every usage read that resolves ids to names must special-case it.
ADMIN_USAGE_USER_ID = 0

# The pool_id a usage row carries when the sender was not in a pool at the time.
NO_POOL_ID = 0


class RequestUsage(Base):
    """Per-day request usage counters keyed by (date, user_id, model, server, pool_id).

    user_id is the attribution key: renaming a user leaves the rows where they are, and
    a deleted user's rows are purged with the account, so a later account reusing the
    name inherits nothing. The config admin, which has no users row, is stored under
    ADMIN_USAGE_USER_ID.

    pool_id is stamped at write time from the sender's pool membership at the moment the
    request completed (NO_POOL_ID when unpooled). It is the whole basis of pool usage
    views: "what did this pool consume" is a filter on this column, exact to the
    request, with no reconstruction from membership history.

    user_identity and user_type are display labels only, never part of the key. The
    identity is the username at the time of the last write and is relabelled in place
    on rename; user_type is whichever delivery path last wrote the row, and reads report
    "mixed" when a person used more than one.
    """
    __tablename__ = "request_usage"

    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True, nullable=False)
    user_id = Column(Integer, nullable=False)
    pool_id = Column(Integer, default=NO_POOL_ID, nullable=False)
    user_identity = Column(String(200), nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('date', 'user_id', 'model', 'server', 'pool_id', name='uq_usage_day'),
        Index('ix_usage_user_date', 'user_id', 'date'),
        Index('ix_usage_pool_date', 'pool_id', 'date'),
    )


class RequestUsageHourly(Base):
    """Per-hour request usage counters; retained for ~48 hours to serve the rolling-24h window.

    Keyed by (date, hour, user_id, model, server, pool_id). See RequestUsage for the
    meaning of each key column and of the user_identity / user_type labels.
    """
    __tablename__ = "request_usage_hourly"

    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True, nullable=False)
    hour = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    pool_id = Column(Integer, default=NO_POOL_ID, nullable=False)
    user_identity = Column(String(200), nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('date', 'hour', 'user_id', 'model', 'server', 'pool_id', name='uq_usage_hour'),
        Index('ix_usage_hourly_date_hour', 'date', 'hour'),
        Index('ix_usage_hourly_user_date', 'user_id', 'date'),
        Index('ix_usage_hourly_pool_date', 'pool_id', 'date'),
    )


class RequestUsageMonthly(Base):
    """Per-month rolled-up request usage; retained forever.

    Keyed by (year, month, user_id, model, server, pool_id). See RequestUsage for the
    meaning of each key column and of the user_identity / user_type labels. Because
    pool_id travels with the row into the rollup, pool usage stays exact for rolled-up
    months; only rows that predate the pool_id column carry a month-granular backfill.
    """
    __tablename__ = "request_usage_monthly"

    id = Column(Integer, primary_key=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    pool_id = Column(Integer, default=NO_POOL_ID, nullable=False)
    user_identity = Column(String(200), nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('year', 'month', 'user_id', 'model', 'server', 'pool_id', name='uq_usage_month'),
        Index('ix_usage_monthly_year_month', 'year', 'month'),
        Index('ix_usage_monthly_user_ym', 'user_id', 'year', 'month'),
        Index('ix_usage_monthly_pool_ym', 'pool_id', 'year', 'month'),
    )


class UsageMeta(Base):
    """Facts about how the usage tables were written that a read must not guess at.

    'timezone' is the IANA zone every date/hour bucket in the usage
    tables was computed in. Rows carry local dates with no zone, so changing TIMEZONE
    after deployment would silently mix two calendars; startup compares the configured
    zone against this row and logs an error on mismatch.

    'pool_id_high_water' is the highest pool id ever allocated. Usage rows outlive
    their pool, so an id must never be handed out twice (see allocate_pool_id).
    """
    __tablename__ = "usage_meta"

    key = Column(String(64), primary_key=True)
    value = Column(String(200), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelGroup(Base):
    """Named group of models sharing a rate-limit bucket.

    Members are explicit models (ModelGroupMember) and/or whole provider instances
    (ModelGroupInstanceMember). Instance membership is live: every model on the
    instance belongs to the group, including ones synced later, unless another group
    claims that model explicitly. Models no group claims form the built-in Default
    group, which has no row (app.profiles.DEFAULT_GROUP_ID).
    """
    __tablename__ = "model_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    # Off = the group's models are not served (cache._passes_global_gate), like a
    # disabled model. Members and per-profile limits are kept for when it is back on.
    enabled = Column(Boolean, default=True, nullable=False, server_default="1")
    rpm_default = Column(Integer, nullable=True)  # null = unlimited
    rpd_default = Column(Integer, nullable=True)  # null = unlimited
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    members = relationship("ModelGroupMember", back_populates="group", cascade="all, delete-orphan")
    instance_members = relationship("ModelGroupInstanceMember", back_populates="group", cascade="all, delete-orphan")
    user_limits = relationship("UserModelGroupRateLimit", back_populates="group", cascade="all, delete-orphan")
    profile_limits = relationship("ProfileModelGroupLimit", back_populates="group", cascade="all, delete-orphan")


class ModelGroupMember(Base):
    """Maps a model_id to exactly one ModelGroup (unique on model_id)."""
    __tablename__ = "model_group_members"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(200), unique=True, nullable=False)  # unique enforces single-group membership
    created_at = Column(DateTime, default=datetime.utcnow)

    group = relationship("ModelGroup", back_populates="members")

    __table_args__ = (
        Index("ix_model_group_members_group_id", "group_id"),
    )


class ModelGroupInstanceMember(Base):
    """Puts every model of one provider instance in a ModelGroup (unique on provider_key)."""
    __tablename__ = "model_group_instance_members"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    provider_key = Column(String(100), unique=True, nullable=False)  # unique enforces single-group membership
    created_at = Column(DateTime, default=datetime.utcnow)

    group = relationship("ModelGroup", back_populates="instance_members")

    __table_args__ = (
        Index("ix_model_group_instance_members_group_id", "group_id"),
    )


class UserModelGroupRateLimit(Base):
    """Per-user override for a model group's rate limits. Absent row = inherit group default."""
    __tablename__ = "user_model_group_rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    rpm_limit = Column(Integer, nullable=True)  # null = inherit group default
    rpd_limit = Column(Integer, nullable=True)  # null = inherit group default
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    group = relationship("ModelGroup", back_populates="user_limits")

    __table_args__ = (
        UniqueConstraint("user_id", "group_id", name="uq_user_model_group_rate_limit"),
    )


class UserModelAccessPolicy(Base):
    """Per-user model-access policy. Absent row = 'default' mode (obey global config).

    mode is one of:
      - default : strictly follow the global model config (no per-user override).
      - allow   : every model ON, overriding the global gate (incl. future models).
      - deny    : every model OFF, overriding the global gate (incl. future models).
      - custom  : per-model overrides layered on the global config; untoggled/new
                  models follow global, explicit overrides win (can beat the gate).
    """
    __tablename__ = "user_model_access_policies"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    mode = Column(String(16), default="default", nullable=False)  # default|allow|deny|custom
    default_allow = Column(Boolean, default=True, nullable=False)  # legacy; kept in sync for back-compat
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class UserModelAccessException(Base):
    """Explicit per-(user, model) override against the user's default policy."""
    __tablename__ = "user_model_access_exceptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(200), nullable=False)  # e.g. "azure:primary/gpt-4"
    is_allowed = Column(Boolean, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "model_id", name="uq_user_model_access_exception"),
        Index("ix_user_model_access_exceptions_user_id", "user_id"),
    )


class ProfileModelGroupLimit(Base):
    """A profile's RPM/RPD for one model group. Absent row or NULL = unlimited."""
    __tablename__ = "profile_model_group_limits"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    rpm_limit = Column(Integer, nullable=True)
    rpd_limit = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    group = relationship("ModelGroup", back_populates="profile_limits")

    __table_args__ = (
        UniqueConstraint("profile_id", "group_id", name="uq_profile_model_group_limit"),
    )


class ProfileModelAccessException(Base):
    """Explicit per-(profile, model) override, honoured when the profile's mode is custom."""
    __tablename__ = "profile_model_access_exceptions"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(200), nullable=False)
    is_allowed = Column(Boolean, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    __table_args__ = (
        UniqueConstraint("profile_id", "model_id", name="uq_profile_model_access_exception"),
        Index("ix_profile_model_access_exceptions_profile_id", "profile_id"),
    )


ANTHROPIC_AUTH_STYLES = ("x-api-key", "bearer")


class ProviderCredentials(Base):
    """Provider credentials model for storing provider configurations."""
    __tablename__ = "provider_credentials"
    
    id = Column(Integer, primary_key=True, index=True)
    provider_key = Column(String(100), unique=True, index=True, nullable=False)  # e.g., "llamacpp:site24x7-fgpu" or "azure:primary"
    provider_type = Column(String(50), nullable=False)  # e.g., "azure", "openai_compatible"
    instance_name = Column(String(100), nullable=False)  # e.g., "primary", "site24x7-fgpu" (customizable)
    enabled = Column(Boolean, default=True)
    
    # Provider-specific configuration fields (nullable, used based on provider_type)
    endpoint = Column(String(500), nullable=True)          # OpenAI, Azure
    api_key = Column(String(500), nullable=True)           # OpenAI, Azure, Google
    discovery_api_version = Column(String(50), nullable=True)  # Azure: Management API version for dynamic discovery
    azure_backend = Column(String(50), nullable=True)      # Azure: "openai" or "foundry"
    region = Column(String(50), nullable=True)             # Bedrock
    access_key_id = Column(String(200), nullable=True)     # Bedrock
    secret_access_key = Column(String(500), nullable=True) # Bedrock
    base_url = Column(String(500), nullable=True)          # Ollama, OpenAI-compatible
    deployments_json = Column(Text, nullable=True)         # Azure (JSON array)
    provider_name = Column(String(100), nullable=False)    # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")

    dynamic_discovery = Column(Boolean, nullable=True)     # Azure: True to discover via the models API, False to use manual deployments
    
    # Supported API formats for custom providers (JSON array, e.g., '["openai", "anthropic"]')
    supported_apis = Column(Text, nullable=True, default='["openai"]')
    # Custom providers with a native Anthropic endpoint: its base URL when it
    # differs from base_url (e.g. https://api.deepseek.com/anthropic), and the
    # auth header style ("x-api-key" default, or "bearer").
    anthropic_base_url = Column(String(500), nullable=True)
    anthropic_auth_style = Column(String(20), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to models
    models = relationship("ModelConfiguration", back_populates="provider", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Request pooling: a group of users sharing one daily request quota (RPD only)
# ---------------------------------------------------------------------------

# Default cap on pool size, used until an admin saves PoolSettings. The admin can
# move it (globally or per pool) up to POOL_MEMBERS_HARD_CAP, the bound that keeps a
# single settlement's apportionment loop (O(members^2) in the worst case, when every
# member caps out) cheap -- the MAX_BULK_USER_IDS precedent in app/routes/admin.py.
MAX_POOL_MEMBERS = 25
POOL_MEMBERS_HARD_CAP = 100
POOL_MEMBERS_MIN = 2

# Invitation states. 'superseded' is written when the invitee joins some other pool,
# so a stale invite can never later move them silently.
POOL_INVITE_STATUSES = frozenset({"pending", "accepted", "declined", "cancelled", "superseded"})

# Scope kinds for the ledger/carry tables. Mirrors the three rate-limit tiers.
POOL_SCOPE_OVERALL = "overall"
POOL_SCOPE_MODEL_GROUP = "model_group"


class RequestPool(Base):
    """A group of users sharing one daily request quota (RPD only; RPM stays per-user).

    Joining contributes the member's own RPD limit and their own consumption so far
    today, and costs them nothing for what the pool spent before they arrived. Every
    composition change settles the interval that just closed -- see settle_pool() in
    app/auth/pools.py.
    """
    __tablename__ = "request_pools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    owner_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # Admin overrides of PoolSettings for this pool alone; NULL inherits the global value.
    max_members = Column(Integer, nullable=True)
    max_member_share_pct = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    members = relationship("RequestPoolMember", back_populates="pool", cascade="all, delete-orphan")
    invitations = relationship("RequestPoolInvitation", back_populates="pool", cascade="all, delete-orphan")

    __table_args__ = (
        # The column's own UNIQUE is case-sensitive, so it admits "Alpha" alongside
        # "alpha". The routes reject those as a clash (ilike), but a pre-check is not
        # a constraint: two concurrent creates can both pass it. This index makes the
        # database agree with the routes, which is also what lets the IntegrityError
        # handlers there turn a lost race into a 409 instead of a duplicate pool.
        Index("uq_request_pools_name_lower", func.lower(name), unique=True),
    )


class RequestPoolMember(Base):
    """Membership of a user in a pool. At most one pool per user."""
    __tablename__ = "request_pool_members"

    id = Column(Integer, primary_key=True, index=True)
    pool_id = Column(Integer, ForeignKey("request_pools.id", ondelete="CASCADE"), nullable=False)
    # UNIQUE enforces "one pool per user" at the schema level -- the same device
    # ModelGroupMember.model_id uses for single-group membership.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)

    pool = relationship("RequestPool", back_populates="members")

    __table_args__ = (
        Index("ix_request_pool_members_pool_id", "pool_id"),
    )


class PoolMembershipInterval(Base):
    """One continuous stint of a user in a pool, in LOCAL usage dates.

    RequestPoolMember answers "who is in the pool now"; this answers "who was in it
    when", which is what any usage window reaching into the past needs. Without it,
    pool usage is reconstructed as "current members x all their rows", so a user who
    burned 900 requests before joining drags all 900 into the pool's total, and a
    member who leaves retroactively erases consumption that really was the pool's.

    Dates are time_utils.local_* so they compare directly against request_usage.date;
    RequestPoolMember.joined_at is a UTC datetime and is not comparable to one.

    Both ends are INCLUSIVE. Usage is day-grained, so a mid-day join cannot be split:
    the whole day counts toward the pool, which is what settlement already charges.
    Rows are append-only -- a rejoin opens a second interval rather than reopening the
    closed one, and reads take the union of the spans.
    """
    __tablename__ = "pool_membership_intervals"

    id = Column(Integer, primary_key=True, index=True)
    pool_id = Column(Integer, ForeignKey("request_pools.id", ondelete="CASCADE"), nullable=False)
    # Keyed by user_id, resolved to a username at read time, so a rename cannot
    # orphan the history the way the usage tables' identity strings can.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    joined_on = Column(Date, nullable=False)
    left_on = Column(Date, nullable=True)   # NULL = still a member

    __table_args__ = (
        Index("ix_pmi_pool_user", "pool_id", "user_id"),
    )


class RequestPoolInvitation(Base):
    """An invitation for a user to join a pool. The invitee must accept."""
    __tablename__ = "request_pool_invitations"

    id = Column(Integer, primary_key=True, index=True)
    pool_id = Column(Integer, ForeignKey("request_pools.id", ondelete="CASCADE"), nullable=False)
    inviter_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    invitee_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    status = Column(String(16), default="pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    responded_at = Column(DateTime, nullable=True)

    pool = relationship("RequestPool", back_populates="invitations")

    __table_args__ = (
        # At most one *pending* invite per (pool, invitee); declined/cancelled rows are
        # kept as history and must not block a re-invite.
        Index("uq_pool_invite_pending", "pool_id", "invitee_user_id",
              unique=True, sqlite_where=text("status = 'pending'")),
    )


class RequestPoolLedger(Base):
    """Per-member, per-scope running charge for today, maintained by settlement accrual.

    `charged` is how many of the pool's requests this member has been held responsible
    for so far today: their baseline at join, plus their limit-share of each interval
    that has closed since. It exists so a member is charged only for consumption that
    happened while they were in the pool.

    INVARIANT: after every composition change, SUM(charged) over current members equals
    the pool's effective used. That is why the pool's "usage at last settlement" is
    never stored -- it *is* SUM(charged).

    Rows are day-scoped and die with the membership; a missing row reads as 0.
    """
    __tablename__ = "request_pool_ledger"

    id = Column(Integer, primary_key=True, index=True)
    pool_id = Column(Integer, ForeignKey("request_pools.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    usage_date = Column(Date, index=True, nullable=False)    # time_utils.local_today()
    scope_kind = Column(String(16), nullable=False)          # overall (Default group) | model_group
    scope_id = Column(Integer, default=0, nullable=False)    # 0 for overall, else group id
    charged = Column(Integer, default=0, nullable=False)     # whole requests, >= 0
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("pool_id", "user_id", "usage_date", "scope_kind", "scope_id",
                         name="uq_pool_ledger"),
    )


class UserRpdCarry(Base):
    """Day-scoped RPD adjustment written when pool composition changes.

        effective_used(user, scope, today) = SUM(request_usage rows) + carry

    `carry` is NEGATIVE for a member who consumed more than the share they were charged
    and POSITIVE for one who consumed less. It is a RATE-LIMIT construct only: no usage
    view ever adds it, so attribution and history stay exactly as recorded. Rows expire
    at local midnight.

    Unlike the ledger this lives on the USER, not the membership -- it must outlive the
    pool the user just left, and it follows them into the next one.
    """
    __tablename__ = "user_rpd_carries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    usage_date = Column(Date, index=True, nullable=False)    # time_utils.local_today()
    scope_kind = Column(String(16), nullable=False)          # overall (Default group) | model_group
    scope_id = Column(Integer, default=0, nullable=False)    # 0 for overall, else group id
    carry = Column(Integer, default=0, nullable=False)       # whole requests; may be negative
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "usage_date", "scope_kind", "scope_id",
                         name="uq_rpd_carry"),
    )


# Pydantic models for API requests/responses
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool
    is_pending_approval: bool = False
    profile_id: Optional[int] = None

    class Config:
        from_attributes = True


class AdminUserCreate(UserCreate):
    """Admin-created account. A separate model so public signup, which shares
    UserCreate, can never choose its own profile."""
    profile_id: Optional[int] = None


class APIKeyCreate(BaseModel):
    name: str


class APIKeyResponse(BaseModel):
    id: int
    name: str
    api_key: str
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    id: int
    name: str
    api_key_preview: str  # Only show first 8 characters
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    is_admin: Optional[bool] = False


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str


class AdminPasswordReset(BaseModel):
    new_password: str


class AdminUserModify(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    new_password: Optional[str] = None


def _clean_name(name: str) -> str:
    """Validate a profile, model group or instance group name."""
    name = name.strip()
    if not name or len(name) > 64:
        raise ValueError("name must be 1–64 characters")
    if not re.match(r'^[A-Za-z0-9_\- ]+$', name):
        raise ValueError("name may only contain letters, digits, spaces, hyphens, and underscores")
    return name


class ProfileCreate(BaseModel):
    name: str
    description: Optional[str] = None
    # Copy limits, model access and mappings from this profile. None starts empty
    # (everything unlimited, no mappings, access follows the global switches).
    clone_from_id: Optional[int] = None

    @model_validator(mode="after")
    def validate_fields(self):
        self.name = _clean_name(self.name)
        if self.description is not None:
            self.description = self.description.strip()[:256]
        return self


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_fields(self):
        if self.name is not None:
            self.name = _clean_name(self.name)
        if self.description is not None:
            self.description = self.description.strip()[:256]
        return self


class ProfileLimitsUpdate(BaseModel):
    """RPM/RPD for a profile (overall, or one group). null = unlimited; an omitted
    field is left unchanged."""
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_limit is not None and self.rpm_limit < 0:
            raise ValueError("rpm_limit must be >= 0")
        if self.rpd_limit is not None and self.rpd_limit < 0:
            raise ValueError("rpd_limit must be >= 0")
        return self


class ProfileOverallLimitsUpdate(ProfileLimitsUpdate):
    """A profile's own limits: Default-group RPM/RPD plus the profile-wide concurrency
    cap, which has no per-group form. Same null/omitted rules."""
    max_concurrent_requests: Optional[int] = None

    @model_validator(mode="after")
    def validate_concurrency(self):
        # 0 would refuse every request a member sends; leave the field empty for unlimited.
        if self.max_concurrent_requests is not None and self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        return self


class ProfileResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    is_default: bool = False
    member_count: int = 0
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    max_concurrent_requests: Optional[int] = None
    model_access_mode: str = "default"
    exception_count: int = 0
    alias_count: int = 0
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class ProfileGroupLimitResponse(BaseModel):
    group_id: int                  # 0 = the built-in Default group
    name: str
    is_default: bool = False
    enabled: bool = True           # off = the group's models are not served
    member_count: int = 0          # models the group currently governs
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None


class ProfileLimitsResponse(BaseModel):
    profile: ProfileResponse
    groups: List[ProfileGroupLimitResponse] = []   # Default first


class ProfileMemberResponse(BaseModel):
    user_id: int
    username: str
    email: str
    is_active: bool = True
    current_rpm_count: int = 0
    current_rpd_count: int = 0


class UserProfileAssign(BaseModel):
    profile_id: int


class UserManagementSettingsResponse(BaseModel):
    allow_self_delete: bool = True
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True


class UserManagementSettingsUpdate(BaseModel):
    allow_self_delete: bool


class GlobalRateLimitUpdate(BaseModel):
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_default is not None and self.rpm_default < 0:
            raise ValueError("rpm_default must be >= 0")
        if self.rpd_default is not None and self.rpd_default < 0:
            raise ValueError("rpd_default must be >= 0")
        return self


WEBSEARCH_API_SURFACES = ("anthropic_messages", "chat_completions", "responses")
WEBSEARCH_SAFESEARCH_VALUES = (0, 1, 2)
WEBSEARCH_TIME_RANGES = ("day", "week", "month", "year")

WEBSEARCH_PROVIDER_SEARXNG = "searxng"
WEBSEARCH_PROVIDER_FOURGET = "fourget"
# Display names, for messages and the admin UI.
WEBSEARCH_PROVIDER_LABELS = {
    WEBSEARCH_PROVIDER_SEARXNG: "SearXNG",
    WEBSEARCH_PROVIDER_FOURGET: "4get",
}
WEBSEARCH_PROVIDERS = tuple(WEBSEARCH_PROVIDER_LABELS)

# 4get scrapers that implement a web search (scraper/*.php with a web() method,
# as listed by its own frontend). A scraper the instance does not know falls
# back to its default silently, so the admin form validates against this.
# Kept in 4get's order: the first entry is its own default and ours.
FOURGET_WEB_SCRAPERS = (
    "ddg", "brave", "google", "google_api", "google_cse", "startpage",
    "yandex", "yahoo_japan", "yep", "mojeek", "mwmbl", "marginalia",
    "naver", "baidu", "coccoc", "solofield", "purili", "wiby",
)

# Search parameters each backend type reads, named like WebSearchConfig's fields
# and the websearch_backends columns. 4get maps safesearch and time_range too.
WEBSEARCH_BACKEND_PARAMS = {
    WEBSEARCH_PROVIDER_SEARXNG: ("engines", "categories", "language", "safesearch", "time_range"),
    WEBSEARCH_PROVIDER_FOURGET: ("fourget_scraper", "fourget_lang", "fourget_country", "safesearch", "time_range"),
}
WEBSEARCH_PARAM_FIELDS = tuple(dict.fromkeys(f for fields in WEBSEARCH_BACKEND_PARAMS.values() for f in fields))
# Parameters a user overrides together: setting any one of a group takes the whole
# group from the user. SearXNG ignores categories whenever engines is sent, and
# 4get's lang/country codes are specific to the scraper.
WEBSEARCH_PARAM_GROUPS = (
    ("engines", "categories"),
    ("fourget_scraper", "fourget_lang", "fourget_country"),
)
# Bounds of the limits users may override, shared with the admin form.
WEBSEARCH_LIMITS = {"max_results": (1, 20), "max_snippet_chars": (50, 5000)}

# Effective values used when a websearch_settings or websearch_backends column is NULL.
WEBSEARCH_DEFAULTS = {
    "enabled": False,
    "provider": WEBSEARCH_PROVIDER_SEARXNG,
    "engines": None,
    "categories": "general",
    "language": "auto",
    "safesearch": 1,
    "time_range": None,
    "fourget_scraper": FOURGET_WEB_SCRAPERS[0],
    "fourget_lang": None,
    "fourget_country": None,
    "max_results": 5,
    "max_snippet_chars": 500,
    "timeout_seconds": 10,
    "max_agentic_loops": 3,
    "max_queries_per_turn": 5,
    "apply_to": list(WEBSEARCH_API_SURFACES),
    "enabled_providers": [],
}


def _check_url(v: Optional[str], field_name: str) -> Optional[str]:
    """Trimmed http(s) URL without a trailing slash; blank -> None."""
    if isinstance(v, str):
        v = v.strip() or None
    if v is None:
        return v
    if not re.match(r"^https?://[^\s/]+", v):
        raise ValueError(f"{field_name} must start with http:// or https://")
    return v.rstrip("/")


class WebSearchParamsMixin(BaseModel):
    """Search parameters of a backend, shared by the admin and user forms.

    None means "not set": the backend's own value for a user, WEBSEARCH_DEFAULTS
    for a backend.
    """
    engines: Optional[str] = None
    categories: Optional[str] = None
    language: Optional[str] = None
    fourget_scraper: Optional[str] = None
    fourget_lang: Optional[str] = None
    fourget_country: Optional[str] = None
    safesearch: Optional[int] = None
    time_range: Optional[str] = None

    @field_validator(
        "engines", "categories", "language", "time_range",
        "fourget_scraper", "fourget_lang", "fourget_country",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("fourget_scraper")
    @classmethod
    def _validate_fourget_scraper(cls, v):
        # 4get does not reject an unknown scraper -- it quietly falls back to the
        # instance default -- so the typo has to be caught here.
        if v is not None and v not in FOURGET_WEB_SCRAPERS:
            raise ValueError(f"fourget_scraper must be one of {', '.join(FOURGET_WEB_SCRAPERS)}")
        return v

    @field_validator("fourget_lang", "fourget_country")
    @classmethod
    def _validate_fourget_locale(cls, v, info):
        if v is not None and not re.match(r"^[A-Za-z0-9_-]{2,16}$", v):
            raise ValueError(f"{info.field_name} must be 2-16 letters, digits, '-' or '_'")
        return v

    @field_validator("safesearch")
    @classmethod
    def _validate_safesearch(cls, v):
        if v is not None and v not in WEBSEARCH_SAFESEARCH_VALUES:
            raise ValueError("safesearch must be 0, 1 or 2")
        return v

    @field_validator("time_range")
    @classmethod
    def _validate_time_range(cls, v):
        if v is not None and v not in WEBSEARCH_TIME_RANGES:
            raise ValueError(f"time_range must be one of {', '.join(WEBSEARCH_TIME_RANGES)}")
        return v

    def params_for(self, backend_type: str) -> dict:
        """The parameters a backend of this type reads; the others are dropped."""
        return {f: getattr(self, f) for f in WEBSEARCH_BACKEND_PARAMS.get(backend_type, ())}


class WebSearchBackendCreate(WebSearchParamsMixin):
    """A backend's full definition (create, or full replacement on update)."""
    name: str
    type: str
    base_url: str
    enabled: bool = True
    user_selectable: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v):
        return _clean_name(v)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v):
        if v not in WEBSEARCH_PROVIDERS:
            raise ValueError(f"type must be one of {', '.join(WEBSEARCH_PROVIDERS)}")
        return v

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_url(cls, v):
        v = _check_url(v, "base_url")
        if v is None:
            raise ValueError("base_url is required")
        return v

    @model_validator(mode="after")
    def _drop_other_type_params(self):
        # A type switch in the form leaves the other type's fields filled in; they
        # would never be read, so they are not stored either.
        keep = WEBSEARCH_BACKEND_PARAMS[self.type]
        for f in WEBSEARCH_PARAM_FIELDS:
            if f not in keep:
                setattr(self, f, None)
        return self

    def column_values(self) -> dict:
        return self.model_dump()


class WebSearchBackendUpdate(WebSearchBackendCreate):
    pass


class WebSearchBackendResponse(BaseModel):
    id: int
    name: str
    type: str
    base_url: str
    enabled: bool = True
    user_selectable: bool = True
    engines: Optional[str] = None
    categories: Optional[str] = None
    language: Optional[str] = None
    fourget_scraper: Optional[str] = None
    fourget_lang: Optional[str] = None
    fourget_country: Optional[str] = None
    safesearch: Optional[int] = None
    time_range: Optional[str] = None
    is_default: bool = False
    user_count: int = 0                   # users who picked it
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class WebSearchSettingsResponse(BaseModel):
    enabled: bool = False
    default_backend_id: Optional[int] = None
    default_backend: Optional[dict] = None  # {id, name, type}
    max_results: int = 5
    max_snippet_chars: int = 500
    timeout_seconds: int = 10
    max_agentic_loops: int = 3
    max_queries_per_turn: int = 5
    apply_to: List[str] = Field(default_factory=lambda: list(WEBSEARCH_API_SURFACES))
    enabled_providers: List[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class WebSearchSettingsUpdate(BaseModel):
    """Full replacement of the web search settings (backends are edited separately).

    Unknown fields are refused: a pre-multiple-backends payload (provider,
    searxng_base_url...) would otherwise pass and clear the default backend.
    """
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    default_backend_id: Optional[int] = None
    max_results: int = Field(default=5, ge=WEBSEARCH_LIMITS["max_results"][0], le=WEBSEARCH_LIMITS["max_results"][1])
    max_snippet_chars: int = Field(
        default=500, ge=WEBSEARCH_LIMITS["max_snippet_chars"][0], le=WEBSEARCH_LIMITS["max_snippet_chars"][1],
    )
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    max_agentic_loops: int = Field(default=3, ge=1, le=10)
    max_queries_per_turn: int = Field(default=5, ge=1, le=10)
    apply_to: List[str] = Field(default_factory=lambda: list(WEBSEARCH_API_SURFACES))
    enabled_providers: List[str] = Field(default_factory=list)

    @field_validator("apply_to")
    @classmethod
    def _validate_apply_to(cls, v):
        unknown = [s for s in v if s not in WEBSEARCH_API_SURFACES]
        if unknown:
            raise ValueError(f"unknown API surface(s): {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @field_validator("enabled_providers")
    @classmethod
    def _validate_providers(cls, v):
        return list(dict.fromkeys(p.strip() for p in v if isinstance(p, str) and p.strip()))

    @model_validator(mode="after")
    def _require_backend_when_enabled(self):
        if self.enabled and self.default_backend_id is None:
            raise ValueError("default_backend_id is required when web search is enabled")
        return self


class UserWebSearchUpdate(WebSearchParamsMixin):
    """A user's full web search choices; None inherits the admin default.

    The parameters are for `backend_id`'s backend, or the default backend when
    it is None. Unknown fields are refused, so a user cannot reach the admin-only
    knobs (enabled, timeout, loops...) by sending them.
    """
    model_config = ConfigDict(extra="forbid")

    backend_id: Optional[int] = None
    max_results: Optional[int] = Field(
        default=None, ge=WEBSEARCH_LIMITS["max_results"][0], le=WEBSEARCH_LIMITS["max_results"][1],
    )
    max_snippet_chars: Optional[int] = Field(
        default=None, ge=WEBSEARCH_LIMITS["max_snippet_chars"][0], le=WEBSEARCH_LIMITS["max_snippet_chars"][1],
    )


class WebSearchEnginesRequest(BaseModel):
    # An unsaved URL (the backend form), else a stored SearXNG backend, else the
    # default backend (or the first SearXNG one when the default is 4get).
    # searxng_base_url is the pre-multiple-backends name.
    base_url: Optional[str] = Field(default=None, validation_alias=AliasChoices("base_url", "searxng_base_url"))
    backend_id: Optional[int] = None

    @field_validator("base_url", mode="before")
    @classmethod
    def _validate_url(cls, v):
        return _check_url(v, "base_url")


class WebSearchTestRequest(BaseModel):
    query: str = Field(default="web search", min_length=1, max_length=500)
    # What to test: an unsaved backend (the backend form), else a stored one, else
    # the default backend.
    backend: Optional[WebSearchBackendCreate] = None
    backend_id: Optional[int] = None
    # Optional unsaved settings (limits, timeout); omitted -> stored settings.
    settings: Optional[WebSearchSettingsUpdate] = None


class AccountDelete(BaseModel):
    confirmation: str


# ---------------------------------------------------------------------------
# Model-group Pydantic models
# ---------------------------------------------------------------------------

def _clean_keys(values: List[str], what: str, max_len: int) -> List[str]:
    seen = set()
    clean = []
    for v in values:
        v = v.strip()
        if not v or len(v) > max_len:
            raise ValueError(f"{what} '{v}' is invalid (must be 1–{max_len} chars)")
        if v not in seen:
            seen.add(v)
            clean.append(v)
    return clean


class ModelGroupMembersUpdate(BaseModel):
    """A group's full member list: explicit models and whole provider instances."""
    model_ids: List[str] = []
    provider_keys: List[str] = []

    @model_validator(mode="after")
    def validate_members(self):
        self.model_ids = _clean_keys(self.model_ids, "model_id", 200)
        self.provider_keys = _clean_keys(self.provider_keys, "provider_key", 100)
        return self


class ModelGroupCreate(ModelGroupMembersUpdate):
    # No limit fields: a new group starts with the strictest limits its members had
    # (database.create_model_group), so moving models in never loosens them.
    name: str
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_fields(self):
        self.name = _clean_name(self.name)
        if self.description is not None:
            self.description = self.description.strip()[:256]
        return self


class ModelGroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    # Both lists together replace the members, in the same commit as the fields; leave
    # both out to keep the members as they are.
    model_ids: Optional[List[str]] = None
    provider_keys: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_fields(self):
        if self.name is not None:
            self.name = _clean_name(self.name)
        if self.description is not None:
            self.description = self.description.strip()[:256]
        if self.model_ids is not None:
            self.model_ids = _clean_keys(self.model_ids, "model_id", 200)
        if self.provider_keys is not None:
            self.provider_keys = _clean_keys(self.provider_keys, "provider_key", 100)
        return self

    def members(self) -> Optional["ModelGroupMembersUpdate"]:
        if self.model_ids is None and self.provider_keys is None:
            return None
        return ModelGroupMembersUpdate(model_ids=self.model_ids or [], provider_keys=self.provider_keys or [])


class ModelGroupMemberResponse(BaseModel):
    kind: str                      # model | instance
    key: str                       # model_id or provider_key
    # ok | model_disabled | provider_disabled | missing (deleted or renamed upstream)
    status: str = "ok"
    covers: List[str] = []         # instance members: the models it currently brings in
    overrides_group: Optional[str] = None  # model members: the group whose instance it is carved out of


class ModelGroupResponse(BaseModel):
    # Limits live on profiles: see ProfileGroupLimitResponse.
    id: int                        # 0 = the built-in Default group
    name: str
    description: Optional[str] = None
    is_default: bool = False
    enabled: bool = True           # off = the group's models are not served
    member_count: int = 0          # explicit members (models + instances)
    model_ids: List[str] = []      # explicit model members
    provider_keys: List[str] = []  # instance members
    members: List[ModelGroupMemberResponse] = []
    effective_models: List[str] = []  # every model the group governs right now
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


# ---------------------------------------------------------------------------
# User-facing quota response models (GET /auth/quotas)
# ---------------------------------------------------------------------------

class QuotaSectionResponse(BaseModel):
    """A single unified quota section for the user dashboard.

    Named model groups and the built-in Default group are represented the same way.
    `models` lists the model ids governed by this section, resolved the way
    enforcement resolves them (rate_limit_tracker.group_for).
    """
    name: str
    description: Optional[str] = None
    rpm_limit: Optional[int] = None      # effective; null = unlimited
    rpd_limit: Optional[int] = None      # effective; null = unlimited
    rpd_count: int = 0                   # requests used today (local day) in this section
    rpd_remaining: Optional[int] = None  # null = unlimited
    models: List[str] = []               # model ids listed under this section
    is_default: bool = False             # true for the built-in Default group
    # Pooling. rpm_limit above always stays the caller's OWN per-minute limit; only the
    # daily numbers are shared, so the UI can say "shared across N members" against the
    # daily figures without implying the per-minute one moved.
    is_pooled: bool = False              # true when rpd_limit/rpd_count are pool-wide
    pooled_from_members: int = 0         # how many members contribute; 0 when unpooled
    carry_adjustment: int = 0            # this user's settlement adjustment inside rpd_count;
                                         # explains a count that doesn't match their own history
    # Per-member share cap inside a pool: the most of the pooled daily limit this user
    # may consume. null when unpooled or uncapped. rpd_remaining already accounts for it.
    share_cap: Optional[int] = None
    share_used: Optional[int] = None


class PoolSummary(BaseModel):
    """Identity of the pool a user belongs to, for headers and chips."""
    id: int
    name: str
    member_count: int = 0

    class Config:
        from_attributes = True


class PoolCreate(BaseModel):
    """Create a pool. The creator becomes its owner and first member."""
    name: str
    description: Optional[str] = None


class PoolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class PoolInviteCreate(BaseModel):
    username: str


class PoolMemberScope(BaseModel):
    """One member's position on one quota tier.

    `sent` is what they actually consumed -- the number every usage view reports.
    `charged` is what the pool has held them responsible for, and it is what governs
    their quota. The two differ whenever the pool's composition changed today, which is
    the whole reason both are surfaced rather than just one.
    """
    scope_kind: str                       # overall (Default group) | model_group
    scope_id: int = 0
    name: str = "Default"
    limit: Optional[int] = None           # this member's own RPD; null means unlimited
    sent: int = 0                         # raw request_usage rows today
    charged: int = 0                      # settled share of the pool's consumption
    carry: int = 0                        # charged - sent; the adjustment the limiter applies
    net_contribution: int = 0             # limit - charged; headroom this member still brings
    # The per-member share cap: the most of the pool this member may consume on this
    # tier today. null means uncapped. share_used is max(0, sent + carry) -- the number
    # the cap is enforced against.
    share_cap: Optional[int] = None
    share_used: int = 0


class PoolMemberResponse(BaseModel):
    user_id: int
    username: str
    is_owner: bool = False
    is_active: bool = True
    joined_at: Optional[datetime] = None
    scopes: List[PoolMemberScope] = []

    class Config:
        from_attributes = True


class PoolScopeResponse(BaseModel):
    """The pool's own numbers on one tier."""
    scope_kind: str
    scope_id: int = 0
    name: str = "Overall"
    limit: Optional[int] = None           # sum of member limits; null when any is unlimited
    used: int = 0
    remaining: Optional[int] = None
    is_unlimited: bool = False


class PoolInvitationResponse(BaseModel):
    id: int
    pool_id: int
    pool_name: str
    inviter_username: str
    invitee_username: str
    status: str
    created_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    pool_member_count: int = 0
    pool_limit: Optional[int] = None      # overall tier, for the "you won't be charged" copy
    pool_used: int = 0
    counterparty_limit: Optional[int] = None
    counterparty_used: int = 0

    class Config:
        from_attributes = True


class PoolLeavePreview(BaseModel):
    """What leaving right now would cost the caller, per tier.

    Computed by running settlement and rolling it back, never derived in the frontend --
    a leave dialog that disagrees with what actually happens is worse than no dialog.
    """
    scope_kind: str
    scope_id: int = 0
    name: str = "Overall"
    limit: Optional[int] = None
    sent: int = 0
    charged: int = 0
    remaining: Optional[int] = None


class MyPoolResponse(BaseModel):
    pool: Optional[PoolSummary] = None
    description: Optional[str] = None
    owner_username: Optional[str] = None
    is_owner: bool = False
    members: List[PoolMemberResponse] = []
    scopes: List[PoolScopeResponse] = []
    if_i_leave_now: List[PoolLeavePreview] = []
    incoming_invites: List[PoolInvitationResponse] = []
    outgoing_invites: List[PoolInvitationResponse] = []
    max_members: int = MAX_POOL_MEMBERS          # effective for this pool
    max_member_share_pct: Optional[int] = None   # effective for this pool; null = uncapped


class AdminPoolResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    owner_username: Optional[str] = None
    created_at: Optional[datetime] = None
    members: List[PoolMemberResponse] = []
    scopes: List[PoolScopeResponse] = []
    member_count: int = 0
    # The pool's own overrides (null = inherits) and what actually applies.
    max_members_override: Optional[int] = None
    max_member_share_pct_override: Optional[int] = None
    effective_max_members: int = MAX_POOL_MEMBERS
    effective_max_member_share_pct: Optional[int] = None

    class Config:
        from_attributes = True


def _check_pool_max_members(value: Optional[int]) -> Optional[int]:
    if value is not None and not (POOL_MEMBERS_MIN <= value <= POOL_MEMBERS_HARD_CAP):
        raise ValueError(
            f"max_members must be between {POOL_MEMBERS_MIN} and {POOL_MEMBERS_HARD_CAP}"
        )
    return value


def _check_pool_share_pct(value: Optional[int]) -> Optional[int]:
    if value is not None and not (0 <= value <= 100):
        raise ValueError("max_member_share_pct must be between 0 and 100")
    return value


class PoolSettingsResponse(BaseModel):
    max_members: int = MAX_POOL_MEMBERS
    max_member_share_pct: Optional[int] = None
    hard_cap: int = POOL_MEMBERS_HARD_CAP
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True


class PoolSettingsUpdate(BaseModel):
    """Global pool policy. max_member_share_pct null removes the per-member cap."""
    max_members: int
    max_member_share_pct: Optional[int] = None

    @field_validator("max_members")
    @classmethod
    def _validate_max_members(cls, v):
        return _check_pool_max_members(v)

    @field_validator("max_member_share_pct")
    @classmethod
    def _validate_share_pct(cls, v):
        return _check_pool_share_pct(v)


class AdminPoolLimitsUpdate(BaseModel):
    """Per-pool overrides. An omitted field is left alone; an explicit null clears the
    override so the pool inherits the global value again (see model_fields_set)."""
    max_members: Optional[int] = None
    max_member_share_pct: Optional[int] = None

    @field_validator("max_members")
    @classmethod
    def _validate_max_members(cls, v):
        return _check_pool_max_members(v)

    @field_validator("max_member_share_pct")
    @classmethod
    def _validate_share_pct(cls, v):
        return _check_pool_share_pct(v)


class MyQuotasResponse(BaseModel):
    is_admin: bool = False                     # true ⇒ exempt from all limits
    sections: List[QuotaSectionResponse] = []  # unified list; empty when is_admin
    pool: Optional[PoolSummary] = None         # null when the user is not in a pool
    profile_name: Optional[str] = None         # the profile the limits come from
    max_concurrent_requests: Optional[int] = None  # requests in progress at once; null = unlimited
    in_flight: int = 0                         # the caller's requests in progress right now


class ModelAliasUpsert(BaseModel):
    # When set, update this row in place (allows renaming the alias text).
    id: Optional[int] = None
    alias: str
    target_model_id: str
    enabled: bool = True
    apis: List[str] = Field(default_factory=lambda: list(MODEL_ALIAS_API_SURFACES))
    # None keeps an existing row's type ("exact" for a new row). The endpoint
    # re-validates against that effective type, since it is unknown here.
    match_type: Optional[Literal[MODEL_ALIAS_MATCH_TYPES]] = None
    # The profile this mapping belongs to.
    profile_id: int = DEFAULT_PROFILE_ID
    # The list it belongs to. None keeps an existing row's list ("alias" for a new row);
    # with `id`, a different kind moves the row to that list.
    kind: Optional[Literal[MODEL_ALIAS_KINDS]] = None

    @model_validator(mode="after")
    def validate_fields(self):
        self.alias = self.alias.strip()
        self.target_model_id = self.target_model_id.strip()
        if not self.alias or len(self.alias) > 200:
            raise ValueError("alias must be 1–200 characters")
        if not self.target_model_id or len(self.target_model_id) > 200:
            raise ValueError("target_model_id must be 1–200 characters")
        if self.match_type is not None:
            validate_alias_pattern(self.alias, self.target_model_id, self.match_type)
        # Normalise APIs: strip, dedupe, reject unknown, preserve canonical order.
        seen = {a.strip() for a in self.apis}
        unknown = seen - set(MODEL_ALIAS_API_SURFACES)
        if unknown:
            raise ValueError(f"unknown API surface(s): {', '.join(sorted(unknown))}")
        self.apis = [a for a in MODEL_ALIAS_API_SURFACES if a in seen]
        if not self.apis:
            raise ValueError("at least one API surface must be selected")
        return self


class ModelAliasResponse(BaseModel):
    id: int
    alias: str
    target_model_id: str
    enabled: bool
    apis: List[str]
    match_type: str
    priority: int
    profile_id: int = DEFAULT_PROFILE_ID
    kind: str = "alias"
    # For exact "alias" rows only: whether the alias names a model the profile's
    # members can use. False means the rule never applies (it belongs in the
    # fallback list); None means not computed.
    source_available: Optional[bool] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("apis", mode="before")
    @classmethod
    def _decode_apis(cls, value):
        # from_attributes hands through the raw Text column (a JSON string) or
        # None; NULL/empty/malformed all mean "all surfaces".
        if isinstance(value, list):
            return value
        if not value:
            return list(MODEL_ALIAS_API_SURFACES)
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return list(MODEL_ALIAS_API_SURFACES)
        if not isinstance(decoded, list) or not decoded:
            return list(MODEL_ALIAS_API_SURFACES)
        return [a for a in MODEL_ALIAS_API_SURFACES if a in set(decoded)] or list(MODEL_ALIAS_API_SURFACES)


class ModelAliasReorder(BaseModel):
    """Every alias id of one list of a profile, in the desired evaluation order (highest priority first)."""
    ids: List[int]
    profile_id: int = DEFAULT_PROFILE_ID
    kind: Literal[MODEL_ALIAS_KINDS] = "alias"

    @model_validator(mode="after")
    def validate_ids(self):
        if not self.ids:
            raise ValueError("ids must not be empty")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must not contain duplicates")
        return self


class ModelAliasTestMatch(BaseModel):
    """Which mapping (if any) a model name hits on one API surface."""
    api: str
    # Whether the name is available to the profile's members; picks the list below.
    available: bool = True
    # The list that was tried: "alias" when available, "fallback" otherwise.
    kind: str = "alias"
    matched: bool
    resolved_model: str
    id: Optional[int] = None
    alias: Optional[str] = None
    match_type: Optional[str] = None
    target_model_id: Optional[str] = None


# Model Management Pydantic models (updated to use ProviderCredentials)
class ProviderConfigurationResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider_key: str
    provider_type: str
    instance_name: str
    provider_name: Optional[str] = None  # Only for custom providers
    enabled: bool  # Changed from is_enabled to enabled for consistency
    model_count: int
    enabled_model_count: int
    supported_apis: Optional[List[str]] = None
    # Outcome of the last model sync (in-memory; None until the provider is first synced)
    sync_state: Optional[str] = None  # "ok" | "failed"
    sync_error: Optional[str] = None
    last_sync_at: Optional[float] = None  # epoch seconds, so the browser renders it in local time
    created_at: datetime
    updated_at: datetime


class ModelConfigurationResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    model_id: str
    provider_key: str
    model_name: str
    is_enabled: bool
    created_at: datetime
    updated_at: datetime


class ModelManagementTree(BaseModel):
    providers: List[ProviderConfigurationResponse]
    total_models: int
    enabled_models: int


class ToggleRequest(BaseModel):
    enabled: bool


class BulkToggleRequest(BaseModel):
    action: str  # "enable_all" or "disable_all"


class BulkUserActionRequest(BaseModel):
    user_ids: List[int]
    action: str  # "approve", "deactivate", "activate", "delete" or "assign_profile"
    profile_id: Optional[int] = None  # required for "assign_profile"


class ModelSearchResponse(BaseModel):
    models: List[ModelConfigurationResponse]
    providers: List[ProviderConfigurationResponse]
    total_results: int


# Provider Credentials Pydantic models
class AzureDeploymentGroup(BaseModel):
    """One row of Azure deployments and the upstream APIs they support.

    A ``legacy_mode`` row comes from the per-bucket lists that predate
    deployment groups: it classifies its deployments without selecting APIs,
    so ``apis`` stays empty and nothing is restricted.
    """
    deployments: List[str]
    apis: List[str] = []
    legacy_mode: Optional[Literal["openai", "responses", "anthropic"]] = None

    @field_validator('deployments')
    @classmethod
    def validate_names(cls, v):
        for name in v:
            stripped = name.strip()
            if stripped and not DEPLOYMENT_NAME_PATTERN.match(stripped):
                raise ValueError(
                    f"Invalid deployment name '{stripped}'. "
                    "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
                )
        return v

    @field_validator('apis')
    @classmethod
    def validate_apis(cls, v):
        from app.providers.azure_deployments import DEPLOYMENT_APIS
        unknown = [a for a in v if a not in DEPLOYMENT_APIS]
        if unknown:
            raise ValueError(f"Unknown API(s) {', '.join(unknown)}; must be one of {', '.join(DEPLOYMENT_APIS)}")
        return v

    @model_validator(mode='after')
    def validate_row(self):
        if self.legacy_mode is not None and self.apis:
            raise ValueError(
                f"A legacy deployment row cannot also select APIs: {', '.join(self.deployments)}"
            )
        if any(d.strip() for d in self.deployments) and not self.apis and self.legacy_mode is None:
            raise ValueError(f"Select at least one API for deployments: {', '.join(self.deployments)}")
        return self


_DEPLOYMENT_LIST_FIELDS = ("deployments", "openai_deployments", "anthropic_deployments")


def _drop_shadowed_deployment_lists(data):
    """Ignore the per-bucket lists when ``deployment_groups`` is given.

    The groups (even ``[]``) are then the only deployment source, so stale
    lists sent alongside them must neither fail validation nor survive.
    """
    if isinstance(data, dict) and data.get("deployment_groups") is not None:
        data = {k: v for k, v in data.items() if k not in _DEPLOYMENT_LIST_FIELDS}
    return data


def _apply_deployment_groups(model) -> None:
    """Fill all three deployment lists from ``deployment_groups``."""
    if model.deployment_groups is None:
        return
    from app.providers.azure_deployments import (
        merge_azure_deployments,
        normalize_azure_deployments,
        normalize_deployment_groups,
    )
    rows = normalize_deployment_groups(model.deployment_groups, strict=True)
    derived = normalize_azure_deployments({"groups": rows})
    model.openai_deployments = derived["openai"]
    model.anthropic_deployments = derived["anthropic"]
    model.deployments = merge_azure_deployments(derived)


class ProviderCredentialsCreate(BaseModel):
    provider_type: str
    instance_name: str
    provider_name: str  # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")
    enabled: bool = True
    
    # Provider-specific fields (all optional)
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Will be converted to JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    # Azure: rows of deployments + the upstream APIs they support. When given,
    # openai_deployments / anthropic_deployments are derived from it.
    deployment_groups: Optional[List["AzureDeploymentGroup"]] = None
    dynamic_discovery: Optional[bool] = None

    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]
    anthropic_base_url: Optional[str] = None  # custom providers: native Anthropic base URL override
    anthropic_auth_style: Optional[str] = None  # "x-api-key" (default) or "bearer"

    @field_validator('anthropic_auth_style')
    @classmethod
    def validate_anthropic_auth_style(cls, v):
        if v in (None, ""):
            return v
        v = v.strip().lower()
        if v not in ANTHROPIC_AUTH_STYLES:
            raise ValueError(f"anthropic_auth_style must be one of {', '.join(ANTHROPIC_AUTH_STYLES)}")
        return v

    @field_validator('anthropic_base_url')
    @classmethod
    def validate_anthropic_base_url(cls, v):
        if v in (None, ""):
            return v
        from urllib.parse import urlparse
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("anthropic_base_url must be an http(s) URL with a host")
        return v

    @field_validator('azure_backend')
    @classmethod
    def validate_azure_backend(cls, v):
        if v is not None and v not in VALID_AZURE_BACKENDS:
            raise ValueError(f"azure_backend must be one of {VALID_AZURE_BACKENDS}, got '{v}'")
        return v

    @field_validator('openai_deployments', 'anthropic_deployments')
    @classmethod
    def validate_deployment_names(cls, v):
        if v is None:
            return v
        for name in v:
            stripped = name.strip()
            if stripped and not DEPLOYMENT_NAME_PATTERN.match(stripped):
                raise ValueError(
                    f"Invalid deployment name '{stripped}'. "
                    "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
                )
        return v

    @model_validator(mode='before')
    @classmethod
    def drop_shadowed_deployment_lists(cls, data):
        return _drop_shadowed_deployment_lists(data)

    @model_validator(mode='after')
    def validate_discovery_deployments(self):
        _apply_deployment_groups(self)
        if self.dynamic_discovery is True:
            has_deployments = (
                (self.deployments and len(self.deployments) > 0)
                or (self.openai_deployments and len(self.openai_deployments) > 0)
                or (self.anthropic_deployments and len(self.anthropic_deployments) > 0)
            )
            if has_deployments:
                raise ValueError(
                    "Cannot specify deployments when dynamic_discovery is enabled. "
                    "Either disable dynamic_discovery or remove deployment lists."
                )
            # Azure dynamic discovery uses GET {endpoint}/openai/models?api-version=...
            # for both backends, so an api-version is always required.
            if self.provider_type == "azure" and not self.discovery_api_version:
                raise ValueError(
                    "discovery_api_version is required when dynamic_discovery is enabled "
                    "for an Azure provider."
                )
        return self


class ProviderCredentialsUpdate(BaseModel):
    instance_name: Optional[str] = None
    enabled: Optional[bool] = None
    
    # Provider-specific fields (all optional)
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Will be converted to JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    # Azure: rows of deployments + the upstream APIs they support. When given,
    # openai_deployments / anthropic_deployments are derived from it.
    deployment_groups: Optional[List["AzureDeploymentGroup"]] = None
    provider_name: Optional[str] = None  # OpenAI-compatible provider name (e.g., "ollama", "llamacpp", "openai")
    dynamic_discovery: Optional[bool] = None

    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]
    anthropic_base_url: Optional[str] = None  # custom providers: native Anthropic base URL override
    anthropic_auth_style: Optional[str] = None  # "x-api-key" (default) or "bearer"

    @field_validator('anthropic_auth_style')
    @classmethod
    def validate_anthropic_auth_style(cls, v):
        if v in (None, ""):
            return v
        v = v.strip().lower()
        if v not in ANTHROPIC_AUTH_STYLES:
            raise ValueError(f"anthropic_auth_style must be one of {', '.join(ANTHROPIC_AUTH_STYLES)}")
        return v

    @field_validator('anthropic_base_url')
    @classmethod
    def validate_anthropic_base_url(cls, v):
        if v in (None, ""):
            return v
        from urllib.parse import urlparse
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("anthropic_base_url must be an http(s) URL with a host")
        return v

    @field_validator('azure_backend')
    @classmethod
    def validate_azure_backend(cls, v):
        if v is not None and v not in VALID_AZURE_BACKENDS:
            raise ValueError(f"azure_backend must be one of {VALID_AZURE_BACKENDS}, got '{v}'")
        return v

    @field_validator('openai_deployments', 'anthropic_deployments')
    @classmethod
    def validate_deployment_names(cls, v):
        if v is None:
            return v
        for name in v:
            stripped = name.strip()
            if stripped and not DEPLOYMENT_NAME_PATTERN.match(stripped):
                raise ValueError(
                    f"Invalid deployment name '{stripped}'. "
                    "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
                )
        return v

    @model_validator(mode='before')
    @classmethod
    def drop_shadowed_deployment_lists(cls, data):
        return _drop_shadowed_deployment_lists(data)

    @model_validator(mode='after')
    def validate_discovery_deployments(self):
        _apply_deployment_groups(self)
        if self.dynamic_discovery is True:
            has_deployments = (
                (self.deployments and len(self.deployments) > 0)
                or (self.openai_deployments and len(self.openai_deployments) > 0)
                or (self.anthropic_deployments and len(self.anthropic_deployments) > 0)
            )
            if has_deployments:
                raise ValueError(
                    "Cannot specify deployments when dynamic_discovery is enabled. "
                    "Either disable dynamic_discovery or remove deployment lists."
                )
            # Update payloads may omit discovery_api_version when it already exists
            # in the DB, so only reject if it is explicitly set to empty/None here.
            if self.discovery_api_version is not None and not self.discovery_api_version:
                raise ValueError(
                    "discovery_api_version cannot be empty when dynamic_discovery is enabled."
                )
        return self


class ProviderCredentialsResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider_key: str
    provider_type: str
    instance_name: str
    enabled: bool
    
    # Provider-specific fields
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Parsed from JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    # Azure: rows of deployments + the upstream APIs they support. When given,
    # openai_deployments / anthropic_deployments are derived from it.
    deployment_groups: Optional[List["AzureDeploymentGroup"]] = None
    provider_name: str  # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")
    dynamic_discovery: Optional[bool] = None
    
    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]
    anthropic_base_url: Optional[str] = None  # custom providers: native Anthropic base URL override
    anthropic_auth_style: Optional[str] = None  # "x-api-key" (default) or "bearer"
    
    created_at: datetime
    updated_at: datetime


# OAuth models
class OAuthUserCreate(BaseModel):
    provider: str
    provider_user_id: str
    email: str
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture: Optional[str] = None
    raw_data: Optional[str] = None


class OAuthUserResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider: str
    provider_user_id: str
    email: str
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ZohoOAuthCallback(BaseModel):
    code: str
    state: Optional[str] = None
    location: Optional[str] = None

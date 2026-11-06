"""BYOK (bring your own key): provider instances owned by one user.

A BYOK instance is an ordinary ``provider_credentials`` row whose
``owner_user_id`` is set. It is reached only by its owner: admins, other users,
allow-mode profiles and identity-less principals can never resolve, list,
dispatch to or probe it. It is exempt from the admin's quotas but its traffic is
still recorded in the usage tables like any other request.

Naming. The instance name is ``byok-{user_id}-{slug}``, so the provider key is
``{provider_name}:byok-{user_id}-{slug}`` and model ids are
``{provider_name}:byok-{user_id}-{slug}/{model}``. The ``byok-`` prefix is
reserved: admin instances may not use it, which is what lets usage counting
(SQL, where no owner is at hand) recognise a BYOK row by its name alone.

Authority. Access decisions never trust the name: they read the owner map, built
from the ``owner_user_id`` column by the provider manager. A key that looks like
BYOK but is not in the map (an orphan, a row loaded mid-refresh) is denied to
everyone -- fail closed.

This module imports nothing from the app so the cache, the access checks, the
rate limiter and the usage tracker can all use it without import cycles.
"""

import re
from typing import Any, Dict, Optional

BYOK_INSTANCE_PREFIX = "byok-"

# Provider types a user may add (the same set the admin form offers).
BYOK_PROVIDER_TYPES = ("azure", "google", "bedrock", "platformai", "custom")
# Types whose provider_name is fixed to the type itself.
SPECIALIZED_PROVIDER_TYPES = frozenset({"azure", "google", "bedrock", "platformai"})

MAX_BYOK_INSTANCES_PER_USER = 10

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
# provider_name for custom providers: no ':' or '/', which the key parsers split on.
PROVIDER_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
BEDROCK_REGION_PATTERN = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")

# Sentinel owner for a key that looks like BYOK but has no known owner: deny.
UNKNOWN_OWNER = -1


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def instance_part(provider_key: Optional[str]) -> str:
    """The instance name inside a provider key ('openai:byok-4-x' -> 'byok-4-x')."""
    if not provider_key or ":" not in provider_key:
        return ""
    return provider_key.split(":", 1)[1]


def is_byok_instance_name(instance_name: Optional[str]) -> bool:
    return bool(instance_name) and instance_name.startswith(BYOK_INSTANCE_PREFIX)


def is_byok_provider_key(provider_key: Optional[str]) -> bool:
    """Whether a provider key names a BYOK instance (by the reserved prefix)."""
    return is_byok_instance_name(instance_part(provider_key))


def provider_key_of_model(model_id: Optional[str]) -> Optional[str]:
    """Same parse as app.rate_limit.provider_key_of: everything before the first '/'."""
    if not model_id:
        return None
    return model_id.split("/", 1)[0]


def is_byok_model(model_id: Optional[str]) -> bool:
    """Whether a canonical model id belongs to a BYOK instance (by name).

    Mirrored exactly in SQL by app.request_tracker._byok_match.
    """
    return is_byok_provider_key(provider_key_of_model(model_id))


def byok_instance_name(user_id: int, slug: str) -> str:
    return f"{BYOK_INSTANCE_PREFIX}{int(user_id)}-{slug}"


def slug_of(instance_name: str, user_id: int) -> str:
    """Inverse of byok_instance_name, for display."""
    prefix = f"{BYOK_INSTANCE_PREFIX}{int(user_id)}-"
    return instance_name[len(prefix):] if instance_name.startswith(prefix) else instance_name


# ---------------------------------------------------------------------------
# Validation (raise ValueError with a user-facing message)
# ---------------------------------------------------------------------------

def validate_slug(slug: Optional[str]) -> str:
    slug = (slug or "").strip().lower()
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            "Name must be 1-40 characters: lowercase letters, digits and hyphens, "
            "starting with a letter or digit."
        )
    return slug


def validate_provider_name(provider_name: Optional[str]) -> str:
    """A provider name safe to embed in a provider key (no ':' or '/')."""
    name = (provider_name or "").strip().lower()
    if not PROVIDER_NAME_PATTERN.match(name):
        raise ValueError(
            "Provider name must be 1-40 characters: lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit."
        )
    return name


def validate_provider_type(provider_type: Optional[str]) -> str:
    if provider_type not in BYOK_PROVIDER_TYPES:
        raise ValueError(f"provider_type must be one of {', '.join(BYOK_PROVIDER_TYPES)}")
    return provider_type


def validate_bedrock_region(region: Optional[str]) -> Optional[str]:
    if region in (None, ""):
        return region
    region = region.strip()
    if not BEDROCK_REGION_PATTERN.match(region):
        raise ValueError("region must look like an AWS region, e.g. us-west-2")
    return region


# ---------------------------------------------------------------------------
# Owner registry
# ---------------------------------------------------------------------------

class ByokRegistry:
    """provider_key -> owner user id, for every BYOK row (enabled or not).

    Swapped whole on a full reload and edited per key on single-instance changes.
    Every method is synchronous and await-free, so it is atomic on the event loop.
    """

    def __init__(self) -> None:
        self._owners: Dict[str, int] = {}

    # -- writers ------------------------------------------------------------
    def replace(self, owners: Dict[str, int]) -> None:
        self._owners = dict(owners)

    def set(self, provider_key: str, owner_user_id: int) -> None:
        owners = dict(self._owners)
        owners[provider_key] = int(owner_user_id)
        self._owners = owners

    def discard(self, provider_key: str) -> None:
        if provider_key in self._owners:
            owners = dict(self._owners)
            owners.pop(provider_key, None)
            self._owners = owners

    # -- readers ------------------------------------------------------------
    def owner_of(self, provider_key: Optional[str]) -> Optional[int]:
        """The owner's id; UNKNOWN_OWNER for a BYOK-looking key with no owner; None if global."""
        if not provider_key:
            return None
        owner = self._owners.get(provider_key)
        if owner is not None:
            return owner
        if is_byok_provider_key(provider_key):
            return UNKNOWN_OWNER
        return None

    def owner_of_model(self, model_id: Optional[str]) -> Optional[int]:
        return self.owner_of(provider_key_of_model(model_id))

    def is_byok(self, provider_key: Optional[str]) -> bool:
        return self.owner_of(provider_key) is not None

    def is_byok_model(self, model_id: Optional[str]) -> bool:
        return self.owner_of_model(model_id) is not None

    def keys_for(self, user_id: int) -> list:
        return sorted(k for k, uid in self._owners.items() if uid == user_id)

    def all_keys(self) -> set:
        return set(self._owners)


byok_registry = ByokRegistry()


def caller_user_id(auth: Any) -> Optional[int]:
    """The user id behind an auth result, or None for the admin / identity-less callers.

    Mirrors the attribution in app.auth.middleware._resolve_identity: an API key's
    owner (user_id), else a user's own id. The config admin (AdminUser) has no user
    row and never owns a BYOK instance.
    """
    if auth is None:
        return None
    if getattr(auth, "is_admin", False) is True and getattr(auth, "id", None) is None:
        return None
    uid = getattr(auth, "user_id", None)
    if uid is None:
        uid = getattr(auth, "id", None)
    try:
        return int(uid) if uid is not None else None
    except (TypeError, ValueError):
        return None


def is_owner(auth: Any, owner: Optional[int]) -> bool:
    """True when ``auth`` is the (known) owner of a BYOK instance."""
    if owner is None or owner == UNKNOWN_OWNER:
        return False
    from app.auth.admin import AdminUser  # local: keep this module import-light
    if isinstance(auth, AdminUser):
        return False
    uid = caller_user_id(auth)
    return uid is not None and uid == owner

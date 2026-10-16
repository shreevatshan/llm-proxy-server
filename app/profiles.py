"""In-memory snapshot of user profiles: which profile each user is in, and what it allows.

Rate limiting (app.rate_limit) and model access (app.cache) both read limit values and
access rules from here on every request, so reads never touch the database. The
snapshot is reloaded by RateLimitTracker._load_config -- on its periodic refresh and on
every refresh_now() an admin edit triggers -- and admin endpoints that move users
between profiles also patch it directly so the move applies to the very next request.

A leaf module: it imports models lazily and nothing from cache.py or rate_limit.py,
which both import it.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple

DEFAULT_PROFILE_ID = 1  # mirrors app.auth.models.DEFAULT_PROFILE_ID without importing it
# The built-in Default model group: every model no named group claims. It has no row;
# its limits are the profile's own rpm/rpd columns.
DEFAULT_GROUP_ID = 0

_NO_LIMITS: Tuple[Optional[int], Optional[int]] = (None, None)


@dataclass(frozen=True)
class ProfileSnapshot:
    id: int
    name: str
    rpm: Optional[int] = None    # Default group; None = unlimited
    rpd: Optional[int] = None
    model_group_limits: Mapping[int, Tuple[Optional[int], Optional[int]]] = field(
        default_factory=lambda: MappingProxyType({}))
    access_mode: str = "default"  # default | allow | deny | custom
    access_exceptions: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({}))

    def model_group_limit(self, group_id: int) -> Tuple[Optional[int], Optional[int]]:
        if group_id == DEFAULT_GROUP_ID:
            return (self.rpm, self.rpd)
        return self.model_group_limits.get(group_id, _NO_LIMITS)


# What an empty registry (before the first load, or a DB with no Default row)
# answers: no limits, global model access. Same as a fresh Default profile.
EMPTY_DEFAULT = ProfileSnapshot(DEFAULT_PROFILE_ID, "Default")


class ProfileRegistry:
    def __init__(self) -> None:
        # Replaced as one tuple, never mutated, so a reader never sees a user map
        # from one load paired with profiles from another.
        self._state: Tuple[Dict[int, int], Dict[int, ProfileSnapshot]] = ({}, {})

    # -- reads ---------------------------------------------------------------

    def profile_id_for(self, user_id: Optional[int]) -> int:
        """A user's profile; users not in the snapshot yet are in the default one."""
        if user_id is None:
            return DEFAULT_PROFILE_ID
        return self._state[0].get(user_id, DEFAULT_PROFILE_ID)

    def get(self, profile_id: Optional[int]) -> ProfileSnapshot:
        profiles = self._state[1]
        snap = profiles.get(profile_id) if profile_id is not None else None
        if snap is None:
            snap = profiles.get(DEFAULT_PROFILE_ID, EMPTY_DEFAULT)
        return snap

    def for_user(self, user_id: Optional[int]) -> ProfileSnapshot:
        return self.get(self.profile_id_for(user_id))

    def name_of(self, profile_id: int) -> Optional[str]:
        snap = self._state[1].get(profile_id)
        return snap.name if snap else None

    # -- writes --------------------------------------------------------------

    def install(self, user_to_profile: Mapping[int, int], profiles: Iterable[ProfileSnapshot]) -> None:
        """Replace the whole snapshot (used by load(), and by tests)."""
        self._state = (dict(user_to_profile), {p.id: p for p in profiles})

    def set_users_profile(self, user_ids: Iterable[int], profile_id: int) -> None:
        """Move users right after the DB commit, ahead of the next reload."""
        user_map, profiles = self._state
        user_map = dict(user_map)
        for uid in user_ids:
            user_map[uid] = profile_id
        self._state = (user_map, profiles)

    def reset(self) -> None:
        self._state = ({}, {})

    async def load(self, db) -> None:
        """Re-read every profile and every user's assignment from *db*."""
        from sqlalchemy.future import select
        from app.auth.models import (
            ProfileModelAccessException, ProfileModelGroupLimit, User, UserProfile,
        )

        profile_rows = (await db.execute(select(UserProfile))).scalars().all()
        mg_rows = (await db.execute(select(ProfileModelGroupLimit))).scalars().all()
        ex_rows = (await db.execute(select(ProfileModelAccessException))).scalars().all()
        user_rows = (await db.execute(select(User.id, User.profile_id))).all()

        mg: Dict[int, Dict[int, tuple]] = {}
        for r in mg_rows:
            mg.setdefault(r.profile_id, {})[r.group_id] = (r.rpm_limit, r.rpd_limit)
        ex: Dict[int, Dict[str, bool]] = {}
        for r in ex_rows:
            ex.setdefault(r.profile_id, {})[r.model_id] = bool(r.is_allowed)

        snaps = [
            ProfileSnapshot(
                id=p.id, name=p.name, rpm=p.rpm_limit, rpd=p.rpd_limit,
                model_group_limits=MappingProxyType(mg.get(p.id, {})),
                access_mode=p.model_access_mode or "default",
                access_exceptions=MappingProxyType(ex.get(p.id, {})),
            )
            for p in profile_rows
        ]
        self.install({uid: (pid or DEFAULT_PROFILE_ID) for uid, pid in user_rows}, snaps)


profile_registry = ProfileRegistry()


def profile_id_for_auth(auth) -> Optional[int]:
    """The caller's profile, or None for the config admin (who has none).

    AdminUser is checked first: its id is None, and usage records it under user id 0,
    which must never be looked up as a real user.
    """
    from app.auth.admin import AdminUser
    if auth is None or isinstance(auth, AdminUser):
        return None
    user_id = getattr(auth, "user_id", None) or getattr(auth, "id", None)
    if user_id is None:
        return None
    return profile_registry.profile_id_for(user_id)

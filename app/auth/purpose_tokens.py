"""Short-lived, single-purpose signed tokens (OAuth link, pending withdraw, flash).

These are JWTs signed with the session key but scoped by a ``purpose`` claim
and a matching ``aud`` (``llmproxy:<purpose>``). They never carry ``sub`` or
``is_admin``, and ``verify_token`` rejects any payload with ``aud``/``purpose``,
so a purpose token can never be replayed as a session.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from .auth import SECRET_KEY, ALGORITHM

logger = logging.getLogger(__name__)

PURPOSE_OAUTH_LINK = "oauth_link"
PURPOSE_PENDING_WITHDRAW = "pending_withdraw"
PURPOSE_FLASH = "auth_flash"

# Claims reserved for session tokens; a purpose token must never carry them.
_FORBIDDEN_CLAIMS = ("sub", "is_admin")


def _audience(purpose: str) -> str:
    return f"llmproxy:{purpose}"


def issue_purpose_token(purpose: str, claims: dict, ttl: timedelta) -> str:
    """Sign *claims* as a token valid only for *purpose*, expiring after *ttl*."""
    # Explicit raise (not `assert`) so the guard survives `python -O`.
    for key in _FORBIDDEN_CLAIMS:
        if key in claims:
            raise AssertionError(f"purpose tokens must not carry {key!r}")
    now = datetime.now(timezone.utc)
    payload = dict(claims)
    payload.update({
        "purpose": purpose,
        "aud": _audience(purpose),
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    })
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def read_purpose_token(token: Optional[str], purpose: str) -> Optional[dict]:
    """Return the claims of a valid *purpose* token, or None.

    None covers missing, malformed, tampered, expired, and wrong-purpose tokens.
    """
    if not token or not isinstance(token, str):
        return None
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM], audience=_audience(purpose)
        )
    except JWTError:
        return None
    except Exception:
        logger.exception("Unexpected error decoding purpose token")
        return None
    if payload.get("purpose") != purpose:
        return None
    if any(key in payload for key in _FORBIDDEN_CLAIMS):
        return None
    return payload


def _created_at_str(user) -> Optional[str]:
    created_at = getattr(user, "created_at", None)
    if created_at is None:
        return None
    if hasattr(created_at, "isoformat"):
        return created_at.isoformat()
    return str(created_at)


def user_binding_claims(user) -> dict:
    """Claims binding a token to one specific user row.

    SQLite can reuse ids after a delete, so the id alone is not enough; the
    username and creation timestamp pin the exact account.
    """
    return {
        "uid": user.id,
        "uname": user.username,
        "ucat": _created_at_str(user),
    }


def claims_match_user(claims: dict, user) -> bool:
    """Whether *claims* (from user_binding_claims) still identify *user*."""
    if not claims or user is None:
        return False
    return (
        claims.get("uid") == user.id
        and claims.get("uname") == user.username
        and claims.get("ucat") == _created_at_str(user)
    )

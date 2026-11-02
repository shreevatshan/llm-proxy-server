"""One-time flash messages for the auth pages.

A flash travels in a signed, HttpOnly cookie (a purpose token with a short
TTL) and is rendered once into the shared alert slot, then cleared. Levels and
action buttons are allowlisted; JS-initiated redirects use ``?reason=<code>``
with a fixed message from ``REASONS`` instead of free text in URLs.
"""

from datetime import timedelta
from typing import Iterable, Optional

from .cookies import cookie_secure
from .purpose_tokens import PURPOSE_FLASH, issue_purpose_token, read_purpose_token

FLASH_COOKIE = "auth_flash"
FLASH_TTL = timedelta(seconds=120)
FLASH_COOKIE_PATH = "/"
_MAX_MESSAGE_LEN = 500
_MAX_SOURCE_LEN = 32

LEVELS = {"success", "info", "warning", "danger"}

ACTIONS = {"withdraw_pending": "Withdraw signup & delete my data"}

REASONS = {
    "session_expired": ("warning", "Your session has expired. Please sign in again."),
    "signin_required": ("info", "Please sign in to continue."),
    "session_invalid": ("warning", "Your session is no longer valid. Please sign in again."),
    "admin_required": ("warning", "That page requires the administrator account. Please sign in as admin."),
    "username_changed": ("success", "Username updated. Please sign in with your new username."),
    "password_changed": ("success", "Password updated. Please sign in again."),
    "account_deleted": ("success", "Your account has been deleted."),
}


def _clean_level(level) -> str:
    return level if level in LEVELS else "danger"


def _clean_actions(actions) -> list:
    if not isinstance(actions, (list, tuple, set, frozenset)):
        return []
    seen = []
    for key in actions:
        if key in ACTIONS and key not in seen:
            seen.append(key)
    return seen


def _clean_source(source) -> Optional[str]:
    if not isinstance(source, str) or not source:
        return None
    return source[:_MAX_SOURCE_LEN]


def _build(level, message, actions, source) -> dict:
    return {
        "level": _clean_level(level),
        "message": message,
        "actions": [{"key": key, "label": ACTIONS[key]} for key in _clean_actions(actions)],
        "source": _clean_source(source),
    }


def set_flash(
    response,
    message: str,
    level: str = "danger",
    actions: Iterable[str] = (),
    source: Optional[str] = None,
) -> None:
    """Attach a one-time flash message cookie to *response*."""
    token = issue_purpose_token(
        PURPOSE_FLASH,
        {
            "msg": str(message)[:_MAX_MESSAGE_LEN],
            "lvl": _clean_level(level),
            "act": _clean_actions(list(actions or ())),
            "src": _clean_source(source),
        },
        FLASH_TTL,
    )
    response.set_cookie(
        key=FLASH_COOKIE,
        value=token,
        max_age=int(FLASH_TTL.total_seconds()),
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path=FLASH_COOKIE_PATH,
    )


def pop_flash(request, response) -> Optional[dict]:
    """Read and clear the flash cookie; return the flash dict or None."""
    raw = request.cookies.get(FLASH_COOKIE)
    if raw is None:
        return None
    response.delete_cookie(
        key=FLASH_COOKIE,
        path=FLASH_COOKIE_PATH,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
    )
    claims = read_purpose_token(raw, PURPOSE_FLASH)
    if not claims:
        return None
    message = claims.get("msg")
    if not isinstance(message, str) or not message:
        return None
    return _build(claims.get("lvl"), message[:_MAX_MESSAGE_LEN], claims.get("act"), claims.get("src"))


def flash_from_reason(code: Optional[str]) -> Optional[dict]:
    """Map an allowlisted ``?reason=`` code to a flash dict, or None."""
    if not code or code not in REASONS:
        return None
    level, message = REASONS[code]
    return _build(level, message, (), None)

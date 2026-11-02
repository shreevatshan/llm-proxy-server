"""Cookie helpers for the session cookie and short-lived auth-flow cookies."""

from .auth import ACCESS_TOKEN_EXPIRE_MINUTES
from .csrf import cookie_secure

__all__ = [
    "SESSION_COOKIE", "OAUTH_LINK_COOKIE", "PENDING_WITHDRAW_COOKIE", "TEMP_COOKIE_PATH",
    "SESSION_COOKIE_GRACE_SECONDS", "cookie_secure",
    "set_session_cookie", "clear_session_cookie", "set_temp_cookie", "clear_temp_cookie",
]

SESSION_COOKIE = "access_token"
OAUTH_LINK_COOKIE = "oauth_link"
PENDING_WITHDRAW_COOKIE = "pending_withdraw"
TEMP_COOKIE_PATH = "/auth/"

# The cookie outlives the JWT by this much so an expired token still reaches the
# server and can be reported as "session expired" rather than "not signed in".
# The JWT `exp` remains the real limit.
SESSION_COOKIE_GRACE_SECONDS = 7 * 24 * 3600


def set_session_cookie(response, token: str) -> None:
    """Attach the HttpOnly session cookie carrying *token* to *response*."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60 + SESSION_COOKIE_GRACE_SECONDS,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
    )


def set_temp_cookie(response, name: str, value: str, max_age: int, path: str = TEMP_COOKIE_PATH) -> None:
    """Attach a short-lived HttpOnly cookie scoped to *path*."""
    response.set_cookie(
        key=name,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path=path,
    )


def clear_temp_cookie(response, name: str, path: str = TEMP_COOKIE_PATH) -> None:
    response.delete_cookie(
        key=name,
        path=path,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
    )

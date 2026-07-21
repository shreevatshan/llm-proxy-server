"""Double-submit-cookie CSRF protection for the management app.

State-changing management requests (``/auth/*``, ``/admin/*``, ``/dashboard/*``)
authenticate via the ``access_token`` cookie, which a cross-site form could ride
on. To defend against that we use the stateless double-submit pattern:

* a non-HttpOnly ``csrf_token`` cookie is issued to the browser, and
* the frontend echoes its value in the ``X-CSRF-Token`` header on every unsafe
  request (see ``core/base.js``); the two must match.

An attacker's site cannot read the victim's cookie (same-origin policy) nor set
the matching header on a cross-origin request, so the check holds. Programmatic
clients that authenticate with an ``Authorization: Bearer`` header (or the Azure
``api-key`` header) are exempt — they are not browsers and carry no ambient
cookie to abuse.
"""

import secrets

from starlette.requests import Request

CSRF_COOKIE_NAME = "csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"

# Only these path prefixes are cookie-authenticated management routes that need
# CSRF protection. Everything else on the management app (the mounted /openai,
# /anthropic, /azure-openai provider APIs, /static, health) is out of scope.
CSRF_PROTECTED_PREFIXES = ("/auth/", "/admin/", "/dashboard/")

# Bootstrap endpoints that run before a session (and thus a CSRF cookie) exists.
CSRF_EXEMPT_PATHS = frozenset({
    "/auth/login",
    "/auth/login/form",
    "/auth/signup",
    "/admin/login",
})

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def cookie_secure() -> bool:
    """Whether cookies should carry the Secure flag.

    Defaults to True (production-safe); set ``COOKIE_SECURE=false`` for local
    plaintext-HTTP development. Mirrors the auth-cookie helper.
    """
    import os
    return os.getenv("COOKIE_SECURE", "true").lower() != "false"


def generate_csrf_token() -> str:
    """Return a fresh, unguessable CSRF token."""
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response, token: str) -> None:
    """Attach the (JS-readable) double-submit CSRF cookie to *response*."""
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,  # must be readable by JS to echo into the header
        secure=cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_csrf_cookie(response) -> None:
    response.delete_cookie(key=CSRF_COOKIE_NAME, path="/")


def _is_programmatic(request: Request) -> bool:
    """True when the caller authenticates as a non-browser API client."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return True
    # Azure OpenAI SDK clients authenticate with the api-key header.
    if request.headers.get("api-key"):
        return True
    return False


def csrf_should_protect(request: Request) -> bool:
    """Whether this request must pass the CSRF check."""
    if request.method not in _UNSAFE_METHODS:
        return False
    path = request.url.path
    if not path.startswith(CSRF_PROTECTED_PREFIXES):
        return False
    if path in CSRF_EXEMPT_PATHS:
        return False
    if _is_programmatic(request):
        return False
    return True


def csrf_token_valid(request: Request) -> bool:
    """Constant-time compare of the CSRF cookie and header."""
    cookie = request.cookies.get(CSRF_COOKIE_NAME)
    header = request.headers.get(CSRF_HEADER_NAME)
    if not cookie or not header:
        return False
    # Compare as bytes: compare_digest raises TypeError on non-ASCII str, and
    # both values are attacker-controlled (would turn hostile input into a 500).
    return secrets.compare_digest(cookie.encode("utf-8"), header.encode("utf-8"))


# Browser entry pages that need a CSRF cookie before any /auth|/admin|/dashboard
# request is made.
_CSRF_ISSUE_PAGES = frozenset({"/", "/login", "/signup"})


def csrf_should_issue(request: Request) -> bool:
    """Whether a GET response should (re)issue the CSRF cookie.

    Issued on browser navigations to management pages when the cookie is absent,
    so both fresh and pre-existing sessions get a token without needing to log in
    again. Everything else (provider APIs, static assets, health) gets no cookie.
    """
    if request.method not in ("GET", "HEAD"):
        return False
    if CSRF_COOKIE_NAME in request.cookies:
        return False
    path = request.url.path
    return path in _CSRF_ISSUE_PAGES or path.startswith(CSRF_PROTECTED_PREFIXES)

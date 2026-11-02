"""Redirect unauthenticated HTML page loads to the right login page.

Page routes (``response_class=HTMLResponse``) authenticate with the same
dependencies as the JSON API, so an expired or missing session would otherwise
render a raw ``{"detail": ...}`` body. For browser page loads we turn the
401/403 into a 303 to ``/login`` (or ``/admin/`` for admin pages) carrying a
validated ``?next=`` and a fixed flash message. Every other request, including
all JSON API routes, keeps FastAPI's default HTTPException response.
"""

import logging
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.datastructures import DefaultPlaceholder
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from .cookies import clear_session_cookie
from .flash import REASONS, set_flash

logger = logging.getLogger(__name__)

_PAGE_METHODS = frozenset({"GET", "HEAD"})
_DEFAULT_REASON = "signin_required"
_ADMIN_REASON = "admin_required"


def _response_class(route: APIRoute):
    cls = route.response_class
    if isinstance(cls, DefaultPlaceholder):
        cls = cls.value
    return cls


def is_html_page_request(request: Request) -> bool:
    """True for a GET/HEAD request matched to an APIRoute that renders HTML."""
    if request.method not in _PAGE_METHODS:
        return False
    route = request.scope.get("route")
    if not isinstance(route, APIRoute):
        return False
    if request.method not in (route.methods or ()):
        return False
    cls = _response_class(route)
    return isinstance(cls, type) and issubclass(cls, HTMLResponse)


def _is_admin_path(path: str) -> bool:
    return path == "/admin" or path.startswith("/admin/")


def _login_redirect(request: Request, exc: StarletteHTTPException) -> RedirectResponse:
    # Imported lazily: auth_pages pulls in templates/routes at import time.
    from app.routes.auth_pages import safe_next, ADMIN_NEXT_PREFIXES, USER_NEXT_PREFIXES

    path = request.url.path
    admin_page = _is_admin_path(path)
    target = "/admin/" if admin_page else "/login"
    current = path + (f"?{request.url.query}" if request.url.query else "")
    next_url = safe_next(current, ADMIN_NEXT_PREFIXES if admin_page else USER_NEXT_PREFIXES)
    if next_url:
        target = f"{target}?{urlencode({'next': next_url})}"

    response = RedirectResponse(url=target, status_code=303)
    if exc.status_code == 401:
        code = (exc.headers or {}).get("X-Auth-Error")
        if code not in REASONS:
            code = _DEFAULT_REASON
        # The session cookie is stale (expired, unknown user) or absent.
        clear_session_cookie(response)
    else:
        # 403: signed in, but not as the administrator. Keep the session.
        code = _ADMIN_REASON
    level, message = REASONS[code]
    set_flash(response, message, level)
    response.headers["Cache-Control"] = "no-store"
    return response


def _should_redirect(request: Request, exc: StarletteHTTPException) -> bool:
    if exc.status_code == 401:
        return is_html_page_request(request)
    if exc.status_code == 403:
        # Only admin pages raise 403 for "wrong principal"; other 403s keep
        # their meaning (and response).
        return _is_admin_path(request.url.path) and is_html_page_request(request)
    return False


def install_page_auth_redirects(app: FastAPI) -> None:
    """Register the HTML-page 401/403 -> login redirect on *app*."""

    async def _handler(request: Request, exc: StarletteHTTPException):
        if _should_redirect(request, exc):
            try:
                return _login_redirect(request, exc)
            except Exception:
                logger.exception("Failed to build login redirect for %s", request.url.path)
        return await http_exception_handler(request, exc)

    app.add_exception_handler(StarletteHTTPException, _handler)

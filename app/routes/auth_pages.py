"""Auth page rendering (login / signup / admin login) with the shared alert slot."""

import logging
import time
from typing import Optional
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.auth.flash import ACTIONS, pop_flash, flash_from_reason

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/frontend/templates")

USER_NEXT_PREFIXES = ("/dashboard/",)
ADMIN_NEXT_PREFIXES = ("/admin/",)

_NEXT_MAX_LEN = 512
# Admin entry points that would loop back to the login page.
_NEXT_REJECT_EXACT = frozenset({"/admin/", "/admin/login"})


def safe_next(raw: Optional[str], allowed_prefixes) -> Optional[str]:
    """Return *raw* if it is a safe same-site relative path under an allowed prefix."""
    if not raw or not isinstance(raw, str):
        return None
    if len(raw) > _NEXT_MAX_LEN:
        return None
    if not raw.startswith("/") or raw.startswith("//") or raw.startswith("/\\"):
        return None
    if "\\" in raw:
        return None
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme or parts.netloc:
        return None
    # Browsers resolve "." / ".." segments (including the %2e forms) before
    # navigating, which would let the path escape the allowed prefix.
    if any(unquote(seg) in (".", "..") for seg in parts.path.split("/")):
        return None
    if parts.path in _NEXT_REJECT_EXACT:
        return None
    if not any(parts.path.startswith(prefix) for prefix in allowed_prefixes):
        return None
    return raw


def render_auth_page(
    request: Request,
    template_name: str,
    context: Optional[dict] = None,
    *,
    next_prefixes=USER_NEXT_PREFIXES,
) -> HTMLResponse:
    """Render an auth page with the one-time flash (or ``?reason=``) and a validated ``next``."""
    ctx = dict(context or {})
    # Render into a placeholder first so pop_flash can clear the cookie on the
    # final response; the cookie header is copied across below.
    cookie_sink = HTMLResponse()
    flash = pop_flash(request, cookie_sink)
    if flash is None:
        flash = flash_from_reason(request.query_params.get("reason"))
    ctx.update({
        "request": request,
        "title": ctx.get("title", "LLM Proxy Server"),
        "cache_version": str(int(time.time())),
        "flash": flash,
        "next": safe_next(request.query_params.get("next"), next_prefixes),
        "auth_actions": ACTIONS,
    })
    response = templates.TemplateResponse(template_name, ctx)
    for key, value in cookie_sink.raw_headers:
        if key.lower() == b"set-cookie":
            response.raw_headers.append((key, value))
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return render_auth_page(request, "auth/login.html", {"title": "Login - LLM Proxy Server"})


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return render_auth_page(request, "auth/signup.html", {"title": "Sign Up - LLM Proxy Server"})

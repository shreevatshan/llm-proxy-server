"""Shared fixture for the auth-flow route tests (OAuth link, pending withdraw, flash, redirects).

Each test gets a throwaway SQLite database wired into the management app through a
``get_db`` override, and a TestClient that never follows redirects so every 303 and its
cookies can be inspected. ``COOKIE_SECURE`` is forced off so the client's cookie jar
sends the Secure-less cookies back over the test's plain-HTTP base URL.

The route tests drive a synchronous TestClient, so they are plain ``unittest.TestCase``
cases; database setup runs through ``asyncio.run`` on a ``NullPool`` engine (a fresh
connection per session, never shared across event loops).
"""

import asyncio
import os
import re
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.auth.cache import auth_cache
from app.auth.database import get_db, get_password_hash
from app.auth.flash import FLASH_COOKIE
from app.auth.models import Base, OAuthUser, User
from app.auth.purpose_tokens import PURPOSE_FLASH, read_purpose_token

ADMIN_EMAIL = "owner@example.com"
CSRF = "csrf-test-token"
PASSWORD = "secret1"

_APP = None


def management_app():
    """One management app for the whole module run; routes are stateless."""
    global _APP
    if _APP is None:
        from app.main import create_management_app
        _APP = create_management_app(FastAPI(), FastAPI(), FastAPI())
    return _APP


def set_cookie_headers(response, name=None):
    headers = response.headers
    if hasattr(headers, "multi_items"):  # httpx (TestClient) response
        items = headers.multi_items()
    else:  # Starlette response built in-process
        items = headers.items()
    values = [v for k, v in items if k.lower() == "set-cookie"]
    if name is not None:
        values = [v for v in values if v.startswith(name + "=")]
    return values


def cookie_cleared(response, name):
    """Whether *response* deletes cookie *name* (Starlette's delete_cookie: Max-Age=0)."""
    return any("max-age=0" in v.lower() for v in set_cookie_headers(response, name))


def cookie_set(response, name):
    """Whether *response* sets cookie *name* to a non-empty value."""
    for v in set_cookie_headers(response, name):
        value = v.split(";", 1)[0].split("=", 1)[1].strip('"')
        if value and "max-age=0" not in v.lower():
            return True
    return False


def flash_claims(response):
    """Decode the flash cookie set by *response* ({"msg","lvl","act","src"}) or None."""
    for v in set_cookie_headers(response, FLASH_COOKIE):
        token = v.split(";", 1)[0].split("=", 1)[1].strip('"')
        claims = read_purpose_token(token, PURPOSE_FLASH)
        if claims:
            return claims
    return None


def rendered_alert(html):
    """(message, [action keys], source, level) of the server-rendered #authAlert, or Nones."""
    message = re.search(r'class="auth-alert-message">([^<]*)<', html)
    actions = re.findall(r'data-auth-action="([^"]+)"', html)
    source = re.search(r'data-source="([^"]*)"', html)
    level = re.search(r'class="alert alert-([a-z]+)"', html)
    return (
        message.group(1) if message else None,
        actions,
        source.group(1) if source else None,
        level.group(1) if level else None,
    )


class AuthFlowTestCase(unittest.TestCase):
    """A temp database behind the real management app, plus user / client helpers."""

    def setUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self._db_path}", poolclass=NullPool)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

        async def _init():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        asyncio.run(_init())

        self.app = management_app()

        async def _override_db():
            async with self.Session() as session:
                yield session
        self.app.dependency_overrides[get_db] = _override_db

        # Middleware user lookups (cache miss) open sessions from the module factory.
        from app.auth import database as auth_database
        self._patches = [
            patch.dict(os.environ, {"COOKIE_SECURE": "false"}),
            patch("app.routes.auth.get_admin_email", return_value=ADMIN_EMAIL),
            patch.object(auth_database, "AsyncSessionLocal", self.Session),
            patch("app.auth.middleware.AsyncSessionLocal", self.Session),
        ]
        for p in self._patches:
            p.start()
        auth_cache.clear()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.app.dependency_overrides.pop(get_db, None)
        auth_cache.clear()
        asyncio.run(self.engine.dispose())
        os.unlink(self._db_path)

    # -- helpers ---------------------------------------------------------

    def run_async(self, coro):
        return asyncio.run(coro)

    def client(self):
        c = TestClient(self.app, follow_redirects=False)
        c.cookies.set("csrf_token", CSRF)
        return c

    def post(self, client, url, json=None, **kwargs):
        headers = {"X-CSRF-Token": CSRF, **kwargs.pop("headers", {})}
        return client.post(url, json=json if json is not None else {}, headers=headers, **kwargs)

    def add_user(self, username, email=None, password=PASSWORD, *, active=True, pending=False,
                 oauth_provider=None, oauth_sub=None):
        async def _add():
            async with self.Session() as s:
                user = User(
                    username=username,
                    email=email or f"{username}@example.com",
                    hashed_password=get_password_hash(password) if password else None,
                    is_active=active,
                    is_pending_approval=pending,
                    oauth_provider=oauth_provider,
                    oauth_sub=oauth_sub,
                )
                s.add(user)
                await s.commit()
                await s.refresh(user)
                return user
        return self.run_async(_add())

    def add_oauth_row(self, user_id, sub, email, provider="zoho"):
        async def _add():
            async with self.Session() as s:
                s.add(OAuthUser(user_id=user_id, provider=provider, provider_user_id=sub,
                                email=email, name="Z User"))
                await s.commit()
        self.run_async(_add())

    def users(self):
        async def _q():
            async with self.Session() as s:
                return (await s.execute(select(User).order_by(User.id))).scalars().all()
        return self.run_async(_q())

    def user(self, username):
        return next((u for u in self.users() if u.username == username), None)

    def oauth_rows(self, sub=None):
        async def _q():
            async with self.Session() as s:
                stmt = select(OAuthUser)
                if sub is not None:
                    stmt = stmt.where(OAuthUser.provider_user_id == sub)
                return (await s.execute(stmt)).scalars().all()
        return self.run_async(_q())

    def delete_user_row(self, user_id):
        async def _d():
            async with self.Session() as s:
                user = await s.get(User, user_id)
                await s.delete(user)
                await s.commit()
        self.run_async(_d())

    def login_page_alert(self, client):
        """Load /login with the client's cookies and return its rendered alert."""
        return rendered_alert(client.get("/login").text)

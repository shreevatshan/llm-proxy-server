"""Tests for redirecting unauthenticated HTML page loads to the login pages.

Page routes authenticate with the same dependencies as the JSON API, so without
the handler an expired or missing session renders a raw ``{"detail": ...}`` body.
For browser page loads the 401/403 becomes a 303 to ``/login`` (or ``/admin/``)
with a validated ``next`` and a fixed flash; JSON APIs keep their responses.
"""

import re
import unittest
from datetime import timedelta
from urllib.parse import parse_qs, urlsplit

from fastapi.datastructures import DefaultPlaceholder
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from app.auth.auth import create_access_token
from app.auth.cookies import SESSION_COOKIE
from app.auth.flash import REASONS

from tests.auth_flow_base import AuthFlowTestCase, cookie_cleared, cookie_set, flash_claims


def _is_html_route(route):
    cls = route.response_class
    if isinstance(cls, DefaultPlaceholder):
        cls = cls.value
    return isinstance(cls, type) and issubclass(cls, HTMLResponse)


class PageAuthRedirectTests(AuthFlowTestCase):

    def assert_redirect(self, r, target, next_url=None):
        self.assertEqual(r.status_code, 303, r.text)
        location = urlsplit(r.headers["location"])
        self.assertEqual(location.path, target)
        query = parse_qs(location.query)
        if next_url is None:
            self.assertNotIn("next", query)
        else:
            self.assertEqual(query.get("next"), [next_url])
        self.assertEqual(set(query) - {"next"}, set(), "no free text in the redirect URL")
        self.assertEqual(r.headers.get("cache-control"), "no-store")
        return flash_claims(r)

    def assert_reason(self, flash, code):
        level, message = REASONS[code]
        self.assertEqual((flash["lvl"], flash["msg"]), (level, message))

    # -- user pages ----------------------------------------------------------

    def test_no_cookie_dashboard_redirects_to_login_with_next(self):
        r = self.client().get("/dashboard/")
        flash = self.assert_redirect(r, "/login", "/dashboard/")
        self.assert_reason(flash, "signin_required")
        self.assertTrue(cookie_cleared(r, SESSION_COOKIE))

    def test_next_keeps_the_query_string(self):
        r = self.client().get("/dashboard/profile?tab=keys")
        self.assert_redirect(r, "/login", "/dashboard/profile?tab=keys")

    def test_expired_cookie_gives_session_expired_and_clears_cookie(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-10)))
        r = c.get("/dashboard/")
        flash = self.assert_redirect(r, "/login", "/dashboard/")
        self.assert_reason(flash, "session_expired")
        self.assertTrue(cookie_cleared(r, SESSION_COOKIE))

    def test_deleted_user_gives_session_invalid(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "ghost"}))
        r = c.get("/dashboard/endpoints")
        flash = self.assert_redirect(r, "/login", "/dashboard/endpoints")
        self.assert_reason(flash, "session_invalid")
        self.assertTrue(cookie_cleared(r, SESSION_COOKIE))

    def test_valid_session_still_renders_the_page(self):
        self.add_user("alice")
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "alice"}))
        r = c.get("/dashboard/endpoints")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(cookie_cleared(r, SESSION_COOKIE))

    # -- admin pages ---------------------------------------------------------

    def test_no_cookie_admin_page_redirects_to_admin_login(self):
        r = self.client().get("/admin/dashboard")
        flash = self.assert_redirect(r, "/admin/", "/admin/dashboard")
        self.assert_reason(flash, "signin_required")

    def test_expired_admin_cookie_gives_session_expired(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token(
            {"sub": "admin"}, expires_delta=timedelta(seconds=-10), is_admin=True))
        r = c.get("/admin/models/search")
        flash = self.assert_redirect(r, "/admin/", "/admin/models/search")
        self.assert_reason(flash, "session_expired")
        self.assertTrue(cookie_cleared(r, SESSION_COOKIE))

    def test_user_token_on_admin_page_redirects_and_keeps_cookie(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "alice"}))
        r = c.get("/admin/dashboard")
        flash = self.assert_redirect(r, "/admin/", "/admin/dashboard")
        self.assert_reason(flash, "admin_required")
        self.assertFalse(cookie_cleared(r, SESSION_COOKIE))
        self.assertFalse(cookie_set(r, SESSION_COOKIE))

    # -- JSON APIs and other errors are unchanged ---------------------------

    def test_json_apis_stay_401_json(self):
        c = self.client()
        for path in ("/dashboard/api/models", "/admin/users", "/auth/me"):
            with self.subTest(path=path):
                r = c.get(path)
                self.assertEqual(r.status_code, 401)
                self.assertTrue(r.headers["content-type"].startswith("application/json"))
                self.assertIn("detail", r.json())
                self.assertIsNone(flash_claims(r))

    def test_expired_cookie_on_json_api_keeps_header(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-10)))
        r = c.get("/auth/me")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.headers.get("x-auth-error"), "session_expired")

    def test_user_token_on_admin_json_api_stays_403_json(self):
        c = self.client()
        c.cookies.set(SESSION_COOKIE, create_access_token({"sub": "alice"}))
        r = c.get("/admin/users")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json(), {"detail": "Admin access required"})

    def test_404_and_405_unchanged(self):
        c = self.client()
        r = c.get("/dashboard/no-such-page")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json(), {"detail": "Not Found"})
        r = self.post(c, "/dashboard/")
        self.assertEqual(r.status_code, 405)
        self.assertEqual(r.json(), {"detail": "Method Not Allowed"})

    # -- guard ---------------------------------------------------------------

    def test_every_protected_html_page_redirects(self):
        """New HTML pages under /dashboard or /admin must not render JSON when signed out."""
        pages = [
            route for route in self.app.routes
            if isinstance(route, APIRoute) and "GET" in route.methods and _is_html_route(route)
            and (route.path.startswith("/dashboard") or route.path.startswith("/admin"))
            and route.path != "/admin/"
        ]
        self.assertGreaterEqual(len(pages), 5)
        c = self.client()
        for route in pages:
            path = re.sub(r"\{[^}]+\}", "1", route.path)
            with self.subTest(path=path):
                r = c.get(path)
                self.assertEqual(r.status_code, 303, f"{path} returned {r.status_code}: {r.text[:200]}")
                expected = "/admin/" if path.startswith("/admin") else "/login"
                self.assertEqual(urlsplit(r.headers["location"]).path, expected)


if __name__ == "__main__":
    unittest.main()

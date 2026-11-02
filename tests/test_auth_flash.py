"""Tests for the one-time auth flash, ``?reason=`` codes, ``safe_next`` and admin login.

Messages reach the auth pages only through a signed, HttpOnly, one-time cookie or
an allowlisted ``?reason=`` code, never as free text in the URL. The flash must be
HTML-escaped, shown once, and ignored when tampered with. ``next`` must only ever
be a same-site relative path under an allowed prefix.
"""

import unittest
from datetime import timedelta
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from fastapi.responses import Response

from app.auth.flash import ACTIONS, FLASH_COOKIE, REASONS, flash_from_reason, set_flash
from app.auth.purpose_tokens import PURPOSE_FLASH, PURPOSE_OAUTH_LINK, issue_purpose_token
from app.config import config
from app.routes.auth_pages import ADMIN_NEXT_PREFIXES, USER_NEXT_PREFIXES, safe_next

from tests.auth_flow_base import (
    AuthFlowTestCase, cookie_cleared, cookie_set, flash_claims, rendered_alert, set_cookie_headers,
)


def flash_cookie(message, level="danger", actions=(), source=None):
    resp = Response()
    set_flash(resp, message, level, actions=actions, source=source)
    return set_cookie_headers(resp, FLASH_COOKIE)[0].split(";", 1)[0].split("=", 1)[1].strip('"')


class SetFlashTests(unittest.TestCase):

    def test_cookie_attributes(self):
        resp = Response()
        set_flash(resp, "hello", "info")
        [header] = set_cookie_headers(resp, FLASH_COOKIE)
        self.assertIn("HttpOnly", header)
        self.assertIn("Path=/", header)
        self.assertIn("Max-Age=120", header)
        self.assertIn("SameSite=lax", header)

    def test_level_and_actions_are_allowlisted_and_message_truncated(self):
        resp = Response()
        set_flash(resp, "x" * 2000, "evil-level", actions=["withdraw_pending", "rm -rf", "withdraw_pending"])
        claims = flash_claims(resp)
        self.assertEqual(claims["lvl"], "danger")
        self.assertEqual(claims["act"], ["withdraw_pending"])
        self.assertEqual(len(claims["msg"]), 500)

    def test_flash_from_reason(self):
        for code, (level, message) in REASONS.items():
            flash = flash_from_reason(code)
            self.assertEqual((flash["level"], flash["message"], flash["actions"]), (level, message, []))
        self.assertIsNone(flash_from_reason("nope"))
        self.assertIsNone(flash_from_reason(None))
        self.assertIsNone(flash_from_reason(""))


class SafeNextTests(unittest.TestCase):

    def test_accepts_relative_paths_under_prefix(self):
        self.assertEqual(safe_next("/dashboard/", USER_NEXT_PREFIXES), "/dashboard/")
        self.assertEqual(safe_next("/dashboard/profile?tab=keys", USER_NEXT_PREFIXES), "/dashboard/profile?tab=keys")
        self.assertEqual(safe_next("/admin/dashboard", ADMIN_NEXT_PREFIXES), "/admin/dashboard")

    def test_rejects_unsafe_values(self):
        for raw in (
            None, "", "//evil.com", "//evil.com/dashboard/", "/\\evil.com", "/dashboard\\..\\x",
            "https://x", "https://evil.com/dashboard/", "javascript:alert(1)", "/login", "dashboard/",
            "/dashboard", "/dashboard/\nSet-Cookie:x", "/dashboard/\x00", "/dashboard/" + "a" * 600,
            "/admin/dashboard", "/dashboard/../x", "/dashboard/%2e%2e/x", "/dashboard/%2E%2e/x",
            "/dashboard/../..//evil.com", "/dashboard/%2e%2e/%2e%2e//evil.com", "/dashboard/./x",
            "/dashboard/%2e/x", "/dashboard/..", "/dashboard/%2e%2e/auth/zoho/login",
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(safe_next(raw, USER_NEXT_PREFIXES))

    def test_accepts_dots_inside_segments(self):
        self.assertEqual(safe_next("/dashboard/a..b", USER_NEXT_PREFIXES), "/dashboard/a..b")
        self.assertEqual(safe_next("/dashboard/.well", USER_NEXT_PREFIXES), "/dashboard/.well")

    def test_rejects_admin_login_loops_and_user_pages_for_admin(self):
        for raw in ("/admin/", "/admin/login", "/dashboard/", "//admin/", "/admin/%2E%2E/admin/", "/admin/../admin/login"):
            with self.subTest(raw=raw):
                self.assertIsNone(safe_next(raw, ADMIN_NEXT_PREFIXES))


class AuthPageFlashTests(AuthFlowTestCase):

    def test_flash_renders_escaped_and_is_cleared_after_one_read(self):
        c = self.client()
        c.cookies.set(FLASH_COOKIE, flash_cookie('<script>alert("x")</script> & co', "warning"))
        r = c.get("/login")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertNotIn("<script>alert", r.text)
        self.assertIn("&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt; &amp; co", r.text)
        message, actions, _source, level = rendered_alert(r.text)
        self.assertIsNotNone(message)
        self.assertEqual(level, "warning")
        self.assertEqual(actions, [])
        self.assertTrue(cookie_cleared(r, FLASH_COOKIE))

    def test_flash_from_a_real_redirect_is_shown_exactly_once(self):
        c = self.client()
        r = c.get("/auth/zoho/link")  # no link request -> 303 /login with a flash
        self.assertEqual(r.status_code, 303)
        self.assertIsNotNone(c.cookies.get(FLASH_COOKIE))
        first = c.get("/login")
        self.assertIn("expired", rendered_alert(first.text)[0])
        self.assertIsNone(c.cookies.get(FLASH_COOKIE))
        self.assertIsNone(rendered_alert(c.get("/login").text)[0])

    def test_flash_actions_and_source_render(self):
        c = self.client()
        c.cookies.set(FLASH_COOKIE, flash_cookie("Pending", "warning", ["withdraw_pending"], "oauth"))
        r = c.get("/login")
        message, actions, source, _level = rendered_alert(r.text)
        self.assertEqual((message, actions, source), ("Pending", ["withdraw_pending"], "oauth"))
        self.assertIn(ACTIONS["withdraw_pending"].replace("&", "&amp;"), r.text)

    def test_flash_also_renders_on_signup(self):
        c = self.client()
        c.cookies.set(FLASH_COOKIE, flash_cookie("Hello signup", "info"))
        self.assertEqual(rendered_alert(c.get("/signup").text)[0], "Hello signup")

    def test_tampered_or_foreign_flash_is_ignored_and_cleared(self):
        good = flash_cookie("Real message")
        header, _payload, sig = good.split(".")
        forged_payload = flash_cookie("Forged message").split(".")[1]
        for token in (
            f"{header}.{forged_payload}.{sig}",
            "garbage",
            issue_purpose_token(PURPOSE_OAUTH_LINK, {"msg": "wrong purpose", "lvl": "danger"}, timedelta(minutes=1)),
            issue_purpose_token(PURPOSE_FLASH, {"msg": "expired", "lvl": "danger"}, timedelta(seconds=-5)),
        ):
            with self.subTest(token=token[:20]):
                c = self.client()
                c.cookies.set(FLASH_COOKIE, token)
                r = c.get("/login")
                self.assertEqual(r.status_code, 200)
                self.assertIsNone(rendered_alert(r.text)[0])
                self.assertNotIn("Forged message", r.text)
                self.assertTrue(cookie_cleared(r, FLASH_COOKIE))

    def test_unknown_action_and_level_in_signed_flash_are_dropped(self):
        c = self.client()
        c.cookies.set(FLASH_COOKIE, issue_purpose_token(
            PURPOSE_FLASH, {"msg": "hi", "lvl": "x\" onclick=\"", "act": ["delete_everything"]}, timedelta(minutes=1)))
        message, actions, _source, level = rendered_alert(c.get("/login").text)
        self.assertEqual((message, actions, level), ("hi", [], "danger"))

    def test_reason_code_renders_its_fixed_message(self):
        level, text = REASONS["session_expired"]
        message, _actions, _source, rendered_level = rendered_alert(
            self.client().get("/login?reason=session_expired").text)
        self.assertEqual((message, rendered_level), (text, level))

    def test_unknown_reason_and_legacy_error_render_nothing(self):
        c = self.client()
        for url in ("/login?reason=<b>pwned</b>", "/login?error=Something%20bad", "/signup?error=x"):
            with self.subTest(url=url):
                r = c.get(url)
                self.assertEqual(r.status_code, 200)
                self.assertIsNone(rendered_alert(r.text)[0])
                self.assertNotIn("pwned", r.text)
                self.assertNotIn("Something bad", r.text)

    def test_flash_cookie_wins_over_reason(self):
        c = self.client()
        c.cookies.set(FLASH_COOKIE, flash_cookie("From cookie", "info"))
        self.assertEqual(rendered_alert(c.get("/login?reason=session_expired").text)[0], "From cookie")

    def test_login_form_carries_only_a_validated_next(self):
        c = self.client()
        self.assertIn('data-next="/dashboard/profile"', c.get("/login?next=%2Fdashboard%2Fprofile").text)
        for bad in ("//evil.com", "https%3A%2F%2Fevil.com", "%2Flogin"):
            with self.subTest(next=bad):
                r = c.get(f"/login?next={bad}")
                self.assertIn('data-next=""', r.text)
                self.assertNotIn("evil.com", r.text)


class AdminLoginFlashTests(AuthFlowTestCase):

    def setUp(self):
        super().setUp()
        self._admin_patches = [
            patch.object(config.admin, "enabled", True),
            patch.object(config.admin, "username", "admin"),
            patch.object(config.admin, "password", "admin-pass"),
        ]
        for p in self._admin_patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._admin_patches):
            p.stop()
        super().tearDown()

    HTML = {"Accept": "text/html,application/xhtml+xml"}

    def test_bad_password_html_is_303_with_flash(self):
        r = self.client().post("/admin/login", headers=self.HTML,
                               data={"username": "admin", "password": "wrong", "next": "/admin/models/search"})
        self.assertEqual(r.status_code, 303)
        location = urlsplit(r.headers["location"])
        self.assertEqual(location.path, "/admin/")
        self.assertEqual(parse_qs(location.query), {"next": ["/admin/models/search"]})
        flash = flash_claims(r)
        self.assertEqual((flash["lvl"], flash["msg"]), ("danger", "Invalid admin credentials"))
        self.assertFalse(cookie_set(r, "access_token"))

    def test_bad_password_html_drops_an_unsafe_next(self):
        r = self.client().post("/admin/login", headers=self.HTML,
                               data={"username": "admin", "password": "wrong", "next": "//evil.com"})
        self.assertEqual(r.headers["location"], "/admin/")

    def test_flash_renders_on_admin_login_page(self):
        c = self.client()
        c.post("/admin/login", headers=self.HTML, data={"username": "admin", "password": "wrong"})
        r = c.get("/admin/?next=%2Fadmin%2Fdashboard")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(rendered_alert(r.text)[0], "Invalid admin credentials")
        self.assertIn('name="next" value="/admin/dashboard"', r.text)

    def test_bad_password_without_html_accept_is_401_json(self):
        r = self.client().post("/admin/login", data={"username": "admin", "password": "wrong"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json(), {"detail": "Invalid admin credentials"})
        self.assertIsNone(flash_claims(r))

    def test_missing_fields(self):
        r = self.client().post("/admin/login", headers=self.HTML, data={"username": "", "password": ""})
        self.assertEqual(r.status_code, 303)
        self.assertIn("required", flash_claims(r)["msg"])
        r = self.client().post("/admin/login", data={"username": "", "password": ""})
        self.assertEqual(r.status_code, 400)

    def test_success_html_goes_to_validated_next(self):
        r = self.client().post("/admin/login", headers=self.HTML,
                               data={"username": "admin", "password": "admin-pass", "next": "/admin/models/search"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/admin/models/search")
        self.assertTrue(cookie_set(r, "access_token"))
        r = self.client().post("/admin/login", headers=self.HTML,
                               data={"username": "admin", "password": "admin-pass", "next": "https://evil.com"})
        self.assertEqual(r.headers["location"], "/admin/dashboard")

    def test_success_without_html_accept_keeps_302(self):
        r = self.client().post("/admin/login", data={"username": "admin", "password": "admin-pass"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers["location"], "/admin/dashboard")
        self.assertTrue(cookie_set(r, "access_token"))


if __name__ == "__main__":
    unittest.main()

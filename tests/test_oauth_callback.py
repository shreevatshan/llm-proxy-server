"""Tests for the ZOHO OAuth callback.

Every exit is a 303 that clears the one-time state cookie and carries a fixed
flash message; provider and exception text must never reach the user. An email
that already belongs to an account is never linked silently: it either gets an
explanatory flash or the signed ``oauth_link`` cookie and the link page.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.auth.cookies import OAUTH_LINK_COOKIE, PENDING_WITHDRAW_COOKIE, SESSION_COOKIE
from app.auth.purpose_tokens import (
    PURPOSE_OAUTH_LINK, PURPOSE_PENDING_WITHDRAW, claims_match_user, read_purpose_token,
)
from app.auth.zoho_oauth import ZohoUserInfo

from tests.auth_flow_base import (
    ADMIN_EMAIL, AuthFlowTestCase, cookie_cleared, cookie_set, flash_claims, set_cookie_headers,
)

STATE = "state-123"


def zoho_info(sub="z-1", email="bob@example.com", verified=True, name="Bob Zoho"):
    return ZohoUserInfo(sub=sub, email=email, email_verified=verified, name=name,
                        iss="https://accounts.zoho.com", aud="client", exp=0, iat=0)


class OAuthCallbackTestBase(AuthFlowTestCase):

    def setUp(self):
        super().setUp()
        self.zoho = MagicMock()
        self.zoho.get_user_info = AsyncMock()
        self.zoho.get_authorization_url = AsyncMock(return_value="https://accounts.zoho.com/oauth")
        self.webhook = AsyncMock()
        self._cb_patches = [
            patch("app.auth.zoho_oauth.get_zoho_oauth", return_value=self.zoho),
            patch("app.routes.auth.send_signup_webhook", self.webhook),
        ]
        for p in self._cb_patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._cb_patches):
            p.stop()
        super().tearDown()

    def callback(self, info=None, client=None, **params):
        if info is not None:
            self.zoho.get_user_info.return_value = info
        c = client or self.client()
        c.cookies.set("zoho_oauth_state", STATE)
        query = {"state": STATE, "code": "auth-code", **params}
        return c, c.get("/auth/zoho/callback", params=query)

    def assert_login_redirect(self, response):
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login")
        self.assertTrue(cookie_cleared(response, "zoho_oauth_state"))
        self.assertNotIn("?", response.headers["location"])
        return flash_claims(response)


class CallbackFailureTests(OAuthCallbackTestBase):

    def test_cancel_on_consent_screen(self):
        c = self.client()
        r = c.get("/auth/zoho/callback", params={"error": "access_denied", "error_description": "<b>user said no</b>"})
        flash = self.assert_login_redirect(r)
        self.assertEqual(flash["msg"], "ZOHO sign-in was cancelled.")
        self.assertEqual(flash["lvl"], "info")
        self.zoho.get_user_info.assert_not_called()

    def test_other_provider_error_is_not_echoed(self):
        r = self.client().get("/auth/zoho/callback", params={"error": "server_error", "error_description": "LEAKY"})
        flash = self.assert_login_redirect(r)
        self.assertNotIn("LEAKY", flash["msg"])
        self.assertNotIn("server_error", flash["msg"])

    def test_no_code_and_no_params_is_a_redirect_not_a_422(self):
        r = self.client().get("/auth/zoho/callback")
        flash = self.assert_login_redirect(r)
        self.assertIsNotNone(flash)
        c = self.client()
        c.cookies.set("zoho_oauth_state", STATE)
        r = c.get("/auth/zoho/callback", params={"state": STATE})
        flash = self.assert_login_redirect(r)
        self.assertIn("did not complete", flash["msg"])
        self.zoho.get_user_info.assert_not_called()

    def test_state_mismatch(self):
        c = self.client()
        c.cookies.set("zoho_oauth_state", "other-state")
        r = c.get("/auth/zoho/callback", params={"state": STATE, "code": "x"})
        flash = self.assert_login_redirect(r)
        self.assertIn("expired or was invalid", flash["msg"])
        self.zoho.get_user_info.assert_not_called()

    def test_missing_state_cookie(self):
        r = self.client().get("/auth/zoho/callback", params={"state": STATE, "code": "x"})
        flash = self.assert_login_redirect(r)
        self.assertIn("expired or was invalid", flash["msg"])

    def test_user_info_exception_text_does_not_leak(self):
        self.zoho.get_user_info.side_effect = RuntimeError("SECRET provider body token=abc")
        with self.assertLogs("app.routes.auth", level="ERROR"):
            _c, r = self.callback()
        flash = self.assert_login_redirect(r)
        self.assertNotIn("SECRET", flash["msg"])
        self.assertNotIn("abc", flash["msg"])
        self.assertEqual(flash["lvl"], "danger")

    def test_server_error_after_user_info_does_not_leak(self):
        with patch("app.routes.auth.get_oauth_user_by_provider_id",
                   AsyncMock(side_effect=RuntimeError("DB EXPLODED at /var/db"))):
            with self.assertLogs("app.routes.auth", level="ERROR"):
                _c, r = self.callback(zoho_info())
        flash = self.assert_login_redirect(r)
        self.assertNotIn("EXPLODED", flash["msg"])
        self.assertIn("server error", flash["msg"])

    def test_not_configured(self):
        with patch("app.auth.zoho_oauth.get_zoho_oauth", return_value=None):
            r = self.client().get("/auth/zoho/callback", params={"state": STATE, "code": "x"})
            flash = self.assert_login_redirect(r)
            self.assertIn("not available", flash["msg"])
            r = self.client().get("/auth/zoho/login")
            self.assertEqual(r.status_code, 303)
            self.assertEqual(r.headers["location"], "/login")
            self.assertIn("not available", flash_claims(r)["msg"])

    def test_zoho_login_sets_state_and_redirects_to_zoho(self):
        r = self.client().get("/auth/zoho/login")
        self.assertIn(r.status_code, (302, 307))
        self.assertEqual(r.headers["location"], "https://accounts.zoho.com/oauth")
        self.assertTrue(cookie_set(r, "zoho_oauth_state"))

    def test_zoho_login_authorization_url_failure_is_a_flash(self):
        self.zoho.get_authorization_url.side_effect = RuntimeError("INTERNAL")
        with self.assertLogs("app.routes.auth", level="ERROR"):
            r = self.client().get("/auth/zoho/login")
        flash = self.assert_login_redirect(r)
        self.assertNotIn("INTERNAL", flash["msg"])


class CallbackNewUserTests(OAuthCallbackTestBase):

    def test_new_pending_user_gets_withdraw_cookie_and_action(self):
        c, r = self.callback(zoho_info(sub="n-1", email="newbie@example.com"))
        flash = self.assert_login_redirect(r)
        self.assertEqual(flash["lvl"], "warning")
        self.assertEqual(flash["act"], ["withdraw_pending"])
        self.assertEqual(flash["src"], "oauth")
        self.assertIn(ADMIN_EMAIL, flash["msg"])
        self.assertFalse(cookie_set(r, SESSION_COOKIE))

        user = self.user("newbie")
        self.assertIsNotNone(user)
        self.assertFalse(user.is_active)
        self.assertTrue(user.is_pending_approval)
        self.assertEqual(len(self.oauth_rows("n-1")), 1)

        [withdraw_header] = set_cookie_headers(r, PENDING_WITHDRAW_COOKIE)
        self.assertIn("Path=/auth/", withdraw_header)
        self.assertIn("HttpOnly", withdraw_header)
        claims = read_purpose_token(c.cookies.get(PENDING_WITHDRAW_COOKIE), PURPOSE_PENDING_WITHDRAW)
        self.assertTrue(claims_match_user(claims, user))
        self.assertNotIn("sub", claims)

        self.webhook.assert_awaited_once()
        self.assertEqual(self.webhook.await_args.kwargs["signup_mode"], "oauth")
        self.assertTrue(self.webhook.await_args.kwargs["is_pending"])

    def test_new_zohocorp_user_gets_a_session(self):
        c, r = self.callback(zoho_info(sub="n-2", email="emp@zohocorp.com"))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/dashboard/")
        self.assertTrue(cookie_set(r, SESSION_COOKIE))
        self.assertIsNone(flash_claims(r))
        self.assertFalse(cookie_set(r, PENDING_WITHDRAW_COOKIE))
        user = self.user("emp")
        self.assertTrue(user.is_active)
        self.assertFalse(user.is_pending_approval)
        self.webhook.assert_awaited_once()

    def test_returning_linked_user_signs_in_without_a_new_account(self):
        self.callback(zoho_info(sub="n-3", email="emp3@zohocorp.com"))
        self.webhook.reset_mock()
        _c, r = self.callback(zoho_info(sub="n-3", email="emp3@zohocorp.com"))
        self.assertEqual(r.headers["location"], "/dashboard/")
        self.assertEqual(len(self.users()), 1)
        self.webhook.assert_not_called()

    def test_deactivated_linked_user_gets_deactivated_message(self):
        user = self.add_user("gone", "gone@example.com", password=None, active=False, pending=False,
                             oauth_provider="zoho", oauth_sub="z-gone")
        self.add_oauth_row(user.id, "z-gone", "gone@example.com")
        _c, r = self.callback(zoho_info(sub="z-gone", email="gone@example.com"))
        flash = self.assert_login_redirect(r)
        self.assertIn("deactivated", flash["msg"])
        self.assertNotIn("pending", flash["msg"])
        self.assertEqual(flash["lvl"], "warning")  # same level login.js uses for a deactivated 403
        self.assertEqual(flash["act"], [])
        self.assertFalse(cookie_set(r, SESSION_COOKIE))
        self.assertFalse(cookie_set(r, PENDING_WITHDRAW_COOKIE))

    def test_pending_linked_user_gets_pending_message(self):
        user = self.add_user("waits", "waits@example.com", password=None, active=False, pending=True,
                             oauth_provider="zoho", oauth_sub="z-waits")
        self.add_oauth_row(user.id, "z-waits", "waits@example.com")
        _c, r = self.callback(zoho_info(sub="z-waits", email="waits@example.com"))
        flash = self.assert_login_redirect(r)
        self.assertIn("pending", flash["msg"])
        self.assertEqual(flash["act"], ["withdraw_pending"])
        self.assertTrue(cookie_set(r, PENDING_WITHDRAW_COOKIE))


class CallbackEmailMatchTests(OAuthCallbackTestBase):

    def test_email_match_sets_link_cookie_and_creates_no_identity(self):
        bob = self.add_user("bobby", "Bob@Example.com", pending=True, active=False)
        c, r = self.callback(zoho_info(sub="z-bob", email="bob@example.com"))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/auth/zoho/link")
        self.assertTrue(cookie_cleared(r, "zoho_oauth_state"))
        self.assertIsNone(flash_claims(r))
        self.assertFalse(cookie_set(r, SESSION_COOKIE))

        [link_header] = set_cookie_headers(r, OAUTH_LINK_COOKIE)
        self.assertIn("Path=/auth/", link_header)
        self.assertIn("HttpOnly", link_header)
        self.assertIn("Max-Age=600", link_header)
        claims = read_purpose_token(c.cookies.get(OAUTH_LINK_COOKIE), PURPOSE_OAUTH_LINK)
        self.assertTrue(claims_match_user(claims, bob))
        self.assertEqual(claims["psub"], "z-bob")
        self.assertEqual(claims["provider"], "zoho")
        self.assertNotIn("sub", claims)

        self.assertEqual(self.oauth_rows(), [])
        self.assertEqual(len(self.users()), 1)
        refreshed = self.user("bobby")
        self.assertIsNone(refreshed.oauth_provider)
        self.assertFalse(refreshed.is_active)
        self.webhook.assert_not_called()

    def _assert_contact_admin(self, r):
        flash = self.assert_login_redirect(r)
        self.assertIn(ADMIN_EMAIL, flash["msg"])
        self.assertIn("didn't create", flash["msg"])
        self.assertFalse(cookie_set(r, OAUTH_LINK_COOKIE))
        self.assertFalse(cookie_set(r, SESSION_COOKIE))
        self.assertEqual(self.oauth_rows(), [])
        return flash

    def test_unverified_email_is_not_offered_a_link(self):
        self.add_user("dave", "dave@example.com")
        _c, r = self.callback(zoho_info(sub="z-dave", email="dave@example.com", verified=False))
        flash = self._assert_contact_admin(r)
        self.assertIn("password", flash["msg"])

    def test_passwordless_account_is_not_offered_a_link(self):
        self.add_user("eve", "eve@example.com", password=None)
        _c, r = self.callback(zoho_info(sub="z-eve", email="eve@example.com"))
        self._assert_contact_admin(r)

    def test_account_linked_to_a_different_zoho_identity(self):
        carol = self.add_user("carol", "carol@example.com", oauth_provider="zoho", oauth_sub="z-carol-1")
        self.add_oauth_row(carol.id, "z-carol-1", "carol@example.com")
        _c, r = self.callback(zoho_info(sub="z-carol-2", email="carol@example.com"))
        flash = self.assert_login_redirect(r)
        self.assertIn("different ZOHO account", flash["msg"])
        self.assertIn(ADMIN_EMAIL, flash["msg"])
        self.assertFalse(cookie_set(r, OAUTH_LINK_COOKIE))
        self.assertEqual(len(self.oauth_rows("z-carol-2")), 0)

    def test_same_identity_without_row_is_restored_even_if_email_unverified(self):
        # Legacy OAuth-only account whose OAuthUser row is missing; ZOHO omits email_verified.
        self.add_user("legacy", "legacy@example.com", password=None,
                      oauth_provider="zoho", oauth_sub="z-legacy")
        _c, r = self.callback(zoho_info(sub="z-legacy", email="legacy@example.com", verified=None))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/dashboard/")
        self.assertTrue(cookie_set(r, SESSION_COOKIE))
        self.assertEqual(len(self.oauth_rows("z-legacy")), 1)

    def test_same_identity_restore_race_still_signs_in(self):
        from app.auth import database as auth_db
        from app.auth.models import OAuthUser

        legacy = self.add_user("legacy", "legacy@example.com", password=None,
                               oauth_provider="zoho", oauth_sub="z-legacy")
        real_link = auth_db.link_oauth_identity

        async def racing_link(db, user, provider, sub, **kw):
            # A concurrent callback inserts the row after our pre-check passed, so
            # our insert hits the unique constraint and the session rolls back.
            async with self.Session() as other:
                other.add(OAuthUser(user_id=legacy.id, provider=provider, provider_user_id=sub,
                                    email="legacy@example.com", name="Z User"))
                await other.commit()
            with patch.object(auth_db, "get_oauth_user_by_provider_id", AsyncMock(return_value=None)):
                return await real_link(db, user, provider, sub, **kw)

        with patch("app.routes.auth.link_oauth_identity", racing_link):
            _c, r = self.callback(zoho_info(sub="z-legacy", email="legacy@example.com"))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/dashboard/")
        self.assertTrue(cookie_set(r, SESSION_COOKIE))

    def test_different_identity_via_row_only(self):
        # oauth_provider is empty on the user, but an OAuthUser row exists.
        carol = self.add_user("carol", "carol@example.com")
        self.add_oauth_row(carol.id, "z-carol-1", "carol@example.com")
        _c, r = self.callback(zoho_info(sub="z-carol-2", email="carol@example.com"))
        flash = self.assert_login_redirect(r)
        self.assertIn("different ZOHO account", flash["msg"])


if __name__ == "__main__":
    unittest.main()

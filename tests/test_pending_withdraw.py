"""Tests for withdrawing (deleting) a signup that is still pending approval.

A pending user never gets a session, so the withdraw endpoint identifies them by
username + password, or by the short-lived ``pending_withdraw`` cookie set after a
ZOHO sign-in. Only a pending, inactive account can be withdrawn, and the admin's
self-delete switch does not apply (the account was never granted access).
"""

import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from app.auth.cookies import PENDING_WITHDRAW_COOKIE
from app.auth.purpose_tokens import (
    PURPOSE_OAUTH_LINK, PURPOSE_PENDING_WITHDRAW, issue_purpose_token, user_binding_claims,
)

from tests.auth_flow_base import (
    PASSWORD, AuthFlowTestCase, cookie_cleared, flash_claims, set_cookie_headers,
)

WITHDRAW = "/auth/pending/withdraw"


class PendingWithdrawTests(AuthFlowTestCase):

    def withdraw(self, c=None, **body):
        return self.post(c or self.client(), WITHDRAW, json=body)

    def assert_withdrawn(self, r, username):
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["redirect"], "/login")
        self.assertTrue(cookie_cleared(r, PENDING_WITHDRAW_COOKIE))
        flash = flash_claims(r)
        self.assertEqual(flash["lvl"], "success")
        self.assertIn("withdrawn", flash["msg"])
        self.assertIsNone(self.user(username))

    def test_credentials_delete_the_pending_user(self):
        self.add_user("pend", active=False, pending=True)
        self.add_user("keeper")
        r = self.withdraw(username="pend", password=PASSWORD)
        self.assert_withdrawn(r, "pend")
        self.assertIsNotNone(self.user("keeper"))

    def test_wrong_password_is_401(self):
        self.add_user("pend", active=False, pending=True)
        r = self.withdraw(username="pend", password="nope-nope")
        self.assertEqual(r.status_code, 401)
        self.assertIsNotNone(self.user("pend"))

    def test_only_one_credential_is_400(self):
        self.add_user("pend", active=False, pending=True)
        self.assertEqual(self.withdraw(username="pend").status_code, 400)
        self.assertEqual(self.withdraw(password=PASSWORD).status_code, 400)
        self.assertIsNotNone(self.user("pend"))

    def test_active_user_is_409(self):
        self.add_user("alive")
        r = self.withdraw(username="alive", password=PASSWORD)
        self.assertEqual(r.status_code, 409)
        self.assertIsNotNone(self.user("alive"))

    def test_deactivated_user_is_409(self):
        self.add_user("dead", active=False, pending=False)
        r = self.withdraw(username="dead", password=PASSWORD)
        self.assertEqual(r.status_code, 409)
        self.assertIsNotNone(self.user("dead"))

    def test_oauth_cookie_path_works_without_credentials(self):
        user = self.add_user("zpend", password=None, active=False, pending=True,
                             oauth_provider="zoho", oauth_sub="z-1")
        self.add_oauth_row(user.id, "z-1", user.email)
        c = self.client()
        c.cookies.set(PENDING_WITHDRAW_COOKIE, issue_purpose_token(
            PURPOSE_PENDING_WITHDRAW, user_binding_claims(user), timedelta(minutes=15)))
        r = self.withdraw(c)
        self.assert_withdrawn(r, "zpend")
        self.assertEqual(self.oauth_rows(), [])

    def test_oauth_callback_cookie_end_to_end(self):
        from app.auth.zoho_oauth import ZohoUserInfo
        zoho = type("Z", (), {})()
        zoho.get_user_info = AsyncMock(return_value=ZohoUserInfo(
            sub="z-e2e", email="e2e@example.com", email_verified=True, name="E", iss="i", aud="a", exp=0, iat=0))
        with patch("app.auth.zoho_oauth.get_zoho_oauth", return_value=zoho), \
                patch("app.routes.auth.send_signup_webhook", AsyncMock()):
            c = self.client()
            c.cookies.set("zoho_oauth_state", "st")
            r = c.get("/auth/zoho/callback", params={"state": "st", "code": "c"})
        self.assertEqual(r.headers["location"], "/login")
        self.assertIsNotNone(self.user("e2e"))
        self.assert_withdrawn(self.withdraw(c), "e2e")

    def test_no_cookie_and_no_credentials_is_400(self):
        r = self.withdraw()
        self.assertEqual(r.status_code, 400)
        self.assertIn("expired", r.json()["detail"])

    def test_expired_or_wrong_purpose_cookie_is_400(self):
        user = self.add_user("zpend", password=None, active=False, pending=True)
        for token in (
            issue_purpose_token(PURPOSE_PENDING_WITHDRAW, user_binding_claims(user), timedelta(seconds=-5)),
            issue_purpose_token(PURPOSE_OAUTH_LINK, user_binding_claims(user), timedelta(minutes=5)),
        ):
            c = self.client()
            c.cookies.set(PENDING_WITHDRAW_COOKIE, token)
            self.assertEqual(self.withdraw(c).status_code, 400)
        self.assertIsNotNone(self.user("zpend"))

    def test_cookie_for_a_reused_uid_is_400(self):
        old = self.add_user("old", password=None, active=False, pending=True)
        token = issue_purpose_token(PURPOSE_PENDING_WITHDRAW, user_binding_claims(old), timedelta(minutes=5))
        self.delete_user_row(old.id)
        new = self.add_user("new", active=False, pending=True)
        self.assertEqual(new.id, old.id)
        c = self.client()
        c.cookies.set(PENDING_WITHDRAW_COOKIE, token)
        self.assertEqual(self.withdraw(c).status_code, 400)
        self.assertIsNotNone(self.user("new"))

    def test_works_when_self_delete_is_disabled(self):
        from app.auth.models import UserManagementSettings
        async def _disable():
            async with self.Session() as s:
                s.add(UserManagementSettings(id=1, allow_self_delete=False))
                await s.commit()
        self.run_async(_disable())

        async def _allowed():
            from app.auth.database import is_self_delete_allowed
            async with self.Session() as s:
                return await is_self_delete_allowed(s)
        self.assertFalse(self.run_async(_allowed()))

        self.add_user("pend", active=False, pending=True)
        self.assert_withdrawn(self.withdraw(username="pend", password=PASSWORD), "pend")

    def test_requires_csrf(self):
        self.add_user("pend", active=False, pending=True)
        r = self.client().post(WITHDRAW, json={"username": "pend", "password": PASSWORD})
        self.assertEqual(r.status_code, 403)
        self.assertIsNotNone(self.user("pend"))

    def test_delete_failure_is_a_generic_500(self):
        self.add_user("pend", active=False, pending=True)
        with patch("app.routes.pools.delete_user_account", AsyncMock(side_effect=RuntimeError("SQL boom"))):
            with self.assertLogs("app.routes.auth", level="ERROR"):
                r = self.withdraw(username="pend", password=PASSWORD)
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("boom", r.text)
        self.assertIsNotNone(self.user("pend"))


class LoginAccountStatusTests(AuthFlowTestCase):

    def test_login_403_carries_account_status(self):
        self.add_user("pend", active=False, pending=True)
        self.add_user("dead", active=False, pending=False)
        c = self.client()
        r = c.post("/auth/login", json={"username": "pend", "password": PASSWORD})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.headers.get("x-account-status"), "pending")
        self.assertIn("pending admin approval", r.json()["detail"])
        r = c.post("/auth/login", json={"username": "dead", "password": PASSWORD})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.headers.get("x-account-status"), "deactivated")
        self.assertIn("deactivated", r.json()["detail"])

    def test_login_form_403_carries_account_status(self):
        self.add_user("pend", active=False, pending=True)
        self.add_user("dead", active=False, pending=False)
        c = self.client()
        r = c.post("/auth/login/form", data={"username": "pend", "password": PASSWORD})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.headers.get("x-account-status"), "pending")
        r = c.post("/auth/login/form", data={"username": "dead", "password": PASSWORD})
        self.assertEqual(r.headers.get("x-account-status"), "deactivated")

    def test_wrong_password_has_no_account_status(self):
        self.add_user("pend", active=False, pending=True)
        r = self.client().post("/auth/login", json={"username": "pend", "password": "wrong-pw"})
        self.assertEqual(r.status_code, 401)
        self.assertIsNone(r.headers.get("x-account-status"))

    def test_successful_login_sets_session_cookie_with_grace(self):
        from app.auth.auth import ACCESS_TOKEN_EXPIRE_MINUTES
        from app.auth.cookies import SESSION_COOKIE_GRACE_SECONDS
        self.add_user("alive")
        r = self.client().post("/auth/login", json={"username": "alive", "password": PASSWORD})
        self.assertEqual(r.status_code, 200)
        [header] = set_cookie_headers(r, "access_token")
        self.assertIn(f"Max-Age={ACCESS_TOKEN_EXPIRE_MINUTES * 60 + SESSION_COOKIE_GRACE_SECONDS}", header)
        self.assertIn("HttpOnly", header)
        self.assertIn("Path=/", header)


if __name__ == "__main__":
    unittest.main()

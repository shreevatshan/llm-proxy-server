"""Tests for the explicit ZOHO account-link page and endpoints.

The link is only made after the existing account's password is confirmed, and it
never changes the account's status: an active account is signed in, a pending one
stays pending (with the withdraw option). The link request lives in a signed,
user-bound cookie, so a tampered, expired or reused-id request is refused.
"""

import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from app.auth.cookies import OAUTH_LINK_COOKIE, PENDING_WITHDRAW_COOKIE, SESSION_COOKIE
from app.auth.purpose_tokens import (
    PURPOSE_FLASH, PURPOSE_OAUTH_LINK, issue_purpose_token, user_binding_claims,
)

from tests.auth_flow_base import (
    ADMIN_EMAIL, PASSWORD, AuthFlowTestCase, cookie_cleared, cookie_set, flash_claims,
)


def link_token(user, sub="z-link", email=None, ttl=timedelta(minutes=10), **overrides):
    claims = user_binding_claims(user)
    claims.update({"provider": "zoho", "psub": sub, "pemail": email or user.email, "pname": "Zed"})
    claims.update(overrides)
    return issue_purpose_token(PURPOSE_OAUTH_LINK, claims, ttl)


class OAuthLinkTests(AuthFlowTestCase):

    def link_client(self, user, **kwargs):
        c = self.client()
        c.cookies.set(OAUTH_LINK_COOKIE, link_token(user, **kwargs))
        return c

    def confirm(self, c, password=PASSWORD):
        return self.post(c, "/auth/zoho/link", json={"password": password})

    # -- page --------------------------------------------------------------

    def test_link_page_renders_masked_details(self):
        bob = self.add_user("bobby", "bob@example.com")
        r = self.link_client(bob).get("/auth/zoho/link")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertIn('id="authAlert"', r.text)
        self.assertIn("bo***", r.text)
        self.assertIn("b***@example.com", r.text)
        self.assertNotIn("bob@example.com", r.text)

    def test_link_page_without_a_valid_request_redirects_with_flash(self):
        r = self.client().get("/auth/zoho/link")
        self.assertEqual(r.status_code, 303)
        self.assertEqual(r.headers["location"], "/login")
        self.assertIn("expired", flash_claims(r)["msg"])

    # -- confirm -------------------------------------------------------------

    def test_wrong_password_is_401_and_keeps_the_cookie(self):
        bob = self.add_user("bobby", "bob@example.com")
        c = self.link_client(bob)
        r = self.confirm(c, "wrong-password")
        self.assertEqual(r.status_code, 401)
        self.assertFalse(cookie_cleared(r, OAUTH_LINK_COOKIE))
        self.assertIsNotNone(c.cookies.get(OAUTH_LINK_COOKIE))
        self.assertEqual(self.oauth_rows(), [])
        # The kept cookie still allows a retry with the right password.
        r = self.confirm(c)
        self.assertEqual(r.status_code, 200)

    def test_active_target_is_linked_and_signed_in(self):
        bob = self.add_user("bobby", "bob@example.com")
        c = self.link_client(bob, sub="z-bob")
        r = self.confirm(c)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["redirect"], "/dashboard/")
        self.assertTrue(cookie_set(r, SESSION_COOKIE))
        self.assertTrue(cookie_cleared(r, OAUTH_LINK_COOKIE))
        [row] = self.oauth_rows("z-bob")
        self.assertEqual(row.user_id, bob.id)
        refreshed = self.user("bobby")
        self.assertEqual((refreshed.oauth_provider, refreshed.oauth_sub), ("zoho", "z-bob"))
        self.assertTrue(refreshed.is_active)

    def test_pending_target_is_linked_but_stays_pending(self):
        pat = self.add_user("patty", "pat@example.com", active=False, pending=True)
        c = self.link_client(pat, sub="z-pat")
        r = self.confirm(c)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["redirect"], "/login")
        self.assertFalse(cookie_set(r, SESSION_COOKIE))
        self.assertTrue(cookie_set(r, PENDING_WITHDRAW_COOKIE))
        self.assertTrue(cookie_cleared(r, OAUTH_LINK_COOKIE))
        flash = flash_claims(r)
        self.assertEqual(flash["lvl"], "warning")
        self.assertEqual(flash["act"], ["withdraw_pending"])
        self.assertEqual(flash["src"], "oauth")
        self.assertIn("linked", flash["msg"])
        self.assertIn("pending", flash["msg"])
        self.assertEqual(len(self.oauth_rows("z-pat")), 1)
        refreshed = self.user("patty")
        self.assertFalse(refreshed.is_active)
        self.assertTrue(refreshed.is_pending_approval)

    def test_pending_linked_zohocorp_account_stays_pending(self):
        emp = self.add_user("emp", "emp@zohocorp.com", active=False, pending=True)
        r = self.confirm(self.link_client(emp, sub="z-emp"))
        self.assertEqual(r.json()["redirect"], "/login")
        self.assertFalse(self.user("emp").is_active)

    def test_tampered_token_is_400(self):
        bob = self.add_user("bobby", "bob@example.com")
        c = self.client()
        token = link_token(bob)
        header, payload, sig = token.split(".")
        c.cookies.set(OAUTH_LINK_COOKIE, f"{header}.{payload}.{sig[::-1]}")
        r = self.confirm(c)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.oauth_rows(), [])

    def test_expired_token_is_400(self):
        bob = self.add_user("bobby", "bob@example.com")
        r = self.confirm(self.link_client(bob, ttl=timedelta(seconds=-5)))
        self.assertEqual(r.status_code, 400)
        self.assertIn("expired", r.json()["detail"])

    def test_wrong_purpose_token_is_400(self):
        bob = self.add_user("bobby", "bob@example.com")
        c = self.client()
        claims = user_binding_claims(bob)
        claims.update({"provider": "zoho", "psub": "z", "pemail": bob.email})
        c.cookies.set(OAUTH_LINK_COOKIE, issue_purpose_token(PURPOSE_FLASH, claims, timedelta(minutes=5)))
        self.assertEqual(self.confirm(c).status_code, 400)

    def test_reused_uid_is_400(self):
        old = self.add_user("olduser", "shared@example.com")
        token = link_token(old)
        self.delete_user_row(old.id)
        new = self.add_user("newuser", "shared@example.com")
        self.assertEqual(new.id, old.id)  # SQLite reuses the rowid
        c = self.client()
        c.cookies.set(OAUTH_LINK_COOKIE, token)
        r = self.confirm(c)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.oauth_rows(), [])

    def test_email_changed_since_request_is_400(self):
        bob = self.add_user("bobby", "bob@example.com")
        r = self.confirm(self.link_client(bob, email="someone-else@example.com"))
        self.assertEqual(r.status_code, 400)

    def test_double_link_is_409(self):
        bob = self.add_user("bobby", "bob@example.com")
        token = link_token(bob, sub="z-bob")
        c = self.client()
        c.cookies.set(OAUTH_LINK_COOKIE, token)
        self.assertEqual(self.confirm(c).status_code, 200)
        c2 = self.client()
        c2.cookies.set(OAUTH_LINK_COOKIE, token)
        r = self.confirm(c2)
        self.assertEqual(r.status_code, 409)
        self.assertEqual(len(self.oauth_rows("z-bob")), 1)

    def test_identity_already_linked_elsewhere_is_409(self):
        other = self.add_user("other", "other@example.com", oauth_provider="zoho", oauth_sub="z-shared")
        self.add_oauth_row(other.id, "z-shared", "other@example.com")
        bob = self.add_user("bobby", "bob@example.com")
        r = self.confirm(self.link_client(bob, sub="z-shared"))
        self.assertEqual(r.status_code, 409)
        self.assertIsNone(self.user("bobby").oauth_provider)

    def test_confirm_requires_csrf(self):
        bob = self.add_user("bobby", "bob@example.com")
        c = self.link_client(bob)
        r = c.post("/auth/zoho/link", json={"password": PASSWORD})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.oauth_rows(), [])

    # -- decline -------------------------------------------------------------

    def decline(self, user):
        c = self.link_client(user)
        r = self.post(c, "/auth/zoho/link/decline")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["redirect"], "/login")
        self.assertTrue(cookie_cleared(r, OAUTH_LINK_COOKIE))
        self.assertEqual(self.oauth_rows(), [])
        flash = flash_claims(r)
        self.assertEqual(flash["lvl"], "info")
        self.assertIn("No account was linked or created", flash["msg"])
        self.assertIn(ADMIN_EMAIL, flash["msg"])
        return flash["msg"]

    def test_decline_creates_no_user(self):
        bob = self.add_user("bobby", "bob@example.com")
        self.decline(bob)
        self.assertEqual([u.username for u in self.users()], ["bobby"])

    def test_decline_text_for_pending_account(self):
        pat = self.add_user("patty", "pat@example.com", active=False, pending=True)
        self.assertIn("Withdraw signup", self.decline(pat))

    def test_decline_text_for_active_account_with_self_delete(self):
        bob = self.add_user("bobby", "bob@example.com")
        self.assertIn("Profile", self.decline(bob))

    def test_decline_text_for_active_account_without_self_delete(self):
        bob = self.add_user("bobby", "bob@example.com")
        with patch("app.routes.auth.is_self_delete_allowed", AsyncMock(return_value=False)):
            message = self.decline(bob)
        self.assertNotIn("Profile", message)
        self.assertIn(f"contact {ADMIN_EMAIL} to remove", message)

    def test_decline_text_for_deactivated_account(self):
        dead = self.add_user("dead", "dead@example.com", active=False, pending=False)
        message = self.decline(dead)
        self.assertNotIn("Withdraw", message)
        self.assertIn("to remove the existing account", message)


if __name__ == "__main__":
    unittest.main()

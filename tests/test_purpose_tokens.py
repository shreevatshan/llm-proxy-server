"""Tests for the single-purpose signed tokens (OAuth link, pending withdraw, flash).

A purpose token is signed with the session key, so the property that matters most
is that it can never be replayed as a session: ``verify_token`` must refuse every
purpose token, and a forged token that carries both ``sub`` and an ``aud`` too.
An expired session must still be told apart from a missing one (X-Auth-Error).
"""

import os
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-not-placeholder-abc123")

from fastapi import HTTPException
from jose import jwt

from app.auth.auth import ALGORITHM, SECRET_KEY, create_access_token, verify_token
from app.auth.purpose_tokens import (
    PURPOSE_FLASH, PURPOSE_OAUTH_LINK, PURPOSE_PENDING_WITHDRAW,
    claims_match_user, issue_purpose_token, read_purpose_token, user_binding_claims,
)

ALL_PURPOSES = (PURPOSE_OAUTH_LINK, PURPOSE_PENDING_WITHDRAW, PURPOSE_FLASH)


def _user(uid=7, username="alice", created_at=datetime(2026, 1, 2, 3, 4, 5)):
    return SimpleNamespace(id=uid, username=username, created_at=created_at)


class PurposeTokenTests(unittest.TestCase):

    def test_round_trip_keeps_claims_and_scopes_audience(self):
        token = issue_purpose_token(PURPOSE_OAUTH_LINK, {"uid": 7, "x": "y"}, timedelta(minutes=5))
        claims = read_purpose_token(token, PURPOSE_OAUTH_LINK)
        self.assertEqual(claims["uid"], 7)
        self.assertEqual(claims["x"], "y")
        self.assertEqual(claims["purpose"], PURPOSE_OAUTH_LINK)
        self.assertEqual(claims["aud"], f"llmproxy:{PURPOSE_OAUTH_LINK}")
        self.assertNotIn("sub", claims)

    def test_wrong_purpose_returns_none(self):
        token = issue_purpose_token(PURPOSE_OAUTH_LINK, {"uid": 7}, timedelta(minutes=5))
        self.assertIsNone(read_purpose_token(token, PURPOSE_PENDING_WITHDRAW))
        self.assertIsNone(read_purpose_token(token, PURPOSE_FLASH))

    def test_matching_aud_but_wrong_purpose_claim_returns_none(self):
        payload = {
            "purpose": PURPOSE_FLASH, "aud": f"llmproxy:{PURPOSE_OAUTH_LINK}",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        self.assertIsNone(read_purpose_token(token, PURPOSE_OAUTH_LINK))

    def test_expired_returns_none(self):
        token = issue_purpose_token(PURPOSE_FLASH, {"msg": "hi"}, timedelta(seconds=-10))
        self.assertIsNone(read_purpose_token(token, PURPOSE_FLASH))

    def test_missing_malformed_and_tampered_return_none(self):
        self.assertIsNone(read_purpose_token(None, PURPOSE_FLASH))
        self.assertIsNone(read_purpose_token("", PURPOSE_FLASH))
        self.assertIsNone(read_purpose_token("not-a-jwt", PURPOSE_FLASH))
        token = issue_purpose_token(PURPOSE_FLASH, {"msg": "hi"}, timedelta(minutes=1))
        header, _payload, signature = token.split(".")
        forged_payload = jwt.encode(
            {"msg": "pwned", "purpose": PURPOSE_FLASH, "aud": f"llmproxy:{PURPOSE_FLASH}",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=1)},
            "x", algorithm=ALGORITHM,
        ).split(".")[1]
        self.assertIsNone(read_purpose_token(f"{header}.{forged_payload}.{signature}", PURPOSE_FLASH))
        other_key = jwt.encode(
            {"msg": "hi", "purpose": PURPOSE_FLASH, "aud": f"llmproxy:{PURPOSE_FLASH}",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=1)},
            "some-other-key", algorithm=ALGORITHM,
        )
        self.assertIsNone(read_purpose_token(other_key, PURPOSE_FLASH))

    def test_issuer_refuses_session_claims(self):
        for forbidden in ({"sub": "alice"}, {"is_admin": True}):
            with self.assertRaises(AssertionError):
                issue_purpose_token(PURPOSE_FLASH, forbidden, timedelta(minutes=1))

    def test_reader_refuses_a_forged_purpose_token_with_sub(self):
        token = jwt.encode(
            {"sub": "alice", "purpose": PURPOSE_FLASH, "aud": f"llmproxy:{PURPOSE_FLASH}",
             "exp": datetime.now(timezone.utc) + timedelta(minutes=1)},
            SECRET_KEY, algorithm=ALGORITHM,
        )
        self.assertIsNone(read_purpose_token(token, PURPOSE_FLASH))

    def test_user_binding_matches_only_the_same_row(self):
        user = _user()
        claims = user_binding_claims(user)
        self.assertEqual(claims, {"uid": 7, "uname": "alice", "ucat": "2026-01-02T03:04:05"})
        self.assertTrue(claims_match_user(claims, user))
        # SQLite id reuse: same id, different account.
        self.assertFalse(claims_match_user(claims, _user(username="mallory")))
        self.assertFalse(claims_match_user(claims, _user(created_at=datetime(2026, 5, 5))))
        self.assertFalse(claims_match_user(claims, _user(uid=8)))
        self.assertFalse(claims_match_user(claims, None))
        self.assertFalse(claims_match_user({}, user))

    def test_binding_survives_a_token_round_trip(self):
        user = _user()
        token = issue_purpose_token(PURPOSE_PENDING_WITHDRAW, user_binding_claims(user), timedelta(minutes=5))
        self.assertTrue(claims_match_user(read_purpose_token(token, PURPOSE_PENDING_WITHDRAW), user))


class VerifyTokenTests(unittest.TestCase):

    def _assert_rejected(self, token):
        with self.assertRaises(HTTPException) as ctx:
            verify_token(token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Could not validate credentials")
        return ctx.exception

    def test_session_token_still_verifies(self):
        data = verify_token(create_access_token({"sub": "alice"}))
        self.assertEqual(data.username, "alice")
        self.assertFalse(data.is_admin)

    def test_every_purpose_token_is_rejected_as_a_session(self):
        for purpose in ALL_PURPOSES:
            with self.subTest(purpose=purpose):
                token = issue_purpose_token(purpose, {"uid": 1, "uname": "alice"}, timedelta(minutes=5))
                exc = self._assert_rejected(token)
                self.assertNotIn("X-Auth-Error", exc.headers or {})

    def test_forged_sub_plus_aud_token_is_rejected(self):
        for extra in ({"aud": "llmproxy:oauth_link"}, {"purpose": "oauth_link"},
                      {"aud": "llmproxy:auth_flash", "purpose": "auth_flash", "is_admin": True}):
            with self.subTest(extra=extra):
                payload = {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=5), **extra}
                self._assert_rejected(jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM))

    def test_expired_session_is_flagged_session_expired(self):
        token = create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-10))
        exc = self._assert_rejected(token)
        self.assertEqual(exc.headers, {"WWW-Authenticate": "Bearer", "X-Auth-Error": "session_expired"})

    def test_garbage_token_is_not_flagged_expired(self):
        exc = self._assert_rejected("garbage")
        self.assertEqual(exc.headers, {"WWW-Authenticate": "Bearer"})


if __name__ == "__main__":
    unittest.main()

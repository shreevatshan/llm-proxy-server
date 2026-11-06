"""Encryption at rest for BYOK provider secrets.

Users hand the proxy their own upstream credentials, so those are stored
encrypted (admin-entered secrets are unchanged). Fernet (AES-128-CBC + HMAC) from
``cryptography``, which python-jose[cryptography] already installs.

Key: ``LLMPROXY_BYOK_ENCRYPTION_KEY`` (any string; it is stretched with HKDF) or,
when unset, derived with HKDF from ``JWT_SECRET_KEY`` under a BYOK-specific info
label. Rotating the source secret makes stored BYOK secrets unreadable: those
instances are skipped at load and the owner is asked to re-enter them.

Derived lazily and cached per source value, so tests (and a changed env) get a
fresh key without a restart of the module.
"""

import base64
import os
from typing import Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PREFIX = "enc:v1:"
_INFO = b"llmproxy-byok-v1"

# The provider_credentials columns that hold secrets (mirrors admin._MASKED_SECRET_FIELDS).
SECRET_FIELDS = ("api_key", "secret_access_key", "oauth_client_secret", "oauth_refresh_token")

_cached: Optional[Tuple[str, Fernet]] = None


class SecretDecryptError(Exception):
    """A stored BYOK secret could not be decrypted (wrong or rotated key)."""


def _source_secret() -> str:
    source = os.getenv("LLMPROXY_BYOK_ENCRYPTION_KEY") or os.getenv("JWT_SECRET_KEY") or ""
    if not source:
        raise RuntimeError("No key material for BYOK secret encryption (set JWT_SECRET_KEY)")
    return source


def _fernet() -> Fernet:
    global _cached
    source = _source_secret()
    if _cached is not None and _cached[0] == source:
        return _cached[1]
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_INFO).derive(source.encode("utf-8"))
    fernet = Fernet(base64.urlsafe_b64encode(key))
    _cached = (source, fernet)
    return fernet


def is_encrypted(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def encrypt(value: Optional[str]) -> Optional[str]:
    """Encrypt a secret; None/empty pass through, an already-encrypted value is kept."""
    if value is None or value == "" or is_encrypted(value):
        return value
    token = _fernet().encrypt(value.encode("utf-8")).decode("ascii")
    return PREFIX + token


def decrypt(value: Optional[str]) -> Optional[str]:
    """Decrypt a stored secret; a value without the prefix is returned unchanged."""
    if not is_encrypted(value):
        return value
    try:
        return _fernet().decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as e:
        raise SecretDecryptError("stored secret cannot be decrypted with the current key") from e


def encrypt_fields(data: dict) -> dict:
    """Return ``data`` with every present, non-empty secret field encrypted."""
    out = dict(data)
    for field in SECRET_FIELDS:
        if out.get(field):
            out[field] = encrypt(out[field])
    return out


def can_decrypt(cred) -> bool:
    """Whether every stored secret on a credentials row decrypts with the current key."""
    try:
        for field in SECRET_FIELDS:
            decrypt(getattr(cred, field, None))
        return True
    except SecretDecryptError:
        return False

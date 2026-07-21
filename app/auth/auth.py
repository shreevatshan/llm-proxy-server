"""Authentication utilities and JWT handling."""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status
from .models import TokenData

# JWT Configuration
# Fail fast if the signing key is unset or still a shipped placeholder: a known
# key lets anyone forge admin tokens.
_PLACEHOLDER_SECRETS = frozenset({
    "your-secret-key-change-this-in-production",
    "your-secret-key-here",  # shipped in .env.example
})
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY or SECRET_KEY in _PLACEHOLDER_SECRETS:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable must be set to a strong, unique value. "
        "Refusing to start with an unset or placeholder signing key."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, is_admin: bool = False):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    # Add admin flag to token
    if is_admin:
        to_encode.update({"is_admin": True})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify JWT token and return token data."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        is_admin: bool = payload.get("is_admin", False)
        token_data = TokenData(username=username, is_admin=is_admin)
    except JWTError:
        raise credentials_exception
    
    return token_data

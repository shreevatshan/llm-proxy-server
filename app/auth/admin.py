"""Admin user class and utilities for config-based authentication."""

from typing import Optional
from datetime import datetime
from app.config import config
from .database import verify_password, get_password_hash


class AdminUser:
    """Admin user class that doesn't use database storage."""
    
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email
        self.is_admin = True
        self.is_active = True
        self.created_at = datetime.utcnow()
        # Admin user doesn't have an ID since it's not stored in DB
        self.id = None


def get_admin_config():
    """Get admin configuration from config."""
    return config.admin


def authenticate_admin(username: str, password: str) -> Optional[AdminUser]:
    """Authenticate admin user against config credentials."""
    admin_config = get_admin_config()
    
    # Check if admin is enabled
    if not admin_config.enabled:
        return None
    
    # Check username match
    if username != admin_config.username:
        return None
    
    # For admin, we'll do a simple password comparison
    # In production, you might want to hash the admin password in config
    if password != admin_config.password:
        return None
    
    # Return admin user object
    return AdminUser(admin_config.username, admin_config.email)


def is_admin_enabled() -> bool:
    """Check if admin account is enabled."""
    return get_admin_config().enabled


def get_admin_username() -> str:
    """Get admin username from config."""
    return get_admin_config().username


def get_admin_email() -> str:
    """Get admin email from config."""
    return get_admin_config().email

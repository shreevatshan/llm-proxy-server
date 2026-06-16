"""Helper for per-model-group rate limit enforcement, called from route handlers."""

from typing import Optional, Union

from fastapi import Request

from app.auth.admin import AdminUser
from app.auth.models import APIKey, User
from app.rate_limit import RateLimitExceeded, rate_limit_tracker


def _envelope_for(path: str, override: Optional[str]) -> str:
    if override:
        return override
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/openai/"):
        return "azure"
    return "openai"


async def enforce_group_rate_limit(
    request: Request,
    auth: Union[User, AdminUser, APIKey],
    model_id: str,
    envelope_override: Optional[str] = None,
) -> None:
    """Check model-group limits for the authenticated user and model.

    Request-level RPM/RPD limits are already enforced in auth middleware.
    This function checks only the group limits for the resolved model.
    Raises RateLimitExceeded if the group limit is hit.
    """
    if isinstance(auth, AdminUser):
        return
    if not model_id:
        return

    user_id = getattr(auth, "user_id", None) or getattr(auth, "id", None)
    username = (
        getattr(auth, "username", None)
        or (f"key:{auth.id}" if hasattr(auth, "id") else None)
    )
    if user_id is None or username is None:
        return

    decision = await rate_limit_tracker.check_group_limit(user_id, username, model_id)
    if decision is not None and not decision.allowed:
        envelope = _envelope_for(request.url.path, envelope_override)
        if envelope == "anthropic":
            raise RateLimitExceeded.anthropic(decision)
        if envelope == "azure":
            raise RateLimitExceeded.azure(decision)
        raise RateLimitExceeded.openai(decision)

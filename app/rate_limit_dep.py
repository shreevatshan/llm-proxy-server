"""Helper for per-group rate limit enforcement, called from route handlers.

Precedence (most-specific wins): instance group > model group > overall.
If the request's instance (provider_key) belongs to an instance group, only the
instance-group limit applies and the model-group check is skipped. Otherwise the
model-group limit applies. The overall request limit is handled in auth middleware
and skipped there whenever either group governs the request.
"""

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


def _provider_key_of(model_id: str) -> Optional[str]:
    """Extract the instance identifier (provider_key) from a model id.

    Model ids are '{provider_key}/{model_name}' (e.g. 'azure:primary/gpt-4').
    """
    if model_id and "/" in model_id:
        return model_id.split("/", 1)[0]
    return None


async def enforce_group_rate_limit(
    request: Request,
    auth: Union[User, AdminUser, APIKey],
    model_id: str,
    envelope_override: Optional[str] = None,
) -> None:
    """Check group limits for the authenticated user, with instance-group precedence.

    Request-level RPM/RPD limits are already enforced in auth middleware.
    If the model's instance belongs to an instance group, only that limit is
    checked; otherwise the model-group limit is checked.
    Raises RateLimitExceeded if a group limit is hit.
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

    # Instance group takes precedence over model group.
    provider_key = _provider_key_of(model_id)
    if provider_key and rate_limit_tracker.instance_belongs_to_group(provider_key):
        decision = await rate_limit_tracker.check_instance_group_limit(user_id, username, provider_key)
    else:
        decision = await rate_limit_tracker.check_group_limit(user_id, username, model_id)

    if decision is not None and not decision.allowed:
        envelope = _envelope_for(request.url.path, envelope_override)
        if envelope == "anthropic":
            raise RateLimitExceeded.anthropic(decision)
        if envelope == "azure":
            raise RateLimitExceeded.azure(decision)
        raise RateLimitExceeded.openai(decision)

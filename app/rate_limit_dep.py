"""Access check and rate limiting for a resolved model, called from route handlers.

Every model belongs to exactly one model group: a named group that claims it (an
explicit model member beats an instance member), or else the built-in Default group,
whose limits are the profile's own rpm/rpd. See rate_limit_tracker.group_for.

Routes call enforce_access_and_limits once the model is resolved. It checks access
first, so a request denied with 403 never spends quota, then charges exactly one group.
Endpoints that do this are marked @enforces_limits, and the auth middleware leaves them
alone; anything unmarked is charged to the Default group in the middleware instead.
"""

from typing import Callable, Optional, TypeVar, Union

from fastapi import Request

from app.api_envelope import envelope_for
from app.auth.admin import AdminUser
from app.auth.models import APIKey, User
from app.concurrency_slots import charge_with_slot
from app.model_access_dep import enforce_model_access
from app.profiles import DEFAULT_GROUP_ID
from app.rate_limit import RateLimitExceeded, rate_limit_tracker

F = TypeVar("F", bound=Callable)

ROUTE_ENFORCES_LIMITS = "_route_enforces_limits"


def enforces_limits(endpoint: F) -> F:
    """Mark an endpoint that calls enforce_access_and_limits itself.

    The auth middleware reads the mark off request.scope["endpoint"] and skips its own
    Default-group charge, so the request is charged once, after the model is resolved
    and access is checked. Deciding by endpoint rather than by whether the tracking
    middleware sniffed a model matters: it never does for large multipart uploads or
    count_tokens, both of which still enforce here.
    """
    setattr(endpoint, ROUTE_ENFORCES_LIMITS, True)
    return endpoint


def _raise(request: Request, decision, envelope_override: Optional[str]) -> None:
    envelope = envelope_for(request.url.path, envelope_override)
    if envelope == "anthropic":
        raise RateLimitExceeded.anthropic(decision)
    if envelope == "azure":
        raise RateLimitExceeded.azure(decision)
    raise RateLimitExceeded.openai(decision)


async def enforce_group_rate_limit(
    request: Request,
    auth: Union[User, AdminUser, APIKey],
    model_id: str,
    envelope_override: Optional[str] = None,
) -> None:
    """Charge the request to the group governing `model_id` -- a named group, or Default.

    Raises RateLimitExceeded if that group's limit is hit. Prefer
    enforce_access_and_limits, which runs the access check first.
    """
    if isinstance(auth, AdminUser):
        return

    user_id = getattr(auth, "user_id", None) or getattr(auth, "id", None)
    username = (
        getattr(auth, "username", None)
        or (f"key:{auth.id}" if hasattr(auth, "id") else None)
    )
    if user_id is None or username is None:
        return

    if rate_limit_tracker.group_for(model_id) == DEFAULT_GROUP_ID:
        charge = lambda: rate_limit_tracker.check_and_increment(user_id, username)
    else:
        charge = lambda: rate_limit_tracker.check_group_limit(user_id, username, model_id)

    decision = await charge_with_slot(request, user_id, charge)
    if decision is not None:
        _raise(request, decision, envelope_override)


async def enforce_access_and_limits(
    request: Request,
    auth: Union[User, AdminUser, APIKey],
    model_id: str,
    envelope_override: Optional[str] = None,
    consume_quota: bool = True,
) -> None:
    """Deny (403) if the caller may not use the model, then charge its group (429).

    consume_quota=False checks access only, for endpoints that do no upstream work
    worth metering (count_tokens).
    """
    await enforce_model_access(request, auth, model_id, envelope_override=envelope_override)
    if consume_quota:
        await enforce_group_rate_limit(request, auth, model_id, envelope_override=envelope_override)

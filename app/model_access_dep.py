"""Per-user model access enforcement, called from route handlers.

Mirrors app/rate_limit_dep.py: same call site (right after the group
rate-limit check) so `auth` and the request model are already in scope.

Access decision (see app/cache.py:is_model_allowed_for_user), resolved by the
user's access mode:
  - allow  : always allowed, even for a globally disabled model/provider
             (the per-user allow mode overrides the global gate).
  - deny   : always denied.
  - custom : an explicit per-model exception wins; otherwise the global gate applies.
  - default: follow the global gate (globally disabled model/provider -> denied).
AdminUser bypasses all checks.
"""

from typing import Optional, Union

from fastapi import Request

from app.api_envelope import envelope_for
from app.auth.admin import AdminUser
from app.auth.models import APIKey, User
from app.providers.provider_manager import provider_manager


class ModelAccessDenied(Exception):
    """Raised when a user is not permitted to use the requested model.

    Caught by a per-app exception handler (registered in app/main.py) which
    serializes `body` with `status_code`. Mirrors RateLimitExceeded.
    """

    def __init__(self, body: dict, status_code: int = 403):
        self.body = body
        self.status_code = status_code
        super().__init__("Model access denied")

    @classmethod
    def openai(cls, model_id: str) -> "ModelAccessDenied":
        return cls(body={"error": {
            "message": _message(model_id),
            "type": "permission_error",
            "code": "model_access_denied",
            "param": None,
        }})

    @classmethod
    def azure(cls, model_id: str) -> "ModelAccessDenied":
        # Azure uses the OpenAI error envelope.
        return cls(body={"error": {
            "message": _message(model_id),
            "type": "permission_error",
            "code": "model_access_denied",
            "param": None,
        }})

    @classmethod
    def anthropic(cls, model_id: str) -> "ModelAccessDenied":
        return cls(body={"type": "error", "error": {
            "type": "permission_error",
            "message": _message(model_id),
        }})


def _message(model_id: str) -> str:
    return (
        f"Model '{model_id}' is not enabled for your account. "
        f"Contact your administrator to request access."
    )


async def enforce_model_access(
    request: Request,
    auth: Union[User, AdminUser, APIKey],
    model_id: str,
    envelope_override: Optional[str] = None,
) -> None:
    """Deny the request if the authenticated user may not use `model_id`.

    Raises ModelAccessDenied (403) in the format matching the API envelope.
    """
    if isinstance(auth, AdminUser):
        return
    if not model_id:
        return

    user_id = getattr(auth, "user_id", None) or getattr(auth, "id", None)
    if user_id is None:
        return

    if provider_manager.model_cache.is_model_allowed_for_user(user_id, model_id):
        return

    envelope = envelope_for(request.url.path, envelope_override)
    if envelope == "anthropic":
        raise ModelAccessDenied.anthropic(model_id)
    if envelope == "azure":
        raise ModelAccessDenied.azure(model_id)
    raise ModelAccessDenied.openai(model_id)

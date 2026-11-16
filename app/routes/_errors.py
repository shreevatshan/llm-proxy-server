"""Shared helpers for surfacing upstream provider errors from route handlers.

When the provider layer raises :class:`ProviderHTTPError`, routes should relay
the *upstream* HTTP status (and a safe subset of headers such as ``Retry-After``)
rather than collapsing everything to a generic 500. This module centralises the
header allowlist and the per-protocol JSON envelope builders.
"""

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.providers.base import ProviderHTTPError

# Response headers that are safe to relay from the upstream provider. Notably
# includes Retry-After and the rate-limit family so clients can back off.
# Deliberately excludes content-type: the body we send is always the
# re-serialized JSON envelope (JSONResponse sets application/json itself),
# so relaying an upstream content-type (e.g. text/html from an HTML 502
# page) would mislabel it.
SAFE_UPSTREAM_HEADERS = {
    "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests", "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
}


def relay_headers(headers, allowed=SAFE_UPSTREAM_HEADERS) -> Dict[str, str]:
    """The upstream response headers in *allowed* (case-insensitive)."""
    return {
        k: v for k, v in (headers or {}).items()
        if isinstance(k, str) and isinstance(v, str) and k.lower() in allowed
    }


def _safe_headers(error: ProviderHTTPError) -> Dict[str, str]:
    return relay_headers(error.headers)


def openai_provider_error_response(error: ProviderHTTPError) -> JSONResponse:
    """Build a JSONResponse preserving the upstream status for OpenAI-format routes.

    Uses the upstream error body verbatim when it is a dict (already an OpenAI
    ``{"error": {...}}`` envelope); otherwise wraps ``error.message`` in the
    OpenAI error shape.
    """
    body = error.body
    if not isinstance(body, dict):
        body = {
            "error": {
                "message": error.message,
                "type": "api_error",
            }
        }
    return JSONResponse(
        status_code=error.status_code,
        content=body,
        headers=_safe_headers(error),
    )


def openai_error_response(
    status_code: int,
    message: str,
    *,
    type: str,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    """A locally raised error in the OpenAI ``{"error": {...}}`` envelope.

    The OpenAI, Anthropic and Azure apps register no HTTPException handler, so
    ``HTTPException`` reaches clients as ``{"detail": ...}``, which OpenAI SDKs
    cannot parse. (The TypeSafe app does register one; see the TypeSafe helpers
    below.)
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": type, "param": param, "code": code}},
    )


def azure_provider_error_response(error: ProviderHTTPError) -> JSONResponse:
    """Build a JSONResponse preserving the upstream status for Azure OpenAI routes.

    Relays the upstream error body verbatim when it is a dict; otherwise wraps
    ``error.message`` in the Azure ``{"error": {"code", "message"}}`` shape.
    """
    body = error.body
    if not isinstance(body, dict):
        body = {
            "error": {
                "code": str(error.status_code),
                "message": error.message,
            }
        }
    return JSONResponse(
        status_code=error.status_code,
        content=body,
        headers=_safe_headers(error),
    )


# ==================== TypeSafe (System One) envelope ====================
#
# The TypeSafe app (port 11366, /typesafe on the management port) answers in
# TypeSafe's own shape: {"detail": {"error_type": ..., "message": ...}}, with a
# FastAPI detail[] list for 422 and an x-typesafe-request-id header on every
# /v1/systemone and /v1/models response and every error. Shared code keeps raising OpenAI-shaped bodies (envelope_for picks
# "openai" for /v1/* paths); the TypeSafe app's exception handlers convert them,
# so the other ports are unaffected.

# Response headers relayed from a System One upstream (TypeSafe app only; the
# other apps keep SAFE_UPSTREAM_HEADERS). Includes the unsuffixed rate-limit
# names some gateways send.
TYPESAFE_RELAYED_HEADERS = frozenset(
    SAFE_UPSTREAM_HEADERS | {
        "retry-after-ms", "x-typesafe-request-id",
        "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    }
)
TYPESAFE_REQUEST_ID_HEADER = "x-typesafe-request-id"

# error_type per status. Only authentication_error and api_usage_error are
# confirmed against api.typesafe.ai; the rest follow the same naming. Correct
# them here, in one place, if TypeSafe documents other names.
_TS_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "api_usage_error",
    413: "request_too_large",
    415: "api_usage_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def typesafe_error_type(status_code: int) -> str:
    return _TS_TYPES.get(
        status_code, "api_error" if status_code >= 500 else "invalid_request_error"
    )


def _message_of(body: Any) -> Optional[str]:
    """The human-readable message in an OpenAI/Anthropic/TypeSafe-ish error body."""
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    for key in ("error", "detail", "message"):
        value = body.get(key)
        if isinstance(value, dict):
            value = value.get("message")
        if isinstance(value, str) and value:
            return value
    return None


def to_typesafe_body(body: Any, status_code: int, default_message: Optional[str] = None) -> Dict[str, Any]:
    """Convert an error body to TypeSafe's ``{"detail": ...}`` shape.

    A body that already carries ``detail`` is passed through unchanged (it is
    TypeSafe-shaped, including a 422 detail[] list).
    """
    if isinstance(body, dict) and "detail" in body:
        return body
    message = _message_of(body) or default_message or f"Request failed with status {status_code}"
    return {"detail": {"error_type": typesafe_error_type(status_code), "message": message}}


def typesafe_request_id(request: Optional[Request] = None) -> str:
    """``req_<32 hex>``: the tracker's request id when there is one."""
    tracking_id = None
    if request is not None:
        tracking_id = getattr(getattr(request, "state", None), "tracking_request_id", None)
    if isinstance(tracking_id, str) and len(tracking_id) == 32:
        try:
            int(tracking_id, 16)
            return f"req_{tracking_id}"
        except ValueError:
            pass
    return f"req_{uuid.uuid4().hex}"


def typesafe_json_response(status_code: int, body: Any, headers: Optional[Dict[str, str]] = None) -> Response:
    """Serialize a relayed body without ever failing.

    ``ensure_ascii`` keeps a lone surrogate an escape instead of an encode
    error, and ``allow_nan`` relays an upstream NaN as sent: nothing after a
    billed upstream answer may turn it into a 500 (which the SDK would retry
    and bill again).
    """
    content = json.dumps(body, ensure_ascii=True, allow_nan=True, separators=(",", ":"))
    return Response(content=content, status_code=status_code,
                    media_type="application/json", headers=headers or {})


def typesafe_error_response(
    request: Optional[Request],
    status_code: int,
    message: str,
    *,
    error_type: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """A locally raised error in TypeSafe's shape, with a request id."""
    out = dict(headers or {})
    out.setdefault(TYPESAFE_REQUEST_ID_HEADER, typesafe_request_id(request))
    body = {"detail": {"error_type": error_type or typesafe_error_type(status_code), "message": message}}
    return typesafe_json_response(status_code, body, out)


def typesafe_provider_error_response(request: Optional[Request], error: ProviderHTTPError) -> Response:
    """Relay an upstream System One error.

    A TypeSafe-shaped body (one with ``detail``) is relayed verbatim; any other
    shape (OpenRouter ``{error:{...}}``, Ollama ``{error:"..."}``, a non-JSON
    page) is converted to ``{"detail": {"error_type", "message"}}``. Only the
    allow-listed headers travel, plus a local request id when the upstream sent
    none.
    """
    body = to_typesafe_body(error.body, error.status_code, default_message=error.message)
    headers = relay_headers(error.headers, TYPESAFE_RELAYED_HEADERS)
    if not any(k.lower() == TYPESAFE_REQUEST_ID_HEADER for k in headers):
        headers[TYPESAFE_REQUEST_ID_HEADER] = typesafe_request_id(request)
    return typesafe_json_response(error.status_code, body, headers)

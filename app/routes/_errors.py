"""Shared helpers for surfacing upstream provider errors from route handlers.

When the provider layer raises :class:`ProviderHTTPError`, routes should relay
the *upstream* HTTP status (and a safe subset of headers such as ``Retry-After``)
rather than collapsing everything to a generic 500. This module centralises the
header allowlist and the per-protocol JSON envelope builders.
"""

from typing import Dict

from fastapi.responses import JSONResponse

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


def _safe_headers(error: ProviderHTTPError) -> Dict[str, str]:
    return {
        k: v for k, v in (error.headers or {}).items()
        if isinstance(k, str) and isinstance(v, str) and k.lower() in SAFE_UPSTREAM_HEADERS
    }


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

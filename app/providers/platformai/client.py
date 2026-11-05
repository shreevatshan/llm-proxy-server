"""HTTP transport for the PlatformAI v3 chat API: auth, headers, errors, SSE.

Two authentication modes, each with its own path:

- API key:    ``Authorization: Bearer <key>``          on ``/pai/key_auth/v3/ai/chat``
- Zoho OAuth: ``Authorization: Zoho-oauthtoken <tok>`` on ``/pai/api/v3/ai/chat``

Gateway errors are ``{"error": {"code": "<CODE>", "message": "..."}}`` where
CODE is ``PAI-<AREA>-<NNN>`` (model/provider/token/... failures) or
``API-<reason>`` (request validation by the API layer); internal failures
carry a numeric code instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import httpx

from app.providers.base import ProviderHTTPError
from app.providers.platformai.auth import PlatformAITokenManager

logger = logging.getLogger(__name__)

OAUTH_CHAT_PATH = "/pai/api/v3/ai/chat"
KEY_CHAT_PATH = "/pai/key_auth/v3/ai/chat"

# Z-Pai-Metadata is rejected unless it is a JSON object with all of these keys,
# each an alphanumeric value of 1-20 characters; unset keys are sent as "na".
METADATA_KEYS = ("custom_param_a", "custom_param_b", "custom_param_c")
METADATA_PLACEHOLDER = "na"

_UPSTREAM_SNIPPET = 400


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class ApiKeyAuth:
    """A PlatformAI API key. A 401 is final: there is nothing to refresh."""

    chat_path = KEY_CHAT_PATH

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("PlatformAI API key is required")
        self._key = api_key

    async def header(self) -> str:
        return f"Bearer {self._key}"

    def invalidate(self, header: str) -> bool:
        return False


class OAuthAuth:
    """A Zoho OAuth access token minted from the stored refresh token."""

    chat_path = OAUTH_CHAT_PATH
    _PREFIX = "Zoho-oauthtoken "

    def __init__(self, tokens: PlatformAITokenManager):
        self.tokens = tokens

    async def header(self) -> str:
        return self._PREFIX + await self.tokens.get_token()

    def invalidate(self, header: str) -> bool:
        return self.tokens.invalidate(header[len(self._PREFIX):])


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

@dataclass
class GatewayError:
    code: Optional[str]
    message: str


def parse_error(text: str) -> GatewayError:
    """The gateway code and message from an error body (or SSE error frame)."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return GatewayError(None, text[:_UPSTREAM_SNIPPET].strip())
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, str):
        return GatewayError(None, err)
    if not isinstance(err, dict):
        message = parsed.get("message") if isinstance(parsed, dict) else None
        return GatewayError(None, str(message) if message else text[:_UPSTREAM_SNIPPET].strip())
    code = err.get("code")
    message = err.get("message") or err.get("description") or ""
    return GatewayError(str(code) if isinstance(code, str) and code else None, str(message))


def upstream_message(text: str) -> str:
    """Best human-readable message from a gateway error body."""
    return parse_error(text).message


_CODE_STATUS: Dict[str, Tuple[int, str]] = {
    "PAI-MODEL-001": (404, "not_found_error"),
    "PAI-PROVIDER-003": (404, "not_found_error"),
    # Unknown model names come back as "API Key not found": no credential
    # (managed or BYOK) resolves for them.
    "PAI-TOKEN-003": (404, "not_found_error"),
    "PAI-MODEL-007": (403, "permission_error"),
    "PAI-PROVIDER-004": (403, "permission_error"),
    "PAI-PROVIDER-006": (403, "permission_error"),
    "PAI-PROVIDER-001": (429, "rate_limit_error"),
    "PAI-PROVIDER-000": (503, "upstream_error"),
    "PAI-PROVIDER-005": (503, "upstream_error"),
    "PAI-PROVIDER-007": (503, "upstream_error"),
}

_AUTH_CODES = {"API-invalid_oauth_token"}


def _classify(status: int, code: Optional[str]) -> Tuple[int, str]:
    if code in _CODE_STATUS:
        return _CODE_STATUS[code]
    if code and code.startswith("PAI-GUARDRAIL-") or status == 424:
        return 400, "invalid_request_error"
    if code in _AUTH_CODES or status in (401, 403) or (code and code.startswith("PAI-TOKEN-")):
        # The client's proxy key is fine; the provider's credentials are not.
        return 502, "upstream_auth_error"
    if code and code.startswith("PAI-WORKSPACE-") or code == "API-url_not_found":
        return 502, "upstream_config_error"
    if code and (code.startswith("PAI-") or code.startswith("API-")):
        # PAI-MODEL-*, PAI-PROVIDER-002, PAI-OBSERVABILITY-*, API-* validation.
        return 400, "invalid_request_error"
    if status == 429:
        return 429, "rate_limit_error"
    if status == 404:
        return 404, "not_found_error"
    if status == 400:
        return 400, "invalid_request_error"
    return 502, "upstream_error"


def gateway_error(status: int, err: GatewayError, headers: Optional[httpx.Headers] = None) -> ProviderHTTPError:
    """Map a gateway error onto the status the proxy's client should see."""
    mapped, err_type = _classify(status, err.code)
    label = f"{err.code}: " if err.code else ""
    message = f"PlatformAI {label}{err.message or f'HTTP {status}'}"
    if err_type == "upstream_error" and status >= 500 and not err.code:
        message = f"PlatformAI request failed with HTTP {status}: {err.message}"
    body: Dict[str, Any] = {"error": {"message": message, "type": err_type, "code": mapped}}
    if err.code:
        body["error"]["param"] = err.code
    if err.code and err.code.startswith("PAI-GUARDRAIL-") or status == 424:
        body["error"]["code"] = "content_filter"
    out_headers = None
    if mapped == 429 and headers is not None and headers.get("retry-after"):
        out_headers = {"retry-after": headers["retry-after"]}
    error = ProviderHTTPError(mapped, message, body=body, headers=out_headers)
    error.pai_code = err.code  # type: ignore[attr-defined]
    error.upstream_message = err.message  # type: ignore[attr-defined]
    return error


def map_error(status: int, text: str, headers: Optional[httpx.Headers] = None) -> ProviderHTTPError:
    return gateway_error(status, parse_error(text), headers)


def pai_code(err: ProviderHTTPError) -> Optional[str]:
    return getattr(err, "pai_code", None)


def _error(status: int, message: str, err_type: str) -> ProviderHTTPError:
    return ProviderHTTPError(status, message, body={"error": {"message": message, "type": err_type, "code": status}})


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------

@dataclass
class SSEEvent:
    event: str
    data: str


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Parse ``event:``/``data:`` SSE lines; a blank line dispatches the event."""
    event = ""
    data_parts: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data_parts:
                yield SSEEvent(event=event, data="".join(data_parts))
            event, data_parts = "", []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value.strip()
        if name == "event":
            event = value
        elif name == "data":
            data_parts.append(value)
    if data_parts:
        yield SSEEvent(event=event, data="".join(data_parts))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def metadata_header(metadata: Any) -> Optional[str]:
    """``Z-Pai-Metadata`` from a request's metadata, or None when it has no
    ``custom_param_*`` keys. The gateway requires all three keys, so unset ones
    are filled with ``"na"``; invalid values are left for the gateway to reject."""
    if not isinstance(metadata, dict):
        return None
    picked = {k: metadata.get(k) for k in METADATA_KEYS}
    if all(v in (None, "") for v in picked.values()):
        return None
    return json.dumps({k: METADATA_PLACEHOLDER if v in (None, "") else str(v) for k, v in picked.items()})


class PlatformAIClient:
    """Authenticated chat calls to one PlatformAI gateway."""

    def __init__(
        self,
        *,
        base_url: str,
        auth,
        token_id: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.token_id = (token_id or "").strip() or None
        self.http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
        )

    @property
    def chat_path(self) -> str:
        return self.auth.chat_path

    async def aclose(self) -> None:
        await self.http.aclose()

    def _headers(self, authorization: str, *, stream: bool, metadata: Optional[str]) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": authorization,
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.token_id:
            headers["Z-Token-Id"] = self.token_id
        if metadata:
            headers["Z-Pai-Metadata"] = metadata
        return headers

    async def _send(self, body: Dict[str, Any], *, stream: bool, metadata: Optional[str]) -> httpx.Response:
        url = f"{self.base_url}{self.chat_path}"

        async def attempt() -> Tuple[str, httpx.Response]:
            authorization = await self.auth.header()
            request = self.http.build_request(
                "POST", url, headers=self._headers(authorization, stream=stream, metadata=metadata), json=body
            )
            try:
                return authorization, await self.http.send(request, stream=stream)
            except httpx.HTTPError as exc:
                raise _error(502, f"Could not reach PlatformAI: {type(exc).__name__}", "upstream_error")

        authorization, response = await attempt()
        if response.status_code == 401 and self.auth.invalidate(authorization):
            await response.aread()
            await response.aclose()
            _, response = await attempt()
        if response.status_code >= 400:
            text = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            raise map_error(response.status_code, text, response.headers)
        return response

    async def chat(self, body: Dict[str, Any], *, metadata: Optional[str] = None) -> Dict[str, Any]:
        response = await self._send(body, stream=False, metadata=metadata)
        try:
            payload = response.json()
        except ValueError:
            raise _error(502, "PlatformAI returned a non-JSON response", "upstream_error")
        if not isinstance(payload, dict):
            raise _error(502, "PlatformAI returned an unexpected response shape", "upstream_error")
        return payload

    async def open_stream(self, body: Dict[str, Any], *, metadata: Optional[str] = None) -> httpx.Response:
        """Open a streaming chat call. HTTP errors raise here, before any event
        is read. The caller must ``await response.aclose()``."""
        return await self._send(body, stream=True, metadata=metadata)

    @staticmethod
    async def events(response: httpx.Response) -> AsyncIterator[SSEEvent]:
        try:
            async for event in iter_sse(response.aiter_lines()):
                yield event
        except httpx.HTTPError as exc:
            raise _error(502, f"PlatformAI stream ended unexpectedly: {type(exc).__name__}", "upstream_error")

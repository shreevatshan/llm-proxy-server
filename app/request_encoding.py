"""Request-body Content-Encoding for the TypeSafe API app (port 11366).

TypeSafe SDKs (typesafe-sdk >= 0.7.5) zstd-compress every request body of
1 KiB or more and never fall back on a 415, so the TypeSafe app accepts what
api.typesafe.ai accepts: one layer of gzip, x-gzip or zstd, at most 32 MiB
compressed and 32 MiB decoded, with a zstd window of at most 8 MiB. The limits
apply to compressed bodies; uncompressed ones are handled as on every other port.

Only the TypeSafe app decodes (``_add_request_tracking(...,
decode_content_encoding=True)`` in app/main.py); the OpenAI, Anthropic and Azure
ports keep their behaviour exactly.

zstd comes from ``pyzstd`` (or the stdlib ``compression.zstd`` on Python 3.14+).
Deliberately not ``zstandard`` or ``backports.zstd``: httpx detects the former,
and urllib3/requests and aiohttp (used by google-genai) detect the latter, and
each would then start offering zstd to upstreams on every port.

Abuse limits: a compressed body is only read and decoded once the request's
credentials have been validated (without auth's side effects), so anonymous or
forged callers can never trigger a decompression. Decoding runs in a worker
thread, and at most a few decodes run at once process-wide (all servers share
one event loop).
"""

import asyncio
import logging
import uuid
import zlib
from typing import Optional

from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

try:
    import pyzstd as _zstd

    def _zstd_decompressor():
        return _zstd.ZstdDecompressor(option={_zstd.DParameter.windowLogMax: ZSTD_WINDOW_LOG_MAX})
except ImportError:  # pragma: no cover - depends on the interpreter
    try:  # Python 3.14+
        from compression import zstd as _zstd

        def _zstd_decompressor():
            return _zstd.ZstdDecompressor(
                options={_zstd.DecompressionParameter.window_log_max: ZSTD_WINDOW_LOG_MAX}
            )
    except ImportError:
        _zstd = None
        _zstd_decompressor = None

MAX_ENCODED_BODY_BYTES = 32 * 1024 * 1024
MAX_DECODED_BODY_BYTES = 32 * 1024 * 1024
ZSTD_WINDOW_LOG_MAX = 23  # 2**23 = 8 MiB, TypeSafe's limit
MAX_CONCURRENT_DECODES = 4

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
_GZIP_CODINGS = ("gzip", "x-gzip")


def zstd_available() -> bool:
    return _zstd is not None


def _decodable() -> tuple:
    return _GZIP_CODINGS + (("zstd",) if _zstd is not None else ())


class BodyEncodingError(Exception):
    """A request body this server refuses; rendered as a TypeSafe error."""

    def __init__(self, status_code: int, error_type: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message

    def to_response(self) -> JSONResponse:
        # Raised from inside track_requests, which must *return* a response:
        # an exception raised in middleware becomes a bare 500.
        return JSONResponse(
            status_code=self.status_code,
            content={"detail": {"error_type": self.error_type, "message": self.message}},
            headers={"x-typesafe-request-id": f"req_{uuid.uuid4().hex}"},
        )


def _too_large(what: str) -> BodyEncodingError:
    return BodyEncodingError(413, "request_too_large", f"{what} request body exceeds 32 MiB")


def _invalid(coding: str, why: str) -> BodyEncodingError:
    return BodyEncodingError(400, "api_usage_error", f"Request body is not valid {coding} data ({why})")


def requested_coding(values) -> Optional[str]:
    """The single coding named by the Content-Encoding header line(s), or None.

    ``identity`` is ignored. Raises 415 for several layers or a coding this
    server cannot decode.
    """
    codings = [c.strip().lower() for v in values for c in v.split(",")]
    codings = [c for c in codings if c and c != "identity"]
    if not codings:
        return None
    if len(codings) > 1:
        raise BodyEncodingError(415, "api_usage_error", "Only one Content-Encoding layer is supported")
    supported = _decodable()
    if codings[0] not in supported:
        raise BodyEncodingError(
            415,
            "api_usage_error",
            f"Unsupported Content-Encoding '{codings[0]}'; supported: {', '.join(supported)}",
        )
    return codings[0]


def _gunzip(raw: bytes) -> bytes:
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)  # gzip framing only
    out = d.decompress(raw, MAX_DECODED_BODY_BYTES + 1)
    if len(out) > MAX_DECODED_BODY_BYTES:
        raise _too_large("Decoded")
    if not d.eof:
        raise _invalid("gzip", "truncated")
    if d.unused_data:
        raise _invalid("gzip", "trailing data or more than one member")
    return out


def _unzstd(raw: bytes) -> bytes:
    d = _zstd_decompressor()
    out = bytearray(d.decompress(raw, max_length=MAX_DECODED_BODY_BYTES + 1))
    while not d.eof and not d.needs_input and len(out) <= MAX_DECODED_BODY_BYTES:
        more = d.decompress(b"", max_length=MAX_DECODED_BODY_BYTES + 1 - len(out))
        if not more:
            break
        out += more
    if len(out) > MAX_DECODED_BODY_BYTES:
        raise _too_large("Decoded")
    if not d.eof:
        raise _invalid("zstd", "truncated")
    if d.unused_data:
        raise _invalid("zstd", "trailing data or more than one frame")
    return bytes(out)


def decode_body(coding: str, raw: bytes) -> bytes:
    """Decode *raw* with *coding* under the size and window limits."""
    try:
        return _gunzip(raw) if coding in _GZIP_CODINGS else _unzstd(raw)
    except BodyEncodingError:
        raise
    except Exception as e:  # zlib.error, ZstdError (corrupt, window too large)
        raise _invalid(coding, "corrupt data or zstd window over 8 MiB") from e


_decode_slots: Optional[asyncio.Semaphore] = None
_decode_slots_loop = None


def _slots() -> asyncio.Semaphore:
    """A per-event-loop semaphore (tests run several loops in one process)."""
    global _decode_slots, _decode_slots_loop
    loop = asyncio.get_running_loop()
    if _decode_slots is None or _decode_slots_loop is not loop:
        _decode_slots = asyncio.Semaphore(MAX_CONCURRENT_DECODES)
        _decode_slots_loop = loop
    return _decode_slots


async def _require_valid_credentials(request) -> None:
    """403 without credentials, 401 for invalid ones -- before any body is read.

    Checked here and not only by the route's auth dependency because FastAPI
    parses the (decoded) body before it runs dependencies.
    """
    from app.auth.middleware import credentials_are_valid

    if not (request.headers.get("authorization") or request.cookies.get("access_token")):
        raise BodyEncodingError(
            403, "authentication_error", "Must supply an API key! Check your request and try again."
        )
    try:
        valid = await credentials_are_valid(request)
    except Exception:
        logger.exception("Credential check before decoding failed")
        raise BodyEncodingError(500, "api_error", "Authentication error")
    if not valid:
        raise BodyEncodingError(
            401, "authentication_error",
            "Cannot authenticate with the server. Please check your API key and try again.",
        )


def _replace_body(request, body: bytes) -> None:
    # BaseHTTPMiddleware's _CachedRequest re-serves _body to the downstream app
    # (starlette/middleware/base.py), the mechanism track_requests' alias
    # rewrite already relies on.
    request._body = body
    request.scope["headers"] = [
        (k, v) for k, v in request.scope["headers"]
        if k not in (b"content-encoding", b"content-length", b"transfer-encoding")
    ] + [(b"content-length", str(len(body)).encode("latin-1"))]
    request.__dict__.pop("_headers", None)  # Request caches its Headers object


async def decode_request_body(request) -> Optional[str]:
    """Decode a Content-Encoding'd request body in place.

    Returns the coding that was decoded, or None when there was nothing to
    decode (no Content-Encoding, or identity only). Raises BodyEncodingError
    (415, 413, 403, 401, 400 or 500).
    """
    if request.method not in _BODY_METHODS or "content-encoding" not in request.headers:
        return None
    coding = requested_coding(request.headers.getlist("content-encoding"))  # 415 before auth, as TypeSafe
    if coding is None:
        return None  # identity only
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_ENCODED_BODY_BYTES:
        raise _too_large("Encoded")
    await _require_valid_credentials(request)
    raw = bytearray()
    async for chunk in request.stream():
        raw += chunk
        if len(raw) > MAX_ENCODED_BODY_BYTES:
            raise _too_large("Encoded")
    if not raw:
        _replace_body(request, b"")
        return coding
    async with _slots():
        decoded = await run_in_threadpool(decode_body, coding, bytes(raw))
    _replace_body(request, decoded)
    return coding

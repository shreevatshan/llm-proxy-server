"""Shared streaming utilities for SSE responses.

Provides timeout, disconnect detection, and trace context preservation
for streaming endpoints (chat completions, responses API, etc.).
"""

import os
import asyncio
import json
import time
import logging
from typing import Any, Optional

from fastapi import Request
from opentelemetry.context import attach

from app import diagnostics
from app.tracing import safe_detach

logger = logging.getLogger(__name__)

# Streaming configuration
STREAM_TIMEOUT_SECONDS = int(os.getenv("STREAM_TIMEOUT_SECONDS", "600"))  # 10 minutes default
# Per-chunk (mid-stream) gap budget. This bounds silence *after* the first token
# has arrived. It must tolerate legitimate long gaps — e.g. Claude extended
# thinking on Bedrock, where the Anthropic SDK silently swallows `ping`
# keepalive events (anthropic/lib/streaming/_messages.py::build_events has no
# `ping` case), so a thinking pause looks like dead air to this loop. The FIRST
# chunk gets the much shorter STREAM_FIRST_CHUNK_TIMEOUT_SECONDS budget below,
# which is what actually catches the "accepted but never responds" hang — so
# this value no longer needs to be aggressive.
STREAM_CHUNK_TIMEOUT_SECONDS = int(os.getenv("STREAM_CHUNK_TIMEOUT_SECONDS", "300"))  # 5 minutes per chunk
# Dedicated time-to-first-token (TTFT) budget. A provider that accepts the
# request but never emits a first event is the most common stall (see the stall
# watchdog dumps). Give the first chunk a much shorter budget than subsequent
# chunks so the proxy surfaces a fast, actionable error well before the client's
# own (shorter) timeout fires. Capped at STREAM_CHUNK_TIMEOUT_SECONDS so a tiny
# per-chunk timeout (e.g. in tests) still wins.
STREAM_FIRST_CHUNK_TIMEOUT_SECONDS = int(os.getenv("STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", "45"))  # TTFT
# Independent stall-dump deadline: how long a stream may make no progress before
# the watchdog dumps all-thread stacks. Deliberately decoupled from (and much
# shorter than) STREAM_CHUNK_TIMEOUT_SECONDS so the dump fires *before* the
# client gives up and disconnects — otherwise the disconnect disarms the
# watchdog first and no dump is ever written. Tune below a typical client
# timeout (most are 30-120s).
STREAM_STALL_DUMP_SECONDS = float(os.getenv("STREAM_STALL_DUMP_SECONDS", "20"))
DISCONNECT_CHECK_INTERVAL = int(os.getenv("DISCONNECT_CHECK_INTERVAL", "10"))  # Check disconnect every N chunks
STREAM_DISCONNECT_POLL_SECONDS = float(os.getenv("STREAM_DISCONNECT_POLL_SECONDS", "5"))
STREAM_DISCONNECT_CHECK_TIMEOUT_SECONDS = float(
    os.getenv("STREAM_DISCONNECT_CHECK_TIMEOUT_SECONDS", "0.5")
)
STREAM_EARLY_DISCONNECT_CHECK_CHUNKS = int(
    os.getenv("STREAM_EARLY_DISCONNECT_CHECK_CHUNKS", "5")
)


def _set_stream_state(stream_state: dict[str, Any], *, status: str, termination_reason: str) -> None:
    """Update stream status + termination reason together."""
    stream_state["final_status"] = status
    stream_state["termination_reason"] = termination_reason


def _stall_watchdog_seconds() -> float:
    """Deadline for the stall watchdog (an independent, short stall budget).

    Capped at the per-chunk timeout so a deliberately tiny STREAM_CHUNK_TIMEOUT
    (e.g. in tests) still tears the stream down before the dump would fire.
    """
    return min(STREAM_STALL_DUMP_SECONDS, STREAM_CHUNK_TIMEOUT_SECONDS)


def _chunk_budget_seconds(chunks_seen: int) -> float:
    """Per-chunk wait budget: a short TTFT budget for the first chunk, the full
    per-chunk budget thereafter.

    The first-token budget is capped at the per-chunk budget so a deliberately
    tiny STREAM_CHUNK_TIMEOUT (e.g. in tests) still bounds the first chunk too.
    """
    if chunks_seen == 0:
        return min(STREAM_FIRST_CHUNK_TIMEOUT_SECONDS, STREAM_CHUNK_TIMEOUT_SECONDS)
    return STREAM_CHUNK_TIMEOUT_SECONDS


def _stall_watchdog_key(request: Request, fallback: str) -> str:
    """Build a stable watchdog key from the tracking request id + a path label."""
    request_id = getattr(getattr(request, "state", None), "tracking_request_id", None)
    return f"{fallback}:{request_id or id(request)}"


async def _cancel_pending_task(
    task: Optional[asyncio.Task],
    *,
    task_name: str,
    timeout_seconds: float = 1.0,
) -> None:
    """Cancel an async task and log if it does not stop promptly."""
    if task is None or task.done():
        return

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout_seconds)
    except asyncio.CancelledError:
        logger.debug("Cancelled %s", task_name)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for %s cancellation", task_name)
    except Exception as exc:
        logger.warning("Error while cancelling %s: %s", task_name, exc, exc_info=True)


async def _close_async_iterator(gen_iter, *, iterator_name: str) -> None:
    """Best-effort async iterator cleanup with logging."""
    aclose = getattr(gen_iter, "aclose", None)
    if not callable(aclose):
        return

    try:
        await aclose()
    except Exception as exc:
        logger.warning("Failed to close %s: %s", iterator_name, exc, exc_info=True)


async def _request_is_disconnected(request: Request) -> bool:
    """Check whether the client disconnected without hanging the stream loop."""
    try:
        return await asyncio.wait_for(
            request.is_disconnected(),
            timeout=min(STREAM_DISCONNECT_POLL_SECONDS, STREAM_DISCONNECT_CHECK_TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        logger.warning("Disconnect check timed out")
        return False


def _format_stream_log_context(log_context: Optional[dict[str, Any]]) -> str:
    """Return a compact JSON suffix for stream lifecycle logs."""
    if not log_context:
        return ""
    try:
        return f" context={json.dumps(log_context, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f" context={log_context!r}"


def set_request_tracking_outcome(
    request: Request,
    *,
    status: str,
    termination_reason: str,
    error: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Record the final request-tracking outcome on request.state.

    `errored` outcomes always win over `completed` unless overwrite=True.
    """
    if not hasattr(request, "state"):
        return

    existing = getattr(request.state, "tracking_final", None)
    if isinstance(existing, dict) and not overwrite:
        existing_status = existing.get("status")
        if existing_status == "errored" and status != "errored":
            return
        if existing_status == "errored" and status == "errored":
            if error and not existing.get("error"):
                existing["error"] = error
            if termination_reason and not existing.get("termination_reason"):
                existing["termination_reason"] = termination_reason
            return

    request.state.tracking_final = {
        "status": status,
        "termination_reason": termination_reason,
        "error": error,
    }


async def stream_with_context_and_timeout(
    generator,
    context_token,
    request: Request,
    timeout: int = STREAM_TIMEOUT_SECONDS,
    request_started_at: Optional[float] = None,
):
    """
    Wrapper to ensure streaming happens within the parent trace context with timeout and disconnect detection.
    
    This wrapper provides:
    1. Trace context preservation across async boundaries
    2. Overall streaming timeout to prevent infinite hangs
    3. Per-chunk timeout to detect slow providers
    4. Client disconnect detection to release resources early
    """
    # Attach the parent context
    token = attach(context_token)
    start_time = request_started_at if request_started_at is not None else time.monotonic()
    chunks_yielded = 0
    watchdog_key = _stall_watchdog_key(request, "chat")
    diagnostics.arm(watchdog_key, _stall_watchdog_seconds())

    try:
        async for chunk in _stream_with_timeout_and_disconnect(generator, request, timeout):
            chunks_yielded += 1
            # Progress made — push the stall deadline out before yielding.
            diagnostics.reset(watchdog_key, _stall_watchdog_seconds())

            # Check overall timeout
            elapsed = time.monotonic() - start_time
            timed_out = elapsed > timeout
            if timed_out:
                logger.warning(
                    "Stream exceeded timeout of %ss after %s chunks; emitting final chunk before terminating",
                    timeout,
                    chunks_yielded,
                )

            yield chunk

            if elapsed > timeout:
                # Send error chunk before closing (OpenAI-compatible error format)
                error_data = {
                    "error": {
                        "message": f"Stream timeout exceeded ({timeout}s)",
                        "type": "server_error",
                        "param": None,
                        "code": "stream_timeout"
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"  # Proper stream termination
                break
                
    except asyncio.CancelledError:
        logger.info(f"Stream cancelled after {chunks_yielded} chunks")
        raise
    except Exception as e:
        logger.error(f"Stream error after {chunks_yielded} chunks: {e}")
        raise
    finally:
        diagnostics.disarm(watchdog_key)
        elapsed = time.monotonic() - start_time
        logger.debug(f"Stream completed: {chunks_yielded} chunks in {elapsed:.2f}s")
        # Detach the context when done
        safe_detach(token)


async def _stream_with_timeout_and_disconnect(generator, request: Request, timeout: int):
    """
    Stream generator with per-chunk timeout and client disconnect detection.

    Disconnect check is debounced to reduce overhead - only checks every N chunks
    (configurable via DISCONNECT_CHECK_INTERVAL environment variable).
    """
    # Create an async iterator from the generator
    gen_iter = generator.__aiter__()
    # Own the in-flight __anext__ as an explicit task so a cancel/timeout can
    # cancel it *before* we aclose() the generator. Awaiting __anext__ bare
    # inside asyncio.wait_for leaves the generator mid-run on cancellation, so
    # the finally-block aclose() raises "asynchronous generator is already
    # running". This mirrors _stream_with_timeout_and_disconnect_anthropic.
    next_chunk_task: Optional[asyncio.Task] = None
    pending_chunk_started_at: Optional[float] = None
    try:
        chunk_count = 0
        chunks_yielded = 0  # drives first-chunk (TTFT) vs subsequent-chunk budget

        while True:
            # Check if client disconnected (debounced - every N chunks).
            # This is the post-chunk check; disconnect is ALSO polled during the
            # per-chunk wait below so a hang before the first/next chunk is caught
            # within STREAM_DISCONNECT_POLL_SECONDS rather than after the full
            # STREAM_CHUNK_TIMEOUT_SECONDS budget (matches the Anthropic path).
            chunk_count += 1
            if (
                chunk_count <= STREAM_EARLY_DISCONNECT_CHECK_CHUNKS
                or chunk_count % DISCONNECT_CHECK_INTERVAL == 0
            ):
                if await _request_is_disconnected(request):
                    logger.info(f"Client disconnected after {chunk_count} chunks, stopping stream")
                    break

            if next_chunk_task is None:
                next_chunk_task = asyncio.create_task(gen_iter.__anext__())
                pending_chunk_started_at = time.monotonic()

            # Enforce the per-chunk timeout across multiple short waits so we can
            # poll for client disconnect in between. The first chunk uses the
            # shorter TTFT budget; subsequent chunks use the full per-chunk budget.
            assert pending_chunk_started_at is not None
            chunk_budget = _chunk_budget_seconds(chunks_yielded)
            elapsed_wait = time.monotonic() - pending_chunk_started_at
            remaining_chunk_budget = chunk_budget - elapsed_wait
            if remaining_chunk_budget <= 0:
                budget_label = "first chunk (TTFT)" if chunks_yielded == 0 else "chunk"
                logger.warning(f"{budget_label} timeout after {chunk_budget}s")
                await _cancel_pending_task(
                    next_chunk_task, task_name="next chunk task after chunk timeout"
                )
                next_chunk_task = None
                pending_chunk_started_at = None
                # Send timeout notification (OpenAI-compatible error format)
                error_data = {
                    "error": {
                        "message": f"Provider response timeout ({chunk_budget}s)",
                        "type": "server_error",
                        "param": None,
                        "code": "chunk_timeout"
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"  # Proper stream termination
                break

            wait_timeout = min(STREAM_DISCONNECT_POLL_SECONDS, remaining_chunk_budget)
            done, _ = await asyncio.wait({next_chunk_task}, timeout=wait_timeout)
            if not done:
                # Provider still producing the next chunk; check whether the
                # client gave up before we burn the whole chunk budget waiting.
                if await _request_is_disconnected(request):
                    logger.info(
                        f"Client disconnected after {chunk_count} chunks while waiting for next chunk, stopping stream"
                    )
                    break
                continue

            task = next_chunk_task
            next_chunk_task = None
            pending_chunk_started_at = None
            try:
                chunk = task.result()
            except StopAsyncIteration:
                # Generator exhausted normally
                break
            chunks_yielded += 1
            yield chunk

    except asyncio.CancelledError:
        logger.info("Stream task cancelled")
        raise
    finally:
        # Cancel any in-flight chunk fetch before closing the generator, so
        # aclose() does not race a running __anext__.
        await _cancel_pending_task(
            next_chunk_task,
            task_name="next chunk task during cleanup",
            timeout_seconds=2.0,
        )
        # Close the upstream generator to release HTTP connections back to the pool.
        # Without this, abandoned streams (client disconnect, timeout) leak sockets.
        await _close_async_iterator(gen_iter, iterator_name="stream generator")


async def stream_with_context(generator, context_token):
    """
    Legacy wrapper for backward compatibility (without timeout).
    Use stream_with_context_and_timeout for new code.
    """
    # No stall watchdog here: this wrapper has no per-chunk timeout, so there is
    # no pre-timeout margin to dump against. Instrument the timeout wrappers only.
    # Attach the parent context
    token = attach(context_token)
    try:
        async for chunk in generator:
            yield chunk
    finally:
        # Detach the context when done
        safe_detach(token)


# ==================== Anthropic SSE Helpers ====================

def format_anthropic_sse_event(event_type: str, data: dict) -> str:
    """Format a single Anthropic SSE event.
    
    Anthropic streaming uses typed events like:
        event: message_start
        data: {"type":"message_start","message":{...}}
    """
    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    return f"event: {event_type}\ndata: {json_str}\n\n"


async def anthropic_stream_with_context_and_timeout(
    generator,
    context_token,
    request: Request,
    timeout: int = STREAM_TIMEOUT_SECONDS,
    log_context: Optional[dict[str, Any]] = None,
    request_started_at: Optional[float] = None,
):
    """
    Wrapper for Anthropic streaming with trace context preservation, timeout,
    and disconnect detection.
    
    Similar to stream_with_context_and_timeout but uses Anthropic error format.
    """
    token = attach(context_token)
    start_time = request_started_at if request_started_at is not None else time.monotonic()
    chunks_yielded = 0
    watchdog_key = _stall_watchdog_key(request, "anthropic")
    diagnostics.arm(watchdog_key, _stall_watchdog_seconds())
    stream_state: dict[str, Any] = {
        "termination_reason": "completed",
        "final_status": "completed",
    }

    logger.info("Anthropic stream started%s", _format_stream_log_context(log_context))

    try:
        async for chunk in _stream_with_timeout_and_disconnect_anthropic(
            generator,
            request,
            timeout,
            log_context=log_context,
            stream_state=stream_state,
        ):
            chunks_yielded += 1
            # Progress made — push the stall deadline out before yielding.
            diagnostics.reset(watchdog_key, _stall_watchdog_seconds())
            yield chunk
        set_request_tracking_outcome(
            request,
            status=stream_state.get("final_status", "completed"),
            termination_reason=stream_state.get("termination_reason", "completed"),
        )

    except asyncio.CancelledError:
        _set_stream_state(
            stream_state,
            status="cancelled",
            termination_reason="cancelled",
        )
        logger.info(
            "Anthropic stream cancelled after %s chunks%s",
            chunks_yielded,
            _format_stream_log_context(log_context),
        )
        raise
    except Exception as e:
        _set_stream_state(
            stream_state,
            status="errored",
            termination_reason="error",
        )
        set_request_tracking_outcome(
            request,
            status="errored",
            termination_reason="stream_error",
            error=str(e),
        )
        logger.error(
            "Anthropic stream error after %s chunks: %s%s",
            chunks_yielded,
            e,
            _format_stream_log_context(log_context),
        )
        raise
    finally:
        diagnostics.disarm(watchdog_key)
        elapsed = time.monotonic() - start_time
        logger.info(
            "Anthropic stream finished reason=%s chunks=%s duration_seconds=%.2f%s",
            stream_state.get("termination_reason", "completed"),
            chunks_yielded,
            elapsed,
            _format_stream_log_context(log_context),
        )
        safe_detach(token)


async def _stream_with_timeout_and_disconnect_anthropic(
    generator,
    request: Request,
    timeout: int,
    log_context: Optional[dict[str, Any]] = None,
    stream_state: Optional[dict[str, Any]] = None,
):
    """
    Anthropic stream generator with per-chunk timeout and client disconnect detection.
    """
    gen_iter = generator.__aiter__()
    if stream_state is None:
        stream_state = {
            "termination_reason": "completed",
            "final_status": "completed",
        }
    next_chunk_task: Optional[asyncio.Task] = None
    pending_chunk_started_at: Optional[float] = None
    stream_started_at = time.monotonic()
    chunks_yielded = 0  # drives first-chunk (TTFT) vs subsequent-chunk budget
    try:
        while True:
            now = time.monotonic()
            if now - stream_started_at >= timeout:
                _set_stream_state(
                    stream_state,
                    status="errored",
                    termination_reason="stream_timeout",
                )
                set_request_tracking_outcome(
                    request,
                    status="errored",
                    termination_reason="stream_timeout",
                    error=f"Stream timeout exceeded ({timeout}s)",
                )
                await _cancel_pending_task(
                    next_chunk_task,
                    task_name="anthropic next chunk task after stream timeout",
                )
                next_chunk_task = None
                pending_chunk_started_at = None
                logger.warning(
                    "Anthropic stream exceeded timeout of %ss%s",
                    timeout,
                    _format_stream_log_context(log_context),
                )
                yield format_anthropic_sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "message": f"Stream timeout exceeded ({timeout}s)",
                        },
                    },
                )
                break

            if next_chunk_task is None:
                next_chunk_task = asyncio.create_task(gen_iter.__anext__())
                pending_chunk_started_at = time.monotonic()

            # The first chunk uses the shorter TTFT budget; subsequent chunks use
            # the full per-chunk budget. A provider that accepts the request but
            # never emits a first event is the most common stall.
            assert pending_chunk_started_at is not None
            chunk_budget = _chunk_budget_seconds(chunks_yielded)
            elapsed_wait = time.monotonic() - pending_chunk_started_at
            remaining_chunk_budget = chunk_budget - elapsed_wait
            if remaining_chunk_budget <= 0:
                budget_reason = "first_chunk_timeout" if chunks_yielded == 0 else "chunk_timeout"
                _set_stream_state(
                    stream_state,
                    status="errored",
                    termination_reason=budget_reason,
                )
                set_request_tracking_outcome(
                    request,
                    status="errored",
                    termination_reason=budget_reason,
                    error=f"Provider response timeout ({chunk_budget}s)",
                )
                await _cancel_pending_task(
                    next_chunk_task,
                    task_name="anthropic next chunk task after chunk timeout",
                )
                next_chunk_task = None
                pending_chunk_started_at = None
                logger.warning(
                    "Anthropic %s after %ss%s",
                    budget_reason,
                    chunk_budget,
                    _format_stream_log_context(log_context),
                )
                yield format_anthropic_sse_event(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "message": f"Provider response timeout ({chunk_budget}s)",
                        },
                    },
                )
                break

            wait_timeout = min(STREAM_DISCONNECT_POLL_SECONDS, remaining_chunk_budget)
            done, _ = await asyncio.wait({next_chunk_task}, timeout=wait_timeout)
            if not done:
                if await _request_is_disconnected(request):
                    _set_stream_state(
                        stream_state,
                        status="cancelled",
                        termination_reason="client_disconnect",
                    )
                    set_request_tracking_outcome(
                        request,
                        status="cancelled",
                        termination_reason="client_disconnect",
                    )
                    await _cancel_pending_task(
                        next_chunk_task,
                        task_name="anthropic next chunk task after client disconnect",
                    )
                    next_chunk_task = None
                    pending_chunk_started_at = None
                    logger.info(
                        "Anthropic client disconnected while waiting for next chunk%s",
                        _format_stream_log_context(log_context),
                    )
                    break
                continue

            task = next_chunk_task
            next_chunk_task = None
            pending_chunk_started_at = None
            try:
                chunk = task.result()
            except StopAsyncIteration:
                stream_state.setdefault("termination_reason", "completed")
                break

            chunks_yielded += 1
            yield chunk

    except asyncio.CancelledError:
        _set_stream_state(
            stream_state,
            status="cancelled",
            termination_reason="cancelled",
        )
        logger.info("Anthropic stream task cancelled")
        raise
    finally:
        await _cancel_pending_task(
            next_chunk_task,
            task_name="anthropic next chunk task during cleanup",
            timeout_seconds=2.0,
        )
        # Close the upstream generator to release HTTP connections back to the pool.
        await _close_async_iterator(gen_iter, iterator_name="anthropic stream generator")

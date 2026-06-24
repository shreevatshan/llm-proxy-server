"""Stall watchdog: timeout-triggered all-thread stack dumps.

Instead of dumping thread stacks on a fixed interval (which mostly captures
idle threads), this watchdog lets a caller *arm* a deadline for a unit of work
(e.g. a streaming request). Making progress *resets* the deadline; finishing
*disarms* it. If the work makes no progress until its deadline, a single daemon
thread writes an all-thread ``faulthandler`` dump — capturing the live blocking
frame a few seconds *before* the work's own timeout fires.

This is the diagnostic instrument for "request active server-side but no output
reaches the client" hangs: the dump names the exact frame the request is parked
on (provider connect, credential/region resolution, middleware, etc.).

Self-contained and dependency-free so both ``run.py`` (entry script) and
``app/routes/stream_utils.py`` can use it without import cycles. All public
functions are safe no-ops until :func:`init_watchdog` is called, so tests and
non-diagnostic runs are unaffected.
"""

import faulthandler
import threading
import time
from typing import Optional, TextIO

# Module-global singleton state, guarded by ``_cond``'s lock.
_cond = threading.Condition()
_deadlines: dict[str, float] = {}   # key -> monotonic deadline
_fp: Optional[TextIO] = None        # faulthandler dump destination
_max_bytes: int = 0                 # truncate the file once it grows past this
_thread: Optional[threading.Thread] = None
_started = False
_shutdown = False


def init_watchdog(fp: TextIO, max_bytes: int) -> None:
    """Enable the watchdog, writing dumps to *fp* (an already-open file handle).

    Idempotent: a second call is ignored. *max_bytes* bounds the dump file —
    it is truncated before a dump once it exceeds this size.
    """
    global _fp, _max_bytes, _thread, _started, _shutdown
    with _cond:
        if _started:
            return
        _fp = fp
        _max_bytes = max_bytes
        _shutdown = False
        _started = True
        _thread = threading.Thread(
            target=_run, name="stall-watchdog", daemon=True
        )
        _thread.start()


def arm(key: str, seconds: float) -> None:
    """Register/refresh a stall deadline for *key* at ``now + seconds``.

    No-op until :func:`init_watchdog` has been called.
    """
    if not _started or seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    with _cond:
        _deadlines[key] = deadline
        _cond.notify_all()


# Refreshing a deadline is semantically identical to arming it again.
reset = arm


def disarm(key: str) -> None:
    """Clear the stall deadline for *key* (work finished or progressed away)."""
    if not _started:
        return
    with _cond:
        if _deadlines.pop(key, None) is not None:
            _cond.notify_all()


def shutdown() -> None:
    """Stop the watchdog thread (best-effort, for clean process exit)."""
    global _shutdown
    if not _started:
        return
    with _cond:
        _shutdown = True
        _cond.notify_all()


def _run() -> None:
    """Daemon loop: sleep until the earliest deadline, dump if it passes."""
    while True:
        with _cond:
            if _shutdown:
                return
            if not _deadlines:
                # Nothing armed — wait until something is.
                _cond.wait()
                continue
            now = time.monotonic()
            key, deadline = min(_deadlines.items(), key=lambda kv: kv[1])
            remaining = deadline - now
            if remaining > 0:
                # Wake when the earliest deadline is due, or sooner if the set
                # changes (arm/disarm/reset call notify_all()).
                _cond.wait(timeout=remaining)
                continue
            # Deadline reached while still armed → consume it (one-shot) and dump.
            elapsed_since_deadline = now - deadline
            del _deadlines[key]
            fp = _fp

        if fp is not None:
            _dump(fp, key, elapsed_since_deadline)


def _dump(fp: TextIO, key: str, overshoot: float) -> None:
    """Write a headed all-thread stack dump. Never raises."""
    try:
        # Keep the file bounded: truncate once it grows past the cap so only the
        # most recent dumps are retained (mirrors run.py's periodic-dump logic).
        if _max_bytes and fp.tell() >= _max_bytes:
            fp.seek(0)
            fp.truncate()
        fp.write(
            f"\n----- stall watchdog: no progress for key={key!r} "
            f"(deadline passed {overshoot:.1f}s ago) -----\n"
        )
        fp.flush()
        faulthandler.dump_traceback(file=fp, all_threads=True)
        fp.flush()
    except Exception:  # noqa: BLE001 - a watchdog must never crash the process
        pass

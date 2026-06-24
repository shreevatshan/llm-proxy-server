#!/usr/bin/env python3
"""
LLM Proxy Server - Startup Script

This script starts the LLM Proxy Server that provides OpenAI-compatible,
Anthropic-compatible, and Azure OpenAI-compatible API endpoints for multiple
LLM providers including Ollama, Azure OpenAI, AWS Bedrock, and GCP Gemini.
"""

# CRITICAL: Apply OpenTelemetry context.detach() patch FIRST!
# This must be imported before any OTel instrumentation libraries (traceloop, etc.)
# to prevent "Failed to detach context" errors in async code.
import app.otel_patch  # noqa: F401

# IMPORTANT: Monkey-patch json.dumps to handle Pydantic models
# This must be done before any other imports that might use json.dumps
import json
from pydantic import BaseModel

_original_json_dumps = json.dumps


def _pydantic_default(o):
    """Default handler for Pydantic models. Defined once at module level (no closure)."""
    if isinstance(o, BaseModel):
        if hasattr(o, 'model_dump'):
            return o.model_dump()
        elif hasattr(o, 'dict'):
            return o.dict()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class _ChainedDefault:
    """Callable that tries Pydantic serialization, then falls back to user default."""
    __slots__ = ('_user_default',)

    def __init__(self, user_default):
        self._user_default = user_default

    def __call__(self, o):
        if isinstance(o, BaseModel):
            if hasattr(o, 'model_dump'):
                return o.model_dump()
            elif hasattr(o, 'dict'):
                return o.dict()
        return self._user_default(o)


def _patched_json_dumps(obj, *args, **kwargs):
    """Patched json.dumps that handles Pydantic models."""
    if 'default' not in kwargs:
        kwargs['default'] = _pydantic_default
    else:
        kwargs['default'] = _ChainedDefault(kwargs['default'])
    return _original_json_dumps(obj, *args, **kwargs)

# Apply the monkey patch
json.dumps = _patched_json_dumps


# llm instrumentation packages 
#from langtrace_python_sdk import langtrace
from traceloop.sdk import Traceloop


from app.tracing import init_tracing
from app.config import config

import uvicorn
import os

# ---------------------------------------------------------------------------
# Crash / exception / hang diagnostics → log file (size-bounded)
#
# All four servers share one event loop, so a single blocking call freezes
# every endpoint at once.  We capture three kinds of evidence to a file so it
# survives container restarts and can be inspected after the fact:
#
#   1. Application errors/exceptions logged via the `logging` module
#      (e.g. the bedrock stream traceback) → rotating file `app.log`.
#   2. Uncaught exceptions on any thread (main, worker, asyncio) → `app.log`.
#   3. Low-level fatal faults (segfault, abort) + on-demand all-thread stack
#      dumps for diagnosing a hang → `faulthandler.log`.
#
# All files live under LOG_DIR (default: ./logs, mounted via docker-compose).
# Sizes are bounded: app.log rotates (LOG_MAX_BYTES × LOG_BACKUP_COUNT), and
# faulthandler.log is truncated at startup + before each dump.
#
# Getting an all-thread stack dump for a hang:
#   • On demand: `kill -SIGUSR1 <pid>` (host: `docker kill --signal=SIGUSR1 <ctr>`;
#     inside the container if run.py is PID 1: `kill -SIGUSR1 1`).
#   • Automatically when a stream stalls: the stall watchdog (app/diagnostics.py)
#     dumps all-thread stacks when a streaming request makes no progress for
#     STREAM_STALL_DUMP_SECONDS (default 20) — short enough to fire BEFORE the
#     client gives up and disconnects, capturing the live blocking frame.
# ---------------------------------------------------------------------------
import atexit
import faulthandler
import logging
import signal
import sys
import threading
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.getenv("LOG_DIR", "logs")
_LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))   # 10 MiB / file
_LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))               # → ~60 MiB cap
_FAULT_MAX_BYTES = int(os.getenv("FAULTHANDLER_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MiB cap

# Keep the faulthandler file handle alive for the whole process lifetime.
_fault_fp = None


def _setup_diagnostics() -> None:
    """Wire up file-based crash/exception/hang logging. Best-effort: never fatal."""
    global _fault_fp
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception as e:  # noqa: BLE001 - diagnostics must not break startup
        print(f"[diagnostics] could not create LOG_DIR={_LOG_DIR!r}: {e}", file=sys.stderr)
        return

    # (1) Application logs → rotating file. Attached to the root logger so every
    #     module's logger (and exc_info=True tracebacks) is captured. We add a
    #     handler rather than replacing existing ones so `docker logs` still works.
    app_log_path = os.path.join(_LOG_DIR, "app.log")
    try:
        file_handler = RotatingFileHandler(
            app_log_path,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ))
        file_handler.setLevel(logging.WARNING)  # warnings + errors + tracebacks
        root_logger = logging.getLogger()
        if root_logger.level == logging.NOTSET or root_logger.level > logging.WARNING:
            root_logger.setLevel(logging.WARNING)
        root_logger.addHandler(file_handler)
    except Exception as e:  # noqa: BLE001
        print(f"[diagnostics] could not open {app_log_path}: {e}", file=sys.stderr)

    _diag_logger = logging.getLogger("diagnostics")

    # (2) Uncaught exceptions on every thread → app.log (+ still printed to stderr).
    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        _diag_logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = _log_uncaught

    def _log_thread_uncaught(args):
        if args.exc_type is SystemExit:
            return
        _diag_logger.critical(
            "Uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _log_thread_uncaught

    # (3) faulthandler → dedicated file. Truncated at startup so it never carries
    #     unbounded history across restarts.
    fault_log_path = os.path.join(_LOG_DIR, "faulthandler.log")
    try:
        _fault_fp = open(fault_log_path, "w", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_fault_fp, all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            # chain=False: do NOT call the previous handler. SIGUSR1's default OS
            # disposition is to terminate the process, so chaining would kill the
            # server right after dumping. We only want the dump.
            faulthandler.register(
                signal.SIGUSR1, file=_fault_fp, all_threads=True, chain=False
            )
    except Exception as e:  # noqa: BLE001
        print(f"[diagnostics] could not open {fault_log_path}: {e}", file=sys.stderr)
        faulthandler.enable()  # fall back to stderr so we still get fatal dumps
        return

    # (4) Stall watchdog: dump all-thread stacks just before a stalled streaming
    #     request hits its per-chunk timeout. Streaming wrappers arm/reset/disarm
    #     a deadline per request (see app/routes/stream_utils.py); if a request
    #     makes no progress until the deadline, the watchdog captures the live
    #     blocking frame — far more useful than periodic snapshots of idle threads.
    try:
        from app import diagnostics

        diagnostics.init_watchdog(_fault_fp, _FAULT_MAX_BYTES)
        atexit.register(diagnostics.shutdown)  # stop the daemon thread on clean exit
        print(
            "[diagnostics] stall watchdog armed; stalled streams dump to "
            f"{fault_log_path}"
        )
    except Exception as e:  # noqa: BLE001 - diagnostics must never break startup
        print(f"[diagnostics] could not start stall watchdog: {e}", file=sys.stderr)


_setup_diagnostics()

# start tracing only if OTEL_EXPORTER_OTLP_ENDPOINT is set
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):

    # Initialize LLM Instrumentor (it will set up the TracerProvider and exporters)
    #langtrace.init()

    Traceloop.init(
        app_name=os.getenv("OTEL_SERVICE_NAME"),
        api_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        headers=os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    )
    
    # Initialize OpenTelemetry tracing utilities
    init_tracing()


async def _run_multi_server():
    """Run four servers concurrently in one process."""
    import asyncio

    from app.main import create_openai_app, create_anthropic_app, create_azure_openai_app, create_management_app

    servers = []

    # OpenAI API server
    openai_app = create_openai_app()
    openai_cfg = uvicorn.Config(
        openai_app,
        host=config.server.host,
        port=config.server.openai_port,
        log_level="info",
    )
    servers.append(uvicorn.Server(openai_cfg))

    # Anthropic API server
    anthropic_app = create_anthropic_app()
    anthropic_cfg = uvicorn.Config(
        anthropic_app,
        host=config.server.host,
        port=config.server.anthropic_port,
        log_level="info",
    )
    servers.append(uvicorn.Server(anthropic_cfg))

    # Azure OpenAI API server
    azure_openai_app = create_azure_openai_app()
    azure_openai_cfg = uvicorn.Config(
        azure_openai_app,
        host=config.server.host,
        port=config.server.azure_openai_port,
        log_level="info",
    )
    servers.append(uvicorn.Server(azure_openai_cfg))

    # Management server
    mgmt_app = create_management_app()
    mgmt_cfg = uvicorn.Config(
        mgmt_app,
        host=config.server.host,
        port=config.server.management_port,
        log_level="info",
    )
    servers.append(uvicorn.Server(mgmt_cfg))

    print("Starting LLM Proxy Server (multi-server mode)...")
    print(f"  OpenAI API         → http://{config.server.host}:{config.server.openai_port}")
    print(f"  Anthropic API      → http://{config.server.host}:{config.server.anthropic_port}")
    print(f"  Azure OpenAI API   → http://{config.server.host}:{config.server.azure_openai_port}")
    print(f"  Management         → http://{config.server.host}:{config.server.management_port}")
    print()

    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    import asyncio
    asyncio.run(_run_multi_server())

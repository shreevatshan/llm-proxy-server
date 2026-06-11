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

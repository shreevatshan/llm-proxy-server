"""
OpenTelemetry Context Detach Patch - MUST BE IMPORTED FIRST!

This module patches OpenTelemetry's context.detach() function to handle
the "different Context" ValueError that occurs in async Python applications.

IMPORTANT: This module MUST be imported before ANY other OpenTelemetry code,
including instrumentation libraries like FastAPIInstrumentor, Traceloop, etc.

Usage:
    # In run.py or at the very top of your entry point:
    import app.otel_patch  # noqa: F401  # Must be first!
    
    # Then your other imports...
    from app.main import app

The Problem:
    Python's contextvars have a limitation where tokens become invalid when
    execution crosses async boundaries (task switches, event loop yields).
    When OpenTelemetry's instrumentation tries to detach tokens after these
    async boundaries, a ValueError is raised:
    
        ValueError: <Token...> was created in a different Context
    
    This error is benign - the context has already been cleaned up by the
    async runtime. This patch suppresses these expected errors.

Reference:
    https://github.com/open-telemetry/opentelemetry-python/issues/2606
"""

import logging

# Set up logging early - before any patches
logger = logging.getLogger(__name__)

# =============================================================================
# CRITICAL: Patch must happen before any other OpenTelemetry imports!
# =============================================================================

# Import only the context module - avoid importing anything else from OTel
from opentelemetry import context as otel_context

# Store the original detach function
_original_detach = otel_context.detach
_patch_applied = False


def _safe_global_detach(token):
    """
    Safe wrapper for otel_context.detach() that handles async context issues.
    
    In async Python, contextvars tokens can become invalid when execution 
    crosses async boundaries. This is expected behavior and the error is benign.
    
    This function wraps the original detach() to catch and suppress only the
    specific "different Context" ValueError that occurs in async scenarios.
    All other errors are propagated normally.
    """
    try:
        _original_detach(token)
    except ValueError as e:
        # Only suppress the specific "different Context" error
        error_msg = str(e)
        if "different Context" in error_msg or "was created in a different Context" in error_msg:
            # This is the expected async context error - safe to ignore
            # Use print to avoid any logging issues at import time
            pass  # Silently ignore expected error
        else:
            # Unexpected ValueError - re-raise it
            raise


def apply_patch():
    """Apply the monkey-patch to otel_context.detach if not already applied."""
    global _patch_applied

    if _patch_applied:
        return

    # Check if already patched (e.g., by app/tracing.py)
    if otel_context.detach is not _original_detach and otel_context.detach.__name__ == '_safe_global_detach':
        _patch_applied = True
        return

    otel_context.detach = _safe_global_detach
    _patch_applied = True

    # Use print since logging might not be fully configured yet
    print("✓ OpenTelemetry context.detach() patched for async safety (early patch)")


def patch_bedrock_cross_region():
    """Add 'global' to bedrock cross-region prefixes so Anthropic span attributes work.

    The OTel bedrock instrumentor's _cross_region_check only knows about
    us/us-gov/eu/apac prefixes. Without 'global', model IDs like
    'global.anthropic.claude-haiku-4-5-...' are parsed as vendor='global',
    which prevents the prompt/response/token attributes from being set.
    """
    try:
        import opentelemetry.instrumentation.bedrock as bedrock_init

        def _cross_region_check_with_global(value):
            prefixes = ["us", "us-gov", "eu", "apac", "global"]
            if any(value.startswith(prefix + ".") for prefix in prefixes):
                parts = value.split(".")
                if len(parts) > 2:
                    parts.pop(0)
                return parts[0], parts[1]
            return value.split(".", 1)

        bedrock_init._cross_region_check = _cross_region_check_with_global
        print("✓ Bedrock cross-region prefix patched to include 'global'")
    except Exception as e:
        print(f"⚠ Failed to patch bedrock cross-region check: {e}")


# Apply patch immediately when module is imported
apply_patch()
patch_bedrock_cross_region()

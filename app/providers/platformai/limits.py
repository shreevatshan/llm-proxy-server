"""Per-model limits learned from PlatformAI refusals.

The gateway publishes no model catalogue, so what a model accepts is learned
from its 400s and remembered (per provider and model, with a TTL so a
gateway-side upgrade is picked up again):

- ``PAI-MODEL-002 "<param> not supported for the model <m>"``: the parameter
  is dropped from later requests (``response_format`` falls back to a prompt
  instruction);
- reasoning-effort refusals: the effort steps down a level;
- ``max_tokens`` refusals naming a ceiling: the ceiling is applied.

Sampling params a model family rejects everywhere (Claude >= 4.7, GPT-5
reasoning models) come from ``app.model_capabilities`` and are applied in
``translate.build_chat_body``. ``DEFAULT_UNSUPPORTED_PARAMS`` holds only what
this gateway refuses on top of that, so known refusals need no retry.

State is module-level because provider instances are rebuilt on every admin
change.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

LEARNED_LIMIT_TTL_SECONDS = 7 * 24 * 3600

_Key = Tuple[str, str]

_EFFORT_CEILINGS: Dict[_Key, Tuple[str, float]] = {}
_MAX_TOKEN_CEILINGS: Dict[_Key, Tuple[int, float]] = {}
_UNSUPPORTED_PARAMS: Dict[_Key, Dict[str, float]] = {}

# Gateway-specific refusals (PAI-MODEL-002, verified live 2026-11-04) that
# app.model_capabilities does not cover because the model itself accepts them
# elsewhere. Keyed by upstream model name; these never expire, and anything not
# listed is still learned at runtime.
DEFAULT_UNSUPPORTED_PARAMS: Dict[str, FrozenSet[str]] = {
    "claude-haiku-5-5": frozenset({"response_format", "variations"}),
}


def reset_limits() -> None:
    """Forget every learned limit (tests)."""
    _EFFORT_CEILINGS.clear()
    _MAX_TOKEN_CEILINGS.clear()
    _UNSUPPORTED_PARAMS.clear()


# ---------------------------------------------------------------------------
# Unsupported parameters
# ---------------------------------------------------------------------------

def unsupported_params(provider_key: str, model: str) -> FrozenSet[str]:
    """Gateway refusals known for ``model`` plus the ones learned and not yet expired."""
    now = time.time()
    entries = _UNSUPPORTED_PARAMS.get((provider_key, model)) or {}
    learned = {p for p, expires in entries.items() if expires > now}
    return DEFAULT_UNSUPPORTED_PARAMS.get(model.lower(), frozenset()) | learned


def learn_unsupported_param(provider_key: str, model: str, param: str) -> None:
    _UNSUPPORTED_PARAMS.setdefault((provider_key, model), {})[param] = time.time() + LEARNED_LIMIT_TTL_SECONDS
    logger.warning("PlatformAI: model %s does not support %s; dropping it from later requests", model, param)


_UNSUPPORTED_PARAM = re.compile(r"^\s*([A-Za-z_][\w.]*) not supported for the model\b", re.I)


def parse_unsupported_param(message: str) -> Optional[str]:
    """The top-level body key a ``PAI-MODEL-002`` message names, if any."""
    match = _UNSUPPORTED_PARAM.search(message or "")
    return match.group(1).split(".", 1)[0] if match else None


# ---------------------------------------------------------------------------
# Reasoning effort
# ---------------------------------------------------------------------------

def effort_ceiling(provider_key: str, model: str) -> Optional[str]:
    hit = _EFFORT_CEILINGS.get((provider_key, model))
    if hit and hit[1] > time.time():
        return hit[0]
    return None


def learn_effort_ceiling(provider_key: str, model: str, ceiling: Optional[str]) -> None:
    """``None`` means the model takes no effort at all (stored as ``"none"``)."""
    _EFFORT_CEILINGS[(provider_key, model)] = (ceiling or "none", time.time() + LEARNED_LIMIT_TTL_SECONDS)
    logger.warning("PlatformAI: model %s accepts reasoning effort up to %s; remembered", model, ceiling or "none")


_EFFORT_STEP_DOWN = re.compile(
    r"does not support effort level|does not support '?(?:minimal|low|medium|high|xhigh)'?", re.I
)
# "Supported values are ..." names the accepted values of whichever parameter was
# wrong; it is an effort refusal only when the message is about effort.
_SUPPORTED_VALUES = re.compile(r"Supported values are", re.I)
_EFFORT_MENTION = re.compile(r"effort|reasoning", re.I)


def is_effort_refusal(message: str) -> bool:
    return bool(_EFFORT_STEP_DOWN.search(message) or (
        _SUPPORTED_VALUES.search(message) and _EFFORT_MENTION.search(message)
    ))


# ---------------------------------------------------------------------------
# max_tokens
# ---------------------------------------------------------------------------

def max_tokens_ceiling(provider_key: str, model: str) -> Optional[int]:
    hit = _MAX_TOKEN_CEILINGS.get((provider_key, model))
    if hit and hit[1] > time.time():
        return hit[0]
    return None


def learn_max_tokens_ceiling(provider_key: str, model: str, ceiling: int) -> None:
    _MAX_TOKEN_CEILINGS[(provider_key, model)] = (ceiling, time.time() + LEARNED_LIMIT_TTL_SECONDS)
    logger.warning("PlatformAI: model %s caps max_tokens at %d; remembered", model, ceiling)


_MAX_TOKENS_MENTION = re.compile(r"max[_ ]?tokens|maximum (?:allowed )?(?:number of )?output tokens|output token", re.I)
# Input + max_tokens over the context window depends on this request's prompt
# size, so it says nothing about the model's output ceiling.
_CONTEXT_OVERFLOW = re.compile(
    r"context (?:limit|length|window)|input length|prompt is too long|maximum context|\d\s*\+\s*\d", re.I
)


def parse_max_tokens_limit(message: str, requested: int) -> Optional[int]:
    """The ceiling a max_tokens refusal names: the largest number below ``requested``."""
    if not _MAX_TOKENS_MENTION.search(message) or _CONTEXT_OVERFLOW.search(message):
        return None
    numbers = [int(n) for n in re.findall(r"\d{3,}", message.replace(",", ""))]
    below = [n for n in numbers if 0 < n < requested]
    return max(below) if below else None


__all__ = [
    "DEFAULT_UNSUPPORTED_PARAMS", "reset_limits", "unsupported_params", "learn_unsupported_param", "parse_unsupported_param",
    "effort_ceiling", "learn_effort_ceiling", "is_effort_refusal",
    "max_tokens_ceiling", "learn_max_tokens_ceiling", "parse_max_tokens_limit",
]


"""Per-model request-parameter capabilities.

Providers must not forward request parameters the target model rejects — each
one comes back as an upstream 400. Historically each model release added its own
ad-hoc guard inside whichever request builder happened to break first, so the
four builders drifted: the same model got different scrubbing depending on which
API surface the client used, and a Claude 5 request kept a ``temperature`` that
Bedrock's Converse path had been stripping since 4.7.

This module is the single declarative source for "which params does this model
reject". Every request builder consults it, so adding a model is one row in
``RULES`` rather than an edit in three provider files.

Parameter names are the canonical Anthropic-native spellings (``top_p``, not
``topP``); callers working in another dialect pass a ``spellings`` map to
:func:`scrub`.

``is_claude_at_least`` lives here rather than in ``app.anthropic_models`` so that
module can depend on this one without a cycle; it is re-exported from
``app.anthropic_models`` for existing callers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Request surfaces. The dialect differs (field names, nesting), and so do some
# of the restrictions, so rules may be scoped to one of them.
SURFACE_NATIVE = "native"      # Anthropic-shaped bodies (messages.create kwargs, native JSON)
SURFACE_CONVERSE = "converse"  # Bedrock Converse args

# Model families, including the ones that arrived after the sonnet/opus/haiku
# trio: "claude-fable-5-1" and "claude-mythos-5-1" reject the same params as the
# rest of the 5 family, so they must parse rather than silently match no rule.
_CLAUDE_FAMILIES = ("sonnet", "opus", "haiku", "fable", "mythos")
_CLAUDE_FAMILY = r'(?:' + '|'.join(_CLAUDE_FAMILIES) + r')'

# Two naming shapes, both still in service:
#   family-first  — "claude-opus-4-7", "claude-fable-5-1"  (current)
#   version-first — "claude-3-5-sonnet-20241022"           (Claude 3 era)
# The Claude 3 shape needs no scrubbing, but it must still parse so that
# warn_if_version_unparseable does not flag a correctly-named deployment.
_CLAUDE_VERSION_RE = re.compile(
    rf'claude-{_CLAUDE_FAMILY}-(?P<major>\d+)(?:-(?P<minor>\d+))?'
    rf'|claude-(?P<vmajor>\d+)(?:-(?P<vminor>\d+))?-{_CLAUDE_FAMILY}'
)


def _parse_claude_version(model_id: str) -> Optional[Tuple[int, int]]:
    """Parse ``(major, minor)`` out of a Claude model id, or None."""
    if not model_id:
        return None
    lower = model_id.lower()
    if "claude" not in lower:
        return None
    match = _CLAUDE_VERSION_RE.search(lower)
    if not match:
        return None
    major_raw = match.group("major") or match.group("vmajor")
    minor_raw = match.group("minor") or match.group("vminor")
    # An 8-digit segment is a date snapshot ("claude-sonnet-5-20250101"), not a
    # minor version. A missing minor is treated as 0.
    minor = int(minor_raw) if (minor_raw is not None and len(minor_raw) <= 2) else 0
    return int(major_raw), minor


def is_claude_at_least(model_id: str, min_major: int, min_minor: int) -> bool:
    """True if ``model_id`` is a Claude model at or above ``min_major.min_minor``.

    Handles both the "major-minor" naming (e.g. "claude-sonnet-4-5") and the
    newer "major only" naming (e.g. "claude-sonnet-5"), and ignores any trailing
    date snapshot (e.g. "claude-sonnet-5-20250101") — an 8-digit segment is a
    date, not a minor version. A missing minor is treated as 0.
    """
    version = _parse_claude_version(model_id)
    if version is None:
        return False
    return version >= (min_major, min_minor)


def is_claude_model(model_id: str) -> bool:
    """True if ``model_id`` names a Claude model with a parseable version."""
    return _parse_claude_version(model_id) is not None


def claude_at_least(min_major: int, min_minor: int) -> Callable[[str], bool]:
    """Build a rule predicate matching Claude models >= the given version."""
    def _match(model_id: str) -> bool:
        return is_claude_at_least(model_id, min_major, min_minor)
    return _match


def is_grok(model_id: str) -> bool:
    """True if ``model_id`` refers to an xAI Grok model."""
    lower = (model_id or "").lower()
    return "xai." in lower or "grok" in lower


# Thinking configuration styles.
THINKING_BUDGET = "budget"      # {"type": "enabled", "budget_tokens": N}
THINKING_ADAPTIVE = "adaptive"  # {"type": "adaptive"} — budget_tokens is rejected


@dataclass(frozen=True)
class CapabilityRule:
    """One "this family rejects these params" fact.

    ``surfaces=None`` means the rule holds on every surface. Rules are additive:
    :func:`rejected_params` unions every rule that matches.
    """

    match: Callable[[str], bool]
    rejects: FrozenSet[str] = field(default_factory=frozenset)
    surfaces: Optional[FrozenSet[str]] = None
    thinking_style: Optional[str] = None
    note: str = ""

    def applies_to(self, model_id: str, surface: str) -> bool:
        if self.surfaces is not None and surface not in self.surfaces:
            return False
        return self.match(model_id)


RULES: Tuple[CapabilityRule, ...] = (
    # 4.7 is the boundary for the whole current generation, on every surface:
    # temperature, top_p, top_k and thinking.budget_tokens were all *removed*
    # (not deprecated) in 4.7 and each returns a 400 on 4.7, 4.8 and the 5
    # family, adaptive being the only on-mode for thinking. 4.6 still accepts
    # all four (budget_tokens deprecated but functional) and Haiku 4.5 still
    # requires budget_tokens, so neither is covered here.
    #
    # This deliberately spans both surfaces. The earlier split — temperature
    # scrubbed only on Converse, budget_tokens only from 5.0 — was the drift
    # this module exists to remove: it left 4.7/4.8 forwarding temperature and
    # budget_tokens on all three native paths.
    CapabilityRule(
        match=claude_at_least(4, 7),
        rejects=frozenset({"temperature", "top_p", "top_k"}),
        thinking_style=THINKING_ADAPTIVE,
        note="Claude >= 4.7 removed temperature/top_p/top_k and the thinking token budget",
    ),
    # Grok on Bedrock accepts only maxTokens in inferenceConfig; each of these
    # comes back as "This model doesn't support the <field> field".
    CapabilityRule(
        match=is_grok,
        rejects=frozenset({"temperature", "top_p", "stop_sequences"}),
        surfaces=frozenset({SURFACE_CONVERSE}),
        note="Grok on Bedrock accepts only maxTokens in inferenceConfig",
    ),
)

# Converse spells the sampling params differently from the native API.
CONVERSE_SPELLINGS: Dict[str, str] = {
    "top_p": "topP",
    "stop_sequences": "stopSequences",
}


def rejected_params(model_id: str, surface: str = SURFACE_NATIVE) -> FrozenSet[str]:
    """Canonical names of the params ``model_id`` rejects on ``surface``."""
    rejected: set = set()
    for rule in RULES:
        if rule.applies_to(model_id, surface):
            rejected |= rule.rejects
    return frozenset(rejected)


def thinking_style(model_id: str) -> str:
    """How ``model_id`` expects extended thinking to be configured."""
    for rule in RULES:
        if rule.thinking_style and rule.match(model_id):
            return rule.thinking_style
    return THINKING_BUDGET


def scrub(
    mapping: MutableMapping[str, Any],
    model_id: str,
    surface: str = SURFACE_NATIVE,
    *,
    spellings: Optional[Dict[str, str]] = None,
    only: Optional[FrozenSet[str]] = None,
) -> List[str]:
    """Remove rejected params from ``mapping`` in place.

    ``spellings`` maps a canonical name to this dialect's key (see
    :data:`CONVERSE_SPELLINGS`). ``only`` restricts the scrub to a subset of the
    canonical names, for callers that hold the remaining params elsewhere.

    Returns the canonical names actually removed, sorted — suitable both for the
    providers' ``dropped_fields`` metadata and for their debug logging.
    """
    if mapping is None:
        return []
    rejected = rejected_params(model_id, surface)
    if only is not None:
        rejected &= only
    spellings = spellings or {}

    removed = [
        name for name in rejected
        if mapping.pop(spellings.get(name, name), None) is not None
    ]
    return sorted(removed)


def normalize_thinking(
    thinking: Optional[Dict[str, Any]],
    model_id: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Reshape a thinking config for ``model_id``.

    On adaptive-thinking models an explicit token budget is rejected, so
    ``{"type": "enabled", "budget_tokens": N}`` becomes ``{"type": "adaptive"}``.
    ``disabled`` and already-``adaptive`` configs pass through untouched.

    Only ``budget_tokens`` is removed — every sibling key is preserved. Dropping
    them all would silently discard ``display``, and since ``omitted`` is the
    default on these models the client would get empty thinking blocks instead
    of the summary it asked for, with nothing reported in ``dropped``.

    Returns the config to send and the list of dropped field names.
    """
    if not isinstance(thinking, dict):
        return thinking, []
    if thinking_style(model_id) != THINKING_ADAPTIVE:
        return thinking, []
    if thinking.get("type") != "enabled":
        return thinking, []

    dropped = ["thinking.budget_tokens"] if "budget_tokens" in thinking else []
    if dropped:
        logger.debug(
            "Converted thinking budget to adaptive for %s (budget_tokens is rejected)",
            model_id,
        )
    normalized = {k: v for k, v in thinking.items() if k != "budget_tokens"}
    normalized["type"] = "adaptive"
    return normalized, dropped


def warn_if_version_unparseable(model_id: str, context: str) -> bool:
    """Warn when a Claude-looking id has no parseable version, and report it.

    Every rule above keys off the version parsed out of the model string. On
    Azure Foundry that string is the operator-chosen deployment name, so a
    deployment called "opus5-prod" matches no rule and silently keeps params the
    model will reject. Surfacing that at startup beats debugging it per request.
    """
    lower = (model_id or "").lower()
    # Derived from the family list rather than repeated, so a family added there is
    # never one this heuristic silently stops recognising — a deployment named
    # "fable5-prod" would otherwise get neither the scrub nor the warning.
    looks_like_claude = any(
        marker in lower for marker in ("claude",) + _CLAUDE_FAMILIES
    )
    if not looks_like_claude or is_claude_model(model_id):
        return False
    logger.warning(
        "%s: model/deployment name %r has no parseable Claude version, so "
        "per-model parameter restrictions cannot be applied to it. Rename it to "
        "include e.g. 'claude-opus-5' to enable them.",
        context,
        model_id,
    )
    return True

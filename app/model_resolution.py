"""Prefix-less model resolution, called from route handlers after authentication.

Models are addressed canonically as ``{provider_key}/{bare_name}`` (e.g.
``azure:foundry/gpt-5.4``). This module lets a client name a model without
knowing the proxy's provider topology: ``gpt-5.4`` resolves to whichever
configured provider serves it.

A name is first *evaluated* for the caller (_evaluate), in this order:

    1. falsy                                      -> unchanged
    2. an exact canonical id                      -> itself, if the caller may
       use it (else denied)
    3. '{head}/{rest}' where head names a live provider (or an ambiguous family
       of them) -> that provider (canonicalised when the family names one
       instance), if the caller may use it. Honoured even when the model is
       absent from the cache, as Azure deployments frequently are, and
       _get_provider asks for disambiguation when it has to.
    4. the whole string as a bare name            -> round-robin over candidates
    5. `rest` as a bare name (strip head exactly once) -> round-robin
    6. otherwise                                  -> unavailable

Step 4 precedes step 5 because a bare name may itself contain '/': an
OpenAI-compatible backend reporting ``meta-llama/Llama-3.1-8B`` produces the id
``lmstudio:box/meta-llama/Llama-3.1-8B``.

Admin mappings come in two lists per profile, and the evaluation of the
*client's* name picks one (map_model_name, resolve_model_for_request):

    available to the caller   -> the regular list; its target is evaluated in
                                 turn, and its failures are reported against it
    unavailable or denied     -> the fallback list; if nothing matches, or the
                                 target cannot be served either, the client's
                                 name fails: ModelUnavailable (404) or
                                 ModelAccessDenied (403)

Resolution happens in the route rather than in middleware so the candidate pool
can be narrowed by the caller's per-user access *before* a provider is picked --
otherwise a user entitled to one of two providers would fail on every other
request. It must run before ``enforce_group_rate_limit`` /
``enforce_model_access``, which key on the provider prefix.

Layering: this module imports ``provider_manager`` at module scope, exactly as
``app/model_access_dep.py`` does. Nothing under ``app/providers/`` or
``app/auth/`` may import it at module scope in return -- use a local import
(``app/auth/middleware.py`` does).
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from app.api_envelope import envelope_for
from app.model_access_dep import ModelAccessDenied, caller_may_use
from app.model_alias import (
    alias_source_name, apply_alias, current_api_surface, current_profile_id, model_alias_resolver,
    original_model_name,
)
from app.profiles import profile_id_for_auth, profile_registry
from app.providers.provider_manager import provider_manager

logger = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    """No configured provider serves the requested model.

    Caught by a per-app exception handler (registered in app/main.py) which
    serializes `body` with `status_code`. Mirrors ModelAccessDenied.
    """

    def __init__(self, body: dict, status_code: int = 404):
        self.body = body
        self.status_code = status_code
        super().__init__("Model unavailable")

    @classmethod
    def openai(cls, model_id: str) -> "ModelUnavailable":
        return cls(body={"error": {
            "message": _message(model_id),
            "type": "invalid_request_error",
            "code": "model_not_found",
            "param": "model",
        }})

    @classmethod
    def azure(cls, model_id: str) -> "ModelUnavailable":
        # Azure uses the OpenAI error envelope.
        return cls(body={"error": {
            "message": _message(model_id),
            "type": "invalid_request_error",
            "code": "model_not_found",
            "param": "model",
        }})

    @classmethod
    def anthropic(cls, model_id: str) -> "ModelUnavailable":
        return cls(body={"type": "error", "error": {
            "type": "not_found_error",
            "message": _message(model_id),
        }})


def _message(model_id: str) -> str:
    return f"The model '{model_id}' does not exist or you do not have access to it."


class _RoundRobin:
    """Rotation cursors for bare-name -> canonical-id selection.

    Keyed on the sorted candidate tuple rather than the bare name: the pool is a
    function of (cache contents x API surface x per-user access), so two users or
    two surfaces are genuinely different rotations, and an index valid for a
    3-entry pool would mean something else for a 2-entry pool. A changed pool
    yields a new key starting at 0, so a cache refresh is self-healing.

    pick() contains no `await`, so its read-modify-write runs to completion
    within one event-loop tick and needs no lock -- the same argument documented
    for the ModelCache sync mutators (app/cache.py). Rotation is per-process: a
    multi-worker deployment rotates independently per worker.
    """

    _MAX_KEYS = 4096

    def __init__(self) -> None:
        self._cursors: Dict[Tuple[str, ...], int] = {}

    def pick(self, candidates: List[str]) -> str:
        """Return the next candidate. `candidates` must be non-empty and sorted."""
        if len(candidates) == 1:
            return candidates[0]
        key = tuple(candidates)
        # Modulo on read keeps a stale cursor in range for any pool size.
        index = self._cursors.get(key, 0) % len(candidates)
        if key not in self._cursors and len(self._cursors) >= self._MAX_KEYS:
            # Rotation is a load-spreading heuristic, not a correctness property;
            # dropping the cursors only restarts rotation.
            self._cursors.clear()
        self._cursors[key] = (index + 1) % len(candidates)
        return candidates[index]


_round_robin = _RoundRobin()


def candidate_ids(
    name: Optional[str],
    *,
    api: Optional[str] = None,
    allow_prefix_strip: bool = True,
) -> List[str]:
    """Sorted canonical ids a prefix-less `name` could resolve to.

    Applies only the caller-independent filters, so this is safe to call before
    authentication. Returns [] when `name` is falsy, is already a canonical id,
    names an available provider, or matches nothing. Consumes no rotation state.
    """
    if not name:
        return []

    cache = provider_manager.model_cache
    if cache.has_model_id(name):
        return []

    head, rest = name.split('/', 1) if '/' in name else ('', '')
    if rest and provider_manager.has_provider_prefix(head):
        return []

    pool = _filter_pool(cache.bare_model_candidates(name), api)
    if not pool and allow_prefix_strip and rest:
        pool = _filter_pool(cache.bare_model_candidates(rest), api)
    return pool


def _filter_pool(pool: List[str], api: Optional[str]) -> List[str]:
    """Drop candidates that could only fail, then globally disabled ones."""
    if not pool:
        return []

    # 1. The provider must still be registered; otherwise _get_provider raises
    #    "not available" and every Nth request would fail deterministically.
    live = []
    for model_id in pool:
        provider = provider_manager.providers.get(model_id.split('/', 1)[0])
        if provider is not None:
            live.append((model_id, provider))
    if not live:
        return []

    # 2. The model must serve the surface the request arrived on -- a model that
    #    does not serve /v1/messages is a guaranteed downstream 404. This
    #    predicate is partly heuristic (BaseProvider._is_chat_capable_model is
    #    documented best-effort and matches substrings like "speech"), so it must
    #    never be the reason a pool empties: a name the user could reach with an
    #    explicit canonical id must not 404 just because it was written bare.
    if api:
        supported = []
        for model_id, provider in live:
            try:
                if provider.supports_api_for_model(model_id, api):
                    supported.append((model_id, provider))
            except Exception:
                logger.debug("supports_api_for_model failed for %s", model_id, exc_info=True)
                supported.append((model_id, provider))
        if supported:
            live = supported

    # 3. Globally disabled models are out of rotation. No backstop here: if an
    #    all-disabled pool still resolved, "disabled" would mean nothing.
    cache = provider_manager.model_cache
    return [model_id for model_id, _ in live if cache._passes_global_gate(model_id)]


# ----------------------------------------------------------------------
# Evaluation: can this caller be served this name? (no side effects)
# ----------------------------------------------------------------------

# Stands for "not authenticated yet" (the tracking middleware): only the global gate
# is known. Distinct from None, which resolve_model_for_request treats as an
# identity-less caller that bypasses the per-user check.
PRE_AUTH = object()


@dataclass(frozen=True)
class _Evaluation:
    status: str                     # "ok" | "unavailable" | "denied"
    candidates: Tuple[str, ...] = ()  # canonical ids the caller may use (ok only)


_UNAVAILABLE = _Evaluation("unavailable")
_DENIED = _Evaluation("denied")


def _may_use_for(auth) -> Callable[[str], bool]:
    """The per-model access predicate for `auth` (PRE_AUTH: the global gate alone).

    After auth, the global gate on bare names is applied by _filter_pool, as before;
    this adds the per-user decision. AdminUser and identity-less principals bypass
    the per-user check, mirroring enforce_model_access (app/model_access_dep.py).
    """
    if auth is PRE_AUTH:
        return provider_manager.model_cache._passes_global_gate
    return lambda model_id: caller_may_use(auth, model_id)


def may_use_for_profile(profile_id: Optional[int]) -> Callable[[str], bool]:
    """The access predicate for a member of `profile_id` (the admin test endpoint)."""
    profile = profile_registry.get(profile_id)
    cache = provider_manager.model_cache
    return lambda model_id: cache.is_model_allowed_for_profile(profile, model_id)


def _canonical_for_prefix(model: str) -> str:
    """'azure/gpt-4' -> 'azure:primary/gpt-4' when the family names one instance, else `model`."""
    head, rest = model.split('/', 1) if '/' in model else ('', '')
    provider_key = provider_manager.find_provider_key(head) if rest else None
    if provider_key is None or provider_key == head:
        return model
    return f"{provider_key}/{rest}"


def _evaluate(
    name: str,
    may_use: Callable[[str], bool],
    api: Optional[str],
    allow_prefix_strip: bool = True,
) -> _Evaluation:
    """Whether a caller with `may_use` can be served `name`, in the module's resolution order."""
    cache = provider_manager.model_cache
    if cache.has_model_id(name):
        # A disabled model or provider fails may_use (the global gate) for everyone
        # but admins and allow-mode profiles, which may still call it by id.
        return _Evaluation("ok", (name,)) if may_use(name) else _DENIED

    head, rest = name.split('/', 1) if '/' in name else ('', '')
    if rest and provider_manager.has_provider_prefix(head):
        canonical = _canonical_for_prefix(name)
        return _Evaluation("ok", (canonical,)) if may_use(canonical) else _DENIED

    pool = _filter_pool(cache.bare_model_candidates(name), api)
    if not pool and allow_prefix_strip and rest:
        pool = _filter_pool(cache.bare_model_candidates(rest), api)
    if not pool:
        return _UNAVAILABLE
    allowed = tuple(model_id for model_id in pool if may_use(model_id))
    return _Evaluation("ok", allowed) if allowed else _DENIED


def is_model_available(name: str, may_use: Callable[[str], bool], api: Optional[str]) -> bool:
    """True if a caller with `may_use` can be served `name` on `api` (picks the regular list)."""
    return bool(name) and _evaluate(name, may_use, api).status == "ok"


def map_model_name(name: Optional[str], auth=PRE_AUTH, *, allow_prefix_strip: bool = True) -> Optional[str]:
    """Apply the regular mappings when the client's name is available to the caller,
    the fallback mappings otherwise. Never raises; resolve_model_for_request does.

    For layers that map without resolving: the tracking middleware (PRE_AUTH), the
    auth middleware, and the Azure deployment routes. A name this request already
    mapped is judged by the client's name behind it. Once the caller is known, as in
    resolve_model_for_request, a fallback whose target cannot be served either leaves
    the client's name, so the failure is reported against the name the client sent.
    (Before auth the choice is provisional, so the fallback is applied unchecked.)
    """
    if not name:
        return name
    source = alias_source_name(name)
    may_use, api = _may_use_for(auth), current_api_surface.get()
    if _evaluate(source, may_use, api, allow_prefix_strip).status == "ok":
        return apply_alias(source, kind="alias")
    if auth is not PRE_AUTH:
        # Peeked without apply_alias, which would record the client's name for the echo.
        target = model_alias_resolver.resolve(source, api, current_profile_id.get(), "fallback")
        if target != source and _evaluate(target, may_use, api, allow_prefix_strip).status != "ok":
            return source
    return apply_alias(source, kind="fallback")


def _raise_unavailable(request_obj, model: str, envelope_override: Optional[str]):
    envelope = envelope_for(_path_of(request_obj), envelope_override)
    if envelope == "anthropic":
        raise ModelUnavailable.anthropic(model)
    if envelope == "azure":
        raise ModelUnavailable.azure(model)
    raise ModelUnavailable.openai(model)


def _raise_access_denied(request_obj, model: str, envelope_override: Optional[str]):
    envelope = envelope_for(_path_of(request_obj), envelope_override)
    if envelope == "anthropic":
        raise ModelAccessDenied.anthropic(model)
    if envelope == "azure":
        raise ModelAccessDenied.azure(model)
    raise ModelAccessDenied.openai(model)


def _path_of(request_obj) -> str:
    try:
        return request_obj.url.path
    except Exception:
        return ""


async def resolve_model_for_request(
    request_obj,
    auth,
    model: Optional[str],
    *,
    envelope_override: Optional[str] = None,
    allow_prefix_strip: bool = True,
) -> Optional[str]:
    """Resolve a client-supplied model name to a canonical '{provider_key}/{name}' id.

    Judges the client's name first (an earlier layer may already have mapped it;
    apply_alias recognises its own output): available to the caller -> the regular
    mappings apply, otherwise the fallback mappings. This also covers paths the
    middleware skips, such as /v1/messages/count_tokens. Raises ModelUnavailable
    (404) when nothing serves the model, and ModelAccessDenied (403) when it exists
    but the caller may not use it -- so a user asking for 'gpt-5.4' gets the same
    answer they would get asking for 'azure:foundry/gpt-5.4'.
    """
    if not model:
        return model
    # Normally bound at auth already; repeated here for auth paths that bypass
    # _update_tracking_identity (e.g. overridden dependencies). Idempotent.
    current_profile_id.set(profile_id_for_auth(auth))
    source = alias_source_name(model)
    may_use = _may_use_for(auth)
    api = current_api_surface.get()

    ev = _evaluate(source, may_use, api, allow_prefix_strip)
    failed_name = source
    if ev.status == "ok":
        target = apply_alias(source, kind="alias")
        if target != source:
            # A regular mapping is the admin's choice for an available model; when
            # its target cannot be served, that is reported, not papered over.
            ev = _evaluate(target, may_use, api, allow_prefix_strip)
            failed_name = target
    else:
        target = apply_alias(source, kind="fallback")
        if target != source:
            fallback = _evaluate(target, may_use, api, allow_prefix_strip)
            if fallback.status == "ok":
                logger.info("Fallback mapping: %s -> %s (requested model %s)", source, target, ev.status)
                ev = fallback
            else:
                logger.warning(
                    "Fallback mapping %s -> %s not used: the target is %s too",
                    source, target, fallback.status,
                )

    if ev.status == "unavailable":
        _raise_unavailable(request_obj, failed_name, envelope_override)
    if ev.status == "denied":
        _raise_access_denied(request_obj, failed_name, envelope_override)

    chosen = _round_robin.pick(list(ev.candidates))

    # Echo the name the client sent. Only set when unset, so an admin alias that
    # already recorded the client's original name keeps priority, and only when it
    # differs from what is served: a canonical id passes through untouched.
    if original_model_name.get() is None and chosen != source:
        original_model_name.set(source)

    await _record_resolved(request_obj, chosen)
    return chosen


async def canonicalize_provider_prefix(request_obj, model: Optional[str]) -> Optional[str]:
    """'azure/gpt-4' -> 'azure:primary/gpt-4' when the family names a single instance.

    A provider family that names one instance is that instance's model: canonicalise
    it, so group limits, access rules, the enable flags and usage all see the
    instance's id rather than a name none of them know (which would land in the
    Default group and skip the global gate). An exact provider_key, an ambiguous
    family, or no provider at all passes through for dispatch to accept or reject.
    Callers that build the id from a URL (the Azure deployment routes) call this
    directly; resolve_model_for_request canonicalises through _evaluate.
    """
    if not model:
        return model
    canonical = _canonical_for_prefix(model)
    if canonical == model:
        return model
    if original_model_name.get() is None:
        original_model_name.set(model)
    await _record_resolved(request_obj, canonical)
    return canonical


async def _record_resolved(request_obj, chosen: str) -> None:
    """Point request.state.model and the in-flight tracker entry at `chosen`, if they differ."""
    state = getattr(request_obj, "state", None)
    if getattr(state, "model", None) == chosen:
        return
    try:
        request_obj.state.model = chosen
    except Exception:
        logger.debug("Unable to record resolved model on request state", exc_info=True)
    await _retag_tracker(request_obj, chosen)


async def _retag_tracker(request_obj, chosen: str) -> None:
    """Point the in-flight tracker entry at the canonical id.

    end_request builds the usage key from entry.model, so without this the same
    logical model splits across a bare row and a prefixed row. A no-op on
    untracked paths (count_tokens), and never
    allowed to break the request.
    """
    request_id = getattr(getattr(request_obj, "state", None), "tracking_request_id", None)
    if not request_id:
        return
    try:
        from app.request_tracker import request_tracker
        await request_tracker.update_model(request_id, chosen)
    except Exception:
        logger.debug("Unable to retag request tracker model", exc_info=True)

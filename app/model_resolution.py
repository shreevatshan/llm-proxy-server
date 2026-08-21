"""Prefix-less model resolution, called from route handlers after authentication.

Models are addressed canonically as ``{provider_key}/{bare_name}`` (e.g.
``azure:foundry/gpt-5.4``). This module lets a client name a model without
knowing the proxy's provider topology: ``gpt-5.4`` resolves to whichever
configured provider serves it.

Resolution order for a name and the request's API surface:

    1. falsy                                      -> unchanged
    2. an exact canonical id                      -> unchanged
    3. '{head}/{rest}' where head names a live provider (or an ambiguous family
       of them) -> unchanged (the caller asked for that provider; honour it even
       when the model is absent from the cache, as Azure deployments frequently
       are, and let _get_provider ask for disambiguation when it has to)
    4. the whole string as a bare name            -> round-robin over candidates
    5. `rest` as a bare name (strip head exactly once) -> round-robin
    6. otherwise                                  -> ModelUnavailable (404)

Step 4 precedes step 5 because a bare name may itself contain '/': an
OpenAI-compatible backend reporting ``meta-llama/Llama-3.1-8B`` produces the id
``lmstudio:box/meta-llama/Llama-3.1-8B``.

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
from typing import Dict, List, Optional, Tuple

from app.api_envelope import envelope_for
from app.auth.admin import AdminUser
from app.model_access_dep import ModelAccessDenied
from app.model_alias import current_api_surface, original_model_name
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
    authentication (app/auth/middleware.py uses it to decide whether the overall
    rate limit applies). Returns [] when `name` is falsy, is already a canonical
    id, names an available provider, or matches nothing. Consumes no rotation
    state.
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


def _filter_for_caller(pool: List[str], auth) -> List[str]:
    """Narrow `pool` to what this caller may use.

    AdminUser and identity-less principals bypass the per-user check, mirroring
    enforce_model_access (app/model_access_dep.py). The global gate is NOT
    bypassed -- it was already applied in _filter_pool, and a globally disabled
    model is invisible to admins today.
    """
    if isinstance(auth, AdminUser):
        return pool
    user_id = getattr(auth, "user_id", None) or getattr(auth, "id", None)
    if user_id is None:
        return pool
    cache = provider_manager.model_cache
    return [model_id for model_id in pool if cache.is_model_allowed_for_user(user_id, model_id)]


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

    Assumes admin aliases have already been applied (middleware, or apply_alias in
    the handler). Raises ModelUnavailable (404) when nothing serves the model, and
    ModelAccessDenied (403) when candidates exist but the caller may use none of
    them -- so a user asking for 'gpt-5.4' gets the same answer they would get
    asking for 'azure:foundry/gpt-5.4'.
    """
    if not model:
        return model

    cache = provider_manager.model_cache
    if cache.has_model_id(model):
        return model

    head, rest = model.split('/', 1) if '/' in model else ('', '')
    if rest and provider_manager.has_provider_prefix(head):
        return model

    api = current_api_surface.get()
    pool = _filter_pool(cache.bare_model_candidates(model), api)
    if not pool and allow_prefix_strip and rest:
        pool = _filter_pool(cache.bare_model_candidates(rest), api)
    if not pool:
        _raise_unavailable(request_obj, model, envelope_override)

    allowed = _filter_for_caller(pool, auth)
    if not allowed:
        _raise_access_denied(request_obj, model, envelope_override)

    chosen = _round_robin.pick(allowed)

    # Echo the name the client sent. Only set when unset, so an admin alias that
    # already recorded the client's original name keeps priority.
    if original_model_name.get() is None:
        original_model_name.set(model)

    try:
        request_obj.state.model = chosen
    except Exception:
        logger.debug("Unable to record resolved model on request state", exc_info=True)

    await _retag_tracker(request_obj, chosen)
    return chosen


async def _retag_tracker(request_obj, chosen: str) -> None:
    """Point the in-flight tracker entry at the canonical id.

    end_request builds the usage key from entry.model, so without this the same
    logical model splits across a bare row and a prefixed row. A no-op on
    untracked paths (count_tokens, everything under /openai/v1/*), and never
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

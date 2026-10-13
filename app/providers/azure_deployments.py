"""Helpers for storing and reading Azure deployment configuration."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

AzureDeploymentGroups = Dict[str, List[str]]


def _normalize_list(values: Optional[Iterable[Any]]) -> List[str]:
    if values is None:
        return []

    normalized: List[str] = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


_DEPLOYMENT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')


def validate_deployment_names(names: Optional[Iterable[str]]) -> None:
    """Raise ValueError if any deployment name has invalid characters."""
    if names is None:
        return
    for name in names:
        text = str(name).strip()
        if text and not _DEPLOYMENT_NAME_PATTERN.match(text):
            raise ValueError(
                f"Invalid deployment name '{text}'. "
                "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
            )


# Upstream APIs a deployment can be served over.
API_CHAT = "chat"
API_RESPONSES = "responses"
API_ANTHROPIC = "anthropic"
API_COMPLETIONS = "completions"
API_EMBEDDINGS = "embeddings"
API_IMAGES = "images"
API_SPEECH = "speech"
API_TRANSCRIPTION = "transcription"
DEPLOYMENT_APIS = (
    API_CHAT, API_RESPONSES, API_ANTHROPIC, API_COMPLETIONS,
    API_EMBEDDINGS, API_IMAGES, API_SPEECH, API_TRANSCRIPTION,
)
# APIs that make a deployment a conversational model (offered on /v1/messages).
CONVERSATIONAL_APIS = frozenset({API_CHAT, API_RESPONSES, API_ANTHROPIC})

# Rows carried over from the per-bucket lists that predate deployment groups.
# They classify their deployments (OpenAI, OpenAI over Responses, Anthropic)
# but select no APIs, so they never restrict which endpoints a deployment
# serves; ``apis`` is always empty on such a row.
LEGACY_OPENAI = "openai"
LEGACY_RESPONSES = "responses"
LEGACY_ANTHROPIC = "anthropic"
LEGACY_MODES = (LEGACY_OPENAI, LEGACY_RESPONSES, LEGACY_ANTHROPIC)


def _empty_groups() -> AzureDeploymentGroups:
    return {"openai": [], "anthropic": [], "responses": [], "groups": []}


def normalize_deployment_groups(raw: Any, *, strict: bool = False) -> List[Dict[str, Any]]:
    """Normalize ``[{"deployments": [...], "apis": [...]}, ...]`` rows.

    A row either selects APIs or carries a ``legacy_mode`` (see LEGACY_MODES)
    with no APIs. Rows with no deployments, or with neither, are dropped; with
    ``strict`` an invalid name, unknown API or mode, a row that is neither or
    both, or a deployment listed in both a legacy and an explicit row raises
    ValueError instead.
    """
    rows: List[Dict[str, Any]] = []
    for row in raw or ():
        if hasattr(row, "model_dump"):
            row = row.model_dump()
        if not isinstance(row, dict):
            if strict:
                raise ValueError("Each deployment group must be an object with 'deployments' and 'apis'")
            continue
        deployments = _normalize_list(row.get("deployments"))
        apis = _normalize_list(row.get("apis"))
        legacy_mode = row.get("legacy_mode") or None
        if strict:
            validate_deployment_names(deployments)
            unknown = [a for a in apis if a not in DEPLOYMENT_APIS]
            if unknown:
                raise ValueError(
                    f"Unknown API(s) {', '.join(unknown)}; must be one of {', '.join(DEPLOYMENT_APIS)}"
                )
            if legacy_mode is not None and legacy_mode not in LEGACY_MODES:
                raise ValueError(f"Unknown legacy_mode '{legacy_mode}'; must be one of {', '.join(LEGACY_MODES)}")
            if legacy_mode is not None and apis:
                raise ValueError(
                    f"A legacy deployment row cannot also select APIs: {', '.join(deployments)}"
                )
            if deployments and not apis and legacy_mode is None:
                raise ValueError(f"Select at least one API for deployments: {', '.join(deployments)}")
        apis = [a for a in DEPLOYMENT_APIS if a in apis]
        if not deployments:
            continue
        if apis:
            # Selected APIs are explicit even if a stale marker came along.
            rows.append({"deployments": deployments, "apis": apis})
        elif legacy_mode in LEGACY_MODES:
            rows.append({"deployments": deployments, "apis": [], "legacy_mode": legacy_mode})
    if strict:
        legacy = {n for r in rows if r.get("legacy_mode") for n in r["deployments"]}
        explicit = {n for r in rows if not r.get("legacy_mode") for n in r["deployments"]}
        both = [n for n in _normalize_list(n for r in rows for n in r["deployments"]) if n in legacy and n in explicit]
        if both:
            raise ValueError(
                f"Deployments listed in both a legacy row and a row with selected APIs: {', '.join(both)}. "
                "Select APIs for them in one row only."
            )
    return rows


def _lists_from_groups(rows: List[Dict[str, Any]]) -> AzureDeploymentGroups:
    """Derive the per-API deployment lists the rest of the code reads.

    ``openai``: deployments served over any OpenAI API.
    ``anthropic``: deployments served over the native Anthropic API.
    ``responses``: OpenAI deployments with Responses but not Chat Completions;
    Anthropic requests for these are adapted onto the Responses API.

    Legacy rows contribute their classification: an ``openai`` or
    ``responses`` row lists OpenAI deployments (the latter also marks them
    Responses-only), an ``anthropic`` row lists Anthropic deployments.
    """
    apis_by_name: Dict[str, set] = {}
    modes_by_name: Dict[str, set] = {}
    order: List[str] = []
    for row in rows:
        for name in row["deployments"]:
            if name not in apis_by_name:
                apis_by_name[name] = set()
                modes_by_name[name] = set()
                order.append(name)
            apis_by_name[name].update(row["apis"])
            if row.get("legacy_mode"):
                modes_by_name[name].add(row["legacy_mode"])

    def is_openai(n: str) -> bool:
        return bool(apis_by_name[n] - {API_ANTHROPIC}) or bool(modes_by_name[n] & {LEGACY_OPENAI, LEGACY_RESPONSES})

    def is_responses_only(n: str) -> bool:
        apis, modes = apis_by_name[n], modes_by_name[n]
        if API_CHAT in apis or LEGACY_OPENAI in modes:
            return False
        return API_RESPONSES in apis or LEGACY_RESPONSES in modes

    return {
        "openai": [n for n in order if is_openai(n)],
        "anthropic": [n for n in order if API_ANTHROPIC in apis_by_name[n] or LEGACY_ANTHROPIC in modes_by_name[n]],
        "responses": [n for n in order if is_responses_only(n)],
        "groups": rows,
    }


def deployment_apis(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Map each deployment listed in a row with selected APIs to those APIs.

    Legacy rows select nothing, so their deployments are left out entirely
    (an empty entry would forbid every API) and keep the classification-based
    routing they had before deployment groups.
    """
    result: Dict[str, List[str]] = {}
    for row in rows:
        if row.get("legacy_mode") or not row.get("apis"):
            continue
        for name in row["deployments"]:
            apis = result.setdefault(name, [])
            apis.extend(a for a in row["apis"] if a not in apis)
    return result


def _groups_from_lists(openai: List[str], anthropic: List[str], responses: List[str]) -> List[Dict[str, Any]]:
    """Legacy rows for configs stored before deployment groups existed."""
    rows: List[Dict[str, Any]] = []
    chat = [n for n in openai if n not in responses]
    if chat:
        rows.append({"deployments": chat, "apis": [], "legacy_mode": LEGACY_OPENAI})
    only_responses = [n for n in openai if n in responses]
    if only_responses:
        rows.append({"deployments": only_responses, "apis": [], "legacy_mode": LEGACY_RESPONSES})
    if anthropic:
        rows.append({"deployments": list(anthropic), "apis": [], "legacy_mode": LEGACY_ANTHROPIC})
    return rows


def normalize_azure_deployments(raw: Any) -> AzureDeploymentGroups:
    """Normalize stored Azure deployment config into grouped lists.

    Legacy list payloads map to ``openai`` deployments. Dict payloads carry
    ``groups`` rows (deployments + the APIs they support), from which the
    ``openai`` / ``anthropic`` / ``responses`` lists are derived; older dicts
    carry only ``openai`` and ``anthropic`` arrays. ``responses`` marks
    deployments rather than adding them, so it is never part of the merged
    deployment list.
    """
    if raw is None:
        return _empty_groups()

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return _empty_groups()

    if isinstance(raw, list):
        openai = _normalize_list(raw)
        return {"openai": openai, "anthropic": [], "responses": [], "groups": _groups_from_lists(openai, [], [])}

    if isinstance(raw, dict):
        if "groups" in raw:
            return _lists_from_groups(normalize_deployment_groups(raw.get("groups")))
        openai_values = raw.get("openai")
        if openai_values is None and "deployments" in raw:
            openai_values = raw.get("deployments")
        openai = _normalize_list(openai_values)
        anthropic = _normalize_list(raw.get("anthropic"))
        responses = [n for n in _normalize_list(raw.get("responses")) if n in openai]
        return {
            "openai": openai,
            "anthropic": anthropic,
            "responses": responses,
            "groups": _groups_from_lists(openai, anthropic, responses),
        }

    return _empty_groups()


def merge_azure_deployments(groups: AzureDeploymentGroups, *, include_anthropic: bool = True) -> List[str]:
    """Return the combined deployment names in stable order."""
    merged = list(groups.get("openai", []))
    if include_anthropic:
        for name in groups.get("anthropic", []):
            if name not in merged:
                merged.append(name)
    return merged


def build_azure_config_fields(creds) -> Dict[str, Any]:
    """Build the Azure-specific provider config dict from DB credentials.

    This is the single source of truth for converting stored Azure credentials
    into the config dict consumed by AzureProvider.__init__.
    """
    deployment_groups = normalize_azure_deployments(getattr(creds, 'deployments_json', None))
    azure_backend = getattr(creds, 'azure_backend', None) or 'openai'

    dynamic_discovery = getattr(creds, 'dynamic_discovery', None)
    if dynamic_discovery is None:
        dynamic_discovery = not bool(merge_azure_deployments(deployment_groups))

    return {
        'endpoint': creds.endpoint,
        'api_key': creds.api_key,
        'discovery_api_version': getattr(creds, 'discovery_api_version', None),
        'azure_backend': azure_backend,
        'deployments': merge_azure_deployments(
            deployment_groups,
            include_anthropic=azure_backend == 'foundry',
        ),
        'openai_deployments': deployment_groups.get('openai', []),
        'anthropic_deployments': deployment_groups.get('anthropic', []),
        'responses_deployments': deployment_groups.get('responses', []),
        'deployment_apis': deployment_apis(deployment_groups.get('groups', [])),
        'dynamic_discovery': dynamic_discovery,
    }


def serialize_azure_deployments(
    *,
    deployments: Optional[List[str]] = None,
    openai_deployments: Optional[List[str]] = None,
    anthropic_deployments: Optional[List[str]] = None,
    deployment_groups: Optional[List[Any]] = None,
) -> Optional[str]:
    """Serialize Azure deployment config to JSON for storage.

    ``deployment_groups`` takes precedence; the derived ``openai``,
    ``anthropic`` and ``responses`` lists are stored alongside it for older
    readers.
    """
    if deployment_groups is not None:
        derived = _lists_from_groups(normalize_deployment_groups(deployment_groups))
        if not derived["groups"]:
            return None
        stored: Dict[str, Any] = {"openai": derived["openai"], "anthropic": derived["anthropic"]}
        if derived["responses"]:
            stored["responses"] = derived["responses"]
        stored["groups"] = derived["groups"]
        return json.dumps(stored)

    if openai_deployments is None and anthropic_deployments is None:
        if deployments is None:
            return None
        groups = normalize_azure_deployments(deployments)
    else:
        groups = {
            "openai": _normalize_list(openai_deployments if openai_deployments is not None else deployments),
            "anthropic": _normalize_list(anthropic_deployments),
        }

    if not groups["openai"] and not groups["anthropic"]:
        return None

    return json.dumps({"openai": groups["openai"], "anthropic": groups["anthropic"]})

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


def normalize_azure_deployments(raw: Any) -> AzureDeploymentGroups:
    """Normalize stored Azure deployment config into grouped lists.

    Legacy list payloads map to ``openai`` deployments. New payloads may be a
    dict containing ``openai`` and ``anthropic`` arrays.
    """
    if raw is None:
        return {"openai": [], "anthropic": []}

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return {"openai": [], "anthropic": []}

    if isinstance(raw, list):
        return {
            "openai": _normalize_list(raw),
            "anthropic": [],
        }

    if isinstance(raw, dict):
        openai_values = raw.get("openai")
        if openai_values is None and "deployments" in raw:
            openai_values = raw.get("deployments")
        anthropic_values = raw.get("anthropic")
        return {
            "openai": _normalize_list(openai_values),
            "anthropic": _normalize_list(anthropic_values),
        }

    return {"openai": [], "anthropic": []}


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
        'dynamic_discovery': dynamic_discovery,
    }


def serialize_azure_deployments(
    *,
    deployments: Optional[List[str]] = None,
    openai_deployments: Optional[List[str]] = None,
    anthropic_deployments: Optional[List[str]] = None,
) -> Optional[str]:
    """Serialize Azure deployment config to JSON for storage."""
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

    return json.dumps(groups)

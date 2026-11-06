"""Pure helpers shared by the admin and the user (BYOK) provider routers.

No side effects here -- no database writes, no auto-sync, no registry refresh.
Those stay in the route layer, where tests patch them (e.g.
``app.routes.admin.auto_sync_on_provider_change``).
"""

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel

from app.auth.models import ProviderCredentialsResponse, _platformai_model_names
from app.providers.azure_deployments import merge_azure_deployments, normalize_azure_deployments

# Placeholder returned instead of stored secrets (see mask_secret).
SECRET_MASK = "********"

# Secret fields that mask_secret masks in read/detail responses. A client
# echoing the mask back on update means "leave unchanged", so these values
# must never be persisted (see strip_masked_secrets).
MASKED_SECRET_FIELDS = ("api_key", "secret_access_key", "oauth_client_secret", "oauth_refresh_token")


def mask_secret(value: Optional[str]) -> Optional[str]:
    """Mask a stored secret for read/detail responses (write-only semantics).

    Returns a fixed placeholder when a value is set (so the UI knows a secret
    exists) and None when it is absent. The real value is never returned.
    """
    return SECRET_MASK if value else None


def strip_masked_secrets(update_data: dict) -> dict:
    """Drop secret fields whose submitted value is the mask placeholder.

    Detail responses return secrets as ``SECRET_MASK``; if a client round-trips
    that literal on an update, persisting it would corrupt the stored credential.
    Treat it as "unchanged" and strip it before writing.
    """
    for field in MASKED_SECRET_FIELDS:
        if update_data.get(field) == SECRET_MASK:
            update_data.pop(field)
    return update_data


def platformai_response_fields(credentials) -> dict:
    """PlatformAI credential fields for a provider response, secrets masked."""
    return {
        "oauth_client_id": getattr(credentials, "oauth_client_id", None),
        "oauth_client_secret": mask_secret(getattr(credentials, "oauth_client_secret", None)),
        "oauth_refresh_token": mask_secret(getattr(credentials, "oauth_refresh_token", None)),
        "oauth_accounts_url": getattr(credentials, "oauth_accounts_url", None),
        "pai_token_id": getattr(credentials, "pai_token_id", None),
    }


def provider_update_data(provider_data: BaseModel, *, exclude: tuple = ()) -> dict:
    """Column values an update/upsert request writes.

    Only fields the client set (or validators derived) that are not None;
    masked secrets mean "unchanged" and are dropped; ``supported_apis`` is
    stored as JSON text.
    """
    update_data = {
        field: value
        for field, value in provider_data.model_dump(exclude_unset=True).items()
        if value is not None and field not in exclude
    }
    strip_masked_secrets(update_data)
    if isinstance(update_data.get('supported_apis'), list):
        update_data['supported_apis'] = json.dumps(update_data['supported_apis'])
    return update_data


def parse_azure_deployment_fields(raw_deployments):
    groups = normalize_azure_deployments(raw_deployments)
    return groups, merge_azure_deployments(groups)


def parse_supported_apis(raw: Optional[str]) -> Optional[list]:
    """Stored supported_apis JSON text -> list (['openai'] when unreadable)."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ["openai"]


def infer_azure_supported_apis(provider_data) -> None:
    """Azure: derive supported_apis from the backend and deployments when unset."""
    if provider_data.provider_type == "azure" and not provider_data.supported_apis:
        inferred_apis = ["openai"]
        if (
            provider_data.azure_backend == "foundry"
            and provider_data.anthropic_deployments
            and len(provider_data.anthropic_deployments) > 0
        ):
            inferred_apis.append("anthropic")
        provider_data.supported_apis = inferred_apis


def check_platformai_create(provider_data) -> None:
    """PlatformAI create: an api_key, or OAuth client id + refresh token (ValueError).

    Also defaults supported_apis. Required only on create: an upsert or update of an
    existing provider keeps the stored values.
    """
    if provider_data.provider_type != "platformai":
        return

    def _given(name):
        value = (getattr(provider_data, name) or "").strip()
        return bool(value) and value != SECRET_MASK

    if not _given("api_key"):
        missing = [n for n in ("oauth_client_id", "oauth_refresh_token") if not _given(n)]
        if missing:
            raise ValueError(
                "A PlatformAI provider needs an api_key, or oauth_client_id and "
                f"oauth_refresh_token (missing: {' and '.join(missing)})."
            )
    if not provider_data.supported_apis:
        provider_data.supported_apis = ["openai", "anthropic"]


def platformai_update_deployments(update_data: dict, stored) -> None:
    """Normalise a PlatformAI model list on update (ValueError when invalid)."""
    if update_data.get("deployments") is not None and stored is not None and stored.provider_type == "platformai":
        update_data["deployments"] = _platformai_model_names(update_data["deployments"])


def credentials_response_fields(credentials, dynamic_discovery: Any = ...) -> Dict[str, Any]:
    """The ProviderCredentialsResponse fields for a credentials row, secrets masked.

    ``dynamic_discovery`` overrides the stored value when given (the detail view
    infers it for legacy Azure rows).
    """
    deployment_groups, deployments = parse_azure_deployment_fields(credentials.deployments_json)
    return dict(
        id=credentials.id,
        provider_key=credentials.provider_key,
        provider_type=credentials.provider_type,
        instance_name=credentials.instance_name,
        provider_name=credentials.provider_name,
        enabled=credentials.enabled,
        endpoint=credentials.endpoint,
        api_key=mask_secret(credentials.api_key),
        discovery_api_version=credentials.discovery_api_version,
        azure_backend=credentials.azure_backend or ("openai" if credentials.provider_type == "azure" else None),
        region=credentials.region,
        access_key_id=credentials.access_key_id,
        secret_access_key=mask_secret(credentials.secret_access_key),
        base_url=credentials.base_url,
        deployments=deployments,
        openai_deployments=deployment_groups.get("openai", []),
        anthropic_deployments=deployment_groups.get("anthropic", []),
        deployment_groups=deployment_groups.get("groups", []),
        dynamic_discovery=credentials.dynamic_discovery if dynamic_discovery is ... else dynamic_discovery,
        supported_apis=parse_supported_apis(credentials.supported_apis),
        anthropic_base_url=credentials.anthropic_base_url,
        anthropic_auth_style=credentials.anthropic_auth_style,
        **platformai_response_fields(credentials),
        created_at=credentials.created_at,
        updated_at=credentials.updated_at,
    )


def build_credentials_response(credentials, dynamic_discovery: Any = ...) -> ProviderCredentialsResponse:
    return ProviderCredentialsResponse(**credentials_response_fields(credentials, dynamic_discovery))

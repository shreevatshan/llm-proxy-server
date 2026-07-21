import os
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, validator, model_validator

# Single source of truth for the transformation config models. Importing them
# here (rather than redefining) ensures initialize_transformation_manager()
# receives the exact type its TransformationManager is annotated with.
from app.transformation import ProcessorConfig, TransformationConfig


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    domain: str = "localhost"  # Public-facing domain used for redirects
    openai_port: int = 11440  # OpenAI API server port
    anthropic_port: int = 2027  # Anthropic API server port
    azure_openai_port: int = 11439  # Azure OpenAI API server port
    management_port: int = 8765  # Management (admin + user login) server port
    timezone: str = "UTC"  # IANA timezone name, e.g. "Asia/Kolkata"
    # Credentialed CORS allowlist for the cookie-authenticated management app.
    # Populated from LLMPROXY_CORS_ALLOW_ORIGINS (comma-separated); defaults to the
    # management origin derived from domain/management_port plus localhost.
    cors_allow_origins: List[str] = []


class DebugConfig(BaseModel):
    """Debug configuration."""
    pass  # Debug configuration settings (currently none)


class ModelConfig(BaseModel):
    pass  # Model configuration settings (currently none)


class AdminConfig(BaseModel):
    enabled: bool = False
    username: str = "admin"
    email: str = "admin@localhost"
    password: str = ""  # No default password shipped; must be set via env when enabled


class OAuthConfig(BaseModel):
    zoho_enabled: bool = False
    zoho_client_id: Optional[str] = None
    zoho_client_secret: Optional[str] = None
    zoho_redirect_uri: Optional[str] = None


class WebhookConfig(BaseModel):
    notification_webhook_url: Optional[str] = None


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    admin: AdminConfig = AdminConfig()
    oauth: OAuthConfig = OAuthConfig()
    webhook: WebhookConfig = WebhookConfig()
    transformation: TransformationConfig = TransformationConfig()
    debug: DebugConfig = DebugConfig()


def load_config_from_env() -> Config:
    """Load configuration from environment variables with defaults."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    tz_name = os.getenv("TIMEZONE", "UTC")
    try:
        ZoneInfo(tz_name)  # validate early — fail fast on bad names
    except (ZoneInfoNotFoundError, KeyError) as e:
        raise ValueError(f"Invalid TIMEZONE '{tz_name}': {e}") from e

    # Server configuration from environment
    server_config = ServerConfig(
        host=os.getenv("LLMPROXY_HOST", "0.0.0.0"),
        domain=os.getenv("LLMPROXY_DOMAIN", "localhost"),
        openai_port=int(os.getenv("OPENAI_SERVER_PORT", "11440")),
        anthropic_port=int(os.getenv("ANTHROPIC_SERVER_PORT", "2027")),
        azure_openai_port=int(os.getenv("AZURE_OPENAI_SERVER_PORT", "11439")),
        management_port=int(os.getenv("MANAGEMENT_SERVER_PORT", "8765")),
        timezone=tz_name,
    )

    # Credentialed-CORS allowlist for the management app. Comma-separated origins
    # in LLMPROXY_CORS_ALLOW_ORIGINS override the default, which is the management
    # origin (from LLMPROXY_DOMAIN + management port) plus its localhost variant.
    cors_env = os.getenv("LLMPROXY_CORS_ALLOW_ORIGINS")
    if cors_env:
        server_config.cors_allow_origins = [
            o.strip() for o in cors_env.split(",") if o.strip()
        ]
    else:
        default_origins = [
            f"http://{server_config.domain}:{server_config.management_port}"
        ]
        localhost_origin = f"http://localhost:{server_config.management_port}"
        if localhost_origin not in default_origins:
            default_origins.append(localhost_origin)
        server_config.cors_allow_origins = default_origins

    # Model configuration from environment
    model_config = ModelConfig()
    
    # Admin configuration from environment.
    # Default OFF (matches the AdminConfig model default) so a no-env deployment
    # does not expose an admin panel on all interfaces. When explicitly enabled,
    # a secure LLMPROXY_ADMIN_PASSWORD MUST be provided — we refuse to start with
    # an empty password, the historical "admin123" default, or the .env.example
    # placeholder.
    _placeholder_admin_passwords = {"admin123", "change-this-strong-password"}
    admin_enabled = os.getenv("LLMPROXY_ADMIN_ENABLED", "false").lower() == "true"
    admin_password = os.getenv("LLMPROXY_ADMIN_PASSWORD", "")
    if admin_enabled and (not admin_password or admin_password in _placeholder_admin_passwords):
        raise ValueError(
            "LLMPROXY_ADMIN_ENABLED is true but LLMPROXY_ADMIN_PASSWORD is unset or "
            "set to an insecure default/placeholder. Set LLMPROXY_ADMIN_PASSWORD to "
            "a strong, non-default value to enable the admin account."
        )
    admin_config = AdminConfig(
        enabled=admin_enabled,
        username=os.getenv("LLMPROXY_ADMIN_USERNAME", "admin"),
        email=os.getenv("LLMPROXY_ADMIN_EMAIL", "admin@localhost"),
        password=admin_password,
    )
    
    # OAuth configuration from environment
    oauth_config = OAuthConfig(
        zoho_enabled=bool(os.getenv("ZOHO_CLIENT_ID") and os.getenv("ZOHO_CLIENT_SECRET")),
        zoho_client_id=os.getenv("ZOHO_CLIENT_ID"),
        zoho_client_secret=os.getenv("ZOHO_CLIENT_SECRET"),
        zoho_redirect_uri=os.getenv(
            "ZOHO_REDIRECT_URI",
            f"http://{server_config.domain}:{server_config.management_port}/auth/zoho/callback"
        )
    )

    # Webhook configuration from environment
    webhook_config = WebhookConfig(
        notification_webhook_url=os.getenv("NOTIFICATION_WEBHOOK_URL")
    )

    # Transformation config (keep defaults for now)
    transformation_config = TransformationConfig()
    
    # Debug configuration from environment
    debug_config = DebugConfig()
    
    return Config(
        server=server_config,
        model=model_config,
        admin=admin_config,
        oauth=oauth_config,
        webhook=webhook_config,
        transformation=transformation_config,
        debug=debug_config
    )


# Global config instance - loads from environment variables only
config = load_config_from_env()

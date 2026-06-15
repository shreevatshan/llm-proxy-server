import os
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, validator, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    domain: str = "localhost"  # Public-facing domain used for redirects
    openai_port: int = 11440  # OpenAI API server port
    anthropic_port: int = 2027  # Anthropic API server port
    azure_openai_port: int = 11439  # Azure OpenAI API server port
    management_port: int = 8765  # Management (admin + user login) server port
    timezone: str = "UTC"  # IANA timezone name, e.g. "Asia/Kolkata"


class DebugConfig(BaseModel):
    """Debug configuration."""
    pass  # Debug configuration settings (currently none)


class ModelConfig(BaseModel):
    pass  # Model configuration settings (currently none)


class AdminConfig(BaseModel):
    enabled: bool = False
    username: str = "admin"
    email: str = "admin@localhost"
    password: str = "admin123"


class OAuthConfig(BaseModel):
    zoho_enabled: bool = False
    zoho_client_id: Optional[str] = None
    zoho_client_secret: Optional[str] = None
    zoho_redirect_uri: Optional[str] = None


class WebhookConfig(BaseModel):
    notification_webhook_url: Optional[str] = None


class ProcessorConfig(BaseModel):
    name: str
    enabled: bool = True
    priority: int = 100
    config: Dict[str, Any] = {}


class TransformationConfig(BaseModel):
    enabled: bool = True
    request_processors: List[ProcessorConfig] = []
    response_processors: List[ProcessorConfig] = []


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
    
    # Model configuration from environment  
    model_config = ModelConfig()
    
    # Admin configuration from environment
    admin_config = AdminConfig(
        enabled=os.getenv("LLMPROXY_ADMIN_ENABLED", "true").lower() == "true",
        username=os.getenv("LLMPROXY_ADMIN_USERNAME", "admin"),
        email=os.getenv("LLMPROXY_ADMIN_EMAIL", "admin@localhost"),
        password=os.getenv("LLMPROXY_ADMIN_PASSWORD", "admin123")
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

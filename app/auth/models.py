"""Database models for authentication."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, Date, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, date
import re
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List

VALID_AZURE_BACKENDS = {"openai", "foundry"}
DEPLOYMENT_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')

Base = declarative_base()


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Nullable for OAuth users
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_pending_approval = Column(Boolean, default=False)

    # OAuth fields
    oauth_provider = Column(String(50), nullable=True)  # e.g., "zoho", "google"
    oauth_sub = Column(String(255), nullable=True)      # OAuth subject identifier
    oauth_data = Column(Text, nullable=True)            # JSON data from OAuth provider
    
    # Relationship to API keys
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Relationship to OAuth accounts
    oauth_accounts = relationship("OAuthUser", back_populates="user", cascade="all, delete-orphan")

    # Relationship to rate limit override (one-to-one, optional)
    rate_limit = relationship("UserRateLimit", uselist=False, cascade="all, delete-orphan")


class OAuthUser(Base):
    """OAuth user information for external authentication providers."""
    __tablename__ = "oauth_users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(50), nullable=False)  # e.g., "zoho", "google"
    provider_user_id = Column(String(255), nullable=False)  # OAuth sub
    email = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    picture = Column(String(500), nullable=True)
    raw_data = Column(Text, nullable=True)  # JSON data from OAuth provider
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user - changed from backref to back_populates for consistency
    user = relationship("User", back_populates="oauth_accounts")
    
    # Unique constraint on provider + provider_user_id
    __table_args__ = (
        Column('provider_user_id_unique', String(255), unique=True),
    )


class APIKey(Base):
    """API Key model for authentication."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationship to user
    user = relationship("User", back_populates="api_keys")


class ModelConfiguration(Base):
    """Model configuration model for enable/disable individual models."""
    __tablename__ = "model_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(200), unique=True, index=True, nullable=False)  # e.g., "ollama:msi-ai-test-01/llama2"
    provider_key = Column(String(100), ForeignKey("provider_credentials.provider_key"), nullable=False)
    model_name = Column(String(100), nullable=False)  # e.g., "llama2"
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to provider
    provider = relationship("ProviderCredentials", back_populates="models")


class ResponseProviderMapping(Base):
    """Maps Responses API response IDs to the provider that created them.
    Used to route retrieve/delete/cancel/input_items requests to the correct upstream."""
    __tablename__ = "response_provider_mappings"

    id = Column(Integer, primary_key=True, index=True)
    response_id = Column(String(200), unique=True, index=True, nullable=False)
    provider_key = Column(String(100), nullable=False)  # e.g., "openai:primary"
    model_name = Column(String(200), nullable=True)      # Original model string from request
    created_at = Column(DateTime, default=datetime.utcnow)


class UserRateLimit(Base):
    """Per-user rate limit overrides. Absent row = inherit global defaults."""
    __tablename__ = "user_rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    rpm_limit = Column(Integer, nullable=True)  # null = inherit global default
    rpd_limit = Column(Integer, nullable=True)  # null = inherit global default
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class GlobalRateLimit(Base):
    """Global rate limit defaults. Single row (id=1). null = unlimited."""
    __tablename__ = "global_rate_limits"

    id = Column(Integer, primary_key=True)  # always 1
    rpm_default = Column(Integer, nullable=True)
    rpd_default = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)


class RequestUsage(Base):
    """Per-day request usage counters keyed by (date, user_identity, model, server)."""
    __tablename__ = "request_usage"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    user_identity = Column(String(200), index=True, nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('date', 'user_identity', 'model', 'server', name='uq_usage_day'),
    )


class RequestUsageHourly(Base):
    """Per-hour request usage counters; retained for ~48 hours to serve the rolling-24h window."""
    __tablename__ = "request_usage_hourly"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    hour = Column(Integer, nullable=False)
    user_identity = Column(String(200), index=True, nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('date', 'hour', 'user_identity', 'model', 'server', name='uq_usage_hour'),
        Index('ix_usage_hourly_date_hour', 'date', 'hour'),
    )


class RequestUsageMonthly(Base):
    """Per-month rolled-up request usage; retained forever."""
    __tablename__ = "request_usage_monthly"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    user_identity = Column(String(200), index=True, nullable=False)
    user_type = Column(String(20), nullable=False)
    model = Column(String(200), index=True, nullable=False)
    server = Column(String(20), nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint('year', 'month', 'user_identity', 'model', 'server', name='uq_usage_month'),
        Index('ix_usage_monthly_year_month', 'year', 'month'),
    )


class ModelGroup(Base):
    """Named group of models sharing a rate-limit bucket."""
    __tablename__ = "model_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), unique=True, nullable=False)
    description = Column(String(256), nullable=True)
    rpm_default = Column(Integer, nullable=True)  # null = unlimited
    rpd_default = Column(Integer, nullable=True)  # null = unlimited
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    members = relationship("ModelGroupMember", back_populates="group", cascade="all, delete-orphan")
    user_limits = relationship("UserModelGroupRateLimit", back_populates="group", cascade="all, delete-orphan")


class ModelGroupMember(Base):
    """Maps a model_id to exactly one ModelGroup (unique on model_id)."""
    __tablename__ = "model_group_members"

    id = Column(Integer, primary_key=True, index=True)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String(200), unique=True, nullable=False)  # unique enforces single-group membership
    created_at = Column(DateTime, default=datetime.utcnow)

    group = relationship("ModelGroup", back_populates="members")

    __table_args__ = (
        Index("ix_model_group_members_group_id", "group_id"),
    )


class UserModelGroupRateLimit(Base):
    """Per-user override for a model group's rate limits. Absent row = inherit group default."""
    __tablename__ = "user_model_group_rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("model_groups.id", ondelete="CASCADE"), nullable=False)
    rpm_limit = Column(Integer, nullable=True)  # null = inherit group default
    rpd_limit = Column(Integer, nullable=True)  # null = inherit group default
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(50), nullable=True)

    group = relationship("ModelGroup", back_populates="user_limits")

    __table_args__ = (
        UniqueConstraint("user_id", "group_id", name="uq_user_model_group_rate_limit"),
    )


class ProviderCredentials(Base):
    """Provider credentials model for storing provider configurations."""
    __tablename__ = "provider_credentials"
    
    id = Column(Integer, primary_key=True, index=True)
    provider_key = Column(String(100), unique=True, index=True, nullable=False)  # e.g., "llamacpp:site24x7-fgpu" or "azure:primary"
    provider_type = Column(String(50), nullable=False)  # e.g., "azure", "openai_compatible"
    instance_name = Column(String(100), nullable=False)  # e.g., "primary", "site24x7-fgpu" (customizable)
    enabled = Column(Boolean, default=True)
    
    # Provider-specific configuration fields (nullable, used based on provider_type)
    endpoint = Column(String(500), nullable=True)          # OpenAI, Azure
    api_key = Column(String(500), nullable=True)           # OpenAI, Azure, Google
    discovery_api_version = Column(String(50), nullable=True)  # Azure: Management API version for dynamic discovery
    azure_backend = Column(String(50), nullable=True)      # Azure: "openai" or "foundry"
    region = Column(String(50), nullable=True)             # Bedrock
    access_key_id = Column(String(200), nullable=True)     # Bedrock
    secret_access_key = Column(String(500), nullable=True) # Bedrock
    base_url = Column(String(500), nullable=True)          # Ollama, OpenAI-compatible
    deployments_json = Column(Text, nullable=True)         # Azure (JSON array)
    provider_name = Column(String(100), nullable=False)    # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")

    dynamic_discovery = Column(Boolean, nullable=True)     # Azure: True to discover via the models API, False to use manual deployments
    
    # Supported API formats for custom providers (JSON array, e.g., '["openai", "anthropic"]')
    supported_apis = Column(Text, nullable=True, default='["openai"]')
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to models
    models = relationship("ModelConfiguration", back_populates="provider", cascade="all, delete-orphan")


# Pydantic models for API requests/responses
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool
    is_pending_approval: bool = False

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    name: str


class APIKeyResponse(BaseModel):
    id: int
    name: str
    api_key: str
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True


class APIKeyListResponse(BaseModel):
    id: int
    name: str
    api_key_preview: str  # Only show first 8 characters
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    is_admin: Optional[bool] = False


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str


class AdminPasswordReset(BaseModel):
    new_password: str


class GlobalRateLimitResponse(BaseModel):
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True


class GlobalRateLimitUpdate(BaseModel):
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_default is not None and self.rpm_default < 0:
            raise ValueError("rpm_default must be >= 0")
        if self.rpd_default is not None and self.rpd_default < 0:
            raise ValueError("rpd_default must be >= 0")
        return self


class UserRateLimitResponse(BaseModel):
    user_id: int
    username: str
    email: str
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    effective_rpm: Optional[int] = None
    effective_rpd: Optional[int] = None
    current_rpm_count: int = 0
    current_rpd_count: int = 0

    class Config:
        from_attributes = True


class UserRateLimitUpdate(BaseModel):
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_limit is not None and self.rpm_limit < 0:
            raise ValueError("rpm_limit must be >= 0")
        if self.rpd_limit is not None and self.rpd_limit < 0:
            raise ValueError("rpd_limit must be >= 0")
        return self


class AccountDelete(BaseModel):
    confirmation: str


# ---------------------------------------------------------------------------
# Model-group Pydantic models
# ---------------------------------------------------------------------------

class ModelGroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None

    @model_validator(mode="after")
    def validate_fields(self):
        import re
        self.name = self.name.strip()
        if not self.name or len(self.name) > 64:
            raise ValueError("name must be 1–64 characters")
        if not re.match(r'^[A-Za-z0-9_\- ]+$', self.name):
            raise ValueError("name may only contain letters, digits, spaces, hyphens, and underscores")
        if self.description is not None:
            self.description = self.description.strip()[:256]
        if self.rpm_default is not None and self.rpm_default < 0:
            raise ValueError("rpm_default must be >= 0")
        if self.rpd_default is not None and self.rpd_default < 0:
            raise ValueError("rpd_default must be >= 0")
        return self


class ModelGroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def validate_fields(self):
        import re
        if self.name is not None:
            self.name = self.name.strip()
            if not self.name or len(self.name) > 64:
                raise ValueError("name must be 1–64 characters")
            if not re.match(r'^[A-Za-z0-9_\- ]+$', self.name):
                raise ValueError("name may only contain letters, digits, spaces, hyphens, and underscores")
        if self.description is not None:
            self.description = self.description.strip()[:256]
        return self


class ModelGroupLimitsUpdate(BaseModel):
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_default is not None and self.rpm_default < 0:
            raise ValueError("rpm_default must be >= 0")
        if self.rpd_default is not None and self.rpd_default < 0:
            raise ValueError("rpd_default must be >= 0")
        return self


class ModelGroupMembersUpdate(BaseModel):
    model_ids: List[str]

    @model_validator(mode="after")
    def validate_model_ids(self):
        seen = set()
        clean = []
        for mid in self.model_ids:
            mid = mid.strip()
            if not mid or len(mid) > 200:
                raise ValueError(f"model_id '{mid}' is invalid (must be 1–200 chars)")
            if mid not in seen:
                seen.add(mid)
                clean.append(mid)
        self.model_ids = clean
        return self


class ModelGroupMemberResponse(BaseModel):
    model_id: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelGroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    rpm_default: Optional[int] = None
    rpd_default: Optional[int] = None
    member_count: int = 0
    members: List[str] = []
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True


class UserModelGroupRateLimitResponse(BaseModel):
    user_id: int
    username: str
    email: str
    group_id: int
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    effective_rpm: Optional[int] = None
    effective_rpd: Optional[int] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# User-facing quota response models (GET /auth/quotas)
# ---------------------------------------------------------------------------

class QuotaOverallResponse(BaseModel):
    rpm_limit: Optional[int] = None      # null = unlimited
    rpm_count: int = 0                   # requests used this minute
    rpm_remaining: Optional[int] = None  # null = unlimited
    rpd_limit: Optional[int] = None      # null = unlimited
    rpd_count: int = 0                   # requests used today (UTC)
    rpd_remaining: Optional[int] = None  # null = unlimited


class QuotaGroupResponse(BaseModel):
    name: str
    description: Optional[str] = None
    rpm_limit: Optional[int] = None      # effective: override if set, else group default; null = unlimited
    rpd_limit: Optional[int] = None      # effective; null = unlimited
    models: List[str] = []               # member model_ids in this group


class MyQuotasResponse(BaseModel):
    is_admin: bool = False                             # true ⇒ exempt from all limits
    overall: Optional[QuotaOverallResponse] = None    # null when is_admin
    groups: List[QuotaGroupResponse] = []


class UserModelGroupRateLimitUpdate(BaseModel):
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None

    @model_validator(mode="after")
    def validate_non_negative(self):
        if self.rpm_limit is not None and self.rpm_limit < 0:
            raise ValueError("rpm_limit must be >= 0")
        if self.rpd_limit is not None and self.rpd_limit < 0:
            raise ValueError("rpd_limit must be >= 0")
        return self


# Model Management Pydantic models (updated to use ProviderCredentials)
class ProviderConfigurationResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider_key: str
    provider_type: str
    instance_name: str
    provider_name: Optional[str] = None  # Only for custom providers
    enabled: bool  # Changed from is_enabled to enabled for consistency
    model_count: int
    enabled_model_count: int
    supported_apis: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime


class ModelConfigurationResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    model_id: str
    provider_key: str
    model_name: str
    is_enabled: bool
    created_at: datetime
    updated_at: datetime


class ModelManagementTree(BaseModel):
    providers: List[ProviderConfigurationResponse]
    total_models: int
    enabled_models: int


class ToggleRequest(BaseModel):
    enabled: bool


class BulkToggleRequest(BaseModel):
    action: str  # "enable_all" or "disable_all"


class ModelSearchResponse(BaseModel):
    models: List[ModelConfigurationResponse]
    providers: List[ProviderConfigurationResponse]
    total_results: int


# Provider Credentials Pydantic models
class ProviderCredentialsCreate(BaseModel):
    provider_type: str
    instance_name: str
    provider_name: str  # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")
    enabled: bool = True
    
    # Provider-specific fields (all optional)
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Will be converted to JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    dynamic_discovery: Optional[bool] = None

    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]

    @field_validator('azure_backend')
    @classmethod
    def validate_azure_backend(cls, v):
        if v is not None and v not in VALID_AZURE_BACKENDS:
            raise ValueError(f"azure_backend must be one of {VALID_AZURE_BACKENDS}, got '{v}'")
        return v

    @field_validator('openai_deployments', 'anthropic_deployments')
    @classmethod
    def validate_deployment_names(cls, v):
        if v is None:
            return v
        for name in v:
            stripped = name.strip()
            if stripped and not DEPLOYMENT_NAME_PATTERN.match(stripped):
                raise ValueError(
                    f"Invalid deployment name '{stripped}'. "
                    "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
                )
        return v

    @model_validator(mode='after')
    def validate_discovery_deployments(self):
        if self.dynamic_discovery is True:
            has_deployments = (
                (self.deployments and len(self.deployments) > 0)
                or (self.openai_deployments and len(self.openai_deployments) > 0)
                or (self.anthropic_deployments and len(self.anthropic_deployments) > 0)
            )
            if has_deployments:
                raise ValueError(
                    "Cannot specify deployments when dynamic_discovery is enabled. "
                    "Either disable dynamic_discovery or remove deployment lists."
                )
            # Azure dynamic discovery uses GET {endpoint}/openai/models?api-version=...
            # for both backends, so an api-version is always required.
            if self.provider_type == "azure" and not self.discovery_api_version:
                raise ValueError(
                    "discovery_api_version is required when dynamic_discovery is enabled "
                    "for an Azure provider."
                )
        return self


class ProviderCredentialsUpdate(BaseModel):
    instance_name: Optional[str] = None
    enabled: Optional[bool] = None
    
    # Provider-specific fields (all optional)
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Will be converted to JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    provider_name: Optional[str] = None  # OpenAI-compatible provider name (e.g., "ollama", "llamacpp", "openai")
    dynamic_discovery: Optional[bool] = None

    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]

    @field_validator('azure_backend')
    @classmethod
    def validate_azure_backend(cls, v):
        if v is not None and v not in VALID_AZURE_BACKENDS:
            raise ValueError(f"azure_backend must be one of {VALID_AZURE_BACKENDS}, got '{v}'")
        return v

    @field_validator('openai_deployments', 'anthropic_deployments')
    @classmethod
    def validate_deployment_names(cls, v):
        if v is None:
            return v
        for name in v:
            stripped = name.strip()
            if stripped and not DEPLOYMENT_NAME_PATTERN.match(stripped):
                raise ValueError(
                    f"Invalid deployment name '{stripped}'. "
                    "Names must start with alphanumeric and contain only letters, digits, hyphens, underscores, or dots."
                )
        return v

    @model_validator(mode='after')
    def validate_discovery_deployments(self):
        if self.dynamic_discovery is True:
            has_deployments = (
                (self.deployments and len(self.deployments) > 0)
                or (self.openai_deployments and len(self.openai_deployments) > 0)
                or (self.anthropic_deployments and len(self.anthropic_deployments) > 0)
            )
            if has_deployments:
                raise ValueError(
                    "Cannot specify deployments when dynamic_discovery is enabled. "
                    "Either disable dynamic_discovery or remove deployment lists."
                )
            # Update payloads may omit discovery_api_version when it already exists
            # in the DB, so only reject if it is explicitly set to empty/None here.
            if self.discovery_api_version is not None and not self.discovery_api_version:
                raise ValueError(
                    "discovery_api_version cannot be empty when dynamic_discovery is enabled."
                )
        return self


class ProviderCredentialsResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider_key: str
    provider_type: str
    instance_name: str
    enabled: bool
    
    # Provider-specific fields
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    discovery_api_version: Optional[str] = None
    azure_backend: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    base_url: Optional[str] = None
    deployments: Optional[List[str]] = None  # Parsed from JSON
    openai_deployments: Optional[List[str]] = None
    anthropic_deployments: Optional[List[str]] = None
    provider_name: str  # Provider name (e.g., "ollama", "azure", "openai", "bedrock", "google")
    dynamic_discovery: Optional[bool] = None
    
    # Supported API formats for custom providers
    supported_apis: Optional[List[str]] = None  # e.g., ["openai", "anthropic"]
    
    created_at: datetime
    updated_at: datetime


# OAuth models
class OAuthUserCreate(BaseModel):
    provider: str
    provider_user_id: str
    email: str
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture: Optional[str] = None
    raw_data: Optional[str] = None


class OAuthUserResponse(BaseModel):
    model_config = {"protected_namespaces": (), "from_attributes": True}
    
    id: int
    provider: str
    provider_user_id: str
    email: str
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    picture: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ZohoOAuthCallback(BaseModel):
    code: str
    state: Optional[str] = None
    location: Optional[str] = None

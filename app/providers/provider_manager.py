from typing import List, Dict, Any, Optional, AsyncGenerator, Set
import json
import asyncio
import logging
import time
import os
import weakref
from app.config import config
from app.cache import ModelCache
from app.openai_models import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    ModelInfo,
    ResponsesCreateRequest,
    ResponsesCompactRequest,
    ResponsesInputTokensRequest,
    ResponseObject,
    ResponseDeletedObject,
    ResponseInputTokensResult,
    CompactedResponseObject,
    ResponseItemList
)
from app.providers.base import BaseProvider
from app.providers.custom_providers import create_custom_provider
from app.providers.azure_provider import AzureProvider
from app.providers.bedrock_provider import BedrockProvider
from app.providers.google_provider import GoogleProvider
from app.providers.azure_deployments import build_azure_config_fields, merge_azure_deployments, normalize_azure_deployments
from opentelemetry import trace
from opentelemetry import context as otel_context
from app.tracing import (
    create_span,
    add_span_attributes,
    set_span_error,
    safe_detach
)

logger = logging.getLogger(__name__)

# Constants
MODEL_FETCH_TIMEOUT_SECONDS = 180  # 3 minutes
SYNC_TRACKING_TTL_SECONDS = 300    # 5 minutes - TTL for tracking recently synced providers

class ProviderManager:
    """Manages all LLM providers and routes requests.
    
    Includes background task tracking for proper cleanup on shutdown.
    """
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.model_cache = ModelCache()
        self.model_cache.set_provider_manager(self)
        self._initialized = False
        self._recently_synced_providers: Dict[str, float] = {}  # Track providers synced via auto-sync with timestamps
        
        # Background task tracking for proper cleanup
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Per-provider lock to prevent concurrent DB syncs for the same provider
        self._provider_sync_locks: Dict[str, asyncio.Lock] = {}
    
    def _track_task(self, task: asyncio.Task) -> None:
        """Track a background task for proper cleanup on shutdown."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def cleanup_background_tasks(self) -> None:
        """Cancel all background tasks on shutdown."""
        if not self._background_tasks:
            return

        print(f"Cancelling {len(self._background_tasks)} background tasks...")
        for task in self._background_tasks:
            task.cancel()

        # Wait for all tasks to complete (with cancellation)
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        print("Background tasks cleaned up")

    async def close_provider_clients(self) -> None:
        """Close HTTP clients for all providers to release file descriptors."""
        for name, provider in self.providers.items():
            try:
                # Close AsyncOpenAI / AsyncAzureOpenAI client
                client = getattr(provider, 'client', None)
                if client and hasattr(client, 'close'):
                    await client.close()
                # Close separate responses client (e.g. Azure v1 client)
                responses_client = getattr(provider, '_responses_client', None)
                if responses_client and responses_client is not client and hasattr(responses_client, 'close'):
                    await responses_client.close()
                # Close separate v1 client if distinct
                v1_client = getattr(provider, '_v1_client', None)
                if v1_client and v1_client is not client and v1_client is not responses_client and hasattr(v1_client, 'close'):
                    await v1_client.close()
                # Close Anthropic async client — try aclose() first (anthropic SDK), then close()
                anthropic_client = getattr(provider, '_anthropic_client', None)
                if anthropic_client:
                    if hasattr(anthropic_client, 'aclose'):
                        await anthropic_client.aclose()
                    elif hasattr(anthropic_client, 'close'):
                        await anthropic_client.close()
                # Close boto3 sync clients (urllib3 connection pools)
                for boto_attr in ('bedrock_runtime', 'bedrock_runtime_native_stream', 'bedrock_client'):
                    boto_client = getattr(provider, boto_attr, None)
                    if boto_client is not None:
                        try:
                            boto_client._endpoint.http_session.close()
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error closing client for provider {name}: {e}")
    
    async def _initialize_providers(self):
        """Initialize all enabled providers from database (async)."""
        try:
            # Load providers from database only (no YAML fallback)
            await self._load_providers_from_database()
        except Exception as e:
            print(f"Failed to load providers from database: {e}")
            print("Database-only mode: No providers will be loaded. Use the admin panel to configure providers.")
            # No fallback to YAML - database-only mode
    
    async def _load_providers_from_database(self):
        """Load providers from database (async)."""
        try:
            from app.auth.database import AsyncSessionLocal, get_all_provider_credentials
            
            # Use async session for initialization
            async with AsyncSessionLocal() as db:
                # Get all provider credentials from database
                credentials_list = await get_all_provider_credentials(db)
                
                if not credentials_list:
                    print("No provider credentials found in database. Use the admin panel to configure providers.")
                    return
                
                # Initialize providers from database credentials
                # Map specific provider types to their implementations
                specialized_providers = {
                    'azure': AzureProvider,
                    'bedrock': BedrockProvider,
                    'google': GoogleProvider,
                }
                
                for cred in credentials_list:
                    if cred.enabled:
                        try:
                            # Use specialized provider if available, otherwise use custom/OpenAI-compatible
                            if cred.provider_type in specialized_providers:
                                provider_factory = specialized_providers[cred.provider_type]
                            else:
                                # All other providers (including 'custom' and legacy 'openai_compatible')
                                provider_factory = create_custom_provider
                            
                            provider_config = self._create_provider_config(cred)
                            self.providers[cred.provider_key] = provider_factory(provider_config)
                            print(f"Initialized {cred.provider_key} provider from database")
                        except Exception as e:
                            print(f"Failed to initialize {cred.provider_key} provider: {e}")
                
                print(f"Loaded {len(self.providers)} providers from database")
                
        except Exception as e:
            print(f"Database provider loading failed: {e}")
            raise
    
    def _create_provider_config(self, cred) -> Dict[str, Any]:
        """Create provider configuration dict from database credentials."""
        config_dict = {
            'name': cred.instance_name,
            'enabled': cred.enabled
        }
        
        # Add provider_name for OpenAI-compatible providers
        if hasattr(cred, 'provider_name') and cred.provider_name:
            config_dict['provider_name'] = cred.provider_name
            config_dict['custom_provider_name'] = cred.provider_name  # For backward compatibility
        
        # Add provider-specific fields based on type
        if cred.provider_type == 'azure':
            config_dict.update(build_azure_config_fields(cred))
            config_dict.update({
                # Azure AD fields for dynamic deployment discovery
                'subscription_id': cred.subscription_id,
                'resource_group': cred.resource_group,
                'account_name': cred.account_name,
                'client_id': cred.client_id,
                'client_secret': cred.client_secret,
                'tenant_id': cred.tenant_id
            })
        elif cred.provider_type == 'google':
            config_dict.update({
                'api_key': cred.api_key,
                'base_url': cred.base_url
            })
        elif cred.provider_type == 'bedrock':
            config_dict.update({
                'region': cred.region or 'us-west-2',
                'access_key_id': cred.access_key_id,
                'secret_access_key': cred.secret_access_key,
                'api_key': cred.api_key,  # Support for OpenAI-compatible mode
                'base_url': cred.base_url,  # Support for configurable base URL
                # Enable inference profiles by default (required for Claude Sonnet 4 and similar models)
                'enable_cross_region_inference': True,
                'enable_application_inference_profiles': True
            })
        else:
            # All other providers are custom (OpenAI/Anthropic compatible)
            # Parse supported_apis from database
            supported_apis = ['openai']  # default
            if hasattr(cred, 'supported_apis') and cred.supported_apis:
                try:
                    import json as _json
                    parsed = _json.loads(cred.supported_apis)
                    if isinstance(parsed, list):
                        supported_apis = parsed
                except (ValueError, TypeError):
                    pass
            
            config_dict.update({
                'base_url': cred.base_url or cred.endpoint,
                'api_key': cred.api_key,
                'supported_apis': supported_apis,
            })
        
        return config_dict
    
    
    async def _fetch_all_models(self) -> List[ModelInfo]:
        """Fetch all available models from all providers (internal method)."""
        all_models = []
        
        # Create tasks for all providers with timeout
        tasks = []
        for provider_name, provider in self.providers.items():
            task = asyncio.create_task(
                self._fetch_models_with_timeout(provider_name, provider)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete (they have individual timeouts)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"Provider task failed: {result}")
                    continue
                if isinstance(result, list):
                    all_models.extend(result)
        
        print(f"Total models fetched from all providers: {len(all_models)}")
        return all_models
    
    async def _fetch_models_with_timeout(self, provider_name: str, provider: BaseProvider, timeout: int = MODEL_FETCH_TIMEOUT_SECONDS) -> List[ModelInfo]:
        """Fetch models from a single provider with timeout (3 minutes for all providers)."""
        with create_span(
            "provider.fetch_models",
            attributes={
                "provider.name": provider_name,
                "provider.type": getattr(provider, 'provider_type', 'unknown'),
                "provider.timeout_seconds": timeout
            }
        ) as span:
            try:
                print(f"Fetching models from {provider_name} (timeout: {timeout}s)...")
                
                # For Azure provider, add extra debugging and error handling
                if hasattr(provider, 'provider_type') and provider.provider_type == 'azure':
                    print(f"Azure provider config - endpoint: {getattr(provider, 'endpoint', 'None')}")
                    print(f"Azure provider config - dynamic_discovery: {getattr(provider, 'dynamic_discovery', 'None')}")
                    print(f"Azure provider config - deployments: {getattr(provider, 'deployments', 'None')}")
                    
                    add_span_attributes(span, {
                        "provider.azure.endpoint": getattr(provider, 'endpoint', None),
                        "provider.azure.dynamic_discovery": getattr(provider, 'dynamic_discovery', None)
                    })
                    
                    # Try to fetch deployments first to see if that's working
                    try:
                        deployments = await provider._fetch_deployments()
                        print(f"Azure deployments fetched: {deployments}")
                        add_span_attributes(span, {
                            "provider.azure.deployments_fetched": len(deployments) if deployments else 0
                        })
                    except Exception as deploy_error:
                        print(f"Error fetching Azure deployments: {deploy_error}")
                        add_span_attributes(span, {
                            "provider.azure.deployment_fetch_error": str(deploy_error)
                        })
                        # If deployment fetching fails, try with a fallback configuration
                        if hasattr(provider, 'deployments') and provider.deployments:
                            print(f"Using fallback deployments from config: {provider.deployments}")
                            models = []
                            for deployment_name in provider.deployments:
                                models.append(provider.create_model_info(deployment_name, "azure"))
                            print(f"Created {len(models)} models from fallback deployments")
                            add_span_attributes(span, {
                                "provider.models_count": len(models),
                                "provider.fallback_used": True
                            })
                            return models
                
                # Use asyncio.wait_for to add timeout
                models = await asyncio.wait_for(
                    provider.get_available_models(),
                    timeout=timeout
                )
                
                print(f"Successfully fetched {len(models)} models from {provider_name}")
                add_span_attributes(span, {
                    "provider.models_count": len(models),
                    "provider.status": "success"
                })
                return models
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout fetching models from {provider_name} after {timeout}s"
                print(error_msg)
                add_span_attributes(span, {
                    "provider.models_count": 0,
                    "provider.status": "timeout"
                })
                set_span_error(span, error_msg)
                return []
            except Exception as e:
                print(f"Error getting models from {provider_name}: {e}")
                import traceback
                print(f"Full traceback for {provider_name}:")
                traceback.print_exc()
                
                set_span_error(span, e)
                
                # For Azure provider, try a fallback approach
                if hasattr(provider, 'provider_type') and provider.provider_type == 'azure':
                    try:
                        if hasattr(provider, 'deployments') and provider.deployments:
                            print(f"Attempting Azure fallback with deployments: {provider.deployments}")
                            models = []
                            for deployment_name in provider.deployments:
                                models.append(provider.create_model_info(deployment_name, "azure"))
                            print(f"Azure fallback created {len(models)} models")
                            add_span_attributes(span, {
                                "provider.models_count": len(models),
                                "provider.fallback_used": True,
                                "provider.status": "fallback_success"
                            })
                            return models
                    except Exception as fallback_error:
                        print(f"Azure fallback also failed: {fallback_error}")
                        add_span_attributes(span, {
                            "provider.fallback_error": str(fallback_error)
                        })
                
                add_span_attributes(span, {
                    "provider.models_count": 0,
                    "provider.status": "error"
                })
                return []
    
    async def initialize(self) -> None:
        """Initialize providers and model cache during startup."""
        if self._initialized:
            return
            
        print("Initializing provider manager...")
        try:
            # First initialize providers
            await self._initialize_providers()
            self._initialized = True
            
            # Start background task to initialize model cache (non-blocking)
            task = asyncio.create_task(self._background_initialize_models())
            # Track task for proper cleanup and add exception handler
            self._track_task(task)
            task.add_done_callback(self._handle_background_task_completion)
            print("Provider manager initialized. Model cache loading in background...")
            
        except Exception as e:
            print(f"Error initializing provider manager: {e}")
            raise
    
    async def _background_initialize_models(self) -> None:
        """Initialize model cache in background (non-blocking)."""
        # Detach from parent trace context so this background task gets its own trace
        token = otel_context.attach(otel_context.Context())
        try:
            with create_span("provider.background_initialize_models") as span:
                try:
                    # First, load existing model configurations from database (fast)
                    await self._load_model_configurations()
                    print("✓ Loaded existing model configurations from database")
                    
                    # Then fetch models from all providers concurrently (slow)
                    await self._fetch_and_sync_all_models()
                    
                    add_span_attributes(span, {
                        "status": "success"
                    })
                except Exception as e:
                    print(f"Background model initialization error: {e}")
                    set_span_error(span, e)
        finally:
            safe_detach(token)
    
    def _handle_background_task_completion(self, task: asyncio.Task) -> None:
        """Handle completion of background task and log any exceptions."""
        try:
            # This will raise any exception that occurred in the task
            task.result()
        except Exception as e:
            print(f"Background task failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    async def _fetch_and_sync_all_models(self) -> None:
        """Fetch models from all providers and sync to database as each completes."""
        with create_span("provider.fetch_and_sync_all_models") as parent_span:
            try:
                print("Starting background model fetch and sync for all providers...")
                
                add_span_attributes(parent_span, {
                    "provider.count": len(self.providers),
                    "provider.names": ",".join(self.providers.keys())
                })
                
                # Create tasks for all providers
                tasks = []
                provider_names = []
                for provider_name, provider in self.providers.items():
                    task = asyncio.create_task(
                        self._fetch_and_sync_provider_models(provider_name, provider)
                    )
                    tasks.append(task)
                    provider_names.append(provider_name)
                
                # Wait for all tasks to complete (each updates cache/DB independently)
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    success_count = 0
                    failed_providers = []
                    for provider_name, result in zip(provider_names, results):
                        if isinstance(result, Exception):
                            print(f"❌ Provider {provider_name} failed: {result}")
                            failed_providers.append(provider_name)
                        else:
                            success_count += 1
                            print(f"✅ Provider {provider_name} completed successfully")
                    
                    print(f"Background sync completed: {success_count}/{len(provider_names)} providers successful")
                    
                    # Update parent span with results
                    add_span_attributes(parent_span, {
                        "provider.success_count": success_count,
                        "provider.failed_count": len(failed_providers),
                        "provider.failed_names": ",".join(failed_providers) if failed_providers else None
                    })
                    
                    if failed_providers:
                        set_span_error(parent_span, f"Some providers failed: {', '.join(failed_providers)}")
            except Exception as e:
                set_span_error(parent_span, e)
                raise
    
    async def _fetch_and_sync_provider_models(self, provider_name: str, provider: BaseProvider) -> None:
        """Fetch models from a provider and immediately sync to cache and database."""
        with create_span(
            f"provider.fetch_and_sync_models",
            attributes={
                "provider.name": provider_name,
                "provider.type": getattr(provider, 'provider_type', 'unknown')
            }
        ) as span:
            try:
                # Clean up old tracking entries
                self._cleanup_sync_tracking()
                
                # Skip if this provider was recently synced via auto-sync
                if provider_name in self._recently_synced_providers:
                    sync_time = self._recently_synced_providers[provider_name]
                    age = time.time() - sync_time
                    if age < SYNC_TRACKING_TTL_SECONDS:
                        print(f"⏭️  Skipping {provider_name} - recently synced {age:.1f}s ago")
                        add_span_attributes(span, {
                            "provider.skipped": True,
                            "provider.skip_reason": "recently_synced",
                            "provider.last_sync_age_seconds": age
                        })
                        return
                
                # Fetch models with timeout
                models = await self._fetch_models_with_timeout(provider_name, provider)
                
                if not models:
                    print(f"No models fetched from {provider_name}")
                    add_span_attributes(span, {
                        "provider.models_fetched": 0,
                        "provider.status": "no_models"
                    })
                    return
                
                add_span_attributes(span, {
                    "provider.models_fetched": len(models)
                })
                
                # Update in-memory cache by adding to existing models (not replacing)
                existing_models = self.model_cache.get_models()
                # Remove old models from this provider
                other_provider_models = [m for m in existing_models if not m.id.startswith(f"{provider_name}/")]
                # Add new models from this provider
                updated_models = other_provider_models + models
                self.model_cache.update_models(updated_models)
                print(f"✓ Cache updated with {len(models)} models from {provider_name} (total: {len(updated_models)})")
                
                add_span_attributes(span, {
                    "provider.cache_updated": True,
                    "provider.total_models_in_cache": len(updated_models)
                })
                
                # Sync to database
                await self._sync_provider_to_database(provider_name, models)
                
                add_span_attributes(span, {
                    "provider.status": "success"
                })
                
            except Exception as e:
                print(f"Error fetching/syncing models for {provider_name}: {e}")
                set_span_error(span, e)
                raise
    
    async def _sync_provider_to_database(self, provider_name: str, models: List[ModelInfo]) -> None:
        """Sync provider models to database with per-provider locking to prevent races."""
        # Get or create a lock for this provider
        if provider_name not in self._provider_sync_locks:
            self._provider_sync_locks[provider_name] = asyncio.Lock()
        
        async with self._provider_sync_locks[provider_name]:
            await self._sync_provider_to_database_locked(provider_name, models)

    async def _sync_provider_to_database_locked(self, provider_name: str, models: List[ModelInfo]) -> None:
        """Sync provider models to database (must be called under lock).
        
        Uses upsert logic: update existing models, insert new ones, remove stale ones.
        """
        with create_span(
            "provider.sync_to_database",
            attributes={
                "provider.name": provider_name,
                "provider.models_to_sync": len(models),
            }
        ) as span:
            try:
                from app.auth.database import AsyncSessionLocal, create_or_update_model_configuration
                
                async with AsyncSessionLocal() as db:
                    try:
                        # Get existing models to preserve enabled states and detect stale models
                        from app.auth.database import get_models_by_provider
                        existing_models = await get_models_by_provider(db, provider_name)
                        enabled_states = {m.model_id: m.is_enabled for m in existing_models}
                        existing_ids = set(enabled_states.keys())
                        
                        # Build set of new model IDs
                        new_ids = {m.id for m in models}
                        
                        # Delete stale models (in DB but no longer from provider)
                        stale_ids = existing_ids - new_ids
                        if stale_ids:
                            for m in existing_models:
                                if m.model_id in stale_ids:
                                    await db.delete(m)
                            await db.commit()
                        
                        # Upsert current models
                        created_count = 0
                        for model in models:
                            model_name = model.id.split('/', 1)[1] if '/' in model.id else model.id
                            is_enabled = enabled_states.get(model.id, True)
                            
                            await create_or_update_model_configuration(
                                db=db,
                                model_id=model.id,
                                provider_key=provider_name,
                                model_name=model_name,
                                is_enabled=is_enabled
                            )
                            created_count += 1
                        
                        await db.commit()
                        print(f"✓ Database synced with {created_count} models for {provider_name}" +
                              (f" (removed {len(stale_ids)} stale)" if stale_ids else ""))
                        
                        add_span_attributes(span, {
                            "provider.models_created": created_count,
                            "provider.models_stale_removed": len(stale_ids),
                            "provider.db_sync_status": "success"
                        })
                        
                    except Exception as e:
                        await db.rollback()
                        print(f"Database sync failed for {provider_name}: {e}")
                        set_span_error(span, e)
                        raise
                        
            except Exception as e:
                print(f"Error syncing {provider_name} to database: {e}")
                set_span_error(span, e)

    
    async def initialize_models(self) -> None:
        """Initialize model cache during startup."""
        print("Initializing model cache...")
        try:
            models = await self._fetch_all_models()
            self.model_cache.warm_cache(models)
            
            # Load model configurations from database
            await self._load_model_configurations()
            
            print(f"Model cache initialized with {len(models)} models")
            
        except Exception as e:
            print(f"Error initializing model cache: {e}")
            raise
    
    async def _load_model_configurations(self) -> None:
        """Load model and provider configurations from database into cache."""
        with create_span("provider.load_model_configurations") as span:
            try:
                from app.auth.database import AsyncSessionLocal, get_model_configurations_dict, get_provider_configurations_dict
                
                # Create database session directly using session factory
                async with AsyncSessionLocal() as db:
                    try:
                        # Load configurations from database
                        model_configs = await get_model_configurations_dict(db)
                        provider_configs = await get_provider_configurations_dict(db)
                        
                        # Update cache
                        self.model_cache.update_model_configurations(model_configs, provider_configs)
                        
                        print(f"Loaded {len(model_configs)} model configs and {len(provider_configs)} provider configs")
                        
                        add_span_attributes(span, {
                            "config.models_loaded_count": len(model_configs),
                            "config.providers_loaded_count": len(provider_configs),
                            "config.status": "success"
                        })
                        
                    except Exception as e:
                        print(f"Database operation failed during model configuration loading: {e}")
                        print("Falling back to default behavior (all models enabled)")
                        # Continue with empty configurations (all enabled by default)
                        self.model_cache.update_model_configurations({}, {})
                        
                        add_span_attributes(span, {
                            "config.models_loaded_count": 0,
                            "config.providers_loaded_count": 0,
                            "config.status": "fallback",
                            "config.fallback_reason": "database_operation_failed"
                        })
                        set_span_error(span, e)
                        
            except ImportError as e:
                print(f"Database module import failed: {e}")
                print("Falling back to default behavior (all models enabled)")
                self.model_cache.update_model_configurations({}, {})
                
                add_span_attributes(span, {
                    "config.models_loaded_count": 0,
                    "config.providers_loaded_count": 0,
                    "config.status": "fallback",
                    "config.fallback_reason": "import_failed"
                })
                set_span_error(span, e)
            except Exception as e:
                print(f"Database connection failed: {e}")
                print("Falling back to default behavior (all models enabled)")
                # Continue with empty configurations (all enabled by default)
                self.model_cache.update_model_configurations({}, {})
                
                add_span_attributes(span, {
                    "config.models_loaded_count": 0,
                    "config.providers_loaded_count": 0,
                    "config.status": "fallback",
                    "config.fallback_reason": "connection_failed"
                })
                set_span_error(span, e)
    
    async def refresh_model_configurations(self) -> None:
        """Refresh model configurations from database."""
        await self._load_model_configurations()
    
    def mark_provider_synced(self, provider_key: str) -> None:
        """Mark a provider as recently synced (to avoid re-syncing on startup)."""
        self._recently_synced_providers[provider_key] = time.time()
        print(f"Provider {provider_key} marked as recently synced")
    
    def _cleanup_sync_tracking(self) -> None:
        """Remove expired entries from sync tracking."""
        current_time = time.time()
        expired_keys = [
            key for key, sync_time in self._recently_synced_providers.items()
            if current_time - sync_time > SYNC_TRACKING_TTL_SECONDS
        ]
        for key in expired_keys:
            del self._recently_synced_providers[key]
        if expired_keys:
            print(f"Cleaned up {len(expired_keys)} expired sync tracking entries")
    
    def clear_sync_tracking(self) -> None:
        """Clear the recently synced providers tracking."""
        self._recently_synced_providers.clear()
    
    async def remove_provider(self, provider_key: str) -> bool:
        """Remove a provider from the manager and invalidate its models from cache."""
        if provider_key in self.providers:
            provider = self.providers.pop(provider_key)
            print(f"Provider {provider_key} removed from provider manager")

            # Close HTTP clients before discarding the provider to release FDs
            try:
                for attr in ('client', '_responses_client', '_v1_client', '_anthropic_client'):
                    client = getattr(provider, attr, None)
                    if client and hasattr(client, 'close'):
                        await client.close()
                for boto_attr in ('bedrock_runtime', 'bedrock_runtime_native_stream', 'bedrock_client'):
                    boto_client = getattr(provider, boto_attr, None)
                    if boto_client is not None:
                        try:
                            boto_client._endpoint.http_session.close()
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error closing clients for removed provider {provider_key}: {e}")

            # Remove all models for this provider from cache
            self.model_cache.invalidate_provider(provider_key)
            print(f"Models for provider {provider_key} removed from cache")

            return True
        return False
    
    async def refresh_providers_from_database(self) -> None:
        """Refresh the providers list from database (reload all providers)."""
        try:
            # Close HTTP clients of current providers before discarding them
            await self.close_provider_clients()
            # Clear current providers
            self.providers.clear()
            
            # Reload providers from database
            await self._load_providers_from_database()
            
            # Refresh model configurations (without fetching models from providers)
            await self._load_model_configurations()
            
            print(f"Provider manager refreshed with {len(self.providers)} providers")
        except Exception as e:
            print(f"Error refreshing providers from database: {e}")
            raise
    
    async def get_all_models(self, api_filter: str = None) -> List[ModelInfo]:
        """Get all available models (from cache, filtered by configuration).
        
        Args:
            api_filter: If set (e.g., "openai" or "anthropic"), only return models
                       from providers that support that API format.
        """
        models = self.model_cache.get_enabled_models()
        
        if api_filter:
            filtered = []
            for model in models:
                # Find the provider for this model
                provider = self.providers.get(model.provider)
                if provider:
                    if provider.supports_api_for_model(model.id, api_filter):
                        filtered.append(model)
            return filtered
        
        return models
    
    async def get_anthropic_provider_for_model(self, model_name: str) -> Optional[BaseProvider]:
        """Get the provider that serves a model via the Anthropic API.
        
        Routes to the appropriate provider based on model name prefix,
        but only if that provider supports the Anthropic API.
        """
        try:
            provider_name, model_id = self._parse_model_name(model_name)
            provider = self._get_provider(provider_name)
            
            # Verify this provider supports Anthropic API
            if provider.supports_api_for_model(model_name, "anthropic"):
                return provider
            else:
                return None
        except (ValueError, Exception) as e:
            logger.debug(f"Could not find Anthropic provider for model '{model_name}': {e}")
            return None
    
    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse provider and model from model name (e.g., 'ollama/llama2' -> ('ollama', 'llama2'))."""
        with create_span("provider.parse_model_name") as span:
            try:
                if '/' in model_name:
                    provider_name, model_id = model_name.split('/', 1)
                    return provider_name, model_id
                else:
                    # If no provider specified, try to find the model in available providers
                    error_msg = f"Model name must include provider prefix (e.g., 'ollama/{model_name}')"
                    raise ValueError(error_msg)
            except Exception as e:
                set_span_error(span, e)
                raise
    
    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider by name."""
        with create_span("provider.get_provider") as span:
            try:
                # Direct lookup for full provider names (e.g., "azure:primary", "openai_compatible:my-server")
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    return provider
                
                # Try to find a provider that starts with the requested name
                # This handles cases like "openai_compatible" matching "openai_compatible:my-server"
                for full_name in self.providers.keys():
                    if full_name.startswith(f"{provider_name}:"):
                        provider = self.providers[full_name]
                        return provider
                
                # Provider not found
                error_msg = f"Provider '{provider_name}' not available or not enabled"
                raise ValueError(error_msg)
            except Exception as e:
                set_span_error(span, e)
                raise
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route chat completion request to appropriate provider."""
        with create_span("provider.chat_completion") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                response = await provider.chat_completion(request)
                return response
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Chat completion error: {str(e)}")
    
    async def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Route completion request to appropriate provider."""
        with create_span("provider.completion") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                return await provider.completion(request)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Completion error: {str(e)}")
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        """Route streaming chat completion request to appropriate provider."""
        # Create span for the streaming setup
        with create_span("provider.chat_completion_stream") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                async for chunk in provider.chat_completion_stream(request):
                    # Add type checking here as well
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        print(f"Warning: Non-string chunk received: {type(chunk)}, {chunk}")
                        continue
                        
            except Exception as e:
                set_span_error(span, e)
                print(f"Error in chat_completion_stream: {e}")
                import traceback
                traceback.print_exc()
                # Send error as SSE format
                error_data = {
                    "error": {
                        "message": f"Chat completion stream error: {str(e)}",
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n"
    
    async def completion_stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Route streaming completion request to appropriate provider."""
        # Create span for the streaming setup
        with create_span("provider.completion_stream") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                async for chunk in provider.completion_stream(request):
                    yield chunk
            except Exception as e:
                set_span_error(span, e)
                # Send error as SSE format
                error_data = {
                    "error": {
                        "message": f"Completion stream error: {str(e)}",
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n"
    
    # ==================== RESPONSES API ====================
    
    async def _get_provider_for_response_id(self, response_id: str) -> BaseProvider:
        """Resolve provider for a response_id by looking up the DB mapping."""
        from app.auth.database import AsyncSessionLocal, get_response_provider_mapping
        
        async with AsyncSessionLocal() as db:
            mapping = await get_response_provider_mapping(db, response_id)
        
        if not mapping:
            raise ValueError(f"No provider mapping found for response_id '{response_id}'. The response may have been created before this proxy instance started, or it may not exist.")
        
        return self._get_provider(mapping.provider_key)
    
    async def _store_response_mapping(self, response_id: str, provider_name: str, model_name: str = None):
        """Store response_id -> provider mapping in the database."""
        from app.auth.database import AsyncSessionLocal, store_response_provider_mapping
        
        try:
            async with AsyncSessionLocal() as db:
                await store_response_provider_mapping(db, response_id, provider_name, model_name)
        except Exception as e:
            # Log but don't fail the request if mapping storage fails
            print(f"Warning: Failed to store response provider mapping: {e}")
    
    async def _delete_response_mapping(self, response_id: str):
        """Delete response_id -> provider mapping from the database."""
        from app.auth.database import AsyncSessionLocal, delete_response_provider_mapping
        
        try:
            async with AsyncSessionLocal() as db:
                await delete_response_provider_mapping(db, response_id)
        except Exception as e:
            print(f"Warning: Failed to delete response provider mapping: {e}")
    
    async def responses_create(self, request: ResponsesCreateRequest) -> ResponseObject:
        """Route Responses API create request to appropriate provider."""
        with create_span("provider.responses_create") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                response = await provider.responses_create(request)
                
                # Store response_id -> provider mapping for future retrieve/delete/cancel
                if response and response.id:
                    await self._store_response_mapping(response.id, provider_name, request.model)
                
                return response
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses create error: {str(e)}")
    
    async def responses_create_stream(self, request: ResponsesCreateRequest) -> AsyncGenerator[str, None]:
        """Route streaming Responses API create request to appropriate provider."""
        with create_span("provider.responses_create_stream") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                
                async for chunk in provider.responses_create_stream(request):
                    if isinstance(chunk, str):
                        # Intercept response.created event to cache the response ID
                        if 'event: response.created' in chunk:
                            try:
                                # Parse the data line to extract response ID
                                for line in chunk.split('\n'):
                                    if line.startswith('data: '):
                                        import json as _json
                                        event_data = _json.loads(line[6:])
                                        resp_id = None
                                        if 'response' in event_data and 'id' in event_data['response']:
                                            resp_id = event_data['response']['id']
                                        elif 'id' in event_data:
                                            resp_id = event_data['id']
                                        if resp_id:
                                            await self._store_response_mapping(resp_id, provider_name, request.model)
                                        break
                            except Exception as parse_err:
                                print(f"Warning: Could not parse response.created event for caching: {parse_err}")
                        
                        yield chunk
                    else:
                        print(f"Warning: Non-string chunk in responses stream: {type(chunk)}")
                        continue
                        
            except Exception as e:
                set_span_error(span, e)
                print(f"Error in responses_create_stream: {e}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "error": {
                        "message": f"Responses stream error: {str(e)}",
                        "type": "server_error"
                    }
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
    
    async def responses_retrieve(self, response_id: str, **kwargs) -> ResponseObject:
        """Route Responses API retrieve request to appropriate provider."""
        with create_span("provider.responses_retrieve") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                return await provider.responses_retrieve(response_id, **kwargs)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses retrieve error: {str(e)}")
    
    async def responses_delete(self, response_id: str) -> ResponseDeletedObject:
        """Route Responses API delete request to appropriate provider."""
        with create_span("provider.responses_delete") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                result = await provider.responses_delete(response_id)
                
                # Clean up the mapping
                await self._delete_response_mapping(response_id)
                
                return result
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses delete error: {str(e)}")
    
    async def responses_cancel(self, response_id: str) -> ResponseObject:
        """Route Responses API cancel request to appropriate provider."""
        with create_span("provider.responses_cancel") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                return await provider.responses_cancel(response_id)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses cancel error: {str(e)}")
    
    async def responses_list_input_items(self, response_id: str, **kwargs) -> ResponseItemList:
        """Route Responses API list input items request to appropriate provider."""
        with create_span("provider.responses_list_input_items") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                return await provider.responses_list_input_items(response_id, **kwargs)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses list input items error: {str(e)}")
    
    async def responses_input_tokens(self, request: ResponsesInputTokensRequest) -> ResponseInputTokensResult:
        """Route Responses API input tokens request to appropriate provider."""
        with create_span("provider.responses_input_tokens") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                return await provider.responses_input_tokens(request)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses input tokens error: {str(e)}")
    
    async def responses_compact(self, request: ResponsesCompactRequest) -> CompactedResponseObject:
        """Route Responses API compact request to appropriate provider."""
        with create_span("provider.responses_compact") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)
                return await provider.responses_compact(request)
            except Exception as e:
                set_span_error(span, e)
                raise Exception(f"Responses compact error: {str(e)}")
    
    def get_provider_for_model(self, model_name: str) -> BaseProvider:
        """Get provider for a specific model."""
        provider_name, model_id = self._parse_model_name(model_name)
        return self._get_provider(provider_name)
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names."""
        return list(self.providers.keys())
    
    async def refresh_models_from_providers(self) -> Dict[str, int]:
        """Refresh all models by fetching from providers and updating database."""
        try:
            from app.auth.database import AsyncSessionLocal, refresh_models_from_providers
            
            # Fetch fresh models from all providers
            fresh_models = await self._fetch_all_models()
            
            if not fresh_models:
                print("No models fetched from providers. Database will be cleared but no new models added.")
                return {"cleared": 0, "created": 0, "error": "No models available from providers"}
            
            # Convert ModelInfo objects to database format
            models_data = []
            for model in fresh_models:
                # Parse provider from model ID (e.g., "azure:primary/gpt-4" -> "azure:primary")
                if '/' in model.id:
                    provider_key = model.id.split('/', 1)[0]
                else:
                    # Fallback for models without provider prefix
                    provider_key = "unknown"
                
                models_data.append({
                    'model_id': model.id,
                    'provider_key': provider_key,
                    'model_name': model.id.split('/')[-1] if '/' in model.id else model.id,
                    'is_enabled': True
                })
            
            # Update database
            async with AsyncSessionLocal() as db:
                result = await refresh_models_from_providers(db, models_data)
                
                # Update model cache
                self.model_cache.update_models(fresh_models)
                await self._load_model_configurations()
                
                print(f"Model refresh completed: cleared {result['cleared']}, created {result['created']}")
                return result
                
        except Exception as e:
            print(f"Error refreshing models from providers: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


# Global provider manager instance
provider_manager = ProviderManager()

from typing import List, Dict, Any, Optional, AsyncGenerator, Set
import json
import asyncio
import logging
import time
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
from app.providers.base import BaseProvider, ProviderHTTPError
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

    async def close_provider_clients(self, providers: Optional[Dict[str, BaseProvider]] = None) -> None:
        """Close HTTP clients for providers to release file descriptors.

        Closes the given ``providers`` mapping (defaults to the live registry).
        """
        target = providers if providers is not None else self.providers
        for name, provider in list(target.items()):
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
                # Close Azure per-api-version deployment clients (otherwise these
                # AsyncAzureOpenAI instances leak on shutdown and every refresh).
                deployment_clients = getattr(provider, '_deployment_clients', None)
                if isinstance(deployment_clients, dict):
                    for dep_client in list(deployment_clients.values()):
                        try:
                            if hasattr(dep_client, 'aclose'):
                                await dep_client.aclose()
                            elif hasattr(dep_client, 'close'):
                                await dep_client.close()
                        except Exception:
                            pass
                    deployment_clients.clear()
                # Close Anthropic async client — try aclose() first (anthropic SDK), then close()
                anthropic_client = getattr(provider, '_anthropic_client', None)
                if anthropic_client:
                    if hasattr(anthropic_client, 'aclose'):
                        await anthropic_client.aclose()
                    elif hasattr(anthropic_client, 'close'):
                        await anthropic_client.close()
                # Close boto3 sync clients (urllib3 connection pools)
                for boto_attr in ('bedrock_runtime', 'bedrock_client'):
                    boto_client = getattr(provider, boto_attr, None)
                    if boto_client is not None:
                        try:
                            boto_client._endpoint.http_session.close()
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error closing client for provider {name}: {e}")

    async def _deferred_close_providers(self, providers: Dict[str, BaseProvider], grace_seconds: float = 5.0) -> None:
        """Close old provider clients after a grace period.

        Called after the registry is swapped so in-flight requests still holding
        a reference to an old provider can finish before its clients close.
        """
        try:
            await asyncio.sleep(grace_seconds)
        except asyncio.CancelledError:
            # On shutdown, close immediately rather than skipping cleanup.
            pass
        await self.close_provider_clients(providers)
    
    async def _initialize_providers(self):
        """Initialize all enabled providers from database (async)."""
        try:
            # Load providers from database only (no YAML fallback)
            await self._load_providers_from_database()
        except Exception as e:
            print(f"Failed to load providers from database: {e}")
            print("Database-only mode: No providers will be loaded. Use the admin panel to configure providers.")
            # No fallback to YAML - database-only mode
    
    async def _load_providers_from_database(self, target: Optional[Dict[str, BaseProvider]] = None):
        """Load providers from database (async).

        Populates ``target`` when given (used by refresh to build a fresh
        registry off to the side), otherwise ``self.providers``.
        """
        registry = self.providers if target is None else target
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
                            registry[cred.provider_key] = provider_factory(provider_config)
                            print(f"Initialized {cred.provider_key} provider from database")
                        except Exception as e:
                            print(f"Failed to initialize {cred.provider_key} provider: {e}")

                print(f"Loaded {len(registry)} providers from database")
                
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

                # Use asyncio.wait_for to add timeout. (The former Azure "debug
                # pre-fetch" that called _fetch_deployments() here was removed —
                # get_available_models() fetches deployments itself, so it was a
                # duplicate upstream call; the except branch below still provides
                # the Azure deployment fallback on failure.)
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
                if isinstance(provider, AzureProvider):
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
                # Propagate so _fetch_and_sync_provider_models counts this as a
                # failure instead of reporting the provider as succeeded while
                # cache and DB have silently diverged.
                raise


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
                from app.auth.database import (
                    AsyncSessionLocal,
                    get_model_configurations_dict,
                    get_provider_configurations_dict,
                    get_all_user_model_policies,
                    get_all_user_model_exceptions,
                )

                # Create database session directly using session factory
                async with AsyncSessionLocal() as db:
                    try:
                        # Load configurations from database
                        model_configs = await get_model_configurations_dict(db)
                        provider_configs = await get_provider_configurations_dict(db)

                        # Update cache
                        self.model_cache.update_model_configurations(model_configs, provider_configs)

                        # Load per-user model access (policies + exceptions)
                        user_policies = {
                            p.user_id: (p.mode or "default")
                            for p in await get_all_user_model_policies(db)
                        }
                        user_exceptions: dict = {}
                        for ex in await get_all_user_model_exceptions(db):
                            user_exceptions.setdefault(ex.user_id, {})[ex.model_id] = ex.is_allowed
                        self.model_cache.update_user_model_access(user_policies, user_exceptions)

                        print(f"Loaded {len(model_configs)} model configs and {len(provider_configs)} provider configs")
                        print(f"Loaded model access for {len(user_policies)} user policies and {len(user_exceptions)} users with exceptions")
                        
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
                    if not client:
                        continue
                    # Prefer aclose() (anthropic SDK), fall back to close().
                    if hasattr(client, 'aclose'):
                        await client.aclose()
                    elif hasattr(client, 'close'):
                        await client.close()
                # Close Azure per-api-version deployment clients too.
                deployment_clients = getattr(provider, '_deployment_clients', None)
                if isinstance(deployment_clients, dict):
                    for dep_client in list(deployment_clients.values()):
                        try:
                            if hasattr(dep_client, 'aclose'):
                                await dep_client.aclose()
                            elif hasattr(dep_client, 'close'):
                                await dep_client.close()
                        except Exception:
                            pass
                    deployment_clients.clear()
                for boto_attr in ('bedrock_runtime', 'bedrock_client'):
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
        """Refresh the providers list from database (reload all providers).

        Swaps in the freshly-loaded registry BEFORE closing the old providers'
        clients, so a request currently streaming/awaiting an upstream call
        (which holds a reference to an old provider) is not torn down mid-flight.
        Old clients are then closed after a short grace period (best-effort).
        """
        try:
            old_providers = self.providers
            # Build the fresh registry off to the side, then swap it in with a
            # single reference assignment — concurrent requests always see
            # either the old registry or the new one whole, never an
            # empty/partial map. On failure the swap never happens, so the old
            # registry keeps serving.
            new_providers: Dict[str, BaseProvider] = {}
            await self._load_providers_from_database(target=new_providers)
            self.providers = new_providers

            # Refresh model configurations (without fetching models from providers)
            await self._load_model_configurations()

            # Registry is swapped; close the OLD providers' clients after a grace
            # period so in-flight requests holding old references can finish.
            if old_providers:
                task = asyncio.create_task(self._deferred_close_providers(old_providers))
                self._track_task(task)

            print(f"Provider manager refreshed with {len(self.providers)} providers")
        except Exception as e:
            print(f"Error refreshing providers from database: {e}")
            raise
    
    async def get_all_models(self, api_filter: str = None, user_id: Optional[int] = None) -> List[ModelInfo]:
        """Get all available models (from cache, filtered by configuration).

        Args:
            api_filter: If set (e.g., "openai" or "anthropic"), only return models
                       from providers that support that API format.
            user_id: If set, further restrict to models the user is allowed to
                     access (per-user policy + exceptions). None => no per-user
                     filter (e.g. admin callers).
        """
        if user_id is not None:
            models = self.model_cache.get_enabled_models_for_user(user_id)
        else:
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
        except Exception as e:
            logger.debug(f"Could not find Anthropic provider for model '{model_name}': {e}")
            return None
    
    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse provider and model from a model name.

        Canonical names carry the provider prefix ('azure:primary/gpt-5.4' ->
        ('azure:primary', 'gpt-5.4')). A prefix-less name ('gpt-5.4') is resolved
        against the model cache instead, so callers need not know the proxy's
        provider topology.

        This is the defensive net for direct callers and the dispatch methods
        below; routes resolve the name earlier (app/model_resolution.py) so that
        rate limiting, access checks and usage attribution all see the canonical
        id. Resolution here is deterministic (first candidate) -- it does not
        round-robin, because it has no request context to key rotation on.
        """
        with create_span("provider.parse_model_name") as span:
            try:
                if not model_name:
                    raise ValueError("Model name is required")

                head, rest = model_name.split('/', 1) if '/' in model_name else ('', '')

                # An explicit prefix naming an available provider is authoritative,
                # even when the model itself is absent from the cache (Azure
                # deployments are frequently not discoverable). An *ambiguous*
                # prefix counts too, and is handed on so _get_provider can ask the
                # caller to disambiguate -- falling through to the cache would
                # silently route an explicit 'azure/...' to whichever provider
                # happens to serve that bare name, possibly not an azure one.
                if rest and self.has_provider_prefix(head):
                    return head, rest

                # Prefix-less, or a prefix whose provider is gone: fall back to the
                # cache. The whole string is tried first -- a bare name may itself
                # contain '/' (e.g. 'meta-llama/Llama-3.1-8B').
                for candidate_name in (model_name, rest):
                    if not candidate_name:
                        continue
                    candidates = self.model_cache.bare_model_candidates(candidate_name)
                    for candidate in candidates:
                        provider_key = candidate.split('/', 1)[0]
                        if provider_key in self.providers:
                            return provider_key, candidate_name

                raise ValueError(
                    f"Model '{model_name}' is not available on any configured provider"
                )
            except Exception as e:
                set_span_error(span, e)
                raise

    def find_provider_key(self, provider_name: str) -> Optional[str]:
        """Resolve `provider_name` to a registered provider key, or None.

        Non-raising counterpart to _get_provider's lookup, for callers that need
        to ask "is this prefix a live provider?" without handling an exception.
        An ambiguous bare prefix returns None -- _get_provider stays the single
        owner of the "ambiguous" / "not available" error messages.
        """
        if not provider_name:
            return None
        if provider_name in self.providers:
            return provider_name
        candidates = [
            full_name for full_name in self.providers.keys()
            if full_name.startswith(f"{provider_name}:")
        ]
        return candidates[0] if len(candidates) == 1 else None

    def has_provider_prefix(self, provider_name: str) -> bool:
        """Whether `provider_name` names a registered provider or provider family.

        Separates "ambiguous bare prefix" from "not a provider at all" -- both
        make find_provider_key return None, but only the latter may fall through
        to bare-name cache resolution. An ambiguous prefix must still be treated
        as an explicit provider choice so the caller is asked to disambiguate
        (_get_provider) rather than being routed somewhere it did not name.
        """
        if not provider_name:
            return False
        if provider_name in self.providers:
            return True
        return any(
            full_name.startswith(f"{provider_name}:") for full_name in self.providers
        )

    def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider by name."""
        with create_span("provider.get_provider") as span:
            try:
                # Direct lookup for full provider names (e.g., "azure:primary", "openai_compatible:my-server")
                if provider_name in self.providers:
                    provider = self.providers[provider_name]
                    return provider

                # Bare-prefix fallback (e.g. "azure" -> "azure:primary"). Only
                # resolve when EXACTLY ONE instance matches; routing to an
                # arbitrary instance (dict-insertion order) is nondeterministic
                # and may pick an instance that doesn't serve the model.
                candidates = [
                    full_name for full_name in self.providers.keys()
                    if full_name.startswith(f"{provider_name}:")
                ]
                if len(candidates) == 1:
                    return self.providers[candidates[0]]
                if len(candidates) > 1:
                    raise ValueError(
                        f"Provider '{provider_name}' is ambiguous; specify the full "
                        f"provider key. Candidates: {', '.join(sorted(candidates))}"
                    )

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
                # Preserve ValueError (bad model name / provider not found) so
                # the route maps it to a 400/404 client error rather than 500.
                # Preserve ProviderHTTPError so the real upstream status/body
                # survive instead of collapsing to a generic 500.
                if isinstance(e, (ValueError, ProviderHTTPError)):
                    raise
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
                if isinstance(e, (ValueError, ProviderHTTPError)):
                    raise
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
                # Send error as SSE format. Preserve status/body for
                # ProviderHTTPError; preserve the message for ValueError (bad
                # model name / provider not found — first-party and
                # client-actionable, not upstream leakage); otherwise emit a
                # sanitized generic error (never str(e), which can leak
                # upstream URLs). Terminate with [DONE] so clients reading
                # until the sentinel don't hang.
                if isinstance(e, ProviderHTTPError):
                    err = e.body.get("error", {}) if isinstance(e.body, dict) else {}
                    error_data = {
                        "error": {
                            "message": err.get("message", "Chat completion stream error"),
                            "type": err.get("type", "upstream_error"),
                            "code": e.status_code,
                        }
                    }
                elif isinstance(e, ValueError):
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    }
                else:
                    error_data = {
                        "error": {
                            "message": "Chat completion stream error",
                            "type": "server_error"
                        }
                    }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
    
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
                # Send sanitized error as SSE + [DONE] (see chat_completion_stream).
                if isinstance(e, ProviderHTTPError):
                    err = e.body.get("error", {}) if isinstance(e.body, dict) else {}
                    error_data = {
                        "error": {
                            "message": err.get("message", "Completion stream error"),
                            "type": err.get("type", "upstream_error"),
                            "code": e.status_code,
                        }
                    }
                elif isinstance(e, ValueError):
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    }
                else:
                    error_data = {
                        "error": {
                            "message": "Completion stream error",
                            "type": "server_error"
                        }
                    }
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
    
    # ==================== RESPONSES API ====================
    
    async def _get_provider_for_response_id(self, response_id: str) -> BaseProvider:
        """Resolve provider for a response_id by looking up the DB mapping."""
        from app.auth.database import AsyncSessionLocal, get_response_provider_mapping
        
        async with AsyncSessionLocal() as db:
            mapping = await get_response_provider_mapping(db, response_id)
        
        if not mapping:
            raise ValueError(f"No provider mapping found for response_id '{response_id}'. The response may have been created before this proxy instance started, or it may not exist.")
        
        return self._get_provider(mapping.provider_key)
    
    async def _store_response_mapping(self, response_id: str, provider_name: str, model_name: str = None, user_id: int = None):
        """Store response_id -> provider mapping in the database."""
        from app.auth.database import AsyncSessionLocal, store_response_provider_mapping

        try:
            async with AsyncSessionLocal() as db:
                await store_response_provider_mapping(db, response_id, provider_name, model_name, user_id=user_id)
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
    
    async def responses_create(self, request: ResponsesCreateRequest, user_id: int = None) -> ResponseObject:
        """Route Responses API create request to appropriate provider."""
        with create_span("provider.responses_create") as span:
            try:
                provider_name, model_id = self._parse_model_name(request.model)
                provider = self._get_provider(provider_name)

                response = await provider.responses_create(request)

                # Store response_id -> provider mapping for future retrieve/delete/cancel
                if response and response.id:
                    await self._store_response_mapping(response.id, provider_name, request.model, user_id=user_id)
                
                return response
            except Exception as e:
                set_span_error(span, e)
                # Preserve typed errors so the route layer can map them to the
                # correct status (ValueError->400, NotImplementedError->501).
                if isinstance(e, (ValueError, NotImplementedError)):
                    raise
                raise Exception(f"Responses create error: {str(e)}")

    async def responses_create_stream(self, request: ResponsesCreateRequest, user_id: int = None) -> AsyncGenerator[str, None]:
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
                                            await self._store_response_mapping(resp_id, provider_name, request.model, user_id=user_id)
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
                # Preserve status/body for ProviderHTTPError and the message
                # for ValueError (first-party, client-actionable); otherwise
                # emit a sanitized generic error — never raw str(e), which can
                # leak upstream URLs/bodies (see chat_completion_stream).
                if isinstance(e, ProviderHTTPError):
                    err = e.body.get("error", {}) if isinstance(e.body, dict) else {}
                    error_data = {
                        "error": {
                            "message": err.get("message", "Responses stream error"),
                            "type": err.get("type", "upstream_error"),
                            "code": e.status_code,
                        }
                    }
                elif isinstance(e, ValueError):
                    error_data = {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    }
                else:
                    error_data = {
                        "error": {
                            "message": "Responses stream error",
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
                if isinstance(e, ValueError):
                    raise
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
                if isinstance(e, ValueError):
                    raise
                raise Exception(f"Responses delete error: {str(e)}")
    
    async def responses_cancel(self, response_id: str) -> ResponseObject:
        """Route Responses API cancel request to appropriate provider."""
        with create_span("provider.responses_cancel") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                return await provider.responses_cancel(response_id)
            except Exception as e:
                set_span_error(span, e)
                if isinstance(e, ValueError):
                    raise
                raise Exception(f"Responses cancel error: {str(e)}")
    
    async def responses_list_input_items(self, response_id: str, **kwargs) -> ResponseItemList:
        """Route Responses API list input items request to appropriate provider."""
        with create_span("provider.responses_list_input_items") as span:
            try:
                provider = await self._get_provider_for_response_id(response_id)
                return await provider.responses_list_input_items(response_id, **kwargs)
            except Exception as e:
                set_span_error(span, e)
                if isinstance(e, ValueError):
                    raise
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
                if isinstance(e, ValueError):
                    raise
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
                if isinstance(e, ValueError):
                    raise
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

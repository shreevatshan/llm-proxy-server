import asyncio
import threading
import time
import logging
import os
from typing import List, Optional, Dict, Set
from opentelemetry import context as otel_context
from app.openai_models import ModelInfo
from app.tracing import create_span, add_span_attributes, set_span_error, safe_detach

logger = logging.getLogger(__name__)


class ModelCache:
    """Async-safe cache for model lists with snapshot-based reads to minimize lock contention.
    
    This implementation uses copy-on-write semantics for reads to avoid blocking concurrent
    requests while still maintaining data consistency for writes.
    
    Uses asyncio.Lock for async-safe locking instead of threading.RLock to prevent
    potential deadlocks when mixing async and sync code.
    """
    
    # Refresh interval in seconds (configurable, default 1 minute)
    REFRESH_INTERVAL = int(os.getenv("MODEL_CACHE_REFRESH_INTERVAL", "300"))
    
    def __init__(self):
        self._models: List[ModelInfo] = []
        self._last_updated: float = 0
        # Use asyncio.Lock for async-safe writes (prevents blocking event loop)
        self._write_lock: Optional[asyncio.Lock] = None
        self._provider_manager = None
        
        # Model configuration cache (use separate lock for config to reduce contention)
        self._model_configs: Dict[str, bool] = {}  # model_id -> enabled status
        self._provider_configs: Dict[str, bool] = {}  # provider_key -> enabled status
        self._config_last_updated: float = 0
        self._config_lock: Optional[asyncio.Lock] = None

        # Per-user model access cache (guarded by _config_lock)
        self._user_model_policies: Dict[int, str] = {}  # user_id -> mode (absent -> "default")
        self._user_model_exceptions: Dict[int, Dict[str, bool]] = {}  # user_id -> {model_id -> is_allowed}
        
        # Threading lock for safe initialization of asyncio locks
        self._init_lock = threading.Lock()
        
        # Background refresh task management
        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False
        self._refresh_in_progress = False  # Simple flag to prevent concurrent refreshes
    
    def _ensure_locks(self):
        """Ensure asyncio locks are initialized (must be called from async context).
        
        Uses double-checked locking pattern with threading lock for thread-safety.
        """
        if self._write_lock is None or self._config_lock is None:
            with self._init_lock:
                if self._write_lock is None:
                    self._write_lock = asyncio.Lock()
                if self._config_lock is None:
                    self._config_lock = asyncio.Lock()
    
    def set_provider_manager(self, provider_manager):
        """Set the provider manager reference for cache refresh."""
        self._provider_manager = provider_manager
    
    async def start(self):
        """Start the background periodic refresh task."""
        if self._running:
            return
        self._running = True
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("ModelCache started with %d second refresh interval", self.REFRESH_INTERVAL)
    
    async def stop(self):
        """Stop the background refresh task gracefully."""
        logger.info("Stopping ModelCache periodic refresh...")
        self._running = False
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info("ModelCache periodic refresh stopped")
    
    async def _refresh_loop(self):
        """Background loop to refresh model cache every REFRESH_INTERVAL seconds."""
        while self._running:
            try:
                await asyncio.sleep(self.REFRESH_INTERVAL)
                if self._running:  # Check again after sleep
                    # Detach from parent trace context so each refresh gets its own trace
                    token = otel_context.attach(otel_context.Context())
                    try:
                        await self._periodic_refresh()
                    finally:
                        safe_detach(token)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in model cache refresh loop: %s", e)
    
    async def _periodic_refresh(self):
        """Perform periodic refresh of models from all providers."""
        # Skip if a refresh is already in progress (non-blocking check)
        if self._refresh_in_progress:
            logger.debug("Skipping periodic refresh - already in progress")
            return
        
        self._refresh_in_progress = True
        with create_span(
            "model_cache.periodic_refresh",
            attributes={
                "model_cache.refresh_interval": self.REFRESH_INTERVAL,
                "model_cache.provider_count": len(self._provider_manager.providers) if self._provider_manager else 0
            }
        ) as span:
            try:
                if self._provider_manager:
                    logger.info("Starting periodic model refresh...")
                    
                    # Capture pre-refresh state (snapshot read - no lock needed)
                    pre_refresh_count = len(self._models)
                    add_span_attributes(span, {
                        "model_cache.pre_refresh_model_count": pre_refresh_count
                    })
                    
                    # Use the existing fetch and sync method that updates both cache and DB
                    await self._provider_manager._fetch_and_sync_all_models()
                    
                    # Capture post-refresh state
                    post_refresh_count = len(self._models)
                    add_span_attributes(span, {
                        "model_cache.post_refresh_model_count": post_refresh_count,
                        "model_cache.models_changed": post_refresh_count - pre_refresh_count,
                        "model_cache.status": "success"
                    })
                    
                    logger.info("Periodic model refresh completed. Models: %d -> %d", pre_refresh_count, post_refresh_count)
                else:
                    logger.warning("No provider manager set, skipping periodic refresh")
                    add_span_attributes(span, {
                        "model_cache.status": "skipped",
                        "model_cache.skip_reason": "no_provider_manager"
                    })
            except Exception as e:
                logger.error("Error during periodic model refresh: %s", e)
                set_span_error(span, e)
            finally:
                self._refresh_in_progress = False

    def get_models(self) -> List[ModelInfo]:
        """Get cached models (returns snapshot - no lock held during iteration)."""
        # Return a copy to prevent modification during iteration
        return list(self._models)
    
    def update_models(self, models: List[ModelInfo]) -> None:
        """Update cached models (atomic assignment - thread-safe for simple reference swap)."""
        # Atomic reference swap is thread-safe in Python
        self._models = models
        self._last_updated = time.time()
    
    async def update_models_async(self, models: List[ModelInfo]) -> None:
        """Update cached models with async lock (for use in async context)."""
        self._ensure_locks()
        async with self._write_lock:
            self._models = models
            self._last_updated = time.time()
    
    def get_cache_age(self) -> float:
        """Get cache age in seconds."""
        return time.time() - self._last_updated
    
    def warm_cache(self, models: List[ModelInfo]) -> None:
        """Warm the cache with initial model data (used during startup)."""
        # Atomic reference swap is thread-safe in Python
        self._models = models
        self._last_updated = time.time()
        print(f"Cache warmed with {len(models)} models")
    
    async def warm_cache_async(self, models: List[ModelInfo]) -> None:
        """Warm the cache with initial model data (async version)."""
        self._ensure_locks()
        async with self._write_lock:
            self._models = models
            self._last_updated = time.time()
            print(f"Cache warmed with {len(models)} models")
    
    def invalidate_model(self, model_id: str) -> None:
        """Remove a specific model from cache (sync version for backward compatibility)."""
        # Create new list and do atomic reference swap
        new_models = [model for model in self._models if model.id != model_id]
        self._models = new_models
        self._last_updated = time.time()
        print(f"Invalidated model from cache: {model_id}")
    
    async def invalidate_model_async(self, model_id: str) -> None:
        """Remove a specific model from cache (async version)."""
        self._ensure_locks()
        async with self._write_lock:
            self._models = [model for model in self._models if model.id != model_id]
            self._last_updated = time.time()
            print(f"Invalidated model from cache: {model_id}")
    
    def invalidate_provider(self, provider_key: str) -> None:
        """Remove all models for a specific provider from cache (sync version)."""
        # Create new list and do atomic reference swap
        initial_count = len(self._models)
        new_models = [model for model in self._models if not model.id.startswith(f"{provider_key}/")]
        removed_count = initial_count - len(new_models)
        self._models = new_models
        self._last_updated = time.time()
        print(f"Invalidated {removed_count} models for provider: {provider_key}")
    
    async def invalidate_provider_async(self, provider_key: str) -> None:
        """Remove all models for a specific provider from cache (async version)."""
        self._ensure_locks()
        async with self._write_lock:
            initial_count = len(self._models)
            self._models = [model for model in self._models if not model.id.startswith(f"{provider_key}/")]
            removed_count = initial_count - len(self._models)
            self._last_updated = time.time()
            print(f"Invalidated {removed_count} models for provider: {provider_key}")
    
    async def refresh_cache_from_database(self) -> None:
        """Refresh entire cache by fetching fresh data from providers."""
        if self._provider_manager:
            try:
                print("Refreshing cache from database...")
                models = await self._provider_manager._fetch_all_models()
                self.update_models(models)
                print(f"Cache refreshed with {len(models)} models")
            except Exception as e:
                print(f"Error refreshing cache from database: {e}")
    
    # Model Configuration Methods
    
    def update_model_configurations(self, model_configs: Dict[str, bool], provider_configs: Dict[str, bool]) -> None:
        """Update model and provider configurations (atomic assignment - thread-safe)."""
        # Atomic reference swaps are thread-safe in Python
        self._model_configs = model_configs
        self._provider_configs = provider_configs
        self._config_last_updated = time.time()
    
    async def update_model_configurations_async(self, model_configs: Dict[str, bool], provider_configs: Dict[str, bool]) -> None:
        """Update model and provider configurations (async version with lock)."""
        self._ensure_locks()
        async with self._config_lock:
            self._model_configs = model_configs
            self._provider_configs = provider_configs
            self._config_last_updated = time.time()
    
    def is_model_enabled(self, model_id: str) -> bool:
        """Check if a model is enabled (default True if not in config)."""
        # Snapshot read - no lock needed for simple dict lookup
        return self._model_configs.get(model_id, True)
    
    def is_provider_enabled(self, provider_key: str) -> bool:
        """Check if a provider is enabled (default True if not in config)."""
        # Snapshot read - no lock needed for simple dict lookup
        return self._provider_configs.get(provider_key, True)
    
    def update_single_model_config(self, model_id: str, enabled: bool) -> None:
        """Update single model configuration (sync version - uses dict mutation)."""
        # Dict item assignment is atomic in Python for simple keys
        self._model_configs[model_id] = enabled
        self._config_last_updated = time.time()
    
    async def update_single_model_config_async(self, model_id: str, enabled: bool) -> None:
        """Update single model configuration (async version with lock)."""
        self._ensure_locks()
        async with self._config_lock:
            self._model_configs[model_id] = enabled
            self._config_last_updated = time.time()
    
    def update_single_provider_config(self, provider_key: str, enabled: bool) -> None:
        """Update single provider configuration (sync version - uses dict mutation)."""
        # Dict item assignment is atomic in Python for simple keys
        self._provider_configs[provider_key] = enabled
        self._config_last_updated = time.time()
    
    async def update_single_provider_config_async(self, provider_key: str, enabled: bool) -> None:
        """Update single provider configuration (async version with lock)."""
        self._ensure_locks()
        async with self._config_lock:
            self._provider_configs[provider_key] = enabled
            self._config_last_updated = time.time()
    
    def update_provider_and_models_config(self, provider_key: str, enabled: bool) -> None:
        """Update provider configuration and all its models (sync version)."""
        # Update provider config (atomic)
        self._provider_configs[provider_key] = enabled
        
        # Get snapshot of models for iteration
        models_snapshot = list(self._models)
        
        # Update all models belonging to this provider (each dict assignment is atomic)
        for model in models_snapshot:
            if '/' in model.id:
                model_provider_key = model.id.split('/', 1)[0]
                if model_provider_key == provider_key:
                    self._model_configs[model.id] = enabled
        
        self._config_last_updated = time.time()
        print(f"Updated provider {provider_key} and its models to {'enabled' if enabled else 'disabled'}")
    
    async def update_provider_and_models_config_async(self, provider_key: str, enabled: bool) -> None:
        """Update provider configuration and all its models (async version with lock)."""
        self._ensure_locks()
        async with self._config_lock:
            # Update provider config
            self._provider_configs[provider_key] = enabled
            
            # Get snapshot of models for iteration
            models_snapshot = list(self._models)
            
            # Update all models belonging to this provider
            for model in models_snapshot:
                if '/' in model.id:
                    model_provider_key = model.id.split('/', 1)[0]
                    if model_provider_key == provider_key:
                        self._model_configs[model.id] = enabled
            
            self._config_last_updated = time.time()
            print(f"Updated provider {provider_key} and its models to {'enabled' if enabled else 'disabled'}")
    
    def get_enabled_models(self) -> List[ModelInfo]:
        """Get only enabled models from cache (snapshot-based, minimal lock time)."""
        # Take snapshots outside of heavy computation
        models_snapshot = list(self._models)
        model_configs_snapshot = dict(self._model_configs)
        provider_configs_snapshot = dict(self._provider_configs)
        
        # Filter using snapshots (no locks held during filtering)
        enabled_models = []
        for model in models_snapshot:
            # Check if model is enabled
            model_enabled = model_configs_snapshot.get(model.id, True)
            if model_enabled:
                # Extract provider key from model id
                if '/' in model.id:
                    provider_key = model.id.split('/', 1)[0]
                    provider_enabled = provider_configs_snapshot.get(provider_key, True)
                    if provider_enabled:
                        enabled_models.append(model)
                else:
                    enabled_models.append(model)
        return enabled_models
    
    def filter_enabled_models(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Filter out disabled models from a list (snapshot-based, minimal lock time)."""
        # Take snapshots
        model_configs_snapshot = dict(self._model_configs)
        provider_configs_snapshot = dict(self._provider_configs)
        
        # Filter using snapshots (no locks held during filtering)
        enabled_models = []
        for model in models:
            # Check if model is enabled
            model_enabled = model_configs_snapshot.get(model.id, True)
            if model_enabled:
                # Extract provider key from model id
                if '/' in model.id:
                    provider_key = model.id.split('/', 1)[0]
                    provider_enabled = provider_configs_snapshot.get(provider_key, True)
                    if provider_enabled:
                        enabled_models.append(model)
                else:
                    enabled_models.append(model)
        return enabled_models
    
    # ==================== Per-User Model Access ====================

    def update_user_model_access(
        self,
        user_policies: Dict[int, str],
        user_exceptions: Dict[int, Dict[str, bool]],
    ) -> None:
        """Replace the full per-user access maps (sync; used on refresh).

        user_policies maps user_id -> mode ("default"|"allow"|"deny"|"custom").
        """
        self._user_model_policies = user_policies
        self._user_model_exceptions = user_exceptions
        self._config_last_updated = time.time()

    def set_user_model_mode(self, user_id: int, mode: str) -> None:
        """Update a single user's access mode in place (atomic dict assignment)."""
        self._user_model_policies[user_id] = mode
        self._config_last_updated = time.time()

    def update_user_model_access_for_user(
        self, user_id: int, mode: str, overrides: Dict[str, bool]
    ) -> None:
        """Replace a single user's mode and full override map (used when seeding
        a fresh custom baseline from allow/deny)."""
        self._user_model_policies[user_id] = mode
        self._user_model_exceptions[user_id] = dict(overrides)
        self._config_last_updated = time.time()

    def set_user_model_exception(self, user_id: int, model_id: str, is_allowed: bool) -> None:
        """Set/replace a single per-model exception for a user."""
        user_map = self._user_model_exceptions.get(user_id)
        if user_map is None:
            user_map = {}
            self._user_model_exceptions[user_id] = user_map
        user_map[model_id] = is_allowed
        self._config_last_updated = time.time()

    def clear_user_model_exception(self, user_id: int, model_id: str) -> None:
        """Remove a single per-model exception for a user (revert to policy default)."""
        user_map = self._user_model_exceptions.get(user_id)
        if user_map is not None:
            user_map.pop(model_id, None)
        self._config_last_updated = time.time()

    def clear_user_model_exceptions(self, user_id: int) -> None:
        """Remove all per-model exceptions for a user (bulk allow-all/deny-all)."""
        self._user_model_exceptions.pop(user_id, None)
        self._config_last_updated = time.time()

    def _passes_global_gate(self, model_id: str) -> bool:
        """True if the model and its provider are globally enabled."""
        if not self.is_model_enabled(model_id):
            return False
        if '/' in model_id:
            provider_key = model_id.split('/', 1)[0]
            if not self.is_provider_enabled(provider_key):
                return False
        return True

    def is_model_allowed_for_user(self, user_id: int, model_id: str) -> bool:
        """Effective per-user access decision for a model, by mode:

          default : follow the global gate only.
          allow   : always allowed (overrides the global gate).
          deny    : always denied (overrides the global gate).
          custom  : explicit override wins (beats the gate); otherwise follow global.
        """
        mode = self._user_model_policies.get(user_id, "default")

        if mode == "allow":
            return True
        if mode == "deny":
            return False
        if mode == "custom":
            user_map = self._user_model_exceptions.get(user_id)
            if user_map is not None and model_id in user_map:
                return user_map[model_id]
            return self._passes_global_gate(model_id)

        # "default" (or any unknown mode) -> obey global config.
        return self._passes_global_gate(model_id)

    def get_enabled_models_for_user(self, user_id: int) -> List[ModelInfo]:
        """All models filtered by the user's effective access (may include
        globally-disabled models when the user's mode overrides the gate)."""
        models_snapshot = list(self._models)
        return [m for m in models_snapshot if self.is_model_allowed_for_user(user_id, m.id)]

    def get_config_cache_age(self) -> float:
        """Get configuration cache age in seconds."""
        return time.time() - self._config_last_updated

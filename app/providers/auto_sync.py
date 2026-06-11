"""
Automatic model synchronization for providers.

This module provides functionality to automatically sync models whenever
a provider is created or edited, with special handling for Azure providers
that use deployment names instead of dynamic model discovery.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.database import (
    get_provider_credentials, 
    create_or_update_model_configuration,
    get_models_by_provider,
    get_model_configuration
)
from app.providers.provider_manager import provider_manager
from app.openai_models import ModelInfo
from app.providers.azure_deployments import build_azure_config_fields, merge_azure_deployments, normalize_azure_deployments
from app.tracing import (
    create_span,
    add_span_attributes,
    set_span_error
)


async def clear_models_for_provider(db: AsyncSession, provider_key: str) -> int:
    """Clear all models for a specific provider."""
    try:
        models = await get_models_by_provider(db, provider_key)
        count = len(models)
        
        for model in models:
            await db.delete(model)
        
        await db.commit()
        return count
    except Exception as e:
        await db.rollback()
        raise e


async def sync_provider_models(db: AsyncSession, provider_key: str) -> Dict[str, Any]:
    """
    Automatically sync models for a specific provider.
    
    Args:
        db: Database session
        provider_key: Provider key (e.g., "azure:primary", "ollama:local")
        
    Returns:
        Dict with sync results including counts and any errors
    """
    with create_span(
        "auto_sync.sync_provider_models",
        attributes={
            "provider.key": provider_key
        }
    ) as span:
        try:
            # Get provider credentials from database
            provider_creds = await get_provider_credentials(db, provider_key)
            if not provider_creds:
                error_msg = f"Provider not found: {provider_key}"
                add_span_attributes(span, {"error": error_msg})
                return {"error": error_msg}
            
            add_span_attributes(span, {
                "provider.type": provider_creds.provider_type
            })
            
            # Handle different provider types
            if provider_creds.provider_type == "azure":
                result = await _sync_azure_provider_models(db, provider_creds)
            else:
                result = await _sync_dynamic_provider_models(db, provider_creds)
            
            # Add result metrics to span
            if "error" in result:
                set_span_error(span, result["error"])
            else:
                add_span_attributes(span, {
                    "provider.models_cleared": result.get("cleared", 0),
                    "provider.models_created": result.get("created", 0),
                    "provider.sync_status": "success"
                })
            
            return result
                
        except Exception as e:
            error_msg = f"Failed to sync models for {provider_key}: {str(e)}"
            set_span_error(span, e)
            return {"error": error_msg}


async def _sync_azure_provider_models(db: AsyncSession, provider_creds) -> Dict[str, Any]:
    """
    Sync models for Azure provider using deployment names.
    
    For Azure providers, we use deployment names from the configuration
    instead of dynamic model discovery, as Azure doesn't expose models
    through a standard API.
    """
    provider_key = provider_creds.provider_key
    
    with create_span(
        "auto_sync.sync_azure_provider",
        attributes={
            "provider.key": provider_key,
            "provider.type": "azure"
        }
    ) as span:
        try:
            # Capture enabled states BEFORE clearing models
            existing_models = await get_models_by_provider(db, provider_key)
            enabled_states = {model.model_id: model.is_enabled for model in existing_models}
            
            # Clear existing models for this provider
            cleared_count = await clear_models_for_provider(db, provider_key)
            
            add_span_attributes(span, {
                "provider.models_cleared": cleared_count
            })
            
            deployment_groups = normalize_azure_deployments(provider_creds.deployments_json)
            azure_backend = getattr(provider_creds, "azure_backend", None) or "openai"
            deployments = merge_azure_deployments(
                deployment_groups,
                include_anthropic=azure_backend == "foundry",
            )

            dynamic_discovery = provider_creds.dynamic_discovery
            if dynamic_discovery is None:
                dynamic_discovery = not bool(deployments)

            if dynamic_discovery:
                provider_instance = await _create_provider_instance(provider_creds)
                if not provider_instance:
                    error_msg = f"Failed to create Azure provider instance for {provider_key}"
                    set_span_error(span, error_msg)
                    return {"error": error_msg}

                timeout = 180
                try:
                    models = await asyncio.wait_for(
                        provider_instance.get_available_models(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    error_msg = f"Timeout ({timeout}s) fetching models from {provider_key}. Provider may be slow or unavailable."
                    set_span_error(span, error_msg)
                    return {"error": error_msg}
                except Exception as e:
                    error_msg = f"Failed to fetch models from {provider_key}: {str(e)}"
                    set_span_error(span, e)
                    return {"error": error_msg}

                deployments = []
                for model in models or []:
                    model_name = model.id.split("/", 1)[1] if "/" in model.id else model.id
                    if model_name not in deployments:
                        deployments.append(model_name)

            if not deployments:
                add_span_attributes(span, {
                    "provider.deployments_count": 0,
                    "provider.status": "no_deployments"
                })
                return {
                    "provider_key": provider_key,
                    "provider_type": "azure",
                    "cleared": cleared_count,
                    "created": 0,
                    "message": "No Azure models discovered or configured for this provider"
                }
            
            add_span_attributes(span, {
                "provider.deployments_count": len(deployments)
            })
            
            # Create model configurations for each deployment
            created_count = 0
            for deployment_name in deployments:
                # Create model ID in the format: provider_key/deployment_name
                model_id = f"{provider_key}/{deployment_name}"
                
                # Use cached enabled state (before models were cleared)
                is_enabled = enabled_states.get(model_id, True)  # Default True for new models
                
                await create_or_update_model_configuration(
                    db=db,
                    model_id=model_id,
                    provider_key=provider_key,
                    model_name=deployment_name,
                    is_enabled=is_enabled
                )
                created_count += 1
            
            add_span_attributes(span, {
                "provider.models_created": created_count,
                "provider.sync_status": "success"
            })
            
            return {
                "provider_key": provider_key,
                "provider_type": "azure",
                "cleared": cleared_count,
                "created": created_count,
                "models": [f"{provider_key}/{deployment}" for deployment in deployments],
                "deployments": deployments,
                "deployment_groups": {
                    "openai": deployment_groups.get("openai", []),
                    "anthropic": deployment_groups.get("anthropic", []),
                },
                "message": f"Successfully synced {created_count} Azure models"
            }
            
        except Exception as e:
            await db.rollback()
            error_msg = f"Failed to sync Azure provider models: {str(e)}"
            set_span_error(span, e)
            return {"error": error_msg}


async def _sync_dynamic_provider_models(db: AsyncSession, provider_creds) -> Dict[str, Any]:
    """
    Sync models for providers that support dynamic model discovery.
    
    This includes providers like Ollama, OpenAI, Google, Bedrock that
    can dynamically discover available models through their APIs.
    """
    provider_key = provider_creds.provider_key
    
    with create_span(
        "auto_sync.sync_dynamic_provider",
        attributes={
            "provider.key": provider_key,
            "provider.type": provider_creds.provider_type
        }
    ) as span:
        try:
            # Capture enabled states BEFORE clearing models
            existing_models = await get_models_by_provider(db, provider_key)
            enabled_states = {model.model_id: model.is_enabled for model in existing_models}
            
            # Clear existing models for this provider
            cleared_count = await clear_models_for_provider(db, provider_key)
            
            add_span_attributes(span, {
                "provider.models_cleared": cleared_count
            })
            
            # Create provider instance for model discovery
            provider_instance = await _create_provider_instance(provider_creds)
            if not provider_instance:
                error_msg = f"Failed to create provider instance for {provider_key}"
                set_span_error(span, error_msg)
                return {"error": error_msg}
            
            # Use 3 minutes timeout for all providers
            timeout = 180
            
            add_span_attributes(span, {
                "provider.fetch_timeout_seconds": timeout
            })
            
            # Fetch models from the provider
            try:
                models = await asyncio.wait_for(
                    provider_instance.get_available_models(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                error_msg = f"Timeout ({timeout}s) fetching models from {provider_key}. Provider may be slow or unavailable."
                set_span_error(span, error_msg)
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"Failed to fetch models from {provider_key}: {str(e)}"
                set_span_error(span, e)
                return {"error": error_msg}
            
            if not models:
                add_span_attributes(span, {
                    "provider.models_fetched": 0,
                    "provider.status": "no_models"
                })
                return {
                    "provider_key": provider_key,
                    "provider_type": provider_creds.provider_type,
                    "cleared": cleared_count,
                    "created": 0,
                    "message": "No models available from provider"
                }
            
            add_span_attributes(span, {
                "provider.models_fetched": len(models)
            })
            
            # Create model configurations
            created_count = 0
            for model in models:
                # Extract model name from full model ID
                if '/' in model.id:
                    model_name = model.id.split('/', 1)[1]
                else:
                    model_name = model.id
                
                # Use cached enabled state (before models were cleared)
                is_enabled = enabled_states.get(model.id, True)  # Default True for new models
                
                await create_or_update_model_configuration(
                    db=db,
                    model_id=model.id,
                    provider_key=provider_key,
                    model_name=model_name,
                    is_enabled=is_enabled
                )
                created_count += 1
            
            add_span_attributes(span, {
                "provider.models_created": created_count,
                "provider.sync_status": "success"
            })
            
            return {
                "provider_key": provider_key,
                "provider_type": provider_creds.provider_type,
                "cleared": cleared_count,
                "created": created_count,
                "models": [model.id for model in models],
                "message": f"Successfully synced {created_count} models"
            }
            
        except Exception as e:
            await db.rollback()
            error_msg = f"Failed to sync dynamic provider models: {str(e)}"
            set_span_error(span, e)
            return {"error": error_msg}


async def _create_provider_instance(provider_creds):
    """
    Create a provider instance from credentials for model discovery.
    """
    try:
        # Import provider classes
        from app.providers.azure_provider import AzureProvider
        from app.providers.bedrock_provider import BedrockProvider
        from app.providers.google_provider import GoogleProvider
        from app.providers.custom_providers import create_custom_provider
        
        # Map specific provider types to their implementations
        specialized_providers = {
            'azure': AzureProvider,
            'bedrock': BedrockProvider,
            'google': GoogleProvider,
        }
        
        # Create provider configuration
        config = _create_provider_config_from_creds(provider_creds)
        
        # Use specialized provider if available, otherwise use custom provider
        if provider_creds.provider_type in specialized_providers:
            provider_class = specialized_providers[provider_creds.provider_type]
            return provider_class(config)
        else:
            # All other providers (custom / openai_compatible) 
            return create_custom_provider(config)
        
    except Exception as e:
        print(f"Error creating provider instance: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_provider_config_from_creds(creds) -> Dict[str, Any]:
    """Create provider configuration dict from database credentials."""
    config_dict = {
        'name': creds.instance_name,
        'enabled': creds.enabled
    }
    
    # Add provider_name for OpenAI-compatible providers
    if hasattr(creds, 'provider_name') and creds.provider_name:
        config_dict['provider_name'] = creds.provider_name
        config_dict['custom_provider_name'] = creds.provider_name  # For backward compatibility
    
    # Add provider-specific fields based on type
    if creds.provider_type == 'azure':
        config_dict.update(build_azure_config_fields(creds))
    elif creds.provider_type == 'openai':
        config_dict.update({
            'endpoint': creds.endpoint or 'https://api.openai.com/v1',
            'api_key': creds.api_key
        })
    elif creds.provider_type == 'google':
        config_dict.update({
            'api_key': creds.api_key,
            'base_url': creds.base_url,
        })
    elif creds.provider_type == 'bedrock':
        config_dict.update({
            'region': creds.region or 'us-west-2',
            'access_key_id': creds.access_key_id,
            'secret_access_key': creds.secret_access_key
        })
    else:
        # All other providers are custom (including custom, ollama, etc.)
        config_dict.update({
            'base_url': creds.base_url or creds.endpoint,
            'api_key': creds.api_key
        })
    
    # Add supported_apis if available (for custom providers)
    if hasattr(creds, 'supported_apis') and creds.supported_apis:
        try:
            config_dict['supported_apis'] = json.loads(creds.supported_apis)
        except (json.JSONDecodeError, TypeError):
            config_dict['supported_apis'] = ['openai']
    
    return config_dict


async def sync_all_provider_models(db: AsyncSession) -> Dict[str, Any]:
    """
    Sync models for all enabled providers.
    
    Returns:
        Dict with overall sync results
    """
    with create_span("auto_sync.sync_all_providers") as span:
        try:
            from app.auth.database import get_all_provider_credentials
            
            # Get all provider credentials
            all_providers = await get_all_provider_credentials(db)
            enabled_providers = [p for p in all_providers if p.enabled]
            
            add_span_attributes(span, {
                "provider.total_count": len(all_providers),
                "provider.enabled_count": len(enabled_providers)
            })
            
            if not enabled_providers:
                add_span_attributes(span, {
                    "provider.sync_status": "no_providers"
                })
                return {
                    "message": "No enabled providers found",
                    "total_providers": 0,
                    "synced_providers": 0,
                    "results": []
                }
            
            # Sync each provider
            results = []
            synced_count = 0
            failed_providers = []
            
            for provider in enabled_providers:
                result = await sync_provider_models(db, provider.provider_key)
                results.append(result)
                
                if "error" not in result:
                    synced_count += 1
                else:
                    failed_providers.append(provider.provider_key)
            
            # Update provider manager cache
            try:
                await provider_manager.refresh_model_configurations()
            except Exception as e:
                print(f"Warning: Failed to refresh provider manager cache: {e}")
            
            add_span_attributes(span, {
                "provider.synced_count": synced_count,
                "provider.failed_count": len(failed_providers),
                "provider.sync_status": "success" if synced_count > 0 else "all_failed"
            })
            
            if failed_providers:
                add_span_attributes(span, {
                    "provider.failed_names": ",".join(failed_providers)
                })
            
            return {
                "message": f"Synced models for {synced_count}/{len(enabled_providers)} providers",
                "total_providers": len(enabled_providers),
                "synced_providers": synced_count,
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Failed to sync all provider models: {str(e)}"
            set_span_error(span, e)
            return {"error": error_msg}


async def auto_sync_on_provider_change(db: AsyncSession, provider_key: str, action: str = "update") -> Dict[str, Any]:
    """
    Automatically sync models when a provider is created or updated.
    
    Args:
        db: Database session
        provider_key: Provider key that was changed
        action: Type of action ("create", "update", "delete")
        
    Returns:
        Dict with sync results
    """
    with create_span(
        "auto_sync.on_provider_change",
        attributes={
            "provider.key": provider_key,
            "provider.action": action
        }
    ) as span:
        try:
            if action == "delete":
                # For deletions, just clear models (they should be cleaned up by cascade)
                add_span_attributes(span, {
                    "provider.sync_status": "deleted"
                })
                return {
                    "provider_key": provider_key,
                    "action": action,
                    "message": "Provider deleted, models cleaned up"
                }
            
            # For create and update, force a complete provider manager refresh first
            # This is crucial for handling provider renames
            try:
                print(f"Auto-sync: Forcing provider manager refresh for {action} of {provider_key}")
                await provider_manager.refresh_providers_from_database()
                
                # For updates (which might be renames), wait a moment for the refresh to complete
                if action == "update":
                    import asyncio
                    await asyncio.sleep(0.1)  # Small delay to ensure refresh completes
                    
            except Exception as e:
                print(f"Warning: Provider manager refresh failed during auto-sync: {e}")
                add_span_attributes(span, {
                    "provider.refresh_error": str(e)
                })
            
            # For create and update, sync the models
            result = await sync_provider_models(db, provider_key)
            result["action"] = action
            
            print(f"🔍 Sync result for {provider_key}: {result}")
            
            # Mark provider as synced to avoid re-syncing on startup
            if "error" not in result:
                provider_manager.mark_provider_synced(provider_key)
            
            # Update provider manager cache if sync was successful
            if "error" not in result:
                try:
                    # Get the models that were just synced from the result
                    model_ids = result.get("models", [])
                    
                    print(f"🔍 Model IDs from sync result: {model_ids}")
                    
                    if model_ids:
                        from app.openai_models import ModelInfo
                        import time
                        
                        synced_models = []
                        for model_id in model_ids:
                            synced_models.append(ModelInfo(
                                id=model_id,
                                object="model",
                                created=int(time.time()),
                                owned_by=provider_key,
                                provider=provider_key
                            ))
                        
                        print(f"🔍 Created {len(synced_models)} ModelInfo objects")
                        
                        # Update cache by adding these models to existing ones
                        existing_models = provider_manager.model_cache.get_models()
                        print(f"🔍 Existing models in cache before update: {len(existing_models)}")
                        
                        # Remove old models from this provider
                        other_provider_models = [m for m in existing_models if not m.id.startswith(f"{provider_key}/")]
                        print(f"🔍 Models from other providers: {len(other_provider_models)}")
                        
                        # Add new models from this provider
                        updated_models = other_provider_models + synced_models
                        print(f"🔍 Total models after combining: {len(updated_models)}")
                        
                        provider_manager.model_cache.update_models(updated_models)
                        print(f"🔍 Cache updated successfully")
                        
                        # Verify cache was updated
                        cache_models_after = provider_manager.model_cache.get_models()
                        print(f"🔍 Models in cache after update: {len(cache_models_after)}")
                        
                        # Refresh providers from database (reloads provider instances and configs)
                        await provider_manager.refresh_providers_from_database()
                        print(f"🔍 Provider manager refreshed from database")
                        
                        # Verify cache after refresh
                        cache_models_final = provider_manager.model_cache.get_models()
                        print(f"🔍 Models in cache after refresh: {len(cache_models_final)}")
                        
                        result["cache_refreshed"] = True
                        result["models_in_cache"] = len(cache_models_final)
                        
                        add_span_attributes(span, {
                            "provider.models_synced": len(synced_models),
                            "provider.cache_refreshed": True,
                            "provider.total_cache_models": len(cache_models_final),
                            "provider.sync_status": "success"
                        })
                        
                        print(f"✅ Auto-sync completed for {provider_key}: {result.get('message', 'Success')}")
                        print(f"   Cache has {len(cache_models_final)} total models ({len(synced_models)} from this provider)")
                    else:
                        print(f"⚠️  Auto-sync for {provider_key} completed but no models were returned")
                        result["cache_refreshed"] = False
                        result["warning"] = "No models returned from sync"
                        
                        add_span_attributes(span, {
                            "provider.models_synced": 0,
                            "provider.cache_refreshed": False,
                            "provider.sync_status": "no_models"
                        })
                        
                except Exception as e:
                    result["cache_refresh_error"] = str(e)
                    print(f"❌ Warning: Cache refresh failed after auto-sync: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    add_span_attributes(span, {
                        "provider.cache_refresh_error": str(e)
                    })
            
            return result
            
        except Exception as e:
            error_msg = f"Auto-sync failed: {str(e)}"
            set_span_error(span, e)
            return {
                "provider_key": provider_key,
                "action": action,
                "error": error_msg
            }

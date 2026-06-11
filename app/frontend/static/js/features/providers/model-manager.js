/**
 * Model Management Module
 * Handles model operations, toggles, and bulk actions
 */


class ModelManager {
    async toggleProviderModels(providerKey) {
        const modelsSection = document.getElementById(`models-section-${providerKey}`);
        const modelsBtn = document.getElementById(`models-btn-${providerKey}`);
        const btnIcon = modelsBtn.querySelector('i');

        if (modelsSection.style.display === 'none') {
            // Show models section
            modelsSection.style.display = 'block';
            btnIcon.classList.remove('fa-list');
            btnIcon.classList.add('fa-list-ul');
            btnIcon.classList.add('expanded');

            // Load models if not already loaded
            await this.loadProviderModels(providerKey);
        } else {
            // Hide models section
            modelsSection.style.display = 'none';
            btnIcon.classList.remove('fa-list-ul', 'expanded');
            btnIcon.classList.add('fa-list');
        }
    }

    async loadProviderModels(providerKey) {
        const loadingEl = document.getElementById(`models-loading-${providerKey}`);
        const contentEl = document.getElementById(`models-content-${providerKey}`);
        const errorEl = document.getElementById(`models-error-${providerKey}`);

        // Show loading state
        loadingEl.style.display = 'block';
        contentEl.style.display = 'none';
        errorEl.style.display = 'none';

        try {
            console.log(`🔍 Loading models for provider: ${providerKey}`);

            const response = await fetch(`/admin/models/provider?provider_key=${encodeURIComponent(providerKey)}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const models = await response.json();
            console.log(`📊 Loaded ${models.length} models for ${providerKey}:`, models);

            // Render models
            this.renderModelsList(providerKey, models);

            // Show content
            loadingEl.style.display = 'none';
            contentEl.style.display = 'block';

        } catch (error) {
            console.error(`❌ Error loading models for ${providerKey}:`, error);

            // Show error state
            loadingEl.style.display = 'none';
            errorEl.style.display = 'block';
        }
    }

    renderModelsList(providerKey, models) {
        const listEl = document.getElementById(`models-list-${providerKey}`);

        if (models.length === 0) {
            listEl.innerHTML = `
                <div class="text-center py-4">
                    <i class="fas fa-info-circle text-muted mb-2" style="font-size: 2rem;"></i>
                    <p class="text-muted mb-0">No models found for this provider</p>
                </div>
            `;
            return;
        }

        let html = '';
        models.forEach(model => {
            const statusClass = model.is_enabled ? 'status-active' : 'status-inactive';
            const statusText = model.is_enabled ? 'Enabled' : 'Disabled';
            const itemClass = model.is_enabled ? '' : 'disabled';

            html += `
                <div class="model-item ${itemClass}" id="model-item-${model.id}">
                    <div class="model-info">
                        <div class="model-name">${model.model_name}</div>
                        <div class="model-id">${model.model_id}</div>
                    </div>
                    <div class="model-actions">
                        <div class="model-status">
                            <span class="status-badge ${statusClass}">${statusText}</span>
                        </div>
                        <label class="model-toggle-switch">
                            <input type="checkbox" ${model.is_enabled ? 'checked' : ''} 
                                   onchange="window.ModelManager.toggleIndividualModel('${model.model_id}', this.checked, '${providerKey}')"
                                   id="toggle-${model.id}">
                            <span class="model-toggle-slider" id="slider-${model.id}"></span>
                        </label>
                    </div>
                </div>
            `;
        });

        listEl.innerHTML = html;
    }

    async toggleIndividualModel(modelId, enabled, providerKey) {
        // Find all matching model items (could be in regular list and/or search results)
        const allModelItems = document.querySelectorAll(`.model-item, .search-model-result`);
        const matchingItems = Array.from(allModelItems).filter(item => {
            const modelIdEl = item.querySelector('.model-id');
            return modelIdEl && modelIdEl.textContent.trim() === modelId;
        });

        // Find all matching sliders and status badges
        const allSliders = [];
        const allStatusBadges = [];
        const allCheckboxes = [];

        matchingItems.forEach(item => {
            const slider = item.querySelector('.model-toggle-slider');
            const statusBadge = item.querySelector('.status-badge');
            const checkbox = item.querySelector('input[type="checkbox"]');
            
            if (slider) allSliders.push(slider);
            if (statusBadge) allStatusBadges.push(statusBadge);
            if (checkbox) allCheckboxes.push(checkbox);
        });

        // Show loading state on all sliders
        allSliders.forEach(slider => slider.classList.add('loading'));

        try {
            console.log(`🔄 Toggling model: ${modelId} to ${enabled ? 'enabled' : 'disabled'}`);

            const response = await fetch(`/admin/models/toggle?model_id=${encodeURIComponent(modelId)}`, {
                method: 'PUT',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enabled: enabled })
            });

            if (response.ok) {
                const result = await response.json();
                console.log(`✅ Model toggle result:`, result);

                // Update all status badges
                allStatusBadges.forEach(statusBadge => {
                    statusBadge.textContent = enabled ? 'Enabled' : 'Disabled';
                    statusBadge.className = `status-badge ${enabled ? 'status-active' : 'status-inactive'}`;
                });

                // Update all model items
                matchingItems.forEach(modelItem => {
                    if (enabled) {
                        modelItem.classList.remove('disabled');
                    } else {
                        modelItem.classList.add('disabled');
                    }
                });

                window.UIUtils.showToast(
                    `Model ${enabled ? 'enabled' : 'disabled'} successfully`, 
                    'success'
                );

                // Update provider model counts
                await this.updateProviderModelCounts(providerKey);
            } else {
                const error = await response.json();
                window.UIUtils.showToast(`Failed to toggle model: ${error.detail}`, 'error');

                // Revert all checkboxes
                allCheckboxes.forEach(checkbox => {
                    checkbox.checked = !enabled;
                });
            }
        } catch (error) {
            console.error(`❌ Error toggling model ${modelId}:`, error);
            window.UIUtils.showToast(`Error: ${error.message}`, 'error');

            // Revert all checkboxes
            allCheckboxes.forEach(checkbox => {
                checkbox.checked = !enabled;
            });
        } finally {
            // Remove loading state from all sliders
            allSliders.forEach(slider => slider.classList.remove('loading'));
        }
    }

    async bulkToggleProviderModels(providerKey, enable) {
        console.log('Bulk toggling provider models:', { providerKey, enable });
        
        try {
            const response = await fetch(`/admin/models/provider/toggle?provider_key=${encodeURIComponent(providerKey)}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'include',
                body: JSON.stringify({
                    enabled: enable
                })
            });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Bulk toggle successful:', result);
                window.UIUtils.showToast(result.message, 'success');
                
                // Clear search manager's cached models data
                if (window.SearchManager) {
                    window.SearchManager.clearModelsCache();
                }
                
                // Update the UI for this provider
                await this.updateProviderModelCounts(providerKey);
            } else {
                const error = await response.json();
                console.error('❌ Bulk toggle failed:', error);
                window.UIUtils.showToast(`Failed to toggle provider models: ${error.detail}`, 'error');
            }
        } catch (error) {
            console.error('❌ Error in bulk toggle:', error);
            window.UIUtils.showToast(`Error: ${error.message}`, 'error');
        }
    }

    async updateProviderModelCounts(providerKey) {
        try {
            // Reload the provider data to get updated counts
            await window.ProviderManager.loadUnifiedProviderData();
        } catch (error) {
            console.warn('Failed to update provider model counts:', error);
        }
    }

    async syncModels() {
        const syncBtn = document.querySelector('[aria-label="Sync Models"]');
        const syncIcon = syncBtn?.querySelector('i');

        const confirmed = await window.UIUtils.showConfirmModal(
            'Sync Models',
            'This will sync all models from providers. Continue?'
        );
        if (!confirmed) return;

        if (syncBtn) syncBtn.disabled = true;
        if (syncIcon) { syncIcon.classList.add('fa-spin'); }

        window.ProviderManager.showProvidersLoading();
        try {
            const response = await fetch('/admin/models/sync', {
                method: 'POST',
                credentials: 'include'
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Sync response:', result);

                const providersCount = result.providers_synced || 0;
                const modelsCount = result.models_synced || 0;
                const staleCount = result.stale_count || 0;

                window.UIUtils.showToast(`Synced ${providersCount} providers and ${modelsCount} models`, 'success');

                if (window.SearchManager) {
                    window.SearchManager.clearModelsCache();
                }

                await window.ProviderManager.loadUnifiedProviderData();

                if (staleCount > 0 && result.stale_models && result.stale_models.length > 0) {
                    this.showStaleModelsDialog(result.stale_models);
                }
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
                window.ProviderManager.showProvidersList();
            }
        } catch (error) {
            window.UIUtils.showToast('Error syncing models: ' + error.message, 'error');
            window.ProviderManager.showProvidersList();
        } finally {
            if (syncBtn) syncBtn.disabled = false;
            if (syncIcon) { syncIcon.classList.remove('fa-spin'); }
        }
    }

    showStaleModelsDialog(staleModels) {
        // Create modal HTML
        const modalHtml = `
            <div class="modal fade" id="staleModelsModal" tabindex="-1" aria-labelledby="staleModelsModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header bg-warning">
                            <h5 class="modal-title" id="staleModelsModalLabel">
                                <i class="fas fa-exclamation-triangle"></i> Stale Models Detected
                            </h5>
                            <button class="btn btn-sm btn-outline-secondary" data-bs-dismiss="modal">
                                <i class="fas fa-times"></i> 
                            </button>
                        </div>
                        <div class="modal-body">
                            <p class="mb-3">Found <strong>${staleModels.length}</strong> stale model(s) in the database that are no longer available from providers.</p>
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle"></i> These models exist in your database but are not currently available from any provider. You can safely remove them.
                            </div>
                            <div class="stale-models-list" style="max-height: 400px; overflow-y: auto;">
                                <table class="table table-striped table-sm mb-0" style="margin-top: 0;">
                                    <thead style="position: sticky; top: -1px; background-color: var(--mono-1); z-index: 10; box-shadow: 0 2px 2px -1px rgba(0, 0, 0, 0.1); border-top: none;">
                                        <tr style="background-color: var(--mono-1) !important;">
                                            <th style="border-top: none; padding-top: 0.5rem; background-color: var(--mono-1);">Model ID</th>
                                            <th style="border-top: none; padding-top: 0.5rem; background-color: var(--mono-1);">Provider</th>
                                            <th style="border-top: none; padding-top: 0.5rem; background-color: var(--mono-1);">Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${staleModels.map(model => `
                                            <tr>
                                                <td><code>${this.escapeHtml(model.model_id)}</code></td>
                                                <td><span class="badge bg-secondary">${this.escapeHtml(model.provider_key)}</span></td>
                                                <td><span class="badge ${model.is_enabled ? 'bg-success' : 'bg-danger'}">${model.is_enabled ? 'Enabled' : 'Disabled'}</span></td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Keep Models</button>
                            <button type="button" class="btn btn-danger" onclick="window.ModelManager.removeStaleModels()">
                                <i class="fas fa-trash"></i> Remove Stale Models
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove any existing modal
        const existingModal = document.getElementById('staleModelsModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Store stale models for removal
        this.currentStaleModels = staleModels;

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('staleModelsModal'));
        modal.show();

        // Clean up when modal is hidden
        document.getElementById('staleModelsModal').addEventListener('hidden.bs.modal', () => {
            document.getElementById('staleModelsModal').remove();
        }, { once: true });
    }

    async removeStaleModels() {
        if (!this.currentStaleModels || this.currentStaleModels.length === 0) {
            return;
        }

        // Get the stale models modal element before showing confirmation
        const staleModalElement = document.getElementById('staleModelsModal');
        const staleModal = bootstrap.Modal.getInstance(staleModalElement);

        // Temporarily hide the stale models modal to show confirmation modal on top
        if (staleModal) {
            staleModal.hide();
        }

        // Wait for modal to hide before showing confirmation
        await new Promise(resolve => setTimeout(resolve, 150));

        const confirmed = await window.UIUtils.showConfirmModal(
            'Remove Stale Models',
            `Are you sure you want to remove ${this.currentStaleModels.length} stale model(s) from the database?`,
            'danger'
        );

        if (!confirmed) {
            // If cancelled, show the stale models modal again
            if (staleModal) {
                staleModal.show();
            }
            return;
        }

        try {
            const modelIds = this.currentStaleModels.map(m => m.model_id);
            
            const response = await fetch('/admin/models/remove-stale', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_ids: modelIds })
            });

            if (response.ok) {
                const result = await response.json();
                
                // Modal is already hidden, just show success message
                window.UIUtils.showToast(result.message, 'success');
                
                // Reload provider data
                await window.ProviderManager.loadUnifiedProviderData();
                
                // Clear stored stale models
                this.currentStaleModels = null;
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
                // Show the stale models modal again on error
                if (staleModal) {
                    staleModal.show();
                }
            }
        } catch (error) {
            window.UIUtils.showToast('Error removing stale models: ' + error.message, 'error');
            // Show the stale models modal again on error
            if (staleModal) {
                staleModal.show();
            }
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async bulkToggle(action) {
        const actionText = action === 'enable_all' ? 'enable' : 'disable';
        const confirmed = await window.UIUtils.showConfirmModal(
            `${actionText.charAt(0).toUpperCase() + actionText.slice(1)} All Models`,
            `Are you sure you want to ${actionText} all models?`
        );
        if (!confirmed) return;

        window.ProviderManager.showProvidersLoading();
        try {
            const response = await fetch('/admin/models/bulk-toggle', {
                method: 'PUT',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: action })
            });

            if (response.ok) {
                const result = await response.json();
                window.UIUtils.showToast(result.message, 'success');
                
                // Clear search manager's cached models data
                if (window.SearchManager) {
                    window.SearchManager.clearModelsCache();
                }
                
                await window.ProviderManager.loadUnifiedProviderData(); // Reload all data
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error with bulk toggle: ' + error.message, 'error');
        }
    }

    async reinitSystem() {
        const confirmed = await window.UIUtils.showConfirmModal(
            'Reinitialize System',
            '⚠️ WARNING: This will completely clear ALL provider and model configurations from the database. This action cannot be undone!\n\nAre you absolutely sure you want to proceed?'
        );
        if (!confirmed) return;

        window.ProviderManager.showProvidersLoading();

        try {
            const response = await fetch('/admin/system/reinit', {
                method: 'POST',
                credentials: 'include'
            });

            if (response.ok) {
                const result = await response.json();
                window.UIUtils.showToast(`System reinitialized: cleared ${result.providers_cleared} providers and ${result.models_cleared} models`, 'success');
                
                // Clear search manager's cached models data
                if (window.SearchManager) {
                    window.SearchManager.clearModelsCache();
                }
                
                await window.ProviderManager.loadUnifiedProviderData(); // Reload all data
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
                window.ProviderManager.showProvidersEmpty();
            }
        } catch (error) {
            console.error('Reinit system error:', error);
            window.UIUtils.showToast('Error reinitializing system: ' + error.message, 'error');
            window.ProviderManager.showProvidersEmpty();
        }
    }
}

// Create global instance
window.ModelManager = new ModelManager();

// Export functions for backward compatibility
window.toggleProviderModels = (key) => window.ModelManager.toggleProviderModels(key);
window.loadProviderModels = (key) => window.ModelManager.loadProviderModels(key);
window.toggleIndividualModel = (id, enabled, provider) => window.ModelManager.toggleIndividualModel(id, enabled, provider);
window.bulkToggleProviderModels = (key, enabled) => window.ModelManager.bulkToggleProviderModels(key, enabled);
window.syncModels = () => window.ModelManager.syncModels();
window.bulkToggle = (action) => window.ModelManager.bulkToggle(action);
window.reinitSystem = () => window.ModelManager.reinitSystem();

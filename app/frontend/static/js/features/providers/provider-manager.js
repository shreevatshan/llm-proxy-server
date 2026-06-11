/**
 * Provider Management Module
 * Handles provider CRUD operations, forms, and provider-specific functionality
 */

// Global variables
let providersData = null;
let modelsData = null;
let currentEditingProvider = null;


class ProviderManager {
    async loadUnifiedProviderData() {
        this.showProvidersLoading();

        try {
            console.log('🔍 Loading provider and model data...');

            // Fetch providers and models data
            const [providersResponse, modelsResponse] = await Promise.all([
                fetch('/admin/providers', { credentials: 'include' }),
                fetch('/admin/models', { credentials: 'include' })
            ]);

            console.log('📡 API Response Status:', {
                providers: providersResponse.status,
                models: modelsResponse.status
            });

            if (!providersResponse.ok || !modelsResponse.ok) {
                const providersError = providersResponse.ok ? 'OK' : await providersResponse.text();
                const modelsError = modelsResponse.ok ? 'OK' : await modelsResponse.text();
                console.error('❌ API Errors:', { providersError, modelsError });
                throw new Error('Failed to fetch provider or model data');
            }

            const providers = await providersResponse.json();
            const modelsData = await modelsResponse.json();

            console.log('📊 API Data Received:', {
                providers: providers.length,
                modelsData: modelsData,
                enabledModels: modelsData.enabled_models,
                totalModels: modelsData.total_models
            });

            // Store data globally
            providersData = providers;
            window.providersData = providers;

            // Update SearchManager's cached data
            if (window.SearchManager) {
                window.SearchManager.refreshCachedData();
            }

            // Update stats
            this.updateProviderStats(providers, modelsData);

            if (providers.length === 0) {
                this.showProvidersEmpty();
            } else {
                this.renderProvidersList(providers);
                this.showProvidersList();
            }

        } catch (error) {
            console.error('❌ Error loading provider data:', error);
            window.UIUtils.showToast('Failed to load provider data: ' + error.message, 'error');
            this.showProvidersEmpty();
        }
    }

    updateProviderStats(providers, modelsData) {
        const enabledProviders = providers.filter(p => p.enabled).length;
        const enabledModels = modelsData.enabled_models || 0;

        console.log('🔧 Updating provider stats:', {
            enabledProviders,
            enabledModels,
            providersLength: providers.length,
            modelsDataStructure: Object.keys(modelsData)
        });

        const providerStatsEl = document.getElementById('enabled-providers');
        const modelStatsEl = document.getElementById('enabled-models');

        console.log('🎯 DOM Elements found:', {
            providerStatsEl: !!providerStatsEl,
            modelStatsEl: !!modelStatsEl,
            providerCurrentText: providerStatsEl?.textContent,
            modelCurrentText: modelStatsEl?.textContent
        });

        if (providerStatsEl) {
            console.log('🔄 Updating provider element...', { before: providerStatsEl.textContent, after: enabledProviders });
            providerStatsEl.textContent = enabledProviders;
            console.log('✅ Provider element updated. New text:', providerStatsEl.textContent);
        } else {
            console.error('❌ Provider stats element not found!');
        }

        if (modelStatsEl) {
            console.log('🔄 Updating model element...', { before: modelStatsEl.textContent, after: enabledModels });
            modelStatsEl.textContent = enabledModels;
            console.log('✅ Model element updated. New text:', modelStatsEl.textContent);
        } else {
            console.error('❌ Model stats element not found!');
        }
    }

    renderProvidersList(providers) {
        const listContainer = document.getElementById('providers-list');

        // Save collapsed state before replacing DOM
        const collapsedSections = new Set();
        listContainer.querySelectorAll('.provider-type-content').forEach(el => {
            if (el.style.display === 'none') {
                collapsedSections.add(el.id);
            }
        });

        if (providers.length === 0) {
            listContainer.innerHTML = '';
            return;
        }

        const groupedProviders = {};
        providers.forEach(provider => {
            // For custom providers, group by provider_name
            // For other providers, group by provider_type
            let groupKey;
            if (provider.provider_type === 'custom' && provider.provider_name) {
                groupKey = `custom:${provider.provider_name}`;
            } else {
                groupKey = provider.provider_type;
            }
            
            if (!groupedProviders[groupKey]) {
                groupedProviders[groupKey] = [];
            }
            groupedProviders[groupKey].push(provider);
        });

        let html = '';

        Object.entries(groupedProviders).forEach(([groupKey, typeProviders]) => {
            // Determine provider type and title for the section header
            let providerType, typeTitle;
            
            if (groupKey.startsWith('custom:')) {
                providerType = 'custom';
                const providerName = groupKey.split(':')[1];
                typeTitle = providerName.toUpperCase();
            } else {
                providerType = groupKey;
                typeTitle = providerType.charAt(0).toUpperCase() + providerType.slice(1);
            }
            
            const enabledCount = typeProviders.filter(p => p.enabled).length;

            html += `
                <div class="provider-type-section">
                    <div class="provider-type-header" onclick="toggleProviderType('${groupKey}')">
                        <i class="fas fa-chevron-down provider-type-icon" id="icon-${groupKey}"></i>
                        <h3>${typeTitle}</h3>
                        <span class="provider-count">${enabledCount}/${typeProviders.length}</span>
                    </div>
                    <div class="provider-type-content" id="content-${groupKey}">
            `;

            typeProviders.forEach(provider => {
                const statusClass = provider.enabled ? 'status-active' : 'status-inactive';
                const statusText = provider.enabled ? 'Enabled' : 'Disabled';
                
                // For card header: always show instance_name
                // For badge: show friendly provider type name + supported APIs
                let badgeName;
                if (provider.provider_type === 'custom') {
                    badgeName = 'Custom';
                } else {
                    badgeName = provider.provider_type.charAt(0).toUpperCase() + provider.provider_type.slice(1);
                }

                // Show supported API badges
                let apiBadges = '';
                const supportedApis = provider.supported_apis || [];
                if (supportedApis.length > 0) {
                    apiBadges = supportedApis.map(api => 
                        `<span class="badge bg-info ms-1" style="font-size: 0.65em;">${api}</span>`
                    ).join('');
                }

                html += `
                    <div class="unified-provider-card ${!provider.enabled ? 'disabled' : ''}" id="provider-${provider.provider_key}">
                        <div class="provider-card-header">
                            <div class="provider-info">
                                <div class="provider-title">
                                    <h4>${provider.instance_name}</h4>
                                    <span class="provider-type-badge">${badgeName}</span>${apiBadges}
                                </div>
                                <div class="provider-status">
                                    <span class="status-badge ${statusClass}">${statusText}</span>
                                </div>
                                <div class="provider-stats">
                                    <span class="model-stats">Models: ${provider.model_count || 0} (${provider.enabled_model_count || 0} enabled)</span>
                                </div>
                            </div>
                            <div class="provider-actions">
                                <button class="btn btn-sm btn-outline-info" onclick="window.ModelManager.toggleProviderModels('${provider.provider_key}')" title="Show/Hide Models" id="models-btn-${provider.provider_key}">
                                    <i class="fas fa-list"></i> Models (${provider.model_count || 0})
                                </button>
                                <label class="toggle-switch">
                                    <input type="checkbox" ${provider.enabled ? 'checked' : ''} 
                                           onchange="window.ProviderManager.toggleProviderEnabled('${provider.provider_key}', this.checked)">
                                    <span class="toggle-slider"></span>
                                </label>
                                <button class="btn btn-sm btn-outline-primary" onclick="window.ProviderManager.editProvider('${provider.provider_key}')" title="Edit">
                                    <i class="fas fa-edit"></i>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="window.ProviderManager.deleteProvider('${provider.provider_key}')" title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                        
                        <!-- Models Section -->
                        <div class="provider-models-section" id="models-section-${provider.provider_key}" style="display: none;">
                            <div class="models-loading" id="models-loading-${provider.provider_key}">
                                <div class="text-center py-3">
                                    <i class="fas fa-spinner fa-spin me-2"></i>
                                    Loading models...
                                </div>
                            </div>
                            <div class="models-content" id="models-content-${provider.provider_key}" style="display: none;">
                                <div class="models-header">
                                    <div class="d-flex justify-content-between align-items-center p-3 border-bottom">
                                        <h6 class="mb-0">
                                            <i class="fas fa-cog me-2"></i>
                                            Individual Model Controls
                                        </h6>
                                        <div class="model-bulk-actions">
                                            <button class="btn btn-sm btn-success" onclick="window.ModelManager.bulkToggleProviderModels('${provider.provider_key}', true)" title="Enable All Models">
                                                <i class="fas fa-check-circle"></i> Enable All
                                            </button>
                                            <button class="btn btn-sm btn-danger" onclick="window.ModelManager.bulkToggleProviderModels('${provider.provider_key}', false)" title="Disable All Models">
                                                <i class="fas fa-times-circle"></i> Disable All
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                <div class="models-list" id="models-list-${provider.provider_key}">
                                    <!-- Models will be loaded here -->
                                </div>
                            </div>
                            <div class="models-error" id="models-error-${provider.provider_key}" style="display: none;">
                                <div class="alert alert-warning m-3">
                                    <i class="fas fa-exclamation-triangle me-2"></i>
                                    Failed to load models. <a href="#" onclick="window.ModelManager.loadProviderModels('${provider.provider_key}')">Retry</a>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        });

        listContainer.innerHTML = html;

        // Restore collapsed state
        collapsedSections.forEach(id => {
            const content = document.getElementById(id);
            if (content) {
                content.style.display = 'none';
                const groupKey = id.replace('content-', '');
                const icon = document.getElementById(`icon-${groupKey}`);
                if (icon) {
                    icon.classList.remove('fa-chevron-down');
                    icon.classList.add('fa-chevron-right');
                }
            }
        });
    }

    showProvidersLoading() {
        document.getElementById('providers-loading').style.display = 'block';
        document.getElementById('providers-list').style.display = 'none';
        document.getElementById('providers-empty').style.display = 'none';
    }

    showProvidersEmpty() {
        document.getElementById('providers-loading').style.display = 'none';
        document.getElementById('providers-list').style.display = 'none';
        document.getElementById('providers-empty').style.display = 'block';
    }

    showProvidersList() {
        document.getElementById('providers-loading').style.display = 'none';
        document.getElementById('providers-list').style.display = 'block';
        document.getElementById('providers-empty').style.display = 'none';
    }

    showAddProviderForm() {
        console.log('🎯 showAddProviderForm called');
        
        // First ensure we're on the providers tab
        const providersTab = document.getElementById('providers-tab');
        console.log('📋 Providers tab element:', providersTab);
        
        if (providersTab && providersTab.style.display === 'none') {
            window.UIUtils.showTab('providers');
        }

        window.currentEditingProvider = null;
        
        const formTitle = document.getElementById('form-title');
        const submitText = document.getElementById('submit-text');
        const providerType = document.getElementById('provider_type');
        const instanceName = document.getElementById('instance_name');
        const specificFields = document.getElementById('provider-specific-fields');
        const configSection = document.getElementById('provider-config-section');
        
        console.log('🔍 Form elements check:', {
            formTitle: !!formTitle,
            submitText: !!submitText,
            providerType: !!providerType,
            instanceName: !!instanceName,
            specificFields: !!specificFields,
            configSection: !!configSection,
            ProviderFormManager: !!window.ProviderFormManager
        });
        
        if (formTitle) formTitle.textContent = 'Add Provider';
        if (submitText) submitText.textContent = 'Create Provider';

        // Reset form
        if (providerType) providerType.value = '';
        if (instanceName) instanceName.value = '';
        if (specificFields) specificFields.innerHTML = '';
        if (configSection) configSection.style.display = 'none';
        if (window.ProviderFormManager) {
            window.ProviderFormManager.clearFormAlerts();
        }
        
        // Reset button state to ensure it's not stuck in loading
        const submitBtn = document.getElementById('submit-btn');
        if (submitBtn) {
            const btnContent = submitBtn.querySelector('.btn-content');
            const btnLoading = submitBtn.querySelector('.btn-loading');
            if (btnContent && btnLoading) {
                btnContent.style.display = 'flex';
                btnLoading.classList.add('d-none');
                submitBtn.disabled = false;
            }
        }
        
        // Reset submission flag
        if (window.ProviderFormManager) {
            window.ProviderFormManager.isSubmitting = false;
        }

        // Show form and hide other sections
        this.showProviderForm();
    }

    showProviderForm() {
        document.getElementById('provider-form-container').style.display = 'block';
        document.getElementById('providers-list').style.display = 'none';
        document.getElementById('providers-empty').style.display = 'none';
        document.getElementById('providers-loading').style.display = 'none';

        // Ensure form event listeners are properly bound
        setTimeout(() => {
            console.log('🔄 Re-binding form event listeners...');
            const providerForm = document.getElementById('provider-form');
            console.log('📋 Form check after showProviderForm:', {
                formExists: !!providerForm,
                adminDashboardExists: !!window.AdminDashboard,
                formVisible: providerForm?.style.display !== 'none'
            });
            
            if (providerForm && window.AdminDashboard) {
                window.AdminDashboard.setupEventListeners();
                
                // Also add a direct click handler to the submit button as backup
                const submitBtn = document.getElementById('submit-btn');
                if (submitBtn) {
                    console.log('🔧 Adding backup click handler to submit button');
                    submitBtn.onclick = (e) => {
                        console.log('🖱️ Submit button clicked directly');
                        e.preventDefault();
                        
                        // Create a form submit event
                        const form = document.getElementById('provider-form');
                        if (form) {
                            const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
                            form.dispatchEvent(submitEvent);
                        } else {
                            console.error('❌ Form not found when button clicked');
                        }
                    };
                }
            }
        }, 100);
    }

    cancelProviderForm() {
        document.getElementById('provider-form-container').style.display = 'none';
        window.ProviderFormManager.clearFormAlerts();

        // Show appropriate view based on providers data
        if (providersData && providersData.length > 0) {
            this.showProvidersList();
        } else {
            this.showProvidersEmpty();
        }
    }

    async toggleProviderEnabled(providerKey, enabled) {
        try {
            const response = await fetch(`/admin/models/provider/toggle?provider_key=${encodeURIComponent(providerKey)}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled })
            });

            if (response.ok) {
                window.UIUtils.showToast(
                    `Provider ${enabled ? 'enabled' : 'disabled'} successfully`, 
                    'success'
                );
                await this.loadUnifiedProviderData();
            } else {
                const error = await response.json();
                window.UIUtils.showToast(`Failed to ${enabled ? 'enable' : 'disable'} provider: ${error.detail}`, 'error');
                // Revert checkbox
                const checkbox = document.querySelector(`input[onchange*="${providerKey}"]`);
                if (checkbox) checkbox.checked = !enabled;
            }
        } catch (error) {
            window.UIUtils.showToast(`Error: ${error.message}`, 'error');
            // Revert checkbox
            const checkbox = document.querySelector(`input[onchange*="${providerKey}"]`);
            if (checkbox) checkbox.checked = !enabled;
        }
    }

    async editProvider(providerKey) {
        try {
            const response = await fetch(`/admin/providers/detail?provider_key=${encodeURIComponent(providerKey)}`, {
                credentials: 'include'
            });

            if (response.ok) {
                const provider = await response.json();
                window.currentEditingProvider = provider;

                // Update form title
                document.getElementById('form-title').textContent = 'Edit Provider';
                document.getElementById('submit-text').textContent = 'Update Provider';

                // Populate form fields
                const providerTypeField = document.getElementById('provider_type');
                const instanceNameField = document.getElementById('instance_name');
                
                if (providerTypeField) providerTypeField.value = provider.provider_type;
                if (instanceNameField) instanceNameField.value = provider.instance_name;

                // Trigger form update to show provider-specific fields
                this.updateProviderForm();

                // Wait a bit for fields to be created, then populate them
                setTimeout(() => {
                    window.ProviderFormManager.populateProviderFields(provider);
                }, 100);

                this.showProviderForm();
            } else {
                const error = await response.json();
                window.UIUtils.showToast(`Failed to load provider: ${error.detail}`, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast(`Error loading provider: ${error.message}`, 'error');
        }
    }

    async deleteProvider(providerKey) {
        const confirmed = await window.UIUtils.showConfirmModal(
            'Delete Provider',
            `⚠️ WARNING: This will permanently delete the provider "${providerKey}" and all its associated models. This action cannot be undone!\n\nAre you absolutely sure you want to proceed?`,
            'danger'
        );
        if (!confirmed) return;

        try {
            const response = await fetch(`/admin/providers?provider_key=${encodeURIComponent(providerKey)}`, {
                method: 'DELETE',
                credentials: 'include'
            });

            if (response.ok) {
                window.UIUtils.showToast('Provider deleted successfully', 'success');
                await this.loadUnifiedProviderData();
            } else {
                const error = await response.json();
                window.UIUtils.showToast(`Failed to delete provider: ${error.detail}`, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast(`Error deleting provider: ${error.message}`, 'error');
        }
    }

    // Additional provider management methods would go here...

    toggleProviderType(providerType) {
        const icon = document.getElementById(`icon-${providerType}`);
        const content = document.getElementById(`content-${providerType}`);

        if (content.style.display === 'none') {
            content.style.display = 'block';
            icon.classList.remove('fa-chevron-right');
            icon.classList.add('fa-chevron-down');
        } else {
            content.style.display = 'none';
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-right');
        }
    }

    updateProviderForm() {
        return window.ProviderFormManager.updateProviderForm();
    }
}

// Create global instance
window.ProviderManager = new ProviderManager();

// Export for backward compatibility
window.loadUnifiedProviderData = () => window.ProviderManager.loadUnifiedProviderData();
window.showAddProviderForm = () => window.ProviderManager.showAddProviderForm();
window.cancelProviderForm = () => window.ProviderManager.cancelProviderForm();
window.toggleProviderEnabled = (key, enabled) => window.ProviderManager.toggleProviderEnabled(key, enabled);
window.editProvider = (key) => window.ProviderManager.editProvider(key);
window.deleteProvider = (key) => window.ProviderManager.deleteProvider(key);
window.toggleProviderType = (type) => window.ProviderManager.toggleProviderType(type);

// ── Export / Import ──────────────────────────────────────────────────────────

window._importFile = null;

async function exportProviderConfig() {
    try {
        const response = await fetch('/admin/providers/export', { credentials: 'include' });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
            window.UIUtils.showToast('Export failed: ' + err.detail, 'error');
            return;
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'llm-proxy-config.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        window.UIUtils.showToast('Config exported successfully', 'success');
    } catch (error) {
        window.UIUtils.showToast('Export error: ' + error.message, 'error');
    }
}

function importProviderConfig(input) {
    const file = input.files[0];
    if (!file) return;
    // Reset input so the same file can be re-selected later
    input.value = '';

    window._importFile = file;

    // Show modal
    document.getElementById('importFileName').textContent = file.name;
    document.getElementById('importOverwrite').checked = false;
    document.getElementById('importSyncModels').checked = true;
    const alertEl = document.getElementById('importResultAlert');
    alertEl.className = 'alert d-none';
    alertEl.textContent = '';

    // Reset button state
    const btn = document.getElementById('importConfirmBtn');
    btn.querySelector('.btn-content').classList.remove('d-none');
    btn.querySelector('.btn-loading').classList.add('d-none');
    btn.disabled = false;

    const modal = new bootstrap.Modal(document.getElementById('importConfigModal'));
    modal.show();
}

async function confirmImport() {
    const file = window._importFile;
    if (!file) return;

    const overwrite = document.getElementById('importOverwrite').checked;
    const syncModels = document.getElementById('importSyncModels').checked;
    const alertEl = document.getElementById('importResultAlert');
    const btn = document.getElementById('importConfirmBtn');

    btn.querySelector('.btn-content').classList.add('d-none');
    btn.querySelector('.btn-loading').classList.remove('d-none');
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams({ overwrite, sync_models: syncModels });
        const response = await fetch(`/admin/providers/import?${params}`, {
            method: 'POST',
            credentials: 'include',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            alertEl.className = 'alert alert-danger';
            alertEl.textContent = result.detail || 'Import failed';
        } else {
            const parts = [
                `Imported: ${result.imported}`,
                `Overwritten: ${result.overwritten}`,
                `Skipped: ${result.skipped}`,
            ];
            if (result.errors && result.errors.length > 0) {
                parts.push(`Errors: ${result.errors.length}`);
            }
            alertEl.className = result.errors && result.errors.length > 0 ? 'alert alert-warning' : 'alert alert-success';
            alertEl.innerHTML = parts.join(' &bull; ') +
                (result.errors && result.errors.length > 0
                    ? '<ul class="mb-0 mt-1">' + result.errors.map(e => `<li>${e}</li>`).join('') + '</ul>'
                    : '');

            // Reload providers list
            await window.ProviderManager.loadUnifiedProviderData();
        }
    } catch (error) {
        alertEl.className = 'alert alert-danger';
        alertEl.textContent = 'Import error: ' + error.message;
    } finally {
        btn.querySelector('.btn-content').classList.remove('d-none');
        btn.querySelector('.btn-loading').classList.add('d-none');
        btn.disabled = false;
    }
}

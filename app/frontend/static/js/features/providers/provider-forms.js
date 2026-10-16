/**
 * Provider Forms Module
 * Handles provider form generation, validation, and submission
 */

// The backend never returns real secrets: GET /admin/providers/detail masks
// them with this fixed placeholder (see _mask_secret in app/routes/admin.py).
// These fields must never be pre-filled with (or submitted as) the mask.
const SECRET_MASK = '********';
const SECRET_FIELDS = ['api_key', 'secret_access_key'];

class ProviderFormManager {
    constructor() {
        this.isSubmitting = false;
    }

    updateProviderForm() {
        const providerType = document.getElementById('provider_type').value;
        const fieldsContainer = document.getElementById('provider-specific-fields');
        const configSection = document.getElementById('provider-config-section');

        // Clear any existing alerts
        this.clearFormAlerts();

        if (!providerType) {
            fieldsContainer.innerHTML = '';
            configSection.style.display = 'none';
            return;
        }

        // Show the configuration section
        configSection.style.display = 'block';

        let formFields = '';

        switch (providerType.toLowerCase()) {
            case 'azure':
                formFields = this.getAzureFormFields();
                break;
            case 'google':
                formFields = this.getGoogleFormFields();
                break;
            case 'bedrock':
                formFields = this.getBedrockFormFields();
                break;
            case 'custom':
                formFields = this.getCustomFormFields();
                break;
            default:
                formFields = '<p class="text-muted">Select a provider type to see configuration options.</p>';
        }

        fieldsContainer.innerHTML = formFields;

        // Initialize Azure-specific functionality if Azure provider is selected
        if (providerType.toLowerCase() === 'azure') {
            // Small delay to ensure DOM is updated
            setTimeout(() => {
                if (window.AzureManager && window.AzureManager.toggleDynamicDiscovery) {
                    window.AzureManager.toggleDynamicDiscovery();
                }
            }, 100);
        }
    }

    getAzureFormFields() {
        return `
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="azure_backend" class="form-label">
                            <i class="fas fa-sitemap"></i> Azure Backend <span class="text-danger">*</span>
                        </label>
                        <select id="azure_backend" class="form-select" onchange="window.AzureManager.toggleDynamicDiscovery()" required>
                            <option value="openai" selected>Azure OpenAI</option>
                            <option value="foundry">Azure Foundry</option>
                        </select>
                        <div class="form-text">Choose whether this Azure provider routes to Azure OpenAI or Foundry.</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="endpoint" class="form-label">
                            <i class="fas fa-globe"></i> Endpoint <span class="text-danger">*</span>
                        </label>
                        <input type="text" id="endpoint" class="form-control" 
                               placeholder="https://your-resource.openai.azure.com" required>
                    </div>
                </div>
            </div>
            <div class="mb-3">
                <label for="api_key" class="form-label">
                    <i class="fas fa-key"></i> API Key <span class="text-danger">*</span>
                </label>
                <input type="password" id="api_key" class="form-control" 
                       placeholder="Your Azure OpenAI API key" required>
            </div>
            
            <!-- Dynamic Discovery Section -->
            <div class="mb-4">
                <h6 class="text-primary mb-3">
                    <i class="fas fa-search"></i> Model Discovery Settings
                </h6>
                <div class="mb-3">
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" id="dynamic_discovery" onchange="window.AzureManager.toggleDynamicDiscovery()">
                        <label class="form-check-label" for="dynamic_discovery">
                            <strong>Enable Dynamic Discovery</strong>
                        </label>
                        <div class="form-text">Automatically discover Azure models or deployments for the selected backend.</div>
                    </div>
                </div>
                
                <!-- Manual Deployments (shown when dynamic discovery is disabled) -->
                <div id="manual-deployments" class="mb-3" style="display: block;">
                    <label class="form-label">
                        <i class="fas fa-robot"></i> Deployments
                    </label>
                    <div id="deployment-groups" class="deployment-groups"></div>
                    <button type="button" class="btn btn-outline-secondary btn-sm mt-2" id="add-deployment-group"
                            onclick="window.AzureManager.addDeploymentGroup()" title="Add deployment row">
                        <i class="fas fa-plus"></i> Add
                    </button>
                    <div class="form-text">
                        Comma-separated deployments per row, served only over the APIs selected for that row.
                        Calls to unselected APIs are rejected. Only rows with <code>/chat/completions</code>,
                        <code>/responses</code> or <code>/messages</code> are offered on <code>/v1/messages</code>;
                        without <code>/messages</code> those requests are adapted onto <code>/chat/completions</code>,
                        or <code>/responses</code> when chat is not selected. <code>/messages</code> (Foundry) calls
                        the deployment natively.
                    </div>
                </div>
                
                <!-- Dynamic Discovery API Version (shown for both backends when dynamic discovery is enabled) -->
                <div id="azure-management-config" class="border rounded p-3 bg-light" style="display: none;">
                    <h6 class="text-secondary mb-3">
                        <i class="fas fa-search"></i> Dynamic Discovery Settings
                    </h6>
                    <div class="mb-3">
                        <label for="discovery_api_version" class="form-label">
                            <i class="fas fa-code-branch"></i> API Version <span class="text-danger">*</span>
                        </label>
                        <input type="text" id="discovery_api_version" class="form-control"
                               placeholder="2024-10-21">
                        <div class="form-text">
                            api-version passed to <code>GET {endpoint}/openai/models?api-version=…</code>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getGoogleFormFields() {
        return `
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="api_key" class="form-label">
                            <i class="fas fa-key"></i> API Key <span class="text-danger">*</span>
                        </label>
                        <input type="password" id="api_key" class="form-control" 
                               placeholder="Your Google AI API key" required>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="base_url" class="form-label">
                            <i class="fas fa-link"></i> Base URL
                        </label>
                        <input type="text" id="base_url" class="form-control" 
                               placeholder="https://generativelanguage.googleapis.com/v1beta/openai/" 
                               value="https://generativelanguage.googleapis.com/v1beta/openai/">
                    </div>
                </div>
            </div>
        `;
    }

    getCustomFormFields() {
        return `
            <div class="mb-3">
                <label for="provider_name" class="form-label">
                    <i class="fas fa-tag"></i> Provider Name <span class="text-danger">*</span>
                </label>
                <input type="text" id="provider_name" class="form-control" 
                       placeholder="e.g., openai, ollama, llamacpp, my-custom-server" 
                       required>
                <div class="form-text">
                    Enter a name to identify this provider type (e.g., openai, ollama, llamacpp, or any custom name).
                </div>
            </div>
            <div class="form-text mb-3">Fill in the base URL for each API the server supports natively; at least one is required.
                OpenAI-only providers are still served on the Anthropic Messages API through the built-in converter.</div>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="base_url" class="form-label">
                            <i class="fas fa-robot"></i> OpenAI base URL
                        </label>
                        <input type="url" id="base_url" class="form-control"
                               placeholder="e.g., https://api.openai.com/v1 or http://localhost:11434/v1">
                        <div class="form-text">Complete server URL including /v1 path. Leave blank if the server has no OpenAI API.</div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="anthropic_base_url" class="form-label">
                            <i class="fas fa-brain"></i> Anthropic base URL
                        </label>
                        <input type="url" id="anthropic_base_url" class="form-control"
                               placeholder="e.g., https://api.deepseek.com/anthropic">
                        <div class="form-text">Complete server URL without the /v1/messages path. Leave blank if the server has no Anthropic API.</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="anthropic_auth_style" class="form-label">
                            <i class="fas fa-key"></i> Anthropic auth header
                        </label>
                        <select id="anthropic_auth_style" class="form-control form-select">
                            <option value="">x-api-key:</option>
                            <option value="bearer">Authorization: Bearer</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="api_key" class="form-label">
                            <i class="fas fa-key"></i> API Key
                        </label>
                        <input type="password" id="api_key" class="form-control"
                               placeholder="Optional (leave blank if not required)">
                        <div class="form-text">Leave blank if your server doesn't require authentication</div>
                    </div>
                </div>
            </div>
        `;
    }

    getBedrockFormFields() {
        return `
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="region" class="form-label">
                            <i class="fas fa-map-marker-alt"></i> Region <span class="text-danger">*</span>
                        </label>
                        <input type="text" id="region" class="form-control" 
                               placeholder="us-west-2" value="us-west-2" required>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="access_key_id" class="form-label">
                            <i class="fas fa-user"></i> Access Key ID <span class="text-danger">*</span>
                        </label>
                        <input type="text" id="access_key_id" class="form-control" 
                               placeholder="Your AWS Access Key ID" required>
                    </div>
                </div>
            </div>
            <div class="mb-3">
                <label for="secret_access_key" class="form-label">
                    <i class="fas fa-lock"></i> Secret Access Key <span class="text-danger">*</span>
                </label>
                <input type="password" id="secret_access_key" class="form-control" 
                       placeholder="Your AWS Secret Access Key" required>
            </div>
        `;
    }

    async submitProviderForm(event) {
        console.log('🚀 Form submission started - event:', event);
        event.preventDefault();

        // Prevent multiple simultaneous submissions
        if (this.isSubmitting) {
            console.log('⚠️ Form submission already in progress, ignoring...');
            return;
        }

        this.isSubmitting = true;

        // Clear existing alerts
        this.clearFormAlerts();

        // Show loading state
        const submitBtn = document.getElementById('submit-btn');
        const btnContent = submitBtn.querySelector('.btn-content');
        const btnLoading = submitBtn.querySelector('.btn-loading');

        console.log('🔍 DOM Elements check:', {
            submitBtn: !!submitBtn,
            btnContent: !!btnContent,
            btnLoading: !!btnLoading,
            submitBtnId: submitBtn?.id,
            submitBtnType: submitBtn?.type
        });

        if (!submitBtn || !btnContent || !btnLoading) {
            console.error('❌ Submit button elements not found');
            this.showFormAlert('danger', 'Form elements not found. Please refresh the page.');
            return;
        }

        console.log('⏳ Setting loading state...');
        btnContent.style.display = 'none';
        btnLoading.classList.remove('d-none');
        submitBtn.disabled = true;

        try {
            const providerType = document.getElementById('provider_type').value;
            const instanceName = document.getElementById('instance_name').value;

            console.log('📋 Form data collected:', { 
                providerType, 
                instanceName,
                providerTypeElement: !!document.getElementById('provider_type'),
                instanceNameElement: !!document.getElementById('instance_name')
            });

            // Validate required fields
            if (!providerType) {
                console.log('❌ Provider type validation failed');
                this.showFormAlert('danger', 'Please select a provider type');
                this.resetButtonState(submitBtn, btnContent, btnLoading);
                return;
            }

            if (!instanceName) {
                console.log('❌ Instance name validation failed');
                this.showFormAlert('danger', 'Please enter an instance name');
                this.resetButtonState(submitBtn, btnContent, btnLoading);
                return;
            }

            const isEdit = window.currentEditingProvider != null; // Using != to catch both null and undefined
            console.log('📝 Edit mode:', isEdit, 'currentEditingProvider:', window.currentEditingProvider);

            const formData = {
                provider_type: providerType,
                instance_name: instanceName,
            };
            // A new provider starts on. An edit leaves the on/off switch alone: it is the
            // provider toggle's to set, and sending true here re-enabled a disabled
            // provider (and with it every model) whenever its form was saved.
            if (!isEdit) formData.enabled = true;

            // Collect provider-specific fields
            const fields = ['endpoint', 'api_key', 'azure_backend', 'region', 'access_key_id', 'secret_access_key', 'base_url', 'provider_name'];
            fields.forEach(field => {
                const element = document.getElementById(field);
                if (element && element.value.trim()) {
                    const value = element.value.trim();
                    // Never submit the masked placeholder as a credential; omitting
                    // the field keeps the currently stored secret unchanged.
                    if (SECRET_FIELDS.includes(field) && value === SECRET_MASK) {
                        return;
                    }
                    formData[field] = value;
                }
            });

            // For providers, set provider_name to provider_type if not already set
            if (!formData.provider_name) {
                formData.provider_name = providerType;
            }

            // Handle Azure-specific fields
            if (providerType === 'azure') {
                const azureBackendElement = document.getElementById('azure_backend');
                if (azureBackendElement && azureBackendElement.value) {
                    formData.azure_backend = azureBackendElement.value;
                }

                // Handle dynamic discovery setting
                const dynamicDiscoveryElement = document.getElementById('dynamic_discovery');
                if (dynamicDiscoveryElement) {
                    formData.dynamic_discovery = dynamicDiscoveryElement.checked;
                }

                // Handle Azure Management API fields
                const azureFields = ['discovery_api_version'];
                azureFields.forEach(field => {
                    const element = document.getElementById(field);
                    if (element && element.value.trim()) {
                        formData[field] = element.value.trim();
                    }
                });

                const deploymentGroups = formData.dynamic_discovery === true
                    ? []
                    : window.AzureManager.getDeploymentGroups();
                const allDeployments = deploymentGroups.flatMap(group => group.deployments);

                // Validate deployment name format
                const deploymentNamePattern = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;
                const invalidNames = allDeployments.filter(d => !deploymentNamePattern.test(d));
                if (invalidNames.length > 0) {
                    this.showFormAlert('danger', `Invalid deployment name(s): ${invalidNames.join(', ')}. Names must start with a letter or digit and contain only letters, digits, hyphens, underscores, or dots.`);
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }

                const noApis = deploymentGroups.filter(group => group.apis.length === 0 && !group.legacy_mode);
                if (noApis.length > 0) {
                    this.showFormAlert('danger', `Select at least one API for: ${noApis.flatMap(group => group.deployments).join(', ')}`);
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }

                if (formData.dynamic_discovery !== true && allDeployments.length === 0) {
                    this.showFormAlert('danger', 'Please provide at least one deployment when manual discovery is selected.');
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }

                formData.deployment_groups = deploymentGroups;
            }

            // Custom providers: a filled base URL means that API is supported natively.
            if (providerType === 'custom') {
                const openaiBaseUrl = (document.getElementById('base_url')?.value || '').trim();
                const anthropicBaseUrl = (document.getElementById('anthropic_base_url')?.value || '').trim();
                if (!openaiBaseUrl && !anthropicBaseUrl) {
                    this.showFormAlert('danger', 'Please provide an OpenAI base URL, an Anthropic base URL, or both');
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }
                const supportedApis = [];
                if (openaiBaseUrl) supportedApis.push('openai');
                if (anthropicBaseUrl) supportedApis.push('anthropic');
                formData.supported_apis = supportedApis;
                // Empty strings clear previously saved values.
                formData.base_url = openaiBaseUrl;
                formData.anthropic_base_url = anthropicBaseUrl;
                const anthropicAuthStyle = document.getElementById('anthropic_auth_style');
                formData.anthropic_auth_style = anthropicAuthStyle ? anthropicAuthStyle.value : '';
            }

            const url = isEdit && window.currentEditingProvider ? 
                `/admin/providers?provider_key=${encodeURIComponent(window.currentEditingProvider.provider_key)}` : 
                '/admin/providers';
            const method = isEdit ? 'PUT' : 'POST';

            // NOTE: never log `formData` here — it contains provider API keys/secrets.
            console.log('🌐 Sending request:', { url, method });

            const response = await fetch(url, {
                method: method,
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            console.log('📡 Response received:', { status: response.status, ok: response.ok });

            if (response.ok) {
                const result = await response.json();
                console.log('✅ Success response:', result);

                let successMsg = `Provider ${isEdit ? 'updated' : 'created'} successfully.`;
                if (Array.isArray(result.deployment_groups)) {
                    const apiLabels = {
                        chat: '/chat/completions', responses: '/responses', anthropic: '/messages',
                        completions: '/completions', embeddings: '/embeddings', images: '/images/*',
                        speech: '/audio/speech', transcription: '/audio/transcriptions',
                    };
                    result.deployment_groups.forEach(group => {
                        const apis = (group.apis || []).length
                            ? group.apis.map(api => apiLabels[api] || api).join(', ')
                            : `legacy ${group.legacy_mode || ''} (all APIs)`;
                        successMsg += `<br>${apis}: ${(group.deployments || []).join(', ')}`;
                    });
                }
                if (result.supported_apis && result.supported_apis.length > 0) {
                    successMsg += `<br>Supported APIs: ${result.supported_apis.join(', ')}`;
                }

                this.showFormAlert('success', successMsg);
                setTimeout(() => {
                    window.ProviderManager.cancelProviderForm();
                    window.ProviderManager.loadUnifiedProviderData();
                }, 2500);
            } else {
                const error = await response.json();
                console.error('❌ Error response:', error);
                
                // Special handling for "Provider already exists" error
                if (response.status === 400 && error.detail && error.detail.includes('Provider already exists')) {
                    const providerKey = `${formData.provider_type}:${formData.provider_name}`;
                    await this.showProviderExistsModal(formData.provider_name, providerKey);
                } else {
                    this.showFormAlert('danger', `Failed to ${isEdit ? 'update' : 'create'} provider: ${error.detail || 'Unknown error'}`);
                }
            }
        } catch (error) {
            console.error('❌ Network/JS error:', error);
            this.showFormAlert('danger', `Network error: ${error.message}`);
        } finally {
            console.log('🔄 Resetting button state');
            // Reset button state
            this.resetButtonState(submitBtn, btnContent, btnLoading);
            // Reset submission flag
            this.isSubmitting = false;
        }
    }

    async switchToUpdateMode(providerKey) {
        console.log('🔄 Switching to update mode for provider:', providerKey);
        this.clearFormAlerts();
        
        try {
            // Load the existing provider data
            const response = await fetch(`/admin/providers/detail?provider_key=${encodeURIComponent(providerKey)}`, {
                credentials: 'include'
            });

            if (response.ok) {
                const provider = await response.json();
                
                // Switch to edit mode
                window.currentEditingProvider = provider;
                document.getElementById('form-title').textContent = 'Edit Provider';
                document.getElementById('submit-text').textContent = 'Update Provider';
                
                // Populate the form with existing data
                this.populateProviderFields(provider);
                
                this.showFormAlert('info', 'Switched to update mode. You can now modify the existing provider.');
            } else {
                const error = await response.json();
                this.showFormAlert('danger', `Failed to load provider: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('❌ Error loading provider for update:', error);
            this.showFormAlert('danger', `Error loading provider: ${error.message}`);
        }
    }

    async showProviderExistsModal(providerName, providerKey) {
        const title = 'Provider Already Exists';
        const message = `Provider "${providerName}" already exists in the system. Would you like to update the existing provider instead?`;
        
        // Create a custom modal for this specific case
        const modal = document.getElementById('confirmModal');
        const modalTitle = document.getElementById('confirmModalTitle');
        const modalMessage = document.getElementById('confirmModalMessage');
        const confirmBtn = document.getElementById('confirmActionBtn');
        const cancelBtn = document.getElementById('confirmCancelBtn');

        // Set modal content
        modalTitle.textContent = title;
        modalMessage.textContent = message;

        // Update button text and styles
        confirmBtn.innerHTML = '<i class="fas fa-edit me-1"></i>Update Existing';
        confirmBtn.className = 'btn btn-primary';
        cancelBtn.innerHTML = '<i class="fas fa-times me-1"></i>Cancel';

        return new Promise((resolve) => {
            const handleConfirm = async () => {
                cleanup();
                try {
                    await this.switchToUpdateMode(providerKey);
                    resolve(true);
                } catch (error) {
                    console.error('Error switching to update mode:', error);
                    this.showFormAlert('danger', `Error switching to update mode: ${error.message}`);
                    resolve(false);
                }
            };

            const handleCancel = () => {
                cleanup();
                this.showFormAlert('info', 'Provider creation cancelled.');
                resolve(false);
            };

            const cleanup = () => {
                confirmBtn.removeEventListener('click', handleConfirm);
                cancelBtn.removeEventListener('click', handleCancel);
                modal.removeEventListener('hidden.bs.modal', handleCancel);
                const bootstrapModal = bootstrap.Modal.getInstance(modal);
                if (bootstrapModal) {
                    bootstrapModal.hide();
                }
            };

            // Add event listeners
            confirmBtn.addEventListener('click', handleConfirm);
            cancelBtn.addEventListener('click', handleCancel);
            modal.addEventListener('hidden.bs.modal', handleCancel, { once: true });

            // Show modal
            const bootstrapModal = new bootstrap.Modal(modal);
            bootstrapModal.show();
        });
    }

    resetButtonState(submitBtn, btnContent, btnLoading) {
        if (submitBtn && btnContent && btnLoading) {
            btnContent.style.display = 'flex';
            btnLoading.classList.add('d-none');
            submitBtn.disabled = false;
        }
    }

    showFormAlert(type, message) {
        // Clear existing alerts first to prevent duplicates
        this.clearFormAlerts();
        
        const alertsContainer = document.getElementById('form-alerts');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;
        alertsContainer.appendChild(alert);

        // Auto-remove after 10 seconds (increased for complex messages with buttons)
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 10000);
    }

    clearFormAlerts() {
        const alertsContainer = document.getElementById('form-alerts');
        if (alertsContainer) {
            alertsContainer.innerHTML = '';
        }
    }

    populateProviderFields(provider) {
        // Populate common fields
        const fields = ['endpoint', 'api_key', 'azure_backend', 'region', 'access_key_id', 'secret_access_key', 'base_url', 'provider_name'];
        fields.forEach(field => {
            const element = document.getElementById(field);
            if (element && provider[field]) {
                // Secret fields come back as the fixed mask, never the real value.
                // Leave them empty so an untouched edit keeps the stored secret
                // instead of overwriting it with the mask string.
                if (SECRET_FIELDS.includes(field) && provider[field] === SECRET_MASK) {
                    element.value = '';
                    element.placeholder = 'Leave blank to keep the current value';
                    element.required = false;
                    return;
                }
                element.value = provider[field];
            }
        });

        // Handle provider-specific fields
        if (provider.provider_type === 'azure') {
            const azureBackendElement = document.getElementById('azure_backend');
            if (azureBackendElement) {
                azureBackendElement.value = provider.azure_backend || 'openai';
            }

            // Handle dynamic discovery
            const dynamicDiscoveryElement = document.getElementById('dynamic_discovery');
            if (dynamicDiscoveryElement) {
                // Explicitly check for true - if undefined/null, default to unchecked
                dynamicDiscoveryElement.checked = provider.dynamic_discovery === true;
                window.AzureManager.toggleDynamicDiscovery();
            }

            // Handle Azure Management API fields
            const azureFields = ['discovery_api_version'];
            azureFields.forEach(field => {
                const element = document.getElementById(field);
                if (element && provider[field]) {
                    element.value = provider[field];
                }
            });

            if (provider.dynamic_discovery !== true) {
                window.AzureManager.setDeploymentGroups(provider.deployment_groups);
            }
        }

        // Custom providers: each filled base URL marks a natively supported API.
        if (provider.provider_type === 'custom') {
            const apis = Array.isArray(provider.supported_apis) ? provider.supported_apis : ['openai'];
            const baseUrl = document.getElementById('base_url');
            const anthropicBaseUrl = document.getElementById('anthropic_base_url');
            const anthropicAuthStyle = document.getElementById('anthropic_auth_style');
            if (baseUrl) baseUrl.value = apis.includes('openai') ? (provider.base_url || '') : '';
            if (anthropicBaseUrl) {
                // Older records enabled native Anthropic without a separate URL
                // (it was derived from base_url); show that URL explicitly.
                anthropicBaseUrl.value = apis.includes('anthropic')
                    ? (provider.anthropic_base_url || provider.base_url || '')
                    : '';
            }
            if (anthropicAuthStyle) anthropicAuthStyle.value = provider.anthropic_auth_style || '';
        }
    }



}

// Create global instance
window.ProviderFormManager = new ProviderFormManager();

// Export functions for global access
window.updateProviderForm = () => window.ProviderFormManager.updateProviderForm();
window.submitProviderForm = (event) => window.ProviderFormManager.submitProviderForm(event);
window.showFormAlert = (type, message) => window.ProviderFormManager.showFormAlert(type, message);
window.clearFormAlerts = () => window.ProviderFormManager.clearFormAlerts();

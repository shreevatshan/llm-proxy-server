/**
 * Provider Forms Module
 * Handles provider form generation, validation, and submission
 */

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
                    <div class="mb-3">
                        <label for="openai_deployments" class="form-label">
                            <i class="fas fa-robot"></i> OpenAI Deployments
                        </label>
                        <input type="text" id="openai_deployments" class="form-control" 
                               placeholder="gpt-4o, gpt-4.1, o4-mini">
                        <div class="form-text">Comma-separated Azure OpenAI or Foundry chat model deployments.</div>
                    </div>
                    <div class="mb-3" id="anthropic-deployments-group" style="display: none;">
                        <label for="anthropic_deployments" class="form-label">
                            <i class="fas fa-brain"></i> Anthropic Deployments
                        </label>
                        <input type="text" id="anthropic_deployments" class="form-control" 
                               placeholder="claude-sonnet-4-5, claude-3-7-sonnet">
                        <div class="form-text">Comma-separated Claude / Anthropic model deployments for Azure Foundry.</div>
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
            <div class="row">
                <div class="col-md-6">
                    <div class="mb-3">
                        <label for="base_url" class="form-label">
                            <i class="fas fa-link"></i> Base URL <span class="text-danger">*</span>
                        </label>
                        <input type="text" id="base_url" class="form-control" 
                               placeholder="e.g., https://api.openai.com/v1 or http://localhost:11434/v1" required>
                        <div class="form-text">Complete server URL including /v1 path</div>
                    </div>
                </div>
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
            <div class="mb-3">
                <label class="form-label">
                    <i class="fas fa-plug"></i> Supported APIs <span class="text-danger">*</span>
                </label>
                <div class="form-text mb-2">Select which API formats this provider supports</div>
                <div class="d-flex gap-3">
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="supported_api_openai" value="openai" checked>
                        <label class="form-check-label" for="supported_api_openai">
                            <i class="fas fa-robot me-1"></i> OpenAI API
                        </label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="supported_api_anthropic" value="anthropic">
                        <label class="form-check-label" for="supported_api_anthropic">
                            <i class="fas fa-brain me-1"></i> Anthropic API
                        </label>
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
                enabled: true
            };

            // Collect provider-specific fields
            const fields = ['endpoint', 'api_key', 'azure_backend', 'region', 'access_key_id', 'secret_access_key', 'base_url', 'provider_name'];
            fields.forEach(field => {
                const element = document.getElementById(field);
                if (element && element.value.trim()) {
                    formData[field] = element.value.trim();
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

                const openaiDeploymentsElement = document.getElementById('openai_deployments');
                const anthropicDeploymentsElement = document.getElementById('anthropic_deployments');
                const openaiDeployments = openaiDeploymentsElement && openaiDeploymentsElement.value.trim()
                    ? openaiDeploymentsElement.value.split(',').map(d => d.trim()).filter(d => d)
                    : [];
                const anthropicDeployments = anthropicDeploymentsElement && anthropicDeploymentsElement.value.trim()
                    ? anthropicDeploymentsElement.value.split(',').map(d => d.trim()).filter(d => d)
                    : [];

                // Validate deployment name format
                const deploymentNamePattern = /^[a-zA-Z0-9][a-zA-Z0-9._-]*$/;
                const allDeployments = [...openaiDeployments, ...anthropicDeployments];
                const invalidNames = allDeployments.filter(d => !deploymentNamePattern.test(d));
                if (invalidNames.length > 0) {
                    this.showFormAlert('danger', `Invalid deployment name(s): ${invalidNames.join(', ')}. Names must start with a letter or digit and contain only letters, digits, hyphens, underscores, or dots.`);
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }

                if (formData.dynamic_discovery !== true) {
                    if (formData.azure_backend === 'openai' && openaiDeployments.length === 0) {
                        this.showFormAlert('danger', 'Please provide at least one OpenAI deployment when manual discovery is selected.');
                        this.resetButtonState(submitBtn, btnContent, btnLoading);
                        return;
                    }
                    if (formData.azure_backend === 'foundry' && openaiDeployments.length === 0 && anthropicDeployments.length === 0) {
                        this.showFormAlert('danger', 'Please provide at least one OpenAI or Anthropic deployment when manual discovery is selected.');
                        this.resetButtonState(submitBtn, btnContent, btnLoading);
                        return;
                    }
                }

                formData.openai_deployments = openaiDeployments;
                formData.anthropic_deployments = anthropicDeployments;
                formData.deployments = [...openaiDeployments, ...anthropicDeployments.filter(d => !openaiDeployments.includes(d))];
            }

            // Handle supported APIs checkboxes (for custom provider type)
            if (providerType === 'custom') {
                const supportedApis = [];
                const openaiCheck = document.getElementById('supported_api_openai');
                const anthropicCheck = document.getElementById('supported_api_anthropic');
                if (openaiCheck && openaiCheck.checked) supportedApis.push('openai');
                if (anthropicCheck && anthropicCheck.checked) supportedApis.push('anthropic');
                if (supportedApis.length === 0) {
                    this.showFormAlert('danger', 'Please select at least one supported API');
                    this.resetButtonState(submitBtn, btnContent, btnLoading);
                    return;
                }
                formData.supported_apis = supportedApis;
            }

            const url = isEdit && window.currentEditingProvider ? 
                `/admin/providers?provider_key=${encodeURIComponent(window.currentEditingProvider.provider_key)}` : 
                '/admin/providers';
            const method = isEdit ? 'PUT' : 'POST';

            console.log('🌐 Sending request:', { url, method, formData });

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
                if (result.openai_deployments && result.openai_deployments.length > 0) {
                    successMsg += `<br>OpenAI deployments: ${result.openai_deployments.join(', ')}`;
                }
                if (result.anthropic_deployments && result.anthropic_deployments.length > 0) {
                    successMsg += `<br>Anthropic deployments: ${result.anthropic_deployments.join(', ')}`;
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

            const openaiDeploymentsElement = document.getElementById('openai_deployments');
            if (openaiDeploymentsElement && provider.openai_deployments && Array.isArray(provider.openai_deployments)) {
                openaiDeploymentsElement.value = provider.openai_deployments.join(', ');
            }
            const anthropicDeploymentsElement = document.getElementById('anthropic_deployments');
            if (anthropicDeploymentsElement && provider.anthropic_deployments && Array.isArray(provider.anthropic_deployments)) {
                anthropicDeploymentsElement.value = provider.anthropic_deployments.join(', ');
            }
        }

        // Handle supported APIs checkboxes for custom providers
        if (provider.provider_type === 'custom' && provider.supported_apis) {
            const apis = Array.isArray(provider.supported_apis) ? provider.supported_apis : [];
            const openaiCheck = document.getElementById('supported_api_openai');
            const anthropicCheck = document.getElementById('supported_api_anthropic');
            if (openaiCheck) openaiCheck.checked = apis.includes('openai');
            if (anthropicCheck) anthropicCheck.checked = apis.includes('anthropic');
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

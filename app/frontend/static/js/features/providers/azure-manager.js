/**
 * Azure Provider Management Module
 * Handles Azure-specific functionality including dynamic discovery and setup instructions
 */

class AzureManager {
    toggleDynamicDiscovery() {
        console.log('toggleDynamicDiscovery called'); // Debug log
        
        const dynamicDiscoveryCheckbox = document.getElementById('dynamic_discovery');
        const azureBackendSelect = document.getElementById('azure_backend');
        const manualDeployments = document.getElementById('manual-deployments');
        const azureManagementConfig = document.getElementById('azure-management-config');
        const anthropicDeploymentsGroup = document.getElementById('anthropic-deployments-group');
        const openaiDeploymentsField = document.getElementById('openai_deployments');
        const anthropicDeploymentsField = document.getElementById('anthropic_deployments');
        const azureBackend = azureBackendSelect?.value || 'openai';

        console.log('Elements found:', {
            checkbox: !!dynamicDiscoveryCheckbox,
            backend: azureBackend,
            manual: !!manualDeployments,
            config: !!azureManagementConfig,
            checked: dynamicDiscoveryCheckbox?.checked
        }); // Debug log

        if (anthropicDeploymentsGroup) {
            const showAnthropic = azureBackend === 'foundry';
            anthropicDeploymentsGroup.style.display = showAnthropic ? 'block' : 'none';
            if (!showAnthropic && anthropicDeploymentsField) {
                anthropicDeploymentsField.value = '';
            }
        }

        if (dynamicDiscoveryCheckbox && manualDeployments && azureManagementConfig) {
            if (dynamicDiscoveryCheckbox.checked) {
                // Dynamic discovery enabled - Foundry uses the models API, Azure OpenAI uses the Management API.
                manualDeployments.style.display = 'none';
                azureManagementConfig.style.display = azureBackend === 'openai' ? 'block' : 'none';

                if (openaiDeploymentsField) {
                    openaiDeploymentsField.value = '';
                    openaiDeploymentsField.removeAttribute('required');
                    openaiDeploymentsField.disabled = true;
                }
                if (anthropicDeploymentsField) {
                    anthropicDeploymentsField.value = '';
                    anthropicDeploymentsField.removeAttribute('required');
                    anthropicDeploymentsField.disabled = true;
                }
            } else {
                // Dynamic discovery disabled - show manual deployments, hide Azure Management API fields
                manualDeployments.style.display = 'block';
                azureManagementConfig.style.display = 'none';

                if (openaiDeploymentsField) {
                    openaiDeploymentsField.disabled = false;
                    if (azureBackend === 'openai') {
                        openaiDeploymentsField.setAttribute('required', 'required');
                    } else {
                        openaiDeploymentsField.removeAttribute('required');
                    }
                }
                if (anthropicDeploymentsField) {
                    anthropicDeploymentsField.disabled = false;
                    anthropicDeploymentsField.removeAttribute('required');
                }
            }

            const managementFieldIds = ['subscription_id', 'resource_group', 'account_name', 'client_id', 'client_secret', 'tenant_id'];
            managementFieldIds.forEach(fieldId => {
                const element = document.getElementById(fieldId);
                if (!element) return;
                if (dynamicDiscoveryCheckbox.checked && azureBackend === 'openai') {
                    element.setAttribute('required', 'required');
                } else {
                    element.removeAttribute('required');
                }
            });
        } else {
            console.error('One or more required elements not found for toggleDynamicDiscovery');
        }
    }

    showAzureSetupInstructions() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'azureSetupModal';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'azureSetupModalLabel');
        modal.setAttribute('aria-hidden', 'true');

        modal.innerHTML = `
            <div class="modal-dialog modal-lg modal-dialog-scrollable">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="azureSetupModalLabel">
                            <i class="fas fa-cloud text-primary me-2"></i>
                            Azure AD App Registration Setup Instructions
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            <strong>Prerequisites:</strong> You need Azure subscription access and permissions to create App Registrations and assign IAM roles.
                        </div>
                        
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-step-forward"></i> Step 1: Create Azure AD App Registration
                        </h6>
                        <ol class="mb-4">
                            <li>Go to <strong>Azure Portal</strong> → <strong>Azure Active Directory</strong> → <strong>App registrations</strong></li>
                            <li>Click <strong>"New registration"</strong></li>
                            <li>Enter details:
                                <ul>
                                    <li><strong>Name:</strong> "LLM-Proxy-Server" (or your preferred name)</li>
                                    <li><strong>Account types:</strong> "Accounts in this organizational directory only"</li>
                                    <li><strong>Redirect URI:</strong> Leave blank</li>
                                </ul>
                            </li>
                            <li>Click <strong>"Register"</strong></li>
                        </ol>
                        
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-step-forward"></i> Step 2: Create Client Secret
                        </h6>
                        <ol class="mb-4">
                            <li>In your app registration, go to <strong>"Certificates & secrets"</strong></li>
                            <li>Click <strong>"New client secret"</strong></li>
                            <li>Enter details:
                                <ul>
                                    <li><strong>Description:</strong> "LLM Proxy Server Secret"</li>
                                    <li><strong>Expires:</strong> Choose appropriate duration (recommended: 24 months)</li>
                                </ul>
                            </li>
                            <li>Click <strong>"Add"</strong></li>
                            <li><strong>⚠️ IMPORTANT:</strong> Copy the secret value immediately - you won't be able to see it again!</li>
                        </ol>
                        
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-step-forward"></i> Step 3: Note Application Details
                        </h6>
                        <p class="mb-3">From the App registration <strong>Overview</strong> page, copy these values:</p>
                        <ul class="mb-4">
                            <li><strong>Application (client) ID</strong> → Use as <code>client_id</code></li>
                            <li><strong>Directory (tenant) ID</strong> → Use as <code>tenant_id</code></li>
                            <li><strong>Client secret value</strong> (from Step 2) → Use as <code>client_secret</code></li>
                        </ul>
                        
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-step-forward"></i> Step 4: Assign Permissions to Cognitive Services Resource
                        </h6>
                        <ol class="mb-4">
                            <li>Go to your <strong>Cognitive Services resource</strong> in Azure Portal</li>
                            <li>Click <strong>"Access control (IAM)"</strong></li>
                            <li>Click <strong>"Add role assignment"</strong></li>
                            <li>Configure the role assignment:
                                <ul>
                                    <li><strong>Role:</strong> "Cognitive Services Contributor" (or "Reader" for read-only access)</li>
                                    <li><strong>Assign access to:</strong> "User, group, or service principal"</li>
                                    <li><strong>Select:</strong> Search for and select your app registration name</li>
                                </ul>
                            </li>
                            <li>Click <strong>"Save"</strong></li>
                        </ol>
                        
                        <h6 class="text-primary mb-3">
                            <i class="fas fa-step-forward"></i> Step 5: Configure Provider
                        </h6>
                        <p class="mb-3">Use these values in the Azure provider configuration:</p>
                        <div class="table-responsive">
                            <table class="table table-sm table-bordered">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th>Source</th>
                                        <th>Description</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><code>subscription_id</code></td>
                                        <td>Azure Subscription</td>
                                        <td>Your Azure subscription ID (found in subscription overview)</td>
                                    </tr>
                                    <tr>
                                        <td><code>resource_group</code></td>
                                        <td>Cognitive Services Resource</td>
                                        <td>Resource group containing your Cognitive Services resource</td>
                                    </tr>
                                    <tr>
                                        <td><code>account_name</code></td>
                                        <td>Cognitive Services Resource</td>
                                        <td>Name of your Cognitive Services resource</td>
                                    </tr>
                                    <tr>
                                        <td><code>client_id</code></td>
                                        <td>App Registration</td>
                                        <td>Application (client) ID from Step 3</td>
                                    </tr>
                                    <tr>
                                        <td><code>client_secret</code></td>
                                        <td>App Registration</td>
                                        <td>Client secret value from Step 2</td>
                                    </tr>
                                    <tr>
                                        <td><code>tenant_id</code></td>
                                        <td>App Registration</td>
                                        <td>Directory (tenant) ID from Step 3</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="alert alert-success mt-3">
                            <i class="fas fa-check-circle me-2"></i>
                            <strong>Success!</strong> Once configured, the system will automatically discover your Azure OpenAI deployments using the Azure Management API.
                        </div>
                        
                        <div class="alert alert-warning mt-3">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            <strong>Security Note:</strong> Store your client secret securely. Consider using Azure Key Vault for production deployments.
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                            <i class="fas fa-times me-1"></i>
                            Close
                        </button>
                        <button type="button" class="btn btn-primary" onclick="window.open('https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade', '_blank')">
                            <i class="fas fa-external-link-alt me-1"></i>
                            Open Azure Portal
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add modal to document body
        document.body.appendChild(modal);

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();

        // Remove modal from DOM when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }
}

// Create global instance
window.AzureManager = new AzureManager();

// Export functions for backward compatibility
window.toggleDynamicDiscovery = () => window.AzureManager.toggleDynamicDiscovery();
window.showAzureSetupInstructions = () => window.AzureManager.showAzureSetupInstructions();

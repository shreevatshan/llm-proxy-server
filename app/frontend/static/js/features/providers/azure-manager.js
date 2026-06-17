/**
 * Azure Provider Management Module
 * Handles Azure-specific form toggling for dynamic discovery.
 */

class AzureManager {
    toggleDynamicDiscovery() {
        const dynamicDiscoveryCheckbox = document.getElementById('dynamic_discovery');
        const azureBackendSelect = document.getElementById('azure_backend');
        const manualDeployments = document.getElementById('manual-deployments');
        const azureDiscoveryConfig = document.getElementById('azure-management-config');
        const anthropicDeploymentsGroup = document.getElementById('anthropic-deployments-group');
        const openaiDeploymentsField = document.getElementById('openai_deployments');
        const anthropicDeploymentsField = document.getElementById('anthropic_deployments');
        const discoveryApiVersionField = document.getElementById('discovery_api_version');
        const azureBackend = azureBackendSelect?.value || 'openai';

        // Show Anthropic deployments field only for Foundry backend
        if (anthropicDeploymentsGroup) {
            const showAnthropic = azureBackend === 'foundry';
            anthropicDeploymentsGroup.style.display = showAnthropic ? 'block' : 'none';
            if (!showAnthropic && anthropicDeploymentsField) {
                anthropicDeploymentsField.value = '';
            }
        }

        if (!dynamicDiscoveryCheckbox || !manualDeployments || !azureDiscoveryConfig) {
            console.error('Required elements not found for toggleDynamicDiscovery');
            return;
        }

        if (dynamicDiscoveryCheckbox.checked) {
            // Dynamic discovery: show api-version field, hide manual deployment lists
            manualDeployments.style.display = 'none';
            azureDiscoveryConfig.style.display = 'block';

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
            if (discoveryApiVersionField) {
                discoveryApiVersionField.setAttribute('required', 'required');
            }
        } else {
            // Manual mode: show deployment lists, hide discovery config
            manualDeployments.style.display = 'block';
            azureDiscoveryConfig.style.display = 'none';

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
            if (discoveryApiVersionField) {
                discoveryApiVersionField.removeAttribute('required');
            }
        }
    }
}

// Create global instance
window.AzureManager = new AzureManager();

// Export for backward compatibility
window.toggleDynamicDiscovery = () => window.AzureManager.toggleDynamicDiscovery();

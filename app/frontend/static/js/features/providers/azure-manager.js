/**
 * Azure Provider Management Module
 * Handles Azure-specific form toggling for dynamic discovery and the
 * deployment-group rows (deployments + the upstream APIs they support).
 */

// Upstream APIs a deployment can be served over. `foundryOnly` APIs are hidden
// for the Azure OpenAI backend.
const AZURE_DEPLOYMENT_APIS = [
    { value: 'chat', label: '/chat/completions', short: '/chat/completions' },
    { value: 'responses', label: '/responses', short: '/responses' },
    { value: 'anthropic', label: '/messages', short: '/messages', foundryOnly: true },
    { value: 'completions', label: '/completions', short: '/completions' },
    { value: 'embeddings', label: '/embeddings', short: '/embeddings' },
    { value: 'images', label: '/images/generations, /edits, /variations', short: '/images/*' },
    { value: 'speech', label: '/audio/speech', short: '/audio/speech' },
    { value: 'transcription', label: '/audio/transcriptions, /translations', short: '/audio/transcriptions' },
];
const AZURE_DEFAULT_APIS = ['chat', 'responses'];
// Rows carried over from configs saved before deployment groups: they
// classify deployments without restricting their APIs, until APIs are picked.
const AZURE_LEGACY_MODE_LABELS = {
    openai: 'Legacy: OpenAI (all APIs)',
    responses: 'Legacy: OpenAI via /responses (all APIs)',
    anthropic: 'Legacy: Anthropic (all APIs)',
};

function azureEscape(text) {
    return window.UIUtils?.escapeHtml ? window.UIUtils.escapeHtml(String(text ?? '')) : String(text ?? '');
}

class AzureManager {
    constructor() {
        this._rowSeq = 0;
    }

    _backend() {
        return document.getElementById('azure_backend')?.value || 'openai';
    }

    _rowsContainer() {
        return document.getElementById('deployment-groups');
    }

    // ==================== Deployment group rows ====================

    addDeploymentGroup(deployments = [], apis = AZURE_DEFAULT_APIS, legacyMode = null) {
        const container = this._rowsContainer();
        if (!container) return null;
        const id = `deployment-group-${++this._rowSeq}`;
        const isLegacy = Boolean(AZURE_LEGACY_MODE_LABELS[legacyMode]);
        const selected = new Set(isLegacy ? [] : apis);
        const options = AZURE_DEPLOYMENT_APIS.map(api => `
            <li class="deployment-api-option" data-api="${api.value}"${api.foundryOnly ? ' data-foundry-only="true"' : ''}>
                <label class="dropdown-item form-check mb-0">
                    <input class="form-check-input" type="checkbox" value="${api.value}"${selected.has(api.value) ? ' checked' : ''}>
                    <span class="form-check-label">${azureEscape(api.label)}</span>
                </label>
            </li>`).join('');

        const row = document.createElement('div');
        row.className = 'deployment-group-row';
        row.id = id;
        row.innerHTML = `
            <input type="text" class="form-control deployment-group-names" aria-label="Deployments"
                   placeholder="gpt-4o, gpt-4.1, o4-mini" value="${azureEscape(deployments.join(', '))}">
            <div class="dropdown deployment-api-picker">
                <button class="form-select deployment-api-toggle" type="button" data-bs-toggle="dropdown"
                        data-bs-auto-close="outside" aria-expanded="false" aria-label="Supported APIs">
                    <span class="deployment-api-summary"></span>
                </button>
                <ul class="dropdown-menu dropdown-menu-end deployment-api-menu">${options}</ul>
            </div>
            <button type="button" class="btn btn-outline-secondary deployment-group-remove" title="Remove row" aria-label="Remove row">
                <i class="fas fa-times"></i>
            </button>`;

        if (isLegacy) row.dataset.legacyMode = legacyMode;

        row.querySelector('.deployment-api-menu').addEventListener('change', () => {
            // Picking APIs turns a legacy row into an explicit one.
            if (this._selectedApis(row).length) delete row.dataset.legacyMode;
            this._applyBackendToRow(row);
        });
        row.querySelector('.deployment-group-remove').addEventListener('click', () => {
            row.remove();
            if (!container.querySelector('.deployment-group-row')) {
                this.addDeploymentGroup();
            }
        });

        container.appendChild(row);
        this._applyBackendToRow(row);
        return row;
    }

    setDeploymentGroups(groups) {
        const container = this._rowsContainer();
        if (!container) return;
        container.innerHTML = '';
        (Array.isArray(groups) ? groups : []).forEach(group => {
            const names = Array.isArray(group?.deployments) ? group.deployments : [];
            const apis = Array.isArray(group?.apis) ? group.apis : [];
            if (names.length) this.addDeploymentGroup(names, apis, group?.legacy_mode || null);
        });
        if (!container.querySelector('.deployment-group-row')) {
            this.addDeploymentGroup();
        }
    }

    /** Rows as [{deployments, apis[, legacy_mode]}], skipping rows with no deployments. */
    getDeploymentGroups() {
        const container = this._rowsContainer();
        if (!container) return [];
        return Array.from(container.querySelectorAll('.deployment-group-row')).map(row => {
            const group = {
                deployments: row.querySelector('.deployment-group-names').value
                    .split(',').map(d => d.trim()).filter(d => d),
                apis: this._selectedApis(row),
            };
            if (!group.apis.length && row.dataset.legacyMode) group.legacy_mode = row.dataset.legacyMode;
            return group;
        }).filter(group => group.deployments.length > 0);
    }

    /** Every checked API, including ones the current backend hides: switching
     *  backends only changes what is offered, never what a row stores. */
    _selectedApis(row) {
        return Array.from(row.querySelectorAll('.deployment-api-option input:checked')).map(cb => cb.value);
    }

    _renderSummary(row) {
        const summary = row.querySelector('.deployment-api-summary');
        const apis = this._selectedApis(row);
        const foundry = this._backend() === 'foundry';
        const labels = AZURE_DEPLOYMENT_APIS.filter(api => apis.includes(api.value))
            .map(api => (api.foundryOnly && !foundry ? `${api.short} (Foundry only)` : api.short));
        const legacyLabel = AZURE_LEGACY_MODE_LABELS[row.dataset.legacyMode];
        summary.textContent = labels.length ? labels.join(', ') : (legacyLabel || 'Select APIs');
        summary.classList.toggle('is-placeholder', labels.length === 0 && !legacyLabel);
        summary.classList.toggle('is-legacy', labels.length === 0 && Boolean(legacyLabel));
    }

    _applyBackendToRow(row) {
        const foundry = this._backend() === 'foundry';
        row.querySelectorAll('.deployment-api-option[data-foundry-only="true"]').forEach(option => {
            // A stored selection stays visible (and uncheckable) on any backend.
            const checked = option.querySelector('input').checked;
            option.style.display = foundry || checked ? '' : 'none';
        });
        this._renderSummary(row);
    }

    // ==================== Form toggling ====================

    toggleDynamicDiscovery() {
        const dynamicDiscoveryCheckbox = document.getElementById('dynamic_discovery');
        const manualDeployments = document.getElementById('manual-deployments');
        const azureDiscoveryConfig = document.getElementById('azure-management-config');
        const discoveryApiVersionField = document.getElementById('discovery_api_version');
        const container = this._rowsContainer();

        if (container) {
            if (!container.querySelector('.deployment-group-row')) {
                this.addDeploymentGroup();
            }
            container.querySelectorAll('.deployment-group-row').forEach(row => this._applyBackendToRow(row));
        }

        if (!dynamicDiscoveryCheckbox || !manualDeployments || !azureDiscoveryConfig) {
            console.error('Required elements not found for toggleDynamicDiscovery');
            return;
        }

        const dynamicDiscovery = dynamicDiscoveryCheckbox.checked;
        // Dynamic discovery: show api-version field, hide manual deployment rows
        manualDeployments.style.display = dynamicDiscovery ? 'none' : 'block';
        azureDiscoveryConfig.style.display = dynamicDiscovery ? 'block' : 'none';
        manualDeployments.querySelectorAll('input, button').forEach(el => {
            el.disabled = dynamicDiscovery;
        });
        if (dynamicDiscovery) {
            this.setDeploymentGroups([]);
            manualDeployments.querySelectorAll('input, button').forEach(el => { el.disabled = true; });
        }
        if (discoveryApiVersionField) {
            if (dynamicDiscovery) {
                discoveryApiVersionField.setAttribute('required', 'required');
            } else {
                discoveryApiVersionField.removeAttribute('required');
            }
        }
    }
}

// Create global instance
window.AzureManager = new AzureManager();

// Export for backward compatibility
window.toggleDynamicDiscovery = () => window.AzureManager.toggleDynamicDiscovery();

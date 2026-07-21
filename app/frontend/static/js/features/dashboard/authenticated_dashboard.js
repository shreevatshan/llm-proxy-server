let createApiKeyModal, apiKeyDisplayModal;
let currentUser = null;

document.addEventListener('DOMContentLoaded', function () {
    createApiKeyModal = new bootstrap.Modal(document.getElementById('createApiKeyModal'));
    apiKeyDisplayModal = new bootstrap.Modal(document.getElementById('apiKeyDisplayModal'));

    // tables.css hides all .tab-content by default; activate the first tab explicitly
    // so the API keys panel is visible without needing a tab switch first.
    showTab('api-keys');

    // Check authentication and load user data
    checkAuthAndLoadData();

    // Populate endpoints and models from embedded server-rendered data
    initEndpointsTab();
    initModelsTab();
});

async function checkAuthAndLoadData() {
    try {
        // Get current user info — authenticated via the HttpOnly cookie.
        const response = await makeAuthenticatedRequest('/auth/me');

        if (response.ok) {
            currentUser = await response.json();
            document.getElementById('welcomeMessage').textContent = `Welcome back, ${currentUser.username}!`;

            // Load API keys for regular users (admin users are redirected server-side)
            loadApiKeys();
        } else {
            // Not authenticated, redirect to login
            window.location.href = '/login';
        }
    } catch (error) {
        // Network error or invalid session, redirect to login
        window.location.href = '/login';
    }
}

function showCreateApiKeyModal() {
    createApiKeyModal.show();
}

async function loadApiKeys() {
    try {
        // Add cache-busting parameter to ensure fresh data
        const response = await makeAuthenticatedRequest(`/auth/api-keys?t=${Date.now()}`);
        const apiKeys = await response.json();

        const container = document.getElementById('apiKeysContainer');

        if (apiKeys.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-key fa-2x mb-3"></i>
                    <p>No API keys yet. Create your first one!</p>
                </div>
            `;
        } else {
            container.innerHTML = apiKeys.map((key, index) => `
                <div class="d-flex justify-content-between align-items-center py-3" style="${index < apiKeys.length - 1 ? 'border-bottom: 1px solid var(--mono-4);' : ''}">
                    <div class="flex-grow-1">
                        <strong>${escapeHtml(key.name)}</strong><br>
                        <div class="api-key-display" id="api-key-${escapeHtml(key.id)}">
                            <div class="mt-1" style="max-width: 400px;">
                                <input type="text" class="form-control form-control-sm" id="full-api-key-${escapeHtml(key.id)}" value="${escapeHtml(key.api_key_preview)}" readonly>
                            </div>
                        </div>
                        <small class="text-muted">
                            Created: ${new Date(key.created_at).toLocaleDateString()}
                            ${key.last_used ? `| Last used: ${new Date(key.last_used).toLocaleDateString()}` : ''}
                        </small>
                    </div>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-danger" data-action="delete-api-key" data-key-id="${escapeHtml(key.id)}" data-key-name="${escapeHtml(key.name)}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `).join('');

            // Bind handlers via data-* so no dynamic value is ever interpolated
            // into an inline handler.
            container.querySelectorAll('[data-action="delete-api-key"]').forEach(el => {
                el.addEventListener('click', () => deleteApiKey(el.dataset.keyId, el.dataset.keyName));
            });
        }
    } catch (error) {
        document.getElementById('apiKeysContainer').innerHTML = `
            <div class="alert alert-danger">
                Failed to load API keys. Please refresh the page.
            </div>
        `;
    }
}

document.getElementById('createApiKeyForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const name = formData.get('name');

    try {
        const response = await makeAuthenticatedRequest('/auth/api-keys', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: name })
        });

        const data = await response.json();

        if (response.ok) {
            createApiKeyModal.hide();
            document.getElementById('newApiKeyValue').value = data.api_key;
            apiKeyDisplayModal.show();
            await loadApiKeys(); // Refresh the list
            this.reset(); // Clear the form
        } else {
            window.UIUtils.showToast(data.detail || 'Failed to create API key', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    }
});

function copyApiKey() {
    const input = document.getElementById('newApiKeyValue');
    input.select();
    document.execCommand('copy');
    window.UIUtils.showToast('API key copied to clipboard!', 'success');
}

async function deleteApiKey(keyId, keyName) {
    const confirmed = await window.UIUtils.showConfirmModal(
        'Delete API Key',
        `Are you sure you want to delete the API key "${keyName}"? This action cannot be undone.`
    );
    if (!confirmed) return;

    try {
        const response = await makeAuthenticatedRequest(`/auth/api-keys?api_key_id=${keyId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            window.UIUtils.showToast('API key deleted successfully!', 'success');
            // Force a complete refresh of the API keys list
            await loadApiKeys();
        } else {
            const data = await response.json();
            window.UIUtils.showToast(data.detail || 'Failed to delete API key', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    }
}

// ================================================================ //
// Endpoints Tab — base-URL cards + collapsible endpoint lists
// Ported from app/frontend/templates/dashboard/endpoints.html
// ================================================================ //

function initEndpointsTab() {
    if (typeof _domain === 'undefined') return;

    const openaiBaseUrl    = `http://${_domain}:${_openaiPort}/v1`;
    const anthropicBaseUrl = `http://${_domain}:${_anthropicPort}`;
    const azureBaseUrl     = `http://${_domain}:${_azurePort}`;

    _setUrl('openai-base-url', openaiBaseUrl);
    _setUrl('anthropic-base-url', anthropicBaseUrl);
    _setUrl('azure-openai-base-url', azureBaseUrl);

    // Unified base URLs: every provider is also reachable through the single
    // management port under a path prefix (see create_management_app).
    if (typeof _managementPort !== 'undefined') {
        _setUrl('openai-unified-url',       `http://${_domain}:${_managementPort}/openai/v1`);
        _setUrl('anthropic-unified-url',    `http://${_domain}:${_managementPort}/anthropic`);
        _setUrl('azure-openai-unified-url', `http://${_domain}:${_managementPort}/azure-openai`);
    }

    _setCount('openai-count', endpointsData);
    _setCount('anthropic-count', anthropicEndpointsData);
    _setCount('azure-openai-count', azureEndpointsData);

    _renderEndpoints(endpointsData,          'openai-endpoints',    openaiBaseUrl);
    _renderEndpoints(anthropicEndpointsData,  'anthropic-endpoints', anthropicBaseUrl);
    _renderEndpoints(azureEndpointsData,      'azure-openai-endpoints', azureBaseUrl);
}

// Render a URL with syntax highlighting: dim the scheme, mute the shared host,
// and burn the distinguishing tail (port + path) bright. The raw URL is stashed
// on dataset.url so Copy still yields plain text.
function _setUrl(elementId, url) {
    const elm = document.getElementById(elementId);
    if (!elm) return;
    elm.dataset.url = url;
    const m = url.match(/^(https?:\/\/)([^\/:]+)(.*)$/);
    if (!m) { elm.textContent = url; return; }
    const [, scheme, host, rest] = m;
    elm.innerHTML =
        `<span class="u-scheme">${escapeHtml(scheme)}</span>` +
        `<span class="u-host">${escapeHtml(host)}</span>` +
        `<span class="u-key">${escapeHtml(rest)}</span>`;
}

function _setCount(elementId, endpoints) {
    const elm = document.getElementById(elementId);
    if (elm) elm.textContent = (endpoints || []).length;
}

function _renderEndpoints(endpoints, containerId, baseUrl) {
    const container = document.getElementById(containerId);
    if (!container) return;
    let html = '';
    (endpoints || []).forEach(ep => {
        const fullUrl = baseUrl + ep.path;
        html += `
            <div class="endpoint-list-item">
                <span class="endpoint-method ${escapeHtml(ep.method.toLowerCase())}">${escapeHtml(ep.method)}</span>
                <span class="endpoint-path">${escapeHtml(ep.path)}</span>
                <span class="endpoint-desc">${escapeHtml(ep.desc || '')}</span>
                <button class="copy-btn" data-url="${escapeHtml(fullUrl)}" onclick="copyText(this.getAttribute('data-url'), this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>`;
    });
    container.innerHTML = html;
}

function toggleEndpoints(section) {
    const container = document.getElementById(section + '-endpoints');
    const block     = document.getElementById(section + '-block');
    if (!container || !block) return;
    const foot   = block.querySelector('.provider-foot');
    const isOpen = block.classList.contains('expanded');
    if (isOpen) {
        container.style.maxHeight = container.scrollHeight + 'px';
        requestAnimationFrame(() => { container.style.maxHeight = '0'; });
        block.classList.remove('expanded');
        if (foot) foot.setAttribute('aria-expanded', 'false');
    } else {
        block.classList.add('expanded');
        container.style.maxHeight = container.scrollHeight + 'px';
        if (foot) foot.setAttribute('aria-expanded', 'true');
        container.addEventListener('transitionend', function handler() {
            if (block.classList.contains('expanded')) container.style.maxHeight = 'none';
            container.removeEventListener('transitionend', handler);
        });
    }
}

function handleEndpointCardKeydown(event, section) {
    if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggleEndpoints(section);
    }
}

function copyText(text, button) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        navigator.clipboard.writeText(text).then(() => showCopied(button, text)).catch(() => _fallbackCopyText(text, button));
    } else {
        _fallbackCopyText(text, button);
    }
}

function _fallbackCopyText(text, button) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    if (ok) showCopied(button, text);
    else showDashboardToast('Failed to copy to clipboard', 'error');
}

function showCopied(button, text) {
    const orig = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
    button.classList.add('copied');
    showDashboardToast(`Copied: ${text}`, 'success');
    setTimeout(() => { button.innerHTML = orig; button.classList.remove('copied'); }, 2000);
}

// ================================================================ //
// Models Tab — search input + live filtering
// Ported from app/frontend/templates/dashboard/models_search.html
// ================================================================ //

let _allModels = [];
let _filteredModels = [];
let _modelApiSupport = {};

function initModelsTab() {
    if (typeof modelsData === 'undefined') return;

    // Build support map and model list from embedded data
    _modelApiSupport = {};
    modelsData.forEach(m => { if (m.supported_apis) _modelApiSupport[m.id] = m.supported_apis; });

    _allModels = modelsData
        .map(m => ({ model_id: m.id, model_name: m.id }))
        .sort((a, b) => a.model_name.toLowerCase().localeCompare(b.model_name.toLowerCase()));

    _filteredModels = [..._allModels];
    _displayModels();

    const search = document.getElementById('model-search');
    if (search) search.addEventListener('input', _filterModels);
}

function _filterModels() {
    const term = (document.getElementById('model-search').value || '').toLowerCase().trim();
    _filteredModels = term
        ? _allModels.filter(m => m.model_name.toLowerCase().includes(term))
        : [..._allModels];
    _displayModels();
}

function _displayModels() {
    const listBody = document.querySelector('#model-list .card-body');
    const stats    = document.getElementById('search-stats');
    if (!listBody) return;

    const n = _filteredModels.length;
    if (stats) stats.textContent = `Showing ${n} enabled model${n !== 1 ? 's' : ''}`;

    if (n === 0) {
        listBody.innerHTML = `<div class="text-center py-5 text-muted"><i class="fas fa-search fa-3x mb-3"></i><p>No models found</p></div>`;
        return;
    }

    let html = '';
    _filteredModels.forEach(model => {
        const name = model.model_name;
        const apis = _modelApiSupport[name] || ['openai'];
        const badges = apis.map(a => `<span class="api-badge ${escapeHtml(a)}">${escapeHtml(a)}</span>`).join('');
        html += `
            <div class="model-list-item">
                <div class="model-info">
                    <div class="model-name">${escapeHtml(name)}</div>
                    <div class="api-badges">${badges}</div>
                </div>
                <button class="copy-button" data-model-name="${escapeHtml(name)}" onclick="copyModelName(this)">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>`;
    });
    listBody.innerHTML = html;
}

function copyModelName(button) {
    const name = button.getAttribute('data-model-name');
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        navigator.clipboard.writeText(name).then(() => _modelCopied(button, name)).catch(() => _modelFallbackCopy(name, button));
    } else {
        _modelFallbackCopy(name, button);
    }
}

function _modelCopied(button, text) {
    const orig = button.innerHTML;
    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
    button.classList.add('copied');
    showDashboardToast(`Copied: ${text}`, 'success');
    setTimeout(() => { button.innerHTML = orig; button.classList.remove('copied'); }, 2000);
}

function _modelFallbackCopy(text, button) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); ta.setSelectionRange(0, 99999);
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    if (ok) _modelCopied(button, text);
    else showDashboardToast('Failed to copy to clipboard', 'error');
}

// ================================================================ //
// Shared toast (used by endpoints + models tabs)
// Uses the same #toast-container already on the page
// ================================================================ //

function showDashboardToast(message, type) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    if (type === 'error') { toast.style.borderLeftColor = 'var(--mono-text-secondary)'; toast.style.borderLeftStyle = 'dashed'; }
    toast.innerHTML = `<i class="fas fa-${type === 'error' ? 'exclamation-circle' : 'check-circle'}"></i> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ================================================================ //
// Shared helpers
// ================================================================ //

// Quote-safe HTML escaping (encodes " ' ` in addition to < > &) so output is
// safe inside quoted attribute values as well as element text.
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}
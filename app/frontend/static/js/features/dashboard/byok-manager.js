/**
 * "My Providers" tab: a user's own provider instances (BYOK).
 *
 * Backed by /auth/providers (app/routes/user_providers.py). The add/edit form is
 * the admin provider form (provider-forms.js) in BYOK mode: the name input holds
 * a slug the server turns into `byok-{user_id}-{slug}`, and secrets are write-only
 * (the server returns them masked). Note: provider-forms.js declares the global
 * SECRET_MASK / SECRET_FIELDS constants; this file must not redeclare them.
 */

class ByokManager {
    constructor() {
        this._loaded = false;
        this._data = null;
    }

    _esc(value) {
        return window.UIUtils?.escapeHtml ? window.UIUtils.escapeHtml(String(value ?? '')) : String(value ?? '');
    }

    _toast(message, type = 'info') {
        window.UIUtils?.showToast(message, type);
    }

    async _request(url, options = {}) {
        const init = { ...options };
        if (init.body && typeof init.body !== 'string') {
            init.body = JSON.stringify(init.body);
            init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
        }
        const response = await makeAuthenticatedRequest(url, init);
        let payload = null;
        try { payload = await response.json(); } catch (_) { /* empty body */ }
        if (!response.ok) {
            const detail = payload && payload.detail;
            const message = typeof detail === 'string' ? detail
                : Array.isArray(detail) ? detail.map(d => d.msg || '').join('; ')
                : `Request failed (${response.status})`;
            throw new Error(message);
        }
        return payload;
    }

    _configureForm() {
        window.ProviderFormManager.configure({
            mode: 'byok',
            createUrl: '/auth/providers',
            updateUrl: key => `/auth/providers?provider_key=${encodeURIComponent(key)}`,
            detailUrl: key => `/auth/providers/detail?provider_key=${encodeURIComponent(key)}`,
            onSaved: () => {
                this.closeForm();
                this.load();
            },
        });
        const form = document.getElementById('provider-form');
        if (form && !form.dataset.byokBound) {
            form.dataset.byokBound = '1';
            form.addEventListener('submit', event => window.ProviderFormManager.submitProviderForm(event));
        }
    }

    async load() {
        this._loaded = true;
        this._configureForm();
        const loading = document.getElementById('byok-loading');
        try {
            this._data = await this._request('/auth/providers');
        } catch (error) {
            if (loading) loading.hidden = true;
            document.getElementById('byok-list').innerHTML =
                `<div class="alert alert-danger">Could not load your providers: ${this._esc(error.message)}</div>`;
            return;
        }
        if (loading) loading.hidden = true;
        const prefix = document.getElementById('byok-prefix');
        if (prefix) prefix.textContent = this._data.instance_prefix;
        this.render();
    }

    render() {
        const { providers, max_instances: max } = this._data;
        const list = document.getElementById('byok-list');
        const empty = document.getElementById('byok-empty');
        const addBtn = document.getElementById('byok-add-btn');
        document.getElementById('byok-count').textContent = `${providers.length} of ${max} providers`;
        if (addBtn) addBtn.disabled = providers.length >= max;
        empty.style.display = providers.length ? 'none' : 'block';
        list.innerHTML = providers.map(p => this._card(p)).join('');
    }

    _status(p) {
        if (p.admin_disabled) {
            return '<span class="badge bg-danger"><i class="fas fa-lock me-1"></i>Disabled by admin</span>';
        }
        if (!p.enabled) return '<span class="badge bg-secondary">Off</span>';
        if (p.load_error) return '<span class="badge bg-warning text-dark">Not serving</span>';
        if (p.sync_state === 'failed') return '<span class="badge bg-warning text-dark">Sync failed</span>';
        return '<span class="badge bg-success">On</span>';
    }

    _card(p) {
        const key = this._esc(p.provider_key);
        const jsKey = this._esc(JSON.stringify(p.provider_key));
        const typeLabel = {
            custom: 'OpenAI / Anthropic compatible', azure: 'Azure', google: 'Google AI',
            bedrock: 'AWS Bedrock', platformai: 'Zoho PlatformAI',
        }[p.provider_type] || p.provider_type;
        const problem = p.load_error || (p.sync_state === 'failed' ? p.sync_error : null);
        const locked = p.admin_disabled ? ' disabled' : '';
        return `
        <div class="card mb-3${p.enabled && !p.admin_disabled ? '' : ' opacity-75'}">
            <div class="card-body">
                <div class="d-flex flex-wrap justify-content-between align-items-start gap-2">
                    <div class="me-auto" style="min-width: 0;">
                        <h5 class="mb-1 text-break">
                            ${this._esc(p.slug)}
                            <span class="badge bg-info text-dark ms-1">BYOK</span>
                            ${this._status(p)}
                        </h5>
                        <div class="small text-muted">${this._esc(typeLabel)} · ${p.enabled_model_count}/${p.model_count} models on</div>
                        <div class="small mt-1">
                            Model id prefix:
                            <code class="text-break">${key}/</code>
                            <button type="button" class="btn btn-link btn-sm p-0 ms-1 align-baseline"
                                    onclick='window.ByokManager.copy(${jsKey} + "/")' title="Copy">
                                <i class="fas fa-copy"></i>
                            </button>
                        </div>
                        ${problem ? `<div class="small text-warning mt-1"><i class="fas fa-triangle-exclamation me-1"></i>${this._esc(problem)}</div>` : ''}
                    </div>
                    <div class="d-flex flex-wrap gap-1">
                        <button type="button" class="btn btn-sm btn-outline-secondary" onclick='window.ByokManager.showModels(${jsKey})'>
                            <i class="fas fa-list me-1"></i>Models
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary"${locked} onclick='window.ByokManager.sync(${jsKey}, this)'>
                            <i class="fas fa-rotate me-1"></i>Sync
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-secondary"${locked} onclick='window.ByokManager.edit(${jsKey})'>
                            <i class="fas fa-pen me-1"></i>Edit
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-${p.enabled ? 'warning' : 'success'}"${locked}
                                onclick='window.ByokManager.toggle(${jsKey}, ${!p.enabled})'>
                            <i class="fas fa-power-off me-1"></i>${p.enabled ? 'Turn off' : 'Turn on'}
                        </button>
                        <button type="button" class="btn btn-sm btn-outline-danger" onclick='window.ByokManager.remove(${jsKey})'>
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>`;
    }

    copy(text) {
        navigator.clipboard?.writeText(text).then(
            () => this._toast('Copied', 'success'),
            () => this._toast('Copy failed', 'error'),
        );
    }

    // ---------------------------------------------------------------- form

    _resetForm() {
        window.currentEditingProvider = null;
        const form = document.getElementById('provider-form');
        form.reset();
        ['provider_type', 'instance_name'].forEach(id => { document.getElementById(id).disabled = false; });
        document.getElementById('provider-specific-fields').innerHTML = '';
        document.getElementById('provider-config-section').style.display = 'none';
        window.ProviderFormManager.clearFormAlerts();
    }

    openForm() {
        this._configureForm();
        this._resetForm();
        document.getElementById('form-title').textContent = 'Add provider';
        document.getElementById('submit-text').textContent = 'Add provider';
        const card = document.getElementById('byok-form-card');
        card.style.display = 'block';
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    closeForm() {
        this._resetForm();
        document.getElementById('byok-form-card').style.display = 'none';
    }

    async edit(providerKey) {
        this._configureForm();
        let provider;
        try {
            provider = await this._request(`/auth/providers/detail?provider_key=${encodeURIComponent(providerKey)}`);
        } catch (error) {
            this._toast(error.message, 'error');
            return;
        }
        this._resetForm();
        window.currentEditingProvider = provider;
        document.getElementById('form-title').textContent = `Edit ${provider.slug}`;
        document.getElementById('submit-text').textContent = 'Save changes';
        const type = document.getElementById('provider_type');
        type.value = provider.provider_type;
        window.ProviderFormManager.updateProviderForm();
        const name = document.getElementById('instance_name');
        name.value = provider.slug;
        // A provider's identity is fixed once created.
        type.disabled = true;
        name.disabled = true;
        window.ProviderFormManager.populateProviderFields(provider);
        const providerName = document.getElementById('provider_name');
        if (providerName) providerName.disabled = true;
        const card = document.getElementById('byok-form-card');
        card.style.display = 'block';
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ------------------------------------------------------------- actions

    _replace(provider) {
        const list = this._data.providers;
        const i = list.findIndex(p => p.provider_key === provider.provider_key);
        if (i >= 0) list[i] = provider;
        this.render();
    }

    async toggle(providerKey, enabled) {
        try {
            const provider = await this._request(
                `/auth/providers/toggle?provider_key=${encodeURIComponent(providerKey)}&enabled=${enabled}`,
                { method: 'PUT' },
            );
            this._replace(provider);
            this._toast(enabled ? 'Provider turned on' : 'Provider turned off', 'success');
        } catch (error) {
            this._toast(error.message, 'error');
        }
    }

    async sync(providerKey, button) {
        if (button) button.disabled = true;
        try {
            const provider = await this._request(
                `/auth/providers/sync?provider_key=${encodeURIComponent(providerKey)}`, { method: 'POST' },
            );
            this._replace(provider);
            if (provider.sync_state === 'failed' || provider.load_error) {
                this._toast(`Sync failed: ${provider.load_error || provider.sync_error || 'no models returned'}`, 'error');
            } else {
                this._toast(`Synced ${provider.model_count} models`, 'success');
            }
        } catch (error) {
            this._toast(error.message, 'error');
        } finally {
            if (button) button.disabled = false;
        }
    }

    async remove(providerKey) {
        const provider = this._data.providers.find(p => p.provider_key === providerKey);
        const ok = await window.UIUtils.showConfirmModal(
            'Delete provider',
            `Delete "${provider ? provider.slug : providerKey}" and its models? Its stored keys are deleted. Past usage stays in your history.`,
            'danger',
        );
        if (!ok) return;
        try {
            await this._request(`/auth/providers?provider_key=${encodeURIComponent(providerKey)}`, { method: 'DELETE' });
            this._toast('Provider deleted', 'success');
            this.load();
        } catch (error) {
            this._toast(error.message, 'error');
        }
    }

    async showModels(providerKey) {
        const body = document.getElementById('byokModelsBody');
        const provider = this._data.providers.find(p => p.provider_key === providerKey);
        document.getElementById('byokModelsTitle').textContent = `Models · ${provider ? provider.slug : providerKey}`;
        body.innerHTML = '<div class="text-muted">Loading…</div>';
        bootstrap.Modal.getOrCreateInstance(document.getElementById('byokModelsModal')).show();
        let data;
        try {
            data = await this._request(`/auth/providers/models?provider_key=${encodeURIComponent(providerKey)}`);
        } catch (error) {
            body.innerHTML = `<div class="alert alert-danger">${this._esc(error.message)}</div>`;
            return;
        }
        if (!data.models.length) {
            body.innerHTML = '<p class="text-muted mb-0">No models yet. Use Sync to discover them.</p>';
            return;
        }
        body.innerHTML = `
            <p class="small text-muted">Call a model by its full id, or by the bare name to share the
            load with the proxy's own instances of it.</p>
            <ul class="list-group">
            ${data.models.map((m, i) => `
                <li class="list-group-item d-flex justify-content-between align-items-center gap-2">
                    <code class="text-break small">${this._esc(m.model_id)}</code>
                    <div class="form-check form-switch mb-0">
                        <input class="form-check-input" type="checkbox" id="byok-model-${i}"
                               data-model-id="${this._esc(m.model_id)}" ${m.is_enabled ? 'checked' : ''}
                               aria-label="Enable ${this._esc(m.model_name)}">
                    </div>
                </li>`).join('')}
            </ul>`;
        body.querySelectorAll('input[data-model-id]').forEach(input => {
            input.addEventListener('change', async () => {
                input.disabled = true;
                try {
                    await this._request('/auth/providers/models/toggle', {
                        method: 'PUT', body: { model_id: input.dataset.modelId, enabled: input.checked },
                    });
                } catch (error) {
                    input.checked = !input.checked;
                    this._toast(error.message, 'error');
                } finally {
                    input.disabled = false;
                }
            });
        });
        // Counts on the cards follow the toggles once the modal closes.
        document.getElementById('byokModelsModal')
            .addEventListener('hidden.bs.modal', () => this.load(), { once: true });
    }
}

window.ByokManager = new ByokManager();

// Load when the tab is first opened (same pattern as websearch-prefs-manager.js).
document.addEventListener('DOMContentLoaded', function () {
    const originalShowTab = window.showTab;
    window.showTab = function (tabName) {
        originalShowTab(tabName);
        if (tabName === 'byok' && !window.ByokManager._loaded) {
            window.ByokManager.load();
        }
    };
});

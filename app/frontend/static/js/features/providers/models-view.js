/**
 * The Models view of the Providers & Models tab: every model, grouped by provider,
 * with its global on/off switch. A model is served only while it, its provider and
 * its model group are all on; the row says which of the other two is holding it back.
 * Profiles can still override all of this in Model Access.
 */
class ModelsView {
    constructor() {
        this._models = [];              // [{model_id, model_name, provider_key, is_enabled}]
        this._providerOff = new Set();  // provider_keys switched off
        this._newModelsOff = new Set(); // provider_keys whose new models start disabled
        this._groupOf = {};             // model_id -> {name, enabled} for named groups
        this._loadSeq = 0;
    }

    async load() {
        const seq = ++this._loadSeq;
        const list = document.getElementById('pm-model-list');
        try {
            const [mResp, pResp, gResp] = await Promise.all([
                fetch('/admin/models/all', { credentials: 'include' }),
                fetch('/admin/providers', { credentials: 'include' }),
                fetch('/admin/model-groups', { credentials: 'include' }),
            ]);
            if (seq !== this._loadSeq) return;   // a newer load is on its way
            if (!mResp.ok) {
                if (list) list.innerHTML = '<div class="text-danger small py-4 text-center">Couldn\'t load models. Retry.</div>';
                return;
            }
            this._models = (await mResp.json())
                .sort((a, b) => a.provider_key.localeCompare(b.provider_key) || a.model_id.localeCompare(b.model_id));
            const providers = pResp.ok ? await pResp.json() : [];
            this._providerOff = new Set(providers.filter(p => p.enabled === false).map(p => p.provider_key));
            this._newModelsOff = new Set(
                providers.filter(p => p.auto_enable_new_models === false).map(p => p.provider_key));
            this._groupOf = {};
            if (gResp.ok) {
                (await gResp.json()).filter(g => !g.is_default).forEach(g => {
                    (g.effective_models || []).forEach(mid => { this._groupOf[mid] = { name: g.name, enabled: g.enabled !== false }; });
                });
            }
            this.render();
        } catch (e) {
            console.error('ModelsView: failed to load models', e);
            if (list) list.innerHTML = '<div class="text-danger small py-4 text-center">Error loading models.</div>';
        }
    }

    render() {
        const container = document.getElementById('pm-model-list');
        if (!container) return;
        const summary = document.getElementById('pm-model-summary');
        const enabled = this._models.filter(m => m.is_enabled).length;
        if (summary) summary.textContent = `${enabled} of ${this._models.length} enabled`;

        const q = (document.getElementById('pm-model-search')?.value || '').trim().toLowerCase();
        const status = document.getElementById('pm-model-status')?.value || '';
        const shown = this._models.filter(m =>
            (!q || m.model_id.toLowerCase().includes(q) || (m.model_name || '').toLowerCase().includes(q))
            && (!status || (status === 'on') === !!m.is_enabled));

        if (!shown.length) {
            const why = this._models.length ? 'No models match.' : 'No models yet. Add a provider, then sync.';
            container.innerHTML = `<div class="text-muted small py-4 text-center">${why}</div>`;
            return;
        }

        const byProvider = new Map();
        shown.forEach(m => {
            if (!byProvider.has(m.provider_key)) byProvider.set(m.provider_key, []);
            byProvider.get(m.provider_key).push(m);
        });
        container.innerHTML = [...byProvider].map(([pk, models]) => this._section(pk, models)).join('');

        // Bind via data-* rather than interpolating ids into inline handlers.
        container.querySelectorAll('[data-action="toggle-model"]').forEach(el => {
            el.addEventListener('change', () => this.toggle(el.dataset.modelId, el.checked));
        });
        container.querySelectorAll('[data-action="provider-bulk"]').forEach(el => {
            el.addEventListener('click', () => this.bulkProvider(el.dataset.providerKey, el.dataset.bulk));
        });
        container.querySelectorAll('[data-action="provider-settings"]').forEach(el => {
            el.addEventListener('click', () => window.ProviderSettings?.open(el.dataset.providerKey));
        });
    }

    _section(providerKey, models) {
        const esc = window.UIUtils.escapeHtml;
        const pk = esc(providerKey);
        const total = this._models.filter(m => m.provider_key === providerKey);
        const on = total.filter(m => m.is_enabled).length;
        const off = this._providerOff.has(providerKey)
            ? '<span class="ma-tag ma-tag-off" title="Switched off in the Providers view: none of its models are served"><i class="fas fa-ban me-1"></i>provider off</span>'
            : '';
        const autoOn = !this._newModelsOff.has(providerKey);
        return `
        <div class="pm-provider-head">
            <span class="pm-provider-name"><i class="fas fa-server me-2"></i>${pk}${off}</span>
            <span class="pm-provider-tools">
                <span class="pm-provider-count">${on}/${total.length} enabled</span>
                <button type="button" class="btn btn-outline-success btn-sm" data-action="provider-bulk" data-bulk="enable_all"
                    data-provider-key="${pk}" title="Enable every model of ${pk}">
                    <i class="fas fa-check-circle me-1"></i>Enable all
                </button>
                <button type="button" class="btn btn-outline-danger btn-sm" data-action="provider-bulk" data-bulk="disable_all"
                    data-provider-key="${pk}" title="Disable every model of ${pk}">
                    <i class="fas fa-times-circle me-1"></i>Disable all
                </button>
                <button type="button" class="btn btn-outline-secondary btn-sm pm-settings-btn${autoOn ? '' : ' is-held'}"
                    data-action="provider-settings" data-provider-key="${pk}"
                    title="${this._gearTitle(autoOn)}" aria-label="Settings for ${pk}">
                    <i class="fas fa-gear"></i>
                </button>
            </span>
        </div>
        ${models.map(m => this._row(m)).join('')}`;
    }

    _gearTitle(autoOn) {
        return autoOn ? 'Provider settings'
            : 'Provider settings (new models stay disabled until you enable them)';
    }

    /** The Provider settings dialog saved: reflect what this view shows of it. */
    applyProviderSettings(providerKey, settings) {
        if (settings.auto_enable_new_models === false) this._newModelsOff.add(providerKey);
        else this._newModelsOff.delete(providerKey);
        this.render();
    }

    _row(m) {
        const esc = window.UIUtils.escapeHtml;
        const group = this._groupOf[m.model_id];
        const held = !m.is_enabled || this._providerOff.has(m.provider_key) || (group && !group.enabled);
        let tag = '';
        if (group) {
            tag = group.enabled
                ? `<span class="ma-tag ma-tag-off" title="Model group">${esc(group.name)}</span>`
                : `<span class="ma-tag ma-tag-off" title="Its model group is switched off in Model Groups"><i class="fas fa-ban me-1"></i>group ${esc(group.name)} off</span>`;
        }
        return `
        <div class="ma-model-row${held ? ' is-off' : ''}">
            <div class="ma-model-id" title="${esc(m.model_id)}">${esc(m.model_name || m.model_id)}${tag}</div>
            <div class="ma-model-controls">
                <label class="ma-switch" title="${m.is_enabled ? 'Disable' : 'Enable'} for everyone">
                    <input type="checkbox"${m.is_enabled ? ' checked' : ''}
                        data-action="toggle-model" data-model-id="${esc(m.model_id)}">
                    <span class="ma-slider"></span>
                </label>
            </div>
        </div>`;
    }

    /** Other views cache counts and model lists; bring them up to date. */
    _afterChange() {
        window.SearchManager?.clearModelsCache();
        window.ProviderManager?.refreshProviderCounts().catch(() => {});
    }

    async toggle(modelId, enabled) {
        try {
            const resp = await fetch(`/admin/models/toggle?model_id=${encodeURIComponent(modelId)}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled }),
            });
            if (resp.ok) {
                const model = this._models.find(m => m.model_id === modelId);
                if (model) model.is_enabled = enabled;
                this.render();
                window.UIUtils?.showToast(`Model ${enabled ? 'enabled' : 'disabled'}.`, 'success');
                this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update model.', 'error');
                await this.load();
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating model.', 'error');
            await this.load();
        }
    }

    async bulkProvider(providerKey, action) {
        const enabling = action === 'enable_all';
        const confirmed = await window.UIUtils?.showConfirmModal(
            enabling ? 'Enable all models' : 'Disable all models',
            `${enabling ? 'Enable' : 'Disable'} every model of ${providerKey} for all users? Other providers are unchanged.`,
        );
        if (!confirmed) return;
        try {
            const resp = await fetch('/admin/models/bulk-toggle', {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, provider_key: providerKey }),
            });
            if (resp.ok) {
                const result = await resp.json();
                window.UIUtils?.showToast(result.message || 'Models updated.', 'success');
                await this.load();
                this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update models.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating models.', 'error');
        }
    }

    async bulk(action) {
        const enabling = action === 'enable_all';
        const confirmed = await window.UIUtils?.showConfirmModal(
            enabling ? 'Enable all models' : 'Disable all models',
            `${enabling ? 'Enable' : 'Disable'} every model for all users? Provider and group switches are unchanged.`,
        );
        if (!confirmed) return;
        try {
            const resp = await fetch('/admin/models/bulk-toggle', {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action }),
            });
            if (resp.ok) {
                const result = await resp.json();
                window.UIUtils?.showToast(result.message || 'Models updated.', 'success');
                await this.load();
                this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update models.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating models.', 'error');
        }
    }
}

window.ModelsView = new ModelsView();

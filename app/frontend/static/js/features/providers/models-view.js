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
            this._providerOff = new Set(pResp.ok
                ? (await pResp.json()).filter(p => p.enabled === false).map(p => p.provider_key)
                : []);
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
    }

    _section(providerKey, models) {
        const esc = window.UIUtils.escapeHtml;
        const total = this._models.filter(m => m.provider_key === providerKey);
        const on = total.filter(m => m.is_enabled).length;
        const off = this._providerOff.has(providerKey)
            ? '<span class="ma-tag ma-tag-off" title="Switched off in the Providers view: none of its models are served"><i class="fas fa-ban me-1"></i>provider off</span>'
            : '';
        return `
        <div class="pm-provider-head">
            <span class="pm-provider-name"><i class="fas fa-server me-2"></i>${esc(providerKey)}${off}</span>
            <span class="pm-provider-count">${on}/${total.length} enabled</span>
        </div>
        ${models.map(m => this._row(m)).join('')}`;
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

/**
 * Unified Models manager.
 *
 * One view, two scopes selected from the left column:
 *   - "Global"   → enable/disable each model for everyone (global availability).
 *   - a profile  → set that profile's access mode (Default / Allow all / Deny all /
 *                  Custom) plus per-model allow/deny exceptions. Every member of the
 *                  profile gets exactly this access.
 *
 * The right pane swaps between the global panel (#ma-global) and the profile
 * panel (#ma-detail); #ma-empty is the pre-selection placeholder.
 */
class ModelsManager {
    constructor() {
        this._scope = 'global';        // 'global' or 'profile'
        this._selectedProfileId = null;
        this._detail = null;           // ProfileModelAccessResponse for the selected profile
        this._globalModels = [];       // [{model_id, model_name, provider_key, is_enabled}]
        document.addEventListener('profiles:changed', () => this._renderProfiles());
    }

    async load() {
        await window.ProfileStore.list({ force: true });
        // A quick link from the Profiles tab (or the last visit) picks the scope.
        const wanted = window.ProfileStore.selected('model-access', null);
        if (wanted != null && window.ProfileStore.byId(wanted)) {
            await this.selectProfile(wanted);
        } else {
            await this.selectScope('global');
        }
    }

    // ── Left column: scope list ──────────────────────────────────────────────

    filterProfiles() {
        this._renderProfiles();
    }

    _renderProfiles() {
        const list = document.getElementById('ma-profile-list');
        if (!list) return;
        const profiles = window.ProfileStore._profiles || [];
        const q = (document.getElementById('ma-profile-search')?.value || '').trim().toLowerCase();
        const filtered = q ? profiles.filter(p => (p.name || '').toLowerCase().includes(q)) : profiles;
        if (!filtered.length) {
            list.innerHTML = `<div class="text-muted small py-3 text-center">${profiles.length ? 'No profiles match your search.' : 'No profiles found.'}</div>`;
            return;
        }
        list.innerHTML = filtered.map(p => this._renderProfileRow(p)).join('');
    }

    _renderProfileRow(p) {
        const esc = window.UIUtils.escapeHtml;
        const active = (this._scope === 'profile' && p.id === this._selectedProfileId) ? ' active' : '';
        const exceptions = p.model_access_mode === 'custom' && p.exception_count
            ? ` · ${p.exception_count} exception${p.exception_count === 1 ? '' : 's'}` : '';
        return `
        <button type="button" class="ma-user-item${active}" data-profile-id="${p.id}"
            onclick="window.ModelsManager.selectProfile(${p.id})">
            <span class="ma-user-text">
                <span class="ma-user-name">${esc(p.name)}</span>
                <span class="ma-user-email">${p.member_count} member${p.member_count === 1 ? '' : 's'} · ${esc(p.model_access_mode)}${exceptions}</span>
            </span>
            <span class="ma-user-trailing">
                <span class="ma-editing-flag"><i class="fas fa-pen me-1"></i>Editing</span>
            </span>
        </button>`;
    }

    _updateScopeHighlight() {
        const globalBtn = document.getElementById('ma-scope-global');
        if (globalBtn) globalBtn.classList.toggle('active', this._scope === 'global');
        this._renderProfiles();  // refresh the per-profile active highlight
    }

    _showPanel(which) {
        const map = { empty: 'ma-empty', global: 'ma-global', detail: 'ma-detail' };
        Object.entries(map).forEach(([key, id]) => {
            const el = document.getElementById(id);
            if (el) el.style.display = key === which ? '' : 'none';
        });
    }

    // ── Global scope ─────────────────────────────────────────────────────────

    async selectScope(scope) {
        if (scope !== 'global') return;
        this._scope = 'global';
        this._selectedProfileId = null;
        window.ProfileStore.select('model-access', null);
        this._updateScopeHighlight();
        await this._loadGlobal();
    }

    async _loadGlobal() {
        const list = document.getElementById('ma-global-list');
        try {
            const resp = await fetch('/admin/models/all', { credentials: 'include' });
            if (!resp.ok) {
                this._showPanel('global');
                if (list) list.innerHTML = '<div class="text-danger small py-4 text-center">Couldn\'t load models. Retry.</div>';
                return;
            }
            this._globalModels = await resp.json();
            this._showPanel('global');
            this._renderGlobal();
        } catch (e) {
            console.error('ModelsManager: failed to load global models', e);
            this._showPanel('global');
            if (list) list.innerHTML = '<div class="text-danger small py-4 text-center">Error loading models.</div>';
        }
    }

    filterGlobal() {
        this._renderGlobal();
    }

    _renderGlobal() {
        const container = document.getElementById('ma-global-list');
        const summary = document.getElementById('ma-global-summary');
        if (!container) return;

        const total = this._globalModels.length;
        const enabled = this._globalModels.filter(m => m.is_enabled).length;
        if (summary) summary.textContent = `${enabled} of ${total} enabled`;

        const q = (document.getElementById('ma-global-search')?.value || '').trim().toLowerCase();
        const filtered = q
            ? this._globalModels.filter(m =>
                (m.model_id || '').toLowerCase().includes(q)
                || (m.model_name || '').toLowerCase().includes(q))
            : this._globalModels;

        if (!filtered.length) {
            container.innerHTML = '<div class="text-muted small py-4 text-center">No models match your search.</div>';
            return;
        }
        container.innerHTML = filtered.map(m => this._renderGlobalRow(m)).join('');
    }

    _renderGlobalRow(m) {
        const safeId = window.UIUtils.escapeHtml(m.model_id);
        const rowCls = ['ma-model-row'];
        if (!m.is_enabled) rowCls.push('is-off');
        const checked = m.is_enabled ? ' checked' : '';
        return `
        <div class="${rowCls.join(' ')}" data-model-id="${safeId}">
            <div class="ma-model-id">${safeId}</div>
            <div class="ma-model-controls">
                <label class="ma-switch">
                    <input type="checkbox"${checked}
                        onchange="window.ModelsManager.toggleGlobal('${encodeURIComponent(m.model_id)}', this.checked)">
                    <span class="ma-slider"></span>
                </label>
            </div>
        </div>`;
    }

    async toggleGlobal(encodedModelId, enabled) {
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(`/admin/models/toggle?model_id=${encodeURIComponent(modelId)}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled }),
            });
            if (resp.ok) {
                const model = this._globalModels.find(m => m.model_id === modelId);
                if (model) model.is_enabled = enabled;
                if (window.SearchManager) window.SearchManager.clearModelsCache();
                this._renderGlobal();
                window.UIUtils?.showToast(`Model ${enabled ? 'enabled' : 'disabled'}.`, 'success');
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update model.', 'error');
                await this._loadGlobal();
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating model.', 'error');
            await this._loadGlobal();
        }
    }

    async bulkGlobal(action) {
        const enabling = action === 'enable_all';
        const confirmed = await window.UIUtils?.showConfirmModal(
            enabling ? 'Enable all models' : 'Disable all models',
            `${enabling ? 'Enable' : 'Disable'} every model for all users?`,
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
                if (window.SearchManager) window.SearchManager.clearModelsCache();
                window.UIUtils?.showToast(result.message || 'Models updated.', 'success');
                await this._loadGlobal();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update models.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating models.', 'error');
        }
    }

    async syncGlobal() {
        const confirmed = await window.UIUtils?.showConfirmModal(
            'Sync models',
            'Refresh the model list from all providers. Continue?',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch('/admin/models/sync', { method: 'POST', credentials: 'include' });
            if (resp.ok) {
                const result = await resp.json();
                window.UIUtils?.showToast(
                    `Synced ${result.providers_synced || 0} providers and ${result.models_synced || 0} models.`,
                    'success',
                );
                if (window.SearchManager) window.SearchManager.clearModelsCache();
                await this._loadGlobal();
                if (result.stale_count > 0 && result.stale_models?.length
                    && window.ModelManager?.showStaleModelsDialog) {
                    window.ModelManager.showStaleModelsDialog(result.stale_models);
                }
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to sync models.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error syncing models.', 'error');
        }
    }

    // ── Profile scope ────────────────────────────────────────────────────────

    async selectProfile(profileId) {
        this._scope = 'profile';
        this._selectedProfileId = profileId;
        window.ProfileStore.select('model-access', profileId);
        this._updateScopeHighlight();
        await this._loadDetail(profileId);
    }

    _url(path, modelId) {
        let url = `/admin/profiles/model-access${path}?profile_id=${this._selectedProfileId}`;
        if (modelId) url += `&model_id=${encodeURIComponent(modelId)}`;
        return url;
    }

    async _loadDetail(profileId) {
        try {
            const resp = await fetch(`/admin/profiles/model-access?profile_id=${profileId}`, { credentials: 'include' });
            if (!resp.ok) {
                this._showPanel('empty');
                const empty = document.getElementById('ma-empty');
                if (empty) empty.querySelector('p').textContent = "Couldn't load model access. Retry.";
                return;
            }
            this._detail = await resp.json();
            this._showPanel('detail');
            this._renderDetail();
        } catch (e) {
            console.error('ModelsManager: failed to load detail', e);
            this._showPanel('empty');
            const empty = document.getElementById('ma-empty');
            if (empty) empty.querySelector('p').textContent = "Couldn't load model access. Retry.";
        }
    }

    /** The server's view changed (mode, exceptions): refresh the left-column counts too. */
    async _afterChange() {
        this._renderDetail();
        await window.ProfileStore.notifyChanged();
    }

    _renderDetail() {
        if (!this._detail) return;
        const nameEl = document.getElementById('ma-selected-username');
        if (nameEl) nameEl.textContent = this._detail.name;
        const membersEl = document.getElementById('ma-selected-members');
        if (membersEl) {
            const n = this._detail.member_count;
            membersEl.textContent = `Applies to ${n} member${n === 1 ? '' : 's'}`;
        }

        // Reflect the mode in the 4-option segmented control.
        const mode = this._detail.mode || 'default';
        const map = {
            default: 'ma-policy-default',
            allow: 'ma-policy-allow',
            deny: 'ma-policy-deny',
            custom: 'ma-policy-custom',
        };
        Object.entries(map).forEach(([m, id]) => {
            const btn = document.getElementById(id);
            if (btn) btn.classList.toggle('active', m === mode);
        });

        const hint = document.getElementById('ma-mode-hint');
        if (hint) hint.textContent = this._modeHint(mode);

        this._renderModels();
    }

    _modeHint(mode) {
        switch (mode) {
            case 'allow':
                return 'Every model is enabled for every member of this profile, overriding the global config (including future models).';
            case 'deny':
                return 'Every model is disabled for every member of this profile, overriding the global config (including future models).';
            case 'custom':
                return 'Per-model overrides win. New models follow the global config; overrides can enable a globally-disabled model.';
            default:
                return 'Follows the global model config. Toggle any model to switch this profile to Custom.';
        }
    }

    filterModels() {
        this._renderModels();
    }

    _renderModels() {
        const container = document.getElementById('ma-model-list');
        const summary = document.getElementById('ma-summary');
        if (!container || !this._detail) return;

        const models = this._detail.models;
        const enabledCount = models.filter(m => m.effective_allowed).length;
        const disabledCount = models.length - enabledCount;
        if (summary) {
            summary.textContent = `${enabledCount} enabled · ${disabledCount} disabled`;
        }

        const q = (document.getElementById('ma-model-search')?.value || '').trim().toLowerCase();
        const filtered = q
            ? models.filter(m => (m.model_id || '').toLowerCase().includes(q))
            : models;

        if (!filtered.length) {
            container.innerHTML = '<div class="text-muted small py-4 text-center">No models match your search.</div>';
            return;
        }
        container.innerHTML = filtered.map(m => this._renderModelRow(m)).join('');
    }

    _renderModelRow(m) {
        const safeId = window.UIUtils.escapeHtml(m.model_id);
        const disabledGlobally = !m.globally_enabled;
        const rowCls = ['ma-model-row'];
        if (m.is_exception) rowCls.push('is-exception');
        if (disabledGlobally) rowCls.push('is-global-off');

        // Toggle reflects effective state. Toggles stay active in every mode so an
        // admin can enable a globally-disabled model for one profile (switches to custom).
        const checked = m.effective_allowed ? ' checked' : '';

        const tag = disabledGlobally
            ? '<span class="ma-tag ma-tag-off"><i class="fas fa-ban me-1"></i>globally disabled</span>'
            : '';

        return `
        <div class="${rowCls.join(' ')}" data-model-id="${safeId}">
            <div class="ma-model-id">
                ${safeId} ${tag}
            </div>
            <div class="ma-model-controls">
                <label class="ma-switch">
                    <input type="checkbox"${checked}
                        onchange="window.ModelsManager.toggleModel('${encodeURIComponent(m.model_id)}', this.checked)">
                    <span class="ma-slider"></span>
                </label>
            </div>
        </div>`;
    }

    async setMode(mode) {
        if (this._selectedProfileId == null) return;
        const labels = { default: 'Default', allow: 'Allow all', deny: 'Deny all', custom: 'Custom' };
        try {
            const resp = await fetch(this._url('/policy'), {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            });
            if (resp.ok) {
                this._detail = await resp.json();
                window.UIUtils?.showToast(`Mode set to ${labels[mode] || mode}.`, 'success');
                await this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update mode.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating mode.', 'error');
        }
    }

    async toggleModel(encodedModelId, isAllowed) {
        if (this._selectedProfileId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(this._url('', modelId), {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_allowed: isAllowed }),
            });
            if (resp.ok) {
                // Response is the full view; toggling may have flipped the profile to
                // custom mode, so re-render the mode control and list from server truth.
                this._detail = await resp.json();
                await this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update access.', 'error');
                // Reconcile with server truth.
                await this._loadDetail(this._selectedProfileId);
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating access.', 'error');
            await this._loadDetail(this._selectedProfileId);
        }
    }

    async resetModel(encodedModelId) {
        if (this._selectedProfileId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(this._url('', modelId), {
                method: 'DELETE',
                credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Reset to global default.', 'success');
                await this._loadDetail(this._selectedProfileId);
                await window.ProfileStore.notifyChanged();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to reset.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error resetting access.', 'error');
        }
    }
}

window.ModelsManager = new ModelsManager();

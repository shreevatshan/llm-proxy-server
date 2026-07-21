/**
 * Unified Models manager.
 *
 * One view, two scopes selected from the left column:
 *   - "Global"  → enable/disable each model for everyone (global availability).
 *   - a user    → set that user's default policy (Allow all / Deny all) plus
 *                 per-model allow/deny exceptions. Globally disabled models stay
 *                 locked off regardless of the per-user setting.
 *
 * The right pane swaps between the global panel (#ma-global) and the per-user
 * panel (#ma-detail); #ma-empty is the pre-selection placeholder.
 */
class ModelsManager {
    constructor() {
        this._scope = 'global';        // 'global' or 'user'
        this._users = [];
        this._selectedUserId = null;
        this._detail = null;           // UserModelAccessResponse for the selected user
        this._exceptionCounts = {};    // user_id -> number of exceptions (for badges)
        this._globalModels = [];       // [{model_id, model_name, provider_key, is_enabled}]
    }

    async load() {
        await this._loadUsers();
        // Re-apply the current scope so the tab remembers what you were editing.
        if (this._scope === 'user' && this._selectedUserId != null) {
            this._updateScopeHighlight();
            await this._loadDetail(this._selectedUserId);
        } else {
            await this.selectScope('global');
        }
    }

    // ── Left column: scope list ──────────────────────────────────────────────

    async _loadUsers() {
        const list = document.getElementById('ma-user-list');
        if (!list) return;
        try {
            const resp = await fetch('/admin/users', { credentials: 'include' });
            if (!resp.ok) {
                list.innerHTML = '<div class="text-danger small py-3 text-center">Couldn\'t load users. Retry.</div>';
                return;
            }
            this._users = await resp.json();
            this._renderUsers();
        } catch (e) {
            console.error('ModelsManager: failed to load users', e);
            list.innerHTML = '<div class="text-danger small py-3 text-center">Error loading users.</div>';
        }
    }

    filterUsers() {
        this._renderUsers();
    }

    _renderUsers() {
        const list = document.getElementById('ma-user-list');
        if (!list) return;
        if (!this._users.length) {
            list.innerHTML = '<div class="text-muted small py-3 text-center">No users found.</div>';
            return;
        }
        const q = (document.getElementById('ma-user-search')?.value || '').trim().toLowerCase();
        const filtered = q
            ? this._users.filter(u =>
                (u.username || '').toLowerCase().includes(q)
                || (u.email || '').toLowerCase().includes(q))
            : this._users;
        if (!filtered.length) {
            list.innerHTML = '<div class="text-muted small py-3 text-center">No users match your search.</div>';
            return;
        }
        list.innerHTML = filtered.map(u => this._renderUserRow(u)).join('');
    }

    _renderUserRow(u) {
        const safeUsername = window.UIUtils.escapeHtml(u.username);
        const safeEmail = window.UIUtils.escapeHtml(u.email || '');
        const active = (this._scope === 'user' && u.id === this._selectedUserId) ? ' active' : '';
        return `
        <button type="button" class="ma-user-item${active}" data-user-id="${u.id}"
            onclick="window.ModelsManager.selectUser(${u.id})">
            <span class="ma-user-text">
                <span class="ma-user-name">${safeUsername}</span>
                <span class="ma-user-email">${safeEmail}</span>
            </span>
            <span class="ma-user-trailing">
                <span class="ma-editing-flag"><i class="fas fa-pen me-1"></i>Editing</span>
            </span>
        </button>`;
    }

    _updateScopeHighlight() {
        const globalBtn = document.getElementById('ma-scope-global');
        if (globalBtn) globalBtn.classList.toggle('active', this._scope === 'global');
        this._renderUsers();  // refresh the per-user active highlight
    }

    _showPanel(which) {
        const map = { empty: 'ma-empty', global: 'ma-global', detail: 'ma-detail' };
        Object.entries(map).forEach(([key, id]) => {
            const el = document.getElementById(id);
            if (el) el.style.display = key === which ? '' : 'none';
        });
    }
    
    _setScopeBanner(kind, name) {
        const el = document.getElementById('ma-scope-banner');
        if (!el) return;
    }

    // ── Global scope ─────────────────────────────────────────────────────────

    async selectScope(scope) {
        if (scope !== 'global') return;
        this._scope = 'global';
        this._selectedUserId = null;
        this._updateScopeHighlight();
        this._setScopeBanner('global');
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

    // ── Per-user scope ───────────────────────────────────────────────────────

    async selectUser(userId) {
        this._scope = 'user';
        this._selectedUserId = userId;
        this._updateScopeHighlight();
        await this._loadDetail(userId);
    }

    async _loadDetail(userId) {
        try {
            const resp = await fetch(`/admin/users/model-access?user_id=${userId}`, { credentials: 'include' });
            if (!resp.ok) {
                this._showPanel('empty');
                this._setScopeBanner(null);
                const empty = document.getElementById('ma-empty');
                if (empty) empty.querySelector('p').textContent = "Couldn't load model access. Retry.";
                return;
            }
            this._detail = await resp.json();
            this._exceptionCounts[userId] = this._detail.models.filter(m => m.is_exception).length;
            this._showPanel('detail');
            this._setScopeBanner('user', this._detail.username);
            this._renderDetail();
            this._renderUsers();  // update badge count
        } catch (e) {
            console.error('ModelsManager: failed to load detail', e);
            this._showPanel('empty');
            this._setScopeBanner(null);
            const empty = document.getElementById('ma-empty');
            if (empty) empty.querySelector('p').textContent = "Couldn't load model access. Retry.";
        }
    }

    _renderDetail() {
        if (!this._detail) return;
        const usernameEl = document.getElementById('ma-selected-username');
        if (usernameEl) usernameEl.textContent = this._detail.username;

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
                return 'Every model is enabled for this user, overriding the global config (including future models).';
            case 'deny':
                return 'Every model is disabled for this user, overriding the global config (including future models).';
            case 'custom':
                return 'Per-model overrides win. New models follow the global config; overrides can enable a globally-disabled model.';
            default:
                return 'Follows the global model config. Toggle any model to switch to Custom for this user.';
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
        // admin can enable a globally-disabled model for a single user (switches to custom).
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
        if (this._selectedUserId == null) return;
        const labels = { default: 'Default', allow: 'Allow all', deny: 'Deny all', custom: 'Custom' };
        try {
            const resp = await fetch(`/admin/users/model-access/policy?user_id=${this._selectedUserId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            });
            if (resp.ok) {
                this._detail = await resp.json();
                this._exceptionCounts[this._selectedUserId] = this._detail.models.filter(m => m.is_exception).length;
                window.UIUtils?.showToast(`Mode set to ${labels[mode] || mode}.`, 'success');
                this._renderDetail();
                this._renderUsers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update mode.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating mode.', 'error');
        }
    }

    async toggleModel(encodedModelId, isAllowed) {
        if (this._selectedUserId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(
                `/admin/users/model-access?user_id=${this._selectedUserId}&model_id=${encodeURIComponent(modelId)}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_allowed: isAllowed }),
            });
            if (resp.ok) {
                // Response is the full view; toggling may have flipped the user to
                // custom mode, so re-render the mode control and list from server truth.
                this._detail = await resp.json();
                this._exceptionCounts[this._selectedUserId] = this._detail.models.filter(m => m.is_exception).length;
                this._renderDetail();
                this._renderUsers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update access.', 'error');
                // Reconcile with server truth.
                await this._loadDetail(this._selectedUserId);
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating access.', 'error');
            await this._loadDetail(this._selectedUserId);
        }
    }

    async resetModel(encodedModelId) {
        if (this._selectedUserId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(
                `/admin/users/model-access?user_id=${this._selectedUserId}&model_id=${encodeURIComponent(modelId)}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Reset to global default.', 'success');
                await this._loadDetail(this._selectedUserId);
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

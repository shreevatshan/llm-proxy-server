/**
 * Admin: users' own provider instances (BYOK), in the Providers tab.
 *
 * Read-only apart from two actions: switch an instance off (the owner cannot
 * switch it back on) and delete it. Backed by /admin/byok/providers; secrets are
 * never sent. Loaded from ProviderManager.loadTab (not the shared showTab, which
 * the user dashboard also uses).
 */

class ByokAdminManager {
    _esc(value) {
        return window.UIUtils?.escapeHtml ? window.UIUtils.escapeHtml(String(value ?? '')) : String(value ?? '');
    }

    async _request(url, options = {}) {
        const response = await fetch(url, { credentials: 'include', ...options });
        let payload = null;
        try { payload = await response.json(); } catch (_) { /* empty body */ }
        if (!response.ok) {
            throw new Error((payload && typeof payload.detail === 'string') ? payload.detail : `Request failed (${response.status})`);
        }
        return payload;
    }

    async load() {
        const body = document.getElementById('byok-admin-body');
        if (!body) return;
        let rows;
        try {
            rows = await this._request('/admin/byok/providers');
        } catch (error) {
            body.innerHTML = `<div class="alert alert-danger m-3">Could not load BYOK instances: ${this._esc(error.message)}</div>`;
            return;
        }
        if (!rows.length) {
            body.innerHTML = '<p class="text-center text-muted py-4 mb-0">No user has added a provider.</p>';
            return;
        }
        body.innerHTML = `
            <div class="table-responsive">
            <table class="table table-hover mb-0 align-middle">
                <thead><tr>
                    <th>Owner</th><th>Instance</th><th>Type</th><th>Upstream</th>
                    <th class="text-end">Models</th><th>Status</th><th class="text-end">Actions</th>
                </tr></thead>
                <tbody>${rows.map(r => this._row(r)).join('')}</tbody>
            </table>
            </div>`;
    }

    _status(r) {
        if (r.admin_disabled) return '<span class="badge bg-danger">Disabled by admin</span>';
        if (!r.enabled) return '<span class="badge bg-secondary">Off (by owner)</span>';
        if (r.load_error) return `<span class="badge bg-warning text-dark" title="${this._esc(r.load_error)}">Not serving</span>`;
        return '<span class="badge bg-success">On</span>';
    }

    _row(r) {
        const jsKey = this._esc(JSON.stringify(r.provider_key));
        return `
            <tr>
                <td>${this._esc(r.owner_username || `user #${r.owner_user_id}`)}</td>
                <td><code class="small text-break">${this._esc(r.provider_key)}</code>
                    <span class="badge bg-info text-dark ms-1">BYOK</span></td>
                <td>${this._esc(r.provider_type)}</td>
                <td class="small">${this._esc(r.endpoint_host || '—')}</td>
                <td class="text-end">${r.model_count}</td>
                <td>${this._status(r)}</td>
                <td class="text-end text-nowrap">
                    <button type="button" class="btn btn-sm btn-outline-${r.admin_disabled ? 'success' : 'warning'}"
                            onclick='window.ByokAdminManager.setDisabled(${jsKey}, ${!r.admin_disabled})'>
                        ${r.admin_disabled ? 'Enable' : 'Disable'}
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-danger" title="Delete"
                            onclick='window.ByokAdminManager.remove(${jsKey})'>
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            </tr>`;
    }

    async setDisabled(providerKey, disabled) {
        try {
            await this._request(
                `/admin/byok/providers/disable?provider_key=${encodeURIComponent(providerKey)}&disabled=${disabled}`,
                { method: 'PUT' },
            );
            window.UIUtils?.showToast(disabled ? 'Instance disabled' : 'Instance re-enabled', 'success');
        } catch (error) {
            window.UIUtils?.showToast(error.message, 'error');
        }
        this.load();
    }

    async remove(providerKey) {
        const ok = await window.UIUtils.showConfirmModal(
            'Delete BYOK instance',
            `Delete ${providerKey} and its models? The owner's stored keys are deleted. Usage history is kept.`,
            'danger',
        );
        if (!ok) return;
        try {
            await this._request(`/admin/byok/providers?provider_key=${encodeURIComponent(providerKey)}`, { method: 'DELETE' });
            window.UIUtils?.showToast('Instance deleted', 'success');
        } catch (error) {
            window.UIUtils?.showToast(error.message, 'error');
        }
        this.load();
    }
}

window.ByokAdminManager = new ByokAdminManager();

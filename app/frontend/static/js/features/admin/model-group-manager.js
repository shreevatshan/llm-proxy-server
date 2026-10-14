class ModelGroupManager {
    constructor() {
        this._allModels = [];
        this._groups = [];
        this._modal = null;
        this._selectedIds = new Set();
        // The 'Limits for profile' card; its profile is set by RateLimitManager from
        // the Rate Limits tab's profile bar.
        this._profileLimits = new window.ProfileGroupLimits({
            kind: 'model', prefix: 'mg', noun: 'models', manager: 'ModelGroupManager',
        });
    }

    async load() {
        await Promise.all([this._loadGroups(), this._loadAllModels()]);
        await this.renderProfileLimits();
    }

    // ---- Models cache ----

    async _loadAllModels() {
        try {
            const resp = await fetch('/admin/models/all', { credentials: 'include' });
            if (!resp.ok) return;
            const models = await resp.json();
            this._allModels = models.map(m => ({
                model_id: m.model_id,
                label: m.model_name,
                provider: m.provider_key,
            }));
        } catch (e) {
            console.error('ModelGroupManager: failed to load models', e);
        }
    }

    // ---- Groups list pane ----

    async _loadGroups() {
        const tbody = document.getElementById('model-groups-tbody');
        if (!tbody) return;
        try {
            const resp = await fetch('/admin/model-groups', { credentials: 'include' });
            if (!resp.ok) {
                tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger py-4">Failed to load model groups.</td></tr>';
                return;
            }
            this._groups = await resp.json();
            if (!this._groups.length) {
                tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-4">No model groups yet. Click "Create Model Group" to add one.</td></tr>';
                return;
            }
            tbody.innerHTML = this._groups.map(g => this._renderRow(g)).join('');
        } catch (e) {
            console.error('ModelGroupManager: failed to load groups', e);
            tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger py-4">Error loading data.</td></tr>';
        }
    }

    _renderRow(g) {
        const esc = window.UIUtils?.escapeHtml || (s => s);
        return `
        <tr data-group-id="${g.id}">
            <td>
                <div class="fw-semibold">${esc(g.name)}</div>
                ${g.description ? `<div class="text-muted small">${esc(g.description)}</div>` : ''}
            </td>
            <td><span class="badge bg-secondary">${g.member_count}</span></td>
            <td>
                <button class="btn btn-outline-primary btn-sm me-1" onclick="window.ModelGroupManager?.openEditModal(${g.id})">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelGroupManager?.deleteGroup(${g.id}, '${esc(g.name)}')">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>`;
    }

    // ---- Create/Edit modal (group identity + members only) ----

    openCreateModal() {
        document.getElementById('mg-group-id').value = '';
        document.getElementById('mg-name').value = '';
        document.getElementById('mg-description').value = '';
        document.getElementById('mg-rpm').value = '';
        document.getElementById('mg-rpd').value = '';
        document.getElementById('modelGroupModalTitle').textContent = 'New Model Group';
        const search = document.getElementById('mg-model-search');
        if (search) search.value = '';
        this._renderModelPicker([]);
        this._getModal()?.show();
    }

    async openEditModal(groupId) {
        try {
            const resp = await fetch(`/admin/model-groups?group_id=${groupId}`, { credentials: 'include' });
            if (!resp.ok) { window.UIUtils?.showToast('Failed to load group.', 'error'); return; }
            const g = await resp.json();

            document.getElementById('mg-group-id').value = g.id;
            document.getElementById('mg-name').value = g.name;
            document.getElementById('mg-description').value = g.description || '';
            document.getElementById('mg-rpm').value = '';
            document.getElementById('mg-rpd').value = '';
            document.getElementById('modelGroupModalTitle').textContent = `Edit: ${g.name}`;
            const search = document.getElementById('mg-model-search');
            if (search) search.value = '';

            if (!this._allModels.length) await this._loadAllModels();
            this._renderModelPicker(g.members || []);
            this._getModal()?.show();
        } catch (e) {
            window.UIUtils?.showToast('Error loading group.', 'error');
        }
    }

    _renderModelPicker(selectedIds, filter = '') {
        this._selectedIds = new Set(selectedIds);
        const container = document.getElementById('mg-model-picker');
        if (!container) return;
        if (!this._allModels.length) {
            container.innerHTML = '<div class="text-muted small">No models found.</div>';
            return;
        }
        const q = filter.trim().toLowerCase();
        const visible = q
            ? this._allModels.filter(m =>
                m.label.toLowerCase().includes(q) ||
                m.model_id.toLowerCase().includes(q) ||
                m.provider.toLowerCase().includes(q))
            : this._allModels;
        if (!visible.length) {
            container.innerHTML = '<div class="text-muted small">No models match your search.</div>';
            return;
        }
        const esc = window.UIUtils?.escapeHtml || (s => s);
        container.innerHTML = visible.map(m => {
            const checked = this._selectedIds.has(m.model_id) ? 'checked' : '';
            return `
            <div class="form-check">
                <input class="form-check-input" type="checkbox" id="mgm-${CSS.escape(m.model_id)}"
                    value="${esc(m.model_id)}" ${checked}
                    onchange="window.ModelGroupManager?._onModelToggle(this)">
                <label class="form-check-label small" for="mgm-${CSS.escape(m.model_id)}">
                    <span class="text-muted">${esc(m.provider)}</span>
                    / ${esc(m.label)}
                </label>
            </div>`;
        }).join('');
    }

    _onModelToggle(checkbox) {
        if (checkbox.checked) {
            this._selectedIds.add(checkbox.value);
        } else {
            this._selectedIds.delete(checkbox.value);
        }
    }

    filterModels(query) {
        this._renderModelPicker([...this._selectedIds], query);
    }

    async saveGroup() {
        const groupId = document.getElementById('mg-group-id').value;
        const name = document.getElementById('mg-name').value.trim();
        const description = document.getElementById('mg-description').value.trim();
        const rpmRaw = document.getElementById('mg-rpm').value;
        const rpdRaw = document.getElementById('mg-rpd').value;
        // Optional: when filled in, these become every profile's limit for the group.
        // A blank field is omitted so each profile keeps its own value for it.
        const limits = {};
        if (rpmRaw !== '') limits.rpm_limit = parseInt(rpmRaw, 10);
        if (rpdRaw !== '') limits.rpd_limit = parseInt(rpdRaw, 10);
        const applyToAll = Object.keys(limits).length > 0;

        if (!name) { window.UIUtils?.showToast('Name is required.', 'error'); return; }

        try {
            let group;
            if (!groupId) {
                const resp = await fetch('/admin/model-groups', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ name, description: description || null }),
                });
                const data = await resp.json();
                if (!resp.ok) { window.UIUtils?.showToast(data.detail || 'Failed to create group.', 'error'); return; }
                group = data;
                window.UIUtils?.showToast(`Group '${name}' created.`, 'success');
            } else {
                const resp = await fetch(`/admin/model-groups?group_id=${groupId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ name, description: description || null }),
                });
                if (!resp.ok) { const e = await resp.json(); window.UIUtils?.showToast(e.detail || 'Failed to update group.', 'error'); return; }
                group = { id: parseInt(groupId) };
                window.UIUtils?.showToast(`Group '${name}' updated.`, 'success');
            }

            // Save member list (use _selectedIds to preserve selections made while search was filtered)
            const model_ids = [...(this._selectedIds || [])];
            const memResp = await fetch(`/admin/model-groups/members?group_id=${group.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ model_ids }),
            });
            if (!memResp.ok) {
                const e = await memResp.json();
                const detail = e.detail;
                if (detail && detail.conflicts) {
                    const list = detail.conflicts.map(c => {
                        const g = this._groups.find(g => g.id === c.current_group_id);
                        return `${c.model_id} (${g ? g.name : `group ${c.current_group_id}`})`;
                    }).join(', ');
                    window.UIUtils?.showToast(`Model conflicts: ${list}`, 'error');
                } else {
                    window.UIUtils?.showToast(typeof detail === 'string' ? detail : 'Failed to save members.', 'error');
                }
                return;
            }

            if (applyToAll) {
                const limResp = await fetch(`/admin/profiles/model-group-limits/all?group_id=${group.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify(limits),
                });
                if (!limResp.ok) { const e = await limResp.json().catch(() => ({})); window.UIUtils?.showToast(e.detail || 'Failed to set limits.', 'error'); return; }
            }

            this._getModal()?.hide();
            await this._loadGroups();
            await this.renderProfileLimits();
        } catch (e) {
            window.UIUtils?.showToast('Network error saving group.', 'error');
        }
    }

    async deleteGroup(groupId, name) {
        const confirmed = await window.UIUtils?.showConfirmModal(
            'Delete Model Group',
            `Delete model group "${name}"? This removes the group and every profile's limit for it. Members will no longer be rate-limited by this group.`,
            'danger',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/model-groups?group_id=${groupId}`, {
                method: 'DELETE', credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast(`Group '${name}' deleted.`, 'success');
                await this._loadGroups();
                await this.renderProfileLimits();
            } else {
                const e = await resp.json();
                window.UIUtils?.showToast(e.detail || 'Failed to delete.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    // ---- Limits for the selected profile ----

    /** Called by RateLimitManager when the profile bar changes; *data* is its prefetched limits. */
    setProfile(profileId, data = null) {
        return this._profileLimits.setProfile(profileId, data);
    }

    renderProfileLimits() {
        return this._profileLimits.render();
    }

    saveProfileGroupLimit(groupId) {
        return this._profileLimits.save(groupId);
    }

    clearProfileGroupLimit(groupId) {
        return this._profileLimits.clear(groupId);
    }

    _getModal() {
        if (!this._modal) {
            const el = document.getElementById('modelGroupModal');
            if (el && window.bootstrap) this._modal = new window.bootstrap.Modal(el);
        }
        return this._modal;
    }
}

window.ModelGroupManager = new ModelGroupManager();

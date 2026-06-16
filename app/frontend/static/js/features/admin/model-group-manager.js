class ModelGroupManager {
    constructor() {
        this._allModels = [];
        this._groups = [];
        this._modal = null;
        this._selectedIds = new Set();
        this._selectedGroupId = '';
    }

    async load() {
        await Promise.all([this._loadGroups(), this._loadAllModels()]);
        this._populateGroupSelect();
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

    // ---- Global Defaults pane ----

    async _loadGroups() {
        const tbody = document.getElementById('model-groups-tbody');
        if (!tbody) return;
        try {
            const resp = await fetch('/admin/model-groups', { credentials: 'include' });
            if (!resp.ok) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">Failed to load model groups.</td></tr>';
                return;
            }
            this._groups = await resp.json();
            if (!this._groups.length) {
                tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-4">No model groups yet. Click "Create Model Group" to add one.</td></tr>';
                return;
            }
            tbody.innerHTML = this._groups.map(g => this._renderRow(g)).join('');
        } catch (e) {
            console.error('ModelGroupManager: failed to load groups', e);
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">Error loading data.</td></tr>';
        }
    }

    _renderRow(g) {
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const rpm = g.rpm_default !== null && g.rpm_default !== undefined ? g.rpm_default : '';
        const rpd = g.rpd_default !== null && g.rpd_default !== undefined ? g.rpd_default : '';
        return `
        <tr data-group-id="${g.id}">
            <td>
                <div class="fw-semibold">${esc(g.name)}</div>
                ${g.description ? `<div class="text-muted small">${esc(g.description)}</div>` : ''}
            </td>
            <td><span class="badge bg-secondary">${g.member_count}</span></td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm mg-rpm-input"
                    data-group-id="${g.id}" placeholder="Unlimited" value="${rpm}" style="width:120px;">
            </td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm mg-rpd-input"
                    data-group-id="${g.id}" placeholder="Unlimited" value="${rpd}" style="width:120px;">
            </td>
            <td>
                <button class="btn btn-primary btn-sm me-1" onclick="window.ModelGroupManager?.saveGroupLimits(${g.id})">
                    <i class="fas fa-save me-1"></i>Save
                </button>
                <button class="btn btn-outline-primary btn-sm me-1" onclick="window.ModelGroupManager?.openEditModal(${g.id})">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelGroupManager?.deleteGroup(${g.id}, '${esc(g.name)}')">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>`;
    }

    async saveGroupLimits(groupId) {
        const rpmInput = document.querySelector(`.mg-rpm-input[data-group-id="${groupId}"]`);
        const rpdInput = document.querySelector(`.mg-rpd-input[data-group-id="${groupId}"]`);
        const rpmRaw = rpmInput?.value.trim();
        const rpdRaw = rpdInput?.value.trim();
        const body = {
            rpm_default: rpmRaw !== '' ? parseInt(rpmRaw, 10) : null,
            rpd_default: rpdRaw !== '' ? parseInt(rpdRaw, 10) : null,
        };
        try {
            const resp = await fetch(`/admin/model-groups/limits?group_id=${groupId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Group limits saved.', 'success');
                await this._loadGroups();
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(e.detail || 'Failed to save limits.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving limits.', 'error');
        }
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
            document.getElementById('mg-rpm').value = g.rpm_default !== null && g.rpm_default !== undefined ? g.rpm_default : '';
            document.getElementById('mg-rpd').value = g.rpd_default !== null && g.rpd_default !== undefined ? g.rpd_default : '';
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
        const rpm_default = rpmRaw !== '' ? parseInt(rpmRaw, 10) : null;
        const rpd_default = rpdRaw !== '' ? parseInt(rpdRaw, 10) : null;

        if (!name) { window.UIUtils?.showToast('Name is required.', 'error'); return; }

        try {
            let group;
            if (!groupId) {
                const resp = await fetch('/admin/model-groups', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ name, description: description || null, rpm_default, rpd_default }),
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
                const limResp = await fetch(`/admin/model-groups/limits?group_id=${groupId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ rpm_default, rpd_default }),
                });
                if (!limResp.ok) { const e = await limResp.json(); window.UIUtils?.showToast(e.detail || 'Failed to update limits.', 'error'); return; }
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

            this._getModal()?.hide();
            await this._loadGroups();
            this._populateGroupSelect();
        } catch (e) {
            window.UIUtils?.showToast('Network error saving group.', 'error');
        }
    }

    async deleteGroup(groupId, name) {
        const confirmed = await window.UIUtils?.showConfirmModal(
            'Delete Model Group',
            `Delete model group "${name}"? This removes the group and all its per-user overrides. Members will no longer be rate-limited by this group.`,
            'danger',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/model-groups?group_id=${groupId}`, {
                method: 'DELETE', credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast(`Group '${name}' deleted.`, 'success');
                if (String(this._selectedGroupId) === String(groupId)) {
                    this._selectedGroupId = '';
                }
                await this._loadGroups();
                this._populateGroupSelect();
            } else {
                const e = await resp.json();
                window.UIUtils?.showToast(e.detail || 'Failed to delete.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    // ---- Per-User Overrides pane ----

    _populateGroupSelect() {
        const sel = document.getElementById('mg-user-group-select');
        if (!sel) return;
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const current = this._selectedGroupId || sel.value || '';
        sel.innerHTML = '<option value="">— Select a model group —</option>' +
            this._groups.map(g => `<option value="${g.id}">${esc(g.name)}</option>`).join('');
        if (current && this._groups.some(g => String(g.id) === String(current))) {
            sel.value = current;
        }
    }

    onGroupSelect(groupId) {
        this._selectedGroupId = groupId;
        this._loadUserOverrides();
    }

    refreshUserOverrides() {
        this._loadUserOverrides();
    }

    async _loadUserOverrides() {
        const tbody = document.getElementById('mg-user-overrides-tbody');
        if (!tbody) return;
        const groupId = this._selectedGroupId;
        if (!groupId) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">Select a model group to manage per-user overrides.</td></tr>';
            return;
        }
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">Loading…</td></tr>';
        try {
            const [usersResp, ovResp] = await Promise.all([
                fetch('/admin/users', { credentials: 'include' }),
                fetch(`/admin/model-groups/users?group_id=${groupId}`, { credentials: 'include' }),
            ]);
            const allUsers = usersResp.ok ? await usersResp.json() : [];
            const overrides = ovResp.ok ? await ovResp.json() : [];
            const ovByUser = new Map(overrides.map(o => [o.user_id, o]));

            if (!allUsers.length) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">No users found.</td></tr>';
                return;
            }
            tbody.innerHTML = allUsers
                .map(u => this._renderOverrideRow(u, ovByUser.get(u.id), groupId))
                .join('');
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger py-4">Error loading overrides.</td></tr>';
        }
    }

    _renderOverrideRow(user, ov, groupId) {
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const rpm = ov && ov.rpm_limit !== null && ov.rpm_limit !== undefined ? ov.rpm_limit : '';
        const rpd = ov && ov.rpd_limit !== null && ov.rpd_limit !== undefined ? ov.rpd_limit : '';
        const hasOverride = !!ov;
        return `
        <tr data-user-id="${user.id}" data-username="${esc(user.username)}">
            <td>
                <div class="fw-semibold">${esc(user.username)}</div>
                <div class="text-muted small">${esc(user.email || '')}</div>
            </td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm mg-override-rpm"
                    data-user-id="${user.id}" placeholder="Default" value="${rpm}" style="width:120px;">
            </td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm mg-override-rpd"
                    data-user-id="${user.id}" placeholder="Default" value="${rpd}" style="width:120px;">
            </td>
            <td>
                <button class="btn btn-primary btn-sm me-1" onclick="window.ModelGroupManager?.saveUserOverride(${groupId}, ${user.id})">
                    <i class="fas fa-save me-1"></i>Save
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelGroupManager?.removeUserOverride(${groupId}, ${user.id}, '${esc(user.username)}')" ${hasOverride ? '' : 'disabled'}>
                    <i class="fas fa-times me-1"></i>Clear
                </button>
            </td>
        </tr>`;
    }

    async saveUserOverride(groupId, userId) {
        const rpmInput = document.querySelector(`.mg-override-rpm[data-user-id="${userId}"]`);
        const rpdInput = document.querySelector(`.mg-override-rpd[data-user-id="${userId}"]`);
        const body = {};
        if (rpmInput) body.rpm_limit = rpmInput.value.trim() !== '' ? parseInt(rpmInput.value, 10) : null;
        if (rpdInput) body.rpd_limit = rpdInput.value.trim() !== '' ? parseInt(rpdInput.value, 10) : null;
        try {
            const resp = await fetch(`/admin/model-groups/users?group_id=${groupId}&user_id=${userId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Override saved.', 'success');
                await this._loadUserOverrides();
            } else {
                const e = await resp.json();
                window.UIUtils?.showToast(e.detail || 'Failed to save override.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    async removeUserOverride(groupId, userId, username) {
        try {
            const resp = await fetch(`/admin/model-groups/users?group_id=${groupId}&user_id=${userId}`, {
                method: 'DELETE', credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast(`Override removed for ${username}.`, 'success');
                await this._loadUserOverrides();
            } else {
                const e = await resp.json();
                window.UIUtils?.showToast(e.detail || 'Failed to remove override.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
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

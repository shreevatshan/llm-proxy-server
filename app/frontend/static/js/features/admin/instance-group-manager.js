class InstanceGroupManager {
    constructor() {
        this._allInstances = [];
        this._groups = [];
        this._modal = null;
        this._selectedKeys = new Set();
        // The 'Limits for profile' card; its profile is set by RateLimitManager from
        // the Rate Limits tab's profile bar.
        this._profileLimits = new window.ProfileGroupLimits({
            kind: 'instance', prefix: 'ig', noun: 'instances', manager: 'InstanceGroupManager',
        });
    }

    async load() {
        await Promise.all([this._loadGroups(), this._loadAllInstances()]);
        await this.renderProfileLimits();
    }

    // ---- Instances cache ----

    async _loadAllInstances() {
        try {
            const resp = await fetch('/admin/providers', { credentials: 'include' });
            if (!resp.ok) return;
            const providers = await resp.json();
            this._allInstances = providers.map(p => ({
                provider_key: p.provider_key,
                label: p.instance_name || p.provider_key,
                provider: p.provider_name || p.provider_type || '',
            }));
        } catch (e) {
            console.error('InstanceGroupManager: failed to load instances', e);
        }
    }

    // ---- Groups list pane ----

    async _loadGroups() {
        const tbody = document.getElementById('instance-groups-tbody');
        if (!tbody) return;
        try {
            const resp = await fetch('/admin/instance-groups', { credentials: 'include' });
            if (!resp.ok) {
                tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger py-4">Failed to load instance groups.</td></tr>';
                return;
            }
            this._groups = await resp.json();
            if (!this._groups.length) {
                tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-4">No instance groups yet. Click "Create Instance Group" to add one.</td></tr>';
                return;
            }
            tbody.innerHTML = this._groups.map(g => this._renderRow(g)).join('');
        } catch (e) {
            console.error('InstanceGroupManager: failed to load groups', e);
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
                <button class="btn btn-outline-primary btn-sm me-1" onclick="window.InstanceGroupManager?.openEditModal(${g.id})">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.InstanceGroupManager?.deleteGroup(${g.id}, '${esc(g.name)}')">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        </tr>`;
    }

    // ---- Create/Edit modal (group identity + members only) ----

    openCreateModal() {
        document.getElementById('ig-group-id').value = '';
        document.getElementById('ig-name').value = '';
        document.getElementById('ig-description').value = '';
        document.getElementById('ig-rpm').value = '';
        document.getElementById('ig-rpd').value = '';
        document.getElementById('instanceGroupModalTitle').textContent = 'New Instance Group';
        const search = document.getElementById('ig-instance-search');
        if (search) search.value = '';
        this._renderInstancePicker([]);
        this._getModal()?.show();
    }

    async openEditModal(groupId) {
        try {
            const resp = await fetch(`/admin/instance-groups?group_id=${groupId}`, { credentials: 'include' });
            if (!resp.ok) { window.UIUtils?.showToast('Failed to load group.', 'error'); return; }
            const g = await resp.json();

            document.getElementById('ig-group-id').value = g.id;
            document.getElementById('ig-name').value = g.name;
            document.getElementById('ig-description').value = g.description || '';
            document.getElementById('ig-rpm').value = '';
            document.getElementById('ig-rpd').value = '';
            document.getElementById('instanceGroupModalTitle').textContent = `Edit: ${g.name}`;
            const search = document.getElementById('ig-instance-search');
            if (search) search.value = '';

            if (!this._allInstances.length) await this._loadAllInstances();
            this._renderInstancePicker(g.members || []);
            this._getModal()?.show();
        } catch (e) {
            window.UIUtils?.showToast('Error loading group.', 'error');
        }
    }

    _renderInstancePicker(selectedKeys, filter = '') {
        this._selectedKeys = new Set(selectedKeys);
        const container = document.getElementById('ig-instance-picker');
        if (!container) return;
        if (!this._allInstances.length) {
            container.innerHTML = '<div class="text-muted small">No instances found.</div>';
            return;
        }
        const q = filter.trim().toLowerCase();
        const visible = q
            ? this._allInstances.filter(m =>
                m.label.toLowerCase().includes(q) ||
                m.provider_key.toLowerCase().includes(q) ||
                m.provider.toLowerCase().includes(q))
            : this._allInstances;
        if (!visible.length) {
            container.innerHTML = '<div class="text-muted small">No instances match your search.</div>';
            return;
        }
        const esc = window.UIUtils?.escapeHtml || (s => s);
        container.innerHTML = visible.map(m => {
            const checked = this._selectedKeys.has(m.provider_key) ? 'checked' : '';
            return `
            <div class="form-check">
                <input class="form-check-input" type="checkbox" id="igm-${CSS.escape(m.provider_key)}"
                    value="${esc(m.provider_key)}" ${checked}
                    onchange="window.InstanceGroupManager?._onInstanceToggle(this)">
                <label class="form-check-label small" for="igm-${CSS.escape(m.provider_key)}">
                    <span class="text-muted">${esc(m.provider)}</span>
                    / ${esc(m.label)}
                    <span class="text-muted">(${esc(m.provider_key)})</span>
                </label>
            </div>`;
        }).join('');
    }

    _onInstanceToggle(checkbox) {
        if (checkbox.checked) {
            this._selectedKeys.add(checkbox.value);
        } else {
            this._selectedKeys.delete(checkbox.value);
        }
    }

    filterInstances(query) {
        this._renderInstancePicker([...this._selectedKeys], query);
    }

    async saveGroup() {
        const groupId = document.getElementById('ig-group-id').value;
        const name = document.getElementById('ig-name').value.trim();
        const description = document.getElementById('ig-description').value.trim();
        const rpmRaw = document.getElementById('ig-rpm').value;
        const rpdRaw = document.getElementById('ig-rpd').value;
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
                const resp = await fetch('/admin/instance-groups', {
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
                const resp = await fetch(`/admin/instance-groups?group_id=${groupId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ name, description: description || null }),
                });
                if (!resp.ok) { const e = await resp.json(); window.UIUtils?.showToast(e.detail || 'Failed to update group.', 'error'); return; }
                group = { id: parseInt(groupId) };
                window.UIUtils?.showToast(`Group '${name}' updated.`, 'success');
            }

            // Save member list (use _selectedKeys to preserve selections made while search was filtered)
            const provider_keys = [...(this._selectedKeys || [])];
            const memResp = await fetch(`/admin/instance-groups/members?group_id=${group.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ provider_keys }),
            });
            if (!memResp.ok) {
                const e = await memResp.json();
                const detail = e.detail;
                if (detail && detail.conflicts) {
                    const list = detail.conflicts.map(c => {
                        const g = this._groups.find(g => g.id === c.current_group_id);
                        return `${c.provider_key} (${g ? g.name : `group ${c.current_group_id}`})`;
                    }).join(', ');
                    window.UIUtils?.showToast(`Instance conflicts: ${list}`, 'error');
                } else {
                    window.UIUtils?.showToast(typeof detail === 'string' ? detail : 'Failed to save members.', 'error');
                }
                return;
            }

            if (applyToAll) {
                const limResp = await fetch(`/admin/profiles/instance-group-limits/all?group_id=${group.id}`, {
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
            'Delete Instance Group',
            `Delete instance group "${name}"? This removes the group and every profile's limit for it. Members will no longer be rate-limited by this group.`,
            'danger',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/instance-groups?group_id=${groupId}`, {
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
            const el = document.getElementById('instanceGroupModal');
            if (el && window.bootstrap) this._modal = new window.bootstrap.Modal(el);
        }
        return this._modal;
    }
}

window.InstanceGroupManager = new InstanceGroupManager();

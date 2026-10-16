/**
 * The Rate Limits tab's 'Limits for profile' table: one row per model group, the
 * built-in Default group first, with the selected profile's RPM/RPD for it. Every model
 * is in exactly one group, so these rows are every limit the profile has.
 * Membership is edited in the Model Groups tab, not here.
 */
class ProfileGroupLimits {
    /**
     * @param {object} opts
     * @param {string} opts.manager  window global the row buttons call back into
     */
    constructor({ manager }) {
        this._manager = manager;
        this.profileId = null;
        this.profileName = '';
        this.limits = [];
    }

    /** *data*, when given, is a GET /admin/profiles/limits body the caller already fetched. */
    setProfile(profileId, data = null) {
        this.profileId = profileId;
        return this.render(data);
    }

    async render(data = null) {
        const tbody = document.getElementById('mg-profile-limits-tbody');
        if (!tbody) return;
        if (!this.profileId) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">Select a profile above.</td></tr>';
            return;
        }
        try {
            if (!data) {
                const resp = await fetch(`/admin/profiles/limits?profile_id=${this.profileId}`, { credentials: 'include' });
                if (!resp.ok) {
                    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger py-4">Failed to load profile limits.</td></tr>';
                    return;
                }
                data = await resp.json();
            }
            this.limits = data.groups || [];
            this.profileName = data.profile?.name || '';
            const esc = window.UIUtils?.escapeHtml || (s => s);
            const label = document.getElementById('mg-profile-name');
            if (label) label.textContent = this.profileName;
            const mgr = `window.${this._manager}`;
            tbody.innerHTML = this.limits.map(g => {
                const rpm = g.rpm_limit ?? '';
                const rpd = g.rpd_limit ?? '';
                const unlimited = g.rpm_limit == null && g.rpd_limit == null;
                // Each group's limit is the only one its models are under, so an
                // unlimited group is genuinely unlimited for this profile.
                const warn = unlimited && g.member_count
                    ? ' · <span class="text-warning"><i class="fas fa-triangle-exclamation me-1"></i>unlimited for this profile</span>'
                    : '';
                const off = g.enabled === false
                    ? ' <span class="badge bg-secondary ms-1" title="Switched off in Model Groups: its models are not served">Disabled</span>'
                    : '';
                const name = g.is_default
                    ? `<i class="fas fa-lock me-1 text-muted"></i>${esc(g.name)}`
                    : esc(g.name) + off;
                const what = g.is_default ? 'models in no other group' : 'models';
                return `
                <tr>
                    <td>
                        <div class="fw-semibold">${name}</div>
                        <div class="text-muted small">${g.member_count} ${what}${warn}</div>
                    </td>
                    <td>
                        <input type="number" min="0" class="form-control form-control-sm mg-pl-rpm"
                            data-group-id="${g.group_id}" placeholder="Unlimited" value="${rpm}" style="width:120px;">
                    </td>
                    <td>
                        <input type="number" min="0" class="form-control form-control-sm mg-pl-rpd"
                            data-group-id="${g.group_id}" placeholder="Unlimited" value="${rpd}" style="width:120px;">
                    </td>
                    <td class="text-nowrap">
                        <button class="btn btn-primary btn-sm me-1" onclick="${mgr}?.saveProfileGroupLimit(${g.group_id})">
                            <i class="fas fa-save me-1"></i>Save
                        </button>
                        <button class="btn btn-outline-danger btn-sm" onclick="${mgr}?.clearProfileGroupLimit(${g.group_id})" ${unlimited ? 'disabled' : ''}>
                            <i class="fas fa-times me-1"></i>Clear
                        </button>
                    </td>
                </tr>`;
            }).join('');
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger py-4">Error loading profile limits.</td></tr>';
        }
    }

    _read(groupId) {
        const rpm = document.querySelector(`.mg-pl-rpm[data-group-id="${groupId}"]`)?.value.trim() ?? '';
        const rpd = document.querySelector(`.mg-pl-rpd[data-group-id="${groupId}"]`)?.value.trim() ?? '';
        return {
            rpm_limit: rpm !== '' ? parseInt(rpm, 10) : null,
            rpd_limit: rpd !== '' ? parseInt(rpd, 10) : null,
        };
    }

    async _put(url, body, okMessage) {
        try {
            const resp = await fetch(url, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast(okMessage, 'success');
                await this.render();
                // Profile cards show the Default limits; keep them current.
                await window.ProfileStore?.notifyChanged?.();
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(typeof e.detail === 'string' ? e.detail : 'Failed to save limit.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    save(groupId) {
        return this._put(`/admin/profiles/model-group-limits?profile_id=${this.profileId}&group_id=${groupId}`,
            this._read(groupId), 'Limit saved.');
    }

    clear(groupId) {
        return this._put(`/admin/profiles/model-group-limits?profile_id=${this.profileId}&group_id=${groupId}`,
            { rpm_limit: null, rpd_limit: null }, 'Limit cleared (unlimited).');
    }
}

window.ProfileGroupLimits = ProfileGroupLimits;

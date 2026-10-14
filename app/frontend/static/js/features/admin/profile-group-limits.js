/**
 * The 'Limits for profile' card shared by the Model Groups and Instance Groups panes:
 * one row per group with the selected profile's RPM/RPD for it.
 */
class ProfileGroupLimits {
    /**
     * @param {object} opts
     * @param {'model'|'instance'} opts.kind  picks the endpoint and the response key
     * @param {string} opts.prefix            element-id / class prefix ('mg' | 'ig')
     * @param {string} opts.noun              what a group's members are ('models' | 'instances')
     * @param {string} opts.manager           window global the row buttons call back into
     */
    constructor({ kind, prefix, noun, manager }) {
        this._kind = kind;
        this._prefix = prefix;
        this._noun = noun;
        this._manager = manager;
        this.profileId = null;
        this.limits = [];
    }

    /** *data*, when given, is a GET /admin/profiles/limits body the caller already fetched. */
    setProfile(profileId, data = null) {
        this.profileId = profileId;
        return this.render(data);
    }

    async render(data = null) {
        const p = this._prefix;
        const tbody = document.getElementById(`${p}-profile-limits-tbody`);
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
            this.limits = data[`${this._kind}_groups`] || [];
            const esc = window.UIUtils?.escapeHtml || (s => s);
            const label = document.getElementById(`${p}-profile-name`);
            if (label) label.textContent = data.profile?.name || '';
            if (!this.limits.length) {
                tbody.innerHTML = `<tr><td colspan="4" class="text-center text-muted py-4">No ${this._kind} groups yet.</td></tr>`;
                return;
            }
            const mgr = `window.${this._manager}`;
            tbody.innerHTML = this.limits.map(g => {
                const rpm = g.rpm_limit ?? '';
                const rpd = g.rpd_limit ?? '';
                const unlimited = g.rpm_limit == null && g.rpd_limit == null;
                // A grouped model/instance skips the overall limit, so an unlimited group
                // is genuinely unlimited for this profile -- worth calling out.
                const warn = unlimited && g.member_count
                    ? ' · <span class="text-warning"><i class="fas fa-triangle-exclamation me-1"></i>unlimited for this profile</span>'
                    : '';
                return `
                <tr>
                    <td>
                        <div class="fw-semibold">${esc(g.name)}</div>
                        <div class="text-muted small">${g.member_count} ${this._noun}${warn}</div>
                    </td>
                    <td>
                        <input type="number" min="0" class="form-control form-control-sm ${p}-pl-rpm"
                            data-group-id="${g.group_id}" placeholder="Unlimited" value="${rpm}" style="width:120px;">
                    </td>
                    <td>
                        <input type="number" min="0" class="form-control form-control-sm ${p}-pl-rpd"
                            data-group-id="${g.group_id}" placeholder="Unlimited" value="${rpd}" style="width:120px;">
                    </td>
                    <td>
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

    async _put(groupId, body, okMessage) {
        try {
            const resp = await fetch(`/admin/profiles/${this._kind}-group-limits?profile_id=${this.profileId}&group_id=${groupId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast(okMessage, 'success');
                await this.render();
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(e.detail || 'Failed to save limit.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    save(groupId) {
        const p = this._prefix;
        const rpm = document.querySelector(`.${p}-pl-rpm[data-group-id="${groupId}"]`)?.value.trim() ?? '';
        const rpd = document.querySelector(`.${p}-pl-rpd[data-group-id="${groupId}"]`)?.value.trim() ?? '';
        return this._put(groupId, {
            rpm_limit: rpm !== '' ? parseInt(rpm, 10) : null,
            rpd_limit: rpd !== '' ? parseInt(rpd, 10) : null,
        }, 'Limit saved.');
    }

    clear(groupId) {
        return this._put(groupId, { rpm_limit: null, rpd_limit: null }, 'Limit cleared (unlimited).');
    }
}

window.ProfileGroupLimits = ProfileGroupLimits;

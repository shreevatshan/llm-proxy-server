/**
 * Rate Limits tab. Every limit belongs to a user profile: the profile bar at the top
 * picks which one the Overall, Model and Instance panes edit.
 */
class RateLimitManager {
    constructor() {
        this._refreshTimer = null;
        this._REFRESH_INTERVAL = 10000;
        this._members = [];
        this._profileId = null;
        document.addEventListener('profiles:changed', () => this._onProfilesChanged());
    }

    async _onProfilesChanged() {
        const before = this._profileId;
        await this._fillProfileSelect();
        // The shown profile was deleted and the picker fell back to another: reload the
        // panes, or the next Save would write the old profile's values into this one.
        if (before !== null && this._profileId !== before) await this._loadProfile();
    }

    async load() {
        await this._fillProfileSelect();
        await this._loadProfile();
        this._startAutoRefresh();
    }

    async _fillProfileSelect() {
        const sel = document.getElementById('rl-profile-select');
        if (!sel) return;
        const want = window.ProfileStore.selected('rate-limits', window.ProfileStore.DEFAULT_ID);
        await window.ProfileStore.fillSelect(sel, { selectedId: want });
        this._profileId = sel.value ? parseInt(sel.value, 10) : null;
        if (!window.ProfileStore.byId(this._profileId)) this._profileId = window.ProfileStore.DEFAULT_ID;
    }

    async onProfileChange(value) {
        this._profileId = value ? parseInt(value, 10) : null;
        window.ProfileStore.select('rate-limits', this._profileId);
        await this._loadProfile();
    }

    async _loadProfile() {
        const members = this._loadMembers();
        // One GET serves all three panes (overall, model groups, instance groups).
        // On failure the group panes fall back to fetching for themselves.
        const limits = await this._fetchLimits();
        this._renderOverall(limits);
        await Promise.all([
            members,
            window.ModelGroupManager?.setProfile(this._profileId, limits),
            window.InstanceGroupManager?.setProfile(this._profileId, limits),
        ]);
    }

    stopAutoRefresh() {
        if (this._refreshTimer) {
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    }

    _startAutoRefresh() {
        this.stopAutoRefresh();
        this._refreshTimer = setInterval(() => {
            // Don't re-render while the admin is typing in the search box,
            // otherwise their input/focus would be wiped every tick.
            if (document.activeElement?.id === 'rl-user-search') return;
            this._loadMembers();
        }, this._REFRESH_INTERVAL);
    }

    async _fetchLimits() {
        if (!this._profileId) return null;
        try {
            const resp = await fetch(`/admin/profiles/limits?profile_id=${this._profileId}`, { credentials: 'include' });
            return resp.ok ? await resp.json() : null;
        } catch (e) {
            console.error('RateLimitManager: failed to load profile limits', e);
            return null;
        }
    }

    _renderOverall(limits) {
        const profile = limits?.profile;
        if (!profile) return;
        document.getElementById('rl-profile-rpm').value = profile.rpm_limit ?? '';
        document.getElementById('rl-profile-rpd').value = profile.rpd_limit ?? '';
        document.getElementById('rl-profile-title').textContent = profile.name;
        document.getElementById('rl-profile-members').textContent =
            `${profile.member_count} member${profile.member_count === 1 ? '' : 's'}`;
    }

    async _loadMembers() {
        const tbody = document.getElementById('rate-limits-tbody');
        if (!tbody || !this._profileId) return;
        try {
            const resp = await fetch(`/admin/profiles/members?profile_id=${this._profileId}`, { credentials: 'include' });
            if (!resp.ok) {
                this._members = [];
                tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger py-4">Failed to load members.</td></tr>';
                return;
            }
            this._members = await resp.json();
            this._applyFilter();
        } catch (e) {
            console.error('RateLimitManager: failed to load members', e);
            this._members = [];
            tbody.innerHTML = '<tr><td colspan="3" class="text-center text-danger py-4">Error loading data.</td></tr>';
        }
    }

    filterUsers() {
        this._applyFilter();
    }

    _applyFilter() {
        const tbody = document.getElementById('rate-limits-tbody');
        if (!tbody) return;
        if (!this._members.length) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-4">No users in this profile.</td></tr>';
            return;
        }
        const q = (document.getElementById('rl-user-search')?.value || '').trim().toLowerCase();
        const filtered = q
            ? this._members.filter(u =>
                (u.username || '').toLowerCase().includes(q)
                || (u.email || '').toLowerCase().includes(q))
            : this._members;
        if (!filtered.length) {
            tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-4">No users match your search.</td></tr>';
            return;
        }
        const esc = window.UIUtils.escapeHtml;
        tbody.innerHTML = filtered.map(u => `
            <tr>
                <td>
                    <div class="fw-semibold">${esc(u.username)}${u.is_active ? '' : ' <span class="status-badge status-inactive ms-1">Inactive</span>'}</div>
                    <div class="text-muted small">${esc(u.email)}</div>
                </td>
                <td>${u.current_rpm_count}</td>
                <td>${u.current_rpd_count}</td>
            </tr>`).join('');
    }

    async saveProfileLimits() {
        if (!this._profileId) return;
        const rpmRaw = document.getElementById('rl-profile-rpm')?.value.trim();
        const rpdRaw = document.getElementById('rl-profile-rpd')?.value.trim();
        const body = {
            rpm_limit: rpmRaw !== '' ? parseInt(rpmRaw, 10) : null,
            rpd_limit: rpdRaw !== '' ? parseInt(rpdRaw, 10) : null,
        };
        try {
            const resp = await fetch(`/admin/profiles/limits?profile_id=${this._profileId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Profile limits saved.', 'success');
                await window.ProfileStore.notifyChanged();
                await this._loadMembers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to save limits.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving limits.', 'error');
        }
    }
}

window.RateLimitManager = new RateLimitManager();

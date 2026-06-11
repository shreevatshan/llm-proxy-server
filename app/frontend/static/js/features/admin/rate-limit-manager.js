class RateLimitManager {
    constructor() {
        this._refreshTimer = null;
        this._REFRESH_INTERVAL = 10000;
    }

    async load() {
        await Promise.all([this._loadDefaults(), this._loadUsers()]);
        this._startAutoRefresh();
    }

    stopAutoRefresh() {
        if (this._refreshTimer) {
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    }

    _startAutoRefresh() {
        this.stopAutoRefresh();
        this._refreshTimer = setInterval(() => this._loadUsers(), this._REFRESH_INTERVAL);
    }

    async _loadDefaults() {
        try {
            const resp = await fetch('/admin/rate-limits/defaults', { credentials: 'include' });
            if (!resp.ok) return;
            const data = await resp.json();
            const rpmInput = document.getElementById('rl-default-rpm');
            const rpdInput = document.getElementById('rl-default-rpd');
            if (rpmInput) rpmInput.value = data.rpm_default !== null && data.rpm_default !== undefined ? data.rpm_default : '';
            if (rpdInput) rpdInput.value = data.rpd_default !== null && data.rpd_default !== undefined ? data.rpd_default : '';
        } catch (e) {
            console.error('RateLimitManager: failed to load defaults', e);
        }
    }

    async _loadUsers() {
        const tbody = document.getElementById('rate-limits-tbody');
        if (!tbody) return;
        try {
            const resp = await fetch('/admin/rate-limits/users', { credentials: 'include' });
            if (!resp.ok) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger py-4">Failed to load rate limit data.</td></tr>';
                return;
            }
            const users = await resp.json();
            if (!users.length) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">No users found.</td></tr>';
                return;
            }
            tbody.innerHTML = users.map(u => this._renderRow(u)).join('');
        } catch (e) {
            console.error('RateLimitManager: failed to load users', e);
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger py-4">Error loading data.</td></tr>';
        }
    }

    _renderRow(u) {
        const rpmVal = u.rpm_limit !== null && u.rpm_limit !== undefined ? u.rpm_limit : '';
        const rpdVal = u.rpd_limit !== null && u.rpd_limit !== undefined ? u.rpd_limit : '';
        const safeUsername = window.UIUtils.escapeHtml(u.username);
        const safeEmail = window.UIUtils.escapeHtml(u.email);
        return `
        <tr data-user-id="${u.user_id}" data-username="${safeUsername}">
            <td>
                <div class="fw-semibold">${safeUsername}</div>
                <div class="text-muted small">${safeEmail}</div>
            </td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm rl-rpm-input"
                    data-user-id="${u.user_id}" placeholder="Default" value="${rpmVal}" style="width:120px;">
            </td>
            <td>
                <input type="number" min="0" class="form-control form-control-sm rl-rpd-input"
                    data-user-id="${u.user_id}" placeholder="Default" value="${rpdVal}" style="width:120px;">
            </td>
            <td>
                <button class="btn btn-primary btn-sm me-1" onclick="window.RateLimitManager.saveUserLimits(${u.user_id})">
                    <i class="fas fa-save me-1"></i>Save
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.RateLimitManager.clearUserOverride(${u.user_id})">
                    <i class="fas fa-times me-1"></i>Clear
                </button>
            </td>
        </tr>`;
    }

    async saveDefaults() {
        const rpmRaw = document.getElementById('rl-default-rpm')?.value.trim();
        const rpdRaw = document.getElementById('rl-default-rpd')?.value.trim();
        const body = {
            rpm_default: rpmRaw !== '' ? parseInt(rpmRaw, 10) : null,
            rpd_default: rpdRaw !== '' ? parseInt(rpdRaw, 10) : null,
        };
        try {
            const resp = await fetch('/admin/rate-limits/defaults', {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Global defaults saved.', 'success');
                await this._loadUsers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to save defaults.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving defaults.', 'error');
        }
    }

    async saveUserLimits(userId) {
        const rpmInput = document.querySelector(`.rl-rpm-input[data-user-id="${userId}"]`);
        const rpdInput = document.querySelector(`.rl-rpd-input[data-user-id="${userId}"]`);
        const rpmRaw = rpmInput?.value.trim();
        const rpdRaw = rpdInput?.value.trim();
        const body = {
            rpm_limit: rpmRaw !== '' ? parseInt(rpmRaw, 10) : null,
            rpd_limit: rpdRaw !== '' ? parseInt(rpdRaw, 10) : null,
        };
        try {
            const resp = await fetch(`/admin/rate-limits/users?user_id=${userId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Rate limit saved.', 'success');
                await this._loadUsers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to save rate limit.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving rate limit.', 'error');
        }
    }

    async clearUserOverride(userId) {
        const row = document.querySelector(`tr[data-user-id="${userId}"]`);
        const username = row?.dataset.username || `user ${userId}`;
        const confirmed = await window.UIUtils?.showConfirmModal(
            'Clear Override',
            `Remove rate limit override for ${username}? They will fall back to global defaults.`,
            'warning',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/rate-limits/users?user_id=${userId}`, {
                method: 'DELETE',
                credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast(`Override removed for ${username}.`, 'success');
                await this._loadUsers();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to remove override.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error removing override.', 'error');
        }
    }
}

window.RateLimitManager = new RateLimitManager();

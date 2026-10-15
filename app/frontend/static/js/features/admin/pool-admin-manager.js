/**
 * Request pools (admin).
 *
 * Two jobs: the global pool policy (max members per pool, the most of a pool's daily
 * limit one member may use), and the list of every pool -- each row drills down to its
 * members, as Usage › By Pool does -- with the admin's force actions: per-pool limit
 * overrides, kick, delete.
 * Pool usage lives in Usage › By Pool and is not repeated here.
 */

class PoolAdminManager {
    constructor() {
        this._settings = null;
        this._pools = [];
        this._openPoolId = null;         // the pool drilled into, or null for the list
        this._limitsModal = null;
    }

    async load() {
        const container = document.getElementById('pools-table-container');
        try {
            const [settingsResp, poolsResp] = await Promise.all([
                fetch('/admin/pools/settings', { credentials: 'include' }),
                fetch('/admin/pools', { credentials: 'include' }),
            ]);
            if (!settingsResp.ok || !poolsResp.ok) throw new Error();
            this._settings = await settingsResp.json();
            this._pools = await poolsResp.json();
        } catch (e) {
            if (container) container.innerHTML = '<p class="text-center text-danger py-4">Failed to load pools.</p>';
            return;
        }
        this._renderSettings();
        this._renderPools();
    }

    // ---- global settings ----

    _renderSettings() {
        const s = this._settings;
        if (!s) return;
        const members = document.getElementById('pool-settings-max-members');
        const share = document.getElementById('pool-settings-share');
        if (members) {
            members.value = s.max_members;
            members.max = s.hard_cap;
        }
        if (share) share.value = s.max_member_share_pct ?? '';
        const hint = document.getElementById('pool-settings-max-members-hint');
        if (hint) hint.textContent = `Between 2 and ${s.hard_cap}. Lowering it removes nobody; full pools just stop accepting members.`;
    }

    async saveSettings() {
        const maxMembers = this._intOrNull(document.getElementById('pool-settings-max-members')?.value);
        const share = this._intOrNull(document.getElementById('pool-settings-share')?.value);
        if (maxMembers === null) {
            window.UIUtils?.showToast('Max members is required.', 'error');
            return;
        }
        try {
            const resp = await fetch('/admin/pools/settings', {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ max_members: maxMembers, max_member_share_pct: share }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                window.UIUtils?.showToast(this._detail(data, 'Failed to save pool settings.'), 'error');
                return;
            }
            this._settings = data;
            window.UIUtils?.showToast('Pool settings saved.', 'success');
            await this.load();
        } catch (e) {
            window.UIUtils?.showToast('Network error saving pool settings.', 'error');
        }
    }

    // ---- pool list ----
    //
    // Two levels, like Usage › By Pool: the table of pools, and one pool's members with
    // a breadcrumb and Back to return. Usage is deliberately not repeated here.

    _renderPools() {
        const container = document.getElementById('pools-table-container');
        if (!container) return;
        // A pool that has gone (deleted, or its last member kicked) drops back to the list.
        const open = this._pools.find(p => p.id === this._openPoolId);
        if (!open) this._openPoolId = null;
        const back = document.getElementById('pools-back');
        if (back) back.style.display = open ? '' : 'none';
        container.innerHTML = open ? this._membersHtml(open) : this._poolTableHtml();
    }

    _poolTableHtml() {
        if (!this._pools.length) {
            return '<p class="text-center text-muted py-4">No pools yet. Users create them from their dashboard.</p>';
        }
        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Pool</th><th>Owner</th><th>Members</th><th>Share cap</th><th style="width:1%">Actions</th></tr></thead>
                    <tbody>${this._pools.map(p => this._poolRowHtml(p)).join('')}</tbody>
                </table>
            </div>`;
    }

    _poolRowHtml(p) {
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const over = p.member_count > p.effective_max_members;
        const full = p.member_count >= p.effective_max_members;
        const pct = p.effective_max_member_share_pct;
        const override = '<span class="badge bg-secondary ms-1" title="Set on this pool; overrides the global setting">override</span>';
        const capacity = over ? '<span class="badge bg-warning text-dark ms-1">over capacity</span>'
            : full ? '<span class="badge bg-secondary ms-1">full</span>' : '';
        const share = pct === null || pct === undefined || pct >= 100 ? 'No cap' : `${pct}%`;
        return `
            <tr role="button" tabindex="0" style="cursor:pointer;" title="Click to see members"
                onclick="window.PoolAdminManager.open(${p.id})"
                onkeydown="if (event.target === this && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); window.PoolAdminManager.open(${p.id}); }">
                <td>
                    <div class="fw-semibold">${esc(p.name)}</div>
                    ${p.description ? `<div class="text-muted small">${esc(p.description)}</div>` : ''}
                </td>
                <td>${esc(p.owner_username || '—')}</td>
                <td class="text-nowrap">${p.member_count} / ${p.effective_max_members}${p.max_members_override !== null ? override : ''}${capacity}</td>
                <td class="text-nowrap">${share}${p.max_member_share_pct_override !== null ? override : ''}</td>
                <td class="text-nowrap" onclick="event.stopPropagation()">
                    <button class="btn btn-outline-primary btn-sm" title="Override this pool's limits"
                        onclick="window.PoolAdminManager.openLimits(${p.id})"><i class="fas fa-sliders"></i></button>
                    <button class="btn btn-outline-danger btn-sm" title="Delete pool"
                        onclick="window.PoolAdminManager.deletePool(${p.id})"><i class="fas fa-trash"></i></button>
                </td>
            </tr>`;
    }

    /** One pool's members, under the same `Pools › name` trail the Usage tab uses. */
    _membersHtml(p) {
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const crumb = `
            <div class="mb-3 d-flex align-items-center flex-wrap" style="gap:.5rem;">
                <span role="button" tabindex="0" onclick="window.PoolAdminManager.back()"
                    onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.PoolAdminManager.back(); }"
                    style="cursor:pointer;text-decoration:underline;text-underline-offset:3px;color:var(--mono-text-muted);">All pools</span>
                <span class="text-muted">&rsaquo;</span>
                <span class="fw-semibold">${esc(p.name)}</span>
            </div>`;
        const rows = (p.members || []).map(m => `
            <tr>
                <td>${esc(m.username)}</td>
                <td>${m.is_owner ? 'Owner' : 'Member'}</td>
                <td>${m.is_active
                    ? '<span class="status-badge status-active">Active</span>'
                    : '<span class="status-badge status-inactive">Inactive</span>'}</td>
                <td>${this._date(m.joined_at)}</td>
                <td class="text-end">
                    <button class="btn btn-outline-danger btn-sm" title="Remove ${esc(m.username)} from ${esc(p.name)}"
                        onclick="window.PoolAdminManager.kick(${p.id}, ${m.user_id})">
                        <i class="fas fa-user-minus me-1"></i>Kick
                    </button>
                </td>
            </tr>`).join('');
        return `${crumb}
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Member</th><th>Role</th><th>Status</th><th>Joined</th><th style="width:1%"></th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }

    open(poolId) {
        this._openPoolId = poolId;
        this._renderPools();
    }

    back() {
        this._openPoolId = null;
        this._renderPools();
    }

    // ---- actions ----

    async kick(poolId, userId) {
        const pool = this._pools.find(p => p.id === poolId);
        const member = pool?.members.find(m => m.user_id === userId);
        if (!pool || !member) return;
        let consequence = 'Their share of what the pool used today stays charged to them.';
        if (pool.member_count === 1) consequence += ` They are the last member, so the pool will be deleted.`;
        else if (member.is_owner) consequence += ` They own the pool, so ownership passes to the longest-standing member.`;
        const ok = await window.UIUtils?.showConfirmModal(
            'Kick from pool', `Remove ${member.username} from ${pool.name}? ${consequence}`, 'danger',
        );
        if (!ok) return;
        await this._send(`/admin/pools/members?user_id=${userId}`, 'DELETE', null, 'Failed to remove member.');
    }

    async deletePool(poolId) {
        const pool = this._pools.find(p => p.id === poolId);
        if (!pool) return;
        const ok = await window.UIUtils?.showConfirmModal(
            'Delete pool',
            `Delete ${pool.name} and release its ${pool.member_count} member${pool.member_count === 1 ? '' : 's'}? `
            + 'Each keeps their settled share of what the pool used today.',
            'danger',
        );
        if (!ok) return;
        await this._send(`/admin/pools?pool_id=${poolId}`, 'DELETE', null, 'Failed to delete pool.');
    }

    openLimits(poolId) {
        const pool = this._pools.find(p => p.id === poolId);
        if (!pool) return;
        const s = this._settings || {};
        document.getElementById('pool-limits-id').value = poolId;
        document.getElementById('poolLimitsModalTitle').textContent = `Limits — ${pool.name}`;
        const members = document.getElementById('pool-limits-max-members');
        members.value = pool.max_members_override ?? '';
        members.placeholder = `Global: ${s.max_members}`;
        if (s.hard_cap) members.max = s.hard_cap;
        const share = document.getElementById('pool-limits-share');
        share.value = pool.max_member_share_pct_override ?? '';
        share.placeholder = s.max_member_share_pct === null || s.max_member_share_pct === undefined
            ? 'Global: no cap' : `Global: ${s.max_member_share_pct}%`;
        const hint = document.getElementById('pool-limits-members-hint');
        hint.textContent = `${pool.member_count} member${pool.member_count === 1 ? '' : 's'} now. Lowering below that removes nobody.`;
        if (!this._limitsModal && window.bootstrap) {
            this._limitsModal = new window.bootstrap.Modal(document.getElementById('poolLimitsModal'));
        }
        this._limitsModal?.show();
    }

    async saveLimits() {
        const poolId = parseInt(document.getElementById('pool-limits-id').value, 10);
        const body = {
            // Blank sends null, which clears the override so the pool inherits again.
            max_members: this._intOrNull(document.getElementById('pool-limits-max-members').value),
            max_member_share_pct: this._intOrNull(document.getElementById('pool-limits-share').value),
        };
        const ok = await this._send(`/admin/pools/limits?pool_id=${poolId}`, 'PUT', body, 'Failed to save pool limits.');
        if (ok) this._limitsModal?.hide();
    }

    async _send(url, method, body, failure) {
        try {
            const resp = await fetch(url, {
                method,
                credentials: 'include',
                headers: body ? { 'Content-Type': 'application/json' } : undefined,
                body: body ? JSON.stringify(body) : undefined,
            });
            const data = await resp.json().catch(() => ({}));
            const esc = window.UIUtils?.escapeHtml || (s => s);
            if (!resp.ok) {
                window.UIUtils?.showToast(esc(this._detail(data, failure)), 'error');
                return false;
            }
            // Messages name pools and users; the toast renders HTML.
            window.UIUtils?.showToast(esc(data.message || 'Done.'), 'success');
            await this.load();
            return true;
        } catch (e) {
            window.UIUtils?.showToast(failure, 'error');
            return false;
        }
    }

    // ---- helpers ----

    _intOrNull(raw) {
        const v = String(raw ?? '').trim();
        if (v === '') return null;
        const n = parseInt(v, 10);
        return Number.isNaN(n) ? null : n;
    }

    _date(iso) {
        if (!iso) return '';
        return new Date(iso + (iso.endsWith('Z') ? '' : 'Z')).toLocaleDateString();
    }

    _detail(data, fallback) {
        const d = data && data.detail;
        if (!d) return fallback;
        if (typeof d === 'string') return d;
        if (Array.isArray(d)) return d.map(e => (e.msg || '').replace(/^Value error, /, '')).filter(Boolean).join('; ') || fallback;
        return fallback;
    }
}

window.PoolAdminManager = new PoolAdminManager();

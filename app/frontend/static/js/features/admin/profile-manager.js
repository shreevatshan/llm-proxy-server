/**
 * User profiles.
 *
 * ProfileStore is the shared list every tab picks a profile from (Users, Rate Limits,
 * Models, Model Mapping). Each tab remembers its own choice for the session, and every
 * tab rebuilds its picker on the `profiles:changed` event fired after any profile edit.
 *
 * ProfileManager drives the Profiles tab: create (optionally as a copy), rename, delete
 * with member reassignment, and the Members table that moves users between profiles.
 */

class ProfileStore {
    constructor() {
        this._profiles = null;
        this._pending = null;
        this._fetchSeq = 0;
        this.DEFAULT_ID = 1;
    }

    async list({ force = false } = {}) {
        if (this._profiles && !force) return this._profiles;
        if (!this._pending || force) {
            const seq = ++this._fetchSeq;
            this._pending = fetch('/admin/profiles', { credentials: 'include' })
                .then(r => (r.ok ? r.json() : []))
                .catch(() => [])
                .then(list => {
                    // A later forced fetch supersedes this one: never let an older
                    // response overwrite a newer list.
                    if (seq !== this._fetchSeq) return this._pending || this._profiles;
                    this._profiles = list;
                    this._pending = null;
                    return list;
                });
        }
        return this._pending;
    }

    byId(id) {
        return (this._profiles || []).find(p => String(p.id) === String(id)) || null;
    }

    /**
     * Fill a <select> with profiles, keeping `selectedId` when it still exists.
     * `emptyLabel` adds a first option with value '' (no profile).
     */
    async fillSelect(el, { emptyLabel = null, selectedId = null, showCounts = true } = {}) {
        if (!el) return;
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const profiles = await this.list();
        const opts = profiles.map(p => {
            const count = showCounts ? ` (${p.member_count})` : '';
            return `<option value="${p.id}">${esc(p.name)}${count}</option>`;
        });
        if (emptyLabel !== null) opts.unshift(`<option value="">${esc(emptyLabel)}</option>`);
        el.innerHTML = opts.join('');
        const want = selectedId === null || selectedId === undefined ? '' : String(selectedId);
        if ([...el.options].some(o => o.value === want)) el.value = want;
        else if (emptyLabel === null && profiles.length) el.value = String(this.DEFAULT_ID);
    }

    /** The profile a tab last showed, remembered per browser session. */
    selected(feature, fallback = null) {
        try {
            const v = sessionStorage.getItem(`profile:${feature}`);
            if (v === null) return fallback;
            return v === '' ? null : parseInt(v, 10);
        } catch (e) {
            return fallback;
        }
    }

    select(feature, id) {
        try { sessionStorage.setItem(`profile:${feature}`, id === null || id === undefined ? '' : String(id)); } catch (e) { /* storage blocked */ }
    }

    async notifyChanged() {
        await this.list({ force: true });
        document.dispatchEvent(new CustomEvent('profiles:changed'));
    }
}


class ProfileManager {
    constructor() {
        this._modal = null;
        this._deleteModal = null;
        this._deletingId = null;
        this._users = [];                 // [{id, username, email, is_active, is_pending_approval, profile_id}]
        this._selected = new Set();       // user ids ticked in the Members table
        // Every profile change can move members (a delete reassigns them), so the
        // Members table is reloaded along with the profile list.
        document.addEventListener('profiles:changed', () => {
            this._render();
            this._loadMembers();
        });
    }

    async load() {
        await window.ProfileStore.list({ force: true });
        this._render();
        await this._loadMembers();
    }

    _render() {
        const tbody = document.getElementById('profiles-tbody');
        if (!tbody) return;
        const profiles = window.ProfileStore._profiles || [];
        const esc = window.UIUtils?.escapeHtml || (s => s);
        if (!profiles.length) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-4">No profiles.</td></tr>';
            return;
        }
        tbody.innerHTML = profiles.map(p => `
            <tr>
                <td>
                    <div class="fw-semibold">${esc(p.name)}
                        ${p.is_default && p.name.toLowerCase() !== 'default' ? '<span class="badge profile-badge-default ms-1">Default</span>' : ''}</div>
                    ${p.description ? `<div class="text-muted small">${esc(p.description)}</div>` : ''}
                </td>
                <td><span class="badge bg-secondary profile-count-link" role="button" tabindex="0"
                    title="Show members of ${esc(p.name)}" aria-label="Show members of ${esc(p.name)}"
                    onclick="window.ProfileManager.showMembersOf(${p.id})"
                    onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.ProfileManager.showMembersOf(${p.id}); }">${p.member_count}</span></td>
                <td>${p.alias_count}</td>
                <td class="text-nowrap">
                    <button class="btn btn-outline-primary btn-sm" title="Edit"
                        onclick="window.ProfileManager.openEdit(${p.id})"><i class="fas fa-edit"></i></button>
                    <button class="btn btn-outline-primary btn-sm" title="Duplicate"
                        onclick="window.ProfileManager.openCreate(${p.id})"><i class="fas fa-copy"></i></button>
                    <button class="btn btn-outline-danger btn-sm" title="${p.is_default ? 'The default profile cannot be deleted' : 'Delete'}"
                        ${p.is_default ? 'disabled' : ''}
                        onclick="window.ProfileManager.openDelete(${p.id})"><i class="fas fa-trash"></i></button>
                </td>
            </tr>`).join('');
    }

    // ---- members ----

    async _loadMembers() {
        const tbody = document.getElementById('profile-members-tbody');
        if (!tbody) return;
        try {
            const resp = await fetch('/admin/users', { credentials: 'include' });
            if (!resp.ok) throw new Error();
            this._users = (await resp.json()).sort((a, b) => a.username.localeCompare(b.username));
        } catch (e) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">Failed to load users.</td></tr>';
            return;
        }
        const known = new Set(this._users.map(u => String(u.id)));
        this._selected = new Set([...this._selected].filter(id => known.has(id)));
        await this._fillMemberSelects();
        this._renderMembers();
    }

    async _fillMemberSelects() {
        const store = window.ProfileStore;
        const filter = document.getElementById('profile-members-filter');
        const want = filter?.value ?? '';
        await store.fillSelect(filter, { emptyLabel: 'All profiles', selectedId: want === '' ? null : want });
        const target = document.getElementById('profile-members-target');
        await store.fillSelect(target, { selectedId: target?.value || store.DEFAULT_ID, showCounts: false });
    }

    _profileOf(user) {
        return user.profile_id ?? window.ProfileStore.DEFAULT_ID;
    }

    /** Users the search box and profile filter currently show. */
    _shownMembers() {
        const q = (document.getElementById('profile-members-search')?.value || '').trim().toLowerCase();
        const only = document.getElementById('profile-members-filter')?.value || '';
        return this._users.filter(u =>
            (!only || String(this._profileOf(u)) === only)
            && (!q || u.username.toLowerCase().includes(q) || (u.email || '').toLowerCase().includes(q)));
    }

    _renderMembers() {
        const tbody = document.getElementById('profile-members-tbody');
        if (!tbody) return;
        const esc = window.UIUtils?.escapeHtml || (s => s);
        const shown = this._shownMembers();
        if (!shown.length) {
            const msg = this._users.length ? 'No users match.' : 'No users yet.';
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-muted py-4">${msg}</td></tr>`;
        } else {
            const options = (window.ProfileStore._profiles || [])
                .map(p => `<option value="${p.id}">${esc(p.name)}</option>`).join('');
            const busy = this._selected.size > 0;
            tbody.innerHTML = shown.map(u => {
                const status = u.is_pending_approval ? ['status-pending', 'Pending']
                    : u.is_active ? ['status-active', 'Active'] : ['status-inactive', 'Inactive'];
                return `
                <tr class="profile-member-row${this._selected.has(String(u.id)) ? ' selected' : ''}">
                    <td class="col-select"><input type="checkbox" class="form-check-input"
                        aria-label="Select ${esc(u.username)}" ${this._selected.has(String(u.id)) ? 'checked' : ''}
                        onchange="window.ProfileManager.toggleMember(${u.id}, this.checked)"></td>
                    <td>${esc(u.username)}</td>
                    <td>${esc(u.email || '')}</td>
                    <td><span class="status-badge ${status[0]}">${status[1]}</span></td>
                    <td><select class="form-select form-select-sm user-profile-select" data-user-id="${u.id}"
                        aria-label="Profile for ${esc(u.username)}" ${busy ? 'disabled' : ''}
                        onchange="window.ProfileManager.moveOne(${u.id}, this)">${options}</select></td>
                </tr>`;
            }).join('');
            // Set after rendering: an <option selected> per row would need the id in the loop above.
            tbody.querySelectorAll('select[data-user-id]').forEach(sel => {
                const user = this._users.find(u => String(u.id) === sel.dataset.userId);
                sel.value = String(this._profileOf(user));
            });
        }
        this._syncMemberSelection(shown);
    }

    _syncMemberSelection(shown = this._shownMembers()) {
        const n = this._selected.size;
        const bar = document.getElementById('profile-members-bulk');
        if (bar) bar.style.display = n ? '' : 'none';
        const count = document.getElementById('profile-members-count');
        if (count) count.textContent = `${n} selected — move to`;
        const all = document.getElementById('profile-members-select-all');
        if (all) {
            const picked = shown.filter(u => this._selected.has(String(u.id))).length;
            all.checked = shown.length > 0 && picked === shown.length;
            all.indeterminate = picked > 0 && picked < shown.length;
        }
    }

    filterMembers() {
        this._renderMembers();
    }

    /** Members count in the profiles table: narrow the Members table to that profile. */
    showMembersOf(profileId) {
        const filter = document.getElementById('profile-members-filter');
        if (filter) filter.value = String(profileId);
        this._renderMembers();
        document.getElementById('profile-members-tbody')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    toggleMember(userId, checked) {
        const was = this._selected.size;
        if (checked) this._selected.add(String(userId));
        else this._selected.delete(String(userId));
        // Row dropdowns stand down while the bulk bar owns the selection.
        if ((was === 0) !== (this._selected.size === 0)) this._renderMembers();
        else this._syncMemberSelection();
    }

    toggleAllMembers(checked) {
        // Only the rows shown, so a filtered select-all never sweeps in hidden users.
        this._shownMembers().forEach(u => {
            if (checked) this._selected.add(String(u.id));
            else this._selected.delete(String(u.id));
        });
        this._renderMembers();
    }

    /** Row dropdown: move one user. Reverts on failure. */
    async moveOne(userId, select) {
        const user = this._users.find(u => u.id === userId);
        const previous = String(this._profileOf(user));
        const profileId = parseInt(select.value, 10);
        select.disabled = true;
        try {
            const resp = await fetch(`/admin/users/profile?user_id=${userId}`, {
                method: 'PUT', credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile_id: profileId }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(this._detail(data, `Request failed (${resp.status})`));
            const name = window.ProfileStore.byId(profileId)?.name || 'profile';
            window.UIUtils?.showToast(`${user.username} moved to ${name}.`, 'success');
            await window.ProfileStore.notifyChanged();
        } catch (e) {
            select.value = previous;
            select.disabled = false;
            window.UIUtils?.showToast('Could not change profile: ' + e.message, 'error');
        }
    }

    async moveSelected() {
        const profileId = parseInt(document.getElementById('profile-members-target').value, 10);
        const userIds = [...this._selected].map(id => parseInt(id, 10));
        if (!userIds.length || !profileId) return;
        try {
            const resp = await fetch('/admin/users/bulk-action', {
                method: 'POST', credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_ids: userIds, action: 'assign_profile', profile_id: profileId }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(this._detail(data, `Request failed (${resp.status})`));
            const failed = data.failed || [];
            if (failed.length) {
                const detail = failed.map(f => `${f.username || 'user ' + f.id}: ${f.error}`).join('; ');
                window.UIUtils?.showToast(`${data.message}. ${detail}`, 'error');
            } else {
                window.UIUtils?.showToast(`${data.message}.`, 'success');
            }
            this._selected.clear();
            await window.ProfileStore.notifyChanged();
        } catch (e) {
            window.UIUtils?.showToast('Could not move users: ' + e.message, 'error');
        }
    }

    // ---- create / edit ----

    _getModal() {
        if (!this._modal) {
            const el = document.getElementById('profileModal');
            if (el && window.bootstrap) this._modal = new window.bootstrap.Modal(el);
        }
        return this._modal;
    }

    async openCreate(cloneFromId = window.ProfileStore.DEFAULT_ID) {
        document.getElementById('profile-id').value = '';
        document.getElementById('profile-name').value = '';
        document.getElementById('profile-description').value = '';
        document.getElementById('profileModalTitle').textContent = 'New Profile';
        document.getElementById('profile-clone-row').style.display = '';
        const sel = document.getElementById('profile-clone-from');
        await window.ProfileStore.fillSelect(sel, {
            emptyLabel: '— Start empty (unlimited) —', selectedId: cloneFromId, showCounts: false,
        });
        this._syncCloneHint();
        this._getModal()?.show();
    }

    openEdit(id) {
        const p = window.ProfileStore.byId(id);
        if (!p) return;
        document.getElementById('profile-id').value = p.id;
        document.getElementById('profile-name').value = p.name;
        document.getElementById('profile-description').value = p.description || '';
        document.getElementById('profileModalTitle').textContent = `Edit: ${p.name}`;
        document.getElementById('profile-clone-row').style.display = 'none';
        this._getModal()?.show();
    }

    _syncCloneHint() {
        const sel = document.getElementById('profile-clone-from');
        const hint = document.getElementById('profile-clone-hint');
        if (!sel || !hint) return;
        hint.innerHTML = sel.value
            ? 'Copies rate limits, model access and model mappings. Members are not copied.'
            : '<span class="text-warning"><i class="fas fa-triangle-exclamation me-1"></i>A new empty profile has no rate limits or model mappings, and follows the global model switches.</span>';
    }

    onCloneChange() {
        this._syncCloneHint();
    }

    async save() {
        const id = document.getElementById('profile-id').value;
        const name = document.getElementById('profile-name').value.trim();
        const description = document.getElementById('profile-description').value.trim() || null;
        if (!name) { window.UIUtils?.showToast('Name is required.', 'error'); return; }
        const body = { name, description };
        let url = '/admin/profiles';
        let method = 'POST';
        if (id) {
            url = `/admin/profiles?profile_id=${id}`;
            method = 'PUT';
        } else {
            const clone = document.getElementById('profile-clone-from').value;
            if (clone) body.clone_from_id = parseInt(clone, 10);
        }
        try {
            const resp = await fetch(url, {
                method, credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                window.UIUtils?.showToast(this._detail(data, 'Failed to save profile.'), 'error');
                return;
            }
            window.UIUtils?.showToast(id ? 'Profile updated.' : `Profile '${name}' created.`, 'success');
            this._getModal()?.hide();
            await window.ProfileStore.notifyChanged();
        } catch (e) {
            window.UIUtils?.showToast('Network error saving profile.', 'error');
        }
    }

    // ---- delete ----

    async openDelete(id) {
        const p = window.ProfileStore.byId(id);
        if (!p) return;
        this._deletingId = id;
        const esc = window.UIUtils?.escapeHtml || (s => s);
        document.getElementById('profile-delete-message').innerHTML = p.member_count
            ? `Delete <strong>${esc(p.name)}</strong>? Its ${p.member_count} member${p.member_count === 1 ? '' : 's'} move to the profile below, and its model mappings are deleted.`
            : `Delete <strong>${esc(p.name)}</strong>? Its model mappings are deleted too.`;
        const sel = document.getElementById('profile-delete-reassign');
        await window.ProfileStore.fillSelect(sel, { selectedId: window.ProfileStore.DEFAULT_ID });
        [...sel.options].forEach(o => { o.disabled = o.value === String(id); });
        if (!this._deleteModal) {
            this._deleteModal = new window.bootstrap.Modal(document.getElementById('deleteProfileModal'));
        }
        this._deleteModal.show();
    }

    async confirmDelete() {
        const id = this._deletingId;
        const to = document.getElementById('profile-delete-reassign').value;
        try {
            const resp = await fetch(`/admin/profiles?profile_id=${id}&reassign_to=${to}`, {
                method: 'DELETE', credentials: 'include',
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                window.UIUtils?.showToast(this._detail(data, 'Failed to delete profile.'), 'error');
                return;
            }
            window.UIUtils?.showToast(`${data.message}; ${data.moved_count} user(s) reassigned.`, 'success');
            this._deleteModal?.hide();
            await window.ProfileStore.notifyChanged();
        } catch (e) {
            window.UIUtils?.showToast('Network error deleting profile.', 'error');
        }
    }

    _detail(data, fallback) {
        const d = data && data.detail;
        if (typeof d === 'string') return d;
        if (Array.isArray(d) && d.length) return d.map(x => x.msg || '').join('; ');
        return fallback;
    }
}

window.ProfileStore = new ProfileStore();
window.ProfileManager = new ProfileManager();

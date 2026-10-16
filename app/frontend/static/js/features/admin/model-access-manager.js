/**
 * Model Access tab. Every rule belongs to a user profile: the profile switcher at the
 * top (the same one Rate Limits and Model Mapping use) picks whose access the cards
 * edit -- the access mode (Default / Allow all / Deny all / Custom) plus per-model
 * allow/deny exceptions. Every member of the profile gets exactly this access.
 *
 * The models list switches by model group (one switch per group, writing an exception
 * for each model it governs) or by individual model; a group row expands to its models.
 *
 * Global availability (provider and model switches) lives in Providers & Models, and
 * model groups in their own tab.
 */
class ModelsManager {
    constructor() {
        this._selectedProfileId = null;
        this._detail = null;           // ProfileModelAccessResponse for the selected profile
        this._view = 'groups';         // models list: 'groups' or 'models' (individual)
        try { if (sessionStorage.getItem('model-access:view') === 'models') this._view = 'models'; } catch (e) { /* storage blocked */ }
        this._expanded = new Set();    // group ids opened in the groups view
        this._switcher = new window.ProfileSwitcher(document.getElementById('ma-profile-switcher'), {
            label: 'Model access for',
            hint: 'Every member of this profile gets this access.',
            stats: p => [
                [p.member_count, 'member'],
                ...(p.model_access_mode === 'custom' ? [[p.exception_count || 0, 'exception']] : []),
            ],
            onSelect: id => this.selectProfile(id),
        });
        document.addEventListener('profiles:changed', () => this._onProfilesChanged());
    }

    async _onProfilesChanged() {
        const before = this._selectedProfileId;
        await this._fillProfileSwitcher();
        // The shown profile was deleted and the switcher fell back to another: reload the
        // cards, or the next toggle would write into the profile the switcher now shows.
        if (before !== null && this._selectedProfileId !== before) await this._loadDetail(this._selectedProfileId);
    }

    async load() {
        await window.ProfileStore.list({ force: true });
        await this._fillProfileSwitcher();
        await this._loadDetail(this._selectedProfileId);
    }

    async _fillProfileSwitcher() {
        // A stored '' is the old "Global" scope, now in Providers & Models: show Default.
        const want = this._selectedProfileId
            ?? window.ProfileStore.selected('model-access', null)
            ?? window.ProfileStore.DEFAULT_ID;
        this._selectedProfileId = await this._switcher.render(want);
    }

    async selectProfile(profileId) {
        this._selectedProfileId = profileId;
        window.ProfileStore.select('model-access', profileId);
        await this._loadDetail(profileId);
    }

    _showPanel(which) {
        const map = { error: 'ma-error', detail: 'ma-detail' };
        Object.entries(map).forEach(([key, id]) => {
            const el = document.getElementById(id);
            if (el) el.style.display = key === which ? '' : 'none';
        });
    }

    _url(path, modelId) {
        let url = `/admin/profiles/model-access${path}?profile_id=${this._selectedProfileId}`;
        if (modelId) url += `&model_id=${encodeURIComponent(modelId)}`;
        return url;
    }

    async _loadDetail(profileId) {
        if (profileId == null) return;
        try {
            const resp = await fetch(`/admin/profiles/model-access?profile_id=${profileId}`, { credentials: 'include' });
            // A slower response for a profile the switcher has since moved off: drop it.
            if (profileId !== this._selectedProfileId) return;
            if (!resp.ok) {
                this._showPanel('error');
                return;
            }
            this._detail = await resp.json();
            this._showPanel('detail');
            this._renderDetail();
        } catch (e) {
            console.error('ModelsManager: failed to load detail', e);
            this._showPanel('error');
        }
    }

    /** The server's view changed (mode, exceptions): refresh the switcher's counts too. */
    async _afterChange() {
        this._renderDetail();
        await window.ProfileStore.notifyChanged();
    }

    _renderDetail() {
        if (!this._detail) return;

        // Reflect the mode in the 4-option segmented control.
        const mode = this._detail.mode || 'default';
        const map = {
            default: 'ma-policy-default',
            allow: 'ma-policy-allow',
            deny: 'ma-policy-deny',
            custom: 'ma-policy-custom',
        };
        Object.entries(map).forEach(([m, id]) => {
            const btn = document.getElementById(id);
            if (btn) btn.classList.toggle('active', m === mode);
        });

        const hint = document.getElementById('ma-mode-hint');
        if (hint) hint.textContent = this._modeHint(mode);

        this._renderModels();
    }

    _modeHint(mode) {
        switch (mode) {
            case 'allow':
                return 'Every model is enabled for every member of this profile, overriding the global config (including future models).';
            case 'deny':
                return 'Every model is disabled for every member of this profile, overriding the global config (including future models).';
            case 'custom':
                return 'Per-model overrides win. New models follow the global config; overrides can enable a globally-disabled model.';
            default:
                return 'Follows the global config in Providers & Models. Toggle any group or model to switch this profile to Custom.';
        }
    }

    filterModels() {
        this._renderModels();
    }

    showView(view) {
        this._view = view === 'models' ? 'models' : 'groups';
        try { sessionStorage.setItem('model-access:view', this._view); } catch (e) { /* storage blocked */ }
        this._renderModels();
    }

    _renderModels() {
        const container = document.getElementById('ma-model-list');
        const summary = document.getElementById('ma-summary');
        if (!container || !this._detail) return;
        ['groups', 'models'].forEach(v => {
            const btn = document.getElementById(`ma-view-${v}-btn`);
            if (btn) {
                btn.classList.toggle('active', v === this._view);
                btn.setAttribute('aria-selected', String(v === this._view));
            }
        });

        const models = this._detail.models;
        const enabledCount = models.filter(m => m.effective_allowed).length;
        const disabledCount = models.length - enabledCount;
        if (summary) {
            summary.textContent = `${enabledCount} enabled · ${disabledCount} disabled`;
        }

        const q = (document.getElementById('ma-model-search')?.value || '').trim().toLowerCase();
        const matches = m => !q || (m.model_id || '').toLowerCase().includes(q);

        if (this._view === 'groups') {
            this._renderGroups(container, q, matches);
            return;
        }

        const filtered = models.filter(matches);
        if (!filtered.length) {
            container.innerHTML = `<div class="text-muted small py-4 text-center">${models.length ? 'No models match your search.' : 'No models found.'}</div>`;
            return;
        }
        container.innerHTML = filtered.map(m => this._renderModelRow(m)).join('');
    }

    /** One row per model group with a switch for all its models; a row expands to them. */
    _renderGroups(container, q, matches) {
        const esc = window.UIUtils.escapeHtml;
        const rows = [];
        const indeterminate = [];
        (this._detail.groups || []).forEach(g => {
            const members = this._detail.models.filter(m => m.group_id === g.group_id);
            // A search shows groups whose name matches, or that hold a matching model.
            const nameHit = !q || g.name.toLowerCase().includes(q);
            const shown = nameHit ? members : members.filter(matches);
            if (!nameHit && !shown.length) return;

            const allowed = members.filter(m => m.effective_allowed).length;
            const all = members.length > 0 && allowed === members.length;
            if (allowed > 0 && !all) indeterminate.push(g.group_id);
            const open = this._expanded.has(g.group_id) || (!!q && !nameHit);
            const tags = [
                g.is_default ? '<span class="ma-tag ma-tag-off">models in no other group</span>' : '',
                g.enabled ? '' : '<span class="ma-tag ma-tag-off" title="Switched off in Model Groups: only profiles that allow its models get them"><i class="fas fa-ban me-1"></i>group off</span>',
            ].join('');
            const count = members.length
                ? `${allowed}/${members.length} allowed`
                : 'no models';
            rows.push(`
            <div class="ma-model-row ma-group-row${allowed ? '' : ' is-off'}">
                <button type="button" class="ma-group-toggle" aria-expanded="${open}"
                    onclick="window.ModelsManager.toggleGroupExpand(${g.group_id})">
                    <i class="fas fa-chevron-${open ? 'down' : 'right'}"></i>
                    <i class="fas fa-layer-group"></i><span class="ma-group-name">${esc(g.name)}</span>${tags}
                    <span class="ma-group-count">${count}</span>
                </button>
                <div class="ma-model-controls">
                    <label class="ma-switch" title="${all ? 'Deny' : 'Allow'} every model in this group for this profile">
                        <input type="checkbox" data-group-id="${g.group_id}"${all ? ' checked' : ''}${members.length ? '' : ' disabled'}
                            onchange="window.ModelsManager.toggleGroup(${g.group_id}, this.checked)">
                        <span class="ma-slider"></span>
                    </label>
                </div>
            </div>`);
            if (open) {
                rows.push(shown.length
                    ? `<div class="ma-group-members">${shown.map(m => this._renderModelRow(m)).join('')}</div>`
                    : '<div class="ma-group-members text-muted small py-2 px-4">No models in this group.</div>');
            }
        });
        container.innerHTML = rows.join('')
            || '<div class="text-muted small py-4 text-center">No groups or models match your search.</div>';
        // Some but not all of a group's models allowed: show the switch as mixed.
        indeterminate.forEach(gid => {
            const cb = container.querySelector(`input[data-group-id="${gid}"]`);
            if (cb) cb.indeterminate = true;
        });
    }

    toggleGroupExpand(groupId) {
        if (this._expanded.has(groupId)) this._expanded.delete(groupId); else this._expanded.add(groupId);
        this._renderModels();
    }

    async toggleGroup(groupId, isAllowed) {
        if (this._selectedProfileId == null) return;
        try {
            const resp = await fetch(`/admin/profiles/model-access/group?profile_id=${this._selectedProfileId}&group_id=${groupId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_allowed: isAllowed }),
            });
            if (resp.ok) {
                this._detail = await resp.json();
                const g = (this._detail.groups || []).find(x => x.group_id === groupId);
                window.UIUtils?.showToast(`${g ? `Group '${g.name}'` : 'Group'} ${isAllowed ? 'allowed' : 'denied'} for this profile.`, 'success');
                await this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update group access.', 'error');
                await this._loadDetail(this._selectedProfileId);
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating group access.', 'error');
            await this._loadDetail(this._selectedProfileId);
        }
    }

    _renderModelRow(m) {
        const esc = window.UIUtils.escapeHtml;
        const safeId = esc(m.model_id);
        const disabledGlobally = !m.globally_enabled;
        const rowCls = ['ma-model-row'];
        if (m.is_exception) rowCls.push('is-exception');
        if (disabledGlobally || m.blocked_by_group) rowCls.push('is-global-off');

        // Toggle reflects effective state. Toggles stay active in every mode so an
        // admin can enable a globally-disabled model for one profile (switches to custom).
        const checked = m.effective_allowed ? ' checked' : '';

        let tag = '';
        if (disabledGlobally) {
            tag = '<span class="ma-tag ma-tag-off"><i class="fas fa-ban me-1"></i>globally disabled</span>';
        } else if (m.blocked_by_group) {
            tag = `<span class="ma-tag ma-tag-off" title="Its model group is switched off in Model Groups"><i class="fas fa-ban me-1"></i>group ${esc(m.blocked_by_group)} disabled</span>`;
        }

        return `
        <div class="${rowCls.join(' ')}" data-model-id="${safeId}">
            <div class="ma-model-id">
                ${safeId} ${tag}
            </div>
            <div class="ma-model-controls">
                <label class="ma-switch">
                    <input type="checkbox"${checked}
                        onchange="window.ModelsManager.toggleModel('${encodeURIComponent(m.model_id)}', this.checked)">
                    <span class="ma-slider"></span>
                </label>
            </div>
        </div>`;
    }

    async setMode(mode) {
        if (this._selectedProfileId == null) return;
        const labels = { default: 'Default', allow: 'Allow all', deny: 'Deny all', custom: 'Custom' };
        try {
            const resp = await fetch(this._url('/policy'), {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            });
            if (resp.ok) {
                this._detail = await resp.json();
                window.UIUtils?.showToast(`Mode set to ${labels[mode] || mode}.`, 'success');
                await this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update mode.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating mode.', 'error');
        }
    }

    async toggleModel(encodedModelId, isAllowed) {
        if (this._selectedProfileId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(this._url('', modelId), {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_allowed: isAllowed }),
            });
            if (resp.ok) {
                // Response is the full view; toggling may have flipped the profile to
                // custom mode, so re-render the mode control and list from server truth.
                this._detail = await resp.json();
                await this._afterChange();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to update access.', 'error');
                // Reconcile with server truth.
                await this._loadDetail(this._selectedProfileId);
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error updating access.', 'error');
            await this._loadDetail(this._selectedProfileId);
        }
    }

    async resetModel(encodedModelId) {
        if (this._selectedProfileId == null) return;
        const modelId = decodeURIComponent(encodedModelId);
        try {
            const resp = await fetch(this._url('', modelId), {
                method: 'DELETE',
                credentials: 'include',
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Reset to global default.', 'success');
                await this._loadDetail(this._selectedProfileId);
                await window.ProfileStore.notifyChanged();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to reset.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error resetting access.', 'error');
        }
    }
}

window.ModelsManager = new ModelsManager();

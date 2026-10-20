/**
 * Model groups, managed in the Model Groups tab. A group holds whole provider instances
 * (live: models synced from them later join by themselves) and/or individual models;
 * an individual model wins over an instance in another group. Every model no group
 * claims is in the built-in Default group, which is listed but cannot be edited.
 *
 * A group can be switched off: its models are then not served, like a disabled model,
 * except to profiles whose access overrides the global config. Members and limits stay.
 *
 * The per-profile limits live in the Rate Limits tab (ProfileGroupLimits), which this
 * manager also hosts so both tabs refresh together after an edit.
 */
class ModelGroupManager {
    constructor() {
        this._groups = [];
        this._models = [];      // [{model_id, label, provider, enabled}]
        this._instances = [];   // [{provider_key, enabled, model_count}]
        this._modal = null;
        this._editingId = null;
        this._selectedModels = new Set();
        this._selectedInstances = new Set();
        this._staleMembers = [];   // members of the edited group that no longer exist
        this._filter = '';           // the picker's search, lower-cased
        this._selectedOnly = false;  // the picker shows only what the group holds
        this._expanded = new Set();  // instances opened in the picker to show their models
        this._openGroupId = null;    // the group drilled into, or null for the list
        this._openInstance = null;   // an instance member of that group drilled into, or null
        this._profileLimits = new window.ProfileGroupLimits({ manager: 'ModelGroupManager' });
    }

    async load() {
        await Promise.all([this._loadGroups(), this._loadCatalogue()]);
        // A drilled-in model list reads each model's status from the catalogue.
        if (this._openGroupId !== null) this._renderGroups();
    }

    // ---- Catalogue: every instance and model the picker can offer ----

    async _loadCatalogue() {
        try {
            const [mResp, pResp] = await Promise.all([
                fetch('/admin/models/all', { credentials: 'include' }),
                fetch('/admin/providers', { credentials: 'include' }),
            ]);
            if (mResp.ok) {
                this._models = (await mResp.json()).map(m => ({
                    model_id: m.model_id, label: m.model_name, provider: m.provider_key,
                    enabled: m.is_enabled !== false,
                }));
            }
            if (pResp.ok) {
                this._instances = (await pResp.json()).map(p => ({
                    provider_key: p.provider_key,
                    enabled: p.enabled !== false,
                    model_count: p.model_count || 0,
                })).sort((a, b) => a.provider_key.localeCompare(b.provider_key));
            }
        } catch (e) {
            console.error('ModelGroupManager: failed to load models/providers', e);
        }
    }

    // ---- Groups list ----

    async _loadGroups() {
        const container = document.getElementById('model-groups-container');
        try {
            const resp = await fetch('/admin/model-groups', { credentials: 'include' });
            if (!resp.ok) {
                if (container) container.innerHTML = '<p class="text-center text-danger py-4">Failed to load model groups.</p>';
                return;
            }
            this._groups = await resp.json();
            this._renderGroups();
        } catch (e) {
            console.error('ModelGroupManager: failed to load groups', e);
            if (container) container.innerHTML = '<p class="text-center text-danger py-4">Error loading data.</p>';
        }
    }

    /** A badge for a member that isn't serving; *count* prefixes it in the list's summary. */
    _statusBadge(status, count = null) {
        const map = {
            missing: ['bg-danger', 'missing'],
            provider_disabled: ['bg-warning text-dark', 'provider off'],
            model_disabled: ['bg-secondary', 'model off'],
        };
        const hit = map[status];
        return hit ? ` <span class="badge ${hit[0]}">${count === null ? '' : `${count} `}${hit[1]}</span>` : '';
    }

    _plural(n, word) {
        return `${n} ${word}${n === 1 ? '' : 's'}`;
    }

    /** The Members cell of the list: counts by kind, and how many aren't serving. */
    _memberSummary(g) {
        if (g.is_default) {
            return '<span class="text-muted small">Every model not in another group</span>';
        }
        if (!g.members.length) return '<span class="text-muted small">No members</span>';
        const instances = g.members.filter(m => m.kind === 'instance').length;
        const models = g.members.length - instances;
        const parts = [];
        if (instances) parts.push(this._plural(instances, 'instance'));
        if (models) parts.push(this._plural(models, 'model'));
        const issues = {};
        g.members.forEach(m => {
            if (m.status && m.status !== 'ok') issues[m.status] = (issues[m.status] || 0) + 1;
        });
        const badges = Object.entries(issues).map(([s, n]) => this._statusBadge(s, n)).join('');
        return `<span class="small text-nowrap">${parts.join(' · ')}</span>${badges}`;
    }

    _renderRow(g) {
        const esc = window.UIUtils.escapeHtml;
        const models = (g.effective_models || []).length;
        const on = g.is_default || g.enabled !== false;
        const status = g.is_default
            ? '<span class="text-muted small">Always on</span>'
            : `<span class="status-badge ${on ? 'status-active' : 'status-inactive'}">${on ? 'Enabled' : 'Disabled'}</span>`;
        const actions = g.is_default
            ? '<span class="text-muted small"><i class="fas fa-lock me-1"></i>Built in</span>'
            : `<div class="d-flex align-items-center gap-2">
                <button class="btn btn-outline-primary btn-sm" onclick="window.ModelGroupManager?.openEditModal(${g.id})" title="Edit">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelGroupManager?.deleteGroup(${g.id})" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
               </div>`;
        return `
        <tr data-group-id="${g.id}"${on ? '' : ' class="mg-row-off"'} role="button" tabindex="0" style="cursor:pointer;"
            title="Click to see ${g.is_default ? 'models' : 'members'}"
            onclick="window.ModelGroupManager?.open(${g.id})"
            onkeydown="if (event.target === this && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); window.ModelGroupManager?.open(${g.id}); }">
            <td>
                <div class="fw-semibold">${g.is_default ? '<i class="fas fa-lock me-1 text-muted"></i>' : ''}${esc(g.name)}</div>
                ${g.description ? `<div class="text-muted small">${esc(g.description)}</div>` : ''}
            </td>
            <td>${status}</td>
            <td>${this._memberSummary(g)}</td>
            <td class="text-nowrap">${this._plural(models, 'model')}</td>
            <td class="text-nowrap" onclick="event.stopPropagation()">${actions}</td>
        </tr>`;
    }

    // ---- Drill-down ----
    //
    // Up to three levels, like Usage › By Pool: the table of groups, one group's members,
    // and one instance member's models, under an `All groups › opus › bedrock:primary`
    // trail with Back to climb out. The Default group has no members, so it opens
    // straight onto its models.

    _renderGroups() {
        const container = document.getElementById('model-groups-container');
        if (!container) return;
        // A group deleted elsewhere, or an instance no longer in it, drops back a level.
        const open = this._groups.find(g => g.id === this._openGroupId);
        if (!open) this._openGroupId = null;
        if (!open || !open.members.some(m => m.kind === 'instance' && m.key === this._openInstance)) {
            this._openInstance = null;
        }
        const back = document.getElementById('model-groups-back');
        if (back) back.style.display = open ? '' : 'none';
        container.innerHTML = open ? this._detailHtml(open) : this._groupTableHtml();
    }

    _groupTableHtml() {
        if (!this._groups.length) return '<p class="text-center text-muted py-4">No model groups.</p>';
        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Name</th><th>Status</th><th>Members</th><th>Models</th><th>Actions</th></tr></thead>
                    <tbody>${this._groups.map(g => this._renderRow(g)).join('')}</tbody>
                </table>
            </div>`;
    }

    /** The trail; every segment but the last is a link back to its own depth. */
    _crumbHtml(g) {
        const esc = window.UIUtils.escapeHtml;
        const link = (text, depth) => `
            <span role="button" tabindex="0" onclick="window.ModelGroupManager?._goToDepth(${depth})"
                onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.ModelGroupManager?._goToDepth(${depth}); }"
                style="cursor:pointer;text-decoration:underline;text-underline-offset:3px;color:var(--mono-text-muted);">${esc(text)}</span>`;
        const parts = [link('All groups', 0)];
        if (this._openInstance) {
            parts.push(link(g.name, 1), `<span class="fw-semibold">${esc(this._openInstance)}</span>`);
        } else {
            parts.push(`<span class="fw-semibold">${esc(g.name)}</span>`);
        }
        const off = !g.is_default && g.enabled === false
            ? '<span class="status-badge status-inactive ms-2">Disabled</span>' : '';
        return `
            <div class="px-3 pt-3 mb-3 d-flex align-items-center flex-wrap" style="gap:.5rem;">
                ${parts.join('<span class="text-muted">&rsaquo;</span>')}${off}
            </div>`;
    }

    _detailHtml(g) {
        let body;
        if (this._openInstance) {
            const inst = g.members.find(m => m.kind === 'instance' && m.key === this._openInstance);
            body = this._modelsTableHtml(inst.covers, false);
        } else if (g.is_default) {
            body = this._modelsTableHtml(g.effective_models || [], true);
        } else {
            body = this._membersTableHtml(g);
        }
        return this._crumbHtml(g) + body;
    }

    _membersTableHtml(g) {
        const esc = window.UIUtils.escapeHtml;
        if (!g.members.length) return '<p class="text-center text-muted py-4">No members.</p>';
        const rows = g.members.map(m => {
            const isInstance = m.kind === 'instance';
            const drill = isInstance && m.covers.length;
            const ok = !m.status || m.status === 'ok';
            const note = m.overrides_group
                ? ` <span class="text-muted small" title="Carved out of the instance in '${esc(m.overrides_group)}'">overrides ${esc(m.overrides_group)}</span>`
                : '';
            const attrs = drill
                ? `role="button" tabindex="0" style="cursor:pointer;" title="Click to see its models"
                    data-key="${esc(m.key)}" onclick="window.ModelGroupManager?.openInstance(this.dataset.key)"
                    onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.ModelGroupManager?.openInstance(this.dataset.key); }"`
                : '';
            return `
            <tr ${attrs}>
                <td class="text-break"><i class="fas ${isInstance ? 'fa-server' : 'fa-cube'} me-2 text-muted"></i>${esc(m.key)}</td>
                <td>${isInstance ? 'Instance' : 'Model'}</td>
                <td class="text-nowrap">${isInstance ? this._plural(m.covers.length, 'model') : '<span class="text-muted">—</span>'}</td>
                <td class="text-nowrap">${ok ? '<span class="status-badge status-active">Active</span>' : this._statusBadge(m.status)}${note}</td>
            </tr>`;
        }).join('');
        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Member</th><th>Type</th><th>Models</th><th>Status</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }

    /** Models as rows; *withInstance* adds the instance column when they span several. */
    _modelsTableHtml(modelIds, withInstance) {
        const esc = window.UIUtils.escapeHtml;
        if (!modelIds.length) return '<p class="text-center text-muted py-4">No models.</p>';
        const catalogue = new Map(this._models.map(m => [m.model_id, m]));
        const rows = modelIds.map(mid => {
            const cut = mid.indexOf('/');
            const instance = cut < 0 ? '' : mid.slice(0, cut);
            const name = cut < 0 ? mid : mid.slice(cut + 1);
            const known = catalogue.get(mid);
            const status = known && !known.enabled
                ? this._statusBadge('model_disabled')
                : '<span class="status-badge status-active">Active</span>';
            return `
            <tr>
                <td class="text-break"><i class="fas fa-cube me-2 text-muted"></i>${esc(name)}</td>
                ${withInstance ? `<td class="text-nowrap">${esc(instance)}</td>` : ''}
                <td class="text-nowrap">${status}</td>
            </tr>`;
        }).join('');
        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Model</th>${withInstance ? '<th>Instance</th>' : ''}<th>Status</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }

    open(groupId) {
        this._openGroupId = groupId;
        this._openInstance = null;
        this._renderGroups();
    }

    openInstance(providerKey) {
        this._openInstance = providerKey;
        this._renderGroups();
    }

    back() {
        this._goToDepth(this._openInstance ? 1 : 0);
    }

    /** 0 is the list of groups, 1 the open group's members. */
    _goToDepth(depth) {
        if (depth < 1) this._openGroupId = null;
        this._openInstance = null;
        this._renderGroups();
    }

    // ---- Create/Edit modal ----

    async openCreateModal() {
        await this._openModal(null);
    }

    async openEditModal(groupId) {
        await this._openModal(this._groups.find(g => g.id === groupId) || null);
    }

    async _openModal(group) {
        if (!this._models.length || !this._instances.length) await this._loadCatalogue();
        this._editingId = group ? group.id : null;
        this._selectedModels = new Set(group ? group.model_ids : []);
        this._selectedInstances = new Set(group ? group.provider_keys : []);
        this._staleMembers = group ? group.members.filter(m => m.status === 'missing') : [];
        document.getElementById('mg-group-id').value = group ? group.id : '';
        document.getElementById('mg-name').value = group ? group.name : '';
        document.getElementById('mg-description').value = group ? (group.description || '') : '';
        document.getElementById('modelGroupModalTitle').textContent = group ? 'Edit group' : 'New group';
        document.getElementById('mg-save').textContent = group ? 'Save changes' : 'Create group';
        // The limits note is about where a new group's limits come from; an edit keeps its own.
        document.getElementById('mg-limits-note').hidden = !!group;
        const search = document.getElementById('mg-member-search');
        if (search) search.value = '';
        this._filter = '';
        this._setSelectedOnly(false);
        this._autoExpand();
        this._renderTree();
        this._getModal()?.show();
    }

    /** Which other group already owns each instance and each model (explicitly). */
    _owners() {
        const instances = {}, models = {};
        this._groups.forEach(g => {
            if (g.is_default || g.id === this._editingId) return;
            (g.provider_keys || []).forEach(k => { instances[k] = g.name; });
            (g.model_ids || []).forEach(k => { models[k] = g.name; });
        });
        return { instances, models };
    }

    // ---- Member picker ----
    //
    // One tree: each instance is a row whose checkbox takes the whole instance, and opens
    // onto its models, each with its own checkbox. A checked instance shows its models as
    // included; a model another group names stays there and can't be checked here.

    /** Instances in key order, each with its models; providers known only from models come last. */
    _tree() {
        const byProvider = new Map();
        this._models.forEach(m => {
            if (!byProvider.has(m.provider)) byProvider.set(m.provider, []);
            byProvider.get(m.provider).push(m);
        });
        byProvider.forEach(list => list.sort((a, b) => a.label.localeCompare(b.label)));
        const nodes = this._instances.map(i => ({ ...i, listed: true, models: byProvider.get(i.provider_key) || [] }));
        byProvider.forEach((models, key) => {
            if (!this._instances.some(i => i.provider_key === key)) {
                nodes.push({ provider_key: key, enabled: true, model_count: models.length, listed: false, models });
            }
        });
        return nodes;
    }

    _matches(...xs) {
        return !this._filter || xs.some(x => (x || '').toLowerCase().includes(this._filter));
    }

    /** Open the instances whose models the filter picked out, or that hold single picks. */
    _autoExpand() {
        this._expanded = new Set();
        this._tree().forEach(node => {
            const key = node.provider_key;
            const byModel = this._filter && !this._matches(key) && node.models.some(m => this._matches(m.model_id, m.label));
            const partial = !this._selectedInstances.has(key) && node.models.some(m => this._selectedModels.has(m.model_id));
            if (byModel || partial) this._expanded.add(key);
        });
    }

    _renderTree() {
        const el = document.getElementById('mg-member-tree');
        if (!el) return;
        const owners = this._owners();
        const nodes = this._tree();
        const html = [this._staleHtml(), ...nodes.map((node, i) => this._instanceHtml(node, i, owners))].join('');
        let empty = 'No instances or models yet.';
        if (this._filter) empty = `Nothing matches “${window.UIUtils.escapeHtml(this._filter)}”.`;
        else if (this._selectedOnly) empty = 'Nothing selected yet. Check an instance, or open one to pick models.';
        el.innerHTML = html || `<div class="mg-empty">${empty}</div>`;
        el.querySelectorAll('input[data-partial]').forEach(box => { box.indeterminate = true; });
        this._renderReadout(owners);
        this._renderWarnings(owners);
    }

    /** Re-render after a click, keeping keyboard focus on the control that had it. */
    _refreshTree() {
        const focus = document.activeElement?.dataset?.focus;
        this._renderTree();
        if (focus) document.querySelector(`#mg-member-tree [data-focus="${CSS.escape(focus)}"]`)?.focus();
    }

    _instanceHtml(node, index, owners) {
        const esc = window.UIUtils.escapeHtml;
        const key = node.provider_key;
        const on = this._selectedInstances.has(key);
        const owner = on ? null : owners.instances[key];
        const total = node.models.length || node.model_count;
        const picked = node.models.filter(m => this._selectedModels.has(m.model_id)).length;
        const keptElsewhere = node.models.filter(m => owners.models[m.model_id]).length;

        // A match on the instance shows all of it; otherwise only the models that match.
        let models = this._matches(key) ? node.models : node.models.filter(m => this._matches(m.model_id, m.label));
        if (this._selectedOnly) models = models.filter(m => on || this._selectedModels.has(m.model_id));
        if (!models.length && !(this._matches(key) && (!this._selectedOnly || on))) return '';

        const open = this._expanded.has(key) && models.length > 0;
        const listId = `mg-inst-${index}-models`;
        const state = on ? 'is-on' : picked ? 'is-partial' : owner ? 'is-taken' : '';
        const count = on
            ? (keptElsewhere ? `${total - keptElsewhere} of ${total}` : this._plural(total, 'model'))
            : picked ? `${picked} of ${total}` : this._plural(total, 'model');

        const box = node.listed
            ? `<input type="checkbox" class="mg-box" id="mg-inst-${index}" value="${esc(key)}" data-kind="instance"
                    data-focus="i:${esc(key)}" ${on ? 'checked' : ''} ${owner ? 'disabled' : ''}
                    ${!on && picked ? 'data-partial' : ''} onchange="window.ModelGroupManager?._onToggle(this)">`
            : '<span class="mg-box-spacer"></span>';
        const tags = [
            node.enabled ? '' : '<span class="mg-tag">provider off</span>',
            owner ? `<span class="mg-where">in <span class="mg-owner">${esc(owner)}</span></span>` : '',
        ].join('');

        return `
        <div class="mg-inst ${state}">
            <div class="mg-inst__row">
                <button type="button" class="mg-twisty" aria-expanded="${open}" aria-controls="${listId}"
                    aria-label="${open ? 'Hide' : 'Show'} models of ${esc(key)}" data-key="${esc(key)}" data-focus="t:${esc(key)}"
                    ${models.length ? '' : 'disabled'} onclick="window.ModelGroupManager?._toggleExpand(this.dataset.key)">
                    <i class="fas fa-chevron-right"></i>
                </button>
                <label class="mg-check" ${node.listed ? `for="mg-inst-${index}"` : ''}>
                    ${box}<span class="mg-inst__key">${esc(key)}</span>${tags}
                </label>
                <span class="mg-count">${count}</span>
            </div>
            ${open ? `<ul class="mg-models" id="${listId}">${models.map((m, j) => this._modelHtml(m, `${index}-${j}`, on, owners)).join('')}</ul>` : ''}
        </div>`;
    }

    _modelHtml(m, id, instanceOn, owners) {
        const esc = window.UIUtils.escapeHtml;
        const named = this._selectedModels.has(m.model_id);
        const owner = named ? null : owners.models[m.model_id];
        // Included with its instance: checked, and only changed by unchecking the instance.
        const covered = instanceOn && !owner;
        const state = owner ? 'is-taken' : covered ? 'is-covered' : named ? 'is-on' : '';
        const title = owner
            ? `${owner} picks this model by name; remove it there first`
            : covered ? `Included with ${m.provider}` : '';
        return `
            <li class="mg-model ${state}"${title ? ` title="${esc(title)}"` : ''}>
                <label class="mg-check" for="mg-model-${id}">
                    <input type="checkbox" class="mg-box" id="mg-model-${id}" value="${esc(m.model_id)}" data-kind="model"
                        data-focus="m:${esc(m.model_id)}" ${named || covered ? 'checked' : ''} ${owner || covered ? 'disabled' : ''}
                        onchange="window.ModelGroupManager?._onToggle(this)">
                    <span class="mg-model__name">${esc(m.label)}</span>
                    ${m.enabled ? '' : '<span class="mg-tag">model off</span>'}
                    ${owner ? `<span class="mg-where">in <span class="mg-owner">${esc(owner)}</span></span>` : ''}
                    ${covered ? `<span class="visually-hidden">, included with ${esc(m.provider)}</span>` : ''}
                </label>
            </li>`;
    }

    /** Members of the edited group whose instance or model is gone; unchecking drops them. */
    _staleHtml() {
        const esc = window.UIUtils.escapeHtml;
        const stale = this._staleMembers.filter(m => this._matches(m.key));
        if (!stale.length) return '';
        const rows = stale.map((m, i) => {
            const set = m.kind === 'instance' ? this._selectedInstances : this._selectedModels;
            return `
            <li class="mg-model${set.has(m.key) ? ' is-on' : ''}">
                <label class="mg-check" for="mg-stale-${i}">
                    <input type="checkbox" class="mg-box" id="mg-stale-${i}" value="${esc(m.key)}" data-kind="${m.kind}"
                        data-focus="s:${esc(m.key)}" ${set.has(m.key) ? 'checked' : ''}
                        onchange="window.ModelGroupManager?._onToggle(this)">
                    <span class="mg-model__name">${esc(m.key)}</span><span class="mg-tag">${m.kind === 'instance' ? 'instance ' : ''}missing</span>
                </label>
            </li>`;
        }).join('');
        return `
        <div class="mg-inst is-stale">
            <div class="mg-inst__row">
                <span class="mg-twisty"></span>
                <span class="mg-check"><span class="mg-inst__key">No longer available</span></span>
                <span class="mg-count">${stale.length}</span>
            </div>
            <ul class="mg-models">${rows}</ul>
        </div>`;
    }

    /** What the group adds up to: the models it would serve, and how they got in. */
    _renderReadout(owners) {
        const el = document.getElementById('mg-readout');
        if (!el) return;
        // Counted from the catalogue, so members that no longer exist add nothing.
        const instanceKeys = new Set(this._instances.map(i => i.provider_key));
        const serves = this._models.filter(m => this._selectedModels.has(m.model_id)
            || (this._selectedInstances.has(m.provider) && !owners.models[m.model_id])).length;
        const instances = [...this._selectedInstances].filter(k => instanceKeys.has(k)).length;
        const singles = this._models.filter(m => this._selectedModels.has(m.model_id)
            && !this._selectedInstances.has(m.provider)).length;
        if (!instances && !singles) { el.innerHTML = '<span class="mg-readout__how">Nothing selected</span>'; return; }
        const how = [];
        if (instances) how.push(this._plural(instances, 'instance'));
        if (singles) how.push(`${singles} single pick${singles === 1 ? '' : 's'}`);
        el.innerHTML = `<span class="mg-readout__n">${this._plural(serves, 'model')}</span>
            <span class="mg-readout__how">from ${how.join(' and ')}</span>`;
    }

    /** Explain the precedence the admin is setting up, before they save it. */
    _renderWarnings(owners) {
        const esc = window.UIUtils.escapeHtml;
        const el = document.getElementById('mg-member-warnings');
        if (!el) return;
        const tally = (counts, k) => { counts[k] = (counts[k] || 0) + 1; return counts; };
        const notes = [];

        // Models picked here whose instance another group holds.
        const takenFrom = [...this._selectedModels]
            .map(mid => owners.instances[mid.split('/')[0]]).filter(Boolean).reduce(tally, {});
        Object.entries(takenFrom).forEach(([group, n]) => {
            notes.push(`${this._plural(n, 'model')} ${n === 1 ? 'moves' : 'move'} out of <span class="mg-owner">${esc(group)}</span>,
                which holds ${n === 1 ? 'its' : 'their'} instance. This group's limits apply to ${n === 1 ? 'it' : 'them'}.`);
        });
        // Instances checked here with models another group picks by name.
        this._selectedInstances.forEach(pk => {
            const keptBy = Object.entries(owners.models)
                .filter(([mid]) => mid.split('/')[0] === pk).map(([, group]) => group).reduce(tally, {});
            Object.entries(keptBy).forEach(([group, n]) => {
                notes.push(`${this._plural(n, 'model')} of ${esc(pk)} ${n === 1 ? 'stays' : 'stay'} in <span class="mg-owner">${esc(group)}</span>,
                    which picks ${n === 1 ? 'it' : 'them'} by name.`);
            });
        });
        el.innerHTML = notes.map(n => `<p><i class="fas fa-circle-info"></i><span>${n}</span></p>`).join('');
    }

    _onToggle(checkbox) {
        const set = checkbox.dataset.kind === 'instance' ? this._selectedInstances : this._selectedModels;
        if (checkbox.checked) set.add(checkbox.value); else set.delete(checkbox.value);
        this._refreshTree();
    }

    _toggleExpand(key) {
        if (this._expanded.has(key)) this._expanded.delete(key); else this._expanded.add(key);
        this._refreshTree();
    }

    filterMembers(query) {
        this._filter = query.trim().toLowerCase();
        this._autoExpand();
        this._renderTree();
    }

    toggleSelectedOnly() {
        this._setSelectedOnly(!this._selectedOnly);
        this._renderTree();
    }

    _setSelectedOnly(on) {
        this._selectedOnly = on;
        document.getElementById('mg-selected-only')?.setAttribute('aria-pressed', String(on));
    }

    _errorText(detail, fallback) {
        if (!detail) return fallback;
        if (typeof detail === 'string') return detail;
        if (detail.message) return detail.message;
        if (Array.isArray(detail)) return detail.map(d => d.msg || '').join('; ') || fallback;
        return fallback;
    }

    async saveGroup() {
        const groupId = document.getElementById('mg-group-id').value;
        const name = document.getElementById('mg-name').value.trim();
        const description = document.getElementById('mg-description').value.trim() || null;
        if (!name) {
            window.UIUtils.showToast('Give the group a name.', 'error');
            document.getElementById('mg-name').focus();
            return;
        }
        const members = {
            model_ids: [...this._selectedModels],
            provider_keys: [...this._selectedInstances],
        };
        const send = (url, method, body) => fetch(url, {
            method, credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        try {
            if (!groupId) {
                const resp = await send('/admin/model-groups', 'POST', { name, description, ...members });
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok) { window.UIUtils.showToast(this._errorText(data.detail, 'Failed to create group.'), 'error'); return; }
                window.UIUtils.showToast(`Group '${name}' created with the limits its models already had.`, 'success');
            } else {
                // One request: the name and members are saved together or not at all.
                const resp = await send(`/admin/model-groups?group_id=${groupId}`, 'PUT', { name, description, ...members });
                if (!resp.ok) {
                    const e = await resp.json().catch(() => ({}));
                    window.UIUtils.showToast(this._errorText(e.detail, 'Failed to update group.'), 'error');
                    return;
                }
                window.UIUtils.showToast(`Group '${name}' saved.`, 'success');
            }
            this._getModal()?.hide();
            await this._afterChange();
        } catch (e) {
            window.UIUtils.showToast('Network error saving group.', 'error');
        }
    }

    async deleteGroup(groupId) {
        const g = this._groups.find(x => x.id === groupId);
        const name = g ? g.name : `group ${groupId}`;
        const confirmed = await window.UIUtils.showConfirmModal(
            'Delete Model Group',
            `Delete model group "${name}"? Its models return to the Default group and its per-profile limits are removed.`,
            'danger',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/model-groups?group_id=${groupId}`, { method: 'DELETE', credentials: 'include' });
            if (resp.ok) {
                window.UIUtils.showToast(`Group '${name}' deleted.`, 'success');
                await this._afterChange();
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils.showToast(this._errorText(e.detail, 'Failed to delete.'), 'error');
            }
        } catch (e) {
            window.UIUtils.showToast('Network error.', 'error');
        }
    }

    async _afterChange() {
        await this._loadGroups();
        // The Rate Limits table lists every group; keep it current if it has been opened.
        if (this._profileLimits.profileId) await this._profileLimits.render();
    }

    // ---- Limits for the selected profile (Rate Limits tab) ----

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

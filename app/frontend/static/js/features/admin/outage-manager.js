/**
 * Outages, managed in the Outages tab. An outage takes a whole provider instance (live:
 * models synced from it later are covered too) or single models out of service for
 * every caller, admins included. Requests for them follow the Fallback list in Model
 * Mapping, or fail with 503 quoting the description, which users also see on hover in
 * their Models page. Resolving an outage deletes it.
 *
 * The picker is the Model Groups one (ModelGroupManager), minus group ownership: an
 * instance or model already in outage is tagged, and reporting it again only replaces
 * its description.
 */
class OutageManager {
    constructor() {
        this._outages = [];
        this._models = [];      // [{model_id, label, provider, enabled}]
        this._instances = [];   // [{provider_key, enabled, model_count}]
        this._modal = null;
        this._editingId = null;     // the outage whose description is being edited
        this._selected = new Set(); // outage ids ticked in the list
        this._search = '';          // the list's search, lower-cased
        this._selectedModels = new Set();
        this._selectedInstances = new Set();
        this._filter = '';           // the picker's search, lower-cased
        this._selectedOnly = false;  // the picker shows only what is ticked
        this._expanded = new Set();  // instances opened in the picker to show their models
    }

    async load() {
        await Promise.all([this._loadOutages(), this._loadCatalogue()]);
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
            console.error('OutageManager: failed to load models/providers', e);
        }
    }

    // ---- Outage list ----

    async _loadOutages() {
        const tbody = document.getElementById('outages-tbody');
        try {
            const resp = await fetch('/admin/outages', { credentials: 'include' });
            if (!resp.ok) {
                if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="text-center text-danger py-4">Failed to load outages.</td></tr>';
                return;
            }
            this._outages = await resp.json();
            const ids = new Set(this._outages.map(o => o.id));
            this._selected = new Set([...this._selected].filter(id => ids.has(id)));
            this._render();
        } catch (e) {
            console.error('OutageManager: failed to load outages', e);
            if (tbody) tbody.innerHTML = '<tr><td colspan="8" class="text-center text-danger py-4">Error loading data.</td></tr>';
        }
    }

    filter(value) {
        this._search = (value || '').trim().toLowerCase();
        this._render();
    }

    _shownRows() {
        if (!this._search) return this._outages;
        return this._outages.filter(o => [o.target, o.description, o.created_by, o.scope]
            .some(x => (x || '').toLowerCase().includes(this._search)));
    }

    _plural(n, word) {
        return `${n} ${word}${n === 1 ? '' : 's'}`;
    }

    /** "5 min ago" for the table; the exact local time goes in the title. */
    _since(epoch) {
        if (!epoch) return { text: '—', title: '' };
        const when = new Date(epoch * 1000);
        const secs = Math.max(0, (Date.now() - when.getTime()) / 1000);
        let text;
        if (secs < 60) text = 'just now';
        else if (secs < 3600) text = `${Math.floor(secs / 60)} min ago`;
        else if (secs < 86400) text = this._plural(Math.floor(secs / 3600), 'hour') + ' ago';
        else text = this._plural(Math.floor(secs / 86400), 'day') + ' ago';
        return { text, title: when.toLocaleString() };
    }

    _render() {
        const tbody = document.getElementById('outages-tbody');
        if (!tbody) return;
        const esc = window.UIUtils.escapeHtml;
        this._syncSelection();
        if (!this._outages.length) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4"><i class="fas fa-circle-check me-2 text-success"></i>No active outages.</td></tr>';
            return;
        }
        const rows = this._shownRows();
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No outages match your search.</td></tr>';
            return;
        }
        tbody.innerHTML = rows.map(o => {
            const isInstance = o.scope === 'provider';
            const selected = this._selected.has(o.id);
            const since = this._since(o.created_at);
            const gone = o.target_exists
                ? ''
                : ' <span class="badge bg-secondary" title="The instance or model no longer exists">no longer exists</span>';
            return `
            <tr data-id="${o.id}" class="outage-row${selected ? ' selected' : ''}">
                <td class="col-select"><input type="checkbox" class="form-check-input" aria-label="Select ${esc(o.target)}"
                    ${selected ? 'checked' : ''} onchange="window.OutageManager?.toggleRow(${o.id}, this.checked)"></td>
                <td class="text-break"><i class="fas ${isInstance ? 'fa-server' : 'fa-cube'} me-2 text-muted"></i><code>${esc(o.target)}</code>${gone}</td>
                <td><span class="outage-scope">${isInstance ? 'Instance' : 'Model'}</span></td>
                <td class="outage-description">${esc(o.description)}</td>
                <td class="text-nowrap">${this._plural(o.affected_models || 0, 'model')}</td>
                <td class="text-nowrap" title="${esc(since.title)}">${esc(since.text)}</td>
                <td class="text-nowrap">${esc(o.created_by || '—')}</td>
                <td><div class="d-flex align-items-center gap-2">
                    <button class="btn btn-outline-primary btn-sm" title="Edit description" aria-label="Edit description of ${esc(o.target)}"
                        onclick="window.OutageManager?.openEditModal(${o.id})"><i class="fas fa-pencil-alt"></i></button>
                    <button class="btn btn-outline-success btn-sm text-nowrap" title="Resolve: serve these models again"
                        onclick="window.OutageManager?.resolve(${o.id})"><i class="fas fa-check me-1"></i>Resolve</button>
                </div></td>
            </tr>`;
        }).join('');
    }

    // ---- Bulk selection ----

    toggleRow(id, checked) {
        if (checked) this._selected.add(id);
        else this._selected.delete(id);
        document.querySelector(`#outages-tbody tr[data-id="${id}"]`)?.classList.toggle('selected', checked);
        this._syncSelection();
    }

    toggleSelectAll(checked) {
        // Only the rows shown, so a filtered select-all never sweeps in hidden outages.
        this._shownRows().forEach(o => {
            if (checked) this._selected.add(o.id);
            else this._selected.delete(o.id);
        });
        this._render();
    }

    clearSelection() {
        this._selected.clear();
        this._render();
    }

    _syncSelection() {
        const n = this._selected.size;
        const bar = document.getElementById('outage-bulk');
        if (bar) bar.style.display = n ? '' : 'none';
        const count = document.getElementById('outage-bulk-count');
        if (count) count.textContent = `${n} selected`;
        const all = document.getElementById('outage-select-all');
        if (all) {
            const shown = this._shownRows();
            const picked = shown.filter(o => this._selected.has(o.id)).length;
            all.checked = shown.length > 0 && picked === shown.length;
            all.indeterminate = picked > 0 && picked < shown.length;
            all.disabled = !shown.length;
        }
    }

    // ---- Report / edit modal ----

    async openCreateModal() {
        if (!this._models.length || !this._instances.length) await this._loadCatalogue();
        this._editingId = null;
        this._selectedModels = new Set();
        this._selectedInstances = new Set();
        this._openModal('', false);
    }

    openEditModal(outageId) {
        const outage = this._outages.find(o => o.id === outageId);
        if (!outage) return;
        this._editingId = outageId;
        this._openModal(outage.description, true, outage.target);
    }

    _openModal(description, editing, target = '') {
        document.getElementById('og-description').value = description;
        document.getElementById('outageModalTitle').textContent = editing ? `Edit outage of ${target}` : 'Report outage';
        document.getElementById('og-save').textContent = editing ? 'Save description' : 'Report outage';
        // An edit changes the description only; the target is fixed.
        document.getElementById('og-members').hidden = editing;
        this.updateCounter();
        if (!editing) {
            const search = document.getElementById('og-member-search');
            if (search) search.value = '';
            this._filter = '';
            this._setSelectedOnly(false);
            this._autoExpand();
            this._renderTree();
        }
        this._getModal()?.show();
        setTimeout(() => document.getElementById('og-description')?.focus(), 300);
    }

    updateCounter() {
        const input = document.getElementById('og-description');
        const counter = document.getElementById('og-description-count');
        if (input && counter) counter.textContent = `${input.value.length}/${input.maxLength}`;
    }

    /** Which instances and models are in outage now, by key. */
    _inOutage() {
        const instances = {}, models = {};
        this._outages.forEach(o => {
            (o.scope === 'provider' ? instances : models)[o.target] = o.description;
        });
        return { instances, models };
    }

    // ---- Member picker (as in ModelGroupManager) ----

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
        const el = document.getElementById('og-member-tree');
        if (!el) return;
        const down = this._inOutage();
        const html = this._tree().map((node, i) => this._instanceHtml(node, i, down)).join('');
        let empty = 'No instances or models yet.';
        if (this._filter) empty = `Nothing matches “${window.UIUtils.escapeHtml(this._filter)}”.`;
        else if (this._selectedOnly) empty = 'Nothing selected yet. Check an instance, or open one to pick models.';
        el.innerHTML = html || `<div class="mg-empty">${empty}</div>`;
        el.querySelectorAll('input[data-partial]').forEach(box => { box.indeterminate = true; });
        this._renderReadout();
    }

    /** Re-render after a click, keeping keyboard focus on the control that had it. */
    _refreshTree() {
        const focus = document.activeElement?.dataset?.focus;
        this._renderTree();
        if (focus) document.querySelector(`#og-member-tree [data-focus="${CSS.escape(focus)}"]`)?.focus();
    }

    _outageTag(description) {
        const esc = window.UIUtils.escapeHtml;
        return `<span class="mg-tag outage-tag" title="In outage: ${esc(description)}"><i class="fas fa-triangle-exclamation me-1"></i>in outage</span>`;
    }

    _instanceHtml(node, index, down) {
        const esc = window.UIUtils.escapeHtml;
        const key = node.provider_key;
        const on = this._selectedInstances.has(key);
        const total = node.models.length || node.model_count;
        const picked = node.models.filter(m => this._selectedModels.has(m.model_id)).length;

        // A match on the instance shows all of it; otherwise only the models that match.
        let models = this._matches(key) ? node.models : node.models.filter(m => this._matches(m.model_id, m.label));
        if (this._selectedOnly) models = models.filter(m => on || this._selectedModels.has(m.model_id));
        if (!models.length && !(this._matches(key) && (!this._selectedOnly || on))) return '';

        const open = this._expanded.has(key) && models.length > 0;
        const listId = `og-inst-${index}-models`;
        const state = on ? 'is-on' : picked ? 'is-partial' : '';
        const count = on ? this._plural(total, 'model') : picked ? `${picked} of ${total}` : this._plural(total, 'model');

        const box = node.listed
            ? `<input type="checkbox" class="mg-box" id="og-inst-${index}" value="${esc(key)}" data-kind="instance"
                    data-focus="i:${esc(key)}" ${on ? 'checked' : ''}
                    ${!on && picked ? 'data-partial' : ''} onchange="window.OutageManager?._onToggle(this)">`
            : '<span class="mg-box-spacer"></span>';
        const tags = [
            node.enabled ? '' : '<span class="mg-tag">provider off</span>',
            key in down.instances ? this._outageTag(down.instances[key]) : '',
        ].join('');

        return `
        <div class="mg-inst ${state}">
            <div class="mg-inst__row">
                <button type="button" class="mg-twisty" aria-expanded="${open}" aria-controls="${listId}"
                    aria-label="${open ? 'Hide' : 'Show'} models of ${esc(key)}" data-key="${esc(key)}" data-focus="t:${esc(key)}"
                    ${models.length ? '' : 'disabled'} onclick="window.OutageManager?._toggleExpand(this.dataset.key)">
                    <i class="fas fa-chevron-right"></i>
                </button>
                <label class="mg-check" ${node.listed ? `for="og-inst-${index}"` : ''}>
                    ${box}<span class="mg-inst__key">${esc(key)}</span>${tags}
                </label>
                <span class="mg-count">${count}</span>
            </div>
            ${open ? `<ul class="mg-models" id="${listId}">${models.map((m, j) => this._modelHtml(m, `${index}-${j}`, on, down)).join('')}</ul>` : ''}
        </div>`;
    }

    _modelHtml(m, id, instanceOn, down) {
        const esc = window.UIUtils.escapeHtml;
        const named = this._selectedModels.has(m.model_id);
        // Included with its instance: checked, and only changed by unchecking the instance.
        const covered = instanceOn;
        const state = covered ? 'is-covered' : named ? 'is-on' : '';
        return `
            <li class="mg-model ${state}"${covered ? ` title="${esc(`Included with ${m.provider}`)}"` : ''}>
                <label class="mg-check" for="og-model-${id}">
                    <input type="checkbox" class="mg-box" id="og-model-${id}" value="${esc(m.model_id)}" data-kind="model"
                        data-focus="m:${esc(m.model_id)}" ${named || covered ? 'checked' : ''} ${covered ? 'disabled' : ''}
                        onchange="window.OutageManager?._onToggle(this)">
                    <span class="mg-model__name">${esc(m.label)}</span>
                    ${m.enabled ? '' : '<span class="mg-tag">model off</span>'}
                    ${m.model_id in down.models ? this._outageTag(down.models[m.model_id]) : ''}
                    ${covered ? `<span class="visually-hidden">, included with ${esc(m.provider)}</span>` : ''}
                </label>
            </li>`;
    }

    /** How many models the outage would take out of service. */
    _renderReadout() {
        const el = document.getElementById('og-readout');
        if (!el) return;
        const affected = this._models.filter(m => this._selectedModels.has(m.model_id)
            || this._selectedInstances.has(m.provider)).length;
        const instances = this._selectedInstances.size;
        const singles = this._models.filter(m => this._selectedModels.has(m.model_id)
            && !this._selectedInstances.has(m.provider)).length;
        if (!instances && !singles) { el.innerHTML = '<span class="mg-readout__how">Nothing selected</span>'; return; }
        const how = [];
        if (instances) how.push(this._plural(instances, 'instance'));
        if (singles) how.push(`${singles} single pick${singles === 1 ? '' : 's'}`);
        el.innerHTML = `<span class="mg-readout__n">${this._plural(affected, 'model')}</span>
            <span class="mg-readout__how">from ${how.join(' and ')}</span>`;
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
        document.getElementById('og-selected-only')?.setAttribute('aria-pressed', String(on));
    }

    // ---- Saving and resolving ----

    /** Server error text, escaped: showToast renders HTML. */
    _errorText(detail, fallback) {
        let text = fallback;
        if (typeof detail === 'string') text = detail;
        else if (detail?.message) text = detail.message;
        else if (Array.isArray(detail)) text = detail.map(d => (d.msg || '').replace(/^Value error, /, '')).join('; ') || fallback;
        return window.UIUtils.escapeHtml(text);
    }

    async save() {
        const input = document.getElementById('og-description');
        const description = input.value.trim();
        if (!description) {
            window.UIUtils.showToast('Describe the outage: users see this text.', 'error');
            input.focus();
            return;
        }
        const editing = this._editingId !== null;
        if (!editing && !this._selectedInstances.size && !this._selectedModels.size) {
            window.UIUtils.showToast('Select at least one instance or model.', 'error');
            return;
        }
        // Models of a ticked instance are covered by it; send them only as the instance.
        const model_ids = [...this._selectedModels].filter(mid => {
            const m = this._models.find(x => x.model_id === mid);
            return !(m && this._selectedInstances.has(m.provider));
        });
        const request = editing
            ? [`/admin/outages?outage_id=${this._editingId}`, 'PUT', { description }]
            : ['/admin/outages', 'POST', { description, provider_keys: [...this._selectedInstances], model_ids }];
        try {
            const resp = await fetch(request[0], {
                method: request[1], credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request[2]),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                window.UIUtils.showToast(this._errorText(data.detail, editing ? 'Failed to update outage.' : 'Failed to report outage.'), 'error');
                return;
            }
            window.UIUtils.showToast(editing
                ? 'Outage description saved.'
                : `Outage reported for ${this._plural(data.length, 'target')}.`, 'success');
            this._getModal()?.hide();
            await this._loadOutages();
        } catch (e) {
            window.UIUtils.showToast('Network error saving outage.', 'error');
        }
    }

    async resolve(outageId) {
        const outage = this._outages.find(o => o.id === outageId);
        const target = outage ? outage.target : `outage ${outageId}`;
        const confirmed = await window.UIUtils.showConfirmModal(
            'Resolve Outage',
            `Resolve the outage of "${target}"? Its models are served again right away.`,
            'warning',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch(`/admin/outages?outage_id=${outageId}`, { method: 'DELETE', credentials: 'include' });
            if (resp.ok) {
                window.UIUtils.showToast(`Outage of ${window.UIUtils.escapeHtml(target)} resolved.`, 'success');
                this._selected.delete(outageId);
                await this._loadOutages();
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils.showToast(this._errorText(e.detail, 'Failed to resolve.'), 'error');
            }
        } catch (e) {
            window.UIUtils.showToast('Network error.', 'error');
        }
    }

    async bulkResolve() {
        const ids = this._outages.filter(o => this._selected.has(o.id)).map(o => o.id);
        if (!ids.length) return;
        const confirmed = await window.UIUtils.showConfirmModal(
            'Resolve Outages',
            `Resolve ${this._plural(ids.length, 'selected outage')}? Their models are served again right away.`,
            'warning',
        );
        if (!confirmed) return;
        try {
            const resp = await fetch('/admin/outages/bulk-resolve', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids }),
            });
            const result = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                window.UIUtils.showToast(this._errorText(result.detail, 'Failed to resolve outages.'), 'error');
                return;
            }
            const missing = (result.missing || []).length;
            window.UIUtils.showToast(
                missing ? `${result.message}; ${missing} were already resolved.` : `${result.message}.`,
                missing ? 'warning' : 'success');
            this._selected.clear();
            await this._loadOutages();
        } catch (e) {
            window.UIUtils.showToast('Network error.', 'error');
        }
    }

    _getModal() {
        if (!this._modal) {
            const el = document.getElementById('outageModal');
            if (el && window.bootstrap) this._modal = new window.bootstrap.Modal(el);
        }
        return this._modal;
    }
}

window.OutageManager = new OutageManager();

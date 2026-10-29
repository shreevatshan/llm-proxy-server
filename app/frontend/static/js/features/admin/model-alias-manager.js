class ModelAliasManager {
    static API_SURFACES = ['openai', 'anthropic', 'azure_openai'];
    static API_LABELS = { openai: 'OpenAI', anthropic: 'Anthropic', azure_openai: 'Azure OpenAI' };
    static MATCH_LABELS = { exact: 'Exact', contains: 'Contains', regex: 'Regex' };
    static MATCH_ICONS = { exact: 'fa-equals', contains: 'fa-search', regex: 'fa-code' };
    static MATCH_HELP = {
        exact: 'Exact: the requested model name must equal this value (case-sensitive).',
        contains: 'Contains: matches any requested model name that includes this text (case-insensitive), e.g. "opus" matches "claude-opus-4-1" and "anthropic/claude-opus-4-1".',
        regex: 'Regex: matches when the pattern is found anywhere in the requested name (case-insensitive, Python syntax). Anchor with ^ and $ for a full match.',
    };
    static ORDER_NOTE = ' Mappings are checked top to bottom and the first match wins.';
    static PATTERN_NOTE = ModelAliasManager.ORDER_NOTE + ' Patterns apply to every endpoint on the selected APIs, including audio and images; on Azure deployment routes the name is "provider/deployment".';
    static KINDS = ['alias', 'fallback'];
    static KIND_STORE_KEY = 'model-mapping-kind';

    constructor() {
        // Both lists of the shown profile; this.aliases is the one on screen.
        this.lists = { alias: [], fallback: [] };
        this.aliases = this.lists.alias;
        this._kind = 'alias';
        try {
            const stored = window.localStorage.getItem(ModelAliasManager.KIND_STORE_KEY);
            if (ModelAliasManager.KINDS.includes(stored)) this._kind = stored;
        } catch (_) { /* storage unavailable: start on the regular list */ }
        this.models = [];
        this._comboInit = false;
        this._dndInit = false;
        this._activeIndex = -1;
        this._search = '';
        this._editingId = null;
        this._dragId = null;
        this._drop = null;
        this._orderSaving = false;
        this._orderDirty = false;
        // Ids of the shown list ticked for bulk delete; cleared when the list changes.
        this._selected = new Set();
        // The profile whose mappings are shown. Each profile has its own list; its
        // members are mapped with that list alone.
        this._profileId = window.ProfileStore.DEFAULT_ID;
        this._switcher = new window.ProfileSwitcher(document.getElementById('model-alias-switcher'), {
            label: 'Model mappings for',
            hint: 'Mappings for members of this profile only. Other profiles have their own lists.',
            // The lists shown are fresher than the profile's alias_count once a mapping is saved.
            stats: p => [
                [p.member_count, 'member'],
                [p.id === this._aliasesFor ? this.lists.alias.length + this.lists.fallback.length : p.alias_count, 'mapping'],
            ],
            onSelect: id => this.setScope(id),
        });
        this._aliasesFor = null;
        document.addEventListener('profiles:changed', () => this._onProfilesChanged());
    }

    async _onProfilesChanged() {
        const before = this._profileId;
        await this._fillScope();
        // The shown profile was deleted and the switcher fell back to Default: show
        // Default's mappings, not the deleted profile's.
        if (this._profileId !== before) await this.setScope(this._profileId);
    }

    async load() {
        this._initSelects();
        this._initCombobox();
        this._initDragAndDrop();
        this._renderKind();
        this.updateMatchHelp();
        this._profileId = window.ProfileStore.selected('model-mapping', window.ProfileStore.DEFAULT_ID);
        await this._fillScope();
        await Promise.all([this._loadAliases(), this._loadModels()]);
    }

    async _fillScope() {
        this._profileId = await this._switcher.render(this._profileId);
    }

    async setScope(value) {
        this._profileId = value ? parseInt(value, 10) : window.ProfileStore.DEFAULT_ID;
        this._switcher.value = this._profileId;
        this._switcher.refresh();
        window.ProfileStore.select('model-mapping', this._profileId);
        this._clearView();
        await this._loadAliases();
    }

    /** Show the regular ('alias') or fallback list of the shown profile. */
    setKind(kind) {
        if (!ModelAliasManager.KINDS.includes(kind) || kind === this._kind) return;
        this._kind = kind;
        try {
            window.localStorage.setItem(ModelAliasManager.KIND_STORE_KEY, kind);
        } catch (_) { /* not remembered; harmless */ }
        this.aliases = this.lists[kind];
        this._clearView();
        this._renderKind();
        this._render();
    }

    _clearView() {
        this._resetForm();
        this._selected.clear();
        this._search = '';
        const search = document.getElementById('model-alias-search');
        if (search) search.value = '';
        const out = document.getElementById('model-alias-test-result');
        if (out) out.innerHTML = '';
    }

    _isFallback() {
        return this._kind === 'fallback';
    }

    /** "mapping" or "fallback mapping", for messages. */
    _noun() {
        return this._isFallback() ? 'fallback mapping' : 'mapping';
    }

    // The list toggle, its counts and the list title.
    _renderKind() {
        for (const kind of ModelAliasManager.KINDS) {
            const btn = document.getElementById(`model-alias-kind-${kind}`);
            if (btn) {
                btn.classList.toggle('active', kind === this._kind);
                btn.setAttribute('aria-pressed', String(kind === this._kind));
            }
            const count = document.getElementById(`model-alias-count-${kind}`);
            if (count) count.textContent = this._aliasesFor === null ? '' : String(this.lists[kind].length);
        }
        const title = document.getElementById('model-alias-list-title');
        if (title) title.innerHTML = `<i class="fas fa-list me-2"></i>Configured ${this._isFallback() ? 'Fallback ' : ''}Mappings`;
        if (this._editingId === null) this._setEditing(null);
    }

    /** Query string selecting the shown profile's list of *kind* (the shown list unless given). */
    _scopeQuery(prefix = '?', kind = this._kind) {
        return `${prefix}profile_id=${this._profileId}&kind=${kind}`;
    }

    async _loadAliases() {
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        const profileId = this._profileId;
        try {
            // Both lists: the other one's size shows on the toggle, and moving a row
            // between them changes both.
            const lists = await Promise.all(ModelAliasManager.KINDS.map(async kind => {
                const response = await fetch(`/admin/model-aliases${this._scopeQuery('?', kind)}`, { credentials: 'include' });
                if (!response.ok) throw new Error('load failed');
                return response.json();
            }));
            ModelAliasManager.KINDS.forEach((kind, i) => { this.lists[kind] = lists[i]; });
            this.aliases = this.lists[this._kind];
            this._aliasesFor = profileId;
            // Drop ticks on rows that no longer exist.
            const ids = new Set(this.aliases.map(row => row.id));
            this._selected.forEach(id => { if (!ids.has(id)) this._selected.delete(id); });
            this._switcher.refresh();
            this._renderKind();
            this._render();
        } catch (_) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-danger py-4">Failed to load mappings.</td></tr>';
        }
    }

    filter(value) {
        this._search = value || '';
        this._render();
    }

    // Fill the match-type and test-API dropdowns from the label maps above.
    _initSelects() {
        const fill = (id, labels) => {
            const select = document.getElementById(id);
            if (!select || select.dataset.filled) return;
            select.dataset.filled = '1';
            for (const [value, label] of Object.entries(labels)) select.add(new Option(label, value));
        };
        fill('model-alias-match-type', ModelAliasManager.MATCH_LABELS);
        fill('model-alias-test-api', ModelAliasManager.API_LABELS);
    }

    _canReorder() {
        return !this._search.trim() && this.aliases.length > 1;
    }

    /** Rows of the shown list that pass the search, with their list index. */
    _shownRows() {
        const q = this._search.trim().toLowerCase();
        return this.aliases
            .map((row, index) => ({ row, index }))
            .filter(({ row }) => !q
                || (row.alias || '').toLowerCase().includes(q)
                || (row.target_model_id || '').toLowerCase().includes(q)
                || (ModelAliasManager.MATCH_LABELS[row.match_type] || '').toLowerCase().includes(q));
    }

    _render() {
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        const esc = window.UIUtils?.escapeHtml || (value => value);
        const hint = document.getElementById('model-alias-reorder-hint');
        if (hint) hint.style.display = this._search.trim() && this.aliases.length > 1 ? '' : 'none';
        this._syncSelection();
        if (!this.aliases.length) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted py-4">No ${this._noun()}s configured.</td></tr>`;
            return;
        }
        const rows = this._shownRows();
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No mappings match your search.</td></tr>';
            return;
        }
        const reorder = this._canReorder();
        const last = this.aliases.length - 1;
        const other = this._isFallback() ? 'mappings' : 'fallback mappings';
        tbody.innerHTML = rows.map(({ row, index }) => {
            const type = row.match_type || 'exact';
            // A regular mapping only applies to an available model, so one whose
            // name is no model the profile can use never fires.
            const neverApplies = row.kind !== 'fallback' && row.source_available === false;
            const name = esc(row.alias) + (neverApplies
                ? ` <span class="alias-warn" title="'${esc(row.alias)}' is not a model available to this profile, and mappings only apply to available models. Move it to the fallback mappings to apply it to unavailable names."><i class="fas fa-exclamation-triangle"></i>Never applies</span>`
                : '');
            const selected = this._selected.has(row.id);
            return `
            <tr data-id="${row.id}" class="alias-row${selected ? ' selected' : ''}">
            <td class="col-select"><input type="checkbox" class="form-check-input" aria-label="Select ${esc(row.alias)}"
                ${selected ? 'checked' : ''} onchange="window.ModelAliasManager?.toggleRow(${row.id}, this.checked)"></td>
            <td class="alias-order-cell">
                <span class="alias-grip ${reorder ? '' : 'is-disabled'}" title="${reorder ? 'Drag to reorder' : 'Clear search to reorder'}"><i class="fas fa-grip-vertical"></i></span>
                <span class="alias-position">${index + 1}</span>
                <button class="btn btn-link btn-sm alias-move-btn" title="Move up" aria-label="Move up" ${reorder && index > 0 ? '' : 'disabled'} onclick="window.ModelAliasManager?.move(${index}, -1)"><i class="fas fa-chevron-up"></i></button>
                <button class="btn btn-link btn-sm alias-move-btn" title="Move down" aria-label="Move down" ${reorder && index < last ? '' : 'disabled'} onclick="window.ModelAliasManager?.move(${index}, 1)"><i class="fas fa-chevron-down"></i></button>
            </td>
            <td>${name}</td>
            <td><span class="alias-match alias-match-${esc(type)}"><i class="fas ${ModelAliasManager.MATCH_ICONS[type] || 'fa-question'}"></i>${esc(ModelAliasManager.MATCH_LABELS[type] || type)}</span></td>
            <td><code>${esc(row.target_model_id)}</code></td>
            <td>${(row.apis || []).map(api => `<span class="ma-tag">${esc(ModelAliasManager.API_LABELS[api] || api)}</span>`).join('')}</td>
            <td><span class="status-badge ${row.enabled ? 'status-active' : 'status-inactive'}">${row.enabled ? 'Enabled' : 'Disabled'}</span></td>
            <td><div class="d-flex align-items-center gap-3">
                <label class="ma-switch" title="${row.enabled ? 'Disable' : 'Enable'} mapping">
                    <input type="checkbox" ${row.enabled ? 'checked' : ''} onchange="window.ModelAliasManager?.toggle(${index})">
                    <span class="ma-slider"></span>
                </label>
                <button class="btn btn-outline-secondary btn-sm" title="Edit mapping" onclick="window.ModelAliasManager?.edit(${index})"><i class="fas fa-pencil-alt"></i></button>
                <button class="btn btn-outline-secondary btn-sm" title="Move to ${other}" aria-label="Move to ${other}" onclick="window.ModelAliasManager?.moveToOtherList(${index})"><i class="fas fa-exchange-alt"></i></button>
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelAliasManager?.remove(${index})"><i class="fas fa-trash"></i></button>
            </div></td></tr>`;
        }).join('');
    }

    // ==================== Bulk selection ====================

    toggleRow(id, checked) {
        if (checked) this._selected.add(id);
        else this._selected.delete(id);
        document.querySelector(`#model-aliases-tbody tr[data-id="${id}"]`)?.classList.toggle('selected', checked);
        this._syncSelection();
    }

    toggleSelectAll(checked) {
        // Only the rows shown, so a filtered select-all never sweeps in hidden mappings.
        this._shownRows().forEach(({ row }) => {
            if (checked) this._selected.add(row.id);
            else this._selected.delete(row.id);
        });
        this._render();
    }

    clearSelection() {
        this._selected.clear();
        this._render();
    }

    _syncSelection() {
        const n = this._selected.size;
        const bar = document.getElementById('model-alias-bulk');
        if (bar) bar.style.display = n ? '' : 'none';
        const count = document.getElementById('model-alias-bulk-count');
        if (count) count.textContent = `${n} selected`;
        const all = document.getElementById('model-alias-select-all');
        if (all) {
            const shown = this._shownRows();
            const picked = shown.filter(({ row }) => this._selected.has(row.id)).length;
            all.checked = shown.length > 0 && picked === shown.length;
            all.indeterminate = picked > 0 && picked < shown.length;
            all.disabled = !shown.length;
        }
    }

    async bulkRemove() {
        const ids = this.aliases.filter(row => this._selected.has(row.id)).map(row => row.id);
        if (!ids.length) return;
        const n = ids.length;
        const noun = `${this._noun()}${n === 1 ? '' : 's'}`;
        const confirmed = await window.UIUtils?.showConfirmModal(
            this._isFallback() ? 'Delete Fallback Mappings' : 'Delete Model Mappings',
            `Delete ${n} selected ${noun}? This cannot be undone.`, 'danger');
        if (!confirmed) return;
        try {
            const response = await fetch('/admin/model-aliases/bulk-delete', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids, profile_id: this._profileId, kind: this._kind }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, `Failed to delete ${noun}.`));
            const missing = (result.missing || []).length;
            window.UIUtils?.showToast(
                missing ? `${result.message}; ${missing} were already gone.` : `${result.message}.`,
                missing ? 'warning' : 'success');
            if (this._editingId !== null && (result.deleted || []).includes(this._editingId)) this._resetForm();
            this._selected.clear();
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || `Failed to delete ${noun}.`, 'error');
        }
    }

    _initDragAndDrop() {
        if (this._dndInit) return;
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        this._dndInit = true;

        // Rows only become draggable while the grip is held, so text in the
        // row stays selectable and buttons keep working.
        tbody.addEventListener('mousedown', (e) => {
            const grip = e.target.closest('.alias-grip');
            if (!grip || grip.classList.contains('is-disabled') || !this._canReorder()) return;
            const tr = grip.closest('tr[data-id]');
            if (tr) tr.draggable = true;
        });
        // Listen on the document: the button may be released outside the table.
        document.addEventListener('mouseup', () => {
            if (this._dragId !== null) return;
            tbody.querySelectorAll('tr[draggable="true"]').forEach(tr => { tr.draggable = false; });
        });
        tbody.addEventListener('dragstart', (e) => {
            const tr = e.target.closest('tr[data-id]');
            if (!tr || !tr.draggable) return;
            this._dragId = Number(tr.dataset.id);
            tr.classList.add('alias-dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', tr.dataset.id);
        });
        tbody.addEventListener('dragover', (e) => {
            if (this._dragId === null) return;
            const tr = e.target.closest('tr[data-id]');
            if (!tr) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            const rect = tr.getBoundingClientRect();
            const after = e.clientY > rect.top + rect.height / 2;
            this._clearDropMarkers(tbody);
            if (Number(tr.dataset.id) === this._dragId) {
                this._drop = null;
                return;
            }
            tr.classList.add(after ? 'alias-drop-after' : 'alias-drop-before');
            this._drop = { id: Number(tr.dataset.id), after };
        });
        tbody.addEventListener('drop', (e) => {
            if (this._dragId === null) return;
            e.preventDefault();
            const drop = this._drop;
            const dragId = this._dragId;
            this._endDrag(tbody);
            if (!drop) return;
            const from = this.aliases.findIndex(row => row.id === dragId);
            if (from < 0) return;
            const [moved] = this.aliases.splice(from, 1);
            let to = this.aliases.findIndex(row => row.id === drop.id);
            if (to < 0) {
                this.aliases.splice(from, 0, moved);
                return;
            }
            if (drop.after) to += 1;
            this.aliases.splice(to, 0, moved);
            if (to === from) return;
            this._render();
            this._persistOrder();
        });
        tbody.addEventListener('dragend', () => this._endDrag(tbody));
    }

    _clearDropMarkers(tbody) {
        tbody.querySelectorAll('.alias-drop-before, .alias-drop-after')
            .forEach(el => el.classList.remove('alias-drop-before', 'alias-drop-after'));
    }

    _endDrag(tbody) {
        this._clearDropMarkers(tbody);
        tbody.querySelectorAll('tr[data-id]').forEach(tr => {
            tr.classList.remove('alias-dragging');
            tr.draggable = false;
        });
        this._dragId = null;
        this._drop = null;
    }

    move(index, delta) {
        if (!this._canReorder()) return;
        const to = index + delta;
        if (index < 0 || to < 0 || to >= this.aliases.length) return;
        [this.aliases[index], this.aliases[to]] = [this.aliases[to], this.aliases[index]];
        this._render();
        this._persistOrder();
    }

    // One order PUT at a time: moves made while a save is in flight are sent
    // as a single follow-up with the latest order, so neither the server nor
    // this.aliases can end up on an older order from a late response.
    _persistOrder() {
        this._orderDirty = true;
        if (!this._orderSaving) this._flushOrder();
    }

    async _flushOrder() {
        this._orderSaving = true;
        try {
            while (this._orderDirty) {
                this._orderDirty = false;
                const response = await fetch('/admin/model-aliases/order', {
                    method: 'PUT', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids: this.aliases.map(row => row.id), profile_id: this._profileId, kind: this._kind }),
                });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(this._errorMessage(result, 'Failed to save mapping order.'));
                if (!this._orderDirty) {
                    this.aliases = this.lists[this._kind] = result;
                    this._render();
                    window.UIUtils?.showToast('Mapping order saved.', 'success');
                }
            }
        } catch (error) {
            this._orderDirty = false;
            window.UIUtils?.showToast(error.message || 'Failed to save mapping order.', 'error');
            await this._loadAliases();
        } finally {
            this._orderSaving = false;
        }
    }

    _errorMessage(result, fallback) {
        const detail = result && result.detail;
        if (Array.isArray(detail)) {
            // Pydantic 422: [{loc, msg, ...}]
            const text = detail.map(d => String(d.msg || '').replace(/^Value error, /, '')).filter(Boolean).join('; ');
            return text || fallback;
        }
        return detail || fallback;
    }

    async _loadModels() {
        try {
            const response = await fetch('/admin/models/all', { credentials: 'include' });
            if (!response.ok) return;
            this.models = await response.json();
        } catch (_) { /* The save endpoint still performs authoritative validation. */ }
    }

    _initCombobox() {
        if (this._comboInit) return;
        const input = document.getElementById('model-alias-target');
        const menu = document.getElementById('model-alias-target-menu');
        const box = document.getElementById('model-alias-combobox');
        if (!input || !menu || !box) return;
        this._comboInit = true;

        const open = () => this._renderMenu(input.value);
        input.addEventListener('focus', open);
        input.addEventListener('input', open);
        input.addEventListener('keydown', (e) => this._onKeydown(e));
        document.addEventListener('click', (e) => {
            if (!box.contains(e.target)) this._closeMenu();
        });
    }

    _renderMenu(query) {
        const menu = document.getElementById('model-alias-target-menu');
        const input = document.getElementById('model-alias-target');
        if (!menu || !input) return;
        const esc = window.UIUtils?.escapeHtml || (value => value);
        const q = (query || '').trim().toLowerCase();
        const matches = this.models
            .map(m => m.model_id)
            .filter(id => !q || id.toLowerCase().includes(q))
            .slice(0, 50);
        this._activeIndex = -1;
        if (!matches.length) {
            menu.innerHTML = '<li class="alias-combobox-empty">No matching models</li>';
        } else {
            menu.innerHTML = matches.map((id, i) => `
                <li class="alias-combobox-item" role="option" data-index="${i}" data-value="${esc(id)}"
                    onmousedown="window.ModelAliasManager?._pick('${esc(id).replace(/'/g, "\\'")}')">${esc(id)}</li>`).join('');
        }
        menu.classList.add('is-open');
        input.setAttribute('aria-expanded', 'true');
    }

    _pick(value) {
        const input = document.getElementById('model-alias-target');
        if (input) input.value = value;
        this._closeMenu();
    }

    _closeMenu() {
        const menu = document.getElementById('model-alias-target-menu');
        const input = document.getElementById('model-alias-target');
        if (menu) menu.classList.remove('is-open');
        if (input) input.setAttribute('aria-expanded', 'false');
        this._activeIndex = -1;
    }

    _onKeydown(e) {
        const menu = document.getElementById('model-alias-target-menu');
        if (!menu || !menu.classList.contains('is-open')) return;
        const items = Array.from(menu.querySelectorAll('.alias-combobox-item'));
        if (!items.length) return;
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            const delta = e.key === 'ArrowDown' ? 1 : -1;
            this._activeIndex = (this._activeIndex + delta + items.length) % items.length;
            items.forEach((el, i) => el.classList.toggle('is-active', i === this._activeIndex));
            items[this._activeIndex].scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'Enter' && this._activeIndex >= 0) {
            e.preventDefault();
            this._pick(items[this._activeIndex].dataset.value);
        } else if (e.key === 'Escape') {
            this._closeMenu();
        }
    }

    _readApis() {
        return ModelAliasManager.API_SURFACES.filter(api => {
            const box = document.getElementById(`model-alias-api-${api}`);
            return box && box.checked;
        });
    }

    _setApis(apis) {
        const selected = new Set(apis || ModelAliasManager.API_SURFACES);
        ModelAliasManager.API_SURFACES.forEach(api => {
            const box = document.getElementById(`model-alias-api-${api}`);
            if (box) box.checked = selected.has(api);
        });
    }

    _matchType() {
        return document.getElementById('model-alias-match-type')?.value || 'exact';
    }

    updateMatchHelp() {
        const type = this._matchType();
        const help = document.getElementById('model-alias-match-help');
        if (help) {
            const note = type === 'exact' ? ModelAliasManager.ORDER_NOTE : ModelAliasManager.PATTERN_NOTE;
            help.textContent = (ModelAliasManager.MATCH_HELP[type] || '') + note;
        }
        const input = document.getElementById('model-alias-name');
        if (input) {
            input.placeholder = { exact: 'e.g. gpt-5.5', contains: 'e.g. opus', regex: 'e.g. ^claude-(opus|sonnet)-4' }[type] || '';
        }
    }

    _setEditing(row) {
        this._editingId = row ? row.id : null;
        const cancel = document.getElementById('model-alias-cancel-btn');
        const saveBtn = document.getElementById('model-alias-save-btn');
        const title = document.getElementById('model-alias-form-title');
        if (cancel) cancel.style.display = row ? '' : 'none';
        if (saveBtn) saveBtn.innerHTML = row ? '<i class="fas fa-save me-1"></i>Update' : '<i class="fas fa-save me-1"></i>Save';
        if (title) {
            const esc = window.UIUtils?.escapeHtml || (value => value);
            const noun = this._isFallback() ? 'Fallback Mapping' : 'Mapping';
            title.innerHTML = row
                ? `<i class="fas fa-pencil-alt me-2"></i>Edit ${noun} <code>${esc(row.alias)}</code>`
                : `<i class="fas fa-plus me-2"></i>Add or Update ${noun}`;
        }
    }

    _resetForm() {
        document.getElementById('model-alias-name').value = '';
        document.getElementById('model-alias-target').value = '';
        const select = document.getElementById('model-alias-match-type');
        if (select) select.value = 'exact';
        this._setApis(ModelAliasManager.API_SURFACES);
        this._closeMenu();
        this._setEditing(null);
        this.updateMatchHelp();
    }

    cancelEdit() {
        this._resetForm();
    }

    async save(event) {
        event.preventDefault();
        const alias = document.getElementById('model-alias-name').value.trim();
        const target_model_id = document.getElementById('model-alias-target').value.trim();
        const match_type = this._matchType();
        const apis = this._readApis();
        if (!apis.length) {
            window.UIUtils?.showToast('Select at least one API surface.', 'error');
            return;
        }
        if (match_type !== 'exact' && alias.length <= 2) {
            const confirmed = await window.UIUtils?.showConfirmModal(
                'Very Broad Pattern',
                `The ${ModelAliasManager.MATCH_LABELS[match_type].toLowerCase()} pattern '${alias}' is very short and will likely match many models. Save it anyway?`,
                'warning',
            );
            if (!confirmed) return;
        }
        const existing = this._editingId !== null
            ? this.aliases.find(row => row.id === this._editingId)
            : this.aliases.find(row => row.alias === alias);
        // Saving from the add form updates a same-named row; make sure the
        // form's match type (which defaults to Exact) is meant to replace its type.
        const existingType = existing?.match_type || 'exact';
        if (existing && this._editingId === null && existingType !== match_type) {
            const label = type => (ModelAliasManager.MATCH_LABELS[type] || type).toLowerCase();
            const confirmed = await window.UIUtils?.showConfirmModal(
                'Change Match Type',
                `A ${label(existingType)} mapping for '${alias}' already exists. Saving will change it to ${label(match_type)}. Continue?`,
                'warning',
            );
            if (!confirmed) return;
        }
        const enabled = existing ? existing.enabled : true;
        const payload = { alias, target_model_id, enabled, apis, match_type, profile_id: this._profileId, kind: this._kind };
        if (this._editingId !== null) payload.id = this._editingId;
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, `Failed to save ${this._noun()}.`));
            if (result.source_available === false) {
                window.UIUtils?.showToast(`Mapping '${alias}' saved, but '${alias}' is not an available model, so it will never apply. Move it to the fallback mappings to map unavailable names.`, 'warning');
            } else {
                window.UIUtils?.showToast(`${this._isFallback() ? 'Fallback mapping' : 'Mapping'} '${alias}' saved.`, 'success');
            }
            this._resetForm();
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || `Failed to save ${this._noun()}.`, 'error');
        }
    }

    /** Move a row to the other list, at the bottom of it. */
    async moveToOtherList(index) {
        const row = this.aliases[index];
        if (!row) return;
        const target = this._isFallback() ? 'alias' : 'fallback';
        const targetLabel = target === 'fallback' ? 'fallback mappings' : 'mappings';
        if (this.lists[target].some(other => other.alias === row.alias)) {
            window.UIUtils?.showToast(`The ${targetLabel} already have an entry for '${row.alias}'.`, 'error');
            return;
        }
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: row.id, alias: row.alias, target_model_id: row.target_model_id,
                    enabled: row.enabled, apis: row.apis, match_type: row.match_type || 'exact',
                    profile_id: this._profileId, kind: target,
                }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, 'Failed to move mapping.'));
            window.UIUtils?.showToast(`'${row.alias}' moved to the ${targetLabel}.`, 'success');
            if (this._editingId === row.id) this._resetForm();
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to move mapping.', 'error');
        }
    }

    edit(index) {
        const row = this.aliases[index];
        if (!row) return;
        document.getElementById('model-alias-name').value = row.alias;
        document.getElementById('model-alias-target').value = row.target_model_id;
        const select = document.getElementById('model-alias-match-type');
        if (select) select.value = row.match_type || 'exact';
        this._setApis(row.apis);
        this._closeMenu();
        this._setEditing(row);
        this.updateMatchHelp();
        document.getElementById('model-alias-name').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    async toggle(index) {
        const row = this.aliases[index];
        if (!row) return;
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: row.id, alias: row.alias, target_model_id: row.target_model_id,
                    enabled: !row.enabled, apis: row.apis, match_type: row.match_type || 'exact',
                    profile_id: this._profileId, kind: this._kind,
                }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, 'Failed to update mapping.'));
            window.UIUtils?.showToast(`Mapping '${row.alias}' ${row.enabled ? 'disabled' : 'enabled'}.`, 'success');
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to update mapping.', 'error');
            this._render();
        }
    }

    async remove(index) {
        const row = this.aliases[index];
        if (!row) return;
        const { alias } = row;
        const confirmed = await window.UIUtils?.showConfirmModal(
            this._isFallback() ? 'Delete Fallback Mapping' : 'Delete Model Mapping', `Delete ${this._noun()} '${alias}'?`, 'danger');
        if (!confirmed) return;
        try {
            const response = await fetch(`/admin/model-aliases/${encodeURIComponent(alias)}${this._scopeQuery()}`, { method: 'DELETE', credentials: 'include' });
            if (!response.ok) throw new Error((await response.json()).detail || 'Failed to delete mapping.');
            window.UIUtils?.showToast(`Mapping '${alias}' deleted.`, 'success');
            if (this._editingId === row.id) this._resetForm();
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to delete mapping.', 'error');
        }
    }

    async test(event) {
        if (event) event.preventDefault();
        const model = document.getElementById('model-alias-test-name')?.value.trim();
        const api = document.getElementById('model-alias-test-api')?.value || '';
        const out = document.getElementById('model-alias-test-result');
        if (!model || !out) return;
        const esc = window.UIUtils?.escapeHtml || (value => value);
        const params = new URLSearchParams({ model });
        if (api) params.set('api', api);
        params.set('profile_id', this._profileId);
        try {
            const response = await fetch(`/admin/model-aliases/test?${params}`, { credentials: 'include' });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, 'Test failed.'));
            out.innerHTML = result.map(item => {
                const label = esc(ModelAliasManager.API_LABELS[item.api] || item.api);
                const fallback = item.kind === 'fallback';
                const state = item.available
                    ? '<span class="alias-match alias-match-exact">Available</span> → mappings:'
                    : '<span class="alias-warn">Unavailable</span> → fallback mappings:';
                if (!item.matched) {
                    const outcome = fallback ? 'no match — the request fails as unavailable' : 'no match — passed through unchanged';
                    return `<li><strong>${label}:</strong> ${state} <span class="text-muted">${outcome}</span></li>`;
                }
                const position = (this.lists[item.kind] || []).findIndex(row => row.id === item.id) + 1;
                const noun = fallback ? 'Fallback' : 'Mapping';
                const which = position > 0 ? `${noun} #${position}` : noun;
                const type = esc(ModelAliasManager.MATCH_LABELS[item.match_type] || item.match_type);
                return `<li><strong>${label}:</strong> ${state} ${which} <code>${esc(item.alias)}</code> (${type}) → <code>${esc(item.target_model_id)}</code></li>`;
            }).join('');
        } catch (error) {
            out.innerHTML = `<li class="text-danger">${esc(error.message || 'Test failed.')}</li>`;
        }
    }
}

window.ModelAliasManager = new ModelAliasManager();

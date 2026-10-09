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

    constructor() {
        this.aliases = [];
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
    }

    async load() {
        this._initSelects();
        this._initCombobox();
        this._initDragAndDrop();
        this.updateMatchHelp();
        await Promise.all([this._loadAliases(), this._loadModels()]);
    }

    async _loadAliases() {
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        try {
            const response = await fetch('/admin/model-aliases', { credentials: 'include' });
            if (!response.ok) throw new Error('load failed');
            this.aliases = await response.json();
            this._render();
        } catch (_) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-danger py-4">Failed to load mappings.</td></tr>';
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

    _render() {
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        const esc = window.UIUtils?.escapeHtml || (value => value);
        const hint = document.getElementById('model-alias-reorder-hint');
        if (hint) hint.style.display = this._search.trim() && this.aliases.length > 1 ? '' : 'none';
        if (!this.aliases.length) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">No mappings configured.</td></tr>';
            return;
        }
        const q = this._search.trim().toLowerCase();
        const rows = this.aliases
            .map((row, index) => ({ row, index }))
            .filter(({ row }) => !q
                || (row.alias || '').toLowerCase().includes(q)
                || (row.target_model_id || '').toLowerCase().includes(q)
                || (ModelAliasManager.MATCH_LABELS[row.match_type] || '').toLowerCase().includes(q));
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">No mappings match your search.</td></tr>';
            return;
        }
        const reorder = this._canReorder();
        const last = this.aliases.length - 1;
        tbody.innerHTML = rows.map(({ row, index }) => {
            const type = row.match_type || 'exact';
            const name = type === 'exact' ? esc(row.alias) : `<code>${esc(row.alias)}</code>`;
            return `
            <tr data-id="${row.id}">
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
                <button class="btn btn-outline-danger btn-sm" onclick="window.ModelAliasManager?.remove(${index})"><i class="fas fa-trash"></i></button>
            </div></td></tr>`;
        }).join('');
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
                    body: JSON.stringify({ ids: this.aliases.map(row => row.id) }),
                });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(this._errorMessage(result, 'Failed to save mapping order.'));
                if (!this._orderDirty) {
                    this.aliases = result;
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
            title.innerHTML = row
                ? `<i class="fas fa-pencil-alt me-2"></i>Edit Mapping <code>${esc(row.alias)}</code>`
                : '<i class="fas fa-plus me-2"></i>Add or Update Mapping';
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
        const payload = { alias, target_model_id, enabled, apis, match_type };
        if (this._editingId !== null) payload.id = this._editingId;
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, 'Failed to save mapping.'));
            window.UIUtils?.showToast(`Mapping '${alias}' saved.`, 'success');
            this._resetForm();
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to save mapping.', 'error');
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
        const confirmed = await window.UIUtils?.showConfirmModal('Delete Model Mapping', `Delete mapping '${alias}'?`, 'danger');
        if (!confirmed) return;
        try {
            const response = await fetch(`/admin/model-aliases/${encodeURIComponent(alias)}`, { method: 'DELETE', credentials: 'include' });
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
        try {
            const response = await fetch(`/admin/model-aliases/test?${params}`, { credentials: 'include' });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(this._errorMessage(result, 'Test failed.'));
            out.innerHTML = result.map(item => {
                const label = esc(ModelAliasManager.API_LABELS[item.api] || item.api);
                if (!item.matched) {
                    return `<li><strong>${label}:</strong> <span class="text-muted">no mapping — passed through unchanged</span></li>`;
                }
                const position = this.aliases.findIndex(row => row.id === item.id) + 1;
                const which = position > 0 ? `Mapping #${position}` : 'Mapping';
                const type = esc(ModelAliasManager.MATCH_LABELS[item.match_type] || item.match_type);
                return `<li><strong>${label}:</strong> ${which} <code>${esc(item.alias)}</code> (${type}) → <code>${esc(item.target_model_id)}</code></li>`;
            }).join('');
        } catch (error) {
            out.innerHTML = `<li class="text-danger">${esc(error.message || 'Test failed.')}</li>`;
        }
    }
}

window.ModelAliasManager = new ModelAliasManager();

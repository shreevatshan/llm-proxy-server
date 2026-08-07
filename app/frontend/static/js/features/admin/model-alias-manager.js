class ModelAliasManager {
    static API_SURFACES = ['openai', 'anthropic', 'azure_openai'];
    static API_LABELS = { openai: 'OpenAI', anthropic: 'Anthropic', azure_openai: 'Azure OpenAI' };

    constructor() {
        this.aliases = [];
        this.models = [];
        this._comboInit = false;
        this._activeIndex = -1;
    }

    async load() {
        this._initCombobox();
        await Promise.all([this._loadAliases(), this._loadModels()]);
    }

    async _loadAliases() {
        const tbody = document.getElementById('model-aliases-tbody');
        if (!tbody) return;
        try {
            const response = await fetch('/admin/model-aliases', { credentials: 'include' });
            if (!response.ok) throw new Error('load failed');
            this.aliases = await response.json();
            const esc = window.UIUtils?.escapeHtml || (value => value);
            tbody.innerHTML = this.aliases.length ? this.aliases.map((row, index) => `
                <tr><td>${esc(row.alias)}</td><td><code>${esc(row.target_model_id)}</code></td>
                <td>${(row.apis || []).map(api => `<span class="ma-tag">${esc(ModelAliasManager.API_LABELS[api] || api)}</span>`).join('')}</td>
                <td><span class="status-badge ${row.enabled ? 'status-active' : 'status-inactive'}">${row.enabled ? 'Enabled' : 'Disabled'}</span></td>
                <td><div class="d-flex align-items-center gap-3">
                    <label class="ma-switch" title="${row.enabled ? 'Disable' : 'Enable'} mapping">
                        <input type="checkbox" ${row.enabled ? 'checked' : ''} onchange="window.ModelAliasManager?.toggle(${index})">
                        <span class="ma-slider"></span>
                    </label>
                    <button class="btn btn-outline-secondary btn-sm" title="Edit mapping" onclick="window.ModelAliasManager?.edit(${index})"><i class="fas fa-pencil-alt"></i></button>
                    <button class="btn btn-outline-danger btn-sm" onclick="window.ModelAliasManager?.remove(${index})"><i class="fas fa-trash"></i></button>
                </div></td></tr>`).join('')
                : '<tr><td colspan="5" class="text-center text-muted py-4">No mappings configured.</td></tr>';
        } catch (_) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">Failed to load mappings.</td></tr>';
        }
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

    async save(event) {
        event.preventDefault();
        const alias = document.getElementById('model-alias-name').value.trim();
        const target_model_id = document.getElementById('model-alias-target').value.trim();
        const apis = this._readApis();
        if (!apis.length) {
            window.UIUtils?.showToast('Select at least one API surface.', 'error');
            return;
        }
        const existing = this.aliases.find(row => row.alias === alias);
        const enabled = existing ? existing.enabled : true;
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alias, target_model_id, enabled, apis }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.detail || 'Failed to save mapping.');
            window.UIUtils?.showToast(`Mapping '${alias}' saved.`, 'success');
            document.getElementById('model-alias-name').value = '';
            document.getElementById('model-alias-target').value = '';
            this._setApis(ModelAliasManager.API_SURFACES);
            this._closeMenu();
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
        this._setApis(row.apis);
        this._closeMenu();
        document.getElementById('model-alias-name').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    async toggle(index) {
        const row = this.aliases[index];
        if (!row) return;
        try {
            const response = await fetch('/admin/model-aliases', {
                method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alias: row.alias, target_model_id: row.target_model_id, enabled: !row.enabled, apis: row.apis }),
            });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.detail || 'Failed to update mapping.');
            window.UIUtils?.showToast(`Mapping '${row.alias}' ${row.enabled ? 'disabled' : 'enabled'}.`, 'success');
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to update mapping.', 'error');
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
            await this._loadAliases();
        } catch (error) {
            window.UIUtils?.showToast(error.message || 'Failed to delete mapping.', 'error');
        }
    }
}

window.ModelAliasManager = new ModelAliasManager();
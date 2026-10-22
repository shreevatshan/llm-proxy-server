// Admin Web Search tab: interception settings, the defaults users inherit, and
// the list of backends. WsPicker, WsParamsForm and the WS_* constants come from
// features/shared/websearch-params.js.

const WS_SURFACES = [
    { value: 'anthropic_messages', label: 'Anthropic Messages', hint: '/messages' },
    { value: 'chat_completions', label: 'Chat Completions', hint: '/chat/completions' },
    { value: 'responses', label: 'Responses', hint: '/responses' },
];

const WS_URL_HELP = {
    searxng: 'Every query the model writes is sent here. The instance must list <code>json</code> under <code>search.formats</code> in its <code>settings.yml</code>.',
    fourget: 'Queried at <code>/api/v1/web</code>. The instance must have its API enabled and bot protection off.',
};
const WS_URL_PLACEHOLDER = { searxng: 'http://searxng:8080', fourget: 'http://fourget' };

class WebSearchManager {
    constructor() {
        this._settings = null;
        this._providers = [];
        this._backends = [];
        this._pickers = null;
        this._params = null;       // WsParamsForm in the backend modal
        this._editing = null;      // backend the modal edits; null = adding one
        this._modalType = 'searxng';
        this._returnFocus = null;
        this._testRun = 0;
    }

    async load() {
        this._init();
        await Promise.all([this._loadSettings(), this._loadProviders(), this._loadBackends()]);
        if (this._settings) this._fill(this._settings);
        this._renderProviders();
        this._renderBackends();
    }

    _el(id) {
        return document.getElementById(id);
    }

    _init() {
        if (this._pickers) return;
        const allProviders = () => this._el('ws-all-providers')?.checked;
        this._pickers = {
            surfaces: new WsPicker(this._el('ws-surfaces-picker'), {
                searchable: false,
                placeholder: 'None, interception never runs',
                summarize: picked => picked.length === WS_SURFACES.length ? 'All' : null,
            }),
            providers: new WsPicker(this._el('ws-providers-picker'), {
                searchPlaceholder: 'Search providers',
                placeholder: 'None selected',
                emptyText: 'No providers configured.',
                isDisabled: allProviders,
                summarize: () => allProviders() ? 'All' : null,
                extraHtml: `
                    <div class="form-check form-switch mb-0">
                        <input class="form-check-input" type="checkbox" role="switch" id="ws-all-providers">
                        <label class="form-check-label" for="ws-all-providers">All providers, including ones added later</label>
                    </div>`,
            }),
        };
        this._el('ws-all-providers').addEventListener('change', () => this._pickers.providers.render());

        const body = this._el('ws-backends-body');
        body.addEventListener('click', e => this._onRowClick(e));

        let scrapers = [];
        try { scrapers = JSON.parse(this._el('ws-fourget-scrapers').textContent); } catch (e) { /* none */ }
        this._params = new WsParamsForm(this._el('wsb-params'), {
            idPrefix: 'wsb',
            scrapers,
            // The modal body would clip the pickers' menus otherwise.
            fixedMenus: true,
            fetchEngines: () => this._fetchEngines(),
        });
        document.querySelectorAll('input[name="wsb-type"]').forEach(radio => {
            radio.addEventListener('change', () => this._setModalType(radio.value));
        });
        const modal = this._el('wsBackendModal');
        modal.addEventListener('hidden.bs.modal', () => {
            this._returnFocus?.focus();
            this._returnFocus = null;
        });
    }

    // ---- loading -------------------------------------------------------

    async _get(url, failure) {
        try {
            const resp = await fetch(url, { credentials: 'include' });
            if (resp.ok) return await resp.json();
            if (failure) window.UIUtils?.showToast(failure, 'error');
        } catch (e) {
            console.error('WebSearchManager:', url, e);
            if (failure) window.UIUtils?.showToast(failure, 'error');
        }
        return null;
    }

    async _loadSettings() {
        const data = await this._get('/admin/websearch/settings', 'Could not load the web search settings.');
        if (data) this._settings = data;
    }

    async _loadProviders() {
        const data = await this._get('/admin/websearch/providers');
        this._providers = data?.providers || [];
    }

    async _loadBackends() {
        const data = await this._get('/admin/websearch/backends', 'Could not load the web search backends.');
        if (data) this._backends = data;
    }

    // Write requests: a toast on failure, the parsed body on success, else null.
    async _send(url, method, body, failure) {
        try {
            const resp = await fetch(url, {
                method,
                credentials: 'include',
                headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
                body: body === undefined ? undefined : JSON.stringify(body),
            });
            const data = await resp.json().catch(() => ({}));
            if (resp.ok) return data;
            window.UIUtils?.showToast(wsErrorMessage(data, failure), 'error');
        } catch (e) {
            window.UIUtils?.showToast(`${failure} Check your connection and try again.`, 'error');
        }
        return null;
    }

    // ---- settings ------------------------------------------------------

    _enabledBackends() {
        return this._backends.filter(b => b.enabled);
    }

    _defaultBackend() {
        return this._backends.find(b => b.is_default) || null;
    }

    _fill(s) {
        this._el('ws-enabled').checked = !!s.enabled;
        this._el('ws-max-loops').value = s.max_agentic_loops;
        this._el('ws-max-queries').value = s.max_queries_per_turn;
        this._el('ws-timeout').value = s.timeout_seconds;
        this._el('ws-max-results').value = s.max_results;
        this._el('ws-snippet-chars').value = s.max_snippet_chars;
        this._pickers.surfaces.set(WS_SURFACES, s.apply_to || []);
        this._el('ws-all-providers').checked = (s.enabled_providers || []).includes('*');
        this._renderDefaultSelect();
        const updated = this._el('ws-updated');
        updated.textContent = s.updated_at
            ? `Settings last saved ${new Date(s.updated_at + (s.updated_at.endsWith('Z') ? '' : 'Z')).toLocaleString()}${s.updated_by ? ' by ' + s.updated_by : ''}.`
            : 'Settings not saved yet.';
    }

    _renderDefaultSelect() {
        const select = this._el('ws-default-backend');
        const enabled = this._enabledBackends();
        const current = this._settings?.default_backend_id ?? null;
        select.innerHTML = enabled.length
            ? enabled.map(b => `<option value="${b.id}">${wsEscape(b.name)} (${WS_TYPE_LABELS[b.type] || b.type})</option>`).join('')
            : '<option value="">Add a backend first</option>';
        select.disabled = !enabled.length;
        if (enabled.some(b => b.id === current)) select.value = String(current);
        // Without a backend, interception has nowhere to send searches.
        const toggle = this._el('ws-enabled');
        toggle.disabled = !enabled.length && !toggle.checked;
        this._el('ws-enabled-help').hidden = enabled.length > 0;
    }

    _renderProviders() {
        const selected = (this._settings?.enabled_providers || []).filter(k => k !== '*');
        // Keep stored keys whose provider no longer exists listed, so saving does not silently drop them.
        const known = new Set(this._providers.map(p => p.provider_key));
        const missing = selected.filter(k => !known.has(k));
        const items = [
            ...missing.map(k => ({ value: k, label: k, group: 'Not loaded' })),
            ...this._providers.map(p => ({
                value: p.provider_key,
                label: p.provider_key,
                hint: p.provider_type,
                group: missing.length ? 'Loaded' : undefined,
            })),
        ];
        this._pickers.providers.set(items, selected);
    }

    _int(id) {
        const raw = this._el(id).value.trim();
        return raw === '' ? undefined : parseInt(raw, 10);
    }

    _collect() {
        const defaultId = this._el('ws-default-backend').value;
        const body = {
            enabled: this._el('ws-enabled').checked,
            default_backend_id: defaultId ? parseInt(defaultId, 10) : null,
            timeout_seconds: this._int('ws-timeout'),
            max_results: this._int('ws-max-results'),
            max_snippet_chars: this._int('ws-snippet-chars'),
            max_agentic_loops: this._int('ws-max-loops'),
            max_queries_per_turn: this._int('ws-max-queries'),
            apply_to: WS_SURFACES.map(s => s.value).filter(v => this._pickers.surfaces.selected.has(v)),
            enabled_providers: this._el('ws-all-providers').checked
                ? ['*']
                : [...this._pickers.providers.selected],
        };
        Object.keys(body).forEach(k => body[k] === undefined && delete body[k]);
        return body;
    }

    async save() {
        const btn = this._el('ws-save-btn');
        btn.disabled = true;
        try {
            const data = await this._send('/admin/websearch/settings', 'PUT', this._collect(), 'Settings were not saved.');
            if (!data) return;
            this._settings = data;
            await this._loadBackends();   // the default marker may have moved
            this._fill(data);
            this._renderProviders();
            this._renderBackends();
            window.UIUtils?.showToast('Settings saved.', 'success');
        } finally {
            btn.disabled = false;
        }
    }

    // ---- backends list -------------------------------------------------

    _renderBackends() {
        const rows = this._backends;
        this._el('ws-backends-table').hidden = !rows.length;
        this._el('ws-backends-empty').hidden = rows.length > 0;
        this._el('ws-backends-body').innerHTML = rows.map(b => {
            const name = wsEscape(b.name);
            const locked = b.is_default ? ' aria-disabled="true" title="Choose another default backend first"' : '';
            return `
                <tr data-id="${b.id}">
                    <td>
                        <div class="ws-backends__name">${name}</div>
                        <div class="ws-backends__meta">
                            <span class="ws-backends__url">${wsEscape(b.base_url)}</span>
                            ${b.user_selectable ? '' : '<span class="ws-backends__flag">Hidden from users</span>'}
                        </div>
                    </td>
                    <td class="d-none d-md-table-cell">${WS_TYPE_LABELS[b.type] || wsEscape(b.type)}</td>
                    <td class="text-end">${b.user_count}</td>
                    <td>
                        <div class="form-check form-switch mb-0">
                            <input class="form-check-input" type="checkbox" role="switch" data-action="toggle"
                                   aria-label="Enable ${name}" ${b.enabled ? 'checked' : ''}${locked}>
                        </div>
                    </td>
                    <td class="text-end text-nowrap">
                        <button type="button" class="btn btn-sm btn-outline-secondary" data-action="edit"
                                aria-label="Edit ${name}" title="Edit"><i class="fas fa-pen"></i></button>
                        <button type="button" class="btn btn-sm btn-outline-danger" data-action="delete"
                                aria-label="Delete ${name}"${locked || ' title="Delete"'}><i class="fas fa-trash"></i></button>
                    </td>
                </tr>`;
        }).join('');
    }

    _onRowClick(e) {
        const control = e.target.closest('[data-action]');
        if (!control) return;
        const backend = this._backends.find(b => String(b.id) === control.closest('tr').dataset.id);
        if (!backend) return;
        if (control.getAttribute('aria-disabled') === 'true') {
            // Reachable by keyboard so the reason can be given; the action itself is refused.
            e.preventDefault();
            window.UIUtils?.showToast('Choose another default backend first.', 'info');
            return;
        }
        const action = control.dataset.action;
        if (action === 'toggle') this.toggleBackend(backend, control);
        else if (action === 'edit') this.openBackend(backend, control);
        else if (action === 'delete') this.deleteBackend(backend);
    }

    _usersSentence(backend) {
        const n = backend.user_count;
        return `${n} user${n === 1 ? '' : 's'} picked ${backend.name}`;
    }

    async toggleBackend(backend, input) {
        const enabled = input.checked;
        if (!enabled && backend.user_count > 0) {
            const fallback = this._defaultBackend()?.name || 'the default';
            const ok = await window.UIUtils.showConfirmModal(
                `Turn off ${backend.name}?`,
                `${this._usersSentence(backend)}. Until you turn it back on they'll use the default, ${fallback}.`,
                'warning',
            );
            if (!ok) {
                input.checked = true;
                return;
            }
        }
        input.disabled = true;
        const data = await this._send(
            `/admin/websearch/backends/toggle?backend_id=${backend.id}`, 'PUT', { enabled },
            `${backend.name} was not ${enabled ? 'turned on' : 'turned off'}.`,
        );
        input.disabled = false;
        if (!data) {
            input.checked = !enabled;
            return;
        }
        window.UIUtils?.showToast(`${backend.name} turned ${enabled ? 'on' : 'off'}.`, 'success');
        await this._refreshBackends();
    }

    async deleteBackend(backend) {
        const fallback = this._defaultBackend()?.name || 'the default';
        const users = backend.user_count > 0
            ? ` ${this._usersSentence(backend)}. They'll use the default, ${fallback}, and their settings for this backend are removed.`
            : '';
        const ok = await window.UIUtils.showConfirmModal(`Delete ${backend.name}?`, `This can't be undone.${users}`, 'danger');
        if (!ok) return;
        const data = await this._send(`/admin/websearch/backends?backend_id=${backend.id}`, 'DELETE', undefined,
            `${backend.name} was not deleted.`);
        if (!data) return;
        window.UIUtils?.showToast(`${backend.name} deleted.`, 'success');
        await this._refreshBackends();
    }

    // Backends and the default select change together.
    async _refreshBackends() {
        await Promise.all([this._loadBackends(), this._loadSettings()]);
        this._renderBackends();
        this._renderDefaultSelect();
    }

    // ---- backend modal -------------------------------------------------

    openBackend(backend = null, trigger = null) {
        this._editing = backend;
        this._returnFocus = trigger || document.activeElement;
        const editing = backend !== null;
        const type = backend?.type || 'searxng';
        this._el('wsBackendModalTitle').textContent = editing ? `Edit ${backend.name}` : 'Add backend';
        this._el('wsb-name').value = backend?.name || '';
        this._el('wsb-url').value = backend?.base_url || '';
        this._el('wsb-enabled').checked = backend ? backend.enabled : true;
        this._el('wsb-selectable').checked = backend ? backend.user_selectable : true;
        // A backend's type is fixed: users' parameters for it only fit that type.
        this._el('wsb-type-choice').hidden = editing;
        const fixed = this._el('wsb-type-fixed');
        fixed.hidden = !editing;
        fixed.textContent = WS_TYPE_LABELS[type] || type;
        this._el('wsb-type-help').textContent = editing ? 'To use a different type, add a new backend.' : '';
        // The default must stay on: it is what everyone falls back to.
        const enabledSwitch = this._el('wsb-enabled');
        enabledSwitch.disabled = !!backend?.is_default;
        enabledSwitch.title = backend?.is_default ? 'The default backend stays on' : '';
        document.querySelectorAll('input[name="wsb-type"]').forEach(r => { r.checked = r.value === type; });
        this._setModalType(type);
        this._params.resetCatalog();
        this._params.fill(backend || {});
        bootstrap.Modal.getOrCreateInstance(this._el('wsBackendModal')).show();
        if (type === 'searxng' && backend?.base_url) this._params.loadCatalog({ quiet: true });
    }

    _setModalType(type) {
        this._modalType = type;
        this._params.setType(type);
        this._el('wsb-url').placeholder = WS_URL_PLACEHOLDER[type];
        this._el('wsb-url-help').innerHTML = WS_URL_HELP[type];
        this._el('wsb-engines-fetch-btn').hidden = type !== 'searxng';
        // A test result belongs to the backend it ran against.
        this._testRun += 1;
        this._el('wsb-test-result').innerHTML = '';
    }

    async _fetchEngines() {
        const url = this._el('wsb-url').value.trim();
        if (!WS_URL_RE.test(url)) throw new Error('Enter the base URL first.');
        const resp = await fetch('/admin/websearch/engines', {
            method: 'POST',
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_url: url }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(wsErrorMessage(data, 'Could not load the engine list.'));
        return data;
    }

    async reloadCatalog() {
        const btn = this._el('wsb-engines-fetch-btn');
        btn.disabled = true;
        try {
            await this._params.loadCatalog();
        } finally {
            btn.disabled = false;
        }
    }

    _backendBody() {
        return {
            name: this._el('wsb-name').value.trim(),
            type: this._modalType,
            base_url: this._el('wsb-url').value.trim(),
            enabled: this._el('wsb-enabled').checked,
            user_selectable: this._el('wsb-selectable').checked,
            ...this._params.collect(),
        };
    }

    _checkBackendBody(body) {
        if (!body.name) {
            this._el('wsb-name').focus();
            window.UIUtils?.showToast('Give the backend a name.', 'error');
            return false;
        }
        if (!WS_URL_RE.test(body.base_url)) {
            this._el('wsb-url').focus();
            window.UIUtils?.showToast('Enter a base URL that starts with http:// or https://.', 'error');
            return false;
        }
        return true;
    }

    async saveBackend() {
        const body = this._backendBody();
        if (!this._checkBackendBody(body)) return;
        const btn = this._el('wsb-save-btn');
        btn.disabled = true;
        try {
            const data = this._editing
                ? await this._send(`/admin/websearch/backends?backend_id=${this._editing.id}`, 'PUT', body, 'Backend was not saved.')
                : await this._send('/admin/websearch/backends', 'POST', body, 'Backend was not saved.');
            if (!data) return;
            bootstrap.Modal.getInstance(this._el('wsBackendModal'))?.hide();
            window.UIUtils?.showToast('Backend saved.', 'success');
            await this._refreshBackends();
        } finally {
            btn.disabled = false;
        }
    }

    async testBackend() {
        const body = this._backendBody();
        if (!WS_URL_RE.test(body.base_url)) {
            this._el('wsb-url').focus();
            window.UIUtils?.showToast('Enter a base URL that starts with http:// or https://.', 'error');
            return;
        }
        // The test needs a valid backend, not a finished one.
        body.name = body.name || 'Unsaved backend';
        const label = wsEscape(body.name);
        const btn = this._el('wsb-test-btn');
        const out = this._el('wsb-test-result');
        const run = this._testRun += 1;
        btn.disabled = true;
        out.innerHTML = '<p class="ws-test__status"><i class="fas fa-spinner fa-spin me-1"></i>Searching…</p>';
        try {
            const resp = await fetch('/admin/websearch/test', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: this._el('wsb-test-query').value.trim() || 'web search',
                    backend: body,
                    backend_id: this._editing?.id ?? null,
                }),
            });
            const data = await resp.json().catch(() => ({}));
            // The type was switched (or another test started) while this one ran.
            if (run !== this._testRun) return;
            if (!resp.ok) {
                out.innerHTML = `<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>${wsEscape(wsErrorMessage(data, 'Test failed.'))}</p>`;
                return;
            }
            if (!data.ok) {
                out.innerHTML = `<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>${label}: <strong>${wsEscape(data.error_code)}</strong>, ${wsEscape(data.error)} (${data.latency_ms} ms)</p>`;
                return;
            }
            const items = (data.results || []).map(r => `
                <li>
                    <a href="${wsEscape(r.url)}" target="_blank" rel="noopener noreferrer">${wsEscape(r.title)}</a>
                    <div class="ws-test__snippet">${wsEscape(r.snippet)}</div>
                </li>`).join('');
            out.innerHTML = `
                <p class="ws-test__status is-ok"><i class="fas fa-circle-check me-1"></i>Connected to ${label}: ${data.result_count} result${data.result_count === 1 ? '' : 's'} in ${data.latency_ms} ms.</p>
                ${items ? `<ol class="ws-test__results">${items}</ol>` : ''}`;
        } catch (e) {
            if (run !== this._testRun) return;
            out.innerHTML = '<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>The test could not reach the server. Check your connection and try again.</p>';
        } finally {
            btn.disabled = false;
        }
    }
}

window.WebSearchManager = new WebSearchManager();

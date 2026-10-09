function wsEscape(text) {
    return window.UIUtils?.escapeHtml ? window.UIUtils.escapeHtml(String(text ?? '')) : String(text ?? '');
}

// Searchable multi-select built on a Bootstrap dropdown. Items are
// {value, label, hint?, group?, hidden?}; a hidden item is listed only while selected.
class WsPicker {
    constructor(root, opts = {}) {
        this.root = root;
        this.opts = opts;
        this.items = [];
        this.selected = new Set();
        const id = root.id;
        const labelledBy = root.dataset.labelledby || '';
        const clearBtn = '<button type="button" class="btn btn-outline-secondary btn-sm ws-picker__clear">Clear all</button>';
        root.innerHTML = `
            <button class="form-select ws-picker__toggle" type="button" data-bs-toggle="dropdown"
                    data-bs-auto-close="outside" aria-expanded="false" aria-labelledby="${labelledBy} ${id}-summary">
                <span class="ws-picker__summary" id="${id}-summary"></span>
            </button>
            <div class="dropdown-menu ws-picker__menu">
                ${opts.searchable === false ? '' : `
                <div class="ws-picker__search">
                    <div class="ws-picker__search-field">
                        <i class="fas fa-magnifying-glass"></i>
                        <input type="search" class="form-control form-control-sm" aria-label="${wsEscape(opts.searchPlaceholder || 'Search')}"
                               placeholder="${wsEscape(opts.searchPlaceholder || 'Search')}">
                    </div>
                    ${clearBtn}
                </div>`}
                ${opts.extraHtml ? `<div class="ws-picker__extra">${opts.extraHtml}</div>` : ''}
                <div class="ws-picker__list" role="group" aria-labelledby="${labelledBy}"></div>
                <div class="ws-picker__foot">
                    <span class="ws-picker__count"></span>
                    ${opts.searchable === false ? clearBtn : ''}
                </div>
            </div>`;
        this._summary = root.querySelector('.ws-picker__summary');
        this._list = root.querySelector('.ws-picker__list');
        this._search = root.querySelector('.ws-picker__search input');
        this._count = root.querySelector('.ws-picker__count');
        this._clear = root.querySelector('.ws-picker__clear');

        this._search?.addEventListener('input', () => this.renderList());
        this._list.addEventListener('change', e => {
            const cb = e.target.closest('input[type="checkbox"]');
            if (!cb) return;
            if (cb.checked) this.selected.add(cb.value); else this.selected.delete(cb.value);
            this._emit();
        });
        this._clear.addEventListener('click', () => {
            this.selected = new Set();
            this.renderList();
            this._emit();
        });
        root.addEventListener('shown.bs.dropdown', () => this._search?.focus());
        root.addEventListener('hidden.bs.dropdown', () => {
            if (this._search && this._search.value) {
                this._search.value = '';
                this.renderList();
            }
        });
    }

    set(items, selected) {
        this.items = items;
        this.selected = new Set(selected);
        this.render();
    }

    render() {
        this.renderSummary();
        this.renderList();
    }

    _disabled() {
        return !!this.opts.isDisabled?.();
    }

    // In selection order, so a comma-separated value reads back as stored.
    _selectedItems() {
        const byValue = new Map(this.items.map(it => [it.value, it]));
        return [...this.selected].map(v => byValue.get(v)).filter(Boolean);
    }

    renderSummary() {
        const picked = this._selectedItems();
        const custom = this.opts.summarize?.(picked, this);
        let text = custom;
        if (text == null) {
            const labels = picked.map(it => it.label);
            text = labels.length === 0 ? null
                : labels.length <= 3 ? labels.join(', ')
                : `${labels.slice(0, 2).join(', ')} and ${labels.length - 2} more`;
        }
        this._summary.textContent = text ?? (this.opts.placeholder || 'None selected');
        this._summary.classList.toggle('is-placeholder', text == null);
    }

    renderList() {
        const q = (this._search?.value || '').trim().toLowerCase();
        const disabled = this._disabled();
        const visible = this.items.filter(it =>
            (!it.hidden || this.selected.has(it.value))
            && (!q || `${it.label} ${it.hint || ''}`.toLowerCase().includes(q)));
        let html = '';
        let group;
        visible.forEach(it => {
            if (it.group && it.group !== group) {
                html += `<div class="ws-picker__group">${wsEscape(it.group)}</div>`;
            }
            group = it.group;
            html += `
                <label class="ws-picker__item${disabled ? ' is-disabled' : ''}">
                    <input class="form-check-input" type="checkbox" value="${wsEscape(it.value)}"
                           ${this.selected.has(it.value) ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
                    <span class="ws-picker__label">${wsEscape(it.label)}</span>
                    ${it.hint ? `<span class="ws-picker__hint">${wsEscape(it.hint)}</span>` : ''}
                </label>`;
        });
        if (!visible.length) {
            html = `<div class="ws-picker__empty">${q
                ? `Nothing matches “${wsEscape(q)}”.`
                : wsEscape(this.opts.emptyText || 'Nothing to pick.')}</div>`;
        }
        this._list.innerHTML = html;
        this._list.classList.toggle('is-disabled', disabled);
        const n = this.selected.size;
        this._count.textContent = disabled ? '' : (n ? `${n} selected` : (this.opts.noneText || 'None selected'));
        this._clear.disabled = disabled || n === 0;
    }

    _emit() {
        this.renderSummary();
        this._count.textContent = this.selected.size ? `${this.selected.size} selected` : (this.opts.noneText || 'None selected');
        this._clear.disabled = this.selected.size === 0;
        this.opts.onChange?.(new Set(this.selected));
    }
}

const WS_SURFACES = [
    { value: 'anthropic_messages', label: 'Anthropic Messages', hint: '/messages' },
    { value: 'chat_completions', label: 'Chat Completions', hint: '/chat/completions' },
    { value: 'responses', label: 'Responses', hint: '/responses' },
];

// Same check as the server's base URL validator.
const WS_URL_RE = /^https?:\/\/[^\s/]+/;

class WebSearchManager {
    constructor() {
        this._settings = null;
        this._providers = [];
        this._engines = null;  // [{name, categories, enabled, shortcut}] from SearXNG /config
        this._categories = null;  // category names from SearXNG /config
        this._pickers = null;
        this._connOpened = false;
        this._source = 'categories';  // 'all' | 'categories' | 'engines'
        this._sourceFromSaved = true;  // re-derive once the catalog loads
        this._activeProvider = null;  // backend the form currently shows
    }

    async load() {
        this._initPickers();
        await Promise.all([this._loadSettings(), this._loadProviders()]);
        this._renderProviders();
        // Populate the category and engine pickers automatically once a URL is saved.
        if (this._settings?.provider !== 'fourget' && this._settings?.searxng_base_url && this._engines === null) {
            this.fetchEngines({ quiet: true });
        }
    }

    _initPickers() {
        if (this._pickers) return;
        const allProviders = () => this._el('ws-all-providers')?.checked;
        this._pickers = {
            surfaces: new WsPicker(this._el('ws-surfaces-picker'), {
                searchable: false,
                placeholder: 'None, interception never runs',
                summarize: picked => picked.length === WS_SURFACES.length ? 'All' : null,
                onChange: () => this._changed(),
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
                onChange: () => this._changed(),
            }),
            categories: new WsPicker(this._el('ws-categories-picker'), {
                searchPlaceholder: 'Search categories',
                placeholder: 'None selected, uses general',
                onChange: names => {
                    this._setCsv('ws-categories', names);
                    this.renderCategories();
                    this._changed();
                },
            }),
            engines: new WsPicker(this._el('ws-engines-picker'), {
                searchPlaceholder: 'Search engines or categories',
                placeholder: 'None selected',
                emptyText: 'The instance has no engines.',
                onChange: names => {
                    this._setCsv('ws-engines', names);
                    this._changed();
                },
            }),
        };
        this._el('ws-all-providers').addEventListener('change', () => {
            this._pickers.providers.render();
            this._changed();
        });
        this._el('ws-engines').addEventListener('input', () => this.renderEngines());
        this._el('ws-fourget-scraper').addEventListener('change', () => this._changed());
        document.querySelectorAll('input[name="ws-provider"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this._setProvider(radio.value);
                // load() only fetches the catalog when SearXNG was the saved backend.
                if (radio.value === 'searxng' && this._engines === null) this.fetchEngines({ quiet: true });
            });
        });
        document.querySelectorAll('input[name="ws-source"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this._sourceFromSaved = false;
                this._setSource(radio.value);
            });
        });
        // Any plain field edit refreshes the collapsed section summaries.
        const tab = this._el('websearch-tab');
        tab.addEventListener('input', () => this._changed());
        tab.addEventListener('change', () => this._changed());
    }

    _provider() {
        return document.querySelector('input[name="ws-provider"]:checked')?.value || 'searxng';
    }

    // Display name, from the backend's radio label.
    _label(provider) {
        return document.querySelector(`label[for="ws-provider-${provider}"]`)?.textContent.trim() || provider;
    }

    // The base URL input of a backend, the selected one by default.
    _urlEl(provider = this._provider()) {
        return this._el(`ws-${provider}-url`);
    }

    // The hidden backend's URL field can't be fixed from view, so an invalid
    // value there keeps the saved URL rather than failing the save.
    _urlValue(provider) {
        const value = this._urlEl(provider).value.trim() || null;
        if (provider === this._provider() || value === null || WS_URL_RE.test(value)) return value;
        return this._settings?.[`${provider}_base_url`] ?? null;
    }

    _setProvider(provider) {
        const isFourget = provider === 'fourget';
        if (provider !== this._activeProvider) {
            this._activeProvider = provider;
            // A test result belongs to the backend it ran against; don't leave it under the other one.
            this._testRun = (this._testRun || 0) + 1;
            this._el('ws-test-result').innerHTML = '';
        }
        this._el('ws-provider-searxng').checked = !isFourget;
        this._el('ws-provider-fourget').checked = isFourget;
        this._el('ws-conn-searxng').hidden = isFourget;
        this._el('ws-conn-fourget').hidden = !isFourget;
        this._el('ws-params-searxng').hidden = isFourget;
        this._el('ws-params-fourget').hidden = !isFourget;
        // Engine/category discovery is a SearXNG endpoint.
        this._el('ws-engines-fetch-btn').hidden = isFourget;
        this._el('ws-safesearch-help').innerHTML = isFourget
            ? "Sent as 4get's <code>nsfw</code> filter: off \u2192 yes, moderate \u2192 maybe, strict \u2192 no."
            : "Sent as SearXNG's <code>safesearch</code>.";
        this.updateParamsVisibility();
        this._changed();
    }

    updateParamsVisibility() {
        const hasUrl = WS_URL_RE.test(this._urlEl().value.trim());
        this._el('ws-params-card').style.display = hasUrl ? '' : 'none';
    }

    _csv(id) {
        return this._el(id).value.split(',').map(e => e.trim()).filter(Boolean);
    }

    _setCsv(id, names) {
        // Keep the user's order, append new picks at the end.
        const kept = this._csv(id).filter(n => names.has(n));
        const added = [...names].filter(n => !kept.includes(n));
        this._el(id).value = [...kept, ...added].join(',');
    }

    // Categories and engines are exclusive: SearXNG would union them.
    _deriveSource() {
        if (this._csv('ws-engines').length) return 'engines';
        const picked = new Set(this._csv('ws-categories'));
        if (this._categories?.length && this._categories.every(c => picked.has(c))) return 'all';
        return 'categories';
    }

    _setSource(source) {
        // "All" needs the instance's category list.
        if (source === 'all' && !this._categories?.length) source = 'categories';
        this._source = source;
        ['all', 'categories', 'engines'].forEach(name => {
            this._el(`ws-source-${name}`).checked = name === source;
            this._el(`ws-source-${name}-field`).hidden = name !== source;
        });
        const allRadio = this._el('ws-source-all');
        allRadio.disabled = !this._categories?.length;
        allRadio.nextElementSibling.title = allRadio.disabled ? 'Reload from SearXNG to list its categories' : '';
        if (this._categories?.length) {
            this._el('ws-source-all-note').textContent =
                `All ${this._categories.length} categories (${this._categories.join(', ')}), each with the engines the instance turns on for it.`;
        }
        this._changed();
    }

    // Once SearXNG's catalog is loaded, the pickers replace the comma-separated inputs.
    _showCatalogPickers(on) {
        ['categories', 'engines'].forEach(name => {
            this._el(`ws-${name}`).hidden = on;
            this._el(`ws-${name}-picker`).hidden = !on;
        });
    }

    renderCategories() {
        if (!this._pickers || !this._categories) return;
        const counts = {};
        (this._engines || []).forEach(e => {
            if (!e.enabled) return;
            e.categories.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
        });
        // Most useful first: categories with more default-on engines.
        const ordered = [...this._categories].sort((a, b) => (counts[b] || 0) - (counts[a] || 0) || a.localeCompare(b));
        const unknown = this._csv('ws-categories').filter(c => !this._categories.includes(c));
        const plural = n => `${n} engine${n === 1 ? '' : 's'} on by default`;
        const items = [
            ...unknown.map(c => ({ value: c, label: c, group: 'Not on this instance' })),
            ...ordered.map(c => ({ value: c, label: c, hint: plural(counts[c] || 0), group: unknown.length ? 'On this instance' : undefined })),
        ];
        this._pickers.categories.set(items, this._csv('ws-categories'));
    }

    async fetchEngines({ quiet = false } = {}) {
        const btn = this._el('ws-engines-fetch-btn');
        const baseUrl = this._el('ws-searxng-url').value.trim();
        if (!baseUrl) {
            if (!quiet) window.UIUtils?.showToast('Enter the SearXNG base URL first.', 'error');
            return;
        }
        btn.disabled = true;
        try {
            const resp = await fetch('/admin/websearch/engines', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ searxng_base_url: baseUrl }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                if (!quiet) window.UIUtils?.showToast(this._errorMessage(data, 'Failed to load engines.'), 'error');
                return;
            }
            this._engines = data.engines || [];
            this._categories = data.categories || [];
            this._showCatalogPickers(true);
            this.renderCategories();
            this.renderEngines();
            this._setSource(this._sourceFromSaved ? this._deriveSource() : this._source);
            if (!quiet) {
                window.UIUtils?.showToast(
                    `Loaded ${this._categories.length} categories and ${this._engines.length} engines from SearXNG.`, 'success');
            }
        } catch (e) {
            if (!quiet) window.UIUtils?.showToast('Network error loading engines.', 'error');
        } finally {
            btn.disabled = false;
        }
    }

    renderEngines() {
        if (!this._pickers || !this._engines) return;
        const known = new Set(this._engines.map(e => e.name));
        const unknown = this._csv('ws-engines').filter(n => !known.has(n));
        // One alphabetical list: "on by default" only matters when SearXNG picks engines for a category.
        const items = [
            ...unknown.map(n => ({ value: n, label: n, group: 'Not on this instance' })),
            ...[...this._engines].sort((a, b) => a.name.localeCompare(b.name)).map(e => ({
                value: e.name,
                label: e.name,
                hint: e.categories.join(', '),
                group: unknown.length ? 'On this instance' : undefined,
            })),
        ];
        this._pickers.engines.set(items, this._csv('ws-engines'));
        this._renderSummaries();
    }

    _el(id) {
        return document.getElementById(id);
    }

    _escape(text) {
        return wsEscape(text);
    }

    async _loadSettings() {
        try {
            const resp = await fetch('/admin/websearch/settings', { credentials: 'include' });
            if (!resp.ok) {
                window.UIUtils?.showToast('Failed to load web search settings.', 'error');
                return;
            }
            this._settings = await resp.json();
            this._fill(this._settings);
        } catch (e) {
            console.error('WebSearchManager: failed to load settings', e);
        }
    }

    async _loadProviders() {
        try {
            const resp = await fetch('/admin/websearch/providers', { credentials: 'include' });
            if (!resp.ok) return;
            const data = await resp.json();
            this._providers = data.providers || [];
        } catch (e) {
            console.error('WebSearchManager: failed to load providers', e);
        }
    }

    _fill(s) {
        this._el('ws-enabled').checked = !!s.enabled;
        this._el('ws-searxng-url').value = s.searxng_base_url || '';
        this._el('ws-fourget-url').value = s.fourget_base_url || '';
        this._el('ws-fourget-lang').value = s.fourget_lang || '';
        this._el('ws-fourget-country').value = s.fourget_country || '';
        this._setScraper(s.fourget_scraper);
        // Must come after both URLs are set: it decides which one to test.
        this._setProvider(s.provider || 'searxng');
        this._el('ws-timeout').value = s.timeout_seconds;
        this._el('ws-engines').value = s.engines || '';
        this._el('ws-categories').value = s.categories || '';
        this.renderCategories();
        this.renderEngines();
        this._sourceFromSaved = true;
        this._setSource(this._deriveSource());
        this._el('ws-language').value = s.language || '';
        this._el('ws-safesearch').value = String(s.safesearch);
        this._el('ws-time-range').value = s.time_range || '';
        this._el('ws-max-results').value = s.max_results;
        this._el('ws-snippet-chars').value = s.max_snippet_chars;
        this._el('ws-max-loops').value = s.max_agentic_loops;
        this._el('ws-max-queries').value = s.max_queries_per_turn;
        this._pickers.surfaces.set(WS_SURFACES, s.apply_to || []);
        this._el('ws-all-providers').checked = (s.enabled_providers || []).includes('*');
        // With nothing to connect to yet, start with the connection section open.
        const savedUrl = s.provider === 'fourget' ? s.fourget_base_url : s.searxng_base_url;
        if (!this._connOpened && !savedUrl && window.bootstrap) {
            bootstrap.Collapse.getOrCreateInstance(this._el('ws-conn-body'), { toggle: false }).show();
        }
        this._connOpened = true;
        const updated = this._el('ws-updated');
        updated.textContent = s.updated_at
            ? `Last saved ${new Date(s.updated_at + (s.updated_at.endsWith('Z') ? '' : 'Z')).toLocaleString()}${s.updated_by ? ' by ' + s.updated_by : ''}.`
            : 'Not saved yet.';
    }

    // A stored scraper the list no longer offers stays selected and visible (the
    // server then rejects it on save) instead of the select going blank and
    // silently saving the default.
    _setScraper(name) {
        const select = this._el('ws-fourget-scraper');
        select.querySelectorAll('option[data-unsupported]').forEach(o => o.remove());
        if (name && ![...select.options].some(o => o.value === name)) {
            const option = new Option(`${name} (not supported)`, name);
            option.dataset.unsupported = '';
            select.add(option);
        }
        select.value = name || select.options[0]?.value || '';
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
        this._renderSummaries();
    }

    _int(id) {
        const raw = this._el(id).value.trim();
        return raw === '' ? undefined : parseInt(raw, 10);
    }

    _collect() {
        const body = {
            enabled: this._el('ws-enabled').checked,
            provider: this._provider(),
            // Both are always sent, so switching backends never drops the other's setup.
            searxng_base_url: this._urlValue('searxng'),
            fourget_base_url: this._urlValue('fourget'),
            fourget_scraper: this._el('ws-fourget-scraper').value,
            fourget_lang: this._el('ws-fourget-lang').value.trim() || null,
            fourget_country: this._el('ws-fourget-country').value.trim() || null,
            engines: this._source === 'engines' ? this._el('ws-engines').value.trim() || null : null,
            categories: this._source === 'all' ? this._categories.join(',')
                : this._source === 'categories' ? this._el('ws-categories').value.trim() || null
                : null,
            language: this._el('ws-language').value.trim() || null,
            safesearch: parseInt(this._el('ws-safesearch').value, 10),
            time_range: this._el('ws-time-range').value || null,
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

    // Keeps the collapsed section headers in step with the fields.
    _changed() {
        if (!this._pickers) return;
        this._renderSummaries();
    }

    _host(url) {
        try { return new URL(url).host; } catch (e) { return url; }
    }

    _renderSummaries() {
        if (!this._pickers) return;
        const s = this._collect();
        const isFourget = s.provider === 'fourget';
        const url = isFourget ? s.fourget_base_url : s.searxng_base_url;
        this._el('ws-conn-summary').textContent =
            `${this._label(s.provider)} \u00b7 ${url ? this._host(url) : 'Not set'}`;
        if (isFourget) {
            this._el('ws-params-summary').textContent = `Scraper: ${s.fourget_scraper}`;
            return;
        }
        const engines = (s.engines || '').split(',').filter(Boolean);
        this._el('ws-params-summary').textContent =
            this._source === 'all' ? 'All categories'
            : this._source === 'engines'
                ? (engines.length ? `${engines.length} engine${engines.length === 1 ? '' : 's'}` : 'No engines, general category')
            : (s.categories || 'general').split(',').join(', ');
    }

    _errorMessage(err, fallback) {
        if (Array.isArray(err.detail)) {
            return err.detail.map(d => (d.msg || '').replace(/^Value error, /, '')).filter(Boolean).join('; ') || fallback;
        }
        return err.detail || fallback;
    }

    async save() {
        const btn = this._el('ws-save-btn');
        btn.disabled = true;
        try {
            const resp = await fetch('/admin/websearch/settings', {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this._collect()),
            });
            if (resp.ok) {
                this._settings = await resp.json();
                this._fill(this._settings);
                this._renderProviders();
                window.UIUtils?.showToast('Web search settings saved.', 'success');
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(this._errorMessage(err, 'Failed to save web search settings.'), 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving web search settings.', 'error');
        } finally {
            btn.disabled = false;
        }
    }

    async test() {
        const btn = this._el('ws-test-btn');
        const out = this._el('ws-test-result');
        const settings = this._collect();
        const label = this._label(settings.provider);
        const run = this._testRun = (this._testRun || 0) + 1;
        // The test should not fail validation just because interception is off.
        settings.enabled = false;
        btn.disabled = true;
        out.innerHTML = '<p class="ws-test__status"><i class="fas fa-spinner fa-spin me-1"></i>Searching…</p>';
        try {
            const resp = await fetch('/admin/websearch/test', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: this._el('ws-test-query').value.trim() || 'web search', settings }),
            });
            const data = await resp.json().catch(() => ({}));
            // The backend was switched (or another test started) while this one ran.
            if (run !== this._testRun) return;
            if (!resp.ok) {
                out.innerHTML = `<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>${this._escape(this._errorMessage(data, 'Test failed.'))}</p>`;
                return;
            }
            if (!data.ok) {
                out.innerHTML = `<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>${label} \u00b7 <strong>${this._escape(data.error_code)}</strong>: ${this._escape(data.error)} (${data.latency_ms} ms)</p>`;
                return;
            }
            const items = (data.results || []).map(r => `
                <li>
                    <a href="${this._escape(r.url)}" target="_blank" rel="noopener noreferrer">${this._escape(r.title)}</a>
                    <div class="ws-test__snippet">${this._escape(r.snippet)}</div>
                </li>`).join('');
            out.innerHTML = `
                <p class="ws-test__status is-ok"><i class="fas fa-circle-check me-1"></i>Connected to ${label}: ${data.result_count} result${data.result_count === 1 ? '' : 's'} in ${data.latency_ms} ms.</p>
                ${items ? `<ol class="ws-test__results">${items}</ol>` : ''}`;
        } catch (e) {
            if (run !== this._testRun) return;
            out.innerHTML = '<p class="ws-test__status is-error"><i class="fas fa-circle-xmark me-1"></i>Network error running the test.</p>';
        } finally {
            btn.disabled = false;
        }
    }
}

window.WebSearchManager = new WebSearchManager();

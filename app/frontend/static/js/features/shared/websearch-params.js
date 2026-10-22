// Web search pieces shared by the admin backend form and the user's Web Search tab.

function wsEscape(text) {
    return window.UIUtils?.escapeHtml ? window.UIUtils.escapeHtml(String(text ?? '')) : String(text ?? '');
}

// Same check as the server's base URL validator.
const WS_URL_RE = /^https?:\/\/[^\s/]+/;

const WS_SAFESEARCH_LABELS = { 0: 'Off', 1: 'Moderate', 2: 'Strict' };
const WS_TIME_RANGE_LABELS = { '': 'Any time', day: 'Past day', week: 'Past week', month: 'Past month', year: 'Past year' };
const WS_TYPE_LABELS = { searxng: 'SearXNG', fourget: '4get' };
// The parameters each backend type reads (mirrors WEBSEARCH_BACKEND_PARAMS).
const WS_TYPE_PARAMS = {
    searxng: ['engines', 'categories', 'language', 'safesearch', 'time_range'],
    fourget: ['fourget_scraper', 'fourget_lang', 'fourget_country', 'safesearch', 'time_range'],
};
const WS_PARAM_FIELDS = [...new Set(Object.values(WS_TYPE_PARAMS).flat())];

function wsErrorMessage(err, fallback) {
    if (Array.isArray(err?.detail)) {
        return err.detail.map(d => (d.msg || '').replace(/^Value error, /, '')).filter(Boolean).join('; ') || fallback;
    }
    if (err?.detail && typeof err.detail === 'object') return err.detail.message || fallback;
    return err?.detail || fallback;
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
                    data-bs-auto-close="outside" aria-expanded="false" aria-labelledby="${labelledBy} ${id}-summary"
                    ${opts.fixed ? `data-bs-popper-config='{"strategy":"fixed"}'` : ''}>
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

// The search parameters of one backend: SearXNG's source, categories, engines and
// language, or 4get's scraper, language and country, plus safe search and time range.
//
// With `inherit`, every field may be left unset (collects as null) and shows the
// value it inherits instead: selects get a "… (default)" first option, text fields
// the inherited value as placeholder and help text, and "Search with" a Default
// segment. Engines and categories are one choice, as are 4get's scraper and its
// language/country: the server takes each group whole from the user.
class WsParamsForm {
    constructor(container, opts = {}) {
        this.root = container;
        this.opts = opts;
        this.p = opts.idPrefix;
        this.type = 'searxng';
        this.inherited = {};       // effective values a field falls back to (inherit mode)
        this._engines = null;      // [{name, categories, enabled}] from SearXNG /config
        this._categories = null;   // category names from SearXNG /config
        this._source = 'categories';
        this._sourceFromSaved = true;
        this._render();
        this._initPickers();
    }

    _el(name) {
        return document.getElementById(`${this.p}-${name}`);
    }

    _render() {
        const p = this.p;
        const inherit = !!this.opts.inherit;
        const scrapers = (this.opts.scrapers || []).map(n => `<option value="${wsEscape(n)}">${wsEscape(n)}</option>`).join('');
        const segment = (value, label) => `
            <input type="radio" class="btn-check" name="${p}-source" id="${p}-source-${value}" value="${value}" autocomplete="off">
            <label class="btn btn-outline-secondary" for="${p}-source-${value}">${label}</label>`;
        this.root.innerHTML = `
            <div class="ws-params">
                <div data-ws-type="searxng">
                    <div class="row g-3">
                        <div class="col-lg-5 col-md-6">
                            <label class="form-label" id="${p}-source-label">Search with</label>
                            <div class="btn-group ws-segmented" role="radiogroup" aria-labelledby="${p}-source-label">
                                ${inherit ? segment('default', 'Default') : ''}
                                ${segment('all', 'All')}
                                ${segment('categories', 'Categories')}
                                ${segment('engines', 'Engines')}
                            </div>
                        </div>
                        <div class="col-lg-7 col-md-6">
                            <div id="${p}-source-default-field">
                                <label class="form-label">Covers</label>
                                <p class="ws-source-note" id="${p}-source-default-note"></p>
                            </div>
                            <div id="${p}-source-all-field">
                                <label class="form-label">Covers</label>
                                <p class="ws-source-note" id="${p}-source-all-note">Every category on the instance, using the engines it turns on by default.</p>
                            </div>
                            <div id="${p}-source-categories-field">
                                <label class="form-label" id="${p}-categories-label" for="${p}-categories">Categories</label>
                                <input type="text" id="${p}-categories" class="form-control" placeholder="general (comma-separated)">
                                <div class="dropdown ws-picker" id="${p}-categories-picker" data-labelledby="${p}-categories-label" hidden></div>
                                <div class="form-text">Each category searches the engines the instance turns on for it.</div>
                            </div>
                            <div id="${p}-source-engines-field">
                                <label class="form-label" id="${p}-engines-label" for="${p}-engines">Engines</label>
                                <input type="text" id="${p}-engines" class="form-control" placeholder="Comma-separated, e.g. bing,duckduckgo">
                                <div class="dropdown ws-picker" id="${p}-engines-picker" data-labelledby="${p}-engines-label" hidden></div>
                                <div class="form-text">Only these engines are searched. With none picked, searches use the general category.</div>
                            </div>
                        </div>
                    </div>
                    <div class="row g-3 mt-0">
                        <div class="col-lg-5 col-md-6">
                            <label class="form-label" for="${p}-language">Language</label>
                            <input type="text" id="${p}-language" class="form-control" placeholder="auto">
                            <div class="form-text" id="${p}-language-help">SearXNG codes, e.g. <code>auto</code> or <code>en-US</code>.</div>
                        </div>
                    </div>
                </div>

                <div data-ws-type="fourget">
                    <div class="row g-3">
                        <div class="col-lg-4 col-md-6">
                            <label class="form-label" for="${p}-fourget-scraper">Scraper</label>
                            <select id="${p}-fourget-scraper" class="form-select">
                                ${inherit ? '<option value="" data-default></option>' : ''}${scrapers}
                            </select>
                            <div class="form-text">The upstream engine 4get scrapes for each query. Some need credentials or are blocked from a datacenter IP.</div>
                        </div>
                        <div class="col-lg-4 col-md-3 col-6">
                            <label class="form-label" for="${p}-fourget-lang">Language</label>
                            <input type="text" id="${p}-fourget-lang" class="form-control" placeholder="any">
                        </div>
                        <div class="col-lg-4 col-md-3 col-6">
                            <label class="form-label" for="${p}-fourget-country">Country</label>
                            <input type="text" id="${p}-fourget-country" class="form-control" placeholder="any">
                            <div class="form-text" id="${p}-fourget-locale-help">Scraper-specific; ddg uses locales like <code>us-en</code>.</div>
                        </div>
                    </div>
                </div>

                <div class="ws-subhead">Results</div>
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label" for="${p}-safesearch">Safe search</label>
                        <select id="${p}-safesearch" class="form-select">
                            ${inherit ? '<option value="" data-default></option>' : ''}
                            ${Object.entries(WS_SAFESEARCH_LABELS).map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}
                        </select>
                        <div class="form-text" id="${p}-safesearch-help"></div>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label" for="${p}-time-range">Time range</label>
                        <select id="${p}-time-range" class="form-select">
                            ${inherit ? '<option value="" data-default></option>' : ''}
                            ${Object.entries(WS_TIME_RANGE_LABELS).filter(([v]) => !inherit || v)
                                .map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}
                        </select>
                    </div>
                </div>
            </div>`;
    }

    _initPickers() {
        const fixed = !!this.opts.fixedMenus;
        this._pickers = {
            categories: new WsPicker(this._el('categories-picker'), {
                fixed,
                searchPlaceholder: 'Search categories',
                placeholder: 'None selected, uses general',
                onChange: names => {
                    this._setCsv('categories', names);
                    this.renderCategories();
                    this._changed();
                },
            }),
            engines: new WsPicker(this._el('engines-picker'), {
                fixed,
                searchPlaceholder: 'Search engines or categories',
                placeholder: 'None selected',
                emptyText: 'The instance has no engines.',
                onChange: names => {
                    this._setCsv('engines', names);
                    this._changed();
                },
            }),
        };
        this._el('engines').addEventListener('input', () => this.renderEngines());
        this._el('categories').addEventListener('input', () => this.renderCategories());
        this._el('fourget-scraper').addEventListener('change', () => this._syncLocaleFields());
        this.root.querySelectorAll(`input[name="${this.p}-source"]`).forEach(radio => {
            radio.addEventListener('change', () => {
                this._sourceFromSaved = false;
                this._setSource(radio.value);
            });
        });
        this.root.addEventListener('input', () => this._changed());
        this.root.addEventListener('change', () => this._changed());
    }

    _changed() {
        this.opts.onChange?.();
    }

    setType(type) {
        this.type = type;
        this.root.querySelectorAll('[data-ws-type]').forEach(el => { el.hidden = el.dataset.wsType !== type; });
        const fourget = type === 'fourget';
        this._el('safesearch-help').innerHTML = fourget
            ? "Sent as 4get's <code>nsfw</code> filter: off → yes, moderate → maybe, strict → no."
            : "Sent as SearXNG's <code>safesearch</code>.";
    }

    // Fill from stored values. In inherit mode `inherited` holds the backend's
    // effective parameters, shown wherever a value is left unset.
    fill(values, inherited = {}) {
        values = values || {};
        this.inherited = inherited;
        if (this.opts.inherit) this._fillInheritLabels();
        this._el('engines').value = values.engines || '';
        this._el('categories').value = values.categories || '';
        this._el('language').value = values.language || '';
        this._setScraper(values.fourget_scraper);
        this._el('fourget-lang').value = values.fourget_lang || '';
        this._el('fourget-country').value = values.fourget_country || '';
        const safesearch = values.safesearch ?? (this.opts.inherit ? '' : 1);
        this._el('safesearch').value = String(safesearch);
        this._el('time-range').value = values.time_range || '';
        this.renderCategories();
        this.renderEngines();
        this._sourceFromSaved = true;
        this._setSource(this._deriveSource());
        this._syncLocaleFields();
    }

    _fillInheritLabels() {
        const inh = this.inherited;
        const defaultOption = (name, label) => {
            const option = this._el(name).querySelector('option[data-default]');
            if (option) option.textContent = `${label} (default)`;
        };
        defaultOption('safesearch', WS_SAFESEARCH_LABELS[inh.safesearch ?? 1] || 'Moderate');
        defaultOption('time-range', WS_TIME_RANGE_LABELS[inh.time_range || '']);
        defaultOption('fourget-scraper', inh.fourget_scraper || 'instance’s own');
        this._el('language').placeholder = inh.language || 'auto';
        this._el('language-help').innerHTML =
            `Leave empty to use the default: <code>${wsEscape(inh.language || 'auto')}</code>.`;
        const source = inh.engines ? `engines ${inh.engines.split(',').join(', ')}`
            : `categories ${(inh.categories || 'general').split(',').join(', ')}`;
        this._el('source-default-note').textContent = `Uses your administrator's choice: ${source}.`;
    }

    // 4get's language and country codes belong to the scraper, so while the
    // scraper is inherited they are too.
    _syncLocaleFields() {
        if (!this.opts.inherit) return;
        const inheritScraper = this._el('fourget-scraper').value === '';
        ['lang', 'country'].forEach(name => {
            const input = this._el(`fourget-${name}`);
            input.disabled = inheritScraper;
            if (inheritScraper) input.value = '';
            input.placeholder = inheritScraper ? (this.inherited[`fourget_${name}`] || 'any') : 'any';
        });
        this._el('fourget-locale-help').innerHTML = inheritScraper
            ? 'Follows the default scraper. Pick a scraper to set your own.'
            : 'Scraper-specific; ddg uses locales like <code>us-en</code>.';
    }

    // A stored scraper the list no longer offers stays selected and visible (the
    // server then rejects it on save) instead of the select silently changing.
    _setScraper(name) {
        const select = this._el('fourget-scraper');
        select.querySelectorAll('option[data-unsupported]').forEach(o => o.remove());
        if (name && ![...select.options].some(o => o.value === name)) {
            const option = new Option(`${name} (not supported)`, name);
            option.dataset.unsupported = '';
            select.add(option);
        }
        select.value = name || select.options[0]?.value || '';
    }

    _csv(name) {
        return this._el(name).value.split(',').map(e => e.trim()).filter(Boolean);
    }

    _setCsv(name, names) {
        // Keep the user's order, append new picks at the end.
        const kept = this._csv(name).filter(n => names.has(n));
        const added = [...names].filter(n => !kept.includes(n));
        this._el(name).value = [...kept, ...added].join(',');
    }

    // Categories and engines are exclusive: SearXNG would union them.
    _deriveSource() {
        if (this._csv('engines').length) return 'engines';
        const picked = this._csv('categories');
        if (this.opts.inherit && !picked.length) return 'default';
        if (this._categories?.length && this._categories.every(c => picked.includes(c))) return 'all';
        return 'categories';
    }

    _setSource(source) {
        // "All" needs the instance's category list.
        if (source === 'all' && !this._categories?.length) source = 'categories';
        this._source = source;
        ['default', 'all', 'categories', 'engines'].forEach(name => {
            const radio = this._el(`source-${name}`);
            if (radio) radio.checked = name === source;
            const field = this._el(`source-${name}-field`);
            if (field) field.hidden = name !== source;
        });
        const allRadio = this._el('source-all');
        allRadio.disabled = !this._categories?.length;
        allRadio.nextElementSibling.title = allRadio.disabled ? 'Load the instance’s categories to use All' : '';
        if (this._categories?.length) {
            this._el('source-all-note').textContent =
                `All ${this._categories.length} categories (${this._categories.join(', ')}), each with the engines the instance turns on for it.`;
        }
        this._changed();
    }

    // Once the catalog is loaded, the pickers replace the comma-separated inputs.
    _showCatalogPickers(on) {
        ['categories', 'engines'].forEach(name => {
            this._el(name).hidden = on;
            this._el(`${name}-picker`).hidden = !on;
        });
    }

    resetCatalog() {
        this._engines = null;
        this._categories = null;
        this._showCatalogPickers(false);
        this._setSource(this._source);
    }

    hasCatalog() {
        return this._engines !== null;
    }

    // `opts.fetchEngines()` resolves to SearXNG's {engines, categories}.
    async loadCatalog({ quiet = false } = {}) {
        if (this.type !== 'searxng' || !this.opts.fetchEngines) return false;
        try {
            const data = await this.opts.fetchEngines();
            this._engines = data.engines || [];
            this._categories = data.categories || [];
        } catch (e) {
            if (!quiet) window.UIUtils?.showToast(e.message || 'Could not load the engine list.', 'error');
            return false;
        }
        this._showCatalogPickers(true);
        this.renderCategories();
        this.renderEngines();
        this._setSource(this._sourceFromSaved ? this._deriveSource() : this._source);
        if (!quiet) {
            window.UIUtils?.showToast(
                `Loaded ${this._categories.length} categories and ${this._engines.length} engines.`, 'success');
        }
        return true;
    }

    renderCategories() {
        if (!this._categories) return;
        const counts = {};
        (this._engines || []).forEach(e => {
            if (!e.enabled) return;
            e.categories.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
        });
        // Most useful first: categories with more default-on engines.
        const ordered = [...this._categories].sort((a, b) => (counts[b] || 0) - (counts[a] || 0) || a.localeCompare(b));
        const unknown = this._csv('categories').filter(c => !this._categories.includes(c));
        const plural = n => `${n} engine${n === 1 ? '' : 's'} on by default`;
        const items = [
            ...unknown.map(c => ({ value: c, label: c, group: 'Not on this instance' })),
            ...ordered.map(c => ({ value: c, label: c, hint: plural(counts[c] || 0), group: unknown.length ? 'On this instance' : undefined })),
        ];
        this._pickers.categories.set(items, this._csv('categories'));
    }

    renderEngines() {
        if (!this._engines) return;
        const known = new Set(this._engines.map(e => e.name));
        const unknown = this._csv('engines').filter(n => !known.has(n));
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
        this._pickers.engines.set(items, this._csv('engines'));
    }

    // The current type's parameters; every other parameter is null.
    collect() {
        const out = Object.fromEntries(WS_PARAM_FIELDS.map(f => [f, null]));
        const text = name => this._el(name).value.trim() || null;
        const safesearch = this._el('safesearch').value;
        out.safesearch = safesearch === '' ? null : parseInt(safesearch, 10);
        out.time_range = this._el('time-range').value || null;
        if (this.type === 'fourget') {
            out.fourget_scraper = this._el('fourget-scraper').value || null;
            out.fourget_lang = text('fourget-lang');
            out.fourget_country = text('fourget-country');
        } else {
            out.language = text('language');
            out.engines = this._source === 'engines' ? text('engines') : null;
            out.categories = this._source === 'all' ? this._categories.join(',')
                : this._source === 'categories' ? text('categories')
                : null;
        }
        return out;
    }
}

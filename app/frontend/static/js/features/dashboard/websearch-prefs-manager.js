// User dashboard Web Search tab: the backend searches go to, result limits and
// search parameters. Anything left empty follows the administrator's defaults.
// WsParamsForm and the WS_* helpers come from features/shared/websearch-params.js.

class WebSearchPrefsManager {
    constructor() {
        this._data = null;      // GET /auth/websearch
        this._params = null;
        this._loaded = false;
    }

    _el(id) {
        return document.getElementById(id);
    }

    _init() {
        if (this._params) return;
        let scrapers = [];
        try { scrapers = JSON.parse(this._el('wsu-fourget-scrapers').textContent); } catch (e) { /* none */ }
        this._params = new WsParamsForm(this._el('wsu-params'), {
            idPrefix: 'wsu',
            inherit: true,
            scrapers,
            fetchEngines: () => this._fetchEngines(),
        });
        this._el('wsu-backend').addEventListener('change', () => this._showBackend());
        this._el('wsu-save-btn').addEventListener('click', () => this.save());
        this._el('wsu-reset-btn').addEventListener('click', () => this.reset());
    }

    async load() {
        this._init();
        this._el('wsu-loading').hidden = false;
        this._el('wsu-unavailable').hidden = true;
        try {
            const resp = await makeAuthenticatedRequest('/auth/websearch');
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                this._showUnavailable(resp.status === 403
                    ? wsErrorMessage(data, 'Web search settings are not available for this account.')
                    : 'Your web search settings did not load. Reload the page to try again.');
                return;
            }
            this._data = data;
            this._loaded = true;
            this._fill();
        } catch (e) {
            this._showUnavailable('Your web search settings did not load. Check your connection and reload the page.');
        } finally {
            this._el('wsu-loading').hidden = true;
        }
    }

    _showUnavailable(message) {
        const box = this._el('wsu-unavailable');
        box.textContent = message;
        box.hidden = false;
        this._el('wsu-form').hidden = true;
    }

    _defaultBackend() {
        return this._data.backends.find(b => b.is_default) || null;
    }

    // The backend the form shows: the one picked in the select, else the default
    // (also while the pick is unavailable, as searches use the default then).
    _selectedBackend() {
        const id = this._el('wsu-backend').value;
        return (id && this._data.backends.find(b => String(b.id) === id)) || this._defaultBackend();
    }

    _fill() {
        const d = this._data;
        this._el('wsu-form').hidden = false;
        if (!this._defaultBackend()) {
            this._showUnavailable('Web search has no backend yet. Your administrator needs to add one before you can choose settings.');
            return;
        }
        const def = this._defaultBackend();
        const others = d.backends.filter(b => !b.is_default);
        const select = this._el('wsu-backend');
        const chosen = d.selection.backend_id;
        // A pick the list cannot show (it became the default, or is unavailable)
        // gets its own option, so saving other fields does not drop it.
        let kept = '';
        if (chosen != null && !others.some(b => b.id === chosen)) {
            const label = chosen === def.id
                ? `${wsEscape(def.name)} (kept if the default changes)`
                : `${wsEscape(d.effective.chosen_backend_name || 'Your pick')} (unavailable)`;
            kept = `<option value="${chosen}">${label}</option>`;
        }
        select.innerHTML = `<option value="">${wsEscape(def.name)} (default)</option>` + kept
            + others.map(b => `<option value="${b.id}">${wsEscape(b.name)}</option>`).join('');
        select.value = chosen == null ? '' : String(chosen);

        [['wsu-max-results', 'max_results', ''], ['wsu-snippet-chars', 'max_snippet_chars', ' characters']].forEach(([id, key, unit]) => {
            const input = this._el(id);
            const { min, max } = d.limits[key];
            input.min = min;
            input.max = max;
            input.value = d.selection[key] ?? '';
            input.placeholder = d.defaults[key];
            this._el(`${id}-help`).textContent =
                `${min}–${max}. Leave empty to use the default: ${d.defaults[key]}${unit}.`;
        });
        this._showBackend();
        this._renderCallout();
    }

    // Parameters are kept per backend, so switching shows the saved ones for it.
    _showBackend() {
        const backend = this._selectedBackend();
        this._el('wsu-params-title').textContent = `Search parameters for ${backend.name}`;
        this._params.setType(backend.type);
        this._params.resetCatalog();
        this._params.fill(this._data.selection.params_by_backend[String(backend.id)] || {}, backend.params);
        if (backend.type === 'searxng') this._params.loadCatalog({ quiet: true });
    }

    _renderCallout() {
        const d = this._data;
        const box = this._el('wsu-callout');
        let html = '';
        if (!d.enabled) {
            html = '<i class="fas fa-circle-info" aria-hidden="true"></i>Web search is turned off by your administrator. '
                + 'Your settings are kept and apply when it’s turned back on.';
        } else if (d.effective.fell_back) {
            html = '<i class="fas fa-triangle-exclamation" aria-hidden="true"></i>The backend you picked isn’t available right now, '
                + `so your searches use the default, ${wsEscape(this._defaultBackend().name)}. Your pick is kept and applies again `
                + 'if it comes back. To change it, pick another backend or the default.';
        }
        box.innerHTML = html;
        box.hidden = !html;
    }

    _int(id) {
        const raw = this._el(id).value.trim();
        return raw === '' ? null : parseInt(raw, 10);
    }

    async _fetchEngines() {
        const backend = this._selectedBackend();
        const resp = await makeAuthenticatedRequest(`/auth/websearch/engines?backend_id=${backend.id}`);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(wsErrorMessage(data, 'The engine list did not load.'));
        return data;
    }

    _collect() {
        const id = this._el('wsu-backend').value;
        return {
            backend_id: id ? parseInt(id, 10) : null,
            max_results: this._int('wsu-max-results'),
            max_snippet_chars: this._int('wsu-snippet-chars'),
            ...this._params.collect(),
        };
    }

    async _send(method, body, failure) {
        try {
            const resp = await makeAuthenticatedRequest('/auth/websearch', {
                method,
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

    async save() {
        const btn = this._el('wsu-save-btn');
        btn.disabled = true;
        try {
            const data = await this._send('PUT', this._collect(), 'Settings were not saved.');
            if (!data) return;
            this._data = data;
            this._fill();
            window.UIUtils?.showToast('Settings saved.', 'success');
        } finally {
            btn.disabled = false;
        }
    }

    async reset() {
        const ok = await window.UIUtils.showConfirmModal(
            'Reset your web search settings?',
            'You’ll use the administrator’s defaults for every backend.',
            'warning',
        );
        if (!ok) return;
        const btn = this._el('wsu-reset-btn');
        btn.disabled = true;
        try {
            const data = await this._send('DELETE', undefined, 'Settings were not reset.');
            if (!data) return;
            this._data = data;
            this._fill();
            window.UIUtils?.showToast('Settings reset.', 'success');
        } finally {
            btn.disabled = false;
        }
    }
}

window.WebSearchPrefsManager = new WebSearchPrefsManager();

// Load when the tab is first opened (same pattern as quotas-manager.js).
document.addEventListener('DOMContentLoaded', function () {
    const originalShowTab = window.showTab;
    window.showTab = function (tabName) {
        originalShowTab(tabName);
        if (tabName === 'websearch' && !window.WebSearchPrefsManager._loaded) {
            window.WebSearchPrefsManager.load();
        }
    };
});

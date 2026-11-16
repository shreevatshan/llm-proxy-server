/**
 * Provider settings dialog (#providerSettingsModal), opened from the gear in each
 * provider's header in the Models view.
 *
 * Every switch in the dialog with data-setting="<field>" is read from
 * GET /admin/providers/settings and saved, as soon as it changes, with
 * PUT /admin/providers/settings {<field>: value}. A new on/off setting therefore
 * needs only its .policy-row in the template and its field on the server.
 */
class ProviderSettingsManager {
    constructor() {
        this.providerKey = null;
        this._bound = false;
    }

    _modal() {
        return document.getElementById('providerSettingsModal');
    }

    _inputs() {
        return [...(this._modal()?.querySelectorAll('[data-setting]') || [])];
    }

    _bind() {
        if (this._bound) return;
        this._inputs().forEach(input => input.addEventListener('change', () => this.save(input)));
        this._bound = true;
    }

    /** The On/Off text beside a switch, so the state never rests on its fill alone.
     *  Blank while the value is unknown (loading, or the load failed). */
    _renderState(input, known = true) {
        const state = input.closest('.policy-row')?.querySelector('.policy-row-state');
        if (!state) return;
        state.textContent = !known ? '' : input.checked ? (state.dataset.on || 'On') : (state.dataset.off || 'Off');
    }

    _showError(message) {
        const box = document.getElementById('provider-settings-error');
        if (!box) return;
        box.textContent = message || '';
        box.hidden = !message;
    }

    async open(providerKey) {
        const modal = this._modal();
        if (!modal) return;
        this._bind();
        this.providerKey = providerKey;
        document.getElementById('provider-settings-key').textContent = providerKey;
        this._showError('');
        // Until the provider's values arrive, show no value at all: a switch left at
        // its default (or the previous provider's value) would read as this one's.
        const inputs = this._inputs();
        inputs.forEach(input => {
            input.disabled = true;
            input.checked = false;
            this._renderState(input, false);
        });
        bootstrap.Modal.getOrCreateInstance(modal).show();

        try {
            const response = await fetch(
                `/admin/providers/settings?provider_key=${encodeURIComponent(providerKey)}`,
                { credentials: 'include' },
            );
            const settings = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(settings.detail || `Request failed (${response.status})`);
            if (this.providerKey !== providerKey) return;   // another provider was opened meanwhile
            inputs.forEach(input => {
                input.checked = !!settings[input.dataset.setting];
                input.disabled = false;
                this._renderState(input);
            });
        } catch (error) {
            if (this.providerKey !== providerKey) return;
            // Shown inside the dialog rather than by closing it: Bootstrap ignores
            // hide() while the dialog is still fading in, which a fast error beats.
            this._showError(`Could not load the settings of ${providerKey}: ${error.message}. ` +
                'Nothing was changed. Close this dialog and try again.');
        }
    }

    async save(input) {
        const providerKey = this.providerKey;
        const field = input.dataset.setting;
        const value = input.checked;
        const title = input.closest('.policy-row')?.querySelector('.policy-row-title')?.textContent.trim() || field;
        this._renderState(input);
        input.disabled = true;
        try {
            const response = await fetch(
                `/admin/providers/settings?provider_key=${encodeURIComponent(providerKey)}`,
                {
                    method: 'PUT',
                    credentials: 'include',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ [field]: value }),
                },
            );
            const settings = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(settings.detail || `Request failed (${response.status})`);
            window.ModelsView?.applyProviderSettings(providerKey, settings);
            window.UIUtils.showToast(
                window.UIUtils.escapeHtml(`${title}: ${value ? 'on' : 'off'} for ${providerKey}`), 'success',
            );
        } catch (error) {
            input.checked = !value;
            this._renderState(input);
            window.UIUtils.showToast(window.UIUtils.escapeHtml(`Could not save "${title}": ${error.message}`), 'error');
        } finally {
            input.disabled = false;
        }
    }
}

window.ProviderSettings = new ProviderSettingsManager();

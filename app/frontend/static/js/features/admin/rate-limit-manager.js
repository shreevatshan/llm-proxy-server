/**
 * Rate Limits tab. Every limit belongs to a user profile: the profile switcher at the
 * top picks whose limits the tab edits -- the profile-wide concurrency cap, then the
 * table with one row per model group, Default first.
 * Groups themselves are managed in the Model Groups tab.
 */
const CONCURRENCY_INVALID_MSG =
    'Max concurrent requests must be a whole number of at least 1, or empty for unlimited.';

class RateLimitManager {
    constructor() {
        this._profileId = null;
        this._switcher = new window.ProfileSwitcher(document.getElementById('rl-profile-switcher'), {
            label: 'Rate limits for',
            hint: 'Limits apply to every member of this profile.',
            stats: p => [[p.member_count, 'member']],
            onSelect: id => this.onProfileChange(id),
        });
        this._concInput = document.getElementById('rl-max-concurrent');
        this._concSave = document.getElementById('rl-max-concurrent-save');
        this._concClear = document.getElementById('rl-max-concurrent-clear');
        this._concSave?.addEventListener('click', () => this._saveTypedConcurrency());
        this._concClear?.addEventListener('click', () => this._saveConcurrency(''));
        this._concInput?.addEventListener('keydown', e => {
            if (e.key === 'Enter') this._saveTypedConcurrency();
        });
        document.addEventListener('profiles:changed', () => this._onProfilesChanged());
    }

    async _onProfilesChanged() {
        const before = this._profileId;
        await this._fillProfileSwitcher();
        // The shown profile was deleted and the switcher fell back to another: reload the
        // panes, or the next Save would write the old profile's values into this one.
        if (before !== null && this._profileId !== before) await this._loadProfile();
    }

    async load() {
        await this._fillProfileSwitcher();
        await this._loadProfile();
    }

    async _fillProfileSwitcher() {
        const want = this._profileId ?? window.ProfileStore.selected('rate-limits', window.ProfileStore.DEFAULT_ID);
        this._profileId = await this._switcher.render(want);
    }

    async onProfileChange(id) {
        this._profileId = id;
        window.ProfileStore.select('rate-limits', this._profileId);
        await this._loadProfile();
    }

    async _loadProfile() {
        // On failure the table falls back to fetching for itself.
        const limits = await this._fetchLimits();
        this._renderConcurrency(limits?.profile);
        await window.ModelGroupManager?.setProfile(this._profileId, limits);
    }

    /** *profile* is a ProfileResponse, or null when there is none to edit. */
    _renderConcurrency(profile) {
        if (!this._concInput) return;
        const value = profile?.max_concurrent_requests ?? null;
        this._concInput.value = value ?? '';
        this._concInput.disabled = !profile;
        this._concSave.disabled = !profile;
        this._concClear.disabled = !profile || value === null;
    }

    /** Save what is typed. A number input reports text it cannot parse ('-', '1e') as
     *  an empty value, which must be refused rather than read as Clear. */
    _saveTypedConcurrency() {
        if (this._concInput.validity.badInput) {
            window.UIUtils?.showToast(CONCURRENCY_INVALID_MSG, 'error');
            return;
        }
        this._saveConcurrency(this._concInput.value.trim());
    }

    async _saveConcurrency(raw) {
        const profileId = this._profileId;
        if (!profileId) return;
        let value = null;
        if (raw !== '') {
            value = Number(raw);
            if (!Number.isInteger(value) || value < 1) {
                window.UIUtils?.showToast(CONCURRENCY_INVALID_MSG, 'error');
                return;
            }
        }
        try {
            const resp = await fetch(`/admin/profiles/limits?profile_id=${profileId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ max_concurrent_requests: value }),
            });
            if (resp.ok) {
                const profile = await resp.json();
                // The admin may have switched profiles while this was in flight.
                if (this._profileId === profileId) this._renderConcurrency(profile);
                window.UIUtils?.showToast(value === null
                    ? 'Concurrency limit cleared (unlimited).' : 'Concurrency limit saved.', 'success');
            } else {
                const e = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(typeof e.detail === 'string' ? e.detail : 'Failed to save concurrency limit.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error.', 'error');
        }
    }

    async _fetchLimits() {
        if (!this._profileId) return null;
        try {
            const resp = await fetch(`/admin/profiles/limits?profile_id=${this._profileId}`, { credentials: 'include' });
            return resp.ok ? await resp.json() : null;
        } catch (e) {
            console.error('RateLimitManager: failed to load profile limits', e);
            return null;
        }
    }
}

window.RateLimitManager = new RateLimitManager();

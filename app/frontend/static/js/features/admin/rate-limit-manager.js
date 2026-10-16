/**
 * Rate Limits tab. Every limit belongs to a user profile: the profile switcher at the
 * top picks whose limits the table edits -- one row per model group, Default first.
 * Groups themselves are managed in the Model Groups tab.
 */
class RateLimitManager {
    constructor() {
        this._profileId = null;
        this._switcher = new window.ProfileSwitcher(document.getElementById('rl-profile-switcher'), {
            label: 'Rate limits for',
            hint: 'Limits apply to every member of this profile.',
            stats: p => [[p.member_count, 'member']],
            onSelect: id => this.onProfileChange(id),
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
        await window.ModelGroupManager?.setProfile(this._profileId, limits);
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

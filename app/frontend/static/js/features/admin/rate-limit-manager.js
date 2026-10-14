/**
 * Rate Limits tab. Every limit belongs to a user profile: the profile switcher at the
 * top picks which one the Overall, Model and Instance panes edit.
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
        // One GET serves all three panes (overall, model groups, instance groups).
        // On failure the group panes fall back to fetching for themselves.
        const limits = await this._fetchLimits();
        this._renderOverall(limits);
        await Promise.all([
            window.ModelGroupManager?.setProfile(this._profileId, limits),
            window.InstanceGroupManager?.setProfile(this._profileId, limits),
        ]);
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

    _renderOverall(limits) {
        const profile = limits?.profile;
        if (!profile) return;
        document.getElementById('rl-profile-rpm').value = profile.rpm_limit ?? '';
        document.getElementById('rl-profile-rpd').value = profile.rpd_limit ?? '';
    }

    async saveProfileLimits() {
        if (!this._profileId) return;
        const rpmRaw = document.getElementById('rl-profile-rpm')?.value.trim();
        const rpdRaw = document.getElementById('rl-profile-rpd')?.value.trim();
        const body = {
            rpm_limit: rpmRaw !== '' ? parseInt(rpmRaw, 10) : null,
            rpd_limit: rpdRaw !== '' ? parseInt(rpdRaw, 10) : null,
        };
        try {
            const resp = await fetch(`/admin/profiles/limits?profile_id=${this._profileId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (resp.ok) {
                window.UIUtils?.showToast('Profile limits saved.', 'success');
                await window.ProfileStore.notifyChanged();
            } else {
                const err = await resp.json().catch(() => ({}));
                window.UIUtils?.showToast(err.detail || 'Failed to save limits.', 'error');
            }
        } catch (e) {
            window.UIUtils?.showToast('Network error saving limits.', 'error');
        }
    }
}

window.RateLimitManager = new RateLimitManager();

/**
 * Usage Manager Module
 * Displays per-user and per-model request counts for the last 30 days.
 */

class UsageManager {
    constructor() {
        this.currentView = 'user'; // 'user' | 'model'
        this.isDrilledIn = false;
        this._cache = null; // top-level API response
        this._refreshTimer = null;
        this._refreshIntervalMs = 60_000; // poll every 60s
        this._lastDrilldown = null; // {axis, id} while drilled in
        this._tabIsActive = false; // true while user is on the Usage tab
    }

    async load() {
        this._tabIsActive = true;
        await this._fetchAndRender({ silent: false });
        this.startAutoRefresh();
    }

    /**
     * Begin polling /admin/usage on an interval while the usage tab is open.
     * Mirrors RequestTracker.connect() shape — start when the user enters the
     * tab, stop on exit. Safe to call repeatedly (idempotent).
     */
    startAutoRefresh() {
        this._clearTimer('restarting');
        console.log(`[UsageManager] auto-refresh started (every ${this._refreshIntervalMs / 1000}s)`);
        this._refreshTimer = setInterval(() => {
            console.log('[UsageManager] tick — refreshing usage');
            this._silentRefresh().catch(err => {
                console.warn('[UsageManager] silent refresh failed:', err);
            });
        }, this._refreshIntervalMs);
    }

    stopAutoRefresh() {
        this._tabIsActive = false;
        this._clearTimer('stopped');
    }

    _clearTimer(reason) {
        if (this._refreshTimer) {
            console.log(`[UsageManager] auto-refresh ${reason}`);
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    }

    async _fetchAndRender({ silent }) {
        try {
            const resp = await fetch('/admin/usage', { credentials: 'include', cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            this._cache = await resp.json();
        } catch (e) {
            console.error('UsageManager: failed to load usage', e);
            if (!silent) this._showError('Failed to load usage data.');
            return;
        }
        if (!silent) {
            this.isDrilledIn = false;
            this._lastDrilldown = null;
            this._setHeaderMode('toggle');
        }
        this._renderStats(this._cache.totals);
        this._updateSince(this._cache.shown_since);
        if (!this.isDrilledIn) {
            this._updateToggle(this.currentView);
            this._renderTopLevel();
        }
    }

    async _silentRefresh() {
        // Stats + since timestamp always refresh. If the user has drilled into a
        // specific user/model, also refresh that view; otherwise refresh the
        // top-level table.
        await this._fetchAndRender({ silent: true });
        if (this.isDrilledIn && this._lastDrilldown) {
            await this._refreshDrilldown();
        }
    }

    async _refreshDrilldown() {
        const { axis, id } = this._lastDrilldown;
        const url = `/admin/usage?view=${encodeURIComponent(axis)}&id=${encodeURIComponent(id)}`;
        try {
            const resp = await fetch(url, { credentials: 'include', cache: 'no-store' });
            if (!resp.ok) return;
            const data = await resp.json();
            this._updateSince(data.shown_since);
            if (axis === 'user') {
                this._renderModelBreakdown(data.breakdown, id);
            } else {
                this._renderUserBreakdown(data.breakdown, id);
            }
        } catch (e) {
            // silent — keep the existing view; next tick will retry.
        }
    }

    setView(view) {
        if (this.isDrilledIn) return;
        this.currentView = view;
        this._updateToggle(view);
        if (this._cache) this._renderTopLevel();
    }

    async drillDown(axis, id) {
        const url = `/admin/usage?view=${encodeURIComponent(axis)}&id=${encodeURIComponent(id)}`;
        let data;
        try {
            const resp = await fetch(url, { credentials: 'include', cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = await resp.json();
        } catch (e) {
            console.error('UsageManager: drill-down failed', e);
            window.UIUtils?.showToast('Failed to load breakdown.', 'error');
            return;
        }
        this.isDrilledIn = true;
        this._lastDrilldown = { axis, id };
        this._updateSince(data.shown_since);
        this._setHeaderMode('back');
        if (axis === 'user') {
            this._renderModelBreakdown(data.breakdown, id);
        } else {
            this._renderUserBreakdown(data.breakdown, id);
        }
    }

    back() {
        if (!this._cache) return;
        this.isDrilledIn = false;
        this._lastDrilldown = null;
        this._updateSince(this._cache.shown_since);
        this._setHeaderMode('toggle');
        this._renderTopLevel();
    }

    _setHeaderMode(mode) {
        const toggle = document.getElementById('usage-toggle');
        const back = document.getElementById('usage-back');
        if (toggle) toggle.style.display = mode === 'back' ? 'none' : '';
        if (back) back.style.display = mode === 'back' ? '' : 'none';
    }

    // ---- private ----

    _renderTopLevel() {
        this._updateToggle(this.currentView);
        if (this.currentView === 'user') {
            this._renderUserTable(this._cache.per_user);
        } else {
            this._renderModelTable(this._cache.per_model);
        }
    }

    _drilldownHeader(name) {
        return `<div class="fw-semibold mb-3">${name}</div>`;
    }

    _renderUserTable(rows) {
        const container = document.getElementById('usage-table-container');
        if (!rows || rows.length === 0) {
            container.innerHTML = this._emptyState();
            return;
        }
        const rowsHtml = rows.map(r => `
            <tr class="usage-drilldown-row" data-axis="user" data-id="${this._esc(r.user_identity)}" style="cursor:pointer;" title="Click to see breakdown by model">
                <td>${this._esc(r.user_identity)}</td>
                <td><span class="badge bg-secondary">${this._esc(r.user_type)}</span></td>
                <td>${r.request_count.toLocaleString()}</td>
            </tr>`).join('');
        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>User</th><th>Type</th><th>Requests</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
        this._bindRowClicks(container);
    }

    _renderModelTable(rows) {
        const container = document.getElementById('usage-table-container');
        if (!rows || rows.length === 0) {
            container.innerHTML = this._emptyState();
            return;
        }
        const rowsHtml = rows.map(r => `
            <tr class="usage-drilldown-row" data-axis="model" data-id="${this._esc(r.model)}" style="cursor:pointer;" title="Click to see breakdown by user">
                <td>${this._esc(r.model)}</td>
                <td>${r.request_count.toLocaleString()}</td>
            </tr>`).join('');
        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Model</th><th>Requests</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
        this._bindRowClicks(container);
    }

    _renderUserBreakdown(rows, id) {
        const container = document.getElementById('usage-table-container');
        const header = this._drilldownHeader(this._esc(id));
        if (!rows || rows.length === 0) {
            container.innerHTML = header + this._emptyState();
            return;
        }
        const rowsHtml = rows.map(r => `
            <tr>
                <td>${this._esc(r.user_identity)}</td>
                <td><span class="badge bg-secondary">${this._esc(r.user_type)}</span></td>
                <td>${r.request_count.toLocaleString()}</td>
            </tr>`).join('');
        container.innerHTML = header + `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>User</th><th>Type</th><th>Requests</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
    }

    _renderModelBreakdown(rows, id) {
        const container = document.getElementById('usage-table-container');
        const header = this._drilldownHeader(this._esc(id));
        if (!rows || rows.length === 0) {
            container.innerHTML = header + this._emptyState();
            return;
        }
        const rowsHtml = rows.map(r => `
            <tr>
                <td>${this._esc(r.model)}</td>
                <td>${r.request_count.toLocaleString()}</td>
            </tr>`).join('');
        container.innerHTML = header + `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Model</th><th>Requests</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
    }

    _renderStats(totals) {
        if (!totals) return;
        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val !== undefined ? val.toLocaleString() : '0';
        };
        set('usage-total-requests', totals.requests);
        set('usage-unique-users', totals.unique_users);
        set('usage-unique-models', totals.unique_models);
    }

    _updateSince(dateStr) {
        const el = document.getElementById('usage-since');
        if (el) el.textContent = dateStr || '—';
    }

    _updateToggle(view) {
        const btnUser = document.getElementById('usage-toggle-user');
        const btnModel = document.getElementById('usage-toggle-model');
        if (!btnUser || !btnModel) return;
        // Inline styles override the dark-theme !important rule that flattens
        // .btn-primary.active and .btn-outline-primary.active to the same look.
        const apply = (btn, selected) => {
            if (selected) {
                btn.style.cssText = 'background-color: var(--mono-text-primary); border-color: var(--mono-text-primary); color: var(--mono-0);';
                btn.classList.add('active');
            } else {
                btn.style.cssText = '';
                btn.classList.remove('active');
            }
        };
        apply(btnUser, view === 'user');
        apply(btnModel, view === 'model');
    }

    _bindRowClicks(container) {
        container.querySelectorAll('.usage-drilldown-row').forEach(row => {
            row.addEventListener('click', () => {
                this.drillDown(row.dataset.axis, row.dataset.id);
            });
        });
    }

    _emptyState() {
        return `
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fas fa-chart-bar"></i></div>
                <h3>No Usage Data</h3>
                <p>Usage data will appear here once requests are made.</p>
            </div>`;
    }

    _showError(msg) {
        const container = document.getElementById('usage-table-container');
        if (container) container.innerHTML = `<div class="alert alert-danger">${this._esc(msg)}</div>`;
    }

    _esc(str) {
        const d = document.createElement('div');
        d.textContent = str || '';
        return d.innerHTML;
    }
}

window.UsageManager = new UsageManager();

// Pause auto-refresh while the browser tab is hidden; resume when visible
// (only resumes if the user was on the Usage section before hiding).
document.addEventListener('visibilitychange', () => {
    const mgr = window.UsageManager;
    if (!mgr) return;
    if (document.hidden) {
        mgr._clearTimer('paused (tab hidden)');
    } else if (mgr._tabIsActive) {
        mgr.startAutoRefresh();
    }
});

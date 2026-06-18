/**
 * Usage Manager Module
 * Displays per-user and per-model request counts for a selectable time window.
 */

const MONTH_NAMES = [
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
];

class UsageManager {
    constructor() {
        this.currentView = 'user'; // 'user' | 'model'
        this.isDrilledIn = false;
        this._cache = null;
        this._refreshTimer = null;
        this._refreshIntervalMs = 60_000;
        this._lastDrilldown = null;
        this._tabIsActive = false;
        this._chart = null; // Chart.js instance for the usage timeseries graph

        // Window state
        this._window = '30d';   // active window key
        this._year = null;      // set when _window === 'month'
        this._month = null;
        this._popoverOpen = false;
        this._yearsLoaded = false;
        this._popoverDismiss = null; // stored outside-click handler
        this._resizeObserver = null;
    }

    // ------------------------------------------------------------------ //
    // Public lifecycle
    // ------------------------------------------------------------------ //

    async load() {
        this._tabIsActive = true;
        this._initResizeObserver();
        this._bindMonthChips();
        await this._fetchAndRender({ silent: false });
        this._startRefreshIfLive();
        // Re-measure after fetch+render so layout is stable
        requestAnimationFrame(() => this._positionIndicator(this._window));
    }

    _bindMonthChips() {
        const grid = document.getElementById('tw-months-grid');
        if (!grid) return;
        grid.addEventListener('click', (e) => {
            const chip = e.target.closest('.tw-month-chip');
            if (!chip || grid.dataset.disabled === 'true') return;
            this._selectMonth(parseInt(chip.dataset.month, 10));
        });
    }

    startAutoRefresh() {
        this._clearTimer('restarting');
        console.log(`[UsageManager] auto-refresh started (every ${this._refreshIntervalMs / 1000}s)`);
        this._refreshTimer = setInterval(() => {
            this._silentRefresh().catch(err => console.warn('[UsageManager] silent refresh failed:', err));
        }, this._refreshIntervalMs);
    }

    stopAutoRefresh() {
        this._tabIsActive = false;
        this._clearTimer('stopped');
    }

    // ------------------------------------------------------------------ //
    // Window selector
    // ------------------------------------------------------------------ //

    setWindow(win) {
        this._window = win;
        this._year = null;
        this._month = null;
        this._closeOlderPopover();
        this._updateWindowButtons(win);
        this.isDrilledIn = false;
        this._lastDrilldown = null;
        this._setHeaderMode('toggle');
        this._fetchAndRender({ silent: false });
        this._clearTimer('window changed');
        this._startRefreshIfLive();
    }

    toggleOlderPopover() {
        if (this._popoverOpen) {
            this._closeOlderPopover();
        } else {
            this._openOlderPopover();
        }
    }

    // Legacy aliases kept so any other callers don't break
    toggleOlderPanel() { this.toggleOlderPopover(); }
    onYearChange() {}
    onMonthChange() {}

    // ------------------------------------------------------------------ //
    // View + drill-down
    // ------------------------------------------------------------------ //

    setView(view) {
        if (this.isDrilledIn) return;
        this.currentView = view;
        this._updateToggle(view);
        if (this._cache) this._renderTopLevel();
    }

    async drillDown(axis, id) {
        const url = this._buildUrl({ view: axis, id });
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
        this._setHeaderMode('back');
        this._renderChart(data.timeseries);
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
        this._setHeaderMode('toggle');
        this._renderChart(this._cache.timeseries);
        this._renderTopLevel();
    }

    // ------------------------------------------------------------------ //
    // Private — fetching
    // ------------------------------------------------------------------ //

    async _fetchYears() {
        try {
            const resp = await fetch('/admin/usage/years', { credentials: 'include', cache: 'no-store' });
            if (!resp.ok) return;
            const data = await resp.json();
            this._populateYearList(data.years || []);
            this._yearsLoaded = true;
        } catch (e) {
            console.warn('[UsageManager] failed to load years', e);
        }
    }

    _buildUrl(extra = {}) {
        const params = new URLSearchParams();
        params.set('window', this._window);
        if (this._window === 'month' && this._year && this._month) {
            params.set('year', this._year);
            params.set('month', this._month);
        }
        if (extra.view) params.set('view', extra.view);
        if (extra.id != null) params.set('id', extra.id);
        return `/admin/usage?${params}`;
    }

    async _fetchAndRender({ silent }) {
        try {
            const resp = await fetch(this._buildUrl(), { credentials: 'include', cache: 'no-store' });
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
        this._renderEarliestDate(this._cache.earliest_date);
        if (!this.isDrilledIn) {
            // When drilled in, _refreshDrilldown renders the filtered series; rendering
            // the unfiltered series here first would flash the wrong bars on each refresh.
            this._renderChart(this._cache.timeseries);
            this._updateToggle(this.currentView);
            this._renderTopLevel();
        }
    }

    async _silentRefresh() {
        await this._fetchAndRender({ silent: true });
        if (this.isDrilledIn && this._lastDrilldown) {
            await this._refreshDrilldown();
        }
    }

    async _refreshDrilldown() {
        const { axis, id } = this._lastDrilldown;
        const url = this._buildUrl({ view: axis, id });
        try {
            const resp = await fetch(url, { credentials: 'include', cache: 'no-store' });
            if (!resp.ok) return;
            const data = await resp.json();
            this._renderChart(data.timeseries);
            if (axis === 'user') {
                this._renderModelBreakdown(data.breakdown, id);
            } else {
                this._renderUserBreakdown(data.breakdown, id);
            }
        } catch (e) {
            // silent — keep existing view; next tick will retry
        }
    }

    _startRefreshIfLive() {
        const liveWindows = ['24h', 'today'];
        if (liveWindows.includes(this._window)) {
            this.startAutoRefresh();
        }
    }

    // ------------------------------------------------------------------ //
    // Private — older popover
    // ------------------------------------------------------------------ //

    async _openOlderPopover() {
        this._popoverOpen = true;
        const popover = document.getElementById('usage-older-popover');
        const btn = document.getElementById('usage-window-older-btn');
        if (popover) popover.classList.add('is-open');
        if (btn) btn.setAttribute('aria-expanded', 'true');

        if (!this._yearsLoaded) await this._fetchYears();

        // Outside-click + Escape dismissal
        const dismiss = (e) => {
            if (e.type === 'keydown' && e.key !== 'Escape') return;
            const container = document.getElementById('usage-window-container');
            if (e.type === 'click' && container && container.contains(e.target)) return;
            this._closeOlderPopover();
        };
        this._popoverDismiss = dismiss;
        setTimeout(() => {
            document.addEventListener('click', dismiss);
            document.addEventListener('keydown', dismiss);
        }, 0);
    }

    _closeOlderPopover() {
        this._popoverOpen = false;
        const popover = document.getElementById('usage-older-popover');
        const btn = document.getElementById('usage-window-older-btn');
        if (popover) popover.classList.remove('is-open');
        if (btn) btn.setAttribute('aria-expanded', 'false');
        if (this._popoverDismiss) {
            document.removeEventListener('click', this._popoverDismiss);
            document.removeEventListener('keydown', this._popoverDismiss);
            this._popoverDismiss = null;
        }
    }

    _populateYearList(years) {
        const list = document.getElementById('tw-years-list');
        if (!list) return;
        list.innerHTML = years.map(y =>
            `<button class="tw-year-item" data-year="${y}" onclick="window.UsageManager?._selectYear(${y})">${y}</button>`
        ).join('');
    }

    _selectYear(y) {
        this._year = y;
        this._month = null;

        // Update year active state
        document.querySelectorAll('.tw-year-item').forEach(el => {
            el.classList.toggle('is-active', parseInt(el.dataset.year, 10) === y);
        });

        // Enable month chips
        const grid = document.getElementById('tw-months-grid');
        if (grid) {
            grid.dataset.disabled = 'false';
            grid.querySelectorAll('.tw-month-chip').forEach(c => c.classList.remove('is-active'));
        }
    }

    _selectMonth(m) {
        if (!this._year) return;
        this._month = m;
        this._window = 'month';

        // Update month chip active state
        document.querySelectorAll('.tw-month-chip').forEach(c => {
            c.classList.toggle('is-active', parseInt(c.dataset.month, 10) === m);
        });

        // Update the "Older" trigger label and style
        const label = document.getElementById('usage-older-label');
        if (label) label.textContent = `${MONTH_NAMES[m].slice(0, 3)} ${this._year}`;
        const btn = document.getElementById('usage-window-older-btn');
        if (btn) btn.classList.add('is-active');

        // Switch container mode so indicator hides
        const container = document.getElementById('usage-window-container');
        if (container) container.dataset.windowMode = 'older';

        // Deactivate preset buttons
        document.querySelectorAll('.time-window__preset').forEach(b => b.classList.remove('is-active'));

        this._closeOlderPopover();
        this.isDrilledIn = false;
        this._lastDrilldown = null;
        this._setHeaderMode('toggle');
        this._fetchAndRender({ silent: false });
        this._clearTimer('historical window');
    }

    // ------------------------------------------------------------------ //
    // Private — rendering
    // ------------------------------------------------------------------ //

    _setHeaderMode(mode) {
        const toggle = document.getElementById('usage-toggle');
        const back = document.getElementById('usage-back');
        if (toggle) toggle.style.display = mode === 'back' ? 'none' : '';
        if (back) back.style.display = mode === 'back' ? '' : 'none';
    }

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

    _pct(count, total) {
        if (!total || total === 0) return '—';
        return ((count / total) * 100).toFixed(1) + '%';
    }

    _renderUserTable(rows) {
        const container = document.getElementById('usage-table-container');
        if (!rows || rows.length === 0) { container.innerHTML = this._emptyState(); return; }
        const total = this._cache?.totals?.requests || rows.reduce((s, r) => s + r.request_count, 0);
        const rowsHtml = rows.map(r => `
            <tr class="usage-drilldown-row" data-axis="user" data-id="${this._esc(r.user_identity)}" style="cursor:pointer;" title="Click to see breakdown by model">
                <td>${this._esc(r.user_identity)}</td>
                <td><span class="badge bg-secondary">${this._esc(r.user_type)}</span></td>
                <td>${r.request_count.toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}</td>
            </tr>`).join('');
        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>User</th><th>Type</th><th>Requests</th><th class="text-end">Percentage</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
        this._bindRowClicks(container);
    }

    _renderModelTable(rows) {
        const container = document.getElementById('usage-table-container');
        if (!rows || rows.length === 0) { container.innerHTML = this._emptyState(); return; }
        const total = this._cache?.totals?.requests || rows.reduce((s, r) => s + r.request_count, 0);
        const rowsHtml = rows.map(r => `
            <tr class="usage-drilldown-row" data-axis="model" data-id="${this._esc(r.model)}" style="cursor:pointer;" title="Click to see breakdown by user">
                <td>${this._esc(r.model)}</td>
                <td>${r.request_count.toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}</td>
            </tr>`).join('');
        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Model</th><th>Requests</th><th class="text-end">Percentage</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
        this._bindRowClicks(container);
    }

    _renderUserBreakdown(rows, id) {
        const container = document.getElementById('usage-table-container');
        const header = this._drilldownHeader(this._esc(id));
        if (!rows || rows.length === 0) { container.innerHTML = header + this._emptyState(); return; }
        const total = rows.reduce((s, r) => s + r.request_count, 0);
        const rowsHtml = rows.map(r => `
            <tr>
                <td>${this._esc(r.user_identity)}</td>
                <td><span class="badge bg-secondary">${this._esc(r.user_type)}</span></td>
                <td>${r.request_count.toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}</td>
            </tr>`).join('');
        container.innerHTML = header + `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>User</th><th>Type</th><th>Requests</th><th class="text-end">Percentage</th></tr></thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>`;
    }

    _renderModelBreakdown(rows, id) {
        const container = document.getElementById('usage-table-container');
        const header = this._drilldownHeader(this._esc(id));
        if (!rows || rows.length === 0) { container.innerHTML = header + this._emptyState(); return; }
        const total = rows.reduce((s, r) => s + r.request_count, 0);
        const rowsHtml = rows.map(r => `
            <tr>
                <td>${this._esc(r.model)}</td>
                <td>${r.request_count.toLocaleString()}</td>
                <td class="text-end">${this._pct(r.request_count, total)}</td>
            </tr>`).join('');
        container.innerHTML = header + `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead><tr><th>Model</th><th>Requests</th><th class="text-end">Percentage</th></tr></thead>
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

    _renderEarliestDate(dateStr) {
        const el = document.getElementById('usage-earliest-date');
        if (!el) return;
        if (dateStr) {
            const d = new Date(dateStr + 'T00:00:00');
            const formatted = d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
            el.textContent = `since ${formatted}`;
        } else {
            el.textContent = '';
        }
    }

    _renderChart(timeseries) {
        const canvas = document.getElementById('usage-chart');
        if (!canvas || typeof Chart === 'undefined') return;

        const series = Array.isArray(timeseries) ? timeseries : [];
        const labels = series.map(b => b.label);
        const data = series.map(b => b.count);

        const css = getComputedStyle(document.documentElement);
        const barColor = (css.getPropertyValue('--mono-text-primary') || '#e0e0e0').trim();
        const mutedColor = (css.getPropertyValue('--mono-text-muted') || '#888').trim();
        const gridColor = (css.getPropertyValue('--border-color') || 'rgba(255,255,255,0.08)').trim();
        const fontFamily = (css.getPropertyValue('--font-family-mono') || 'monospace').trim();

        if (this._chart) {
            this._chart.data.labels = labels;
            this._chart.data.datasets[0].data = data;
            this._chart.update();
            return;
        }

        this._chart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Requests',
                    data,
                    backgroundColor: barColor,
                    borderWidth: 0,
                    borderRadius: 2,
                    maxBarThickness: 48,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 200 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        titleFont: { family: fontFamily },
                        bodyFont: { family: fontFamily },
                    },
                },
                scales: {
                    x: {
                        ticks: { color: mutedColor, font: { family: fontFamily, size: 10 }, maxRotation: 0, autoSkip: true },
                        grid: { display: false },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: mutedColor, font: { family: fontFamily, size: 10 }, precision: 0 },
                        grid: { color: gridColor },
                    },
                },
            },
        });
    }

    _updateWindowButtons(activeKey) {
        // Reset container and older trigger to preset mode
        const container = document.getElementById('usage-window-container');
        if (container) container.dataset.windowMode = 'preset';
        const olderBtn = document.getElementById('usage-window-older-btn');
        if (olderBtn) {
            olderBtn.classList.remove('is-active');
            const label = document.getElementById('usage-older-label');
            if (label) label.textContent = 'Older';
        }

        // Preset active class
        document.querySelectorAll('.time-window__preset').forEach(btn => {
            btn.classList.toggle('is-active', btn.dataset.window === activeKey);
        });

        // Slide the indicator to the active preset
        this._positionIndicator(activeKey);
    }

    _positionIndicator(activeKey) {
        const track = document.getElementById('usage-window-btns');
        if (!track) return;
        const active = track.querySelector(`.time-window__preset[data-window="${activeKey}"]`);
        if (!active || active.offsetWidth === 0) return;
        // offsetLeft is relative to offsetParent (the track's border box).
        // The indicator has left:2px in CSS to sit inside the 2px padding;
        // we drive it via translateX from that left:2px origin, so we subtract 2px.
        const x = active.offsetLeft - 2;
        const w = active.offsetWidth;
        track.style.setProperty('--tw-x', `${Math.max(0, x)}px`);
        track.style.setProperty('--tw-w', `${w}px`);
    }

    _initResizeObserver() {
        const track = document.getElementById('usage-window-btns');
        if (!track) return;
        // Position on first render
        requestAnimationFrame(() => this._positionIndicator(this._window));
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(() => this._positionIndicator(this._window));
            this._resizeObserver.observe(track);
        } else {
            window.addEventListener('resize', () => this._positionIndicator(this._window));
        }
    }

    _updateToggle(view) {
        const btnUser = document.getElementById('usage-toggle-user');
        const btnModel = document.getElementById('usage-toggle-model');
        if (!btnUser || !btnModel) return;
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
        // Ensure indicator is positioned when toggle re-renders (view switch doesn't change window)
        if (this._window !== 'month') this._positionIndicator(this._window);
    }

    _bindRowClicks(container) {
        container.querySelectorAll('.usage-drilldown-row').forEach(row => {
            row.addEventListener('click', () => this.drillDown(row.dataset.axis, row.dataset.id));
        });
    }

    _emptyState() {
        return `
            <div class="empty-state">
                <div class="empty-state-icon"><i class="fas fa-chart-bar"></i></div>
                <h3>No Usage Data</h3>
                <p>No data found for the selected time window.</p>
            </div>`;
    }

    _showError(msg) {
        const container = document.getElementById('usage-table-container');
        if (container) container.innerHTML = `<div class="alert alert-danger">${this._esc(msg)}</div>`;
    }

    _clearTimer(reason) {
        if (this._refreshTimer) {
            console.log(`[UsageManager] auto-refresh ${reason}`);
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    }

    _esc(str) {
        const d = document.createElement('div');
        d.textContent = str || '';
        return d.innerHTML;
    }
}

window.UsageManager = new UsageManager();

document.addEventListener('visibilitychange', () => {
    const mgr = window.UsageManager;
    if (!mgr) return;
    if (document.hidden) {
        mgr._clearTimer('paused (tab hidden)');
    } else if (mgr._tabIsActive) {
        mgr._startRefreshIfLive();
    }
});

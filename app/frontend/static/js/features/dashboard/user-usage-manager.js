/**
 * User Usage Manager — per-user usage view
 * Displays the authenticated user's request counts by model over selectable time windows.
 * Mirrors admin UsageManager but simplified: no per-user drill-down, no year picker.
 */

class UserUsageManager {
    constructor() {
        this._cache = null;
        this._loadError = false;
        this._refreshTimer = null;
        this._refreshIntervalMs = 60_000;
        this._tabIsActive = false;
        this._chart = null; // Chart.js instance for the usage timeseries graph

        // Window state
        this._window = '30d';   // active window key
        this._year = null;      // set when _window === 'month'
        this._month = null;
    }

    // ------------------------------------------------------------------ //
    // Lifecycle
    // ------------------------------------------------------------------ //

    async load() {
        this._tabIsActive = true;
        await this._fetchAndRender({ silent: false });
        this._startRefreshIfLive();
        requestAnimationFrame(() => this._positionIndicator(this._window));
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
        this._updateWindowButtons(win);
        this._fetchAndRender({ silent: false });
        this._clearTimer('window changed');
        this._startRefreshIfLive();
    }

    _startRefreshIfLive() {
        const liveWindows = ['24h', 'today'];
        if (liveWindows.includes(this._window)) {
            this.startAutoRefresh();
        }
    }

    startAutoRefresh() {
        this._clearTimer('restarting');
        this._refreshTimer = setInterval(() => {
            this._silentRefresh().catch(err => console.warn('[UserUsageManager] silent refresh failed:', err));
        }, this._refreshIntervalMs);
    }

    // ------------------------------------------------------------------ //
    // Fetch & render
    // ------------------------------------------------------------------ //

    async _fetchAndRender(opts = {}) {
        const { silent = false } = opts;
        try {
            const url = this._buildUrl();
            const response = await makeAuthenticatedRequest(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this._cache = await response.json();
            this._loadError = false;
            this._render(this._cache);
        } catch (error) {
            this._loadError = true;
            if (!silent) {
                console.error('[UserUsageManager] fetch failed:', error);
                document.getElementById('usageContainer').innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Failed to load usage data. Please refresh the page.
                    </div>
                `;
            }
        }
    }

    async _silentRefresh() {
        try {
            const url = this._buildUrl();
            const response = await makeAuthenticatedRequest(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this._cache = await response.json();
            this._loadError = false;
            this._render(this._cache);
        } catch (error) {
            // Silent — keep existing view
        }
    }

    _buildUrl() {
        const params = new URLSearchParams({ window: this._window });
        if (this._year !== null && this._month !== null) {
            params.append('year', this._year);
            params.append('month', this._month);
        }
        return `/auth/usage?${params.toString()}`;
    }

    _render(data) {
        if (!data) return;

        this._renderChart(data.timeseries);

        // get_usage_aggregates returns {breakdown:[...]} when filter_user is set,
        // not {per_model:[...], totals:{...}} — normalise here.
        const perModel = data.per_model || data.breakdown || [];
        const totalReqs = perModel.reduce((s, m) => s + (m.request_count || 0), 0);
        const totals = data.totals || { requests: totalReqs, unique_models: perModel.length };

        // Update stats
        document.getElementById('totalRequests').textContent = (totals.requests || totalReqs).toLocaleString();
        document.getElementById('uniqueModels').textContent = totals.unique_models ?? perModel.length;

        // Show earliest date
        const dateEl = document.getElementById('user-usage-earliest-date');
        if (dateEl) {
            const dateStr = data.earliest_date;
            if (dateStr) {
                const d = new Date(dateStr + 'T00:00:00');
                const formatted = d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
                dateEl.textContent = `since ${formatted}`;
            } else {
                dateEl.textContent = '';
            }
        }

        // Render per-model breakdown
        const container = document.getElementById('usageContainer');

        if (perModel.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-chart-line fa-2x mb-3"></i>
                    <p>No usage data for this period</p>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="table-responsive">
                    <table class="table table-sm table-hover mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>Model</th>
                                <th class="text-end">Requests</th>
                                <th class="text-end">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${perModel.map(m => {
                                const pct = totals.requests > 0
                                    ? ((m.request_count / totals.requests) * 100).toFixed(1)
                                    : 0;
                                return `
                                    <tr>
                                        <td>${m.model}</td>
                                        <td class="text-end">${m.request_count.toLocaleString()}</td>
                                        <td class="text-end">${pct}%</td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        }
    }

    // ------------------------------------------------------------------ //
    // Window buttons & indicator
    // ------------------------------------------------------------------ //

    _renderChart(timeseries) {
        const canvas = document.getElementById('user-usage-chart');
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
        const container = document.getElementById('usage-window-container');
        if (!container) return;

        container.dataset.windowMode = 'preset';

        // Remove is-active from all buttons
        const buttons = container.querySelectorAll('.time-window__preset');
        buttons.forEach(btn => btn.classList.remove('is-active'));

        // Add is-active to the matching button
        const activeBtn = container.querySelector(`[data-window="${activeKey}"]`);
        if (activeBtn) {
            activeBtn.classList.add('is-active');
            this._positionIndicator(activeKey);
        }
    }

    _positionIndicator(activeKey) {
        const activeBtn = document.querySelector(`[data-window="${activeKey}"]`);
        const indicator = document.getElementById('usage-window-indicator');
        const container = document.getElementById('usage-window-btns');

        if (!activeBtn || !indicator || !container) return;

        const x = activeBtn.offsetLeft;
        const w = activeBtn.offsetWidth;

        container.style.setProperty('--tw-x', `${x}px`);
        container.style.setProperty('--tw-w', `${w}px`);
    }

    _clearTimer(reason) {
        if (this._refreshTimer) {
            clearInterval(this._refreshTimer);
            this._refreshTimer = null;
        }
    }
}

// Global instance for use in onclick handlers
window.UserUsageManager = new UserUsageManager();

// Load usage data when the Usage tab is opened.
// Wraps showTab so it's triggered by tab click rather than page load.
document.addEventListener('DOMContentLoaded', function () {
    // showTab is defined in ui-utils.js which is loaded before this script.
    const originalShowTab = window.showTab;
    window.showTab = function(tabName) {
        originalShowTab(tabName);
        if (tabName === 'usage') {
            const mgr = window.UserUsageManager;
            if (!mgr._cache || mgr._loadError) {
                mgr.load().catch(err => console.error('Failed to load usage:', err));
            }
        }
    };
});

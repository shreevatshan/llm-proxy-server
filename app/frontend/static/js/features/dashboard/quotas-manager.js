/**
 * Quotas Manager — per-user read-only quota view
 * Displays the authenticated user's effective request limits (RPM/RPD) and model-group quotas.
 * Fetches GET /auth/quotas once on first tab open (no time window, no auto-refresh needed).
 */

class QuotasManager {
    constructor() {
        this._cache = null;
        this._loadError = false;
    }

    // ------------------------------------------------------------------ //
    // Lifecycle
    // ------------------------------------------------------------------ //

    async load() {
        await this._fetchAndRender();
    }

    // ------------------------------------------------------------------ //
    // Fetch & render
    // ------------------------------------------------------------------ //

    async _fetchAndRender() {
        try {
            const response = await makeAuthenticatedRequest('/auth/quotas');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this._cache = await response.json();
            this._loadError = false;
            this._render(this._cache);
        } catch (error) {
            this._loadError = true;
            console.error('[QuotasManager] fetch failed:', error);
            const errHtml = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Failed to load quota data. Please refresh the page.
                </div>
            `;
            ['quotaOverallContainer', 'quotaGroupsContainer', 'quotaInstanceGroupsContainer'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.innerHTML = errHtml;
            });
        }
    }

    _fmt(value) {
        // null means unlimited; integers rendered as locale string
        return value === null || value === undefined ? '∞' : Number(value).toLocaleString();
    }

    _render(data) {
        if (!data) return;

        if (data.is_admin) {
            // Admin users are exempt from all limits
            const adminNote = `
                <div class="alert alert-info">
                    <i class="fas fa-shield-alt me-2"></i>
                    Admin accounts are exempt from all request limits.
                </div>
            `;
            ['quotaOverallContainer', 'quotaGroupsContainer', 'quotaInstanceGroupsContainer'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.innerHTML = adminNote;
            });
            return;
        }

        // --- Overall quota card ---
        this._renderOverall(data.overall || {});

        // --- Instance-group quota table (precedence over model groups) ---
        this._renderGroupTable(
            document.getElementById('quotaInstanceGroupsContainer'),
            data.instance_groups || [],
            { itemsKey: 'instances', label: 'Instances', emptyIcon: 'fa-server', emptyText: 'No instance groups have been configured.' }
        );

        // --- Model-group quota table ---
        this._renderGroupTable(
            document.getElementById('quotaGroupsContainer'),
            data.groups || [],
            { itemsKey: 'models', label: 'Models', emptyIcon: 'fa-layer-group', emptyText: 'No model groups have been configured.' }
        );
    }

    _renderOverall(o) {
        const container = document.getElementById('quotaOverallContainer');
        if (!container) return;
        const rpdLeft = (o.rpd_remaining === null || o.rpd_remaining === undefined)
            ? ''
            : ` <small class="text-muted">(${Number(o.rpd_remaining).toLocaleString()} left)</small>`;
        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-sm table-hover mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Metric</th>
                            <th class="text-end">Limit</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>RPM</strong></td>
                            <td class="text-end">${this._fmt(o.rpm_limit)}</td>
                        </tr>
                        <tr>
                            <td><strong>RPD</strong></td>
                            <td class="text-end">${this._fmt(o.rpd_limit)}${rpdLeft}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `;
    }

    _renderGroupTable(container, groups, opts) {
        if (!container) return;
        if (groups.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas ${opts.emptyIcon} fa-2x mb-3"></i>
                    <p>${opts.emptyText}</p>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div class="table-responsive">
                <table class="table table-sm table-hover mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Group</th>
                            <th class="text-end">RPM Limit</th>
                            <th class="text-end">RPD Limit</th>
                            <th>${opts.label}</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${groups.map(g => {
                            const items = g[opts.itemsKey] || [];
                            const badges = items.length > 0
                                ? items.map(m => `<span class="badge bg-secondary me-1 mb-1" style="font-size:0.7rem;font-weight:400;">${this._escapeHtml(m)}</span>`).join('')
                                : `<span class="text-muted fst-italic">No ${opts.label.toLowerCase()}</span>`;
                            const rpdLeft = (g.rpd_remaining === null || g.rpd_remaining === undefined)
                                ? ''
                                : `<br><small class="text-muted">${Number(g.rpd_remaining).toLocaleString()} left</small>`;
                            return `
                                <tr>
                                    <td>
                                        <strong>${this._escapeHtml(g.name)}</strong>
                                        ${g.description ? `<br><small class="text-muted">${this._escapeHtml(g.description)}</small>` : ''}
                                    </td>
                                    <td class="text-end">${this._fmt(g.rpm_limit)}</td>
                                    <td class="text-end">${this._fmt(g.rpd_limit)}${rpdLeft}</td>
                                    <td style="max-width:340px;">${badges}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(String(str)));
        return div.innerHTML;
    }
}

// Global instance
window.QuotasManager = new QuotasManager();

// Lazy-load when the Quotas tab is opened (monkey-patch pattern from user-usage-manager.js)
document.addEventListener('DOMContentLoaded', function () {
    // Initialize Bootstrap tooltips for the quota title info icons
    if (window.bootstrap?.Tooltip) {
        document.querySelectorAll('#quotas-tab [data-bs-toggle="tooltip"]').forEach(el => {
            new bootstrap.Tooltip(el, { customClass: 'quota-tooltip' });
        });
    }

    const originalShowTab = window.showTab;
    window.showTab = function(tabName) {
        originalShowTab(tabName);
        if (tabName === 'quotas') {
            const mgr = window.QuotasManager;
            if (!mgr._cache || mgr._loadError) {
                mgr.load().catch(err => console.error('[QuotasManager] Failed to load quotas:', err));
            }
        }
    };
});

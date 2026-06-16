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
            document.getElementById('quotaGroupsContainer').innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Failed to load quota data. Please refresh the page.
                </div>
            `;
            this._renderOverallError();
        }
    }

    _renderOverallError() {
        ['quotaRpmLimit', 'quotaRpdRemaining', 'quotaRpdLimit'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '—';
        });
    }

    _fmt(value) {
        // null means unlimited; integers rendered as locale string
        return value === null || value === undefined ? '∞' : Number(value).toLocaleString();
    }

    _render(data) {
        if (!data) return;

        if (data.is_admin) {
            // Admin users are exempt from all limits
            ['quotaRpmLimit', 'quotaRpdRemaining', 'quotaRpdLimit'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = '∞';
            });
            document.getElementById('quotaGroupsContainer').innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-shield-alt me-2"></i>
                    Admin accounts are exempt from all request limits.
                </div>
            `;
            return;
        }

        // --- Overall stat cards ---
        const o = data.overall || {};
        const rpmLimEl = document.getElementById('quotaRpmLimit');
        const rpdRemEl = document.getElementById('quotaRpdRemaining');
        const rpdLimEl = document.getElementById('quotaRpdLimit');
        if (rpmLimEl) rpmLimEl.textContent = this._fmt(o.rpm_limit);
        if (rpdLimEl) rpdLimEl.textContent = this._fmt(o.rpd_limit);
        if (rpdRemEl) rpdRemEl.textContent = `Remaining: ${this._fmt(o.rpd_remaining)}`;

        // --- Model-group quota table ---
        const container = document.getElementById('quotaGroupsContainer');
        const groups = data.groups || [];

        if (groups.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-layer-group fa-2x mb-3"></i>
                    <p>No model groups have been configured.</p>
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
                            <th>Models</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${groups.map(g => {
                            const modelBadges = (g.models || []).length > 0
                                ? g.models.map(m => `<span class="badge bg-secondary me-1 mb-1" style="font-size:0.7rem;font-weight:400;">${this._escapeHtml(m)}</span>`).join('')
                                : '<span class="text-muted fst-italic">No models</span>';
                            return `
                                <tr>
                                    <td>
                                        <strong>${this._escapeHtml(g.name)}</strong>
                                        ${g.description ? `<br><small class="text-muted">${this._escapeHtml(g.description)}</small>` : ''}
                                    </td>
                                    <td class="text-end">${this._fmt(g.rpm_limit)}</td>
                                    <td class="text-end">${this._fmt(g.rpd_limit)}</td>
                                    <td style="max-width:340px;">${modelBadges}</td>
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

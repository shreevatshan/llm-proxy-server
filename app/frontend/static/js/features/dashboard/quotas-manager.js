/**
 * Quotas Manager — per-user read-only quota view
 * Renders the authenticated user's quota as a single unified list of sections. Each
 * section is one quota bucket (a model group, an instance group, or the ungrouped
 * "Other Models" bucket — indistinguishable in the UI) showing today's usage,
 * remaining, and the models it governs.
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
        const container = document.getElementById('quotaSectionsContainer');
        try {
            const response = await makeAuthenticatedRequest('/auth/quotas');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this._cache = await response.json();
            this._loadError = false;
            this._render(this._cache);
        } catch (error) {
            this._loadError = true;
            console.error('[QuotasManager] fetch failed:', error);
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Failed to load quota data. Please refresh the page.
                    </div>
                `;
            }
        }
    }

    _fmt(value) {
        // null means unlimited; integers rendered as locale string
        return value === null || value === undefined ? '∞' : Number(value).toLocaleString();
    }

    _render(data) {
        if (!data) return;
        const container = document.getElementById('quotaSectionsContainer');
        if (!container) return;

        if (data.is_admin) {
            container.innerHTML = `
                <div class="alert alert-info mb-0">
                    <i class="fas fa-shield-alt me-2"></i>
                    Admin accounts are exempt from all request limits.
                </div>
            `;
            return;
        }

        const sections = data.sections || [];
        if (sections.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-gauge-high fa-2x mb-3"></i>
                    <p class="mb-0">No quota information is available.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = sections.map((s, i) => this._renderSection(s, i)).join('');

        // Animate meter fills in on next frame (skipped under reduced-motion by CSS).
        requestAnimationFrame(() => {
            container.querySelectorAll('.quota-meter__fill').forEach(el => {
                el.style.width = el.dataset.fill || '0%';
            });
        });

        // Wire the footer-bar model dropdowns for large buckets (mirrors the
        // endpoints tab: click or Enter/Space on the bar expands the list).
        container.querySelectorAll('.quota-foot').forEach(foot => {
            foot.addEventListener('click', () => this._toggleModels(foot));
            foot.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this._toggleModels(foot);
                }
            });
        });
    }

    // Buckets with more than this many models tuck their chip wall behind an
    // endpoints-style footer bar so the quota header + meter stay the focus.
    static COLLAPSE_THRESHOLD = 8;

    _renderSection(s, index) {
        const models = s.models || [];
        const count = models.length;
        const eyebrow = `${count} ${count === 1 ? 'model' : 'models'}`;
        const desc = s.description
            ? `<span class="quota-desc">${this._escapeHtml(s.description)}</span>`
            : '';

        const collapsible = count > QuotasManager.COLLAPSE_THRESHOLD;
        const chips = count ? models.map(m => this._renderChip(m)).join('') : '';

        // Small buckets: models sit inline under the RPM label. Large buckets:
        // the foot carries RPM only, and models move to the collapsible footer bar.
        const inlineModels = collapsible
            ? ''
            : (count ? `<div class="quota-models">${chips}</div>` : `<span class="quota-empty">No models</span>`);

        const listId = `quota-models-${index}`;
        const dropdown = collapsible ? `
            <div class="quota-foot" role="button" tabindex="0" aria-expanded="false"
                 aria-controls="${listId}" data-target="${listId}">
                <span class="foot-label"><span class="foot-count">${count}</span> models</span>
                <i class="fas fa-chevron-down foot-chevron"></i>
            </div>
            <div class="quota-models-collapse" id="${listId}">
                <div class="quota-models">${chips}</div>
            </div>
        ` : '';

        return `
            <div class="quota-block">
                <div class="quota-block__head">
                    <div class="quota-ident">
                        <span class="quota-socket"></span>
                        <span class="quota-heading">
                            <span class="quota-name">${this._escapeHtml(s.name)}</span>
                            <span class="quota-eyebrow">${eyebrow}</span>
                            ${desc}
                        </span>
                    </div>
                    ${this._readoutHtml(s)}
                </div>
                ${this._meterHtml(s)}
                ${this._leftHtml(s)}
                <div class="quota-block__foot">
                    ${inlineModels}
                </div>
                ${dropdown}
            </div>
        `;
    }

    _renderChip(m) {
        // Model ids are '{provider}:{instance}/{model_name}'. The provider prefix
        // repeats across a bucket and carries little signal, so dim it and keep the
        // distinguishing model name bright — same idea as the endpoints URL treatment.
        const slash = m.indexOf('/');
        if (slash === -1) {
            return `<span class="quota-chip"><span class="c-name">${this._escapeHtml(m)}</span></span>`;
        }
        const prefix = m.slice(0, slash + 1);
        const name = m.slice(slash + 1);
        return `<span class="quota-chip"><span class="c-prefix">${this._escapeHtml(prefix)}</span><span class="c-name">${this._escapeHtml(name)}</span></span>`;
    }

    _toggleModels(foot) {
        // Mirror the endpoints tab's toggleEndpoints: animate a max-height
        // transition, then release to `none` so the list can reflow freely.
        const list = document.getElementById(foot.dataset.target);
        const block = foot.closest('.quota-block');
        if (!list || !block) return;

        const isOpen = block.classList.contains('expanded');
        if (isOpen) {
            list.style.maxHeight = list.scrollHeight + 'px';
            requestAnimationFrame(() => { list.style.maxHeight = '0'; });
            block.classList.remove('expanded');
            foot.setAttribute('aria-expanded', 'false');
        } else {
            block.classList.add('expanded');
            list.style.maxHeight = list.scrollHeight + 'px';
            foot.setAttribute('aria-expanded', 'true');
            list.addEventListener('transitionend', function handler() {
                if (block.classList.contains('expanded')) list.style.maxHeight = 'none';
                list.removeEventListener('transitionend', handler);
            });
        }
    }

    _readoutHtml(s) {
        // Top-right corner stacks both ceilings together: daily on top, per-minute
        // beneath it. Numbers stay primary; the "N left today" figure lives below
        // the meter instead (see _leftHtml).
        const used = Number(s.rpd_count || 0).toLocaleString();
        const day = (s.rpd_limit === null || s.rpd_limit === undefined)
            ? `<span class="q-used">${used}</span> <span class="q-inf">/ ∞</span> <span class="q-unit">today</span>`
            : `<span class="q-used">${used}</span> <span class="q-limit">/ ${this._fmt(s.rpd_limit)}</span> <span class="q-unit">today</span>`;
        return `
            <div class="quota-readout" title="Requests used today / daily limit · per-minute limit">
                <div class="q-day">${day}</div>
                <div class="q-min"><span class="q-min-val">${this._fmt(s.rpm_limit)}</span> <span class="q-unit">per minute</span></div>
            </div>
        `;
    }

    _leftHtml(s) {
        // Remaining daily allowance, shown directly under the meter. Only meaningful
        // when there's a ceiling and a known remainder.
        if (s.rpd_limit === null || s.rpd_limit === undefined) return '';
        if (s.rpd_remaining === null || s.rpd_remaining === undefined) return '';
        // Flag near-exhaustion by weight + a glyph (no color — the palette is monochrome).
        const critical = s.rpd_limit > 0 && (s.rpd_count / s.rpd_limit) >= 0.9;
        const icon = critical ? '<i class="fas fa-triangle-exclamation"></i>' : '';
        return `
            <div class="quota-left${critical ? ' is-critical' : ''}">
                ${icon}${Number(s.rpd_remaining).toLocaleString()} left today
            </div>
        `;
    }

    _meterHtml(s) {
        // No ceiling ⇒ no meter; the absence itself signals "unlimited".
        if (s.rpd_limit === null || s.rpd_limit === undefined || s.rpd_limit === 0) return '';
        const used = Number(s.rpd_count || 0);
        const pct = Math.max(0, Math.min(100, (used / s.rpd_limit) * 100));
        const rounded = pct.toFixed(1);
        return `
            <div class="quota-meter" role="progressbar"
                 aria-valuenow="${used}" aria-valuemin="0" aria-valuemax="${s.rpd_limit}"
                 aria-label="Daily requests used">
                <div class="quota-meter__fill" data-fill="${rounded}%"></div>
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

/**
 * Dashboard Sidebar
 * Collapse/expand for the dashboard section sidebar (components/_dash_sidebar.html).
 * Desktop: toggles a remembered icon rail. Mobile (≤768px): opens a temporary overlay panel.
 * The collapsed class is applied before paint by an inline script in each dashboard's <head>.
 */
(function () {
    const STORAGE_KEY = 'dashSidebar:collapsed';
    const COLLAPSED = 'dash-sidebar-collapsed';
    const OPEN = 'dash-sidebar-open';

    const root = document.documentElement;
    const sidebar = document.getElementById('dashSidebar');
    if (!sidebar) return;

    const toggle = sidebar.querySelector('.dash-sidebar__toggle');
    const tooltip = sidebar.querySelector('.dash-sidebar__tooltip');
    const buttons = sidebar.querySelectorAll('.tab-button');
    const mobile = window.matchMedia('(max-width: 768px)');

    // Labels are visible on desktop-expanded and mobile-open; hidden everywhere else
    function isRail() {
        return mobile.matches ? !root.classList.contains(OPEN) : root.classList.contains(COLLAPSED);
    }

    function sync() {
        const rail = isRail();
        const text = rail ? 'Expand sidebar' : (mobile.matches ? 'Close sidebar' : 'Collapse sidebar');
        // The toggle is icon-only, so its name always lives in aria-label and the tooltip
        toggle.setAttribute('aria-expanded', String(!rail));
        toggle.setAttribute('aria-label', text);
        toggle.setAttribute('data-label', text);
        // Keep an accessible name while the visible text is hidden
        buttons.forEach(btn => {
            if (rail) btn.setAttribute('aria-label', btn.dataset.label);
            else btn.removeAttribute('aria-label');
        });
        hideTooltip();
    }

    // Charts and sub-tab indicators that listen for window resize re-measure after the width settles
    let reflowTimer = null;
    function scheduleReflow() {
        clearTimeout(reflowTimer);
        reflowTimer = setTimeout(() => window.dispatchEvent(new Event('resize')), 220);
    }
    sidebar.addEventListener('transitionend', (e) => {
        if (e.target !== sidebar || e.propertyName !== 'width') return;
        clearTimeout(reflowTimer);
        window.dispatchEvent(new Event('resize'));
    });

    function setOpen(open) {
        root.classList.toggle(OPEN, open);
        sync();
    }

    toggle.addEventListener('click', () => {
        if (mobile.matches) {
            setOpen(!root.classList.contains(OPEN));
            return;
        }
        const collapsed = root.classList.toggle(COLLAPSED);
        try { localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0'); } catch (e) { /* storage unavailable */ }
        sync();
        scheduleReflow();
    });

    // Mobile overlay closes on a pick, Escape, or a click outside the panel
    buttons.forEach(btn => btn.addEventListener('click', () => {
        if (mobile.matches && root.classList.contains(OPEN)) setOpen(false);
    }));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && root.classList.contains(OPEN)) {
            setOpen(false);
            toggle.focus();
        }
    });
    // The scrim is the sidebar's ::before, so an outside click lands on the aside itself
    const panel = sidebar.querySelector('.dash-sidebar__panel');
    document.addEventListener('click', (e) => {
        if (root.classList.contains(OPEN) && !panel.contains(e.target)) setOpen(false);
    });
    mobile.addEventListener('change', () => {
        root.classList.remove(OPEN);
        sync();
        scheduleReflow();
    });

    // Rail tooltip
    function showTooltip(target) {
        if ((!isRail() && target !== toggle) || !target.dataset.label) return;
        const rect = target.getBoundingClientRect();
        tooltip.textContent = target.dataset.label;
        tooltip.style.left = `${rect.right + 8}px`;
        tooltip.style.top = `${rect.top + rect.height / 2}px`;
        tooltip.hidden = false;
    }
    function hideTooltip() {
        tooltip.hidden = true;
    }
    [...buttons, toggle].forEach(el => {
        el.addEventListener('mouseenter', () => showTooltip(el));
        el.addEventListener('mouseleave', hideTooltip);
        el.addEventListener('focusin', () => showTooltip(el));
        el.addEventListener('focusout', hideTooltip);
    });
    sidebar.querySelector('.dash-sidebar__nav').addEventListener('scroll', hideTooltip, { passive: true });

    sync();
})();

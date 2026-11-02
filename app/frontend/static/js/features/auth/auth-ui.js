// ---------------------------------------------------------------------------
// AuthUI: one message channel for the auth pages (login / signup / admin login
// / account linking). Renders into the shared #authAlert slot
// (templates/components/_auth_alert.html) with the same markup the server
// uses, and provides a fetch helper with CSRF + safe JSON parsing, since
// core/base.js is not loaded on auth pages.
//
// All message text is inserted with textContent - never innerHTML.
// ---------------------------------------------------------------------------
(function () {
    'use strict';

    // Mirrors app/auth/flash.py ACTIONS.
    const ACTION_LABELS = {
        withdraw_pending: 'Withdraw signup & delete my data',
    };

    const LEVELS = ['success', 'info', 'warning', 'danger'];

    const ICONS = {
        success: 'fa-check-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-circle',
        danger: 'fa-exclamation-triangle',
    };

    const GENERIC_ERROR = 'Something went wrong. Please try again.';
    const NETWORK_ERROR = 'Network error. Please check your connection and try again.';

    let hideTimer = null;

    function slot() {
        return document.getElementById('authAlert');
    }

    function readCookie(name) {
        const prefix = name + '=';
        for (const part of document.cookie.split(';')) {
            const c = part.trim();
            if (c.startsWith(prefix)) {
                try {
                    return decodeURIComponent(c.substring(prefix.length));
                } catch (e) {
                    return c.substring(prefix.length);
                }
            }
        }
        return null;
    }

    function normalizeAction(action) {
        if (typeof action === 'string') {
            if (!Object.prototype.hasOwnProperty.call(ACTION_LABELS, action)) return null;
            return { key: action, label: ACTION_LABELS[action] };
        }
        if (action && typeof action.key === 'string') {
            return { key: action.key, label: String(action.label || ACTION_LABELS[action.key] || action.key) };
        }
        return null;
    }

    function clearTimer() {
        if (hideTimer) {
            clearTimeout(hideTimer);
            hideTimer = null;
        }
    }

    function clear() {
        clearTimer();
        const el = slot();
        if (el) el.replaceChildren();
    }

    function show(message, opts) {
        const options = opts || {};
        const level = LEVELS.includes(options.level) ? options.level : 'danger';
        const actions = (options.actions || []).map(normalizeAction).filter(Boolean);
        const source = options.source == null ? '' : String(options.source);
        const autoHideMs = options.autoHideMs;

        const el = slot();
        if (!el) return null;
        clearTimer();

        const alert = document.createElement('div');
        alert.className = 'alert alert-' + level;
        alert.setAttribute('role', (level === 'danger' || level === 'warning') ? 'alert' : 'status');
        alert.dataset.source = source;

        const icon = document.createElement('i');
        icon.className = 'fas ' + (ICONS[level] || ICONS.info) + ' auth-alert-icon';
        icon.setAttribute('aria-hidden', 'true');
        alert.appendChild(icon);

        const body = document.createElement('div');
        body.className = 'auth-alert-body';
        const msg = document.createElement('span');
        msg.className = 'auth-alert-message';
        msg.textContent = message == null ? '' : String(message);
        body.appendChild(msg);

        if (actions.length) {
            const row = document.createElement('div');
            row.className = 'auth-alert-actions';
            for (const a of actions) {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'btn btn-sm btn-outline-' +
                    (a.key === 'withdraw_pending' ? 'danger' : 'secondary') + ' auth-alert-action';
                btn.dataset.authAction = a.key;
                btn.textContent = a.label;
                row.appendChild(btn);
            }
            body.appendChild(row);
        }
        alert.appendChild(body);

        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'btn-close auth-alert-close';
        close.setAttribute('aria-label', 'Close');
        alert.appendChild(close);

        el.replaceChildren(alert);

        // Errors and warnings stay until the user acts on them.
        if (autoHideMs && autoHideMs > 0 && level !== 'danger' && level !== 'warning') {
            hideTimer = setTimeout(function () {
                hideTimer = null;
                if (alert.parentNode) alert.remove();
            }, autoHideMs);
        }
        return alert;
    }

    function formatDetail(detail) {
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
            const msgs = detail
                .map(function (item) {
                    if (typeof item === 'string') return item;
                    return item && typeof item.msg === 'string' ? item.msg : '';
                })
                .filter(Boolean);
            return msgs.length ? msgs.join(' ') : GENERIC_ERROR;
        }
        if (detail && typeof detail === 'object') {
            if (typeof detail.message === 'string' && detail.message) return detail.message;
            if (typeof detail.detail === 'string' && detail.detail) return detail.detail;
            return GENERIC_ERROR;
        }
        return GENERIC_ERROR;
    }

    function fieldErrorsFrom(detail, headers, message) {
        const errors = {};
        if (Array.isArray(detail)) {
            for (const item of detail) {
                if (!item || !Array.isArray(item.loc) || !item.loc.length) continue;
                const field = item.loc[item.loc.length - 1];
                if (typeof field !== 'string' || field === 'body') continue;
                if (!errors[field] && typeof item.msg === 'string') errors[field] = item.msg;
            }
        }
        const headerField = headers && headers.get('X-Error-Field');
        if (headerField && message) errors[headerField] = message;
        return errors;
    }

    async function request(url, opts) {
        const options = opts || {};
        const method = (options.method || 'POST').toUpperCase();
        const headers = { 'Accept': 'application/json' };
        const init = { method: method, credentials: 'same-origin', headers: headers };
        if (options.json !== undefined) {
            headers['Content-Type'] = 'application/json';
            init.body = JSON.stringify(options.json);
        }
        if (method !== 'GET' && method !== 'HEAD') {
            const csrf = readCookie('csrf_token');
            if (csrf) headers['X-CSRF-Token'] = csrf;
        }

        let response;
        try {
            response = await fetch(url, init);
        } catch (e) {
            return { ok: false, status: 0, data: null, headers: null, message: NETWORK_ERROR, fieldErrors: {} };
        }

        let data = null;
        let parsed = false;
        try {
            const text = await response.text();
            if (text) {
                data = JSON.parse(text);
                parsed = true;
            } else {
                parsed = true;
            }
        } catch (e) {
            parsed = false;
        }

        let message = '';
        if (!parsed) {
            message = 'Server error (HTTP ' + response.status + '). Please try again.';
        } else if (!response.ok) {
            message = data && data.detail !== undefined ? formatDetail(data.detail) : formatDetail(data);
            if (!message || message === GENERIC_ERROR) {
                if (response.status >= 500) {
                    message = 'Server error (HTTP ' + response.status + '). Please try again.';
                }
            }
        } else if (data && typeof data.message === 'string') {
            message = data.message;
        }

        const fieldErrors = response.ok ? {} : fieldErrorsFrom(data && data.detail, response.headers, message);

        return {
            ok: response.ok && parsed,
            status: response.status,
            data: parsed ? data : null,
            headers: response.headers,
            message: message,
            fieldErrors: fieldErrors,
        };
    }

    // Delegated handlers: action buttons (server- or client-rendered) and close.
    document.addEventListener('click', function (event) {
        const target = event.target;
        if (!(target instanceof Element)) return;

        const actionBtn = target.closest('#authAlert [data-auth-action]');
        if (actionBtn) {
            const alert = actionBtn.closest('.alert');
            const source = alert && alert.dataset.source ? alert.dataset.source : null;
            document.dispatchEvent(new CustomEvent('authui:action', {
                detail: { action: actionBtn.dataset.authAction, source: source },
            }));
            return;
        }

        const closeBtn = target.closest('#authAlert .auth-alert-close');
        if (closeBtn) {
            const alert = closeBtn.closest('.alert');
            if (alert) {
                clearTimer();
                alert.remove();
            }
        }
    });

    window.AuthUI = {
        show: show,
        clear: clear,
        formatDetail: formatDetail,
        request: request,
    };
})();

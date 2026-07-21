// ---------------------------------------------------------------------------
// CSRF (double-submit cookie): the backend issues a JS-readable `csrf_token`
// cookie and requires its value echoed in the `X-CSRF-Token` header on every
// state-changing management request. Rather than touch every call site, we wrap
// window.fetch once so all same-origin unsafe requests carry the header
// automatically. Cross-origin requests are left untouched (they'd fail the
// same-origin cookie read anyway and could trigger needless CORS preflights).
// ---------------------------------------------------------------------------
(function installCsrfFetch() {
    function readCookie(name) {
        const prefix = name + '=';
        for (const part of document.cookie.split(';')) {
            const c = part.trim();
            if (c.startsWith(prefix)) {
                return decodeURIComponent(c.substring(prefix.length));
            }
        }
        return null;
    }

    function isSameOrigin(input) {
        try {
            const url = (typeof input === 'string') ? input : (input && input.url) || '';
            if (url.startsWith('/') && !url.startsWith('//')) return true;  // relative path
            return new URL(url, window.location.origin).origin === window.location.origin;
        } catch (e) {
            return false;
        }
    }

    const nativeFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        init = init || {};
        const method = (
            (init && init.method) ||
            (typeof input === 'object' && input && input.method) ||
            'GET'
        ).toUpperCase();
        if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method) && isSameOrigin(input)) {
            const token = readCookie('csrf_token');
            if (token) {
                const headers = new Headers(
                    (init && init.headers) ||
                    (typeof input === 'object' && input && input.headers) ||
                    {}
                );
                if (!headers.has('X-CSRF-Token')) {
                    headers.set('X-CSRF-Token', token);
                }
                init.headers = headers;
            }
        }
        return nativeFetch(input, init);
    };
})();

// Global JavaScript functions
async function logout() {
    try {
        // Call logout endpoint to clear server-side cookie
        await fetch('/auth/logout', {
            method: 'POST',
            credentials: 'include'
        });

        // Also try admin logout in case user is admin
        await fetch('/admin/logout', {
            method: 'POST',
            credentials: 'include'
        });
    } catch (error) {
        console.log('Error during logout:', error);
    }

    // The JWT lives only in the server-set HttpOnly `access_token` cookie, which
    // the logout endpoints above clear. Nothing to remove client-side.

    // Context-aware redirect based on current page
    const currentPath = window.location.pathname;
    if (currentPath.startsWith('/admin/')) {
        window.location.href = '/admin';
    } else {
        window.location.href = '/login';
    }
}

function makeAuthenticatedRequest(url, options = {}) {
    // Prevent requests during profile update to avoid token conflicts
    if (window.profileUpdateInProgress && !url.includes('/auth/profile')) {
        console.log('Blocking request during profile update:', url);
        return Promise.reject(new Error('Profile update in progress'));
    }

    // Prevent ALL requests after successful profile update
    if (window.profileUpdateSuccess) {
        console.log('Blocking request after profile update success:', url);
        return Promise.reject(new Error('Profile updated, redirecting to login'));
    }

    // Authentication is carried by the server-set HttpOnly `access_token` cookie.
    // We intentionally do NOT persist the JWT in localStorage or attach a bearer
    // header from JS — keeping the token out of JS-readable storage means an XSS
    // can't exfiltrate it. The backend auth middleware falls back to the cookie
    // when no Authorization header is present, so credentials:'include' suffices.
    //
    // CSRF: state-changing requests are protected by the double-submit cookie
    // scheme. The window.fetch wrapper installed above automatically attaches the
    // X-CSRF-Token header (read from the csrf_token cookie) to same-origin unsafe
    // requests, so callers here need only send credentials.
    options.credentials = 'include';
    return fetch(url, options);
}

// Create toast container if it doesn't exist
function ensureToastContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    return container;
}

// Show toast notifications — delegates to UIUtils.showToast if loaded, else Bootstrap fallback
function showAlert(message, type = 'info') {
    // Normalize 'danger' → 'error' for UIUtils.showToast compatibility
    const normalizedType = type === 'danger' ? 'error' : type;

    if (window.UIUtils && window.UIUtils.showToast) {
        window.UIUtils.showToast(message, normalizedType);
        return;
    }

    // Bootstrap fallback (used on pages that don't load ui-utils.js)
    const container = ensureToastContainer();
    const toastDiv = document.createElement('div');
    toastDiv.className = 'toast show';
    toastDiv.setAttribute('role', 'alert');
    toastDiv.setAttribute('aria-live', 'assertive');
    toastDiv.setAttribute('aria-atomic', 'true');

    const typeConfig = {
        'success': { bgClass: 'bg-success', icon: 'fas fa-check-circle', textClass: 'text-white' },
        'error':   { bgClass: 'bg-danger',  icon: 'fas fa-exclamation-circle', textClass: 'text-white' },
        'warning': { bgClass: 'bg-warning', icon: 'fas fa-exclamation-triangle', textClass: 'text-dark' },
        'info':    { bgClass: 'bg-info',    icon: 'fas fa-info-circle', textClass: 'text-white' }
    };

    const config = typeConfig[normalizedType] || typeConfig['info'];

    toastDiv.innerHTML = `
        <div class="toast-header ${config.bgClass} ${config.textClass}">
            <i class="${config.icon} me-2"></i>
            <strong class="me-auto">Notification</strong>
            <button type="button" class="btn-close ${config.textClass === 'text-white' ? 'btn-close-white' : ''}" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">${message}</div>
    `;

    container.appendChild(toastDiv);
    const bsToast = new bootstrap.Toast(toastDiv, { autohide: true, delay: 4000 });
    bsToast.show();
    toastDiv.addEventListener('hidden.bs.toast', () => toastDiv.remove());
}

// Local HTML escaper (same semantics as UIUtils.escapeHtml in ui-utils.js).
// base.js is loaded on every page but ui-utils.js is not, so we must not rely
// on window.UIUtils being present when escaping user-controlled values.
function baseEscapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}

// Update navigation bar based on authentication status
async function updateNavbar() {
    const navbarAuth = document.getElementById('navbarAuth');

    try {
        // Check if user is authenticated by calling /auth/me (authenticated via the HttpOnly cookie)
        const response = await makeAuthenticatedRequest('/auth/me');
        if (response.ok) {
            const user = await response.json();
            const username = baseEscapeHtml(user.username);
            const initial = baseEscapeHtml((user.username || '').charAt(0).toUpperCase());
            navbarAuth.innerHTML = `
                <div class="dropdown">
                    <button class="btn user-dropdown-toggle dropdown-toggle" type="button"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        <span class="user-avatar">${initial}</span>
                        <span class="user-name">${username}</span>
                    </button>
                    <ul class="dropdown-menu dropdown-menu-end user-dropdown-menu">
                        <li class="dropdown-header-item">
                            <span class="user-avatar user-avatar-lg">${initial}</span>
                            <div>
                                <div class="dropdown-username">${username}</div>
                                <div class="dropdown-role">${user.is_admin ? 'Administrator' : 'User'}</div>
                            </div>
                        </li>
                        <li><hr class="dropdown-divider"></li>
                        ${!user.is_admin ? '<li><a class="dropdown-item" href="/dashboard/profile"><i class="fas fa-user-circle me-2"></i>Profile Settings</a></li>' : ''}
                        ${!user.is_admin ? '<li><hr class="dropdown-divider"></li>' : ''}
                        <li><a class="dropdown-item dropdown-item-danger" href="#" onclick="logout(); return false;"><i class="fas fa-sign-out-alt me-2"></i>Sign Out</a></li>
                    </ul>
                </div>
            `;
        } else {
            navbarAuth.innerHTML = ``;
        }
    } catch (error) {
        // Network error or invalid token
        navbarAuth.innerHTML = ``;
    }
}

// Update navbar on page load
document.addEventListener('DOMContentLoaded', function () {
    updateNavbar();
});
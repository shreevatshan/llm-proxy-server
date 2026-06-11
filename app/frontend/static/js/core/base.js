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

    // Clear client-side token
    localStorage.removeItem('access_token');

    // Context-aware redirect based on current page
    const currentPath = window.location.pathname;
    if (currentPath.startsWith('/admin/')) {
        window.location.href = '/admin';
    } else {
        window.location.href = '/login';
    }
}

function getAuthToken() {
    return localStorage.getItem('access_token');
}

function setAuthToken(token) {
    localStorage.setItem('access_token', token);
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
    
    const token = getAuthToken();
    if (token) {
        options.headers = {
            ...options.headers,
            'Authorization': `Bearer ${token}`
        };
    }
    // Always include credentials to send HTTP-only cookies
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

// Update navigation bar based on authentication status
async function updateNavbar() {
    const navbarAuth = document.getElementById('navbarAuth');

    try {
        // Check if user is authenticated by calling /auth/me (works with both localStorage token and HTTP-only cookies)
        const response = await makeAuthenticatedRequest('/auth/me');
        if (response.ok) {
            const user = await response.json();
            navbarAuth.innerHTML = `
                <div class="dropdown">
                    <button class="btn user-dropdown-toggle dropdown-toggle" type="button"
                            data-bs-toggle="dropdown" aria-expanded="false">
                        <span class="user-avatar">${user.username.charAt(0).toUpperCase()}</span>
                        <span class="user-name">${user.username}</span>
                    </button>
                    <ul class="dropdown-menu dropdown-menu-end user-dropdown-menu">
                        <li class="dropdown-header-item">
                            <span class="user-avatar user-avatar-lg">${user.username.charAt(0).toUpperCase()}</span>
                            <div>
                                <div class="dropdown-username">${user.username}</div>
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
            // Not authenticated - remove any stored token
            localStorage.removeItem('access_token');
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
// Login page. All messages go through window.AuthUI (features/auth/auth-ui.js)
// into the shared #authAlert slot; server-rendered flashes (e.g. from the
// ZOHO callback) arrive in the same slot.

const loginForm = document.getElementById('loginForm');
const loginSubmitBtn = loginForm.querySelector('button[type="submit"]');

// Only same-origin relative paths; the server already validates `next`.
function localPath(path, fallback) {
    return (typeof path === 'string' && path.startsWith('/') && !path.startsWith('//') && !path.includes('\\'))
        ? path
        : fallback;
}

function setBusy(button, busy, busyHtml) {
    if (busy) {
        button.dataset.originalHtml = button.innerHTML;
        button.disabled = true;
        button.innerHTML = busyHtml;
    } else {
        button.disabled = false;
        if (button.dataset.originalHtml !== undefined) {
            button.innerHTML = button.dataset.originalHtml;
            delete button.dataset.originalHtml;
        }
    }
}

loginForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    if (loginSubmitBtn.disabled) return;

    const formData = new FormData(this);
    const username = formData.get('username') || '';
    const password = formData.get('password') || '';

    if (!username.trim() || !password) {
        AuthUI.show('Please enter your username and password.', { level: 'danger' });
        return;
    }

    AuthUI.clear();
    setBusy(loginSubmitBtn, true, '<i class="fas fa-spinner fa-spin me-2"></i>Signing in...');

    // The server sets the JWT as an HttpOnly `access_token` cookie; no
    // client-side token storage is needed.
    const res = await AuthUI.request('/auth/login', { json: { username: username, password: password } });

    if (res.ok) {
        AuthUI.show('Signed in. Redirecting…', { level: 'success' });
        window.location.assign(localPath(this.dataset.next, '/dashboard/'));
        return;
    }

    setBusy(loginSubmitBtn, false);
    const message = res.message || 'Login failed. Please try again.';
    const accountStatus = res.status === 403 && res.headers ? res.headers.get('X-Account-Status') : null;
    if (accountStatus === 'pending') {
        AuthUI.show(message, { level: 'warning', actions: ['withdraw_pending'] });
    } else if (accountStatus === 'deactivated') {
        AuthUI.show(message, { level: 'warning' });
    } else {
        AuthUI.show(message, { level: 'danger' });
    }
});

// ---------------------------------------------------------------------------
// Withdraw a pending signup (alert action "withdraw_pending").
// source === 'oauth': the server holds a short-lived pending_withdraw cookie
// from the ZOHO callback, so no credentials are needed. Otherwise the user
// confirms with the username from the login form and their password.
// ---------------------------------------------------------------------------
const withdrawModalEl = document.getElementById('withdrawModal');
const withdrawForm = document.getElementById('withdrawForm');
const withdrawPasswordGroup = document.getElementById('withdrawPasswordGroup');
const withdrawPassword = document.getElementById('withdrawPassword');
const withdrawPasswordError = document.getElementById('withdrawPassword-error');
const withdrawConfirmBtn = document.getElementById('withdrawConfirmBtn');

// The modal sits inside an animated (transformed) container; move it to <body>
// so Bootstrap's fixed positioning and backdrop work.
document.body.appendChild(withdrawModalEl);
const withdrawModal = bootstrap.Modal.getOrCreateInstance(withdrawModalEl);

let withdrawSource = null;

function setWithdrawPasswordError(message) {
    if (message) {
        withdrawPassword.classList.add('is-invalid');
        withdrawPasswordError.textContent = message;
    } else {
        withdrawPassword.classList.remove('is-invalid');
        withdrawPasswordError.textContent = '';
    }
}

function openWithdrawModal(source) {
    const isOauth = source === 'oauth';
    if (!isOauth && !document.getElementById('username').value.trim()) {
        AuthUI.show('Enter your username in the login form first, then choose withdraw again.',
            { level: 'warning', actions: ['withdraw_pending'], source: source });
        document.getElementById('username').focus();
        return;
    }
    withdrawSource = source;
    withdrawPasswordGroup.classList.toggle('d-none', isOauth);
    withdrawPassword.value = '';
    setWithdrawPasswordError('');
    withdrawModal.show();
}

withdrawModalEl.addEventListener('shown.bs.modal', function () {
    if (!withdrawPasswordGroup.classList.contains('d-none')) {
        withdrawPassword.focus();
    } else {
        withdrawConfirmBtn.focus();
    }
});

withdrawPassword.addEventListener('input', function () {
    if (this.value) setWithdrawPasswordError('');
});

document.addEventListener('authui:action', function (event) {
    const detail = event.detail || {};
    if (detail.action === 'withdraw_pending') {
        openWithdrawModal(detail.source || null);
    }
});

withdrawForm.addEventListener('submit', async function (e) {
    e.preventDefault();
    if (withdrawConfirmBtn.disabled) return;

    const source = withdrawSource;
    const isOauth = source === 'oauth';
    let body = {};
    if (!isOauth) {
        const username = document.getElementById('username').value;
        const password = withdrawPassword.value;
        if (!password) {
            setWithdrawPasswordError('Password is required');
            withdrawPassword.focus();
            return;
        }
        body = { username: username, password: password };
    }

    setBusy(withdrawConfirmBtn, true, '<i class="fas fa-spinner fa-spin me-2"></i>Deleting...');
    const res = await AuthUI.request('/auth/pending/withdraw', { json: body });

    if (res.ok) {
        // The server sets a one-time success flash shown on the next page.
        window.location.assign(localPath(res.data && res.data.redirect, '/login'));
        return;
    }

    setBusy(withdrawConfirmBtn, false);
    withdrawPassword.value = '';
    withdrawModal.hide();
    // Offer a retry for wrong credentials and transient failures; an expired
    // withdraw option (400) or a non-pending account (409) cannot be retried here.
    const retryable = res.status === 401 || res.status === 0 || res.status >= 500;
    AuthUI.show(res.message || 'Could not withdraw the signup. Please try again.', {
        level: 'danger',
        actions: retryable ? ['withdraw_pending'] : [],
        source: source,
    });
});

// ZOHO OAuth login handler
document.getElementById('zohoLoginBtn').addEventListener('click', async function () {
    const res = await AuthUI.request('/auth/zoho/status', { method: 'GET' });
    if (!res.ok || !res.data) {
        AuthUI.show(res.message || 'Failed to connect to ZOHO. Please try again.', { level: 'danger' });
        return;
    }
    if (!res.data.available) {
        AuthUI.show(res.data.message || 'ZOHO sign-in is not available.', { level: 'warning' });
        return;
    }
    window.location.assign('/auth/zoho/login');
});

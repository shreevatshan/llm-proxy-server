// ---------------------------------------------------------------------------
// Link ZOHO account page: confirm the link with the existing account's
// password, or decline. Both endpoints answer {redirect}; messages go through
// the shared alert slot (window.AuthUI).
// ---------------------------------------------------------------------------
(function () {
    'use strict';

    const form = document.getElementById('linkAccountForm');
    const passwordInput = document.getElementById('linkPassword');
    const linkBtn = document.getElementById('linkAccountBtn');
    const declineBtn = document.getElementById('declineLinkBtn');
    if (!form || !passwordInput || !linkBtn || !declineBtn || !window.AuthUI) return;

    let busy = false;

    function setBusy(value) {
        busy = value;
        linkBtn.disabled = value;
        declineBtn.disabled = value;
    }

    // Only same-site relative paths from the server are followed.
    function navigate(data, fallback) {
        const target = data && typeof data.redirect === 'string' ? data.redirect : '';
        const safe = target.startsWith('/') && !target.startsWith('//') && !target.includes('\\');
        window.location.assign(safe ? target : fallback);
    }

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        if (busy) return;
        window.AuthUI.clear();

        if (!passwordInput.value) {
            window.AuthUI.show('Please enter the password of the existing account.', { level: 'warning' });
            passwordInput.focus();
            return;
        }

        setBusy(true);
        const result = await window.AuthUI.request('/auth/zoho/link', {
            json: { password: passwordInput.value },
        });
        if (result.ok) {
            navigate(result.data, '/dashboard/');
            return; // stay disabled while the page changes
        }
        setBusy(false);
        window.AuthUI.show(result.message || 'Could not link the accounts. Please try again.', { level: 'danger' });
        if (result.status === 401) {
            passwordInput.value = '';
            passwordInput.focus();
        }
    });

    declineBtn.addEventListener('click', async function () {
        if (busy) return;
        window.AuthUI.clear();
        setBusy(true);
        const result = await window.AuthUI.request('/auth/zoho/link/decline');
        if (result.ok) {
            navigate(result.data, '/login');
            return;
        }
        setBusy(false);
        window.AuthUI.show(result.message || 'Something went wrong. Please try again.', { level: 'danger' });
    });
})();

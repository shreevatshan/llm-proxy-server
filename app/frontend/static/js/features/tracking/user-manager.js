/**
 * User Management Module
 * Handles user-related operations like activate, deactivate, and remove
 */

class UserManager {
    filterUsers(query) {
        const q = (query || '').trim().toLowerCase();
        const rows = document.querySelectorAll('.user-mgmt-row');
        let visible = 0;
        rows.forEach(row => {
            const username = row.dataset.username || '';
            const email = row.dataset.email || '';
            const match = !q || username.includes(q) || email.includes(q);
            row.style.display = match ? '' : 'none';
            if (match) visible++;
        });
        const noMatch = document.getElementById('user-mgmt-no-match');
        if (noMatch) noMatch.style.display = (rows.length && visible === 0) ? '' : 'none';
    }

    async deactivateUser(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');

        const confirmed = await window.UIUtils.showConfirmModal(
            'Deactivate User',
            `Are you sure you want to deactivate user "${username}"?`
        );
        if (!confirmed) return;

        try {
            const response = await fetch(`/admin/users?user_id=${userId}`, {
                method: 'DELETE',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                window.UIUtils.showToast('User deactivated successfully!', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error deactivating user: ' + error.message, 'error');
        }
    }

    async activateUser(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');

        const confirmed = await window.UIUtils.showConfirmModal(
            'Activate User',
            `Are you sure you want to activate user "${username}"?`
        );
        if (!confirmed) return;

        try {
            const response = await fetch(`/admin/users/activate?user_id=${userId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                window.UIUtils.showToast('User activated successfully!', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error activating user: ' + error.message, 'error');
        }
    }

    async approveUser(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');

        const confirmed = await window.UIUtils.showConfirmModal(
            'Approve User',
            `Approve and activate the account for "${username}"? They will be able to log in immediately.`
        );
        if (!confirmed) return;

        try {
            const response = await fetch(`/admin/users/approve?user_id=${userId}`, {
                method: 'PUT',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                window.UIUtils.showToast('User approved and activated!', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error approving user: ' + error.message, 'error');
        }
    }

    async removeUser(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');

        const confirmed = await window.UIUtils.showConfirmModal(
            'Permanently Delete User',
            `⚠️ WARNING: This will permanently delete user "${username}" and all associated data. This action cannot be undone!\n\nAre you absolutely sure you want to proceed?`,
            'danger'
        );
        if (!confirmed) return;

        try {
            const response = await fetch(`/admin/users/permanent?user_id=${userId}`, {
                method: 'DELETE',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (response.ok) {
                window.UIUtils.showToast('User permanently removed successfully!', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error removing user: ' + error.message, 'error');
        }
    }

    async modifyUser(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');
        const email = button.getAttribute('data-email') || '';
        const isOauth = !!button.getAttribute('data-oauth');

        // Grab modal elements
        const modal = document.getElementById('modifyUserModal');
        const usernameInput = document.getElementById('modifyUsernameInput');
        const emailInput = document.getElementById('modifyEmailInput');
        const passwordInput = document.getElementById('modifyPasswordInput');
        const confirmPasswordInput = document.getElementById('modifyConfirmPasswordInput');
        const nonOauthFields = document.getElementById('modifyNonOauthFields');
        const errorDiv = document.getElementById('modifyUserError');
        const confirmBtn = document.getElementById('modifyUserConfirmBtn');

        // Prefill current values
        usernameInput.value = username;
        emailInput.value = email;
        passwordInput.value = '';
        confirmPasswordInput.value = '';
        errorDiv.classList.add('d-none');

        // OAuth users may only change their username; hide email/password fields.
        nonOauthFields.style.display = isOauth ? 'none' : '';

        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Handle save confirmation
        const handleConfirm = async () => {
            const newUsername = usernameInput.value.trim();

            // Clear previous errors
            errorDiv.classList.add('d-none');

            if (!newUsername) {
                errorDiv.textContent = 'Username cannot be empty';
                errorDiv.classList.remove('d-none');
                return;
            }

            // Build request body with only the fields that apply.
            const body = { username: newUsername };

            if (!isOauth) {
                const newEmail = emailInput.value.trim();
                if (!newEmail) {
                    errorDiv.textContent = 'Email cannot be empty';
                    errorDiv.classList.remove('d-none');
                    return;
                }
                body.email = newEmail;

                const newPassword = passwordInput.value;
                const confirmPassword = confirmPasswordInput.value;
                if (newPassword || confirmPassword) {
                    if (newPassword.length < 6) {
                        errorDiv.textContent = 'Password must be at least 6 characters long';
                        errorDiv.classList.remove('d-none');
                        return;
                    }
                    if (newPassword !== confirmPassword) {
                        errorDiv.textContent = 'Passwords do not match';
                        errorDiv.classList.remove('d-none');
                        return;
                    }
                    body.new_password = newPassword;
                }
            }

            // Disable button during request
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';

            try {
                const response = await fetch(`/admin/users/modify?user_id=${userId}`, {
                    method: 'PUT',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(body)
                });

                if (response.ok) {
                    window.UIUtils.showToast('User updated successfully!', 'success');
                    bsModal.hide();
                    setTimeout(() => location.reload(), 1000);
                } else {
                    const error = await response.json();
                    errorDiv.textContent = error.detail || 'Failed to modify user';
                    errorDiv.classList.remove('d-none');
                }
            } catch (error) {
                errorDiv.textContent = 'Error modifying user: ' + error.message;
                errorDiv.classList.remove('d-none');
            } finally {
                // Re-enable button
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = '<i class="fas fa-save me-1"></i>Save Changes';
            }
        };

        // Add event listener for confirm button
        confirmBtn.onclick = handleConfirm;

        // Handle Enter key in inputs
        const handleEnter = (e) => {
            if (e.key === 'Enter') {
                handleConfirm();
            }
        };
        usernameInput.onkeypress = handleEnter;
        emailInput.onkeypress = handleEnter;
        passwordInput.onkeypress = handleEnter;
        confirmPasswordInput.onkeypress = handleEnter;

        // Focus on first input when modal is shown
        modal.addEventListener('shown.bs.modal', () => {
            usernameInput.focus();
        }, { once: true });

        // Clean up event listeners when modal is hidden
        modal.addEventListener('hidden.bs.modal', () => {
            confirmBtn.onclick = null;
            usernameInput.onkeypress = null;
            emailInput.onkeypress = null;
            passwordInput.onkeypress = null;
            confirmPasswordInput.onkeypress = null;
        }, { once: true });
    }
}

// Create global instance
window.UserManager = new UserManager();

// Export functions for backward compatibility
window.deactivateUser = (button) => window.UserManager.deactivateUser(button);
window.activateUser = (button) => window.UserManager.activateUser(button);
window.approveUser = (button) => window.UserManager.approveUser(button);
window.removeUser = (button) => window.UserManager.removeUser(button);
window.modifyUser = (button) => window.UserManager.modifyUser(button);
window.filterUsers = (query) => window.UserManager.filterUsers(query);

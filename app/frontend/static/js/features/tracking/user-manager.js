/**
 * User Management Module
 * Handles user-related operations like activate, deactivate, and remove
 */

class UserManager {
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

    async resetPassword(button) {
        const userId = button.getAttribute('data-user-id');
        const username = button.getAttribute('data-username');

        // Show password reset modal
        const modal = document.getElementById('passwordResetModal');
        const usernameDisplay = document.getElementById('passwordResetUsername');
        const newPasswordInput = document.getElementById('newPasswordInput');
        const confirmPasswordInput = document.getElementById('confirmPasswordInput');
        const errorDiv = document.getElementById('passwordResetError');
        const confirmBtn = document.getElementById('passwordResetConfirmBtn');

        // Set username
        usernameDisplay.textContent = username;

        // Clear previous values
        newPasswordInput.value = '';
        confirmPasswordInput.value = '';
        errorDiv.classList.add('d-none');

        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Handle password reset confirmation
        const handleConfirm = async () => {
            const newPassword = newPasswordInput.value;
            const confirmPassword = confirmPasswordInput.value;

            // Clear previous errors
            errorDiv.classList.add('d-none');

            // Validate password length
            if (newPassword.length < 6) {
                errorDiv.textContent = 'Password must be at least 6 characters long';
                errorDiv.classList.remove('d-none');
                return;
            }

            // Validate passwords match
            if (newPassword !== confirmPassword) {
                errorDiv.textContent = 'Passwords do not match';
                errorDiv.classList.remove('d-none');
                return;
            }

            // Disable button during request
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Resetting...';

            try {
                const response = await fetch(`/admin/users/reset-password?user_id=${userId}`, {
                    method: 'PUT',
                    credentials: 'include',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        new_password: newPassword
                    })
                });

                if (response.ok) {
                    window.UIUtils.showToast('Password reset successfully!', 'success');
                    bsModal.hide();
                } else {
                    const error = await response.json();
                    errorDiv.textContent = error.detail || 'Failed to reset password';
                    errorDiv.classList.remove('d-none');
                }
            } catch (error) {
                errorDiv.textContent = 'Error resetting password: ' + error.message;
                errorDiv.classList.remove('d-none');
            } finally {
                // Re-enable button
                confirmBtn.disabled = false;
                confirmBtn.innerHTML = '<i class="fas fa-key me-1"></i>Reset Password';
            }
        };

        // Add event listener for confirm button
        confirmBtn.onclick = handleConfirm;

        // Handle Enter key in password inputs
        const handleEnter = (e) => {
            if (e.key === 'Enter') {
                handleConfirm();
            }
        };
        newPasswordInput.onkeypress = handleEnter;
        confirmPasswordInput.onkeypress = handleEnter;

        // Focus on first input when modal is shown
        modal.addEventListener('shown.bs.modal', () => {
            newPasswordInput.focus();
        }, { once: true });

        // Clean up event listeners when modal is hidden
        modal.addEventListener('hidden.bs.modal', () => {
            confirmBtn.onclick = null;
            newPasswordInput.onkeypress = null;
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
window.resetPassword = (button) => window.UserManager.resetPassword(button);

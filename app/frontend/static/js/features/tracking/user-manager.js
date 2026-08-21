/**
 * User Management Module
 * Handles user-related operations like activate, deactivate, and remove
 */

class UserManager {
    constructor() {
        // Selection is held here rather than read off the DOM so that selections
        // survive the search filter hiding and re-showing rows.
        this._selectedIds = new Set();
    }

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
        this._syncSelectionUI();
    }

    // ==================== Bulk selection ====================

    /** Checkboxes on rows the search filter is currently showing. */
    _visibleChecks() {
        return Array.from(document.querySelectorAll('.user-mgmt-check'))
            .filter(cb => {
                const row = cb.closest('.user-mgmt-row');
                return row && row.style.display !== 'none';
            });
    }

    toggleUserSelection(checkbox) {
        const userId = checkbox.getAttribute('data-user-id');
        if (checkbox.checked) {
            this._selectedIds.add(userId);
        } else {
            this._selectedIds.delete(userId);
        }
        this._syncSelectionUI();
    }

    toggleSelectAll(checkbox) {
        // Only touches visible rows, so selecting all while a search is active
        // does not silently sweep in users the admin cannot see.
        this._visibleChecks().forEach(cb => {
            cb.checked = checkbox.checked;
            const userId = cb.getAttribute('data-user-id');
            if (checkbox.checked) {
                this._selectedIds.add(userId);
            } else {
                this._selectedIds.delete(userId);
            }
        });
        this._syncSelectionUI();
    }

    clearSelection() {
        this._selectedIds.clear();
        document.querySelectorAll('.user-mgmt-check').forEach(cb => { cb.checked = false; });
        this._syncSelectionUI();
    }

    _syncSelectionUI() {
        const count = this._selectedIds.size;

        const bar = document.getElementById('user-bulk-actions');
        if (bar) bar.style.display = count > 0 ? '' : 'none';

        const status = document.getElementById('user-bulk-count');
        

        // Tri-state header checkbox, measured against the visible rows only.
        const selectAll = document.getElementById('user-mgmt-select-all');
        if (selectAll) {
            const visible = this._visibleChecks();
            const selectedVisible = visible.filter(cb => cb.checked).length;
            selectAll.checked = visible.length > 0 && selectedVisible === visible.length;
            selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visible.length;
        }

        document.querySelectorAll('.user-mgmt-row').forEach(row => {
            row.classList.toggle('selected', this._selectedIds.has(row.dataset.userId));
        });

        // Once anything is selected the bulk bar owns the actions, so the row
        // buttons stand down — otherwise a row button next to a ticked checkbox
        // reads as "act on the selection" while doing something narrower. A
        // disabled button cannot show a tooltip (pointer-events: none), so the
        // status text above points at the bar instead.
        document.querySelectorAll('.user-row-action').forEach(btn => {
            btn.disabled = count > 0;
        });
    }

    async _bulkAction(action, { title, message, type }) {
        if (this._selectedIds.size === 0) {
            window.UIUtils.showToast('Select at least one user first.', 'warning');
            return;
        }

        const confirmed = await window.UIUtils.showConfirmModal(title, message, type);
        if (!confirmed) return;

        const userIds = [...this._selectedIds].map(id => parseInt(id, 10));

        try {
            const response = await fetch('/admin/users/bulk-action', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_ids: userIds, action })
            });

            if (response.ok) {
                const result = await response.json();
                const failed = result.failed || [];
                if (failed.length) {
                    const detail = failed
                        .map(f => `${f.username || 'user ' + f.id}: ${f.error}`)
                        .join('; ');
                    window.UIUtils.showToast(`${result.message}. ${detail}`, 'error');
                } else {
                    window.UIUtils.showToast(`${result.message}.`, 'success');
                }
                setTimeout(() => location.reload(), 1000);
            } else {
                const error = await response.json();
                window.UIUtils.showToast('Error: ' + error.detail, 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Error updating users: ' + error.message, 'error');
        }
    }

    bulkApprove() {
        const n = this._selectedIds.size;
        return this._bulkAction('approve', {
            title: 'Approve Users',
            message: `Approve and activate ${n} selected user${n === 1 ? '' : 's'}? They will be able to log in immediately. Users that are not awaiting approval are skipped.`
        });
    }

    bulkActivate() {
        const n = this._selectedIds.size;
        return this._bulkAction('activate', {
            title: 'Activate Users',
            message: `Activate ${n} selected user${n === 1 ? '' : 's'}?`
        });
    }

    bulkDeactivate() {
        const n = this._selectedIds.size;
        return this._bulkAction('deactivate', {
            title: 'Deactivate Users',
            message: `Deactivate ${n} selected user${n === 1 ? '' : 's'}? Their API keys stop working immediately.`
        });
    }

    bulkRemove() {
        const n = this._selectedIds.size;
        return this._bulkAction('delete', {
            title: 'Permanently Delete Users',
            message: `Permanently delete ${n} selected user${n === 1 ? '' : 's'} and all their data? This cannot be undone.`,
            type: 'danger'
        });
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

// Browsers restore checkbox state across a reload or a back-navigation, which
// would leave boxes ticked while the selection Set is empty. Start from a clean
// slate instead.
window.addEventListener('pageshow', () => window.UserManager.clearSelection());

// Export functions for backward compatibility
window.deactivateUser = (button) => window.UserManager.deactivateUser(button);
window.activateUser = (button) => window.UserManager.activateUser(button);
window.approveUser = (button) => window.UserManager.approveUser(button);
window.removeUser = (button) => window.UserManager.removeUser(button);
window.modifyUser = (button) => window.UserManager.modifyUser(button);
window.filterUsers = (query) => window.UserManager.filterUsers(query);

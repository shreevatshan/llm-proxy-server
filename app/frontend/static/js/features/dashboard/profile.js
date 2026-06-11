let deleteAccountModal;
let pendingDeleteConfirmation = '';

document.addEventListener('DOMContentLoaded', function () {
    deleteAccountModal = new bootstrap.Modal(document.getElementById('deleteAccountModal'));
});

// Update Profile Form
document.getElementById('updateProfileForm')?.addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const username = formData.get('username');
    const email = formData.get('email');
    const emailField = document.getElementById('newEmail');
    const usernameField = document.getElementById('newUsername');

    // Get current values from the form's original values (set by the template)
    const currentUsername = usernameField.defaultValue;
    const currentEmail = emailField.defaultValue;
    
    console.log('Current username:', currentUsername);
    console.log('New username:', username);
    console.log('Current email:', currentEmail);
    console.log('New email:', email);

    if (username === currentUsername && email === currentEmail) {
        window.UIUtils.showToast('No changes detected', 'info');
        return;
    }

    const updateData = {};
    if (username !== currentUsername) updateData.username = username;
    
    // Only include email in update if the field is not readonly (not an OAuth user)
    if (email !== currentEmail && !emailField.hasAttribute('readonly')) {
        updateData.email = email;
    }

    // If no actual changes to update (e.g., only email changed but user is OAuth)
    if (Object.keys(updateData).length === 0) {
        window.UIUtils.showToast('No changes detected', 'info');
        return;
    }

    // Disable submit button
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Saving...';

    try {
        // Prevent any other requests during profile update
        window.profileUpdateInProgress = true;
        
        const response = await makeAuthenticatedRequest('/auth/profile', {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(updateData)
        });

        const data = await response.json();

        if (response.ok && data.id) {  // Check if response is ok and has user data
            console.log('Profile update successful');
            
            // Check if username was updated (requires login redirect due to JWT token)
            const usernameUpdated = updateData.username && updateData.username !== currentUsername;
            
            if (usernameUpdated) {
                console.log('Username updated, clearing tokens and redirecting to login...');
                
                // Set flag to prevent any other requests
                window.profileUpdateSuccess = true;
                
                window.UIUtils.showToast('Username updated successfully! Redirecting to login...', 'success');
                
                // Clear authentication token to prevent unauthorized requests
                localStorage.removeItem('access_token');
                console.log('Cleared access_token from localStorage');
                
                // Clear any cookies that might contain auth tokens
                document.cookie.split(";").forEach(function(c) { 
                    document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
                });
                console.log('Cleared all cookies');
                
                // Show message for 2 seconds before redirecting
                setTimeout(() => {
                    // Stop all network requests
                    if (window.stop) {
                        window.stop();
                    }
                    
                    // Force immediate redirect - multiple methods for redundancy
                    try {
                        window.location.replace('/login?updated=true&t=' + new Date().getTime());
                    } catch (e) {
                        console.log('Replace failed, trying href:', e);
                        window.location.href = '/login?updated=true&t=' + new Date().getTime();
                    }
                }, 2000);
            } else {
                // Only email was updated - no need to redirect
                console.log('Email updated, staying on profile page');
                window.UIUtils.showToast('Email updated successfully!', 'success');
                
                // Reload the page to show updated information
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            }
            
            return; // Exit immediately
        } else {
            window.UIUtils.showToast(data.detail || 'Failed to update profile', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    } finally {
        window.profileUpdateInProgress = false;
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
});

// Change Password Form
const changePasswordForm = document.getElementById('changePasswordForm');
if (changePasswordForm) {
    changePasswordForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        const formData = new FormData(this);
        const currentPassword = formData.get('current_password');
        const newPassword = formData.get('new_password');

        if (newPassword.length < 6) {
            window.UIUtils.showToast('New password must be at least 6 characters long', 'error');
            return;
        }

        // Disable submit button
        const submitBtn = this.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Updating...';

        try {
            const response = await makeAuthenticatedRequest('/auth/password', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword
                })
            });

            const data = await response.json();

            if (response.ok) {
                window.UIUtils.showToast('Password updated successfully!', 'success');
                this.reset();
            } else {
                window.UIUtils.showToast(data.detail || 'Failed to update password', 'error');
            }
        } catch (error) {
            window.UIUtils.showToast('Network error. Please try again.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    });
}

// Delete Account Form
document.getElementById('deleteAccountForm')?.addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    pendingDeleteConfirmation = formData.get('confirmation');

    if (!pendingDeleteConfirmation) {
        window.UIUtils.showToast('Confirmation required: Please type "DELETE" to proceed with account deletion', 'error');
        return;
    }

    if (pendingDeleteConfirmation !== 'DELETE') {
        window.UIUtils.showToast('Invalid confirmation: Please type "DELETE" exactly (all uppercase) to confirm', 'error');
        return;
    }

    deleteAccountModal.show();
});

// Confirm Delete Account
document.getElementById('confirmDeleteAccount')?.addEventListener('click', async function () {
    const originalText = this.innerHTML;
    this.disabled = true;
    this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Deleting...';

    try {
        const response = await makeAuthenticatedRequest('/auth/account', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                confirmation: pendingDeleteConfirmation
            })
        });

        const data = await response.json();

        if (response.ok) {
            window.UIUtils.showToast('Account deleted successfully. Redirecting...', 'success');
            deleteAccountModal.hide();

            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            window.UIUtils.showToast(data.detail || 'Failed to delete account', 'error');
            deleteAccountModal.hide();
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
        deleteAccountModal.hide();
    } finally {
        this.disabled = false;
        this.innerHTML = originalText;
        pendingDeleteConfirmation = '';
    }
});
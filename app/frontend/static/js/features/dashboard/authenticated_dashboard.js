let createApiKeyModal, apiKeyDisplayModal;
let currentUser = null;

document.addEventListener('DOMContentLoaded', function () {
    createApiKeyModal = new bootstrap.Modal(document.getElementById('createApiKeyModal'));
    apiKeyDisplayModal = new bootstrap.Modal(document.getElementById('apiKeyDisplayModal'));

    // Check authentication and load user data
    checkAuthAndLoadData();
});

async function checkAuthAndLoadData() {
    const token = getAuthToken();
    console.log('Token found:', token ? 'Yes' : 'No');

    try {
        // Get current user info - this will work with both localStorage token and HTTP-only cookies
        console.log('Making request to /auth/me');
        const response = await makeAuthenticatedRequest('/auth/me');
        console.log('Response status:', response.status);

        if (response.ok) {
            currentUser = await response.json();
            console.log('User authenticated:', currentUser.username);
            document.getElementById('welcomeMessage').textContent = `Welcome back, ${currentUser.username}!`;

            // Load API keys for regular users (admin users are redirected server-side)
            loadApiKeys();
        } else {
            // Token is invalid, redirect to login
            console.log('Token invalid, redirecting to login');
            localStorage.removeItem('access_token');
            window.location.href = '/login';
        }
    } catch (error) {
        // Network error or invalid token, redirect to login
        console.log('Error during authentication:', error);
        localStorage.removeItem('access_token');
        window.location.href = '/login';
    }
}

function showCreateApiKeyModal() {
    createApiKeyModal.show();
}

async function loadApiKeys() {
    try {
        // Add cache-busting parameter to ensure fresh data
        const response = await makeAuthenticatedRequest(`/auth/api-keys?t=${Date.now()}`);
        const apiKeys = await response.json();

        const container = document.getElementById('apiKeysContainer');

        if (apiKeys.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-key fa-2x mb-3"></i>
                    <p>No API keys yet. Create your first one!</p>
                </div>
            `;
        } else {
            container.innerHTML = apiKeys.map((key, index) => `
                <div class="d-flex justify-content-between align-items-center py-3" style="${index < apiKeys.length - 1 ? 'border-bottom: 1px solid var(--mono-4);' : ''}">
                    <div class="flex-grow-1">
                        <strong>${key.name}</strong><br>
                        <div class="api-key-display" id="api-key-${key.id}">
                            <div class="mt-1" style="max-width: 400px;">
                                <input type="text" class="form-control form-control-sm" id="full-api-key-${key.id}" value="${key.api_key_preview}" data-preview="${key.api_key_preview}" data-full-key="" readonly>
                            </div>
                        </div>
                        <small class="text-muted">
                            Created: ${new Date(key.created_at).toLocaleDateString()}
                            ${key.last_used ? `| Last used: ${new Date(key.last_used).toLocaleDateString()}` : ''}
                        </small>
                    </div>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-secondary" onclick="copyApiKeyValue(${key.id})" id="copy-btn-${key.id}">
                            <i class="fas fa-copy"></i>
                        </button>
                        <button class="btn btn-outline-primary" onclick="showApiKey(${key.id})" id="show-btn-${key.id}">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="deleteApiKey(${key.id}, '${key.name}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        document.getElementById('apiKeysContainer').innerHTML = `
            <div class="alert alert-danger">
                Failed to load API keys. Please refresh the page.
            </div>
        `;
    }
}

document.getElementById('createApiKeyForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const name = formData.get('name');

    try {
        const response = await makeAuthenticatedRequest('/auth/api-keys', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name: name })
        });

        const data = await response.json();

        if (response.ok) {
            createApiKeyModal.hide();
            document.getElementById('newApiKeyValue').value = data.api_key;
            apiKeyDisplayModal.show();
            await loadApiKeys(); // Refresh the list
            this.reset(); // Clear the form
        } else {
            window.UIUtils.showToast(data.detail || 'Failed to create API key', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    }
});

function copyApiKey() {
    const input = document.getElementById('newApiKeyValue');
    input.select();
    document.execCommand('copy');
    window.UIUtils.showToast('API key copied to clipboard!', 'success');
}

async function copyApiKeyValue(keyId) {
    const input = document.getElementById(`full-api-key-${keyId}`);
    if (input) {
        // Check if we have the full key stored
        let fullKey = input.getAttribute('data-full-key');
        
        // If we don't have the full key yet, fetch it
        if (!fullKey) {
            try {
                const response = await makeAuthenticatedRequest(`/auth/api-keys/detail?api_key_id=${keyId}`);
                if (response.ok) {
                    const data = await response.json();
                    fullKey = data.api_key;
                    // Store it for future use
                    input.setAttribute('data-full-key', fullKey);
                } else {
                    window.UIUtils.showToast('Failed to retrieve API key', 'error');
                    return;
                }
            } catch (error) {
                window.UIUtils.showToast('Network error. Please try again.', 'error');
                return;
            }
        }
        
        // Copy the full key
        if (fullKey) {
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(fullKey).then(() => {
                    window.UIUtils.showToast('API key copied to clipboard!', 'success');
                }).catch(() => {
                    fallbackCopyText(fullKey);
                });
            } else {
                fallbackCopyText(fullKey);
            }
        }
    }
}

function fallbackCopyText(text) {
    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        window.UIUtils.showToast('API key copied to clipboard!', 'success');
    } catch (error) {
        window.UIUtils.showToast('Failed to copy. Please copy manually.', 'warning');
    }
}

async function showApiKey(keyId) {
    try {
        // Show loading state
        const showBtn = document.getElementById(`show-btn-${keyId}`);
        const originalContent = showBtn.innerHTML;
        showBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        showBtn.disabled = true;

        // Fetch the full API key
        const response = await makeAuthenticatedRequest(`/auth/api-keys/detail?api_key_id=${keyId}`);

        if (response.ok) {
            const data = await response.json();

            // Update the input value with the full key and store it
            const fullKeyInput = document.getElementById(`full-api-key-${keyId}`);
            fullKeyInput.value = data.api_key;
            fullKeyInput.setAttribute('data-full-key', data.api_key);

            // Update button to show "Hide"
            showBtn.innerHTML = '<i class="fas fa-eye-slash"></i>';
            showBtn.onclick = () => hideApiKey(keyId);
        } else {
            const errorData = await response.json();
            window.UIUtils.showToast(errorData.detail || 'Failed to retrieve API key', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    } finally {
        // Reset button state
        const showBtn = document.getElementById(`show-btn-${keyId}`);
        showBtn.disabled = false;
        if (showBtn.innerHTML.includes('spinner')) {
            showBtn.innerHTML = '<i class="fas fa-eye"></i>';
        }
    }
}

function hideApiKey(keyId) {
    // Restore the preview value
    const fullKeyInput = document.getElementById(`full-api-key-${keyId}`);
    const previewValue = fullKeyInput.getAttribute('data-preview');
    fullKeyInput.value = previewValue;

    // Update button to show "Show"
    const showBtn = document.getElementById(`show-btn-${keyId}`);
    showBtn.innerHTML = '<i class="fas fa-eye"></i>';
    showBtn.onclick = () => showApiKey(keyId);
}

function copyFullApiKey(keyId) {
    const input = document.getElementById(`full-api-key-${keyId}`);
    if (input && input.value) {
        // Use the modern clipboard API if available
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(input.value).then(() => {
                window.UIUtils.showToast('API key copied to clipboard!', 'success');
            }).catch(() => {
                // Fallback to the older method
                fallbackCopy(input);
            });
        } else {
            // Fallback for older browsers or non-secure contexts
            fallbackCopy(input);
        }
    }
}

function fallbackCopy(input) {
    try {
        input.select();
        input.setSelectionRange(0, 99999); // For mobile devices
        document.execCommand('copy');
        window.UIUtils.showToast('API key copied to clipboard!', 'success');
    } catch (error) {
        window.UIUtils.showToast('Failed to copy API key. Please copy manually.', 'warning');
    }
}

async function deleteApiKey(keyId, keyName) {
    const confirmed = await window.UIUtils.showConfirmModal(
        'Delete API Key',
        `Are you sure you want to delete the API key "${keyName}"? This action cannot be undone.`
    );
    if (!confirmed) return;

    try {
        const response = await makeAuthenticatedRequest(`/auth/api-keys?api_key_id=${keyId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            window.UIUtils.showToast('API key deleted successfully!', 'success');
            // Force a complete refresh of the API keys list
            await loadApiKeys();
        } else {
            const data = await response.json();
            window.UIUtils.showToast(data.detail || 'Failed to delete API key', 'error');
        }
    } catch (error) {
        window.UIUtils.showToast('Network error. Please try again.', 'error');
    }
}
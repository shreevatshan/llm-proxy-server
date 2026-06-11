
function setAuthToken(token) {
    localStorage.setItem('access_token', token);
}

function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = message;
    alertDiv.style.marginTop = '1rem';

    const container = document.querySelector('.auth-card') || document.querySelector('.login-container');
    if (!container) return;
    const existingAlert = container.querySelector('.alert:not(#errorAlert)');
    if (existingAlert) {
        existingAlert.remove();
    }
    const form = container.querySelector('form');
    if (form) {
        container.insertBefore(alertDiv, form);
    } else {
        container.appendChild(alertDiv);
    }

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

document.getElementById('loginForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const username = formData.get('username');
    const password = formData.get('password');

    try {
        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                password: password
            })
        });

        const data = await response.json();

        if (response.ok) {
            console.log('Login successful, token received:', data.access_token ? 'Yes' : 'No');
            setAuthToken(data.access_token);
            console.log('Token stored in localStorage:', localStorage.getItem('access_token') ? 'Yes' : 'No');
            showAlert('Login successful! Redirecting...', 'success');
            setTimeout(() => {
                console.log('Redirecting to dashboard...');
                window.location.href = '/dashboard/';
            }, 1000);
        } else {
            console.log('Login failed:', data.detail);
            const detail = data.detail || 'Login failed';
            // Show pending approval and deactivated messages as warnings for better visibility
            const alertType = (detail.includes('pending admin approval') || detail.includes('deactivated'))
                ? 'warning'
                : 'danger';
            showAlert(detail, alertType);
        }
    } catch (error) {
        showAlert('Network error. Please try again.', 'danger');
    }
});

// ZOHO OAuth login handler
document.getElementById('zohoLoginBtn').addEventListener('click', async function() {
    try {
        // Check if ZOHO OAuth is available
        const statusResponse = await fetch('/auth/zoho/status');
        const statusData = await statusResponse.json();
        
        if (!statusData.available) {
            showAlert(statusData.message, 'warning');
            return;
        }
        
        // Redirect to ZOHO OAuth login
        window.location.href = '/auth/zoho/login';
        
    } catch (error) {
        showAlert('Failed to connect to ZOHO. Please try again.', 'danger');
    }
});

// Check for OAuth callback errors
window.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const error = urlParams.get('error');
    if (error) {
        const errorAlert = document.getElementById('errorAlert');
        const errorMessage = document.getElementById('errorMessage');
        if (errorAlert && errorMessage) {
            errorMessage.textContent = decodeURIComponent(error);
            errorAlert.style.display = 'block';
        } else {
            showAlert(decodeURIComponent(error), 'danger');
        }
    }
});
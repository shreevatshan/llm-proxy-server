
function showFieldError(fieldId, message) {
    const field = document.getElementById(fieldId);
    const errorElement = document.getElementById(fieldId + '-error');
    
    if (field && errorElement) {
        field.classList.add('is-invalid');
        errorElement.textContent = message;
        errorElement.classList.add('show');
        
        // Hide form-text help text when showing error to avoid duplication
        const formGroup = field.closest('.form-group');
        const formText = formGroup ? formGroup.querySelector('.form-text') : null;
        if (formText) {
            formText.classList.add('hidden');
        }
    }
}

function clearFieldError(fieldId) {
    const field = document.getElementById(fieldId);
    const errorElement = document.getElementById(fieldId + '-error');
    
    if (field && errorElement) {
        field.classList.remove('is-invalid');
        errorElement.textContent = '';
        errorElement.classList.remove('show');
        
        // Hide form-text help text when field is valid
        const formGroup = field.closest('.form-group');
        const formText = formGroup ? formGroup.querySelector('.form-text') : null;
        if (formText) {
            formText.classList.add('hidden');
        }
    }
}

function showFieldHelp(fieldId) {
    const field = document.getElementById(fieldId);
    const formGroup = field ? field.closest('.form-group') : null;
    const formText = formGroup ? formGroup.querySelector('.form-text') : null;
    
    if (formText) {
        formText.classList.remove('hidden');
    }
}

function clearAllErrors() {
    const fields = ['username', 'email', 'password', 'confirmPassword'];
    fields.forEach(fieldId => clearFieldError(fieldId));
}

function showGeneralSuccess(message) {
    const container = document.querySelector('.auth-card') || document.querySelector('.login-container');
    if (!container) return;
    const existingMessage = container.querySelector('.general-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    const messageDiv = document.createElement('div');
    messageDiv.className = 'general-message alert alert-success';
    messageDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    messageDiv.style.marginTop = '1rem';
    const form = container.querySelector('form');
    if (form) {
        container.insertBefore(messageDiv, form);
    } else {
        container.appendChild(messageDiv);
    }
}

function validateUsername() {
    const username = document.getElementById('username').value.trim();
    
    if (!username) {
        clearFieldError('username');
        showFieldHelp('username');
        return true; // Show help but don't show error for empty field unless it's form submission
    }
    
    if (username.length < 3) {
        showFieldError('username', 'Username must be at least 3 characters long');
        return false;
    }
    
    clearFieldError('username');
    return true;
}

function validateEmail() {
    const email = document.getElementById('email').value.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (!email) {
        clearFieldError('email');
        return true; // Don't show error for empty field unless it's form submission
    }
    
    if (!emailRegex.test(email)) {
        showFieldError('email', 'Please enter a valid email address');
        return false;
    }
    
    // Email is valid - clear any errors
    clearFieldError('email');
    return true;
}

function validatePasswordLength() {
    const password = document.getElementById('password').value;
    
    if (!password) {
        clearFieldError('password');
        showFieldHelp('password');
        return true; // Show help but don't show error for empty field unless it's form submission
    }
    
    if (password.length < 6) {
        showFieldError('password', 'Password must be at least 6 characters long');
        return false;
    }
    
    clearFieldError('password');
    return true;
}

// Validation functions for form submission (treat empty fields as errors)
function validateUsernameForSubmission() {
    const username = document.getElementById('username').value.trim();
    
    if (!username) {
        showFieldError('username', 'Username is required');
        return false;
    }
    
    if (username.length < 3) {
        showFieldError('username', 'Username must be at least 3 characters long');
        return false;
    }
    
    clearFieldError('username');
    return true;
}

function validateEmailForSubmission() {
    const email = document.getElementById('email').value.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    
    if (!email) {
        showFieldError('email', 'Email is required');
        return false;
    }
    
    if (!emailRegex.test(email)) {
        showFieldError('email', 'Please enter a valid email address');
        return false;
    }
    
    clearFieldError('email');
    return true;
}

function validatePasswordLengthForSubmission() {
    const password = document.getElementById('password').value;
    
    if (!password) {
        showFieldError('password', 'Password is required');
        return false;
    }
    
    if (password.length < 6) {
        showFieldError('password', 'Password must be at least 6 characters long');
        return false;
    }
    
    clearFieldError('password');
    return true;
}

function validatePasswords() {
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    if (!confirmPassword) {
        showFieldError('confirmPassword', 'Please confirm your password');
        return false;
    }
    
    if (password !== confirmPassword) {
        showFieldError('confirmPassword', 'Passwords do not match');
        return false;
    }
    
    clearFieldError('confirmPassword');
    return true;
}

// Real-time validation
document.addEventListener('DOMContentLoaded', function() {
    const usernameField = document.getElementById('username');
    const emailField = document.getElementById('email');
    const passwordField = document.getElementById('password');
    const confirmPasswordField = document.getElementById('confirmPassword');
    
    // Username validation
    usernameField.addEventListener('blur', validateUsername);
    usernameField.addEventListener('input', function() {
        // Always validate if field has content or was previously invalid
        if (this.value.trim() || this.classList.contains('is-invalid')) {
            validateUsername();
        }
    });
    
    // Email validation
    emailField.addEventListener('blur', validateEmail);
    emailField.addEventListener('input', function() {
        // Always validate if field has content or was previously invalid
        if (this.value.trim() || this.classList.contains('is-invalid')) {
            validateEmail();
        }
    });
    
    // Password validation
    passwordField.addEventListener('blur', validatePasswordLength);
    passwordField.addEventListener('input', function() {
        // Always validate if field has content or was previously invalid
        if (this.value || this.classList.contains('is-invalid')) {
            validatePasswordLength();
        }
        // Re-validate confirm password if it has a value
        if (confirmPasswordField.value) {
            validatePasswords();
        }
    });
    
    // Confirm password validation
    confirmPasswordField.addEventListener('input', function() {
        validatePasswords();
    });
    
    confirmPasswordField.addEventListener('blur', function() {
        if (this.value) {
            validatePasswords();
        }
    });
});

document.getElementById('signupForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    // Clear any existing general messages
    const existingMessage = document.querySelector('.general-message');
    if (existingMessage) {
        existingMessage.remove();
    }

    const formData = new FormData(this);
    const username = formData.get('username').trim();
    const email = formData.get('email').trim();
    const password = formData.get('password');
    const confirmPassword = formData.get('confirmPassword');

    // Validate all fields
    const isUsernameValid = validateUsernameForSubmission();
    const isEmailValid = validateEmailForSubmission();
    const isPasswordValid = validatePasswordLengthForSubmission();
    const isConfirmPasswordValid = validatePasswords();

    if (!isUsernameValid || !isEmailValid || !isPasswordValid || !isConfirmPasswordValid) {
        // Focus on the first invalid field
        const firstInvalidField = document.querySelector('.form-control.is-invalid');
        if (firstInvalidField) {
            firstInvalidField.focus();
        }
        return;
    }

    // Disable submit button to prevent double submission
    const submitButton = this.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Account...';

    try {
        const response = await fetch('/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: username,
                email: email,
                password: password
            })
        });

        const data = await response.json();

        if (response.ok) {
            const adminEmail = data.admin_email || 'the administrator';
            // Hide the form and show a pending approval notice
            document.getElementById('signupForm').style.display = 'none';
            const container = document.querySelector('.auth-card') || document.querySelector('.login-container');
            if (!container) return;
            const existingMessage = container.querySelector('.general-message');
            if (existingMessage) existingMessage.remove();
            const pendingDiv = document.createElement('div');
            pendingDiv.className = 'general-message alert alert-warning';
            pendingDiv.style.marginTop = '1rem';
            pendingDiv.innerHTML =
                `<i class="fas fa-clock" style="flex-shrink:0;"></i>` +
                `<span><strong>Account pending approval.</strong> ` +
                `Your account has been created and is awaiting admin approval. ` +
                `Please contact <strong>${adminEmail}</strong> for access.</span>`;
            const backLink = container.querySelector('.back-link');
            if (backLink) {
                container.insertBefore(pendingDiv, backLink);
            } else {
                container.appendChild(pendingDiv);
            }
            const loginLink = document.createElement('div');
            loginLink.className = 'back-link';
            loginLink.style.marginTop = '1rem';
            loginLink.innerHTML = '<small class="text-muted"><a href="/login">Back to Login</a></small>';
            container.appendChild(loginLink);
        } else {
            // Handle server-side field errors
            if (data.detail && typeof data.detail === 'object') {
                let hasFieldErrors = false;
                for (const [field, message] of Object.entries(data.detail)) {
                    if (['username', 'email', 'password', 'confirmPassword'].includes(field)) {
                        showFieldError(field, message);
                        hasFieldErrors = true;
                    }
                }
                
                // Focus on the first error field
                if (hasFieldErrors) {
                    const firstInvalidField = document.querySelector('.form-control.is-invalid');
                    if (firstInvalidField) {
                        firstInvalidField.focus();
                    }
                }
                
                // If there are non-field errors, show them as general errors
                if (!hasFieldErrors) {
                    showFieldError('username', data.detail || 'Signup failed. Please try again.');
                }
            } else {
                // Route plain-string errors to the correct field based on message content
                const errorMsg = data.detail || 'Signup failed. Please try again.';
                const lowerMsg = errorMsg.toLowerCase();
                if (lowerMsg.includes('email')) {
                    showFieldError('email', errorMsg);
                } else if (lowerMsg.includes('password')) {
                    showFieldError('password', errorMsg);
                } else {
                    showFieldError('username', errorMsg);
                }
            }
        }
    } catch (error) {
        showFieldError('username', 'Network error. Please check your connection and try again.');
    } finally {
        // Re-enable submit button
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
    }
});
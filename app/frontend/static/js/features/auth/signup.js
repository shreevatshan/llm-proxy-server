
// Mirrors USERNAME_MIN_LENGTH / USERNAME_MAX_LENGTH in app/auth/models.py.
const USERNAME_MIN_LENGTH = 3;
const USERNAME_MAX_LENGTH = 50;

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

function validateUsername() {
    const username = document.getElementById('username').value.trim();
    
    if (!username) {
        clearFieldError('username');
        showFieldHelp('username');
        return true; // Show help but don't show error for empty field unless it's form submission
    }
    
    if (username.length < USERNAME_MIN_LENGTH) {
        showFieldError('username', `Username must be at least ${USERNAME_MIN_LENGTH} characters long`);
        return false;
    }

    if (username.length > USERNAME_MAX_LENGTH) {
        showFieldError('username', `Username must be at most ${USERNAME_MAX_LENGTH} characters long`);
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
    
    if (username.length < USERNAME_MIN_LENGTH) {
        showFieldError('username', `Username must be at least ${USERNAME_MIN_LENGTH} characters long`);
        return false;
    }

    if (username.length > USERNAME_MAX_LENGTH) {
        showFieldError('username', `Username must be at most ${USERNAME_MAX_LENGTH} characters long`);
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

// Fields the server can attribute an error to (422 loc / X-Error-Field header).
const SERVER_FIELDS = ['username', 'email', 'password'];

document.getElementById('signupForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const submitButton = this.querySelector('button[type="submit"]');
    if (submitButton.disabled) return;

    AuthUI.clear();

    const formData = new FormData(this);
    const username = (formData.get('username') || '').trim();
    const email = (formData.get('email') || '').trim();
    const password = formData.get('password') || '';

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
    const originalText = submitButton.innerHTML;
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Account...';

    const res = await AuthUI.request('/auth/signup', {
        json: { username: username, email: email, password: password },
    });

    if (res.ok) {
        const adminEmail = (res.data && res.data.admin_email) || 'the administrator';
        // Hide the form and show the pending approval notice (rendered as text).
        this.style.display = 'none';
        AuthUI.show(
            `Account created and awaiting admin approval. Please contact ${adminEmail} for access.`,
            { level: 'warning' }
        );
        return;
    }

    submitButton.disabled = false;
    submitButton.innerHTML = originalText;

    // Field errors stay inline; anything not tied to a field goes to the alert slot.
    let hasFieldErrors = false;
    const unmapped = [];
    for (const [field, message] of Object.entries(res.fieldErrors || {})) {
        if (SERVER_FIELDS.includes(field)) {
            showFieldError(field, message);
            hasFieldErrors = true;
        } else if (message) {
            unmapped.push(message);
        }
    }

    if (hasFieldErrors) {
        const firstInvalidField = document.querySelector('.form-control.is-invalid');
        if (firstInvalidField) {
            firstInvalidField.focus();
        }
        if (unmapped.length) {
            AuthUI.show(unmapped.join(' '), { level: 'danger' });
        }
    } else {
        AuthUI.show(res.message || 'Signup failed. Please try again.', { level: 'danger' });
    }
});

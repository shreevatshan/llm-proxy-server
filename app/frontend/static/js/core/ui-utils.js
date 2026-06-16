/**
 * UI Utilities Module
 * Handles toast notifications, modal confirmations, and tab management
 */

// Toast Notification System
function showToast(message, type = 'info', duration = 5000) {
    // Ensure toast container exists
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icon = getToastIcon(type);
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${icon}"></i>
            <span class="toast-message">${message}</span>
        </div>
        <button class="toast-close" onclick="closeToast(this)">
            <i class="fas fa-times"></i>
        </button>
    `;

    toastContainer.appendChild(toast);

    // Show toast with animation
    setTimeout(() => toast.classList.add('show'), 100);

    // Auto-remove toast after duration
    if (duration > 0) {
        setTimeout(() => closeToast(toast.querySelector('.toast-close')), duration);
    }

    return toast;
}

function getToastIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'error':
        case 'danger': return 'fa-exclamation-triangle';
        case 'warning': return 'fa-exclamation-circle';
        case 'info':
        default: return 'fa-info-circle';
    }
}

function closeToast(closeButton) {
    const toast = closeButton.closest('.toast');
    toast.classList.add('hiding');
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 300);
}

// Confirmation Modal System
function showConfirmModal(title, message, type = 'warning') {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        const modalTitle = document.getElementById('confirmModalTitle');
        const modalMessage = document.getElementById('confirmModalMessage');
        const confirmBtn = document.getElementById('confirmActionBtn');
        const cancelBtn = document.getElementById('confirmCancelBtn');

        // Set modal content
        modalTitle.textContent = title;
        modalMessage.textContent = message;

        // Set button style based on type
        confirmBtn.className = 'btn btn-' + (type === 'danger' ? 'danger' : type === 'warning' ? 'warning' : 'primary');

        // Set up event handlers
        const handleConfirm = () => {
            cleanup();
            resolve(true);
        };

        const handleCancel = () => {
            cleanup();
            resolve(false);
        };

        const cleanup = () => {
            confirmBtn.removeEventListener('click', handleConfirm);
            cancelBtn.removeEventListener('click', handleCancel);
            modal.removeEventListener('hidden.bs.modal', handleCancel);
            bootstrap.Modal.getInstance(modal).hide();
        };

        // Add event listeners
        confirmBtn.addEventListener('click', handleConfirm);
        cancelBtn.addEventListener('click', handleCancel);
        modal.addEventListener('hidden.bs.modal', handleCancel, { once: true });

        // Show modal
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
    });
}

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName + '-tab').style.display = 'block';

    // Add active class to selected button (find the button that corresponds to this tab)
    const tabButton = document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (tabButton) {
        tabButton.classList.add('active');
    }

    // Load data when switching to providers tab
    if (tabName === 'providers') {
        window.ProviderManager?.loadUnifiedProviderData();
    }

    // Connect/disconnect SSE for active requests tab
    if (tabName === 'requests') {
        window.RequestTracker?.connect();
    } else {
        window.RequestTracker?.disconnect();
    }

    // Load usage data when switching to usage tab; stop polling when leaving
    if (tabName === 'usage') {
        window.UsageManager?.load();
    } else {
        window.UsageManager?.stopAutoRefresh();
    }

    // Load rate limits when switching to rate-limits tab; stop polling when leaving
    if (tabName === 'rate-limits') {
        window.RateLimitManager?.load();
    } else {
        window.RateLimitManager?.stopAutoRefresh();
    }
}

// Sub-tab switcher inside the Rate Limits tab
function showRateLimitSubTab(name) {
    const panes = ['request', 'model-groups'];
    panes.forEach(p => {
        const el = document.getElementById(`rl-subtab-${p}`);
        if (el) el.style.display = p === name ? '' : 'none';
        const btn = document.getElementById(`rl-subtab-${p}-btn`);
        if (btn) btn.classList.toggle('active', p === name);
    });
    if (name === 'model-groups') {
        window.ModelGroupManager?.load();
    }
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for global access
window.UIUtils = {
    showToast,
    getToastIcon,
    closeToast,
    showConfirmModal,
    showTab,
    showRateLimitSubTab,
    escapeHtml
};

window.showRateLimitSubTab = showRateLimitSubTab;

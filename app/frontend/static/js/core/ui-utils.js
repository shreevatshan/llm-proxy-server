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
    const selectedTab = document.getElementById(tabName + '-tab');
    if (selectedTab) selectedTab.style.display = 'block';

    // Add active class to selected button (find the button that corresponds to this tab)
    const tabButton = document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (tabButton) {
        tabButton.classList.add('active');
    }

    // Load data when switching to providers tab
    if (tabName === 'providers') {
        window.ProviderManager?.loadTab();
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

    // Load rate limits when switching to rate-limits tab
    if (tabName === 'rate-limits') {
        window.RateLimitManager?.load();
    }

    if (tabName === 'model-mapping') {
        window.ModelAliasManager?.load();
    }

    if (tabName === 'profiles') {
        window.ProfileManager?.load();
    }

    if (tabName === 'pools') {
        window.PoolAdminManager?.load();
    }

    if (tabName === 'websearch') {
        window.WebSearchManager?.load();
    }

    // Per-profile model access
    if (tabName === 'models') {
        window.ModelsManager?.load();
    }

    if (tabName === 'model-groups') {
        window.ModelGroupManager?.load();
    }
}

// Utility Functions
// Encodes HTML-special characters including quotes so the result is safe to
// interpolate into both element text and quoted attribute values. The plain
// textContent round-trip does NOT encode " ' or `, so we build the string
// manually with replace().
function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/`/g, '&#96;');
}

// Export functions for global access
window.UIUtils = {
    showToast,
    getToastIcon,
    closeToast,
    showConfirmModal,
    showTab,
    escapeHtml
};

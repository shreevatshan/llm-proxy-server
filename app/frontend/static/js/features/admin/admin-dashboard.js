/**
 * Main Admin Dashboard Module
 * Coordinates all modules and handles initialization
 */

class AdminDashboard {
    constructor() {
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;

        console.log('🚀 Initializing Admin Dashboard...');
        
        // Initialize all modules
        this.initializeModules();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Show the default active tab (users tab is active by default)
        this.showInitialTab();
        
        // Load initial data if on providers tab
        if (document.getElementById('providers-tab')?.style.display !== 'none') {
            window.ProviderManager.loadUnifiedProviderData();
        }


        this.initialized = true;
        console.log('✅ Admin Dashboard initialized successfully');
    }

    initializeModules() {
        // All modules are already initialized via their respective files
        // This method can be extended to perform any additional setup
        console.log('📦 All modules loaded:', {
            UIUtils: !!window.UIUtils,
            ProviderManager: !!window.ProviderManager,
            ProviderFormManager: !!window.ProviderFormManager,
            ModelManager: !!window.ModelManager,
            SearchManager: !!window.SearchManager,
            UserManager: !!window.UserManager,
            AzureManager: !!window.AzureManager,
            RequestTracker: !!window.RequestTracker,
            UsageManager: !!window.UsageManager
        });
    }

    showInitialTab() {
        // Find the active tab button
        const activeButton = document.querySelector('.tab-button.active');
        if (activeButton) {
            // Extract tab name from the onclick attribute
            const onclickAttr = activeButton.getAttribute('onclick');
            const tabNameMatch = onclickAttr.match(/showTab\('([^']+)'\)/);
            if (tabNameMatch) {
                const tabName = tabNameMatch[1];
                console.log(`🎯 Showing initial tab: ${tabName}`);
                window.UIUtils.showTab(tabName);
                return;
            }
        }
        
        // Fallback to users tab if no active button found
        console.log('🎯 Showing fallback tab: users');
        window.UIUtils.showTab('users');
    }

    setupEventListeners() {
        // Set up any global event listeners here
        const setupFormHandling = () => {
            console.log('🔧 Setting up form handling...');
            
            // Initialize form handling
            const providerForm = document.getElementById('provider-form');
            console.log('📋 Provider form element:', {
                exists: !!providerForm,
                id: providerForm?.id,
                tagName: providerForm?.tagName,
                action: providerForm?.action,
                method: providerForm?.method
            });
            
            if (providerForm) {
                // Remove any existing listeners to avoid duplicates
                providerForm.removeEventListener('submit', handleFormSubmit);
                providerForm.addEventListener('submit', handleFormSubmit);
                console.log('✅ Form submit listener attached');
            } else {
                console.warn('⚠️ Provider form not found during setup');
            }

            // Initialize provider type dropdown change
            const providerTypeSelect = document.getElementById('provider_type');
            if (providerTypeSelect) {
                providerTypeSelect.removeEventListener('change', handleProviderTypeChange);
                providerTypeSelect.addEventListener('change', handleProviderTypeChange);
            }

            // Initialize search input
            const searchInput = document.getElementById('provider-search');
            if (searchInput) {
                searchInput.removeEventListener('keyup', handleSearchInput);
                searchInput.addEventListener('keyup', handleSearchInput);
            }
        };

        const handleFormSubmit = (e) => {
            console.log('📝 Form submit event triggered:', e);
            console.log('📋 Event details:', {
                type: e.type,
                target: e.target,
                preventDefault: typeof e.preventDefault,
                targetId: e.target.id,
                targetTagName: e.target.tagName
            });
            window.ProviderFormManager.submitProviderForm(e);
        };

        const handleProviderTypeChange = () => {
            window.ProviderFormManager.updateProviderForm();
        };

        const handleSearchInput = () => {
            window.SearchManager.searchProviders();
        };

        // Setup immediately if DOM is ready, otherwise wait for DOMContentLoaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupFormHandling);
        } else {
            setupFormHandling();
        }
    }

    // Utility methods
    getProvidersData() {
        return window.providersData;
    }

    setProvidersData(data) {
        window.providersData = data;
    }

    getCurrentEditingProvider() {
        return window.currentEditingProvider;
    }

    setCurrentEditingProvider(provider) {
        window.currentEditingProvider = provider;
    }
}

// Create global instance
window.AdminDashboard = new AdminDashboard();

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.AdminDashboard.init();
});

// Export legacy functions for complete backward compatibility
window.showTab = window.UIUtils.showTab;
window.showToast = window.UIUtils.showToast;
window.showConfirmModal = window.UIUtils.showConfirmModal;

// Provider management legacy functions
window.loadUnifiedProviderData = () => window.ProviderManager.loadUnifiedProviderData();
window.showAddProviderForm = () => window.ProviderManager.showAddProviderForm();
window.cancelProviderForm = () => window.ProviderManager.cancelProviderForm();
window.updateProviderForm = () => window.ProviderFormManager.updateProviderForm();
window.submitProviderForm = (event) => window.ProviderFormManager.submitProviderForm(event);

// Debug function for manual testing
window.debugFormSubmission = () => {
    console.log('🧪 Debug form submission triggered');
    const form = document.getElementById('provider-form');
    if (form) {
        const event = new Event('submit', { bubbles: true, cancelable: true });
        form.dispatchEvent(event);
    } else {
        console.error('❌ Form not found for debug submission');
    }
};

// Search legacy functions
window.searchProviders = () => window.SearchManager.searchProviders();
window.clearSearch = () => window.SearchManager.clearSearch();

// Model management legacy functions
window.syncModels = () => window.ModelManager.syncModels();
window.bulkToggle = (action) => window.ModelManager.bulkToggle(action);
window.reinitSystem = () => window.ModelManager.reinitSystem();

// Azure legacy functions
window.toggleDynamicDiscovery = () => window.AzureManager.toggleDynamicDiscovery();
window.showAzureSetupInstructions = () => window.AzureManager.showAzureSetupInstructions();

// User management legacy functions
window.deactivateUser = (button) => window.UserManager.deactivateUser(button);
window.activateUser = (button) => window.UserManager.activateUser(button);
window.approveUser = (button) => window.UserManager.approveUser(button);
window.removeUser = (button) => window.UserManager.removeUser(button);

console.log('🔧 Admin Dashboard main module loaded');

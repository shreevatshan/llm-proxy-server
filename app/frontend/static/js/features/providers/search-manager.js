/**
 * Search Functionality Module
 * Handles provider and model search with debouncing and highlighting
 */

class SearchManager {
    constructor() {
        this.searchTimeout = null;
        this.originalProvidersData = null;
        this.allProviderModels = {};
    }

    searchProviders() {
        // Clear previous timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }

        // Debounce search to avoid excessive filtering
        this.searchTimeout = setTimeout(() => {
            this.performSearch();
        }, 300);
    }

    async performSearch() {
        const searchInput = document.getElementById('provider-search');
        const searchTerm = searchInput.value.toLowerCase().trim();

        // Store original data if not already stored
        if (!this.originalProvidersData && window.providersData) {
            this.originalProvidersData = [...window.providersData];
        }

        if (searchTerm.length === 0) {
            // No search term - show all providers
            if (this.originalProvidersData) {
                window.ProviderManager.renderProvidersList(this.originalProvidersData);
                window.ProviderManager.showProvidersList();
                document.getElementById('search-results').style.display = 'none';
            }
            return;
        }

        if (!this.originalProvidersData || this.originalProvidersData.length === 0) {
            this.showSearchResults([], searchTerm);
            return;
        }

        try {
            // Perform the search
            const searchResults = await this.performProviderAndModelSearch(this.originalProvidersData, searchTerm);
            this.showSearchResults(searchResults, searchTerm);
        } catch (error) {
            console.error('Search error:', error);
            window.UIUtils.showToast('Error performing search: ' + error.message, 'error');
        }
    }

    async performProviderAndModelSearch(providers, searchTerm) {
        const results = [];

        for (const provider of providers) {
            let providerMatches = false;
            let matchingModels = [];

            // Check if provider name matches
            const providerNameMatch = provider.provider_name.toLowerCase().includes(searchTerm);
            const providerTypeMatch = provider.provider_type.toLowerCase().includes(searchTerm);
            const providerKeyMatch = provider.provider_key.toLowerCase().includes(searchTerm);

            if (providerNameMatch || providerTypeMatch || providerKeyMatch) {
                providerMatches = true;
            }

            // Search through provider's models
            try {
                const models = await this.getProviderModels(provider.provider_key);
                if (models && models.length > 0) {
                    matchingModels = models.filter(model => {
                        return model.model_name.toLowerCase().includes(searchTerm) ||
                            model.model_id.toLowerCase().includes(searchTerm);
                    });
                }
            } catch (error) {
                console.warn(`Failed to search models for provider ${provider.provider_key}:`, error);
            }

            // Include provider in results if either provider matches or has matching models
            if (providerMatches || matchingModels.length > 0) {
                results.push({
                    ...provider,
                    matchingModels: matchingModels,
                    providerMatches: providerMatches,
                    modelMatches: matchingModels.length > 0
                });
            }
        }

        return results;
    }

    async getProviderModels(providerKey) {
        // Check if we already have the models cached
        if (this.allProviderModels[providerKey]) {
            return this.allProviderModels[providerKey];
        }

        try {
            const response = await fetch(`/admin/models/provider?provider_key=${encodeURIComponent(providerKey)}`, {
                credentials: 'include'
            });

            if (response.ok) {
                const models = await response.json();
                this.allProviderModels[providerKey] = models; // Cache the models
                return models;
            }
        } catch (error) {
            console.warn(`Failed to fetch models for provider ${providerKey}:`, error);
        }

        return [];
    }

    showSearchResults(results, searchTerm) {
        const searchResultsContainer = document.getElementById('search-results');
        const providersListContainer = document.getElementById('providers-list');

        // Hide the main providers list
        providersListContainer.style.display = 'none';

        if (results.length === 0) {
            // No results found
            searchResultsContainer.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🔍</div>
                    <h3>No Results Found</h3>
                    <p>No providers or models match "${window.UIUtils.escapeHtml(searchTerm)}"</p>
                    <button class="btn btn-outline-primary" onclick="window.SearchManager.clearSearch()">
                        <i class="fas fa-times"></i> Clear Search
                    </button>
                </div>
            `;
            searchResultsContainer.style.display = 'block';
            return;
        }

        // Render search results
        let html = '';

        // Group results by provider type
        const groupedResults = {};
        results.forEach(result => {
            if (!groupedResults[result.provider_type]) {
                groupedResults[result.provider_type] = [];
            }
            groupedResults[result.provider_type].push(result);
        });

        Object.entries(groupedResults).forEach(([providerType, typeResults]) => {
            const typeTitle = providerType.charAt(0).toUpperCase() + providerType.slice(1);

            html += `
                <div class="provider-type-section">
                    <div class="provider-type-header">
                        <i class="fas fa-chevron-down provider-type-icon"></i>
                        <h3>${typeTitle}</h3>
                        <span class="provider-count">${typeResults.length}</span>
                    </div>
                    <div class="provider-type-content">
            `;

            typeResults.forEach(result => {
                html += this.renderSearchResultProvider(result, searchTerm);
            });

            html += `
                    </div>
                </div>
            `;
        });

        searchResultsContainer.innerHTML = html;
        searchResultsContainer.style.display = 'block';
    }

    renderSearchResultProvider(provider, searchTerm) {
        const statusClass = provider.enabled ? 'status-active' : 'status-inactive';
        const statusText = provider.enabled ? 'Enabled' : 'Disabled';

        let html = `
            <div class="unified-provider-card search-result ${!provider.enabled ? 'disabled' : ''}" id="search-provider-${provider.provider_key}">
                <div class="provider-card-header">
                    <div class="provider-info">
                        <div class="provider-title">
                            <h4>${this.highlightSearchTerm(provider.provider_name, searchTerm)}</h4>
                            <span class="provider-type-badge">${this.highlightSearchTerm(provider.provider_type, searchTerm)}</span>
                        </div>
                        <div class="provider-status">
                            <span class="status-badge ${statusClass}">${statusText}</span>
                            ${provider.providerMatches ? '<span class="badge bg-success ms-2"><i class="fas fa-check"></i> Provider Match</span>' : ''}
                            ${provider.modelMatches ? `<span class="badge bg-info ms-2"><i class="fas fa-cog"></i> ${provider.matchingModels.length} Model${provider.matchingModels.length !== 1 ? 's' : ''}</span>` : ''}
                        </div>
                        <div class="provider-stats">
                            <span class="model-stats">Models: ${provider.model_count || 0} (${provider.enabled_model_count || 0} enabled)</span>
                        </div>
                    </div>
                    <div class="provider-actions">
                        <button class="btn btn-sm btn-outline-info" onclick="window.ModelManager.toggleProviderModels('${provider.provider_key}')" title="Show/Hide Models" id="models-btn-${provider.provider_key}">
                            <i class="fas fa-list"></i> Models (${provider.model_count || 0})
                        </button>
                        <label class="toggle-switch">
                            <input type="checkbox" ${provider.enabled ? 'checked' : ''} 
                                   onchange="window.ProviderManager.toggleProviderEnabled('${provider.provider_key}', this.checked)">
                            <span class="toggle-slider"></span>
                        </label>
                        <button class="btn btn-sm btn-outline-primary" onclick="window.ProviderManager.editProvider('${provider.provider_key}')" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="window.ProviderManager.deleteProvider('${provider.provider_key}')" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
        `;

        // Show matching models if any
        if (provider.matchingModels && provider.matchingModels.length > 0) {
            html += `
                <div class="provider-models-section search-models-section">
                    <div class="models-header">
                        <div class="d-flex justify-content-between align-items-center p-3 border-bottom">
                            <h6 class="mb-0">
                                <i class="fas fa-search me-2 text-success"></i>
                                Matching Models (${provider.matchingModels.length})
                            </h6>
                        </div>
                    </div>
                    <div class="models-list">
            `;

            provider.matchingModels.forEach(model => {
                const modelStatusClass = model.is_enabled ? 'status-active' : 'status-inactive';
                const modelStatusText = model.is_enabled ? 'Enabled' : 'Disabled';
                const modelItemClass = model.is_enabled ? '' : 'disabled';

                html += `
                    <div class="model-item search-model-result ${modelItemClass}">
                        <div class="model-info">
                            <div class="model-name">${this.highlightSearchTerm(model.model_name, searchTerm)}</div>
                            <div class="model-id">${this.highlightSearchTerm(model.model_id, searchTerm)}</div>
                        </div>
                        <div class="model-actions">
                            <div class="model-status">
                                <span class="status-badge ${modelStatusClass}">${modelStatusText}</span>
                            </div>
                            <label class="model-toggle-switch">
                                <input type="checkbox" ${model.is_enabled ? 'checked' : ''} 
                                       onchange="window.ModelManager.toggleIndividualModel('${model.model_id}', this.checked, '${provider.provider_key}')"
                                       id="search-toggle-${model.id}">
                                <span class="model-toggle-slider" id="search-slider-${model.id}"></span>
                            </label>
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        html += `</div>`;
        return html;
    }

    highlightSearchTerm(text, searchTerm) {
        if (!searchTerm || !text) return window.UIUtils.escapeHtml(text);

        const escapedText = window.UIUtils.escapeHtml(text);
        const escapedSearchTerm = window.UIUtils.escapeHtml(searchTerm);

        // Case-insensitive highlighting
        const regex = new RegExp(`(${escapedSearchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return escapedText.replace(regex, '<mark class="search-highlight">$1</mark>');
    }

    clearSearch() {
        const searchInput = document.getElementById('provider-search');
        const searchResultsContainer = document.getElementById('search-results');
        const providersListContainer = document.getElementById('providers-list');

        // Clear search input
        searchInput.value = '';

        // Hide search results
        searchResultsContainer.style.display = 'none';

        // Show original providers list
        if (this.originalProvidersData && this.originalProvidersData.length > 0) {
            window.ProviderManager.renderProvidersList(this.originalProvidersData);
            providersListContainer.style.display = 'block';
        } else {
            window.ProviderManager.showProvidersEmpty();
        }

        // Clear search timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = null;
        }
    }

    // Method to refresh cached data when providers data changes
    refreshCachedData() {
        if (window.providersData) {
            console.log('🔄 Refreshing SearchManager cached data');
            this.originalProvidersData = [...window.providersData];
        }
    }

    // Method to clear cached models data (e.g., after bulk operations)
    clearModelsCache() {
        console.log('🧹 Clearing SearchManager models cache');
        this.allProviderModels = {};
    }
}

// Create global instance
window.SearchManager = new SearchManager();

// Export functions for backward compatibility
window.searchProviders = () => window.SearchManager.searchProviders();
window.clearSearch = () => window.SearchManager.clearSearch();

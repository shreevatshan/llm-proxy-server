/**
 * Request Tracker Module
 * Provides real-time visibility into active LLM API requests via SSE.
 */

class RequestTrackerManager {
    constructor() {
        this.eventSource = null;
        this.activeRequests = new Map();
        this.durationTimerId = null;
        this.connected = false;
    }

    connect() {
        if (this.eventSource) return;

        this.eventSource = new EventSource('/admin/requests/stream');

        this.eventSource.onopen = () => {
            this.connected = true;
            this._updateConnectionStatus(true);
        };

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._handleEvent(data);
            } catch (e) {
                console.error('RequestTracker: failed to parse event', e);
            }
        };

        this.eventSource.onerror = () => {
            this.connected = false;
            this._updateConnectionStatus(false);
        };

        this.durationTimerId = setInterval(() => this._updateDurations(), 1000);
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        if (this.durationTimerId) {
            clearInterval(this.durationTimerId);
            this.durationTimerId = null;
        }
        this.connected = false;
        this._updateConnectionStatus(false);
    }

    _handleEvent(data) {
        switch (data.event) {
            case 'snapshot':
                this.activeRequests.clear();
                if (data.active_requests) {
                    for (const req of data.active_requests) {
                        this.activeRequests.set(req.request_id, req);
                    }
                }
                this._renderAll();
                this._updateStats();
                break;
            case 'request_started':
                if (data.request) {
                    this.activeRequests.set(data.request.request_id, data.request);
                    this._addRow(data.request);
                    this._updateStats();
                }
                break;
            case 'request_updated':
                if (data.request) {
                    this.activeRequests.set(data.request.request_id, data.request);
                    this._upsertRow(data.request);
                    this._updateStats();
                }
                break;
            case 'request_completed':
            case 'request_cancelled':
            case 'request_errored':
                if (data.request) {
                    this._maybeShowTerminalToast(data.request);
                    this.activeRequests.delete(data.request.request_id);
                    this._removeRow(data.request.request_id);
                    this._updateStats();
                }
                break;
        }
        this._toggleEmptyState();
    }

    _renderAll() {
        const tbody = document.getElementById('active-requests-body');
        if (!tbody) return;
        tbody.innerHTML = '';
        for (const req of this.activeRequests.values()) {
            tbody.appendChild(this._createRow(req));
        }
    }

    _createRow(req) {
        const tr = document.createElement('tr');
        tr.id = 'req-row-' + req.request_id;
        tr.className = 'request-row-enter';
        tr.innerHTML =
            '<td>' + this._serverBadge(req.server) + '</td>' +
            '<td><code>' + this._escapeHtml(req.endpoint) + '</code></td>' +
            '<td>' + this._escapeHtml(req.model || '-') + '</td>' +
            '<td>' + this._userBadge(req.user_identity, req.user_type) + '</td>' +
            '<td>' + (req.is_streaming ? '<span class="badge bg-info">Stream</span>' : '<span class="badge bg-secondary">Sync</span>') + '</td>' +
            '<td class="request-duration" data-start="' + req.start_time + '">' + this._formatDuration(req.start_time) + '</td>' +
            '<td><span class="status-badge status-active">Active</span></td>';
        requestAnimationFrame(() => tr.classList.remove('request-row-enter'));
        return tr;
    }

    _addRow(req) {
        const tbody = document.getElementById('active-requests-body');
        if (!tbody) return;
        const existing = document.getElementById('req-row-' + req.request_id);
        if (existing) return;
        tbody.prepend(this._createRow(req));
    }

    _upsertRow(req) {
        const tbody = document.getElementById('active-requests-body');
        if (!tbody) return;
        const nextRow = this._createRow(req);
        const existing = document.getElementById('req-row-' + req.request_id);
        if (existing && existing.parentNode) {
            existing.parentNode.replaceChild(nextRow, existing);
            requestAnimationFrame(() => nextRow.classList.remove('request-row-enter'));
            return;
        }
        tbody.prepend(nextRow);
    }

    _removeRow(requestId) {
        const row = document.getElementById('req-row-' + requestId);
        if (!row) return;
        row.classList.add('request-row-exit');
        setTimeout(() => {
            if (row.parentNode) row.parentNode.removeChild(row);
            this._toggleEmptyState();
        }, 400);
    }

    _updateDurations() {
        const cells = document.querySelectorAll('.request-duration');
        const now = Date.now() / 1000;
        cells.forEach(cell => {
            const start = parseFloat(cell.dataset.start);
            if (!isNaN(start)) {
                cell.textContent = this._formatDuration(start, now);
            }
        });
    }

    _updateStats() {
        const total = this.activeRequests.size;
        let openai = 0, anthropic = 0, azure = 0;
        for (const req of this.activeRequests.values()) {
            if (req.server === 'openai') openai++;
            else if (req.server === 'anthropic') anthropic++;
            else if (req.server === 'azure_openai') azure++;
        }
        this._setText('total-active-requests', total);
        this._setText('openai-active-requests', openai);
        this._setText('anthropic-active-requests', anthropic);
        this._setText('azure-active-requests', azure);
    }

    _updateConnectionStatus(connected) {
        const badge = document.getElementById('sse-connection-status');
        if (!badge) return;
        if (connected) {
            badge.className = 'status-badge status-active';
            badge.innerHTML = '<i class="fas fa-circle"></i> Connected';
        } else {
            badge.className = 'status-badge status-inactive';
            badge.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
        }
    }

    _toggleEmptyState() {
        const empty = document.getElementById('requests-empty');
        const table = document.getElementById('active-requests-table');
        if (!empty || !table) return;
        if (this.activeRequests.size === 0) {
            empty.style.display = '';
            table.style.display = 'none';
        } else {
            empty.style.display = 'none';
            table.style.display = '';
        }
    }

    _formatDuration(startTime, now) {
        now = now || (Date.now() / 1000);
        const elapsed = now - startTime;
        if (elapsed < 1) return '<1s';
        if (elapsed < 60) return Math.floor(elapsed) + 's';
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        return mins + 'm ' + secs + 's';
    }

    _serverBadge(server) {
        const map = {
            openai: { cls: 'primary', label: 'OpenAI' },
            anthropic: { cls: 'info', label: 'Anthropic' },
            azure_openai: { cls: 'warning', label: 'Azure' }
        };
        const s = map[server] || { cls: 'secondary', label: server };
        return '<span class="badge bg-' + s.cls + '">' + s.label + '</span>';
    }

    _userBadge(identity, type) {
        if (!identity || identity === 'unknown') return '<span class="text-muted">-</span>';
        const icons = { admin: 'fa-shield-alt', user: 'fa-user', api_key: 'fa-key' };
        const icon = icons[type] || 'fa-user';
        return '<i class="fas ' + icon + ' me-1 text-muted"></i>' + this._escapeHtml(identity);
    }

    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    _setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    _maybeShowTerminalToast(req) {
        const reason = req.termination_reason;
        if (!reason) return;

        const labels = {
            bedrock_native_idle_timeout: 'Bedrock idle timeout',
            bedrock_native_premature_eof: 'Bedrock premature EOF',
            bedrock_native_provider_error: 'Bedrock provider error',
            chunk_timeout: 'Chunk timeout',
            stream_timeout: 'Stream timeout',
            stream_error: 'Stream error'
        };

        const label = labels[reason];
        if (!label) return;

        const model = req.model || 'unknown model';
        const endpoint = req.endpoint || 'unknown endpoint';
        const errorSuffix = req.error ? `: ${req.error}` : '';
        const message = `${label} for ${model} on ${endpoint}${errorSuffix}`;
        window.UIUtils?.showToast(message, 'error', 8000);
    }
}

window.RequestTracker = new RequestTrackerManager();

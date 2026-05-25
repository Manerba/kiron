/**
 * Ollama Monitor - Requests Tab
 * Live-Tabelle aller Ollama-Requests mit Filtern und Pagination.
 */

let requestsData = [];  // Alle geladenen Requests
let requestFilterDebounce = null;

// Pagination-State
let reqCurrentPage = 1;
let reqPageSize = 100;
let reqTotalItems = 0;

// Cache fuer geladene Request-Details (Body)
const requestDetailCache = {};

// ========================================
// Initialisierung
// ========================================

async function initRequestsTab() {
    const container = document.getElementById('contentContainer');
    const filterBar = document.getElementById('filterBar');
    if (!container) return;

    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    // Filter-Bar aufbauen
    if (filterBar) {
        filterBar.style.display = 'flex';
        filterBar.innerHTML = buildRequestFilters();
        initRequestFilterListeners();
    }

    reqCurrentPage = 1;
    await loadRequests();
}

// ========================================
// Filter-Bar
// ========================================

function buildRequestFilters() {
    return `
        <div class="filter-group">
            <label class="filter-label">IP:</label>
            <input type="text" id="filterIp" class="filter-input" placeholder="IP-Adresse...">
        </div>
        <div class="filter-group">
            <label class="filter-label">Modell:</label>
            <select id="filterModel" class="filter-select">
                <option value="">Alle</option>
            </select>
        </div>
        <div class="filter-group">
            <label class="filter-label">Status:</label>
            <select id="filterStatus" class="filter-select">
                <option value="">Alle</option>
                <option value="2xx">2xx Erfolg</option>
                <option value="4xx">4xx Client-Fehler</option>
                <option value="5xx">5xx Server-Fehler</option>
            </select>
        </div>
        <div class="filter-group">
            <label class="toggle-switch" title="Nur aktive Requests anzeigen">
                <input type="checkbox" id="filterActiveOnly">
                <span class="toggle-slider"></span>
            </label>
            <label class="filter-label" for="filterActiveOnly">Nur aktive</label>
        </div>
        <div class="filter-group">
            <button class="action-btn primary" id="btnRefreshRequests" title="Neu laden">&#x21bb; Aktualisieren</button>
        </div>
    `;
}

function initRequestFilterListeners() {
    const ipInput = document.getElementById('filterIp');
    const modelSelect = document.getElementById('filterModel');
    const statusSelect = document.getElementById('filterStatus');
    const activeCheck = document.getElementById('filterActiveOnly');
    const refreshBtn = document.getElementById('btnRefreshRequests');

    if (ipInput) {
        ipInput.addEventListener('input', () => {
            clearTimeout(requestFilterDebounce);
            requestFilterDebounce = setTimeout(() => { reqCurrentPage = 1; loadRequests(); }, 300);
        });
    }

    if (modelSelect) {
        modelSelect.addEventListener('change', () => { reqCurrentPage = 1; loadRequests(); });
    }

    if (statusSelect) {
        statusSelect.addEventListener('change', () => { reqCurrentPage = 1; loadRequests(); });
    }

    if (activeCheck) {
        activeCheck.addEventListener('change', () => { reqCurrentPage = 1; loadRequests(); });
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => loadRequests());
    }
}

// ========================================
// Daten laden
// ========================================

function getFilterParams() {
    const params = new URLSearchParams();
    params.set('limit', String(reqPageSize));
    params.set('offset', String((reqCurrentPage - 1) * reqPageSize));

    const ip = document.getElementById('filterIp')?.value?.trim();
    if (ip) params.set('ip', ip);

    const model = document.getElementById('filterModel')?.value;
    if (model) params.set('model', model);

    const status = document.getElementById('filterStatus')?.value;
    if (status) params.set('status', status);

    const activeOnly = document.getElementById('filterActiveOnly')?.checked;
    params.set('active_only', activeOnly ? 'true' : 'false');

    return params;
}

async function loadRequests() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    try {
        const params = getFilterParams();
        const response = await fetch(`/api/requests/recent?${params}`);
        const data = await response.json();

        requestsData = data.requests || [];
        reqTotalItems = data.total || 0;

        updateModelFilter();
        renderRequestsTable();
    } catch (error) {
        console.error('Fehler beim Laden der Requests:', error);
        if (container) {
            container.innerHTML = '<div class="no-data">Fehler beim Laden der Requests.</div>';
        }
    }
}

/**
 * Befuellt den Modell-Filter dynamisch aus den geladenen Daten.
 */
function updateModelFilter() {
    const modelSelect = document.getElementById('filterModel');
    if (!modelSelect) return;

    const currentValue = modelSelect.value;
    const models = new Set();

    requestsData.forEach(r => {
        if (r.model) models.add(r.model);
    });

    const sortedModels = Array.from(models).sort();

    // Optionen neu aufbauen
    let html = '<option value="">Alle</option>';
    sortedModels.forEach(m => {
        const selected = m === currentValue ? ' selected' : '';
        html += `<option value="${escapeHtml(m)}"${selected}>${escapeHtml(m)}</option>`;
    });

    modelSelect.innerHTML = html;
}

// ========================================
// Pagination
// ========================================

function getTotalPages() {
    return Math.max(1, Math.ceil(reqTotalItems / reqPageSize));
}

function goToPage(page) {
    const totalPages = getTotalPages();
    page = Math.max(1, Math.min(page, totalPages));
    if (page === reqCurrentPage) return;
    reqCurrentPage = page;
    loadRequests();
}

function changePageSize(newSize) {
    reqPageSize = newSize;
    reqCurrentPage = 1;
    loadRequests();
}

function buildPaginationBar() {
    const totalPages = getTotalPages();

    // Seitengroesse-Auswahl
    const sizes = [50, 100, 200, 500];
    const sizeOptions = sizes.map(s =>
        `<option value="${s}"${s === reqPageSize ? ' selected' : ''}>${s}</option>`
    ).join('');

    // Seitenzahlen generieren
    let pageButtons = '';
    const maxVisible = 5;
    let startPage = Math.max(1, reqCurrentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    if (startPage > 1) {
        pageButtons += `<button class="page-btn" onclick="goToPage(1)">1</button>`;
        if (startPage > 2) pageButtons += `<span class="page-ellipsis">...</span>`;
    }

    for (let i = startPage; i <= endPage; i++) {
        const active = i === reqCurrentPage ? ' active' : '';
        pageButtons += `<button class="page-btn${active}" onclick="goToPage(${i})">${i}</button>`;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) pageButtons += `<span class="page-ellipsis">...</span>`;
        pageButtons += `<button class="page-btn" onclick="goToPage(${totalPages})">${totalPages}</button>`;
    }

    const from = reqTotalItems === 0 ? 0 : (reqCurrentPage - 1) * reqPageSize + 1;
    const to = Math.min(reqCurrentPage * reqPageSize, reqTotalItems);

    return `
        <div class="pagination-bar">
            <div class="pagination-info">
                ${from}-${to} von ${reqTotalItems} Requests
            </div>
            <div class="pagination-controls">
                <button class="page-btn nav-btn" onclick="goToPage(${reqCurrentPage - 1})" ${reqCurrentPage <= 1 ? 'disabled' : ''}>&#x25C0;</button>
                ${pageButtons}
                <button class="page-btn nav-btn" onclick="goToPage(${reqCurrentPage + 1})" ${reqCurrentPage >= totalPages ? 'disabled' : ''}>&#x25B6;</button>
            </div>
            <div class="pagination-size">
                <label>Pro Seite:</label>
                <select onchange="changePageSize(Number(this.value))">${sizeOptions}</select>
            </div>
        </div>
    `;
}

// ========================================
// Tabelle rendern
// ========================================

function renderRequestsTable() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    if (requestsData.length === 0 && reqTotalItems === 0) {
        container.innerHTML = '<div class="no-data">Keine Requests gefunden.</div>';
        return;
    }

    let html = buildPaginationBar();

    html += `
        <table class="request-table">
            <thead>
                <tr>
                    <th>Zeit</th>
                    <th>Client-IP</th>
                    <th>Methode</th>
                    <th>Endpunkt</th>
                    <th>Modell</th>
                    <th>Status</th>
                    <th>Dauer</th>
                    <th>Tokens</th>
                </tr>
            </thead>
            <tbody id="requestsTableBody">
    `;

    requestsData.forEach(req => {
        html += buildRequestRow(req);
    });

    html += '</tbody></table>';

    // Pagination auch unten
    html += buildPaginationBar();

    container.innerHTML = html;
}

function buildRequestRow(req) {
    const rowClass = getRowClass(req);
    const statusBadge = buildStatusBadge(req.status_code, req.state);
    const duration = formatDuration(req.duration_ms);
    const tokens = req.tokens_generated != null ? req.tokens_generated : '-';
    const time = formatTime(req.timestamp_iso);
    const model = req.model || '-';
    const method = req.method || '-';
    const path = req.path || '-';
    const ip = req.client_ip || '-';

    return `
        <tr class="${rowClass}" data-request-id="${escapeHtml(String(req.id))}" onclick="toggleRequestDetail(this)">
            <td>${escapeHtml(time)}</td>
            <td>${escapeHtml(ip)}</td>
            <td>${escapeHtml(method)}</td>
            <td title="${escapeHtml(path)}">${escapeHtml(truncatePath(path, 30))}</td>
            <td>${escapeHtml(model)}</td>
            <td>${statusBadge}</td>
            <td>${escapeHtml(duration)}</td>
            <td>${escapeHtml(String(tokens))}</td>
        </tr>
        <tr class="request-detail-row" style="display: none;">
            <td colspan="8">
                <div class="request-detail">
                    <div class="request-detail-grid">
                        <div><strong>Request-ID:</strong> ${escapeHtml(String(req.id))}</div>
                        <div><strong>Request-Groesse:</strong> ${formatBytes(req.request_size)}</div>
                        <div><strong>Response-Groesse:</strong> ${formatBytes(req.response_size)}</div>
                        <div><strong>Streaming:</strong> ${req.is_streaming ? 'Ja' : 'Nein'}</div>
                        <div><strong>Status:</strong> ${escapeHtml(req.state || '-')}</div>
                        ${buildDiagnosticMessage(req)}
                    </div>
                    <div class="request-body-section" data-request-id="${escapeHtml(String(req.id))}"></div>
                </div>
            </td>
        </tr>
    `;
}

async function toggleRequestDetail(rowEl) {
    const detailRow = rowEl.nextElementSibling;
    if (!detailRow || !detailRow.classList.contains('request-detail-row')) return;

    if (detailRow.style.display === 'none') {
        detailRow.style.display = '';

        // Lazy-Load: Body-Daten beim ersten Aufklappen laden
        const requestId = rowEl.getAttribute('data-request-id');
        const bodySection = detailRow.querySelector('.request-body-section');
        if (bodySection && !bodySection.dataset.loaded) {
            bodySection.innerHTML = '<div class="body-loading">Lade Details...</div>';
            try {
                let detail = requestDetailCache[requestId];
                if (!detail) {
                    const resp = await fetch(`/api/requests/${encodeURIComponent(requestId)}`);
                    detail = await resp.json();
                    requestDetailCache[requestId] = detail;
                }
                bodySection.innerHTML = renderBodyContent(detail);
                bodySection.dataset.loaded = 'true';
            } catch (err) {
                bodySection.innerHTML = '<div class="body-loading">Fehler beim Laden der Details.</div>';
                console.error('Detail-Load Fehler:', err);
            }
        }
    } else {
        detailRow.style.display = 'none';
    }
}

/**
 * Rendert Prompt und Antwort als formatierte Bloecke.
 */
function renderBodyContent(detail) {
    if (!detail) return '';

    const requestBody = detail.request_body || '';
    const responseBody = detail.response_body || '';

    if (!requestBody && !responseBody) {
        return '<div class="body-loading">Keine Prompt/Antwort-Daten vorhanden.</div>';
    }

    let html = '';

    if (requestBody) {
        let formattedBody;
        try {
            const parsed = JSON.parse(requestBody);
            formattedBody = syntaxHighlightJson(parsed);
        } catch (e) {
            formattedBody = escapeHtml(requestBody);
        }
        html += `
            <div class="body-block body-prompt">
                <div class="body-label">Prompt (Request-Body)</div>
                <pre class="body-content">${formattedBody}</pre>
            </div>
        `;
    }

    if (responseBody) {
        let formattedResponse;
        try {
            const parsed = JSON.parse(responseBody);
            formattedResponse = syntaxHighlightJson(parsed);
        } catch (e) {
            formattedResponse = escapeHtml(responseBody);
        }
        html += `
            <div class="body-block body-response">
                <div class="body-label">Antwort (LLM-Response)</div>
                <pre class="body-content">${formattedResponse}</pre>
            </div>
        `;
    }

    return html;
}

/**
 * JSON-Syntax-Highlighting: Keys, Strings, Zahlen, Booleans, Null farbig markieren.
 */
function syntaxHighlightJson(obj) {
    const json = JSON.stringify(obj, null, 2);
    // Regex-basiertes Highlighting auf dem escaped JSON-String
    return json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/("(\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")\s*:/g,
            '<span class="json-key">$1</span>:')
        .replace(/"(\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"/g,
            '<span class="json-string">$&</span>')
        .replace(/\b(-?\d+(\.\d+)?([eE][+-]?\d+)?)\b/g,
            '<span class="json-number">$1</span>')
        .replace(/\b(true|false)\b/g,
            '<span class="json-boolean">$1</span>')
        .replace(/\bnull\b/g,
            '<span class="json-null">null</span>');
}

// ========================================
// WebSocket-Handler
// ========================================

/**
 * Fuegt einen neuen Request oben in die Tabelle ein (nur auf Seite 1).
 */
function handleNewRequest(data) {
    if (!data || !data.id) return;

    reqTotalItems++;

    // Nur auf Seite 1 live einfuegen
    if (reqCurrentPage !== 1) {
        updatePaginationBars();
        return;
    }

    // In lokale Daten einfuegen
    requestsData.unshift(data);

    // Ueberschuessige Eintraege entfernen
    if (requestsData.length > reqPageSize) {
        requestsData.pop();
    }

    // Zeile in DOM einfuegen
    const tbody = document.getElementById('requestsTableBody');
    if (!tbody) return;

    const tempContainer = document.createElement('tbody');
    tempContainer.innerHTML = buildRequestRow(data);

    const mainRow = tempContainer.querySelector('tr');
    const detailRow = tempContainer.querySelectorAll('tr')[1];

    if (mainRow) {
        mainRow.classList.add('request-new');
        // Highlight nach 2s entfernen
        setTimeout(() => {
            mainRow.classList.remove('request-new');
        }, 2000);
    }

    // Am Anfang der Tabelle einfuegen
    if (tbody.firstChild) {
        if (detailRow) tbody.insertBefore(detailRow, tbody.firstChild);
        if (mainRow) tbody.insertBefore(mainRow, tbody.firstChild);
    } else {
        if (mainRow) tbody.appendChild(mainRow);
        if (detailRow) tbody.appendChild(detailRow);
    }

    // Ueberschuessige DOM-Zeilen entfernen
    const allDataRows = tbody.querySelectorAll('tr[data-request-id]');
    if (allDataRows.length > reqPageSize) {
        const lastDataRow = allDataRows[allDataRows.length - 1];
        const lastDetailRow = lastDataRow.nextElementSibling;
        if (lastDetailRow && lastDetailRow.classList.contains('request-detail-row')) {
            tbody.removeChild(lastDetailRow);
        }
        tbody.removeChild(lastDataRow);
    }

    // Modell-Filter aktualisieren falls neues Modell
    if (data.model) {
        updateModelFilter();
    }

    updatePaginationBars();
}

/**
 * Aktualisiert einen bestehenden Request in der Tabelle.
 */
function handleUpdateRequest(data) {
    if (!data || !data.id) return;

    // Detail-Cache invalidieren bei Update
    delete requestDetailCache[data.id];

    // Lokale Daten aktualisieren
    const idx = requestsData.findIndex(r => r.id === data.id);
    if (idx !== -1) {
        requestsData[idx] = { ...requestsData[idx], ...data };
    }

    // DOM-Zeile finden und aktualisieren
    const row = document.querySelector(`tr[data-request-id="${data.id}"]`);
    if (!row) return;

    const req = idx !== -1 ? requestsData[idx] : data;

    // Zellen aktualisieren (Status, Dauer, Tokens)
    const cells = row.querySelectorAll('td');
    if (cells.length >= 8) {
        // Status (Index 5)
        cells[5].innerHTML = buildStatusBadge(req.status_code, req.state);
        // Dauer (Index 6)
        cells[6].textContent = formatDuration(req.duration_ms);
        // Tokens (Index 7)
        cells[7].textContent = req.tokens_generated != null ? req.tokens_generated : '-';
    }

    // Zeilen-Klasse aktualisieren
    row.className = getRowClass(req);

    // Detail-Row: Body-Section als nicht geladen markieren (fuer Neuladen beim naechsten Aufklappen)
    const detailRow = row.nextElementSibling;
    if (detailRow && detailRow.classList.contains('request-detail-row')) {
        const detailGrid = detailRow.querySelector('.request-detail-grid');
        if (detailGrid) {
            detailGrid.innerHTML = `
                <div><strong>Request-ID:</strong> ${escapeHtml(String(req.id))}</div>
                <div><strong>Request-Groesse:</strong> ${formatBytes(req.request_size)}</div>
                <div><strong>Response-Groesse:</strong> ${formatBytes(req.response_size)}</div>
                <div><strong>Streaming:</strong> ${req.is_streaming ? 'Ja' : 'Nein'}</div>
                <div><strong>Status:</strong> ${escapeHtml(req.state || '-')}</div>
                ${buildDiagnosticMessage(req)}
            `;
        }
        // Body-Section zuruecksetzen fuer Neuladen
        const bodySection = detailRow.querySelector('.request-body-section');
        if (bodySection) {
            bodySection.removeAttribute('data-loaded');
            bodySection.innerHTML = '';
        }
    }
}

/**
 * Aktualisiert die Pagination-Leisten im DOM ohne Tabelle neu zu rendern.
 */
function updatePaginationBars() {
    const bars = document.querySelectorAll('.pagination-bar');
    if (bars.length === 0) return;

    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = buildPaginationBar();
    const newBar = tempDiv.querySelector('.pagination-bar');

    bars.forEach(bar => {
        bar.innerHTML = newBar.innerHTML;
    });
}

// ========================================
// Hilfsfunktionen
// ========================================

function getRowClass(req) {
    const classes = [];

    if (req.state === 'active') {
        classes.push('request-active');
    }

    if (req.state === 'warning') {
        classes.push('request-warning');
    }

    if (req.status_code && req.status_code >= 400) {
        classes.push('request-error');
    }

    return classes.join(' ');
}

function buildStatusBadge(statusCode, state) {
    if (state === 'active') {
        return '<span class="status-badge status-active">aktiv</span>';
    }
    if (state === 'warning') {
        return '<span class="status-badge status-warning">warning</span>';
    }
    if (!statusCode || statusCode === 0) {
        return `<span class="status-badge">${escapeHtml(String(statusCode || '-'))}</span>`;
    }
    if (statusCode >= 200 && statusCode < 300) {
        return `<span class="status-badge status-2xx">${escapeHtml(String(statusCode))}</span>`;
    }
    if (statusCode >= 400 && statusCode < 500) {
        return `<span class="status-badge status-4xx">${escapeHtml(String(statusCode))}</span>`;
    }
    if (statusCode >= 500) {
        return `<span class="status-badge status-5xx">${escapeHtml(String(statusCode))}</span>`;
    }
    return `<span class="status-badge">${escapeHtml(String(statusCode))}</span>`;
}

function buildDiagnosticMessage(req) {
    if (!req.error_message) return '';
    const isWarning = req.state === 'warning';
    const cls = isWarning ? 'request-detail-warning' : 'request-detail-error';
    const label = isWarning ? 'Warnung' : 'Fehler';
    return `<div class="${cls}"><strong>${label}:</strong> ${escapeHtml(req.error_message)}</div>`;
}

function formatDuration(ms) {
    if (ms === null || ms === undefined) return '-';
    if (ms < 1000) return ms + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
    return (ms / 60000).toFixed(1) + 'min';
}

function formatTime(isoString) {
    if (!isoString) return '-';
    try {
        const d = new Date(isoString);
        const pad = (n) => String(n).padStart(2, '0');
        return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}. ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    } catch (e) {
        return isoString;
    }
}

function formatBytes(bytes) {
    if (bytes === null || bytes === undefined) return '-';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function truncatePath(path, maxLen) {
    if (!path) return '-';
    if (path.length <= maxLen) return path;
    return path.substring(0, maxLen) + '...';
}

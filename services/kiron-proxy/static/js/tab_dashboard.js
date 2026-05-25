/**
 * Ollama Monitor - Dashboard Tab (Uebersicht)
 * Zeigt KPI-Widgets mit Zusammenfassung und System-Status.
 */

async function initDashboardTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        // Beide APIs parallel laden
        const [summaryResp, metricsResp] = await Promise.all([
            fetch('/api/dashboard/summary'),
            fetch('/api/metrics/system')
        ]);
        const summary = await summaryResp.json();
        if (!summaryResp.ok) {
            throw new Error(summary.error || summaryResp.statusText);
        }
        if (!metricsResp.ok) {
            var metricsError = await metricsResp.json().catch(function() { return {}; });
            if (metricsResp.status === 503) {
                var retryAfter = Number(metricsError.retry_after_s) || 2;
                window.setTimeout(function() {
                    if (typeof currentTab === 'undefined' || currentTab === 'dashboard') {
                        initDashboardTab();
                    }
                }, retryAfter * 1000);
                return;
            }
            throw new Error(metricsError.error || metricsResp.statusText);
        }
        const metrics = await metricsResp.json();

        container.innerHTML = renderDashboardOverview(summary, metrics);
    } catch (error) {
        console.error('Dashboard-Fehler:', error);
        container.innerHTML = '<div class="kpi-error"><span class="kpi-error-icon">\u26A0\uFE0F</span><span class="kpi-error-text">Fehler beim Laden</span></div>';
    }
}

/**
 * Formatiert Millisekunden als lesbaren Latenz-String.
 * Unter 1000ms: "123ms", darueber: "1.2s"
 */
function formatLatency(ms) {
    if (ms === null || ms === undefined) return '-';
    const num = parseFloat(ms);
    if (isNaN(num)) return '-';
    if (num < 1000) {
        return Math.round(num) + 'ms';
    }
    return (num / 1000).toFixed(1) + 's';
}

/**
 * Formatiert GB-Werte.
 */
function formatGB(value) {
    if (value === null || value === undefined) return '-';
    const num = parseFloat(value);
    if (isNaN(num)) return '-';
    return num.toFixed(1) + ' GB';
}

/**
 * Rendert das Ollama-Status-Widget (Version + geladene Modelle).
 */
function renderOllamaStatusWidget(metrics) {
    var ollama = metrics.ollama || {};

    var html = '';
    html += '<div class="widget" data-widget="ollama-status">';
    html += '<div class="widget-header"><h3 class="widget-title">Ollama Status</h3></div>';
    html += '<div class="widget-body">';

    // Version
    html += '<div style="padding: 12px 14px; font-size: 0.9rem; border-bottom: 1px solid var(--border-color);">';
    var loadedCountText = (ollama.models_loaded_state === 'unknown' || ollama.models_loaded_count === null || ollama.models_loaded_count === undefined)
        ? 'Unbekannt'
        : String(ollama.models_loaded_count);
    html += '<strong>Version:</strong> ' + escapeHtml(String(ollama.version || 'Unbekannt'));
    html += ' &mdash; <strong>Geladene Modelle:</strong> ' + escapeHtml(loadedCountText);
    html += '</div>';

    // Geladene Modelle Tabelle
    var loadedModels = ollama.models_loaded || [];
    if (loadedModels.length > 0) {
        html += '<table class="widget-table">';
        html += '<thead><tr><th>Modell</th><th>VRAM</th></tr></thead>';
        html += '<tbody>';
        for (var i = 0; i < loadedModels.length; i++) {
            var model = loadedModels[i];
            var vramText = (model.vram_state !== 'unknown' && model.vram_gb !== null && model.vram_gb !== undefined)
                ? parseFloat(model.vram_gb).toFixed(1) + ' GB'
                : 'Unbekannt';
            html += '<tr>';
            html += '<td>' + escapeHtml(String(model.name || '-')) + '</td>';
            html += '<td>' + escapeHtml(vramText) + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table>';
    } else if (ollama.models_loaded_state === 'unknown') {
        html += '<div class="kpi-empty"><span class="kpi-empty-text">Load-Zustand unbekannt</span></div>';
    } else {
        html += '<div class="kpi-empty"><span class="kpi-empty-text">Keine Modelle geladen</span></div>';
    }

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget
    return html;
}

/**
 * Rendert das komplette Dashboard-Overview HTML.
 */
function renderDashboardOverview(summary, metrics) {
    let html = '<div class="widget-dashboard widget-dashboard--masonry">';

    // ========================================
    // 1. Obere KPI-Reihe (volle Breite, 4 Karten)
    // ========================================
    html += '<div class="widget widget--full">';
    html += '<div class="widget-header"><h3 class="widget-title">Uebersicht</h3></div>';
    html += '<div class="widget-body">';
    html += '<div class="kpi-cards">';

    // Requests heute
    html += '<div class="kpi-card" data-kpi-card="requests_today">';
    html += '<div class="kpi-icon">\u{1F4CA}</div>';
    html += '<div class="kpi-value" data-kpi="requests_today">' + escapeHtml(String(summary.requests_today ?? 0)) + '</div>';
    html += '<div class="kpi-title">Requests heute</div>';
    html += '</div>';

    // Aktive Requests
    var activeClass = (summary.active_now > 0) ? 'kpi-green' : '';
    html += '<div class="kpi-card ' + activeClass + '" data-kpi-card="active_now">';
    html += '<div class="kpi-icon">\u26A1</div>';
    html += '<div class="kpi-value" data-kpi="active_now">' + escapeHtml(String(summary.active_now ?? 0)) + '</div>';
    html += '<div class="kpi-title">Aktive Requests</div>';
    html += '</div>';

    // Avg. Latenz
    html += '<div class="kpi-card" data-kpi-card="avg_duration_ms">';
    html += '<div class="kpi-icon">\u23F1</div>';
    html += '<div class="kpi-value" data-kpi="avg_duration_ms">' + escapeHtml(formatLatency(summary.avg_duration_ms)) + '</div>';
    html += '<div class="kpi-title">Avg. Latenz</div>';
    html += '</div>';

    // Fehlerrate
    var errorRate = parseFloat(summary.error_rate) || 0;
    var errorClass = (errorRate > 5) ? 'kpi-red' : '';
    html += '<div class="kpi-card ' + errorClass + '" data-kpi-card="error_rate">';
    html += '<div class="kpi-icon">\u{1F6AB}</div>';
    html += '<div class="kpi-value" data-kpi="error_rate">' + escapeHtml(errorRate.toFixed(1) + '%') + '</div>';
    html += '<div class="kpi-title">Fehlerrate</div>';
    html += '</div>';

    html += '</div>'; // .kpi-cards
    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    // ========================================
    // 2. System-Ressourcen + Ollama Status
    //    Reihenfolge: GPU, Disk I/O, RAM, Ollama Status, CPU
    // ========================================
    html += renderGpuWidget(metrics);
    html += renderDiskWidget(metrics);
    html += renderRamWidget(metrics);
    html += renderOllamaStatusWidget(metrics);
    html += renderCpuWidget(metrics);

    // ========================================
    // 4. Top Clients Tabelle
    // ========================================
    html += '<div class="widget" data-widget="top-clients">';
    html += '<div class="widget-header"><h3 class="widget-title">Top Clients</h3></div>';
    html += '<div class="widget-body">';

    var topClients = summary.top_clients || [];
    if (topClients.length > 0) {
        html += '<table class="widget-table">';
        html += '<thead><tr><th>IP-Adresse</th><th>Anzahl</th></tr></thead>';
        html += '<tbody>';
        for (var c = 0; c < topClients.length; c++) {
            html += '<tr>';
            html += '<td>' + escapeHtml(String(topClients[c].ip || '-')) + '</td>';
            html += '<td>' + escapeHtml(String(topClients[c].count ?? 0)) + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table>';
    } else {
        html += '<div class="kpi-empty"><span class="kpi-empty-text">Keine Daten</span></div>';
    }

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    // ========================================
    // 5. Top Modelle Tabelle
    // ========================================
    html += '<div class="widget" data-widget="top-models">';
    html += '<div class="widget-header"><h3 class="widget-title">Top Modelle</h3></div>';
    html += '<div class="widget-body">';

    var topModels = summary.top_models || [];
    if (topModels.length > 0) {
        html += '<table class="widget-table">';
        html += '<thead><tr><th>Modell</th><th>Anzahl</th><th>Avg. Latenz</th></tr></thead>';
        html += '<tbody>';
        for (var m = 0; m < topModels.length; m++) {
            html += '<tr>';
            html += '<td>' + escapeHtml(String(topModels[m].model || '-')) + '</td>';
            html += '<td>' + escapeHtml(String(topModels[m].count ?? 0)) + '</td>';
            html += '<td>' + escapeHtml(formatLatency(topModels[m].avg_duration_ms)) + '</td>';
            html += '</tr>';
        }
        html += '</tbody></table>';
    } else {
        html += '<div class="kpi-empty"><span class="kpi-empty-text">Keine Daten</span></div>';
    }

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    html += '</div>'; // .widget-dashboard
    return html;
}

/**
 * Live-Update via WebSocket.
 * Aktualisiert KPI-Werte im DOM ohne komplett neu zu rendern.
 * Wird von app_core.js aufgerufen mit data = { system: {...}, summary: {...} }
 */
function updateDashboardFromWS(data) {
    var summary = data.summary || {};

    // --- Uebersicht KPI-Cards ---

    // Requests heute
    var reqEl = document.querySelector('[data-kpi="requests_today"]');
    if (reqEl) {
        reqEl.textContent = String(summary.requests_today ?? 0);
    }

    // Aktive Requests
    var activeEl = document.querySelector('[data-kpi="active_now"]');
    if (activeEl) {
        activeEl.textContent = String(summary.active_now ?? 0);
        var activeCard = document.querySelector('[data-kpi-card="active_now"]');
        if (activeCard) {
            activeCard.classList.toggle('kpi-green', (summary.active_now > 0));
        }
    }

    // Avg. Latenz
    var latEl = document.querySelector('[data-kpi="avg_duration_ms"]');
    if (latEl) {
        latEl.textContent = formatLatency(summary.avg_duration_ms);
    }

    // Fehlerrate
    var errEl = document.querySelector('[data-kpi="error_rate"]');
    if (errEl) {
        var errRate = parseFloat(summary.error_rate) || 0;
        errEl.textContent = errRate.toFixed(1) + '%';
        var errCard = document.querySelector('[data-kpi-card="error_rate"]');
        if (errCard) {
            errCard.classList.toggle('kpi-red', (errRate > 5));
        }
    }

    // --- System KPI-Cards (CPU, RAM, GPU, VRAM, Disk I/O) - geteilt mit System-Tab ---
    updateSystemResourceKpis(data);
}

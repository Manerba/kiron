/**
 * Ollama Monitor - System Tab
 * Zeigt detaillierte System-Metriken (CPU, RAM, GPU, Disk I/O).
 */

async function initSystemTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await fetch('/api/metrics/system');
        if (!resp.ok) {
            var err = await resp.json().catch(function() { return {}; });
            if (resp.status === 503) {
                var retryAfter = Number(err.retry_after_s) || 2;
                window.setTimeout(function() {
                    if (typeof currentTab === 'undefined' || currentTab === 'system') {
                        initSystemTab();
                    }
                }, retryAfter * 1000);
                return;
            }
            throw new Error(err.error || resp.statusText);
        }
        const metrics = await resp.json();
        container.innerHTML = renderSystemMetrics(metrics);
    } catch (error) {
        console.error('System-Fehler:', error);
        container.innerHTML = '<div class="kpi-error"><span class="kpi-error-icon">\u26A0\uFE0F</span><span class="kpi-error-text">Fehler beim Laden</span></div>';
    }
}

// ========================================
// Formatierungshilfen
// ========================================

function sysFormatPercent(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return Math.round(num) + '%';
}

function sysFormatGB(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return num.toFixed(1) + ' GB';
}

function sysFormatMB(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return Math.round(num) + ' MB';
}

function sysIsKnownNumber(value) {
    return typeof value === 'number' && isFinite(value);
}

function sysGetGpuProcessesPayload(metrics) {
    var payload = metrics.gpu_processes || {};
    var state = typeof payload.state === 'string' ? payload.state : 'error';
    return {
        state: state,
        data: Array.isArray(payload.data) ? payload.data : [],
        known: state === 'ok' || state === 'empty'
    };
}

function sysGetMaintenanceView(maintenance) {
    maintenance = maintenance || {};
    var active = maintenance.active === true;
    var transitioning = maintenance.transitioning === true;
    var targetActive = maintenance.target_active === true;
    return {
        active: active,
        transitioning: transitioning,
        targetActive: targetActive,
        label: transitioning
            ? (targetActive ? 'AKTIVIERT...' : 'DEAKTIVIERT...')
            : (active ? 'AKTIV' : 'INAKTIV'),
        activeClass: active || (transitioning && targetActive)
    };
}

function sysFormatMBs(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return num.toFixed(1) + ' MB/s';
}

function sysFormatTemp(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return Math.round(num) + ' \u00B0C';
}

function sysFormatWatt(value) {
    if (value === null || value === undefined) return '-';
    var num = parseFloat(value);
    if (isNaN(num)) return '-';
    return Math.round(num) + ' W';
}

function sysGetPercentColor(value) {
    var num = parseFloat(value);
    if (isNaN(num)) return '';
    if (num > 80) return 'kpi-red';
    if (num > 50) return 'kpi-orange';
    return '';
}

// ========================================
// Haupt-Render-Funktion
// ========================================

/**
 * Rendert das komplette System-Metriken Grid mit 5 Widgets.
 */
/**
 * Einzelne System-Ressourcen-Widgets (CPU, RAM, GPU, Disk I/O). Jede Funktion
 * liefert genau einen .widget-Block, damit Aufrufer (Dashboard) sie in
 * beliebiger Reihenfolge in ihren eigenen Container setzen koennen.
 */
function renderCpuWidget(metrics) {
    var cpu = metrics.cpu || {};

    var html = '';

    html += '<div class="widget" data-widget="sys-cpu">';
    html += '<div class="widget-header"><h3 class="widget-title">CPU</h3></div>';
    html += '<div class="widget-body">';

    // KPI-Card: Auslastung
    var cpuColor = sysGetPercentColor(cpu.usage_percent);
    html += '<div class="kpi-cards">';
    html += '<div class="kpi-card ' + cpuColor + '" data-kpi-card="sys_cpu_usage" style="grid-column: 1 / -1;">';
    html += '<div class="kpi-icon">\u{1F4BB}</div>';
    html += '<div class="kpi-value" data-kpi="sys_cpu_usage">' + escapeHtml(sysFormatPercent(cpu.usage_percent)) + '</div>';
    html += '<div class="kpi-title">Auslastung</div>';
    html += '</div>';
    html += '</div>';

    // Detail-Tabelle
    html += '<table class="widget-table">';
    html += '<tbody>';
    html += '<tr><td>Load 1m</td><td>' + escapeHtml(String(cpu.load_avg_1m != null ? parseFloat(cpu.load_avg_1m).toFixed(2) : '-')) + '</td></tr>';
    html += '<tr><td>Load 5m</td><td>' + escapeHtml(String(cpu.load_avg_5m != null ? parseFloat(cpu.load_avg_5m).toFixed(2) : '-')) + '</td></tr>';
    html += '<tr><td>Load 15m</td><td>' + escapeHtml(String(cpu.load_avg_15m != null ? parseFloat(cpu.load_avg_15m).toFixed(2) : '-')) + '</td></tr>';
    html += '<tr><td>Kerne</td><td>' + escapeHtml(String(cpu.core_count ?? '-')) + '</td></tr>';
    if (cpu.freq_mhz != null) {
        html += '<tr><td>Frequenz</td><td>' + escapeHtml(String(Math.round(cpu.freq_mhz)) + ' MHz') + '</td></tr>';
    }
    html += '</tbody>';
    html += '</table>';

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    return html;
}

function renderRamWidget(metrics) {
    var memory = metrics.memory || {};

    var html = '';

    html += '<div class="widget" data-widget="sys-ram">';
    html += '<div class="widget-header"><h3 class="widget-title">Arbeitsspeicher</h3></div>';
    html += '<div class="widget-body">';

    // KPI-Card: Auslastung
    var ramColor = sysGetPercentColor(memory.usage_percent);
    html += '<div class="kpi-cards">';
    html += '<div class="kpi-card ' + ramColor + '" data-kpi-card="sys_ram_usage" style="grid-column: 1 / -1;">';
    html += '<div class="kpi-icon">\u{1F9E0}</div>';
    html += '<div class="kpi-value" data-kpi="sys_ram_usage">' + escapeHtml(sysFormatPercent(memory.usage_percent)) + '</div>';
    html += '<div class="kpi-title">Auslastung</div>';
    html += '</div>';
    html += '</div>';

    // Detail-Tabelle
    html += '<table class="widget-table">';
    html += '<tbody>';
    html += '<tr><td>Belegt</td><td>' + escapeHtml(sysFormatGB(memory.used_gb)) + '</td></tr>';
    html += '<tr><td>Gesamt</td><td>' + escapeHtml(sysFormatGB(memory.total_gb)) + '</td></tr>';
    html += '<tr><td>Verfuegbar</td><td>' + escapeHtml(sysFormatGB(memory.available_gb)) + '</td></tr>';
    html += '<tr><td>Swap belegt</td><td>' + escapeHtml(sysFormatGB(memory.swap_used_gb)) + '</td></tr>';
    html += '</tbody>';
    html += '</table>';

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    return html;
}

function renderGpuWidget(metrics) {
    var gpu = metrics.gpu || {};
    var gpuAvailable = !gpu.error;

    var html = '';

    html += '<div class="widget" data-widget="sys-gpu">';
    html += '<div class="widget-header"><h3 class="widget-title">GPU</h3></div>';
    html += '<div class="widget-body">';

    if (gpuAvailable) {
        // KPI-Cards: GPU Util + VRAM
        var gpuUtilColor = sysGetPercentColor(gpu.gpu_util_percent);
        var vramColor = sysGetPercentColor(gpu.vram_usage_percent);

        html += '<div class="kpi-cards">';

        html += '<div class="kpi-card ' + gpuUtilColor + '" data-kpi-card="sys_gpu_util">';
        html += '<div class="kpi-icon">\u{1F3AE}</div>';
        html += '<div class="kpi-value" data-kpi="sys_gpu_util">' + escapeHtml(sysFormatPercent(gpu.gpu_util_percent)) + '</div>';
        html += '<div class="kpi-title">GPU Util</div>';
        html += '</div>';

        html += '<div class="kpi-card ' + vramColor + '" data-kpi-card="sys_vram_usage">';
        html += '<div class="kpi-icon">\u{1F4BE}</div>';
        html += '<div class="kpi-value" data-kpi="sys_vram_usage">' + escapeHtml(sysFormatPercent(gpu.vram_usage_percent)) + '</div>';
        html += '<div class="kpi-title">VRAM</div>';
        html += '</div>';

        html += '</div>'; // .kpi-cards

        // Detail-Tabelle
        html += '<table class="widget-table">';
        html += '<tbody>';
        html += '<tr><td>GPU</td><td>' + escapeHtml(String(gpu.name || '-')) + '</td></tr>';
        html += '<tr><td>VRAM belegt</td><td>' + escapeHtml(sysFormatMB(gpu.vram_used_mb)) + '</td></tr>';
        html += '<tr><td>VRAM gesamt</td><td>' + escapeHtml(sysFormatMB(gpu.vram_total_mb)) + '</td></tr>';
        html += '<tr><td>Temperatur</td><td>' + escapeHtml(sysFormatTemp(gpu.temperature_c)) + '</td></tr>';
        html += '<tr><td>Leistung</td><td>' + escapeHtml(sysFormatWatt(gpu.power_draw_w)) + '</td></tr>';
        html += '</tbody>';
        html += '</table>';
    } else {
        // Fehlermeldung bei nicht verfuegbarer GPU
        var errorMsg = (gpu.error) ? String(gpu.error) : 'GPU nicht verfuegbar';
        html += '<div class="kpi-cards">';
        html += '<div class="kpi-card" data-kpi-card="sys_gpu_util">';
        html += '<div class="kpi-icon">\u{1F3AE}</div>';
        html += '<div class="kpi-value" data-kpi="sys_gpu_util">N/A</div>';
        html += '<div class="kpi-title">GPU Util</div>';
        html += '</div>';
        html += '<div class="kpi-card" data-kpi-card="sys_vram_usage">';
        html += '<div class="kpi-icon">\u{1F4BE}</div>';
        html += '<div class="kpi-value" data-kpi="sys_vram_usage">N/A</div>';
        html += '<div class="kpi-title">VRAM</div>';
        html += '</div>';
        html += '</div>'; // .kpi-cards
        html += '<div class="kpi-empty"><span class="kpi-empty-text">' + escapeHtml(errorMsg) + '</span></div>';
    }

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget

    return html;
}

function renderDiskWidget(metrics) {
    var diskIo = metrics.disk_io || {};

    var html = '';

    html += '<div class="widget" data-widget="sys-disk">';
    html += '<div class="widget-header"><h3 class="widget-title">Disk I/O</h3></div>';
    html += '<div class="widget-body">';

    // KPI-Cards: Read + Write
    html += '<div class="kpi-cards">';

    html += '<div class="kpi-card" data-kpi-card="sys_disk_read">';
    html += '<div class="kpi-icon">\u{1F4E5}</div>';
    html += '<div class="kpi-value" data-kpi="sys_disk_read">' + escapeHtml(sysFormatMBs(diskIo.read_mb_s)) + '</div>';
    html += '<div class="kpi-title">Read</div>';
    html += '</div>';

    html += '<div class="kpi-card" data-kpi-card="sys_disk_write">';
    html += '<div class="kpi-icon">\u{1F4E4}</div>';
    html += '<div class="kpi-value" data-kpi="sys_disk_write">' + escapeHtml(sysFormatMBs(diskIo.write_mb_s)) + '</div>';
    html += '<div class="kpi-title">Write</div>';
    html += '</div>';

    html += '</div>'; // .kpi-cards

    // Detail-Tabelle: IOPS (falls vorhanden)
    var hasIops = (diskIo.read_iops != null || diskIo.write_iops != null);
    if (hasIops) {
        html += '<table class="widget-table">';
        html += '<tbody>';
        if (diskIo.read_iops != null) {
            html += '<tr><td>Read IOPS</td><td>' + escapeHtml(String(Math.round(diskIo.read_iops))) + '</td></tr>';
        }
        if (diskIo.write_iops != null) {
            html += '<tr><td>Write IOPS</td><td>' + escapeHtml(String(Math.round(diskIo.write_iops))) + '</td></tr>';
        }
        html += '</tbody>';
        html += '</table>';
    }

    html += '</div>'; // .widget-body
    html += '</div>'; // .widget (Disk I/O)

    return html;
}

function renderSystemMetrics(metrics) {
    var gpu = metrics.gpu || {};
    var gpuProcessesPayload = sysGetGpuProcessesPayload(metrics);
    var gpuProcesses = gpuProcessesPayload.data;
    var docling = metrics.docling || {};
    var gpuAvailable = !gpu.error;

    var html = '<div class="widget-dashboard">';

    // ========================================
    // GPU Memory Manager (volle Breite)
    // ========================================
    if (gpuAvailable) {
        var vramTotal = gpu.vram_total_mb || 1;
        // Aggregate per label
        var gpuAgg = {};
        var unknownVramCount = sysIsKnownNumber(metrics.gpu_process_vram_unknown_count)
            ? metrics.gpu_process_vram_unknown_count
            : 0;
        for (var i = 0; i < gpuProcesses.length; i++) {
            var p = gpuProcesses[i];
            if (!sysIsKnownNumber(p.vram_mb)) {
                if (!sysIsKnownNumber(metrics.gpu_process_vram_unknown_count)) {
                    unknownVramCount += 1;
                }
                continue;
            }
            var lbl = p.label || 'other';
            gpuAgg[lbl] = (gpuAgg[lbl] || 0) + p.vram_mb;
        }
        var ollamaVram = gpuAgg['ollama'] || 0;
        var doclingVram = gpuAgg['docling'] || 0;
        var embedVram = gpuAgg['embedding'] || 0;
        var debertaVram = gpuAgg['deberta'] || 0;
        var otherVram = gpuAgg['other'] || 0;
        if (!gpuProcessesPayload.known && sysIsKnownNumber(gpu.vram_used_mb)) {
            var attributedVram = ollamaVram + doclingVram + embedVram + debertaVram + otherVram;
            otherVram += Math.max(0, gpu.vram_used_mb - attributedVram);
        }
        var usedVram = ollamaVram + doclingVram + embedVram + debertaVram + otherVram;
        var freeVram = Math.max(0, vramTotal - usedVram);

        html += '<div class="widget" data-widget="sys-gpu-mgr">';
        html += '<div class="widget-header"><h3 class="widget-title">GPU Memory Manager</h3></div>';
        html += '<div class="widget-body">';

        // Stacked VRAM Bar
        html += '<div class="gpu-mgr-bar" data-kpi="gpu_mgr_bar">';
        html += '<div class="gpu-mgr-segment gpu-mgr-segment-ollama" style="width:' + (ollamaVram / vramTotal * 100).toFixed(1) + '%" title="Ollama: ' + Math.round(ollamaVram) + ' MB"></div>';
        html += '<div class="gpu-mgr-segment gpu-mgr-segment-docling" style="width:' + (doclingVram / vramTotal * 100).toFixed(1) + '%" title="Docling: ' + Math.round(doclingVram) + ' MB"></div>';
        html += '<div class="gpu-mgr-segment gpu-mgr-segment-embed" style="width:' + (embedVram / vramTotal * 100).toFixed(1) + '%" title="Embedding: ' + Math.round(embedVram) + ' MB"></div>';
        html += '<div class="gpu-mgr-segment gpu-mgr-segment-deberta" style="width:' + (debertaVram / vramTotal * 100).toFixed(1) + '%" title="DeBERTa: ' + Math.round(debertaVram) + ' MB"></div>';
        if (otherVram > 0) {
            html += '<div class="gpu-mgr-segment gpu-mgr-segment-other" style="width:' + (otherVram / vramTotal * 100).toFixed(1) + '%" title="Sonstige: ' + Math.round(otherVram) + ' MB"></div>';
        }
        html += '<div class="gpu-mgr-segment gpu-mgr-segment-free" style="width:' + (freeVram / vramTotal * 100).toFixed(1) + '%" title="Frei: ' + Math.round(freeVram) + ' MB"></div>';
        html += '</div>';

        // Legend
        html += '<div class="gpu-mgr-legend">';
        html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-ollama"></span> Ollama: <span data-kpi="gpu_mgr_ollama">' + Math.round(ollamaVram) + '</span> MB</span>';
        html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-docling"></span> Docling: <span data-kpi="gpu_mgr_docling">' + Math.round(doclingVram) + '</span> MB</span>';
        html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-embed"></span> Embedding: <span data-kpi="gpu_mgr_embed">' + Math.round(embedVram) + '</span> MB</span>';
        html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-deberta"></span> DeBERTa: <span data-kpi="gpu_mgr_deberta">' + Math.round(debertaVram) + '</span> MB</span>';
        if (otherVram > 0) {
            html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-other"></span> Sonstige: <span data-kpi="gpu_mgr_other">' + Math.round(otherVram) + '</span> MB</span>';
        }
        if (unknownVramCount > 0) {
            html += '<span class="gpu-mgr-legend-item">Unbekannter VRAM: <span data-kpi="gpu_mgr_unknown">' + Math.round(unknownVramCount) + '</span> Prozess' + (unknownVramCount !== 1 ? 'e' : '') + '</span>';
        }
        if (!gpuProcessesPayload.known) {
            html += '<span class="gpu-mgr-legend-item">Prozessliste: <span data-kpi="gpu_mgr_process_state">' + escapeHtml(gpuProcessesPayload.state) + '</span></span>';
        }
        html += '<span class="gpu-mgr-legend-item"><span class="gpu-mgr-legend-dot gpu-mgr-segment-free"></span> Frei: <span data-kpi="gpu_mgr_free">' + Math.round(freeVram) + '</span> MB</span>';
        html += '</div>';

        // Service Table
        var ollama = metrics.ollama || {};
        var ollamaModelsCount = (ollama.models_loaded || []).length;
        var doclingRunning = docling.running;
        var doclingDevice = docling.device || 'unknown';

        html += '<table class="widget-table gpu-mgr-table">';
        html += '<thead><tr><th>Service</th><th>VRAM</th><th>Details</th><th>Aktionen</th></tr></thead>';
        html += '<tbody>';

        // Ollama row
        html += '<tr><td>Ollama</td>';
        html += '<td>' + Math.round(ollamaVram) + ' MB</td>';
        html += '<td>' + ollamaModelsCount + ' Modell' + (ollamaModelsCount !== 1 ? 'e' : '') + ' geladen</td>';
        html += '<td></td></tr>';

        // Docling row
        var doclingStatusBadge = doclingRunning
            ? '<span class="models-badge models-badge-vram">Running</span>'
            : '<span class="models-badge models-badge-unloaded">Stopped</span>';
        var cudaActive = doclingDevice === 'cuda' ? ' active' : '';
        var cpuActive = doclingDevice === 'cpu' ? ' active' : '';

        html += '<tr><td>Docling</td>';
        html += '<td>' + Math.round(doclingVram) + ' MB</td>';
        html += '<td>' + doclingStatusBadge + ' ';
        html += '<span class="gpu-mgr-device-toggle">';
        html += '<button class="gpu-mgr-device-btn' + cudaActive + '" onclick="toggleDoclingDevice(\'cuda\')" data-kpi="gpu_mgr_docling_cuda">CUDA</button>';
        html += '<button class="gpu-mgr-device-btn' + cpuActive + '" onclick="toggleDoclingDevice(\'cpu\')" data-kpi="gpu_mgr_docling_cpu">CPU</button>';
        html += '</span></td>';
        html += '<td class="models-actions">';
        if (doclingRunning) {
            html += '<button class="action-btn" onclick="stopDocling()" data-kpi="gpu_mgr_docling_action">Stoppen</button>';
        } else {
            html += '<button class="action-btn primary" onclick="startDocling()" data-kpi="gpu_mgr_docling_action">Starten</button>';
        }
        html += '</td></tr>';

        // Embedding row
        var embedding = metrics.embedding || {};
        var embedRunning = embedding.running;
        var embedModel = embedding.model || '';
        var embedLoading = embedding.loading_model || '';
        var embedLoadedModels = Array.isArray(embedding.loaded_models)
            ? embedding.loaded_models.filter(Boolean)
            : [];
        var embedColbertAvailable = Array.isArray(embedding.available_colbert_models)
            ? embedding.available_colbert_models
            : [];
        var embedCurrentIsColbert = embedLoadedModels.some(function(modelName) {
            return embedColbertAvailable.includes(modelName);
        });
        var embedLoadingIsColbert = embedLoading && embedColbertAvailable.includes(embedLoading);
        var embedStatusBadge = embedRunning
            ? (embedLoading
                ? '<span class="models-badge models-badge-mixed">Loading</span>'
                : '<span class="models-badge models-badge-vram">Running</span>')
            : '<span class="models-badge models-badge-unloaded">Stopped</span>';
        var embedKindBadge = (embedCurrentIsColbert || embedLoadingIsColbert)
            ? ' <span class="models-badge models-badge-type-colbert">ColBERT</span>'
            : '';
        var embedSlotInfo = Number.isFinite(Number(embedding.model_slots))
            ? ' <span class="gpu-mgr-model-name">(' + embedLoadedModels.length + '/' + escapeHtml(String(embedding.model_slots)) + ' Slots)</span>'
            : '';
        var embedModelInfo = embedLoading
            ? ' <span class="gpu-mgr-model-name">' + escapeHtml(embedLoading) + '</span>' + embedSlotInfo
            : (embedLoadedModels.length
                ? ' <span class="gpu-mgr-model-name">' + escapeHtml(embedLoadedModels.join(', ')) + '</span>' + embedSlotInfo
                : embedSlotInfo);

        html += '<tr><td>Embedding</td>';
        html += '<td>' + Math.round(embedVram) + ' MB</td>';
        html += '<td>' + embedStatusBadge + embedKindBadge + embedModelInfo + '</td>';
        html += '<td class="models-actions">';
        if (embedRunning) {
            html += '<button class="action-btn" onclick="stopEmbedding()" data-kpi="gpu_mgr_embed_action">Stoppen</button>';
        } else {
            html += '<button class="action-btn primary" onclick="startEmbedding()" data-kpi="gpu_mgr_embed_action">Starten</button>';
        }
        html += '</td></tr>';

        // DeBERTa row
        var deberta = metrics.deberta || {};
        var debertaRunning = deberta.running;
        var debertaModel = deberta.model || '';
        var debertaLoading = deberta.loading_model || '';
        var debertaAvailable = Array.isArray(deberta.available_models) ? deberta.available_models : [];
        var debertaHasModel = !!debertaModel;
        var debertaStatusBadge;
        if (!debertaRunning) {
            debertaStatusBadge = '<span class="models-badge models-badge-unloaded">Stopped</span>';
        } else if (debertaLoading) {
            debertaStatusBadge = '<span class="models-badge models-badge-mixed">Loading</span>';
        } else if (debertaHasModel) {
            debertaStatusBadge = '<span class="models-badge models-badge-vram">Loaded</span>';
        } else {
            debertaStatusBadge = '<span class="models-badge models-badge-unloaded">Idle</span>';
        }
        var debertaModelInfo = debertaLoading
            ? ' <span class="gpu-mgr-model-name">' + escapeHtml(debertaLoading) + '</span>'
            : (debertaHasModel ? ' <span class="gpu-mgr-model-name">' + escapeHtml(debertaModel) + '</span>' : '');

        html += '<tr><td>DeBERTa</td>';
        html += '<td>' + Math.round(debertaVram) + ' MB</td>';
        html += '<td>' + debertaStatusBadge + debertaModelInfo + '</td>';
        html += '<td class="models-actions">';
        if (debertaRunning) {
            if (debertaHasModel) {
                html += '<button class="action-btn" onclick="unloadDeberta()" data-kpi="gpu_mgr_deberta_model_action">Entladen</button>';
            } else if (debertaAvailable.length > 0) {
                var defaultModel = debertaAvailable[0];
                html += '<button class="action-btn primary" onclick="loadDeberta(\'' + escapeHtml(defaultModel) + '\')" data-kpi="gpu_mgr_deberta_model_action">Laden</button>';
            }
            html += '<button class="action-btn" onclick="stopDeberta()" data-kpi="gpu_mgr_deberta_service_action">Stoppen</button>';
        } else {
            html += '<button class="action-btn primary" onclick="startDeberta()" data-kpi="gpu_mgr_deberta_service_action">Starten</button>';
        }
        html += '</td></tr>';

        html += '</tbody></table>';

        html += '</div>'; // .widget-body
        html += '</div>'; // .widget (GPU Manager)
    }

    // ========================================
    // Widget 6: Wartungsmodus (volle Breite)
    // ========================================
    var maintenance = sysGetMaintenanceView(metrics.maintenance);
    var maintActive = maintenance.active;

    html += '<div class="widget" data-widget="sys-maintenance">';
    html += '<div class="widget-header"><h3 class="widget-title">Wartungsmodus</h3></div>';
    html += '<div class="widget-body">';
    html += '<div class="maintenance-container">';

    // Toggle-Zeile
    html += '<div class="maintenance-toggle-row">';
    html += '<div class="maintenance-info">';
    html += '<div class="maintenance-status ' + (maintenance.activeClass ? 'maintenance-active' : '') + '" data-kpi="maintenance_status">';
    html += maintenance.label;
    html += '</div>';
    html += '<div class="maintenance-desc">Blockiert externe Zugriffe auf Docling (5001), Ollama API (11434) und OpenAI API (11440)</div>';
    html += '</div>';
    html += '<label class="maintenance-switch">';
    html += '<input type="checkbox" ' + (maintActive ? 'checked' : '') + (maintenance.transitioning ? ' disabled' : '') + ' onchange="toggleMaintenance()" data-kpi="maintenance_toggle">';
    html += '<span class="maintenance-slider"></span>';
    html += '</label>';
    html += '</div>';

    // Port-Status-Tabelle
    html += '<table class="widget-table maintenance-ports-table">';
    html += '<thead><tr><th>Port</th><th>Service</th><th>Status</th></tr></thead>';
    html += '<tbody>';
    var maintPorts = [
        {port: 5001, desc: 'Docling'},
        {port: 11434, desc: 'Ollama API'},
        {port: 11440, desc: 'OpenAI API'},
    ];
    for (var mi = 0; mi < maintPorts.length; mi++) {
        var portBadge = maintActive
            ? '<span class="models-badge models-badge-unloaded">Blockiert</span>'
            : '<span class="models-badge models-badge-vram">Offen</span>';
        html += '<tr><td>' + maintPorts[mi].port + '</td><td>' + escapeHtml(maintPorts[mi].desc) + '</td><td>' + portBadge + '</td></tr>';
    }
    html += '</tbody></table>';

    html += '</div>'; // .maintenance-container
    html += '</div>'; // .widget-body
    html += '</div>'; // .widget (Wartungsmodus)

    html += '</div>'; // .widget-dashboard
    return html;
}

// ========================================
// Docling Control Functions
// ========================================

async function toggleDoclingDevice(device) {
    var btns = document.querySelectorAll('.gpu-mgr-device-btn');
    btns.forEach(function(b) { b.disabled = true; });

    try {
        var resp = await fetch('/api/docling/device', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({device: device}),
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('Docling Device: ' + device.toUpperCase(), 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btns.forEach(function(b) { b.disabled = false; });
}

async function startDocling() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Starte...';

    try {
        var resp = await fetch('/api/docling/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('Docling gestartet', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Starten';
}

async function stopDocling() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Stoppe...';

    try {
        var resp = await fetch('/api/docling/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('Docling gestoppt', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Stoppen';
}

// ========================================
// Embedding Control Functions
// ========================================

async function startEmbedding() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Starte...';

    try {
        var resp = await fetch('/api/embedding/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('Embedding-Service gestartet', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Starten';
}

async function stopEmbedding() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Stoppe...';

    try {
        var resp = await fetch('/api/embedding/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('Embedding-Service gestoppt', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Stoppen';
}

// ========================================
// DeBERTa Control Functions
// ========================================

async function loadDeberta(model) {
    var btn = event.target;
    var originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Lade...';

    try {
        var resp = await fetch('/api/deberta/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: model}),
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('DeBERTa-Modell geladen: ' + model, 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = originalText;
}

async function unloadDeberta() {
    var btn = event.target;
    var originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Entlade...';

    try {
        var resp = await fetch('/api/deberta/unload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('DeBERTa-Modell entladen', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = originalText;
}

async function startDeberta() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Starte...';

    try {
        var resp = await fetch('/api/deberta/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('DeBERTa-Service gestartet', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Starten';
}

async function stopDeberta() {
    var btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Stoppe...';

    try {
        var resp = await fetch('/api/deberta/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            showNotification('DeBERTa-Service gestoppt', 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
    }

    btn.disabled = false;
    btn.textContent = 'Stoppen';
}

// ========================================
// Wartungsmodus Control Function
// ========================================

async function toggleMaintenance() {
    var toggle = document.querySelector('[data-kpi="maintenance_toggle"]');
    if (toggle) toggle.disabled = true;

    try {
        var resp = await fetch('/api/maintenance/toggle', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        var data = await resp.json();
        if (resp.ok) {
            var msg = data.active ? 'Wartungsmodus aktiviert' : 'Wartungsmodus deaktiviert';
            showNotification(msg, data.active ? 'error' : 'success');
        } else {
            showNotification('Fehler: ' + (data.error || resp.statusText), 'error');
            if (toggle) toggle.checked = !toggle.checked;
        }
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
        if (toggle) toggle.checked = !toggle.checked;
    }

    if (toggle) toggle.disabled = false;
}

// ========================================
// Live-Update via WebSocket
// ========================================

/**
 * Aktualisiert alle System-KPI-Werte im DOM.
 * Wird von app_core.js aufgerufen mit data = { system: {...}, summary: {...} }
 */
/**
 * Live-Update der Ressourcen-KPI-Karten (CPU, RAM, GPU Util, VRAM,
 * Disk I/O). Geteilt zwischen System-Tab und Dashboard-Tab.
 */
function updateSystemResourceKpis(data) {
    var metrics = data.system || {};
    var cpu = metrics.cpu || {};
    var memory = metrics.memory || {};
    var gpu = metrics.gpu || {};
    var diskIo = metrics.disk_io || {};
    var gpuAvailable = !gpu.error;

    // --- CPU ---
    var cpuEl = document.querySelector('[data-kpi="sys_cpu_usage"]');
    if (cpuEl) {
        cpuEl.textContent = sysFormatPercent(cpu.usage_percent);
        var cpuCard = document.querySelector('[data-kpi-card="sys_cpu_usage"]');
        if (cpuCard) {
            cpuCard.className = 'kpi-card ' + sysGetPercentColor(cpu.usage_percent);
        }
    }

    // --- RAM ---
    var ramEl = document.querySelector('[data-kpi="sys_ram_usage"]');
    if (ramEl) {
        ramEl.textContent = sysFormatPercent(memory.usage_percent);
        var ramCard = document.querySelector('[data-kpi-card="sys_ram_usage"]');
        if (ramCard) {
            ramCard.className = 'kpi-card ' + sysGetPercentColor(memory.usage_percent);
        }
    }

    // --- GPU Util ---
    var gpuEl = document.querySelector('[data-kpi="sys_gpu_util"]');
    if (gpuEl) {
        if (gpuAvailable) {
            gpuEl.textContent = sysFormatPercent(gpu.gpu_util_percent);
            var gpuCard = document.querySelector('[data-kpi-card="sys_gpu_util"]');
            if (gpuCard) {
                gpuCard.className = 'kpi-card ' + sysGetPercentColor(gpu.gpu_util_percent);
            }
        } else {
            gpuEl.textContent = 'N/A';
            var gpuCard2 = document.querySelector('[data-kpi-card="sys_gpu_util"]');
            if (gpuCard2) {
                gpuCard2.className = 'kpi-card';
            }
        }
    }

    // --- VRAM ---
    var vramEl = document.querySelector('[data-kpi="sys_vram_usage"]');
    if (vramEl) {
        if (gpuAvailable) {
            vramEl.textContent = sysFormatPercent(gpu.vram_usage_percent);
            var vramCard = document.querySelector('[data-kpi-card="sys_vram_usage"]');
            if (vramCard) {
                vramCard.className = 'kpi-card ' + sysGetPercentColor(gpu.vram_usage_percent);
            }
        } else {
            vramEl.textContent = 'N/A';
            var vramCard2 = document.querySelector('[data-kpi-card="sys_vram_usage"]');
            if (vramCard2) {
                vramCard2.className = 'kpi-card';
            }
        }
    }

    // --- Disk I/O ---
    var diskReadEl = document.querySelector('[data-kpi="sys_disk_read"]');
    if (diskReadEl) {
        diskReadEl.textContent = sysFormatMBs(diskIo.read_mb_s);
    }

    var diskWriteEl = document.querySelector('[data-kpi="sys_disk_write"]');
    if (diskWriteEl) {
        diskWriteEl.textContent = sysFormatMBs(diskIo.write_mb_s);
    }
}

function updateSystemFromWS(data) {
    var metrics = data.system || {};
    var gpu = metrics.gpu || {};
    var gpuAvailable = !gpu.error;

    // --- GPU Memory Manager ---
    var gpuMgrBar = document.querySelector('[data-kpi="gpu_mgr_bar"]');
    if (gpuMgrBar && gpuAvailable) {
        var gpuProcessesPayload = sysGetGpuProcessesPayload(metrics);
        var gpuProcesses = gpuProcessesPayload.data;
        var docling = metrics.docling || {};
        var vramTotal = gpu.vram_total_mb || 1;
        var gpuAgg = {};
        var unknownVramCount = sysIsKnownNumber(metrics.gpu_process_vram_unknown_count)
            ? metrics.gpu_process_vram_unknown_count
            : 0;
        for (var gi = 0; gi < gpuProcesses.length; gi++) {
            var gp = gpuProcesses[gi];
            if (!sysIsKnownNumber(gp.vram_mb)) {
                if (!sysIsKnownNumber(metrics.gpu_process_vram_unknown_count)) {
                    unknownVramCount += 1;
                }
                continue;
            }
            var gl = gp.label || 'other';
            gpuAgg[gl] = (gpuAgg[gl] || 0) + gp.vram_mb;
        }
        var ollamaV = gpuAgg['ollama'] || 0;
        var doclingV = gpuAgg['docling'] || 0;
        var embedV = gpuAgg['embedding'] || 0;
        var debertaV = gpuAgg['deberta'] || 0;
        var otherV = gpuAgg['other'] || 0;
        if (!gpuProcessesPayload.known && sysIsKnownNumber(gpu.vram_used_mb)) {
            var attributedV = ollamaV + doclingV + embedV + debertaV + otherV;
            otherV += Math.max(0, gpu.vram_used_mb - attributedV);
        }
        var usedV = ollamaV + doclingV + embedV + debertaV + otherV;
        var freeV = Math.max(0, vramTotal - usedV);

        // Update bar segments
        // Reihenfolge im DOM: ollama, docling, embed, deberta, [other?], free.
        // "other" ist optional und nur vorhanden, wenn beim letzten Render >0 war.
        var segments = gpuMgrBar.children;
        var idx = 0;
        if (segments[idx]) { segments[idx].style.width = (ollamaV / vramTotal * 100).toFixed(1) + '%'; segments[idx].title = 'Ollama: ' + Math.round(ollamaV) + ' MB'; idx++; }
        if (segments[idx]) { segments[idx].style.width = (doclingV / vramTotal * 100).toFixed(1) + '%'; segments[idx].title = 'Docling: ' + Math.round(doclingV) + ' MB'; idx++; }
        if (segments[idx]) { segments[idx].style.width = (embedV / vramTotal * 100).toFixed(1) + '%'; segments[idx].title = 'Embedding: ' + Math.round(embedV) + ' MB'; idx++; }
        if (segments[idx]) { segments[idx].style.width = (debertaV / vramTotal * 100).toFixed(1) + '%'; segments[idx].title = 'DeBERTa: ' + Math.round(debertaV) + ' MB'; idx++; }
        // Handle optional "other" segment
        if (segments.length > 5 && segments[idx]) { segments[idx].style.width = (otherV / vramTotal * 100).toFixed(1) + '%'; idx++; }
        if (segments[idx]) { segments[idx].style.width = (freeV / vramTotal * 100).toFixed(1) + '%'; }

        // Update legend values
        var elO = document.querySelector('[data-kpi="gpu_mgr_ollama"]');
        if (elO) elO.textContent = Math.round(ollamaV);
        var elD = document.querySelector('[data-kpi="gpu_mgr_docling"]');
        if (elD) elD.textContent = Math.round(doclingV);
        var elE = document.querySelector('[data-kpi="gpu_mgr_embed"]');
        if (elE) elE.textContent = Math.round(embedV);
        var elDb = document.querySelector('[data-kpi="gpu_mgr_deberta"]');
        if (elDb) elDb.textContent = Math.round(debertaV);
        var elOt = document.querySelector('[data-kpi="gpu_mgr_other"]');
        if (elOt) elOt.textContent = Math.round(otherV);
        var elUnk = document.querySelector('[data-kpi="gpu_mgr_unknown"]');
        if (elUnk) elUnk.textContent = Math.round(unknownVramCount);
        var elProcState = document.querySelector('[data-kpi="gpu_mgr_process_state"]');
        if (elProcState) elProcState.textContent = gpuProcessesPayload.state;
        var elF = document.querySelector('[data-kpi="gpu_mgr_free"]');
        if (elF) elF.textContent = Math.round(freeV);

        // Update docling status
        var cudaBtn = document.querySelector('[data-kpi="gpu_mgr_docling_cuda"]');
        var cpuBtn = document.querySelector('[data-kpi="gpu_mgr_docling_cpu"]');
        if (cudaBtn && cpuBtn) {
            cudaBtn.className = 'gpu-mgr-device-btn' + (docling.device === 'cuda' ? ' active' : '');
            cpuBtn.className = 'gpu-mgr-device-btn' + (docling.device === 'cpu' ? ' active' : '');
        }
    }

    // --- Wartungsmodus ---
    var maintenance = sysGetMaintenanceView(metrics.maintenance);
    var maintActive = maintenance.active;
    var maintStatusEl = document.querySelector('[data-kpi="maintenance_status"]');
    if (maintStatusEl) {
        maintStatusEl.textContent = maintenance.label;
        maintStatusEl.className = 'maintenance-status' + (maintenance.activeClass ? ' maintenance-active' : '');
    }
    var maintToggle = document.querySelector('[data-kpi="maintenance_toggle"]');
    if (maintToggle) {
        maintToggle.checked = maintActive;
        maintToggle.disabled = maintenance.transitioning;
    }
    var portRows = document.querySelectorAll('.maintenance-ports-table tbody tr');
    portRows.forEach(function(row) {
        var statusCell = row.querySelector('td:last-child');
        if (statusCell) {
            statusCell.innerHTML = maintActive
                ? '<span class="models-badge models-badge-unloaded">Blockiert</span>'
                : '<span class="models-badge models-badge-vram">Offen</span>';
        }
    });
}

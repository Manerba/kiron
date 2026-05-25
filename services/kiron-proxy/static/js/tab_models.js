/**
 * Ollama Monitor - Models Tab
 * Verwalten von Ollama-Sprachmodellen: Anzeigen, Laden, Entladen, Pullen, Loeschen.
 */

// ========================================
// State
// ========================================

let _modelsData = [];
let _pullInProgress = false;
let _expandedRegistryModel = null;
let _benchmarksData = {};

// ========================================
// Init
// ========================================

// Sub-Tab "Installiert": lokal installierte Modelle. Wird vom Sub-Tab-System
// (app_core.js SUBTABS) aufgerufen, nicht direkt.
async function initModelsInstalled() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    // Installiert nutzt keine Filter-Bar
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.style.display = 'none';
        filterBar.innerHTML = '';
    }

    container.innerHTML = `
        <div class="models-page">
            <div id="modelsTableContainer">
                <div class="table-loading"><div class="spinner"></div></div>
            </div>
        </div>
    `;

    loadLocalModels();

    // Pruefen ob ein Download aktiv ist (ueberlebt Page-Reload)
    checkActivePulls();
}

// Sub-Tab "Verfuegbar": Registry + manueller Pull.
async function initModelsAvailable() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    // Download-Funktion in die Filter-Bar
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        filterBar.innerHTML = `
            <div class="models-pull-manual">
                <input type="text" id="modelsPullInput" class="filter-input models-pull-input"
                       placeholder="Modellname, z.B. mistral:7b" />
                <button class="action-btn primary" id="modelsPullBtn" onclick="startPullModel()">
                    Herunterladen
                </button>
                <a href="https://ollama.com/search" target="_blank" class="models-search-link">
                    Modelle auf ollama.com suchen
                </a>
            </div>
            <div id="modelsPullProgress" class="models-pull-progress" style="display:none;">
                <div class="models-progress-bar-container">
                    <div class="models-progress-bar" id="modelsPullBar"></div>
                </div>
                <div class="models-progress-text" id="modelsPullText">Starte...</div>
            </div>
        `;
        filterBar.style.display = 'flex';
    }

    container.innerHTML = `
        <div class="models-page">
            <div id="registryTableContainer" class="models-registry-section">
                <div class="table-loading"><div class="spinner"></div></div>
            </div>
        </div>
    `;

    // Benchmarks (Registry-Spalten) + lokale Modelle (Installiert-Check) vor dem
    // Registry-Render bereitstellen.
    await Promise.all([loadBenchmarks(), fetchLocalModels()]);
    loadFeaturedModels();

    // Pruefen ob ein Download aktiv ist (ueberlebt Page-Reload)
    checkActivePulls();
}

// ========================================
// Benchmarks laden
// ========================================

async function loadBenchmarks() {
    try {
        const resp = await fetch('/api/benchmarks');
        const data = await resp.json();
        _benchmarksData = data.models || {};
    } catch {
        _benchmarksData = {};
    }
}

function getBenchmark(modelName, field) {
    const base = modelName.split(':')[0];
    const entry = _benchmarksData[base];
    if (!entry) return null;
    const val = entry[field];
    if (val === null || val === undefined) return null;
    return val;
}

function formatBenchmark(val) {
    if (val === null || val === undefined) return '<span class="text-muted">-</span>';
    return typeof val === 'number' ? val.toFixed(1) : escapeHtml(String(val));
}

function isKnownNumber(val) {
    return typeof val === 'number' && Number.isFinite(val);
}

function formatGbValue(val) {
    return isKnownNumber(val) ? val + ' GB' : '<span class="text-muted">Unbekannt</span>';
}

// ========================================
// Lokale Modelle laden und rendern
// ========================================

// Holt lokale Modelle in _modelsData (container-unabhaengig, auch fuer den
// Verfuegbar-Sub-Tab). Gibt bei Fehler einen Meldungstext zurueck, sonst null.
async function fetchLocalModels() {
    try {
        const resp = await fetch('/api/models/local');
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            return `Fehler: ${escapeHtml(err.error || resp.statusText)}`;
        }
        const data = await resp.json();
        _modelsData = data.models || [];
        return null;
    } catch (e) {
        return 'Ollama nicht erreichbar';
    }
}

async function loadLocalModels() {
    const tableContainer = document.getElementById('modelsTableContainer');
    if (!tableContainer) return;

    const error = await fetchLocalModels();
    if (error) {
        tableContainer.innerHTML = `<div class="no-data">${error}</div>`;
        return;
    }
    renderModelsTable();
}

function renderModelsTable() {
    const tableContainer = document.getElementById('modelsTableContainer');
    if (!tableContainer) return;

    if (_modelsData.length === 0) {
        tableContainer.innerHTML = `
            <div class="no-data">
                Keine Modelle installiert.<br>
                <span class="text-muted">Verwende den Download-Bereich oben, um ein Modell herunterzuladen.</span>
            </div>
        `;
        return;
    }

    // Sortierung: Aktive/Geladene zuerst, laufende Embedding-Loads danach
    const sorted = [..._modelsData].sort((a, b) => {
        const aActive = a.load_state === 'loaded' || a.embedding_active || a.embedding_loading;
        const bActive = b.load_state === 'loaded' || b.embedding_active || b.embedding_loading;
        if (aActive && !bActive) return -1;
        if (!aActive && bActive) return 1;
        if (a.embedding_loading && !b.embedding_loading) return -1;
        if (!a.embedding_loading && b.embedding_loading) return 1;
        return a.name.localeCompare(b.name);
    });

    let html = `<table class="models-table">
        <thead>
            <tr>
                <th>Modell</th>
                <th>Typ</th>
                <th>Familie</th>
                <th>Parameter</th>
                <th>Quant.</th>
                <th>Groesse</th>
                <th>Status</th>
                <th>Speicher (VRAM / RAM)</th>
                <th>Aktionen</th>
            </tr>
        </thead>
        <tbody>`;

    for (const m of sorted) {
        const statusBadge = getStatusBadge(m);
        const memBar = getMemoryBar(m);
        const typeBadge = getTypeBadge(m.model_type);
        const expiresInfo = m.expires_at ? `<div class="models-expires">Entladen: ${formatExpires(m.expires_at)}</div>` : '';
        const ctxInfo = m.context_length ? `<div class="models-ctx">Ctx: ${m.context_length.toLocaleString()}</div>` : '';

        // Ollama Laden/Entladen Buttons
        let ollamaActions = '';
        if (m.loaded) {
            ollamaActions = `<button class="action-btn" onclick="unloadModel('${escapeHtml(m.name)}')" title="Entladen">Entladen</button>`;
        } else {
            ollamaActions = `<div class="models-load-group">
                <button class="action-btn primary" onclick="loadModel('${escapeHtml(m.name)}', true)" title="Laden (GPU)">Laden</button>
                <button class="action-btn" onclick="loadModel('${escapeHtml(m.name)}', false)" title="Ohne GPU laden">CPU</button>
            </div>`;
        }

        // Embedding-Service Button (nur bei embedding-faehigen Modellen)
        let embedAction = '';
        if (m.embedding_capable && m.embedding_loading) {
            embedAction = `<button class="action-btn models-btn-embed" disabled title="Embedding-Load laeuft">Lade...</button>`;
        } else if (m.embedding_capable && !m.embedding_active) {
            embedAction = `<button class="action-btn models-btn-embed" onclick="loadEmbeddingModel('${escapeHtml(m.name)}')" title="Im Embedding-Service laden">Embedding</button>`;
        }

        html += `<tr data-model="${escapeHtml(m.name)}">
            <td class="font-bold">${escapeHtml(m.name)}</td>
            <td>${typeBadge}</td>
            <td>${escapeHtml(m.family)}</td>
            <td>${escapeHtml(m.parameter_size)}</td>
            <td>${escapeHtml(m.quantization_level)}</td>
            <td>${formatGbValue(m.size_gb)}</td>
            <td>${statusBadge}${ctxInfo}${expiresInfo}</td>
            <td>${memBar}</td>
            <td class="models-actions">
                ${embedAction}
                ${ollamaActions}
                <button class="action-btn danger" onclick="deleteModel('${escapeHtml(m.name)}', ${m.loaded})" title="Loeschen">Loeschen</button>
            </td>
        </tr>`;
    }

    html += '</tbody></table>';
    tableContainer.innerHTML = html;
}

function getStatusBadge(model) {
    if (model.load_state === 'unknown') {
        return '<span class="models-badge models-badge-unloaded">Load-Status unbekannt</span>';
    }
    // Embedding-Service aktiv: blauer Badge
    if (model.embedding_loading) {
        return '<span class="models-badge models-badge-mixed">Embedding laedt</span>';
    }
    if (model.embedding_active) {
        if (model.loaded) {
            // Sowohl in Ollama als auch im Embedding-Service geladen
            return '<span class="models-badge models-badge-embed-active">Aktiv (Embedding + Ollama)</span>';
        }
        return '<span class="models-badge models-badge-embed-active">Aktiv (Embedding)</span>';
    }
    if (!model.loaded) {
        return '<span class="models-badge models-badge-unloaded">Nicht geladen</span>';
    }
    if (model.ram_gb <= 0.01) {
        return '<span class="models-badge models-badge-vram">Geladen (VRAM)</span>';
    }
    if (model.vram_gb <= 0.01) {
        return '<span class="models-badge models-badge-ram">Geladen (RAM)</span>';
    }
    return '<span class="models-badge models-badge-mixed">Geladen (VRAM+RAM)</span>';
}

function getTypeBadge(modelType) {
    switch (modelType) {
        case 'embedding':
            return '<span class="models-badge models-badge-type-embed">Embedding</span>';
        case 'vlm':
            return '<span class="models-badge models-badge-type-vlm">VLM</span>';
        default:
            return '<span class="models-badge models-badge-type-llm">LLM</span>';
    }
}

function getMemoryBar(model) {
    if (model.load_state === 'unknown') {
        return '<span class="text-muted">Unbekannt</span>';
    }
    if (!model.loaded) {
        return '<span class="text-muted">-</span>';
    }

    if (!isKnownNumber(model.vram_gb) || !isKnownNumber(model.ram_gb)) {
        return '<span class="text-muted">Unbekannt</span>';
    }

    const total = model.vram_gb + model.ram_gb;
    if (total <= 0) return '<span class="text-muted">-</span>';

    const vramPct = (model.vram_gb / total * 100).toFixed(1);
    const ramPct = (model.ram_gb / total * 100).toFixed(1);

    return `
        <div class="models-mem-bar-wrapper">
            <div class="models-mem-bar">
                <div class="models-mem-vram" style="width:${vramPct}%" title="VRAM: ${model.vram_gb} GB"></div>
                <div class="models-mem-ram" style="width:${ramPct}%" title="RAM: ${model.ram_gb} GB"></div>
            </div>
            <div class="models-mem-labels">
                <span class="models-mem-label-vram">${model.vram_gb} GB VRAM</span>
                ${model.ram_gb > 0.01 ? `<span class="models-mem-label-ram">${model.ram_gb} GB RAM</span>` : ''}
            </div>
        </div>
    `;
}

function formatExpires(isoStr) {
    try {
        const d = new Date(isoStr);
        const now = new Date();
        const diffMs = d - now;
        if (diffMs <= 0) return 'bald';
        const mins = Math.round(diffMs / 60000);
        if (mins < 60) return `in ${mins} Min`;
        const hours = Math.round(mins / 60);
        return `in ${hours} Std`;
    } catch {
        return isoStr;
    }
}

// ========================================
// Featured-Modelle: Tabelle mit aufklappbaren Tags
// ========================================

async function loadFeaturedModels() {
    const container = document.getElementById('registryTableContainer');
    if (!container) return;

    try {
        const resp = await fetch('/api/models/available');
        const data = await resp.json();
        const models = data.models || [];

        if (models.length === 0) {
            container.innerHTML = data.error
                ? `<div class="no-data text-muted">${escapeHtml(data.error)}</div>`
                : '';
            return;
        }

        // Sortieren: nicht-installierte zuerst, dann source "ollama" vor "other", dann alphabetisch
        models.sort((a, b) => {
            if (a.installed && !b.installed) return 1;
            if (!a.installed && b.installed) return -1;
            if (a.source === 'ollama' && b.source !== 'ollama') return -1;
            if (a.source !== 'ollama' && b.source === 'ollama') return 1;
            return a.name.localeCompare(b.name);
        });

        const hasBench = Object.keys(_benchmarksData).length > 0;

        let html = `<table class="models-table models-registry-table">
            <thead>
                <tr>
                    <th></th>
                    <th>Modell</th>
                    <th>Quelle</th>
                    <th>Groesse</th>
                    <th>Status</th>
                    ${hasBench ? `
                    <th class="models-bench-col" title="MMLU-PRO (Open LLM Leaderboard)">MMLU-PRO</th>
                    <th class="models-bench-col" title="HumanEval (EvalPlus)">HumanEval</th>
                    <th class="models-bench-col" title="GPQA (Open LLM Leaderboard)">GPQA</th>
                    <th class="models-bench-col" title="BBH (Open LLM Leaderboard)">BBH</th>
                    <th class="models-bench-col" title="IFEval (Open LLM Leaderboard)">IFEval</th>
                    ` : ''}
                    <th>Aktion</th>
                </tr>
            </thead>
            <tbody>`;

        const colCount = hasBench ? 11 : 6;

        for (const m of models) {
            const baseName = m.name.split(':')[0];
            const sizeStr = m.size_gb ? `${m.size_gb} GB` : '-';
            const installedBadge = m.installed
                ? '<span class="models-badge models-badge-vram">Installiert</span>'
                : '<span class="models-badge models-badge-unloaded">Nicht installiert</span>';
            const sourceBadge = m.source === 'ollama'
                ? '<span class="models-badge models-badge-ollama">ollama</span>'
                : '<span class="models-badge models-badge-other">other</span>';
            const isOllama = m.source === 'ollama';

            html += `<tr class="models-registry-row" data-registry-model="${escapeHtml(m.name)}">
                <td class="models-expand-cell">
                    ${isOllama ? `<button class="models-expand-btn" onclick="toggleRegistryTags('${escapeHtml(baseName)}', this)"
                            title="Varianten anzeigen">&#9654;</button>` : ''}
                </td>
                <td>
                    <span class="font-bold">${escapeHtml(m.name)}</span>
                </td>
                <td>${sourceBadge}</td>
                <td>${sizeStr}</td>
                <td>${installedBadge}</td>
                ${hasBench ? `
                <td class="models-bench-col">${formatBenchmark(getBenchmark(m.name, 'mmlu_pro'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(m.name, 'humaneval'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(m.name, 'gpqa'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(m.name, 'bbh'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(m.name, 'ifeval'))}</td>
                ` : ''}
                <td>
                    ${isOllama && !m.installed
                        ? `<button class="action-btn primary" onclick="pullFromRegistry('${escapeHtml(m.name)}')">Pull</button>`
                        : ''
                    }
                </td>
            </tr>
            ${isOllama ? `
            <tr class="models-tags-row" id="tags-row-${escapeHtml(baseName)}" style="display:none;">
                <td colspan="${colCount}">
                    <div class="models-tags-container" id="tags-container-${escapeHtml(baseName)}">
                        <div class="table-loading"><div class="spinner"></div></div>
                    </div>
                </td>
            </tr>` : ''}`;
        }

        html += '</tbody></table>';
        container.innerHTML = html;

    } catch {
        container.innerHTML = '<div class="no-data text-muted">Registry nicht erreichbar</div>';
    }
}

async function toggleRegistryTags(baseName, btn) {
    const tagsRow = document.getElementById('tags-row-' + baseName);
    if (!tagsRow) return;

    if (tagsRow.style.display !== 'none') {
        // Zuklappen
        tagsRow.style.display = 'none';
        btn.innerHTML = '&#9654;';
        btn.classList.remove('expanded');
        _expandedRegistryModel = null;
        return;
    }

    // Aufklappen
    tagsRow.style.display = '';
    btn.innerHTML = '&#9660;';
    btn.classList.add('expanded');
    _expandedRegistryModel = baseName;

    const container = document.getElementById('tags-container-' + baseName);
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';

    try {
        const resp = await fetch(`/api/models/registry/${encodeURIComponent(baseName)}/tags`);
        const data = await resp.json();

        if (data.error) {
            container.innerHTML = `<div class="text-muted" style="padding:8px;">Fehler: ${escapeHtml(data.error)}</div>`;
            return;
        }

        const tags = data.tags || [];

        if (tags.length === 0) {
            container.innerHTML = '<div class="text-muted" style="padding:8px;">Keine Varianten gefunden</div>';
            return;
        }

        // Lokale Modelle fuer installed-Check
        const localNames = new Set(_modelsData.map(m => m.name));

        const hasBench = Object.keys(_benchmarksData).length > 0;

        let html = `<table class="models-tags-table">
            <thead>
                <tr>
                    <th>Tag</th>
                    <th>Groesse</th>
                    <th>Context</th>
                    <th>Input</th>
                    ${hasBench ? `
                    <th class="models-bench-col">MMLU-PRO</th>
                    <th class="models-bench-col">HumanEval</th>
                    <th class="models-bench-col">GPQA</th>
                    <th class="models-bench-col">BBH</th>
                    <th class="models-bench-col">IFEval</th>
                    ` : ''}
                    <th>Aktion</th>
                </tr>
            </thead>
            <tbody>`;

        for (const t of tags) {
            const isInstalled = localNames.has(t.tag);
            const sizeStr = t.size || '-';
            const ctxStr = t.context || '-';
            const inputStr = t.input_type || 'Text';

            html += `<tr>
                <td class="font-bold">${escapeHtml(t.tag)}</td>
                <td>${escapeHtml(sizeStr)}</td>
                <td>${escapeHtml(ctxStr)}</td>
                <td>${escapeHtml(inputStr)}</td>
                ${hasBench ? `
                <td class="models-bench-col">${formatBenchmark(getBenchmark(t.tag, 'mmlu_pro'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(t.tag, 'humaneval'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(t.tag, 'gpqa'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(t.tag, 'bbh'))}</td>
                <td class="models-bench-col">${formatBenchmark(getBenchmark(t.tag, 'ifeval'))}</td>
                ` : ''}
                <td>
                    <button class="action-btn primary" onclick="pullFromRegistry('${escapeHtml(t.tag)}')"
                            ${isInstalled ? 'disabled title="Bereits installiert"' : ''}>
                        ${isInstalled ? 'Installiert' : 'Pull'}
                    </button>
                </td>
            </tr>`;
        }

        html += '</tbody></table>';
        container.innerHTML = html;

    } catch (e) {
        container.innerHTML = '<div class="text-muted" style="padding:8px;">Fehler beim Laden der Varianten</div>';
    }
}

function pullFromRegistry(tagName) {
    const input = document.getElementById('modelsPullInput');
    if (input) {
        input.value = tagName;
    }
    startPullModel();
}

// ========================================
// Modell-Aktionen
// ========================================

async function loadModel(name, gpu = true) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Lade...';

    try {
        const resp = await fetch('/api/models/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, gpu}),
        });
        const data = await resp.json();

        if (resp.ok) {
            const label = gpu ? 'geladen' : 'geladen (CPU)';
            showNotification(`${name} ${label}`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || resp.statusText}`, 'error');
        }
    } catch (e) {
        showNotification('Fehler beim Laden: ' + e.message, 'error');
    }

    await loadLocalModels();
}

async function unloadModel(name) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Entlade...';

    try {
        const resp = await fetch('/api/models/unload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name}),
        });
        const data = await resp.json();

        if (resp.ok) {
            showNotification(`${name} entladen`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || resp.statusText}`, 'error');
        }
    } catch (e) {
        showNotification('Fehler beim Entladen: ' + e.message, 'error');
    }

    await loadLocalModels();
}

async function deleteModel(name, isLoaded) {
    if (isLoaded) {
        if (!confirm(`"${name}" ist noch geladen. Trotzdem loeschen?`)) return;
        await doDeleteModel(name, true);
    } else {
        if (!confirm(`"${name}" wirklich loeschen?`)) return;
        await doDeleteModel(name, false);
    }
}

async function doDeleteModel(name, force) {
    try {
        const resp = await fetch('/api/models/delete', {
            method: 'DELETE',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, force}),
        });
        const data = await resp.json();

        if (resp.ok) {
            showNotification(`${name} geloescht`, 'success');
        } else if (resp.status === 409) {
            if (confirm(`${data.error}. Trotzdem loeschen?`)) {
                await doDeleteModel(name, true);
                return;
            }
        } else {
            showNotification(`Fehler: ${data.error || resp.statusText}`, 'error');
        }
    } catch (e) {
        showNotification('Fehler beim Loeschen: ' + e.message, 'error');
    }

    await loadLocalModels();
}

// ========================================
// Pull / Download
// ========================================

async function startPullModel() {
    if (_pullInProgress) return;

    const input = document.getElementById('modelsPullInput');
    const modelName = (input ? input.value.trim() : '');

    if (!modelName) {
        showNotification('Bitte Modellname eingeben', 'error');
        return;
    }

    _pullInProgress = true;
    const pullBtn = document.getElementById('modelsPullBtn');
    if (pullBtn) {
        pullBtn.disabled = true;
        pullBtn.textContent = 'Download laeuft...';
    }

    const progressDiv = document.getElementById('modelsPullProgress');
    const progressBar = document.getElementById('modelsPullBar');
    const progressText = document.getElementById('modelsPullText');
    if (progressDiv) progressDiv.style.display = 'block';
    if (progressBar) progressBar.style.width = '0%';
    if (progressText) progressText.textContent = 'Starte...';

    try {
        const response = await fetch('/api/models/pull', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: modelName}),
        });

        if (!response.ok) {
            showNotification('Pull fehlgeschlagen: ' + response.statusText, 'error');
            resetPullUI();
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const data = JSON.parse(line);
                        if (data.error) {
                            let msg = data.error;
                            if (msg.includes('412') || msg.includes('newer version')) {
                                msg = 'Ollama-Update erforderlich! Dieses Modell braucht eine neuere Ollama-Version.';
                            }
                            showNotification('Pull-Fehler: ' + msg, 'error');
                            resetPullUI();
                            return;
                        }
                        updatePullProgress(data, progressBar, progressText);
                    } catch (e) {
                        console.error('NDJSON Parse-Fehler:', e, line);
                    }
                }
            }
        }

        showNotification(`${modelName} erfolgreich heruntergeladen`, 'success');
        if (input) input.value = '';
        await Promise.all([loadLocalModels(), loadFeaturedModels()]);
    } catch (e) {
        showNotification('Download-Fehler: ' + e.message, 'error');
    }

    resetPullUI();
    const pDiv = document.getElementById('modelsPullProgress');
    if (pDiv) setTimeout(() => { pDiv.style.display = 'none'; }, 3000);
}

function updatePullProgress(data, bar, text) {
    if (!bar || !text) return;

    const status = data.status || '';

    if (data.total && data.completed !== undefined) {
        const pct = Math.round((data.completed / data.total) * 100);
        bar.style.width = pct + '%';
        const completedMB = (data.completed / (1024 * 1024)).toFixed(0);
        const totalMB = (data.total / (1024 * 1024)).toFixed(0);
        text.textContent = `${status} — ${completedMB} / ${totalMB} MB (${pct}%)`;
    } else if (status === 'success') {
        bar.style.width = '100%';
        text.textContent = 'Fertig!';
    } else {
        text.textContent = status;
    }
}

function resetPullUI() {
    _pullInProgress = false;
    const pullBtn = document.getElementById('modelsPullBtn');
    if (pullBtn) {
        pullBtn.disabled = false;
        pullBtn.textContent = 'Herunterladen';
    }
}

async function checkActivePulls() {
    try {
        const resp = await fetch('/api/models/pull/status');
        const data = await resp.json();
        const active = data.active_pulls || {};
        const names = Object.keys(active);

        if (names.length === 0) return;

        // Ersten aktiven Pull anzeigen
        const pull = active[names[0]];
        if (pull.status === 'success' || pull.status === 'error') return;

        _pullInProgress = true;
        const pullBtn = document.getElementById('modelsPullBtn');
        if (pullBtn) {
            pullBtn.disabled = true;
            pullBtn.textContent = 'Download laeuft...';
        }
        const input = document.getElementById('modelsPullInput');
        if (input) input.value = pull.model;

        const progressDiv = document.getElementById('modelsPullProgress');
        if (progressDiv) progressDiv.style.display = 'block';

        // Die Pull-UI gibt es nur im Verfuegbar-Sub-Tab. Sind wir woanders,
        // dorthin wechseln — switchSubTab re-initialisiert Verfuegbar und ruft
        // checkActivePulls erneut (dann existiert die UI und das Polling startet).
        if (!document.getElementById('modelsPullBtn')) {
            if (typeof switchSubTab === 'function') switchSubTab('available');
            return;
        }

        // Polling starten
        pollPullStatus(pull.model);
    } catch {
        // Kein aktiver Pull oder Fehler — ignorieren
    }
}

async function pollPullStatus(modelName) {
    const progressBar = document.getElementById('modelsPullBar');
    const progressText = document.getElementById('modelsPullText');

    while (true) {
        try {
            const resp = await fetch('/api/models/pull/status');
            const data = await resp.json();
            const pull = (data.active_pulls || {})[modelName];

            if (!pull) {
                // Pull nicht mehr aktiv (aufgeraeumt)
                showNotification(`${modelName} Download abgeschlossen`, 'success');
                break;
            }

            updatePullProgress(pull, progressBar, progressText);

            if (pull.status === 'success') {
                showNotification(`${modelName} erfolgreich heruntergeladen`, 'success');
                await Promise.all([loadLocalModels(), loadFeaturedModels()]);
                break;
            }
            if (pull.status === 'error') {
                let msg = pull.error || 'Unbekannter Fehler';
                if (msg.includes('412') || msg.includes('newer version')) {
                    msg = 'Ollama-Update erforderlich! Dieses Modell braucht eine neuere Ollama-Version.';
                }
                showNotification('Pull-Fehler: ' + msg, 'error');
                break;
            }
        } catch {
            // Netzwerkfehler — weiter versuchen
        }

        await new Promise(r => setTimeout(r, 500));
    }

    resetPullUI();
    const progressDiv = document.getElementById('modelsPullProgress');
    if (progressDiv) {
        // Progress-Anzeige nach 3s ausblenden
        setTimeout(() => { progressDiv.style.display = 'none'; }, 3000);
    }
}

// ========================================
// Embedding-Service Modell laden
// ========================================

async function loadEmbeddingModel(name) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Lade...';

    // Basename extrahieren (ohne :tag)
    const basename = name.split(':')[0].split('/').pop();

    try {
        const resp = await fetch('/api/embedding/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: basename}),
        });
        const data = await resp.json();

        if (resp.ok) {
            showNotification(`Embedding-Modell ${basename} geladen`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || resp.statusText}`, 'error');
        }
    } catch (e) {
        showNotification('Fehler beim Laden: ' + e.message, 'error');
    }

    await loadLocalModels();
}

// ========================================
// WebSocket Live-Update
// ========================================

function updateModelsFromWS(data) {
    let needsRerender = false;

    // Ollama-Modelle aktualisieren
    const ollamaData = data?.system?.ollama;
    if (ollamaData && ollamaData.models_loaded) {
        const loadedModels = ollamaData.models_loaded;
        const loadedStateKnown = ollamaData.models_loaded_state !== 'unknown';
        const loadedMap = {};
        for (const lm of loadedModels) {
            loadedMap[lm.name] = lm;
        }

        for (const m of _modelsData) {
            const wasLoaded = m.loaded;
            const wasLoadState = m.load_state;
            const lm = loadedMap[m.name];

            if (!loadedStateKnown) {
                m.loaded = false;
                m.load_state = 'unknown';
                m.vram_gb = null;
                m.ram_gb = null;
                m.context_length = null;
                m.expires_at = null;
            } else if (lm) {
                m.loaded = true;
                m.load_state = 'loaded';
                m.vram_gb = isKnownNumber(lm.vram_gb) ? lm.vram_gb : null;
                m.ram_gb = (isKnownNumber(lm.size_gb) && isKnownNumber(lm.vram_gb))
                    ? Math.max(0, lm.size_gb - lm.vram_gb)
                    : null;
            } else {
                m.loaded = false;
                m.load_state = 'unloaded';
                m.vram_gb = 0;
                m.ram_gb = 0;
                m.context_length = null;
                m.expires_at = null;
            }

            if (wasLoaded !== m.loaded || wasLoadState !== m.load_state) {
                needsRerender = true;
            }
        }
    }

    // Embedding-Status aktualisieren
    const embedData = data?.system?.embedding;
    if (embedData) {
        const embedModel = embedData.model || null;
        const embedLoading = embedData.loading_model || null;
        const embedRunning = embedData.running || false;

        for (const m of _modelsData) {
            const basename = m.name.split(':')[0].split('/').pop();
            const wasActive = m.embedding_active;
            const wasLoading = m.embedding_loading;
            m.embedding_active = embedRunning && basename === embedModel;
            m.embedding_loading = embedRunning && basename === embedLoading;
            if (wasActive !== m.embedding_active || wasLoading !== m.embedding_loading) {
                needsRerender = true;
            }
        }
    }

    if (needsRerender) {
        renderModelsTable();
        return;
    }

    // Nur Zellen aktualisieren wenn kein Rerender noetig
    for (const m of _modelsData) {
        const row = document.querySelector(`tr[data-model="${CSS.escape(m.name)}"]`);
        if (row) {
            const statusCell = row.cells[6];
            const memCell = row.cells[7];
            if (statusCell) statusCell.innerHTML = getStatusBadge(m);
            if (memCell) memCell.innerHTML = getMemoryBar(m);
        }
    }
}

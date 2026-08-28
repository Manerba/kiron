/**
 * Konfiguration-Tab.
 */

let _configData = null;
let _configSaving = false;

async function initConfigTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    await loadRuntimeConfig();
}

async function loadRuntimeConfig() {
    try {
        const resp = await fetch('/api/config/runtime');
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || resp.statusText);
        }
        _configData = data;
        renderConfigTab();
    } catch (e) {
        const container = document.getElementById('contentContainer');
        if (container) {
            container.innerHTML = `<div class="widget-error">Konfiguration konnte nicht geladen werden: ${escapeHtml(e.message)}</div>`;
        }
    }
}

function renderConfigTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;
    const embedding = (_configData && _configData.embedding) || {};
    const desiredSlots = Number.isFinite(Number(embedding.model_slots))
        ? Number(embedding.model_slots)
        : 2;
    const effectiveSlots = embedding.effective_model_slots;
    const loadedModels = Array.isArray(embedding.loaded_models) ? embedding.loaded_models : [];
    const serviceRunning = embedding.service_running === true;
    const pendingRestart = embedding.pending_restart === true;
    const loadingModel = embedding.loading_model || '';
    const statusBadge = serviceRunning
        ? '<span class="models-badge models-badge-vram">Running</span>'
        : '<span class="models-badge models-badge-unloaded">Stopped</span>';
    const restartBadge = pendingRestart
        ? '<span class="models-badge models-badge-mixed">Neustart ausstehend</span>'
        : '<span class="models-badge models-badge-embed-active">Aktuell</span>';
    const modelListHtml = loadedModels.length
        ? loadedModels.map(escapeHtml).join(', ')
        : '-';
    const effectiveText = Number.isFinite(Number(effectiveSlots))
        ? String(effectiveSlots)
        : '-';

    container.innerHTML = `
        <div class="config-page">
            <section class="config-section">
                <div class="config-section-header">
                    <h3>Embedding-Service</h3>
                    <div class="config-status">${statusBadge}${restartBadge}</div>
                </div>
                <div class="config-grid">
                    <label class="config-field" for="embeddingSlotsInput">
                        <span>Modellslots</span>
                        <input
                            id="embeddingSlotsInput"
                            class="filter-input config-number-input"
                            type="number"
                            min="${embedding.min_model_slots || 1}"
                            max="${embedding.max_model_slots || 4}"
                            step="1"
                            value="${desiredSlots}"
                        >
                    </label>
                    <div class="config-readout">
                        <span class="config-readout-label">Effektiv</span>
                        <span>${escapeHtml(effectiveText)}</span>
                    </div>
                    <div class="config-readout">
                        <span class="config-readout-label">Geladen</span>
                        <span>${modelListHtml}</span>
                    </div>
                    <div class="config-readout">
                        <span class="config-readout-label">Loading</span>
                        <span>${escapeHtml(loadingModel || '-')}</span>
                    </div>
                </div>
                <div class="config-actions">
                    <button class="action-btn primary" id="saveEmbeddingConfigBtn" ${_configSaving ? 'disabled' : ''}>
                        ${_configSaving ? 'Speichere...' : 'Speichern & neu starten'}
                    </button>
                    <button class="action-btn" id="reloadConfigBtn" ${_configSaving ? 'disabled' : ''}>Aktualisieren</button>
                </div>
            </section>
        </div>
    `;

    const saveBtn = document.getElementById('saveEmbeddingConfigBtn');
    if (saveBtn) saveBtn.addEventListener('click', saveEmbeddingConfig);
    const reloadBtn = document.getElementById('reloadConfigBtn');
    if (reloadBtn) reloadBtn.addEventListener('click', loadRuntimeConfig);
}

async function saveEmbeddingConfig() {
    if (_configSaving) return;
    const input = document.getElementById('embeddingSlotsInput');
    const slots = input ? Number(input.value) : NaN;
    if (!Number.isInteger(slots)) {
        showNotification('Slot-Anzahl muss eine ganze Zahl sein', 'error');
        return;
    }

    _configSaving = true;
    renderConfigTab();
    try {
        const resp = await fetch('/api/config/embedding', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_slots: slots, restart: true}),
        });
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.restart?.error || data.error || resp.statusText);
        }
        _configData = data;
        showNotification('Embedding-Konfiguration gespeichert', 'success');
    } catch (e) {
        showNotification('Fehler: ' + e.message, 'error');
        await loadRuntimeConfig();
        return;
    } finally {
        _configSaving = false;
    }
    renderConfigTab();
}

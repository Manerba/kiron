/**
 * KIron Models: one Catalog and local-registration view.
 */

let _modelsData = [];
let _registrationCandidates = [];
let _registrationErrors = {};
let _registrationProfiles = [];
let registrationDialogReturnFocus = null;
let _modelsTable = null;
let _modelRowIdSequence = 0;
const _modelRowIds = new Map();
let _modelsVerificationTimer = null;
let _modelsVerificationDeadline = 0;
let _modelsRefreshPending = null;
let _modelsLastRefreshAt = null;
let _ollamaRecovery = null;
const MODEL_VERIFICATION_WINDOW_MS = 90000;
const MODEL_VERIFICATION_INTERVAL_MS = 3000;
const MODEL_REFRESH_INTERVAL_MS = 10000;

function initModelsTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    const filterBar = document.getElementById('filterBar');
    let openButton = null;
    if (filterBar) {
        filterBar.replaceChildren();
        filterBar.style.display = 'flex';
        const controlGroup = document.createElement('div');
        controlGroup.className = 'filter-group models-control-group';
        openButton = document.createElement('button');
        openButton.className = 'action-btn primary';
        openButton.id = 'modelRegistrationOpen';
        openButton.type = 'button';
        openButton.textContent = '+ Modell registrieren';
        controlGroup.appendChild(openButton);
        filterBar.appendChild(controlGroup);
    }

    container.innerHTML = `
        <div class="models-page">
            <p id="modelsRuntimeStatus" role="status" aria-live="polite"></p>
            <section id="ollamaRecovery" hidden aria-label="Ollama-Wiederherstellung">
                <p id="ollamaRecoveryMessage" role="status" aria-live="polite"></p>
                <ul id="ollamaRecoveryOperations"></ul>
                <button id="ollamaRecoveryButton" type="button" class="action-btn">Ollama wiederherstellen</button>
            </section>
            <div id="modelsActionTooltip" class="models-action-tooltip" role="tooltip" hidden></div>
            <div class="models-table-viewport">
                <div id="modelsTableContainer">
                    <div class="table-loading"><div class="spinner"></div></div>
                </div>
            </div>
            <div class="models-registration-overlay"
                 id="modelRegistrationDialog" hidden>
                <section class="models-registration-dialog"
                         role="dialog" aria-modal="true" tabindex="-1"
                         aria-labelledby="modelRegistrationTitle"
                         aria-describedby="modelRegistrationHelp">
                    <form id="modelRegistrationForm">
                    <div class="models-dialog-header">
                        <h2 id="modelRegistrationTitle">Lokales Modell registrieren</h2>
                        <button class="models-dialog-close" id="modelRegistrationClose"
                                type="button" aria-label="Dialog schließen">×</button>
                    </div>
                    <p id="modelRegistrationHelp" class="text-muted">
                        Zur Auswahl stehen nur bereits vorhandene Modelle, die KIron
                        noch nicht aus Catalog oder lokaler Registry kennt.
                    </p>
                    <label for="modelRegistrationProvider">Provider</label>
                    <select id="modelRegistrationProvider" required>
                        <option value="ollama">Ollama</option>
                        <option value="prism">Prism (lokale GGUF-Gewichte)</option>
                        <option value="kiron_embeddings">KIron Embeddings (lokale Gewichte)</option>
                        <option value="kiron_deberta">KIron Reranking / NLI (lokale Gewichte)</option>
                    </select>
                    <label id="modelRegistrationReferenceLabel"
                           for="modelRegistrationReference">Vorhandenes lokales Modell</label>
                    <select id="modelRegistrationReference" required>
                        <option value="">Modelle werden geladen…</option>
                    </select>
                    <p id="modelRegistrationCandidateStatus" class="text-muted"
                       aria-live="polite"></p>
                    <div class="models-loader-field" id="modelRegistrationLoaderField" hidden>
                        <label for="modelRegistrationLoader">Loader</label>
                        <select id="modelRegistrationLoader" disabled required>
                            <option value="sentence_transformers">Sentence Transformers</option>
                            <option value="transformers_last_token">Transformers Last Token</option>
                            <option value="colbert_xmod">ColBERT XMOD</option>
                            <option value="cross_encoder">Cross Encoder</option>
                            <option value="mankei_last_token">Mankei Last Token</option>
                        </select>
                    </div>
                    <div id="modelRegistrationProfileField" hidden>
                        <label for="modelRegistrationProfile">Freigegebenes Laufzeitprofil</label>
                        <select id="modelRegistrationProfile" disabled required></select>
                        <label for="modelRegistrationProjector">Vision-Projektor (optional)</label>
                        <select id="modelRegistrationProjector" disabled>
                            <option value="">Ohne Projektor</option>
                        </select>
                        <p class="text-muted">Dateiidentität und Zusammengehörigkeit werden bei der Registrierung geprüft. Die Registrierung bestätigt keine Fähigkeiten.</p>
                    </div>
                    <p class="models-registration-error" id="modelRegistrationError"
                       role="alert" aria-live="polite"></p>
                    <div class="models-dialog-actions">
                        <button class="action-btn" id="modelRegistrationCancel"
                                type="button">Abbrechen</button>
                        <button class="action-btn primary" id="modelRegistrationSubmit"
                                type="submit">Registrieren</button>
                    </div>
                    </form>
                </section>
            </div>
        </div>
    `;

    const overlay = document.getElementById('modelRegistrationDialog');
    const provider = document.getElementById('modelRegistrationProvider');
    const form = document.getElementById('modelRegistrationForm');
    const tableContainer = document.getElementById('modelsTableContainer');

    if (openButton) {
        openButton.addEventListener(
            'click',
            () => showRegistrationDialog(openButton)
        );
    }
    tableContainer.addEventListener('click', handleModelsTableAction, true);
    initModelActionTooltips(container.querySelector('.models-page'), tableContainer);
    _modelsTable = createModelsTable();
    tables.models = _modelsTable;
    document.getElementById('modelRegistrationClose').addEventListener(
        'click',
        closeRegistrationOverlay
    );
    document.getElementById('modelRegistrationCancel').addEventListener(
        'click',
        closeRegistrationOverlay
    );
    provider.addEventListener('change', () => {
        updateRegistrationFields();
        renderRegistrationCandidates();
    });
    form.addEventListener('submit', submitModelRegistration);
    overlay.addEventListener('click', event => {
        if (event.target === overlay) closeRegistrationOverlay();
    });
    overlay.addEventListener('keydown', registrationOverlayKeydown);

    updateRegistrationFields();
    loadLocalModels();
}

function showRegistrationDialog(trigger) {
    const overlay = document.getElementById('modelRegistrationDialog');
    const form = document.getElementById('modelRegistrationForm');
    registrationDialogReturnFocus = trigger;
    form.reset();
    clearRegistrationError();
    updateRegistrationFields();
    overlay.hidden = false;
    document.getElementById('modelRegistrationProvider').focus();
    loadRegistrationCandidates();
}

function closeRegistrationOverlay() {
    const overlay = document.getElementById('modelRegistrationDialog');
    if (!overlay || overlay.hidden) return;
    overlay.hidden = true;
    clearRegistrationError();
    if (
        registrationDialogReturnFocus
        && document.contains(registrationDialogReturnFocus)
    ) {
        registrationDialogReturnFocus.focus();
    }
    registrationDialogReturnFocus = null;
}

function registrationOverlayKeydown(event) {
    const overlay = event.currentTarget;
    if (event.key === 'Escape') {
        event.preventDefault();
        closeRegistrationOverlay();
        return;
    }
    if (event.key !== 'Tab') return;

    const focusable = Array.from(overlay.querySelectorAll(
        'button:not([disabled]), select:not([disabled]), '
        + '[href], [tabindex]:not([tabindex="-1"])'
    )).filter(node => !node.hidden && node.offsetParent !== null);
    if (!focusable.length) {
        event.preventDefault();
        overlay.querySelector('[role="dialog"]')?.focus();
        return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

function clearRegistrationError() {
    const errorNode = document.getElementById('modelRegistrationError');
    if (errorNode) errorNode.textContent = '';
}

function updateRegistrationFields() {
    const provider = document.getElementById('modelRegistrationProvider');
    const loader = document.getElementById('modelRegistrationLoader');
    const loaderField = document.getElementById('modelRegistrationLoaderField');
    const referenceLabel = document.getElementById(
        'modelRegistrationReferenceLabel'
    );
    if (!provider || !loader || !loaderField || !referenceLabel) return;
    const huggingface = ['kiron_embeddings', 'kiron_deberta'].includes(provider.value);
    const reranker = provider.value === 'kiron_deberta';
    for (const option of loader.options) {
        const forReranker = ['cross_encoder', 'mankei_last_token'].includes(option.value);
        option.hidden = forReranker !== reranker;
        option.disabled = option.hidden;
    }
    if (loader.selectedOptions[0]?.disabled) {
        loader.value = reranker ? 'cross_encoder' : 'sentence_transformers';
    }
    loader.disabled = !huggingface;
    loaderField.hidden = !huggingface;
    const profileField = document.getElementById('modelRegistrationProfileField');
    const gguf = _registrationCandidates.some(item => item.runtime_provider === provider.value && item.artifact_format === 'gguf');
    profileField.hidden = !gguf;
    document.getElementById('modelRegistrationProfile').disabled = !gguf;
    document.getElementById('modelRegistrationProjector').disabled = !gguf;
    referenceLabel.textContent = huggingface
        ? 'Vorhandenes lokales Modellverzeichnis'
        : 'Vorhandenes lokales Modell';
}

function renderRegistrationCandidates() {
    const provider = document.getElementById('modelRegistrationProvider');
    const reference = document.getElementById('modelRegistrationReference');
    const status = document.getElementById('modelRegistrationCandidateStatus');
    const submit = document.getElementById('modelRegistrationSubmit');
    if (!provider || !reference || !status || !submit) return;

    const matching = _registrationCandidates.filter(
        candidate => candidate.runtime_provider === provider.value
    );
    reference.replaceChildren();
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.selected = true;
    placeholder.disabled = true;
    placeholder.textContent = matching.length
        ? 'Modell auswählen…'
        : 'Keine unregistrierten lokalen Modelle gefunden';
    reference.appendChild(placeholder);

    for (const candidate of matching) {
        const option = document.createElement('option');
        option.value = candidate.candidate_id;
        option.textContent = candidate.display_name;
        reference.appendChild(option);
    }
    const profile = document.getElementById('modelRegistrationProfile');
    const projector = document.getElementById('modelRegistrationProjector');
    profile.replaceChildren();
    projector.replaceChildren();
    const none = document.createElement('option');
    none.value = ''; none.textContent = 'Ohne Projektor'; projector.appendChild(none);
    for (const item of _registrationProfiles) {
        const option = document.createElement('option');
        option.value = item.id; option.textContent = item.id; profile.appendChild(option);
    }
    for (const candidate of matching.filter(item => item.artifact_format === 'gguf')) {
        const option = document.createElement('option');
        option.value = candidate.candidate_id; option.textContent = candidate.display_name;
        projector.appendChild(option);
    }
    updateRegistrationFields();
    reference.disabled = matching.length === 0;
    submit.disabled = matching.length === 0;
    const providerError = _registrationErrors[provider.value];
    status.textContent = providerError
        ? `Lokaler Provider nicht erreichbar oder Bestand ungültig: ${providerError}`
        : matching.length
        ? `${matching.length} unregistrierte lokale Modelle gefunden.`
        : 'Keine unregistrierten lokalen Modelle gefunden.';
}

async function loadRegistrationCandidates() {
    const reference = document.getElementById('modelRegistrationReference');
    const status = document.getElementById('modelRegistrationCandidateStatus');
    const submit = document.getElementById('modelRegistrationSubmit');
    if (!reference || !status || !submit) return;
    reference.disabled = true;
    submit.disabled = true;
    status.textContent = 'Lokale Modelle werden gelesen…';
    try {
        const response = await fetch('/api/models/registration-candidates');
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data?.error?.message || response.statusText);
        }
        _registrationCandidates = Array.isArray(data.candidates)
            ? data.candidates.filter(candidate => (
                candidate
                && ['ollama', 'prism', 'kiron_embeddings', 'kiron_deberta'].includes(candidate.runtime_provider)
                && ['ollama', 'local', 'huggingface'].includes(candidate.artifact_origin)
                && ['ollama_manifest', 'hf_weights', 'gguf'].includes(candidate.artifact_format)
                && typeof candidate.candidate_id === 'string'
                && typeof candidate.display_name === 'string'
            ))
            : [];
        _registrationProfiles = Array.isArray(data.runtime_profiles) ? data.runtime_profiles : [];
        _registrationErrors = data.errors && typeof data.errors === 'object' ? data.errors : {};
        renderRegistrationCandidates();
    } catch (error) {
        _registrationCandidates = [];
        reference.replaceChildren();
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Lokale Modelle konnten nicht gelesen werden';
        reference.appendChild(option);
        status.textContent = `Kandidaten konnten nicht geladen werden: ${error.message}`;
        reference.disabled = true;
        submit.disabled = true;
    }
}

async function submitModelRegistration(event) {
    event.preventDefault();
    const providerNode = document.getElementById('modelRegistrationProvider');
    const referenceNode = document.getElementById('modelRegistrationReference');
    const loaderNode = document.getElementById('modelRegistrationLoader');
    const submitButton = document.getElementById('modelRegistrationSubmit');
    const errorNode = document.getElementById('modelRegistrationError');
    const payload = {
        candidate_id: referenceNode.value,
    };
    if (['kiron_embeddings', 'kiron_deberta'].includes(providerNode.value)) {
        payload.loader = loaderNode.value;
    }

    const candidate = _registrationCandidates.find(item => item.candidate_id === referenceNode.value);
    if (candidate?.artifact_format === 'gguf') {
        payload.runtime_profile = document.getElementById('modelRegistrationProfile').value;
        const projectorId = document.getElementById('modelRegistrationProjector').value;
        if (projectorId) payload.projector_id = projectorId;
    }
    submitButton.disabled = true;
    errorNode.textContent = '';
    try {
        const response = await fetch('/api/models/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'X-Kiron-Action': 'models'},
            body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const code = data?.error?.code || 'registration_failed';
            const message = data?.error?.message || response.statusText;
            errorNode.textContent = `${code}: ${message}`;
            return;
        }
        await loadLocalModels();
        closeRegistrationOverlay();
        showNotification('Modell lokal registriert', 'success');
    } catch (error) {
        errorNode.textContent = `registration_failed: ${error.message}`;
    } finally {
        submitButton.disabled = !referenceNode.value;
    }
}

async function fetchLocalModels() {
    _modelsLastRefreshAt = Date.now();
    const response = await fetch('/api/models/control');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || response.statusText);
    _modelsData = data.models;
    const status = document.getElementById('modelsRuntimeStatus');
    if (status) status.textContent = (data.warnings || []).join(' · ');
    _ollamaRecovery = data.ollama_recovery;
    const recovery = document.getElementById('ollamaRecovery');
    if (recovery) {
        recovery.hidden = !_ollamaRecovery?.blocked;
        document.getElementById('ollamaRecoveryMessage').textContent = _ollamaRecovery?.message || '';
        const operations = document.getElementById('ollamaRecoveryOperations');
        operations.replaceChildren();
        for (const operation of _ollamaRecovery?.operations || []) {
            const item = document.createElement('li');
            item.textContent = `${operation.model} · ${operation.state} · Besitzer ${operation.owner} · Letztes Lebenszeichen: ${new Date(operation.last_seen_at).toLocaleString()} · Vorgang ${operation.operation_id}`;
            operations.appendChild(item);
        }
        const button = document.getElementById('ollamaRecoveryButton');
        button.disabled = !_ollamaRecovery?.recovery_available;
        button.onclick = recoverOllama;
    }
}

async function recoverOllama() {
    if (!_ollamaRecovery?.recovery_available) return;
    if (!confirm('Ollama wiederherstellen? Dabei werden alle geladenen Ollama-Modelle entladen und der Modelldienst neu gestartet.')) return;
    const button = document.getElementById('ollamaRecoveryButton');
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    try {
        const response = await fetch('/api/ollama/recovery', {
            method: 'POST', headers: {'Content-Type': 'application/json', 'X-Kiron-Action': 'models'},
            body: JSON.stringify({revision: _ollamaRecovery.revision}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error?.message || 'Wiederherstellung fehlgeschlagen.');
        showNotification('Ollama wurde wiederhergestellt. Modelle können wieder geladen werden.', 'success');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        button.removeAttribute('aria-busy');
        await loadLocalModels();
    }
}

function scheduleModelVerificationRefresh() {
    if (_modelsVerificationTimer !== null) clearTimeout(_modelsVerificationTimer);
    _modelsVerificationTimer = null;
    if (!_modelsData.some(model => model.runtime && model.error_code === 'artifact_verification_unconfirmed')) return;
    if (typeof currentTab !== 'string' || currentTab !== 'models' || !document.getElementById('modelsTableContainer')) return;
    const remaining = _modelsVerificationDeadline - Date.now();
    const status = document.getElementById('modelsRuntimeStatus');
    if (remaining <= 0) {
        if (status) status.textContent += ' Artefaktprüfung weiterhin ungeklärt. Bitte „Diagnose aktualisieren“ verwenden.';
        return;
    }
    if (status) status.textContent += ' Artefaktprüfung noch nicht bestätigt; automatische Nachprüfung läuft.';
    _modelsVerificationTimer = setTimeout(() => {
        _modelsVerificationTimer = null;
        if (Date.now() >= _modelsVerificationDeadline) scheduleModelVerificationRefresh();
        else if (typeof currentTab === 'string' && currentTab === 'models' && document.getElementById('modelsTableContainer')) loadLocalModels(true);
    }, Math.min(MODEL_VERIFICATION_INTERVAL_MS, remaining));
}

async function loadLocalModels(automatic = false) {
    const tableContainer = document.getElementById('modelsTableContainer');
    if (!tableContainer) return;
    if (!automatic) _modelsVerificationDeadline = Date.now() + MODEL_VERIFICATION_WINDOW_MS;
    const work = _modelsRefreshPending || fetchLocalModels();
    _modelsRefreshPending = work;
    try {
        await work;
        if (document.getElementById('modelsTableContainer') === tableContainer) renderModelsTable();
    } catch (error) {
        if (document.getElementById('modelsTableContainer') !== tableContainer) return;
        const message = document.createElement('div');
        message.className = 'no-data';
        message.textContent = `Modelle konnten nicht geladen werden: ${error.message}`;
        tableContainer.replaceChildren(message);
    } finally {
        if (_modelsRefreshPending === work) {
            _modelsRefreshPending = null;
            scheduleModelVerificationRefresh();
        }
    }
}

function modelTypeLabel(value) {
    const labels = {
        llm: 'LLM',
        vlm: 'VLM',
        embedding: 'Embedding',
        colbert: 'ColBERT',
        rerank: 'Rerank',
    };
    return labels[value] || String(value || '—');
}

function statusLabel(model) {
    const states = {unknown: 'Status unbekannt', unloaded: 'Nicht geladen', loading: 'Wird geladen',
        loaded: 'Geladen', unloading: 'Wird entladen', failed: 'Laufzeitfehler'};
    return states[model.operation_state] || 'Status unbekannt';
}

function verificationLabel(model) {
    return {verified: 'Nachweise passend', deviating: 'Nachweise abweichend', unverified: 'Ungeprüft'}[model.verification?.status] || 'Ungeprüft';
}

function configurationLabel(model) {
    const configuration = model.configuration || {};
    const device = configuration.device === 'gpu' && configuration.ram_gb > 0
        ? 'GPU + CPU' : {cpu: 'CPU', gpu: 'GPU'}[configuration.device];
    const parts = [];
    if (device) parts.push(device);
    if (Number.isInteger(configuration.context_tokens)) parts.push(`Kontext ${configuration.context_tokens}`);
    return parts.join(' · ') || 'Nicht beobachtet';
}

function renderModelStatus(table, model) {
    return modelBadgeHtml(table, statusLabel(model),
        model.operation_state === 'loaded' ? 'models-badge-loaded' : 'models-badge-unloaded');
}

function modelSignalHtml(table, color, explanation) {
    const symbol = {green: '✓', yellow: '?', red: '!', grey: '–'}[color];
    const label = table.escapeHtml(explanation);
    return `<span class="models-signal models-signal-${color}" tabindex="0" role="img"`
        + ` aria-label="${label}" data-model-tooltip="${label}"><span aria-hidden="true">${symbol}</span></span>`;
}

function renderModelConfiguration(table, model) {
    const config = model.configuration || {};
    const deviceKnown = ['cpu', 'gpu'].includes(config.device);
    const contextKnown = Number.isInteger(config.context_tokens);
    const color = deviceKnown && contextKnown ? 'green' : deviceKnown || contextKnown ? 'yellow' : 'grey';
    const explanation = color === 'green'
        ? 'Gerät und Kontext sind vom laufenden Modell beobachtet. Dies ist keine Bewertung der Fähigkeitsnachweise.'
        : color === 'yellow'
            ? `Teilweise beobachtet: ${deviceKnown ? 'Der Kontext ist' : 'Das Gerät ist'} nicht bekannt.`
            : model.operation_state === 'unloaded'
                ? 'Das Modell ist nicht geladen; es gibt keine aktuelle Betriebskonfiguration zu beobachten.'
                : 'Der Dienst liefert keine belastbaren Angaben zum aktuellen Gerät und Kontext. Das bedeutet nicht, dass das Modell fehlerhaft ist.';
    return modelSignalHtml(table, color, `Konfiguration: ${configurationLabel(model)}. ${explanation}`);
}

function renderModelVerification(table, model) {
    const status = model.verification?.status || 'unverified';
    const color = {verified: 'green', deviating: 'red', unverified: 'yellow'}[status] || 'yellow';
    const explanation = status === 'verified'
        ? 'Die aktuelle Konfiguration passt zu den gültigen Fähigkeitsnachweisen und die Laufzeit-/Identitätsprüfung ist bestätigt.'
        : status === 'deviating'
            ? 'Die aktuelle Konfiguration weicht von den belegten Testbedingungen ab. Das ist kein Ladefehler; Entladen bleibt davon unabhängig.'
            : model.operation_state !== 'loaded'
                ? 'Es ist kein geladenes Modell bestätigt, dessen aktuelle Konfiguration mit den Nachweisen abgeglichen werden kann.'
                : !(model.verification?.checks || []).length
                    ? 'Es liegen keine gültig gebundenen Fähigkeitsnachweise für den Abgleich vor.'
                    : 'Der Abgleich ist unvollständig: beobachtete Konfigurationswerte oder eine bestätigte Laufzeit-/Identitätsprüfung fehlen.';
    const checks = (model.verification?.checks || []).map(check => {
        const expected = Object.entries(check.expected || {}).map(([key, constraint]) => {
            const label = {device: 'Gerät', context_tokens: 'Kontext'}[key] || key;
            const values = constraint.allowed_values?.join(', ')
                || `${constraint.minimum ?? 'offen'} bis ${constraint.maximum ?? 'offen'}`;
            return `${label} ${values}`;
        }).join(', ');
        return expected ? `${check.capability}: ${expected}` : '';
    }).filter(Boolean).join('; ');
    return modelSignalHtml(table, color,
        `${verificationLabel(model)}. ${explanation} Aktuell: ${configurationLabel(model)}.${checks ? ' Belegt für ' + checks + '.' : ''}`);
}

function renderModelProfileVerification(table, model) {
    const profiles = model.embedding_profiles || [];
    if (!profiles.length) {
        return modelSignalHtml(table, 'grey', 'Für dieses Modell ist kein Embedding-Profil ausgewiesen.');
    }
    const verified = profiles.every(profile => profile.verification?.status === 'verified');
    const summary = profiles.map(profile => `${profile.profile_id} (${profile.endpoint}): `
        + (profile.verification?.status === 'verified' ? 'verifiziert' : 'ungeprüft')).join('; ');
    return modelSignalHtml(table, verified ? 'green' : 'yellow', `Embedding-Profilnachweise: ${summary}.`);
}

function renderEmbeddingProfileDetail(table, model) {
    const profiles = model.embedding_profiles || [];
    if (!profiles.length) return '';
    return '<section class="models-detail-section"><h3>Embedding-Profilnachweise</h3>'
        + profiles.map(profile => '<dl>'
            + modelDetailItem(table, 'Profil', profile.profile_id, true)
            + modelDetailItem(table, 'Endpoint', profile.endpoint, true)
            + modelDetailItem(table, 'Profilprüfung', profile.verification?.status === 'verified' ? 'Verifiziert' : 'Ungeprüft')
            + modelDetailItem(table, 'Index-Kompatibilität', profile.index_compatibility_id || 'Nicht belegt', true)
            + modelDetailItem(table, 'Query-Kompatibilität', profile.query_compatibility_id || 'Nicht belegt', true)
            + ((profile.verification?.blocking_reasons || []).length
                ? modelDetailItem(table, 'Fehlende Nachweise', modelListLabel(profile.verification.blocking_reasons), true) : '')
            + '</dl>').join('') + '</section>';
}

function modelServiceMemory(model) {
    const memory = model.service_memory;
    return model.catalog_managed && model.operation_state === 'loaded'
        && Array.isArray(memory?.loaded_models) && memory.loaded_models.includes(model.name)
        ? memory : null;
}

function memoryAmount(value) {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0
        ? value.toFixed(2) : '—';
}

function memoryLabel(model) {
    if (model.operation_state === 'unknown') return 'Unbekannt';
    const service = modelServiceMemory(model);
    const memory = service || model;
    const values = `${memoryAmount(memory.vram_gb)} / ${memoryAmount(memory.ram_gb)}`;
    if (!service) return values === '— / —' ? 'Unbekannt' : values;
    const shared = service.loaded_models.length > 1 ? `, ${service.loaded_models.length} Modelle` : '';
    return `${values} (Dienst${shared})`;
}

function memoryScopeLabel(model) {
    const service = modelServiceMemory(model);
    if (model.backend === 'ollama') {
        return 'Von Ollama gemeldete Speicheraufteilung des geladenen Modells einschließlich Kontextspeicher. '
            + 'RAM ist der Anteil außerhalb des VRAM. Zusätzlicher Prozessspeicher ist darin nicht vollständig enthalten.';
    }
    if (!service) return 'Speicherbelegung des Modells, soweit gemessen.';
    const count = service.loaded_models.length;
    return `Dienstspeicher für ${count} ${count === 1 ? 'geladenes Modell' : 'geladene Modelle'}. `
        + 'VRAM einschließlich CUDA-Kontext und Cache; RAM als residenter Arbeitsspeicher (RSS). '
        + (count > 1 ? 'Der gemeinsame Wert erscheint bei jedem beteiligten Modell und zählt insgesamt einmal.' : '');
}

function sizeLabel(value) {
    return typeof value === 'number' && Number.isFinite(value)
        ? `${value.toFixed(2)} GB`
        : 'Unbekannt';
}

function modelIdentityKey(model) {
    return [
        model.backend || '',
        model.canonical_model_id || '',
        model.registry_id || '',
        model.reference || '',
        model.name || '',
    ].join('\u001f');
}

function modelRowId(model) {
    const identity = modelIdentityKey(model);
    if (!_modelRowIds.has(identity)) {
        _modelRowIdSequence += 1;
        _modelRowIds.set(identity, `model-row-${_modelRowIdSequence}`);
    }
    return _modelRowIds.get(identity);
}

function modelOriginLabel(model) {
    const origins = [];
    if (model.catalog_managed) origins.push('Catalog');
    if (model.locally_registered || model.registered) origins.push('Lokal registriert');
    return origins.length ? origins.join(' + ') : '—';
}

function modelListLabel(value) {
    return Array.isArray(value) && value.length ? value.join(', ') : '—';
}

function tokenCountLabel(value, fallback = 'Nicht ermittelt') {
    return Number.isInteger(value) && value > 0
        ? `${value.toLocaleString('de-DE')} Token`
        : fallback;
}

function dateTimeLabel(value) {
    if (!value) return '—';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime())
        ? String(value)
        : parsed.toLocaleString('de-DE');
}

function yesNoLabel(value) {
    return value === true ? 'Ja' : value === false ? 'Nein' : 'Unbekannt';
}

function modelBadgeHtml(table, label, className) {
    return `<span class="models-badge ${className}">`
        + `${table.escapeHtml(label)}</span>`;
}

function renderModelOrigin(table, model) {
    const badges = [modelBadgeHtml(table, model.runtime_provider || model.backend || 'Unbekannter Provider', 'models-badge-provider')];
    if (model.catalog_managed) {
        badges.push(modelBadgeHtml(table, 'Catalog', 'models-badge-catalog'));
    }
    if (model.locally_registered || model.registered) {
        badges.push(
            modelBadgeHtml(table, 'Lokal registriert', 'models-badge-local')
        );
    }
    return badges.length ? badges.join(' ') : '—';
}

function renderModelActions(table, row) {
    const icons = {
        load: '<path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/>',
        load_cpu: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
        unload: '<path d="M12 16V4m-5 5 5-5 5 5M5 16v5h14v-5"/>',
        delete: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
        health: '<path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-1l2 3M4 15l2 3a7 7 0 0 0 12-1"/>',
        warmup: '<path d="M12 3c1 5 6 6 6 12a6 6 0 0 1-12 0c0-3 2-5 3-6 0 3 1 4 2 4 2-3 1-6 1-10Z"/>',
    };
    const actions = (row._model.controls || []).map(control => {
        const icon = icons[control.id];
        if (!icon) return '';
        const label = `${control.label}: ${row._model.name}`;
        const tooltip = table.escapeHtml(control.reason ? `${label} – ${control.reason}` : label);
        const className = control.id === 'delete' ? 'danger' : control.id === 'load' ? 'primary' : '';
        return `<button type="button" class="action-btn models-action-icon ${className}"`
            + ` data-model-action="${table.escapeHtml(control.id)}" data-model-row="${table.escapeHtml(row.id)}"`
            + ` data-model-tooltip="${tooltip}" aria-label="${table.escapeHtml(label)}" aria-disabled="${!control.enabled}">`
            + '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor"'
            + ` stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${icon}</svg></button>`;
    });
    return actions.length ? `<span class="models-actions">${actions.join('')}</span>` : '—';
}

function modelDetailItem(table, label, value, code = false) {
    const rendered = value === null || value === undefined || value === ''
        ? '—'
        : String(value);
    const valueClass = code ? ' models-detail-code' : '';
    return '<div class="models-detail-item">'
        + `<dt>${table.escapeHtml(label)}</dt>`
        + `<dd class="${valueClass.trim()}">${table.escapeHtml(rendered)}</dd>`
        + '</div>';
}

function renderCatalogTokenLimits(table, model) {
    const limits = Array.isArray(model.catalog_token_limits)
        ? model.catalog_token_limits
        : [];
    if (!limits.length) {
        return '<p class="models-detail-empty">'
            + (model.catalog_managed
                ? 'Keine profilbezogenen Token-Limits deklariert.'
                : 'Nicht im Catalog verwaltet.')
            + '</p>';
    }

    return '<div class="models-token-limits">' + limits.map(limit => {
        const roles = limit.by_role && typeof limit.by_role === 'object'
            ? Object.entries(limit.by_role)
            : [];
        const renderedRoles = roles.map(([role, value]) => {
            const labels = {
                search_query: 'Suchanfrage',
                search_document: 'Dokument',
            };
            return '<span class="models-token-role">'
                + `<span>${table.escapeHtml(labels[role] || role)}</span>`
                + `<strong>${table.escapeHtml(tokenCountLabel(value, 'Unbekannt'))}</strong>`
                + '</span>';
        }).join('');
        const policy = [
            limit.unit,
            limit.counting,
            `Truncation: ${limit.truncation || 'unbekannt'}`,
            `Overflow: ${limit.overflow || 'unbekannt'}`,
        ].filter(Boolean).join(' · ');
        return '<article class="models-token-profile">'
            + `<h4>${table.escapeHtml(limit.profile_id || 'Profil')}</h4>`
            + `<div class="models-token-roles">${renderedRoles || '—'}</div>`
            + `<p>${table.escapeHtml(policy)}</p>`
            + '</article>';
    }).join('') + '</div>';
}

function renderModelDetail(table, model) {
    const ollamaModel = model.backend === 'ollama';
    const nativeContext = ollamaModel
        ? tokenCountLabel(model.native_context_length)
        : 'Nicht über Ollama bereitgestellt';
    const runtimeContext = !ollamaModel
        ? 'Nicht zutreffend'
        : model.operation_state === 'loaded'
            ? tokenCountLabel(model.configuration?.context_tokens)
            : model.operation_state === 'unknown'
                ? 'Load-Status unbekannt'
                : 'Nicht geladen';
    const catalogContext = model.catalog_managed
        ? tokenCountLabel(model.catalog_context_length, 'Nicht deklariert')
        : 'Nicht im Catalog verwaltet';
    const memory = modelServiceMemory(model) || model;
    const memoryVram = memoryAmount(memory.vram_gb);
    const memoryRam = memoryAmount(memory.ram_gb);

    return '<div class="detail-content models-detail-content">'
        + '<section class="models-detail-block">'
        + '<h3>Token &amp; Kontext</h3>'
        + '<dl class="models-detail-grid">'
        + modelDetailItem(
            table,
            'Native Modellgrenze',
            nativeContext
        )
        + modelDetailItem(table, 'Aktiver Ollama-Runner', runtimeContext)
        + modelDetailItem(table, 'Catalog-Kontextwert (aggregiert)', catalogContext)
        + '</dl>'
        + '<p class="models-context-note">Der Runner-Wert gilt für die aktuelle '
        + 'Ollama-Ladung. Ein Request mit abweichendem num_ctx kann den Runner neu laden.</p>'
        + '<div class="models-detail-subheading">Catalog-Limits je Profil/Rolle</div>'
        + renderCatalogTokenLimits(table, model)
        + '</section>'
        + '<section class="models-detail-block">'
        + '<h3>Identität</h3>'
        + '<dl class="models-detail-grid">'
        + modelDetailItem(table, 'Modellname', model.name, true)
        + modelDetailItem(table, 'Canonical-ID', model.canonical_model_id, true)
        + modelDetailItem(table, 'Aliase', modelListLabel(model.aliases), true)
        + modelDetailItem(table, 'Herkunft', modelOriginLabel(model))
        + modelDetailItem(table, 'Backend', model.backend, true)
        + modelDetailItem(table, 'Provider', model.provider, true)
        + modelDetailItem(table, 'Lokale Referenz', model.reference, true)
        + modelDetailItem(table, 'Loader', model.loader, true)
        + modelDetailItem(table, 'Registry-ID', model.registry_id, true)
        + modelDetailItem(table, 'Catalog-Digest', model.catalog_digest, true)
        + '</dl>'
        + '</section>'
        + '<section class="models-detail-block">'
        + '<h3>Runtime</h3>'
        + '<dl class="models-detail-grid">'
        + modelDetailItem(table, 'Betriebszustand', statusLabel(model))
        + modelDetailItem(table, 'Beobachtete Konfiguration', configurationLabel(model))
        + modelDetailItem(table, 'Nachweisstatus', verificationLabel(model))
        + modelDetailItem(table, 'Installationsdiagnose', model.diagnostic || '—')
        + modelDetailItem(table, 'Konfiguriert', yesNoLabel(model.configured))
        + modelDetailItem(table, 'Installiert', yesNoLabel(model.installed))
        + modelDetailItem(table, 'Typ', modelTypeLabel(model.model_type))
        + modelDetailItem(table, 'Familie', model.family)
        + modelDetailItem(table, 'Format', model.format)
        + modelDetailItem(table, 'Parameter', model.parameter_size)
        + modelDetailItem(table, 'Quantisierung', model.quantization_level)
        + modelDetailItem(table, 'Modellgröße', sizeLabel(model.size_gb))
        + modelDetailItem(table, 'VRAM', memoryVram === '—' ? 'Unbekannt' : `${memoryVram} GiB`)
        + modelDetailItem(table, 'RAM', memoryRam === '—' ? 'Unbekannt' : `${memoryRam} GiB`)
        + modelDetailItem(table, 'Speichermessung', memoryScopeLabel(model))
        + modelDetailItem(table, 'Geladen bis', dateTimeLabel(model.expires_at))
        + modelDetailItem(table, 'Tasks', modelListLabel(model.tasks), true)
        + modelDetailItem(table, 'Endpoints', modelListLabel(model.endpoints), true)
        + modelDetailItem(table, 'Profile', modelListLabel(model.profile_ids), true)
        + modelDetailItem(
            table,
            'Deployments',
            modelListLabel(model.deployment_ids),
            true
        )
        + '</dl>'
        + '</section>'
        + '</div>';
}

function createModelsTable() {
    const table = new ExpandableTable({
        id: 'models',
        showHeader: true,
        expandable: true,
        gridTemplate: '30px minmax(180px, 1.6fr) minmax(130px, 1fr) minmax(225px, 1.2fr) '
            + '140px 90px minmax(100px, 1fr) 95px 85px 90px '
            + '112px 70px 80px minmax(205px, auto)',
        defaultSort: {field: 'name', direction: 'asc'},
        columns: [
            {field: 'name', label: 'Modell', sortable: true, align: 'left', renderer: 'modelName'},
            {field: 'operation_state', label: 'Status', sortable: true, align: 'left', renderer: 'modelStatus'},
            {field: 'vram_gb', label: 'Speicher (VRAM/RAM in GiB)', sortable: true, align: 'left', renderer: 'modelMemory'},
            {field: 'source', label: 'Herkunft', sortable: false, align: 'left', renderer: 'modelOrigin'},
            {field: 'model_type', label: 'Typ', sortable: true, align: 'left', renderer: 'modelType'},
            {field: 'family', label: 'Familie', sortable: true, align: 'left'},
            {field: 'parameter_size', label: 'Parameter', sortable: true, align: 'left'},
            {field: 'quantization_level', label: 'Quant.', sortable: true, align: 'left'},
            {field: 'size_gb', label: 'Größe', sortable: true, align: 'right', renderer: 'modelSize'},
            {field: 'configuration', label: 'Konfiguration', sortable: false, align: 'center', renderer: 'modelConfiguration'},
            {field: 'embedding_profiles', label: 'Profil', sortable: false, align: 'center', renderer: 'modelProfileVerification'},
            {field: 'verification', label: 'Laufzeit', sortable: false, align: 'center', renderer: 'modelVerification'},
            {field: 'name', label: 'Aktionen', sortable: false, align: 'right', renderer: 'modelActions'},
        ],
        detailFields: [],
    }, 'modelsTableContainer');

    table.renderers.modelName = (value) => (
        `<span class="models-name">${table.escapeHtml(value || '—')}</span>`
    );
    table.renderers.modelOrigin = (_value, _column, row) => (
        renderModelOrigin(table, row._model)
    );
    table.renderers.modelType = (value) => table.escapeHtml(modelTypeLabel(value));
    table.renderers.modelSize = (value) => table.escapeHtml(sizeLabel(value));
    table.renderers.modelStatus = (_value, _column, row) => renderModelStatus(table, row._model);
    table.renderers.modelConfiguration = (_value, _column, row) => renderModelConfiguration(table, row._model);
    table.renderers.modelProfileVerification = (_value, _column, row) => renderModelProfileVerification(table, row._model);
    table.renderers.modelVerification = (_value, _column, row) => renderModelVerification(table, row._model);
    table.renderers.modelMemory = (_value, _column, row) => (
        `<span title="${table.escapeHtml(memoryScopeLabel(row._model))}">${table.escapeHtml(memoryLabel(row._model))}</span>`
    );
    table.renderers.modelActions = (_value, _column, row) => (
        renderModelActions(table, row)
    );
    table.renderDetail = row => renderModelDetail(table, row._model)
        + renderEmbeddingProfileDetail(table, row._model) + renderRuntimeDetail(table, row._model);
    return table;
}

function handleModelsTableAction(event) {
    const button = event.target.closest('[data-model-action]');
    if (!button || !_modelsTable) return;
    event.preventDefault();
    event.stopPropagation();

    const row = _modelsTable.getAllData().find(
        item => item.id === button.dataset.modelRow
    );
    if (!row) return;
    const model = row._model;
    const control = model.controls.find(item => item.id === button.dataset.modelAction);
    if (!control?.enabled || button.disabled) return;
    void performModelAction(model, control, button);
}

function renderModelsTable() {
    const tooltip = document.getElementById('modelsActionTooltip');
    if (tooltip) tooltip.hidden = true;
    const tableContainer = document.getElementById('modelsTableContainer');
    if (!tableContainer || !_modelsTable) return;
    const expanded = new Set(
        Array.from(tableContainer.querySelectorAll('.table-row-wrapper.expanded'))
            .map(node => node.dataset.rowId)
    );

    if (_modelsData.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'no-data';
        empty.textContent = 'Keine Catalog- oder lokalen Registryeinträge vorhanden.';
        tableContainer.replaceChildren(empty);
        _modelsTable.data = [];
        _modelsTable.filteredData = [];
        return;
    }

    _modelsTable.data = _modelsData.map(model => ({
        ...model,
        vram_gb: modelServiceMemory(model)?.vram_gb ?? model.vram_gb,
        id: modelRowId(model),
        _model: model,
    }));
    _modelsTable.applyFiltersAndRender();

    for (const rowId of expanded) {
        const wrapper = tableContainer.querySelector(`[data-row-id="${rowId}"]`);
        if (wrapper) wrapper.classList.add('expanded');
    }
}

function updateModelsFromWS(data) {
    // Name-only metrics cannot confirm a deployment/configuration/generation.
    // Runtime rows keep their complete observation until fetchLocalModels refreshes it.
    // Service telemetry is independent of model residency and capability evidence.
    for (const model of _modelsData) {
        if (!model.catalog_managed || !['kiron_embeddings', 'kiron_deberta'].includes(model.backend)) continue;
        const memory = data?.system?.service_memory?.[model.backend];
        model.service_memory = Array.isArray(memory?.loaded_models) && memory.loaded_models.includes(model.name)
            ? memory : null;
    }
    const nativeModels = _modelsData.filter(model => !model.runtime);
    const previousStates = new Map(nativeModels.map(model => [model.control_id, model.load_state]));
    const ollamaData = data?.system?.ollama;
    if (ollamaData && Array.isArray(ollamaData.models_loaded)) {
        const known = ollamaData.models_loaded_state !== 'unknown';
        const loadedMap = new Map(
            ollamaData.models_loaded
                .filter(item => item && typeof item.name === 'string')
                .map(item => [item.name, item])
        );
        for (const m of nativeModels) {
            if (m.backend !== 'ollama') continue;
            const runtime = loadedMap.get(m.name);
            m.loaded = known && Boolean(runtime);
            m.loading = false;
            m.load_state = known
                ? (runtime ? 'loaded' : 'unloaded')
                : 'unknown';
            m.runtime_state = m.load_state;
            m.vram_gb = runtime?.vram_gb ?? (known ? 0 : null);
            m.ram_gb = runtime
                && typeof runtime.size_gb === 'number'
                && typeof runtime.vram_gb === 'number'
                ? Math.max(0, runtime.size_gb - runtime.vram_gb)
                : (known ? 0 : null);
            m.runtime_context_length = runtime?.context_length ?? null;
            m.expires_at = runtime?.expires_at ?? null;
        }
    }

    const embedData = data?.system?.embedding;
    if (embedData) {
        const embedRunning = embedData.running === true;
        const embedLoading = embedData.loading_model || null;
        const embedActiveModels = new Set(
            Array.isArray(embedData.loaded_models)
                ? embedData.loaded_models.filter(item => typeof item === 'string')
                : []
        );
        for (const m of nativeModels) {
            if (m.backend !== 'kiron_embeddings') continue;
            m.embedding_active = embedRunning && embedActiveModels.has(m.name);
            m.embedding_loading = embedRunning && m.name === embedLoading;
            m.loaded = m.embedding_active;
            m.loading = m.embedding_loading;
            m.load_state = !embedRunning
                ? 'unknown'
                : m.loading ? 'loading' : m.loaded ? 'loaded' : 'unloaded';
            m.runtime_state = m.load_state;
        }
    }

    const debertaData = data?.system?.deberta;
    if (debertaData) {
        const serviceRunning = debertaData.running === true;
        const loadingModel = debertaData.loading_model || null;
        const loadedModels = new Set(
            Array.isArray(debertaData.loaded_models)
                ? debertaData.loaded_models.filter(
                    item => typeof item === 'string'
                )
                : []
        );
        for (const m of nativeModels) {
            if (m.backend !== 'kiron_deberta') continue;
            m.loaded = serviceRunning && loadedModels.has(m.name);
            m.loading = serviceRunning && loadingModel === m.name;
            m.load_state = !serviceRunning
                ? 'unknown'
                : m.loading ? 'loading' : m.loaded ? 'loaded' : 'unloaded';
            m.runtime_state = m.load_state;
        }
    }

    for (const model of nativeModels) model.operation_state = model.load_state;
    if (typeof currentTab === 'string' && currentTab === 'models') {
        // Refresh complete server observations for registered models too. The
        // metrics stream is only the clock; it cannot grant runtime evidence or
        // action permissions. Coalesce reads and retry failed reads at this rate.
        if (!_modelsRefreshPending && (_modelsLastRefreshAt === null
                || Date.now() - _modelsLastRefreshAt >= MODEL_REFRESH_INTERVAL_MS)) {
            void loadLocalModels(true);
            return;
        }
        if (nativeModels.some(model => model.control_id && previousStates.get(model.control_id) !== model.load_state)) {
            // Obtain new action permissions from the server when metrics change state.
            void loadLocalModels(true);
            return;
        }
        renderModelsTable();
    }
}


function initModelActionTooltips(container, tableContainer) {
    const tooltip = document.getElementById('modelsActionTooltip');
    let activeButton = null;
    const hide = () => {
        tooltip.hidden = true;
        if (activeButton) activeButton.removeAttribute('aria-describedby');
        activeButton = null;
    };
    const show = event => {
        const button = event.target.closest('[data-model-tooltip]');
        if (!button || button.disabled) return;
        hide();
        activeButton = button;
        tooltip.textContent = button.dataset.modelTooltip;
        button.setAttribute('aria-describedby', tooltip.id);
        tooltip.hidden = false;
        const rect = button.getBoundingClientRect();
        tooltip.style.left = `${Math.max(8, Math.min(rect.right - tooltip.offsetWidth, window.innerWidth - tooltip.offsetWidth - 8))}px`;
        tooltip.style.top = `${rect.top > tooltip.offsetHeight + 12 ? rect.top - tooltip.offsetHeight - 6 : rect.bottom + 6}px`;
    };
    tableContainer.addEventListener('mouseover', show);
    tableContainer.addEventListener('focusin', show);
    tableContainer.addEventListener('mouseout', event => {
        if (!activeButton?.contains(event.relatedTarget)) hide();
    });
    tableContainer.addEventListener('focusout', hide);
    tableContainer.addEventListener('click', hide, true);
    container.addEventListener('scroll', hide, true);
    container.addEventListener('keydown', event => {
        if (event.key === 'Escape') hide();
    });
}

function renderRuntimeDetail(table, model) {
    if (!model.runtime) return '';
    if (model.runtime_deployments) {
        return model.runtime_deployments.map(observed => renderRuntimeDetail(table, observed)).join('');
    }
    const capabilityLabels = {supported: 'Belegt', unsupported: 'Nicht unterstützt', unverified: 'Nicht belegt'};
    const caps = Object.entries(model.capabilities || {}).map(([name, value]) => {
        const dates = (value.evidence || []).map(item => item.observed_at).join(', ');
        const evidence = (value.evidence || []).map(item => item.id).join(', ');
        return modelDetailItem(table, name, `${capabilityLabels[value.status] || 'Nicht belegt'}${dates ? ' · ' + dates : ''}${evidence ? ' · Beleg ' + evidence : ''}`, true);
    }).join('');
    const checks = (model.verification?.checks || []).map(check => {
        const expected = Object.entries(check.expected).map(([key, constraint]) => {
            const label = {device: 'Gerät', context_tokens: 'Kontext'}[key] || key;
            const values = constraint.allowed_values?.join(', ') || `${constraint.minimum ?? 'offen'} bis ${constraint.maximum ?? 'offen'}`;
            return `${label}: ${values}`;
        }).join(' · ');
        return modelDetailItem(table, check.capability,
            `${verificationLabel({verification: check})}${expected ? ' · Belegt für ' + expected : ''}`);
    }).join('');
    const profile = model.resource_profile;
    const params = profile ? Object.entries(profile).map(([name, value]) => modelDetailItem(table, name, String(value), true)).join('')
        : modelDetailItem(table, 'Ladeprofil', 'Kein gemessener Ressourcenplan; gemeinsames kaltes Laden ist gesperrt.', true);
    return '<section class="models-detail-section"><h3>Gemeinsame Laufzeit und Fähigkeitsbelege</h3><dl>'
        + modelDetailItem(table, 'Provider', model.runtime_provider, true)
        + modelDetailItem(table, 'Profile', modelListLabel(model.profile_ids), true)
        + modelDetailItem(table, 'Beobachtet', model.observed_at || 'Unbekannt', true)
        + modelDetailItem(table, 'Diagnose', model.diagnostic || '—', true)
        + modelDetailItem(table, 'Diagnosecode', model.error_code || '—', true)
        + modelDetailItem(table, 'API-Modelladressen', modelListLabel(model.api_model_ids), true)
        + modelDetailItem(table, 'Vision-Projektor registriert', model.projector ? 'Ja' : 'Nein', true)
        + caps + '</dl><h3>Abgleich mit der aktuellen Konfiguration</h3><dl>' + checks
        + '</dl><p>Abweichende oder fehlende Fähigkeitsnachweise ändern den Betriebszustand nicht. Für garantierte API-Funktionen gelten weiterhin deren Nachweisprüfungen.</p>'
        + '<h3>Ressourcenplan für gemeinsames Laden</h3><dl>' + params + '</dl></section>';
}

async function performModelAction(model, control, button) {
    if (control.confirmation && !confirm(`${model.name}: ${control.confirmation}`)) return;
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    try {
        const response = await fetch('/api/models/control/action', {
            method: 'POST', headers: {'Content-Type': 'application/json', 'X-Kiron-Action': 'models'},
            body: JSON.stringify({model: model.control_id, revision: model.control_revision, action: control.id}),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || 'Der Modelldienst hat die Aktion nicht bestätigt.');
        showNotification('Modellaktion abgeschlossen', 'success');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        await loadLocalModels();
    }
}

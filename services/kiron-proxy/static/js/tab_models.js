/**
 * KIron Models: one Catalog and local-registration view.
 */

let _modelsData = [];
let _registrationCandidates = [];
let registrationDialogReturnFocus = null;
let _modelsTable = null;
let _modelRowIdSequence = 0;
const _modelRowIds = new Map();

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
                        <option value="huggingface">Hugging Face (lokales Verzeichnis)</option>
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
    const huggingface = provider.value === 'huggingface';
    loader.disabled = !huggingface;
    loaderField.hidden = !huggingface;
    referenceLabel.textContent = huggingface
        ? 'Vorhandene Hugging-Face-Modellwurzel'
        : 'Vorhandenes Ollama-Modell';
}

function renderRegistrationCandidates() {
    const provider = document.getElementById('modelRegistrationProvider');
    const reference = document.getElementById('modelRegistrationReference');
    const status = document.getElementById('modelRegistrationCandidateStatus');
    const submit = document.getElementById('modelRegistrationSubmit');
    if (!provider || !reference || !status || !submit) return;

    const matching = _registrationCandidates.filter(
        candidate => candidate.provider === provider.value
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
        option.value = candidate.reference;
        option.textContent = candidate.provider === 'huggingface'
            ? `${candidate.display_name} — ${candidate.reference}`
            : candidate.display_name;
        reference.appendChild(option);
    }
    reference.disabled = matching.length === 0;
    submit.disabled = matching.length === 0;
    status.textContent = matching.length
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
                && (candidate.provider === 'ollama'
                    || candidate.provider === 'huggingface')
                && typeof candidate.reference === 'string'
                && typeof candidate.display_name === 'string'
            ))
            : [];
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
        provider: providerNode.value,
        reference: referenceNode.value,
    };
    if (providerNode.value === 'huggingface') {
        payload.loader = loaderNode.value;
    }

    submitButton.disabled = true;
    errorNode.textContent = '';
    try {
        const response = await fetch('/api/models/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
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
    const response = await fetch('/api/models/local');
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || data?.error?.message || response.statusText);
    }
    _modelsData = Array.isArray(data.models) ? data.models : [];
}

async function loadLocalModels() {
    const tableContainer = document.getElementById('modelsTableContainer');
    if (!tableContainer) return;
    try {
        await fetchLocalModels();
        renderModelsTable();
    } catch (error) {
        const message = document.createElement('div');
        message.className = 'no-data';
        message.textContent = `Modelle konnten nicht geladen werden: ${error.message}`;
        tableContainer.replaceChildren(message);
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
    if (model.load_state === 'unknown') return 'Load-Status unbekannt';
    if (model.embedding_loading || model.loading) return 'Wird geladen';
    if (model.loaded && model.managed_service) return 'Geladen (Service)';
    if (model.configured && !model.installed) {
        return 'Konfiguriert, nicht installiert';
    }
    if (model.loaded) return 'Geladen';
    return model.installed
        ? 'Installiert, nicht geladen'
        : 'Nicht installiert';
}

function memoryLabel(model) {
    if (model.load_state === 'unknown') return 'Unbekannt';
    const parts = [];
    if (typeof model.vram_gb === 'number' && Number.isFinite(model.vram_gb)) {
        parts.push(`VRAM ${model.vram_gb.toFixed(2)} GB`);
    }
    if (typeof model.ram_gb === 'number' && Number.isFinite(model.ram_gb)) {
        parts.push(`RAM ${model.ram_gb.toFixed(2)} GB`);
    }
    return parts.length ? parts.join(' / ') : '—';
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
    if (model.locally_registered) origins.push('Lokal registriert');
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
    const badges = [];
    if (model.catalog_managed) {
        badges.push(modelBadgeHtml(table, 'Catalog', 'models-badge-catalog'));
    }
    if (model.locally_registered) {
        badges.push(
            modelBadgeHtml(table, 'Lokal registriert', 'models-badge-local')
        );
    }
    return badges.length ? badges.join(' ') : '—';
}

function modelActionButtonHtml(table, row, action, label, className, title, disabled = false) {
    const classes = `action-btn ${className || ''}`.trim();
    const disabledAttribute = disabled ? ' disabled' : '';
    const actionAttribute = action
        ? ` data-model-action="${table.escapeHtml(action)}"`
        : '';
    return `<button type="button" class="${table.escapeHtml(classes)}"`
        + `${actionAttribute} data-model-row="${table.escapeHtml(row.id)}"`
        + ` title="${table.escapeHtml(title)}"${disabledAttribute}>`
        + `${table.escapeHtml(label)}</button>`;
}

function renderModelActions(table, row) {
    const model = row._model;
    const actions = [];

    if (
        model.backend === 'kiron_embeddings'
        && model.installed
        && model.embedding_loading
    ) {
        actions.push(modelActionButtonHtml(
            table,
            row,
            '',
            model.embedding_kind === 'colbert' ? 'ColBERT lädt…' : 'Lädt…',
            'models-btn-embed',
            'Embedding-Ladevorgang läuft',
            true
        ));
    } else if (
        model.backend === 'kiron_embeddings'
        && model.installed
        && !model.embedding_active
    ) {
        actions.push(modelActionButtonHtml(
            table,
            row,
            'load-embedding',
            model.embedding_kind === 'colbert' ? 'ColBERT laden' : 'Embedding',
            'models-btn-embed',
            'Im Embedding-Service laden'
        ));
    } else if (model.embedding_kind === 'colbert') {
        actions.push(modelActionButtonHtml(
            table,
            row,
            'warmup-colbert',
            'Warmup',
            'models-btn-embed',
            'ColBERT-Warmup ausführen'
        ));
    }

    if (model.backend === 'ollama' && model.installed) {
        if (model.loaded) {
            actions.push(modelActionButtonHtml(
                table, row, 'unload', 'Entladen', '',
                'Aus dem Speicher entladen'
            ));
        } else {
            actions.push(modelActionButtonHtml(
                table, row, 'load-gpu', 'Laden', 'primary', 'Mit GPU laden'
            ));
            actions.push(modelActionButtonHtml(
                table, row, 'load-cpu', 'CPU', '', 'Ohne GPU laden'
            ));
        }
        actions.push(modelActionButtonHtml(
            table, row, 'delete', 'Löschen', 'danger',
            'Lokales Ollama-Modell löschen'
        ));
    }

    return actions.length
        ? `<span class="models-actions">${actions.join('')}</span>`
        : '—';
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
        : model.loaded
            ? tokenCountLabel(model.runtime_context_length)
            : model.load_state === 'unknown'
                ? 'Load-Status unbekannt'
                : 'Nicht geladen';
    const catalogContext = model.catalog_managed
        ? tokenCountLabel(model.catalog_context_length, 'Nicht deklariert')
        : 'Nicht im Catalog verwaltet';
    const memoryVram = typeof model.vram_gb === 'number'
        ? `${model.vram_gb.toFixed(2)} GB`
        : '—';
    const memoryRam = typeof model.ram_gb === 'number'
        ? `${model.ram_gb.toFixed(2)} GB`
        : '—';

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
        + modelDetailItem(table, 'Status', statusLabel(model))
        + modelDetailItem(table, 'Konfiguriert', yesNoLabel(model.configured))
        + modelDetailItem(table, 'Installiert', yesNoLabel(model.installed))
        + modelDetailItem(table, 'Typ', modelTypeLabel(model.model_type))
        + modelDetailItem(table, 'Familie', model.family)
        + modelDetailItem(table, 'Format', model.format)
        + modelDetailItem(table, 'Parameter', model.parameter_size)
        + modelDetailItem(table, 'Quantisierung', model.quantization_level)
        + modelDetailItem(table, 'Modellgröße', sizeLabel(model.size_gb))
        + modelDetailItem(table, 'VRAM', memoryVram)
        + modelDetailItem(table, 'RAM', memoryRam)
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
        gridTemplate: '30px minmax(180px, 1.6fr) 140px 90px '
            + 'minmax(100px, 1fr) 95px 85px 90px minmax(155px, 1.1fr) '
            + 'minmax(170px, 1.2fr) minmax(205px, auto)',
        defaultSort: {field: 'name', direction: 'asc'},
        columns: [
            {field: 'name', label: 'Modell', sortable: true, align: 'left', renderer: 'modelName'},
            {field: 'source', label: 'Herkunft', sortable: false, align: 'left', renderer: 'modelOrigin'},
            {field: 'model_type', label: 'Typ', sortable: true, align: 'left', renderer: 'modelType'},
            {field: 'family', label: 'Familie', sortable: true, align: 'left'},
            {field: 'parameter_size', label: 'Parameter', sortable: true, align: 'left'},
            {field: 'quantization_level', label: 'Quant.', sortable: true, align: 'left'},
            {field: 'size_gb', label: 'Größe', sortable: true, align: 'right', renderer: 'modelSize'},
            {field: 'load_state', label: 'Status', sortable: true, align: 'left', renderer: 'modelStatus'},
            {field: 'vram_gb', label: 'Speicher', sortable: true, align: 'left', renderer: 'modelMemory'},
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
    table.renderers.modelStatus = (_value, _column, row) => modelBadgeHtml(
        table,
        statusLabel(row._model),
        row._model.loaded ? 'models-badge-loaded' : 'models-badge-unloaded'
    );
    table.renderers.modelMemory = (_value, _column, row) => (
        table.escapeHtml(memoryLabel(row._model))
    );
    table.renderers.modelActions = (_value, _column, row) => (
        renderModelActions(table, row)
    );
    table.renderDetail = row => renderModelDetail(table, row._model);
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
    const name = String(model.name || '');
    const action = button.dataset.modelAction;

    if (action === 'load-embedding') {
        button.disabled = true;
        button.textContent = 'Lädt…';
        void loadEmbeddingModel(name);
    } else if (action === 'warmup-colbert') {
        void warmupColbertModel(name, button);
    } else if (action === 'load-gpu') {
        void loadModel(name, true, button);
    } else if (action === 'load-cpu') {
        void loadModel(name, false, button);
    } else if (action === 'unload') {
        void unloadModel(name, button);
    } else if (action === 'delete') {
        void deleteModel(name, Boolean(model.loaded));
    }
}

function renderModelsTable() {
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
        id: modelRowId(model),
        _model: model,
    }));
    _modelsTable.applyFiltersAndRender();

    for (const rowId of expanded) {
        const wrapper = tableContainer.querySelector(`[data-row-id="${rowId}"]`);
        if (wrapper) wrapper.classList.add('expanded');
    }
}

async function responseData(response) {
    return response.json().catch(() => ({}));
}

async function loadModel(name, gpu, button) {
    button.disabled = true;
    button.textContent = 'Lädt…';
    try {
        const response = await fetch('/api/models/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, gpu}),
        });
        const data = await responseData(response);
        if (response.ok) {
            showNotification(
                gpu ? `${name} geladen` : `${name} auf CPU geladen`,
                'success'
            );
        } else {
            showNotification(`Fehler: ${data.error || response.statusText}`, 'error');
        }
    } catch (error) {
        showNotification(`Fehler beim Laden: ${error.message}`, 'error');
    }
    await loadLocalModels();
}

async function unloadModel(name, button) {
    button.disabled = true;
    button.textContent = 'Entlädt…';
    try {
        const response = await fetch('/api/models/unload', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name}),
        });
        const data = await responseData(response);
        if (response.ok) {
            showNotification(`${name} entladen`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || response.statusText}`, 'error');
        }
    } catch (error) {
        showNotification(`Fehler beim Entladen: ${error.message}`, 'error');
    }
    await loadLocalModels();
}

async function deleteModel(name, isLoaded) {
    const question = isLoaded
        ? `"${name}" ist geladen. Trotzdem löschen?`
        : `"${name}" wirklich löschen?`;
    if (!confirm(question)) return;
    await doDeleteModel(name, isLoaded);
}

async function doDeleteModel(name, force) {
    try {
        const response = await fetch('/api/models/delete', {
            method: 'DELETE',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, force}),
        });
        const data = await responseData(response);
        if (response.ok) {
            showNotification(`${name} gelöscht`, 'success');
        } else if (response.status === 409 && !force) {
            if (confirm(`${data.error}. Trotzdem löschen?`)) {
                await doDeleteModel(name, true);
                return;
            }
        } else {
            showNotification(`Fehler: ${data.error || response.statusText}`, 'error');
        }
    } catch (error) {
        showNotification(`Fehler beim Löschen: ${error.message}`, 'error');
    }
    await loadLocalModels();
}

async function loadEmbeddingModel(name) {
    try {
        const response = await fetch('/api/embedding/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: name}),
        });
        const data = await responseData(response);
        if (response.ok) {
            showNotification(`Embedding-Modell ${name} geladen`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || response.statusText}`, 'error');
        }
    } catch (error) {
        showNotification(`Fehler beim Laden: ${error.message}`, 'error');
    }
    await loadLocalModels();
}

async function warmupColbertModel(name, button) {
    button.disabled = true;
    button.textContent = 'Warmup…';
    try {
        const response = await fetch('/api/embedding/colbert/warmup', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model: name}),
        });
        const data = await responseData(response);
        if (response.ok) {
            showNotification(`ColBERT-Warmup ${name} erfolgreich`, 'success');
        } else {
            showNotification(`Fehler: ${data.error || response.statusText}`, 'error');
        }
    } catch (error) {
        showNotification(`Fehler beim ColBERT-Warmup: ${error.message}`, 'error');
    }
    await loadLocalModels();
}

function updateModelsFromWS(data) {
    const ollamaData = data?.system?.ollama;
    if (ollamaData && Array.isArray(ollamaData.models_loaded)) {
        const known = ollamaData.models_loaded_state !== 'unknown';
        const loadedMap = new Map(
            ollamaData.models_loaded
                .filter(item => item && typeof item.name === 'string')
                .map(item => [item.name, item])
        );
        for (const m of _modelsData) {
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
        for (const m of _modelsData) {
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
        for (const m of _modelsData) {
            if (m.backend !== 'kiron_deberta') continue;
            m.loaded = serviceRunning && loadedModels.has(m.name);
            m.loading = serviceRunning && loadingModel === m.name;
            m.load_state = !serviceRunning
                ? 'unknown'
                : m.loading ? 'loading' : m.loaded ? 'loaded' : 'unloaded';
            m.runtime_state = m.load_state;
        }
    }

    if (typeof currentTab === 'string' && currentTab === 'models') {
        renderModelsTable();
    }
}

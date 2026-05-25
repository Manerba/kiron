/**
 * Ollama Monitor - API Keys Tab
 * Verwaltung von API-Keys fuer die OpenAI-kompatible API.
 */

// ========================================
// State
// ========================================

let _apikeysLoaded = false;

// ========================================
// Init
// ========================================

function initApiKeysTab() {
    _apikeysLoaded = false;
    const container = document.getElementById('contentContainer');
    if (!container) return;

    container.innerHTML = `
        <div class="apikeys-page">
            <div class="apikeys-header">
                <h2>API Keys</h2>
                <button class="apikeys-create-btn" onclick="showCreateKeyModal()">Neuen API-Key erstellen</button>
            </div>
            <table class="apikeys-table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Key</th>
                        <th>Erstellt</th>
                        <th>Zuletzt verwendet</th>
                        <th>Status</th>
                        <th>Aktionen</th>
                    </tr>
                </thead>
                <tbody id="apikeysTableBody">
                    <tr><td colspan="6" class="apikeys-empty">Lade...</td></tr>
                </tbody>
            </table>
            <div class="apikeys-info">
                <strong>OpenAI-kompatible API</strong> auf Port <code>11440</code><br>
                Beispiel: <code>curl http://localhost:11440/v1/chat/completions -H "Authorization: Bearer sk-..." -H "Content-Type: application/json" -d '{"model":"qwen3:8b","messages":[{"role":"user","content":"Hi"}]}'</code>
            </div>
        </div>
    `;

    loadApiKeys();
}

// ========================================
// Load Keys
// ========================================

async function loadApiKeys() {
    try {
        const resp = await fetch('/api/apikeys');
        const data = await resp.json();
        renderApiKeysTable(data.keys || []);
        _apikeysLoaded = true;
    } catch (e) {
        const tbody = document.getElementById('apikeysTableBody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="6" class="apikeys-empty">Fehler beim Laden</td></tr>';
        }
    }
}

// ========================================
// Render Table
// ========================================

function renderApiKeysTable(keys) {
    const tbody = document.getElementById('apikeysTableBody');
    if (!tbody) return;

    if (keys.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="apikeys-empty">Keine API-Keys vorhanden</td></tr>';
        return;
    }

    tbody.innerHTML = keys.map(k => {
        const created = new Date(k.created_at * 1000).toLocaleString('de-DE');
        const lastUsed = k.last_used_at ? new Date(k.last_used_at * 1000).toLocaleString('de-DE') : 'Nie';
        const statusBadge = k.is_active
            ? '<span class="apikeys-badge-active">Aktiv</span>'
            : '<span class="apikeys-badge-inactive">Inaktiv</span>';
        const toggleLabel = k.is_active ? 'Deaktivieren' : 'Aktivieren';

        // Inline-onclick fuer JS-Args via JSON.stringify (vollwertige JS-Literal-
        // Escapes inkl. Apostroph + Backslash) und anschliessend escapeHtml fuer
        // den HTML-Attribute-Kontext. escapeHtml allein wuerde Apostrophe nicht
        // escapen und eroeffnete einen Stored-XSS-Vektor ueber den Key-Namen.
        const idJs = escapeHtml(JSON.stringify(k.id));
        const nameJs = escapeHtml(JSON.stringify(k.name));
        return `<tr>
            <td>${escapeHtml(k.name)}</td>
            <td><code>${escapeHtml(k.key_prefix)}</code></td>
            <td>${created}</td>
            <td>${lastUsed}</td>
            <td>${statusBadge}</td>
            <td class="apikeys-actions">
                <button onclick="toggleApiKey(${idJs}, ${!k.is_active})">${toggleLabel}</button>
                <button class="btn-danger" onclick="deleteApiKey(${idJs}, ${nameJs})">L\u00f6schen</button>
            </td>
        </tr>`;
    }).join('');
}

// ========================================
// Create Key Modal
// ========================================

function showCreateKeyModal() {
    // Remove existing modal
    const existing = document.querySelector('.apikeys-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.className = 'apikeys-modal';
    modal.innerHTML = `
        <div class="apikeys-modal-content">
            <h3>Neuen API-Key erstellen</h3>
            <div id="apiKeyCreateForm">
                <label for="apiKeyNameInput">Name</label>
                <input type="text" id="apiKeyNameInput" placeholder="z.B. StockPicker, Test-Client..." autofocus>
                <div class="apikeys-modal-buttons">
                    <button class="btn-cancel" onclick="closeCreateKeyModal()">Abbrechen</button>
                    <button class="btn-create" onclick="createApiKey()">Erstellen</button>
                </div>
            </div>
            <div id="apiKeyCreatedResult" style="display: none;"></div>
        </div>
    `;

    document.body.appendChild(modal);

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeCreateKeyModal();
    });

    // Focus input
    setTimeout(() => {
        const input = document.getElementById('apiKeyNameInput');
        if (input) input.focus();
    }, 50);

    // Enter key
    const input = modal.querySelector('#apiKeyNameInput');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') createApiKey();
        });
    }
}

function closeCreateKeyModal() {
    const modal = document.querySelector('.apikeys-modal');
    if (modal) modal.remove();
}

async function createApiKey() {
    const input = document.getElementById('apiKeyNameInput');
    if (!input) return;

    const name = input.value.trim();
    if (!name) {
        input.style.borderColor = 'var(--color-bearish)';
        return;
    }

    try {
        const resp = await fetch('/api/apikeys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        const data = await resp.json();

        if (!resp.ok) {
            alert(data.error || 'Fehler beim Erstellen');
            return;
        }

        // Show the key
        const form = document.getElementById('apiKeyCreateForm');
        const result = document.getElementById('apiKeyCreatedResult');
        if (form) form.style.display = 'none';
        if (result) {
            result.style.display = 'block';
            result.innerHTML = `
                <div class="apikeys-key-display">
                    <div class="key-warning">Dieser Key wird nur einmal angezeigt. Bitte jetzt kopieren!</div>
                    <code id="apiKeyPlaintext">${escapeHtml(data.key)}</code>
                    <button class="btn-copy" onclick="copyApiKey()">Kopieren</button>
                </div>
                <div class="apikeys-modal-buttons" style="margin-top: 16px;">
                    <button class="btn-cancel" onclick="closeCreateKeyModal()">Schlie\u00dfen</button>
                </div>
            `;
        }

        // Reload table
        loadApiKeys();

    } catch (e) {
        alert('Netzwerkfehler: ' + e.message);
    }
}

function copyApiKey() {
    const code = document.getElementById('apiKeyPlaintext');
    if (!code) return;
    const text = code.textContent;

    // execCommand-Fallback (funktioniert auch ohne HTTPS)
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        const btn = code.parentElement.querySelector('.btn-copy');
        if (btn) {
            btn.textContent = 'Kopiert!';
            setTimeout(() => { btn.textContent = 'Kopieren'; }, 2000);
        }
    } catch (e) {
        // Letzter Fallback: Text im Code-Block selektieren
        const range = document.createRange();
        range.selectNodeContents(code);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    }
    document.body.removeChild(textarea);
}

// ========================================
// Toggle / Delete
// ========================================

async function toggleApiKey(keyId, newActive) {
    try {
        const resp = await fetch(`/api/apikeys/${keyId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_active: newActive }),
        });
        if (resp.ok) {
            loadApiKeys();
        } else {
            const data = await resp.json();
            alert(data.error || 'Fehler');
        }
    } catch (e) {
        alert('Netzwerkfehler: ' + e.message);
    }
}

async function deleteApiKey(keyId, keyName) {
    if (!confirm(`API-Key "${keyName}" wirklich l\u00f6schen?`)) return;

    try {
        const resp = await fetch(`/api/apikeys/${keyId}`, { method: 'DELETE' });
        if (resp.ok) {
            loadApiKeys();
        } else {
            const data = await resp.json();
            alert(data.error || 'Fehler');
        }
    } catch (e) {
        alert('Netzwerkfehler: ' + e.message);
    }
}

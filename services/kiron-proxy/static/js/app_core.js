/**
 * Kiron - App Core
 * Zentrales Routing, Theme, Settings, WebSocket-Manager.
 */

// ========================================
// Globaler State
// ========================================

let currentTab = 'dashboard';
let currentSubTab = null;
const domCache = {};

const tabMap = {
    'dashboard': 'dashboard',
    'requests': 'requests',
    'system': 'system',
    'history': 'history',
    'models': 'models',
    'apikeys': 'apikeys',
    'selftest': 'selftest',
};

// Sub-Tabs pro Haupt-Tab (analog kiara): label + globaler init-Funktionsname.
const SUBTABS = {
    'models': {
        'installed': { label: 'Installiert', init: 'initModelsInstalled' },
        'available': { label: 'Verfuegbar', init: 'initModelsAvailable' },
    },
};

// ========================================
// Theme (via SharedUi.theme — ui_core)
// ========================================

function initTheme() {
    // SharedUi.configure muss vor load() laufen.
    if (window.SharedUi && typeof window.SharedUi.configure === 'function') {
        window.SharedUi.configure({ productName: 'Kiron', themeStorageKey: 'ui_theme' });
    }
    const applied = (window.SharedUi && window.SharedUi.theme)
        ? window.SharedUi.theme.load()
        : (localStorage.getItem('ui_theme') || 'light');
    if (!window.SharedUi || !window.SharedUi.theme) {
        document.documentElement.setAttribute('data-theme', applied);
    }
    updateThemePills(applied);
}

function setTheme(theme) {
    if (window.SharedUi && window.SharedUi.theme && window.SharedUi.theme.set(theme)) {
        updateThemePills(theme);
    }
}

function updateThemePills(theme) {
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-theme') === theme);
    });
}

// ========================================
// Kompakte Ansicht (Density, analog kiara)
// ========================================

function initDensity() {
    const raw = localStorage.getItem('kiara-density');
    const savedDensity = raw === 'compact' ? 'compact' : 'normal';
    document.documentElement.setAttribute('data-density', savedDensity);
    updateDensityButton(savedDensity);
}

function toggleDensity() {
    const current = document.documentElement.getAttribute('data-density');
    const newDensity = current === 'compact' ? 'normal' : 'compact';
    document.documentElement.setAttribute('data-density', newDensity);
    localStorage.setItem('kiara-density', newDensity);
    updateDensityButton(newDensity);
}

function updateDensityButton(density) {
    const densityStatus = document.getElementById('densityStatus');
    if (densityStatus) densityStatus.textContent = density === 'compact' ? 'An' : 'Aus';
}

// ========================================
// Settings Dropdown
// ========================================

function toggleSettingsDropdown() {
    const dropdown = document.getElementById('settingsDropdown');
    if (!dropdown) return;

    if (dropdown.style.display === 'none' || !dropdown.style.display) {
        dropdown.style.display = 'block';
        // Klick ausserhalb schliesst Dropdown
        setTimeout(() => {
            document.addEventListener('click', closeSettingsOnClickOutside);
        }, 0);
    } else {
        dropdown.style.display = 'none';
        document.removeEventListener('click', closeSettingsOnClickOutside);
    }
}

function closeSettingsOnClickOutside(event) {
    const dropdown = document.getElementById('settingsDropdown');
    const wrapper = document.querySelector('.settings-dropdown-wrapper');
    if (wrapper && !wrapper.contains(event.target)) {
        if (dropdown) dropdown.style.display = 'none';
        document.removeEventListener('click', closeSettingsOnClickOutside);
    }
}

async function restartDashboard() {
    if (!confirm('Dashboard wirklich neu starten?')) return;
    try {
        await fetch('/api/dashboard/restart', { method: 'POST' });
    } catch (e) { /* Verbindung bricht ab */ }
    setTimeout(() => location.reload(), 3000);
}

// ========================================
// Routing
// ========================================

function parseRoute() {
    const path = window.location.pathname.replace(/^\/+|\/+$/g, '');
    const parts = path.split('/');
    const tab = tabMap[parts[0]] ? parts[0] : 'dashboard';
    const subtab = parts[1] || null;
    return { tab, subtab };
}

function buildPath(tab, subtab) {
    let path = '/' + (tab || 'dashboard');
    if (subtab && SUBTABS[tab] && SUBTABS[tab][subtab]) {
        path += '/' + subtab;
    }
    return path;
}

function navigateTo(tab, subtab) {
    if (!tabMap[tab]) tab = 'dashboard';
    const path = buildPath(tab, subtab);
    history.pushState({ tab, subtab: subtab || null }, '', path);
    switchTab(tab, subtab);
}

// ========================================
// Tab-Wechsel
// ========================================

function saveTabDOM(tab) {
    const container = document.getElementById('contentContainer');
    if (container && container.innerHTML) {
        domCache[tab] = container.innerHTML;
    }
}

function restoreTabDOM(tab) {
    const container = document.getElementById('contentContainer');
    if (container && domCache[tab]) {
        container.innerHTML = domCache[tab];
        return true;
    }
    return false;
}

function invalidateTabCache(tab) {
    if (tab) {
        delete domCache[tab];
    } else {
        Object.keys(domCache).forEach(key => delete domCache[key]);
    }
}

function switchTab(tab, subtab) {
    if (!tabMap[tab]) tab = 'dashboard';

    // History-Charts aufraeumen wenn weg navigiert wird
    if (currentTab === 'history' && tab !== 'history') {
        if (typeof destroyHistoryCharts === 'function') {
            destroyHistoryCharts();
        }
    }

    // Aktuellen Tab DOM speichern
    if (currentTab && currentTab !== tab) {
        saveTabDOM(currentTab);
    }

    currentTab = tab;
    const filterBar = document.getElementById('filterBar');

    // Haupt-Tab-Links aktiv markieren (nur .main-tabs, nicht die Sub-Tabs,
    // die seit dem kiara-Sub-Tab-System ebenfalls .tab nutzen)
    document.querySelectorAll('.main-tabs .tab').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-tab') === tab);
    });

    // Container leeren
    const container = document.getElementById('contentContainer');
    if (container) {
        container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    }

    // Hat der Tab Sub-Tabs? Dann Leiste rendern und an den aktiven Sub-Tab delegieren.
    const activeSub = renderSubTabsBar(tab, subtab);
    if (activeSub) {
        if (filterBar) filterBar.style.display = 'none';
        switchSubTab(activeSub);
        return;
    }

    // Tab initialisieren
    if (tab === 'dashboard') {
        if (filterBar) filterBar.style.display = 'none';
        if (typeof initDashboardTab === 'function') {
            initDashboardTab();
        }
    } else if (tab === 'requests') {
        // filterBar wird von initRequestsTab() selbst gesteuert
        if (typeof initRequestsTab === 'function') {
            initRequestsTab();
        }
    } else if (tab === 'system') {
        if (filterBar) filterBar.style.display = 'none';
        if (typeof initSystemTab === 'function') {
            initSystemTab();
        }
    } else if (tab === 'history') {
        if (filterBar) filterBar.style.display = 'none';
        if (typeof initHistoryTab === 'function') {
            initHistoryTab();
        }
    } else if (tab === 'apikeys') {
        if (filterBar) filterBar.style.display = 'none';
        if (typeof initApiKeysTab === 'function') {
            initApiKeysTab();
        }
    } else if (tab === 'selftest') {
        if (filterBar) filterBar.style.display = 'none';
        if (typeof initSelftestTab === 'function') {
            initSelftestTab();
        }
    }
}

// ========================================
// Sub-Tabs (analog kiara)
// ========================================

function renderSubTabsBar(tab, activeSubtab) {
    const subTabsNav = document.getElementById('subTabs');
    if (!subTabsNav) return null;

    const mainTabs = document.querySelector('.main-tabs');
    const cfg = SUBTABS[tab];
    if (!cfg) {
        subTabsNav.classList.add('is-hidden');
        subTabsNav.innerHTML = '';
        currentSubTab = null;
        if (mainTabs) mainTabs.classList.remove('has-subtabs');
        return null;
    }

    const keys = Object.keys(cfg);
    const active = (activeSubtab && cfg[activeSubtab]) ? activeSubtab : keys[0];

    subTabsNav.innerHTML = keys.map(key =>
        `<a class="tab${key === active ? ' active' : ''}" href="${buildPath(tab, key)}" data-subtab="${key}">${cfg[key].label}</a>`
    ).join('');
    subTabsNav.classList.remove('is-hidden');
    if (mainTabs) mainTabs.classList.add('has-subtabs');

    subTabsNav.querySelectorAll('.tab').forEach(st => {
        st.addEventListener('click', (e) => {
            e.preventDefault();
            switchSubTab(st.dataset.subtab);
        });
    });

    return active;
}

function switchSubTab(subtab) {
    const cfg = SUBTABS[currentTab];
    if (!cfg || !cfg[subtab]) return;

    currentSubTab = subtab;

    document.querySelectorAll('#subTabs .tab').forEach(t => {
        t.classList.toggle('active', t.dataset.subtab === subtab);
    });

    history.replaceState({ tab: currentTab, subtab }, '', buildPath(currentTab, subtab));

    const container = document.getElementById('contentContainer');
    if (container) {
        container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
    }

    const initFn = window[cfg[subtab].init];
    if (typeof initFn === 'function') {
        initFn();
    }
}

// ========================================
// WebSocket-Manager
// ========================================

let ws = null;
let wsReconnectTimer = null;

function initWebSocket() {
    // Bestehende Verbindung schliessen
    if (ws) {
        try { ws.close(); } catch (e) { /* ignorieren */ }
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket verbunden');
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleWSMessage(msg);
        } catch (e) {
            console.error('WebSocket Nachricht konnte nicht geparst werden:', e);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket getrennt, reconnect in 3s...');
        wsReconnectTimer = setTimeout(initWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket Fehler:', err);
    };
}

function handleWSMessage(msg) {
    // Request-Events an Requests-Tab weiterleiten (immer, auch wenn anderer Tab aktiv)
    if (msg.type === 'new_request') {
        if (typeof handleNewRequest === 'function') {
            handleNewRequest(msg.data);
        }
    }
    if (msg.type === 'update_request') {
        if (typeof handleUpdateRequest === 'function') {
            handleUpdateRequest(msg.data);
        }
    }

    // Metriken an aktiven Tab weiterleiten
    if (msg.type === 'metrics') {
        if (currentTab === 'dashboard' && typeof updateDashboardFromWS === 'function') {
            updateDashboardFromWS(msg.data);
        }
        if (currentTab === 'system' && typeof updateSystemFromWS === 'function') {
            updateSystemFromWS(msg.data);
        }
        if (currentTab === 'models' && typeof updateModelsFromWS === 'function') {
            updateModelsFromWS(msg.data);
        }
    }
}

// ========================================
// Notifications
// ========================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ========================================
// Hilfsfunktionen
// ========================================

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function debounce(fn, delay) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function formatPercent(value) {
    if (value === null || value === undefined) return '-';
    const num = parseFloat(value);
    if (isNaN(num)) return '-';
    return num.toFixed(1) + '%';
}

// ========================================
// Initialisierung
// ========================================

function initTabs() {
    const route = parseRoute();
    switchTab(route.tab, route.subtab);
    history.replaceState(
        { tab: route.tab, subtab: route.subtab || null },
        '',
        buildPath(route.tab, route.subtab)
    );

    // WebSocket starten
    initWebSocket();
}

// Event-Listener
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initDensity();

    // Settings-Dropdown (Zahnrad)
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', toggleSettingsDropdown);
    }

    // Theme-Pills (Light/Dark/Kontrast)
    document.querySelectorAll('.theme-option[data-theme]').forEach(btn => {
        btn.addEventListener('click', () => setTheme(btn.getAttribute('data-theme')));
    });

    // Restart-Button
    const restartBtn = document.getElementById('restartBtn');
    if (restartBtn) {
        restartBtn.addEventListener('click', restartDashboard);
    }

    // Kompakte Ansicht (Density)
    const densityToggle = document.getElementById('densityToggle');
    if (densityToggle) {
        densityToggle.addEventListener('click', toggleDensity);
    }

    // History-Charts bei Theme-Wechsel neu zeichnen.
    document.addEventListener('themeChanged', () => {
        if (typeof refreshHistoryTheme === 'function') {
            refreshHistoryTheme();
        }
    });

    // Tab-Klicks abfangen (SPA-Navigation)
    document.querySelectorAll('.tab[data-tab]').forEach(el => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = el.getAttribute('data-tab');
            navigateTo(tab);
        });
    });

    // Browser-Navigation (Zurueck/Vor)
    window.addEventListener('popstate', (e) => {
        const route = parseRoute();
        const tab = e.state?.tab || route.tab;
        const subtab = (e.state && 'subtab' in e.state) ? e.state.subtab : route.subtab;
        switchTab(tab, subtab);
    });

    // Tabs initialisieren
    initTabs();
});

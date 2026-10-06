/**
 * Verlauf-Tab - Historische Metriken mit Zeitreihen-Charts.
 * Zeigt 4 Charts (CPU, RAM, GPU, Disk) mit Zeitraum-Selektor.
 */

// State
let historyCharts = [];
let historyRefreshTimer = null;
let historyActiveRange = '1m';
let historyLoadGeneration = 0;

// Eigenstaendiger 1m-Livepfad (kein History-API-/SQLite-Zugriff).
let historyLiveSocket = null;
let historyLiveReconnectTimer = null;
let historyLiveRenderFrame = null;
let historyLiveGeneration = 0;
let historyLiveSamples = [];
let historyLiveConnectionState = 'disconnected';
let historyLiveBackendStatus = null;
let historyLiveIntervalMs = 250;

const HISTORY_LIVE_PROTOCOL_VERSION = 1;
const HISTORY_LIVE_RECONNECT_MS = 1000;

const HISTORY_RANGES = {
    '1m':  { seconds: 60,       label: '1m',  refreshMs: 0, live: true, maxPoints: 256 },
    '10m': { seconds: 600,      label: '10m', refreshMs: 0, live: true, maxPoints: 2416 },
    '1h':  { seconds: 3600,     label: '1h',  refreshMs: 10000 },
    '6h':  { seconds: 21600,    label: '6h',  refreshMs: 60000 },
    '24h': { seconds: 86400,    label: '24h', refreshMs: 60000 },
    '7d':  { seconds: 604800,   label: '7d',  refreshMs: 0 },
    '30d': { seconds: 2592000,  label: '30d', refreshMs: 0 },
    '90d': { seconds: 7776000,  label: '90d', refreshMs: 0 },
};

const CHART_CONFIGS = [
    {
        id: 'chart-cpu',
        title: 'CPU & Load',
        series: [
            { key: 'cpu_usage', label: 'CPU %', color: '#3498db', yAxisID: 'y' },
            { key: 'cpu_load_1m', label: 'Load 1m', color: '#e67e22', yAxisID: 'y2' },
        ],
        yAxes: [
            { id: 'y', label: '%', position: 'left', min: 0, max: 100 },
            { id: 'y2', label: 'Load', position: 'right', min: 0 },
        ],
    },
    {
        id: 'chart-ram',
        title: 'RAM',
        series: [
            { key: 'memory_usage', label: 'RAM %', color: '#2ecc71', yAxisID: 'y' },
            { key: 'memory_used_gb', label: 'Belegt (GB)', color: '#27ae60', yAxisID: 'y2' },
        ],
        yAxes: [
            { id: 'y', label: '%', position: 'left', min: 0, max: 100 },
            { id: 'y2', label: 'GB', position: 'right', min: 0 },
        ],
    },
    {
        id: 'chart-gpu',
        title: 'GPU & VRAM',
        series: [
            { key: 'gpu_util', label: 'GPU %', color: '#9b59b6', yAxisID: 'y' },
            { key: 'vram_usage', label: 'VRAM %', color: '#8e44ad', yAxisID: 'y' },
        ],
        yAxes: [
            { id: 'y', label: '%', position: 'left', min: 0, max: 100 },
        ],
    },
    {
        id: 'chart-disk',
        title: 'Disk I/O',
        series: [
            { key: 'disk_read_mb_s', label: 'Lesen MB/s', color: '#1abc9c', yAxisID: 'y' },
            { key: 'disk_write_mb_s', label: 'Schreiben MB/s', color: '#e74c3c', yAxisID: 'y' },
        ],
        yAxes: [
            { id: 'y', label: 'MB/s', position: 'left', min: 0 },
        ],
    },
];

const HISTORY_LIVE_METRIC_KEYS = [
    ...new Set(CHART_CONFIGS.flatMap(config => config.series.map(series => series.key))),
];

/**
 * Tab initialisieren (wird von app_core.js aufgerufen).
 */
function initHistoryTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

    destroyHistoryCharts();

    // Zeitraum-Selektor in die Filter-Bar
    const filterBar = document.getElementById('filterBar');
    if (filterBar) {
        let filterHtml = '<div class="time-range-selector"><span class="range-label">Zeitraum:</span>';
        for (const [key, cfg] of Object.entries(HISTORY_RANGES)) {
            const active = key === historyActiveRange ? ' active' : '';
            filterHtml += `<button class="time-range-btn${active}" data-range="${key}">${cfg.label}</button>`;
        }
        filterHtml += '</div>';
        filterBar.innerHTML = filterHtml;
        filterBar.style.display = 'flex';
        filterBar.querySelectorAll('.time-range-btn').forEach(btn => {
            btn.addEventListener('click', () => setHistoryRange(btn.getAttribute('data-range')));
        });
    }

    // Inhalt: DB-Stats + Chart-Grid
    let html = '<div class="history-tab">';
    html += '<div class="history-stats" id="historyStats"></div>';
    html += '<div class="charts-grid">';
    for (const cfg of CHART_CONFIGS) {
        html += `<div class="chart-container">`;
        html += `<div class="chart-title">${cfg.title}</div>`;
        html += `<canvas id="${cfg.id}"></canvas>`;
        html += `</div>`;
    }
    html += '</div></div>';

    container.innerHTML = html;

    // Charts erstellen
    for (const cfg of CHART_CONFIGS) {
        const chart = new TimeSeriesChart(cfg.id, {
            title: cfg.title,
            series: cfg.series,
            yAxes: cfg.yAxes,
        });
        historyCharts.push(chart);
    }

    // Datenquelle passend zum Zeitraum starten.
    if (HISTORY_RANGES[historyActiveRange].live) {
        startHistoryLive();
    } else {
        loadHistoryData();
        loadHistoryStats();
        startHistoryRefresh();
    }
}

/**
 * Zeitraum wechseln.
 */
function setHistoryRange(range) {
    if (!HISTORY_RANGES[range]) return;
    historyLoadGeneration += 1;
    stopHistoryRefresh();
    stopHistoryLive();
    historyActiveRange = range;

    // Button-Styling
    document.querySelectorAll('.time-range-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-range') === range);
    });

    if (HISTORY_RANGES[range].live) {
        startHistoryLive();
    } else {
        loadHistoryData();
        loadHistoryStats();
        startHistoryRefresh();
    }
}

/**
 * Historische Daten vom API laden und Charts aktualisieren.
 */
async function loadHistoryData() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || range.live) return;
    const requestedRange = historyActiveRange;
    const generation = ++historyLoadGeneration;

    const to = Math.floor(Date.now() / 1000);
    const from = to - range.seconds;

    // Alle Metriken in einem Request
    const allMetrics = CHART_CONFIGS.flatMap(c => c.series.map(s => s.key));
    const uniqueMetrics = [...new Set(allMetrics)];

    try {
        const resp = await fetch(
            `/api/history/metrics?metrics=${uniqueMetrics.join(',')}&from=${from}&to=${to}&max_points=500`
        );
        if (!resp.ok) return;
        const data = await resp.json();
        if (generation !== historyLoadGeneration || historyActiveRange !== requestedRange) return;

        // Charts aktualisieren
        historyCharts.forEach(chart => {
            chart.update(data.timestamps || [], data.series || {});
        });
    } catch (e) {
        console.error('History-Daten laden fehlgeschlagen:', e);
    }
}

/**
 * DB-Statistiken laden.
 */
async function loadHistoryStats() {
    const requestedRange = historyActiveRange;
    if (HISTORY_RANGES[requestedRange] && HISTORY_RANGES[requestedRange].live) return;
    try {
        const resp = await fetch('/api/history/stats');
        if (!resp.ok) return;
        const stats = await resp.json();
        if (historyActiveRange !== requestedRange) return;

        const el = document.getElementById('historyStats');
        if (!el) return;

        let html = '';
        html += `<span class="stat-item">Datenpunkte: <span class="stat-value">${(stats.total_rows || 0).toLocaleString()}</span></span>`;
        if (stats.oldest) {
            html += `<span class="stat-item">Seit: <span class="stat-value">${stats.oldest}</span></span>`;
        }
        if (stats.db_size_mb !== undefined) {
            html += `<span class="stat-item">DB: <span class="stat-value">${stats.db_size_mb} MB</span></span>`;
        }
        el.innerHTML = html;
    } catch (e) {
        // Nicht kritisch
    }
}

/**
 * Auto-Refresh starten basierend auf aktivem Zeitraum.
 */
function startHistoryRefresh() {
    stopHistoryRefresh();

    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || range.live || !range.refreshMs) return;

    historyRefreshTimer = setInterval(() => {
        loadHistoryData();
        loadHistoryStats();
    }, range.refreshMs);
}

/**
 * Auto-Refresh stoppen.
 */
function stopHistoryRefresh() {
    if (historyRefreshTimer) {
        clearInterval(historyRefreshTimer);
        historyRefreshTimer = null;
    }
}

function historyLiveSocketUrl(rangeKey) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/history/${rangeKey}`;
}

function normalizeHistoryLiveSample(raw) {
    if (!raw || typeof raw !== 'object') return null;
    if (!Number.isSafeInteger(raw.sequence) || raw.sequence <= 0) return null;
    if (typeof raw.timestamp !== 'number' || !Number.isFinite(raw.timestamp)) return null;
    if (!raw.values || typeof raw.values !== 'object') return null;

    const values = {};
    for (const key of HISTORY_LIVE_METRIC_KEYS) {
        const value = raw.values[key];
        values[key] = typeof value === 'number' && Number.isFinite(value) ? value : null;
    }

    const states = {};
    if (raw.states && typeof raw.states === 'object') {
        for (const key of ['cpu', 'memory', 'disk', 'gpu']) {
            if (typeof raw.states[key] === 'string') states[key] = raw.states[key];
        }
    }
    return { sequence: raw.sequence, timestamp: raw.timestamp, values, states };
}

function pruneHistoryLiveSamples() {
    if (!historyLiveSamples.length) return;
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || !range.live) return;
    const latestTimestamp = historyLiveSamples[historyLiveSamples.length - 1].timestamp;
    const cutoff = latestTimestamp - range.seconds;
    let firstVisible = 0;
    while (
        firstVisible < historyLiveSamples.length
        && historyLiveSamples[firstVisible].timestamp <= cutoff
    ) {
        firstVisible += 1;
    }
    if (firstVisible > 0) historyLiveSamples.splice(0, firstVisible);
    if (historyLiveSamples.length > range.maxPoints) {
        historyLiveSamples.splice(0, historyLiveSamples.length - range.maxPoints);
    }
}

function replaceHistoryLiveSamples(rawSamples) {
    const bySequence = new Map();
    for (const raw of Array.isArray(rawSamples) ? rawSamples : []) {
        const sample = normalizeHistoryLiveSample(raw);
        if (sample) bySequence.set(sample.sequence, sample);
    }
    historyLiveSamples = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence);
    pruneHistoryLiveSamples();
}

function appendHistoryLiveSample(raw) {
    const sample = normalizeHistoryLiveSample(raw);
    if (!sample) return false;
    const latest = historyLiveSamples[historyLiveSamples.length - 1];
    if (latest && sample.sequence <= latest.sequence) return false;
    historyLiveSamples.push(sample);
    pruneHistoryLiveSamples();
    return true;
}

function appendHistoryStat(container, label, value) {
    const item = document.createElement('span');
    item.className = 'stat-item';
    item.appendChild(document.createTextNode(`${label}: `));
    const strong = document.createElement('span');
    strong.className = 'stat-value';
    strong.textContent = value;
    item.appendChild(strong);
    container.appendChild(item);
}

function renderHistoryLiveStats() {
    const el = document.getElementById('historyStats');
    if (!el) return;
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || !range.live) return;
    el.replaceChildren();

    const stateLabels = {
        connecting: 'Verbindet…',
        live: 'Verbunden',
        reconnecting: 'Verbindung unterbrochen – neuer Versuch…',
        incompatible: 'Protokoll nicht kompatibel',
        unavailable: 'Live-Sampler nicht verfügbar',
        disconnected: 'Getrennt',
    };
    appendHistoryStat(el, 'Live', stateLabels[historyLiveConnectionState] || historyLiveConnectionState);
    appendHistoryStat(el, 'Takt', `${historyLiveIntervalMs} ms`);
    appendHistoryStat(el, 'Punkte', historyLiveSamples.length.toLocaleString());
    const missedTicks = Number(historyLiveBackendStatus && historyLiveBackendStatus.missed_ticks);
    if (Number.isSafeInteger(missedTicks) && missedTicks > 0) {
        appendHistoryStat(el, 'Ausgelassene Ticks', missedTicks.toLocaleString());
    }

    let coveredSeconds = 0;
    if (historyLiveSamples.length > 1) {
        coveredSeconds = Math.max(
            0,
            historyLiveSamples[historyLiveSamples.length - 1].timestamp
                - historyLiveSamples[0].timestamp
        );
    }
    if (coveredSeconds < range.seconds - 1) {
        const target = range.seconds >= 60 && range.seconds % 60 === 0
            ? `${range.seconds / 60} min`
            : `${range.seconds} s`;
        appendHistoryStat(el, 'Puffer', `${coveredSeconds.toFixed(1)} s / ${target}`);
    }

    const latest = historyLiveSamples[historyLiveSamples.length - 1];
    const gpuState = latest && latest.states ? latest.states.gpu : null;
    if (gpuState && gpuState !== 'ok' && gpuState !== 'partial') {
        appendHistoryStat(el, 'GPU', 'nicht verfügbar');
    }
}

function renderHistoryLiveCharts() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range || !range.live) return;
    const timestamps = historyLiveSamples.map(sample => sample.timestamp);
    const series = {};
    for (const cfg of CHART_CONFIGS) {
        for (const item of cfg.series) {
            if (!series[item.key]) series[item.key] = [];
        }
    }
    for (const sample of historyLiveSamples) {
        for (const key of Object.keys(series)) {
            series[key].push(sample.values[key]);
        }
    }
    historyCharts.forEach(chart => chart.update(timestamps, series, {
        relativeWindowSeconds: range.seconds,
    }));
    renderHistoryLiveStats();
}

function scheduleHistoryLiveRender() {
    if (historyLiveRenderFrame !== null) return;
    historyLiveRenderFrame = window.requestAnimationFrame(() => {
        historyLiveRenderFrame = null;
        renderHistoryLiveCharts();
    });
}

function startHistoryLive(resetSamples = true) {
    stopHistoryLive(false);
    const rangeKey = historyActiveRange;
    const range = HISTORY_RANGES[rangeKey];
    if (!range || !range.live) return;
    if (resetSamples) historyLiveSamples = [];
    historyLiveConnectionState = 'connecting';
    historyLiveBackendStatus = null;
    const generation = ++historyLiveGeneration;
    renderHistoryLiveCharts();

    const socket = new WebSocket(historyLiveSocketUrl(rangeKey));
    historyLiveSocket = socket;

    socket.onopen = () => {
        if (generation !== historyLiveGeneration) return;
        historyLiveConnectionState = 'connecting';
        renderHistoryLiveStats();
    };

    socket.onmessage = event => {
        if (generation !== historyLiveGeneration || historyActiveRange !== rangeKey) return;
        let message;
        try {
            message = JSON.parse(event.data);
        } catch (e) {
            console.error(`${rangeKey}-Verlauf: Nachricht konnte nicht geparst werden:`, e);
            return;
        }
        if (message.version !== HISTORY_LIVE_PROTOCOL_VERSION) {
            historyLiveConnectionState = 'incompatible';
            renderHistoryLiveStats();
            socket.onclose = null;
            if (historyLiveSocket === socket) historyLiveSocket = null;
            socket.close(1002, 'unsupported history protocol');
            return;
        }
        if (message.type === `history_${rangeKey}_snapshot`) {
            replaceHistoryLiveSamples(message.samples);
            historyLiveBackendStatus = message.status || null;
            if (Number.isFinite(message.interval_ms)) historyLiveIntervalMs = message.interval_ms;
            historyLiveConnectionState = 'live';
            scheduleHistoryLiveRender();
        } else if (message.type === `history_${rangeKey}_point`) {
            if (appendHistoryLiveSample(message.sample)) {
                historyLiveConnectionState = 'live';
                scheduleHistoryLiveRender();
            }
        }
    };

    socket.onerror = () => {
        if (generation !== historyLiveGeneration) return;
        historyLiveConnectionState = 'reconnecting';
        renderHistoryLiveStats();
    };

    socket.onclose = event => {
        if (generation !== historyLiveGeneration) return;
        historyLiveSocket = null;
        if (historyActiveRange !== rangeKey || currentTab !== 'history') return;
        historyLiveConnectionState = event && event.code === 1013
            ? 'unavailable'
            : 'reconnecting';
        renderHistoryLiveStats();
        historyLiveReconnectTimer = setTimeout(() => {
            historyLiveReconnectTimer = null;
            if (historyActiveRange === rangeKey && currentTab === 'history') {
                startHistoryLive(false);
            }
        }, HISTORY_LIVE_RECONNECT_MS);
    };
}

function stopHistoryLive(clearSamples = true) {
    historyLiveGeneration += 1;
    if (historyLiveReconnectTimer) {
        clearTimeout(historyLiveReconnectTimer);
        historyLiveReconnectTimer = null;
    }
    if (historyLiveRenderFrame !== null) {
        window.cancelAnimationFrame(historyLiveRenderFrame);
        historyLiveRenderFrame = null;
    }
    if (historyLiveSocket) {
        const socket = historyLiveSocket;
        historyLiveSocket = null;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        try { socket.close(1000, 'history view closed'); } catch (e) { /* ignorieren */ }
    }
    historyLiveConnectionState = 'disconnected';
    historyLiveBackendStatus = null;
    if (clearSamples) historyLiveSamples = [];
}

/**
 * Alle Chart-Instanzen aufraeumen.
 */
function destroyHistoryCharts() {
    historyLoadGeneration += 1;
    stopHistoryRefresh();
    stopHistoryLive();
    historyCharts.forEach(c => c.destroy());
    historyCharts = [];
}

/**
 * Theme-Refresh fuer alle Charts (nach Theme-Wechsel).
 */
function refreshHistoryTheme() {
    historyCharts.forEach(c => c.refreshTheme());
}

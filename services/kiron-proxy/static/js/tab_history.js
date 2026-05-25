/**
 * Verlauf-Tab - Historische Metriken mit Zeitreihen-Charts.
 * Zeigt 4 Charts (CPU, RAM, GPU, Disk) mit Zeitraum-Selektor.
 */

// State
let historyCharts = [];
let historyRefreshTimer = null;
let historyActiveRange = '1h';

const HISTORY_RANGES = {
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

/**
 * Tab initialisieren (wird von app_core.js aufgerufen).
 */
function initHistoryTab() {
    const container = document.getElementById('contentContainer');
    if (!container) return;

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
    destroyHistoryCharts();
    for (const cfg of CHART_CONFIGS) {
        const chart = new TimeSeriesChart(cfg.id, {
            title: cfg.title,
            series: cfg.series,
            yAxes: cfg.yAxes,
        });
        historyCharts.push(chart);
    }

    // Daten laden
    loadHistoryData();
    loadHistoryStats();
    startHistoryRefresh();
}

/**
 * Zeitraum wechseln.
 */
function setHistoryRange(range) {
    if (!HISTORY_RANGES[range]) return;
    historyActiveRange = range;

    // Button-Styling
    document.querySelectorAll('.time-range-btn').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-range') === range);
    });

    loadHistoryData();
    startHistoryRefresh();
}

/**
 * Historische Daten vom API laden und Charts aktualisieren.
 */
async function loadHistoryData() {
    const range = HISTORY_RANGES[historyActiveRange];
    if (!range) return;

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

        // Charts aktualisieren
        historyCharts.forEach((chart, i) => {
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
    try {
        const resp = await fetch('/api/history/stats');
        if (!resp.ok) return;
        const stats = await resp.json();

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
    if (!range || !range.refreshMs) return;

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

/**
 * Alle Chart-Instanzen aufraeumen.
 */
function destroyHistoryCharts() {
    stopHistoryRefresh();
    historyCharts.forEach(c => c.destroy());
    historyCharts = [];
}

/**
 * Theme-Refresh fuer alle Charts (nach Theme-Wechsel).
 */
function refreshHistoryTheme() {
    historyCharts.forEach(c => c.refreshTheme());
}

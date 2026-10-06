/**
 * Generische TimeSeriesChart-Klasse (Wrapper um Chart.js).
 * Unterstuetzt Dark/Light Theme, Dual-Y-Achsen, Responsive.
 */

class TimeSeriesChart {
    /**
     * @param {string} canvasId - ID des Canvas-Elements
     * @param {object} config
     *   config.title - Chart-Titel (informativ, nicht gerendert von Chart.js)
     *   config.series - Array von {key, label, color, yAxisID}
     *   config.yAxes - Array von {id, label, position, min, max}
     */
    constructor(canvasId, config) {
        this.canvasId = canvasId;
        this.config = config;
        this.chart = null;
        this.xAxisMode = 'time';
        this._createChart();
    }

    _getThemeColors() {
        const style = getComputedStyle(document.documentElement);
        return {
            text: style.getPropertyValue('--text-secondary').trim() || '#666',
            grid: style.getPropertyValue('--border-color').trim() || '#ddd',
            bg: style.getPropertyValue('--bg-secondary').trim() || '#f5f5f5',
        };
    }

    _timeXAxis(theme) {
        return {
            type: 'time',
            time: {
                tooltipFormat: 'dd.MM.yyyy HH:mm:ss',
                displayFormats: {
                    second: 'HH:mm:ss',
                    minute: 'HH:mm',
                    hour: 'HH:mm',
                    day: 'dd.MM.',
                },
            },
            ticks: { color: theme.text, maxTicksLimit: 8 },
            grid: { color: theme.grid + '40' },
        };
    }

    _relativeSecondsXAxis(theme, windowSeconds) {
        const useMinuteTicks = windowSeconds >= 600;
        return {
            type: 'linear',
            min: 0,
            max: windowSeconds,
            reverse: true,
            ticks: {
                color: theme.text,
                stepSize: useMinuteTicks ? 120 : 10,
                autoSkip: false,
                callback: value => useMinuteTicks
                    ? `${(Number(value) / 60).toFixed(0)} min`
                    : `${Number(value).toFixed(0)} s`,
            },
            grid: { color: theme.grid + '40' },
        };
    }

    _setXAxisMode(relativeWindowSeconds) {
        const isRelative = Number.isFinite(relativeWindowSeconds) && relativeWindowSeconds > 0;
        const nextMode = isRelative ? `relative:${relativeWindowSeconds}` : 'time';
        if (!this.chart || this.xAxisMode === nextMode) return;

        const theme = this._getThemeColors();
        this.chart.options.scales.x = isRelative
            ? this._relativeSecondsXAxis(theme, relativeWindowSeconds)
            : this._timeXAxis(theme);
        this.xAxisMode = nextMode;
    }

    _createChart() {
        const canvas = document.getElementById(this.canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const theme = this._getThemeColors();
        const { series, yAxes } = this.config;

        // Datasets
        const datasets = series.map(s => ({
            label: s.label,
            data: [],
            borderColor: s.color,
            backgroundColor: s.color + '18',
            borderWidth: 1.5,
            pointRadius: 0,
            pointHitRadius: 8,
            tension: 0.3,
            fill: series.length <= 2,
            yAxisID: s.yAxisID || 'y',
        }));

        // Y-Achsen
        const scales = {
            x: this._timeXAxis(theme),
        };

        (yAxes || [{ id: 'y', label: '%', position: 'left' }]).forEach(axis => {
            scales[axis.id] = {
                position: axis.position || 'left',
                title: {
                    display: true,
                    text: axis.label,
                    color: theme.text,
                    font: { size: 11 },
                },
                ticks: { color: theme.text, maxTicksLimit: 6 },
                grid: {
                    color: axis.position === 'right' ? 'transparent' : theme.grid + '40',
                },
                min: axis.min,
                max: axis.max,
            };
        });

        this.chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: { color: theme.text, boxWidth: 12, padding: 10 },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleFont: { size: 12 },
                        bodyFont: { size: 11 },
                    },
                },
                scales,
                animation: { duration: 300 },
            },
        });
    }

    /**
     * Neue Daten setzen und Chart aktualisieren.
     * @param {number[]} timestamps - Unix-Timestamps
     * @param {object} seriesData - {key: [values]}
     * @param {object} options - optional: {relativeWindowSeconds: number}
     */
    update(timestamps, seriesData, options = {}) {
        if (!this.chart) return;

        const relativeWindowSeconds = Number(options.relativeWindowSeconds);
        const isRelative = Number.isFinite(relativeWindowSeconds) && relativeWindowSeconds > 0;
        this._setXAxisMode(isRelative ? relativeWindowSeconds : null);

        let xValues;
        if (isRelative) {
            const latestTimestamp = timestamps.length ? timestamps[timestamps.length - 1] : 0;
            xValues = timestamps.map(ts => latestTimestamp - ts);
        } else {
            xValues = timestamps.map(ts => new Date(ts * 1000));
        }

        this.config.series.forEach((s, i) => {
            const values = seriesData[s.key] || [];
            this.chart.data.datasets[i].data = xValues.map((x, idx) => ({
                x,
                y: values[idx] != null ? values[idx] : null,
            }));
        });

        this.chart.update('none');
    }

    /**
     * Theme-Farben aktualisieren (nach Theme-Wechsel aufrufen).
     */
    refreshTheme() {
        if (!this.chart) return;

        const theme = this._getThemeColors();
        const scales = this.chart.options.scales;

        // X-Achse
        if (scales.x) {
            scales.x.ticks.color = theme.text;
            scales.x.grid.color = theme.grid + '40';
        }

        // Y-Achsen
        Object.keys(scales).forEach(key => {
            if (key === 'x') return;
            scales[key].ticks.color = theme.text;
            if (scales[key].title) scales[key].title.color = theme.text;
        });

        // Legende
        this.chart.options.plugins.legend.labels.color = theme.text;

        this.chart.update('none');
    }

    destroy() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }
}

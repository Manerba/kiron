/**
 * Dashboard Template - Widget Dashboard
 *
 * Rendert Dashboard mit generischen Widgets basierend auf
 * Konfiguration vom Server. Jedes Widget lädt seine Daten
 * von einem eigenen API-Endpoint.
 */

class WidgetDashboard {
    constructor(config, containerId) {
        this.config = config;     // { widgets: [...] }
        this.containerId = containerId;
    }

    async init() {
        await this.loadAllWidgets();
    }

    async loadAllWidgets() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        // Loading anzeigen
        container.innerHTML = `
            <div class="kpi-loading">
                <div class="spinner"></div>
                <span>Lädt Dashboard-Daten...</span>
            </div>
        `;

        try {
            // Alle Widget-Daten parallel laden
            const promises = this.config.widgets.map(w =>
                fetch(w.apiEndpoint).then(r => r.json())
            );
            const results = await Promise.all(promises);
            this.renderWidgets(results);
        } catch (error) {
            console.error('Dashboard-Fehler:', error);
            container.innerHTML = `
                <div class="kpi-error">
                    <span class="kpi-error-icon">⚠️</span>
                    <span class="kpi-error-text">Fehler beim Laden der Daten</span>
                </div>
            `;
        }
    }

    renderWidgets(dataArray) {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        const html = this.config.widgets.map((widget, i) =>
            this.renderWidget(widget, dataArray[i])
        ).join('');
        container.innerHTML = `<div class="widget-dashboard">${html}</div>`;
    }

    renderWidget(widget, data) {
        let contentHtml;
        if (widget.type === 'table') {
            contentHtml = this.renderWidgetTable(widget, data);
        } else {
            contentHtml = this.renderWidgetKpis(widget, data);
        }

        // Optionale Info-Zeile zwischen Header und Body
        const infoHtml = data && data.info
            ? `<div class="widget-info">${this.escapeHtml(data.info)}</div>`
            : '';

        return `
            <div class="widget" data-widget="${widget.id}">
                <div class="widget-header">
                    <h3 class="widget-title">${this.escapeHtml(widget.title)}</h3>
                    ${widget.subtitle ? `<div class="widget-subtitle">${this.escapeHtml(widget.subtitle)}</div>` : ''}
                </div>
                ${infoHtml}
                <div class="widget-body">
                    ${contentHtml}
                </div>
            </div>
        `;
    }

    renderWidgetTable(widget, data) {
        if (!data || data.error) {
            return `<div class="kpi-empty"><span class="kpi-empty-text">Keine Daten verfügbar</span></div>`;
        }

        const headerHtml = widget.columns.map(col =>
            `<th>${this.escapeHtml(col.label)}</th>`
        ).join('');

        const rowsHtml = (data.rows || []).map(row => {
            const cells = widget.columns.map(col => {
                const val = row[col.key] ?? '-';
                const formatted = this.formatValue(val, col.format);
                const colorClass = this.getColorClass(val, col);
                return `<td class="${colorClass}">${formatted}</td>`;
            }).join('');
            return `<tr>${cells}</tr>`;
        }).join('');

        if (!data.rows || data.rows.length === 0) {
            return `<div class="kpi-empty"><span class="kpi-empty-text">Keine Daten verfügbar</span></div>`;
        }

        return `
            <table class="widget-table">
                <thead><tr>${headerHtml}</tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    }

    renderWidgetKpis(widget, data) {
        if (!data || !widget.cards) {
            return `<div class="kpi-empty"><span class="kpi-empty-text">Keine Daten verfügbar</span></div>`;
        }

        const cardsHtml = widget.cards.map(card => {
            const value = data[card.valueField] ?? '-';
            const formatted = this.formatValue(value, card.format);
            const colorClass = this.getColorClass(value, card);
            return `
                <div class="kpi-card ${colorClass}" data-card="${card.id}">
                    ${card.icon ? `<div class="kpi-icon">${card.icon}</div>` : ''}
                    <div class="kpi-value">${formatted}</div>
                    <div class="kpi-title">${this.escapeHtml(card.title)}</div>
                </div>
            `;
        }).join('');

        return `<div class="kpi-cards">${cardsHtml}</div>`;
    }

    formatValue(value, format) {
        if (value === null || value === undefined || value === '-') return '-';

        switch (format) {
            case 'percent':
                const num = parseFloat(value);
                if (isNaN(num)) return '-';
                return `${num.toFixed(1)}%`;

            case 'currency':
                const curr = parseFloat(value);
                if (isNaN(curr)) return '-';
                return `$${curr.toFixed(2)}`;

            case 'number':
                if (typeof value === 'number') {
                    if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                    if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
                    if (value % 1 !== 0) return value.toFixed(1);
                    return value.toLocaleString('de-DE');
                }
                return String(value);

            case 'text':
            default:
                return String(value);
        }
    }

    getColorClass(value, col) {
        if (!col.colorRules) return '';

        const numVal = parseFloat(value);
        if (isNaN(numVal)) return '';

        for (const [rule, color] of Object.entries(col.colorRules)) {
            const operator = rule.charAt(0);
            const threshold = parseFloat(rule.slice(1));

            if (operator === '>' && numVal > threshold) return `kpi-${color}`;
            if (operator === '<' && numVal < threshold) return `kpi-${color}`;
            if (operator === '=' && numVal === threshold) return `kpi-${color}`;
        }

        return '';
    }

    escapeHtml(text) {
        if (text === null || text === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    }

    async refresh() {
        await this.loadAllWidgets();
    }
}

// Globale Variable für Dashboard-Instanz
let dashboard = null;

// Initialisierung (wird von app_core.js aufgerufen)
async function initDashboard() {
    try {
        const response = await fetch('/api/dashboard/config');
        const config = await response.json();

        dashboard = new WidgetDashboard(config, 'contentContainer');
        await dashboard.init();

        return dashboard;
    } catch (error) {
        console.error('Dashboard-Initialisierung fehlgeschlagen:', error);
        return null;
    }
}

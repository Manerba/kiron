/**
 * Generische aufklappbare Tabellen-Komponente
 *
 * Verwendung:
 *   const table = new ExpandableTable(config, containerId);
 *   table.loadData();
 */

class ExpandableTable {
    constructor(config, containerId, options = {}) {
        this.config = config;
        this.containerId = containerId;
        this.options = options;

        // State
        this.data = [];
        this.filteredData = [];
        this.sortColumn = config.defaultSort?.field || '';
        this.sortDirection = config.defaultSort?.direction || 'asc';
        this.activeFilters = {};
        this.filterValues = {};
        this.lastParams = {};  // Für Reload bei Seitenwechsel
        this.stanceCounts = {};
        this.v2StanceCounts = {};
        this.filterDebounceTimer = null;  // Debounce für Input-Filter

        // Pagination State
        this.pagination = {
            enabled: config.pagination?.enabled || false,
            currentPage: 1,
            pageSize: config.pagination?.defaultPageSize || 100,
            totalItems: 0,
            totalPages: 0,
            options: config.pagination?.options || [20, 100, 300, 1000]
        };

        // Renderer-Registry
        this.renderers = {
            'icon': this.renderIcon.bind(this),
            'percent': this.renderPercent.bind(this),
            'badge': this.renderBadge.bind(this),
            'link': this.renderLink.bind(this),
            'currency': this.renderCurrency.bind(this),
            'change_pct': this.renderChangePct.bind(this),
            'ratio': this.renderRatio.bind(this),
            'actions': this.renderActions.bind(this),
            'toggle': this.renderToggle.bind(this),
            'default': this.renderDefault.bind(this),
        };

        // Icons für Standard-Renderer
        this.icons = {
            stance: {
                'bullish': '🟢',
                'bearish': '🔴',
                'neutral': '⚪',
                'unclear': '❓',
                'none': '➖'
            },
            provenance: {
                'editorial': '📝',
                'pr_release': '📣',
                'sponsored': '💰',
                'unknown': '❓'
            },
            signal: {
                'BUY': '🟢',
                'WATCH': '🟡',
                'AVOID': '🔴',
                'TOO_LATE': '⚫'
            },
            blocklist_reason: {
                'reverse_split': '<span class="reason-icon reason-reverse-split" title="Reverse Split">🔄</span>',
                'delisted': '<span class="reason-icon reason-delisted" title="Delisted">❌</span>',
                'manual': '<span class="reason-icon reason-manual" title="Manuell gesperrt">✋</span>'
            }
        };

        this.init();
    }

    init() {
        // Filter initialisieren
        this.config.filters?.forEach(filter => {
            if (filter.type === 'buttons') {
                this.activeFilters[filter.field] = new Set();
            } else {
                this.filterValues[filter.id] = filter.default || '';
            }
        });
    }

    // ========================================
    // Daten laden
    // ========================================

    async loadData(params = {}) {
        this.showLoading();

        try {
            // Query-Parameter zusammenbauen
            const queryParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                if (value !== undefined && value !== '') {
                    queryParams.append(key, value);
                }
            }

            // Pagination-Parameter hinzufügen wenn aktiviert
            if (this.pagination.enabled) {
                queryParams.set('page', this.pagination.currentPage);
                queryParams.set('page_size', this.pagination.pageSize);

                // Button-Filter als serverseitiger Filter hinzufügen
                for (const [field, values] of Object.entries(this.activeFilters)) {
                    if (values.size > 0) {
                        const filterParam = field === 'stance' ? 'stance_filter' :
                                           field === 'signal_type' ? 'signal_filter' :
                                           field === 'v2_signal_type' ? 'v2_signal_filter' :
                                           `${field}_filter`;
                        queryParams.set(filterParam, Array.from(values).join(','));
                    }
                }

                // Input-Filter als serverseitiger Filter hinzufügen
                this.config.filters?.forEach(filter => {
                    if (filter.type === 'input') {
                        const value = this.filterValues[filter.id];
                        if (value) {
                            queryParams.set(filter.field, value);
                        }
                    }
                });

                // Sortierung hinzufügen
                if (this.sortColumn) {
                    queryParams.set('sort_field', this.sortColumn);
                    queryParams.set('sort_dir', this.sortDirection);
                }
            }

            // lastParams speichern für Reload (ohne Pagination-Parameter)
            this.lastParams = { ...params };

            const url = queryParams.toString()
                ? `${this.config.apiEndpoint}?${queryParams}`
                : this.config.apiEndpoint;

            const response = await fetch(url);
            const result = await response.json();

            // Daten können direkt ein Array sein oder in einem Wrapper
            this.data = Array.isArray(result) ? result : (result.data || result.articles || result.watchlist || result.predictions || result.variants || result.blocklist || result.rules || result.strategies || result.sources || result.items || []);

            // Zusätzliche Daten für Filter/Stats (optional)
            this.stanceCounts = result.stance_counts || result.signal_counts || result.reason_counts || {};
            this.v2StanceCounts = result.v2_signal_counts || {};
            this.stats = result.stats || {};

            // Pagination-Metadaten verarbeiten
            if (this.pagination.enabled && result.pagination) {
                this.pagination.currentPage = result.pagination.page;
                this.pagination.pageSize = result.pagination.page_size;
                this.pagination.totalItems = result.pagination.total_items;
                this.pagination.totalPages = result.pagination.total_pages;
            }

            // Callback für zusätzliche Daten
            if (this.options.onDataLoaded) {
                this.options.onDataLoaded(result);
            }

            this.applyFiltersAndRender();
        } catch (error) {
            console.error('Fehler beim Laden der Daten:', error);
            this.showError('Fehler beim Laden der Daten');
        }
    }

    // ========================================
    // Pagination
    // ========================================

    goToPage(page) {
        if (page < 1 || page > this.pagination.totalPages) return;
        if (page === this.pagination.currentPage) return;

        this.pagination.currentPage = page;
        this.loadData(this.lastParams);
    }

    setPageSize(size) {
        if (!this.pagination.options.includes(size)) return;
        if (size === this.pagination.pageSize) return;

        this.pagination.pageSize = size;
        this.pagination.currentPage = 1;  // Zurück zu Seite 1
        this.loadData(this.lastParams);
    }

    /**
     * Berechnet die Seitenzahlen für das Sliding-Window (11 Seiten)
     */
    getPageNumbers() {
        const total = this.pagination.totalPages;
        const current = this.pagination.currentPage;
        const windowSize = 11;

        // Weniger als windowSize Seiten → alle anzeigen
        if (total <= windowSize) {
            return Array.from({ length: total }, (_, i) => i + 1);
        }

        const halfWindow = Math.floor(windowSize / 2);  // 5
        let start, end;

        if (current <= halfWindow + 1) {
            // Am Anfang: zeige 1-11
            start = 1;
            end = windowSize;
        } else if (current >= total - halfWindow) {
            // Am Ende: zeige letzte 11
            start = total - windowSize + 1;
            end = total;
        } else {
            // In der Mitte: current ±5
            start = current - halfWindow;
            end = current + halfWindow;
        }

        return Array.from({ length: end - start + 1 }, (_, i) => start + i);
    }

    renderPagination() {
        if (!this.pagination.enabled || this.pagination.totalPages <= 1) {
            return '';
        }

        const tableId = this.config.id;
        const current = this.pagination.currentPage;
        const total = this.pagination.totalPages;
        const pageNumbers = this.getPageNumbers();

        // Prüfen ob Ellipsis nötig ist
        const showStartEllipsis = pageNumbers[0] > 1;
        const showEndEllipsis = pageNumbers[pageNumbers.length - 1] < total;

        let html = '<div class="pagination-wrapper">';

        // Linke Seite: Seitenwahl
        html += '<div class="pagination-container">';
        html += '<div class="pagination-nav">';

        // Erste Seite
        html += `<button class="pagination-btn ${current === 1 ? 'disabled' : ''}"
                         onclick="tables['${tableId}'].goToPage(1)"
                         ${current === 1 ? 'disabled' : ''}
                         title="Erste Seite">&laquo;</button>`;

        // Vorherige Seite
        html += `<button class="pagination-btn ${current === 1 ? 'disabled' : ''}"
                         onclick="tables['${tableId}'].goToPage(${current - 1})"
                         ${current === 1 ? 'disabled' : ''}
                         title="Vorherige Seite">&lt;</button>`;

        // Seiten-Buttons
        html += '<div class="pagination-pages">';

        if (showStartEllipsis) {
            html += '<span class="pagination-ellipsis">...</span>';
        }

        pageNumbers.forEach(page => {
            const isActive = page === current;
            html += `<button class="pagination-btn ${isActive ? 'active' : ''}"
                             onclick="tables['${tableId}'].goToPage(${page})">${page}</button>`;
        });

        if (showEndEllipsis) {
            html += '<span class="pagination-ellipsis">...</span>';
        }

        html += '</div>';  // .pagination-pages

        // Nächste Seite
        html += `<button class="pagination-btn ${current === total ? 'disabled' : ''}"
                         onclick="tables['${tableId}'].goToPage(${current + 1})"
                         ${current === total ? 'disabled' : ''}
                         title="Nächste Seite">&gt;</button>`;

        // Letzte Seite
        html += `<button class="pagination-btn ${current === total ? 'disabled' : ''}"
                         onclick="tables['${tableId}'].goToPage(${total})"
                         ${current === total ? 'disabled' : ''}
                         title="Letzte Seite">&raquo;</button>`;

        html += '</div>';  // .pagination-nav

        // Seiten-Info
        html += `<div class="pagination-info">${current} / ${total}</div>`;
        html += '</div>';  // .pagination-container

        // Rechte Seite: Page-Size Dropdown
        html += '<div class="pagination-page-size">';
        html += '<label>Pro Seite:</label>';
        html += `<select onchange="tables['${tableId}'].setPageSize(parseInt(this.value))">`;

        this.pagination.options.forEach(size => {
            const selected = size === this.pagination.pageSize ? 'selected' : '';
            html += `<option value="${size}" ${selected}>${size}</option>`;
        });

        html += '</select>';
        html += '</div>';  // .pagination-page-size

        html += '</div>';  // .pagination-wrapper

        return html;
    }

    // ========================================
    // Filter
    // ========================================

    setFilter(filterId, value) {
        this.filterValues[filterId] = value;

        // Bei Pagination: Zurück zu Seite 1 und neu laden (serverseitige Filterung)
        if (this.pagination.enabled) {
            // Debounce: 300ms warten bevor API-Call (bei schnellem Tippen)
            clearTimeout(this.filterDebounceTimer);
            this.filterDebounceTimer = setTimeout(() => {
                this.pagination.currentPage = 1;
                this.loadData(this.lastParams);
            }, 300);
        } else {
            this.applyFiltersAndRender();
        }
    }

    toggleButtonFilter(field, value) {
        if (!this.activeFilters[field]) {
            this.activeFilters[field] = new Set();
        }

        if (this.activeFilters[field].has(value)) {
            this.activeFilters[field].delete(value);
        } else {
            this.activeFilters[field].add(value);
        }

        // Bei Pagination: Zurück zu Seite 1 und neu laden
        if (this.pagination.enabled) {
            this.pagination.currentPage = 1;
            this.loadData(this.lastParams);
        } else {
            this.applyFiltersAndRender();
        }
    }

    resetButtonFilter(field) {
        if (this.activeFilters[field]) {
            this.activeFilters[field].clear();
        }

        // Bei Pagination: Zurück zu Seite 1 und neu laden
        if (this.pagination.enabled) {
            this.pagination.currentPage = 1;
            this.loadData(this.lastParams);
        } else {
            this.applyFiltersAndRender();
        }
    }

    applyFiltersAndRender() {
        let filtered = this.data.slice();

        // Bei serverseitiger Pagination: Button-Filter wurden schon serverseitig angewendet
        if (!this.pagination.enabled) {
            // Button-Filter anwenden (nur bei client-seitiger Filterung)
            for (const [field, values] of Object.entries(this.activeFilters)) {
                if (values.size > 0) {
                    filtered = filtered.filter(row => {
                        const rowValue = row[field] || 'none';
                        return values.has(rowValue);
                    });
                }
            }
        }

        // Input/Select-Filter anwenden (nur client-seitige Filter, nicht Select-Filter)
        this.config.filters?.forEach(filter => {
            // Select/Number-Filter werden serverseitig angewendet, nicht hier
            if (filter.type === 'buttons' || filter.type === 'select' || filter.type === 'number') return;

            const value = this.filterValues[filter.id];
            if (!value) return;

            if (filter.wildcard) {
                const searchTerm = value.toLowerCase();
                const fields = [filter.field, ...(filter.wildcardFields || [])];
                filtered = filtered.filter(row => {
                    return fields.some(f => {
                        const fieldValue = (row[f] || '').toLowerCase();
                        return fieldValue.includes(searchTerm);
                    });
                });
            } else {
                filtered = filtered.filter(row => row[filter.field] == value);
            }
        });

        // Sortieren (nur bei client-seitiger Sortierung)
        if (this.sortColumn && !this.pagination.enabled) {
            filtered = this.sortData(filtered);
        }

        this.filteredData = filtered;
        this.render();
    }

    // ========================================
    // Sortierung
    // ========================================

    sortBy(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            // Bei Zeit/Datum standardmäßig absteigend
            this.sortDirection = column.includes('time') || column.includes('date') ? 'desc' : 'asc';
        }

        // Bei Pagination: serverseitige Sortierung
        if (this.pagination.enabled) {
            this.pagination.currentPage = 1;  // Zurück zu Seite 1
            this.loadData(this.lastParams);
        } else {
            this.applyFiltersAndRender();
        }
    }

    sortData(data) {
        const col = this.config.columns.find(c => c.field === this.sortColumn);

        return [...data].sort((a, b) => {
            let aVal = a[this.sortColumn];
            let bVal = b[this.sortColumn];

            // Spezielle Sortierung für bestimmte Renderer
            if (col?.renderer === 'icon' && col?.rendererOptions?.type === 'stance') {
                const order = {'bullish': 1, 'bearish': 2, 'neutral': 3, 'unclear': 4, 'none': 5};
                aVal = order[aVal] || 99;
                bVal = order[bVal] || 99;
                return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            }

            // Datum/Zeit-Sortierung (chronologisch)
            if (col?.renderer === 'datetime' || col?.renderer === 'date') {
                const aDate = this.parseDateValue(aVal);
                const bDate = this.parseDateValue(bVal);
                // null-Werte ans Ende
                if (!aDate && !bDate) return 0;
                if (!aDate) return 1;
                if (!bDate) return -1;
                return this.sortDirection === 'asc'
                    ? aDate.getTime() - bDate.getTime()
                    : bDate.getTime() - aDate.getTime();
            }

            // Numerische Werte
            if (typeof aVal === 'number' || col?.renderer === 'percent') {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
                return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            }

            // String-Vergleich
            aVal = String(aVal || '');
            bVal = String(bVal || '');
            const comparison = aVal.localeCompare(bVal, 'de');
            return this.sortDirection === 'asc' ? comparison : -comparison;
        });
    }

    /**
     * Parst deutsche Datum-/Zeitformate zu JavaScript Date
     * Unterstützte Formate:
     * - DD.MM.YYYY HH:mm (z.B. "03.02.2025 14:30")
     * - DD.MM.YYYY (z.B. "03.02.2025")
     * - DD.MM.YY (z.B. "03.02.25")
     * - ISO 8601 Fallback
     */
    parseDateValue(value) {
        if (!value) return null;
        const str = String(value).trim();

        // Deutsches Format: DD.MM.YYYY HH:mm oder DD.MM.YYYY
        const deMatch = str.match(/^(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?/);
        if (deMatch) {
            let [, day, month, year, hour = '0', minute = '0'] = deMatch;
            if (year.length === 2) year = '20' + year;
            return new Date(year, month - 1, day, hour, minute);
        }

        // ISO 8601 Fallback
        const parsed = new Date(str);
        return isNaN(parsed.getTime()) ? null : parsed;
    }

    // ========================================
    // Rendering
    // ========================================

    render() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            return;
        }

        if (this.filteredData.length === 0) {
            let html = '<div class="table-no-data">Keine Daten gefunden</div>';
            // Pagination trotzdem anzeigen wenn aktiviert
            if (this.pagination.enabled) {
                html += this.renderPagination();
            }
            container.innerHTML = html;
            return;
        }

        let html = '';

        // Header
        if (this.config.showHeader) {
            html += this.renderHeader();
        }

        // Rows
        html += '<div class="table-body">';
        this.filteredData.forEach((row, index) => {
            html += this.renderRow(row, index);
        });
        html += '</div>';

        // Pagination
        if (this.pagination.enabled) {
            html += this.renderPagination();
        }

        container.innerHTML = html;
    }

    renderHeader() {
        let html = `<div class="table-header" style="grid-template-columns: ${this.config.gridTemplate}">`;

        if (this.config.expandable) {
            html += '<span class="header-cell header-expand"></span>';
        }

        this.config.columns.forEach(col => {
            const sortIcon = this.getSortIcon(col.field);
            const clickHandler = col.sortable ? `onclick="tables['${this.config.id}'].sortBy('${col.field}')"` : '';
            const sortableClass = col.sortable ? 'sortable' : '';
            const hasInfo = col.info && col.info.length > 0;
            const infoClass = hasInfo ? 'has-info' : '';

            // Info-Tooltip als data-Attribut
            const infoAttr = hasInfo ? `data-info="${this.escapeHtml(col.info)}"` : '';

            html += `
                <span class="header-cell ${sortableClass} ${infoClass}" ${clickHandler} ${infoAttr}>
                    ${this.escapeHtml(col.label)}${hasInfo ? '<span class="info-indicator">ⓘ</span>' : ''} ${col.sortable ? sortIcon : ''}
                </span>
            `;
        });

        html += '</div>';
        return html;
    }

    renderRow(row, index) {
        const rowId = row.id || index;

        let html = `<div class="table-row-wrapper" data-row-id="${rowId}">`;

        // Haupt-Zeile
        const clickHandler = this.config.expandable
            ? `onclick="tables['${this.config.id}'].toggleRow('${rowId}')"`
            : '';
        const cursorClass = this.config.expandable ? 'clickable' : '';

        html += `<div class="table-row ${cursorClass}" style="grid-template-columns: ${this.config.gridTemplate}" ${clickHandler}>`;

        if (this.config.expandable) {
            html += '<span class="expand-icon">▶</span>';
        }

        this.config.columns.forEach(col => {
            const value = row[col.field];
            const rendered = this.renderCell(value, col, row);
            const alignClass = col.align !== 'left' ? `align-${col.align}` : '';
            const cssClass = col.cssClass || '';

            html += `<span class="table-cell ${alignClass} ${cssClass}">${rendered}</span>`;
        });

        html += '</div>';

        // Detail-Bereich (wenn aufklappbar)
        if (this.config.expandable) {
            html += `<div class="table-row-detail">${this.renderDetail(row)}</div>`;
        }

        html += '</div>';
        return html;
    }

    renderCell(value, col, row) {
        // Truncate
        if (col.truncate > 0 && value && value.length > col.truncate) {
            const truncated = value.substring(0, col.truncate) + '...';
            const tooltip = this.escapeHtml(row[col.tooltipField] || value);
            return `<span title="${tooltip}">${this.escapeHtml(truncated)}</span>`;
        }

        // Renderer
        const renderer = this.renderers[col.renderer] || this.renderers['default'];
        return renderer(value, col, row);
    }

    renderDetail(row) {
        let html = '<div class="detail-content">';

        // Link
        if (this.config.detailLinkField && row[this.config.detailLinkField]) {
            html += `<a href="${this.escapeHtml(row[this.config.detailLinkField])}" target="_blank" class="detail-link">${this.config.detailLinkLabel}</a>`;
        }

        // Detail-Felder (kompakt)
        const compactFields = this.config.detailFields.filter(f => !f.section);
        if (compactFields.length > 0) {
            html += '<div class="detail-fields">';
            compactFields.forEach(field => {
                const value = row[field.field];
                const rendered = field.renderer
                    ? (this.renderers[field.renderer] || this.renderers['default'])(value, field, row)
                    : this.escapeHtml(value || '-');
                html += `
                    <div class="detail-field">
                        <span class="detail-label">${this.escapeHtml(field.label)}</span>
                        <span class="detail-value">${rendered}</span>
                    </div>
                `;
            });
            html += '</div>';
        }

        // Sektionen
        const sectionFields = this.config.detailFields.filter(f => f.section);
        sectionFields.forEach(field => {
            const value = row[field.field];
            if (!value) return;

            html += `<div class="detail-section">`;
            html += `<div class="detail-section-label">${this.escapeHtml(field.label)}</div>`;

            if (field.sectionType === 'quote') {
                html += `<p class="detail-quote">"${this.escapeHtml(value)}"</p>`;
            } else if (field.sectionType === 'code') {
                html += `<pre class="detail-code">${this.escapeHtml(value)}</pre>`;
            } else {
                html += `<div class="detail-text">${this.escapeHtml(value)}</div>`;
            }

            html += '</div>';
        });

        // Langer Text
        if (this.config.detailTextField && row[this.config.detailTextField]) {
            let text = row[this.config.detailTextField];
            if (text.length > this.config.detailTextMaxLength) {
                text = text.substring(0, this.config.detailTextMaxLength) + '\n\n[... gekürzt]';
            }

            html += `<div class="detail-section">`;
            if (this.config.detailTextLabel) {
                html += `<div class="detail-section-label">${this.escapeHtml(this.config.detailTextLabel)}</div>`;
            }
            html += `<div class="detail-longtext">${this.escapeHtml(text)}</div>`;
            html += '</div>';
        }

        html += '</div>';
        return html;
    }

    // ========================================
    // Renderer
    // ========================================

    renderDefault(value, col, row) {
        return this.escapeHtml(value ?? '-');
    }

    renderIcon(value, col, row) {
        const iconType = col.rendererOptions?.type || 'stance';
        const icons = this.icons[iconType] || {};
        return icons[value] || '➖';
    }

    renderPercent(value, col, row) {
        if (value === null || value === undefined) return '-';
        const num = parseFloat(value);
        if (isNaN(num)) return '-';
        // Wenn Wert zwischen 0 und 1, als Prozent interpretieren
        const percent = num <= 1 ? Math.round(num * 100) : Math.round(num);
        return `${percent}%`;
    }

    renderBadge(value, col, row) {
        const badgeClass = col.rendererOptions?.class || value?.toLowerCase() || '';
        return `<span class="badge badge-${badgeClass}">${this.escapeHtml(value || '-')}</span>`;
    }

    renderLink(value, col, row) {
        if (!value) return '-';
        const label = col.rendererOptions?.label || '🔗';
        return `<a href="${this.escapeHtml(value)}" target="_blank">${label}</a>`;
    }

    renderCurrency(value, col, row) {
        if (value === null || value === undefined) return '-';
        const num = parseFloat(value);
        if (isNaN(num)) return '-';
        return `$${num.toFixed(2)}`;
    }

    renderChangePct(value, col, row) {
        if (value === null || value === undefined) return '-';
        const num = parseFloat(value);
        if (isNaN(num)) return '-';
        const sign = num >= 0 ? '+' : '';
        const colorClass = num > 0 ? 'positive' : (num < 0 ? 'negative' : '');
        return `<span class="change-pct ${colorClass}">${sign}${num.toFixed(2)}%</span>`;
    }

    renderRatio(value, col, row) {
        if (value === null || value === undefined) return '-';
        const num = parseFloat(value);
        if (isNaN(num)) return '-';
        return `${num.toFixed(1)}x`;
    }

    renderActions(value, col, row) {
        const id = row.id;
        return `
            <span class="action-btn edit-btn" onclick="event.stopPropagation(); openCommentEditor(${id})" title="Kommentar bearbeiten">✏️</span>
            <span class="action-btn delete-btn" onclick="event.stopPropagation(); deleteWatchlistItem(${id})" title="In Papierkorb">🗑️</span>
        `;
    }

    renderToggle(value, col, row) {
        const codename = row.codename || row.id;
        const checked = value ? 'checked' : '';
        return `
            <label class="toggle-switch" onclick="event.stopPropagation()">
                <input type="checkbox" ${checked}
                       onchange="toggleVariantActive('${this.escapeHtml(codename)}', this)">
                <span class="toggle-slider"></span>
            </label>
        `;
    }

    // ========================================
    // UI Helpers
    // ========================================

    toggleRow(rowId) {
        const wrapper = document.querySelector(`[data-row-id="${rowId}"]`);
        if (wrapper) {
            const wasExpanded = wrapper.classList.contains('expanded');
            wrapper.classList.toggle('expanded');

            // Callback aufrufen wenn Zeile aufgeklappt wird
            if (!wasExpanded && this.options.onRowExpanded) {
                const detailElement = wrapper.querySelector('.table-row-detail');
                this.options.onRowExpanded(rowId, detailElement);
            }
        }
    }

    getSortIcon(field) {
        if (this.sortColumn !== field) {
            return '<span class="sort-icon">⇅</span>';
        }
        return this.sortDirection === 'asc'
            ? '<span class="sort-icon active">↑</span>'
            : '<span class="sort-icon active">↓</span>';
    }

    showLoading() {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = '<div class="table-loading"><div class="spinner"></div></div>';
        }
    }

    showError(message) {
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = `<div class="table-error">${this.escapeHtml(message)}</div>`;
        }
    }

    escapeHtml(text) {
        if (text === null || text === undefined) return '';
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    }

    // ========================================
    // Public API
    // ========================================

    getData() {
        return this.filteredData;
    }

    getAllData() {
        return this.data;
    }

    getStats() {
        return this.stats;
    }

    getStanceCounts() {
        return this.stanceCounts;
    }

    getV2StanceCounts() {
        return this.v2StanceCounts;
    }

    getPagination() {
        return this.pagination;
    }

    refresh() {
        this.loadData(this.lastParams || {});
    }

    /**
     * Setzt Pagination auf Seite 1 zurück (für externe Filter-Änderungen)
     */
    resetPagination() {
        if (this.pagination.enabled) {
            this.pagination.currentPage = 1;
        }
    }
}

// Globale Registry für Tabellen (für onclick-Handler)
window.tables = window.tables || {};

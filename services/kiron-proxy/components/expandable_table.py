"""
Generische aufklappbare Tabellen-Komponente

Verwendung:
    table = ExpandableTable(
        id="articles",
        columns=[
            Column("Zeit", "matched_time", width=115, sortable=True),
            Column("Ticker", "ticker", width=70, sortable=True, css_class="ticker"),
            ...
        ],
        detail_fields=[
            DetailField("Veröffentlicht", "published_at"),
            DetailField("Begründung", "rationale", section=True),
            ...
        ],
        default_sort=("matched_time", "desc"),
    )

    # In Template:
    {{ table.to_json() }}
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json


@dataclass
class Column:
    """Spalten-Definition für die Tabelle."""

    label: str  # Anzeigename im Header
    field: str  # Feldname in den Daten
    width: int = 100  # Breite in Pixeln
    sortable: bool = True  # Sortierbar?
    css_class: str = ""  # Zusätzliche CSS-Klasse
    renderer: str = ""  # Spezieller Renderer (z.B. "icon", "percent", "badge")
    renderer_options: dict = field(default_factory=dict)  # Optionen für Renderer
    align: str = "left"  # Ausrichtung: left, center, right
    truncate: int = 0  # Text nach X Zeichen abschneiden (0 = nicht)
    tooltip_field: str = ""  # Feld für Tooltip (wenn truncate)
    info: str = ""  # Info-Text für Spaltenüberschrift (Tooltip bei Hover)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "field": self.field,
            "width": self.width,
            "sortable": self.sortable,
            "cssClass": self.css_class,
            "renderer": self.renderer,
            "rendererOptions": self.renderer_options,
            "align": self.align,
            "truncate": self.truncate,
            "tooltipField": self.tooltip_field or self.field,
            "info": self.info,
        }


@dataclass
class DetailField:
    """Feld-Definition für den aufgeklappten Bereich."""

    label: str  # Anzeigename
    field: str  # Feldname in den Daten
    renderer: str = ""  # Spezieller Renderer
    renderer_options: dict = field(default_factory=dict)
    section: bool = False  # Als eigene Sektion darstellen?
    section_type: str = "text"  # text, quote, code

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "field": self.field,
            "renderer": self.renderer,
            "rendererOptions": self.renderer_options,
            "section": self.section,
            "sectionType": self.section_type,
        }


@dataclass
class Filter:
    """Filter-Definition."""

    id: str  # HTML-ID
    label: str  # Anzeigename
    type: str  # select, input, buttons
    field: str  # Feld zum Filtern
    options: list = field(default_factory=list)  # Für select/buttons
    placeholder: str = ""  # Für input
    default: str = ""  # Standardwert
    wildcard: bool = False  # Wildcard-Suche?
    wildcard_fields: list = field(default_factory=list)  # Zusätzliche Felder für Wildcard

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "field": self.field,
            "options": self.options,
            "placeholder": self.placeholder,
            "default": self.default,
            "wildcard": self.wildcard,
            "wildcardFields": self.wildcard_fields,
        }


@dataclass
class PaginationConfig:
    """Pagination-Konfiguration für Tabellen."""

    enabled: bool = False  # Pagination aktiviert?
    options: List[int] = field(default_factory=lambda: [20, 100, 300, 1000])  # Page-Size Optionen
    default_page_size: int = 100  # Standard-Seitengröße

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "options": self.options,
            "defaultPageSize": self.default_page_size,
        }


@dataclass
class ExpandableTable:
    """Konfiguration für eine aufklappbare Tabelle."""

    id: str  # Eindeutige ID
    columns: List[Column]  # Spalten-Definitionen
    api_endpoint: str  # API-Endpunkt für Daten

    # Optional: Aufklapp-Funktion
    expandable: bool = False  # Aufklappbar?
    detail_fields: List[DetailField] = field(default_factory=list)  # Felder im Detail
    detail_link_field: str = ""  # Feld für "Öffnen"-Link (z.B. "url")
    detail_link_label: str = "🔗 Öffnen"  # Label für Link
    detail_text_field: str = ""  # Feld für langen Text (z.B. "full_text")
    detail_text_label: str = ""  # Label für Text-Sektion
    detail_text_max_length: int = 3000  # Max. Länge für Text

    # Sortierung
    default_sort: Tuple[str, str] = ("", "asc")  # (field, direction)

    # Filter
    filters: List[Filter] = field(default_factory=list)

    # Pagination
    pagination: PaginationConfig = field(default_factory=PaginationConfig)

    # Darstellung
    row_height: int = 40  # Zeilenhöhe in Pixeln
    show_header: bool = True  # Header anzeigen?
    sticky_header: bool = True  # Sticky Header?

    def __post_init__(self):
        # Wenn detail_fields gesetzt sind, ist die Tabelle aufklappbar
        if self.detail_fields or self.detail_text_field:
            self.expandable = True

    def get_grid_template(self) -> str:
        """Generiert das CSS Grid Template für die Spalten."""
        widths = []
        if self.expandable:
            widths.append("30px")  # Expand-Icon
        for col in self.columns:
            if col.width == 0:
                widths.append("1fr")  # Flex
            else:
                widths.append(f"{col.width}px")
        return " ".join(widths)

    def to_dict(self) -> dict:
        """Konvertiert die Konfiguration zu einem Dictionary für JSON."""
        return {
            "id": self.id,
            "columns": [col.to_dict() for col in self.columns],
            "apiEndpoint": self.api_endpoint,
            "expandable": self.expandable,
            "detailFields": [f.to_dict() for f in self.detail_fields],
            "detailLinkField": self.detail_link_field,
            "detailLinkLabel": self.detail_link_label,
            "detailTextField": self.detail_text_field,
            "detailTextLabel": self.detail_text_label,
            "detailTextMaxLength": self.detail_text_max_length,
            "defaultSort": {
                "field": self.default_sort[0],
                "direction": self.default_sort[1]
            },
            "filters": [f.to_dict() for f in self.filters],
            "pagination": self.pagination.to_dict(),
            "gridTemplate": self.get_grid_template(),
            "rowHeight": self.row_height,
            "showHeader": self.show_header,
            "stickyHeader": self.sticky_header,
        }

    def to_json(self) -> str:
        """Konvertiert die Konfiguration zu JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

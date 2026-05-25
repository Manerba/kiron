"""
KPI-Card Komponenten für Dashboard

Dataclasses für:
- KPICard: Einzelne KPI-Karte (Wert, Icon, Formatierung)
- KPISection: Gruppe von KPI-Karten mit Titel
- WidgetColumn: Spalte einer Widget-Tabelle
- Widget: Generisches Dashboard-Widget
- WidgetDashboardConfig: Dashboard-Konfiguration mit Widgets
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class KPICard:
    """Einzelne KPI-Karte."""

    id: str
    title: str                          # "BUY-Signale"
    value_field: str                    # Feldname in API-Response
    icon: str = ""                      # Emoji (z.B. "🟢")
    subtitle_field: str = ""            # Optional: zusätzliches Feld unter dem Wert
    format: str = "number"              # number, percent, currency, text
    color_field: str = ""               # Feld das Farbe bestimmt
    color_rules: Dict[str, str] = field(default_factory=dict)  # {">0": "green", "<0": "red"}
    link_tab: str = ""                  # Klick-Navigation zu Tab

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "valueField": self.value_field,
            "icon": self.icon,
            "subtitleField": self.subtitle_field,
            "format": self.format,
            "colorField": self.color_field,
            "colorRules": self.color_rules,
            "linkTab": self.link_tab,
        }


@dataclass
class KPISection:
    """Gruppe von KPI-Karten (z.B. pro Scoring-Variante)."""

    id: str
    title: str                          # "baseline" oder "Predictions (heute)"
    cards: List[KPICard] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "type": "cards",
            "cards": [card.to_dict() for card in self.cards],
        }


@dataclass
class KPITableRow:
    """Einzelne Zeile in einer KPI-Tabelle."""

    label: str                          # "BUY", "WATCH", etc.
    value_field: str                    # Feldname in API-Response
    icon: str = ""                      # Emoji (z.B. "🟢")
    format: str = "number"              # number, percent, currency, text
    color_rules: Dict[str, str] = field(default_factory=dict)  # {">0": "green", "<0": "red"}
    tooltip: str = ""                   # Hover-Text

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "valueField": self.value_field,
            "icon": self.icon,
            "format": self.format,
            "colorRules": self.color_rules,
            "tooltip": self.tooltip,
        }


@dataclass
class KPITableSection:
    """Tabellen-Darstellung von KPIs (kompakter als einzelne Karten)."""

    id: str
    title: str                          # "Predictions (heute)"
    rows: List[KPITableRow] = field(default_factory=list)
    columns: List[str] = field(default_factory=lambda: ["Kennzahl", "Wert"])  # Spaltenüberschriften

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "type": "table",
            "columns": self.columns,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass
class WidgetColumn:
    """Spalte einer Widget-Tabelle."""

    key: str              # Feld-Key in den Daten
    label: str            # Spaltenüberschrift
    format: str = "text"  # text, percent, number
    color_rules: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "format": self.format,
            "colorRules": self.color_rules,
        }


@dataclass
class Widget:
    """Generisches Dashboard-Widget."""

    id: str
    title: str
    type: str               # "table" oder "kpi"
    api_endpoint: str
    subtitle: str = ""
    columns: List[WidgetColumn] = field(default_factory=list)  # Für type="table"
    cards: List[KPICard] = field(default_factory=list)          # Für type="kpi"

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "type": self.type,
            "apiEndpoint": self.api_endpoint,
        }
        if self.type == "table":
            result["columns"] = [col.to_dict() for col in self.columns]
        else:
            result["cards"] = [card.to_dict() for card in self.cards]
        return result


@dataclass
class WidgetDashboardConfig:
    """Dashboard-Konfiguration mit Widgets."""

    widgets: List[Widget]

    def to_dict(self) -> dict:
        return {
            "widgets": [w.to_dict() for w in self.widgets],
        }

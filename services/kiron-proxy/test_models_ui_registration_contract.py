from __future__ import annotations

from pathlib import Path

import app as app_module


ROOT = Path(__file__).resolve().parent
JS = ROOT / "static" / "js" / "tab_models.js"
CORE = ROOT / "static" / "js" / "app_core.js"
CSS = ROOT / "static" / "css" / "models.css"
EXPANDABLE_CSS = ROOT / "static" / "css" / "expandable_table.css"
TEMPLATE = ROOT / "templates" / "index.html"


def test_models_surface_has_one_view_and_no_available_or_pull_contracts() -> None:
    js = JS.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    route_paths = {
        route.path for route in app_module.app.routes if hasattr(route, "path")
    }

    assert "initModelsTab" in js
    for removed in (
        "initModelsInstalled",
        "initModelsAvailable",
        "/api/models/available",
        "/api/models/pull",
        "/api/models/pull/status",
        "checkActivePulls",
        "startPullModel",
    ):
        assert removed not in js
    assert "SUBTABS" not in core
    assert "path.split('/')" not in core
    assert "const tab = tabMap[path] ? path : 'dashboard';" in core
    assert "subTabs" not in template
    assert "models-pull" not in css
    assert "/api/models/available" not in route_paths
    assert "/api/models/pull" not in route_paths
    assert "/api/models/pull/status" not in route_paths
    assert not hasattr(app_module, "_active_pulls")


def test_registration_dialog_is_accessible_and_uses_text_only_server_rendering() -> None:
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    for contract in (
        'class="models-registration-overlay"',
        'role="dialog"',
        'aria-modal="true"',
        'aria-labelledby="modelRegistrationTitle"',
        'id="modelRegistrationProvider"',
        'id="modelRegistrationReference"',
        'id="modelRegistrationCandidateStatus"',
        'id="modelRegistrationLoader"',
        'role="alert"',
        "closeRegistrationOverlay",
        "registrationOverlayKeydown",
        "registrationDialogReturnFocus.focus()",
        "'/api/models/registration-candidates'",
        "document.createElement('option')",
        "option.textContent",
        "errorNode.textContent",
    ):
        assert contract in js
    assert "position: fixed" in css
    assert ".models-registration-overlay[hidden]" in css
    assert "overscroll-behavior: contain" in css
    assert "<dialog" not in js
    assert "showModal()" not in js
    assert 'method="dialog"' not in js
    assert "onclick=" not in js
    assert "innerHTML = data" not in js
    assert "insertAdjacentHTML" not in js


def test_registration_dialog_uses_discovered_candidates_not_free_text() -> None:
    js = JS.read_text(encoding="utf-8")

    assert '<select id="modelRegistrationReference" required>' in js
    assert '<input id="modelRegistrationReference"' not in js
    assert "loadRegistrationCandidates" in js
    assert "candidate.runtime_provider === provider.value" in js
    assert "Keine unregistrierten lokalen Modelle gefunden" in js
    assert "candidate_id: referenceNode.value" in js


def test_static_and_dynamic_origins_are_visible_in_the_same_table() -> None:
    js = JS.read_text(encoding="utf-8")

    assert "Catalog" in js
    assert "Lokal registriert" in js
    assert "catalog_managed" in js
    assert "locally_registered" in js
    assert "Herkunft" in js


def test_models_table_uses_dashboard_expandable_standard_and_full_width() -> None:
    js = JS.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")

    for contract in (
        "new ExpandableTable",
        "expandable: true",
        "tables.models = _modelsTable",
        "filterBar.style.display = 'flex'",
        "models-control-group",
        "+ Modell registrieren",
        "models-table-viewport",
        "models-name",
        "Token &amp; Kontext",
        "Identität",
        "Runtime",
        "Native Modellgrenze",
        "Aktiver Ollama-Runner",
        "Catalog-Limits je Profil/Rolle",
        "native_context_length",
        "runtime_context_length",
        "catalog_context_length",
        "catalog_token_limits",
    ):
        assert contract in js

    assert "document.createElement('table')" not in js
    assert "models-toolbar" not in js
    assert "max-width: 1500px" not in css
    assert ".models-page {\n    width: 100%;" in css
    assert ".models-name {" in css
    assert "font-weight: 400;" in css


def test_models_rows_match_dashboard_density_and_compact_toggle() -> None:
    css = CSS.read_text(encoding="utf-8")
    expandable_css = EXPANDABLE_CSS.read_text(encoding="utf-8")

    assert ".models-actions .action-btn {\n    padding: 2px 6px;" in css
    assert ".models-badge {\n    display: inline-block;\n    padding: 2px 8px;" in css
    for contract in (
        '[data-density="compact"] .table-header {',
        '[data-density="compact"] .table-body {',
        '[data-density="compact"] .table-row {',
        '[data-density="compact"] .detail-content {',
        '[data-density="compact"] .detail-fields {',
    ):
        assert contract in expandable_css
    assert 'padding: 3px 12px;' in expandable_css
    assert 'gap: 2px;' in expandable_css


def test_registration_uses_explicit_runtime_and_artifact_identity():
    js = JS.read_text(encoding="utf-8")
    assert 'candidate_id: referenceNode.value' in js
    assert 'runtime_provider: providerNode.value' not in js
    assert 'X-Kiron-Action' in js
    assert 'modelRegistrationProjector' in js
    assert 'modelRegistrationProfile' in js
    assert 'candidate.artifact_origin' in js
    assert 'candidate.artifact_format' in js
    assert '_registrationErrors[provider.value]' in js
    assert 'option.disabled = option.hidden' in js
    assert '<option value="huggingface">' not in js

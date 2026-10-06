from __future__ import annotations

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parent / "static/js/tab_models.js"
).read_text(encoding="utf-8")


def test_managed_actions_and_live_state_use_exact_backend_identity() -> None:
    assert "if (m.backend !== 'ollama') continue;" in SOURCE
    assert "if (m.backend !== 'kiron_embeddings') continue;" in SOURCE
    assert "if (m.backend !== 'kiron_deberta') continue;" in SOURCE
    assert "embedActiveModels.has(m.name)" in SOURCE
    assert "m.name === embedLoading" in SOURCE
    assert "loadedModels.has(m.name)" in SOURCE


def test_control_actions_use_server_identity_without_provider_specific_requests() -> None:
    assert "fetch('/api/models/control/action'" in SOURCE
    assert "model: model.control_id" in SOURCE
    assert "revision: model.control_revision" in SOURCE
    assert "model.controls.find" in SOURCE
    for removed in ("/api/models/load", "/api/models/delete", "/api/embedding/load", "runtime/action", "native_actions"):
        assert removed not in SOURCE


def test_operational_state_configuration_and_evidence_are_separate() -> None:
    for label in ('Betriebszustand', 'Beobachtete Konfiguration', 'Nachweisstatus',
                  'Nachweise abweichend', 'Status unbekannt', 'Geladen'):
        assert label in SOURCE
    assert "states[model.operation_state]" in SOURCE
    assert "field: 'operation_state', label: 'Status'" in SOURCE

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


def test_service_load_actions_do_not_derive_a_basename() -> None:
    load_start = SOURCE.index("async function loadEmbeddingModel(name)")
    ws_start = SOURCE.index("function updateModelsFromWS(data)")
    load_source = SOURCE[load_start:ws_start]

    assert ".split(':')[0].split('/').pop()" not in load_source
    assert "JSON.stringify({model: name})" in load_source


def test_configured_installed_and_runtime_states_are_visibly_distinct() -> None:
    for label in (
        "Konfiguriert, nicht installiert",
        "Installiert, nicht geladen",
        "Wird geladen",
        "Geladen (Service)",
        "Load-Status unbekannt",
    ):
        assert label in SOURCE

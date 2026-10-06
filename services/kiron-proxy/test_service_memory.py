"""Service measurements reach model rows without inventing model allocations."""

import asyncio
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import psutil
import pytest

import metrics
from service_memory import build_service_memory
from test_model_discovery import _health, _payload


MODEL = "bge-reranker-v2-m3"


def process(**changes):
    return {"pid": 4711, "service": "kiron-deberta", "label": "deberta",
            "vram_mb": 1976.0, "vram_state": "known", "rss_bytes": 1806912 * 1024, **changes}


def snapshot(*, processes=None, **health_changes):
    return build_service_memory(
        {"state": "ok", "data": [process()] if processes is None else processes},
        embedding=None, deberta={**_health(loaded=(MODEL,)), **health_changes},
    )


def test_live_measurement_shape_reaches_loaded_inventory_row():
    memory = snapshot()
    rows = _payload(deberta=_health(loaded=(MODEL,)), service_memory=memory)["models"]
    row = next(row for row in rows if row["name"] == MODEL)
    assert row["loaded"] is True
    assert row["vram_gb"] is None and row["ram_gb"] is None
    assert row["service_memory"] == {
        "vram_gb": 1.93, "ram_gb": 1.72, "loaded_models": [MODEL], "process_count": 1,
    }
    assert all(row["service_memory"] is None for row in rows if row["name"] != MODEL)


def test_dashboard_inventory_receives_service_memory_from_metrics_cache(monkeypatch):
    import app as dashboard
    from test_dashboard_colbert import _FakeOllamaClient

    memory = snapshot()
    monkeypatch.setattr(dashboard.httpx, "AsyncClient", _FakeOllamaClient)
    monkeypatch.setattr(dashboard, "get_cached_payload", lambda: {"system": {
        "deberta": _health(loaded=(MODEL,)), "service_memory": memory,
    }})
    monkeypatch.setattr(dashboard, "service_huggingface_inventory", lambda *a, **k: frozenset())
    monkeypatch.setattr(dashboard, "get_model_registration_service", lambda: SimpleNamespace(list_models=lambda: ()))
    payload = asyncio.run(dashboard.get_local_models())
    row = next(row for row in payload["models"] if row["name"] == MODEL)
    assert row["loaded"] is True
    assert row["service_memory"] == memory["kiron_deberta"]
    assert row["vram_gb"] is None and row["ram_gb"] is None


def test_shared_embedding_memory_is_explicit_and_not_assigned_per_model():
    names = ["mankei-326m-embedder", "nomic-embed-text"]
    health = _health(loaded=names)
    memory = build_service_memory(
        {"state": "ok", "data": [process(service="kiron-embeddings"), process(service="kiron-embeddings", pid=4712)]},
        embedding=health, deberta=None,
    )
    rows = _payload(embedding=health, service_memory=memory)["models"]
    loaded = [row for row in rows if row["name"] in names]
    assert len(loaded) == 2
    for row in loaded:
        assert row["vram_gb"] is None and row["ram_gb"] is None
        assert row["service_memory"]["vram_gb"] == 3.86
        assert row["service_memory"]["loaded_models"] == names
        assert row["service_memory"]["process_count"] == 2


@pytest.mark.parametrize("probe", [None, {"state": "timeout", "data": [process()]},
                                  {"state": "empty", "data": []}, {"state": "ok", "data": [None]}])
def test_failed_or_empty_probe_cannot_invent_zero_memory(probe):
    assert build_service_memory(probe, embedding=None, deberta=_health(loaded=(MODEL,))) == {}


@pytest.mark.parametrize("changes", [{"running": False}, {"loading_model": "next"}, {"status": "loading"},
                                    {"loaded_models": []}, {"loaded_models": [MODEL, MODEL]},
                                    {"loaded_models": [None]}])
def test_unknown_or_changing_service_cannot_claim_model_memory(changes):
    assert snapshot(**changes) == {}


@pytest.mark.parametrize("processes", [[], [process(service=None)], [process(pid=True)],
                                     [process(), process()], [process(service="test-kiron-deberta")]])
def test_unowned_or_ambiguous_processes_are_not_attributed(processes):
    assert snapshot(processes=processes) == {}


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_invalid_vram_keeps_ram_measurement_and_unknown_vram(value):
    memory = snapshot(processes=[process(vram_mb=value)])["kiron_deberta"]
    assert memory["vram_gb"] is None
    assert memory["ram_gb"] == 1.72


def test_partial_rss_does_not_turn_into_an_underreported_total():
    memory = snapshot(processes=[process(), process(pid=4712, rss_bytes=None)])["kiron_deberta"]
    assert memory["vram_gb"] == 3.86 and memory["ram_gb"] is None
    zero = snapshot(processes=[process(vram_mb=0, rss_bytes=0)])["kiron_deberta"]
    assert zero["vram_gb"] == zero["ram_gb"] == 0


def test_loaded_service_without_measurement_is_unknown():
    rows = _payload(deberta=_health(loaded=(MODEL,)))["models"]
    row = next(row for row in rows if row["name"] == MODEL)
    assert row["vram_gb"] is None and row["ram_gb"] is None
    assert row["service_memory"] is None
    stale = snapshot()
    stale["kiron_deberta"]["loaded_models"] = ["another-model"]
    row = next(row for row in _payload(deberta=_health(loaded=(MODEL,)), service_memory=stale)["models"]
               if row["name"] == MODEL)
    assert row["service_memory"] is None


def test_gpu_probe_adds_exact_service_and_rss(monkeypatch):
    monkeypatch.setattr(metrics.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="4711, /usr/bin/python, 1976 MiB", stderr=""))
    monkeypatch.setattr(metrics, "_read_proc_cmdline", lambda pid: "python")
    monkeypatch.setattr(metrics.psutil, "Process", lambda pid: SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=1806912 * 1024)))
    with patch("builtins.open", mock_open(read_data="0::/system.slice/kiron-deberta.service\n")):
        probe = metrics.get_gpu_process_metrics()
    assert probe["data"][0]["service"] == "kiron-deberta"
    assert probe["data"][0]["rss_bytes"] == 1806912 * 1024
    assert build_service_memory(probe, embedding=None, deberta=_health(loaded=(MODEL,)))["kiron_deberta"]["vram_gb"] == 1.93


def test_process_read_errors_keep_measurements_unknown(monkeypatch):
    def denied(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(metrics.psutil, "Process", denied)
    with patch("builtins.open", mock_open(read_data="0::/system.slice/kiron-deberta.service\n")):
        assert metrics._gpu_process_host_info(4711) == {"service": "kiron-deberta", "rss_bytes": None}
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert metrics._gpu_process_host_info(4711) == {"service": None, "rss_bytes": None}
    with patch("builtins.open", mock_open(read_data="0::/system.slice/test-kiron-deberta.service\n")):
        assert metrics._gpu_process_host_info(4711) == {"service": None, "rss_bytes": None}


def test_frontend_displays_scope_partial_values_and_refreshes_runtime_memory():
    program = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = vm.createContext({console, currentTab:'dashboard'});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const model = {name:'bge-reranker-v2-m3', backend:'kiron_deberta', catalog_managed:true,
    operation_state:'loaded', load_state:'unknown', runtime:true, vram_gb:null, ram_gb:null,
    configuration:{vram_gb:null, ram_gb:null}, service_memory:{vram_gb:1.93, ram_gb:1.72,
    loaded_models:['bge-reranker-v2-m3'], process_count:1}};
assert.equal(context.memoryLabel(model), '1.93 / 1.72 (Dienst)');
assert.ok(context.memoryScopeLabel(model).includes('CUDA-Kontext'));
model.service_memory.loaded_models.push('another');
assert.equal(context.memoryLabel(model), '1.93 / 1.72 (Dienst, 2 Modelle)');
assert.ok(context.memoryScopeLabel(model).includes('insgesamt einmal'));
model.service_memory.vram_gb = null;
assert.equal(context.memoryLabel(model), '— / 1.72 (Dienst, 2 Modelle)');
assert.equal(context.memoryLabel({...model, operation_state:'unknown'}), 'Unbekannt');
assert.equal(context.memoryLabel({...model, service_memory:null}), 'Unbekannt');
assert.equal(context.memoryLabel({operation_state:'loaded', vram_gb:0, ram_gb:4}), '0.00 / 4.00');
context.fixture = model;
vm.runInContext('_modelsData = [fixture]', context);
context.updateModelsFromWS({system:{service_memory:{kiron_deberta:{vram_gb:2.1, ram_gb:1.8,
    loaded_models:[model.name], process_count:1}}}});
assert.equal(context.memoryLabel(model), '2.10 / 1.80 (Dienst)');
assert.equal(model.load_state, 'unknown');
assert.equal(model.configuration.vram_gb, null);
context.updateModelsFromWS({system:{service_memory:{kiron_deberta:{vram_gb:9, ram_gb:9,
    loaded_models:['different'], process_count:1}}}});
assert.equal(model.service_memory, null);
context.updateModelsFromWS({system:{}});
assert.equal(context.memoryLabel(model), 'Unbekannt');
'''
    subprocess.run(["node", "-e", program, str(Path(__file__).parent / "static/js/tab_models.js")], check=True)

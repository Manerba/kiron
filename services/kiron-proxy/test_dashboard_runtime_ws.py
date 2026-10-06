"""Execute the unified inventory reader and WS caller without network or a browser."""
from pathlib import Path
import subprocess

import pytest


JS = Path(__file__).parent / "static" / "js"


@pytest.mark.parametrize("backend", ["ollama", "kiron_embeddings"])
def test_native_websocket_preserves_complete_runtime_observation(backend):
    program = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const backend = process.argv[3];
const status = {textContent: ''};
const context = vm.createContext({console, document: {
    getElementById: id => id === 'modelsRuntimeStatus' ? status : null, addEventListener: () => {},
}});
for (const path of process.argv.slice(1, 3)) {
    vm.runInContext(fs.readFileSync(path, 'utf8'), context);
}
vm.runInContext("currentTab = 'models'", context);
let renders = 0;
context.renderModelsTable = () => { renders++; };
const native = [
    {name: 'registered', backend, deployment_ids: ['bound'], installed: true},
    {name: 'native-only', backend, installed: true},
];
let observed = {
    name: 'public-alias', backend, runtime: true, registry_ids: [],
    deployment_ids: ['bound'], loaded: false, loading: false, load_state: 'unknown',
    runtime_state: 'unknown', operation_state: 'unknown', provider_health: 'available', error_code: 'conflict',
    diagnostic: 'Configuration and generation differ', actions: ['health'],
    snapshot_revision: 'snapshot-1', observed_at: '2026-09-22T00:00:00Z',
    vram_gb: null, ram_gb: null, runtime_context_length: null,
};
context.fetch = async url => {
    assert.equal(url, '/api/models/control');
    return {ok: true, json: async () => ({models: [
        {...native[0], ...observed, name: 'registered', control_id: 'opaque-bound', controls: []}, native[1]
    ]})};
};
const call = code => vm.runInContext(code, context);
const row = index => JSON.parse(call(`JSON.stringify(_modelsData[${index}])`));
function metrics(loaded, known = true) {
    const names = loaded ? ['registered', 'native-only'] : [];
    return backend === 'ollama'
        ? {system: {ollama: {models_loaded_state: known ? 'known' : 'unknown',
            models_loaded: names.map(name => ({name, vram_gb: 6, size_gb: 7,
                context_length: 4096, expires_at: 'later'}))}}}
        : {system: {embedding: {running: known, loaded_models: names,
            loading_model: loaded ? null : 'registered'}}};
}
function send(loaded, known = true) {
    context.message = {type: 'metrics', data: metrics(loaded, known)};
    call('handleWSMessage(message)');
}
(async () => {
    await call('fetchLocalModels()');
    const conflict = row(0);
    assert.equal(conflict.control_id, 'opaque-bound'); // Server-projected identity.
    assert.equal(conflict.name, 'registered');
    assert.equal(call('statusLabel(_modelsData[0])'), 'Status unbekannt');
    send(true);
    assert.deepEqual(row(0), conflict); // Includes actions, diagnosis, metrics and identity.
    assert.equal(row(1).loaded, true);
    assert.equal(row(1).load_state, 'loaded');
    assert.equal(call('statusLabel(_modelsData[0])'), 'Status unbekannt');

    // A fresh complete runtime observation is still applied through its own source.
    observed = {...observed, loaded: true, load_state: 'loaded', runtime_state: 'loaded', operation_state: 'loaded',
        error_code: null, diagnostic: null, actions: ['health', 'unload'],
        snapshot_revision: 'snapshot-2', observed_at: '2026-09-22T00:01:00Z'};
    await call('fetchLocalModels()');
    const confirmed = row(0);
    assert.equal(confirmed.snapshot_revision, 'snapshot-2');
    assert.equal(call('statusLabel(_modelsData[0])'), 'Geladen');
    send(false);
    assert.deepEqual(row(0), confirmed);
    assert.equal(row(1).loaded, false);
    assert.equal(row(1).load_state, 'unloaded');
    send(false, false);
    assert.deepEqual(row(0), confirmed);
    assert.equal(row(1).load_state, 'unknown');
    assert.equal(renders, 3); // Real app_core WS dispatcher reaches the models tab.
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run(
        ["node", "-e", program, str(JS / "tab_models.js"),
         str(JS / "app_core.js"), backend], check=True, timeout=5,
    )


def test_registered_models_refresh_residency_memory_and_controls_from_server():
    program = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
let now = 0, calls = 0, release = null;
const table = {}, status = {textContent: ''};
const context = vm.createContext({console, currentTab: 'models', Date: {now: () => now},
    document: {getElementById: id => id === 'modelsTableContainer' ? table
        : id === 'modelsRuntimeStatus' ? status : null},
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
context.renderModelsTable = () => {};
const call = code => vm.runInContext(code, context);
const model = {name: 'qwen3:8b', backend: 'ollama', runtime: true,
    control_id: 'original', control_revision: 'old', installed: true,
    operation_state: 'unloaded', load_state: 'unknown', vram_gb: 0, ram_gb: 0,
    configuration: {}, error_code: 'conflict', verification: {status: 'unverified'},
    controls: [{id: 'load', enabled: true}, {id: 'unload', enabled: false}]};
let observed = [model];
context.fetch = async url => {
    assert.equal(url, '/api/models/control');
    calls++;
    if (release) await release;
    return {ok: true, json: async () => ({models: structuredClone(observed)})};
};
const metrics = {system: {ollama: {models_loaded_state: 'known', models_loaded: [
    {name: 'qwen3:8b-cpu33-test', size_gb: 8.5, vram_gb: 5.21},
]}}};
async function tick() {
    context.updateModelsFromWS(metrics);
    const work = call('_modelsRefreshPending');
    if (work) await work;
    await Promise.resolve();
}
(async () => {
    await call('loadLocalModels()');
    assert.equal(calls, 1);
    now = 9999;
    await tick();
    assert.equal(calls, 1); // Metrics never replace the complete observation.
    assert.equal(call('_modelsData.length'), 1);
    const partial = {...model, name: 'qwen3:8b-cpu33-test', control_id: 'test',
        control_revision: 'current', operation_state: 'loaded', vram_gb: 5.21, ram_gb: 3.29,
        configuration: {device: 'gpu', context_tokens: 24576, vram_gb: 5.21, ram_gb: 3.29},
        controls: [{id: 'load', enabled: false}, {id: 'unload', enabled: true}]};
    observed = [model, partial]; // Registration and load happened outside this browser.
    now = 10000;
    let unblock;
    release = new Promise(resolve => {unblock = resolve;});
    context.updateModelsFromWS(metrics);
    context.updateModelsFromWS(metrics);
    assert.equal(calls, 2); // One in-flight refresh, even with more metrics.
    assert.equal(call('_modelsData.length'), 1);
    unblock();
    await call('_modelsRefreshPending');
    await Promise.resolve();
    release = null;
    assert.equal(call('_modelsData.length'), 2);
    assert.equal(call('statusLabel(_modelsData[1])'), 'Geladen');
    assert.equal(call('memoryLabel(_modelsData[1])'), '5.21 / 3.29');
    assert.equal(call('configurationLabel(_modelsData[1])'), 'GPU + CPU · Kontext 24576');
    assert.equal(call('_modelsData[1].error_code'), 'conflict');
    assert.equal(call('_modelsData[1].verification.status'), 'unverified');
    assert.equal(call('_modelsData[1].controls[1].enabled'), true);

    // Same loaded model, changed context and memory: also refreshed.
    observed[1] = {...partial, vram_gb: 4.4, ram_gb: 2.91,
        configuration: {...partial.configuration, context_tokens: 16384, vram_gb: 4.4, ram_gb: 2.91}};
    now = 20000;
    await tick();
    assert.equal(calls, 3);
    assert.equal(call('memoryLabel(_modelsData[1])'), '4.40 / 2.91');
    assert.equal(call('_modelsData[1].configuration.context_tokens'), 16384);
    context.currentTab = 'dashboard';
    now = 30000;
    await tick();
    assert.equal(calls, 3); // No background-tab polling.

    context.currentTab = 'models';
    observed = [{...model, operation_state: 'unknown', vram_gb: null, ram_gb: null}];
    await tick();
    assert.equal(calls, 4);
    assert.equal(call('statusLabel(_modelsData[0])'), 'Status unbekannt');
    assert.equal(call('memoryLabel(_modelsData[0])'), 'Unbekannt');
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    subprocess.run(["node", "-e", program, str(JS / "tab_models.js")], check=True, timeout=5)

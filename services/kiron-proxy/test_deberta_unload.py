"""DeBERTa unload transport and the model target sent by the System tab."""
from pathlib import Path
import subprocess
import unittest
from unittest import mock

import httpx

import app


class DebertaUnloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.post = mock.AsyncMock(return_value=httpx.Response(200, json={
            "status": "unloaded", "model": "bge-reranker-v2-m3", "already": False}))
        client = mock.AsyncMock()
        client.__aenter__.return_value = client
        client.post = self.post
        patch = mock.patch.object(app.httpx, "AsyncClient", return_value=client)
        patch.start()
        self.addCleanup(patch.stop)

    async def test_alias_is_forwarded_without_allocating_or_using_gpu_gate(self):
        with mock.patch.object(app.vram_lease, "gpu_gate_decision", side_effect=AssertionError("allocation gate")), \
             mock.patch.object(app.vram_lease, "write_overlay_marker", side_effect=AssertionError("allocation marker")):
            response = await app.unload_deberta_model({"model": "BAAI/bge-reranker-v2-m3"})
        self.assertEqual(response.status_code, 200)
        self.post.assert_awaited_once_with(app.DEBERTA_UNLOAD_URL, json={"model": "BAAI/bge-reranker-v2-m3"})

    async def test_missing_invalid_or_unrelated_target_never_calls_service(self):
        for body in ({}, {"model": None}, {"model": "unknown"}, {"model": "mankei-326m-embedder"},
                     {"model": "bge-reranker-v2-m3", "force": True}):
            response = await app.unload_deberta_model(body)
            self.assertEqual(response.status_code, 400)
        self.post.assert_not_awaited()

    async def test_success_must_confirm_exact_target(self):
        for payload in ({"status": "unloaded"}, {"status": "unloaded", "model": "other", "already": False},
                        {"status": "unloaded", "model": "bge-reranker-v2-m3", "already": 0}, []):
            self.post.return_value = httpx.Response(200, json=payload)
            response = await app.unload_deberta_model({"model": "bge-reranker-v2-m3"})
            self.assertEqual(response.status_code, 502)
        self.post.return_value = httpx.Response(200, text="invalid JSON")
        self.assertEqual((await app.unload_deberta_model({"model": "bge-reranker-v2-m3"})).status_code, 502)

    async def test_backend_errors_and_timeout_are_reported(self):
        for status in (409, 500, 503):
            self.post.return_value = httpx.Response(status, json={"error": "fixture"})
            response = await app.unload_deberta_model({"model": "bge-reranker-v2-m3"})
            self.assertEqual(response.status_code, status)
        for error, status in ((httpx.ConnectError("fixture"), 503), (httpx.ReadTimeout("fixture"), 504)):
            self.post.side_effect = error
            response = await app.unload_deberta_model({"model": "bge-reranker-v2-m3"})
            self.assertEqual(response.status_code, status)


def test_system_tab_sends_the_model_shown_on_its_unload_button():
    source = Path(__file__).with_name("static") / "js/tab_system.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const calls = [];
const context = vm.createContext({
    escapeHtml: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('"', '&quot;')
        .replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    showNotification: () => {},
    fetch: async (url, options) => {
        calls.push({url, options});
        return {ok: true, json: async () => ({status: 'unloaded'})};
    },
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
async function check() {
    const html = context.renderSystemMetrics({gpu: {vram_total_mb: 12288},
        deberta: {running: true, model: 'bge-reranker-v2-m3'}});
    const button = html.match(/<button[^>]*data-kpi="gpu_mgr_deberta_model_action"[^>]*>/)[0];
    assert.ok(button.includes('onclick="unloadDeberta(this)"'));
    const model = button.match(/data-model="([^"]+)"/)[1];
    const btn = {dataset: {model}, disabled: false, textContent: 'Entladen'};
    await context.unloadDeberta(btn);
    assert.equal(calls[0].url, '/api/deberta/unload');
    assert.deepEqual(JSON.parse(calls[0].options.body), {model: 'bge-reranker-v2-m3'});
    assert.equal(btn.disabled, false);
}
check().catch(error => { console.error(error); process.exitCode = 1; });
"""
    subprocess.run(["node", "-e", script, str(source)], check=True, capture_output=True, text=True)

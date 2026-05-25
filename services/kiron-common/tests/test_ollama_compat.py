import asyncio
import json
import unittest
from unittest import mock

import kiron_common.ollama_compat as compat_mod
from kiron_common.ollama_compat import (
    OllamaCompatClient,
    apply_think_false_bytes,
    canonical_model_name,
    ensure_num_gpu_zero,
    normalize_loaded_model,
    normalize_ps_response,
)


class CompatHelperTests(unittest.TestCase):
    def test_bool_size_vram_is_invalid_for_critical_path(self):
        result = normalize_loaded_model(
            {"name": "m", "size": 1, "size_vram": False},
            require_size_vram=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "size_vram_invalid")

    def test_missing_size_vram_allowed_for_display(self):
        result = normalize_loaded_model({"name": "m", "size": "bad"}, require_size_vram=False)
        self.assertTrue(result.ok)
        self.assertIsNone(result.data.size)
        self.assertIsNone(result.data.size_vram)

    def test_ps_root_shape(self):
        self.assertFalse(normalize_ps_response([], require_size_vram=False).ok)
        self.assertFalse(normalize_ps_response({"models": None}, require_size_vram=False).ok)

    def test_ps_critical_collects_all_failures(self):
        # require_size_vram=True must iterate through every entry so the
        # caller sees every malformed model, not only the first one.
        payload = {
            "models": [
                {"name": "good1:latest", "size": 1, "size_vram": 1},
                {"name": "bad1:latest", "size": 1},
                {"name": "good2:latest", "size": 2, "size_vram": 2},
                {"name": "bad2:latest", "size": 2, "size_vram": "huge"},
            ]
        }
        result = normalize_ps_response(payload, require_size_vram=True)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "size_vram_invalid")
        self.assertEqual([m.name for m in result.data["models"]], ["good1:latest", "good2:latest"])
        self.assertEqual(
            [f.data["name"] for f in result.data["failures"]],
            ["bad1:latest", "bad2:latest"],
        )

    def test_think_false_bytes_preserves_explicit_true(self):
        body = json.dumps({"model": "m", "think": True}).encode()
        self.assertEqual(apply_think_false_bytes(body, "/api/chat"), body)

    def test_think_false_bytes_injects_for_chat_generate(self):
        body = json.dumps({"model": "m"}).encode()
        out = apply_think_false_bytes(body, "/api/generate")
        self.assertFalse(json.loads(out)["think"])

    def test_think_false_bytes_normalizes_path_variants(self):
        body = json.dumps({"model": "m"}).encode()
        for variant in ("/api/chat/", "/api/chat?stream=true", "/api/chat/?stream=true"):
            out = apply_think_false_bytes(body, variant)
            self.assertFalse(json.loads(out)["think"], variant)

    def test_num_gpu_helper_respects_explicit(self):
        opts = {"num_gpu": 7}
        self.assertIs(ensure_num_gpu_zero(opts), opts)
        self.assertEqual(opts["num_gpu"], 7)

    def test_canonical_model_name_tags_simple_names(self):
        self.assertEqual(canonical_model_name("llama2"), "llama2:latest")
        self.assertEqual(canonical_model_name("llama2:7b"), "llama2:7b")

    def test_canonical_model_name_ignores_registry_port_colon(self):
        # ``localhost:5000/llama3`` carries a registry port, not a tag —
        # Ollama lists it as ``localhost:5000/llama3:latest`` in /api/ps.
        self.assertEqual(
            canonical_model_name("localhost:5000/llama3"),
            "localhost:5000/llama3:latest",
        )
        self.assertEqual(
            canonical_model_name("localhost:5000/llama3:latest"),
            "localhost:5000/llama3:latest",
        )


class _FakeAsyncClient:
    def __init__(self, ps_payloads=None, post_payload=None):
        self.closed = False
        self.calls = []
        self.post_timeouts = []
        self.ps_payloads = list(ps_payloads or [{"models": []}])
        self.post_payload = {} if post_payload is None else post_payload

    async def get(self, path):
        self.calls.append(("GET", path))
        if path == "/api/version":
            return _Resp(200, {"version": "0.21.1"})
        if path == "/api/tags":
            return _Resp(200, {"models": [{"name": "m:latest", "size": 1, "details": {}}]})
        if path == "/api/ps":
            if len(self.ps_payloads) > 1:
                payload = self.ps_payloads.pop(0)
            else:
                payload = self.ps_payloads[0]
            return _Resp(200, payload)
        return _Resp(404, {})

    async def post(self, path, json=None, timeout=None):
        self.calls.append(("POST", path, json))
        self.post_timeouts.append(timeout)
        return _Resp(200, self.post_payload)

    async def aclose(self):
        self.closed = True


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class CompatClientTests(unittest.TestCase):
    def test_injected_client_is_not_closed(self):
        async def run():
            fake = _FakeAsyncClient()
            compat = OllamaCompatClient(client=fake)
            self.assertTrue((await compat.version()).ok)
            self.assertTrue((await compat.tags()).ok)
            self.assertTrue((await compat.ps(require_size_vram=True)).ok)
            self.assertTrue((await compat.unload("m:latest")).ok)
            await compat.aclose()
            self.assertFalse(fake.closed)

        asyncio.run(run())

    def test_unload_polls_until_model_disappears_from_ps(self):
        async def run():
            fake = _FakeAsyncClient(ps_payloads=[
                {"models": [{"name": "m:latest", "size": 1, "size_vram": 1}]},
                {"models": []},
            ])
            compat = OllamaCompatClient(client=fake)
            result = await compat.unload("m:latest")
            self.assertTrue(result.ok)
            self.assertEqual(
                fake.calls,
                [
                    ("POST", "/api/generate", {
                        "model": "m:latest",
                        "keep_alive": 0,
                        "stream": False,
                    }),
                    ("GET", "/api/ps"),
                    ("GET", "/api/ps"),
                ],
            )

        asyncio.run(run())

    def test_unload_post_uses_unload_timeout(self):
        async def run():
            fake = _FakeAsyncClient()
            compat = OllamaCompatClient(client=fake)
            result = await compat.unload("m:latest")
            self.assertTrue(result.ok)
            self.assertEqual(fake.post_timeouts, [compat_mod.UNLOAD_VERIFY_TIMEOUT_S])

        asyncio.run(run())

    def test_unload_matches_canonical_tag_for_tagless_input(self):
        async def run():
            fake = _FakeAsyncClient(ps_payloads=[
                {"models": [{"name": "llama2:latest", "size": 1, "size_vram": 1}]},
            ])
            compat = OllamaCompatClient(client=fake)
            with mock.patch.object(compat_mod, "UNLOAD_VERIFY_TIMEOUT_S", 0.0):
                result = await compat.unload("llama2")
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "unload_verify_timeout")

        asyncio.run(run())

    def test_unload_fails_when_model_remains_loaded(self):
        async def run():
            fake = _FakeAsyncClient(ps_payloads=[
                {"models": [{"name": "m:latest", "size": 1, "size_vram": 1}]},
            ])
            compat = OllamaCompatClient(client=fake)
            with mock.patch.object(compat_mod, "UNLOAD_VERIFY_TIMEOUT_S", 0.0):
                result = await compat.unload("m:latest")
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "unload_verify_timeout")

        asyncio.run(run())

    def test_unload_ignores_unrelated_malformed_entry_once_target_gone(self):
        async def run():
            fake = _FakeAsyncClient(ps_payloads=[
                {"models": [
                    {"name": "m:latest", "size": 1, "size_vram": 1},
                    {"size": 1, "size_vram": 1},
                ]},
                {"models": [
                    {"name": "other:latest", "size": 1, "size_vram": 1},
                ]},
            ])
            compat = OllamaCompatClient(client=fake)
            result = await compat.unload("m:latest")
            self.assertTrue(result.ok)
            self.assertEqual(result.code, "ok")

        asyncio.run(run())

    def test_unload_short_circuits_on_200_with_error_body(self):
        async def run():
            fake = _FakeAsyncClient(post_payload={"error": "model 'x' not found"})
            compat = OllamaCompatClient(client=fake)
            result = await compat.unload("x")
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "unload_rejected")
            # Must short-circuit before the verify loop touches /api/ps.
            self.assertEqual([c[0] for c in fake.calls], ["POST"])

        asyncio.run(run())

    def test_rejects_timeout_with_injected_client(self):
        fake = _FakeAsyncClient()
        with self.assertRaises(ValueError):
            OllamaCompatClient(client=fake, timeout=5.0)


if __name__ == "__main__":
    unittest.main()

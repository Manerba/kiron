"""Tests fuer GPU-Prozess-Label-Klassifizierung (#965).

Container-Prozesse wie docling-serve aus quay.io/docling-project/...
landen in nvidia-smi nur mit dem generischen Interpreter-Pfad
`/opt/app-root/bin/python3`. Vor dem Fix wurden sie als "other" gelabelt
und ihre VRAM-Nutzung tauchte im UI faelschlich unter "Sonstige" auf.
Diese Tests fixieren die korrekte Zuordnung via /proc/<pid>/cmdline.
"""

import unittest
from unittest import mock

import metrics


class ClassifyGpuProcessLabelTest(unittest.TestCase):
    def _run(self, process_name: str, cmdline: str) -> str:
        with mock.patch.object(metrics, "_read_proc_cmdline", return_value=cmdline):
            return metrics._classify_gpu_process_label(process_name, pid=4711)

    def test_docling_container_via_cmdline(self):
        # Real-world: quay.io/docling-project/docling-serve-cu130 container.
        label = self._run(
            "/opt/app-root/bin/python3",
            "/opt/app-root/bin/python3 /opt/app-root/bin/docling-serve run",
        )
        self.assertEqual(label, "docling")

    def test_kiron_deberta_via_process_name(self):
        # cmdline ist hier irrelevant, process_name reicht aus.
        label = self._run(
            "/usr/lib/kiron/services/kiron-deberta/venv/bin/python",
            "/usr/lib/kiron/services/kiron-deberta/venv/bin/python main.py",
        )
        self.assertEqual(label, "deberta")

    def test_ollama_process(self):
        label = self._run("/usr/bin/ollama", "/usr/bin/ollama serve")
        self.assertEqual(label, "ollama")

    def test_embedding_service(self):
        label = self._run(
            "/usr/lib/kiron/services/kiron-embeddings/venv/bin/python",
            "/usr/lib/kiron/services/kiron-embeddings/venv/bin/python main.py",
        )
        self.assertEqual(label, "embedding")

    def test_unknown_process_falls_back_to_other(self):
        label = self._run("/usr/bin/python3", "/usr/bin/python3 some_random_script.py")
        self.assertEqual(label, "other")

    def test_cmdline_unreadable_falls_back_to_process_name(self):
        # _read_proc_cmdline gibt bei OSError "" zurueck — process_name allein
        # muss noch korrekt klassifizieren, sonst verlieren wir Robustheit bei
        # Race Conditions (Prozess endet zwischen nvidia-smi und /proc-Lesen).
        label = self._run("/opt/x/kiron-deberta/python", "")
        self.assertEqual(label, "deberta")

    def test_deberta_beats_embedding_when_both_match(self):
        # Falls beide Tokens vorkommen, gewinnt deberta — DeBERTa-Service
        # importiert ggf. sentence-transformers, soll aber "deberta" bleiben.
        label = self._run(
            "/usr/lib/kiron/services/kiron-deberta/venv/bin/python",
            "python -m sentence_transformers nli-deberta-v3-base",
        )
        self.assertEqual(label, "deberta")


class ReadProcCmdlineTest(unittest.TestCase):
    def test_returns_empty_on_missing_pid(self):
        # PID 0 existiert nie als Userspace-Prozess.
        self.assertEqual(metrics._read_proc_cmdline(0), "")

    def test_reads_own_cmdline(self):
        import os
        out = metrics._read_proc_cmdline(os.getpid())
        # Eigene cmdline enthaelt zumindest "python" (Test-Runner).
        self.assertIn("python", out.lower())


if __name__ == "__main__":
    unittest.main()

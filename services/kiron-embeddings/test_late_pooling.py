"""Numerical regressions for the late profile's token selection contract."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main


class TokenModel:
    max_seq_length = 8192
    device = torch.device("cpu")

    def __init__(self, role):
        self.prefix = len(role + ": ")
        self.tokenizer = mock.Mock(side_effect=self.tokenize)
        self.tokenizer.all_special_ids = [101, 102]
        self.encoded = []

    def tokenize(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [[101, 7, 102] for _ in text]}
        p = self.prefix
        return {
            "input_ids": torch.tensor([[101, 7, 8, 102, 9, 102]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "offset_mapping": torch.tensor([
                [(0, 0), (0, p - 1), (p, p + 6), (p + 7, p + 12),
                 (p + 13, p + 20), (0, 0)]
            ]),
        }

    def __getitem__(self, index):
        return SimpleNamespace(auto_model=self)

    def __call__(self, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.tensor([
            [[50., 40., 30.], [40., 30., 20.], [2., 0., 0.],
             [0., 0., 20.], [0., 2., 0.], [30., 20., 10.]]
        ]))

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        return np.array([[0., 0., 3.] for _ in texts], dtype=np.float32)


class LatePoolingTests(unittest.TestCase):
    def run_chunks(self, spans, role="search_document"):
        document = "Berlin [SEP] Hamburg"
        model = TokenModel(role)
        manager = main.ModelManager()
        manager.set_worker_thread()
        manager.device = "cpu"
        chunks = [{"text": document[a:b], "char_start": a, "char_end": b}
                  for a, b in spans]
        with mock.patch.object(manager, "_ensure_model_sync", return_value=(model, 0)):
            result = manager._late_embed_sync("nomic-embed-text", document, chunks, role)
        return result, model

    def test_empty_span_inside_token_uses_fallback(self):
        result, model = self.run_chunks([(1, 1)])
        self.assertEqual(result.fallback_count, 1)
        self.assertEqual(model.encoded, ["search_document: "])
        np.testing.assert_array_equal(result.embeddings, [[0., 0., 1.]])

    def test_literal_special_token_with_nonzero_offsets_is_excluded(self):
        result, _ = self.run_chunks([(0, 20), (7, 12)])
        self.assertEqual(result.fallback_count, 1)
        np.testing.assert_allclose(result.embeddings,
                                   [[2**-.5, 2**-.5, 0.], [0., 0., 1.]], atol=1e-7)

    def test_prefix_and_inserted_special_tokens_are_excluded_for_both_roles(self):
        for role in ("search_document", "search_query"):
            with self.subTest(role=role):
                result, _ = self.run_chunks([(0, 6), (13, 20)], role)
                np.testing.assert_array_equal(result.embeddings, [[1., 0., 0.], [0., 1., 0.]])
                self.assertEqual(result.fallback_count, 0)

    def test_partial_token_overlap_and_output_order(self):
        result, _ = self.run_chunks([(14, 15), (1, 3), (6, 7), (0, 6)])
        np.testing.assert_array_equal(result.embeddings,
                                      [[0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.]])
        self.assertEqual(result.fallback_count, 1)


if __name__ == "__main__":
    unittest.main()

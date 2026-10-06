"""CPU tensor fixtures: actual forward tokens, never estimates or model loads."""
import unittest
import torch

from token_usage import ForwardUsage, TokenUsageError, capture_forward_usage


def features(mask):
    values = torch.tensor(mask, dtype=torch.int64)
    return {"input_ids": values.clone(), "attention_mask": values}


class ForwardUsageTests(unittest.TestCase):
    def test_batches_count_special_and_truncated_tokens_but_not_padding(self):
        usage = ForwardUsage()
        usage.observe(features([[1, 1, 1, 0], [1, 1, 1, 1]]))
        usage.observe(features([[1, 1]]))
        self.assertEqual(usage.finish(3), 9)
        with self.assertRaises(TokenUsageError):
            usage.finish(4)

    def test_missing_or_invalid_forward_evidence_fails_closed(self):
        invalid = [None, {}, features([[0, 0]]), features([[1, 2]]),
                   {"input_ids": torch.zeros((1, 2)), "attention_mask": torch.ones((1, 2))},
                   {"input_ids": torch.ones((1, 2), dtype=torch.int64),
                    "attention_mask": torch.ones((2,), dtype=torch.int64)}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(TokenUsageError):
                ForwardUsage().observe(value)
        with self.assertRaises(TokenUsageError):
            ForwardUsage().finish(1)

    def test_capture_restores_class_or_instance_forward_even_on_failure(self):
        class Encoder:
            def forward(self, value):
                return value
        model = Encoder()
        with capture_forward_usage(model) as usage:
            model.forward(features([[1, 1, 0]]))
            self.assertEqual(usage.finish(1), 2)
        self.assertNotIn("forward", model.__dict__)
        replacement = lambda _: (_ for _ in ()).throw(RuntimeError("fixture"))
        model.forward = replacement
        with self.assertRaises(RuntimeError), capture_forward_usage(model):
            model.forward(features([[1]]))
        self.assertIs(model.forward, replacement)

    def test_last_token_encoder_exposes_the_actual_backbone_mask(self):
        from loaders import LastTokenEmbeddingModel
        from test_mankei_embedder import _Tokenizer, _Backbone
        model = LastTokenEmbeddingModel(tokenizer=_Tokenizer(), backbone=_Backbone(), max_seq_length=256)
        with capture_forward_usage(model) as usage:
            result = model.encode(["eins", "zwei"], batch_size=2, convert_to_numpy=True)
            self.assertEqual(usage.finish(2), 5)
        self.assertEqual(result.tolist(), [[2., 20.], [5., 50.]])

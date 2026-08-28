"""Modellspezifische Regressionstests fuer den Mankei-Embedder."""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main  # noqa: E402
import loaders  # noqa: E402

from catalog_view import TransformersLastTokenLoaderParameters  # noqa: E402
from kiron_common.model_catalog import LoaderType, ModelEndpoint  # noqa: E402


class _Tokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return {
            "input_ids": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, **_kwargs):
        return SimpleNamespace(last_hidden_state=torch.tensor([
            [[1.0, 10.0], [2.0, 20.0], [99.0, 99.0]],
            [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
        ]))


class MankeiEmbedderTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_is_revision_pinned(self):
        config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model(
            "mankei-326m-embedder"
        )
        self.assertEqual(
            config.artifact.repository,
            "keyvan-ai/Mankei-326M-Embedder",
        )
        self.assertEqual(
            config.artifact.revision,
            "86f562cf94d5510175b546e6e9156f99bbd790b5",
        )
        self.assertEqual(config.embedding_length, 960)
        self.assertIs(config.loader_type, LoaderType.TRANSFORMERS_LAST_TOKEN)
        self.assertIsInstance(
            config.loader_parameters,
            TransformersLastTokenLoaderParameters,
        )
        self.assertIs(
            main.EMBEDDING_SERVICE_VIEW.loader_for(config),
            loaders.load_transformers_last_token_cpu,
        )

    def test_prefixes_match_model_card(self):
        config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model(
            "mankei-326m-embedder"
        )
        self.assertEqual(
            main._apply_prefix(["Frage"], config, "search_query"),
            ["query: Frage"],
        )
        self.assertEqual(
            main._apply_prefix(["Absatz"], config, "search_document"),
            ["passage: Absatz"],
        )
        self.assertEqual(
            main._apply_prefix(["Absatz"], config, None),
            ["passage: Absatz"],
        )

    def test_last_token_pooling_uses_attention_mask(self):
        tokenizer = _Tokenizer()
        model = loaders.LastTokenEmbeddingModel(
            tokenizer=tokenizer,
            backbone=_Backbone(),
            max_seq_length=256,
        )

        result = model.encode(
            ["eins", "zwei"],
            batch_size=2,
            convert_to_numpy=True,
        )

        np.testing.assert_array_equal(result, [[2.0, 20.0], [5.0, 50.0]])
        self.assertEqual(tokenizer.calls[0][1]["max_length"], 256)

    def test_loader_is_local_only_and_pinned(self):
        config = main.EMBEDDING_SERVICE_VIEW.require_runtime_model(
            "mankei-326m-embedder"
        )
        tokenizer = _Tokenizer()
        backbone = _Backbone()
        with (
            mock.patch.object(
                loaders.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_load,
            mock.patch.object(
                loaders.AutoModel,
                "from_pretrained",
                return_value=backbone,
            ) as model_load,
        ):
            loaded = loaders.load_transformers_last_token_cpu(config)

        self.assertIsInstance(loaded, loaders.LastTokenEmbeddingModel)
        self.assertEqual(tokenizer.padding_side, "right")
        for call in (tokenizer_load.call_args, model_load.call_args):
            self.assertTrue(call.kwargs["local_files_only"])
            self.assertEqual(
                call.kwargs["revision"],
                "86f562cf94d5510175b546e6e9156f99bbd790b5",
            )
        self.assertEqual(
            loaded.max_seq_length,
            config.require_profile(ModelEndpoint.EMBED).document_max_tokens,
        )

    async def test_invalid_input_type_is_rejected_before_worker(self):
        req = main.EmbedRequest(
            model="mankei-326m-embedder",
            input=["Text"],
            input_type="classification",
        )
        response = await main.embed(req)
        self.assertEqual(response.status_code, 400)
        payload = json.loads(bytes(response.body))
        self.assertEqual(
            payload["valid_input_types"],
            ["search_document", "search_query"],
        )

    def test_show_exposes_embedding_dimension_for_kiara(self):
        response = main.show_model_endpoint(
            main.ShowRequest(model="mankei-326m-embedder:latest")
        )
        self.assertEqual(response["model"], "mankei-326m-embedder")
        self.assertEqual(response["model_info"]["llama.embedding_length"], 960)
        self.assertEqual(response["details"]["quantization_level"], "BF16")


if __name__ == "__main__":
    unittest.main()

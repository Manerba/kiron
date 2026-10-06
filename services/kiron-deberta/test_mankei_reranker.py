"""Modellspezifische Regressionstests fuer den Mankei-Reranker."""

import os
import sys
import unittest
from dataclasses import replace
from types import MappingProxyType
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_view  # noqa: E402
import loaders  # noqa: E402
import main  # noqa: E402
from kiron_common.model_catalog import LoaderType  # noqa: E402


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


class _LoadBackbone:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=960)
        self.to_calls = []

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class _LoadHead:
    def __init__(self, in_features, out_features):
        self.shape = (in_features, out_features)
        self.loaded = None
        self.to_calls = []

    def load_state_dict(self, state, strict=True):
        self.loaded = (state, strict)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class MankeiRerankerTests(unittest.TestCase):
    def test_registry_is_revision_pinned(self):
        config = main.DEBERTA_CATALOG_VIEW.require_runtime_model(
            "mankei-326m-reranker"
        )
        self.assertEqual(
            config.artifact.repository,
            "keyvan-ai/Mankei-326M-Reranker",
        )
        self.assertEqual(
            config.artifact.revision,
            "e1219e343a9ebdcf2e381c058a480e34defbeb95",
        )
        self.assertEqual(config.loader_type, LoaderType.MANKEI_LAST_TOKEN)
        self.assertIsInstance(
            config.loader_parameters,
            catalog_view.MankeiLastTokenLoaderParameters,
        )
        self.assertEqual(config.max_length, 192)
        self.assertEqual(
            config.artifact.weights[0].sha256,
            "34566de5240d16995f2830c00ea5981d5f3013f165b2c75a440100b7c49375d8",
        )
        self.assertEqual(
            config.artifact.auxiliary[0].sha256,
            "ce0cf93891cbee260a7acfb9c5d66bf3d9c13c68e81fd9eb5cac99838b543edf",
        )

    def test_predict_formats_pair_and_uses_last_token(self):
        tokenizer = _Tokenizer()
        head = torch.nn.Linear(2, 1)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0]]))
            head.bias.zero_()
        model = loaders.MankeiReranker(
            tokenizer=tokenizer,
            backbone=_Backbone(),
            head=head,
            max_length=192,
            pair_template="Frage: {query}\nPassage: {passage}",
            batch_size=2,
        )

        scores = model.predict([("Frage 1", "Text 1"), ("Frage 2", "Text 2")])

        np.testing.assert_array_equal(scores, [2.0, 5.0])
        self.assertEqual(
            tokenizer.calls[0][0],
            ["Frage: Frage 1\nPassage: Text 1", "Frage: Frage 2\nPassage: Text 2"],
        )
        self.assertEqual(tokenizer.calls[0][1]["max_length"], 192)

    def test_loader_uses_pinned_local_files_and_safe_head_load(self):
        config = main.DEBERTA_CATALOG_VIEW.require_runtime_model(
            "mankei-326m-reranker"
        )
        tokenizer = _Tokenizer()
        backbone = _LoadBackbone()
        head = _LoadHead(960, 1)
        state = {"weight": torch.zeros((1, 960)), "bias": torch.zeros(1)}

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
            mock.patch.object(
                loaders,
                "hf_hub_download",
                return_value="/cache/head.pt",
            ) as head_download,
            mock.patch.object(loaders.torch, "load", return_value=state) as torch_load,
            mock.patch.object(loaders.torch.nn, "Linear", return_value=head),
        ):
            loaded = loaders.load_mankei_last_token_cuda(config)

        self.assertIsInstance(loaded, loaders.MankeiReranker)
        self.assertEqual(tokenizer.padding_side, "right")
        for call in (tokenizer_load.call_args, model_load.call_args):
            self.assertTrue(call.kwargs["local_files_only"])
            self.assertEqual(call.kwargs["revision"], config.artifact.revision)
        self.assertTrue(head_download.call_args.kwargs["local_files_only"])
        self.assertEqual(
            head_download.call_args.kwargs["revision"],
            config.artifact.revision,
        )
        self.assertEqual(
            torch_load.call_args.kwargs,
            {"map_location": "cpu", "weights_only": True},
        )
        self.assertEqual(head.loaded, (state, True))

    def test_model_manager_selects_custom_loader(self):
        sentinel = object()
        custom_load = mock.Mock(return_value=sentinel)
        registry = dict(loaders.LOADER_REGISTRY)
        registry[LoaderType.MANKEI_LAST_TOKEN] = custom_load
        service_view = replace(
            main.DEBERTA_CATALOG_VIEW,
            _loader_registry=MappingProxyType(registry),
        )
        manager = main.ModelManager(service_view)
        with (
            mock.patch.object(main.torch.cuda, "is_available", return_value=True),
            mock.patch.object(main.torch.cuda, "mem_get_info", return_value=(12 * 1024**3, 12 * 1024**3)),
            mock.patch.object(main.torch.cuda, "empty_cache"),
        ):
            loaded, config = manager._load_model("mankei-326m-reranker")

        self.assertIs(loaded, sentinel)
        self.assertIs(
            config,
            main.DEBERTA_CATALOG_VIEW.require_runtime_model(
                "mankei-326m-reranker"
            ),
        )
        custom_load.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()

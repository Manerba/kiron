"""Service-local, explicit model loaders for kiron-deberta."""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer

from kiron_common.model_catalog import LoaderType

from catalog_view import (
    CrossEncoderLoaderParameters,
    DebertaServiceCatalogError,
    DebertaServiceModel,
    LoaderCallable,
    MankeiLastTokenLoaderParameters,
)


class MankeiReranker:
    """CrossEncoder-compatible adapter for Mankei's backbone and ``head.pt``."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        backbone: torch.nn.Module,
        head: torch.nn.Linear,
        max_length: int,
        pair_template: str,
        batch_size: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.head = head
        self.max_length = max_length
        self.pair_template = pair_template
        self.batch_size = batch_size

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> np.ndarray:
        if apply_softmax:
            raise ValueError("Mankei ist ein Single-Score-Reranker ohne Softmax-Labels")
        if not pairs:
            return np.empty((0,), dtype=np.float32)

        device = next(self.backbone.parameters()).device
        batches: list[torch.Tensor] = []
        for start in range(0, len(pairs), self.batch_size):
            texts = [
                self.pair_template.format(query=query, passage=passage)
                for query, passage in pairs[start : start + self.batch_size]
            ]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        hidden = self.backbone(**encoded).last_hidden_state
                else:
                    hidden = self.backbone(**encoded).last_hidden_state
                last_token = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[
                    torch.arange(hidden.size(0), device=device),
                    last_token,
                ]
                if device.type == "cuda":
                    # The published linear head is FP32. The surrounding endpoint
                    # autocast must not silently pull it into FP16.
                    with torch.amp.autocast("cuda", enabled=False):
                        scores = self.head(pooled.float()).squeeze(-1)
                else:
                    scores = self.head(pooled.float()).squeeze(-1)
            batches.append(scores.float().cpu())

        return torch.cat(batches).numpy()


def _artifact_coordinates(model: DebertaServiceModel) -> tuple[str, str]:
    repository = model.artifact.repository
    revision = model.artifact.revision
    if repository is None or revision is None:
        raise DebertaServiceCatalogError(
            f"/service/models/{model.model_name}/artifact",
            "loader requires a pinned HuggingFace repository and revision",
        )
    return repository, revision


def load_cross_encoder_cuda(model: DebertaServiceModel) -> object:
    parameters = model.loader_parameters
    if not isinstance(parameters, CrossEncoderLoaderParameters):
        raise DebertaServiceCatalogError(
            f"/service/models/{model.model_name}/loader/parameters",
            "cross_encoder loader received incompatible parameters",
        )
    repository, revision = _artifact_coordinates(model)
    dtype_by_name = {"float16": torch.float16}
    return CrossEncoder(
        repository,
        cache_folder=os.environ.get("HF_HUB_CACHE") or None,
        max_length=parameters.max_length,
        device="cuda",
        trust_remote_code=model.artifact.trust_remote_code,
        revision=revision,
        local_files_only=True,
        model_kwargs={"torch_dtype": dtype_by_name[parameters.torch_dtype]},
    )


def load_mankei_last_token_cuda(model: DebertaServiceModel) -> object:
    parameters = model.loader_parameters
    if not isinstance(parameters, MankeiLastTokenLoaderParameters):
        raise DebertaServiceCatalogError(
            f"/service/models/{model.model_name}/loader/parameters",
            "mankei_last_token loader received incompatible parameters",
        )
    repository, revision = _artifact_coordinates(model)
    cache_dir = os.environ.get("HF_HUB_CACHE") or None
    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
        use_fast=parameters.tokenizer_use_fast,
    )
    tokenizer.padding_side = parameters.padding_side
    backbone = AutoModel.from_pretrained(
        repository,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    head_path = hf_hub_download(
        repo_id=repository,
        filename=parameters.head_filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    head = torch.nn.Linear(backbone.config.hidden_size, 1)
    head.load_state_dict(state, strict=True)
    head.eval()

    # BF16 backbone and the small FP32 head move separately to preserve the
    # score precision intended by the model author.
    backbone = backbone.to("cuda")
    head = head.to("cuda", dtype=torch.float32)
    return MankeiReranker(
        tokenizer=tokenizer,
        backbone=backbone,
        head=head,
        max_length=parameters.max_length,
        pair_template=parameters.pair_template,
        batch_size=parameters.batch_size,
    )


LOADER_REGISTRY: MappingProxyType[LoaderType, LoaderCallable] = MappingProxyType(
    {
        LoaderType.CROSS_ENCODER: load_cross_encoder_cuda,
        LoaderType.MANKEI_LAST_TOKEN: load_mankei_last_token_cuda,
    }
)


__all__ = [
    "LOADER_REGISTRY",
    "MankeiReranker",
    "load_cross_encoder_cuda",
    "load_mankei_last_token_cuda",
]

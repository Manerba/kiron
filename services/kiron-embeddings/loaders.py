"""Service-local, explicit model loaders for kiron-embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from kiron_common.model_catalog import LoaderType, ModelEndpoint

from catalog_view import (
    ColbertXmodLoaderParameters,
    EmbeddingServiceCatalogError,
    EmbeddingServiceModel,
    LoaderCallable,
    SentenceTransformersLoaderParameters,
    TransformersLastTokenLoaderParameters,
)


@dataclass(slots=True)
class ColbertModelBundle:
    tokenizer: Any
    backbone: torch.nn.Module
    linear: torch.nn.Linear


class LastTokenEmbeddingModel:
    """Minimal SentenceTransformer-compatible decoder embedding adapter."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        backbone: torch.nn.Module,
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.max_seq_length = max_seq_length

    def half(self) -> "LastTokenEmbeddingModel":
        self.backbone = self.backbone.to(dtype=torch.bfloat16)
        return self

    def to(self, device: str | torch.device) -> "LastTokenEmbeddingModel":
        self.backbone = self.backbone.to(device)
        return self

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
    ) -> np.ndarray | torch.Tensor:
        if not texts:
            empty = torch.empty(
                (0, self.backbone.config.hidden_size), dtype=torch.float32
            )
            return empty.numpy() if convert_to_numpy else empty

        device = next(self.backbone.parameters()).device
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
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
            batches.append(pooled.float().cpu())

        result = torch.cat(batches, dim=0)
        return result.numpy() if convert_to_numpy else result


def _artifact_coordinates(model: EmbeddingServiceModel) -> tuple[str, str]:
    repository = model.artifact.repository
    revision = model.artifact.revision
    if repository is None or revision is None:
        raise EmbeddingServiceCatalogError(
            f"/service/models/{model.model_name}/artifact",
            "loader requires a pinned HuggingFace repository and revision",
        )
    return repository, revision


def load_sentence_transformers_cpu(model: EmbeddingServiceModel) -> object:
    parameters = model.loader_parameters
    if not isinstance(parameters, SentenceTransformersLoaderParameters):
        raise EmbeddingServiceCatalogError(
            f"/service/models/{model.model_name}/loader/parameters",
            "sentence_transformers loader received incompatible parameters",
        )
    repository, revision = _artifact_coordinates(model)
    return SentenceTransformer(
        repository,
        cache_folder=os.environ.get("HF_HUB_CACHE") or None,
        device="cpu",
        trust_remote_code=model.artifact.trust_remote_code,
        revision=revision,
        local_files_only=True,
    )


def load_transformers_last_token_cpu(model: EmbeddingServiceModel) -> object:
    parameters = model.loader_parameters
    if not isinstance(parameters, TransformersLastTokenLoaderParameters):
        raise EmbeddingServiceCatalogError(
            f"/service/models/{model.model_name}/loader/parameters",
            "transformers_last_token loader received incompatible parameters",
        )
    repository, revision = _artifact_coordinates(model)
    dtype_by_name = {"bfloat16": torch.bfloat16}
    torch_dtype = dtype_by_name[parameters.torch_dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        local_files_only=True,
        use_fast=parameters.tokenizer_use_fast,
    )
    profile = model.require_profile(ModelEndpoint.EMBED)
    if profile.padding_side is None:
        raise EmbeddingServiceCatalogError(
            f"/service/models/{model.model_name}/profiles",
            "transformers_last_token requires a profile padding side",
        )
    tokenizer.padding_side = profile.padding_side
    backbone = AutoModel.from_pretrained(
        repository,
        revision=revision,
        local_files_only=True,
        torch_dtype=torch_dtype,
    ).eval()
    max_seq_length = max(
        profile.document_max_tokens,
        profile.query_max_tokens,
    )
    return LastTokenEmbeddingModel(
        tokenizer=tokenizer,
        backbone=backbone,
        max_seq_length=max_seq_length,
    )


def load_colbert_xmod_cpu(model: EmbeddingServiceModel) -> object:
    parameters = model.loader_parameters
    if not isinstance(parameters, ColbertXmodLoaderParameters):
        raise EmbeddingServiceCatalogError(
            f"/service/models/{model.model_name}/loader/parameters",
            "colbert_xmod loader received incompatible parameters",
        )
    repository, revision = _artifact_coordinates(model)
    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        local_files_only=True,
        use_fast=parameters.tokenizer_use_fast,
    )
    backbone = AutoModel.from_pretrained(
        repository,
        revision=revision,
        local_files_only=True,
    )
    projection_path = hf_hub_download(
        repo_id=repository,
        filename=model.artifact.weights[0].path,
        revision=revision,
        local_files_only=True,
    )
    with safe_open(projection_path, framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(parameters.projection_weight_tensor)
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight.to(dtype=torch.float32))
    backbone.eval()
    linear.eval()
    return ColbertModelBundle(
        tokenizer=tokenizer,
        backbone=backbone,
        linear=linear,
    )


LOADER_REGISTRY: MappingProxyType[LoaderType, LoaderCallable] = MappingProxyType(
    {
        LoaderType.SENTENCE_TRANSFORMERS: load_sentence_transformers_cpu,
        LoaderType.TRANSFORMERS_LAST_TOKEN: load_transformers_last_token_cpu,
        LoaderType.COLBERT_XMOD: load_colbert_xmod_cpu,
    }
)


__all__ = [
    "ColbertModelBundle",
    "LOADER_REGISTRY",
    "LastTokenEmbeddingModel",
    "load_colbert_xmod_cpu",
    "load_sentence_transformers_cpu",
    "load_transformers_last_token_cpu",
]

#!/usr/bin/env python3
"""Pinned, real-weight reference vectors for kiron-nomic-late-v1.

Run with the embedding venv and HF_HOME=/var/cache/kiron/huggingface.
--write-reference explicitly freezes the independent FP32 oracle, never the
service result. Normal runs compare the service with that frozen witness.
--base-url additionally checks live discovery, request guards and GPU vectors.
No downloads, model installation, index writes or production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

os.environ.setdefault("HF_HOME", "/var/cache/kiron/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services/kiron-embeddings"), str(ROOT / "services/kiron-common")]

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

MODEL = "nomic-embed-text:latest"
PROFILE = "kiron-nomic-late-v1"
REPOSITORY = "nomic-ai/nomic-embed-text-v1.5"
REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
WEIGHT_SHA = "9e7d262b1fe5ea350782829496efa831901b77486bbde1cea54a4c822d010d5c"
FIXTURE = ROOT / "scripts/fixtures/nomic-late-v1.reference.json"
MAX_ABS = 1e-3
MAX_COSINE_DISTANCE = 1e-5


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases():
    document = ("Berlin ist die Hauptstadt von Deutschland. "
                "Paris ist die Hauptstadt von Frankreich. "
                "Bienen produzieren Honig. Straße 😀 e\u0301.")
    paris = document.index("Paris")
    bees = document.index("Bienen")
    street = document.index("Straße")

    def case(name, text, spans, role="search_document"):
        return {"name": name, "request": {
            "model": MODEL, "document": text, "input_type": role,
            "chunks": [{"text": text[a:b], "char_start": a, "char_end": b}
                       for a, b in spans],
        }}

    specials = "Berlin [SEP] Hamburg [CLS] München [PAD] Ende."
    long_doc = "a " * 8190 + "b " * 8194
    result = [
        case("document_overlap_unicode", document,
             [(0, paris - 1), (paris, bees - 1), (bees, street - 1),
              (1, 3), (4, paris + 5), (0, len(document)),
              (6, 7), (1, 1), (len(document), len(document)),
              (street, len(document))]),
        case("literal_special_tokens", specials, [(0, len(specials)), (7, 12), (0, 6)]),
        case("query_special_tokens", specials, [(0, len(specials)), (7, 12)], "search_query"),
        case("empty_document", "", [(0, 0)]),
        case("window_and_fallback_truncation", long_doc,
             [(16366, 16384), (16380, 16390), (16380, len(long_doc)), (0, 2)]),
    ]
    for name, query in (
        ("query_berlin", "Was ist die Hauptstadt von Deutschland?"),
        ("query_paris", "Was ist die Hauptstadt von Frankreich?"),
    ):
        result.append(case(name, query, [(0, len(query))], "search_query"))
    return result


def load_model():
    torch.set_num_threads(4)
    return SentenceTransformer(
        REPOSITORY, revision=REVISION, device="cpu", local_files_only=True,
        trust_remote_code=True, model_kwargs={"use_safetensors": True},
    ).eval()


def provenance(model):
    snapshot = Path(hf_hub_download(REPOSITORY, "config.json", revision=REVISION,
                                    local_files_only=True)).parent
    weights = sha(snapshot / "model.safetensors")
    if weights != WEIGHT_SHA:
        raise ValueError("Nomic weight digest does not match the pinned artifact")
    files = {p: sha(snapshot / p) for p in (
        "config.json", "modules.json", "sentence_bert_config.json",
        "1_Pooling/config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt",
    )}
    backbone = model[0].auto_model
    for cls in (type(backbone), type(backbone.config)):
        source = Path(inspect.getfile(cls))
        files["remote_code/" + source.parent.name + "/" + source.name] = sha(source)
    if model.max_seq_length != 8192 or model.tokenizer.truncation_side != "right":
        raise ValueError("Unexpected Nomic token window")
    return {
        "repository": REPOSITORY, "revision": REVISION, "weight_sha256": weights,
        "files_sha256": files,
        "packages": {p: importlib.metadata.version(p) for p in (
            "sentence-transformers", "transformers", "torch", "tokenizers", "numpy",
        )},
        "reference_device": "cpu", "reference_dtype": "float32",
    }


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def token_forward(model, text):
    encoded = model.tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                              truncation=True, max_length=8192, padding=False)
    with torch.inference_mode():
        hidden = model[0].auto_model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
        ).last_hidden_state[0].float().cpu().numpy()
    return encoded, hidden


def dense_reference(model, formatted):
    # Independent ST mean-pooling oracle: fallback includes prefix and specials.
    encoded, hidden = token_forward(model, formatted)
    attended = encoded["attention_mask"][0].numpy().astype(bool)
    return normalize(np.mean(hidden[attended], axis=0, dtype=np.float32))


def reference(model, request):
    # Deliberately does not use _apply_prefix, _late_embed_sync or model.encode.
    prefix = request["input_type"] + ": "
    formatted = prefix + request["document"]
    encoded, hidden = token_forward(model, formatted)
    offsets = encoded["offset_mapping"][0].numpy()
    ids = encoded["input_ids"][0].numpy()
    eligible = (encoded["attention_mask"][0].numpy().astype(bool)
                & ~np.isin(ids, model.tokenizer.all_special_ids)
                & (offsets[:, 1] > len(prefix)))
    vectors, counts, fallback = [], [], []
    for i, chunk in enumerate(request["chunks"]):
        start, end = len(prefix) + chunk["char_start"], len(prefix) + chunk["char_end"]
        # Positive intersection of half-open character intervals. Zero-width
        # chunks/tokens cannot intersect; partially covered tokens contribute.
        selected = eligible & (np.minimum(offsets[:, 1], end) > np.maximum(offsets[:, 0], start))
        counts.append(int(selected.sum()))
        if selected.any():
            vector = normalize(np.mean(hidden[selected], axis=0, dtype=np.float32))
        else:
            fallback.append(i)
            vector = dense_reference(model, prefix + chunk["text"])
        vectors.append(vector.tolist())
    return {
        "embeddings": vectors, "selected_token_counts": counts,
        "fallback_indices": fallback, "document_token_count": len(ids),
    }


def compare(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 768:
        raise ValueError(f"Unexpected vector shape: {a.shape}, expected {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite reference or service vector")
    norms = np.linalg.norm(a, axis=1)
    abs_error = float(np.max(np.abs(a - b)))
    cosine = 1 - np.sum(a * b, axis=1) / (norms * np.linalg.norm(b, axis=1))
    max_distance = max(0., float(np.max(cosine)))
    return {"passed": bool(abs_error <= MAX_ABS and max_distance <= MAX_COSINE_DISTANCE
                            and np.max(np.abs(norms - 1)) <= 1e-6),
            "vectors": len(a), "max_absolute_error": abs_error,
            "max_cosine_distance": max_distance}


def http(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(base.rstrip("/") + path, data=payload,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=240) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def discovery_and_guards(base):
    from kiron_common.embedding_contract import canonical_json, validate_capabilities
    from kiron_common.embedding_registry import EMBEDDING_REGISTRY

    status, tags = http(base, "/api/tags")
    assert status == 200, tags
    caps = next(row["kiron_capabilities"] for row in tags["models"]
                if row.get("kiron_capabilities", {}).get("canonical_model_id") == MODEL)
    validate_capabilities(caps)
    assert caps == EMBEDDING_REGISTRY.require(MODEL).capabilities
    for alias in (MODEL, "nomic-embed-text"):
        status, shown = http(base, "/api/show", {"model": alias})
        assert status == 200 and canonical_json(caps) == canonical_json(shown["kiron_capabilities"])
    profile = next(p for p in caps["profiles"] if p["profile_id"] == PROFILE)
    assert profile["verification"]["status"] == "verified"
    assert profile["verification"]["blocking_reasons"] == []
    dense = next(p for p in caps["profiles"] if p["profile_id"] == "kiron-nomic-dense-v1")
    for field in ("index_compatibility_id", "query_compatibility_id"):
        assert profile[field] is not None and profile[field] != dense[field]
    sample = cases()[-1]["request"]
    for role, code in ((None, "missing_required_input_type"), ("query", "unsupported_input_type")):
        body = {**sample, "input_type": role}
        status, result = http(base, "/api/embed_late", body)
        assert status == 400 and result["error"]["code"] == code, result
    status, result = http(base, "/api/embed_late", {
        **sample, "chunks": [{"text": "wrong", "char_start": 0, "char_end": 1}],
    })
    assert status == 400, result
    return {"passed": True, "alias_tags_show_equal": True, "request_guards": 3,
            "index_compatibility_id": profile["index_compatibility_id"],
            "query_compatibility_id": profile["query_compatibility_id"]}


def run(args):
    model = load_model()
    identity = provenance(model)
    if args.write_reference:
        frozen_cases = []
        for case in cases():
            print("Reference:", case["name"], file=sys.stderr, flush=True)
            frozen_cases.append({**case, **reference(model, case["request"])})
        # Negative control proves that a dense query cannot stand in for late.
        query = cases()[-2]["request"]
        dense = dense_reference(model, "search_query: " + query["document"])
        artifact = {"version": 1, "profile_id": PROFILE, "provenance": identity,
                    "tolerances": {"max_absolute_error": MAX_ABS,
                                   "max_cosine_distance": MAX_COSINE_DISTANCE},
                    "cases": frozen_cases, "dense_query_negative_control": dense.tolist()}
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
        return {"reference_written": str(FIXTURE), "sha256": sha(FIXTURE),
                "cases": len(frozen_cases)}

    frozen = json.loads(FIXTURE.read_text())
    assert frozen["provenance"] == identity, "Local artifact/runtime drift from frozen reference"
    assert [{k: c[k] for k in ("name", "request")} for c in frozen["cases"]] == cases()
    report = {"profile_id": PROFILE, "reference_sha256": sha(FIXTURE),
              "provenance": identity, "mode": args.base_url or "cpu_worker", "cases": []}
    if args.base_url:
        report["discovery"] = discovery_and_guards(args.base_url)
    else:
        import main
        manager = main.ModelManager()
        manager.set_worker_thread()
        manager.device = "cpu"
    actual_vectors = {}
    for case in frozen["cases"]:
        print("Check:", case["name"], file=sys.stderr, flush=True)
        request = case["request"]
        if args.base_url:
            status, output = http(args.base_url, "/api/embed_late", request)
            assert status == 200, output
            vectors = output["embeddings"]
        else:
            expected = reference(model, request)
            assert compare(expected["embeddings"], case["embeddings"])["passed"]
            for key in ("selected_token_counts", "fallback_indices", "document_token_count"):
                assert expected[key] == case[key], (case["name"], key)
            with mock.patch.object(manager, "_ensure_model_sync", return_value=(model, 0)):
                result = manager._late_embed_sync("nomic-embed-text", request["document"],
                                                 request["chunks"], request["input_type"])
            vectors = result.embeddings
            assert result.fallback_count == len(case["fallback_indices"]), case["name"]
        actual_vectors[case["name"]] = vectors
        report["cases"].append({"name": case["name"], **compare(vectors, case["embeddings"])})
    documents = np.asarray(actual_vectors["document_overlap_unicode"][:3])
    rankings = {}
    for name, expected_index in (("query_berlin", 0), ("query_paris", 1)):
        scores = documents @ np.asarray(actual_vectors[name][0])
        assert int(np.argmax(scores)) == expected_index, (name, scores)
        rankings[name] = {"scores": scores.tolist(), "top_chunk": int(np.argmax(scores))}
    negative = compare(actual_vectors["query_berlin"], [frozen["dense_query_negative_control"]])
    assert not negative["passed"], "Dense negative control must differ from late query"
    report["retrieval"] = rankings
    report["dense_query_negative_control"] = negative
    report["actual_vectors"] = actual_vectors
    report["passed"] = all(c["passed"] for c in report["cases"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = run(args)
    if args.report:
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("provenance", "actual_vectors")}, indent=2))
    sys.exit(0 if result.get("passed", True) else 1)

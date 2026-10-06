#!/usr/bin/env python3
"""Independent, pinned GGUF witnesses for the Hellord dense embedding profile.

The reference is a CPU llama.cpp server loading the exact installed GGUF with
--embedding --pooling last --embd-normalize -1. Python normalizes the returned
hidden vector. --unpooled-url additionally checks that vector against the last
row of a token-level forward pass (--pooling none), including negative controls.
Normal runs never rewrite the fixture. --base-url checks the KIron public path,
discovery, role/option guards, token limits, batching and query retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts/fixtures/hellord-e5-mistral-v1.reference.json"
MODEL = "hellord/e5-mistral-7b-instruct:Q4_0"
PROFILE = "ollama-e5-mistral-dense-v1"
WEIGHT_SHA = "f75c0e30696fa7f1656320905943fb42e489e7c4156c37f37e9a17060b5e3e89"
MANIFEST_SHA = "c53c78f9f295165b7b4311cec0063639e1675f699584fd955253944bb35b50b5"
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
LIMIT = 4096
DIMENSIONS = 4096
# Q4_0 CPU and CUDA use different quantized matrix kernels. Both bounds must
# pass; negative controls below prove that prefix/EOS/pooling errors fail them.
MAX_ABS = 0.01
MAX_COSINE_DISTANCE = 0.001


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = Request(base.rstrip("/") + path, data=data,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=900) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def post(base, path, payload):
    status, body = request(base, path, payload)
    if status != 200:
        raise ValueError(f"{path}: HTTP {status}: {body}")
    return body


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_runtime(props, binary):
    libraries = [{"path": p.name, "sha256": sha(p)} for p in sorted(binary.parent.glob("*.so*"))
                 if p.is_file() and not p.is_symlink()]
    if not libraries:
        raise ValueError("reference shared libraries must accompany the pinned server binary")
    return {"build_info": props["build_info"], "binary_sha256": sha(binary),
            "libraries": libraries, "required_device": "cpu", "pooling": "last",
            "normalization": "python_l2", "reference_context": 8192,
            "executed_token_limit": LIMIT}


def normalize(values):
    if len(values) != DIMENSIONS or not all(math.isfinite(x) for x in values):
        raise ValueError("invalid reference embedding")
    norm = math.sqrt(sum(x*x for x in values))
    if norm == 0:
        raise ValueError("zero reference embedding")
    return [x / norm for x in values]


def compare(actual, expected, *, must_match=True):
    if len(actual) != DIMENSIONS or not all(math.isfinite(x) for x in actual):
        raise ValueError("invalid embedding dimensions or non-finite values")
    norm = math.sqrt(sum(x*x for x in actual))
    normalized = normalize(actual)
    distance = max(0.0, 1 - sum(x*y for x,y in zip(normalized, normalize(expected))))
    max_abs = max(abs(x-y) for x,y in zip(actual, expected))
    matches = max_abs <= MAX_ABS and distance <= MAX_COSINE_DISTANCE and abs(norm-1) < 1e-5
    if matches != must_match:
        raise AssertionError(f"vector comparison: match={matches}, cosine_distance={distance}, max_abs={max_abs}, norm={norm}")
    return {"max_abs": max_abs, "cosine_distance": distance, "norm": norm, "matches": matches}


def cases():
    doc = "search_document"
    query = "search_query"
    specs = [
        ("document_berlin", doc, "Berlin ist die Hauptstadt von Deutschland."),
        ("document_paris", doc, "Paris ist die Hauptstadt von Frankreich."),
        ("document_bees", doc, "Bienen produzieren Honig."),
        ("query_berlin", query, "Was ist die Hauptstadt von Deutschland?"),
        ("query_paris", query, "Was ist die Hauptstadt von Frankreich?"),
        ("document_unicode", doc, "Straße 😀 e\u0301 東京\nZweite Zeile {text}."),
        ("query_unicode", query, "Straße 😀 e\u0301 東京\nZweite Zeile {text}."),
        ("document_special", doc, "Berlin <s> Hamburg </s> München <unk> Ende."),
        ("query_special", query, "Berlin <s> Hamburg </s> München <unk> Ende."),
        ("empty_document", doc, ""),
        ("empty_query", query, ""),
        ("boundary_document", doc, "a " * 4093 + "b"),
    ]
    return [{"name": name, "request": {"model": MODEL, "input_type": role, "input": text}}
            for name,role,text in specs]


def format_text(case):
    req = case["request"]
    return (QUERY_PREFIX if req["input_type"] == "search_query" else "") + req["input"] + "</s>"


def tokens(base, content, *, special=True):
    return post(base, "/tokenize", {"content": content, "add_special": special,
                                    "parse_special": True})["tokens"]


def oracle(base, case):
    text = format_text(case)
    before = tokens(base, text)
    assert len(before) <= LIMIT
    result = post(base, "/embedding", {"content": text, "embd_normalize": -1})
    vector = result[0]["embedding"][0]
    return {**case, "formatted_token_count": len(before), "executed_tokens": before,
            "embedding": normalize(vector)}


def reference_controls(pooled, unpooled):
    case = cases()[0]
    text = format_text(case)
    hidden = post(unpooled, "/embedding", {"content": text, "embd_normalize": -1})[0]["embedding"]
    expected = oracle(pooled, case)["embedding"]
    result = {"last_token": compare(normalize(hidden[-1]), expected),
              "wrong_first_token": compare(normalize(hidden[0]), expected, must_match=False),
              "wrong_mean_pooling": compare(normalize([sum(v)/len(hidden) for v in zip(*hidden)]),
                                             expected, must_match=False)}
    missing_eos = post(pooled, "/embedding", {"content": case["request"]["input"],
                                              "embd_normalize": -1})[0]["embedding"][0]
    result["missing_eos"] = compare(normalize(missing_eos), expected, must_match=False)
    query = cases()[3]
    missing_prefix = post(pooled, "/embedding", {"content": query["request"]["input"] + "</s>",
                                                 "embd_normalize": -1})[0]["embedding"][0]
    result["missing_query_prefix"] = compare(normalize(missing_prefix), oracle(pooled, query)["embedding"],
                                              must_match=False)
    return result


def check_fixture(fixture):
    assert fixture["model"] == MODEL and fixture["profile_id"] == PROFILE
    assert fixture["artifact"]["weight_sha256"] == WEIGHT_SHA
    assert fixture["artifact"]["manifest_sha256"] == MANIFEST_SHA
    assert [{"name": c["name"], "request": c["request"]} for c in fixture["cases"]] == cases()
    for case in fixture["cases"]:
        compare(case["embedding"], case["embedding"])
        assert len(case["executed_tokens"]) <= LIMIT


def check_live(base, fixture):
    sys.path.insert(0, str(ROOT / "services/kiron-common"))
    from kiron_common.embedding_contract import canonical_json, validate_capabilities
    status, tags = request(base, "/api/tags")
    assert status == 200
    rows = [r for r in tags["models"] if r["name"] == MODEL]
    assert len(rows) == 1
    cap = rows[0]["kiron_capabilities"]
    validate_capabilities(cap)
    for alias in [MODEL, "e5-mistral-7b-instruct", "hellord/e5-mistral-7b-instruct"]:
        show = post(base, "/api/show", {"model": alias})
        assert canonical_json(show["kiron_capabilities"]) == canonical_json(cap)
    profile = next(p for p in cap["profiles"] if p["profile_id"] == PROFILE)
    assert profile["verification"]["status"] == "verified"
    assert profile["index_compatibility_id"] and profile["query_compatibility_id"]
    assert "reference-vectors:scripts/fixtures/hellord-e5-mistral-v1.reference.json@sha256:" + sha(FIXTURE) in profile["verification"]["evidence"]
    vectors = {}
    checks = []
    for case in fixture["cases"]:
        response = post(base, "/api/embed", {**case["request"], "keep_alive": "10m"})
        assert len(response["embeddings"]) == 1
        assert response["prompt_eval_count"] == len(case["executed_tokens"])
        vector = response["embeddings"][0]
        vectors[case["name"]] = vector
        checks.append({"name": case["name"], **compare(vector, case["embedding"])})
        print(case["name"], "passed", flush=True)
    for role in ("search_document", "search_query"):
        batch_cases = [c for c in fixture["cases"] if c["request"]["input_type"] == role][:3]
        body = {"model": "e5-mistral-7b-instruct", "input_type": role,
                "input": [c["request"]["input"] for c in batch_cases], "truncate": False}
        response = post(base, "/api/embed", body)
        assert len(response["embeddings"]) == len(batch_cases)
        assert response["prompt_eval_count"] == sum(len(c["executed_tokens"]) for c in batch_cases)
        for case, vector in zip(batch_cases, response["embeddings"]):
            checks.append({"name": "batch_" + case["name"], **compare(vector, case["embedding"])})
    rankings = {}
    for city in ("berlin", "paris"):
        query = vectors["query_" + city]
        scores = {name: sum(x*y for x,y in zip(query, vectors["document_" + name]))
                  for name in ("berlin", "paris", "bees")}
        assert max(scores, key=scores.get) == city
        rankings[city] = scores
    valid = cases()[0]["request"]
    invalid = [({k:v for k,v in valid.items() if k != "input_type"}, "missing_required_input_type"),
               ({**valid, "input_type": "classification"}, "unsupported_input_type"),
               ({**valid, "dimensions": 128}, "embedding_profile_parameter_conflict"),
               ({**valid, "truncate": True}, "embedding_profile_parameter_conflict"),
               ({**valid, "options": {"num_ctx": 8192}}, "embedding_profile_parameter_conflict"),
               ({**valid, "options": {"rope_frequency_scale": 2}}, "embedding_profile_parameter_conflict")]
    guard_results = []
    for body,code in invalid:
        status, response = request(base, "/api/embed", body)
        assert status == 400 and response["error"]["code"] == code
        guard_results.append(code)
    overflow_results = []
    for role in ("search_document", "search_query"):
        status, response = request(base, "/api/embed", {"model": MODEL, "input_type": role,
                                                        "input": "a " * 4094 + "b"})
        assert status == 400 and "context length" in response["error"]
        overflow_results.append({"role": role, "http_status": status})
    return {"profile": profile, "checks": checks, "vector_count": len(checks),
            "query_rankings": rankings, "guards": guard_results,
            "overflow_rejection": overflow_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-url")
    parser.add_argument("--unpooled-url")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"profile_id": PROFILE, "tolerances": {"max_abs": MAX_ABS,
               "max_cosine_distance": MAX_COSINE_DISTANCE}}
    if args.write_reference and (not args.reference_url or not args.unpooled_url):
        parser.error("freezing requires both pooled and unpooled reference servers")
    if args.reference_url:
        if not args.model_file or not args.reference_binary:
            parser.error("CPU verification requires the model file and pinned reference binary")
        assert sha(args.model_file) == WEIGHT_SHA
        props = request(args.reference_url, "/props")[1]
        assert Path(props["model_path"]).resolve() == args.model_file.resolve()
        artifact = {"manifest_sha256": MANIFEST_SHA, "weight_sha256": WEIGHT_SHA,
                    "size_bytes": args.model_file.stat().st_size}
        assert props["default_generation_settings"]["n_ctx"] == 8192
        runtime = reference_runtime(props, args.reference_binary)
        if args.write_reference:
            controls = reference_controls(args.reference_url, args.unpooled_url)
            frozen = []
            for case in cases():
                frozen.append(oracle(args.reference_url, case))
                print(case["name"], "reference computed", flush=True)
            fixture = {"schema_version": 1, "model": MODEL, "profile_id": PROFILE,
                       "artifact": artifact, "reference_runtime": runtime,
                       "negative_controls": controls, "cases": frozen}
            check_fixture(fixture)
            FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
        else:
            fixture = json.loads(FIXTURE.read_text())
            assert runtime == fixture["reference_runtime"]
            comparisons = []
            for case in fixture["cases"]:
                actual = oracle(args.reference_url, case)
                assert actual["executed_tokens"] == case["executed_tokens"]
                comparisons.append({"name": case["name"], **compare(actual["embedding"], case["embedding"])})
                print(case["name"], "CPU reference passed", flush=True)
            report["cpu_checks"] = comparisons
        report["reference_runtime"] = runtime
    fixture = json.loads(FIXTURE.read_text())
    check_fixture(fixture)
    if args.base_url:
        report["live"] = check_live(args.base_url, fixture)
    if not args.reference_url and not args.base_url:
        parser.error("choose a CPU reference or live verification")
    report.update(status="passed", fixture_sha256=sha(FIXTURE))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

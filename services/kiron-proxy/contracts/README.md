# OpenAI SDK contracts for Local Inference Profile v1

This directory freezes the SDK side of Tareas #285, preserves direct-upstream
fixtures from #286, and tests KIron's text API boundary for #288. Baseline and
upstream fixture tests use `httpx.MockTransport`; the API integration tests use
`httpx.ASGITransport` with the real application in-process. All tests are offline;
none demonstrate live provider or GPU capabilities.

Use an isolated Python 3.12 test environment outside the repository and the
production service environments:

```bash
python3 -m venv /usr/lib/kiron/test-venvs/local-inference
/usr/lib/kiron/test-venvs/local-inference/bin/python -m pip install -r services/kiron-proxy/requirements-contract.txt
/usr/lib/kiron/test-venvs/local-inference/bin/python -m pip install --no-deps starlette==0.52.1 pymysql==1.2.3 Pillow==12.3.0
PYTHONPATH=services/kiron-common:services/kiron-proxy /usr/lib/kiron/test-venvs/local-inference/bin/python -m unittest discover -s services/kiron-proxy/contracts -p 'test_*.py' -v
```

`requirements-contract.txt` pins the SDK and its complete runtime dependencies;
it is independent of the proxy's production requirements. The source-wheel
SHA-256 and selected installed SDK source hashes in `openai-sdk-snapshot.json`
freeze the normative schema/serialization snapshot. The snapshot covers only
the profile's selected types and SDK behavior, not the whole OpenAI platform.
No upstream source code is vendored. Changing the SDK requires a reviewed
snapshot, lock and profile update; the tests deliberately reject drift.

The SDK baseline fixtures cover Models, Chat function-call roundtrips and interleaved tool
streams with usage, Responses text/function/reasoning items and streaming,
embedding encodings, Bearer auth, JSON/SSE parsing and standard errors.
Pydantic validates response/event fixtures and the SDK runs with response
validation enabled, except for a tested upstream limitation: SDK 2.29.0 types
`Embedding.embedding` as `list[float]`, so strict validation rejects a base64
vector before its decoding post-parser. Base64/default-encoding tests therefore
use ordinary SDK parsing and separately verify the binary float32 payload.
The SDK requests base64 when the caller omits `encoding_format`.

The SDK's Chat `.stream()` helper also attempts to finish a tool's strict JSON
arguments when a different tool index arrives. Interleaved parallel calls can
therefore raise `JSONDecodeError` with `strict=True`. The baseline checks strict
tool transport via `.create(stream=True)` and final assembly via `.stream()`
with non-strict tools as a diagnostic of upstream behavior. The public KIron
profile requires contiguous argument blocks in ascending call-index order,
without returning to a completed call. Providers may interleave internally;
the server must buffer within the profile's limits before public emission.
A separate positive test exercises parallel `strict=True` tools through the
ordinary `.stream()` helper with that emission order, including parsed final
arguments and usage. Clients need no workaround. These baseline fixtures isolate the required SDK ordering.
`test_openai_tools_sdk.py` and the actual wire/API tests now exercise KIron's
bounded buffering and strict helper assembly; baseline fixtures alone do not
substitute for that implementation coverage.

Responses streaming helpers assemble text and function arguments. They expose
reasoning delta events but obtain final reasoning items from the complete
`response.completed` envelope; do not omit final output items. Text delta/done
events include required `logprobs`, and function-argument done events include
the required function `name`.

`test_prism_wire_fixtures.py` additionally repeats 13 SDK replays of five real
direct-upstream responses from Tareas #286: text and one tool, each as JSON
and SSE, plus a parallel-tool SSE stream. `fixtures/prism-b10709/` preserves
the original request, response metadata and raw body bytes. Its manifest
records their source paths and SHA-256 hashes, the original offline replay
reports, release commit and binary identity. Tests need only these committed
fixtures, not the original host paths or a running model. Ordinary and strict
SDK parsing, strict streaming helpers, tool arguments/IDs, usage and the
observed contiguous parallel-call order are checked without rewriting the
responses. These successful samples do not establish general tool-schema
enforcement, named tool-choice enforcement or KIron API conformance.

`test_openai_api_sdk.py` sends ordinary pinned `AsyncOpenAI` requests, with strict
response validation, through the real ASGI application and `RuntimeService`.
An actual resolver and isolated admission store connect typed Ollama/Prism fake
providers. Tests check canonical Models list/detail/alias records, text completion,
the SDK streaming helper with usage, authentication, unknown/unavailable models,
and capability/syntax rejection before any provider I/O. A foreign completion
request ID and a partial stream without a terminal event produce SDK errors and
retain unknown admission tickets. Successful executions require admission before
provider execution and release their request tickets afterward. The additional
ASGI/log-store import dependencies above belong only to the test environment;
the frozen SDK dependency pins remain unchanged.

`test_openai_wire_sdk.py` separately checks the actual completion/SSE serializer
bytes through strict SDK parsing with an in-memory mock transport.

`test_openai_responses_sdk.py` checks the actual Responses codec with SDK 2.29.0
strict validation: nonstream output, raw SSE and the stream helper, complete
reasoning items, incomplete/failed terminals, exact usage details, parallel strict
tool output replay without removing SDK annotations, and structured parsing.
Its vision case sends SDK input through the shared bounded Pillow decoder and
checks the original text/image order. These tests use MockTransport, with no
model or GPU. Pillow is a test dependency for that decoder case; it is not part
of the SDK's own frozen dependency closure.

These are offline API/codec contracts with controlled providers, not a live
backend acceptance result. Each feature still needs its explicit provider gate. Passing the SDK baseline alone is never
a KIron conformance result. Acceptance-test ownership and the remaining cases are in
[`docs/local-inference-validation-v1.md`](../../../docs/local-inference-validation-v1.md).

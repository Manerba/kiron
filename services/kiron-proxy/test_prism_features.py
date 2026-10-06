"""Adapter-boundary proofs; these fixtures do not enable any native capability."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import unittest

import httpx

from kiron_common.local_inference import (
    Capability, CapabilityEvidence, CapabilityName, CapabilitySet, CapabilityStatus,
    ErrorCode, EventKind, FinishReason, GenerationOptions, LocalInferenceError,
    OutputFormat, OutputFormatKind, ParameterConstraint, ReasoningOptions, SamplingOptions,
)
import test_prism_provider as baseline


SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}},
          "required": ["n"], "additionalProperties": False}


class PrismFeatureTests(unittest.IsolatedAsyncioTestCase):
    setUp = baseline.PrismProviderTests.setUp
    make = baseline.PrismProviderTests.make

    def capability(self, adapter, name, values):
        artifact = self.deployment.artifact_identity
        implementation = adapter.implementation
        evidence = CapabilityEvidence(implementation.provider_revision, artifact.fingerprint,
            artifact.sha256, None, implementation.template_revision, implementation.parser_revision,
            self.deployment.configuration_fingerprint, "fixture-only", datetime.now(timezone.utc))
        previous = adapter._capabilities.get(self.deployment.id, CapabilitySet())
        cap = Capability(CapabilityStatus.SUPPORTED,
            {key: ParameterConstraint(allowed_values=value) for key, value in values.items()}, (evidence,))
        adapter._capabilities[self.deployment.id] = CapabilitySet({**previous.by_name, name: cap})

    def structured(self, adapter):
        self.capability(adapter, CapabilityName.STRUCTURED_OUTPUT, {
            "formats": ("json_object", "json_schema"), "strict": (False, True),
            "schema_keywords": ("type", "properties", "required", "additionalProperties"),
            "schema_types": ("object", "integer"), "schema_variants": (),
        })
        return replace(self.request, options=GenerationOptions(8, output_format=OutputFormat(
            OutputFormatKind.JSON_SCHEMA, SCHEMA, "number", True)))

    def reasoning(self, adapter):
        self.capability(adapter, CapabilityName.REASONING, {
            "efforts": ("low",), "token_rule": ("initial-block-token-ids-v1",),
            "template_sha256": ("e" * 64,), "budget_tokens.low": (32,),
            "generation_prefix": ("<think>\n",), "initial_state": ("prefilled",),
            "opening_id": (9,), "opening_text": ("<think>",),
            "closing_id": (11,), "closing_text": ("</think>",),
            "eos_ids": (14,), "eos_texts": ("<|im_end|>",), "tool_ids": (), "tool_texts": (),
        })
        return replace(self.request, options=GenerationOptions(8, reasoning=ReasoningOptions(effort="low")))

    @staticmethod
    def completion(text="OK", reasoning=None):
        message = {"role": "assistant", "content": text}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {"model": "bonsai-probe", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    @staticmethod
    def verbose(stream=False):
        return {"tokens": [10, 11, 12, 13, 14], "tokens_predicted": 5, "stop": True, "stop_type": "eos",
            "generation_settings": {"stream": stream, "generation_prompt": "<think>\n"},
            "content": "I</think>\n\nOK"}

    @staticmethod
    def pieces():
        return {"tokens": [{"id": token, "piece": piece} for token, piece in
            [(10, "I"), (11, "</think>"), (12, "\n\n"), (13, "OK"), (14, "<|im_end|>")]]}

    @staticmethod
    def stream_bytes(text="OK", reasoning=None, verbose=None):
        delta = {"content": text}
        if reasoning is not None:
            delta["reasoning_content"] = reasoning
        first = {"model": "bonsai-probe", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        last = {"model": "bonsai-probe", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        if verbose is not None:
            last["__verbose"] = verbose
        return ("".join("data: " + json.dumps(frame) + "\n\n" for frame in [first, last]) + "data: [DONE]\n\n").encode()

    async def test_structured_native_mapping_and_success_validation(self):
        seen = []
        async def native(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=self.completion('{"n":3}'))
        adapter = await self.make(inference_handler=native)
        result = await adapter.chat(self.structured(adapter))
        self.assertEqual(result.content[0].text, '{"n":3}')
        self.assertEqual(seen[0]["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertFalse(seen[0]["chat_template_kwargs"]["enable_thinking"])

    async def test_invalid_structured_success_is_not_completed(self):
        async def native(request):
            body = json.loads(request.content)
            if body["stream"]:
                return httpx.Response(200, content=self.stream_bytes('{"n":"wrong"}'))
            return httpx.Response(200, json=self.completion('{"n":"wrong"}'))
        adapter = await self.make(inference_handler=native)
        request = self.structured(adapter)
        with self.assertRaises(LocalInferenceError) as raised:
            await adapter.chat(request)
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)
        events = [event async for event in adapter.stream(request)]
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertFalse(any(event.kind is EventKind.COMPLETED for event in events))

    async def test_reasoning_original_id_lookup_keeps_generation_and_exact_usage(self):
        seen = []
        async def native(request):
            body = json.loads(request.content)
            seen.append((request.url.path, body, request.headers.get("authorization")))
            if request.url.path == "/tokenize":
                return httpx.Response(200, json=self.pieces())
            value = self.completion(reasoning="I")
            value["__verbose"] = self.verbose()
            return httpx.Response(200, json=value)
        adapter = await self.make(inference_handler=native)
        result = await adapter.chat(self.reasoning(adapter))
        self.assertEqual(result.reasoning[0].text, "I")
        self.assertEqual(result.usage.reasoning_output_tokens, 2)
        self.assertEqual(result.usage.output_tokens, 5)
        self.assertEqual(seen[0][1]["reasoning_budget_tokens"], 32)
        self.assertEqual(seen[1][1], {"content": [10, 11, 12, 13, 14], "with_pieces": True,
                                    "add_special": False, "parse_special": False})
        self.assertEqual([item[2] for item in seen], ["Bearer kiron-prism-child"] * 2)

    async def test_stream_reasoning_requires_final_original_ids_before_success(self):
        async def native(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json=self.pieces())
            return httpx.Response(200, content=self.stream_bytes(reasoning="I", verbose=self.verbose(True)))
        adapter = await self.make(inference_handler=native)
        events = [event async for event in adapter.stream(self.reasoning(adapter))]
        self.assertEqual(events[-1].kind, EventKind.COMPLETED)
        self.assertEqual(events[-2].usage.reasoning_output_tokens, 2)
        self.assertEqual([event.text for event in events if event.kind is EventKind.REASONING_DELTA], ["I"])

    async def test_missing_ids_or_changed_lookup_never_fabricate_usage(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                async def native(request):
                    if request.url.path == "/tokenize":
                        return httpx.Response(401, json={"error": "old generation"})
                    return httpx.Response(200, content=self.stream_bytes(reasoning="I",
                        verbose=None if missing else self.verbose(True)))
                adapter = await self.make(inference_handler=native)
                events = [event async for event in adapter.stream(self.reasoning(adapter))]
                self.assertEqual(events[-1].kind, EventKind.FAILED)
                self.assertFalse(any(event.kind in {EventKind.USAGE, EventKind.COMPLETED} for event in events))

    async def test_cancelled_original_id_lookup_does_not_complete(self):
        entered = asyncio.Event()
        async def native(request):
            if request.url.path == "/tokenize":
                entered.set()
                await asyncio.Event().wait()
            return httpx.Response(200, content=self.stream_bytes(reasoning="I", verbose=self.verbose(True)))
        adapter = await self.make(inference_handler=native)
        request = self.reasoning(adapter)
        async def collect():
            return [event async for event in adapter.stream(request)]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 1)
        self.context.cancellation.set()
        events = await asyncio.wait_for(task, 1)
        self.assertEqual(events[-1].kind, EventKind.CANCELLED)

    async def test_unverified_effort_and_ambiguous_stop_rejected_before_inference(self):
        adapter = await self.make()
        request = self.reasoning(adapter)
        for options in (replace(request.options, reasoning=ReasoningOptions(effort="high")),
                        replace(request.options, sampling=SamplingOptions(stop=("X",)))):
            with self.subTest(options=options), self.assertRaises(LocalInferenceError):
                await adapter.chat(replace(request, options=options))
        self.assertFalse(any(path == "/v1/chat/completions" for path, _ in self.calls))

    async def test_duplicate_native_json_is_rejected(self):
        async def native(request):
            value = json.dumps(self.completion())
            return httpx.Response(200, content=value[:-1] + ',"model":"foreign"}')
        adapter = await self.make(inference_handler=native)
        with self.assertRaises(LocalInferenceError) as raised:
            await adapter.chat(self.request)
        self.assertEqual(raised.exception.failure.code, ErrorCode.PROVIDER_ERROR)

    async def test_direct_base_profile_and_streaming_constraints_precede_io(self):
        adapter = await self.make()
        for options in (GenerationOptions(4096), GenerationOptions(8, sampling=SamplingOptions(frequency_penalty=1.0))):
            with self.subTest(options=options), self.assertRaises(LocalInferenceError):
                await adapter.chat(replace(self.request, options=options))
        current = adapter._capabilities[self.deployment.id]
        adapter._capabilities[self.deployment.id] = CapabilitySet({
            CapabilityName.CHAT: current.by_name[CapabilityName.CHAT]})
        with self.assertRaises(LocalInferenceError):
            await anext(adapter.stream(self.request))
        adapter._capabilities.clear()
        with self.assertRaises(LocalInferenceError):
            await adapter.chat(self.request)
        self.assertFalse(any(path == "/v1/chat/completions" for path, _ in self.calls))

    async def test_incomplete_or_ambiguous_reasoning_policies_fail_before_io(self):
        adapter = await self.make()
        request = self.reasoning(adapter)
        original = adapter._capabilities[self.deployment.id]
        cap = original.by_name[CapabilityName.REASONING]
        for key, constraint in (
                ("template_sha256", ParameterConstraint(allowed_values=("unknown",))),
                ("budget_tokens.low", ParameterConstraint(allowed_values=(True,))),
                ("eos_ids", ParameterConstraint(allowed_values=(14, 15))),
                ("opening_id", ParameterConstraint(allowed_values=(11,))),
                ("token_rule", ParameterConstraint(allowed_values=("unknown",))),
                ("initial_state", ParameterConstraint(allowed_values=("disabled",)))):
            with self.subTest(key=key):
                invalid = replace(cap, constraints={**cap.constraints, key: constraint})
                adapter._capabilities[self.deployment.id] = CapabilitySet({**original.by_name,
                    CapabilityName.REASONING: invalid})
                with self.assertRaises(LocalInferenceError) as raised:
                    await adapter.chat(request)
                self.assertEqual(raised.exception.failure.code, ErrorCode.INVALID_CONFIGURATION)
        self.assertFalse(any(path == "/v1/chat/completions" for path, _ in self.calls))

    async def test_explicit_disabled_count_policy_proves_zero_from_original_ids(self):
        seen = []
        async def native(request):
            seen.append(request.url.path)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [{"id": 13, "piece": "OK"},
                                                           {"id": 14, "piece": "<|im_end|>"}]})
            value = self.completion()
            value["usage"].update(completion_tokens=2, total_tokens=12)
            value["__verbose"] = {**self.verbose(), "tokens": [13, 14], "tokens_predicted": 2,
                "content": "OK", "generation_settings": {"generation_prompt": "<think></think>\n", "stream": False}}
            return httpx.Response(200, json=value)
        adapter = await self.make(inference_handler=native)
        self.reasoning(adapter)
        caps = adapter._capabilities[self.deployment.id]
        constraints = {**caps.by_name[CapabilityName.CHAT].constraints,
                       **caps.by_name[CapabilityName.REASONING].constraints,
            "usage_fields": ParameterConstraint(allowed_values=("reasoning_output_tokens",)),
            "token_rule": ParameterConstraint(allowed_values=("disabled-token-ids-v1",)),
            "generation_prefix": ParameterConstraint(allowed_values=("<think></think>\n",)),
            "initial_state": ParameterConstraint(allowed_values=("disabled",))}
        adapter._capabilities[self.deployment.id] = CapabilitySet({**caps.by_name,
            CapabilityName.CHAT: replace(caps.by_name[CapabilityName.CHAT], constraints=constraints)})
        result = await adapter.chat(self.request)
        self.assertEqual(result.usage.reasoning_output_tokens, 0)
        self.assertEqual(seen, ["/v1/chat/completions", "/tokenize"])

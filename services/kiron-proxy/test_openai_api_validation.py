"""Closed v1 HTTP validation; replaces direct-Ollama translation tests.

The API resolves canonical requests through RuntimeService. Native CPU options,
lease-marker shortcuts and Ollama HTTP fixtures belong to the native port tests.
Their public replacements verify rejection before resolution/inference, typed
runtime failures, real canonical completion and ASGI-owned cancellation cleanup.
"""
import asyncio
import json
from unittest import mock

from test_openai_runtime_api import RuntimeApiFixture, CHAT
import openai_api


class ApiValidationTests(RuntimeApiFixture):
    async def assert_rejected(self, body, *, code="invalid_request", status=400):
        before = list(self.runtime.calls)
        result = await self.client.post("/v1/chat/completions",json=body)
        self.assertEqual(result.status_code,status,result.text)
        self.assertEqual(result.json()["error"]["code"],code)
        self.assertEqual(self.runtime.calls,before)
        self.assertIn("x-request-id",result.headers)

    async def test_false_is_not_an_unset_container_or_integer(self):
        for name in ("tools","functions","response_format","top_logprobs","max_tokens","n"):
            with self.subTest(name=name):
                await self.assert_rejected({**CHAT,name:False})

    async def test_documented_noops_and_nullable_sampling_are_accepted(self):
        fields = {"tools":[],"tool_choice":"none","parallel_tool_calls":False,
                  "logit_bias":{},"logprobs":False,"top_logprobs":0,"store":False,
                  "response_format":{"type":"text"},"temperature":None,"reasoning_effort":None}
        response = await self.client.post("/v1/chat/completions",json={**CHAT,**fields})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.runtime.calls,["resolve","validate_chat","public_model_for","chat"])

    async def test_unknown_fields_are_rejected_at_top_and_message_and_part_levels(self):
        requests = [{**CHAT,"extra":None}, {**CHAT,"options":{"num_gpu":0}},
            {**CHAT,"messages":[{"role":"user","content":"hi","extra":None}]},
            {**CHAT,"messages":[{"role":"user","content":[{"type":"text","text":"hi","extra":False}]}]},
            {**CHAT,"response_format":{"type":"text","extra":None}},
            {**CHAT,"stream":True,"stream_options":{"include_usage":True,"extra":[]}}]
        for body in requests:
            with self.subTest(body=body):
                await self.assert_rejected(body,code="unsupported_parameter")

    async def test_supported_feature_syntax_reaches_capability_check_before_provider_io(self):
        from types import SimpleNamespace
        from kiron_common.local_inference import CapabilitySet
        from runtime_service import RuntimeService
        async def validate(parsed, model, context):
            self.runtime.check("validate_chat", context)
            request = parsed.to_request(model, context, 16)
            RuntimeService._validate_features(request, CapabilitySet({}), SimpleNamespace(implementation=None))
            return request
        self.runtime.validate_chat = validate
        tool = {"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}
        for field,value in (("tools",[tool]), ("response_format",{"type":"json_object"}),
                            ("parallel_tool_calls",True), ("reasoning_effort","high")):
            with self.subTest(field=field):
                self.runtime.calls.clear()
                result = await self.client.post("/v1/chat/completions",json={**CHAT,field:value})
                self.assertEqual(result.status_code,400,result.text)
                self.assertEqual(result.json()["error"]["code"],"unsupported_capability")
                self.assertEqual(self.runtime.calls,["resolve","validate_chat"])
        self.runtime.calls.clear()
        for field,value in (("modalities",["audio"]),("store",True),("user","secret")):
            with self.subTest(field=field):
                await self.assert_rejected({**CHAT,field:value},code="unsupported_parameter")

    async def test_strict_types_ranges_and_conflicting_budgets(self):
        for fields in ({"temperature":True},{"temperature":3},{"top_p":-1},{"max_tokens":0},
                       {"max_tokens":4,"max_completion_tokens":4},{"seed":2**63},{"stream":None},
                       {"stream_options":{"include_usage":False}},{"stop":[]},{"messages":[]}):
            with self.subTest(fields=fields):
                await self.assert_rejected({**CHAT,**fields})

    async def test_duplicate_keys_invalid_utf8_nonfinite_and_depth_fail_before_resolution(self):
        raw = [b'{"model":"alias","model":"other","messages":[]}', b'\xff',
               b'{"model":"alias","messages":[],"temperature":NaN}',
               b'{"x":'+b'['*33+b'0'+b']'*33+b'}', b'[]', b'{}']
        for body in raw:
            with self.subTest(body=body[:80]):
                response = await self.client.post("/v1/chat/completions",content=body,
                                                  headers={"Content-Type":"application/json"})
                self.assertEqual(response.status_code,400,response.text)
                self.assertEqual(response.json()["error"]["code"],"invalid_request")
        self.assertEqual(self.runtime.calls,[])

    async def test_content_type_charset_compression_and_duplicate_headers(self):
        for headers in ({"Content-Type":"text/plain"}, {"Content-Type":"application/json; charset=latin-1"},
                        {"Content-Type":"application/json; version=1"},
                        {"Content-Type":"application/json","Content-Encoding":"gzip"},
                        [("Content-Type","application/json"),("Content-Type","application/json")]):
            with self.subTest(headers=headers):
                response = await self.client.post("/v1/chat/completions",content=json.dumps(CHAT),headers=headers)
                self.assertEqual(response.status_code,415,response.text)
                self.assertEqual(response.json()["error"]["code"],"unsupported_media_type")
        self.assertEqual(self.runtime.calls,[])
        response = await self.client.post("/v1/chat/completions",content=json.dumps(CHAT),
                                          headers={"Content-Type":'application/json; charset="UTF-8"'})
        self.assertEqual(response.status_code,200,response.text)

    async def test_declared_and_chunked_body_limits_without_large_allocations(self):
        body = json.dumps(CHAT).encode()
        async def chunks():
            for index in range(0,len(body),7):
                yield body[index:index+7]
        with mock.patch.object(openai_api,"MAX_BODY_SIZE",len(body)-1):
            declared = await self.client.post("/v1/chat/completions",content=body,
                                             headers={"Content-Type":"application/json"})
            chunked = await self.client.post("/v1/chat/completions",content=chunks(),
                                            headers={"Content-Type":"application/json"})
        self.assertEqual(declared.status_code,413)
        self.assertEqual(chunked.status_code,413)
        self.assertEqual(self.runtime.calls,[])
        with mock.patch.object(openai_api,"MAX_BODY_SIZE",len(body)):
            boundary = await self.client.post("/v1/chat/completions",content=chunks(),
                                             headers={"Content-Type":"application/json"})
        self.assertEqual(boundary.status_code,200,boundary.text)

    async def test_invalid_content_length_is_json_400(self):
        for length in ("-1","nope"):
            with self.subTest(length=length):
                response = await self.client.post("/v1/chat/completions",content=b'{}',
                    headers={"Content-Type":"application/json","Content-Length":length})
                self.assertEqual(response.status_code,400)
                self.assertEqual(response.json()["error"]["code"],"invalid_request")
        self.assertEqual(self.runtime.calls,[])

    async def test_vision_errors_keep_public_status_type_and_retry_contract(self):
        from openai_vision import VisionError
        for status, code, kind in ((400,"invalid_request","invalid_request_error"),
                                   (429,"overloaded","rate_limit_error"),
                                   (503,"model_unavailable","server_error"),
                                   (504,"timeout","server_error")):
            with self.subTest(status=status), mock.patch.object(openai_api,"decode_chat_images",
                    side_effect=VisionError(code,"messages[0].content",status)):
                response = await self.client.post("/v1/chat/completions",json=CHAT)
                self.assertEqual(response.status_code,status,response.text)
                self.assertEqual(response.json()["error"]["type"],kind)
                self.assertEqual(response.json()["error"]["code"],code)
                self.assertIn("x-request-id",response.headers)
                if status == 429:
                    self.assertEqual(response.headers["retry-after"],"1")
        self.assertEqual(self.runtime.calls,[])

    async def test_request_id_is_server_owned_and_never_reuses_client_operation_id(self):
        responses = [await self.client.post("/v1/chat/completions",json=CHAT,
                     headers={"X-Request-ID":"same-untrusted-id"}) for _ in range(2)]
        ids = [response.headers["x-request-id"] for response in responses]
        self.assertNotEqual(ids[0],ids[1])
        self.assertTrue(all(len(value)==16 and value!="same-untrusted-id" for value in ids))

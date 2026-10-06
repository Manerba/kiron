"""Embedding protocol validation without tokenizers, models or GPU access."""
import asyncio
import base64
from dataclasses import replace
import math
import struct
import time
import unittest
from unittest.mock import patch

from kiron_common.local_inference import (
    ArtifactIdentity, EmbeddingRequest, EmbeddingResult, EmbeddingRole, ErrorCode,
    LocalInferenceError, RequestContext, ResolvedDeployment, ResolvedModel,
    RuntimeFailure, TokenUsage,
)
from kiron_common.model_catalog import ArtifactFormat, ArtifactType, BackendType, LoaderType, ModelEndpoint, ModelTask
import openai_embeddings as wire
from openai_wire import ApiError
from test_openai_runtime_api import RuntimeApiFixture


def model():
    deployment = ResolvedDeployment("embed",BackendType.OLLAMA,"fixture:latest",
        ArtifactIdentity(ArtifactType.OLLAMA,ArtifactFormat.OLLAMA_MANIFEST,manifest_digest="sha256:"+"a"*64),
        LoaderType.OLLAMA,None,"b"*64)
    return ResolvedModel("alias",deployment,"dense-v1",EmbeddingRole.QUERY,"c"*64,
        task=ModelTask.EMBEDDING,endpoint=ModelEndpoint.EMBED,
        profile_metadata={"kind":"dense","dimensions":3},created=123,canonical_model_id="dense-v1.query")


def canonical_request(inputs=("hello",), dimensions=None):
    return EmbeddingRequest(model(),inputs,RequestContext("embedding-request",time.monotonic()+10,asyncio.Event()),dimensions=dimensions)


class EmbeddingParserTests(unittest.TestCase):
    def test_scalar_batch_whitespace_and_encoding_defaults_are_preserved(self):
        for value, expected in ((" hello ",(" hello ",)), (["x","世界"],("x","世界"))):
            parsed = wire.parse_embeddings({"model":"dense-v1.query","input":value})
            self.assertEqual(parsed.inputs,expected)
            self.assertEqual(parsed.encoding_format,"float")
            self.assertIsNone(parsed.dimensions)
        parsed = wire.parse_embeddings({"model":"dense-v1.document","input":"x","encoding_format":"base64","dimensions":3})
        self.assertEqual((parsed.encoding_format,parsed.dimensions),("base64",3))

    def test_token_ids_rejected_without_text_conversion_or_backend(self):
        for value in ([1,2,3],[[1],[2,3]]):
            with self.subTest(value=value),self.assertRaises(ApiError) as caught:
                wire.parse_embeddings({"model":"m","input":value})
            self.assertEqual(caught.exception.code,"unsupported_capability")

    def test_closed_fields_types_lengths_and_empty_inputs(self):
        bad = [{"input":value} for value in ("",[],[""],False,1,None,["x",1],[["x"]],[False],[-1],[[]])]
        bad += [{"dimensions":value} for value in (0,-1,False,1.5,None)]
        bad += [{"encoding_format":value} for value in (None,False,"int8",[])]
        for changes in bad:
            with self.subTest(changes=changes),self.assertRaises(ApiError) as caught:
                wire.parse_embeddings({"model":"m","input":"x",**changes})
            self.assertEqual(caught.exception.code,"invalid_request")
        for key in ("input_type","user","stream","unknown"):
            with self.subTest(key=key),self.assertRaises(ApiError) as caught:
                wire.parse_embeddings({"model":"m","input":"x",key:False})
            self.assertEqual(caught.exception.code,"unsupported_parameter")
        with patch.object(wire,"MAX_INPUTS",2),patch.object(wire,"MAX_CHARACTERS",3):
            wire.parse_embeddings({"model":"m","input":["世界","abc"]})
            for value in (["a"]*3,"abcd"):
                with self.assertRaises(ApiError):
                    wire.parse_embeddings({"model":"m","input":value})


class EmbeddingOutputTests(unittest.TestCase):
    def test_float_and_base64_share_order_float32_bits_and_actual_token_usage(self):
        request = canonical_request(("query1","query2"))
        result = EmbeddingResult(request.context.request_id,((.1,-0.0,1.5),(2.0,3.0,4.0)),TokenUsage(17,0))
        floats = wire.serialize_embeddings(result,request=request)
        encoded = wire.serialize_embeddings(result,request=request,encoding_format="base64")
        self.assertEqual(floats["model"],"dense-v1.query")
        self.assertEqual(floats["usage"],{"prompt_tokens":17,"total_tokens":17})
        self.assertEqual([row["index"] for row in floats["data"]],[0,1])
        for float_row,base64_row in zip(floats["data"],encoded["data"]):
            data = base64.b64decode(base64_row["embedding"],validate=True)
            self.assertEqual(list(struct.unpack("<fff",data)),float_row["embedding"])
        self.assertEqual(math.copysign(1,floats["data"][0]["embedding"][1]),-1)

    def test_dimension_count_identity_usage_and_float32_overflow_fail_as_protocol(self):
        request = canonical_request()
        good = EmbeddingResult(request.context.request_id,((1.,2.,3.),),TokenUsage(3,0))
        malformed = [replace(good,request_id="other"),replace(good,vectors=((1.,2.),)),
                     replace(good,vectors=((1.,2.,3.),(4.,5.,6.))),replace(good,vectors=((1e100,2.,3.),))]
        for result in malformed:
            with self.subTest(result=result),self.assertRaises(ApiError) as caught:
                wire.serialize_embeddings(result,request=request)
            self.assertEqual((caught.exception.status,caught.exception.code),(502,"backend_protocol_error"))
        for vector in ((float("nan"),2.,3.),(True,2.,3.)):
            forged = replace(good)
            object.__setattr__(forged,"vectors",(vector,))
            with self.assertRaises(ApiError):
                wire.serialize_embeddings(forged,request=request)
        forged_usage = TokenUsage(3,0)
        object.__setattr__(forged_usage,"input_tokens",True)
        with self.assertRaises(ApiError):
            wire.serialize_embeddings(replace(good,usage=forged_usage),request=request)
        with patch.object(wire,"MAX_OUTPUT_BYTES",100),self.assertRaises(ApiError):
            wire.serialize_embeddings(good,request=request)


class EmbeddingApiTests(RuntimeApiFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.runtime.model = model()
        self.runtime.public = lambda: {"id":self.runtime.model.api_model_id,"created":123,"object":"model","owned_by":"ollama"}
        async def validate(parsed,resolved,context):
            self.runtime.check("validate_embedding",context)
            if parsed.dimensions not in (None,3):
                raise LocalInferenceError(RuntimeFailure(ErrorCode.UNSUPPORTED_VALUE,"dimension mismatch"))
            return parsed.to_request(resolved,context)
        async def embed(request):
            self.runtime.check("embed",request.context)
            self.runtime.requests.append(request)
            return EmbeddingResult(request.context.request_id,tuple((.1,0.,1.) for _ in request.inputs),TokenUsage(7,0))
        self.runtime.validate_embedding,self.runtime.embed = validate,embed

    async def test_real_route_preserves_unformatted_text_and_role_bound_profile(self):
        response = await self.client.post("/v1/embeddings",json={"model":"alias","input":["query: x","document: y"],"dimensions":3})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.runtime.calls,["resolve","validate_embedding","public_model_for","embed"])
        self.assertEqual(self.runtime.requests[-1].inputs,("query: x","document: y"))
        self.assertIs(self.runtime.requests[-1].model.embedding_role,EmbeddingRole.QUERY)
        self.assertEqual(response.json()["model"],"dense-v1.query")
        self.assertEqual(response.json()["usage"],{"prompt_tokens":7,"total_tokens":7})
        self.assertIn("x-request-id",response.headers)

    async def test_dimension_rejected_before_availability_or_embedding_io(self):
        response = await self.client.post("/v1/embeddings",json={"model":"alias","input":"x","dimensions":2})
        self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(response.json()["error"]["code"],"unsupported_value")
        self.assertEqual(self.runtime.calls,["resolve","validate_embedding"])

    async def test_input_type_stream_and_token_arrays_rejected_before_resolution(self):
        for changes,code in (({"input_type":"search_query"},"unsupported_parameter"),
                             ({"stream":False},"unsupported_parameter"),({"input":[1,2]},"unsupported_capability")):
            with self.subTest(changes=changes):
                response = await self.client.post("/v1/embeddings",json={"model":"alias","input":"x",**changes})
                self.assertEqual(response.status_code,400,response.text)
                self.assertEqual(response.json()["error"]["code"],code)
        self.assertEqual(self.runtime.calls,[])

    async def test_auth_content_type_and_method_use_shared_json_boundary(self):
        response = await self.client.post("/v1/embeddings",json={"model":"alias","input":"x"},headers={"Authorization":"Bearer wrong"})
        self.assertEqual(response.status_code,401)
        response = await self.client.get("/v1/embeddings")
        self.assertEqual(response.status_code,405)
        response = await self.client.post("/v1/embeddings",content="{}",headers={"Content-Type":"text/plain"})
        self.assertEqual(response.status_code,415)
        self.assertEqual(self.runtime.calls,[])

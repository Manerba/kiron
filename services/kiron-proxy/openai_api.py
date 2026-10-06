"""Authenticated OpenAI HTTP boundary over the shared local inference service."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import time
from uuid import uuid4

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from kiron_common.local_inference import EventKind, LocalInferenceError, RequestContext
from openai_wire import ApiError, chat_events, decode_json, parse_chat, serialize_completion
from openai_vision import VisionError, decode_chat_images
from openai_embeddings import parse_embeddings, serialize_embeddings
from openai_responses import prepare_response, parse_response, serialize_response, response_events
from openai_responses_stream import ResponseStreamState
from request_store import RequestRecord
from runtime_service import RuntimeStreamingResponse

logger = logging.getLogger(__name__)
MAX_BODY_SIZE = 4 * 1024 * 1024
MAX_RESPONSE_SIZE = 16 * 1024 * 1024
REQUEST_TIMEOUT = 600.0
LOG_TIMEOUT = 1.0
CANCEL_WAIT = 0.25
ERROR_SEND_TIMEOUT = 1.0


def _observe_late(task):
    if not task.cancelled():
        with suppress(Exception):
            task.exception()


async def _cancel_task(task):
    if task is None:
        return
    if task.done():
        _observe_late(task)
        return
    task.cancel()
    await asyncio.wait({task}, timeout=CANCEL_WAIT)
    task.add_done_callback(_observe_late)


def _runtime_error(exc):
    """Public messages deliberately exclude backend paths and raw exceptions."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, LocalInferenceError):
        code = exc.failure.code.value
        mappings = {
            "model_not_found": (404, "model_not_found", "invalid_request_error", "Model was not found"),
            "provider_unavailable": (503, "provider_unavailable", "server_error", "Model provider is unavailable"),
            "invalid_configuration": (503, "model_unavailable", "server_error", "Model configuration is unavailable"),
            "unsupported_capability": (400, "unsupported_capability", "invalid_request_error", "Model capability is not supported"),
            "unsupported_parameter": (400, "unsupported_parameter", "invalid_request_error", "Model parameter is not supported"),
            "unsupported_value": (400, "unsupported_value", "invalid_request_error", "Model parameter value is not supported"),
            "invalid_request": (400, "invalid_request", "invalid_request_error", "Request is invalid"),
            "context_length_exceeded": (400, "context_length_exceeded", "invalid_request_error", "Request exceeds the model context window"),
            "conflict": (503, "resource_busy", "server_error", "Inference resources are busy"),
            "cancelled": (503, "resource_busy", "server_error", "Inference was cancelled"),
            "overloaded": (429, "overloaded", "rate_limit_error", "Inference capacity is exhausted"),
            "provider_error": (502, "backend_protocol_error", "server_error", "Model provider returned an invalid result"),
            "timeout": (504, "timeout", "server_error", "Inference deadline exceeded"),
        }
        status, code, kind, message = mappings.get(code, (500, "internal_error", "server_error", "Internal inference error"))
        # Runtime failures may originate in an adapter; its parameter string is
        # no more trusted as a public field path than its message.
        return ApiError(message, code, None, status, kind)
    if isinstance(exc, TimeoutError):
        return ApiError("Request deadline exceeded", "timeout", status=504, error_type="server_error")
    return ApiError("Internal inference error", "internal_error", status=500, error_type="server_error")


def _error_response(error):
    headers = {}
    if error.status == 401:
        headers["WWW-Authenticate"] = 'Bearer realm="api"'
    if error.status == 429:
        headers["Retry-After"] = "1"
    return JSONResponse(error.envelope(), status_code=error.status, headers=headers)


async def _safe_log(store, method, *args, **kwargs):
    pending = None
    try:
        pending = asyncio.create_task(getattr(store, method)(*args, **kwargs))
        done, _ = await asyncio.wait({pending}, timeout=LOG_TIMEOUT)
        if not done:
            raise TimeoutError
        pending.result()
    except Exception:
        # Logging failures cannot replace the inference result or leak headers.
        logger.warning("OpenAI request logging failed operation=%s", method)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            pending.add_done_callback(_observe_late)


class _ApiBoundary:
    """Auth and lifecycle surround routing, including 404/405 and ASGI headers."""
    def __init__(self, app, *, request_store, api_key_store):
        self.app, self.store, self.keys = app, request_store, api_key_store

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started_at = time.monotonic()
        context = RequestContext(uuid4().hex[:16], started_at + REQUEST_TIMEOUT, asyncio.Event())
        record = RequestRecord(id=context.request_id, method=scope["method"], path=scope["path"],
                               client_ip=(scope.get("client") or ("", 0))[0])
        scope.setdefault("state", {}).update(openai_context=context, openai_record=record)
        body_ended, disconnected = asyncio.Event(), asyncio.Event()
        started = complete = is_sse = abandoned = False
        error_send_open = True

        async def watched_receive():
            if body_ended.is_set():
                await disconnected.wait()
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected.set()
                context.cancellation.set()
            elif not message.get("more_body", False):
                body_ended.set()
            return message

        async def wrapped_send(message, *, error_response=False):
            nonlocal started, complete, is_sse
            if abandoned and not error_response:
                raise ClientDisconnect
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", context.request_id.encode("ascii")))
                message = {**message, "headers": headers}
                is_sse = any(k.lower() == b"content-type" and v.startswith(b"text/event-stream") for k, v in headers)
                record.status_code = message["status"]
                started = True
            elif message["type"] == "http.response.body":
                record.response_size += len(message.get("body", b""))
                complete = not message.get("more_body", False)
            try:
                await send(message)
            except OSError:
                raise ClientDisconnect from None

        async def fallback_send(message):
            if not error_send_open:
                raise ClientDisconnect
            await wrapped_send(message, error_response=True)

        async def deliver_error(awaitable):
            # The main request deadline has already expired on this path.
            # A blocked client send must not make its error handling unbounded.
            nonlocal error_send_open
            pending = asyncio.create_task(awaitable)
            try:
                done, _ = await asyncio.wait({pending}, timeout=ERROR_SEND_TIMEOUT)
                if pending in done:
                    return pending.result()
            finally:
                error_send_open = False
                await _cancel_task(pending)

        async def run():
            request = Request(scope)
            authorizations = request.headers.getlist("authorization")
            if len(authorizations) != 1:
                raise ApiError("A Bearer API key is required", "invalid_api_key", status=401, error_type="authentication_error")
            parts = authorizations[0].split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise ApiError("A Bearer API key is required", "invalid_api_key", status=401, error_type="authentication_error")
            try:
                valid = await asyncio.to_thread(self.keys.validate_key, parts[1])
            except Exception:
                raise ApiError("API key verification is unavailable", "auth_backend_unavailable", status=503,
                               error_type="server_error") from None
            if not valid:
                raise ApiError("API key is invalid", "invalid_api_key", status=401, error_type="authentication_error")
            await self.app(scope, watched_receive, wrapped_send)

        task = None
        watcher = None
        disconnect_waiter = None
        try:
            await _safe_log(self.store, "add_request", record)
            task = asyncio.create_task(run())
            async def watch_disconnect():
                nonlocal abandoned
                await body_ended.wait()
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        if complete:
                            return
                        disconnected.set()
                        context.cancellation.set()
                        abandoned = True
                        task.cancel()
                        return
            watcher = asyncio.create_task(watch_disconnect())
            disconnect_waiter = asyncio.create_task(disconnected.wait())
            done, _ = await asyncio.wait({task, disconnect_waiter},
                timeout=max(0, context.deadline_monotonic - time.monotonic()),
                return_when=asyncio.FIRST_COMPLETED)
            if disconnected.is_set():
                context.cancellation.set()
                abandoned = True
                record.state, record.error_message = "error", "cancelled"
                await _cancel_task(task)
                return
            if not done:
                context.cancellation.set()
                abandoned = True
                await _cancel_task(task)
                raise TimeoutError
            await task
        except ClientDisconnect:
            context.cancellation.set()
            disconnected.set()
            abandoned = True
            record.state, record.error_message = "error", "cancelled"
        except asyncio.CancelledError:
            context.cancellation.set()
            abandoned = True
            record.state, record.error_message = "error", "cancelled"
            await _cancel_task(task)
            if not disconnected.is_set():
                raise
        except Exception as exc:
            abandoned = True
            error = _runtime_error(exc)
            record.state, record.error_message = "error", error.code
            if not started and not disconnected.is_set():
                await deliver_error(_error_response(error)(scope, watched_receive, fallback_send))
            elif is_sse and not complete and not disconnected.is_set():
                fail_stream = scope["state"].get("openai_stream_failure")
                body = (fail_stream(error) if fail_stream is not None else
                        b"data: " + json.dumps(error.envelope(), separators=(",", ":")).encode() + b"\n\ndata: [DONE]\n\n")
                await deliver_error(fallback_send({"type": "http.response.body", "body": body, "more_body": False}))
            elif not disconnected.is_set():
                raise  # A failed ASGI transport cannot be repaired by another HTTP response.
        finally:
            await _cancel_task(disconnect_waiter)
            await _cancel_task(watcher)
            if record.state != "error":
                record.state = "completed" if complete and record.status_code < 400 else "error"
            record.duration_ms = round((time.monotonic() - started_at) * 1000, 1)
            await _safe_log(self.store, "update_request", record.id, model=record.model,
                status_code=record.status_code, state=record.state, error_message=record.error_message,
                duration_ms=record.duration_ms, request_size=record.request_size,
                response_size=record.response_size, tokens_generated=record.tokens_generated,
                is_streaming=record.is_streaming)


async def _body(request, *, json_required=False):
    if json_required:
        content_types = request.headers.getlist("content-type")
        parts = content_types[0].split(";") if len(content_types) == 1 else []
        if (not parts or len(parts) > 2 or parts[0].strip().lower() != "application/json"
                or any(part.strip().lower() not in {"charset=utf-8", 'charset="utf-8"'} for part in parts[1:])):
            raise ApiError("Content-Type must be application/json with optional UTF-8 charset",
                           "unsupported_media_type", status=415)
        if request.headers.get("content-encoding") is not None:
            raise ApiError("Request compression is unsupported", "unsupported_media_type", status=415)
    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1:
        raise ApiError("Repeated Content-Length is invalid", "invalid_request")
    length = lengths[0] if lengths else None
    if length is not None:
        try:
            if not length.isascii() or not length.isdecimal():
                raise ValueError
            if int(length) > MAX_BODY_SIZE:
                raise ApiError("Request body exceeds 4 MiB", "request_too_large", status=413)
        except ValueError:
            raise ApiError("Content-Length is invalid", "invalid_request") from None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_SIZE:
            raise ApiError("Request body exceeds 4 MiB", "request_too_large", status=413)
        chunks.append(chunk)
    request.state.openai_record.request_size = size
    return b"".join(chunks)


async def _image_parts(data, context):
    try:
        return await decode_chat_images(data, context)
    except VisionError as exc:
        raise ApiError(str(exc), exc.code, exc.param, exc.status,
                       "server_error" if exc.status >= 500 else
                       "rate_limit_error" if exc.status == 429 else "invalid_request_error") from None


def _service(request):
    service = getattr(request.app.state, "local_inference", None)
    if service is None:
        raise ApiError("Local inference service is unavailable", "provider_unavailable", status=503,
                       error_type="server_error")
    return service


def create_openai_api_app(request_store, api_key_store):
    async def models(request):
        if request.method != "GET":
            raise HTTPException(405, headers={"Allow": "GET"})
        await _body(request)
        values = await _service(request).public_models(request.state.openai_context)
        return JSONResponse({"object": "list", "data": values})

    async def model(request):
        if request.method != "GET":
            raise HTTPException(405, headers={"Allow": "GET"})
        await _body(request)
        _, value = await _service(request).public_model(request.path_params["model_id"], request.state.openai_context)
        return JSONResponse(value)

    async def chat(request):
        raw = await _body(request, json_required=True)
        service, context, record = _service(request), request.state.openai_context, request.state.openai_record
        data = decode_json(raw)
        image_parts = await _image_parts(data, context)
        parsed = parse_chat(data, image_parts=image_parts)
        resolved = await service.resolve(parsed.model_id, context)
        record.model, record.is_streaming = resolved.api_model_id, parsed.stream
        inference = await service.validate_chat(parsed, resolved, context)
        public = await service.public_model_for(resolved, context)
        completion_id, created = "chatcmpl-" + uuid4().hex, int(time.time())
        if not parsed.stream:
            result = await service.chat(inference)
            payload = serialize_completion(result, public["id"], completion_id, created, request=inference)
            response = JSONResponse(payload)
            if len(response.body) > MAX_RESPONSE_SIZE:
                raise ApiError("Model output exceeds the response limit", "backend_protocol_error", status=502,
                               error_type="server_error")
            record.tokens_generated = result.usage.output_tokens
            return response
        operation = await service.prepare(inference, streaming=True)
        try:
            async def events():
                async for event in operation.events():
                    if event.kind is EventKind.USAGE:
                        record.tokens_generated = event.usage.output_tokens
                    yield event
            async def content():
                async for chunk in chat_events(events(), public["id"], completion_id, created, parsed.include_usage, request=inference):
                    # Inspect only our serializer's complete frames for logging.
                    if chunk.startswith(b"data: {"):
                        value = json.loads(chunk[6:].strip())
                        if "error" in value:
                            record.state, record.error_message = "error", value["error"]["code"]
                        if value.get("usage"):
                            record.tokens_generated = value["usage"]["completion_tokens"]
                    yield chunk
            return RuntimeStreamingResponse(content(), operation=operation, media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        except BaseException:
            await operation.close()
            raise

    async def embeddings(request):
        raw = await _body(request, json_required=True)
        service, context, record = _service(request), request.state.openai_context, request.state.openai_record
        parsed = parse_embeddings(decode_json(raw))
        resolved = await service.resolve(parsed.model_id, context)
        record.model = resolved.api_model_id
        inference = await service.validate_embedding(parsed, resolved, context)
        await service.public_model_for(resolved, context)
        result = await service.embed(inference)
        payload = serialize_embeddings(result, request=inference, encoding_format=parsed.encoding_format)
        return JSONResponse(payload)

    async def responses(request):
        raw = await _body(request, json_required=True)
        service, context, record = _service(request), request.state.openai_context, request.state.openai_record
        prepared = prepare_response(decode_json(raw))
        try:
            image_parts = await _image_parts(prepared.chat_data, context)
        except ApiError as exc:
            raise ApiError(exc.message, exc.code, prepared.parameter(exc.param), exc.status, exc.error_type) from None
        parsed = parse_response(prepared, image_parts=image_parts)
        resolved = await service.resolve(parsed.model_id, context)
        record.model, record.is_streaming = resolved.api_model_id, parsed.stream
        inference = await service.validate_response(parsed, resolved, context)
        await service.public_model_for(resolved, context)
        response_id, created = "resp_" + uuid4().hex, int(time.time())
        if not parsed.stream:
            result = await service.chat(inference)
            payload = serialize_response(result, request=inference, response_id=response_id, created=created)
            response = JSONResponse(payload)
            if len(response.body) > MAX_RESPONSE_SIZE:
                raise ApiError("Model output exceeds the response limit", "backend_protocol_error", status=502,
                               error_type="server_error")
            record.tokens_generated = result.usage.output_tokens
            return response
        operation = await service.prepare(inference, streaming=True)
        try:
            state = ResponseStreamState(inference, response_id, created)
            request.scope["state"]["openai_stream_failure"] = state.fail
            async def events():
                async for event in operation.events():
                    if event.kind is EventKind.USAGE:
                        record.tokens_generated = event.usage.output_tokens
                    yield event
            async def content():
                async for chunk in response_events(events(), request=inference,
                        response_id=response_id, created=created, state=state):
                    # Only locally serialized event metadata reaches the log.
                    for line in chunk.splitlines():
                        if line.startswith(b"data: {"):
                            value = json.loads(line[6:])
                            if value.get("type") == "response.failed":
                                record.state = "error"
                                record.error_message = value["response"]["error"]["code"]
                    yield chunk
            return RuntimeStreamingResponse(content(), operation=operation, media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        except BaseException:
            await operation.close()
            raise

    async def route_error(request, exc):
        if exc.status_code == 405:
            error = ApiError("Method is not supported", "method_not_allowed", status=405)
        else:
            error = ApiError("Endpoint was not found", "endpoint_not_found", status=404)
        request.state.openai_record.error_message = error.code
        response = _error_response(error)
        if exc.status_code == 405 and exc.headers and "Allow" in exc.headers:
            response.headers["Allow"] = exc.headers["Allow"]
        return response

    app = Starlette(routes=[Route("/v1/models", models, methods=["GET"]),
        Route("/v1/models/{model_id:path}", model, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/v1/responses", responses, methods=["POST"])],
        exception_handlers={HTTPException: route_error},
        middleware=[Middleware(_ApiBoundary, request_store=request_store, api_key_store=api_key_store)])
    app.router.redirect_slashes = False
    return app

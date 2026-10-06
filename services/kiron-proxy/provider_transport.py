"""Bounded async transport helpers shared by local runtime adapters."""
from __future__ import annotations

import asyncio
import json
import time
import httpx

from kiron_common.local_inference import ErrorCode, LocalInferenceError, RequestContext, RuntimeFailure


def decode_provider_json(raw):
    """Decode bounded native object JSON strictly; transport owns the byte cap."""
    try:
        if type(raw) in (bytes, bytearray):
            raw = raw.decode("utf-8")
        if type(raw) is not str:
            raise ValueError()
        depth, quoted, escaped = 0, False, False
        for char in raw:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 32:
                    raise ValueError()
            elif char in "]}":
                depth -= 1

        def pairs(items):
            value = {}
            for key, child in items:
                if key in value:
                    raise ValueError()
                value[key] = child
            return value

        def invalid(_):
            raise ValueError()

        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
        if type(value) is not dict:
            raise ValueError()
        # Catch overflowing JSON floats and lone Unicode surrogates, including
        # nested values which a provider-specific projection would not inspect.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Invalid provider JSON") from None


def failure(code, message, parameter=None):
    return LocalInferenceError(RuntimeFailure(code, message, parameter))


class RequestRejected(LocalInferenceError):
    """A fully consumed, request-bound native rejection before execution.

    Adapters may construct this only from their explicit rejection protocol.
    An HTTP status, idle health, transport error or arbitrary failure code is
    insufficient evidence that backend work never started.
    """


def _discard_late_result(task):
    if task.cancelled():
        return
    try:
        result = task.result()
    except BaseException:
        return
    if isinstance(result, httpx.Response):
        cleanup = asyncio.create_task(result.aclose())
        cleanup.add_done_callback(lambda done: None if done.cancelled() else done.exception())


async def with_context(awaitable, context):
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            if context.cancellation.is_set():
                raise failure(ErrorCode.CANCELLED, "Request cancelled")
            remaining = context.deadline_monotonic - time.monotonic()
            if remaining <= 0:
                raise failure(ErrorCode.TIMEOUT, "Request deadline exceeded")
            done, _ = await asyncio.wait({task}, timeout=min(remaining, 0.05))
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait({task}, timeout=0.25)
            finally:
                # A cancellation-resistant transport must not extend the public
                # deadline indefinitely. This is not evidence of backend end.
                task.add_done_callback(_discard_late_result)


async def bounded_lines(response, context, *, limit=1024 * 1024):
    """Bound buffering before a line delimiter arrives, including split UTF-8."""
    buffer = bytearray()
    chunks = response.aiter_bytes(65536).__aiter__()
    while True:
        try:
            chunk = await with_context(chunks.__anext__(), context)
        except StopAsyncIteration:
            if buffer:
                raise failure(ErrorCode.PROVIDER_ERROR, "Unterminated provider stream frame")
            return
        buffer.extend(chunk)
        while True:
            end = buffer.find(b"\n")
            if end < 0:
                break
            if end > limit:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider stream frame limit exceeded")
            line = bytes(buffer[:end]).removesuffix(b"\r")
            del buffer[:end + 1]
            try:
                yield line.decode("utf-8")
            except UnicodeError as exc:
                raise failure(ErrorCode.PROVIDER_ERROR, "Invalid provider stream encoding") from exc
        if len(buffer) > limit:
            raise failure(ErrorCode.PROVIDER_ERROR, "Provider stream frame limit exceeded")


async def bounded_request(client, method, path, context, *, limit, **kwargs):
    """Bound the bytes while receiving, before allocating or parsing the body."""
    request = client.build_request(method, path, **kwargs)
    response = await with_context(client.send(request, stream=True), context)
    try:
        content = bytearray()
        iterator = response.aiter_bytes(65536).__aiter__()
        while True:
            try:
                chunk = await with_context(iterator.__anext__(), context)
            except StopAsyncIteration:
                break
            if len(content) + len(chunk) > limit:
                raise failure(ErrorCode.PROVIDER_ERROR, "Provider response limit exceeded")
            content.extend(chunk)
        return response.status_code, bytes(content)
    finally:
        cleanup = RequestContext(context.request_id, time.monotonic() + 1, asyncio.Event())
        try:
            await with_context(response.aclose(), cleanup)
        except LocalInferenceError:
            # Admission is decided from terminal backend evidence separately.
            pass

"""Pinned SDK probe against the real API and isolated native RuntimeService.

ASGI replaces only the public TCP transport. Authentication, validation,
serialization, lifecycle, admission, controller UDS and native inference run.
No production API key, registry or service is used.
"""
import hashlib
import json
from pathlib import Path
import secrets


CONTEXT_PROMPT = "boundary " * 4096


def context_error(raw, *, streaming):
    """Require an exact error terminal; no generated text, success or usage."""
    if streaming:
        blocks = [block for block in raw.split(b'\n\n') if block]
        if not blocks or blocks[-1] != b'data: [DONE]':
            raise ValueError('context SSE lacks final DONE')
        values = []
        for block in blocks[:-1]:
            if not block.startswith(b'data: ') or b'\n' in block:
                raise ValueError('unexpected context SSE framing')
            value = json.loads(block[6:])
            if values and 'error' in values[-1]:
                raise ValueError('data after context error')
            if 'error' not in value:
                if (value.get('object') != 'chat.completion.chunk' or value.get('usage') is not None
                        or value.get('choices') != [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]):
                    raise ValueError('context rejection emitted content, usage or success')
            values.append(value)
        if not values or 'error' not in values[-1] or sum('error' in v for v in values) != 1:
            raise ValueError('context SSE requires exactly one final error')
        value = values[-1]
    else:
        value = json.loads(raw)
    error = value.get('error') if type(value) is dict else None
    if (set(value) != {'error'} or type(error) is not dict
            or set(error) != {'message', 'type', 'param', 'code'}
            or error['code'] != 'context_length_exceeded' or error['type'] != 'invalid_request_error'
            or error['param'] is not None or type(error['message']) is not str):
        raise ValueError('unexpected public context error')
    return value


async def probe(service, model, store, *, report=None):
    import httpx
    import openai
    from openai_api import create_openai_api_app

    if openai.__version__ != "2.29.0":
        raise ValueError("public API smoke requires the pinned SDK 2.29.0")
    if model.deployment.resource_profile.context_tokens != 1024 or len(CONTEXT_PROMPT.encode()) > 65536:
        raise ValueError('context probe requires the fixed bounded 1024-token profile')
    key = secrets.token_hex(32)

    class Keys:
        def validate_key(self, value):
            return secrets.compare_digest(value, key)

    class Records:
        def __init__(self):
            self.values = {}

        async def add_request(self, record):
            self.values[record.id] = record

        async def update_request(self, request_id, **changes):
            record = self.values[request_id]
            for name, value in changes.items():
                setattr(record, name, value)

    records = Records()
    app = create_openai_api_app(records, Keys())
    app.state.local_inference = service
    transport = httpx.ASGITransport(app=app)
    async with openai.AsyncOpenAI(api_key=key, base_url="http://isolated-kiron/v1", max_retries=0,
            _strict_response_validation=True, http_client=httpx.AsyncClient(transport=transport)) as client:
        models = await client.models.list()
        detail = await client.models.retrieve(model.api_model_id)
        if [item.model_dump() for item in models.data] != [detail.model_dump()]:
            raise ValueError("public models/detail do not agree")
        if (detail.id != model.api_model_id or detail.created != model.created
                or detail.owned_by != "prism"):
            raise ValueError("public model identity changed")
        if store.snapshot():
            raise ValueError("discovery mutated isolated admission")
        try:
            await client.chat.completions.create(model=model.api_model_id,
                messages=[{"role": "user", "content": "Reply with exactly OK."}],
                tools=[{"type": "function", "function": {"name": "not_executed", "parameters": {"type": "object"}}}])
        except openai.BadRequestError as error:
            if error.code != "unsupported_capability":
                raise
            rejected = error.code
        else:
            raise ValueError("unimplemented tools were accepted")
        if store.snapshot():
            raise ValueError("capability rejection mutated isolated admission")
        context_rejections = []
        for streaming in (False, True):
            messages = [{"role": "user", "content": CONTEXT_PROMPT}]
            if not streaming:
                try:
                    await client.chat.completions.create(model=model.api_model_id, messages=messages,
                                                        max_tokens=1, stream=False)
                except openai.BadRequestError as error:
                    if error.code != 'context_length_exceeded' or error.status_code != 400:
                        raise
                    response = error.response
                    body = await response.aread()
                else:
                    raise ValueError('overlong JSON prompt was accepted')
            else:
                raw = await client.chat.completions.with_raw_response.create(model=model.api_model_id,
                    messages=messages, max_tokens=1, stream=True)
                response = raw.http_response
                body = await response.aread()
                if response.status_code != 200 or not response.headers.get('content-type', '').startswith('text/event-stream'):
                    raise ValueError('context stream has no public SSE response')
                try:
                    async for _ in raw.parse():
                        pass
                except openai.APIError as error:
                    if error.code != 'context_length_exceeded':
                        raise
                else:
                    raise ValueError('SDK did not expose the context stream error')
            envelope = context_error(body, streaming=streaming)
            request_id = response.headers.get('x-request-id', '')
            if len(request_id) != 16 or request_id not in records.values:
                raise ValueError('context rejection request ID missing')
            tickets = [ticket.operation_id for ticket in store.snapshot() if ticket.kind == 'request']
            if tickets:
                raise ValueError('complete context rejection retained request tickets')
            case = {'streaming': streaming, 'http_status': response.status_code, 'error': envelope['error'],
                    'request_id': request_id, 'request_tickets_after_eof': tickets,
                    'public_body_sha256': hashlib.sha256(body).hexdigest()}
            context_rejections.append(case)
            if report is not None:
                stem = Path(report) / ('context-stream' if streaming else 'context-json')
                with stem.with_suffix('.public.raw').open('xb') as output:
                    output.write(body)
                with stem.with_suffix('.json').open('x') as output:
                    json.dump(case, output, indent=2)
        raw = await client.chat.completions.with_raw_response.create(model=model.api_model_id,
            messages=[{"role": "user", "content": "Reply with exactly OK."}], max_tokens=16)
        text = raw.parse()
        request_id = raw.headers.get("x-request-id", "")
        if len(request_id) != 16 or request_id not in records.values:
            raise ValueError("API request identity missing from SDK response")
        if not text.choices[0].message.content or text.choices[0].finish_reason != "stop":
            raise ValueError("text completion failed")
        events = []
        async with client.chat.completions.stream(model=model.api_model_id,
                messages=[{"role": "user", "content": "Reply with exactly OK."}],
                max_completion_tokens=16, stream_options={"include_usage": True}) as stream:
            async for event in stream:
                events.append(event.type)
            final = await stream.get_final_completion()
        if (not final.choices[0].message.content or final.choices[0].finish_reason != "stop"
                or final.usage is None or final.usage.completion_tokens < 1):
            raise ValueError("SDK streaming helper did not assemble exact text/usage")
        if any(ticket.kind == "request" for ticket in store.snapshot()):
            raise ValueError("finished public API requests retained admission tickets")
    return {"sdk_version": openai.__version__, "public_transport": "httpx.ASGITransport",
        "models": [item.model_dump() for item in models.data], "model": detail.model_dump(),
        "context_rejections": context_rejections,
        "context_prompt_sha256": hashlib.sha256(CONTEXT_PROMPT.encode()).hexdigest(),
        "context_prompt_bytes": len(CONTEXT_PROMPT.encode()),
        "unsupported_tools": rejected, "text": text.model_dump(), "stream": final.model_dump(),
        "stream_event_types": events,
        "request_records": [{"id": value.id, "path": value.path, "status": value.status_code,
            "state": value.state, "error_code": value.error_message, "tokens": value.tokens_generated}
            for value in records.values.values()]}

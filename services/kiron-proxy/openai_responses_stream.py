"""Bounded Responses SSE aggregation; caller owns source cancellation/cleanup."""
import io

from kiron_common.local_inference import (
    EventIndexState, EventKind, FinishReason, InferenceEvent, LocalInferenceError, ReasoningKind, ToolCall,
)
from kiron_common.local_inference.json_schema import InstanceValidationError
from openai_generation import validate_output
from openai_responses import protocol, response_base, response_usage
from openai_tools import MAX_ARGUMENT_BYTES, MAX_CALLS, NAME
from openai_wire import (ApiError, MAX_OUTPUT_BYTES, _checked_calls, _encoded,
                         _finish, _runtime_error, _source_events)


class ResponseStreamState:
    """One response, including a synchronous timeout terminal for the HTTP owner.

    Text buffers retain exact deltas. A running encoded-size reservation leaves
    room for a full failed response without repeatedly serializing all buffers.
    No source await or transport action is performed by fail().
    """
    def __init__(self, request, response_id, created):
        self.request, self.response_id, self.created = request, response_id, created
        self.base = response_base(request, response_id, created)
        self.sequence = self.emitted = 0
        self.pending = bytearray()
        self.started = self.canonical_started = self.terminal = False
        self.finish = self.usage = None
        self.items, self.calls, self.ids = [], {}, set()
        self.indices = EventIndexState()
        self.reserved = len(_encoded(self.base)) + 4096

    def _frame(self, kind, fields, sequence):
        return b'event: ' + kind.encode('ascii') + b'\ndata: ' + _encoded(
            {'type': kind, 'sequence_number': sequence, **fields}) + b'\n\n'

    def _publish(self, events, reserve=0, *, terminal=False):
        frames = b''.join(self._frame(kind, fields, self.sequence + index)
                          for index, (kind, fields) in enumerate(events))
        if self.emitted + len(frames) + (0 if terminal else self.reserved + reserve) > MAX_OUTPUT_BYTES:
            raise protocol()
        self.sequence += len(events)
        self.emitted += len(frames)
        self.reserved += reserve
        self.pending.extend(frames)
        return frames

    def _take(self):
        frames = bytes(self.pending)
        self.pending.clear()
        return frames

    def start(self):
        if self.started or self.terminal:
            return b''
        frames = self._publish([('response.created', {'response': self.base}),
                                ('response.in_progress', {'response': self.base})])
        self.started = True
        return self._take()

    def _add(self, item, index):
        if index != len(self.items):
            raise protocol()
        frame = self._publish([('response.output_item.added', {'output_index': index, 'item': item})],
                              len(_encoded(item)) + 256)
        entry = {'item': item, 'index': index, 'buffers': {}, 'complete': False, 'bytes': 0}
        self.items.append(entry)
        return entry, frame

    def _delta(self, entry, key, text, kind, indices):
        if type(text) is not str:
            raise protocol()
        # Validate/measure before touching state; invalid Unicode cannot poison
        # the failed response that includes all previously accepted fragments.
        encoded = _encoded(text)
        frame = self._publish([(kind, {'item_id': entry['item']['id'],
            'output_index': entry['index'], **indices, 'delta': text})], len(encoded) - 2)
        entry['buffers'][key].write(text)
        return frame

    def _text(self, event):
        frames = b''
        index, part_index = event.output_item_index, event.part_index
        if index == len(self.items):
            item = {'id': self.response_id + '_message_' + str(index), 'type': 'message', 'role': 'assistant',
                    'status': 'in_progress', 'content': []}
            _, frames = self._add(item, index)
        entry = self.items[index]
        item = entry['item']
        if part_index == len(item['content']):
            part = {'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []}
            frames += self._publish([('response.content_part.added', {'item_id': item['id'],
                'output_index': index, 'content_index': part_index, 'part': part})], len(_encoded(part)) + 64)
            item['content'].append(part)
            entry['buffers'][('content', part_index)] = io.StringIO()
        return frames + self._delta(entry, ('content', part_index), event.text, 'response.output_text.delta',
                                   {'content_index': part_index, 'logprobs': []})

    def _thought(self, event):
        options = self.request.options.reasoning
        if (options.effort in (None, 'none') or event.reasoning_kind not in (ReasoningKind.TEXT, ReasoningKind.SUMMARY)
                or event.reasoning_kind is ReasoningKind.SUMMARY and options.summary is None):
            raise protocol()
        frames = b''
        index = event.output_item_index
        if index == len(self.items):
            _, frames = self._add({'id': self.response_id + '_reasoning_' + str(index), 'type': 'reasoning',
                'status': 'in_progress', 'summary': [], 'content': []}, index)
        entry = self.items[index]
        summary = event.reasoning_kind is ReasoningKind.SUMMARY
        key, part_type = ('summary', 'summary_text') if summary else ('content', 'reasoning_text')
        part_index = event.summary_index if summary else event.part_index
        indices = {'summary_index' if summary else 'content_index': part_index}
        buffer_key = (key, part_index)
        if part_index == len(entry['item'][key]):
            part = {'type': part_type, 'text': ''}
            frames += self._publish([('response.reasoning_summary_part.added' if summary else 'response.content_part.added',
                {'item_id': entry['item']['id'], 'output_index': entry['index'], **indices, 'part': part})],
                len(_encoded(part)) + 64)
            entry['item'][key].append(part)
            entry['buffers'][buffer_key] = io.StringIO()
        kind = 'response.reasoning_summary_text.delta' if summary else 'response.reasoning_text.delta'
        return frames + self._delta(entry, buffer_key, event.text, kind, indices)

    def accept(self, event):
        self._accept(event)
        return self._take()

    def _accept(self, event):
        if self.terminal:
            return b''
        if (not isinstance(event, InferenceEvent) or event.request_id != self.request.context.request_id
                or self.finish is not None):
            raise protocol()
        try:
            self.indices.accept(event)
        except (TypeError, ValueError):
            raise protocol() from None
        if not self.canonical_started:
            if event.kind is not EventKind.STARTED:
                raise protocol()
            self.canonical_started = True
            return b''
        if event.kind is EventKind.FAILED:
            raise _runtime_error(event.error)
        if event.kind is EventKind.CANCELLED:
            raise ApiError('The inference request was cancelled.', 'resource_busy', status=503, error_type='server_error')
        if event.kind is EventKind.COMPLETED and self.usage is not None:
            _finish(event.finish_reason)
            self.finish = event.finish_reason
            return b''
        if self.usage is not None:
            raise protocol()
        if event.kind is EventKind.USAGE:
            usage = response_usage(event.usage)
            # TokenUsage accepts arbitrarily large exact Python integers. Reserve
            # their actual JSON length before failed-terminal state adopts them.
            self._publish([], len(_encoded(usage)))
            self.usage = usage
            return b''
        if event.kind is EventKind.TEXT_DELTA:
            return self._text(event)
        if event.kind is EventKind.REASONING_DELTA:
            return self._thought(event)
        if event.kind is EventKind.TOOL_CALL_STARTED:
            index = event.call_index
            if (type(index) is not int or not 0 <= index < MAX_CALLS or index in self.calls
                    or type(event.call_id) is not str or not event.call_id or event.call_id in self.ids
                    or type(event.name) is not str or not NAME.fullmatch(event.name)):
                raise protocol()
            entry, frame = self._add({'id': self.response_id + '_call_' + str(index),
                'type': 'function_call', 'call_id': event.call_id, 'name': event.name,
                'arguments': '', 'status': 'in_progress'}, event.output_item_index)
            entry['buffers']['arguments'] = io.StringIO()
            self.calls[index] = entry
            self.ids.add(event.call_id)
            return frame
        if event.kind is EventKind.TOOL_ARGUMENTS_DELTA:
            entry = self.calls.get(event.call_index)
            if entry is None or entry['complete'] or type(event.text) is not str:
                raise protocol()
            size = len(event.text.encode('utf-8'))
            if entry['bytes'] + size > MAX_ARGUMENT_BYTES:
                raise protocol()
            frame = self._delta(entry, 'arguments', event.text, 'response.function_call_arguments.delta', {})
            entry['bytes'] += size
            return frame
        if event.kind is EventKind.TOOL_CALL_COMPLETED:
            call = event.tool_call
            entry = self.calls.get(call.index) if isinstance(call, ToolCall) else None
            if (entry is None or entry['complete'] or not call.complete
                    or (call.id, call.name, call.arguments) != (entry['item']['call_id'],
                        entry['item']['name'], entry['buffers']['arguments'].getvalue())):
                raise protocol()
            _checked_calls((ToolCall(0, call.id, call.name, call.arguments),), FinishReason.TOOL_CALLS)
            entry['complete'] = True
            return b''
        raise protocol()

    def _output(self, status):
        output = []
        for entry in self.items:
            item = dict(entry['item'], status=status)
            if item['type'] == 'function_call':
                item['arguments'] = entry['buffers']['arguments'].getvalue()
                if status == 'completed' and not entry['complete']:
                    raise protocol()
            elif item['type'] == 'message':
                item['content'] = [{**part, 'text': entry['buffers'][('content', index)].getvalue()}
                                   for index, part in enumerate(item['content'])]
            else:
                for key in ('summary', 'content'):
                    item[key] = [{**part, 'text': entry['buffers'][(key, index)].getvalue()}
                                 for index, part in enumerate(item[key])]
            output.append(item)
        return output

    def complete(self):
        if self.terminal:
            return b''
        if not self.canonical_started or self.finish is None or self.usage is None:
            raise protocol()
        calls = tuple(ToolCall(index, entry['item']['call_id'], entry['item']['name'],
            entry['buffers']['arguments'].getvalue(), complete=entry['complete'])
            for index, entry in sorted(self.calls.items()))
        _checked_calls(calls, self.finish, self.request)
        try:
            validate_output(''.join(entry['buffers'][('content', index)].getvalue()
                            for entry in self.items if entry['item']['type'] == 'message'
                            for index in range(len(entry['item']['content']))),
                            self.request.options.output_format, self.finish)
        except InstanceValidationError:
            raise protocol() from None
        status = 'incomplete' if self.finish is FinishReason.LENGTH else 'completed'
        output = self._output(status)
        events = []
        for index, item in enumerate(output):
            common = {'item_id': item['id'], 'output_index': index}
            if item['type'] == 'function_call':
                events.append(('response.function_call_arguments.done', {**common,
                    'name': item['name'], 'arguments': item['arguments']}))
            else:
                for key in (('content',) if item['type'] == 'message' else ('content', 'summary')):
                    for part_index, part in enumerate(item[key]):
                        summary = key == 'summary'
                        fields = {**common, 'summary_index' if summary else 'content_index': part_index}
                        name = ('reasoning_summary_text' if summary else
                                'output_text' if item['type'] == 'message' else 'reasoning_text')
                        events.append(('response.' + name + '.done', {**fields, 'text': part['text'],
                            **({'logprobs': []} if name == 'output_text' else {})}))
                        events.append(('response.reasoning_summary_part.done' if summary else 'response.content_part.done',
                                       {**fields, 'part': part}))
            events.append(('response.output_item.done', {'output_index': index, 'item': item}))
        value = response_base(self.request, self.response_id, self.created,
                              status=status, output=output, usage=self.usage)
        events.append(('response.' + status, {'response': value}))
        frames = self._publish(events, terminal=True)
        self.terminal = True
        return self._take()

    def fail(self, error):
        if self.terminal:
            return b''
        frames = self.start()
        # ResponseError has an SDK-defined enum, distinct from HTTP error.code.
        code = 'rate_limit_exceeded' if error.status == 429 else 'invalid_prompt' if error.status < 500 else 'server_error'
        value = response_base(self.request, self.response_id, self.created, status='failed',
            output=self._output('incomplete'), usage=self.usage,
            error={'code': code, 'message': 'The inference request could not be completed.'})
        self._publish([('response.failed', {'response': value})], terminal=True)
        self.terminal = True
        return frames + self._take()


async def response_events(events, *, request, response_id, created, state=None):
    state = state if state is not None else ResponseStreamState(request, response_id, created)
    if (state.request is not request or state.response_id != response_id or state.created != created):
        raise protocol()
    try:
        frame = state.start()
        if frame:
            yield frame
        async for event in _source_events(events):
            if state.terminal:
                return
            frame = state.accept(event)
            if frame:
                yield frame
        frame = state.complete()
        if frame:
            yield frame
    except Exception as error:
        safe = _runtime_error(error.failure) if isinstance(error, LocalInferenceError) else error if isinstance(error, ApiError) else protocol()
        frame = state.fail(safe)
        if frame:
            yield frame

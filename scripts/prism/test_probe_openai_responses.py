"""Offline safety/semantic boundaries for the isolated Responses live probe."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

spec = importlib.util.spec_from_file_location('responses_probe', Path(__file__).with_name('probe-openai-responses.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Call(NS):
    def model_dump(self, **kwargs):
        return dict(vars(self))


def result(name='text', **usage):
    values = dict(input_tokens=9,output_tokens=2,total_tokens=11,
                  input_tokens_details=NS(cached_tokens=0),output_tokens_details=NS(reasoning_tokens=0))
    values.update(usage)
    return NS(status='completed',usage=NS(**values),output=[],output_text='OK')


class ProbeTests(unittest.TestCase):
    def test_finite_output_budget_and_unknown_or_missing_native_usage_fail(self):
        self.assertEqual(sum(probe.BUDGETS.values()),160)
        probe.validate_result(result(),'text')
        for changes in ({'output_tokens_details':None}, {'input_tokens_details':NS(cached_tokens=None)},
                        {'output_tokens':True}, {'total_tokens':12}, {'output_tokens':9,'total_tokens':18},
                        {'output_tokens_details':NS(reasoning_tokens=1)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe.validate_result(result(**changes),'text')

    def test_reasoning_requires_actual_native_text_count_and_answer(self):
        value = result(output_tokens=38,total_tokens=47,output_tokens_details=NS(reasoning_tokens=33))
        value.output_text = '323'
        value.output = [NS(type='reasoning',content=[NS(text='calculation')],summary=[])]
        probe.validate_result(value,'reasoning-stream')
        for bad in ('summary','content'):
            prior = getattr(value.output[0],bad)
            setattr(value.output[0],bad,[NS(text='invented')] if bad == 'summary' else [])
            with self.assertRaises(ValueError):
                probe.validate_result(value,'reasoning-stream')
            setattr(value.output[0],bad,prior)

    def test_budget_requires_exact_usage_terminal_last_and_no_summary(self):
        events = [{'type': 'response.created', 'sequence_number': 0},
                  {'type': 'response.incomplete', 'sequence_number': 1, 'response': {
                    'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
                    'usage': {'input_tokens': 93, 'input_tokens_details': {'cached_tokens': 0},
                              'output_tokens': 48, 'output_tokens_details': {'reasoning_tokens': 32},
                              'total_tokens': 141},
                    'output': [{'type': 'reasoning', 'summary': [],
                                'content': [{'type': 'reasoning_text', 'text': 'actual native trace'}]}]}}]
        self.assertEqual(probe.validate_budget_events(events)['usage']['total_tokens'], 141)
        for field, value in (('total_tokens', 0), ('input_tokens', -1), ('input_tokens', True), ('output_tokens', 47)):
            broken = deepcopy(events)
            broken[-1]['response']['usage'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.validate_budget_events(broken)
        for field, value in (('cached_tokens', 777), ('cached_tokens', -1), ('cached_tokens', None),
                             ('reasoning_tokens', 33), ('reasoning_tokens', True)):
            broken = deepcopy(events)
            key = 'input_tokens_details' if field == 'cached_tokens' else 'output_tokens_details'
            broken[-1]['response']['usage'][key][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.validate_budget_events(broken)
        for alteration in ('late_event', 'summary', 'empty_reasoning', 'duplicate_terminal', 'boolean_sequence'):
            broken = deepcopy(events)
            if alteration == 'late_event':
                broken.append({'type': 'response.output_text.delta', 'sequence_number': 2})
            elif alteration == 'summary':
                broken[-1]['response']['output'][0]['summary'] = [{'type': 'summary_text', 'text': 'invented'}]
            elif alteration == 'empty_reasoning':
                broken[-1]['response']['output'][0]['content'][0]['text'] = ''
            elif alteration == 'duplicate_terminal':
                broken.append(deepcopy(broken[-1]))
                broken[-1]['sequence_number'] = 2
            else:
                broken[0]['sequence_number'] = False
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                probe.validate_budget_events(broken)

    def test_replay_preserves_full_sdk_annotations_and_reverses_distinct_results(self):
        calls = [Call(type='function_call',id='item-'+city,call_id='call-'+city,name='weather',
                      arguments=json.dumps({'city':city}),status='completed',parsed_arguments={'city':city})
                 for city in ('Berlin','Paris')]
        value = NS(output=calls)
        history = [{'role':'user','content':'both'}]
        replay = probe.replay_input(history,value)
        self.assertEqual(replay[1:3],[call.model_dump() for call in calls])
        self.assertIn('parsed_arguments',replay[1])
        self.assertEqual([item['call_id'] for item in replay[3:5]],['call-Paris','call-Berlin'])
        self.assertEqual([json.loads(item['output']) for item in replay[3:5]],[{'value':'BETA'},{'value':'ALPHA'}])
        ordered = probe.ordered_replay_input(history, value)
        self.assertEqual([item['type'] for item in ordered[1:4]], ['function_call', 'message', 'function_call'])
        self.assertEqual([ordered[1], ordered[3]], [call.model_dump() for call in calls])
        self.assertEqual(ordered[2]['content'][0]['text'], probe.ORDERED_HISTORY_TEXT)
        self.assertEqual(ordered[4:], replay[3:])
        calls[1].call_id = calls[0].call_id
        with self.assertRaises(ValueError):
            probe.replay_input(history,value)


if __name__ == '__main__':
    unittest.main()

"""No native model/process starts: exercise exact pidfd and ownership guards."""
import asyncio
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('fault_probe',Path(__file__).with_name('probe-controller-faults.py'))
fault = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fault)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.binary = Path(self.directory.name)/'binary'
        self.binary.write_bytes(b'immutable test fixture, not executed')
        info = self.binary.stat()
        self.identity = {'pid':777,'state':'S','ppid':410,'pgid':777,'session':777,'start_ticks':12345,
            'uids':[65534]*4,'gids':[982]*4,'groups':[],'no_new_privs':1,'exe_device':info.st_dev,'exe_inode':info.st_ino}
        self.child = NS(process=NS(pid=777,returncode=None))
        self.generation = NS(boot_id='boot',process_id='spawn')
        self.controller = NS(child=self.child,generation=self.generation,policy=NS(binary=self.binary))
        for name,result in (('geteuid',65534),('getegid',982),('getpid',410),('getgroups',[])):
            p = patch.object(fault.os,name,return_value=result)
            p.start();self.addCleanup(p.stop)

    def test_only_verified_pidfd_is_signalled_and_closed_once(self):
        with patch.object(fault,'process_identity',side_effect=[dict(self.identity,state='R'),self.identity]), \
                patch.object(fault.os,'pidfd_open',return_value=88) as opened, \
                patch.object(fault.signal,'pidfd_send_signal') as sent, patch.object(fault.os,'close') as closed:
            fault.signal_owned_child(self.controller,self.generation,self.child,self.identity)
        opened.assert_called_once_with(777,0)
        sent.assert_called_once_with(88,signal.SIGKILL,None,0)
        closed.assert_called_once_with(88)

    def test_foreign_uid_gid_parent_group_session_exe_dead_and_root_never_signal(self):
        for field,value in (('uids',[0]*4),('gids',[0]*4),('ppid',411),('pgid',778),('session',778),
                            ('exe_inode',0),('exe_device',0),('state','Z'),('state','X'),('groups',[982]),('no_new_privs',0)):
            bad = {**self.identity,field:value}
            with self.subTest(field=field,value=value), patch.object(fault,'process_identity',return_value=bad), \
                    patch.object(fault.os,'pidfd_open') as opened, patch.object(fault.signal,'pidfd_send_signal') as sent:
                with self.assertRaises(ValueError):
                    fault.signal_owned_child(self.controller,self.generation,self.child,self.identity)
                opened.assert_not_called();sent.assert_not_called()
        for name,result in (('geteuid',0),('getgroups',[982])):
            with self.subTest(name=name), patch.object(fault.os,name,return_value=result), self.assertRaises(ValueError):
                fault.guard_identity(self.identity,self.binary)

    def test_identity_swap_after_pidfd_open_closes_without_signal(self):
        for field,value in (('start_ticks',12346),('exe_inode',0),('uids',[0]*4),('state','Z'),('groups',[982]),('no_new_privs',0)):
            with self.subTest(field=field), patch.object(fault,'process_identity',side_effect=[self.identity,{**self.identity,field:value}]), \
                    patch.object(fault.os,'pidfd_open',return_value=88), patch.object(fault.signal,'pidfd_send_signal') as sent, \
                    patch.object(fault.os,'close') as closed:
                with self.assertRaises(ValueError):
                    fault.signal_owned_child(self.controller,self.generation,self.child,self.identity)
                sent.assert_not_called();closed.assert_called_once_with(88)

    def test_controller_generation_and_child_identity_are_rechecked_after_fd_acquisition(self):
        for mutation in ('generation','child'):
            controller = NS(**vars(self.controller))
            def opened(pid,flags):
                setattr(controller,mutation,object())
                return 88
            with self.subTest(mutation=mutation), patch.object(fault,'process_identity',return_value=self.identity), \
                    patch.object(fault.os,'pidfd_open',side_effect=opened), patch.object(fault.signal,'pidfd_send_signal') as sent, \
                    patch.object(fault.os,'close') as closed:
                with self.assertRaises(ValueError):
                    fault.signal_owned_child(controller,self.generation,self.child,self.identity)
                sent.assert_not_called();closed.assert_called_once_with(88)
        self.child.process.returncode=0
        with self.assertRaises(ValueError):
            fault.signal_owned_child(self.controller,self.generation,self.child,self.identity)

    def test_signal_failure_still_closes_descriptor(self):
        with patch.object(fault,'process_identity',return_value=self.identity), \
                patch.object(fault.os,'pidfd_open',return_value=88), \
                patch.object(fault.signal,'pidfd_send_signal',side_effect=ProcessLookupError), patch.object(fault.os,'close') as closed:
            with self.assertRaises(ProcessLookupError):
                fault.signal_owned_child(self.controller,self.generation,self.child,self.identity)
        closed.assert_called_once_with(88)

    def test_only_confirmed_reap_allows_monitor_to_have_already_released_ticket(self):
        generation=self.generation
        unknown=NS(operation_id='req',owner='kiron-proxy',kind='request',phase='unknown',
            deployment_id='model',generation='["boot","spawn"]')
        self.assertEqual(fault.post_failure_state([unknown],'req','model',generation,cleanup_confirmed=False),'unknown')
        self.assertEqual(fault.post_failure_state([],'req','model',generation,cleanup_confirmed=True),'terminated')
        for tickets,confirmed in (([],False),([NS(**{**vars(unknown),'phase':'active'})],False),
                ([NS(**{**vars(unknown),'generation':'old'})],True)):
            with self.subTest(tickets=tickets),self.assertRaises(ValueError):
                fault.post_failure_state(tickets,'req','model',generation,cleanup_confirmed=confirmed)

    def test_active_ticket_requires_exact_operation_deployment_and_generation(self):
        ticket=NS(operation_id='request',owner='kiron-proxy',kind='request',phase='active',
                  deployment_id='model',generation='["boot","spawn"]')
        self.assertTrue(fault.active_ticket([ticket],'request','model',self.generation))
        for field,value in (('operation_id','other'),('owner','other'),('kind','load'),('phase','unknown'),
                            ('deployment_id','other'),('generation','["boot","old"]')):
            with self.subTest(field=field):
                changed=NS(**{**vars(ticket),field:value})
                self.assertFalse(fault.active_ticket([changed],'request','model',self.generation))
        self.assertEqual(fault.RECOVERY_TOKENS,8)
        self.assertLessEqual(fault.MAX_SECONDS,600)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_fake_crash_reconcile_new_generation_recovery_and_unload(self):
        import httpx
        from kiron_common.local_inference import EventKind, InferenceEvent, RuntimeGeneration, RuntimeFailure, ErrorCode, InferenceResult, TextPart, TokenUsage, FinishReason
        from test_openai_tools import request_for
        request = request_for()
        model=request.model
        old=RuntimeGeneration('boot','old')
        new=RuntimeGeneration('boot','new')
        dead=asyncio.Event()
        tickets=[]
        child=NS(process=NS(pid=777,returncode=None))
        controller=NS(child=child,generation=old,policy=NS(binary=Path('/unused')))
        store=NS(snapshot=lambda:tuple(tickets))
        operations=[]
        loads=[]
        class Service:
            async def load(self,model,context):
                loads.append(context.request_id)
                controller.generation=old if len(loads)==1 else new
                controller.child=child if len(loads)==1 else NS(process=NS(pid=888,returncode=None))
                return NS(observation=NS(generation=controller.generation))
            async def prepare(self,req,streaming):
                ticket=NS(operation_id=req.context.request_id,owner='kiron-proxy',kind='request',phase='active',
                          deployment_id=model.deployment.id,generation='["boot","old"]')
                # Actual dataclass is required for persistent proof serialization.
                from kiron_common.gpu_admission import Ticket
                tickets.append(Ticket(**vars(ticket),gpu_bytes=0,host_bytes=0,heartbeat_monotonic=1,ttl_seconds=600))
                class Operation:
                    async def events(self):
                        yield InferenceEvent(EventKind.STARTED,req.context.request_id)
                        await dead.wait()
                        yield InferenceEvent(EventKind.FAILED,req.context.request_id,error=RuntimeFailure(ErrorCode.PROVIDER_ERROR,'crash'))
                    async def close(self):
                        from dataclasses import replace
                        if tickets:
                            tickets[0]=replace(tickets[0],phase='unknown')
                operations.append(Operation())
                return operations[-1]
            async def chat(self,req):
                self.recovery_budget=req.options.max_output_tokens
                return InferenceResult(req.context.request_id,(TextPart('OK'),),(),(),TokenUsage(7,2),FinishReason.STOP)
            async def unload(self,model,context):
                controller.child=None
                tickets.clear()
        service=Service()
        async def control(req):
            if dead.is_set() and controller.generation==old:
                controller.child=None
                tickets.clear()
            return httpx.Response(200,json={'generation':asdict(controller.generation),
                'state':'loaded' if controller.child else 'unloaded','active_requests':int(bool(tickets)),
                'slot_task_id':7})
        async def native(req):
            self.assertEqual(req.headers['authorization'],'Bearer kiron-prism-old')
            self.assertEqual(controller.generation,new)
            return httpx.Response(401,json={'error':'unauthorized'})
        def crash(*args):
            self.assertTrue(tickets and tickets[0].phase=='active')
            child.process.returncode=-9
            dead.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(control),base_url='http://control',timeout=None) as control_client, \
                httpx.AsyncClient(transport=httpx.MockTransport(native),base_url='http://native',timeout=None) as native_client:
            provider=NS(control=control_client,inference=native_client)
            with patch.object(fault.os,'geteuid',return_value=65534), patch.object(fault.os,'getgroups',return_value=[]), \
                    patch.object(fault,'process_identity',return_value={'pid':777}),patch.object(fault,'guard_identity'), \
                    patch.object(fault,'signal_owned_child',side_effect=crash) as signalled,patch.object(fault.Path,'exists',return_value=False):
                value=await fault.probe(service,model,store,'native-crash',controller=controller,provider=provider)
        self.assertEqual(value['native_generations'],2)
        self.assertEqual(service.recovery_budget,8)
        self.assertEqual(len(loads),2)
        self.assertEqual([s['step'] for s in value['steps']],['loaded','before-crash','signal','stream-failed','reconciled','recovered','unloaded'])
        self.assertEqual(value['steps'][3]['work_state'],'unknown')
        signalled.assert_called_once()
        self.assertEqual(tickets,[])


if __name__=='__main__':
    unittest.main()

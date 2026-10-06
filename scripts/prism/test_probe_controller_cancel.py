"""No model or process starts: bounded native-cancel probe contracts."""
import asyncio
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx

spec = importlib.util.spec_from_file_location("cancel_probe", Path(__file__).with_name("probe-controller-cancel.py"))
cancel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cancel)


class CancelProbeTests(unittest.IsolatedAsyncioTestCase):
    async def workflow(self, *, defect=None):
        from kiron_common.gpu_admission import Ticket
        from kiron_common.local_inference import EventKind, InferenceEvent, RuntimeGeneration
        from test_openai_tools import request_for
        model = request_for().model
        generation = RuntimeGeneration("boot", "spawn")
        tickets = []
        child = NS(process=NS(pid=777, returncode=None))
        shutdowns, cancellation_seen = [], []
        controller = NS(child=child, generation=generation, policy=NS(binary=Path("/unused")))
        key = '["boot","spawn"]'
        async def shutdown():
            shutdowns.append(len(shutdowns))
            if controller.child is not None:
                self.assertTrue(cancellation_seen)
                self.assertTrue(any(t.phase == "unknown" for t in tickets))
                if defect == "cleanup":
                    return
                child.process.returncode = 0
                controller.child = None
                tickets.clear()
        controller.close = shutdown
        class Service:
            async def load(self, model, context):
                tickets.append(Ticket(context.request_id,"kiron-proxy",key,model.deployment.id,"load","resident",1,1,1,600))
                return NS(observation=NS(generation=generation))
            async def prepare(self, request, *, streaming):
                self.budget = request.options.max_output_tokens
                tickets.append(Ticket(request.context.request_id,"kiron-proxy",key,model.deployment.id,"request","active",0,0,1,600))
                class Operation:
                    async def events(self):
                        yield InferenceEvent(EventKind.STARTED,request.context.request_id)
                        if defect == "already_done":
                            return
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            cancellation_seen.append(request.context.cancellation.is_set())
                            raise
                    async def close(self):
                        for i, ticket in reversed(list(enumerate(tickets))):
                            if ticket.kind == "request":
                                if defect == "false_release":
                                    tickets.pop(i)
                                else:
                                    tickets[i] = replace(ticket,phase="unknown")
                return Operation()
        service = Service()
        async def control(request):
            return httpx.Response(200,json={"generation":asdict(generation),
                "state":"loaded" if controller.child else "unloaded",
                "active_requests":int(any(ticket.phase == "active" for ticket in tickets)),"slot_task_id":7})
        identity = {"pid":777}
        async with httpx.AsyncClient(transport=httpx.MockTransport(control),base_url="http://control",timeout=None) as client:
            with patch.object(cancel.os,"geteuid",return_value=65534), patch.object(cancel.os,"getgroups",return_value=[]), \
                    patch.object(cancel.faults,"process_identity",return_value=identity), \
                    patch.object(cancel.faults,"guard_identity"), patch.object(cancel.Path,"exists",return_value=False):
                value = await cancel.probe(service,model,NS(snapshot=lambda:tuple(tickets)),"native-cancel",
                                           controller=controller,provider=NS(control=client))
        return value, tickets, shutdowns, service, cancellation_seen

    async def test_realistic_fake_active_cancel_unknown_shutdown_and_idempotent_close(self):
        value,tickets,shutdowns,service,seen = await self.workflow()
        self.assertEqual([step["step"] for step in value["steps"]], ["loaded","before-cancel","cancelled","shutdown-recovered"])
        self.assertEqual(value["steps"][2]["work_state"], "unknown")
        self.assertEqual(seen,[True])
        self.assertEqual(service.budget,64)
        self.assertEqual(len(shutdowns),2)
        self.assertEqual(tickets,[])

    async def test_no_pass_when_request_ended_before_injection(self):
        with self.assertRaisesRegex(ValueError,"ended before"):
            await self.workflow(defect="already_done")

    async def test_no_pass_for_false_release_or_missing_reap(self):
        for defect in ("false_release","cleanup"):
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                await self.workflow(defect=defect)

    async def test_root_groups_unknown_kind_never_start(self):
        from unittest.mock import AsyncMock
        service = NS(load=AsyncMock())
        for uid,groups,kind in ((0,[],"native-cancel"),(65534,[982],"native-cancel"),(65534,[],"other")):
            with self.subTest(uid=uid,groups=groups,kind=kind), patch.object(cancel.os,"geteuid",return_value=uid), \
                    patch.object(cancel.os,"getgroups",return_value=groups), self.assertRaises(ValueError):
                await cancel.probe(service,None,None,kind,controller=None,provider=None)
        service.load.assert_not_awaited()

    def test_reap_requires_owned_controller_empty_native_exit_and_pid_absent(self):
        identity={"pid":777}
        child=NS(process=NS(returncode=0))
        controller=NS(child=None)
        with patch.object(cancel.Path,"exists",return_value=False):
            self.assertTrue(cancel.reaped(controller,child,identity))
            child.process.returncode=None
            self.assertFalse(cancel.reaped(controller,child,identity))
        child.process.returncode=0
        with patch.object(cancel.Path,"exists",return_value=True):
            self.assertFalse(cancel.reaped(controller,child,identity))
        self.assertEqual(cancel.MAX_SECONDS,600)


if __name__ == "__main__":
    unittest.main()

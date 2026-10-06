"""Completed embedding jobs must return unused CUDA blocks to other services."""
import asyncio
from contextlib import ExitStack
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import main
from model_worker import SerialModelWorker


class WorkspaceReleaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_error_release_workspace_before_completion_without_unloading(self):
        for fail in (False, True):
            with self.subTest(fail=fail), ExitStack() as patches:
                manager = main.ModelManager()
                model = object()
                manager.models['fixture'] = model
                manager.model = model
                manager.current_model_name = 'fixture'
                manager.device = 'cuda'
                manager._model_epoch = 3
                events = []
                reserved = [4096]

                def encode(*args, **kwargs):
                    manager.assert_worker_thread()
                    events.append('encode')
                    if fail:
                        raise ValueError('inference failed')
                    return 'encoded'

                def release():
                    manager.assert_worker_thread()
                    events.append('release')
                    reserved[0] = 256

                def reconcile(*args):
                    self.assertEqual(reserved[0], 256)
                    self.assertIs(manager.model, model)
                    events.append('reconcile')

                manager.residency = SimpleNamespace(complete=reconcile, heartbeat=lambda _: None)
                for name, value in {
                    'is_initialized': lambda: True,
                    'synchronize': lambda: events.append('synchronize'),
                    'memory_reserved': lambda: reserved[0],
                    'memory_allocated': lambda: 200,
                    'empty_cache': release,
                }.items():
                    patches.enter_context(mock.patch.object(main.torch.cuda, name, side_effect=value))
                patches.enter_context(mock.patch.object(manager, '_encode_sync', side_effect=encode))
                patches.enter_context(mock.patch.object(manager, '_drop_model_sync'))
                worker = SerialModelWorker(manager)
                worker.start()
                try:
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'inference failed'):
                            await worker.encode('fixture', ['text'], None)
                    else:
                        self.assertEqual(await worker.encode('fixture', ['text'], None), 'encoded')
                    self.assertEqual(events, ['encode', 'synchronize', 'release', 'reconcile'])
                    self.assertEqual(manager._model_epoch, 3)
                    self.assertEqual(list(manager.models), ['fixture'])
                    self.assertNotEqual(manager.owner_thread_id, threading.get_ident())
                finally:
                    await worker.stop()

    async def test_cpu_boundary_does_not_initialize_cuda(self):
        manager = main.ModelManager()
        manager.set_worker_thread()
        with mock.patch.object(main.torch.cuda, 'is_initialized', return_value=False), \
             mock.patch.object(main.torch.cuda, 'synchronize') as synchronize, \
             mock.patch.object(main.torch.cuda, 'empty_cache') as release:
            manager.residency_boundary()
        synchronize.assert_not_called()
        release.assert_not_called()

    async def test_cleanup_failure_cannot_confirm_residency(self):
        manager = main.ModelManager()
        manager.set_worker_thread()
        manager.residency = mock.Mock()
        with mock.patch.object(main.torch.cuda, 'is_initialized', return_value=True), \
             mock.patch.object(main.torch.cuda, 'synchronize', side_effect=RuntimeError('CUDA failed')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA failed'):
                manager.residency_boundary()
        manager.residency.complete.assert_not_called()
        manager.residency.unknown.assert_called_once_with()

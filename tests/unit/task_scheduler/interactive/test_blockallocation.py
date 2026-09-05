import queue
import unittest
from concurrent.futures import Future
from threading import Event, Lock
from unittest.mock import patch

from executorlib.standalone.interactive.communication import ExecutorlibSocketError
from executorlib.task_scheduler.interactive.blockallocation import (
    BlockAllocationTaskScheduler,
    _drain_dead_worker,
)


class TestBlockAllocationResize(unittest.TestCase):
    def test_increase_workers_passes_worker_context(self):
        scheduler = object.__new__(BlockAllocationTaskScheduler)
        scheduler._future_queue = queue.Queue()
        scheduler._process = []
        scheduler._process_kwargs = {"future_queue": scheduler._future_queue}
        scheduler._max_workers = 1
        scheduler._self_id = 1
        scheduler._alive_workers = [1]
        scheduler._alive_workers_lock = Lock()
        scheduler._bootup_events = [Event()]

        class FakeThread:
            instances = []

            def __init__(self, target, kwargs):
                self.target = target
                self.kwargs = kwargs
                self.started = False
                self.instances.append(self)

            def start(self):
                self.started = True

        with patch(
            "executorlib.task_scheduler.interactive.blockallocation.Thread",
            FakeThread,
        ):
            scheduler.max_workers = 2

        worker = FakeThread.instances[-1]
        self.assertEqual(worker.kwargs["worker_id"], 1)
        self.assertIn("stop_function", worker.kwargs)
        self.assertIn("bootup_event", worker.kwargs)
        self.assertIn("next_bootup_event", worker.kwargs)
        self.assertIs(worker.kwargs["alive_workers"], scheduler._alive_workers)
        self.assertTrue(worker.started)
        self.assertEqual(scheduler._alive_workers[0], 2)


class TestDrainDeadWorker(unittest.TestCase):
    def test_fail_tasks_when_no_workers_remain(self):
        future_queue = queue.Queue()
        alive_workers = [1]
        alive_workers_lock = Lock()
        future = Future()

        # Add a task and then the shutdown sentinel
        future_queue.put({"fn": lambda: 42, "future": future})
        future_queue.put({"shutdown": True})

        _drain_dead_worker(
            future_queue=future_queue,
            alive_workers=alive_workers,
            alive_workers_lock=alive_workers_lock,
        )

        # Worker count should be decremented
        self.assertEqual(alive_workers[0], 0)

        # Task should fail with ExecutorlibSocketError
        with self.assertRaises(ExecutorlibSocketError):
            future.result()

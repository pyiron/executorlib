import queue
import time
import unittest
from concurrent.futures import Future
from threading import Event, Lock, Thread
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

    def test_shrink_wakes_worker_blocked_on_empty_queue(self):
        """
        Regression test: shrinking max_workers while a worker is already blocked in
        future_queue.get() used to hang forever, because the shutdown sentinel was
        spliced directly into the queue's internal deque without waking the blocked
        consumer (see executorlib.standalone.queue.put_front).
        """
        future_queue = queue.Queue()
        received = []

        def worker_loop():
            received.append(future_queue.get())
            future_queue.task_done()

        worker = Thread(target=worker_loop, daemon=True)
        worker.start()
        time.sleep(0.2)  # let the worker actually block inside future_queue.get()

        scheduler = object.__new__(BlockAllocationTaskScheduler)
        scheduler._future_queue = future_queue
        scheduler._process = [worker]
        scheduler._max_workers = 1

        shrink_done = Event()

        def shrink():
            scheduler.max_workers = 0
            shrink_done.set()

        shrink_thread = Thread(target=shrink, daemon=True)
        shrink_thread.start()
        shrink_thread.join(timeout=5)

        self.assertTrue(shrink_done.is_set(), "max_workers setter hung while shrinking")
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received, [{"shutdown": True, "wait": True}])
        self.assertEqual(scheduler._process, [])


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

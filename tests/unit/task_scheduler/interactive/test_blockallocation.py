import queue
import unittest
from threading import Lock
from concurrent.futures import Future

from executorlib.task_scheduler.interactive.blockallocation import _drain_dead_worker
from executorlib.task_scheduler.interactive.shared import task_done
from executorlib.standalone.interactive.communication import ExecutorlibSocketError


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

from concurrent.futures import Future
import os
from queue import Queue
import shutil
import unittest

from executorlib.shared.thread import RaisingThread

try:
    from executorlib import FileExecutor
    from executorlib.cache.shared import execute_tasks_h5, execute_in_subprocess

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True


def my_funct(a, b):
    return a + b


@unittest.skipIf(
    skip_h5io_test, "h5io is not installed, so the h5io tests are skipped."
)
class TestCacheExecutorSerial(unittest.TestCase):
    def test_executor_mixed(self):
        with FileExecutor() as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), 3)
            self.assertTrue(fs1.done())

    def test_executor_dependence_mixed(self):
        with FileExecutor() as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            fs2 = exe.submit(my_funct, 1, b=fs1)
            self.assertFalse(fs2.done())
            self.assertEqual(fs2.result(), 4)
            self.assertTrue(fs2.done())

    def test_executor_function(self):
        fs1 = Future()
        q = Queue()
        q.put({"fn": my_funct, "args": (), "kwargs": {"a": 1, "b": 2}, "future": fs1})
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "cores_per_worker": 1,
            },
        )
        process.start()
        self.assertFalse(fs1.done())
        self.assertEqual(fs1.result(), 3)
        self.assertTrue(fs1.done())
        q.put({"shutdown": True, "wait": True})

    def test_executor_function_dependence_kwargs(self):
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put({"fn": my_funct, "args": (), "kwargs": {"a": 1, "b": 2}, "future": fs1})
        q.put({"fn": my_funct, "args": (), "kwargs": {"a": 1, "b": fs1}, "future": fs2})
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "cores_per_worker": 1,
            },
        )
        process.start()
        self.assertFalse(fs2.done())
        self.assertEqual(fs2.result(), 4)
        self.assertTrue(fs2.done())
        q.put({"shutdown": True, "wait": True})

    def test_executor_function_dependence_args(self):
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put({"fn": my_funct, "args": (), "kwargs": {"a": 1, "b": 2}, "future": fs1})
        q.put({"fn": my_funct, "args": [fs1], "kwargs": {"b": 2}, "future": fs2})
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "cores_per_worker": 1,
            },
        )
        process.start()
        self.assertFalse(fs2.done())
        self.assertEqual(fs2.result(), 5)
        self.assertTrue(fs2.done())
        q.put({"shutdown": True, "wait": True})

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")

from concurrent.futures import Future
import os
from queue import Queue
import shutil
import unittest

from executorlib.cache.subprocess_spawner import (
    execute_in_subprocess,
    terminate_subprocess,
)
from executorlib.standalone.thread import RaisingThread

try:
    from executorlib.cache.executor import FileExecutor, create_file_executor
    from executorlib.cache.shared import execute_tasks_h5

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def my_funct(a, b):
    return a + b


def list_files_in_working_directory():
    return os.listdir(os.getcwd())


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5py tests are skipped."
)
class TestCacheExecutorSerial(unittest.TestCase):
    def test_executor_mixed(self):
        with FileExecutor(execute_function=execute_in_subprocess) as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), 3)
            self.assertTrue(fs1.done())

    def test_executor_dependence_mixed(self):
        with FileExecutor(execute_function=execute_in_subprocess) as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            fs2 = exe.submit(my_funct, 1, b=fs1)
            self.assertFalse(fs2.done())
            self.assertEqual(fs2.result(), 4)
            self.assertTrue(fs2.done())

    def test_create_file_executor_error(self):
        with self.assertRaises(ValueError):
            create_file_executor(block_allocation=True)
        with self.assertRaises(ValueError):
            create_file_executor(init_function=True)

    def test_executor_dependence_error(self):
        with self.assertRaises(ValueError):
            with FileExecutor(
                execute_function=execute_in_subprocess, disable_dependencies=True
            ) as exe:
                exe.submit(my_funct, 1, b=exe.submit(my_funct, 1, b=2))

    def test_executor_working_directory(self):
        cwd = os.path.join(os.path.dirname(__file__), "executables")
        with FileExecutor(
            resource_dict={"cwd": cwd}, execute_function=execute_in_subprocess
        ) as exe:
            fs1 = exe.submit(list_files_in_working_directory)
            self.assertEqual(fs1.result(), os.listdir(cwd))

    def test_executor_function(self):
        fs1 = Future()
        q = Queue()
        q.put(
            {
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": 1, "b": 2},
                "future": fs1,
                "resource_dict": {},
            }
        )
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1, "cwd": None},
                "terminate_function": terminate_subprocess,
            },
        )
        process.start()
        self.assertFalse(fs1.done())
        self.assertEqual(fs1.result(), 3)
        self.assertTrue(fs1.done())
        q.put({"shutdown": True, "wait": True})
        process.join()

    def test_executor_function_dependence_kwargs(self):
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put(
            {
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": 1, "b": 2},
                "future": fs1,
                "resource_dict": {},
            }
        )
        q.put(
            {
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": 1, "b": fs1},
                "future": fs2,
                "resource_dict": {},
            }
        )
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1, "cwd": None},
                "terminate_function": terminate_subprocess,
            },
        )
        process.start()
        self.assertFalse(fs2.done())
        self.assertEqual(fs2.result(), 4)
        self.assertTrue(fs2.done())
        q.put({"shutdown": True, "wait": True})
        process.join()

    def test_executor_function_dependence_args(self):
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put(
            {
                "fn": my_funct,
                "args": (),
                "kwargs": {"a": 1, "b": 2},
                "future": fs1,
                "resource_dict": {},
            }
        )
        q.put(
            {
                "fn": my_funct,
                "args": [fs1],
                "kwargs": {"b": 2},
                "future": fs2,
                "resource_dict": {},
            }
        )
        cache_dir = os.path.abspath("cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = RaisingThread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "cache_directory": cache_dir,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1},
                "terminate_function": terminate_subprocess,
            },
        )
        process.start()
        self.assertFalse(fs2.done())
        self.assertEqual(fs2.result(), 5)
        self.assertTrue(fs2.done())
        q.put({"shutdown": True, "wait": True})
        process.join()

    def test_execute_in_subprocess_errors(self):
        with self.assertRaises(ValueError):
            execute_in_subprocess(
                file_name=__file__, command=[], config_directory="test"
            )
        with self.assertRaises(ValueError):
            execute_in_subprocess(file_name=__file__, command=[], backend="flux")

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")

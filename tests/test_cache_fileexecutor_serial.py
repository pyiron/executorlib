from concurrent.futures import Future
import os
from queue import Queue
import shutil
import unittest
from threading import Thread

try:
    from executorlib.task_scheduler.file.subprocess_spawner import (
        execute_in_subprocess,
        terminate_subprocess,
    )
    from executorlib.task_scheduler.file.task_scheduler import FileTaskScheduler, create_file_executor
    from executorlib.task_scheduler.file.shared import execute_tasks_h5

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def my_funct(a, b):
    return a + b


def list_files_in_working_directory():
    return os.listdir(os.getcwd())


def get_error(a):
    raise ValueError(a)


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5py tests are skipped."
)
class TestCacheExecutorSerial(unittest.TestCase):
    def test_executor_mixed(self):
        with FileTaskScheduler(execute_function=execute_in_subprocess) as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), 3)
            self.assertTrue(fs1.done())

    def test_executor_mixed_cache_key(self):
        with FileTaskScheduler(execute_function=execute_in_subprocess) as exe:
            fs1 = exe.submit(my_funct, 1, b=2, resource_dict={"cache_key": "a/b/c"})
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), 3)
            self.assertTrue(fs1.done())

    def test_executor_dependence_mixed(self):
        with FileTaskScheduler(execute_function=execute_in_subprocess) as exe:
            fs1 = exe.submit(my_funct, 1, b=2)
            fs2 = exe.submit(my_funct, 1, b=fs1)
            self.assertFalse(fs2.done())
            self.assertEqual(fs2.result(), 4)
            self.assertTrue(fs2.done())

    def test_create_file_executor_error(self):
        with self.assertRaises(TypeError):
            create_file_executor()
        with self.assertRaises(ValueError):
            create_file_executor(block_allocation=True, resource_dict={})
        with self.assertRaises(ValueError):
            create_file_executor(init_function=True, resource_dict={})

    def test_executor_dependence_error(self):
        with self.assertRaises(ValueError):
            with FileTaskScheduler(
                execute_function=execute_in_subprocess, disable_dependencies=True
            ) as exe:
                fs = exe.submit(my_funct, 1, b=exe.submit(my_funct, 1, b=2))
                fs.result()

    def test_executor_working_directory(self):
        cwd = os.path.join(os.path.dirname(__file__), "executables")
        with FileTaskScheduler(
            resource_dict={"cwd": cwd}, execute_function=execute_in_subprocess
        ) as exe:
            fs1 = exe.submit(list_files_in_working_directory)
            self.assertEqual(fs1.result(), os.listdir(cwd))

    def test_executor_error(self):
        cwd = os.path.join(os.path.dirname(__file__), "executables")
        with FileTaskScheduler(
            resource_dict={"cwd": cwd}, execute_function=execute_in_subprocess, write_error_file=False,
        ) as exe:
            fs1 = exe.submit(get_error, a=1)
            with self.assertRaises(ValueError):
                fs1.result()
        self.assertEqual(len(os.listdir(cwd)), 1)

    def test_executor_error_file(self):
        cwd = os.path.join(os.path.dirname(__file__), "executables")
        with FileTaskScheduler(
            resource_dict={"cwd": cwd}, execute_function=execute_in_subprocess, write_error_file=True,
        ) as exe:
            fs1 = exe.submit(get_error, a=1)
            with self.assertRaises(ValueError):
                fs1.result()
        working_directory_file_lst = os.listdir(cwd)
        self.assertEqual(len(working_directory_file_lst), 2)
        self.assertTrue("error.out" in working_directory_file_lst)
        os.remove(os.path.join(cwd, "error.out"))

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
        cache_dir = os.path.abspath("executorlib_cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = Thread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1, "cwd": None, "cache_directory": cache_dir},
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
        cache_dir = os.path.abspath("executorlib_cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = Thread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1, "cwd": None, "cache_directory": cache_dir},
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
        cache_dir = os.path.abspath("executorlib_cache")
        os.makedirs(cache_dir, exist_ok=True)
        process = Thread(
            target=execute_tasks_h5,
            kwargs={
                "future_queue": q,
                "execute_function": execute_in_subprocess,
                "resource_dict": {"cores": 1, "cache_directory": cache_dir},
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
        file_name = os.path.abspath(os.path.join(__file__, "..", "executorlib_cache", "test.h5"))
        os.makedirs(os.path.dirname(file_name))
        with open(file_name, "w") as f:
            f.write("test")
        with self.assertRaises(ValueError):
            execute_in_subprocess(
                file_name=file_name,
                data_dict={},
                command=[],
                config_directory="test",
            )
        with self.assertRaises(ValueError):
            execute_in_subprocess(
                file_name=file_name,
                data_dict={},
                command=[],
                backend="flux",
            )

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)

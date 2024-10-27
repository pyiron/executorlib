from concurrent.futures import Future
import subprocess
import queue
import unittest

from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.interactive.shared import execute_parallel_tasks
from executorlib.standalone.interactive.spawner import MpiExecSpawner


def submit_shell_command(
    command: list, universal_newlines: bool = True, shell: bool = False
):
    return subprocess.check_output(
        command, universal_newlines=universal_newlines, shell=shell
    )


class SubprocessExecutorTest(unittest.TestCase):
    def test_execute_single_task(self):
        test_queue = queue.Queue()
        f = Future()
        test_queue.put(
            {
                "fn": submit_shell_command,
                "args": [["echo", "test"]],
                "kwargs": {"universal_newlines": True},
                "future": f,
            }
        )
        test_queue.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        self.assertFalse(f.done())
        execute_parallel_tasks(
            future_queue=test_queue,
            cores=1,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
        )
        self.assertTrue(f.done())
        self.assertEqual("test\n", f.result())
        test_queue.join()

    def test_wrong_error(self):
        test_queue = queue.Queue()
        f = Future()
        test_queue.put(
            {
                "fn": submit_shell_command,
                "args": [["echo", "test"]],
                "kwargs": {"wrong_key": True},
                "future": f,
            }
        )
        cloudpickle_register(ind=1)
        with self.assertRaises(TypeError):
            execute_parallel_tasks(
                future_queue=test_queue,
                cores=1,
                openmpi_oversubscribe=False,
                spawner=MpiExecSpawner,
            )

    def test_broken_executable(self):
        test_queue = queue.Queue()
        f = Future()
        test_queue.put(
            {
                "fn": submit_shell_command,
                "args": [["/executable/does/not/exist"]],
                "kwargs": {"universal_newlines": True},
                "future": f,
            }
        )
        cloudpickle_register(ind=1)
        with self.assertRaises(FileNotFoundError):
            execute_parallel_tasks(
                future_queue=test_queue,
                cores=1,
                openmpi_oversubscribe=False,
                spawner=MpiExecSpawner,
            )

    def test_shell_static_executor_args(self):
        with Executor(max_workers=1) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                submit_shell_command,
                ["echo", "test"],
                universal_newlines=True,
                shell=False,
            )
            self.assertFalse(future.done())
            self.assertEqual("test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_static_executor_binary(self):
        with Executor(max_workers=1) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                submit_shell_command,
                ["echo", "test"],
                universal_newlines=False,
                shell=False,
            )
            self.assertFalse(future.done())
            self.assertEqual(b"test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_static_executor_shell(self):
        with Executor(max_workers=1) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                submit_shell_command, "echo test", universal_newlines=True, shell=True
            )
            self.assertFalse(future.done())
            self.assertEqual("test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_executor(self):
        with Executor(max_workers=2) as exe:
            cloudpickle_register(ind=1)
            f_1 = exe.submit(
                submit_shell_command, ["echo", "test_1"], universal_newlines=True
            )
            f_2 = exe.submit(
                submit_shell_command, ["echo", "test_2"], universal_newlines=True
            )
            f_3 = exe.submit(
                submit_shell_command, ["echo", "test_3"], universal_newlines=True
            )
            f_4 = exe.submit(
                submit_shell_command, ["echo", "test_4"], universal_newlines=True
            )
            self.assertFalse(f_1.done())
            self.assertFalse(f_2.done())
            self.assertFalse(f_3.done())
            self.assertFalse(f_4.done())
            self.assertEqual("test_1\n", f_1.result())
            self.assertEqual("test_2\n", f_2.result())
            self.assertTrue(f_1.done())
            self.assertTrue(f_2.done())
            self.assertEqual("test_3\n", f_3.result())
            self.assertEqual("test_4\n", f_4.result())
            self.assertTrue(f_1.done())
            self.assertTrue(f_2.done())
            self.assertTrue(f_3.done())
            self.assertTrue(f_4.done())

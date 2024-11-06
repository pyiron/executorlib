from concurrent.futures import Future
import os
import subprocess
import queue
import unittest

from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.interactive.shared import execute_parallel_tasks
from executorlib.standalone.interactive.spawner import MpiExecSpawner


executable_path = os.path.join(os.path.dirname(__file__), "executables", "count.py")


def init_process():
    return {
        "process": subprocess.Popen(
            ["python", executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=False,
        )
    }


def interact(shell_input, process, lines_to_read=None, stop_read_pattern=None):
    process.stdin.write(shell_input)
    process.stdin.flush()
    lines_count = 0
    output = ""
    while True:
        output_current = process.stdout.readline()
        output += output_current
        lines_count += 1
        if stop_read_pattern is not None and stop_read_pattern in output_current:
            break
        elif lines_to_read is not None and lines_to_read == lines_count:
            break
    return output


def shutdown(process):
    process.stdin.write("shutdown\n")
    process.stdin.flush()


class ShellInteractiveExecutorTest(unittest.TestCase):
    def test_execute_single_task(self):
        test_queue = queue.Queue()
        future_lines = Future()
        future_pattern = Future()
        future_shutdown = Future()
        test_queue.put(
            {
                "fn": interact,
                "future": future_lines,
                "args": (),
                "kwargs": {
                    "shell_input": "4\n",
                    "lines_to_read": 5,
                    "stop_read_pattern": None,
                },
            }
        )
        test_queue.put(
            {
                "fn": interact,
                "future": future_pattern,
                "args": (),
                "kwargs": {
                    "shell_input": "4\n",
                    "lines_to_read": None,
                    "stop_read_pattern": "done",
                },
            }
        )
        test_queue.put(
            {
                "fn": shutdown,
                "future": future_shutdown,
                "args": (),
                "kwargs": {},
            }
        )
        test_queue.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        self.assertFalse(future_lines.done())
        self.assertFalse(future_pattern.done())
        execute_parallel_tasks(
            future_queue=test_queue,
            cores=1,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
            init_function=init_process,
        )
        self.assertTrue(future_lines.done())
        self.assertTrue(future_pattern.done())
        self.assertTrue(future_shutdown.done())
        self.assertEqual("0\n1\n2\n3\ndone\n", future_lines.result())
        self.assertEqual("0\n1\n2\n3\ndone\n", future_pattern.result())
        test_queue.join()

    def test_shell_interactive_executor(self):
        cloudpickle_register(ind=1)
        with Executor(
            max_workers=1,
            init_function=init_process,
            block_allocation=True,
        ) as exe:
            future_lines = exe.submit(
                interact, shell_input="4\n", lines_to_read=5, stop_read_pattern=None
            )
            future_pattern = exe.submit(
                interact,
                shell_input="4\n",
                lines_to_read=None,
                stop_read_pattern="done",
            )
            self.assertFalse(future_lines.done())
            self.assertFalse(future_pattern.done())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_lines.result())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_pattern.result())
            self.assertTrue(future_lines.done())
            self.assertTrue(future_pattern.done())
            future_shutdown = exe.submit(shutdown)
            self.assertIsNone(future_shutdown.result())
            self.assertTrue(future_shutdown.done())

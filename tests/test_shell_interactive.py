from concurrent.futures import Future
import os
import queue

from unittest import TestCase

from pympipool.shell.interactive import ShellExecutor, execute_single_task


class ShellInteractiveExecutorTest(TestCase):
    def setUp(self):
        self.executable_path = os.path.join(os.path.dirname(__file__), "executables", "count.py")

    def test_execute_single_task(self):
        test_queue = queue.Queue()
        future_lines = Future()
        future_pattern = Future()
        test_queue.put({"init": True, "args": [["python", self.executable_path]], "kwargs": {"universal_newlines": True}})
        test_queue.put({"future": future_lines, "input": "4\n", "lines_to_read": 5, "stop_read_pattern": None})
        test_queue.put({"future": future_pattern, "input": "4\n", "lines_to_read": None, "stop_read_pattern": "done"})
        test_queue.put({"shutdown": True})
        self.assertFalse(future_lines.done())
        self.assertFalse(future_pattern.done())
        execute_single_task(future_queue=test_queue)
        self.assertTrue(future_lines.done())
        self.assertTrue(future_pattern.done())
        self.assertEqual("0\n1\n2\n3\ndone\n", future_lines.result())
        self.assertEqual("0\n1\n2\n3\ndone\n", future_pattern.result())

    def test_shell_interactive_executor(self):
        with ShellExecutor(["python", self.executable_path], universal_newlines=True) as exe:
            future_lines = exe.submit(string_input="4", lines_to_read=5, stop_read_pattern=None)
            future_pattern = exe.submit(string_input="4", lines_to_read=None, stop_read_pattern="done")
            self.assertFalse(future_lines.done())
            self.assertFalse(future_pattern.done())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_lines.result())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_pattern.result())
            self.assertTrue(future_lines.done())
            self.assertTrue(future_pattern.done())

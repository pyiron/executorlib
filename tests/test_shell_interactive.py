import os

from unittest import TestCase

from pympipool.shell.interactive import ShellInteractiveExecutor


class InteractiveExecutorTest(TestCase):
    def test_shell_interactive(self):
        executable_path = os.path.join(os.path.dirname(__file__), "executables", "count.py")
        with ShellInteractiveExecutor(["python", executable_path], universal_newlines=True) as exe:
            future_lines = exe.submit(string_input="4", lines_to_read=5, stop_read_pattern=None)
            future_pattern = exe.submit(string_input="4", lines_to_read=None, stop_read_pattern="done")
            self.assertFalse(future_lines.done())
            self.assertFalse(future_pattern.done())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_lines.result())
            self.assertEqual("0\n1\n2\n3\ndone\n", future_pattern.result())
            self.assertTrue(future_lines.done())
            self.assertTrue(future_pattern.done())

from concurrent.futures import Future
import queue

from unittest import TestCase

from pympipool.shell.executor import SubprocessSingleExecutor, SubprocessExecutor, execute_single_task


class SubprocessExecutorTest(TestCase):
    def test_execute_single_task(self):
        test_queue = queue.Queue()
        f = Future()
        test_queue.put({"future": f, "args": [["echo", "test"]], "kwargs": {"universal_newlines": True}})
        test_queue.put({"shutdown": True})
        self.assertFalse(f.done())
        execute_single_task(future_queue=test_queue)
        self.assertTrue(f.done())
        self.assertEqual("test\n", f.result())

    def test_shell_static_executor_args(self):
        with SubprocessSingleExecutor() as exe:
            future = exe.submit(["echo", "test"], universal_newlines=True, shell=False)
            self.assertFalse(future.done())
            self.assertEqual("test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_static_executor_binary(self):
        with SubprocessSingleExecutor() as exe:
            future = exe.submit(["echo", "test"], universal_newlines=False, shell=False)
            self.assertFalse(future.done())
            self.assertEqual(b"test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_static_executor_shell(self):
        with SubprocessSingleExecutor() as exe:
            future = exe.submit("echo test", universal_newlines=True, shell=True)
            self.assertFalse(future.done())
            self.assertEqual("test\n", future.result())
            self.assertTrue(future.done())

    def test_shell_executor(self):
        with SubprocessExecutor(max_workers=2) as exe:
            f_1 = exe.submit(["echo", "test_1"], universal_newlines=True)
            f_2 = exe.submit(["echo", "test_2"], universal_newlines=True)
            f_3 = exe.submit(["echo", "test_3"], universal_newlines=True)
            f_4 = exe.submit(["echo", "test_4"], universal_newlines=True)
            self.assertFalse(f_1.done())
            self.assertFalse(f_2.done())
            self.assertFalse(f_3.done())
            self.assertFalse(f_4.done())
            self.assertEqual("test_1\n", f_1.result())
            self.assertEqual("test_2\n", f_2.result())
            self.assertTrue(f_1.done())
            self.assertTrue(f_2.done())
            self.assertFalse(f_3.done())
            self.assertFalse(f_4.done())
            self.assertEqual("test_3\n", f_3.result())
            self.assertEqual("test_4\n", f_4.result())
            self.assertTrue(f_1.done())
            self.assertTrue(f_2.done())
            self.assertTrue(f_3.done())
            self.assertTrue(f_4.done())

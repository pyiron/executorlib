import unittest
from concurrent.futures import Future
from executorlib import SingleNodeExecutor
from executorlib.api import cloudpickle_register
from executorlib.standalone.split import split, SplitFuture


def function_with_multiple_outputs(i):
    return "a", "b", i


def function_with_exception(i):
    raise RuntimeError()


class TestSplitFuture(unittest.TestCase):
    def test_integration_base(self):
        with SingleNodeExecutor() as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(function_with_multiple_outputs, 15)
            f1, f2, f3 = split(future=future, n=3)
            self.assertEqual(f1.result(), "a")
            self.assertEqual(f2.result(), "b")
            self.assertEqual(f3.result(), 15)
            self.assertTrue(f1.done())
            self.assertTrue(f2.done())
            self.assertTrue(f3.done())

    def test_integration_exception(self):
        with SingleNodeExecutor() as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(function_with_exception, 15)
            f1, f2, f3 = split(future=future, n=3)
            with self.assertRaises(RuntimeError):
                f3.result()

    def test_split_future_object(self):
        f1 = Future()
        fs1 = SplitFuture(future=f1, selector=1)
        fs1.set_running_or_notify_cancel()
        self.assertTrue(fs1.running())
        fs1.set_result([1, 2])
        self.assertEqual(fs1.result(), 2)
        f2 = Future()
        fs2 = SplitFuture(future=f2, selector=1)
        fs2.cancel()
        self.assertTrue(fs2.cancelled())
        f3 = Future()
        fs3 = SplitFuture(future=f3, selector=1)
        fs3.set_exception(RuntimeError())
        with self.assertRaises(RuntimeError):
            fs3.result()

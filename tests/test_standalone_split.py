import unittest
from concurrent.futures import Future
from executorlib import SingleNodeExecutor, split_future, get_item_from_future
from executorlib.api import cloudpickle_register
from executorlib.standalone.split import FutureSelector


def function_returns_tuple(i):
    return "a", "b", i


def function_returns_dict(i):
    return {"a": 1, "b": 2, "c": i}


def function_with_exception(i):
    raise RuntimeError()


def callback(future):
    print("callback:", future.result())


class TestSplitFuture(unittest.TestCase):
    def test_integration_return_tuple(self):
        with SingleNodeExecutor() as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(function_returns_tuple, 15)
            f1, f2, f3 = split_future(future=future, n=3)
            self.assertEqual(f1.result(), "a")
            self.assertEqual(f2.result(), "b")
            self.assertEqual(f3.result(), 15)
            self.assertTrue(f1.done())
            self.assertTrue(f2.done())
            self.assertTrue(f3.done())

    def test_integration_return_dict(self):
        with SingleNodeExecutor() as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(function_returns_dict, 15)
            f1 = get_item_from_future(future=future, key="a")
            f2 = get_item_from_future(future=future, key="b")
            f3 = get_item_from_future(future=future, key="c")
            self.assertEqual(f1.result(), 1)
            self.assertEqual(f2.result(), 2)
            self.assertEqual(f3.result(), 15)
            self.assertTrue(f1.done())
            self.assertTrue(f2.done())
            self.assertTrue(f3.done())

    def test_integration_exception(self):
        with SingleNodeExecutor() as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(function_with_exception, 15)
            f1, f2, f3 = split_future(future=future, n=3)
            with self.assertRaises(RuntimeError):
                f3.result()

    def test_split_future_object(self):
        f1 = Future()
        fs1 = FutureSelector(future=f1, selector=1)
        fs1.add_done_callback(callback)
        fs1.set_running_or_notify_cancel()
        self.assertTrue(fs1.running())
        fs1.set_result([1, 2])
        self.assertEqual(fs1.result(), 2)
        f2 = Future()
        fs2 = FutureSelector(future=f2, selector=1)
        fs2.cancel()
        self.assertTrue(fs2.cancelled())
        f3 = Future()
        fs3 = FutureSelector(future=f3, selector=1)
        fs3.set_running_or_notify_cancel()
        self.assertTrue(fs3.running())
        fs3.set_result(None)
        self.assertEqual(fs3.result(), None)
        f4 = Future()
        fs4 = FutureSelector(future=f4, selector=1)
        fs4.set_exception(RuntimeError())
        self.assertEqual(type(fs4.exception()), RuntimeError)
        with self.assertRaises(RuntimeError):
            fs4.result()

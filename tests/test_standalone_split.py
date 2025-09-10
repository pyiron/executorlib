import unittest
from executorlib import SingleNodeExecutor
from executorlib.standalone.split import split


def function_with_multiple_outputs(i):
    return "a", "b", i


def function_with_exception(i):
    raise RuntimeError()


class TestSplitFuture(unittest.TestCase):
    def test_integration_base(self):
        with SingleNodeExecutor() as exe:
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
            future = exe.submit(function_with_exception, 15)
            f1, f2, f3 = split(future=future, n=3)
            with self.assertRaises(RuntimeError):
                f3.result()
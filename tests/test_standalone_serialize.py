import unittest
from executorlib.standalone.serialize import _get_function_name


def my_function(a: int, b: int) -> int:
    return a + b


class MyClass:
    def __call__(self, a: int, b: int) -> int:
        return a + b


class TestSerialization(unittest.TestCase):
    def test_serialization(self):
        fn = _get_function_name(fn=my_function)
        self.assertEqual(fn, "my_function")
        fn = _get_function_name(fn=MyClass())
        self.assertEqual(fn, "MyClass")
        fn = _get_function_name(fn=None)
        self.assertEqual(fn, "None")

import unittest
import numpy as np
from executorlib.task_scheduler.interactive.dependency_plot import _short_object_name


class MyClass:
    def __init__(self, i):
        self._i = i


def my_function(i):
    return i


class TestShortObjectName(unittest.TestCase):
    def test_short_object_name(self):
        result = _short_object_name(node=[MyClass("a"), MyClass("b")])
        self.assertEqual("['MyClass()', 'MyClass()']", result)
        result = _short_object_name(node=(MyClass("a"), MyClass("b")))
        self.assertEqual("('MyClass()', 'MyClass()')", result)
        result = _short_object_name(node=[np.array([[1,2], [4,3]]), np.array([[1,2], [4,3]])])
        self.assertEqual("['ndarray()', 'ndarray()']", result)
        result = _short_object_name(node={"this is a very long string far too long for a dictionary key": my_function})
        self.assertEqual("{'this is a very long s...': 'my_function()'}", result)

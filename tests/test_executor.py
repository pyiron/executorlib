import unittest
from pympipool.shared.backend import call_funct


def function_multi_args(a, b):
    return a + b


class TestExecutor(unittest.TestCase):
    def test_funct_call_default(self):
        self.assertEqual(call_funct(input_dict={
            "fn": sum,
            "args": [[1, 2, 3]],
            "kwargs": {}
        }), 6)

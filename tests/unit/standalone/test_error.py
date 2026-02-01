import os
import unittest
from executorlib.standalone.error import backend_write_error_file


class TestErrorWriter(unittest.TestCase):
    def test_backend_write_error_file(self):
        backend_write_error_file(
            error=ValueError(),
            apply_dict={
                "error_log_file": "error.out",
                "fn": 1,
                "args": (1, 2, 3),
                "kwargs": {"a": 1, "b": 2, "c": 3},
            }
        )
        error_file_content = [
            'function: 1\n',
            'args: (1, 2, 3)\n',
            "kwargs: {'a': 1, 'b': 2, 'c': 3}\n",
            'ValueError\n'
        ]
        with open("error.out", "r") as f:
            content = f.readlines()
        self.assertEqual(error_file_content, content)

    def tearDown(self):
        if os.path.exists("error.out"):
            os.remove("error.out")
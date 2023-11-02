from functools import partialmethod
from concurrent.futures import TimeoutError
from time import sleep
import unittest

from pympipool.mpi.executor import PyMPISingleTaskExecutor


class Foo:
    """
    A base class to be dynamically modified for testing CloudpickleProcessPoolExecutor.
    """
    def __init__(self, fnc: callable):
        self.fnc = fnc
        self.result = None

    @property
    def run(self):
        return self.fnc

    def process_result(self, future):
        self.result = future.result()


def dynamic_foo():
    """
    A decorator for dynamically modifying the Foo class to test
    CloudpickleProcessPoolExecutor.

    Overrides the `fnc` input of `Foo` with the decorated function.
    """
    def as_dynamic_foo(fnc: callable):
        return type(
            "DynamicFoo",
            (Foo,),  # Define parentage
            {
                "__init__": partialmethod(
                    Foo.__init__,
                    fnc
                )
            },
        )

    return as_dynamic_foo


class TestUnpickleableElements(unittest.TestCase):
    def test_timeout(self):
        fortytwo = 42

        @dynamic_foo()
        def slow():
            sleep(0.1)
            return fortytwo

        f = slow()
        executor = PyMPISingleTaskExecutor()
        fs = executor.submit(f.run)
        self.assertEqual(
            fs.result(timeout=30),
            fortytwo,
            msg="waiting long enough should get the result"
        )

        with self.assertRaises(TimeoutError):
            fs = executor.submit(f.run)
            fs.result(timeout=0.0001)

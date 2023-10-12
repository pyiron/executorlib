from functools import partialmethod
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
    def test_unpickleable_return(self):
        """
        We should be able to use an unpickleable return value -- in this case, a
        method of a dynamically defined class.
        """

        @dynamic_foo()
        def does_nothing():
            return

        @dynamic_foo()
        def slowly_returns_unpickleable():
            """
            Returns a complex, dynamically defined variable
            """
            sleep(0.1)
            inside_variable = does_nothing()
            inside_variable.result = "it was an inside job!"
            return inside_variable

        dynamic_dynamic = slowly_returns_unpickleable()
        executor = PyMPISingleTaskExecutor()
        fs = executor.submit(dynamic_dynamic.run)
        self.assertIsInstance(
            fs.result(),
            Foo,
        )
        self.assertEqual(fs.result().result, "it was an inside job!")
"""
The purpose of these tests is the check executor behaviour when the python objects
are dynamically generated.
This is a special (and rather difficult) case for serializing objects which cannot
be pickled using the standard pickle module, and thus poses a relatively thorough test
for the general un-pickle-able case.
"""
from functools import partialmethod
from time import sleep
import unittest

from pympipool.mpi.executor import PyMPIExecutor as Executor


class Foo:
    """
    A base class to be dynamically modified for putting an executor/serializer through
    its paces.
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


class TestDynamicallyDefinedObjects(unittest.TestCase):
    def test_dynamic_args(self):
        """
        We should be able to use a dynamically defined return value.
        """

        @dynamic_foo()
        def does_nothing():
            return

        @dynamic_foo()
        def slowly_returns_dynamic(dynamic_arg):
            """
            Returns a complex, dynamically defined variable
            """
            sleep(0.1)
            dynamic_arg.result = "input updated"
            return dynamic_arg

        dynamic_dynamic = slowly_returns_dynamic()
        executor = Executor()
        dynamic_object = does_nothing()
        fs = executor.submit(dynamic_dynamic.run, dynamic_object)
        self.assertEqual(fs.result().result, "input updated")

    def test_dynamic_callable(self):
        """
        We should be able to use a dynamic callable -- in this case, a method of
        a dynamically defined class.
        """
        fortytwo = 42  # No magic numbers; we use it in a couple places so give it a var

        @dynamic_foo()
        def slowly_returns_42():
            sleep(0.1)
            return fortytwo

        dynamic_42 = slowly_returns_42()  # Instantiate the dynamically defined class
        self.assertIsInstance(
            dynamic_42,
            Foo,
            msg="Just a sanity check that the test is set up right"
        )
        self.assertIsNone(
            dynamic_42.result,
            msg="Just a sanity check that the test is set up right"
        )
        executor = Executor()
        fs = executor.submit(dynamic_42.run)
        fs.add_done_callback(dynamic_42.process_result)
        self.assertFalse(fs.done(), msg="Should be running on the executor")
        self.assertEqual(fortytwo, fs.result(), msg="Future must complete")
        self.assertEqual(fortytwo, dynamic_42.result, msg="Callback must get called")

    def test_exception(self):
        @dynamic_foo()
        def raise_error():
            raise RuntimeError

        re = raise_error()
        executor = Executor()
        fs = executor.submit(re.run)
        with self.assertRaises(RuntimeError):
            fs.result()

    def test_dynamic_return(self):
        """
        We should be able to use a dynamic return value -- in this case, a
        method of a dynamically defined class.
        """

        @dynamic_foo()
        def does_nothing():
            return

        @dynamic_foo()
        def slowly_returns_dynamic():
            """
            Returns a complex, dynamically defined variable
            """
            sleep(0.1)
            inside_variable = does_nothing()
            inside_variable.result = "it was an inside job!"
            return inside_variable

        dynamic_dynamic = slowly_returns_dynamic()
        executor = Executor()
        fs = executor.submit(dynamic_dynamic.run)
        self.assertIsInstance(
            fs.result(),
            Foo,
        )
        self.assertEqual(fs.result().result, "it was an inside job!")

    def test_timeout(self):
        fortytwo = 42

        @dynamic_foo()
        def slow():
            sleep(0.1)
            return fortytwo

        f = slow()
        executor = Executor()
        fs = executor.submit(f.run)
        self.assertEqual(
            fs.result(timeout=30),
            fortytwo,
            msg="waiting long enough should get the result"
        )

        with self.assertRaises(TimeoutError):
            fs = executor.submit(f.run)
            fs.result(timeout=0.0001)
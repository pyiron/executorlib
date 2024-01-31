"""
The purpose of these tests is the check executor behaviour when the python objects
are dynamically generated.
This is a special (and rather difficult) case for serializing objects which cannot
be pickled using the standard pickle module, and thus poses a relatively thorough test
for the general un-pickle-able case.
"""
from concurrent.futures._base import TimeoutError as cfbTimeoutError
from functools import partialmethod
from time import sleep
import unittest

from pympipool import Executor


class Foo:
    """
    A base class to be dynamically modified for putting an executor/serializer through
    its paces.
    """
    def __init__(self, fnc: callable):
        self.fnc = fnc
        self.result = None
        self.running = False

    @property
    def run(self):
        self.running = True
        return self.fnc

    def process_result(self, future):
        self.result = future.result()
        self.running = False


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
    def test_args(self):
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
            dynamic_arg.attribute_on_dynamic = "attribute updated"
            return dynamic_arg

        dynamic_dynamic = slowly_returns_dynamic()
        executor = Executor(hostname_localhost=True)
        dynamic_object = does_nothing()
        fs = executor.submit(dynamic_dynamic.run, dynamic_object)
        self.assertEqual(
            fs.result().attribute_on_dynamic,
            "attribute updated",
            msg="The submit callable should have modified the mutable, dynamically "
                "defined object with a new attribute."
        )

    def test_callable(self):
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
        executor = Executor(hostname_localhost=True)
        fs = executor.submit(dynamic_42.run)
        fs.add_done_callback(dynamic_42.process_result)
        self.assertFalse(
            fs.done(),
            msg="The submit callable sleeps long enough that we expect to still be "
                "running here -- did something fail to get submit to an executor??"
        )
        self.assertEqual(
            fortytwo,
            fs.result(),
            msg="The future is expected to behave as usual"
        )
        self.assertEqual(
            fortytwo,
            dynamic_42.result,
            msg="The callback modifies its object and should run by the time the result"
                "is available -- did it fail to get called?"
        )

    def test_callback(self):
        """Make sure the callback methods can modify their owners"""

        @dynamic_foo()
        def returns_42():
            return 42

        dynamic_42 = returns_42()
        self.assertFalse(
            dynamic_42.running,
            msg="Sanity check that the test starts in the expected condition"
        )
        executor = Executor(hostname_localhost=True)
        fs = executor.submit(dynamic_42.run)
        fs.add_done_callback(dynamic_42.process_result)
        self.assertTrue(
            dynamic_42.running,
            msg="Submit method need to be able to modify their owners"
        )
        fs.result()  # Wait for the process to finish
        self.assertFalse(
            dynamic_42.running,
            msg="Callback methods need to be able to modify their owners"
        )

    def test_exception(self):
        """
        Exceptions from dynamically defined callables should get cleanly raised.
        """

        @dynamic_foo()
        def raise_error():
            raise RuntimeError

        re = raise_error()
        executor = Executor(hostname_localhost=True)
        fs = executor.submit(re.run)
        with self.assertRaises(
            RuntimeError,
            msg="The callable just raises an error -- this should get shown to the user"
        ):
            fs.result()

    def test_return(self):
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
        executor = Executor(hostname_localhost=True)
        fs = executor.submit(dynamic_dynamic.run)
        self.assertIsInstance(
            fs.result(),
            Foo,
            msg="Just a sanity check that we're getting the right type of dynamically "
                "defined type of object"
        )
        self.assertEqual(
            fs.result().result,
            "it was an inside job!",
            msg="The submit callable modifies the object that owns it, and this should"
                "be reflected in the main process after deserialziation"
        )

    def test_timeout(self):
        """
        Timeouts for dynamically defined callables should be handled ok.
        """

        fortytwo = 42

        @dynamic_foo()
        def slow():
            sleep(0.1)
            return fortytwo

        f = slow()
        executor = Executor(hostname_localhost=True)
        fs = executor.submit(f.run)
        self.assertEqual(
            fs.result(timeout=30),
            fortytwo,
            msg="waiting long enough should get the result"
        )

        with self.assertRaises(
            (TimeoutError, cfbTimeoutError),
            msg="With a timeout time smaller than our submit callable's sleep time, "
                "we had better get an exception!"
        ):
            fs = executor.submit(f.run)
            fs.result(timeout=0.0001)

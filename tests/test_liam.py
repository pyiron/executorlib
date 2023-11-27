from unittest import TestCase
from pympipool import PyMPIExecutor


def add_two(x):
    return x + 2


class TestPyMPIExecutor(TestCase):

    def test_local(self):
        # Works perfectly
        def add_one(x):
            return x + 1

        class Foo:
            def add(self, x):
                return add_one(x)

        foo = Foo()
        executor = PyMPIExecutor()
        fs = executor.submit(foo.add, 1)
        self.assertEqual(2, fs.result())

    def test_semi_local(self):
        # Hangs
        # CI logs make it look like cloudpickle is trying to import from this module,
        # but can't find it
        class Foo:
            def add(self, x):
                return add_two(x)

        foo = Foo()
        executor = PyMPIExecutor()
        fs = executor.submit(foo.add, 1)
        self.assertEqual(3, fs.result())
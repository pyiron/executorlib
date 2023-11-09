import numpy as np
import unittest
from time import sleep
from pympipool.mpi.executor import PyMPISingleTaskExecutor
from concurrent.futures import Future


def calc(i):
    return np.array(i**2)


class TestFuture(unittest.TestCase):
    def test_pool_serial(self):
        with PyMPISingleTaskExecutor(cores=1) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), np.array(4))

    def test_pool_serial_multi_core(self):
        with PyMPISingleTaskExecutor(cores=2) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), [np.array(4), np.array(4)])

    def test_independence_from_executor(self):
        """
        Ensure that futures are able to live on after the executor gets garbage
        collected.
        """

        with self.subTest("From the main process"):
            mutable = []

            def slow_callable():
                from time import sleep
                sleep(1)
                return True

            def callback(future):
                mutable.append("Called back")

            def submit():
                # Executor only exists in this scope and can get garbage collected after
                # this function is exits
                future = PyMPISingleTaskExecutor().submit(slow_callable)
                future.add_done_callback(callback)
                return future

            self.assertListEqual(
                [],
                mutable,
                msg="Sanity check that test is starting in the expected condition"
            )
            future = submit()

            self.assertFalse(
                future.done(),
                msg="The submit function is slow, it should be running still"
            )
            self.assertListEqual(
                [],
                mutable,
                msg="While running, the mutable should not have been impacted by the "
                    "callback"
            )
            future.result()  # Wait for the calculation to finish
            self.assertListEqual(
                ["Called back"],
                mutable,
                msg="After completion, the callback should modify the mutable data"
            )

        with self.subTest("From inside a class"):
            class Foo:
                def __init__(self):
                    self.running = False

                def run(self):
                    self.running = True

                    future = PyMPISingleTaskExecutor().submit(self.return_42)
                    future.add_done_callback(self.finished)

                    return future

                def return_42(self):
                    from time import sleep
                    sleep(1)
                    return 42

                def finished(self, future):
                    self.running = False

            foo = Foo()
            self.assertFalse(
                foo.running,
                msg="Sanity check that the test starts in the expected condition"
            )
            fs = foo.run()
            self.assertTrue(
                foo.running,
                msg="We should be able to exit the run method before the task completes"
            )
            fs.result()  # Wait for completion
            self.assertFalse(
                foo.running,
                msg="After task completion, we expect the callback to modify the class"
            )
import numpy as np
import unittest
from time import sleep
from pympipool import Executor
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


class TestFuture(unittest.TestCase):
    def test_pool_serial(self):
        with Executor(cores=1) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), np.array(4))

    def test_pool_serial_multi_core(self):
        with Executor(cores=2) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), [np.array(4), np.array(4)])

import numpy as np
import unittest
from time import sleep
from pympipool import Pool
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


class TestFuture(unittest.TestCase):
    def test_pool_serial(self):
        with Pool(cores=1) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
            p.update()
        self.assertTrue(output.done())
        self.assertEqual(output.result(), 4)

    def test_pool_serial_multi_core(self):
        with Pool(cores=2) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
            p.update()
        self.assertTrue(output.done())
        self.assertEqual(output.result(), 4)
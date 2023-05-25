import numpy as np
import unittest
from pympipool import Pool
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


class TestPool(unittest.TestCase):
    def test_pool_serial(self):
        with Pool(cores=1) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            p.update()
        self.assertTrue(output.done())
        self.assertEqual(output.result(), 4)

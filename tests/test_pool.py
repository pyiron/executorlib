import numpy as np
import unittest
from pympipool import Pool


def calc(i):
    return np.array(i ** 2)


def calc_none(i):
    return None


def calc_error(i):
    raise ValueError("calc_error value error")


class TestPool(unittest.TestCase):
    def test_pool_serial(self):
        with Pool(cores=1) as p:
            output = p.map(function=calc, lst=[1, 2, 3, 4])
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4)
        self.assertEqual(output[2], 9)
        self.assertEqual(output[3], 16)

    def test_pool_parallel(self):
        with Pool(cores=2) as p:
            output = p.map(function=calc, lst=[1, 2, 3, 4])
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4)
        self.assertEqual(output[2], 9)
        self.assertEqual(output[3], 16)

    def test_pool_none(self):
        with Pool(cores=2) as p:
            output = p.map(function=calc_none, lst=[1, 2, 3, 4])
        self.assertIsNone(output)

    def test_pool_error(self):
        with self.assertRaises(ValueError):
            with Pool(cores=2) as p:
                p.map(function=calc_error, lst=[1, 2, 3, 4])

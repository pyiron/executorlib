import unittest
from pympipool import Pool


def calc(i):
    import numpy as np
    return np.array(i ** 2)


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

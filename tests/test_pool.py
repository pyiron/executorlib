import numpy as np
import unittest
from pympipool import Pool


def calc(i):
    return np.array(i ** 2)


def calc_add(i, j):
    return i + j


def calc_none(i):
    return None


def calc_error_value_error(i):
    raise ValueError("calc_error value error")


def calc_error_type_error(i):
    raise TypeError("calc_error value error")


class TestPool(unittest.TestCase):
    def test_map_serial(self):
        with Pool(max_workers=1) as p:
            output = p.map(func=calc, iterable=[1, 2, 3, 4])
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4)
        self.assertEqual(output[2], 9)
        self.assertEqual(output[3], 16)

    def test_starmap_serial(self):
        with Pool(max_workers=1) as p:
            output = p.starmap(func=calc_add, iterable=[[1, 2], [3, 4]])
        self.assertEqual(output[0], 3)
        self.assertEqual(output[1], 7)

    def test_map_parallel(self):
        with Pool(max_workers=2) as p:
            output = p.map(func=calc, iterable=[1, 2, 3, 4])
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4)
        self.assertEqual(output[2], 9)
        self.assertEqual(output[3], 16)

    def test_starmap_parallel(self):
        with Pool(max_workers=2) as p:
            output = p.starmap(func=calc_add, iterable=[[1, 2], [3, 4]])
        self.assertEqual(output[0], 3)
        self.assertEqual(output[1], 7)

    def test_pool_none(self):
        with Pool(max_workers=2) as p:
            output = p.map(func=calc_none, iterable=[1, 2, 3, 4])
        self.assertEqual(output, [None, None, None, None])

    def test_pool_error(self):
        with self.assertRaises(ValueError):
            with Pool(max_workers=2) as p:
                p.map(func=calc_error_value_error, iterable=[1, 2, 3, 4])
        with self.assertRaises(TypeError):
            with Pool(max_workers=2) as p:
                p.map(func=calc_error_type_error, iterable=[1, 2, 3, 4])

    def test_shutdown(self):
        p = Pool(max_workers=1)
        output = p.map(func=calc, iterable=[1, 2, 3, 4])
        p.shutdown(wait=True)
        p.shutdown(wait=False)
        self.assertEqual(output[0], 1)
        self.assertEqual(output[1], 4)
        self.assertEqual(output[2], 9)
        self.assertEqual(output[3], 16)

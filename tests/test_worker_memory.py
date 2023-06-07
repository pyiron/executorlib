import unittest
import numpy as np
from pympipool import Worker


def get_global(memory=None):
    return memory


def set_global():
    return np.array([5])


class TestWorkerMemory(unittest.TestCase):
    def test_internal_memory(self):
        with Worker(cores=1, init_function=set_global) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

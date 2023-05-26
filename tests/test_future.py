import numpy as np
import unittest
from time import sleep
from pympipool import Pool
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


def get_ranks(input_parameter, comm=None):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    if comm is not None:
        size_new = comm.Get_size()
        rank_new = comm.Get_rank()
    else:
        size_new = 0
        rank_new = 0
    return size, rank, size_new, rank_new, input_parameter


class TestPool(unittest.TestCase):
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

    # def test_pool_parallel(self):
    #     with Pool(cores=2, cores_per_task=2) as p:
    #         output = p.submit(get_ranks, i=2)
    #         self.assertTrue(isinstance(output, Future))
    #         self.assertFalse(output.done())
    #         p.update()
    #     sleep(10)
    #     self.assertTrue(output.done())
    #     self.assertEqual(output.result(), [2, 0, 2, 0, 2])

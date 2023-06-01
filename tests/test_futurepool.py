import numpy as np
import unittest
from queue import Queue
from time import sleep
from pympipool import PoolFuture, _execute_tasks, _cloudpickle_update
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


def mpi_funct(i):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


class TestFuturePool(unittest.TestCase):
    def test_pool_serial(self):
        with PoolFuture(cores=1) as p:
            output = p.submit(calc, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), np.array(4))

    def test_pool_serial_multi_core(self):
        with PoolFuture(cores=2) as p:
            output = p.submit(mpi_funct, i=2)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
        self.assertTrue(output.done())
        self.assertEqual(output.result(), [(2, 2, 0), (2, 2, 1)])

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"f": calc, 'a': (), "k": {"i": 2}, "l": f})
        q.put({"c": "close"})
        _cloudpickle_update(ind=1)
        _execute_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            enable_flux_backend=False
        )
        self.assertEqual(f.result(), np.array(4))

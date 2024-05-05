import unittest

from pympipool import Executor
from executorlib_core.base import cloudpickle_register


def calc(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial(self):
        with Executor(
            max_cores=2, hostname_localhost=True, backend="mpi", block_allocation=True
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_single(self):
        with Executor(
            max_cores=1, hostname_localhost=True, backend="mpi", block_allocation=True
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_parallel(self):
        with Executor(
            max_cores=2,
            cores_per_worker=2,
            hostname_localhost=True,
            backend="mpi",
            block_allocation=True,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_errors(self):
        with self.assertRaises(TypeError):
            Executor(
                max_cores=1,
                cores_per_worker=1,
                threads_per_core=2,
                hostname_localhost=True,
                backend="mpi",
            )
        with self.assertRaises(TypeError):
            Executor(
                max_cores=1,
                cores_per_worker=1,
                gpus_per_worker=1,
                hostname_localhost=True,
                backend="mpi",
            )

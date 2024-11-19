import importlib.util
import shutil
import time
import unittest

from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def calc(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def mpi_funct_sleep(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    time.sleep(i)
    return i, size, rank


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial(self):
        with Executor(max_cores=2, backend="local", block_allocation=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_single(self):
        with Executor(max_cores=1, backend="local", block_allocation=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_oversubscribe(self):
        with self.assertRaises(ValueError):
            with Executor(max_cores=1, backend="local", block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                fs_1 = exe.submit(calc, 1, resource_dict={"cores": 2})

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_meta_executor_parallel(self):
        with Executor(
            max_workers=2,
            resource_dict={"cores": 2},
            backend="local",
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
                resource_dict={"cores": 1, "gpus_per_core": 1},
                backend="local",
            )


class TestExecutorBackendCache(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("./cache")

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_meta_executor_parallel_cache(self):
        with Executor(
            max_workers=2,
            resource_dict={"cores": 2},
            backend="local",
            block_allocation=True,
            cache_directory="./cache",
        ) as exe:
            cloudpickle_register(ind=1)
            time_1 = time.time()
            fs_1 = exe.submit(mpi_funct_sleep, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())
            time_2 = time.time()
            self.assertTrue(time_2 - time_1 > 1)
            time_3 = time.time()
            fs_2 = exe.submit(mpi_funct_sleep, 1)
            self.assertEqual(fs_2.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_2.done())
            time_4 = time.time()
            self.assertTrue(time_3 - time_4 < 1)

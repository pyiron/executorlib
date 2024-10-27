import importlib.util
import os
import shutil
import unittest


try:
    from executorlib import FileExecutor
    from executorlib.cache.shared import execute_tasks_h5, execute_in_subprocess

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


@unittest.skipIf(
    skip_h5io_test or skip_mpi4py_test,
    "h5io or mpi4py are not installed, so the h5io and mpi4py tests are skipped.",
)
class TestCacheExecutorMPI(unittest.TestCase):
    def test_executor(self):
        with FileExecutor(cores_per_worker=2) as exe:
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")

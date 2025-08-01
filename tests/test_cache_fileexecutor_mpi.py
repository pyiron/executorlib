import importlib.util
import shutil
import unittest


try:
    from executorlib.task_scheduler.file.task_scheduler import FileTaskScheduler
    from executorlib.task_scheduler.file.subprocess_spawner import execute_in_subprocess

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


@unittest.skipIf(
    skip_h5py_test or skip_mpi4py_test,
    "h5py or mpi4py are not installed, so the h5py and mpi4py tests are skipped.",
)
class TestCacheExecutorMPI(unittest.TestCase):
    def test_executor(self):
        with FileTaskScheduler(
            resource_dict={"cores": 2}, execute_function=execute_in_subprocess
        ) as exe:
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())

    def test_batched_error(self):
        with self.assertRaises(NotImplementedError):
            with FileTaskScheduler() as exe:
                exe.batched(iterable=[], n=2)

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)

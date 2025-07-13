import os
import importlib
import unittest
import shutil

from executorlib import SlurmClusterExecutor
from executorlib.standalone.serialize import cloudpickle_register

if shutil.which("srun") is not None:
    skip_slurm_test = False
else:
    skip_slurm_test = True

skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


@unittest.skipIf(
    skip_slurm_test or skip_mpi4py_test,
    "h5py or mpi4py or SLRUM are not installed, so the h5py, slurm and mpi4py tests are skipped.",
)
class TestCacheExecutorPysqa(unittest.TestCase):
    def test_executor(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
            terminate_tasks_on_shutdown=False,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)
            self.assertTrue(fs1.done())

    def test_executor_no_cwd(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2},
            block_allocation=False,
            cache_directory="executorlib_cache",
            terminate_tasks_on_shutdown=True,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 2)
            self.assertTrue(fs1.done())

    def test_executor_existing_files(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)
            for file_name in os.listdir("executorlib_cache"):
                file_path = os.path.join("executorlib_cache", file_name )
                os.remove(file_path)
                if ".h5" in file_path:
                    task_key = file_path[:-5] + "_i.h5"
                    dump(file_name=task_key, data_dict={"a": 1})

        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)

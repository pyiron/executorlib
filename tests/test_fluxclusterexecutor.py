import os
import importlib
import unittest
import shutil

from executorlib import FluxClusterExecutor
from executorlib.standalone.serialize import cloudpickle_register

try:
    import flux.job
    from executorlib.task_scheduler.file.hdf import dump

    skip_flux_test = "FLUX_URI" not in os.environ
    pmi = os.environ.get("EXECUTORLIB_PMIX", None)
except ImportError:
    skip_flux_test = True


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


@unittest.skipIf(
    skip_flux_test or skip_mpi4py_test,
    "h5py or mpi4py or flux are not installed, so the h5py, flux and mpi4py tests are skipped.",
)
class TestCacheExecutorPysqa(unittest.TestCase):
    def test_executor(self):
        with FluxClusterExecutor(
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
        with FluxClusterExecutor(
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
        with FluxClusterExecutor(
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

        with FluxClusterExecutor(
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

import os
import importlib
import unittest
import shutil

from executorlib import Executor
from executorlib.standalone.serialize import cloudpickle_register

try:
    import flux.job

    skip_flux_test = "FLUX_URI" not in os.environ
    pmi = os.environ.get("PYMPIPOOL_PMIX", None)
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
        with Executor(
            backend="flux_submission",
            resource_dict={"cores": 2, "cwd": "cache"},
            block_allocation=False,
            cache_directory="cache",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")

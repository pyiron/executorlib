import os
import unittest

import numpy as np

from executorlib import Executor


try:
    import flux.job
    from executorlib.interactive.flux import FluxPythonSpawner

    skip_flux_test = "FLUX_URI" not in os.environ
    pmi = os.environ.get("PYMPIPOOL_PMIX", None)
except ImportError:
    skip_flux_test = True


def calc(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def get_global(memory=None):
    return memory


def set_global():
    return {"memory": np.array([5])}


@unittest.skipIf(
    skip_flux_test, "Flux is not installed, so the flux tests are skipped."
)
class TestFluxBackend(unittest.TestCase):
    def setUp(self):
        self.executor = flux.job.FluxExecutor()

    def test_flux_executor_serial(self):
        with Executor(
            max_cores=2,
            flux_executor=self.executor,
            backend="flux_allocation",
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_threads(self):
        with Executor(
            max_cores=1,
            resource_dict={"threads_per_core": 2},
            flux_executor=self.executor,
            backend="flux_allocation",
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_parallel(self):
        with Executor(
            max_cores=2,
            resource_dict={"cores": 2},
            flux_executor=self.executor,
            backend="flux_allocation",
            block_allocation=True,
            flux_executor_pmi_mode=pmi,
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_single_task(self):
        with Executor(
            max_cores=2,
            resource_dict={"cores": 2},
            flux_executor=self.executor,
            backend="flux_allocation",
            block_allocation=True,
            flux_executor_pmi_mode=pmi,
        ) as p:
            output = p.map(mpi_funct, [1, 2, 3])
        self.assertEqual(
            list(output),
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_internal_memory(self):
        with Executor(
            max_cores=1,
            resource_dict={"cores": 1},
            init_function=set_global,
            flux_executor=self.executor,
            backend="flux_allocation",
            block_allocation=True,
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

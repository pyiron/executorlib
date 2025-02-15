import os
import unittest

import numpy as np

from executorlib import FluxJobExecutor


try:
    import flux.job
    from executorlib.interactive.fluxspawner import FluxPythonSpawner

    skip_flux_test = "FLUX_URI" not in os.environ
    pmi = os.environ.get("EXECUTORLIB_PMIX", None)
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
        with FluxJobExecutor(
            max_cores=2,
            flux_executor=self.executor,
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_serial_no_depencies(self):
        with FluxJobExecutor(
            max_cores=2,
            flux_executor=self.executor,
            block_allocation=True,
            disable_dependencies=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_threads(self):
        with FluxJobExecutor(
            max_cores=1,
            resource_dict={"threads_per_core": 2},
            flux_executor=self.executor,
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_parallel(self):
        with FluxJobExecutor(
            max_cores=2,
            resource_dict={"cores": 2},
            flux_executor=self.executor,
            block_allocation=True,
            flux_executor_pmi_mode=pmi,
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_single_task(self):
        with FluxJobExecutor(
            max_cores=2,
            resource_dict={"cores": 2},
            flux_executor=self.executor,
            block_allocation=True,
            flux_executor_pmi_mode=pmi,
        ) as p:
            output = p.map(mpi_funct, [1, 2, 3])
        self.assertEqual(
            list(output),
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_output_files_cwd(self):
        dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.makedirs(dirname, exist_ok=True)
        file_stdout = os.path.join(dirname, "flux.out")
        file_stderr = os.path.join(dirname, "flux.err")
        with FluxJobExecutor(
            max_cores=1,
            resource_dict={"cores": 1, "cwd": dirname},
            flux_executor=self.executor,
            block_allocation=True,
            flux_log_files=True,
        ) as p:
            output = p.map(calc, [1, 2, 3])
        self.assertEqual(
            list(output),
            [1, 2, 3],
        )
        self.assertTrue(os.path.exists(file_stdout))
        self.assertTrue(os.path.exists(file_stderr))
        os.remove(file_stdout)
        os.remove(file_stderr)

    def test_output_files_abs(self):
        file_stdout = os.path.abspath("flux.out")
        file_stderr = os.path.abspath("flux.err")
        with FluxJobExecutor(
            max_cores=1,
            resource_dict={"cores": 1},
            flux_executor=self.executor,
            block_allocation=True,
            flux_log_files=True,
        ) as p:
            output = p.map(calc, [1, 2, 3])
        self.assertEqual(
            list(output),
            [1, 2, 3],
        )
        self.assertTrue(os.path.exists(file_stdout))
        self.assertTrue(os.path.exists(file_stderr))
        os.remove(file_stdout)
        os.remove(file_stderr)

    def test_internal_memory(self):
        with FluxJobExecutor(
            max_cores=1,
            resource_dict={"cores": 1},
            init_function=set_global,
            flux_executor=self.executor,
            block_allocation=True,
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

    def test_validate_max_workers(self):
        with self.assertRaises(ValueError):
            FluxJobExecutor(
                max_workers=10,
                resource_dict={"cores": 10, "threads_per_core": 10},
                flux_executor=self.executor,
                block_allocation=True,
            )

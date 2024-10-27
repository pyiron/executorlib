from concurrent.futures import Future
import os
from queue import Queue
import unittest

import numpy as np

from executorlib.interactive.shared import InteractiveExecutor
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.interactive.shared import execute_parallel_tasks


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
class TestFlux(unittest.TestCase):
    def setUp(self):
        self.flux_executor = flux.job.FluxExecutor()

    def test_flux_executor_serial(self):
        with InteractiveExecutor(
            max_workers=2,
            executor_kwargs={"flux_executor": self.flux_executor},
            spawner=FluxPythonSpawner,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_threads(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "flux_executor": self.flux_executor,
                "threads_per_core": 2,
            },
            spawner=FluxPythonSpawner,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_parallel(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "flux_executor": self.flux_executor,
                "cores": 2,
                "flux_executor_pmi_mode": pmi,
            },
            spawner=FluxPythonSpawner,
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_single_task(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "flux_executor": self.flux_executor,
                "cores": 2,
                "flux_executor_pmi_mode": pmi,
            },
            spawner=FluxPythonSpawner,
        ) as p:
            output = p.map(mpi_funct, [1, 2, 3])
        self.assertEqual(
            list(output),
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc, "args": (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            flux_executor=self.flux_executor,
            spawner=FluxPythonSpawner,
        )
        self.assertEqual(f.result(), 2)
        q.join()

    def test_execute_task_threads(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc, "args": (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            threads_per_core=1,
            flux_executor=self.flux_executor,
            spawner=FluxPythonSpawner,
        )
        self.assertEqual(f.result(), 2)
        q.join()

    def test_internal_memory(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "flux_executor": self.flux_executor,
                "cores": 1,
                "init_function": set_global,
            },
            spawner=FluxPythonSpawner,
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

    def test_interface_exception(self):
        with self.assertRaises(ValueError):
            flux_interface = FluxPythonSpawner(
                flux_executor=self.flux_executor, openmpi_oversubscribe=True
            )
            flux_interface.bootup(command_lst=[])

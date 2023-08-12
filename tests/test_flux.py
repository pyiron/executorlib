from concurrent.futures import Future
from queue import Queue

import numpy as np
import unittest

from pympipool.shared.executorbase import cloudpickle_register, executor_broker, execute_parallel_tasks


try:
    import flux.job
    from pympipool.flux.executor import (
        PyFluxExecutor,
        PyFluxSingleTaskExecutor,
        FluxPythonInterface,
    )

    skip_flux_test = False
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
        self.executor = flux.job.FluxExecutor()

    def test_flux_executor_serial(self):
        with PyFluxExecutor(max_workers=2, executor=self.executor) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_threads(self):
        with PyFluxExecutor(
            max_workers=1, threads_per_core=2, executor=self.executor
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_flux_executor_parallel(self):
        with PyFluxExecutor(
            max_workers=1, cores_per_worker=2, executor=self.executor
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_single_task(self):
        with PyFluxSingleTaskExecutor(cores=2, executor=self.executor) as p:
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
            executor=self.executor,
            interface_class=FluxPythonInterface,
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
            executor=self.executor,
            interface_class=FluxPythonInterface,
        )
        self.assertEqual(f.result(), 2)
        q.join()

    def test_internal_memory(self):
        with PyFluxSingleTaskExecutor(
            cores=1, init_function=set_global, executor=self.executor
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

    def test_executor_broker(self):
        q = Queue()
        f = Future()
        q.put({"fn": calc, "args": (1,), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        executor_broker(
            future_queue=q,
            max_workers=1,
            executor=self.executor,
            executor_class=PyFluxSingleTaskExecutor,
        )
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 1)
        q.join()

    def test_executor_broker_threads(self):
        q = Queue()
        f = Future()
        q.put({"fn": calc, "args": (1,), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        executor_broker(
            future_queue=q,
            max_workers=1,
            threads_per_core=2,
            executor=self.executor,
            executor_class=PyFluxSingleTaskExecutor,
        )
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 1)
        q.join()

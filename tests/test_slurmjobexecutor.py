import os
import shutil
import unittest

import numpy as np

from executorlib import SlurmJobExecutor


if shutil.which("srun") is not None:
    skip_slurm_test = False
else:
    skip_slurm_test = True


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
    skip_slurm_test, "Slurm is not installed, so the Slurm tests are skipped."
)
class TestSlurmBackend(unittest.TestCase):
    def test_slurm_executor_serial(self):
        with SlurmJobExecutor(
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_slurm_executor_serial_no_depencies(self):
        with SlurmJobExecutor(
            block_allocation=True,
            disable_dependencies=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_slurm_executor_threads(self):
        with SlurmJobExecutor(
            resource_dict={"threads_per_core": 2},
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_slurm_executor_parallel(self):
        with SlurmJobExecutor(
            resource_dict={"cores": 2},
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_single_task(self):
        with SlurmJobExecutor(
            resource_dict={"cores": 2},
            block_allocation=True,
        ) as p:
            output = p.map(mpi_funct, [1, 2, 3])
        self.assertEqual(
            list(output),
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_internal_memory(self):
        with SlurmJobExecutor(
            resource_dict={"cores": 1},
            init_function=set_global,
            block_allocation=True,
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

import shutil
import unittest

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


@unittest.skipIf(
    skip_slurm_test, "Slurm is not installed, so the Slurm tests are skipped."
)
class TestSlurmBackend(unittest.TestCase):
    def test_slurm_executor_serial(self):
        with SlurmJobExecutor(resource_dict={"slurm_cmd_args": ["--mpi=pmi2"]}) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_slurm_executor_parallel(self):
        with SlurmJobExecutor(
            max_cores=2,
            resource_dict={"cores": 2, "slurm_cmd_args": ["--mpi=pmi2"]},
            block_allocation=True,
        ) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())
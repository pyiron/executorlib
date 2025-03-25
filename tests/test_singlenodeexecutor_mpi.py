import os
import importlib.util
import shutil
import time
import unittest

from executorlib import SingleNodeExecutor, SlurmJobExecutor
from executorlib.standalone.serialize import cloudpickle_register


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def calc(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def mpi_funct_sleep(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    time.sleep(i)
    return i, size, rank


class TestExecutorBackend(unittest.TestCase):
    def test_meta_executor_serial(self):
        with SingleNodeExecutor(max_cores=2, block_allocation=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_single(self):
        with SingleNodeExecutor(max_cores=1, block_allocation=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_oversubscribe(self):
        with self.assertRaises(ValueError):
            with SingleNodeExecutor(max_cores=1, block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                fs_1 = exe.submit(calc, 1, resource_dict={"cores": 2})

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_meta_executor_parallel(self):
        with SingleNodeExecutor(
            max_workers=2,
            resource_dict={"cores": 2},
            block_allocation=True,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_errors(self):
        with self.assertRaises(TypeError):
            SingleNodeExecutor(
                max_cores=1,
                resource_dict={"cores": 1, "gpus_per_core": 1},
            )


class TestExecutorBackendCache(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("./cache")

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_meta_executor_parallel_cache(self):
        with SingleNodeExecutor(
            max_workers=2,
            resource_dict={"cores": 2},
            block_allocation=True,
            cache_directory="./cache",
        ) as exe:
            cloudpickle_register(ind=1)
            time_1 = time.time()
            fs_1 = exe.submit(mpi_funct_sleep, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())
            time_2 = time.time()
            self.assertTrue(time_2 - time_1 > 1)
            time_3 = time.time()
            fs_2 = exe.submit(mpi_funct_sleep, 1)
            self.assertEqual(fs_2.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_2.done())
            time_4 = time.time()
            self.assertTrue(time_3 - time_4 < 1)


class TestWorkingDirectory(unittest.TestCase):
    def test_output_files_cwd(self):
        dirname = os.path.abspath(os.path.dirname(__file__))
        os.makedirs(dirname, exist_ok=True)
        with SingleNodeExecutor(
            max_cores=1,
            resource_dict={"cores": 1, "cwd": dirname},
            block_allocation=True,
        ) as p:
            output = p.map(calc, [1, 2, 3])
        self.assertEqual(
            list(output),
            [1, 2, 3],
        )


class TestSLURMExecutor(unittest.TestCase):
    def test_validate_max_workers(self):
        os.environ["SLURM_NTASKS"] = "6"
        os.environ["SLURM_CPUS_PER_TASK"] = "4"
        with self.assertRaises(ValueError):
            SlurmJobExecutor(
                max_workers=10,
                resource_dict={"cores": 10, "threads_per_core": 10},
                block_allocation=True,
            )

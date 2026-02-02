import os
import importlib
import unittest
import shutil
from time import sleep

from executorlib import FluxClusterExecutor
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.standalone.command import get_cache_execute_command

try:
    import flux.job
    from executorlib import terminate_tasks_in_cache
    from executorlib.standalone.hdf import dump
    from executorlib.task_scheduler.file.spawner_pysqa import execute_with_pysqa
    from executorlib.standalone.scheduler import terminate_with_pysqa
    from executorlib.task_scheduler.interactive.spawner_pysqa import PysqaSpawner

    skip_flux_test = "FLUX_URI" not in os.environ
    pmi = os.environ.get("EXECUTORLIB_PMIX", None)
except ImportError:
    skip_flux_test = True


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def echo(i):
    sleep(1)
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def stop_function():
    return True


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
            pmi_mode=pmi,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)
            self.assertTrue(fs1.done())

    def test_executor_blockallocation(self):
        with FluxClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache"},
            block_allocation=True,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
            max_workers=1,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 2)
            self.assertTrue(fs1.done())

    def test_executor_blockallocation_echo(self):
        with FluxClusterExecutor(
            resource_dict={"cores": 1, "cwd": "executorlib_cache"},
            block_allocation=True,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
            max_workers=2,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(echo, 1)
            fs2 = exe.submit(echo, 2)
            self.assertFalse(fs1.done())
            self.assertFalse(fs2.done())
            self.assertEqual(fs1.result(), 1)
            self.assertEqual(fs2.result(), 2)
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)
            self.assertTrue(fs1.done())
            self.assertTrue(fs2.done())

    def test_executor_cancel_future_on_shutdown(self):
        with FluxClusterExecutor(
            resource_dict={"cores": 1, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
            wait=False,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(echo, 1)
            self.assertFalse(fs1.done())
        self.assertTrue(fs1.cancelled())
        sleep(2)
        with FluxClusterExecutor(
            resource_dict={"cores": 1, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
            wait=False,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(echo, 1)
            self.assertEqual(fs1.result(), 1)
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)
            self.assertTrue(fs1.done())

    def test_executor_no_cwd(self):
        with FluxClusterExecutor(
            resource_dict={"cores": 2},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 2)
            self.assertTrue(fs1.done())

    def test_pysqa_interface(self):
        queue_id = execute_with_pysqa(
            command=get_cache_execute_command(
                file_name="test_i.h5",
                cores=1,
            ),
            file_name="test_i.h5",
            data_dict={"fn": sleep, "args": (10,)},
            resource_dict={"cores": 1},
            cache_directory="executorlib_cache",
            backend="flux"
        )
        self.assertIsNone(terminate_with_pysqa(queue_id=queue_id, backend="flux"))

    def test_executor_existing_files(self):
        with FluxClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache"},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode=pmi,
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
            pmi_mode=pmi,
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())
            self.assertEqual(len(os.listdir("executorlib_cache")), 4)

    def test_terminate_tasks_in_cache(self):
        file = os.path.join("executorlib_cache", "test_i.h5")
        dump(file_name=file, data_dict={"queue_id": 1})
        self.assertIsNone(terminate_tasks_in_cache(
            cache_directory="executorlib_cache",
            backend="flux",
        ))

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)


@unittest.skipIf(
    skip_flux_test,
    "flux is not installed, so the flux tests are skipped.",
)
class TestPysqaSpawner(unittest.TestCase):
    def test_pysqa_spawner_sleep(self):
        interface_flux = PysqaSpawner(backend="flux", cores=1)
        self.assertTrue(interface_flux.bootup(command_lst=["sleep", "1"]))
        self.assertTrue(interface_flux._check_process_helper(command_lst=[]))
        self.assertTrue(interface_flux.poll())
        process_id = interface_flux._process
        interface_flux.shutdown(wait=True)
        interface_flux._process = process_id
        self.assertFalse(interface_flux.poll())
        self.assertFalse(interface_flux._check_process_helper(command_lst=["sleep", "1"]))

    def test_pysqa_spawner_big(self):
        interface_flux = PysqaSpawner(backend="flux", cores=100)
        self.assertFalse(interface_flux.bootup(command_lst=["sleep", "1"], stop_function=stop_function))

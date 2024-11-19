from concurrent.futures import CancelledError, Future
import importlib.util
from queue import Queue
from time import sleep
import shutil
import unittest

import numpy as np

from executorlib.base.executor import ExecutorBase
from executorlib.standalone.interactive.spawner import MpiExecSpawner
from executorlib.interactive.shared import (
    InteractiveExecutor,
    InteractiveStepExecutor,
    execute_parallel_tasks,
)
from executorlib.standalone.interactive.backend import call_funct
from executorlib.standalone.serialize import cloudpickle_register

try:
    import h5py

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True

skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def calc(i):
    return i


def calc_array(i):
    return np.array(i**2)


def echo_funct(i):
    return i


def get_global(memory=None):
    return memory


def set_global():
    return {"memory": np.array([5])}


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def raise_error():
    raise RuntimeError


def sleep_one(i):
    sleep(1)
    return i


class TestPyMpiExecutorSerial(unittest.TestCase):
    def test_pympiexecutor_two_workers(self):
        with InteractiveExecutor(
            max_workers=2,
            executor_kwargs={},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_one_worker(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())


class TestPyMpiExecutorStepSerial(unittest.TestCase):
    def test_pympiexecutor_two_workers(self):
        with InteractiveStepExecutor(
            max_cores=2,
            executor_kwargs={},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_one_worker(self):
        with InteractiveStepExecutor(
            max_cores=1,
            executor_kwargs={},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())


@unittest.skipIf(
    skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
)
class TestPyMpiExecutorMPI(unittest.TestCase):
    def test_pympiexecutor_one_worker_with_mpi(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_pympiexecutor_one_worker_with_mpi_multiple_submissions(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            cloudpickle_register(ind=1)
            fs1 = p.submit(mpi_funct, 1)
            fs2 = p.submit(mpi_funct, 2)
            fs3 = p.submit(mpi_funct, 3)
            output = [
                fs1.result(),
                fs2.result(),
                fs3.result(),
            ]
        self.assertEqual(
            output,
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_pympiexecutor_one_worker_with_mpi_echo(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            cloudpickle_register(ind=1)
            output = p.submit(echo_funct, 2).result()
        self.assertEqual(output, [2, 2])


@unittest.skipIf(
    skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
)
class TestPyMpiStepExecutorMPI(unittest.TestCase):
    def test_pympiexecutor_one_worker_with_mpi(self):
        with InteractiveStepExecutor(
            max_cores=2,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_pympiexecutor_one_worker_with_mpi_multiple_submissions(self):
        with InteractiveStepExecutor(
            max_cores=2,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            cloudpickle_register(ind=1)
            fs1 = p.submit(mpi_funct, 1)
            fs2 = p.submit(mpi_funct, 2)
            fs3 = p.submit(mpi_funct, 3)
            output = [
                fs1.result(),
                fs2.result(),
                fs3.result(),
            ]
        self.assertEqual(
            output,
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_pympiexecutor_one_worker_with_mpi_echo(self):
        with InteractiveStepExecutor(
            max_cores=2,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            cloudpickle_register(ind=1)
            output = p.submit(echo_funct, 2).result()
        self.assertEqual(output, [2, 2])


class TestPyMpiExecutorInitFunction(unittest.TestCase):
    def test_internal_memory(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "cores": 1,
                "init_function": set_global,
            },
            spawner=MpiExecSpawner,
        ) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

    def test_call_funct(self):
        self.assertEqual(
            call_funct(
                input_dict={"fn": get_global, "args": (), "kwargs": {}},
                memory={"memory": 4},
            ),
            4,
        )

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"fn": get_global, "args": (), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
            init_function=set_global,
        )
        self.assertEqual(f.result(), np.array([5]))
        q.join()


class TestFuturePool(unittest.TestCase):
    def test_pool_serial(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 1},
            spawner=MpiExecSpawner,
        ) as p:
            output = p.submit(calc_array, i=2)
            self.assertEqual(len(p), 1)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
            self.assertTrue(output.done())
            self.assertEqual(len(p), 0)
        self.assertEqual(output.result(), np.array(4))

    def test_executor_multi_submission(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 1},
            spawner=MpiExecSpawner,
        ) as p:
            fs_1 = p.submit(calc_array, i=2)
            fs_2 = p.submit(calc_array, i=2)
            self.assertEqual(fs_1.result(), np.array(4))
            self.assertEqual(fs_2.result(), np.array(4))
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_shutdown(self):
        p = InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 1},
            spawner=MpiExecSpawner,
        )
        fs1 = p.submit(sleep_one, i=2)
        fs2 = p.submit(sleep_one, i=4)
        sleep(1)
        p.shutdown(wait=True, cancel_futures=True)
        self.assertTrue(fs1.done())
        self.assertTrue(fs2.done())
        self.assertEqual(fs1.result(), 2)
        with self.assertRaises(CancelledError):
            fs2.result()

    def test_pool_serial_map(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 1},
            spawner=MpiExecSpawner,
        ) as p:
            output = p.map(calc_array, [1, 2, 3])
        self.assertEqual(list(output), [np.array(1), np.array(4), np.array(9)])

    def test_executor_exception(self):
        with self.assertRaises(RuntimeError):
            with InteractiveExecutor(
                max_workers=1,
                executor_kwargs={"cores": 1},
                spawner=MpiExecSpawner,
            ) as p:
                p.submit(raise_error)

    def test_executor_exception_future(self):
        with self.assertRaises(RuntimeError):
            with InteractiveExecutor(
                max_workers=1,
                executor_kwargs={"cores": 1},
                spawner=MpiExecSpawner,
            ) as p:
                fs = p.submit(raise_error)
                fs.result()

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_meta(self):
        meta_data_exe_dict = {
            "cores": 2,
            "spawner": "<class 'executorlib.standalone.interactive.spawner.MpiExecSpawner'>",
            "hostname_localhost": True,
            "init_function": None,
            "cwd": None,
            "openmpi_oversubscribe": False,
            "max_workers": 1,
        }
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={
                "cores": 2,
                "hostname_localhost": True,
                "init_function": None,
                "cwd": None,
                "openmpi_oversubscribe": False,
            },
            spawner=MpiExecSpawner,
        ) as exe:
            for k, v in meta_data_exe_dict.items():
                if k != "spawner":
                    self.assertEqual(exe.info[k], v)
                else:
                    self.assertEqual(str(exe.info[k]), v)
        with ExecutorBase() as exe:
            self.assertIsNone(exe.info)

    def test_meta_step(self):
        meta_data_exe_dict = {
            "cores": 2,
            "spawner": "<class 'executorlib.standalone.interactive.spawner.MpiExecSpawner'>",
            "hostname_localhost": True,
            "cwd": None,
            "openmpi_oversubscribe": False,
            "max_cores": 2,
        }
        with InteractiveStepExecutor(
            max_cores=2,
            executor_kwargs={
                "cores": 2,
                "hostname_localhost": True,
                "cwd": None,
                "openmpi_oversubscribe": False,
            },
            spawner=MpiExecSpawner,
        ) as exe:
            for k, v in meta_data_exe_dict.items():
                if k != "spawner":
                    self.assertEqual(exe.info[k], v)
                else:
                    self.assertEqual(str(exe.info[k]), v)

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_pool_multi_core(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            output = p.submit(mpi_funct, i=2)
            self.assertEqual(len(p), 1)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(2)
            self.assertTrue(output.done())
            self.assertEqual(len(p), 0)
        self.assertEqual(output.result(), [(2, 2, 0), (2, 2, 1)])

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_pool_multi_core_map(self):
        with InteractiveExecutor(
            max_workers=1,
            executor_kwargs={"cores": 2},
            spawner=MpiExecSpawner,
        ) as p:
            output = p.map(mpi_funct, [1, 2, 3])
        self.assertEqual(
            list(output),
            [[(1, 2, 0), (1, 2, 1)], [(2, 2, 0), (2, 2, 1)], [(3, 2, 0), (3, 2, 1)]],
        )

    def test_execute_task_failed_no_argument(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {}, "future": f})
        cloudpickle_register(ind=1)
        with self.assertRaises(TypeError):
            execute_parallel_tasks(
                future_queue=q,
                cores=1,
                openmpi_oversubscribe=False,
                spawner=MpiExecSpawner,
            )
        q.join()

    def test_execute_task_failed_wrong_argument(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {"j": 4}, "future": f})
        cloudpickle_register(ind=1)
        with self.assertRaises(TypeError):
            execute_parallel_tasks(
                future_queue=q,
                cores=1,
                openmpi_oversubscribe=False,
                spawner=MpiExecSpawner,
            )
        q.join()

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
        )
        self.assertEqual(f.result(), np.array(4))
        q.join()

    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_execute_task_parallel(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=2,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
        )
        self.assertEqual(f.result(), [np.array(4), np.array(4)])
        q.join()


class TestFuturePoolCache(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree("./cache")

    @unittest.skipIf(
        skip_h5py_test, "h5py is not installed, so the h5py tests are skipped."
    )
    def test_execute_task_cache(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc, "args": (), "kwargs": {"i": 1}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            openmpi_oversubscribe=False,
            spawner=MpiExecSpawner,
            cache_directory="./cache",
        )
        self.assertEqual(f.result(), 1)
        q.join()

    @unittest.skipIf(
        skip_h5py_test, "h5py is not installed, so the h5py tests are skipped."
    )
    def test_execute_task_cache_failed_no_argument(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {}, "future": f})
        cloudpickle_register(ind=1)
        with self.assertRaises(TypeError):
            execute_parallel_tasks(
                future_queue=q,
                cores=1,
                openmpi_oversubscribe=False,
                spawner=MpiExecSpawner,
                cache_directory="./cache",
            )
        q.join()

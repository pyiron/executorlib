from concurrent.futures import CancelledError, Future
from queue import Queue
from time import sleep
import unittest

import numpy as np

from pympipool.scheduler.mpi import PyMPIExecutor, PyMPIStepExecutor, MpiExecInterface
from pympipool.shared.backend import call_funct
from pympipool.shared.executorbase import (
    cloudpickle_register,
    execute_parallel_tasks,
    ExecutorBase,
)


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
        with PyMPIExecutor(max_workers=2, hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_one_worker(self):
        with PyMPIExecutor(max_workers=1, hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_errors(self):
        with self.assertRaises(TypeError):
            PyMPIExecutor(
                max_workers=1,
                cores_per_worker=1,
                threads_per_core=2,
                hostname_localhost=True,
            )
        with self.assertRaises(TypeError):
            PyMPIExecutor(
                max_workers=1,
                cores_per_worker=1,
                gpus_per_worker=1,
                hostname_localhost=True,
            )


class TestPyMpiExecutorStepSerial(unittest.TestCase):
    def test_pympiexecutor_two_workers(self):
        with PyMPIStepExecutor(max_cores=2, hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_one_worker(self):
        with PyMPIStepExecutor(max_cores=1, hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_pympiexecutor_errors(self):
        with self.assertRaises(TypeError):
            PyMPIStepExecutor(
                max_cores=1,
                cores_per_worker=1,
                threads_per_core=2,
                hostname_localhost=True,
            )
        with self.assertRaises(TypeError):
            PyMPIStepExecutor(
                max_cores=1,
                cores_per_worker=1,
                gpus_per_worker=1,
                hostname_localhost=True,
            )


class TestPyMpiExecutorMPI(unittest.TestCase):
    def test_pympiexecutor_one_worker_with_mpi(self):
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_pympiexecutor_one_worker_with_mpi_multiple_submissions(self):
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
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
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
        ) as p:
            cloudpickle_register(ind=1)
            output = p.submit(echo_funct, 2).result()
        self.assertEqual(output, [2, 2])


class TestPyMpiStepExecutorMPI(unittest.TestCase):
    def test_pympiexecutor_one_worker_with_mpi(self):
        with PyMPIStepExecutor(
            max_cores=2, cores_per_worker=2, hostname_localhost=True
        ) as exe:
            cloudpickle_register(ind=1)
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_pympiexecutor_one_worker_with_mpi_multiple_submissions(self):
        with PyMPIStepExecutor(
            max_cores=2, cores_per_worker=2, hostname_localhost=True
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
        with PyMPIStepExecutor(
            max_cores=2, cores_per_worker=2, hostname_localhost=True
        ) as p:
            cloudpickle_register(ind=1)
            output = p.submit(echo_funct, 2).result()
        self.assertEqual(output, [2, 2])


class TestPyMpiExecutorInitFunction(unittest.TestCase):
    def test_internal_memory(self):
        with PyMPIExecutor(
            max_workers=1,
            cores_per_worker=1,
            init_function=set_global,
            hostname_localhost=True,
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
            oversubscribe=False,
            interface_class=MpiExecInterface,
            hostname_localhost=True,
            init_function=set_global,
        )
        self.assertEqual(f.result(), np.array([5]))
        q.join()


class TestFuturePool(unittest.TestCase):
    def test_pool_serial(self):
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=1, hostname_localhost=True
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
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=1, hostname_localhost=True
        ) as p:
            fs_1 = p.submit(calc_array, i=2)
            fs_2 = p.submit(calc_array, i=2)
            self.assertEqual(fs_1.result(), np.array(4))
            self.assertEqual(fs_2.result(), np.array(4))
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_shutdown(self):
        p = PyMPIExecutor(max_workers=1, cores_per_worker=1, hostname_localhost=True)
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
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=1, hostname_localhost=True
        ) as p:
            output = p.map(calc_array, [1, 2, 3])
        self.assertEqual(list(output), [np.array(1), np.array(4), np.array(9)])

    def test_executor_exception(self):
        with self.assertRaises(RuntimeError):
            with PyMPIExecutor(
                max_workers=1, cores_per_worker=1, hostname_localhost=True
            ) as p:
                p.submit(raise_error)

    def test_executor_exception_future(self):
        with self.assertRaises(RuntimeError):
            with PyMPIExecutor(
                max_workers=1, cores_per_worker=1, hostname_localhost=True
            ) as p:
                fs = p.submit(raise_error)
                fs.result()

    def test_meta(self):
        meta_data_exe_dict = {
            "cores": 2,
            "interface_class": "<class 'pympipool.shared.interface.MpiExecInterface'>",
            "hostname_localhost": True,
            "init_function": None,
            "cwd": None,
            "oversubscribe": False,
            "max_workers": 1,
        }
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
        ) as exe:
            for k, v in meta_data_exe_dict.items():
                if k != "interface_class":
                    self.assertEqual(exe.info[k], v)
                else:
                    self.assertEqual(str(exe.info[k]), v)
        with ExecutorBase() as exe:
            self.assertIsNone(exe.info)

    def test_meta_step(self):
        meta_data_exe_dict = {
            "cores": 2,
            "interface_class": "<class 'pympipool.shared.interface.MpiExecInterface'>",
            "hostname_localhost": True,
            "cwd": None,
            "oversubscribe": False,
            "max_cores": 2,
        }
        with PyMPIStepExecutor(
            max_cores=2, cores_per_worker=2, hostname_localhost=True
        ) as exe:
            for k, v in meta_data_exe_dict.items():
                if k != "interface_class":
                    self.assertEqual(exe.info[k], v)
                else:
                    self.assertEqual(str(exe.info[k]), v)

    def test_pool_multi_core(self):
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
        ) as p:
            output = p.submit(mpi_funct, i=2)
            self.assertEqual(len(p), 1)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
            self.assertTrue(output.done())
            self.assertEqual(len(p), 0)
        self.assertEqual(output.result(), [(2, 2, 0), (2, 2, 1)])

    def test_pool_multi_core_map(self):
        with PyMPIExecutor(
            max_workers=1, cores_per_worker=2, hostname_localhost=True
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
                oversubscribe=False,
                interface_class=MpiExecInterface,
                hostname_localhost=True,
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
                oversubscribe=False,
                interface_class=MpiExecInterface,
                hostname_localhost=True,
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
            oversubscribe=False,
            interface_class=MpiExecInterface,
            hostname_localhost=True,
        )
        self.assertEqual(f.result(), np.array(4))
        q.join()

    def test_execute_task_parallel(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc_array, "args": (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=2,
            oversubscribe=False,
            interface_class=MpiExecInterface,
            hostname_localhost=True,
        )
        self.assertEqual(f.result(), [np.array(4), np.array(4)])
        q.join()

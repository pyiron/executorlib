from concurrent.futures import as_completed, Future, Executor
from queue import Queue
import unittest
from pympipool.shared.executorbase import (
    executor_broker,
    execute_task_dict,
    _get_executor_dict,
    _get_future_done,
)
from pympipool.mpi.executor import PyMPIExecutor, PyMPISingleTaskExecutor


def calc(i):
    return i


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


class TestFutureCreation(unittest.TestCase):
    def test_get_future_done(self):
        f = _get_future_done()
        self.assertTrue(isinstance(f, Future))
        self.assertTrue(f.done())


class TestMetaExecutorFuture(unittest.TestCase):
    def test_meta_executor_future(self):
        meta_future = _get_executor_dict(
            max_workers=1,
            executor_class=PyMPISingleTaskExecutor,
        )
        future_obj = list(meta_future.keys())[0]
        executor_obj = list(meta_future.values())[0]
        self.assertTrue(isinstance(future_obj, Future))
        self.assertTrue(isinstance(executor_obj, Executor))
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj, next(as_completed(meta_future.keys())))
        executor_obj.shutdown(wait=True)

    def test_execute_task_dict(self):
        meta_future_lst = _get_executor_dict(
            max_workers=1,
            executor_class=PyMPISingleTaskExecutor,
        )
        f = Future()
        self.assertTrue(
            execute_task_dict(
                task_dict={"fn": calc, "args": (1,), "kwargs": {}, "future": f},
                meta_future_lst=meta_future_lst,
            )
        )
        self.assertEqual(f.result(), 1)
        self.assertTrue(f.done())
        self.assertFalse(
            execute_task_dict(
                task_dict={"shutdown": True, "wait": True},
                meta_future_lst=meta_future_lst,
            )
        )

    def test_execute_task_dict_error(self):
        meta_future_lst = _get_executor_dict(
            max_workers=1,
            executor_class=PyMPISingleTaskExecutor,
        )
        with self.assertRaises(ValueError):
            execute_task_dict(task_dict={}, meta_future_lst=meta_future_lst)
        list(meta_future_lst.values())[0].shutdown(wait=True)

    def test_executor_broker(self):
        q = Queue()
        f = Future()
        q.put({"fn": calc, "args": (1,), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        executor_broker(future_queue=q, max_workers=1, executor_class=PyMPISingleTaskExecutor)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 1)
        q.join()


class TestMetaExecutor(unittest.TestCase):
    def test_meta_executor_serial(self):
        with PyMPIExecutor(max_workers=2) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_single(self):
        with PyMPIExecutor(max_workers=1) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

    def test_meta_executor_parallel(self):
        with PyMPIExecutor(max_workers=1, cores_per_worker=2) as exe:
            fs_1 = exe.submit(mpi_funct, 1)
            self.assertEqual(fs_1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs_1.done())

    def test_errors(self):
        with self.assertRaises(TypeError):
            PyMPIExecutor(max_workers=1, cores_per_worker=1, threads_per_core=2)
        with self.assertRaises(TypeError):
            PyMPIExecutor(max_workers=1, cores_per_worker=1, gpus_per_worker=1)

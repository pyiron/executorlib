from concurrent.futures import as_completed, Future, Executor
from queue import Queue
import unittest
from pympipool.shared.broker import (
    executor_broker,
    _execute_task_dict,
    _get_future_done,
    _get_executor_list,
)

from pympipool.interfaces.taskbroker import HPCExecutor


def calc(i):
    return i


class TestFutureCreation(unittest.TestCase):
    def test_get_future_done(self):
        f = _get_future_done()
        self.assertTrue(isinstance(f, Future))
        self.assertTrue(f.done())


class TestMetaExecutorFuture(unittest.TestCase):
    def test_meta_executor_future(self):
        meta_future = _get_executor_list(max_workers=1)[0]
        self.assertTrue(isinstance(meta_future.future, Future))
        self.assertTrue(isinstance(meta_future.executor, Executor))
        self.assertTrue(meta_future.done())
        self.assertEqual(meta_future, next(as_completed([meta_future])))
        meta_future.submit(task_dict={"shutdown": True, "wait": True, "future": _get_future_done()})

    def test_execute_task_dict(self):
        meta_future_lst = _get_executor_list(max_workers=1)
        f = Future()
        self.assertTrue(_execute_task_dict(
            task_dict={"fn": calc, "args": (1,), "kwargs": {}, "future": f},
            meta_future_lst=meta_future_lst
        ))
        self.assertEqual(f.result(), 1)
        self.assertTrue(f.done())
        self.assertFalse(_execute_task_dict(
            task_dict={"shutdown": True, "wait": True},
            meta_future_lst=meta_future_lst
        ))

    def test_executor_broker(self):
        q = Queue()
        f = Future()
        q.put({"fn": calc, "args": (1,), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        executor_broker(future_queue=q, max_workers=1)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 1)
        q.join()


class TestMetaExecutor(unittest.TestCase):
    def test_meta_executor(self):
        with HPCExecutor(max_workers=2) as exe:
            fs_1 = exe.submit(calc, 1)
            fs_2 = exe.submit(calc, 2)
            self.assertEqual(fs_1.result(), 1)
            self.assertEqual(fs_2.result(), 2)
            self.assertTrue(fs_1.done())
            self.assertTrue(fs_2.done())

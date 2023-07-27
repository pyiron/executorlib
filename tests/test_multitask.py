import numpy as np
import unittest
from queue import Queue
from time import sleep
from pympipool import PoolExecutor
from pympipool.legacy.shared.interface import execute_serial_tasks
from pympipool.shared.taskexecutor import cloudpickle_register
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


def sleep_one(i):
    sleep(1)
    return i


def wait_and_calc(n):
    sleep(1)
    return n ** 2


def call_back(future):
    global_lst.append(future.result())


global_lst = []


class TestFuturePool(unittest.TestCase):
    def test_pool_serial(self):
        with PoolExecutor(max_workers=1) as p:
            output = p.submit(calc, i=2)
            self.assertEqual(len(p), 1)
            self.assertTrue(isinstance(output, Future))
            self.assertFalse(output.done())
            sleep(1)
            self.assertTrue(output.done())
            self.assertEqual(len(p), 0)
        self.assertEqual(output.result(), np.array(4))

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"fn": calc, 'args': (), "kwargs": {"i": 2}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_serial_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            enable_flux_backend=False
        )
        self.assertEqual(f.result(), np.array(4))
        q.join()

    def test_pool_cancel(self):
        with PoolExecutor(max_workers=2, sleep_interval=0) as p:
            fs1 = p.submit(sleep_one, i=2)
            fs2 = p.submit(sleep_one, i=2)
            fs3 = p.submit(sleep_one, i=2)
            fs4 = p.submit(sleep_one, i=2)
            sleep(1)
            fs1.cancel()
            fs2.cancel()
            fs3.cancel()
            fs4.cancel()
        self.assertTrue(fs1.done())
        self.assertTrue(fs2.done())
        self.assertTrue(fs3.done())
        self.assertTrue(fs4.done())

    def test_cancel_task(self):
        fs1 = Future()
        fs1.cancel()
        q = Queue()
        q.put({"fn": sleep_one, 'args': (), "kwargs": {"i": 1}, "future": fs1})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_serial_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            enable_flux_backend=False
        )
        self.assertTrue(fs1.done())
        self.assertTrue(fs1.cancelled())
        q.join()

    def test_waiting(self):
        exe = PoolExecutor(max_workers=2)
        f1 = exe.submit(wait_and_calc, 42)
        f2 = exe.submit(wait_and_calc, 84)
        f1.add_done_callback(call_back)
        f2.add_done_callback(call_back)
        exe.shutdown(wait=True)
        self.assertTrue([42**2, 84**2], global_lst)

import numpy as np
import unittest
from queue import Queue
from time import sleep
from pympipool import MultiTaskExecutor
from pympipool.share.serial import execute_serial_tasks, cloudpickle_register
from concurrent.futures import Future


def calc(i):
    return np.array(i ** 2)


def sleep_one(i):
    sleep(1)
    return i


class TestFuturePool(unittest.TestCase):
    def test_pool_serial(self):
        with MultiTaskExecutor(cores=1) as p:
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
        q.put({"shutdown": True})
        cloudpickle_register(ind=1)
        execute_serial_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            enable_flux_backend=False
        )
        self.assertEqual(f.result(), np.array(4))

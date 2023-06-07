import unittest
import numpy as np
from queue import Queue
from pympipool import Worker, _cloudpickle_update
from pympipool.share.parallel import call_funct
from pympipool.share.serial import execute_tasks
from concurrent.futures import Future


def get_global(memory=None):
    return memory


def set_global():
    return {"memory": np.array([5])}


class TestWorkerMemory(unittest.TestCase):
    def test_internal_memory(self):
        with Worker(cores=1, init_function=set_global) as p:
            f = p.submit(get_global)
            self.assertFalse(f.done())
            self.assertEqual(f.result(), np.array([5]))
            self.assertTrue(f.done())

    def test_call_funct(self):
        self.assertEqual(call_funct(
            input_dict={"f": get_global, "a": (), "k": {}},
            memory={"memory": 4}
        ), 4)

    def test_execute_task(self):
        f = Future()
        q = Queue()
        q.put({"i": True, "f": set_global, "a": (), "k": {}})
        q.put({"f": get_global, 'a': (), "k": {}, "l": f})
        q.put({"c": "close"})
        _cloudpickle_update(ind=1)
        execute_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            enable_flux_backend=False
        )
        self.assertEqual(f.result(), np.array([5]))

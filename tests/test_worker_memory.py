import unittest
import numpy as np
from queue import Queue
from pympipool.shared.backend import call_funct
from pympipool.shared.executorbase import cloudpickle_register, execute_parallel_tasks
from pympipool.mpi.executor import PyMPISingleTaskExecutor, MpiExecInterface
from concurrent.futures import Future


def get_global(memory=None):
    return memory


def set_global():
    return {"memory": np.array([5])}


class TestWorkerMemory(unittest.TestCase):
    def test_internal_memory(self):
        with PyMPISingleTaskExecutor(cores=1, init_function=set_global) as p:
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
        q.put({"init": True, "fn": set_global, "args": (), "kwargs": {}})
        q.put({"fn": get_global, "args": (), "kwargs": {}, "future": f})
        q.put({"shutdown": True, "wait": True})
        cloudpickle_register(ind=1)
        execute_parallel_tasks(
            future_queue=q,
            cores=1,
            oversubscribe=False,
            interface_class=MpiExecInterface,
        )
        self.assertEqual(f.result(), np.array([5]))
        q.join()

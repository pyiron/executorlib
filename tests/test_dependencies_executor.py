from concurrent.futures import Future
import unittest
from time import sleep
from queue import Queue

from pympipool import Executor
from pympipool.shared.executor import cloudpickle_register
from pympipool.interactive import create_executor
from pympipool.shared.thread import RaisingThread
from pympipool.shared.executor import execute_tasks_with_dependencies
from pympipool.shared.plot import generate_nodes_and_edges


try:
    import pygraphviz

    skip_graphviz_test = False
except ImportError:
    skip_graphviz_test = True


def add_function(parameter_1, parameter_2):
    sleep(0.2)
    return parameter_1 + parameter_2


class TestExecutorWithDependencies(unittest.TestCase):
    def test_executor(self):
        with Executor(max_cores=1, backend="local", hostname_localhost=True) as exe:
            cloudpickle_register(ind=1)
            future_1 = exe.submit(add_function, 1, parameter_2=2)
            future_2 = exe.submit(add_function, 1, parameter_2=future_1)
            self.assertEqual(future_2.result(), 4)

    @unittest.skipIf(
        skip_graphviz_test,
        "graphviz is not installed, so the plot_dependency_graph test is skipped.",
    )
    def test_executor_dependency_plot(self):
        with Executor(
            max_cores=1,
            backend="local",
            hostname_localhost=True,
            plot_dependency_graph=True,
        ) as exe:
            cloudpickle_register(ind=1)
            future_1 = exe.submit(add_function, 1, parameter_2=2)
            future_2 = exe.submit(add_function, 1, parameter_2=future_1)
            self.assertTrue(future_1.done())
            self.assertTrue(future_2.done())
            self.assertEqual(len(exe._future_hash_dict), 2)
            self.assertEqual(len(exe._task_hash_dict), 2)
            nodes, edges = generate_nodes_and_edges(
                task_hash_dict=exe._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 5)
            self.assertEqual(len(edges), 4)

    def test_dependency_steps(self):
        cloudpickle_register(ind=1)
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put(
            {
                "fn": add_function,
                "args": (),
                "kwargs": {"parameter_1": 1, "parameter_2": 2},
                "future": fs1,
                "resource_dict": {"cores": 1},
            }
        )
        q.put(
            {
                "fn": add_function,
                "args": (),
                "kwargs": {"parameter_1": 1, "parameter_2": fs1},
                "future": fs2,
                "resource_dict": {"cores": 1},
            }
        )
        executor = create_executor(
            max_workers=1,
            max_cores=2,
            cores_per_worker=1,
            threads_per_core=1,
            gpus_per_worker=0,
            oversubscribe=False,
            backend="local",
            hostname_localhost=True,
        )
        process = RaisingThread(
            target=execute_tasks_with_dependencies,
            kwargs={
                "future_queue": q,
                "executor_queue": executor._future_queue,
                "executor": executor,
                "refresh_rate": 0.01,
            },
        )
        process.start()
        self.assertFalse(fs1.done())
        self.assertFalse(fs2.done())
        self.assertEqual(fs2.result(), 4)
        self.assertTrue(fs1.done())
        self.assertTrue(fs2.done())
        q.put({"shutdown": True, "wait": True})

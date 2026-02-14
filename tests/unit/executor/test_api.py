import os
import shutil
import unittest
from concurrent.futures import Future

from executorlib import get_cache_data, get_future_from_cache
from executorlib.api import TestClusterExecutor
from executorlib.task_scheduler.interactive.dependency_plot import generate_nodes_and_edges_for_plotting
from executorlib.standalone.serialize import cloudpickle_register

try:
    import h5py

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def add_function(parameter_1, parameter_2):
    return parameter_1 + parameter_2


def foo(x):
    return x + 1


def get_error(i):
    raise ValueError(f"error {i}")


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5io tests are skipped."
)
class TestTestClusterExecutor(unittest.TestCase):
    def test_cache_dir(self):
        with TestClusterExecutor(cache_directory="not_this_dir", resource_dict={}) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                foo,
                1,
                resource_dict={
                    "cache_directory": "rather_this_dir",
                    "cache_key": "foo",
                },
            )
            self.assertEqual(future.result(), 2)
        self.assertFalse(os.path.exists("not_this_dir"))
        cache_lst = get_cache_data(cache_directory="not_this_dir")
        self.assertEqual(len(cache_lst), 0)
        self.assertTrue(os.path.exists("rather_this_dir"))
        cache_lst = get_cache_data(cache_directory="rather_this_dir")
        self.assertEqual(len(cache_lst), 1)
        with TestClusterExecutor(cache_directory="not_this_dir", resource_dict={}) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                foo,
                1,
                resource_dict={
                    "cache_directory": "rather_this_dir",
                    "cache_key": "foo",
                },
            )
            self.assertEqual(future.result(), 2)
        self.assertFalse(os.path.exists("not_this_dir"))
        cache_lst = get_cache_data(cache_directory="not_this_dir")
        self.assertEqual(len(cache_lst), 0)
        self.assertTrue(os.path.exists("rather_this_dir"))
        cache_lst = get_cache_data(cache_directory="rather_this_dir")
        self.assertEqual(len(cache_lst), 1)

    def test_get_future_from_cache(self):
        with TestClusterExecutor(cache_directory="cache_dir", resource_dict={}) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                foo,
                1,
                resource_dict={
                    "cache_directory": "cache_dir",
                    "cache_key": "foo",
                },
            )
            future_error = exe.submit(
                get_error,
                1,
                resource_dict={
                    "cache_directory": "cache_dir",
                    "cache_key": "error",
                },
            )
            self.assertEqual(future.result(), 2)
            with self.assertRaises(ValueError):
                future_error.result()
        future = get_future_from_cache(
            cache_directory="cache_dir",
            cache_key="foo",
        )
        self.assertTrue(isinstance(future, Future))
        self.assertTrue(future.done())
        self.assertEqual(future.result(), 2)
        with self.assertRaises(ValueError):
            get_future_from_cache(
                cache_directory="cache_dir",
                cache_key="error",
            )

    def test_empty(self):
        with TestClusterExecutor(cache_directory="rather_this_dir") as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(foo,1)
            self.assertEqual(future.result(), 2)
        self.assertTrue(os.path.exists("rather_this_dir"))
        cache_lst = get_cache_data(cache_directory="rather_this_dir")
        self.assertEqual(len(cache_lst), 1)

    def test_executor_dependency_plot(self):
        with TestClusterExecutor(
            plot_dependency_graph=True,
        ) as exe:
            cloudpickle_register(ind=1)
            future_1 = exe.submit(add_function, 1, parameter_2=2)
            future_2 = exe.submit(add_function, 1, parameter_2=future_1)
            self.assertTrue(future_1.done())
            self.assertTrue(future_2.done())
            self.assertEqual(len(exe._task_scheduler._future_hash_dict), 2)
            self.assertEqual(len(exe._task_scheduler._task_hash_dict), 2)
            nodes, edges = generate_nodes_and_edges_for_plotting(
                task_hash_dict=exe._task_scheduler._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._task_scheduler._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 4)
            self.assertEqual(len(edges), 4)

    def tearDown(self):
        shutil.rmtree("rather_this_dir", ignore_errors=True)
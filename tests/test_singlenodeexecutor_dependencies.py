from concurrent.futures import Future
import unittest
from time import sleep
from queue import Queue
from threading import Thread

from executorlib import SingleNodeExecutor
from executorlib.interfaces.single import create_single_node_executor
from executorlib.interactive.dependency import _execute_tasks_with_dependencies
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.standalone.interactive.spawner import MpiExecSpawner


try:
    import pygraphviz

    skip_graphviz_test = False
except ImportError:
    skip_graphviz_test = True


def add_function(parameter_1, parameter_2):
    sleep(0.2)
    return parameter_1 + parameter_2


def generate_tasks(length):
    sleep(0.2)
    return range(length)


def calc_from_lst(lst, ind, parameter):
    sleep(0.2)
    return lst[ind] + parameter


def merge(lst):
    sleep(0.2)
    return sum(lst)


def return_input_dict(input_dict):
    return input_dict


def raise_error(parameter):
    raise RuntimeError


class TestExecutorWithDependencies(unittest.TestCase):
    def test_executor(self):
        with SingleNodeExecutor(max_cores=1) as exe:
            cloudpickle_register(ind=1)
            future_1 = exe.submit(add_function, 1, parameter_2=2)
            future_2 = exe.submit(add_function, 1, parameter_2=future_1)
            self.assertEqual(future_2.result(), 4)

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
        executor = create_single_node_executor(
            max_workers=1,
            max_cores=2,
            resource_dict={
                "cores": 1,
                "threads_per_core": 1,
                "gpus_per_core": 0,
                "cwd": None,
                "openmpi_oversubscribe": False,
                "slurm_cmd_args": [],
            },
        )
        process = Thread(
            target=_execute_tasks_with_dependencies,
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

    def test_dependency_steps_error(self):
        cloudpickle_register(ind=1)
        fs1 = Future()
        fs2 = Future()
        q = Queue()
        q.put(
            {
                "fn": raise_error,
                "args": (),
                "kwargs": {"parameter": 0},
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
        executor = create_single_node_executor(
            max_workers=1,
            max_cores=2,
            resource_dict={
                "cores": 1,
                "threads_per_core": 1,
                "gpus_per_core": 0,
                "cwd": None,
                "openmpi_oversubscribe": False,
                "slurm_cmd_args": [],
            },
        )
        process = Thread(
            target=_execute_tasks_with_dependencies,
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
        self.assertTrue(fs1.exception() is not None)
        self.assertTrue(fs2.exception() is not None)
        with self.assertRaises(RuntimeError):
            fs2.result()
        q.put({"shutdown": True, "wait": True})

    def test_dependency_steps_error_before(self):
        cloudpickle_register(ind=1)
        fs1 = Future()
        fs1.set_exception(RuntimeError())
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
        executor = create_single_node_executor(
            max_workers=1,
            max_cores=2,
            resource_dict={
                "cores": 1,
                "threads_per_core": 1,
                "gpus_per_core": 0,
                "cwd": None,
                "openmpi_oversubscribe": False,
                "slurm_cmd_args": [],
            },
        )
        process = Thread(
            target=_execute_tasks_with_dependencies,
            kwargs={
                "future_queue": q,
                "executor_queue": executor._future_queue,
                "executor": executor,
                "refresh_rate": 0.01,
            },
        )
        process.start()
        self.assertTrue(fs1.exception() is not None)
        self.assertTrue(fs2.exception() is not None)
        with self.assertRaises(RuntimeError):
            fs2.result()
        executor.shutdown(wait=True)
        q.put({"shutdown": True, "wait": True})
        q.join()
        process.join()

    def test_many_to_one(self):
        length = 5
        parameter = 1
        with SingleNodeExecutor(max_cores=2) as exe:
            cloudpickle_register(ind=1)
            future_lst = exe.submit(
                generate_tasks,
                length=length,
                resource_dict={"cores": 1},
            )
            lst = []
            for i in range(length):
                lst.append(
                    exe.submit(
                        calc_from_lst,
                        lst=future_lst,
                        ind=i,
                        parameter=parameter,
                        resource_dict={"cores": 1},
                    )
                )
            future_sum = exe.submit(
                merge,
                lst=lst,
                resource_dict={"cores": 1},
            )
            self.assertEqual(future_sum.result(), 15)

    def test_future_input_dict(self):
        with SingleNodeExecutor() as exe:
            fs = exe.submit(
                return_input_dict,
                input_dict={"a": exe.submit(sum, [2, 2])},
            )
            self.assertEqual(fs.result()["a"], 4)


class TestExecutorErrors(unittest.TestCase):
    def test_block_allocation_false_one_worker(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=1, block_allocation=False) as exe:
                cloudpickle_register(ind=1)
                fs = exe.submit(raise_error, parameter=0)
                fs.result()

    def test_block_allocation_true_one_worker(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=1, block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                fs = exe.submit(raise_error, parameter=0)
                fs.result()

    def test_block_allocation_false_two_workers(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=2, block_allocation=False) as exe:
                cloudpickle_register(ind=1)
                fs = exe.submit(raise_error, parameter=0)
                fs.result()

    def test_block_allocation_true_two_workers(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=2, block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                fs = exe.submit(raise_error, parameter=0)
                fs.result()

    def test_block_allocation_false_one_worker_loop(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=1, block_allocation=False) as exe:
                cloudpickle_register(ind=1)
                lst = []
                for i in range(1, 4):
                    lst = exe.submit(
                        raise_error,
                        parameter=lst,
                    )
                lst.result()

    def test_block_allocation_true_one_worker_loop(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=1, block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                lst = []
                for i in range(1, 4):
                    lst = exe.submit(
                        raise_error,
                        parameter=lst,
                    )
                lst.result()

    def test_block_allocation_false_two_workers_loop(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=2, block_allocation=False) as exe:
                cloudpickle_register(ind=1)
                lst = []
                for i in range(1, 4):
                    lst = exe.submit(
                        raise_error,
                        parameter=lst,
                    )
                lst.result()

    def test_block_allocation_true_two_workers_loop(self):
        with self.assertRaises(RuntimeError):
            with SingleNodeExecutor(max_cores=2, block_allocation=True) as exe:
                cloudpickle_register(ind=1)
                lst = []
                for i in range(1, 4):
                    lst = exe.submit(
                        raise_error,
                        parameter=lst,
                    )
                lst.result()


class TestInfo(unittest.TestCase):
    """Test cases for the info property of SingleNodeExecutor."""

    def setUp(self):
        """Set up the expected info dictionary."""
        self.expected_info = {
            'cores': 1,
            'cwd': None,
            'openmpi_oversubscribe': False,
            'cache_directory': None,
            'hostname_localhost': None,
            'spawner': MpiExecSpawner,
            'max_cores': None,
            'max_workers': None,
        }

    def test_info_disable_dependencies_true(self):
        """Test info property with dependencies disabled."""
        with SingleNodeExecutor(disable_dependencies=True) as exe:
            self.assertEqual(exe.info, self.expected_info)

    def test_info_disable_dependencies_false(self):
        """Test info property with dependencies enabled."""
        with SingleNodeExecutor(disable_dependencies=False) as exe:
            self.assertEqual(exe.info, self.expected_info)

    def test_info_error_handling(self):
        """Test info property error handling when executor is not running."""
        exe = SingleNodeExecutor()
        exe.shutdown(wait=True)
        self.assertIsNone(exe.info)

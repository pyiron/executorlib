import os
import unittest
from time import sleep

from executorlib import (
    SingleNodeExecutor,
    SlurmJobExecutor,
    SlurmClusterExecutor,
    split_future,
    get_item_from_future,
)
from executorlib.task_scheduler.interactive.dependency_plot import generate_nodes_and_edges_for_plotting
from executorlib.standalone.serialize import cloudpickle_register


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


def get_tuple(i):
    return 1, 2, i


def get_dict(i):
    return {"a": 1, "b": i}


def echo(i):
    return i

@unittest.skipIf(
    skip_graphviz_test,
    "graphviz is not installed, so the plot_dependency_graph tests are skipped.",
)
class TestLocalExecutorWithDependencies(unittest.TestCase):
    def test_executor_dependency_plot(self):
        with SingleNodeExecutor(
            max_cores=1,
            plot_dependency_graph=True,
            block_allocation=True,
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

    def test_executor_dependency_plot_filename(self):
        graph_file = os.path.join(os.path.dirname(__file__), "test.png")
        with SingleNodeExecutor(
            max_cores=1,
            block_allocation=False,
            plot_dependency_graph=False,
            plot_dependency_graph_filename=graph_file,
        ) as exe:
            cloudpickle_register(ind=1)
            future_1 = exe.submit(add_function, 1, parameter_2=2)
            future_2 = exe.submit(add_function, 1, parameter_2=future_1)
            self.assertTrue(future_1.done())
            self.assertTrue(future_2.done())
        self.assertTrue(os.path.exists(graph_file))
        os.remove(graph_file)

    def test_many_to_one_plot(self):
        length = 5
        parameter = 1
        with SingleNodeExecutor(
            max_cores=2,
            block_allocation=False,
            plot_dependency_graph=True,
        ) as exe:
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
            self.assertTrue(future_lst.done())
            for l in lst:
                self.assertTrue(l.done())
            self.assertTrue(future_sum.done())
            self.assertEqual(len(exe._task_scheduler._future_hash_dict), 7)
            self.assertEqual(len(exe._task_scheduler._task_hash_dict), 7)
            nodes, edges = generate_nodes_and_edges_for_plotting(
                task_hash_dict=exe._task_scheduler._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._task_scheduler._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 14)
            self.assertEqual(len(edges), 22)

    def test_future_input_dict(self):
        with SingleNodeExecutor(plot_dependency_graph=True) as exe:
            exe.submit(
                return_input_dict,
                input_dict={"a": exe.submit(sum, [2, 2])},
            )
            self.assertEqual(len(exe._task_scheduler._future_hash_dict), 2)
            self.assertEqual(len(exe._task_scheduler._task_hash_dict), 2)
            nodes, edges = generate_nodes_and_edges_for_plotting(
                task_hash_dict=exe._task_scheduler._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._task_scheduler._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 4)
            self.assertEqual(len(edges), 3)


@unittest.skipIf(
    skip_graphviz_test,
    "graphviz is not installed, so the plot_dependency_graph tests are skipped.",
)
class TestSlurmAllocationExecutorWithDependencies(unittest.TestCase):
    def test_executor_dependency_plot(self):
        with SlurmJobExecutor(
            max_cores=1,
            block_allocation=False,
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

    def test_many_to_one_plot(self):
        length = 5
        parameter = 1
        with SlurmJobExecutor(
            max_cores=2,
            block_allocation=False,
            plot_dependency_graph=True,
        ) as exe:
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
            self.assertTrue(future_lst.done())
            for l in lst:
                self.assertTrue(l.done())
            self.assertTrue(future_sum.done())
            self.assertEqual(len(exe._task_scheduler._future_hash_dict), 7)
            self.assertEqual(len(exe._task_scheduler._task_hash_dict), 7)
            nodes, edges = generate_nodes_and_edges_for_plotting(
                task_hash_dict=exe._task_scheduler._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._task_scheduler._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 14)
            self.assertEqual(len(edges), 22)


@unittest.skipIf(
    skip_graphviz_test,
    "graphviz is not installed, so the plot_dependency_graph tests are skipped.",
)
class TestSlurmSubmissionExecutorWithDependencies(unittest.TestCase):
    def test_executor_dependency_plot(self):
        with SlurmClusterExecutor(
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

    def test_many_to_one_plot(self):
        length = 5
        parameter = 1
        with SlurmClusterExecutor(
            plot_dependency_graph=True,
        ) as exe:
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
            self.assertTrue(future_lst.done())
            for l in lst:
                self.assertTrue(l.done())
            self.assertTrue(future_sum.done())
            self.assertEqual(len(exe._task_scheduler._future_hash_dict), 7)
            self.assertEqual(len(exe._task_scheduler._task_hash_dict), 7)
            nodes, edges = generate_nodes_and_edges_for_plotting(
                task_hash_dict=exe._task_scheduler._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._task_scheduler._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 14)
            self.assertEqual(len(edges), 22)


@unittest.skipIf(
    skip_graphviz_test,
    "graphviz is not installed, so the plot_dependency_graph tests are skipped.",
)
class TestSelectExecutorPlot(unittest.TestCase):
    def test_split_future(self):
        with SingleNodeExecutor(plot_dependency_graph=True) as exe:
            f = exe.submit(get_tuple, 5)
            f1, f2, f3 = split_future(future=f, n=3)
            f12 = exe.submit(echo, f1)
            f22 = exe.submit(echo, f2)
            f32 = exe.submit(echo, f3)
            self.assertIsNone(f12.result())
            self.assertIsNone(f22.result())
            self.assertIsNone(f32.result())

    def test_get_item_from_future(self):
        with SingleNodeExecutor(plot_dependency_graph=True) as exe:
            f = exe.submit(get_dict, 5)
            f1 = exe.submit(echo, get_item_from_future(future=f, key="a"))
            f2 = exe.submit(echo, get_item_from_future(future=f, key="b"))
            self.assertIsNone(f1.result())
            self.assertIsNone(f2.result())

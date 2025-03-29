import os
import unittest
from time import sleep

from executorlib import FluxJobExecutor, FluxClusterExecutor
from executorlib.standalone.plot import generate_nodes_and_edges
from executorlib.standalone.serialize import cloudpickle_register


try:
    import pygraphviz
    import flux.job
    from executorlib.interactive.fluxspawner import FluxPythonSpawner

    skip_graphviz_flux_test = "FLUX_URI" not in os.environ
except ImportError:
    skip_graphviz_flux_test = True


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


@unittest.skipIf(
    skip_graphviz_flux_test,
    "Either graphviz or flux are not installed, so the plot_dependency_graph tests are skipped.",
)
class TestFluxAllocationExecutorWithDependencies(unittest.TestCase):
    def test_executor_dependency_plot(self):
        with FluxJobExecutor(
            max_cores=1,
            plot_dependency_graph=True,
            block_allocation=False,
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

    def test_many_to_one_plot(self):
        length = 5
        parameter = 1
        with FluxJobExecutor(
            max_cores=2,
            plot_dependency_graph=True,
            block_allocation=True,
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
            self.assertEqual(len(exe._future_hash_dict), 7)
            self.assertEqual(len(exe._task_hash_dict), 7)
            nodes, edges = generate_nodes_and_edges(
                task_hash_dict=exe._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 19)
            self.assertEqual(len(edges), 22)


@unittest.skipIf(
    skip_graphviz_flux_test,
    "Either graphviz or flux are not installed, so the plot_dependency_graph tests are skipped.",
)
class TestFluxSubmissionExecutorWithDependencies(unittest.TestCase):
    def test_executor_dependency_plot(self):
        with FluxClusterExecutor(
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

    def test_many_to_one_plot(self):
        length = 5
        parameter = 1
        with FluxClusterExecutor(
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
            self.assertEqual(len(exe._future_hash_dict), 7)
            self.assertEqual(len(exe._task_hash_dict), 7)
            nodes, edges = generate_nodes_and_edges(
                task_hash_dict=exe._task_hash_dict,
                future_hash_inverse_dict={
                    v: k for k, v in exe._future_hash_dict.items()
                },
            )
            self.assertEqual(len(nodes), 19)
            self.assertEqual(len(edges), 22)

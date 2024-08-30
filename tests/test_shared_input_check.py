import unittest

from executorlib.shared.inputcheck import (
    check_command_line_argument_lst,
    check_gpus_per_worker,
    check_threads_per_core,
    check_oversubscribe,
    check_executor,
    check_init_function,
    check_nested_flux_executor,
    check_pmi,
    check_plot_dependency_graph,
    check_refresh_rate,
    check_resource_dict,
    check_resource_dict_is_empty,
)


class TestInputCheck(unittest.TestCase):
    def test_check_command_line_argument_lst(self):
        with self.assertRaises(ValueError):
            check_command_line_argument_lst(command_line_argument_lst=["a"])

    def test_check_gpus_per_worker(self):
        with self.assertRaises(TypeError):
            check_gpus_per_worker(gpus_per_worker=1)

    def test_check_threads_per_core(self):
        with self.assertRaises(TypeError):
            check_threads_per_core(threads_per_core=2)

    def test_check_oversubscribe(self):
        with self.assertRaises(ValueError):
            check_oversubscribe(oversubscribe=True)

    def test_check_executor(self):
        with self.assertRaises(ValueError):
            check_executor(executor=1)

    def test_check_init_function(self):
        with self.assertRaises(ValueError):
            check_init_function(init_function=1, block_allocation=False)

    def test_check_refresh_rate(self):
        with self.assertRaises(ValueError):
            check_refresh_rate(refresh_rate=1)

    def test_check_resource_dict(self):
        def simple_function(resource_dict):
            return resource_dict

        with self.assertRaises(ValueError):
            check_resource_dict(function=simple_function)

    def test_check_resource_dict_is_empty(self):
        with self.assertRaises(ValueError):
            check_resource_dict_is_empty(resource_dict={"a": 1})

    def test_check_pmi(self):
        with self.assertRaises(ValueError):
            check_pmi(backend="test", pmi="test")
        with self.assertRaises(ValueError):
            check_pmi(backend="flux", pmi="test")

    def test_check_nested_flux_executor(self):
        with self.assertRaises(ValueError):
            check_nested_flux_executor(nested_flux_executor=True)

    def test_check_plot_dependency_graph(self):
        with self.assertRaises(ValueError):
            check_plot_dependency_graph(plot_dependency_graph=True)

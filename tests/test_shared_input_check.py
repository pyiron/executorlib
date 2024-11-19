import unittest

from executorlib.standalone.inputcheck import (
    check_command_line_argument_lst,
    check_gpus_per_worker,
    check_oversubscribe,
    check_executor,
    check_init_function,
    check_nested_flux_executor,
    check_pmi,
    check_plot_dependency_graph,
    check_refresh_rate,
    check_resource_dict,
    check_resource_dict_is_empty,
    check_flux_executor_pmi_mode,
    check_max_workers_and_cores,
    check_hostname_localhost,
    check_pysqa_config_directory,
    check_file_exists,
    validate_number_of_cores,
)


class TestInputCheck(unittest.TestCase):
    def test_check_command_line_argument_lst(self):
        with self.assertRaises(ValueError):
            check_command_line_argument_lst(command_line_argument_lst=["a"])

    def test_check_gpus_per_worker(self):
        with self.assertRaises(TypeError):
            check_gpus_per_worker(gpus_per_worker=1)

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
            check_pmi(backend="flux_allocation", pmi="test")

    def test_check_nested_flux_executor(self):
        with self.assertRaises(ValueError):
            check_nested_flux_executor(nested_flux_executor=True)

    def test_check_plot_dependency_graph(self):
        with self.assertRaises(ValueError):
            check_plot_dependency_graph(plot_dependency_graph=True)

    def test_check_flux_executor_pmi_mode(self):
        with self.assertRaises(ValueError):
            check_flux_executor_pmi_mode(flux_executor_pmi_mode="test")

    def test_check_max_workers_and_cores(self):
        with self.assertRaises(ValueError):
            check_max_workers_and_cores(max_workers=2, max_cores=None)
        with self.assertRaises(ValueError):
            check_max_workers_and_cores(max_workers=None, max_cores=2)
        with self.assertRaises(ValueError):
            check_max_workers_and_cores(max_workers=2, max_cores=2)

    def test_check_hostname_localhost(self):
        with self.assertRaises(ValueError):
            check_hostname_localhost(hostname_localhost=True)
        with self.assertRaises(ValueError):
            check_hostname_localhost(hostname_localhost=False)

    def test_check_pysqa_config_directory(self):
        with self.assertRaises(ValueError):
            check_pysqa_config_directory(pysqa_config_directory="path/to/config")

    def test_check_file_exists(self):
        with self.assertRaises(ValueError):
            check_file_exists(file_name=None)
        with self.assertRaises(ValueError):
            check_file_exists(file_name="/path/does/not/exist")

    def test_validate_number_of_cores(self):
        with self.assertRaises(ValueError):
            validate_number_of_cores(
                max_cores=None, max_workers=None, cores_per_worker=None
            )
        with self.assertRaises(TypeError):
            validate_number_of_cores(
                max_cores=1, max_workers=None, cores_per_worker=None
            )
        self.assertIsInstance(
            validate_number_of_cores(max_cores=1, max_workers=None, cores_per_worker=1),
            int,
        )
        self.assertIsInstance(
            validate_number_of_cores(
                max_cores=None, max_workers=1, cores_per_worker=None
            ),
            int,
        )

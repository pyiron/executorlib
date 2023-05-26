import unittest
from pympipool.common import parse_arguments, command_line_options


class TestParser(unittest.TestCase):
    def test_command(self):
        result_dict = {
            'host': 'localhost',
            'total_cores': '2',
            'zmqport': '22',
            'cores_per_task': '2'
        }
        command_lst = [
            'mpiexec', '-n', '1', 'python', '/',
            '--zmqport', result_dict['zmqport'],
            '--cores-per-task', result_dict['cores_per_task'],
            '--cores-total', result_dict['total_cores']
        ]
        self.assertEqual(command_lst, command_line_options(
            hostname="127.0.0.1",
            port_selected=result_dict['zmqport'],
            path="/",
            cores=result_dict['total_cores'],
            cores_per_task=result_dict['cores_per_task'],
            oversubscribe=False,
            enable_flux_backend=False,
        ))
        self.assertEqual(result_dict, parse_arguments(command_lst))

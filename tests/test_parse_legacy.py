import unittest
from pympipool.legacy.shared.backend import parse_arguments
from pympipool.legacy.shared.interface import command_line_options


class TestParser(unittest.TestCase):
    def test_command_local(self):
        result_dict = {
            'host': 'localhost',
            'total_cores': '2',
            'zmqport': '22',
            'cores_per_task': '1'
        }
        command_lst = [
            'mpiexec', '--oversubscribe',
            '-n', result_dict['total_cores'],
            'python', '-m', 'mpi4py.futures', '/',
            '--zmqport', result_dict['zmqport'],
            '--cores-per-task', result_dict['cores_per_task'],
            '--cores-total', result_dict['total_cores']
        ]
        self.assertEqual(command_lst, command_line_options(
            hostname=result_dict['host'],
            port_selected=result_dict['zmqport'],
            path="/",
            cores=int(result_dict['total_cores']),
            cores_per_task=int(result_dict['cores_per_task']),
            oversubscribe=True,
            enable_flux_backend=False,
        ))
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_flux(self):
        result_dict = {
            'host': "127.0.0.1",
            'total_cores': '2',
            'zmqport': '22',
            'cores_per_task': '2'
        }
        command_lst = [
            'flux', 'run', '-n', '1', 'python', '/',
            '--host', result_dict['host'],
            '--zmqport', result_dict['zmqport'],
            '--cores-per-task', result_dict['cores_per_task'],
            '--cores-total', result_dict['total_cores']
        ]
        self.assertEqual(command_lst, command_line_options(
            hostname=result_dict['host'],
            port_selected=result_dict['zmqport'],
            path="/",
            cores=int(result_dict['total_cores']),
            cores_per_task=int(result_dict['cores_per_task']),
            oversubscribe=False,
            enable_flux_backend=True,
        ))
        self.assertEqual(result_dict, parse_arguments(command_lst))

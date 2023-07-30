import unittest
import os
from pympipool.legacy.shared.backend import parse_arguments
from pympipool.shared.connections import MpiExecInterface, FluxCmdInterface


class TestParser(unittest.TestCase):
    def test_command_local(self):
        result_dict = {
            'host': 'localhost',
            'total_cores': '2',
            'zmqport': '22',
            'cores_per_task': '1'
        }
        command_lst = [
            'mpiexec',
            '-n', result_dict['total_cores'],
            '--oversubscribe',
            'python', '-m', 'mpi4py.futures', '/',
            '--zmqport', result_dict['zmqport'],
            '--cores-per-task', result_dict['cores_per_task'],
            '--cores-total', result_dict['total_cores']
        ]
        interface = MpiExecInterface(
            cwd=None,
            cores=2,
            gpus_per_core=0,
            oversubscribe=True
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[
                    'python', '-m', 'mpi4py.futures', '/',
                    '--zmqport', result_dict['zmqport'],
                    '--cores-per-task', '1', '--cores-total', '2'
                ]
            )
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_flux(self):
        result_dict = {
            'host': "127.0.0.1",
            'total_cores': '2',
            'zmqport': '22',
            'cores_per_task': '2'
        }
        command_lst = [
            'flux', 'run', '-n', '1',
            "--cwd=" + os.path.abspath("."),
            'python', '/',
            '--host', result_dict['host'],
            '--zmqport', result_dict['zmqport'],
            '--cores-per-task', result_dict['cores_per_task'],
            '--cores-total', result_dict['total_cores']
        ]
        interface = FluxCmdInterface(
            cwd=os.path.abspath("."),
            cores=1,
            gpus_per_core=0,
            oversubscribe=False
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[
                    'python', '/', '--host', result_dict['host'],
                    '--zmqport', result_dict['zmqport'],
                    '--cores-per-task', '2', '--cores-total', '2'
                ]
            )
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

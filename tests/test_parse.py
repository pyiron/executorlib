import os
import unittest
from pympipool.shared.backend import parse_arguments
from pympipool.shared.connections import MpiExecInterface, FluxCmdInterface, SlurmSubprocessInterface


class TestParser(unittest.TestCase):
    def test_command_local(self):
        result_dict = {
            'host': 'localhost',
            'zmqport': '22',
        }
        command_lst = [
            'mpiexec',
            '-n', '2',
            '--oversubscribe',
            'python', '/',
            '--zmqport', result_dict['zmqport']
        ]
        interface = MpiExecInterface(cwd=None, cores=2, gpus_per_core=0, oversubscribe=True)
        self.assertEqual(
            command_lst,
            interface.generate_command(command_lst=['python', '/', '--zmqport', result_dict['zmqport']])
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_flux(self):
        result_dict = {
            'host': "127.0.0.1",
            'zmqport': '22',
        }
        command_lst = [
            'flux', 'run', '-n', '2',
            "--cwd=" + os.path.abspath("."),
            '--gpus-per-task=1',
            'python', '/',
            '--host', result_dict['host'],
            '--zmqport', result_dict['zmqport']
        ]
        interface = FluxCmdInterface(cwd=os.path.abspath("."), cores=2, gpus_per_core=1, oversubscribe=False)
        self.assertEqual(
            command_lst,
            interface.generate_command(command_lst=['python', '/', '--host', result_dict['host'], '--zmqport', result_dict['zmqport']])
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_mpiexec_gpu(self):
        interface = MpiExecInterface(cwd=os.path.abspath("."), cores=2, gpus_per_core=1, oversubscribe=True)
        with self.assertRaises(ValueError):
            interface.bootup(command_lst=[])

    def test_command_slurm(self):
        result_dict = {
            'host': "127.0.0.1",
            'zmqport': '22',
        }
        command_lst = [
            'srun', '-n', '2',
            "-D", os.path.abspath("."),
            '--gpus-per-task=1',
            '--oversubscribe',
            'python', '/',
            '--host', result_dict['host'],
            '--zmqport', result_dict['zmqport']
        ]
        interface = SlurmSubprocessInterface(cwd=os.path.abspath("."), cores=2, gpus_per_core=1, oversubscribe=True)
        self.assertEqual(
            command_lst,
            interface.generate_command(command_lst=['python', '/', '--host', result_dict['host'], '--zmqport', result_dict['zmqport']])
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

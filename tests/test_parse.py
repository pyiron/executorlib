import os
import sys
import unittest
from pympipool.shared.backend import parse_arguments
from pympipool.shared.interface import SrunInterface, MpiExecInterface


class TestParser(unittest.TestCase):
    def test_command_local(self):
        result_dict = {
            "host": "localhost",
            "zmqport": "22",
        }
        command_lst = [
            "mpiexec",
            "-n",
            "2",
            "--oversubscribe",
            sys.executable,
            "/",
            "--zmqport",
            result_dict["zmqport"],
        ]
        interface = MpiExecInterface(
            cwd=None, cores=2, oversubscribe=True
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[sys.executable, "/", "--zmqport", result_dict["zmqport"]]
            ),
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_slurm(self):
        result_dict = {
            "host": "127.0.0.1",
            "zmqport": "22",
        }
        command_lst = [
            "srun",
            "-n",
            "2",
            "-D",
            os.path.abspath("."),
            "--gpus-per-task=1",
            "--oversubscribe",
            sys.executable,
            "/",
            "--host",
            result_dict["host"],
            "--zmqport",
            result_dict["zmqport"],
        ]
        interface = SrunInterface(
            cwd=os.path.abspath("."), cores=2, gpus_per_core=1, oversubscribe=True
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[
                    sys.executable,
                    "/",
                    "--host",
                    result_dict["host"],
                    "--zmqport",
                    result_dict["zmqport"],
                ]
            ),
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

import os
import sys

import numpy as np
import unittest
from pympipool.shared.communication import SocketInterface
from pympipool.shared.executorbase import cloudpickle_register
from pympipool.shared.interface import MpiExecInterface


def calc(i):
    return np.array(i**2)


class TestInterface(unittest.TestCase):
    def test_interface(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            interface=MpiExecInterface(
                cwd=None, cores=1, oversubscribe=False
            )
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        __file__, "..", "..", "pympipool", "backend", "mpiexec.py"
                    )
                ),
                "--zmqport",
                str(interface.bind_to_random_port()),
            ]
        )
        self.assertEqual(
            interface.send_and_receive_dict(input_dict=task_dict), np.array(4)
        )
        interface.shutdown(wait=True)

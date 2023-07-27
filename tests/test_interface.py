import os
import socket

import numpy as np
import unittest
from pympipool.shared.communication import SocketInterface
from pympipool.shared.taskexecutor import command_line_options, cloudpickle_register


def calc(i):
    return np.array(i ** 2)


class TestInterface(unittest.TestCase):
    def test_interface(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, 'args': (), "kwargs": {"i": 2}}
        interface = SocketInterface(queue_adapter=None, queue_adapter_kwargs=None)
        interface.bootup(
            command_lst=command_line_options(
                hostname=socket.gethostname(),
                port_selected=interface.bind_to_random_port(),
                path=os.path.abspath(os.path.join(__file__, "..", "..", "pympipool", "backend", "mpiexec.py")),
                cores=1,
                gpus_per_task=0,
                oversubscribe=False,
                enable_flux_backend=False,
                enable_slurm_backend=False,
                enable_multi_host=False,
            )
        )
        self.assertEqual(interface.send_and_receive_dict(input_dict=task_dict), np.array(4))
        interface.shutdown(wait=True)
import os
import sys
import unittest

import numpy as np
import zmq

from pympipool.shared.communication import (
    interface_connect,
    interface_shutdown,
    interface_send,
    interface_receive,
    SocketInterface,
)
from pympipool.shared.executorbase import cloudpickle_register
from pympipool.shared.interface import MpiExecInterface


def calc(i):
    return np.array(i**2)


class TestInterface(unittest.TestCase):
    def test_interface(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            interface=MpiExecInterface(cwd=None, cores=1, oversubscribe=False)
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


class TestZMQ(unittest.TestCase):
    def test_initialize_zmq(self):
        message = "test"
        host = "localhost"

        context_server = zmq.Context()
        socket_server = context_server.socket(zmq.PAIR)
        port = str(socket_server.bind_to_random_port("tcp://*"))
        context_client, socket_client = interface_connect(host=host, port=port)
        interface_send(socket=socket_server, result_dict={"message": message})
        self.assertEqual(interface_receive(socket=socket_client), {"message": message})
        interface_shutdown(socket=socket_client, context=context_client)
        interface_shutdown(socket=socket_server, context=context_server)

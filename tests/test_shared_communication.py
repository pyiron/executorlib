import importlib.util
import os
import sys
import unittest

import numpy as np
import zmq

from executorlib.standalone.interactive.communication import (
    interface_connect,
    interface_shutdown,
    interface_send,
    interface_receive,
    SocketInterface,
)
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.standalone.interactive.spawner import MpiExecSpawner


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def calc(i):
    return np.array(i**2)


class TestInterface(unittest.TestCase):
    @unittest.skipIf(
        skip_mpi4py_test, "mpi4py is not installed, so the mpi4py tests are skipped."
    )
    def test_interface_mpi(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False)
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        __file__,
                        "..",
                        "..",
                        "executorlib",
                        "backend",
                        "interactive_parallel.py",
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

    def test_interface_serial(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False)
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        __file__,
                        "..",
                        "..",
                        "executorlib",
                        "backend",
                        "interactive_serial.py",
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

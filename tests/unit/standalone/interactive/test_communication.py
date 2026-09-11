import importlib.util
import os
import sys
import unittest
from time import sleep
from typing import Callable, Optional

import numpy as np
import zmq

from executorlib.standalone.interactive.communication import (
    interface_connect,
    interface_shutdown,
    interface_send,
    interface_receive,
    SocketInterface,
    ExecutorlibSocketError,
)
from executorlib.standalone.serialize import cloudpickle_register
from executorlib.standalone.interactive.spawner import MpiExecSpawner


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


def calc(i):
    return np.array(i**2)


class BrokenSpawner(MpiExecSpawner):
    def bootup(self, command_lst: list[str], stop_function: Optional[Callable] = None,):
        return False


class DelayedExitSpawner(MpiExecSpawner):
    """Spawner that reports the process as alive for the first poll() call only,
    emulating a worker which exits while shutdown() is waiting for its reply."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._poll_call_count = 0

    def poll(self) -> bool:
        self._poll_call_count += 1
        return self._poll_call_count == 1


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
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "src",
                        "executorlib",
                        "backend",
                        "interactive_parallel.py",
                    )
                ),
                "--zmqport",
                str(interface.bind_to_random_port()),
            ]
        )
        self.assertTrue(interface.status)
        self.assertEqual(
            interface.send_and_receive_dict(input_dict=task_dict)["result"], np.array(4)
        )
        interface.shutdown(wait=True)

    def test_interface_serial_without_debug(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=False,
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "src",
                        "executorlib",
                        "backend",
                        "interactive_serial.py",
                    )
                ),
                "--zmqport",
                str(interface.bind_to_random_port()),
            ]
        )
        self.assertTrue(interface.status)
        self.assertEqual(
            interface.send_and_receive_dict(input_dict=task_dict)["result"], np.array(4)
        )
        interface.shutdown(wait=True)

    def test_interface_serial_with_debug(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=True,
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "src",
                        "executorlib",
                        "backend",
                        "interactive_serial.py",
                    )
                ),
                "--zmqport",
                str(interface.bind_to_random_port()),
            ]
        )
        self.assertTrue(interface.status)
        self.assertEqual(
            interface.send_and_receive_dict(input_dict=task_dict)["result"], np.array(4)
        )
        interface.shutdown(wait=True)

    def test_interface_serial_with_error(self):
        cloudpickle_register(ind=1)
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=True,
        )
        interface.bootup(command_lst=["bash", "exit"])
        self.assertTrue(interface.status)
        while interface._spawner.poll():
            sleep(0.1)
        self.assertFalse(interface._spawner.poll())
        interface.shutdown(wait=True)

    def test_interface_shutdown_with_process_exiting_during_wait(self):
        cloudpickle_register(ind=1)
        interface = SocketInterface(
            spawner=DelayedExitSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=False,
            time_out_ms=100,
        )
        port = interface.bind_to_random_port()
        # Connect a peer so the PAIR socket is not in the ZMQ "mute state" and
        # send_dict() does not block forever. The peer never replies, emulating
        # a worker process that exits while shutdown() is waiting for its reply.
        context, socket = interface_connect(host="localhost", port=str(port))
        try:
            self.assertIsNone(interface.shutdown(wait=True))
        finally:
            socket.close()
            context.term()

    def test_interface_serial_wrong_input(self):
        cloudpickle_register(ind=1)
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=True,
        )
        with self.assertRaises(ValueError):
            interface.bootup(command_lst=None)

    def test_interface_serial_with_broken_spawner(self):
        cloudpickle_register(ind=1)
        interface = SocketInterface(
            spawner=BrokenSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=True,
        )
        interface.bootup(command_lst=["bash", "exit"])
        self.assertFalse(interface.status)

    def test_interface_serial_with_stopped_process(self):
        cloudpickle_register(ind=1)
        task_dict = {"fn": calc, "args": (), "kwargs": {"i": 2}}
        interface = SocketInterface(
            spawner=MpiExecSpawner(cwd=None, cores=1, openmpi_oversubscribe=False),
            log_obj_size=True,
        )
        interface.bootup(
            command_lst=[
                sys.executable,
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        "..",
                        "..",
                        "..",
                        "src",
                        "executorlib",
                        "backend",
                        "interactive_serial.py",
                    )
                ),
                "--zmqport",
                str(interface.bind_to_random_port()),
            ]
        )
        self.assertTrue(interface.status)
        interface.send_dict(input_dict=task_dict)
        interface._spawner._process.terminate()
        output = interface.receive_dict()
        self.assertIsInstance(output["error"], ExecutorlibSocketError)


class TestZMQ(unittest.TestCase):
    def test_zmq_interface_receive_message(self):
        self.assertEqual(len(interface_receive(socket=None)), 0)

    def test_zmq_client_initializes_socket(self):
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

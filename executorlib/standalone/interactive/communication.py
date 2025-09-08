import logging
import sys
from socket import gethostname
from typing import Any, Callable, Optional

import cloudpickle
import zmq


class ExecutorlibSocketError(RuntimeError):
    pass


class SocketInterface:
    """
    The SocketInterface is an abstraction layer on top of the zero message queue.

    Args:
        spawner (executorlib.shared.spawner.BaseSpawner): Interface for starting the parallel process
        log_obj_size (boolean): Enable debug mode which reports the size of the communicated objects.
        time_out_ms (int): Time out for waiting for a message on socket in milliseconds.
    """

    def __init__(
        self, spawner=None, log_obj_size: bool = False, time_out_ms: int = 1000
    ):
        """
        Initialize the SocketInterface.

        Args:
            spawner (executorlib.shared.spawner.BaseSpawner): Interface for starting the parallel process
            log_obj_size (boolean): Enable debug mode which reports the size of the communicated objects.
            time_out_ms (int): Time out for waiting for a message on socket in milliseconds.
        """
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)
        self._process = None
        self._time_out_ms = time_out_ms
        self._logger: Optional[logging.Logger] = None
        if log_obj_size:
            self._logger = logging.getLogger("executorlib")
        self._spawner = spawner
        self._command_lst: list[str] = []
        self._booted_sucessfully: bool = False
        self._stop_function: Optional[Callable] = None

    @property
    def status(self) -> bool:
        return self._booted_sucessfully

    @status.setter
    def status(self, status: bool):
        self._booted_sucessfully = status

    def send_dict(self, input_dict: dict):
        """
        Send a dictionary with instructions to a connected client process.

        Args:
            input_dict (dict): dictionary of commands to be communicated. The key "shutdown" is reserved to stop the
                connected client from listening.
        """
        data = cloudpickle.dumps(input_dict)
        if self._logger is not None:
            self._logger.warning("Send dictionary of size: " + str(sys.getsizeof(data)))
        self._socket.send(data)

    def receive_dict(self) -> dict:
        """
        Receive a dictionary from a connected client process.

        Returns:
            dict: dictionary with response received from the connected client
        """
        response_lst: list[tuple[Any, int]] = []
        while len(response_lst) == 0:
            response_lst = self._poller.poll(self._time_out_ms)
            if not self._spawner.poll():
                raise ExecutorlibSocketError(
                    "SocketInterface crashed during execution."
                )
        data = self._socket.recv(zmq.NOBLOCK)
        if self._logger is not None:
            self._logger.warning(
                "Received dictionary of size: " + str(sys.getsizeof(data))
            )
        output = cloudpickle.loads(data)
        if "result" in output:
            return output["result"]
        else:
            raise output["error"]

    def send_and_receive_dict(self, input_dict: dict) -> dict:
        """
        Combine both the send_dict() and receive_dict() function in a single call.

        Args:
            input_dict (dict): dictionary of commands to be communicated. The key "shutdown" is reserved to stop the
                               connected client from listening.

        Returns:
            dict: dictionary with response received from the connected client
        """
        self.send_dict(input_dict=input_dict)
        return self.receive_dict()

    def bind_to_random_port(self) -> int:
        """
        Identify a random port typically in the range from 49152 to 65536 to bind the SocketInterface instance to. Other
        processes can then connect to this port to receive instructions and send results.

        Returns:
            int: port the SocketInterface instance is bound to.
        """
        return self._socket.bind_to_random_port("tcp://*")

    def bootup(
        self,
        command_lst: Optional[list[str]] = None,
        stop_function: Optional[Callable] = None,
    ):
        """
        Boot up the client process to connect to the SocketInterface.

        Args:
            command_lst (list): list of strings to start the client process
            stop_function (Callable): Function to stop the interface.
        """
        if command_lst is not None:
            self._command_lst = command_lst
        if stop_function is not None:
            self._stop_function = stop_function
        if len(self._command_lst) == 0:
            raise ValueError("No command defined to boot up SocketInterface.")
        if not self._spawner.bootup(
            command_lst=self._command_lst,
            stop_function=self._stop_function,
        ):
            self._reset_socket()
            self._booted_sucessfully = False
        else:
            self._booted_sucessfully = True

    def shutdown(self, wait: bool = True):
        """
        Shutdown the SocketInterface and the connected client process.

        Args:
            wait (bool): Whether to wait for the client process to finish before returning. Default is True.
        """
        result = None
        if self._spawner.poll():
            result = self.send_and_receive_dict(
                input_dict={"shutdown": True, "wait": wait}
            )
            self._spawner.shutdown(wait=wait)
        self._reset_socket()
        return result

    def _reset_socket(self):
        """
        Reset the socket and context of the SocketInterface instance.
        """
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self._process = None
        self._socket = None
        self._context = None

    def __del__(self):
        """
        Destructor for the SocketInterface class.
        Calls the shutdown method with wait=True to ensure proper cleanup.
        """
        self.shutdown(wait=True)


def interface_bootup(
    command_lst: list[str],
    connections,
    hostname_localhost: Optional[bool] = None,
    log_obj_size: bool = False,
    worker_id: Optional[int] = None,
    stop_function: Optional[Callable] = None,
) -> SocketInterface:
    """
    Start interface for ZMQ communication

    Args:
        command_lst (list): List of commands as strings
        connections (executorlib.shared.spawner.BaseSpawner): Interface to start parallel process, like MPI, SLURM
                                                                  or Flux
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true
        log_obj_size (boolean): Enable debug mode which reports the size of the communicated objects.
        worker_id (int): Communicate the worker which ID was assigned to it for future reference and resource
                         distribution.
        stop_function (Callable): Function to stop the interface.

    Returns:
         executorlib.shared.communication.SocketInterface: socket interface for zmq communication
    """
    if hostname_localhost is None and sys.platform != "darwin":
        hostname_localhost = False
    if not hostname_localhost:
        command_lst += [
            "--host",
            gethostname(),
        ]
    if worker_id is not None:
        command_lst += ["--worker-id", str(worker_id)]
    interface = SocketInterface(
        spawner=connections,
        log_obj_size=log_obj_size,
    )
    command_lst += [
        "--zmqport",
        str(interface.bind_to_random_port()),
    ]
    interface.bootup(
        command_lst=command_lst,
        stop_function=stop_function,
    )
    return interface


def interface_connect(host: str, port: str) -> tuple[zmq.Context, zmq.Socket]:
    """
    Connect to an existing SocketInterface instance by providing the hostname and the port as strings.

    Args:
        host (str): hostname of the host running the SocketInterface instance to connect to.
        port (str): port on the host the SocketInterface instance is running on.
    """
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://" + host + ":" + port)
    return context, socket


def interface_send(socket: Optional[zmq.Socket], result_dict: dict):
    """
    Send results to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        result_dict (dict): dictionary to be sent, supported keys are result and error.
    """
    if socket is not None:
        socket.send(cloudpickle.dumps(result_dict))


def interface_receive(socket: Optional[zmq.Socket]) -> dict:
    """
    Receive instructions from a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
    """
    if socket is not None:
        return cloudpickle.loads(socket.recv())
    else:
        return {}


def interface_shutdown(socket: Optional[zmq.Socket], context: Optional[zmq.Context]):
    """
    Close the connection to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        context (zmq.sugar.context.Context): context for the connection
    """
    if socket is not None and context is not None:
        socket.close()
        context.term()

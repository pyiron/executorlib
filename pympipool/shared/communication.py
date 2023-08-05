from socket import gethostname

import cloudpickle
import zmq


class SocketInterface(object):
    """
    The SocketInterface is an abstraction layer on top of the zero message queue.

    Args:
        interface (pympipool.shared.interface.BaseInterface): Interface for starting the parallel process
    """

    def __init__(self, interface=None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._process = None
        self._interface = interface

    def send_dict(self, input_dict):
        """
        Send a dictionary with instructions to a connected client process.

        Args:
            input_dict (dict): dictionary of commands to be communicated. The key "shutdown" is reserved to stop the
                connected client from listening.
        """
        self._socket.send(cloudpickle.dumps(input_dict))

    def receive_dict(self):
        """
        Receive a dictionary from a connected client process.

        Returns:
            dict: dictionary with response received from the connected client
        """
        output = cloudpickle.loads(self._socket.recv())
        if "result" in output.keys():
            return output["result"]
        else:
            error_type = output["error_type"].split("'")[1]
            raise eval(error_type)(output["error"])

    def send_and_receive_dict(self, input_dict):
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

    def bind_to_random_port(self):
        """
        Identify a random port typically in the range from 49152 to 65536 to bind the SocketInterface instance to. Other
        processes can then connect to this port to receive instructions and send results.

        Returns:
            int: port the SocketInterface instance is bound to.
        """
        return self._socket.bind_to_random_port("tcp://*")

    def bootup(self, command_lst):
        """
        Boot up the client process to connect to the SocketInterface.

        Args:
            command_lst (list): list of strings to start the client process
        """
        self._interface.bootup(command_lst=command_lst)

    def shutdown(self, wait=True):
        result = None
        if self._interface.poll():
            result = self.send_and_receive_dict(
                input_dict={"shutdown": True, "wait": wait}
            )
            self._interface.shutdown(wait=wait)
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self._process = None
        self._socket = None
        self._context = None
        return result

    def __del__(self):
        self.shutdown(wait=True)


def interface_bootup(
    command_lst,
    connections,
):
    command_lst += [
        "--host",
        gethostname(),
    ]
    interface = SocketInterface(interface=connections)
    command_lst += [
        "--zmqport",
        str(interface.bind_to_random_port()),
    ]
    interface.bootup(command_lst=command_lst)
    return interface


def interface_connect(host, port):
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


def interface_send(socket, result_dict):
    """
    Send results to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        result_dict (dict): dictionary to be sent, supported keys are result, error and error_type.
    """
    socket.send(cloudpickle.dumps(result_dict))


def interface_receive(socket):
    """
    Receive instructions from a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
    """
    return cloudpickle.loads(socket.recv())


def interface_shutdown(socket, context):
    """
    Close the connection to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        context (zmq.sugar.context.Context): context for the connection
    """
    socket.close()
    context.term()

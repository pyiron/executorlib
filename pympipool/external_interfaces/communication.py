import subprocess

import cloudpickle
import zmq


class SocketInterface(object):
    """
    The SocketInterface is an abstraction layer on top of the zero message queue.

    Args:
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter
    """

    def __init__(self, queue_adapter=None, queue_adapter_kwargs=None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._process = None
        self._queue_adapter = queue_adapter
        self._queue_adapter_kwargs = queue_adapter_kwargs

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

    def bootup(self, command_lst, cwd=None, cores=None):
        """
        Boot up the client process to connect to the SocketInterface.

        Args:
            command_lst (list): list of strings to start the client process
            cwd (str/None): current working directory where the parallel python task is executed
            cores (str/ None): if the job is submitted to a queuing system using the pysqa.queueadapter.QueueAdapter
                then cores defines the number of cores to be used for the specific queuing system allocation to execute
                the client process.
        """
        if self._queue_adapter is not None:
            self._queue_adapter.submit_job(
                working_directory=cwd,
                cores=cores,
                command=" ".join(command_lst),
                **self._queue_adapter_kwargs
            )
        else:
            self._process = subprocess.Popen(
                args=command_lst,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                cwd=cwd,
            )

    def shutdown(self, wait=True):
        result = None
        if self._process is not None and self._process.poll() is None:
            result = self.send_and_receive_dict(
                input_dict={"shutdown": True, "wait": wait}
            )
            self._process_close(wait=wait)
        elif self._queue_adapter is not None and self._socket is not None:
            result = self.send_and_receive_dict(
                input_dict={"shutdown": True, "wait": wait}
            )
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self._process = None
        self._socket = None
        self._context = None
        return result

    def _process_close(self, wait=True):
        self._process.terminate()
        self._process.stdout.close()
        self._process.stdin.close()
        self._process.stderr.close()
        if wait:
            self._process.wait()

    def __del__(self):
        self.shutdown(wait=True)


def connect_to_socket_interface(host, port):
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


def send_result(socket, result_dict):
    """
    Send results to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        result_dict (dict): dictionary to be sent, supported keys are result, error and error_type.
    """
    socket.send(cloudpickle.dumps(result_dict))


def receive_instruction(socket):
    """
    Receive instructions from a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
    """
    return cloudpickle.loads(socket.recv())


def close_connection(socket, context):
    """
    Close the connection to a SocketInterface instance.

    Args:
        socket (zmq.Socket): socket for the connection
        context (zmq.sugar.context.Context): context for the connection
    """
    socket.close()
    context.term()

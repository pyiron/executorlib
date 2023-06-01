import subprocess

import cloudpickle
import zmq


class SocketInterface(object):
    def __init__(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        self._process = None

    def send_dict(self, input_dict):
        self._socket.send(cloudpickle.dumps(input_dict))

    def receive_dict(self):
        output = cloudpickle.loads(self._socket.recv())
        if "r" in output.keys():
            return output["r"]
        else:
            error_type = output["et"].split("'")[1]
            raise eval(error_type)(output["e"])

    def send_and_receive_dict(self, input_dict):
        self.send_dict(input_dict=input_dict)
        return self.receive_dict()

    def bind_to_random_port(self):
        return self._socket.bind_to_random_port("tcp://*")

    def bootup(self, command_lst):
        self._process = subprocess.Popen(
            args=command_lst,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )

    def shutdown(self, wait=True):
        if self._process is not None and self._process.poll() is None:
            self.send_dict(input_dict={"c": "close"})
            self._process_close(wait=wait)
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self._process = None
        self._socket = None
        self._context = None

    def _process_close(self, wait=True):
        self._process.terminate()
        self._process.stdout.close()
        self._process.stdin.close()
        self._process.stderr.close()
        if wait:
            self._process.wait()

    def __del__(self):
        self.shutdown(wait=True)

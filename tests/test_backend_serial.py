import unittest
import cloudpickle
import zmq
from pympipool.backend.serial import main
from threading import Thread


def calc(i, j):
    return i + j


def set_global():
    return {"j": 5}


def submit(socket):
    socket.send(
        cloudpickle.dumps({"init": True, "fn": set_global, "args": (), "kwargs": {}})
    )
    socket.send(cloudpickle.dumps({"fn": calc, "args": (), "kwargs": {"i": 2}}))
    socket.send(cloudpickle.dumps({"shutdown": True, "wait": True}))


def submit_error(socket):
    socket.send(
        cloudpickle.dumps({"init": True, "fn": set_global, "args": (), "kwargs": {}})
    )
    socket.send(cloudpickle.dumps({"fn": calc, "args": (), "kwargs": {}}))
    socket.send(cloudpickle.dumps({"shutdown": True, "wait": True}))


class TestSerial(unittest.TestCase):
    def test_main_as_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=main, kwargs={"argument_lst": ["--zmqport", str(port)]})
        t.start()
        submit(socket=socket)
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": 7})
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": True})
        socket.close()
        context.term()

    def test_main_as_thread_error(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=main, kwargs={"argument_lst": ["--zmqport", str(port)]})
        t.start()
        submit_error(socket=socket)
        self.assertEqual(
            cloudpickle.loads(socket.recv())["error_type"], "<class 'TypeError'>"
        )
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": True})
        socket.close()
        context.term()

    def test_submit_as_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=submit, kwargs={"socket": socket})
        t.start()
        main(argument_lst=["--zmqport", str(port)])
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": 7})
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": True})
        socket.close()
        context.term()

    def test_submit_as_thread_error(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=submit_error, kwargs={"socket": socket})
        t.start()
        main(argument_lst=["--zmqport", str(port)])
        self.assertEqual(
            cloudpickle.loads(socket.recv())["error_type"], "<class 'TypeError'>"
        )
        self.assertEqual(cloudpickle.loads(socket.recv()), {"result": True})
        socket.close()
        context.term()

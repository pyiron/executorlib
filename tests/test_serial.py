import unittest
import cloudpickle
import zmq
from pympipool.backend.serial import main
from threading import Thread


def calc(i):
    return i


def submit_from_thread(socket):
    socket.send(cloudpickle.dumps({"fn": calc, 'args': (), "kwargs": {"i": 2}}))
    socket.send(cloudpickle.dumps({"shutdown": True, "wait": True}))


class TestSerial(unittest.TestCase):
    def test_main_as_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=main, kwargs={"argument_lst": ["--zmqport", str(port)]})
        t.start()
        socket.send(cloudpickle.dumps({"fn": calc, 'args': (), "kwargs": {"i": 2}}))
        self.assertEqual(cloudpickle.loads(socket.recv()), {'result': 2})
        socket.send(cloudpickle.dumps({"shutdown": True, "wait": True}))
        self.assertEqual(cloudpickle.loads(socket.recv()), {'result': True})
        socket.close()
        context.term()

    def test_submit_as_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        port = socket.bind_to_random_port("tcp://*")
        t = Thread(target=submit_from_thread, kwargs={"socket": socket})
        t.start()
        main(argument_lst=["--zmqport", str(port)])
        self.assertEqual(cloudpickle.loads(socket.recv()), {'result': 2})
        self.assertEqual(cloudpickle.loads(socket.recv()), {'result': True})
        socket.close()
        context.term()

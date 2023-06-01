import unittest
import zmq
import cloudpickle
from pympipool.share.parallel import initialize_zmq


class TestZMQ(unittest.TestCase):
    def test_initialize_zmq(self):
        message = "test"
        host = "localhost"

        context_server = zmq.Context()
        socket_server = context_server.socket(zmq.PAIR)
        port = str(socket_server.bind_to_random_port("tcp://*"))
        context_client, socket_client = initialize_zmq(host=host, port=port)
        socket_server.send(cloudpickle.dumps(message))
        self.assertEqual(cloudpickle.loads(socket_client.recv()), message)
        socket_client.close()
        context_client.term()
        socket_server.close()
        context_server.term()

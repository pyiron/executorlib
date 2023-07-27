import unittest
import zmq
from pympipool.shared.communication import (
    connect_to_socket_interface,
    close_connection,
    send_result,
    receive_instruction
)


class TestZMQ(unittest.TestCase):
    def test_initialize_zmq(self):
        message = "test"
        host = "localhost"

        context_server = zmq.Context()
        socket_server = context_server.socket(zmq.PAIR)
        port = str(socket_server.bind_to_random_port("tcp://*"))
        context_client, socket_client = connect_to_socket_interface(host=host, port=port)
        send_result(socket=socket_server, result_dict={"message": message})
        self.assertEqual(receive_instruction(socket=socket_client), {"message": message})
        close_connection(socket=socket_client, context=context_client)
        close_connection(socket=socket_server, context=context_server)

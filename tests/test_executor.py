import unittest
from concurrent.futures import ThreadPoolExecutor
from pympipool.legacy.shared.backend import map_funct, parse_socket_communication
from pympipool.shared.backend import call_funct


def function_multi_args(a, b):
    return a + b


class TestExecutor(unittest.TestCase):
    def test_exec_funct_single_core_map(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = map_funct(
                executor=executor,
                funct=sum,
                lst=[[1, 1], [2, 2]],
                cores_per_task=1,
                chunksize=1,
            )
        self.assertEqual(output, [2, 4])

    def test_exec_funct_single_core_starmap(self):
        with self.assertRaises(AttributeError):
            with ThreadPoolExecutor(max_workers=1) as executor:
                map_funct(
                    executor=executor,
                    funct=sum,
                    lst=[[1, 1], [2, 2]],
                    cores_per_task=1,
                    chunksize=1,
                    map_flag=False
                )

    def test_parse_socket_communication_close(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"shutdown": True, "wait": True},
                future_dict={},
                cores_per_task=1
            )
        self.assertEqual(output, {"exit": True})

    def test_parse_socket_communication_execute(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": sum, "iterable": [[1, 1]], "chunksize": 1, "map": True},
                future_dict={},
                cores_per_task=1
            )
        self.assertEqual(output, {"result": [2]})

    def test_parse_socket_communication_error(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": sum, "iterable": [["a", "b"]], "chunksize": 1, "map": True},
                future_dict={},
                cores_per_task=1
            )
        self.assertEqual(output["error_type"], "<class 'TypeError'>")

    def test_parse_socket_communication_submit_args(self):
        future_dict = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": sum, "args": [[1, 1]], "kwargs": {}},
                future_dict=future_dict,
                cores_per_task=1
            )
        future = future_dict[output['result']]
        self.assertEqual(future.result(), 2)

    def test_parse_socket_communication_submit_kwargs(self):
        future_dict = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": function_multi_args, "args": (), "kwargs": {"a": 1, "b": 2}},
                future_dict=future_dict,
                cores_per_task=1
            )
        future = future_dict[output['result']]
        self.assertEqual(future.result(), 3)

    def test_parse_socket_communication_submit_both(self):
        future_dict = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": function_multi_args, "args": [1], "kwargs": {"b": 2}},
                future_dict=future_dict,
                cores_per_task=1
            )
        future = future_dict[output['result']]
        self.assertEqual(future.result(), 3)

    def test_parse_socket_communication_update(self):
        future_dict = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": sum, "args": [[1, 1]], "kwargs": {}},
                future_dict=future_dict,
                cores_per_task=1
            )
            future_hash = output["result"]
            result = parse_socket_communication(
                executor=executor,
                input_dict={"update": [future_hash]},
                future_dict=future_dict,
                cores_per_task=1
            )
        self.assertEqual(result, {"result": {future_hash: 2}})

    def test_parse_socket_communication_cancel(self):
        future_dict = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            output = parse_socket_communication(
                executor=executor,
                input_dict={"fn": sum, "args": [[1, 1]], "kwargs": {}},
                future_dict=future_dict,
                cores_per_task=1
            )
            future_hash = output["result"]
            result = parse_socket_communication(
                executor=executor,
                input_dict={"cancel": [future_hash]},
                future_dict=future_dict,
                cores_per_task=1
            )
        self.assertEqual(result, {"result": True})

    def test_funct_call_default(self):
        self.assertEqual(call_funct(input_dict={
            "fn": sum,
            "args": [[1, 2, 3]],
            "kwargs": {}
        }), 6)

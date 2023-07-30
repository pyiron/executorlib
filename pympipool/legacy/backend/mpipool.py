from os.path import abspath
import pickle
import sys

import cloudpickle

from pympipool.shared.communication import (
    interface_connect,
    interface_send,
    interface_shutdown,
    interface_receive,
)
from pympipool.legacy.shared.backend import parse_socket_communication, parse_arguments


def main():
    from mpi4py import MPI

    MPI.pickle.__init__(
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )
    from mpi4py.futures import MPIPoolExecutor

    future_dict = {}
    argument_dict = parse_arguments(argument_lst=sys.argv)

    # required for flux interface - otherwise the current path is not included in the python path
    cwd = abspath(".")
    if cwd not in sys.path:
        sys.path.insert(1, cwd)

    with MPIPoolExecutor(
        max_workers=int(argument_dict["total_cores"]),
        path=sys.path,  # required for flux interface - otherwise the current path is not included in the python path
    ) as executor:
        if executor is not None:
            context, socket = interface_connect(
                host=argument_dict["host"], port=argument_dict["zmqport"]
            )
            while True:
                output = parse_socket_communication(
                    executor=executor,
                    input_dict=interface_receive(socket=socket),
                    future_dict=future_dict,
                    cores_per_task=int(argument_dict["cores_per_task"]),
                )
                if "exit" in output.keys() and output["exit"]:
                    if "result" in output.keys():
                        interface_send(
                            socket=socket, result_dict={"result": output["result"]}
                        )
                    else:
                        interface_send(socket=socket, result_dict={"result": True})
                    interface_shutdown(socket=socket, context=context)
                    break
                elif isinstance(output, dict):
                    interface_send(socket=socket, result_dict=output)


if __name__ == "__main__":
    main()

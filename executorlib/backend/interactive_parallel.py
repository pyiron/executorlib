import pickle
import sys
from os.path import abspath
from typing import Optional

import cloudpickle
import zmq

from executorlib.standalone.error import backend_write_error_file
from executorlib.standalone.interactive.backend import call_funct, parse_arguments
from executorlib.standalone.interactive.communication import (
    interface_connect,
    interface_receive,
    interface_send,
    interface_shutdown,
)


def main() -> None:
    """
    Entry point of the program.

    This function initializes MPI, sets up the necessary communication, and executes the requested functions.

    Returns:
        None
    """
    from mpi4py import MPI

    MPI.pickle.__init__(  # type: ignore
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )
    mpi_rank_zero = MPI.COMM_WORLD.Get_rank() == 0
    mpi_size_larger_one = MPI.COMM_WORLD.Get_size() > 1

    argument_dict = parse_arguments(argument_lst=sys.argv)
    context: Optional[zmq.Context] = None
    socket: Optional[zmq.Socket] = None
    if mpi_rank_zero:
        context, socket = interface_connect(
            host=argument_dict["host"], port=argument_dict["zmqport"]
        )

    memory = {"executorlib_worker_id": int(argument_dict["worker_id"])}

    # required for flux interface - otherwise the current path is not included in the python path
    cwd = abspath(".")
    if cwd not in sys.path:
        sys.path.insert(1, cwd)

    while True:
        # Read from socket
        input_dict: dict = {}
        if mpi_rank_zero:
            input_dict = interface_receive(socket=socket)
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)

        # Parse input
        if "shutdown" in input_dict and input_dict["shutdown"]:
            if mpi_rank_zero:
                interface_send(socket=socket, result_dict={"result": True})
                interface_shutdown(socket=socket, context=context)
            MPI.COMM_WORLD.Barrier()
            break
        elif (
            "fn" in input_dict
            and "init" not in input_dict
            and "args" in input_dict
            and "kwargs" in input_dict
        ):
            # Execute function
            try:
                output = call_funct(input_dict=input_dict, funct=None, memory=memory)
                if mpi_size_larger_one:
                    output_reply = MPI.COMM_WORLD.gather(output, root=0)
                else:
                    output_reply = output
            except Exception as error:
                if mpi_rank_zero:
                    interface_send(
                        socket=socket,
                        result_dict={"error": error},
                    )
                    backend_write_error_file(
                        error=error,
                        apply_dict=input_dict,
                    )
            else:
                # Send output
                if mpi_rank_zero:
                    interface_send(socket=socket, result_dict={"result": output_reply})
        elif (
            "init" in input_dict
            and input_dict["init"]
            and "args" in input_dict
            and "kwargs" in input_dict
        ):
            memory.update(call_funct(input_dict=input_dict, funct=None, memory=memory))


if __name__ == "__main__":
    main()

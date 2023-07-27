from os.path import abspath
import pickle
import sys

import cloudpickle

from pympipool.shared.communication import (
    connect_to_socket_interface,
    send_result,
    close_connection,
    receive_instruction,
)
from pympipool.shared.backend import call_funct, parse_arguments


def main():
    from mpi4py import MPI

    MPI.pickle.__init__(
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )
    mpi_rank_zero = MPI.COMM_WORLD.Get_rank() == 0
    mpi_size_larger_one = MPI.COMM_WORLD.Get_size() > 1

    argument_dict = parse_arguments(argument_lst=sys.argv)
    if mpi_rank_zero:
        context, socket = connect_to_socket_interface(
            host=argument_dict["host"], port=argument_dict["zmqport"]
        )
    else:
        context = None
        socket = None

    memory = None

    # required for flux interface - otherwise the current path is not included in the python path
    cwd = abspath(".")
    if cwd not in sys.path:
        sys.path.insert(1, cwd)

    while True:
        # Read from socket
        if mpi_rank_zero:
            input_dict = receive_instruction(socket=socket)
        else:
            input_dict = None
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)

        # Parse input
        if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
            if mpi_rank_zero:
                send_result(socket=socket, result_dict={"result": True})
                close_connection(socket=socket, context=context)
            break
        elif (
            "fn" in input_dict.keys()
            and "init" not in input_dict.keys()
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
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
                    send_result(
                        socket=socket,
                        result_dict={"error": error, "error_type": str(type(error))},
                    )
            else:
                # Send output
                if mpi_rank_zero:
                    send_result(socket=socket, result_dict={"result": output_reply})
        elif (
            "init" in input_dict.keys()
            and input_dict["init"]
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
        ):
            memory = call_funct(input_dict=input_dict, funct=None)


if __name__ == "__main__":
    main()

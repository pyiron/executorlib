import pickle
import sys

import cloudpickle

from pympipool.share.parallel import call_funct, initialize_zmq, parse_arguments


def main():
    from mpi4py import MPI

    MPI.pickle.__init__(
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )
    mpi_rank_zero = MPI.COMM_WORLD.Get_rank() == 0

    argument_dict = parse_arguments(argument_lst=sys.argv)
    if mpi_rank_zero:
        context, socket = initialize_zmq(
            host=argument_dict["host"], port=argument_dict["zmqport"]
        )
    else:
        context = None
        socket = None

    memory = None
    while True:
        # Read from socket
        if mpi_rank_zero:
            input_dict = cloudpickle.loads(socket.recv())
        else:
            input_dict = None
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)

        # Parse input
        if "shutdown" in input_dict.keys() and input_dict["shutdown"]:
            if mpi_rank_zero:
                socket.close()
                context.term()
            break
        elif (
            "fn" in input_dict.keys()
            and "init" not in input_dict.keys()
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
        ):
            # Execute function
            output = call_funct(input_dict=input_dict, funct=None, memory=memory)
            output_reply = MPI.COMM_WORLD.gather(output, root=0)

            # Send output
            if mpi_rank_zero:
                socket.send(cloudpickle.dumps({"result": output_reply}))
        elif (
            "fn" in input_dict.keys()
            and "init" in input_dict.keys()
            and input_dict["init"]
            and "args" in input_dict.keys()
            and "kwargs" in input_dict.keys()
        ):
            memory = call_funct(input_dict=input_dict, funct=None)


if __name__ == "__main__":
    main()

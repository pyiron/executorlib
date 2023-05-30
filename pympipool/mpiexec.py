import pickle
import cloudpickle
import sys
from pympipool.common import parse_arguments, initialize_zmq


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
            host=argument_dict["host"],
            port=argument_dict["zmqport"]
        )
    else:
        context = None
        socket = None

    while True:
        # Read from socket
        if mpi_rank_zero:
            input_dict = cloudpickle.loads(socket.recv())
        else:
            input_dict = None
        input_dict = MPI.COMM_WORLD.bcast(input_dict, root=0)

        # Parse input
        if "c" in input_dict.keys() and input_dict["c"] == "close":
            if mpi_rank_zero:
                socket.close()
                context.term()
            break
        elif "f" in input_dict.keys() and (
            "a" in input_dict.keys() or "k" in input_dict.keys()
        ):
            # Call function
            if "a" in input_dict.keys() and "k" in input_dict.keys():
                output = input_dict["f"](*input_dict["a"], **input_dict["k"])
            elif "a" in input_dict.keys():
                output = input_dict["f"](*input_dict["a"])
            elif "k" in input_dict.keys():
                output = input_dict["f"](**input_dict["k"])
            else:
                raise ValueError("Neither *args nor *kwargs are defined.")

            # Collect output
            output_reply = MPI.COMM_WORLD.gather(output, root=0)

            # Send output
            if mpi_rank_zero:
                socket.send(cloudpickle.dumps({"r": output_reply}))


if __name__ == "__main__":
    main()

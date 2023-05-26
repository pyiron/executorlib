import pickle
import cloudpickle
from mpi4py import MPI

MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import sys
import zmq
from pympipool.common import parse_arguments


def wrap(funct, number_of_cores_per_communicator):
    def functwrapped(*args, **kwargs):
        MPI.COMM_WORLD.Barrier()
        rank = MPI.COMM_WORLD.Get_rank()
        comm_new = MPI.COMM_WORLD.Split(
            rank // number_of_cores_per_communicator,
            rank % number_of_cores_per_communicator,
        )
        comm_new.Barrier()
        return funct(*args, comm=comm_new, **kwargs)

    return functwrapped


def exec_funct(executor, funct, lst, cores_per_task):
    if cores_per_task == 1:
        results = executor.map(funct, lst)
        return list(tqdm(results, desc="Tasks", total=len(lst)))
    else:
        lst_parallel = []
        for input_parameter in lst:
            for _ in range(cores_per_task):
                lst_parallel.append(input_parameter)
        results = executor.map(
            wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
            lst_parallel,
        )
        return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))[
            ::cores_per_task
        ]


def connect_to_message_queue(host, port_selected):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://" + host + ":" + port_selected)
    return context, socket


def main():
    argument_dict = parse_arguments(argument_lst=sys.argv)
    future_dict = {}
    with MPIPoolExecutor(int(argument_dict["total_cores"])) as executor:
        if executor is not None:
            context, socket = connect_to_message_queue(
                host=argument_dict["host"], port_selected=argument_dict["zmqport"]
            )
        while True:
            if executor is not None:
                input_dict = cloudpickle.loads(socket.recv())
                if "c" in input_dict.keys() and input_dict["c"] == "close":
                    socket.close()
                    context.term()
                    break
                elif "f" in input_dict.keys() and "l" in input_dict.keys():
                    try:
                        output = exec_funct(
                            executor=executor,
                            funct=input_dict["f"],
                            lst=input_dict["l"],
                            cores_per_task=int(argument_dict["cores_per_task"]),
                        )
                    except Exception as error:
                        socket.send(
                            cloudpickle.dumps({"e": error, "et": str(type(error))})
                        )
                    else:
                        socket.send(cloudpickle.dumps({"r": output}))
                elif (
                    "f" in input_dict.keys()
                    and "a" in input_dict.keys()
                    and "k" in input_dict.keys()
                ):
                    future = executor.submit(
                        input_dict["f"], *input_dict["a"], **input_dict["k"]
                    )
                    future_hash = hash(future)
                    future_dict[future_hash] = future
                    socket.send(cloudpickle.dumps({"r": future_hash}))
                elif "f" in input_dict.keys() and "k" in input_dict.keys():
                    future = executor.submit(input_dict["f"], **input_dict["k"])
                    future_hash = hash(future)
                    future_dict[future_hash] = future
                    socket.send(cloudpickle.dumps({"r": future_hash}))
                elif "f" in input_dict.keys() and "a" in input_dict.keys():
                    future = executor.submit(input_dict["f"], *input_dict["a"])
                    future_hash = hash(future)
                    future_dict[future_hash] = future
                    socket.send(cloudpickle.dumps({"r": future_hash}))
                elif "u" in input_dict.keys():
                    done_dict = {
                        k: f.result()
                        for k, f in {k: future_dict[k] for k in input_dict["u"]}.items()
                        if f.done()
                    }
                    socket.send(cloudpickle.dumps({"r": done_dict}))
                    for k in done_dict.keys():
                        del future_dict[k]


if __name__ == "__main__":
    main()

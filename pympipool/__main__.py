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


def parse_arguments(argument_lst):
    argument_dict = {
        "total_cores": "--cores-total",
        "zmqport": "--zmqport",
        "cores_per_task": "--cores-per-task",
        "host": "--host",
    }
    parse_dict = {"host": "localhost"}
    parse_dict.update(
        {
            k: argument_lst[argument_lst.index(v) + 1]
            for k, v in argument_dict.items()
            if v in argument_lst
        }
    )
    return parse_dict


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


def parse_socket_communication(executor, input_dict, future_dict, cores_per_task):
    if "c" in input_dict.keys() and input_dict["c"] == "close":
        return "exit"
    elif "f" in input_dict.keys() and "l" in input_dict.keys():
        try:
            output = exec_funct(
                executor=executor,
                funct=input_dict["f"],
                lst=input_dict["l"],
                cores_per_task=cores_per_task,
            )
        except Exception as error:
            return {"e": error, "et": str(type(error))}
        else:
            return {"r": output}
    elif (
        "f" in input_dict.keys()
        and "a" in input_dict.keys()
        and "k" in input_dict.keys()
    ):
        future = executor.submit(input_dict["f"], *input_dict["a"], **input_dict["k"])
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"r": future_hash}
    elif "f" in input_dict.keys() and "k" in input_dict.keys():
        future = executor.submit(input_dict["f"], **input_dict["k"])
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"r": future_hash}
    elif "f" in input_dict.keys() and "a" in input_dict.keys():
        future = executor.submit(input_dict["f"], *input_dict["a"])
        future_hash = hash(future)
        future_dict[future_hash] = future
        return {"r": future_hash}
    elif "u" in input_dict.keys():
        done_dict = {
            k: f.result()
            for k, f in {k: future_dict[k] for k in input_dict["u"]}.items()
            if f.done()
        }
        for k in done_dict.keys():
            del future_dict[k]
        return {"r": done_dict}


def main():
    future_dict = {}
    argument_dict = parse_arguments(argument_lst=sys.argv)
    with MPIPoolExecutor(int(argument_dict["total_cores"])) as executor:
        if executor is not None:
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            socket.connect(
                "tcp://" + argument_dict["host"] + ":" + argument_dict["zmqport"]
            )
            while True:
                output = parse_socket_communication(
                    executor=executor,
                    input_dict=cloudpickle.loads(socket.recv()),
                    future_dict=future_dict,
                    cores_per_task=int(argument_dict["cores_per_task"]),
                )
                if isinstance(output, str) and output == "exit":
                    socket.close()
                    context.term()
                    break
                elif isinstance(output, dict):
                    socket.send(cloudpickle.dumps(output))


if __name__ == "__main__":
    main()

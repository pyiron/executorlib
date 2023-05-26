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


def wrap(funct, number_of_cores_per_communicator):
    def functwrapped(input_parameter):
        MPI.COMM_WORLD.Barrier()
        rank = MPI.COMM_WORLD.Get_rank()
        comm_new = MPI.COMM_WORLD.Split(
            rank // number_of_cores_per_communicator,
            rank % number_of_cores_per_communicator,
        )
        comm_new.Barrier()
        return funct(input_parameter, comm=comm_new)

    return functwrapped


def exec_future(executor, funct, funct_args, funct_kwargs, cores_per_task):
    if cores_per_task == 1:
        if funct_args is not None and funct_kwargs is not None:
            return executor.submit(
                funct, *funct_args, **funct_kwargs
            )
        elif funct_args is not None:
            return executor.submit(
                funct, *funct_args,
            )
        elif funct_kwargs is not None:
            return executor.submit(
                funct, **funct_kwargs
            )
        else:
            raise ValueError("Neither *args nor *kwargs are defined.")
    else:
        if funct_args is not None and funct_kwargs is not None:
            future_lst = [
                executor.submit(
                    wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                    *funct_args,
                    **funct_kwargs
                ) for _ in range(cores_per_task)
            ]
            return future_lst[0]
        elif funct_args is not None:
            future_lst = [
                executor.submit(
                    wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                    *funct_args
                ) for _ in range(cores_per_task)
            ]
            return future_lst[0]
        elif funct_kwargs is not None:
            future_lst = [
                executor.submit(
                    wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
                    **funct_kwargs
                ) for _ in range(cores_per_task)
            ]
            return future_lst[0]
        else:
            raise ValueError("Neither *args nor *kwargs are defined.")


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


def main():
    argument_lst = sys.argv
    total_cores = int(argument_lst[argument_lst.index("--cores-total") + 1])
    future_dict = {}
    with MPIPoolExecutor(total_cores) as executor:
        if executor is not None:
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            port_selected = argument_lst[argument_lst.index("--zmqport") + 1]
            cores_per_task = int(
                argument_lst[argument_lst.index("--cores-per-task") + 1]
            )
            if "--host" in argument_lst:
                host = argument_lst[argument_lst.index("--host") + 1]
            else:
                host = "localhost"
            socket.connect("tcp://" + host + ":" + port_selected)
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
                            cores_per_task=cores_per_task,
                        )
                    except Exception as error:
                        socket.send(
                            cloudpickle.dumps({"e": error, "et": str(type(error))})
                        )
                    else:
                        socket.send(cloudpickle.dumps({"r": output}))
                elif (
                    "f" in input_dict.keys()
                    and ("a" in input_dict.keys() or "k" in input_dict.keys())
                ):
                    if "a" in input_dict.keys():
                        funct_args = input_dict["a"]
                    else:
                        funct_args = None
                    if "a" in input_dict.keys():
                        funct_kwargs = input_dict["k"]
                    else:
                        funct_kwargs = None
                    future = exec_future(
                        executor=executor,
                        funct=input_dict["f"],
                        funct_args=funct_args,
                        funct_kwargs=funct_kwargs,
                        cores_per_task=cores_per_task
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

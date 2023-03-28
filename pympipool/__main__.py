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


def exec_funct(executor, funct, lst, cores_per_task):
    print("cores_per_task:", cores_per_task, file=sys.stderr)
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
        return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))
        # return list(tqdm(results, desc="Tasks", total=len(lst_parallel)))[
        #     ::cores_per_task
        # ]


def main():
    argument_lst = sys.argv
    total_cores = int(argument_lst[argument_lst.index("--cores-total") + 1])
    with MPIPoolExecutor(total_cores) as executor:
        if executor is not None:
            print("mpi world:", MPI.COMM_WORLD.Get_size(), file=sys.stderr)
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            port_selected = argument_lst[argument_lst.index("--zmqport") + 1]
            cores_per_task = int(
                argument_lst[argument_lst.index("--cores-per-task") + 1]
            )
            socket.connect("tcp://localhost:" + port_selected)
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


if __name__ == "__main__":
    main()

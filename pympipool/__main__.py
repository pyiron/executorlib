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
        color = rank // number_of_cores_per_communicator
        key = rank % number_of_cores_per_communicator
        comm_new = MPI.COMM_WORLD.Split(color, key)
        comm_new.Barrier()
        return funct(input_parameter, comm=comm_new)

    return functwrapped


def exec_funct(executor, funct, lst, cores_per_task):
    if cores_per_task == 1:
        results = executor.map(funct, lst)
        return list(tqdm(results, desc="Configs", total=len(lst)))
    else:
        lst_parallel = []
        for l in lst:
            for _ in range(cores_per_task):
                lst_parallel.append(l)
        results = executor.map(
            wrap(funct=funct, number_of_cores_per_communicator=cores_per_task),
            lst_parallel,
        )
        return list(tqdm(results, desc="Configs", total=len(lst)))[::cores_per_task]


def main():
    with MPIPoolExecutor() as executor:
        if executor is not None:
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            argument_lst = sys.argv
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

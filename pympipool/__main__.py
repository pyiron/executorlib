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


def exec_funct(executor, funct, lst):
    results = executor.map(funct, lst)
    return list(tqdm(results, desc="Configs", total=len(lst)))


def main():
    with MPIPoolExecutor() as executor:
        if executor is not None:
            context = zmq.Context()
            socket = context.socket(zmq.PAIR)
            argument_lst = sys.argv
            port_selected = argument_lst[argument_lst.index("--zmqport") + 1]
            socket.connect("tcp://localhost:" + port_selected)
        while True:
            if executor is not None:
                input_dict = cloudpickle.loads(socket.recv())
                if "c" in input_dict.keys() and input_dict["c"] == "close":
                    break
                elif "f" in input_dict.keys() and "l" in input_dict.keys():
                    try:
                        output = exec_funct(
                            executor=executor,
                            funct=input_dict["f"],
                            lst=input_dict["l"],
                        )
                    except Exception as error:
                        socket.send(
                            cloudpickle.dumps({"e": error, "et": str(type(error))})
                        )
                    else:
                        socket.send(cloudpickle.dumps({"r": output}))


if __name__ == "__main__":
    main()

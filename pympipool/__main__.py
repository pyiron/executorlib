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
import os


# Keep the output channel clean
stdout_link = sys.stdout
sys.stdout = open(os.devnull, "w")


def exec_funct(executor, funct, lst):
    results = executor.map(funct, lst)
    return list(tqdm(results, desc="Configs", total=len(lst)))


def check_using_openmpi():
    if MPI.COMM_WORLD.Get_rank() == 0:
        vendor = MPI.get_vendor()[0]
        if vendor != "Open MPI":
            raise ValueError(
                "Currently only OpenMPI is supported. " + vendor + " is not supported."
            )


def main():
    check_using_openmpi()
    with MPIPoolExecutor() as executor:
        while True:
            output = None
            if executor is not None:
                input_dict = cloudpickle.load(sys.stdin.buffer)
                if "c" in input_dict.keys() and input_dict["c"] == "close":
                    break
                elif "f" in input_dict.keys() and "l" in input_dict.keys():
                    output = exec_funct(
                        executor=executor,
                        funct=input_dict["f"],
                        lst=input_dict["l"],
                    )
                if output is not None:
                    cloudpickle.dump(output, stdout_link.buffer)
                    stdout_link.flush()


if __name__ == "__main__":
    main()

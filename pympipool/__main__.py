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


def send(output_dict):
    cloudpickle.dump(
        output_dict,
        stdout_link.buffer,
    )
    stdout_link.flush()


def check_using_openmpi():
    vendor = MPI.get_vendor()[0]
    if vendor != "Open MPI":
        send(
            output_dict={
                "e": "Currently only OpenMPI is supported. "
                + vendor
                + " is not supported."
            }
        )
        return False
    else:
        send(output_dict={"r": True})
        return True


def main():
    with MPIPoolExecutor() as executor:
        if executor is not None:
            if check_using_openmpi():
                while True:
                    input_dict = cloudpickle.load(sys.stdin.buffer)
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
                            send(output_dict={"e": error})
                        else:
                            if output is not None:
                                send(output_dict={"e": error})


if __name__ == "__main__":
    main()

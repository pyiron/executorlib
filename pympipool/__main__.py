import dill
from mpi4py import MPI

MPI.pickle.__init__(
    dill.dumps,
    dill.loads,
    dill.HIGHEST_PROTOCOL,
)
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import sys


def exec_funct(executor, funct, lst):
    results = executor.map(funct, lst)
    return list(tqdm(results, desc="Configs", total=len(lst)))


def main():
    with MPIPoolExecutor() as executor:
        while True:
            output = None
            if executor is not None:
                input_dict = dill.load(sys.stdin.buffer)
                if "c" in input_dict.keys() and input_dict["c"] == "close":
                    break
                elif "f" in input_dict.keys() and "l" in input_dict.keys():
                    output = exec_funct(
                        executor=executor,
                        funct=input_dict["f"],
                        lst=input_dict["l"],
                    )
                if output is not None:
                    dill.dump(output, sys.stdout.buffer)
                    sys.stdout.flush()


if __name__ == "__main__":
    main()

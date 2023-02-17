import dill
import textwrap
from mpi4py import MPI

MPI.pickle.__init__(
    dill.dumps,
    dill.loads,
    dill.HIGHEST_PROTOCOL,
)
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import sys
import os


# Keep the output channel clean
stdout_link = sys.stdout
sys.stdout = open(os.devnull, "w")


def get_function_from_string(function_str):
    function_dedent_str = textwrap.dedent(function_str)
    exec(function_dedent_str)
    return eval(function_dedent_str.split("(")[0][4:])


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
                        funct=get_function_from_string(function_str=input_dict["f"]),
                        lst=input_dict["l"],
                    )
                if output is not None:
                    dill.dump(output, stdout_link.buffer)
                    stdout_link.flush()


if __name__ == "__main__":
    main()

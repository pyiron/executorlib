import pickle
import sys
import time

import cloudpickle

from executorlib.standalone.error import backend_write_error_file
from executorlib.task_scheduler.file.backend import (
    backend_load_file,
    backend_write_file,
)


def main() -> None:
    """
    Main function for executing the cache_parallel script.

    This function uses MPI (Message Passing Interface) to distribute the execution of a function
    across multiple processes. It loads a file, broadcasts the data to all processes, executes
    the function, gathers the results (if there are multiple processes), and writes the output
    to a file.

    Args:
        None

    Returns:
        None
    """
    from mpi4py import MPI

    MPI.pickle.__init__(  # type: ignore
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )
    mpi_rank_zero = MPI.COMM_WORLD.Get_rank() == 0
    mpi_size_larger_one = MPI.COMM_WORLD.Get_size() > 1
    file_name = sys.argv[1]

    time_start = time.time()
    apply_dict = {}
    if mpi_rank_zero:
        apply_dict = backend_load_file(file_name=file_name)
    apply_dict = MPI.COMM_WORLD.bcast(apply_dict, root=0)
    output = apply_dict["fn"].__call__(*apply_dict["args"], **apply_dict["kwargs"])
    try:
        result = (
            MPI.COMM_WORLD.gather(output, root=0) if mpi_size_larger_one else output
        )
    except Exception as error:
        if mpi_rank_zero:
            backend_write_file(
                file_name=file_name,
                output={"error": error},
                runtime=time.time() - time_start,
            )
            backend_write_error_file(
                error=error,
                apply_dict=apply_dict,
            )
    else:
        if mpi_rank_zero:
            backend_write_file(
                file_name=file_name,
                output={"result": result},
                runtime=time.time() - time_start,
            )
    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()

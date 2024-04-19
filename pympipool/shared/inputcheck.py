def check_oversubscribe(oversubscribe):
    if oversubscribe:
        raise ValueError(
            "Oversubscribing is not supported for the pympipool.flux.PyFLuxExecutor backend."
            "Please use oversubscribe=False instead of oversubscribe=True."
        )


def check_command_line_argument_lst(command_line_argument_lst):
    if len(command_line_argument_lst) > 0:
        raise ValueError(
            "The command_line_argument_lst parameter is not supported for the SLURM backend."
        )


def check_gpus_per_worker(gpus_per_worker):
    if gpus_per_worker != 0:
        raise TypeError(
            "GPU assignment is not supported for the pympipool.mpi.PyMPIExecutor backend."
            "Please use gpus_per_worker=0 instead of gpus_per_worker="
            + str(gpus_per_worker)
            + "."
        )


def check_threads_per_core(threads_per_core):
    if threads_per_core != 1:
        raise TypeError(
            "Thread based parallelism is not supported for the pympipool.mpi.PyMPIExecutor backend."
            "Please use threads_per_core=1 instead of threads_per_core="
            + str(threads_per_core)
            + "."
        )

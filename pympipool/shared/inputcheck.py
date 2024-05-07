import inspect


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


def check_executor(executor):
    if executor is not None:
        raise ValueError(
            "The executor parameter is only supported for the flux framework backend."
        )


def check_resource_dict(function):
    if "resource_dict" in inspect.signature(function).parameters.keys():
        raise ValueError(
            "The parameter resource_dict is used internally in pympipool, "
            "so it cannot be used as parameter in the submitted functions."
        )


def check_resource_dict_is_empty(resource_dict):
    if len(resource_dict) > 0:
        raise ValueError(
            "When block_allocation is enabled, the resource requirements have to be defined on the executor level."
        )


def check_refresh_rate(refresh_rate):
    if refresh_rate != 0.01:
        raise ValueError(
            "The sleep_interval parameter is only used when disable_dependencies=False."
        )


def check_backend(backend):
    if backend not in ["auto", "mpi", "slurm", "flux"]:
        raise ValueError(
            'The currently implemented backends are ["flux", "mpi", "slurm"]. '
            'Alternatively, you can select "auto", the default option, to automatically determine the backend. But '
            + backend
            + " is not a valid choice."
        )


def check_init_function(block_allocation, init_function):
    if not block_allocation and init_function is not None:
        raise ValueError("")


def validate_number_of_cores(max_cores, max_workers):
    # only overwrite max_cores when it is set to 1
    if max_workers != 1 and max_cores == 1:
        return max_workers
    return max_cores

import inspect
from concurrent.futures import Executor
from typing import Callable, List, Optional


def check_oversubscribe(oversubscribe: bool) -> None:
    """
    Check if oversubscribe is True and raise a ValueError if it is.
    """
    if oversubscribe:
        raise ValueError(
            "Oversubscribing is not supported for the executorlib.flux.PyFLuxExecutor backend."
            "Please use oversubscribe=False instead of oversubscribe=True."
        )


def check_command_line_argument_lst(command_line_argument_lst: List[str]) -> None:
    """
    Check if command_line_argument_lst is not empty and raise a ValueError if it is.
    """
    if len(command_line_argument_lst) > 0:
        raise ValueError(
            "The command_line_argument_lst parameter is not supported for the SLURM backend."
        )


def check_gpus_per_worker(gpus_per_worker: int) -> None:
    """
    Check if gpus_per_worker is not 0 and raise a TypeError if it is.
    """
    if gpus_per_worker != 0:
        raise TypeError(
            "GPU assignment is not supported for the executorlib.mpi.PyMPIExecutor backend."
            "Please use gpus_per_worker=0 instead of gpus_per_worker="
            + str(gpus_per_worker)
            + "."
        )


def check_threads_per_core(threads_per_core: int) -> None:
    """
    Check if threads_per_core is not 1 and raise a TypeError if it is.
    """
    if threads_per_core != 1:
        raise TypeError(
            "Thread based parallelism is not supported for the executorlib.mpi.PyMPIExecutor backend."
            "Please use threads_per_core=1 instead of threads_per_core="
            + str(threads_per_core)
            + "."
        )


def check_executor(executor: Executor) -> None:
    """
    Check if executor is not None and raise a ValueError if it is.
    """
    if executor is not None:
        raise ValueError(
            "The executor parameter is only supported for the flux framework backend."
        )


def check_nested_flux_executor(nested_flux_executor: bool) -> None:
    """
    Check if nested_flux_executor is True and raise a ValueError if it is.
    """
    if nested_flux_executor:
        raise ValueError(
            "The nested_flux_executor parameter is only supported for the flux framework backend."
        )


def check_resource_dict(function: Callable) -> None:
    """
    Check if the function has a parameter named 'resource_dict' and raise a ValueError if it does.
    """
    if "resource_dict" in inspect.signature(function).parameters.keys():
        raise ValueError(
            "The parameter resource_dict is used internally in executorlib, "
            "so it cannot be used as a parameter in the submitted functions."
        )


def check_resource_dict_is_empty(resource_dict: dict) -> None:
    """
    Check if resource_dict is not empty and raise a ValueError if it is.
    """
    if len(resource_dict) > 0:
        raise ValueError(
            "When block_allocation is enabled, the resource requirements have to be defined on the executor level."
        )


def check_refresh_rate(refresh_rate: float) -> None:
    """
    Check if refresh_rate is not 0.01 and raise a ValueError if it is.
    """
    if refresh_rate != 0.01:
        raise ValueError(
            "The sleep_interval parameter is only used when disable_dependencies=False."
        )


def check_plot_dependency_graph(plot_dependency_graph: bool) -> None:
    """
    Check if plot_dependency_graph is True and raise a ValueError if it is.
    """
    if plot_dependency_graph:
        raise ValueError(
            "The plot_dependency_graph parameter is only used when disable_dependencies=False."
        )


def check_pmi(backend: str, pmi: Optional[str]) -> None:
    """
    Check if pmi is valid for the selected backend and raise a ValueError if it is not.
    """
    if backend != "flux" and pmi is not None:
        raise ValueError("The pmi parameter is currently only implemented for flux.")
    elif backend == "flux" and pmi not in ["pmix", "pmi1", "pmi2", None]:
        raise ValueError(
            "The pmi parameter supports [pmix, pmi1, pmi2], but not: " + pmi
        )


def check_init_function(block_allocation: bool, init_function: Callable) -> None:
    """
    Check if block_allocation is False and init_function is not None, and raise a ValueError if it is.
    """
    if not block_allocation and init_function is not None:
        raise ValueError("")


def validate_number_of_cores(max_cores: int, max_workers: int) -> int:
    """
    Validate the number of cores and return the appropriate value.
    """
    if max_workers != 1 and max_cores == 1:
        return max_workers
    return max_cores

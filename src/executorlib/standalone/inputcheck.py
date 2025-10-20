import inspect
import multiprocessing
import os.path
from concurrent.futures import Executor
from typing import Callable, Optional
from warnings import warn


def check_oversubscribe(oversubscribe: bool) -> None:
    """
    Check if oversubscribe is True and raise a ValueError if it is.
    """
    if oversubscribe:
        raise ValueError(
            "Oversubscribing is not supported for the executorlib.flux.PyFLuxExecutor backend."
            "Please use oversubscribe=False instead of oversubscribe=True."
        )


def check_command_line_argument_lst(command_line_argument_lst: list[str]) -> None:
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
    if "resource_dict" in inspect.signature(function).parameters:
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


def check_pmi(backend: Optional[str], pmi: Optional[str]) -> None:
    """
    Check if pmi is valid for the selected backend and raise a ValueError if it is not.
    """
    if backend is not None:
        if backend != "flux_allocation" and pmi is not None:
            raise ValueError(
                "The pmi parameter is currently only implemented for flux."
            )
        elif backend == "flux_allocation" and pmi not in ["pmix", "pmi1", "pmi2", None]:
            raise ValueError(
                "The pmi parameter supports [pmix, pmi1, pmi2], but not: " + str(pmi)
            )


def check_init_function(
    block_allocation: bool, init_function: Optional[Callable]
) -> None:
    """
    Check if block_allocation is False and init_function is not None, and raise a ValueError if it is.
    """
    if not block_allocation and init_function is not None:
        raise ValueError("")


def check_max_workers_and_cores(
    max_workers: Optional[int], max_cores: Optional[int]
) -> None:
    if max_workers is not None:
        raise ValueError(
            "The number of workers cannot be controlled with the pysqa based backend."
        )
    if max_cores is not None:
        raise ValueError(
            "The number of cores cannot be controlled with the pysqa based backend."
        )


def check_hostname_localhost(hostname_localhost: Optional[bool]) -> None:
    if hostname_localhost is not None:
        raise ValueError(
            "The option to connect to hosts based on their hostname is not available with the pysqa based backend."
        )


def check_pmi_mode(pmi_mode: Optional[str]) -> None:
    if pmi_mode is not None:
        raise ValueError(
            "The option to specify the pmi mode is not available on a local workstation, it requires SLURM or flux."
        )


def check_flux_log_files(flux_log_files: Optional[bool]) -> None:
    """
    Check if flux_log_files is True and raise a ValueError if it is.
    """
    if flux_log_files:
        raise ValueError(
            "The flux_log_files parameter is only supported for the flux framework backend."
        )


def check_pysqa_config_directory(pysqa_config_directory: Optional[str]) -> None:
    """
    Check if pysqa_config_directory is None and raise a ValueError if it is not.
    """
    if pysqa_config_directory is not None:
        raise ValueError(
            "pysqa_config_directory parameter is only supported for pysqa backend."
        )


def validate_number_of_cores(
    max_cores: Optional[int] = None,
    max_workers: Optional[int] = None,
    cores_per_worker: Optional[int] = 1,
    set_local_cores: bool = False,
) -> int:
    """
    Validate the number of cores and return the appropriate value.
    """
    if max_cores is not None and max_workers is None and cores_per_worker is not None:
        return int(max_cores / cores_per_worker)
    elif max_workers is not None:
        return int(max_workers)
    elif max_cores is None and max_workers is None and not set_local_cores:
        raise ValueError(
            "Block allocation requires a fixed set of computational resources. Neither max_cores nor max_workers are defined."
        )
    else:
        max_workers = multiprocessing.cpu_count()
        warn(
            "max_workers parameter is not set, set default based on CPU count to: max_workers="
            + str(max_workers),
            stacklevel=2,
        )
        return max_workers


def check_file_exists(file_name: Optional[str]):
    """
    Check if file exists and raise a ValueError if it does not or file_name is None.
    """
    if file_name is None:
        raise ValueError("file_name is not set.")
    if not os.path.exists(file_name):
        raise ValueError("file_name is not written to the file system.")


def check_log_obj_size(log_obj_size: bool) -> None:
    """
    Check if log_obj_size is True and raise a ValueError if it is.
    """
    if log_obj_size:
        raise ValueError(
            "log_obj_size is not supported for the executorlib.SlurmClusterExecutor and executorlib.FluxClusterExecutor."
            "Please use log_obj_size=False instead of log_obj_size=True."
        )

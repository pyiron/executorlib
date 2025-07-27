import importlib.util
import os
import sys
from typing import Optional


def get_command_path(executable: str) -> str:
    """
    Get path of the backend executable script

    Args:
        executable (str): Name of the backend executable script, either mpiexec.py or serial.py

    Returns:
        str: absolute path to the executable script
    """
    return os.path.abspath(os.path.join(__file__, "..", "..", "backend", executable))


def get_cache_execute_command(
    file_name: str,
    cores: int = 1,
    backend: Optional[str] = None,
    pmi_mode: Optional[str] = None,
) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        file_name (str): The name of the file.
        cores (int, optional): Number of cores used to execute the task. Defaults to 1.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None (Flux only)

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1 and importlib.util.find_spec("mpi4py") is not None:
        if backend is None:
            command_lst = (
                ["mpiexec", "-n", str(cores)]
                + command_lst
                + [get_command_path(executable="cache_parallel.py"), file_name]
            )
        elif backend == "slurm":
            command_prepend = ["srun", "-n", str(cores)]
            if pmi_mode is not None:
                command_prepend += ["--mpi=" + pmi_mode]
            command_lst = (
                command_prepend
                + command_lst
                + [get_command_path(executable="cache_parallel.py"), file_name]
            )
        elif backend == "flux":
            flux_command = ["flux", "run"]
            if pmi_mode is not None:
                flux_command += ["-o", "pmi=" + pmi_mode]
            command_lst = (
                flux_command
                + ["-n", str(cores)]
                + command_lst
                + [get_command_path(executable="cache_parallel.py"), file_name]
            )
        else:
            raise ValueError(f"backend should be None, slurm or flux, not {backend}")
    elif cores > 1:
        raise ImportError(
            "mpi4py is required for parallel calculations. Please install mpi4py."
        )
    else:
        command_lst += [get_command_path(executable="cache_serial.py"), file_name]
    return command_lst


def get_interactive_execute_command(
    cores: int,
) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        cores (int): Number of cores used to execute the task, if it is greater than one use interactive_parallel.py
                     else interactive_serial.py

    Returns:
        list[str]: List of strings containing the python executable path and the backend script to execute
    """
    command_lst = [sys.executable]
    if cores > 1 and importlib.util.find_spec("mpi4py") is not None:
        command_lst += [get_command_path(executable="interactive_parallel.py")]
    elif cores > 1:
        raise ImportError(
            "mpi4py is required for parallel calculations. Please install mpi4py."
        )
    else:
        command_lst += [get_command_path(executable="interactive_serial.py")]
    return command_lst

import importlib.util
import os
import sys
from typing import Optional

SLURM_COMMAND = "srun"


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
    exclusive: bool = False,
    openmpi_oversubscribe: bool = False,
    pmi_mode: Optional[str] = None,
) -> list:
    """
    Get command to call backend as a list of two strings

    Args:
        file_name (str): The name of the file.
        cores (int, optional): Number of cores used to execute the task. Defaults to 1.
        backend (str, optional): name of the backend used to spawn tasks ["slurm", "flux"].
        exclusive (bool): Whether to exclusively reserve the compute nodes, or allow sharing compute notes. Defaults to False.
        openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None

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
            if openmpi_oversubscribe:
                command_prepend += ["--oversubscribe"]
            if exclusive:
                command_prepend += ["--exact"]
            command_lst = (
                command_prepend
                + command_lst
                + [get_command_path(executable="cache_parallel.py"), file_name]
            )
        elif backend == "flux":
            flux_command = ["flux", "run"]
            if pmi_mode is not None:
                flux_command += ["-o", "pmi=" + pmi_mode]
            if openmpi_oversubscribe:
                raise ValueError(
                    "The option openmpi_oversubscribe is not available with the flux backend."
                )
            if exclusive:
                raise ValueError(
                    "The option exclusive is not available with the flux backend."
                )
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


def generate_slurm_command(
    cores: int,
    cwd: Optional[str],
    threads_per_core: int = 1,
    gpus_per_core: int = 0,
    num_nodes: Optional[int] = None,
    exclusive: bool = False,
    openmpi_oversubscribe: bool = False,
    slurm_cmd_args: Optional[list[str]] = None,
    pmi_mode: Optional[str] = None,
) -> list[str]:
    """
    Generate the command list for the SLURM interface.

    Args:
        cores (int): The number of cores.
        cwd (str): The current working directory.
        threads_per_core (int, optional): The number of threads per core. Defaults to 1.
        gpus_per_core (int, optional): The number of GPUs per core. Defaults to 0.
        num_nodes (int, optional): The number of compute nodes to use for executing the task. Defaults to None.
        exclusive (bool): Whether to exclusively reserve the compute nodes, or allow sharing compute notes. Defaults to False.
        openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        slurm_cmd_args (list[str], optional): Additional command line arguments. Defaults to [].
        pmi_mode (str): PMI interface to use (OpenMPI v5 requires pmix) default is None

    Returns:
        list[str]: The generated command list.
    """
    command_prepend_lst = [SLURM_COMMAND, "-n", str(cores)]
    if cwd is not None:
        command_prepend_lst += ["-D", cwd]
    if pmi_mode is not None:
        command_prepend_lst += ["--mpi=" + pmi_mode]
    if num_nodes is not None:
        command_prepend_lst += ["-N", str(num_nodes)]
    if threads_per_core > 1:
        command_prepend_lst += ["--cpus-per-task=" + str(threads_per_core)]
    if gpus_per_core > 0:
        command_prepend_lst += ["--gpus-per-task=" + str(gpus_per_core)]
    if exclusive:
        command_prepend_lst += ["--exact"]
    if openmpi_oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    if slurm_cmd_args is not None and len(slurm_cmd_args) > 0:
        command_prepend_lst += slurm_cmd_args
    return command_prepend_lst

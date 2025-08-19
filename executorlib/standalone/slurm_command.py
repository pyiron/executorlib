from typing import Optional

SLURM_COMMAND = "srun"


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

import os
from typing import Optional

from executorlib.standalone.interactive.spawner import SubprocessSpawner

SLURM_COMMAND = "srun"


def validate_max_workers(max_workers: int, cores: int, threads_per_core: int):
    cores_total = int(os.environ["SLURM_NTASKS"]) * int(
        os.environ["SLURM_CPUS_PER_TASK"]
    )
    cores_requested = max_workers * cores * threads_per_core
    if cores_total < cores_requested:
        raise ValueError(
            "The number of requested cores is larger than the available cores "
            + str(cores_total)
            + " < "
            + str(cores_requested)
        )


class SrunSpawner(SubprocessSpawner):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        threads_per_core: int = 1,
        gpus_per_core: int = 0,
        openmpi_oversubscribe: bool = False,
        slurm_cmd_args: list[str] = [],
    ):
        """
        Srun interface implementation.

        Args:
            cwd (str, optional): The current working directory. Defaults to None.
            cores (int, optional): The number of cores to use. Defaults to 1.
            threads_per_core (int, optional): The number of threads per core. Defaults to 1.
            gpus_per_core (int, optional): The number of GPUs per core. Defaults to 0.
            openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
            slurm_cmd_args (list[str], optional): Additional command line arguments. Defaults to [].
        """
        super().__init__(
            cwd=cwd,
            cores=cores,
            openmpi_oversubscribe=openmpi_oversubscribe,
            threads_per_core=threads_per_core,
        )
        self._gpus_per_core = gpus_per_core
        self._slurm_cmd_args = slurm_cmd_args

    def generate_command(self, command_lst: list[str]) -> list[str]:
        """
        Generate the command list for the Srun interface.

        Args:
            command_lst (list[str]): The command list.

        Returns:
            list[str]: The generated command list.
        """
        command_prepend_lst = generate_slurm_command(
            cores=self._cores,
            cwd=self._cwd,
            threads_per_core=self._threads_per_core,
            gpus_per_core=self._gpus_per_core,
            openmpi_oversubscribe=self._openmpi_oversubscribe,
            slurm_cmd_args=self._slurm_cmd_args,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


def generate_slurm_command(
    cores: int,
    cwd: Optional[str],
    threads_per_core: int = 1,
    gpus_per_core: int = 0,
    openmpi_oversubscribe: bool = False,
    slurm_cmd_args: list[str] = [],
) -> list[str]:
    """
    Generate the command list for the SLURM interface.

    Args:
        cores (int): The number of cores.
        cwd (str): The current working directory.
        threads_per_core (int, optional): The number of threads per core. Defaults to 1.
        gpus_per_core (int, optional): The number of GPUs per core. Defaults to 0.
        openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        slurm_cmd_args (list[str], optional): Additional command line arguments. Defaults to [].

    Returns:
        list[str]: The generated command list.
    """
    command_prepend_lst = [SLURM_COMMAND, "-n", str(cores)]
    if cwd is not None:
        command_prepend_lst += ["-D", cwd]
    if threads_per_core > 1:
        command_prepend_lst += ["--cpus-per-task" + str(threads_per_core)]
    if gpus_per_core > 0:
        command_prepend_lst += ["--gpus-per-task=" + str(gpus_per_core)]
    if openmpi_oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    if len(slurm_cmd_args) > 0:
        command_prepend_lst += slurm_cmd_args
    return command_prepend_lst

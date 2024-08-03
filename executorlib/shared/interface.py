import subprocess
from abc import ABC
from typing import Optional

MPI_COMMAND = "mpiexec"
SLURM_COMMAND = "srun"


class BaseInterface(ABC):
    def __init__(self, cwd: str, cores: int = 1, oversubscribe: bool = False):
        """
        Base class for interface implementations.

        Args:
            cwd (str): The current working directory.
            cores (int, optional): The number of cores to use. Defaults to 1.
            oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        """
        self._cwd = cwd
        self._cores = cores
        self._oversubscribe = oversubscribe

    def bootup(
        self,
        command_lst: list[str],
        prefix_name: Optional[str] = None,
        prefix_path: Optional[str] = None,
    ):
        """
        Method to start the interface.

        Args:
            command_lst (list[str]): The command list to execute.
            prefix_name (str, optional): The prefix name. Defaults to None.
            prefix_path (str, optional): The prefix path. Defaults to None.
        """
        raise NotImplementedError

    def shutdown(self, wait: bool = True):
        """
        Method to shutdown the interface.

        Args:
            wait (bool, optional): Whether to wait for the interface to shutdown. Defaults to True.
        """
        raise NotImplementedError

    def poll(self):
        """
        Method to check if the interface is running.

        Returns:
            bool: True if the interface is running, False otherwise.
        """
        raise NotImplementedError


class SubprocessInterface(BaseInterface):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        oversubscribe: bool = False,
    ):
        """
        Subprocess interface implementation.

        Args:
            cwd (str, optional): The current working directory. Defaults to None.
            cores (int, optional): The number of cores to use. Defaults to 1.
            oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        """
        super().__init__(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        )
        self._process = None

    def bootup(
        self,
        command_lst: list[str],
        prefix_name: Optional[str] = None,
        prefix_path: Optional[str] = None,
    ):
        """
        Method to start the subprocess interface.

        Args:
            command_lst (list[str]): The command list to execute.
            prefix_name (str, optional): The prefix name. Defaults to None.
            prefix_path (str, optional): The prefix path. Defaults to None.
        """
        if prefix_name is None and prefix_path is None:
            self._process = subprocess.Popen(
                args=self.generate_command(command_lst=command_lst),
                cwd=self._cwd,
                stdin=subprocess.DEVNULL,
            )
        else:
            import conda_subprocess

            self._process = conda_subprocess.Popen(
                args=self.generate_command(command_lst=command_lst),
                cwd=self._cwd,
                stdin=subprocess.DEVNULL,
                prefix_path=prefix_path,
                prefix_name=prefix_name,
            )

    def generate_command(self, command_lst: list[str]) -> list[str]:
        """
        Method to generate the command list.

        Args:
            command_lst (list[str]): The command list.

        Returns:
            list[str]: The generated command list.
        """
        return command_lst

    def shutdown(self, wait: bool = True):
        """
        Method to shutdown the subprocess interface.

        Args:
            wait (bool, optional): Whether to wait for the interface to shutdown. Defaults to True.
        """
        self._process.communicate()
        self._process.terminate()
        if wait:
            self._process.wait()
        self._process = None

    def poll(self) -> bool:
        """
        Method to check if the subprocess interface is running.

        Returns:
            bool: True if the interface is running, False otherwise.
        """
        return self._process is not None and self._process.poll() is None


class MpiExecInterface(SubprocessInterface):
    def generate_command(self, command_lst: list[str]) -> list[str]:
        """
        Generate the command list for the MPIExec interface.

        Args:
            command_lst (list[str]): The command list.

        Returns:
            list[str]: The generated command list.
        """
        command_prepend_lst = generate_mpiexec_command(
            cores=self._cores,
            oversubscribe=self._oversubscribe,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


class SrunInterface(SubprocessInterface):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        threads_per_core: int = 1,
        gpus_per_core: int = 0,
        oversubscribe: bool = False,
        command_line_argument_lst: list[str] = [],
    ):
        """
        Srun interface implementation.

        Args:
            cwd (str, optional): The current working directory. Defaults to None.
            cores (int, optional): The number of cores to use. Defaults to 1.
            threads_per_core (int, optional): The number of threads per core. Defaults to 1.
            gpus_per_core (int, optional): The number of GPUs per core. Defaults to 0.
            oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
            command_line_argument_lst (list[str], optional): Additional command line arguments. Defaults to [].
        """
        super().__init__(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._command_line_argument_lst = command_line_argument_lst

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
            oversubscribe=self._oversubscribe,
            command_line_argument_lst=self._command_line_argument_lst,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


def generate_mpiexec_command(cores: int, oversubscribe: bool = False) -> list[str]:
    """
    Generate the command list for the MPIExec interface.

    Args:
        cores (int): The number of cores.
        oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.

    Returns:
        list[str]: The generated command list.
    """
    if cores == 1:
        return []
    else:
        command_prepend_lst = [MPI_COMMAND, "-n", str(cores)]
        if oversubscribe:
            command_prepend_lst += ["--oversubscribe"]
        return command_prepend_lst


def generate_slurm_command(
    cores: int,
    cwd: str,
    threads_per_core: int = 1,
    gpus_per_core: int = 0,
    oversubscribe: bool = False,
    command_line_argument_lst: list[str] = [],
) -> list[str]:
    """
    Generate the command list for the SLURM interface.

    Args:
        cores (int): The number of cores.
        cwd (str): The current working directory.
        threads_per_core (int, optional): The number of threads per core. Defaults to 1.
        gpus_per_core (int, optional): The number of GPUs per core. Defaults to 0.
        oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        command_line_argument_lst (list[str], optional): Additional command line arguments. Defaults to [].

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
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    if len(command_line_argument_lst) > 0:
        command_prepend_lst += command_line_argument_lst
    return command_prepend_lst

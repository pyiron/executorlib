import os
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

MPI_COMMAND = "mpiexec"


class BaseSpawner(ABC):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        openmpi_oversubscribe: bool = False,
    ):
        """
        Base class for interface implementations.

        Args:
            cwd (str): The current working directory.
            cores (int, optional): The number of cores to use. Defaults to 1.
            openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        """
        self._cwd = cwd
        self._cores = cores
        self._openmpi_oversubscribe = openmpi_oversubscribe

    @abstractmethod
    def bootup(
        self,
        command_lst: list[str],
    ):
        """
        Method to start the interface.

        Args:
            command_lst (list[str]): The command list to execute.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, wait: bool = True):
        """
        Method to shutdown the interface.

        Args:
            wait (bool, optional): Whether to wait for the interface to shutdown. Defaults to True.
        """
        raise NotImplementedError

    @abstractmethod
    def poll(self):
        """
        Method to check if the interface is running.

        Returns:
            bool: True if the interface is running, False otherwise.
        """
        raise NotImplementedError


class SubprocessSpawner(BaseSpawner):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        openmpi_oversubscribe: bool = False,
        threads_per_core: int = 1,
    ):
        """
        Subprocess interface implementation.

        Args:
            cwd (str, optional): The current working directory. Defaults to None.
            cores (int, optional): The number of cores to use. Defaults to 1.
            threads_per_core (int, optional): The number of threads per core. Defaults to 1.
            openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.
        """
        super().__init__(
            cwd=cwd,
            cores=cores,
            openmpi_oversubscribe=openmpi_oversubscribe,
        )
        self._process: Optional[subprocess.Popen] = None
        self._threads_per_core = threads_per_core

    def bootup(
        self,
        command_lst: list[str],
    ):
        """
        Method to start the subprocess interface.

        Args:
            command_lst (list[str]): The command list to execute.
        """
        if self._cwd is not None:
            os.makedirs(self._cwd, exist_ok=True)
        self._process = subprocess.Popen(
            args=self.generate_command(command_lst=command_lst),
            cwd=self._cwd,
            stdin=subprocess.DEVNULL,
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
        if self._process is not None:
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


class MpiExecSpawner(SubprocessSpawner):
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
            openmpi_oversubscribe=self._openmpi_oversubscribe,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


def generate_mpiexec_command(
    cores: int, openmpi_oversubscribe: bool = False
) -> list[str]:
    """
    Generate the command list for the MPIExec interface.

    Args:
        cores (int): The number of cores.
        openmpi_oversubscribe (bool, optional): Whether to oversubscribe the cores. Defaults to False.

    Returns:
        list[str]: The generated command list.
    """
    if cores == 1:
        return []
    else:
        command_prepend_lst = [MPI_COMMAND, "-n", str(cores)]
        if openmpi_oversubscribe:
            command_prepend_lst += ["--oversubscribe"]
        return command_prepend_lst

from typing import Optional

from pysqa import QueueAdapter

from executorlib.standalone.interactive.spawner import BaseSpawner
from executorlib.standalone.scheduler import pysqa_execute_command, terminate_with_pysqa


class PysqaSpawner(BaseSpawner):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        openmpi_oversubscribe: bool = False,
        threads_per_core: int = 1,
        config_directory: Optional[str] = None,
        backend: Optional[str] = None,
        submission_kwargs: Optional[dict] = None,
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
        self._process: Optional[int] = None
        self._threads_per_core = threads_per_core
        self._config_directory = config_directory
        self._backend = backend
        self._submission_kwargs = submission_kwargs

    def bootup(
        self,
        command_lst: list[str],
    ):
        """
        Method to start the subprocess interface.

        Args:
            command_lst (list[str]): The command list to execute.
        """
        qa = QueueAdapter(
            directory=self._config_directory,
            queue_type=self._backend,
            execute_command=pysqa_execute_command,
        )
        self._process = qa.submit_job(
            command=" ".join(self.generate_command(command_lst=command_lst)),
            working_directory=self._cwd,
            cores=self._cores,
            **self._submission_kwargs,
        )

    def generate_command(self, command_lst: list[str]) -> list[str]:
        """
        Method to generate the command list.

        Args:
            command_lst (list[str]): The command list.

        Returns:
            list[str]: The generated command list.
        """
        if self._cores > 1 and self._backend is None:
            command_prepend = ["mpiexec", "-n", str(self._cores)]
        elif self._cores > 1 and self._backend == "slurm":
            command_prepend = ["srun", "-n", str(self._cores)]
        elif self._cores > 1 and self._backend == "flux":
            command_prepend = ["flux", "run", "-n", str(self._cores)]
        elif self._cores > 1:
            raise ValueError("backend should be None, slurm or flux, not {}".format(self._backend))
        else:
            command_prepend = []
        return command_prepend + command_lst

    def shutdown(self, wait: bool = True):
        """
        Method to shutdown the subprocess interface.

        Args:
            wait (bool, optional): Whether to wait for the interface to shutdown. Defaults to True.
        """
        if self._process is not None:
            terminate_with_pysqa(
                queue_id=self._process,
                config_directory=self._config_directory,
                backend=self._backend,
            )
        self._process = None

    def poll(self) -> bool:
        """
        Method to check if the subprocess interface is running.

        Returns:
            bool: True if the interface is running, False otherwise.
        """
        qa = QueueAdapter(
            directory=self._config_directory,
            queue_type=self._backend,
            execute_command=pysqa_execute_command,
        )
        if self._process is not None:
            return qa.get_status_of_job(process_id=self._process) in ["running", "pending"]
        else:
            return False

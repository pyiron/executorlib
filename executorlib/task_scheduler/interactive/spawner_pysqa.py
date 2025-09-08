import hashlib
import os
from time import sleep
from typing import Callable, Optional

from pysqa import QueueAdapter

from executorlib.standalone.inputcheck import validate_number_of_cores
from executorlib.standalone.interactive.spawner import BaseSpawner
from executorlib.standalone.scheduler import pysqa_execute_command, terminate_with_pysqa
from executorlib.task_scheduler.interactive.blockallocation import (
    BlockAllocationTaskScheduler,
)


class PysqaSpawner(BaseSpawner):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        threads_per_core: int = 1,
        gpus_per_core: int = 0,
        num_nodes: Optional[int] = None,
        exclusive: bool = False,
        openmpi_oversubscribe: bool = False,
        slurm_cmd_args: Optional[list[str]] = None,
        pmi_mode: Optional[str] = None,
        config_directory: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """
        Subprocess interface implementation.

        Args:
            cwd (str, optional): The current working directory. Defaults to None.
            cores (int): The number of cores to use. Defaults to 1.
            threads_per_core (int): The number of threads per core. Defaults to 1.
            gpus_per_core (int): number of GPUs per worker - defaults to 0
            num_nodes (int, optional): The number of compute nodes to use for executing the task.  Defaults to None.
            exclusive (bool): Whether to exclusively reserve the compute nodes, or allow sharing compute notes. Defaults
                              to False.
            openmpi_oversubscribe (bool): Whether to oversubscribe the cores. Defaults to False.
            slurm_cmd_args (list, optional): Additional command line arguments for the srun call (SLURM only)
            pmi_mode (str, optional): PMI interface to use (OpenMPI v5 requires pmix) default is None
            config_directory (str, optional): path to the pysqa config directory (only for pysqa based backend).
            backend (str): name of the backend used to spawn tasks.
        """
        super().__init__(
            cwd=cwd,
            cores=cores,
            openmpi_oversubscribe=openmpi_oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._num_nodes = num_nodes
        self._exclusive = exclusive
        self._slurm_cmd_args = slurm_cmd_args
        self._pmi_mode = pmi_mode
        self._config_directory = config_directory
        self._backend = backend
        self._pysqa_submission_kwargs = kwargs
        self._process: Optional[int] = None
        self._queue_adapter: Optional[QueueAdapter] = None

    def bootup(
        self,
        command_lst: list[str],
        stop_function: Optional[Callable] = None,
    ):
        """
        Method to start the subprocess interface.

        Args:
            command_lst (list[str]): The command list to execute.
            stop_function (Callable): Function to stop the interface.

        Returns:
            bool: Whether the interface was successfully started.
        """
        self._queue_adapter = QueueAdapter(
            directory=self._config_directory,
            queue_type=self._backend,
            execute_command=pysqa_execute_command,
        )
        self._process = self._start_process_helper(
            command_lst=command_lst,
            queue_adapter=self._queue_adapter,
        )
        while True:
            if self._check_process_helper(command_lst=command_lst):
                return True
            elif stop_function is not None and stop_function():
                self.shutdown(wait=True)
                return False
            else:
                sleep(1)  # Wait for the process to start

    def generate_command(self, command_lst: list[str]) -> list[str]:
        """
        Method to generate the command list.

        Args:
            command_lst (list[str]): The command list.

        Returns:
            list[str]: The generated command list.
        """
        if self._cores > 1 and self._backend == "slurm":
            command_prepend = ["srun", "-n", str(self._cores)]
            if self._pmi_mode is not None:
                command_prepend += ["--mpi=" + self._pmi_mode]
            if self._num_nodes is not None:
                command_prepend += ["-N", str(self._num_nodes)]
            if self._threads_per_core > 1:
                command_prepend += ["--cpus-per-task=" + str(self._threads_per_core)]
            if self._gpus_per_core > 0:
                command_prepend += ["--gpus-per-task=" + str(self._gpus_per_core)]
            if self._exclusive:
                command_prepend += ["--exact"]
            if self._openmpi_oversubscribe:
                command_prepend += ["--oversubscribe"]
            if self._slurm_cmd_args is not None and len(self._slurm_cmd_args) > 0:
                command_prepend += self._slurm_cmd_args
        elif self._cores > 1 and self._backend == "flux":
            command_prepend = ["flux", "run", "-n", str(self._cores)]
            if self._pmi_mode is not None:
                command_prepend += ["-o", "pmi=" + self._pmi_mode]
            if self._num_nodes is not None:
                raise ValueError()
            if self._threads_per_core > 1:
                raise ValueError()
            if self._gpus_per_core > 0:
                raise ValueError()
            if self._exclusive:
                raise ValueError()
            if self._openmpi_oversubscribe:
                raise ValueError()
        elif self._cores > 1:
            raise ValueError(
                f"backend should be None, slurm or flux, not {self._backend}"
            )
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
        if self._process is not None and self._queue_adapter is not None:
            status = self._queue_adapter.get_status_of_job(process_id=self._process)
            return status in ["running", "pending"]
        else:
            return False

    def _start_process_helper(
        self, command_lst: list[str], queue_adapter: QueueAdapter
    ) -> int:
        hash = hashlib.md5(str(self).encode()).hexdigest()
        if self._cwd is not None:
            working_directory = os.path.join(self._cwd, hash)
        else:
            working_directory = os.path.abspath(hash)
        return queue_adapter.submit_job(
            command=" ".join(self.generate_command(command_lst=command_lst)),
            working_directory=working_directory,
            cores=int(self._cores * self._threads_per_core),
            **self._pysqa_submission_kwargs,
        )

    def _check_process_helper(self, command_lst: list[str]) -> bool:
        if self._queue_adapter is not None:
            status = self._queue_adapter.get_status_of_job(process_id=self._process)
        else:
            status = None
        if status == "running":
            return True
        elif status is None:
            raise RuntimeError(
                f"Failed to start the process with command: {command_lst}"
            )
        elif status == "error":
            self._process = self._start_process_helper(
                command_lst=command_lst, queue_adapter=self._queue_adapter
            )
        return False

    def __del__(self):
        self.shutdown(wait=True)


def create_pysqa_block_allocation_scheduler(
    max_cores: Optional[int] = None,
    cache_directory: Optional[str] = None,
    hostname_localhost: Optional[bool] = None,
    log_obj_size: bool = False,
    pmi_mode: Optional[str] = None,
    init_function: Optional[Callable] = None,
    max_workers: Optional[int] = None,
    resource_dict: Optional[dict] = None,
    pysqa_config_directory: Optional[str] = None,
    backend: Optional[str] = None,
):
    if resource_dict is None:
        resource_dict = {}
    cores_per_worker = resource_dict.get("cores", 1)
    if "cwd" in resource_dict and resource_dict["cwd"] is not None:
        resource_dict["cwd"] = os.path.abspath(resource_dict["cwd"])
    if cache_directory is not None:
        resource_dict["cache_directory"] = os.path.abspath(cache_directory)
    else:
        resource_dict["cache_directory"] = os.path.abspath(".")
    resource_dict["hostname_localhost"] = hostname_localhost
    resource_dict["log_obj_size"] = log_obj_size
    resource_dict["pmi_mode"] = pmi_mode
    resource_dict["init_function"] = init_function
    resource_dict["config_directory"] = pysqa_config_directory
    resource_dict["backend"] = backend
    max_workers = validate_number_of_cores(
        max_cores=max_cores,
        max_workers=max_workers,
        cores_per_worker=cores_per_worker,
        set_local_cores=False,
    )
    return BlockAllocationTaskScheduler(
        max_workers=max_workers,
        executor_kwargs=resource_dict,
        spawner=PysqaSpawner,
    )

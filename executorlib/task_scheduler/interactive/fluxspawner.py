import os
from typing import Optional

import flux
import flux.job

from executorlib.standalone.interactive.spawner import BaseSpawner


def validate_max_workers(max_workers: int, cores: int, threads_per_core: int):
    handle = flux.Flux()
    cores_total = flux.resource.list.resource_list(handle).get().up.ncores
    cores_requested = max_workers * cores * threads_per_core
    if cores_total < cores_requested:
        raise ValueError(
            "The number of requested cores is larger than the available cores "
            + str(cores_total)
            + " < "
            + str(cores_requested)
        )


class FluxPythonSpawner(BaseSpawner):
    """
    A class representing the FluxPythonInterface.

    Args:
        cwd (str, optional): The current working directory. Defaults to None.
        cores (int, optional): The number of cores. Defaults to 1.
        threads_per_core (int, optional): The number of threads per base. Defaults to 1.
        gpus_per_core (int, optional): The number of GPUs per base. Defaults to 0.
        num_nodes (int, optional): The number of compute nodes to use for executing the task. Defaults to None.
        exclusive (bool): Whether to exclusively reserve the compute nodes, or allow sharing compute notes. Defaults to
                          False.
        openmpi_oversubscribe (bool, optional): Whether to oversubscribe. Defaults to False.
        priority (int, optional): job urgency 0 (lowest) through 31 (highest) (default is 16). Priorities 0 through 15
                                  are restricted to the instance owner.
        pmi_mode (str, optional): The PMI option. Defaults to None.
        flux_executor (flux.job.FluxExecutor, optional): The FluxExecutor instance. Defaults to None.
        flux_executor_nesting (bool, optional): Whether to use nested FluxExecutor. Defaults to False.
        flux_log_files (bool, optional): Write flux stdout and stderr files. Defaults to False.
    """

    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        threads_per_core: int = 1,
        gpus_per_core: int = 0,
        num_nodes: Optional[int] = None,
        exclusive: bool = False,
        priority: Optional[int] = None,
        openmpi_oversubscribe: bool = False,
        pmi_mode: Optional[str] = None,
        flux_executor: Optional[flux.job.FluxExecutor] = None,
        flux_executor_nesting: bool = False,
        flux_log_files: bool = False,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            openmpi_oversubscribe=openmpi_oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._num_nodes = num_nodes
        self._exclusive = exclusive
        self._flux_executor = flux_executor
        self._pmi_mode = pmi_mode
        self._flux_executor_nesting = flux_executor_nesting
        self._flux_log_files = flux_log_files
        self._priority = priority
        self._future = None

    def bootup(
        self,
        command_lst: list[str],
    ):
        """
        Boot up the client process to connect to the SocketInterface.

        Args:
            command_lst (list[str]): List of strings to start the client process.
        Raises:
            ValueError: If oversubscribing is not supported for the Flux adapter or if conda environments are not supported.
        """
        if self._openmpi_oversubscribe:
            raise ValueError(
                "Oversubscribing is currently not supported for the Flux adapter."
            )
        if self._flux_executor is None:
            self._flux_executor = flux.job.FluxExecutor()
        if not self._flux_executor_nesting:
            jobspec = flux.job.JobspecV1.from_command(
                command=command_lst,
                num_tasks=self._cores,
                cores_per_task=self._threads_per_core,
                gpus_per_task=self._gpus_per_core,
                num_nodes=self._num_nodes,
                exclusive=self._exclusive,
            )
        else:
            jobspec = flux.job.JobspecV1.from_nest_command(
                command=command_lst,
                num_slots=self._cores,
                cores_per_slot=self._threads_per_core,
                gpus_per_slot=self._gpus_per_core,
                num_nodes=self._num_nodes,
                exclusive=self._exclusive,
            )
        jobspec.environment = dict(os.environ)
        if self._pmi_mode is not None:
            jobspec.setattr_shell_option("pmi", self._pmi_mode)
        if self._cwd is not None:
            jobspec.cwd = self._cwd
            os.makedirs(self._cwd, exist_ok=True)
        if self._flux_log_files and self._cwd is not None:
            jobspec.stderr = os.path.join(self._cwd, "flux.err")
            jobspec.stdout = os.path.join(self._cwd, "flux.out")
        elif self._flux_log_files:
            jobspec.stderr = os.path.abspath("flux.err")
            jobspec.stdout = os.path.abspath("flux.out")
        if self._priority is not None:
            self._future = self._flux_executor.submit(
                jobspec=jobspec, urgency=self._priority
            )
        else:
            self._future = self._flux_executor.submit(jobspec=jobspec)

    def shutdown(self, wait: bool = True):
        """
        Shutdown the FluxPythonInterface.

        Args:
            wait (bool, optional): Whether to wait for the execution to complete. Defaults to True.
        """
        if self._future is not None:
            if self.poll():
                self._future.cancel()
            # The flux future objects are not instantly updated,
            # still showing running after cancel was called,
            # so we wait until the execution is completed.
            self._future.result()

    def poll(self):
        """
        Check if the FluxPythonInterface is running.

        Returns:
            bool: True if the interface is running, False otherwise.
        """
        return self._future is not None and not self._future.done()

import os
from typing import Optional

import flux.job

from executorlib.shared.interface import BaseInterface


class FluxPythonInterface(BaseInterface):
    def __init__(
        self,
        cwd: Optional[str] = None,
        cores: int = 1,
        threads_per_core: int = 1,
        gpus_per_core: int = 0,
        oversubscribe: bool = False,
        executor: Optional[flux.job.FluxExecutor] = None,
        pmi: Optional[str] = None,
        nested_flux_executor: bool = False,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._executor = executor
        self._pmi = pmi
        self._future = None
        self._nested_flux_executor = nested_flux_executor

    def bootup(
        self,
        command_lst: list[str],
        prefix_name: Optional[str] = None,
        prefix_path: Optional[str] = None,
    ):
        """
        Boot up the client process to connect to the SocketInterface.

        Args:
            command_lst (list): list of strings to start the client process
            prefix_name (str): name of the conda environment to initialize
            prefix_path (str): path of the conda environment to initialize
            nested (bool): nested flux instance
        """
        if self._oversubscribe:
            raise ValueError(
                "Oversubscribing is currently not supported for the Flux adapter."
            )
        if prefix_name is not None or prefix_path is not None:
            raise ValueError(
                "Conda environments are currently not supported for the Flux adapter."
            )
        if self._executor is None:
            self._executor = flux.job.FluxExecutor()
        if not self._nested_flux_executor:
            jobspec = flux.job.JobspecV1.from_command(
                command=command_lst,
                num_tasks=self._cores,
                cores_per_task=self._threads_per_core,
                gpus_per_task=self._gpus_per_core,
                num_nodes=None,
                exclusive=False,
            )
        else:
            jobspec = flux.job.JobspecV1.from_nest_command(
                command=command_lst,
                num_slots=self._cores,
                cores_per_slot=self._threads_per_core,
                gpus_per_slot=self._gpus_per_core,
                num_nodes=None,
                exclusive=False,
            )
        jobspec.environment = dict(os.environ)
        if self._pmi is not None:
            jobspec.setattr_shell_option("pmi", self._pmi)
        if self._cwd is not None:
            jobspec.cwd = self._cwd
        self._future = self._executor.submit(jobspec)

    def shutdown(self, wait: bool = True):
        if self.poll():
            self._future.cancel()
        # The flux future objects are not instantly updated,
        # still showing running after cancel was called,
        # so we wait until the execution is completed.
        self._future.result()

    def poll(self):
        return self._future is not None and not self._future.done()

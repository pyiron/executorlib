import os
from typing import Optional

import flux.job

from pympipool.shared.interface import BaseInterface


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

    def bootup(self, command_lst: list[str]):
        if self._oversubscribe:
            raise ValueError(
                "Oversubscribing is currently not supported for the Flux adapter."
            )
        if self._executor is None:
            self._executor = flux.job.FluxExecutor()
        jobspec = flux.job.JobspecV1.from_command(
            command=command_lst,
            num_tasks=self._cores,
            cores_per_task=self._threads_per_core,
            gpus_per_task=self._gpus_per_core,
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

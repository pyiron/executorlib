from abc import ABC
import os
import subprocess


class BaseInterface(ABC):
    def __init__(
        self, cwd, cores=1, threads_per_core=1, gpus_per_core=0, oversubscribe=False
    ):
        self._cwd = cwd
        self._cores = cores
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._oversubscribe = oversubscribe

    def bootup(self, command_lst):
        raise NotImplementedError

    def shutdown(self, wait=True):
        raise NotImplementedError

    def poll(self):
        raise NotImplementedError


class FluxPythonInterface(BaseInterface):
    def __init__(
        self,
        cwd=None,
        cores=1,
        threads_per_core=1,
        gpus_per_core=0,
        oversubscribe=False,
        executor=None,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            threads_per_core=threads_per_core,
            oversubscribe=oversubscribe,
        )
        self._executor = executor
        self._future = None

    def bootup(self, command_lst):
        import flux.job

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
        if self._cwd is not None:
            jobspec.cwd = self._cwd
        self._future = self._executor.submit(jobspec)

    def shutdown(self, wait=True):
        if self.poll():
            self._future.cancel()
        # The flux future objects are not instantly updated,
        # still showing running after cancel was called,
        # so we wait until the execution is completed.
        self._future.result()

    def poll(self):
        return self._future is not None and not self._future.done()

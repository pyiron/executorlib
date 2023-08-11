from pympipool.shared.interface import (
    BaseInterface,
    SubprocessInterface,
    generate_slurm_command,
    generate_mpiexec_command
)


class PysqaInterface(BaseInterface):
    def __init__(
        self,
        cwd=None,
        cores=1,
        gpus_per_core=0,
        oversubscribe=False,
        queue_adapter=None,
        queue_type=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )
        self._queue_adapter = queue_adapter
        self._queue_type = queue_type
        self._queue_adapter_kwargs = queue_adapter_kwargs
        self._queue_id = None

    def bootup(self, command_lst):
        if self._queue_type.lower() == "slurm":
            command_prepend_lst = generate_slurm_command(
                cores=self._cores,
                cwd=self._cwd,
                gpus_per_core=self._gpus_per_core,
                oversubscribe=self._oversubscribe,
            )
        else:
            command_prepend_lst = generate_mpiexec_command(
                cores=self._cores,
                gpus_per_core=self._gpus_per_core,
                oversubscribe=self._oversubscribe,
            )
        self._queue_id = self._queue_adapter.submit_job(
            working_directory=self._cwd,
            cores=self._cores,
            command=" ".join(command_prepend_lst + command_lst),
            **self._queue_adapter_kwargs
        )

    def shutdown(self, wait=True):
        self._queue_adapter.delete_job(process_id=self._queue_id)

    def poll(self):
        return self._queue_adapter is not None


class FluxCmdInterface(SubprocessInterface):
    def generate_command(self, command_lst):
        command_prepend_lst = [
            "flux",
            "run",
            "-n",
            str(self._cores),
        ]
        if self._cwd is not None:
            command_prepend_lst += [
                "--cwd=" + self._cwd,
            ]
        if self._threads_per_core > 1:
            command_prepend_lst += ["--cores-per-task=" + str(self._threads_per_core)]
        if self._gpus_per_core > 0:
            command_prepend_lst += ["--gpus-per-task=" + str(self._gpus_per_core)]
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )

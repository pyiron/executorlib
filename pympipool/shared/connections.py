from abc import ABC
import subprocess


class BaseInterface(ABC):
    def __init__(self, cwd, cores=1, gpus_per_core=0, oversubscribe=False):
        self._cwd = cwd
        self._cores = cores
        self._gpus_per_core = gpus_per_core
        self._oversubscribe = oversubscribe

    def bootup(self, command_lst):
        raise NotImplementedError

    def shutdown(self, wait=True):
        raise NotImplementedError

    def poll(self):
        raise NotImplementedError


class SubprocessInterface(BaseInterface):
    def __init__(self, cwd=None, cores=1, gpus_per_core=0, oversubscribe=False):
        super().__init__(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )
        self._process = None

    def bootup(self, command_lst):
        self._process = subprocess.Popen(
            args=self.generate_command(command_lst=command_lst),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=self._cwd,
        )

    def generate_command(self, command_lst):
        return command_lst

    def shutdown(self, wait=True):
        self._process.terminate()
        self._process.stdout.close()
        self._process.stdin.close()
        self._process.stderr.close()
        if wait:
            self._process.wait()
        self._process = None

    def poll(self):
        return self._process is not None and self._process.poll() is None


class MpiExecInterface(SubprocessInterface):
    def generate_command(self, command_lst):
        command_prepend_lst = generate_mpiexec_command(
            cores=self._cores,
            gpus_per_core=self._gpus_per_core,
            oversubscribe=self._oversubscribe,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


class SlurmSubprocessInterface(SubprocessInterface):
    def generate_command(self, command_lst):
        command_prepend_lst = generate_slurm_command(
            cores=self._cores,
            cwd=self._cwd,
            gpus_per_core=self._gpus_per_core,
            oversubscribe=self._oversubscribe,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
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
        if self._gpus_per_core > 0:
            command_prepend_lst += ["--gpus-per-task=" + str(self._gpus_per_core)]
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


class FluxPythonInterface(BaseInterface):
    def __init__(
        self, cwd=None, cores=1, gpus_per_core=0, oversubscribe=False, executor=None
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
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
            command=" ".join(command_lst),
            num_tasks=1,
            cores_per_task=self._cores,
            gpus_per_task=self._gpus_per_core,
            num_nodes=None,
            exclusive=False,
        )
        jobspec.cwd = self._cwd
        self._future = self._executor.submit(jobspec)

    def shutdown(self, wait=True):
        self._executor.shutdown(wait=wait)

    def poll(self):
        return self._executor is not None


def generate_slurm_command(cores, cwd, gpus_per_core=0, oversubscribe=False):
    command_prepend_lst = ["srun", "-n", str(cores), "-D", cwd]
    if gpus_per_core > 0:
        command_prepend_lst += ["--gpus-per-task=" + str(gpus_per_core)]
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    return command_prepend_lst


def generate_mpiexec_command(cores, gpus_per_core=0, oversubscribe=False):
    command_prepend_lst = ["mpiexec", "-n", str(cores)]
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    if gpus_per_core > 0:
        raise ValueError()
    return command_prepend_lst


def get_connection_interface(
    cwd=None,
    cores=1,
    gpus_per_core=0,
    oversubscribe=False,
    enable_flux_backend=False,
    enable_slurm_backend=False,
    queue_adapter=None,
    queue_type=None,
    queue_adapter_kwargs=None,
):
    """
    Backwards compatibility adapter to get the connection interface

    Args:
        cwd (str/None): current working directory where the parallel python task is executed
        cores (int): defines the total number of MPI ranks to use
        gpus_per_core (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only) - default False
        enable_flux_backend (bool): use the flux-framework as backend rather than just calling mpiexec
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_type (str): type of the queuing system
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

    Returns:
        pympipool.shared.connections.BaseInterface: Connection interface
    """
    if queue_adapter is not None:
        connections = PysqaInterface(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
            queue_adapter=queue_adapter,
            queue_type=queue_type,
            queue_adapter_kwargs=queue_adapter_kwargs,
        )
    elif enable_flux_backend:
        connections = FluxCmdInterface(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )
    elif enable_slurm_backend:
        connections = SlurmSubprocessInterface(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )
    else:
        connections = MpiExecInterface(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )
    return connections

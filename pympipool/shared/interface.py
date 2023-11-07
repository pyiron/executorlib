from abc import ABC
import subprocess


MPI_COMMAND = "mpiexec"
SLURM_COMMAND = "srun"


class BaseInterface(ABC):
    def __init__(self, cwd, cores=1, oversubscribe=False):
        self._cwd = cwd
        self._cores = cores
        self._oversubscribe = oversubscribe

    def bootup(self, command_lst):
        raise NotImplementedError

    def shutdown(self, wait=True):
        raise NotImplementedError

    def poll(self):
        raise NotImplementedError


class SubprocessInterface(BaseInterface):
    def __init__(
        self,
        cwd=None,
        cores=1,
        oversubscribe=False,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        )
        self._process = None

    def bootup(self, command_lst):
        self._process = subprocess.Popen(
            args=self.generate_command(command_lst=command_lst),
            cwd=self._cwd,
        )

    def generate_command(self, command_lst):
        return command_lst

    def shutdown(self, wait=True):
        self._process.communicate()
        self._process.terminate()
        if wait:
            self._process.wait()
        self._process = None

    def poll(self):
        return self._process is not None and self._process.poll() is None


class MpiExecInterface(SubprocessInterface):
    def generate_command(self, command_lst):
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
        cwd=None,
        cores=1,
        threads_per_core=1,
        gpus_per_core=0,
        oversubscribe=False,
    ):
        super().__init__(
            cwd=cwd,
            cores=cores,
            oversubscribe=oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core

    def generate_command(self, command_lst):
        command_prepend_lst = generate_slurm_command(
            cores=self._cores,
            cwd=self._cwd,
            threads_per_core=self._threads_per_core,
            gpus_per_core=self._gpus_per_core,
            oversubscribe=self._oversubscribe,
        )
        return super().generate_command(
            command_lst=command_prepend_lst + command_lst,
        )


def generate_mpiexec_command(cores, oversubscribe=False):
    command_prepend_lst = [MPI_COMMAND, "-n", str(cores)]
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    return command_prepend_lst


def generate_slurm_command(
    cores, cwd, threads_per_core=1, gpus_per_core=0, oversubscribe=False
):
    command_prepend_lst = [SLURM_COMMAND, "-n", str(cores)]
    if cwd is not None:
        command_prepend_lst += ["-D", cwd]
    if threads_per_core > 1:
        command_prepend_lst += ["--cpus-per-task" + str(threads_per_core)]
    if gpus_per_core > 0:
        command_prepend_lst += ["--gpus-per-task=" + str(gpus_per_core)]
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    return command_prepend_lst

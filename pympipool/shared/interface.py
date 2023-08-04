from abc import ABC
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


class SubprocessInterface(BaseInterface):
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
            threads_per_core=threads_per_core,
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


def generate_mpiexec_command(cores, gpus_per_core=0, oversubscribe=False):
    command_prepend_lst = ["mpiexec", "-n", str(cores)]
    if oversubscribe:
        command_prepend_lst += ["--oversubscribe"]
    if gpus_per_core > 0:
        raise ValueError()
    return command_prepend_lst

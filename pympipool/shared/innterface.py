from abc import ABC


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

import os

import flux.job

from pympipool.shared.executorbase import (
    cloudpickle_register,
    ExecutorBase,
    executor_broker,
    execute_parallel_tasks,
)
from pympipool.shared.interface import BaseInterface
from pympipool.shared.thread import RaisingThread


class PyFluxExecutor(ExecutorBase):
    """
    The pympipool.flux.PyFluxExecutor leverages the flux framework to distribute python tasks within a queuing system
    allocation. In analogy to the pympipool.slurm.PySlurmExecutur it provides the option to specify the number of
    threads per worker as well as the number of GPUs per worker in addition to specifying the number of cores per
    worker.

    Args:
        max_workers (int): defines the number workers which can execute functions in parallel
        cores_per_worker (int): number of MPI cores to be used for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_worker (int): number of GPUs per worker - defaults to 0
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true

    Examples:

        >>> import numpy as np
        >>> from pympipool.flux import PyFluxExecutor
        >>>
        >>> def calc(i, j, k):
        >>>     from mpi4py import MPI
        >>>     size = MPI.COMM_WORLD.Get_size()
        >>>     rank = MPI.COMM_WORLD.Get_rank()
        >>>     return np.array([i, j, k]), size, rank
        >>>
        >>> def init_k():
        >>>     return {"k": 3}
        >>>
        >>> with PyFluxExecutor(max_workers=2, init_function=init_k) as p:
        >>>     fs = p.submit(calc, 2, j=4)
        >>>     print(fs.result())
        [(array([2, 4, 3]), 2, 0), (array([2, 4, 3]), 2, 1)]

    """

    def __init__(
        self,
        max_workers=1,
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        init_function=None,
        cwd=None,
        executor=None,
        hostname_localhost=False,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=executor_broker,
            kwargs={
                # Broker Arguments
                "future_queue": self._future_queue,
                "max_workers": max_workers,
                "hostname_localhost": hostname_localhost,
                "executor_class": PyFluxSingleTaskExecutor,
                # Executor Arguments
                "cores": cores_per_worker,
                "threads_per_core": threads_per_core,
                "gpus_per_task": int(gpus_per_worker / cores_per_worker),
                "init_function": init_function,
                "cwd": cwd,
                "executor": executor,
            },
        )
        self._process.start()


class PyFluxSingleTaskExecutor(ExecutorBase):
    """
    The pympipool.flux.PyFluxSingleTaskExecutor is the internal worker for the pympipool.flux.PyFluxExecutor.

    Args:
        cores (int): defines the number of MPI ranks to use for each function call
        threads_per_core (int): number of OpenMP threads to be used for each function call
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        init_function (None): optional function to preset arguments for functions which are submitted later
        cwd (str/None): current working directory where the parallel python task is executed
        executor (flux.job.FluxExecutor): Flux Python interface to submit the workers to flux
        hostname_localhost (boolean): use localhost instead of the hostname to establish the zmq connection. In the
                                      context of an HPC cluster this essential to be able to communicate to an
                                      Executor running on a different compute node within the same allocation. And
                                      in principle any computer should be able to resolve that their own hostname
                                      points to the same address as localhost. Still MacOS >= 12 seems to disable
                                      this look up for security reasons. So on MacOS it is required to set this
                                      option to true

    """

    def __init__(
        self,
        cores=1,
        threads_per_core=1,
        gpus_per_task=0,
        init_function=None,
        cwd=None,
        executor=None,
        hostname_localhost=False,
    ):
        super().__init__()
        self._process = RaisingThread(
            target=execute_parallel_tasks,
            kwargs={
                # Executor Arguments
                "future_queue": self._future_queue,
                "cores": cores,
                "interface_class": FluxPythonInterface,
                "hostname_localhost": hostname_localhost,
                # Interface Arguments
                "threads_per_core": threads_per_core,
                "gpus_per_core": gpus_per_task,
                "cwd": cwd,
                "executor": executor,
            },
        )
        self._process.start()
        self._set_init_function(init_function=init_function)
        cloudpickle_register(ind=3)


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
            oversubscribe=oversubscribe,
        )
        self._threads_per_core = threads_per_core
        self._gpus_per_core = gpus_per_core
        self._executor = executor
        self._future = None

    def bootup(self, command_lst):
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
